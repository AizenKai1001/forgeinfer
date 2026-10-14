//! Chat with documents: split attached files into passages, rank the passages against a question by the vectors an embedding model made of them, and write the prompt the way LM Studio does
//! (the citations in front of the question). Nothing here touches the network or the window: `app.rs` embeds and calls these.

use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

use crate::chats::Citation;

/// Characters of a passage and of the overlap between neighbours: about 512 and 100 tokens of English text (LM Studio's chunk size and overlap).
pub const CHUNK_CHARS: usize = 1800;
pub const OVERLAP_CHARS: usize = 350;

/// Below the minimum score a passage does not count as an answer, but a question that matches nothing well (a request for a summary, a vague one) still gets the nearest passages, marked as weak matches, when
/// they score at least this much: the model decides whether they help.
pub const WEAK_FLOOR: f32 = 0.4;

/// The share of the free context the files may take before they are searched instead of sent whole (LM Studio's 70 percent).
const INLINE_SHARE: f32 = 0.7;

/// Send the files whole, or search them?
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Strategy {
    Inline,
    Retrieve,
}

/// LM Studio's rule: the files go whole into the prompt while they and the question take at most 70 percent of what is left of the context.
pub fn choose(file_tokens: usize, prompt_tokens: usize, ctx: usize, occupied_tokens: usize) -> Strategy {
    let free = ctx.saturating_sub(occupied_tokens);
    if (file_tokens + prompt_tokens) as f32 > free as f32 * INLINE_SHARE {
        Strategy::Retrieve
    } else {
        Strategy::Inline
    }
}

/// A recursive character splitter (LangChain's, as LM Studio uses it): split on the coarsest separator that occurs (blank line, line, space, character), pieces that fit are packed together up to
/// `size` characters with the last `overlap` characters of a passage starting the next one, pieces that do not fit are split again with the finer separators. Lengths are in characters.
pub fn split(text: &str, size: usize, overlap: usize) -> Vec<String> {
    let size = size.max(16);
    let overlap = overlap.min(size / 2);
    let mut out = Vec::new();
    split_into(text, &["\n\n", "\n", " ", ""], size, overlap, &mut out);
    out
}

fn len(s: &str) -> usize {
    s.chars().count()
}

fn split_into(text: &str, separators: &[&str], size: usize, overlap: usize, out: &mut Vec<String>) {
    // the first separator that occurs; the empty one (single characters) always does
    let at = separators.iter().position(|s| s.is_empty() || text.contains(s)).unwrap_or(separators.len() - 1);
    let (sep, finer) = (separators[at], &separators[at + 1..]);
    let pieces: Vec<String> = if sep.is_empty() { text.chars().map(|c| c.to_string()).collect() } else { text.split(sep).filter(|p| !p.is_empty()).map(str::to_string).collect() };
    let mut small: Vec<String> = Vec::new();
    for p in pieces {
        if len(&p) < size {
            small.push(p);
            continue;
        }
        if !small.is_empty() {
            merge(&small, sep, size, overlap, out);
            small.clear();
        }
        if finer.is_empty() {
            out.push(p);
        } else {
            split_into(&p, finer, size, overlap, out);
        }
    }
    if !small.is_empty() {
        merge(&small, sep, size, overlap, out);
    }
}

/// Pack small pieces into passages of at most `size` characters (`_merge_splits` of LangChain): a passage ends when the next piece does not fit, and the pieces at its end that add up to `overlap` start the next.
fn merge(pieces: &[String], sep: &str, size: usize, overlap: usize, out: &mut Vec<String>) {
    let sep_len = len(sep);
    let mut window: Vec<&str> = Vec::new();
    let mut total = 0usize;
    let emit = |window: &[&str], out: &mut Vec<String>| {
        let doc = window.join(sep);
        let doc = doc.trim();
        if !doc.is_empty() {
            out.push(doc.to_string());
        }
    };
    for p in pieces {
        let l = len(p);
        let join = if window.is_empty() { 0 } else { sep_len };
        if total + l + join > size && !window.is_empty() {
            emit(&window, out);
            // drop pieces from the front until what is left is within the overlap and the next piece fits
            while !window.is_empty() && (total > overlap || total + l + (if window.len() > 1 { sep_len } else { 0 }) > size) {
                total -= len(window[0]) + if window.len() > 1 { sep_len } else { 0 };
                window.remove(0);
            }
        }
        total += l + if window.is_empty() { 0 } else { sep_len };
        window.push(p);
    }
    emit(&window, out);
}

/// A 64-bit FNV-1a hash of a text: which attached files an index already holds.
pub fn fingerprint(text: &str) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for b in text.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// The passage of a file and, once the embedding model has seen it, its vector.
#[derive(Clone, Debug, PartialEq)]
pub struct Passage {
    pub file: String,
    pub text: String,
}

/// The passages of the files of one chat with their vectors (all of length one: the dot product is the cosine).
#[derive(Clone, Debug, Default)]
pub struct DocIndex {
    docs: HashSet<u64>,
    pub passages: Vec<Passage>,
    pub vectors: Vec<Vec<f32>>,
}

/// One search result: the index of a passage and its score.
#[derive(Clone, Debug, PartialEq)]
pub struct Hit {
    pub passage: usize,
    pub score: f32,
}

impl DocIndex {
    pub fn has(&self, fingerprint: u64) -> bool {
        self.docs.contains(&fingerprint)
    }

    pub fn is_empty(&self) -> bool {
        self.passages.is_empty()
    }

    /// Add the passages of a file with their vectors (one per passage).
    pub fn add(&mut self, fingerprint: u64, file: &str, texts: Vec<String>, vectors: Vec<Vec<f32>>) {
        debug_assert_eq!(texts.len(), vectors.len());
        self.docs.insert(fingerprint);
        for (t, v) in texts.into_iter().zip(vectors) {
            self.passages.push(Passage { file: file.to_string(), text: t });
            self.vectors.push(v);
        }
    }

    /// The best `limit` passages for a question vector whose score is above `threshold`, best first.
    pub fn search(&self, query: &[f32], limit: usize, threshold: f32) -> Vec<Hit> {
        let mut hits: Vec<Hit> = self
            .vectors
            .iter()
            .enumerate()
            .map(|(i, v)| Hit { passage: i, score: v.iter().zip(query).map(|(a, b)| a * b).sum() })
            .filter(|h| h.score > threshold)
            .collect();
        hits.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        hits.truncate(limit);
        hits
    }
}

/// What a search reports while it runs, and how it is told to stop.
#[derive(Clone, Default)]
pub struct Progress {
    /// Passages embedded so far, and passages that need it in all (known once a file is split).
    pub done: Arc<AtomicUsize>,
    pub total: Arc<AtomicUsize>,
    pub cancel: Arc<AtomicBool>,
}

/// Passages per request to the embedding model.
const BATCH: usize = 32;

/// The prefixes an embedding model wants in front of the passages it indexes and the questions it is asked (nomic-embed-text was trained with `search_document: ` and `search_query: `; other models use none).
pub fn prefixes_for(model_file: &str) -> (&'static str, &'static str) {
    if model_file.to_lowercase().contains("nomic") {
        ("search_document: ", "search_query: ")
    } else {
        ("", "")
    }
}

/// Search the files of a chat for a question: the files the index does not hold yet are split and embedded (`embed` turns a list of texts into their vectors), then the question is embedded and the best
/// passages above `threshold` come back as citations, with the index that now holds all the files. `docs` are (name, text) of every file of the chat.
#[allow(clippy::too_many_arguments)]
pub fn search(
    mut index: DocIndex,
    docs: &[(String, String)],
    question: &str,
    limit: usize,
    threshold: f32,
    prefixes: (&str, &str),
    progress: &Progress,
    embed: &mut dyn FnMut(&[String]) -> Result<Vec<Vec<f32>>, String>,
) -> Result<(DocIndex, Vec<Citation>), String> {
    let stop = || progress.cancel.load(Ordering::Relaxed);
    // what has to be embedded: the passages of the files the index has not seen
    let mut todo: Vec<(u64, &str, Vec<String>)> = Vec::new();
    for (name, text) in docs {
        let fp = fingerprint(text);
        if index.has(fp) || todo.iter().any(|t| t.0 == fp) {
            continue;
        }
        let parts = split(text, CHUNK_CHARS, OVERLAP_CHARS);
        progress.total.fetch_add(parts.len(), Ordering::Relaxed);
        todo.push((fp, name, parts));
    }
    for (fp, name, parts) in todo {
        let mut vectors: Vec<Vec<f32>> = Vec::with_capacity(parts.len());
        for batch in parts.chunks(BATCH) {
            if stop() {
                return Err("stopped".to_string());
            }
            let inputs: Vec<String> = batch.iter().map(|p| format!("{}{p}", prefixes.0)).collect();
            let v = embed(&inputs)?;
            if v.len() != batch.len() {
                return Err(format!("{} vectors came back for {} passages", v.len(), batch.len()));
            }
            vectors.extend(v);
            progress.done.fetch_add(batch.len(), Ordering::Relaxed);
        }
        index.add(fp, name, parts, vectors);
    }
    if stop() {
        return Err("stopped".to_string());
    }
    let q = embed(&[format!("{}{question}", prefixes.1)])?;
    let q = q.first().ok_or("the embedding model returned no vector for the question")?;
    let mut hits = index.search(q, limit, threshold);
    let weak = hits.is_empty() && threshold > WEAK_FLOOR;
    if weak {
        hits = index.search(q, limit.min(2), WEAK_FLOOR);
    }
    let citations = hits.iter().map(|h| Citation { file: index.passages[h.passage].file.clone(), text: index.passages[h.passage].text.clone(), score: h.score, weak }).collect();
    Ok((index, citations))
}

/// What the model reads for a question with passages found in the user's files (LM Studio's wording; the file name is added so that the model can say where something comes from).
pub fn citations_prompt(passages: &[(String, String)], weak: bool, question: &str) -> String {
    if passages.is_empty() {
        return format!(
            "No passage of the files provided by the user matched the question closely enough. Respond to the best of your ability, and say so if you would need the files for it.\n\nUser Query:\n\n{question}"
        );
    }
    let mut s = if weak {
        String::from("No passage of the files provided by the user matched the question closely. These are the nearest passages, and they may not be relevant:\n\n")
    } else {
        String::from("The following citations were found in the files provided by the user:\n\n")
    };
    for (i, (file, text)) in passages.iter().enumerate() {
        s.push_str(&format!("Citation {} (from {file}): \"{}\"\n\n", i + 1, text.trim()));
    }
    s.push_str("Use the citations above to respond to the user query, only if they are relevant. Otherwise, respond to the best of your ability without them.\n\nUser Query:\n\n");
    s.push_str(question);
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_text_is_one_passage() {
        assert_eq!(split("hello world", 100, 10), vec!["hello world".to_string()]);
        assert!(split("   \n\n  ", 100, 10).is_empty());
    }

    #[test]
    fn passages_stay_within_the_size_and_cover_the_text() {
        let text: String = (0..200).map(|i| format!("Sentence number {i} says something about the harbour. ")).collect();
        let parts = split(&text, 500, 100);
        assert!(parts.len() > 10);
        for p in &parts {
            assert!(p.chars().count() <= 500, "{} characters", p.chars().count());
        }
        // every sentence is in some passage, whole
        for i in 0..200 {
            let s = format!("Sentence number {i} says something about the harbour.");
            assert!(parts.iter().any(|p| p.contains(&s)), "sentence {i} is missing");
        }
    }

    #[test]
    fn neighbouring_passages_overlap() {
        let words: String = (0..400).map(|i| format!("w{i} ")).collect();
        let parts = split(&words, 300, 80);
        assert!(parts.len() >= 3);
        for pair in parts.windows(2) {
            let last_word = pair[0].split(' ').next_back().unwrap();
            assert!(pair[1].contains(last_word), "{:?} does not start inside {:?}", pair[1], pair[0]);
        }
    }

    #[test]
    fn paragraphs_are_kept_together_when_they_fit() {
        let a = "First paragraph, about apples. ".repeat(4);
        let b = "Second paragraph, about oranges. ".repeat(4);
        let text = format!("{}\n\n{}", a.trim(), b.trim());
        let parts = split(&text, 200, 20);
        assert_eq!(parts.len(), 2, "{parts:?}");
        assert!(parts[0].starts_with("First") && parts[1].starts_with("Second"));
    }

    #[test]
    fn a_word_longer_than_the_size_is_cut_by_characters() {
        let text = "x".repeat(1000);
        let parts = split(&text, 100, 10);
        assert!(parts.len() >= 10);
        assert!(parts.iter().all(|p| p.chars().count() <= 100));
        assert_eq!(parts.iter().map(|p| p.matches('x').count()).sum::<usize>() >= 1000, true);
    }

    #[test]
    fn unicode_is_counted_in_characters() {
        let text = "日本語のテキストです。".repeat(100);
        let parts = split(&text, 120, 20);
        assert!(parts.iter().all(|p| p.chars().count() <= 120));
        assert!(parts.len() >= 8);
    }

    #[test]
    fn the_strategy_follows_the_free_context() {
        // 8,192 tokens free: 70 percent is 5,734
        assert_eq!(choose(3000, 100, 8192, 0), Strategy::Inline);
        assert_eq!(choose(6000, 100, 8192, 0), Strategy::Retrieve);
        // a conversation that already fills half of it leaves less
        assert_eq!(choose(3000, 100, 8192, 4096), Strategy::Retrieve);
        assert_eq!(choose(1, 1, 0, 0), Strategy::Retrieve);
    }

    #[test]
    fn the_index_ranks_by_cosine_and_applies_the_threshold() {
        let mut idx = DocIndex::default();
        assert!(!idx.has(7));
        idx.add(7, "a.txt", vec!["north".into(), "east".into(), "northeast".into()], vec![vec![0.0, 1.0], vec![1.0, 0.0], vec![0.7071, 0.7071]]);
        assert!(idx.has(7));
        assert_eq!(idx.passages.len(), 3);
        let hits = idx.search(&[0.0, 1.0], 2, 0.5);
        assert_eq!(hits.iter().map(|h| h.passage).collect::<Vec<_>>(), vec![0, 2]);
        assert!(hits[0].score > hits[1].score);
        // nothing above a high threshold, limit respected
        assert!(idx.search(&[0.6, 0.6], 3, 2.0).is_empty());
        assert_eq!(idx.search(&[0.6, 0.8], 1, -1.0).len(), 1);
    }

    #[test]
    fn the_prompt_names_the_files_and_the_question() {
        let p = citations_prompt(&[("notes.txt".into(), "The ship docks at six.".into()), ("log.txt".into(), "Cargo: tea.".into())], false, "When does it dock?");
        assert!(p.starts_with("The following citations were found in the files provided by the user:"));
        assert!(p.contains("Citation 1 (from notes.txt): \"The ship docks at six.\""));
        assert!(p.contains("Citation 2 (from log.txt): \"Cargo: tea.\""));
        assert!(p.ends_with("User Query:\n\nWhen does it dock?"));
        let none = citations_prompt(&[], false, "Anything?");
        assert!(none.contains("No passage") && none.ends_with("Anything?"));
        let weak = citations_prompt(&[("a.txt".into(), "Maybe.".into())], true, "Summary?");
        assert!(weak.starts_with("No passage of the files provided by the user matched the question closely.") && weak.contains("Citation 1 (from a.txt): \"Maybe.\"") && weak.ends_with("Summary?"));
    }

    /// A toy embedding: the vector counts the letters a, b and c of the text (so "abc ..." passages are about the same thing as "abc ..." questions).
    fn toy(texts: &[String]) -> Result<Vec<Vec<f32>>, String> {
        Ok(texts
            .iter()
            .map(|t| {
                let t = t.split_once(": ").map_or(t.as_str(), |x| x.1);
                let v = [t.matches('a').count() as f32, t.matches('b').count() as f32, t.matches('c').count() as f32];
                let n = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
                v.iter().map(|x| x / n).collect()
            })
            .collect())
    }

    #[test]
    fn a_search_indexes_new_files_once_and_finds_the_closest_passage() {
        let docs = vec![
            ("alpha.txt".to_string(), "aaaa aaaa aaaa\n\n".repeat(3) + &"bbbb bbbb\n\n".repeat(3)),
            ("gamma.txt".to_string(), "cccc cccc cccc\n\n".repeat(4)),
        ];
        let progress = Progress::default();
        let calls = std::cell::Cell::new(0usize);
        let mut embed = |t: &[String]| {
            calls.set(calls.get() + 1);
            toy(t)
        };
        let (index, cites) = search(DocIndex::default(), &docs, "ccc?", 2, 0.5, ("search_document: ", "search_query: "), &progress, &mut embed).unwrap();
        assert!(!index.is_empty());
        assert!(cites.iter().all(|c| c.file == "gamma.txt" && c.score > 0.9), "{cites:?}");
        assert!(progress.done.load(Ordering::Relaxed) >= 2 && progress.done.load(Ordering::Relaxed) == progress.total.load(Ordering::Relaxed));
        let first_calls = calls.get();
        // the same files again: only the question is embedded
        let (index, cites) = search(index, &docs, "aaa", 1, 0.5, ("", ""), &Progress::default(), &mut embed).unwrap();
        assert_eq!(calls.get(), first_calls + 1);
        assert_eq!(cites.len(), 1);
        assert_eq!(cites[0].file, "alpha.txt");
        // nothing close enough: no citations
        let (_, none) = search(index, &docs, "xyz", 3, 0.5, ("", ""), &Progress::default(), &mut embed).unwrap();
        assert!(none.is_empty());
    }

    #[test]
    fn a_question_that_matches_nothing_well_gets_the_nearest_passages_as_weak_matches() {
        let docs = vec![("alpha.txt".to_string(), "aaaa aaaa\n\n".repeat(4)), ("gamma.txt".to_string(), "cccc cccc\n\n".repeat(4))];
        // "ab": a cosine of 0.7 with the a passage, nothing with the c passage
        let (index, cites) = search(DocIndex::default(), &docs, "ab", 3, 0.9, ("", ""), &Progress::default(), &mut |t| toy(t)).unwrap();
        assert_eq!(cites.len(), 1, "{cites:?}");
        assert!(cites[0].weak && cites[0].file == "alpha.txt" && cites[0].score > 0.4 && cites[0].score < 0.9);
        // a question about nothing in the files (cosine 0) gets no passage at all
        let (_, none) = search(index, &docs, "xyz", 3, 0.9, ("", ""), &Progress::default(), &mut |t| toy(t)).unwrap();
        assert!(none.is_empty());
        // a good match is not weak
        let (_, good) = search(DocIndex::default(), &docs, "cc", 3, 0.9, ("", ""), &Progress::default(), &mut |t| toy(t)).unwrap();
        assert!(good.len() == 1 && !good[0].weak);
    }

    #[test]
    fn a_stopped_search_and_a_failing_model_are_errors() {
        let docs = vec![("a.txt".to_string(), "aaa ".repeat(2000))];
        let p = Progress::default();
        p.cancel.store(true, Ordering::Relaxed);
        assert_eq!(search(DocIndex::default(), &docs, "a", 1, 0.0, ("", ""), &p, &mut |t| toy(t)).unwrap_err(), "stopped");
        let e = search(DocIndex::default(), &docs, "a", 1, 0.0, ("", ""), &Progress::default(), &mut |_| Err("server down".to_string())).unwrap_err();
        assert_eq!(e, "server down");
        let short = search(DocIndex::default(), &docs, "a", 1, 0.0, ("", ""), &Progress::default(), &mut |_| Ok(vec![])).unwrap_err();
        assert!(short.contains("vectors came back"));
    }

    #[test]
    fn nomic_models_get_their_task_prefixes() {
        assert_eq!(prefixes_for("nomic-embed-text-v1.5.Q4_K_M.gguf"), ("search_document: ", "search_query: "));
        assert_eq!(prefixes_for("bge-small-en.gguf"), ("", ""));
    }

    #[test]
    fn fingerprints_differ_for_different_texts() {
        assert_eq!(fingerprint("abc"), fingerprint("abc"));
        assert_ne!(fingerprint("abc"), fingerprint("abd"));
        assert_ne!(fingerprint(""), fingerprint("a"));
    }
}
