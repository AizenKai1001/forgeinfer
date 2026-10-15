//! Chat with documents: split attached files into passages, rank the passages against a question by the vectors an embedding model made of them, and write the prompt the way LM Studio does
//! (the citations in front of the question). Nothing here touches the network or the window: `app.rs` embeds and calls these.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use crate::chats::Citation;

/// Characters of a passage and of the overlap between neighbours: about 512 and 100 tokens of English text (LM Studio's chunk size and overlap).
pub const CHUNK_CHARS: usize = 1800;
pub const OVERLAP_CHARS: usize = 350;

/// Below the minimum score a passage does not count as an answer, but a question that matches nothing well (a request for a summary, a vague one) still gets the nearest passages, marked as weak matches, when
/// they score at least this much (0.4 for a minimum of 0.6): the model decides whether they help.
pub fn weak_floor(threshold: f32) -> f32 {
    (threshold - 0.2).max(0.0)
}

/// The share of the free context the files may take before they are searched instead of sent whole (LM Studio's 70 percent).
const INLINE_SHARE: f32 = 0.7;

/// Is this a question about the document as a whole (what it is about, a summary, its abstract or introduction)? Meaning-based search cannot find "the abstract": such a question gets the opening
/// passages of the files as well.
pub fn is_overview(question: &str) -> bool {
    let q = question.to_lowercase();
    ["summar", "overview", "abstract", "what is this", "what's this", "what is the document", "what is the paper", "what is the file", "about this", "what is it about", "main point", "main idea", "tl;dr", "tldr", "introduction", "outline", "table of contents", "tell me about", "key point", "the gist"]
        .iter()
        .any(|k| q.contains(k))
}

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

/// The passages of a file with the page of each: a text with page breaks (form feeds: what `office` makes of a PDF) is split page by page, so that a passage never crosses a page; any other text is one run.
pub fn split_pages(text: &str, size: usize, overlap: usize) -> Vec<(String, Option<u32>)> {
    if !text.contains('\u{c}') {
        return split(text, size, overlap).into_iter().map(|p| (p, None)).collect();
    }
    let mut out = Vec::new();
    for (i, page) in text.split('\u{c}').enumerate() {
        out.extend(split(page, size, overlap).into_iter().map(|p| (p, Some(i as u32 + 1))));
    }
    out
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
    /// The page of a PDF the passage is on.
    pub page: Option<u32>,
}

/// The words of a text for the keyword search: lower-case letters and digits.
fn words(text: &str) -> impl Iterator<Item = String> + '_ {
    text.split(|c: char| !c.is_alphanumeric()).filter(|w| !w.is_empty()).map(|w| w.to_lowercase())
}

/// The words of a question that name something: a name or a title (a capital letter inside the sentence), an acronym or a mixed-case word (NF4, QLoRA, MMLU), a number or a code (a digit: INV-7731-QX,
/// route R-58), split into the words that `words` makes of them. The words of everyday language that happen to be rare in a file ("limitation", "authors") are not among them: a passage with such a word
/// is not the one the question is about.
fn names_in(question: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for (i, raw) in question.split_whitespace().enumerate() {
        let w = raw.trim_matches(|c: char| !c.is_alphanumeric());
        let letters: Vec<char> = w.chars().filter(|c| c.is_alphabetic()).collect();
        let has_digit = w.chars().any(|c| c.is_ascii_digit());
        let acronym = letters.len() >= 2 && letters.iter().all(|c| c.is_uppercase());
        let mixed = letters.iter().skip(1).any(|c| c.is_uppercase());
        // (the first word of a question is capitalized whatever it is)
        let capitalized = i > 0 && letters.first().is_some_and(|c| c.is_uppercase());
        if has_digit || acronym || mixed || capitalized {
            out.extend(words(w).filter(|t| (t.chars().count() >= 2 || t.chars().all(|c| c.is_ascii_digit())) && !LITTLE_WORDS.contains(&t.as_str())));
        }
    }
    out.sort();
    out.dedup();
    out
}

/// The little words of questions: they say nothing about which passage is meant.
const LITTLE_WORDS: &[&str] = &[
    "the", "a", "an", "of", "to", "in", "on", "at", "for", "and", "or", "is", "are", "was", "were", "be", "been", "what", "who", "whom", "which", "where", "when", "why", "how", "did", "does", "do", "with", "by", "from", "it",
    "its", "this", "that", "these", "those", "as", "about", "into", "than", "then", "there", "their", "they", "he", "she", "his", "her", "you", "your", "me", "my", "we", "our", "can", "could", "would", "should", "will",
    "has", "have", "had", "not", "no", "any", "all", "some", "tell", "give", "show", "please", "say", "says", "said", "mean", "means",
];

/// A word is rare when it is in at most this many passages (and at most a twelfth of them).
const RARE_MAX: usize = 20;

/// A vector scaled to length one (a server that returns vectors of another length, as some do, still gives cosines).
fn unit(v: &[f32]) -> Vec<f32> {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > 0.0 {
        v.iter().map(|x| x / n).collect()
    } else {
        v.to_vec()
    }
}

/// The passages of the files of one chat with their vectors (all of length one: the dot product is the cosine).
#[derive(Clone, Debug, Default)]
pub struct DocIndex {
    /// The files the index holds: the fingerprint of the text and the name.
    docs: Vec<(u64, String)>,
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
        self.docs.iter().any(|d| d.0 == fingerprint)
    }

    /// Forget the files that are not in `names` any more (a message with a file was deleted): their passages must not be found.
    pub fn retain_files(&mut self, names: &[&str]) {
        self.docs.retain(|d| names.contains(&d.1.as_str()));
        let (passages, vectors): (Vec<Passage>, Vec<Vec<f32>>) = std::mem::take(&mut self.passages).into_iter().zip(std::mem::take(&mut self.vectors)).filter(|(p, _)| names.contains(&p.file.as_str())).unzip();
        self.passages = passages;
        self.vectors = vectors;
    }

    pub fn is_empty(&self) -> bool {
        self.passages.is_empty()
    }

    /// Add the passages of a file with their vectors (one per passage).
    pub fn add(&mut self, fingerprint: u64, file: &str, passages: Vec<(String, Option<u32>)>, vectors: Vec<Vec<f32>>) {
        debug_assert_eq!(passages.len(), vectors.len());
        self.docs.push((fingerprint, file.to_string()));
        for ((text, page), v) in passages.into_iter().zip(vectors) {
            self.passages.push(Passage { file: file.to_string(), text, page });
            self.vectors.push(unit(&v));
        }
    }

    /// The first passage of each of the first `files` files (the opening of a document: its title, abstract or introduction).
    pub fn openings(&self, files: usize) -> Vec<usize> {
        let mut seen: Vec<&str> = Vec::new();
        let mut out = Vec::new();
        for (i, p) in self.passages.iter().enumerate() {
            if !seen.contains(&p.file.as_str()) {
                if seen.len() == files {
                    break;
                }
                seen.push(&p.file);
                out.push(i);
            }
        }
        out
    }

    /// The passages that have a name, an identifier or a number of the question (`names_in`) that few passages have, best first, at most `limit`: BM25 over those words only. Embeddings are weak exactly
    /// there (a question that names "invoice INV-7731-QX" or "Marguerite Okafor" scores 0.5 to 0.6 against the right passage, and sometimes far below the rest), a keyword match is not.
    pub fn lexical(&self, question: &str, limit: usize) -> Vec<usize> {
        let n = self.passages.len();
        // in a short file every word is rare
        if n < 8 || limit == 0 {
            return Vec::new();
        }
        let terms = names_in(question);
        if terms.is_empty() {
            return Vec::new();
        }
        let slot: HashMap<&str, usize> = terms.iter().enumerate().map(|(i, t)| (t.as_str(), i)).collect();
        // per passage: how often each word of the question occurs, and how long it is
        let mut counts: Vec<Vec<u32>> = Vec::with_capacity(n);
        let mut lens: Vec<usize> = Vec::with_capacity(n);
        for p in &self.passages {
            let mut c = vec![0u32; terms.len()];
            let mut len = 0;
            for w in words(&p.text) {
                len += 1;
                if let Some(&i) = slot.get(w.as_str()) {
                    c[i] += 1;
                }
            }
            counts.push(c);
            lens.push(len);
        }
        let avg = (lens.iter().sum::<usize>() as f32 / n as f32).max(1.0);
        let df: Vec<usize> = (0..terms.len()).map(|t| counts.iter().filter(|c| c[t] > 0).count()).collect();
        // rare: in at most a twelfth of the passages (and in no more than twenty)
        let cap = (n / 12).clamp(2, RARE_MAX);
        let (k1, b) = (1.2f32, 0.75f32);
        let mut scored: Vec<(f32, usize)> = (0..n)
            .filter_map(|i| {
                let s: f32 = (0..terms.len())
                    .filter(|&t| counts[i][t] > 0 && df[t] <= cap)
                    .map(|t| {
                        let idf = (1.0 + (n - df[t]) as f32 / (df[t] as f32 + 0.5) + 0.5 / (df[t] as f32 + 0.5)).ln();
                        let f = counts[i][t] as f32;
                        idf * f * (k1 + 1.0) / (f + k1 * (1.0 - b + b * lens[i] as f32 / avg))
                    })
                    .sum();
                (s > 0.0).then_some((s, i))
            })
            .collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.into_iter().take(limit).map(|s| s.1).collect()
    }

    /// The best `limit` passages for a question vector whose score is above `threshold`, best first.
    pub fn search(&self, query: &[f32], limit: usize, threshold: f32) -> Vec<Hit> {
        let query = unit(query);
        let mut hits: Vec<Hit> = self
            .vectors
            .iter()
            .enumerate()
            .map(|(i, v)| Hit { passage: i, score: v.iter().zip(&query).map(|(a, b)| a * b).sum() })
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
    /// Passages embedded so far, and passages that need it in all (known once a file is split). While files are summarized: parts summarized so far, and parts that need it.
    pub done: Arc<AtomicUsize>,
    pub total: Arc<AtomicUsize>,
    pub cancel: Arc<AtomicBool>,
    /// The counts above are those of a summary, not of an embedding.
    pub summarizing: Arc<AtomicBool>,
}

/// Words that make a question a request for a summary of the files as a whole. Searching by meaning cannot read a whole file: for such a request every part of the files is summarized and the answer is written
/// from the summaries (`summarize`).
const SUMMARY_WORDS: &[&str] = &[
    "summar", "overview", "outline of", "tl;dr", "tldr", "the gist", "main point", "main idea", "key point", "takeaway", "what is this about", "what's this about", "what is it about", "what's it about", "document about", "paper about",
    "file about", "pdf about", "chapter about", "book about", "article about", "report about", "describe this document", "describe this file", "describe this paper", "explain this document", "explain this file", "explain this paper",
];

pub fn wants_summary(question: &str) -> bool {
    let q = question.to_lowercase();
    if q.contains("summar") {
        return true;
    }
    // "the main idea of section 3", "the key points on page 12": a request about a part of the file, which the passages that match answer better than the summaries of all of it
    if ["section ", "chapter ", "page ", "paragraph", "figure ", "table ", "slide ", "line ", "function", "class "].iter().any(|k| q.contains(k)) {
        return false;
    }
    SUMMARY_WORDS.iter().any(|k| q.contains(k))
}

/// The files a request names (by their name, or by their name without the extension when that is at least four characters); all of them when it names none.
pub fn files_named(question: &str, docs: &[(String, String)]) -> Vec<(String, String)> {
    let q = question.to_lowercase();
    let named: Vec<(String, String)> = docs
        .iter()
        .filter(|(name, _)| {
            let l = name.to_lowercase();
            let stem = l.rsplit_once('.').map_or(l.as_str(), |s| s.0);
            q.contains(&l) || (stem.chars().count() >= 4 && q.contains(stem))
        })
        .cloned()
        .collect();
    if named.is_empty() {
        docs.to_vec()
    } else {
        named
    }
}

/// The largest number of parts, in all the files together, that are summarized: a request for a summary of more is answered from the passages that match, as before.
pub const MAX_SUMMARY_PARTS: usize = 48;

/// How a request for a summary is carried out for some files.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SummaryPlan {
    /// Characters of one part of a file.
    pub part_chars: usize,
    /// Words the summary of a part may have: all the summaries together fill about half of the context.
    pub words: usize,
    /// Parts of all the files together.
    pub parts: usize,
}

/// The plan for summarizing `docs` (name, text) with a context of `ctx` tokens; None when the files are too short to need it (one part) or too long to be read in reasonable time.
pub fn summary_plan(docs: &[(String, String)], ctx: usize) -> Option<SummaryPlan> {
    // a part, the instructions and the summary must fit the context at about three characters to the token; a bigger part than 14,000 characters gains little and takes longer to read
    let part_chars = (ctx.saturating_sub(700) * 9 / 10 * 3).clamp(2_000, 14_000);
    // (a file that is plainly too long is not even cut into parts)
    if docs.iter().map(|d| d.1.chars().count()).sum::<usize>() > MAX_SUMMARY_PARTS * part_chars {
        return None;
    }
    let parts: usize = docs.iter().map(|d| split(&d.1, part_chars, 0).len()).sum();
    if !(2..=MAX_SUMMARY_PARTS).contains(&parts) {
        return None;
    }
    // the summaries are read together with the question: half of the context between them, 260 tokens (about 195 words) at most for one
    let tokens = ((ctx / 2).saturating_sub(300) / parts).clamp(50, 260);
    Some(SummaryPlan { part_chars, words: tokens * 3 / 4, parts })
}

/// What each part of a file was summarized to, by (fingerprint of the file's text, characters of a part, words of a summary): asking for the summary again costs nothing.
pub type SummaryCache = HashMap<(u64, usize, usize), Vec<String>>;

/// A summary that the length limit cut off ends in the middle of a sentence: keep the whole sentences when they are most of the text.
fn whole_sentences(text: &str) -> String {
    let t = text.trim();
    if t.ends_with(['.', '!', '?', ')', '"', '\'', '*', '`', ':', ';', '\u{2019}', '\u{201d}']) {
        return t.to_string();
    }
    let cut = t.char_indices().filter(|&(i, c)| matches!(c, '.' | '!' | '?') && t[i + c.len_utf8()..].starts_with(char::is_whitespace)).map(|(i, c)| i + c.len_utf8()).next_back();
    match cut {
        Some(end) if end * 2 >= t.len() => t[..end].to_string(),
        _ => t.to_string(),
    }
}

/// The request the model gets for one part of a file.
pub fn summary_prompt(file: &str, part: usize, parts: usize, text: &str, words: usize) -> String {
    format!(
        "Summarize this part of the file \"{file}\" (part {part} of {parts}) in at most {words} words. Keep the names, numbers, dates and terms that matter; leave out legal notices, copyright text and page headers. Reply with the summary only.\n\n{}",
        text.replace('\u{c}', "\n\n").trim()
    )
}

/// Summarize every part of the files (those the cache does not hold yet) with `ask`, which sends one request to the chat model and returns its reply; the summaries come back as citations, in the order of
/// the files and of their parts. `progress` counts the parts and can stop it between two of them.
pub fn summarize(cache: &Mutex<SummaryCache>, docs: &[(String, String)], plan: &SummaryPlan, progress: &Progress, ask: &mut dyn FnMut(&str) -> Result<String, String>) -> Result<Vec<Citation>, String> {
    let key = |text: &str| (fingerprint(text), plan.part_chars, plan.words);
    let held = |k: &(u64, usize, usize)| cache.lock().unwrap_or_else(|e| e.into_inner()).get(k).cloned();
    let split_docs: Vec<Vec<String>> = docs.iter().map(|d| split(&d.1, plan.part_chars, 0)).collect();
    let missing: usize = docs.iter().zip(&split_docs).filter(|(d, _)| held(&key(&d.1)).is_none()).map(|(_, p)| p.len()).sum();
    progress.done.store(0, Ordering::Relaxed);
    progress.total.store(missing, Ordering::Relaxed);
    let mut out = Vec::new();
    for ((name, text), parts) in docs.iter().zip(&split_docs) {
        let k = key(text);
        let summaries = match held(&k) {
            Some(s) => s,
            None => {
                let mut s: Vec<String> = Vec::with_capacity(parts.len());
                for (i, part) in parts.iter().enumerate() {
                    if progress.cancel.load(Ordering::Relaxed) {
                        return Err("stopped".to_string());
                    }
                    let reply = whole_sentences(&ask(&summary_prompt(name, i + 1, parts.len(), part, plan.words))?);
                    let reply = reply.as_str();
                    // a model that gave nothing (it spent its answer on reasoning, say): the start of the part stands in for the summary
                    s.push(if reply.is_empty() { part.chars().take(400).collect::<String>() } else { reply.to_string() });
                    progress.done.fetch_add(1, Ordering::Relaxed);
                }
                let mut c = cache.lock().unwrap_or_else(|e| e.into_inner());
                if c.len() >= 32 {
                    c.clear();
                }
                c.insert(k, s.clone());
                s
            }
        };
        let n = summaries.len() as u32;
        for (i, text) in summaries.into_iter().enumerate() {
            out.push(Citation { file: name.clone(), text, score: 1.0, weak: false, page: None, part: Some((i as u32 + 1, n)), keyword: false });
        }
    }
    Ok(out)
}

/// What the model reads for a request to summarize files that were read in parts: the summaries of the parts in front of the request.
pub fn summaries_prompt(parts: &[(String, (u32, u32), String)], question: &str) -> String {
    let mut s = String::from("The user attached files that are too long to read at once. Each was read in parts, one after the other, and every part was summarized. These are the summaries, in the order of the files and of their parts:\n\n");
    for (file, (i, n), text) in parts {
        s.push_str(&format!("File \"{file}\", part {i} of {n}: {}\n\n", text.trim()));
    }
    s.push_str("Use the summaries to respond to the user query. Only if the query asks for a specific detail that the summaries do not contain, say that they do not cover it.\n\nUser Query:\n\n");
    s.push_str(question);
    s
}

/// Passages per request to the embedding model. Stop is felt when the request in flight ends, and on the processor 16 passages of about 450 tokens take about 8 seconds (on a graphics card the batch
/// size hardly matters: 16 or 32 passages are one pass either way).
const BATCH: usize = 16;

/// What an embedding model needs from the search around it, known from its file name: the prefixes it was trained with, and the minimum cosine at which a passage counts as an answer. Models differ in
/// how they spread the cosine (measured on this project's own documentation cut into passages, 14 questions with a known answer and 6 about something else entirely; `related` is the best passage that
/// holds the answer, `unrelated` the best one for a question about bread or football):
///   nomic-embed-text (with its prefixes)  related 0.63 to 0.73, unrelated up to 0.64     minimum 0.60
///   bge-*-en-v1.5                         related 0.60 to 0.74, unrelated up to 0.57     minimum 0.58
///   all-MiniLM-L6-v2                      related 0.23 to 0.48, unrelated up to 0.16     minimum 0.20
///   e5-*-v2 (with its prefixes)           related 0.81 to 0.88, unrelated up to 0.81     minimum 0.82
///   gte-*                                 related 0.78 to 0.88, unrelated up to 0.77     minimum 0.78
/// (e5 and gte hardly separate the two: for them the order of the passages says more than the minimum.)
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Profile {
    /// the family, for the settings page
    pub family: &'static str,
    /// in front of every passage that is indexed
    pub doc_prefix: &'static str,
    /// in front of the question
    pub query_prefix: &'static str,
    pub threshold: f32,
}

/// The profile of the embedding model in this file (an unknown model: no prefixes and the minimum that suits nomic-embed-text, 0.6).
pub fn profile_for(model_file: &str) -> Profile {
    let name = model_file.to_lowercase();
    let e5 = name.starts_with("e5") || ["-e5", "_e5", " e5", "e5-", "e5_"].iter().any(|k| name.contains(k));
    if name.contains("nomic") {
        Profile { family: "nomic-embed-text", doc_prefix: "search_document: ", query_prefix: "search_query: ", threshold: 0.60 }
    } else if name.contains("bge") {
        Profile { family: "bge", doc_prefix: "", query_prefix: "", threshold: 0.58 }
    } else if name.contains("minilm") {
        Profile { family: "MiniLM", doc_prefix: "", query_prefix: "", threshold: 0.20 }
    } else if e5 {
        Profile { family: "e5", doc_prefix: "passage: ", query_prefix: "query: ", threshold: 0.82 }
    } else if name.contains("gte") {
        Profile { family: "gte", doc_prefix: "", query_prefix: "", threshold: 0.78 }
    } else {
        Profile { family: "this model", doc_prefix: "", query_prefix: "", threshold: 0.60 }
    }
}

/// The example lines of the embedding playground with the task prefixes of this model: the first line is the question, the others are the passages to rank.
pub fn sample_for(model: &str) -> String {
    let p = profile_for(model);
    let (q, d) = (p.query_prefix, p.doc_prefix);
    format!("{q}What is the capital of France?
{d}Paris is the capital and largest city of France.
{d}The Eiffel Tower was finished in 1889 and stands in Paris.
{d}The mitochondria is the powerhouse of the cell.
{d}Berlin has been the capital of Germany since 1990.")
}

/// The prefixes an embedding model wants in front of the passages it indexes and the questions it is asked (nomic-embed-text was trained with `search_document: ` and `search_query: `, e5 with
/// `passage: ` and `query: `; the others use none).
pub fn prefixes_for(model_file: &str) -> (&'static str, &'static str) {
    let p = profile_for(model_file);
    (p.doc_prefix, p.query_prefix)
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
    index.retain_files(&docs.iter().map(|d| d.0.as_str()).collect::<Vec<_>>());
    // what has to be embedded: the passages of the files the index has not seen
    let mut todo: Vec<(u64, &str, Vec<(String, Option<u32>)>)> = Vec::new();
    for (name, text) in docs {
        let fp = fingerprint(text);
        if index.has(fp) || todo.iter().any(|t| t.0 == fp) {
            continue;
        }
        let parts = split_pages(text, CHUNK_CHARS, OVERLAP_CHARS);
        progress.total.fetch_add(parts.len(), Ordering::Relaxed);
        todo.push((fp, name, parts));
    }
    for (fp, name, parts) in todo {
        let mut vectors: Vec<Vec<f32>> = Vec::with_capacity(parts.len());
        for batch in parts.chunks(BATCH) {
            if stop() {
                return Err("stopped".to_string());
            }
            let inputs: Vec<String> = batch.iter().map(|p| format!("{}{}", prefixes.0, p.0)).collect();
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
    let q = unit(q.first().ok_or("the embedding model returned no vector for the question")?);
    let q = &q;
    let mut hits = index.search(q, limit, threshold);
    // the passages that have a word of the question that few passages have (a name, an identifier) count whatever their vectors say: one place at most when the vectors already fill the answer
    let keyword = index.lexical(question, if hits.len() >= limit { 1 } else { 2 });
    let floor = weak_floor(threshold);
    let mut weak = hits.is_empty() && keyword.is_empty() && threshold > floor;
    if weak {
        hits = index.search(q, limit.min(2), floor);
    }
    if !keyword.is_empty() {
        let mut merged: Vec<Hit> = keyword.iter().map(|&i| Hit { passage: i, score: index.vectors[i].iter().zip(q).map(|(a, b)| a * b).sum() }).collect();
        for h in hits {
            if !merged.iter().any(|m| m.passage == h.passage) {
                merged.push(h);
            }
        }
        merged.truncate(limit.max(1));
        hits = merged;
    }
    if is_overview(question) {
        // the opening of each file (two files at most) in front of the best matches, which for such a question are mostly weak ones: the matches only count when they are strong
        if weak {
            hits.clear();
            weak = false;
        }
        let mut merged: Vec<Hit> = index.openings(2).into_iter().map(|i| Hit { passage: i, score: index.vectors[i].iter().zip(q).map(|(a, b)| a * b).sum() }).collect();
        for h in hits {
            if !merged.iter().any(|m| m.passage == h.passage) {
                merged.push(h);
            }
        }
        hits = merged;
    }
    let citations = hits.iter().map(|h| Citation { file: index.passages[h.passage].file.clone(), text: index.passages[h.passage].text.clone(), score: h.score, weak, page: index.passages[h.passage].page, part: None, keyword: keyword.contains(&h.passage) }).collect();
    Ok((index, citations))
}

/// What the model reads for a question with passages found in the user's files (LM Studio's wording; the file name is added so that the model can say where something comes from).
pub fn citations_prompt(passages: &[(String, Option<u32>, String)], weak: bool, question: &str) -> String {
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
    for (i, (file, page, text)) in passages.iter().enumerate() {
        let from = match page {
            Some(p) => format!("{file}, page {p}"),
            None => file.clone(),
        };
        s.push_str(&format!("Citation {} (from {from}): \"{}\"\n\n", i + 1, text.trim()));
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
    fn a_text_with_page_breaks_is_split_page_by_page() {
        let text = "The ship docks at six.\u{c}Cargo: tea and rope.\u{c}\u{c}Last page.";
        let parts = split_pages(text, 100, 10);
        assert_eq!(parts, vec![("The ship docks at six.".to_string(), Some(1)), ("Cargo: tea and rope.".to_string(), Some(2)), ("Last page.".to_string(), Some(4))]);
        // no page breaks: one run, no page numbers
        assert!(split_pages("plain text", 100, 10).iter().all(|p| p.1.is_none()));
        // long pages give several passages of the same page
        let long = "word ".repeat(100) + "\u{c}short";
        let p = split_pages(&long, 100, 20);
        assert!(p.iter().filter(|x| x.1 == Some(1)).count() >= 4 && p.last().unwrap() == &("short".to_string(), Some(2)));
    }

    #[test]
    fn a_file_that_left_the_chat_is_not_searched_any_more() {
        let docs = vec![("alpha.txt".to_string(), "aaaa aaaa".to_string()), ("gamma.txt".to_string(), "cccc cccc".to_string())];
        let (index, _) = search(DocIndex::default(), &docs, "cc", 3, 0.0, ("", ""), &Progress::default(), &mut |t| toy(t)).unwrap();
        assert_eq!(index.passages.len(), 2);
        // the message with gamma.txt was deleted: only alpha.txt is asked about, and a later gamma.txt would be read again
        let (index, cites) = search(index, &docs[..1], "cc", 3, 0.0, ("", ""), &Progress::default(), &mut |t| toy(t)).unwrap();
        assert!(index.passages.iter().all(|p| p.file == "alpha.txt") && cites.iter().all(|c| c.file == "alpha.txt"), "{cites:?}");
        assert!(!index.has(fingerprint("cccc cccc")) && index.has(fingerprint("aaaa aaaa")));
    }

    #[test]
    fn vectors_that_are_not_of_length_one_still_give_cosines() {
        let mut idx = DocIndex::default();
        idx.add(1, "a.txt", vec![("x".into(), None), ("y".into(), None)], vec![vec![0.0, 5.0], vec![3.0, 0.0]]);
        let hits = idx.search(&[0.0, 2.0], 2, -1.0);
        assert_eq!(hits[0].passage, 0);
        assert!((hits[0].score - 1.0).abs() < 1e-6 && hits[1].score.abs() < 1e-6, "{hits:?}");
    }

    #[test]
    fn the_index_ranks_by_cosine_and_applies_the_threshold() {
        let mut idx = DocIndex::default();
        assert!(!idx.has(7));
        idx.add(7, "a.txt", vec![("north".into(), None), ("east".into(), None), ("northeast".into(), Some(2))], vec![vec![0.0, 1.0], vec![1.0, 0.0], vec![0.7071, 0.7071]]);
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
        let p = citations_prompt(&[("notes.txt".into(), None, "The ship docks at six.".into()), ("log.pdf".into(), Some(3), "Cargo: tea.".into())], false, "When does it dock?");
        assert!(p.starts_with("The following citations were found in the files provided by the user:"));
        assert!(p.contains("Citation 1 (from notes.txt): \"The ship docks at six.\""));
        assert!(p.contains("Citation 2 (from log.pdf, page 3): \"Cargo: tea.\""));
        assert!(p.ends_with("User Query:\n\nWhen does it dock?"));
        let none = citations_prompt(&[], false, "Anything?");
        assert!(none.contains("No passage") && none.ends_with("Anything?"));
        let weak = citations_prompt(&[("a.txt".into(), None, "Maybe.".into())], true, "Summary?");
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
    fn a_question_about_the_whole_document_gets_its_opening() {
        assert!(is_overview("What does the abstract say?") && is_overview("Summarize this file") && is_overview("what is it about?"));
        assert!(!is_overview("When does the ship dock?"));
        // two files of two pages each: the question matches the c passage of the second file best, and the openings of both files come first
        let docs = vec![("alpha.txt".to_string(), "aaaa aaaa\u{c}bbbb bbbb".to_string()), ("gamma.txt".to_string(), "cccc cccc\u{c}cccc ccab".to_string())];
        let (index, cites) = search(DocIndex::default(), &docs, "Summarize the cccc part", 1, 0.9, ("", ""), &Progress::default(), &mut |t| toy(t)).unwrap();
        let named: Vec<(String, Option<u32>)> = cites.iter().map(|c| (c.file.clone(), c.page)).collect();
        assert_eq!(named[0], ("alpha.txt".to_string(), Some(1)), "{named:?}");
        assert_eq!(named[1], ("gamma.txt".to_string(), Some(1)), "{named:?}");
        assert!(cites.iter().all(|c| !c.weak));
        assert_eq!(index.openings(1), vec![0]);
        // an ordinary question gets no openings
        let (_, plain) = search(index, &docs, "cccc", 1, 0.9, ("", ""), &Progress::default(), &mut |t| toy(t)).unwrap();
        assert!(plain.iter().all(|c| c.file == "gamma.txt"), "{plain:?}");
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
        assert_eq!(prefixes_for("e5-small-v2.Q8_0.gguf"), ("passage: ", "query: "));
        assert_eq!(prefixes_for("multilingual-e5-small.Q8_0.gguf"), ("passage: ", "query: "));
        assert_eq!(prefixes_for("all-MiniLM-L6-v2.Q8_0.gguf"), ("", ""));
    }

    #[test]
    fn the_playground_sample_carries_the_prefixes_of_the_model() {
        assert_eq!(sample_for("nomic-embed-text-v1.5"), crate::app::EMBED_SAMPLE);
        let bge = sample_for("bge-small-en-v1.5-q8_0");
        assert!(bge.starts_with("What is the capital of France?
Paris is the capital") && !bge.contains("search_"));
        let e5 = sample_for("e5-small-v2");
        assert!(e5.starts_with("query: What is the capital of France?
passage: Paris is") && e5.lines().count() == 5);
    }

    #[test]
    fn every_embedding_family_has_its_own_minimum_score() {
        let t = |f: &str| profile_for(f).threshold;
        assert_eq!(t("nomic-embed-text-v1.5.Q4_K_M.gguf"), 0.60);
        assert_eq!(t("bge-small-en-v1.5-q8_0.gguf"), 0.58);
        assert_eq!(t("all-MiniLM-L6-v2.Q8_0.gguf"), 0.20);
        assert_eq!(t("e5-small-v2.Q8_0.gguf"), 0.82);
        assert_eq!(t("gte-small.Q8_0.gguf"), 0.78);
        // a model we do not know keeps the figure that was the only one so far, and no prefixes
        assert_eq!(profile_for("some-new-model.gguf"), Profile { family: "this model", doc_prefix: "", query_prefix: "", threshold: 0.60 });
        // names that merely contain the letters
        assert_eq!(profile_for("qwen3-embedding-0.6b.gguf").doc_prefix, "");
        assert_eq!(profile_for("lne5ger.gguf").family, "this model");
        // the weak matches start 0.2 below the minimum
        assert!((weak_floor(0.6) - 0.4).abs() < 1e-6 && (weak_floor(0.2) - 0.0).abs() < 1e-6 && weak_floor(0.1) == 0.0);
    }

    #[test]
    fn fingerprints_differ_for_different_texts() {
        assert_eq!(fingerprint("abc"), fingerprint("abc"));
        assert_ne!(fingerprint("abc"), fingerprint("abd"));
        assert_ne!(fingerprint(""), fingerprint("a"));
    }

    #[test]
    fn only_requests_for_the_whole_file_want_a_summary() {
        for yes in ["Summarize this PDF", "give me a summary", "What is this document about?", "what's it about", "tl;dr please", "Give an overview of the file", "list the key points", "explain this paper", "Summarize section 3"] {
            assert!(wants_summary(yes), "{yes}");
        }
        for no in ["What does the author say about transformers?", "When does the ship dock?", "what is this variable", "describe this function", "Who wrote it?", "what is the capital of France", "What is the main idea of section 3?", "the key points on page 12"] {
            assert!(!wants_summary(no), "{no}");
        }
    }

    fn long_docs() -> Vec<(String, String)> {
        // 40 paragraphs of 1,000 characters: three parts of 14,000 characters
        let text: String = (0..40).map(|i| format!("Paragraph {i}: {}\n\n", "word ".repeat(195))).collect();
        vec![("report.txt".to_string(), text)]
    }

    #[test]
    fn the_plan_fits_the_parts_and_their_summaries_to_the_context() {
        let docs = long_docs();
        let big = summary_plan(&docs, 8192).unwrap();
        assert_eq!((big.part_chars, big.words), (14_000, 195));
        assert!((3..=4).contains(&big.parts), "{big:?}");
        // a small context makes smaller parts and shorter summaries
        let small = summary_plan(&docs, 2048).unwrap();
        assert!(small.part_chars < 4_000 && small.parts > big.parts && small.words < big.words, "{small:?}");
        // everything has to stay within half the context
        assert!(small.parts * (small.words * 4 / 3) <= 2048 / 2 + small.parts * 20, "{small:?}");
        // a file of one part needs no summary, a file of hundreds of parts is not read part by part
        assert_eq!(summary_plan(&[("a".to_string(), "short".to_string())], 8192), None);
        assert_eq!(summary_plan(&[("a".to_string(), "word ".repeat(400_000))], 8192), None);
    }

    #[test]
    fn every_part_is_summarized_once_and_remembered() {
        let docs = long_docs();
        let plan = summary_plan(&docs, 8192).unwrap();
        let cache = Mutex::new(SummaryCache::new());
        let asked = std::cell::RefCell::new(Vec::<String>::new());
        let mut ask = |p: &str| {
            asked.borrow_mut().push(p.to_string());
            Ok(format!(" summary number {} ", asked.borrow().len()))
        };
        let progress = Progress::default();
        let cites = summarize(&cache, &docs, &plan, &progress, &mut ask).unwrap();
        assert_eq!(cites.len(), plan.parts);
        assert_eq!((progress.done.load(Ordering::Relaxed), progress.total.load(Ordering::Relaxed)), (plan.parts, plan.parts));
        assert_eq!(cites[0].text, "summary number 1");
        assert_eq!((cites[0].file.as_str(), cites[0].part, cites[1].part), ("report.txt", Some((1, plan.parts as u32)), Some((2, plan.parts as u32))));
        // each request names its part and carries the text of it, the first part starts the file
        let first = asked.borrow()[0].clone();
        assert!(first.starts_with("Summarize this part of the file \"report.txt\" (part 1 of ") && first.contains("at most 195 words") && first.contains("Paragraph 0:"), "{}", &first[..200]);
        assert!(asked.borrow()[1].contains("(part 2 of "));
        // the second request for the same file asks nothing
        let before = asked.borrow().len();
        let again = summarize(&cache, &docs, &plan, &Progress::default(), &mut ask).unwrap();
        assert_eq!(asked.borrow().len(), before);
        assert_eq!(again, cites);
        // another size of part is another summary
        let other = SummaryPlan { part_chars: 7_000, words: 100, parts: 8 };
        summarize(&cache, &docs, &other, &Progress::default(), &mut ask).unwrap();
        assert!(asked.borrow().len() > before);
    }

    #[test]
    fn a_stopped_or_failing_summary_is_an_error_and_an_empty_reply_gets_the_start_of_the_part() {
        let docs = long_docs();
        let plan = summary_plan(&docs, 8192).unwrap();
        let stop = Progress::default();
        stop.cancel.store(true, Ordering::Relaxed);
        assert_eq!(summarize(&Mutex::new(SummaryCache::new()), &docs, &plan, &stop, &mut |_| Ok("x".to_string())).unwrap_err(), "stopped");
        assert_eq!(summarize(&Mutex::new(SummaryCache::new()), &docs, &plan, &Progress::default(), &mut |_| Err("server down".to_string())).unwrap_err(), "server down");
        let cites = summarize(&Mutex::new(SummaryCache::new()), &docs, &plan, &Progress::default(), &mut |_| Ok("   ".to_string())).unwrap();
        assert!(cites[0].text.starts_with("Paragraph 0:") && cites[0].text.chars().count() <= 400, "{:?}", cites[0].text);
        // a failure leaves nothing half-done in the cache
        let cache = Mutex::new(SummaryCache::new());
        let mut calls = 0;
        let _ = summarize(&cache, &docs, &plan, &Progress::default(), &mut |_| {
            calls += 1;
            if calls == 2 { Err("broke".to_string()) } else { Ok("ok".to_string()) }
        });
        assert!(cache.lock().unwrap().is_empty());
    }

    /// Against a running ForgeInfer server: FORGE_TEST_DOCS=file;file FORGE_TEST_SERVER=8080 FORGE_TEST_QUESTION="Summarize it" cargo test --lib live_summary -- --ignored --nocapture
    #[test]
    #[ignore]
    fn live_summary() {
        use crate::chats::Role;
        use crate::client::{self, ChatParams, Endpoint};
        use crate::settings::{GenParams, ThinkingMode};
        let port: u16 = std::env::var("FORGE_TEST_SERVER").ok().and_then(|v| v.parse().ok()).unwrap_or(8080);
        let ep = Endpoint::local(port, "");
        let ctx = client::loaded_context(&ep).unwrap_or(8192) as usize;
        let model = client::list_models(&ep).unwrap().remove(0);
        let docs: Vec<(String, String)> = std::env::var("FORGE_TEST_DOCS")
            .unwrap_or_default()
            .split(';')
            .filter(|p| !p.is_empty())
            .map(|p| (std::path::Path::new(p).file_name().unwrap().to_string_lossy().to_string(), crate::office::read_document(std::path::Path::new(p), 24 << 20).unwrap()))
            .collect();
        let plan = summary_plan(&docs, ctx).expect("these files need no summary or are too long for one");
        println!("model {model}, context {ctx}: {plan:?}");
        let progress = Progress::default();
        let cache = Mutex::new(SummaryCache::new());
        let gen = GenParams { temperature: 0.2, max_tokens: (plan.words * 2 + 40) as u32, thinking: ThinkingMode::Off, ..GenParams::default() };
        let mut ask = |prompt: &str| client::complete(&ep, &ChatParams { model: model.clone(), messages: vec![(Role::User, prompt.to_string())], gen: gen.clone() }, &progress.cancel);
        let t0 = std::time::Instant::now();
        let cites = summarize(&cache, &docs, &plan, &progress, &mut ask).unwrap();
        println!("{} parts summarized in {:.1} s", cites.len(), t0.elapsed().as_secs_f32());
        for c in &cites {
            println!("--- {} part {:?} ({} characters)\n{}", c.file, c.part, c.text.chars().count(), c.text);
        }
        let question = std::env::var("FORGE_TEST_QUESTION").unwrap_or_else(|_| "Summarize the file.".to_string());
        let parts: Vec<(String, (u32, u32), String)> = cites.iter().filter_map(|c| c.part.map(|p| (c.file.clone(), p, c.text.clone()))).collect();
        let prompt = summaries_prompt(&parts, &question);
        println!("=== the answer prompt is {} characters", prompt.chars().count());
        let t1 = std::time::Instant::now();
        let params = ChatParams { model, messages: vec![(Role::User, prompt)], gen: GenParams { thinking: ThinkingMode::Off, ..GenParams::default() } };
        let answer = client::complete(&ep, &params, &AtomicBool::new(false)).unwrap();
        println!("=== answer in {:.1} s:\n{answer}", t1.elapsed().as_secs_f32());
    }

    /// What the keyword search finds in real files: FORGE_TEST_DOCS=file;file FORGE_TEST_QUESTIONS="q one|q two" cargo test --lib live_keywords -- --ignored --nocapture
    #[test]
    #[ignore]
    fn live_keywords() {
        let mut idx = DocIndex::default();
        for p in std::env::var("FORGE_TEST_DOCS").unwrap_or_default().split(';').filter(|p| !p.is_empty()) {
            let path = std::path::Path::new(p);
            let text = crate::office::read_document(path, 24 << 20).unwrap();
            let parts = split_pages(&text, CHUNK_CHARS, OVERLAP_CHARS);
            let vectors = vec![vec![1.0f32]; parts.len()];
            idx.add(fingerprint(&text), &path.file_name().unwrap().to_string_lossy(), parts, vectors);
        }
        println!("{} passages", idx.passages.len());
        for q in std::env::var("FORGE_TEST_QUESTIONS").unwrap_or_default().split('|').filter(|q| !q.is_empty()) {
            let hits = idx.lexical(q, 2);
            println!("\n? {q}\n   keyword hits: {}", if hits.is_empty() { "none".to_string() } else { hits.iter().map(|&i| format!("#{i} (page {:?})", idx.passages[i].page)).collect::<Vec<_>>().join(", ") });
            for &i in &hits {
                let t: String = idx.passages[i].text.split_whitespace().collect::<Vec<_>>().join(" ");
                println!("   #{i}: {}", t.chars().take(160).collect::<String>());
            }
        }
    }

    fn ledger() -> DocIndex {
        let mut idx = DocIndex::default();
        let passages: Vec<(String, Option<u32>)> = (0..40)
            .map(|i| match i {
                17 => ("The harbour office paid invoice INV-7731-QX to Marguerite Okafor of Delta Freight.".to_string(), None),
                30 => ("Marguerite Okafor signed for the rope.".to_string(), None),
                _ => (format!("The harbour office counted crates number {i} and filed the copy."), None),
            })
            .collect();
        idx.add(1, "ledger.txt", passages, vec![vec![1.0, 0.0]; 40]);
        idx
    }

    #[test]
    fn the_names_of_a_question_are_its_capitalized_words_acronyms_and_codes() {
        assert_eq!(names_in("What did Marguerite Okafor pay for?"), ["marguerite", "okafor"]);
        assert_eq!(names_in("What happened with invoice INV-7731-QX?"), ["7731", "inv", "qx"]);
        assert_eq!(names_in("Where is route R-58 and PO 440182?"), ["440182", "58", "po"]);
        assert_eq!(names_in("how does the NF4 type work in QLoRA?"), ["nf4", "qlora"]);
        assert_eq!(names_in("I think Fleiss kappa matters"), ["fleiss"]);
        // everyday words, however rare in a file, and the capital of the first word are not names
        assert!(names_in("What is the main limitation of the authors' work?").is_empty());
        assert!(names_in("Summarize the paper").is_empty());
        assert!(names_in("").is_empty());
    }

    #[test]
    fn a_rare_word_of_the_question_finds_its_passage() {
        let idx = ledger();
        assert_eq!(idx.lexical("What happened with invoice INV-7731-QX?", 2), vec![17]);
        // a name that two passages have: the one that has more of the question comes first
        assert_eq!(idx.lexical("What did Marguerite Okafor pay to Delta Freight?", 2), vec![17, 30]);
        // words that nearly every passage has, little words, and a word that no passage has: nothing
        assert!(idx.lexical("What did the harbour office do with the crates?", 3).is_empty());
        assert!(idx.lexical("Who is Zebulon?", 3).is_empty());
        assert!(idx.lexical("What happened with invoice INV-7731-QX?", 0).is_empty());
        // in a short file every word is rare, so none counts
        let mut small = DocIndex::default();
        small.add(2, "s.txt", vec![("alpha beta".into(), None), ("gamma".into(), None)], vec![vec![1.0], vec![1.0]]);
        assert!(small.lexical("alpha", 2).is_empty());
    }

    #[test]
    fn a_word_in_the_question_finds_the_passage_even_when_the_vectors_are_far_away() {
        let text: String = (0..20)
            .map(|i| if i == 7 { format!("{} zorblax77 {}\n\n", "cccc ".repeat(120), "cccc ".repeat(120)) } else { format!("{}\n\n", "cccc ".repeat(240)) })
            .collect();
        let docs = vec![("log.txt".to_string(), text)];
        let (index, cites) = search(DocIndex::default(), &docs, "What is zorblax77?", 3, 0.6, ("", ""), &Progress::default(), &mut |t| toy(t)).unwrap();
        assert!(index.passages.len() >= 8);
        assert!(cites.len() == 1 && cites[0].text.contains("zorblax77") && cites[0].keyword && !cites[0].weak && cites[0].score < 0.1, "{cites:?}");
        // a question without such a word and without close vectors still finds nothing
        let (_, none) = search(index, &docs, "Who signed it?", 3, 0.6, ("", ""), &Progress::default(), &mut |t| toy(t)).unwrap();
        assert!(none.is_empty(), "{none:?}");
    }

    #[test]
    fn a_keyword_match_takes_one_place_when_the_vectors_have_filled_the_answer() {
        // twenty passages that the vectors like, one that has a rare word
        let mut text: String = (0..20).map(|_| format!("{}\n\n", "aaaa ".repeat(240))).collect();
        text.push_str(&format!("{} zorblax77\n\n", "cccc ".repeat(200)));
        let docs = vec![("log.txt".to_string(), text)];
        let (_, cites) = search(DocIndex::default(), &docs, "aaa zorblax77", 3, 0.5, ("", ""), &Progress::default(), &mut |t| toy(t)).unwrap();
        assert_eq!(cites.len(), 3, "{cites:?}");
        assert!(cites[0].text.contains("zorblax77") && cites[0].keyword);
        assert!(cites[1..].iter().all(|c| !c.keyword && c.score > 0.5), "{cites:?}");
    }

    #[test]
    fn a_request_that_names_a_file_is_about_that_file() {
        let docs = vec![("Report 2025.pdf".to_string(), "r".to_string()), ("slides.pptx".to_string(), "s".to_string()), ("a.txt".to_string(), "t".to_string())];
        let names = |q: &str| files_named(q, &docs).into_iter().map(|d| d.0).collect::<Vec<_>>();
        assert_eq!(names("Summarize slides.pptx please"), ["slides.pptx"]);
        assert_eq!(names("summarize the report 2025"), ["Report 2025.pdf"]);
        assert_eq!(names("Summarize the SLIDES"), ["slides.pptx"]);
        // no name, or a name too short to mean anything ("a"): all the files
        assert_eq!(names("Summarize everything"), ["Report 2025.pdf", "slides.pptx", "a.txt"]);
        assert_eq!(names("summarize a story"), ["Report 2025.pdf", "slides.pptx", "a.txt"]);
    }

    #[test]
    fn a_summary_cut_off_by_the_length_limit_keeps_its_whole_sentences() {
        assert_eq!(whole_sentences("  One. Two is longer than one. Three is cut o"), "One. Two is longer than one.");
        // complete, or too little of it is whole sentences: left alone
        assert_eq!(whole_sentences("It ends here."), "It ends here.");
        assert_eq!(whole_sentences("A list:\n- one\n- two (cut"), "A list:\n- one\n- two (cut");
        assert_eq!(whole_sentences("Ok. A very long second sentence that never ends and never stops going on and on"), "Ok. A very long second sentence that never ends and never stops going on and on");
        assert_eq!(whole_sentences("Version 3.5 was cut at the number 4"), "Version 3.5 was cut at the number 4");
        assert_eq!(whole_sentences(""), "");
    }

    #[test]
    fn the_summaries_are_written_into_the_prompt_in_order() {
        let p = summaries_prompt(&[("a.pdf".to_string(), (1, 2), "First half.".to_string()), ("a.pdf".to_string(), (2, 2), " Second half. ".to_string())], "Summarize it");
        assert!(p.contains("File \"a.pdf\", part 1 of 2: First half.\n\nFile \"a.pdf\", part 2 of 2: Second half.\n\n"), "{p}");
        assert!(p.starts_with("The user attached files that are too long to read at once.") && p.ends_with("User Query:\n\nSummarize it"));
    }
}
