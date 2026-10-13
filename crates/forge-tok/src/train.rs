//! Byte-level BPE training (tools/ref_bpe_train.py is the executable specification): count the pre-tokens of the corpus, keep an inverted index pair -> words, repeatedly merge the most frequent
//! pair (ties: the smaller pair) in only the words that contain it, and update only their neighbouring pair counts. Deterministic. The output is a Hugging Face compatible tokenizer.json
//! (ByteLevel BPE; Digits(individual) + GPT-2 pre-tokenizer), so the encoder in `tokenizer.rs` and other tools can load it.
//!
//! Corpus format: documents separated by a NUL byte (see tools/data_fetch.py).

use crate::pretok::PreTokenizer;
use crate::tokenizer::bytes_to_unicode;
use forge_core::{Error, Json, Result};
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};
use std::io::Read;
use std::path::Path;

pub struct TrainConfig {
    /// Total vocabulary size including special tokens and the 256 byte tokens.
    pub vocab_size: usize,
    /// Added at ids 0.. (they are not part of the BPE alphabet).
    pub special_tokens: Vec<String>,
    pub threads: usize,
    /// Print progress to stderr every this many merges (0 = silent).
    pub log_every: usize,
}

pub struct Trained {
    pub special_tokens: Vec<String>,
    /// merge i joins `merges[i].0` and `merges[i].1` (symbol ids: 0..256 are bytes, 256 + i is merge i)
    pub merges: Vec<(u32, u32)>,
    /// bytes of every symbol (0..256 + merges)
    pub symbol_bytes: Vec<Vec<u8>>,
    pub distinct_words: usize,
    pub total_words: u64,
}

fn count_chunk(text: &[u8], pre: PreTokenizer) -> HashMap<Vec<u8>, u64> {
    let mut counts: HashMap<Vec<u8>, u64> = HashMap::new();
    for doc in text.split(|&b| b == 0) {
        if doc.is_empty() {
            continue;
        }
        let s = String::from_utf8_lossy(doc);
        let chars: Vec<char> = s.chars().collect();
        let mut offs = Vec::with_capacity(chars.len() + 1);
        let mut o = 0;
        for c in &chars {
            offs.push(o);
            o += c.len_utf8();
        }
        offs.push(o);
        let sb = s.as_bytes();
        for (a, b) in pre.split(&chars) {
            let piece = &sb[offs[a]..offs[b]];
            if let Some(c) = counts.get_mut(piece) {
                *c += 1;
            } else {
                counts.insert(piece.to_vec(), 1);
            }
        }
    }
    counts
}

/// Count the distinct pre-tokens of a set of corpus files (streamed in blocks of about 64 MB cut at document boundaries, counted on `threads` threads).
pub fn count_words(
    files: &[impl AsRef<Path>],
    pre: PreTokenizer,
    threads: usize,
) -> Result<HashMap<Vec<u8>, u64>> {
    let mut total: HashMap<Vec<u8>, u64> = HashMap::new();
    let block = 64usize << 20;
    for path in files {
        let mut f = std::fs::File::open(path.as_ref())
            .map_err(|e| Error::msg(format!("{}: {e}", path.as_ref().display())))?;
        let mut carry: Vec<u8> = Vec::new();
        loop {
            let mut buf = std::mem::take(&mut carry);
            let have = buf.len();
            buf.resize(have + block, 0);
            let mut got = 0;
            while got < block {
                let n = f.read(&mut buf[have + got..])?;
                if n == 0 {
                    break;
                }
                got += n;
            }
            buf.truncate(have + got);
            if got == 0 {
                if !buf.is_empty() {
                    merge_counts(&mut total, count_chunk(&buf, pre));
                }
                break;
            }
            // cut at the last document boundary; the tail is the start of the next block
            let cut = buf
                .iter()
                .rposition(|&b| b == 0)
                .map(|p| p + 1)
                .unwrap_or(buf.len());
            carry = buf.split_off(cut);
            // split the block into `threads` pieces at document boundaries
            let mut pieces: Vec<&[u8]> = Vec::new();
            let mut start = 0;
            let target = buf.len() / threads.max(1) + 1;
            while start < buf.len() {
                let mut end = (start + target).min(buf.len());
                while end < buf.len() && buf[end - 1] != 0 {
                    end += 1;
                }
                pieces.push(&buf[start..end]);
                start = end;
            }
            let results: Vec<HashMap<Vec<u8>, u64>> = std::thread::scope(|s| {
                let hs: Vec<_> = pieces
                    .iter()
                    .map(|p| s.spawn(move || count_chunk(p, pre)))
                    .collect();
                hs.into_iter().map(|h| h.join().unwrap()).collect()
            });
            for r in results {
                merge_counts(&mut total, r);
            }
        }
    }
    Ok(total)
}

fn merge_counts(into: &mut HashMap<Vec<u8>, u64>, from: HashMap<Vec<u8>, u64>) {
    for (k, v) in from {
        *into.entry(k).or_insert(0) += v;
    }
}

#[inline]
fn pk(a: u32, b: u32) -> u64 {
    ((a as u64) << 32) | b as u64
}

pub fn train_from_counts(counts: HashMap<Vec<u8>, u64>, cfg: &TrainConfig) -> Result<Trained> {
    let n_merges_wanted = cfg
        .vocab_size
        .checked_sub(256 + cfg.special_tokens.len())
        .ok_or_else(|| {
            Error::msg("vocab_size is smaller than the byte alphabet plus special tokens")
        })?;
    let distinct_words = counts.len();
    let total_words: u64 = counts.values().sum();
    // deterministic order: sort the words (HashMap iteration order is random)
    let mut entries: Vec<(Vec<u8>, u64)> = counts.into_iter().collect();
    entries.sort_unstable();
    let mut words: Vec<Vec<u32>> = Vec::with_capacity(entries.len());
    let mut freq: Vec<u64> = Vec::with_capacity(entries.len());
    for (w, f) in entries {
        words.push(w.iter().map(|&b| b as u32).collect());
        freq.push(f);
    }
    let mut pair_count: HashMap<u64, i64> = HashMap::new();
    let mut where_: HashMap<u64, Vec<u32>> = HashMap::new();
    for (wi, w) in words.iter().enumerate() {
        for p in w.windows(2) {
            let k = pk(p[0], p[1]);
            *pair_count.entry(k).or_insert(0) += freq[wi] as i64;
            let v = where_.entry(k).or_default();
            if v.last() != Some(&(wi as u32)) {
                v.push(wi as u32);
            }
        }
    }
    let mut heap: BinaryHeap<(i64, Reverse<u64>)> =
        pair_count.iter().map(|(&k, &c)| (c, Reverse(k))).collect();
    let mut symbol_bytes: Vec<Vec<u8>> = (0..256u32).map(|b| vec![b as u8]).collect();
    // a symbol is identified by its bytes (two different pairs can produce the same string; the vocabulary holds it once, as in Hugging Face's trainer)
    let mut by_bytes: HashMap<Vec<u8>, u32> = symbol_bytes
        .iter()
        .enumerate()
        .map(|(i, b)| (b.clone(), i as u32))
        .collect();
    let mut merges: Vec<(u32, u32)> = Vec::new();
    let t0 = std::time::Instant::now();
    while symbol_bytes.len() < 256 + n_merges_wanted {
        let Some((count, Reverse(key))) = heap.pop() else {
            break;
        };
        let cur = pair_count.get(&key).copied().unwrap_or(0);
        if cur != count {
            if cur > 0 {
                heap.push((cur, Reverse(key)));
            }
            continue;
        }
        if count <= 0 {
            break;
        }
        let (a, b) = ((key >> 32) as u32, key as u32);
        let mut sym = symbol_bytes[a as usize].clone();
        sym.extend_from_slice(&symbol_bytes[b as usize]);
        let new = match by_bytes.get(&sym) {
            Some(&id) => id,
            None => {
                let id = symbol_bytes.len() as u32;
                by_bytes.insert(sym.clone(), id);
                symbol_bytes.push(sym);
                id
            }
        };
        merges.push((a, b));
        let mut affected = where_.remove(&key).unwrap_or_default();
        affected.sort_unstable();
        affected.dedup();
        let mut touched: Vec<u64> = Vec::new();
        for wi in affected {
            let w = &words[wi as usize];
            let f = freq[wi as usize] as i64;
            // the pair may no longer be present in this word (stale index entry)
            if !w.windows(2).any(|p| p[0] == a && p[1] == b) {
                continue;
            }
            for p in w.windows(2) {
                let k = pk(p[0], p[1]);
                *pair_count.get_mut(&k).unwrap() -= f;
                touched.push(k);
            }
            let mut out: Vec<u32> = Vec::with_capacity(w.len());
            let mut i = 0;
            while i < w.len() {
                if i + 1 < w.len() && w[i] == a && w[i + 1] == b {
                    out.push(new);
                    i += 2;
                } else {
                    out.push(w[i]);
                    i += 1;
                }
            }
            for p in out.windows(2) {
                let k = pk(p[0], p[1]);
                *pair_count.entry(k).or_insert(0) += f;
                let v = where_.entry(k).or_default();
                if v.last() != Some(&wi) {
                    v.push(wi);
                }
                touched.push(k);
            }
            words[wi as usize] = out;
        }
        touched.sort_unstable();
        touched.dedup();
        for k in touched {
            let c = pair_count.get(&k).copied().unwrap_or(0);
            if c > 0 {
                heap.push((c, Reverse(k)));
            }
        }
        if cfg.log_every > 0 && merges.len().is_multiple_of(cfg.log_every) {
            eprintln!(
                "[bpe] {} merges, last {:?} ({} occurrences), {:.0} s",
                merges.len(),
                String::from_utf8_lossy(&symbol_bytes[new as usize]),
                count,
                t0.elapsed().as_secs_f32()
            );
        }
    }
    Ok(Trained {
        special_tokens: cfg.special_tokens.clone(),
        merges,
        symbol_bytes,
        distinct_words,
        total_words,
    })
}

pub fn train_bpe(files: &[impl AsRef<Path>], cfg: &TrainConfig) -> Result<Trained> {
    let t0 = std::time::Instant::now();
    let counts = count_words(files, PreTokenizer::Gpt2Digits, cfg.threads)?;
    if cfg.log_every > 0 {
        eprintln!(
            "[bpe] counted {} distinct pre-tokens in {:.1} s",
            counts.len(),
            t0.elapsed().as_secs_f32()
        );
    }
    train_from_counts(counts, cfg)
}

impl Trained {
    /// Token id of symbol `s` (0..256 + merges) in the final vocabulary: special tokens first, then the 256 byte tokens, then the merges.
    pub fn id_of_symbol(&self, s: u32) -> u32 {
        self.special_tokens.len() as u32 + s
    }

    /// A Hugging Face compatible tokenizer.json.
    pub fn to_tokenizer_json(&self) -> Json {
        let b2u = bytes_to_unicode();
        let to_str = |bytes: &[u8]| -> String { bytes.iter().map(|&b| b2u[b as usize]).collect() };
        let ns = self.special_tokens.len();
        let mut vocab: Vec<(String, Json)> = Vec::new();
        for (i, s) in self.special_tokens.iter().enumerate() {
            vocab.push((s.clone(), Json::Int(i as i64)));
        }
        for (i, bytes) in self.symbol_bytes.iter().enumerate() {
            vocab.push((to_str(bytes), Json::Int((ns + i) as i64)));
        }
        let merges: Vec<Json> = self
            .merges
            .iter()
            .map(|&(a, b)| {
                Json::Str(format!(
                    "{} {}",
                    to_str(&self.symbol_bytes[a as usize]),
                    to_str(&self.symbol_bytes[b as usize])
                ))
            })
            .collect();
        let added: Vec<Json> = self
            .special_tokens
            .iter()
            .enumerate()
            .map(|(i, s)| {
                Json::Obj(vec![
                    ("id".into(), Json::Int(i as i64)),
                    ("content".into(), Json::Str(s.clone())),
                    ("single_word".into(), Json::Bool(false)),
                    ("lstrip".into(), Json::Bool(false)),
                    ("rstrip".into(), Json::Bool(false)),
                    ("normalized".into(), Json::Bool(false)),
                    ("special".into(), Json::Bool(true)),
                ])
            })
            .collect();
        let byte_level = |regex: bool| {
            Json::Obj(vec![
                ("type".into(), Json::Str("ByteLevel".into())),
                ("add_prefix_space".into(), Json::Bool(false)),
                ("trim_offsets".into(), Json::Bool(true)),
                ("use_regex".into(), Json::Bool(regex)),
            ])
        };
        Json::Obj(vec![
            ("version".into(), Json::Str("1.0".into())),
            ("truncation".into(), Json::Null),
            ("padding".into(), Json::Null),
            ("added_tokens".into(), Json::Arr(added)),
            ("normalizer".into(), Json::Null),
            (
                "pre_tokenizer".into(),
                Json::Obj(vec![
                    ("type".into(), Json::Str("Sequence".into())),
                    (
                        "pretokenizers".into(),
                        Json::Arr(vec![
                            Json::Obj(vec![
                                ("type".into(), Json::Str("Digits".into())),
                                ("individual_digits".into(), Json::Bool(true)),
                            ]),
                            byte_level(true),
                        ]),
                    ),
                ]),
            ),
            ("post_processor".into(), Json::Null),
            ("decoder".into(), byte_level(true)),
            (
                "model".into(),
                Json::Obj(vec![
                    ("type".into(), Json::Str("BPE".into())),
                    ("dropout".into(), Json::Null),
                    ("unk_token".into(), Json::Null),
                    ("continuing_subword_prefix".into(), Json::Null),
                    ("end_of_word_suffix".into(), Json::Null),
                    ("fuse_unk".into(), Json::Bool(false)),
                    ("byte_fallback".into(), Json::Bool(false)),
                    ("ignore_merges".into(), Json::Bool(false)),
                    ("vocab".into(), Json::Obj(vocab)),
                    ("merges".into(), Json::Arr(merges)),
                ]),
            ),
        ])
    }
}
