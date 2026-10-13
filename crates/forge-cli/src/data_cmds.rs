//! Data preparation commands for pre-training: train a byte-level BPE on text files and encode corpora to uint16 token shards.
//! Corpus files hold documents separated by a NUL byte (tools/data_fetch.py writes them).

use forge_tok::train::{train_bpe, TrainConfig};
use forge_tok::Tokenizer;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::sync_channel;
use std::time::Instant;

/// The special tokens of our models, ids 0..15 (ChatML plus tool-call and reasoning markers; the rest is reserved so that later additions do not change the vocabulary size).
pub const SPECIAL_TOKENS: [&str; 16] = [
    "<|endoftext|>",
    "<|im_start|>",
    "<|im_end|>",
    "<tool_call>",
    "</tool_call>",
    "<tool_response>",
    "</tool_response>",
    "<think>",
    "</think>",
    "<|pad|>",
    "<|reserved_0|>",
    "<|reserved_1|>",
    "<|reserved_2|>",
    "<|reserved_3|>",
    "<|reserved_4|>",
    "<|reserved_5|>",
];

fn flag<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str())
}

/// forge tok-train <corpus files...> --vocab N --out tokenizer.json [--threads T]
pub fn tok_train(args: &[String]) -> i32 {
    let files: Vec<PathBuf> = args
        .iter()
        .skip(1)
        .take_while(|a| !a.starts_with("--"))
        .map(PathBuf::from)
        .collect();
    if files.is_empty() {
        eprintln!(
            "usage: forge tok-train <corpus files...> --vocab N --out tokenizer.json [--threads T]"
        );
        return 2;
    }
    let vocab: usize = flag(args, "--vocab")
        .and_then(|v| v.parse().ok())
        .unwrap_or(32768);
    let threads: usize = flag(args, "--threads")
        .and_then(|v| v.parse().ok())
        .unwrap_or(4);
    let out = flag(args, "--out").unwrap_or("tokenizer.json");
    let cfg = TrainConfig {
        vocab_size: vocab,
        special_tokens: SPECIAL_TOKENS.iter().map(|s| s.to_string()).collect(),
        threads,
        log_every: 1000,
    };
    let t0 = Instant::now();
    let trained = match train_bpe(&files, &cfg) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("{e}");
            return 1;
        }
    };
    eprintln!(
        "trained {} merges ({} tokens with specials) from {} distinct pre-tokens ({} total) in {:.1} s",
        trained.merges.len(),
        trained.symbol_bytes.len() + trained.special_tokens.len(),
        trained.distinct_words,
        trained.total_words,
        t0.elapsed().as_secs_f32()
    );
    let mut text = String::new();
    trained.to_tokenizer_json().write(&mut text);
    std::fs::write(out, text).expect("cannot write the tokenizer file");
    eprintln!("wrote {out}");
    0
}

/// The documents of a block: the pieces between NUL bytes, empty ones left out (the NUL search looks at eight bytes at a time; the plain `split` closure runs byte by byte, about a tenth of the encoder's time).
fn split_documents(buf: &[u8]) -> Vec<&[u8]> {
    let mut docs = Vec::with_capacity(buf.len() / 2000 + 1);
    let mut start = 0usize;
    while start < buf.len() {
        let end = forge_tok::fast::find_byte(buf, 0, start).unwrap_or(buf.len());
        if end > start {
            docs.push(&buf[start..end]);
        }
        start = end + 1;
    }
    docs
}

/// Encoding work is cut into ranges of documents of about this many bytes; the threads take the ranges one by one, so no thread idles while another still has a big piece.
const RANGE_BYTES: usize = 256 << 10;

/// forge tok-encode <tokenizer.json> <corpus.txt> --out shard.u16 [--threads T] [--eos ID]
/// A pipeline of a reader (32 MB blocks cut at a document end), T encoder threads (each keeps its word cache for the whole run) and a writer, so that reading and writing overlap the encoding.
pub fn tok_encode(args: &[String]) -> i32 {
    let (Some(tok_path), Some(corpus)) = (args.get(1), args.get(2)) else {
        eprintln!(
            "usage: forge tok-encode <tokenizer.json> <corpus.txt> --out shard.u16 [--threads T]"
        );
        return 2;
    };
    let out_path = flag(args, "--out").unwrap_or("shard.u16");
    let threads: usize = flag(args, "--threads")
        .and_then(|v| v.parse().ok())
        .unwrap_or(4)
        .max(1);
    let tok = match Tokenizer::from_file(tok_path) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("{e}");
            return 1;
        }
    };
    if tok.vocab_size() > 65536 {
        eprintln!(
            "vocabulary of {} does not fit uint16 shards",
            tok.vocab_size()
        );
        return 1;
    }
    let eos = flag(args, "--eos")
        .and_then(|v| v.parse().ok())
        .or_else(|| tok.token_id("<|endoftext|>"))
        .unwrap_or(0) as u16;
    let mut file = std::fs::File::open(Path::new(corpus)).expect("cannot open the corpus");
    let out_file = std::fs::File::create(out_path).expect("cannot create the output");
    let t0 = Instant::now();
    let block = 32usize << 20;
    let mut encoders: Vec<forge_tok::Encoder> = (0..threads).map(|_| tok.encoder()).collect();
    let n_tokens = AtomicU64::new(0);
    let (mut n_bytes, mut n_docs) = (0u64, 0u64);
    std::thread::scope(|s| {
        let (tx_block, rx_block) = sync_channel::<Vec<u8>>(2);
        let (tx_out, rx_out) = sync_channel::<Vec<Vec<u8>>>(2);
        // reader: blocks of whole documents (a block ends right after a NUL; the rest of the read waits for the next block)
        s.spawn(move || {
            let mut carry: Vec<u8> = Vec::new();
            loop {
                let mut buf = std::mem::take(&mut carry);
                let have = buf.len();
                buf.resize(have + block, 0);
                let mut got = 0;
                while got < block {
                    let n = file.read(&mut buf[have + got..]).expect("read failed");
                    if n == 0 {
                        break;
                    }
                    got += n;
                }
                buf.truncate(have + got);
                if got == 0 {
                    if !buf.is_empty() {
                        let _ = tx_block.send(buf);
                    }
                    break;
                }
                match buf.iter().rposition(|&b| b == 0) {
                    Some(p) => {
                        carry = buf.split_off(p + 1);
                        if tx_block.send(buf).is_err() {
                            break;
                        }
                    }
                    None => carry = buf, // one document longer than a block: keep reading
                }
            }
        });
        // writer
        let counter = &n_tokens;
        s.spawn(move || {
            let mut w = std::io::BufWriter::with_capacity(1 << 20, out_file);
            for parts in rx_out {
                for p in parts {
                    counter.fetch_add(p.len() as u64 / 2, Ordering::Relaxed);
                    w.write_all(&p).expect("write failed");
                }
            }
            w.flush().expect("flush failed");
        });
        // encoding: each block is cut into ranges of documents, the threads take them in turn, the outputs go to the writer in order
        for buf in rx_block {
            let docs = split_documents(&buf);
            n_docs += docs.len() as u64;
            n_bytes += docs.iter().map(|d| d.len() as u64).sum::<u64>();
            let mut ranges: Vec<(usize, usize)> = Vec::new();
            let (mut start, mut size) = (0usize, 0usize);
            for (i, d) in docs.iter().enumerate() {
                size += d.len();
                if size >= RANGE_BYTES {
                    ranges.push((start, i + 1));
                    start = i + 1;
                    size = 0;
                }
            }
            if start < docs.len() {
                ranges.push((start, docs.len()));
            }
            let next = AtomicUsize::new(0);
            let mut parts: Vec<(usize, Vec<u8>)> = std::thread::scope(|s2| {
                let hs: Vec<_> = encoders
                    .iter_mut()
                    .map(|enc| {
                        let (docs, ranges, next) = (&docs, &ranges, &next);
                        s2.spawn(move || {
                            let mut done: Vec<(usize, Vec<u8>)> = Vec::new();
                            let mut ids: Vec<u32> = Vec::new();
                            loop {
                                let r = next.fetch_add(1, Ordering::Relaxed);
                                if r >= ranges.len() {
                                    break;
                                }
                                let mut bytes: Vec<u8> = Vec::new();
                                for d in &docs[ranges[r].0..ranges[r].1] {
                                    let text = String::from_utf8_lossy(d);
                                    ids.clear();
                                    enc.encode_into(&text, &mut ids);
                                    ids.push(eos as u32);
                                    let at = bytes.len();
                                    bytes.resize(at + ids.len() * 2, 0);
                                    for (dst, &id) in
                                        bytes[at..].chunks_exact_mut(2).zip(ids.iter())
                                    {
                                        dst.copy_from_slice(&(id as u16).to_le_bytes());
                                    }
                                }
                                done.push((r, bytes));
                            }
                            done
                        })
                    })
                    .collect();
                hs.into_iter().flat_map(|h| h.join().unwrap()).collect()
            });
            parts.sort_by_key(|p| p.0);
            if tx_out
                .send(parts.into_iter().map(|p| p.1).collect())
                .is_err()
            {
                break;
            }
            eprintln!(
                "[encode] {} docs, {:.2} GB text, {:.2} G tokens written or queued, {:.0} s",
                n_docs,
                n_bytes as f64 / 1e9,
                n_tokens.load(Ordering::Relaxed) as f64 / 1e9,
                t0.elapsed().as_secs_f32()
            );
        }
        drop(tx_out);
    });
    let dt = t0.elapsed().as_secs_f64();
    let n_tok = n_tokens.load(Ordering::Relaxed);
    eprintln!(
        "{} documents, {} bytes -> {} tokens ({:.4} tokens per byte) in {:.1} s ({:.1} MB/s)",
        n_docs,
        n_bytes,
        n_tok,
        n_tok as f64 / n_bytes.max(1) as f64,
        dt,
        n_bytes as f64 / 1e6 / dt
    );
    0
}

/// The token ids of the chat format that matter for masking.
pub struct ChatIds {
    pub eos: u16,
    pub im_start: u16,
    pub im_end: u16,
    /// the tokens of "assistant\n" (the role header that follows <|im_start|> in an assistant turn)
    pub assistant: Vec<u16>,
}

/// Marks (bit 15, "not a target") every token of a ChatML token stream that the assistant does not write: system and user turns, the role headers, the newline after each <|im_end|> and the document
/// separators. The text of an assistant turn and its closing <|im_end|> stay trainable, so the model learns to answer and to stop, not to imitate the user. Returns the number of masked tokens.
pub fn mask_chat(tokens: &mut [u16], ids: &ChatIds) -> usize {
    let mut masked = 0usize;
    let mut in_assistant = false;
    let mut i = 0usize;
    while i < tokens.len() {
        let t = tokens[i] & 0x7fff;
        if t == ids.im_start {
            // the start of a turn: the marker and the role header are prompt; the assistant's turn begins after its header
            let header = ids.assistant.len();
            let is_assistant = i + 1 + header <= tokens.len()
                && tokens[i + 1..i + 1 + header]
                    .iter()
                    .zip(&ids.assistant)
                    .all(|(a, b)| a & 0x7fff == *b);
            let upto = if is_assistant { i + 1 + header } else { i + 1 };
            for tk in &mut tokens[i..upto] {
                *tk |= 0x8000;
            }
            masked += upto - i;
            in_assistant = is_assistant;
            i = upto;
            continue;
        }
        if t == ids.eos {
            in_assistant = false;
        }
        if in_assistant {
            if t == ids.im_end {
                in_assistant = false; // the closing marker is trained; what follows (the newline) is not
            }
        } else {
            tokens[i] |= 0x8000;
            masked += 1;
        }
        i += 1;
    }
    masked
}

/// forge chat-mask <tokenizer.json> <in.u16> --out <out.u16>
/// Adds the "not a target" mark (bit 15) to the prompt tokens of a ChatML shard (see `mask_chat`); `forge train` then trains on the assistant's text only. Needs a vocabulary of at most 32,768 tokens.
pub fn chat_mask(args: &[String]) -> i32 {
    let (Some(tok_path), Some(input)) = (args.get(1), args.get(2)) else {
        eprintln!("usage: forge chat-mask <tokenizer.json> <in.u16> --out <out.u16>");
        return 2;
    };
    let Some(out_path) = flag(args, "--out") else {
        eprintln!("--out is required");
        return 2;
    };
    let tok = match Tokenizer::from_file(tok_path) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("{e}");
            return 1;
        }
    };
    if tok.vocab_size() > 32768 {
        eprintln!(
            "a vocabulary of {} tokens leaves no spare bit in the uint16 tokens",
            tok.vocab_size()
        );
        return 1;
    }
    let (Some(im_start), Some(im_end), Some(eos)) = (
        tok.token_id("<|im_start|>"),
        tok.token_id("<|im_end|>"),
        tok.token_id("<|endoftext|>"),
    ) else {
        eprintln!("the tokenizer has no ChatML special tokens");
        return 1;
    };
    let ids = ChatIds {
        eos: eos as u16,
        im_start: im_start as u16,
        im_end: im_end as u16,
        assistant: tok
            .encode("assistant\n")
            .into_iter()
            .map(|v| v as u16)
            .collect(),
    };
    let bytes = std::fs::read(input).expect("cannot read the input shard");
    let mut tokens: Vec<u16> = bytes
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect();
    drop(bytes);
    if tokens.iter().any(|&t| t & 0x8000 != 0) {
        eprintln!("the shard already carries marks (bit 15 set): refusing to mask twice");
        return 1;
    }
    let masked = mask_chat(&mut tokens, &ids);
    let mut out =
        std::io::BufWriter::new(std::fs::File::create(out_path).expect("cannot create the output"));
    for chunk in tokens.chunks(1 << 24) {
        let b: Vec<u8> = chunk.iter().flat_map(|v| v.to_le_bytes()).collect();
        out.write_all(&b).expect("write failed");
    }
    out.flush().unwrap();
    eprintln!(
        "{} tokens, {} masked ({:.1} %), {} trainable (assistant header = {:?})",
        tokens.len(),
        masked,
        100.0 * masked as f64 / tokens.len().max(1) as f64,
        tokens.len() - masked,
        ids.assistant
    );
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn documents_are_split_at_nul_bytes_like_the_plain_split() {
        let mut state = 12345u64;
        for _ in 0..20_000 {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let n = (state >> 40) as usize % 70;
            let buf: Vec<u8> = (0..n)
                .map(|_| {
                    state = state
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    if (state >> 33) % 5 == 0 {
                        0
                    } else {
                        (state >> 40) as u8 | 1
                    }
                })
                .collect();
            let want: Vec<&[u8]> = buf.split(|&b| b == 0).filter(|d| !d.is_empty()).collect();
            assert_eq!(split_documents(&buf), want, "{buf:?}");
        }
    }

    // ids: 0 eos, 1 <|im_start|>, 2 <|im_end|>, 50 = "assistant", 51 = "\n", 60 = "user", 70.. = text
    fn ids() -> ChatIds {
        ChatIds {
            eos: 0,
            im_start: 1,
            im_end: 2,
            assistant: vec![50, 51],
        }
    }

    fn trained(tokens: &[u16]) -> Vec<u16> {
        tokens
            .iter()
            .filter(|&&t| t & 0x8000 == 0)
            .cloned()
            .collect()
    }

    #[test]
    fn only_the_assistant_text_and_its_end_marker_stay_trainable() {
        // doc 1: user asks (70 71), assistant answers (72 73 74); doc 2: system (75), user (76), assistant (77), user (78), assistant (79 80)
        let mut t: Vec<u16> = vec![
            1, 60, 51, 70, 71, 2, 51, 1, 50, 51, 72, 73, 74, 2, 51, 0, 1, 40, 51, 75, 2, 51, 1, 60,
            51, 76, 2, 51, 1, 50, 51, 77, 2, 51, 1, 60, 51, 78, 2, 51, 1, 50, 51, 79, 80, 2, 51, 0,
        ];
        let n = t.len();
        let masked = mask_chat(&mut t, &ids());
        assert_eq!(
            trained(&t.iter().map(|&x| x).collect::<Vec<_>>()).len(),
            n - masked
        );
        let kept: Vec<u16> = t.iter().filter(|&&x| x & 0x8000 == 0).cloned().collect();
        assert_eq!(kept, vec![72, 73, 74, 2, 77, 2, 79, 80, 2]);
        // the marks sit on top of unchanged ids
        assert_eq!(t[0], 1 | 0x8000);
        assert_eq!(t[10] & 0x7fff, 72);
        assert_eq!(t[10] & 0x8000, 0);
    }

    #[test]
    fn a_turn_cut_off_by_a_document_end_does_not_leak_into_the_next_document() {
        let mut t: Vec<u16> = vec![1, 50, 51, 72, 73, 0, 70, 71, 1, 60, 51, 72, 2, 51, 0];
        mask_chat(&mut t, &ids());
        let kept: Vec<u16> = t.iter().filter(|&&x| x & 0x8000 == 0).cloned().collect();
        assert_eq!(kept, vec![72, 73]);
    }
}
