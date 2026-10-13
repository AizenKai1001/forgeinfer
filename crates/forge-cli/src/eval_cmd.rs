//! Evaluation commands. `forge eval-hellaswag`: HellaSwag validation accuracy of a training checkpoint, scored the way the lm-evaluation-harness does it (log-likelihood of every ending given the
//! context; `acc` = highest total, `acc_norm` = highest total divided by the number of characters of the ending) plus the per-token average used by llm.c.

use forge_core::Json;
use forge_tok::Tokenizer;
use forge_train::{config_from_checkpoint, Trainer};
use std::path::{Path, PathBuf};
use std::time::Instant;

fn flag<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str())
}

/// lm-evaluation-harness preprocessing of HellaSwag text
fn preprocess(text: &str) -> String {
    let mut t = text.trim().replace(" [title]", ". ");
    // remove [ ... ] segments
    let mut out = String::with_capacity(t.len());
    let mut depth = 0;
    for c in t.drain(..) {
        match c {
            '[' => depth += 1,
            ']' if depth > 0 => depth -= 1,
            c if depth == 0 => out.push(c),
            _ => {}
        }
    }
    out.replace("  ", " ")
}

fn capitalize(s: &str) -> String {
    let mut c = s.chars();
    match c.next() {
        Some(f) => f.to_uppercase().collect::<String>() + &c.as_str().to_lowercase(),
        None => String::new(),
    }
}

pub fn hellaswag(args: &[String]) -> i32 {
    let ckpt = PathBuf::from(args.get(1).expect("usage: forge eval-hellaswag <checkpoint> --tokenizer FILE --data hellaswag_val.jsonl [--limit N]"));
    let tok = Tokenizer::from_file(flag(args, "--tokenizer").expect("--tokenizer is required"))
        .expect("cannot load the tokenizer");
    let data = std::fs::read_to_string(flag(args, "--data").expect("--data is required"))
        .expect("cannot read the data");
    let limit: usize = flag(args, "--limit")
        .and_then(|v| v.parse().ok())
        .unwrap_or(usize::MAX);
    let mut cfg = config_from_checkpoint(&ckpt).expect("cannot read the checkpoint header");
    let t_len = 256usize;
    // --batch N (a multiple of 4): sequences per forward pass (default 8 = two examples); a smaller one needs less GPU memory
    let nseq: usize = flag(args, "--batch")
        .and_then(|v| v.parse().ok())
        .unwrap_or(8)
        .max(4)
        / 4
        * 4;
    cfg.seq_len = t_len;
    cfg.batch = nseq;
    cfg.ce_chunk = 2048.min(t_len * nseq);
    let mut tr = Trainer::new_inference(cfg.clone(), 0).expect("cannot create the evaluator");
    tr.load_checkpoint(&ckpt)
        .expect("cannot load the checkpoint");
    let eos = tok.token_id("<|endoftext|>").unwrap_or(0);

    struct Ex {
        seqs: Vec<(Vec<u32>, usize, usize)>, // (tokens, context length, characters of the ending)
        label: usize,
    }
    let mut examples: Vec<Ex> = Vec::new();
    for line in data.lines().take(limit) {
        let j = Json::parse(line).expect("bad jsonl line");
        let ctx = preprocess(&format!(
            "{}: {} {}",
            j["activity_label"].as_str().unwrap_or(""),
            j["ctx_a"].as_str().unwrap_or(""),
            capitalize(j["ctx_b"].as_str().unwrap_or(""))
        ));
        let ctx_ids = tok.encode(&ctx);
        let mut seqs = Vec::new();
        for e in j["endings"].as_array().unwrap() {
            let ending = preprocess(e.as_str().unwrap_or(""));
            let end_ids = tok.encode(&format!(" {ending}"));
            let mut ids = ctx_ids.clone();
            ids.extend_from_slice(&end_ids);
            let mut clen = ctx_ids.len();
            if ids.len() > t_len + 1 {
                let cut = ids.len() - (t_len + 1);
                ids.drain(..cut);
                clen = clen.saturating_sub(cut).max(1);
            }
            seqs.push((ids, clen, ending.chars().count()));
        }
        let label = j["label"]
            .as_str()
            .and_then(|s| s.parse().ok())
            .or_else(|| j["label"].as_i64().map(|v| v as usize))
            .unwrap_or(0);
        examples.push(Ex { seqs, label });
    }
    let t0 = Instant::now();
    let (mut n, mut acc, mut acc_norm, mut acc_tok) = (0usize, 0usize, 0usize, 0usize);
    for pair in examples.chunks(nseq / 4) {
        // nseq / 4 examples = nseq sequences per forward
        let mut batch: Vec<u32> = Vec::with_capacity(nseq * (t_len + 1));
        let mut meta: Vec<(usize, usize, usize, usize)> = Vec::new(); // (example in pair, ending, ctx_len, len)
        for (ei, ex) in pair.iter().enumerate() {
            for (si, (ids, clen, _)) in ex.seqs.iter().enumerate() {
                batch.extend_from_slice(ids);
                batch.resize(batch.len() + (t_len + 1 - ids.len()), eos);
                meta.push((ei, si, *clen, ids.len()));
            }
        }
        while batch.len() < nseq * (t_len + 1) {
            batch.push(eos);
        }
        let nll = tr.token_nll(&batch).expect("forward failed");
        let mut scores = vec![[0f64; 4]; pair.len()];
        let mut tok_scores = vec![[0f64; 4]; pair.len()];
        for (row, (ei, si, clen, len)) in meta.iter().enumerate() {
            // ending tokens are ids[clen..len]; token ids[j + 1] is predicted at position j
            let s: f64 = (clen - 1..len - 1)
                .map(|j| nll[row * t_len + j] as f64)
                .sum();
            scores[*ei][*si] = -s;
            tok_scores[*ei][*si] = -s / (*len - *clen).max(1) as f64;
        }
        for (ei, ex) in pair.iter().enumerate() {
            let argmax = |v: &[f64]| {
                (0..4)
                    .max_by(|&a, &b| v[a].partial_cmp(&v[b]).unwrap())
                    .unwrap()
            };
            let normed: Vec<f64> = (0..4)
                .map(|k| scores[ei][k] / ex.seqs[k].2.max(1) as f64)
                .collect();
            n += 1;
            acc += (argmax(&scores[ei]) == ex.label) as usize;
            acc_norm += (argmax(&normed) == ex.label) as usize;
            acc_tok += (argmax(&tok_scores[ei]) == ex.label) as usize;
        }
    }
    println!(
        "HellaSwag validation, {} examples: acc {:.2} %  acc_norm {:.2} %  acc (per-token mean, llm.c style) {:.2} %   ({:.0} s; chance 25 %)",
        n,
        100.0 * acc as f64 / n as f64,
        100.0 * acc_norm as f64 / n as f64,
        100.0 * acc_tok as f64 / n as f64,
        t0.elapsed().as_secs_f32()
    );
    0
}

/// `forge eval-mc <checkpoint> --tokenizer FILE --data FILE.jsonl [--limit N] [--seq 256] [--batch 8]`: any multiple-choice set in the jsonl format of tools/mc_prepare.py
/// ({"choices": [{"ctx", "cont"}, ...], "label": k}); every choice is scored as the log-likelihood of `cont` given `ctx` (the harness's way: the continuation tokens are the tail of the encoded
/// ctx + cont), `acc` = highest total, `acc_norm` = highest total divided by the characters of the continuation (without its leading space).
pub fn mc(args: &[String]) -> i32 {
    let ckpt = PathBuf::from(args.get(1).expect("usage: forge eval-mc <checkpoint> --tokenizer FILE --data FILE.jsonl [--limit N] [--seq 256] [--batch 8]"));
    let tok = Tokenizer::from_file(flag(args, "--tokenizer").expect("--tokenizer is required"))
        .expect("cannot load the tokenizer");
    let data_path = flag(args, "--data").expect("--data is required");
    let data = std::fs::read_to_string(data_path).expect("cannot read the data");
    let limit: usize = flag(args, "--limit")
        .and_then(|v| v.parse().ok())
        .unwrap_or(usize::MAX);
    let t_len: usize = flag(args, "--seq")
        .and_then(|v| v.parse().ok())
        .unwrap_or(256);
    let batch: usize = flag(args, "--batch")
        .and_then(|v| v.parse().ok())
        .unwrap_or(8);
    let mut cfg = config_from_checkpoint(&ckpt).expect("cannot read the checkpoint header");
    cfg.seq_len = t_len;
    cfg.batch = batch;
    cfg.ce_chunk = 2048.min(t_len * batch);
    let mut tr = Trainer::new_inference(cfg, 0).expect("cannot create the evaluator");
    tr.load_checkpoint(&ckpt)
        .expect("cannot load the checkpoint");
    let eos = tok.token_id("<|endoftext|>").unwrap_or(0);

    struct Seq {
        ids: Vec<u32>,
        clen: usize,
        chars: usize,
    }
    let mut examples: Vec<(Vec<Seq>, usize)> = Vec::new();
    for line in data.lines().take(limit) {
        let j = Json::parse(line).expect("bad jsonl line");
        let mut seqs = Vec::new();
        for c in j["choices"].as_array().expect("choices") {
            let (ctx, cont) = (
                c["ctx"].as_str().unwrap_or(""),
                c["cont"].as_str().unwrap_or(""),
            );
            let ctx_ids = tok.encode(ctx);
            let whole = tok.encode(&format!("{ctx}{cont}"));
            let mut ids = if whole.len() > ctx_ids.len() && whole[..ctx_ids.len()] == ctx_ids[..] {
                whole
            } else {
                let mut v = ctx_ids.clone();
                v.extend(tok.encode(cont));
                v
            };
            let mut clen = ctx_ids.len().max(1);
            if ids.len() > t_len + 1 {
                let cut = ids.len() - (t_len + 1);
                ids.drain(..cut);
                clen = clen.saturating_sub(cut).max(1);
            }
            seqs.push(Seq {
                ids,
                clen,
                chars: cont.trim_start().chars().count().max(1),
            });
        }
        examples.push((seqs, j["label"].as_usize().unwrap_or(0)));
    }
    // all (example, choice) sequences in a flat list, `batch` rows per forward
    let flat: Vec<(usize, usize)> = examples
        .iter()
        .enumerate()
        .flat_map(|(e, (s, _))| (0..s.len()).map(move |c| (e, c)))
        .collect();
    let mut score: Vec<Vec<f64>> = examples.iter().map(|(s, _)| vec![0.0; s.len()]).collect();
    let t0 = Instant::now();
    for chunk in flat.chunks(batch) {
        let mut rows: Vec<u32> = Vec::with_capacity(batch * (t_len + 1));
        for &(e, c) in chunk {
            let s = &examples[e].0[c];
            rows.extend_from_slice(&s.ids);
            rows.resize(rows.len() + (t_len + 1 - s.ids.len()), eos);
        }
        rows.resize(batch * (t_len + 1), eos);
        let nll = tr.token_nll(&rows).expect("forward failed");
        for (r, &(e, c)) in chunk.iter().enumerate() {
            let s = &examples[e].0[c];
            let total: f64 = (s.clen - 1..s.ids.len() - 1)
                .map(|j| nll[r * t_len + j] as f64)
                .sum();
            score[e][c] = -total;
        }
    }
    let (mut n, mut acc, mut acc_norm) = (0usize, 0usize, 0usize);
    for (e, (seqs, label)) in examples.iter().enumerate() {
        let argmax = |v: &[f64]| {
            (0..v.len())
                .max_by(|&a, &b| v[a].partial_cmp(&v[b]).unwrap())
                .unwrap()
        };
        let normed: Vec<f64> = (0..seqs.len())
            .map(|c| score[e][c] / seqs[c].chars as f64)
            .collect();
        n += 1;
        acc += (argmax(&score[e]) == *label) as usize;
        acc_norm += (argmax(&normed) == *label) as usize;
    }
    let chance: f64 = examples
        .iter()
        .map(|(s, _)| 1.0 / s.len() as f64)
        .sum::<f64>()
        / n.max(1) as f64;
    println!(
        "{}: {} examples  acc {:.2} %  acc_norm {:.2} %   (chance {:.1} %, {:.0} s)",
        Path::new(data_path)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("mc"),
        n,
        100.0 * acc as f64 / n as f64,
        100.0 * acc_norm as f64 / n as f64,
        100.0 * chance,
        t0.elapsed().as_secs_f32()
    );
    0
}

/// `forge eval-ppl <checkpoint> --data FILE.u16 [FILE.u16 ...] [--windows 64] [--seq 512] [--batch 4]`: the mean negative log-likelihood per token (nats; perplexity = exp) of a checkpoint on token files
/// (uint16 shards such as `~/forgeinfer/data/val_domains/*.u16`): `windows` windows of `seq` + 1 tokens spread evenly over each file, every position scored.
pub fn ppl(args: &[String]) -> i32 {
    let ckpt = PathBuf::from(args.get(1).expect("usage: forge eval-ppl <checkpoint> --data FILE.u16 [FILE.u16 ...] [--windows 64] [--seq 512] [--batch 4]"));
    let files: Vec<&String> = args
        .iter()
        .skip_while(|a| a.as_str() != "--data")
        .skip(1)
        .take_while(|a| !a.starts_with("--"))
        .collect();
    assert!(!files.is_empty(), "--data needs at least one file");
    let windows: usize = flag(args, "--windows")
        .and_then(|v| v.parse().ok())
        .unwrap_or(64);
    let seq: usize = flag(args, "--seq")
        .and_then(|v| v.parse().ok())
        .unwrap_or(512);
    let batch: usize = flag(args, "--batch")
        .and_then(|v| v.parse().ok())
        .unwrap_or(4);
    let mut cfg = config_from_checkpoint(&ckpt).expect("cannot read the checkpoint header");
    cfg.seq_len = seq;
    cfg.batch = batch;
    cfg.ce_chunk = 2048.min(seq * batch);
    let mut tr = Trainer::new_inference(cfg, 0).expect("cannot create the evaluator");
    tr.load_checkpoint(&ckpt)
        .expect("cannot load the checkpoint");
    for f in files {
        let raw = std::fs::read(f).expect("cannot read the token file");
        let toks: Vec<u32> = raw
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]) as u32)
            .collect();
        assert!(toks.len() > seq + 1, "{f} is too short");
        let span = toks.len() - (seq + 1);
        let (mut sum, mut count) = (0f64, 0usize);
        let n_batches = (windows + batch - 1) / batch;
        for bi in 0..n_batches {
            let mut rows: Vec<u32> = Vec::with_capacity(batch * (seq + 1));
            for r in 0..batch {
                let k = bi * batch + r;
                let start = span * (k % windows) / windows.max(1);
                rows.extend_from_slice(&toks[start..start + seq + 1]);
            }
            let nll = tr.token_nll(&rows).expect("forward failed");
            for (i, v) in nll.iter().enumerate() {
                // windows beyond `windows` (padding of the last batch) are not counted
                if bi * batch + i / seq < windows {
                    sum += *v as f64;
                    count += 1;
                }
            }
        }
        let mean = sum / count.max(1) as f64;
        println!(
            "{}: {:.4} nats per token (perplexity {:.1}) over {} tokens",
            Path::new(f)
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("data"),
            mean,
            mean.exp(),
            count
        );
    }
    0
}
