//! `forge eval-lambada`: LAMBADA the way the lm-evaluation-harness scores `lambada_openai` (zero-shot): the model reads a passage without its last word and has to continue with that word.
//!
//!   forge eval-lambada <checkpoint> --tokenizer FILE --data lambada_openai.jsonl [--limit N] [--seq 256]
//!
//! One line per example in the jsonl (tools/f1_eval_data.py writes it): {"ctx": "passage without its last word", "cont": " word"} (the continuation keeps its leading space).
//!   acc         = the model's greedy continuation reproduces EVERY token of the word: with the true previous tokens given, the arg-max of the logits at each token of the word is that token
//!                 (the harness's `is_greedy`; a word the tokenizer splits into three tokens needs three arg-max hits)
//!   perplexity  = exp(-mean over the examples of the log-likelihood of the whole word)   (a per-WORD perplexity: comparable across tokenizers, unlike a per-token one)
//! The context and the word are tokenized as `forge eval-mc` does it (the word's tokens are the tail of the encoded ctx + cont). No batching: every target token costs one forward pass of
//! `Trainer::logits_at` over the passage (the window grows in steps of 64 positions), about 1.5 passes per example with the Forge-1 tokenizer.
//! The first examples are also scored with `Trainer::token_nll` (the path of eval-mc / eval-ppl) and the two log-likelihoods are compared: the `selfcheck` line must show a difference below ~0.02 nats.

use forge_core::Json;
use forge_tok::Tokenizer;
use forge_train::{config_from_checkpoint, Trainer};
use std::path::{Path, PathBuf};
use std::time::Instant;

fn flag<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).map(|s| s.as_str())
}

/// log-probability of `target` under the softmax of `logits` (f64 accumulation) and whether `target` is the arg-max (the first maximum on ties)
fn log_prob(logits: &[f32], target: usize) -> (f64, bool) {
    let mut mx = f32::NEG_INFINITY;
    let mut arg = 0usize;
    for (i, &l) in logits.iter().enumerate() {
        if l > mx {
            mx = l;
            arg = i;
        }
    }
    let sum: f64 = logits.iter().map(|&l| ((l - mx) as f64).exp()).sum();
    ((logits[target] - mx) as f64 - sum.ln(), arg == target)
}

pub fn lambada(args: &[String]) -> i32 {
    let ckpt = PathBuf::from(args.get(1).expect("usage: forge eval-lambada <checkpoint> --tokenizer FILE --data lambada_openai.jsonl [--limit N] [--seq 256]"));
    let tok = Tokenizer::from_file(flag(args, "--tokenizer").expect("--tokenizer is required")).expect("cannot load the tokenizer");
    let data_path = flag(args, "--data").expect("--data is required");
    let data = std::fs::read_to_string(data_path).expect("cannot read the data");
    let limit: usize = flag(args, "--limit").and_then(|v| v.parse().ok()).unwrap_or(usize::MAX);
    let t_len: usize = flag(args, "--seq").and_then(|v| v.parse().ok()).unwrap_or(256);
    let mut cfg = config_from_checkpoint(&ckpt).expect("cannot read the checkpoint header");
    cfg.seq_len = t_len;
    cfg.batch = 1;
    cfg.ce_chunk = t_len;
    let mut tr = Trainer::new_inference(cfg, 0).expect("cannot create the evaluator");
    tr.load_checkpoint(&ckpt).expect("cannot load the checkpoint");
    let eos = tok.token_id("<|endoftext|>").unwrap_or(0);

    let (mut n, mut hits, mut word_nll, mut target_tokens) = (0usize, 0usize, 0f64, 0usize);
    let (mut check_max, mut check_n) = (0f64, 0usize);
    let t0 = Instant::now();
    for line in data.lines().take(limit) {
        if line.trim().is_empty() {
            continue;
        }
        let j = Json::parse(line).expect("bad jsonl line");
        let (ctx, cont) = (j["ctx"].as_str().unwrap_or(""), j["cont"].as_str().unwrap_or(""));
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
        // the last t_len + 1 tokens (t_len inputs and the last target), as eval-mc truncates
        if ids.len() > t_len + 1 {
            let cut = ids.len() - (t_len + 1);
            ids.drain(..cut);
            clen = clen.saturating_sub(cut).max(1);
        }
        if ids.len() <= clen {
            continue; // an empty continuation cannot be scored
        }
        let (mut ll, mut greedy) = (0f64, true);
        let mut per_pos: Vec<f64> = Vec::new();
        for pos in clen..ids.len() {
            // the logits that follow ids[..pos]; the true token there is ids[pos]
            let logits = tr.logits_at(&ids, pos).expect("forward failed");
            let (lp, hit) = log_prob(&logits, ids[pos] as usize);
            ll += lp;
            greedy &= hit;
            per_pos.push(lp);
        }
        // cross-check of the first examples against the trainer's own per-token negative log-likelihood (the path eval-mc and eval-ppl use)
        if check_n < 8 {
            let mut row = ids.clone();
            row.resize(t_len + 1, eos);
            let nll = tr.token_nll(&row).expect("forward failed");
            for (k, pos) in (clen..ids.len()).enumerate() {
                check_max = check_max.max((-per_pos[k] - nll[pos - 1] as f64).abs());
            }
            check_n += 1;
        }
        n += 1;
        hits += greedy as usize;
        word_nll += -ll;
        target_tokens += ids.len() - clen;
        if n % 1000 == 0 {
            eprintln!("  {n} examples, acc {:.2} %, {:.0} s", 100.0 * hits as f64 / n as f64, t0.elapsed().as_secs_f32());
        }
    }
    println!("selfcheck: largest |nll(logits_at) - nll(token_nll)| over the target tokens of the first {check_n} examples = {check_max:.5} nats (expected below 0.02)");
    let mean = word_nll / n.max(1) as f64;
    println!(
        "{}: {} examples  acc {:.2} %  perplexity {:.2}   (mean {:.4} nats per word, {:.2} target tokens per word, {:.0} s; chance ~0 %)",
        Path::new(data_path).file_stem().and_then(|s| s.to_str()).unwrap_or("lambada"),
        n,
        100.0 * hits as f64 / n.max(1) as f64,
        mean.exp(),
        mean,
        target_tokens as f64 / n.max(1) as f64,
        t0.elapsed().as_secs_f32()
    );
    0
}
