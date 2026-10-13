//! `forge gen-check`: KV-cache generation from a training checkpoint against the no-cache reference (the trainer's forward over the whole text for every token), on real prompts.
//!
//!   forge gen-check <checkpoint> --tokenizer FILE [--prompts "text one|text two|..."] [--tokens 200] [--ctx 1024] [--chunk 512] [--bench] [--long N] [--old-long K] [--q8 --no-ref]
//! (--q8 --no-ref: the speed of the Q8-weights decoder, `forge serve --q8`; --no-ref is needed because the fp16 reference is bitwise the other path)
//!
//! For every prompt: greedy decoding both ways (the cache path is fed the prompt at once, then one token per step), the generated tokens must be identical and the logits are compared at every
//! position (bitwise equality and the largest difference); the prompt fed token by token must leave the same cache and logits as the prompt fed at once. Tokens per second of both paths are printed.
//! --bench adds the cost of a long context: the cache is filled with --long tokens (default: the context minus 80) and 64 tokens are decoded there, the old path is timed on --old-long tokens at that depth.

use forge_tok::Tokenizer;
use forge_train::{config_from_checkpoint, Decoder, Trainer};
use std::path::PathBuf;
use std::time::Instant;

fn flag<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str())
}

/// arg-max, the lowest index wins ties
fn argmax(l: &[f32]) -> u32 {
    let mut best = 0usize;
    for (i, &v) in l.iter().enumerate() {
        if v > l[best] {
            best = i;
        }
    }
    best as u32
}

fn compare(a: &[f32], b: &[f32]) -> (f32, bool) {
    let mut worst = 0f32;
    let mut same = true;
    for (x, y) in a.iter().zip(b) {
        worst = worst.max((x - y).abs());
        same &= x.to_bits() == y.to_bits();
    }
    (worst, same)
}

/// gap between the best and the second-best logit (how close greedy decoding was to a tie)
fn top2_gap(l: &[f32]) -> f32 {
    let (mut a, mut b) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
    for &v in l {
        if v > a {
            b = a;
            a = v;
        } else if v > b {
            b = v;
        }
    }
    a - b
}

pub fn gen_check(args: &[String]) -> i32 {
    let Some(ckpt) = args.get(1).filter(|a| !a.starts_with("--")) else {
        eprintln!("usage: forge gen-check <checkpoint> --tokenizer FILE [--prompts \"a|b\"] [--tokens 200] [--ctx 1024] [--chunk 512] [--bench] [--long N] [--old-long K]");
        return 2;
    };
    let ckpt = PathBuf::from(ckpt);
    let tok = match flag(args, "--tokenizer").map(Tokenizer::from_file) {
        Some(Ok(t)) => t,
        Some(Err(e)) => {
            eprintln!("tokenizer: {e}");
            return 1;
        }
        None => {
            eprintln!("--tokenizer is required");
            return 2;
        }
    };
    let n_new: usize = flag(args, "--tokens")
        .and_then(|v| v.parse().ok())
        .unwrap_or(200);
    let ctx: usize = flag(args, "--ctx")
        .and_then(|v| v.parse().ok())
        .unwrap_or(1024);
    let chunk: usize = flag(args, "--chunk")
        .and_then(|v| v.parse().ok())
        .unwrap_or(512);
    let no_ref = args.iter().any(|a| a == "--no-ref");
    let prompts: Vec<String> = flag(args, "--prompts")
        .map(|p| p.split('|').map(|s| s.to_string()).collect())
        .unwrap_or_else(|| {
            vec![
                "The history of the Roman Empire".to_string(),
                "In mathematics, a prime number is".to_string(),
                "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n".to_string(),
                "<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n".to_string(),
            ]
        });
    let mut cfg = match config_from_checkpoint(&ckpt) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("checkpoint {}: {e}", ckpt.display());
            return 1;
        }
    };
    let ctx = (ctx / 64 * 64).max(64);
    cfg.seq_len = ctx;
    cfg.batch = 1;
    cfg.ce_chunk = 64;
    eprintln!(
        "[gen-check] {} layers, d {}, heads {}/{}, ff {}, vocab {}, structure: {}",
        cfg.n_layer,
        cfg.n_embd,
        cfg.n_head,
        cfg.n_kv,
        cfg.n_ff,
        cfg.vocab,
        cfg.structure()
    );
    let t0 = Instant::now();
    let mut tr = match Trainer::new_inference(cfg, 0)
        .and_then(|mut t| t.load_checkpoint(&ckpt).map(|_| t))
    {
        Ok(t) => t,
        Err(e) => {
            eprintln!("model: {e}");
            return 1;
        }
    };
    let mut dec = match Decoder::new(&tr, chunk) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("decoder: {e}");
            return 1;
        }
    };
    if args.iter().any(|a| a == "--q8") {
        if !no_ref {
            eprintln!("--q8 changes the weights: use it with --no-ref (the comparison with the fp16 forward pass would fail by design); `forge q8-eval` compares the two");
            return 2;
        }
        match dec.enable_q8(&mut tr, false) {
            Ok(i) => eprintln!(
                "[gen-check] Q8 weights: {:.0} MB of matrices quantized in {:.0} ms",
                i.matrix_bytes as f64 / 1e6,
                i.quantize_ms
            ),
            Err(e) => {
                eprintln!("Q8: {e}");
                return 1;
            }
        }
    }
    eprintln!("[gen-check] loaded in {:.1} s on {}; KV cache {:.1} MB for {} positions; single tokens through the {} path", t0.elapsed().as_secs_f64(), tr.device_info(), dec.cache_bytes() as f64 / 1e6, dec.max_ctx(), if dec.fast_path() { "fused" } else { "row" });
    let mut all_ok = true;
    let (mut old_tok, mut old_s, mut new_tok, mut new_s) = (0usize, 0f64, 0usize, 0f64);
    for (pi, prompt) in prompts.iter().enumerate() {
        let ids = tok.encode(prompt);
        if ids.is_empty() || ids.len() + n_new > ctx {
            eprintln!(
                "prompt {pi}: {} tokens + {n_new} do not fit the context {ctx}; skipped",
                ids.len()
            );
            continue;
        }
        if no_ref {
            // timing only: the cache path on its own greedy text
            dec.reset();
            let t = Instant::now();
            let mut l = dec.feed(&tr, &ids).expect("prefill");
            let dt_pre = t.elapsed().as_secs_f64();
            let t = Instant::now();
            for _ in 1..n_new {
                let x = argmax(&l);
                l = dec.feed(&tr, &[x]).expect("decode step");
            }
            let dt_new = t.elapsed().as_secs_f64();
            new_tok += n_new - 1;
            new_s += dt_new;
            eprintln!(
                "prompt {pi} ({} tokens): cache only: prefill {:.1} ms, decode {:.1} tokens/s",
                ids.len(),
                dt_pre * 1e3,
                (n_new - 1) as f64 / dt_new
            );
            continue;
        }
        // the no-cache reference: the trainer's forward over the whole text for every token
        let mut seq = ids.clone();
        let mut ref_logits: Vec<Vec<f32>> = Vec::with_capacity(n_new);
        let t = Instant::now();
        for _ in 0..n_new {
            let l = tr.logits_at(&seq, seq.len()).expect("forward");
            seq.push(argmax(&l));
            ref_logits.push(l);
        }
        let dt_old = t.elapsed().as_secs_f64();
        old_tok += n_new;
        old_s += dt_old;
        // the cache: prompt at once, then one token per step
        dec.reset();
        let t = Instant::now();
        let mut l = dec.feed(&tr, &ids).expect("prefill");
        let dt_pre = t.elapsed().as_secs_f64();
        let mut got: Vec<u32> = Vec::with_capacity(n_new);
        let (mut worst, mut n_same, mut min_gap, mut first_diff) =
            (0f32, 0usize, f32::INFINITY, None);
        let t = Instant::now();
        for i in 0..n_new {
            let (d, same) = compare(&l, &ref_logits[i]);
            worst = worst.max(d);
            n_same += same as usize;
            min_gap = min_gap.min(top2_gap(&ref_logits[i]));
            let next = argmax(&l);
            if next != seq[ids.len() + i] && first_diff.is_none() {
                first_diff = Some(i);
            }
            got.push(next);
            if i + 1 < n_new {
                // continue on the reference's text so that every position stays comparable even after a divergence
                l = dec.feed(&tr, &[seq[ids.len() + i]]).expect("decode step");
            }
        }
        let dt_new = t.elapsed().as_secs_f64();
        new_tok += n_new - 1;
        new_s += dt_new;
        // the prompt token by token must leave the same cache and logits as the prompt at once
        let snap_at_once = {
            let mut d2 = Decoder::new(&tr, chunk).expect("decoder");
            d2.feed(&tr, &ids).expect("prefill");
            d2.cache_snapshot(&tr).expect("snapshot")
        };
        let mut d3 = Decoder::new(&tr, chunk).expect("decoder");
        let mut l3 = Vec::new();
        for &x in &ids {
            l3 = d3.feed(&tr, &[x]).expect("step");
        }
        let snap_steps = d3.cache_snapshot(&tr).expect("snapshot");
        let (dp, samep) = compare(&l3, &ref_logits[0]);
        let cache_same = snap_at_once == snap_steps;
        let ok = first_diff.is_none() && worst <= 2e-3 && cache_same && dp <= 2e-3;
        all_ok &= ok;
        let text = tok.decode(&got, true);
        eprintln!(
            "prompt {pi} ({} tokens): {} greedy tokens {}; logits at {} positions: {} bitwise equal, max |diff| {:.3e}; smallest top-2 gap {:.3e}; prompt fed token by token: cache {}, logits {} (max |diff| {:.1e})",
            ids.len(),
            n_new,
            match first_diff {
                None => "IDENTICAL".to_string(),
                Some(i) => format!("DIFFER from position {i}"),
            },
            n_new,
            n_same,
            worst,
            min_gap,
            if cache_same { "identical" } else { "DIFFERENT" },
            if samep { "bitwise equal" } else { "differ" },
            dp
        );
        eprintln!("   old path {:.1} tokens/s | cache: prefill {:.1} ms, decode {:.1} tokens/s | text: {:?}", n_new as f64 / dt_old, dt_pre * 1e3, (n_new - 1) as f64 / dt_new, text.chars().take(160).collect::<String>());
    }
    if no_ref {
        println!(
            "short context: KV cache {:.1} tokens/s (no reference run)",
            new_tok as f64 / new_s
        );
    } else {
        println!(
            "short context: old path {:.1} tokens/s, KV cache {:.1} tokens/s ({:.1}x)",
            old_tok as f64 / old_s,
            new_tok as f64 / new_s,
            (new_tok as f64 / new_s) / (old_tok as f64 / old_s)
        );
    }
    if args.iter().any(|a| a == "--bench") {
        // a long context: fill the cache with real text (the prompts repeated), decode 64 tokens at that depth; the old path for a few tokens at the same depth
        let long: usize = flag(args, "--long")
            .and_then(|v| v.parse().ok())
            .unwrap_or(ctx.saturating_sub(80))
            .min(ctx - 66);
        let old_long: usize = flag(args, "--old-long")
            .and_then(|v| v.parse().ok())
            .unwrap_or(8);
        let mut fill: Vec<u32> = Vec::new();
        while fill.len() < long {
            for p in &prompts {
                fill.extend(tok.encode(p));
            }
        }
        fill.truncate(long);
        dec.reset();
        let t = Instant::now();
        let mut l = dec.feed(&tr, &fill).expect("prefill");
        let dt_pre = t.elapsed().as_secs_f64();
        let t = Instant::now();
        for _ in 0..64 {
            let x = argmax(&l);
            l = dec.feed(&tr, &[x]).expect("step");
        }
        let dt = t.elapsed().as_secs_f64();
        // the same 64 steps at a short depth for the ratio
        dec.reset();
        let mut l = dec.feed(&tr, &fill[..16]).expect("prefill");
        let t = Instant::now();
        for _ in 0..64 {
            let x = argmax(&l);
            l = dec.feed(&tr, &[x]).expect("step");
        }
        let dt_short = t.elapsed().as_secs_f64();
        let mut seq = fill.clone();
        let t = Instant::now();
        for _ in 0..old_long {
            let l = tr.logits_at(&seq, seq.len()).expect("forward");
            seq.push(argmax(&l));
        }
        let dt_old = t.elapsed().as_secs_f64().max(1e-9);
        println!(
            "long context ({} tokens in the cache): prefill {:.1} ms ({:.0} tokens/s); KV-cache decode {:.1} tokens/s ({:.2} ms per token) vs {:.1} tokens/s at depth 16; old path {:.2} tokens/s ({:.1} ms per token)",
            long,
            dt_pre * 1e3,
            long as f64 / dt_pre,
            64.0 / dt,
            dt * 1e3 / 64.0,
            64.0 / dt_short,
            old_long as f64 / dt_old,
            dt_old * 1e3 / old_long as f64
        );
    }
    if std::env::var("FORGE_KPROF").is_ok() {
        println!("{}", forge_cuda::kprof_report());
    }
    println!(
        "{}",
        if no_ref {
            "GEN-CHECK (timing only)"
        } else if all_ok {
            "GEN-CHECK PASSED"
        } else {
            "GEN-CHECK FAILED"
        }
    );
    if all_ok {
        0
    } else {
        1
    }
}
