//! `forge q8-eval`: Q8 weights against fp16 weights in the KV-cache decoder of a training checkpoint (the engine's `--q8`), on the same tokens.
//!
//!   forge q8-eval <checkpoint> --tokenizer FILE [--data FILE.u16 ...] [--windows 16] [--seq 512] [--tokens 64] [--no-ppl] [--no-greedy]
//!
//! 1. Perplexity and logits: for `--windows` windows of `--seq` + 1 tokens spread over every `--data` file (uint16 shards; a token with bit 15 set is an input but not a target, as in the chat
//!    shards) both decoders are fed the same tokens one at a time (the single-token kernels, as a server runs them); the negative log-likelihood of every target is summed for both and their
//!    paired difference is printed (nats per token, with its standard error), together with the logits error of the Q8 step against the fp16 step on the same prefix: the largest absolute
//!    difference relative to the largest fp16 logit and the mean absolute difference relative to the mean absolute fp16 logit (averaged over the steps), the top-1 agreement and the mean KL divergence.
//! 2. Greedy text: 20 varied chat prompts, `--tokens` tokens each, fp16 against Q8, each decoder continuing on its own text; how many texts are identical and where the others first differ. The
//!    teacher-forced rate (Q8's arg-max after the fp16 text so far) is the divergence-free number.

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

fn argmax(l: &[f32]) -> usize {
    let mut best = 0usize;
    for (i, &v) in l.iter().enumerate() {
        if v > l[best] {
            best = i;
        }
    }
    best
}

/// log-softmax of the logits into `out`
fn log_softmax(l: &[f32], out: &mut Vec<f64>) {
    let m = l.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
    let s: f64 = l.iter().map(|&x| (x as f64 - m).exp()).sum();
    let ls = s.ln() + m;
    out.clear();
    out.extend(l.iter().map(|&x| x as f64 - ls));
}

const PROMPTS: [&str; 20] = [
    "What is the capital of France?",
    "Explain in two sentences why the sky is blue.",
    "Write a Python function that returns the n-th Fibonacci number.",
    "Give me three tips for sleeping better.",
    "What is 17 times 23?",
    "Translate 'good morning, how are you?' into Spanish.",
    "Who wrote Romeo and Juliet, and when?",
    "Summarize the plot of Cinderella in a few lines.",
    "What is the difference between a virus and a bacterium?",
    "Write a haiku about autumn rain.",
    "How do I reverse a string in JavaScript?",
    "List four renewable sources of energy.",
    "Why do cats purr?",
    "What does a CPU do?",
    "Tell me a short joke about computers.",
    "Explain what photosynthesis is to a ten year old.",
    "What are the primary colors?",
    "How many days are there in a leap year?",
    "Describe the water cycle step by step.",
    "I feel stressed about an exam tomorrow. What should I do?",
];

pub fn q8_eval(args: &[String]) -> i32 {
    let Some(ckpt) = args.get(1).filter(|a| !a.starts_with("--")) else {
        eprintln!("usage: forge q8-eval <checkpoint> --tokenizer FILE [--data FILE.u16 ...] [--windows 16] [--seq 512] [--tokens 64] [--no-ppl] [--no-greedy]");
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
    let windows: usize = flag(args, "--windows")
        .and_then(|v| v.parse().ok())
        .unwrap_or(16);
    let seq: usize = flag(args, "--seq")
        .and_then(|v| v.parse().ok())
        .unwrap_or(512);
    let n_new: usize = flag(args, "--tokens")
        .and_then(|v| v.parse().ok())
        .unwrap_or(64);
    let files: Vec<&String> = args
        .iter()
        .skip_while(|a| a.as_str() != "--data")
        .skip(1)
        .take_while(|a| !a.starts_with("--"))
        .collect();
    let ctx = ((seq + 1).max(512 + n_new) + 63) / 64 * 64;
    let mut cfg = match config_from_checkpoint(&ckpt) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("checkpoint {}: {e}", ckpt.display());
            return 1;
        }
    };
    cfg.seq_len = ctx;
    cfg.batch = 1;
    cfg.ce_chunk = 32;
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
    let mut d16 = Decoder::new(&tr, 512).expect("decoder");
    let mut dq = Decoder::new(&tr, 512).expect("decoder");
    let info = match dq.enable_q8(&mut tr, false) {
        Ok(i) => i,
        Err(e) => {
            eprintln!("Q8: {e}");
            return 1;
        }
    };
    eprintln!(
        "[q8-eval] loaded in {:.1} s; Q8 matrices {:.0} MB quantized in {:.0} ms; {} layers, d {}; context {ctx}",
        t0.elapsed().as_secs_f64(),
        info.matrix_bytes as f64 / 1e6,
        info.quantize_ms,
        tr.cfg.n_layer,
        tr.cfg.n_embd
    );
    let vocab = tr.cfg.vocab;
    if !args.iter().any(|a| a == "--no-ppl") {
        for f in &files {
            let raw = std::fs::read(f).expect("cannot read the token file");
            let toks: Vec<u32> = raw
                .chunks_exact(2)
                .map(|c| u16::from_le_bytes([c[0], c[1]]) as u32)
                .collect();
            assert!(toks.len() > seq + 1, "{f} is too short");
            let span = toks.len() - (seq + 1);
            let t1 = Instant::now();
            let (mut n, mut s16, mut sq, mut sd, mut sd2) = (0usize, 0f64, 0f64, 0f64, 0f64);
            let (mut steps, mut agree, mut kl_sum) = (0usize, 0usize, 0f64);
            let (mut rel_max_sum, mut rel_mean_sum, mut rel_max_worst) = (0f64, 0f64, 0f64);
            let (mut l16, mut lq) = (Vec::new(), Vec::new());
            for wi in 0..windows {
                let start = span * wi / windows.max(1);
                let w: Vec<u32> = toks[start..start + seq + 1].to_vec();
                d16.reset();
                dq.reset();
                for i in 0..seq {
                    let id = w[i] & 0x7fff;
                    let z16 = d16.feed(&tr, &[id]).expect("fp16 step");
                    let zq = dq.feed(&tr, &[id]).expect("q8 step");
                    // logits error of this step
                    let (mut mx, mut sum_d, mut sum_a, mut amax) = (0f64, 0f64, 0f64, 0f64);
                    for (a, b) in z16.iter().zip(&zq) {
                        let d = (*a - *b).abs() as f64;
                        mx = mx.max(d);
                        sum_d += d;
                        sum_a += a.abs() as f64;
                        amax = amax.max(a.abs() as f64);
                    }
                    rel_max_sum += mx / amax.max(1e-9);
                    rel_mean_sum += sum_d / sum_a.max(1e-9);
                    rel_max_worst = rel_max_worst.max(mx / amax.max(1e-9));
                    steps += 1;
                    agree += (argmax(&z16) == argmax(&zq)) as usize;
                    log_softmax(&z16, &mut l16);
                    log_softmax(&zq, &mut lq);
                    kl_sum += l16
                        .iter()
                        .zip(&lq)
                        .map(|(a, b)| a.exp() * (a - b))
                        .sum::<f64>();
                    // the target: bit 15 marks an input that is not a target
                    let t = w[i + 1];
                    if t & 0x8000 == 0 && (t as usize) < vocab {
                        let (a, b) = (-l16[t as usize], -lq[t as usize]);
                        n += 1;
                        s16 += a;
                        sq += b;
                        sd += b - a;
                        sd2 += (b - a) * (b - a);
                    }
                }
            }
            let nf = n.max(1) as f64;
            let mean_d = sd / nf;
            let se = ((sd2 / nf - mean_d * mean_d).max(0.0) / nf).sqrt();
            println!(
                "{}: {} scored tokens in {} windows of {}: fp16 {:.4} nats (perplexity {:.2}), Q8 {:.4} nats (perplexity {:.2}); Q8 - fp16 {:+.4} +- {:.4} nats ({:+.3} % perplexity)   [{:.0} s]",
                std::path::Path::new(f).file_stem().and_then(|s| s.to_str()).unwrap_or("data"),
                n,
                windows,
                seq,
                s16 / nf,
                (s16 / nf).exp(),
                sq / nf,
                (sq / nf).exp(),
                mean_d,
                se,
                100.0 * ((sq / nf).exp() / (s16 / nf).exp() - 1.0),
                t1.elapsed().as_secs_f64()
            );
            println!(
                "   logits of the Q8 step against the fp16 step ({} steps): largest |diff| / largest |logit|: mean {:.3e}, worst {:.3e}; mean |diff| / mean |logit|: {:.3e}; top-1 agreement {:.2} %; mean KL(fp16 || Q8) {:.2e} nats",
                steps,
                rel_max_sum / steps as f64,
                rel_max_worst,
                rel_mean_sum / steps as f64,
                100.0 * agree as f64 / steps as f64,
                kl_sum / steps as f64
            );
        }
    }
    if !args.iter().any(|a| a == "--no-greedy") {
        let end_id = tok.token_id("<|im_end|>");
        let t1 = Instant::now();
        let (mut identical, mut diffs) = (0usize, Vec::new());
        let (mut tf_agree, mut tf_total) = (0usize, 0usize);
        let mut first_eos_diff = 0usize;
        for (pi, p) in PROMPTS.iter().enumerate() {
            let ids = tok.encode(&format!(
                "<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n"
            ));
            let run = |d: &mut Decoder, tr: &Trainer| -> Vec<u32> {
                d.reset();
                let mut l = d.feed(tr, &ids).expect("prompt");
                let mut out = Vec::new();
                for i in 0..n_new {
                    let x = argmax(&l) as u32;
                    out.push(x);
                    if i + 1 < n_new {
                        l = d.feed(tr, &[x]).expect("step");
                    }
                }
                out
            };
            let a = run(&mut d16, &tr);
            let b = run(&mut dq, &tr);
            // the text up to the first end-of-turn token is the answer; what follows is whatever the model drifts into
            let eos_at = end_id
                .and_then(|e| a.iter().position(|&x| x == e))
                .unwrap_or(n_new);
            match a.iter().zip(&b).position(|(x, y)| x != y) {
                None => identical += 1,
                Some(i) => {
                    diffs.push((pi, i));
                    if i < eos_at {
                        first_eos_diff += 1;
                    }
                }
            }
            // teacher forcing: the Q8 decoder follows the fp16 text, its arg-max is compared at every step
            dq.reset();
            let mut l = dq.feed(&tr, &ids).expect("prompt");
            for i in 0..n_new {
                tf_total += 1;
                tf_agree += (argmax(&l) as u32 == a[i]) as usize;
                if i + 1 < n_new {
                    l = dq.feed(&tr, &[a[i]]).expect("step");
                }
            }
            if pi < 3 {
                println!(
                    "   prompt {pi} fp16: {:?}",
                    tok.decode(&a, true).chars().take(110).collect::<String>()
                );
                println!(
                    "   prompt {pi} Q8  : {:?}",
                    tok.decode(&b, true).chars().take(110).collect::<String>()
                );
            }
        }
        println!(
            "greedy, {} prompts x {} tokens: {} of {} texts identical; first divergences (prompt: token): {}; {} of the divergences are inside the answer (before the first <|im_end|>); teacher-forced arg-max agreement {:.2} % ({} of {})   [{:.0} s]",
            PROMPTS.len(),
            n_new,
            identical,
            PROMPTS.len(),
            diffs
                .iter()
                .map(|(p, i)| format!("{p}:{i}"))
                .collect::<Vec<_>>()
                .join(" "),
            first_eos_diff,
            100.0 * tf_agree as f64 / tf_total as f64,
            tf_agree,
            tf_total,
            t1.elapsed().as_secs_f64()
        );
    }
    0
}
