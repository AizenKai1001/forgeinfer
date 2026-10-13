//! `forge chat`: talk to a training checkpoint (any Forge-1 structure) through the trainer's own kernels with a KV cache (`forge_train::Decoder`: bitwise the logits of the trainer's forward pass;
//! the conversation so far stays in the cache between turns). `--no-kv` runs the old path, the whole window again for every new token.
//!
//!   forge chat <checkpoint> --tokenizer FILE --prompt "The capital of France is" [--tokens 80] [--temp 0.8] [--top-p 0.95] [--seq 1024] [--seed 1]    (plain continuation)
//!   forge chat <checkpoint> --tokenizer FILE --chat "What is 2 + 2?" [--system "..."]                                                                (ChatML: <|im_start|>user ... <|im_end|> <|im_start|>assistant)
//!   forge chat <checkpoint> --tokenizer FILE --interactive                                                                                           (reads lines from stdin, keeps the conversation)

use forge_tok::Tokenizer;
use forge_train::{config_from_checkpoint, Decoder, Trainer};
use std::io::{BufRead, Write};
use std::path::PathBuf;

fn flag<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str())
}

fn splitmix(x: &mut u64) -> u64 {
    *x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Sample from the logits: temperature, then nucleus (top-p) filtering; temperature 0 = arg-max.
fn sample(logits: &[f32], temp: f32, top_p: f32, rng: &mut u64) -> u32 {
    if temp <= 0.0 {
        return logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i as u32)
            .unwrap();
    }
    let mx = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<(usize, f32)> = logits
        .iter()
        .enumerate()
        .map(|(i, &l)| (i, ((l - mx) / temp).exp()))
        .collect();
    let sum: f32 = probs.iter().map(|p| p.1).sum();
    probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let mut cum = 0.0;
    let mut keep = probs.len();
    for (k, p) in probs.iter().enumerate() {
        cum += p.1 / sum;
        if cum >= top_p {
            keep = k + 1;
            break;
        }
    }
    let total: f32 = probs[..keep].iter().map(|p| p.1).sum();
    let r = (splitmix(rng) >> 11) as f32 / (1u64 << 53) as f32 * total;
    let mut acc = 0.0;
    for p in &probs[..keep] {
        acc += p.1;
        if acc >= r {
            return p.0 as u32;
        }
    }
    probs[0].0 as u32
}

pub fn chat(args: &[String]) -> i32 {
    let ckpt = PathBuf::from(args.get(1).expect("usage: forge chat <checkpoint> --tokenizer FILE (--prompt TEXT | --chat TEXT | --interactive)"));
    let tok = Tokenizer::from_file(flag(args, "--tokenizer").expect("--tokenizer is required"))
        .expect("cannot load the tokenizer");
    let max_new: usize = flag(args, "--tokens")
        .and_then(|v| v.parse().ok())
        .unwrap_or(80);
    let temp: f32 = flag(args, "--temp")
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.8);
    let top_p: f32 = flag(args, "--top-p")
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.95);
    let seq: usize = flag(args, "--seq")
        .and_then(|v| v.parse().ok())
        .unwrap_or(1024);
    let mut rng: u64 = flag(args, "--seed")
        .and_then(|v| v.parse().ok())
        .unwrap_or(1);
    let mut cfg = config_from_checkpoint(&ckpt).expect("cannot read the checkpoint header");
    cfg.seq_len = seq;
    cfg.batch = 1;
    cfg.ce_chunk = 32;
    let mut tr = Trainer::new_inference(cfg, 0).expect("cannot create the model");
    tr.load_checkpoint(&ckpt)
        .expect("cannot load the checkpoint");
    let mut dec = if args.iter().any(|a| a == "--no-kv") {
        None
    } else {
        match Decoder::new(&tr, 512) {
            Ok(d) => Some(d),
            Err(e) => {
                eprintln!("no KV cache: {e}");
                None
            }
        }
    };
    let eos = tok.token_id("<|endoftext|>").unwrap_or(0);
    let im_start = tok.token_id("<|im_start|>");
    let im_end = tok.token_id("<|im_end|>");

    // generate until a stop token or `max_new` tokens; the new tokens are printed as they come
    let mut generate = |tr: &mut Trainer, ids: &mut Vec<u32>, stops: &[u32]| {
        let mut bytes: Vec<u8> = Vec::new();
        let mut printed = 0usize;
        for _ in 0..max_new {
            if ids.len() >= seq {
                break;
            }
            let logits = match dec.as_mut() {
                Some(d) => {
                    // the cache already holds the shared start of `ids` (earlier turns, the tokens generated so far): feed only the rest
                    let lcp = d
                        .ids()
                        .iter()
                        .zip(ids.iter())
                        .take_while(|(a, b)| a == b)
                        .count()
                        .min(ids.len() - 1);
                    d.truncate(lcp);
                    d.feed(tr, &ids[lcp..]).expect("forward failed")
                }
                None => tr.logits_at(ids, ids.len()).expect("forward failed"),
            };
            let next = sample(&logits, temp, top_p, &mut rng);
            if stops.contains(&next) {
                break;
            }
            ids.push(next);
            bytes.extend(tok.decode_bytes(&[next], true));
            let valid = match std::str::from_utf8(&bytes[printed..]) {
                Ok(_) => bytes.len(),
                Err(e) => printed + e.valid_up_to(),
            };
            print!("{}", String::from_utf8_lossy(&bytes[printed..valid]));
            std::io::stdout().flush().ok();
            printed = valid;
        }
        println!();
    };
    let render = |role: &str, text: &str| -> Vec<u32> {
        let mut v = Vec::new();
        v.extend(im_start);
        v.extend(tok.encode(&format!("{role}\n{text}")));
        v.extend(im_end);
        v.extend(tok.encode("\n"));
        v
    };
    let stops: Vec<u32> = [Some(eos), im_end].iter().flatten().cloned().collect();
    if let Some(p) = flag(args, "--prompt") {
        let mut ids = tok.encode(p);
        print!("{p}");
        generate(&mut tr, &mut ids, &[eos]);
    } else {
        let mut ids: Vec<u32> = Vec::new();
        if let Some(sys) = flag(args, "--system") {
            ids.extend(render("system", sys));
        }
        let mut turn = |tr: &mut Trainer, ids: &mut Vec<u32>, user: &str| {
            ids.extend(render("user", user));
            ids.extend(im_start);
            ids.extend(tok.encode("assistant\n"));
            generate(tr, ids, &stops);
            ids.extend(im_end);
            ids.extend(tok.encode("\n"));
        };
        if let Some(m) = flag(args, "--chat") {
            turn(&mut tr, &mut ids, m);
        } else {
            // interactive: one user turn per line of stdin
            eprint!("> ");
            for line in std::io::stdin().lock().lines() {
                let line = line.unwrap_or_default();
                if line.trim().is_empty() {
                    break;
                }
                turn(&mut tr, &mut ids, line.trim());
                eprint!("> ");
            }
        }
    }
    0
}
