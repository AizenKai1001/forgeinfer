//! `forge embed MODEL.gguf [--ids] [--json] [--dims N] [--cpu | --gpu] [--device N] TEXT...`: the embedding of each text, for checking an embedding model against another implementation.
//!   --ids   print the token ids of every text instead of embedding it
//!   --json  print the vectors as one JSON array of arrays (full precision)
//!   --dims  keep the first N values and normalise again
//!   --cpu / --gpu  where the encoder runs (default: the GPU when there is one that works, else the CPU); --device: the GPU's number (default 0)

use forge_embed::{truncate_normalized, Embedder, Encode};
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

/// Load an embedding model where `device` says: "cpu", "gpu" (an error when the card cannot be used) or "auto" (the GPU when it works and has room, else the CPU, with a note).
pub fn load_encoder(path: &Path, device: &str, ordinal: usize) -> Result<Arc<dyn Encode>, String> {
    let place = forge_embed_gpu::Place::parse(device).ok_or_else(|| format!("the embedding device {device:?} is not cpu, gpu or auto"))?;
    forge_embed_gpu::load_encoder(path, place, ordinal)
}

pub fn embed(args: &[String]) -> i32 {
    let Some(model) = args.get(1) else {
        eprintln!("usage: forge embed MODEL.gguf [--ids] [--json] [--dims N] [--repeat N] [--cpu | --gpu] [--device N] TEXT...");
        return 2;
    };
    let want_ids = args.iter().any(|a| a == "--ids");
    let json = args.iter().any(|a| a == "--json");
    let value = |name: &str| args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).and_then(|v| v.parse::<usize>().ok());
    let (dims, repeat, ordinal) = (value("--dims"), value("--repeat").unwrap_or(1).max(1), value("--device").unwrap_or(0));
    let device = if args.iter().any(|a| a == "--cpu") {
        "cpu"
    } else if args.iter().any(|a| a == "--gpu") {
        "gpu"
    } else {
        "auto"
    };
    let mut texts: Vec<&String> = Vec::new();
    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--ids" | "--json" | "--cpu" | "--gpu" => {}
            "--dims" | "--repeat" | "--device" => i += 1,
            _ => texts.push(&args[i]),
        }
        i += 1;
    }
    let t0 = Instant::now();
    if want_ids {
        let e = match Embedder::load(model) {
            Ok(e) => e,
            Err(err) => {
                eprintln!("{err}");
                return 1;
            }
        };
        for t in &texts {
            let ids = e.tokenizer().encode_truncated(t, e.config().ctx);
            let words: Vec<&str> = ids.iter().map(|&i| e.tokenizer().token(i).unwrap_or("?")).collect();
            println!("{:?}
  {:?}
  {}", t, ids, words.join(" "));
        }
        return 0;
    }
    let e = match load_encoder(Path::new(model), device, ordinal) {
        Ok(e) => e,
        Err(err) => {
            eprintln!("{err}");
            return 1;
        }
    };
    eprintln!("{} loaded in {:.2} s on {}: {} layers, {} dimensions, {} tokens of context", e.name(), t0.elapsed().as_secs_f64(), e.backend(), e.config().n_layer, e.dim(), e.config().ctx);
    let mut all = Vec::new();
    for t in &texts {
        let run = || -> Result<forge_embed::Embedding, String> { Ok(e.embed_many(&[t.as_str()])?.remove(0)) };
        let mut r = match run() {
            Ok(r) => r,
            Err(err) => {
                eprintln!("{err}");
                return 1;
            }
        };
        // the first call of a GPU backend pays for the lazy initialisation: time the repeats
        let t1 = Instant::now();
        for _ in 1..repeat {
            r = run().unwrap_or(r);
        }
        let ms = t1.elapsed().as_secs_f64() * 1e3 / (repeat.max(2) - 1) as f64;
        let v = match dims {
            Some(d) => truncate_normalized(&r.vector, d),
            None => r.vector.clone(),
        };
        if repeat > 1 {
            eprintln!("{:?}: {} tokens, {:.2} ms per text ({:.0} tokens/s), first values {:.5?}", t, r.tokens, ms, r.tokens as f64 / (ms / 1e3), &v[..v.len().min(6)]);
        } else {
            eprintln!("{:?}: {} tokens, first values {:.5?}", t, r.tokens, &v[..v.len().min(6)]);
        }
        all.push(v);
    }
    if json {
        let body: Vec<String> = all.iter().map(|v| format!("[{}]", v.iter().map(|x| format!("{x:.9}")).collect::<Vec<_>>().join(","))).collect();
        println!("[{}]", body.join(","));
    } else if all.len() >= 2 {
        for a in 0..all.len() {
            for b in a + 1..all.len() {
                println!("cosine({a}, {b}) = {:.5}", forge_embed::cosine(&all[a], &all[b]));
            }
        }
    }
    0
}
