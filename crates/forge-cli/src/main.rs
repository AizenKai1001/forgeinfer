//! `forge` command line (work in progress): run a model on the CPU reference path, tokenize text, inspect a model file.
//!
//!   forge run <model> [--prompt TEXT] [--tokens N] [--temp T]       generate (model = .gguf file or Hugging Face directory; tokenizer.json next to it for directories)
//!   forge tokenize <tokenizer.json> TEXT                              print the token ids
//!   forge info <model.gguf>                                           print metadata and tensors
//!   forge serve <model> [--native] [--port P] [--ctx N] ...          OpenAI-compatible HTTP server on the GPU

mod chat_cmd;
mod data_cmds;
mod eval_cmd;
mod eval_lambada;
mod gencheck_cmd;
mod kbench_cmd;
mod q8eval_cmd;
mod quant_cmd;
mod serve_cmd;
mod train_cmd;

use forge_gguf::Gguf;
use forge_model::{Model, Sampler, SamplerConfig, Session};
use forge_tok::Tokenizer;
use std::path::{Path, PathBuf};
use std::time::Instant;

fn usage() -> ! {
    eprintln!(
        "usage:\n  forge serve <model> [--port P] [--ctx N] ...   an OpenAI-compatible server (a GGUF brings its own tokenizer and chat template)\n  forge run <model> [--tokenizer FILE] [--prompt TEXT] [--tokens N] [--temp T] [--seed S]\n  forge tokenize <tokenizer.json | model.gguf> TEXT\n  forge info <model.gguf>                       metadata and tensor table of a GGUF file\n  forge tensor <model.gguf> NAME [R0 NR C0 NC]  statistics and a block of numbers of one weight matrix\n  forge gguf-stitch BASE RECIPE OUT             a GGUF mixed from several quantizations of one model\n  forge quantize | imatrix | hessian | ppl | eval-*   the quantizer and the evaluation tools\n  forge train | tok-train | tok-encode | export-hf   the trainer and its data tools"
    );
    std::process::exit(2);
}

fn flag<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str())
}

fn find_tokenizer(model: &Path, explicit: Option<&str>) -> PathBuf {
    if let Some(p) = explicit {
        return PathBuf::from(p);
    }
    if model.is_dir() {
        return model.join("tokenizer.json");
    }
    // a GGUF file: look for a tokenizer.json in the sibling Hugging Face folder of the same model family
    model
        .parent()
        .unwrap_or(Path::new("."))
        .join("tokenizer.json")
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(|s| s.as_str()) {
        Some("run") => {
            let model_path = PathBuf::from(args.get(1).unwrap_or_else(|| usage()));
            let tok_path = find_tokenizer(&model_path, flag(&args, "--tokenizer"));
            let prompt = flag(&args, "--prompt").unwrap_or("The capital of France is");
            let n: usize = flag(&args, "--tokens")
                .and_then(|v| v.parse().ok())
                .unwrap_or(48);
            let temp: f32 = flag(&args, "--temp")
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.0);
            let seed: u64 = flag(&args, "--seed")
                .and_then(|v| v.parse().ok())
                .unwrap_or(1);
            let t0 = Instant::now();
            let tok = Tokenizer::from_file(&tok_path).unwrap_or_else(|e| {
                eprintln!("tokenizer {}: {e}", tok_path.display());
                std::process::exit(1)
            });
            let native = args.iter().any(|a| a == "--native");
            let model = (if native {
                Model::load_native(&model_path)
            } else {
                Model::load(&model_path)
            })
            .unwrap_or_else(|e| {
                eprintln!("model {}: {e}", model_path.display());
                std::process::exit(1)
            });
            eprintln!(
                "loaded in {:.2} s: {:?}, {} layers, d {}, {:.0} M parameters",
                t0.elapsed().as_secs_f32(),
                model.cfg.arch,
                model.cfg.n_layer,
                model.cfg.n_embd,
                model.cfg.approx_params() as f64 / 1e6
            );
            let ids = tok.encode(prompt);
            let mut sampler = Sampler::new(SamplerConfig {
                temperature: temp,
                seed,
                ..Default::default()
            });
            let stop: Vec<u32> = ["<|im_end|>", "<|endoftext|>"]
                .iter()
                .filter_map(|s| tok.token_id(s))
                .collect();
            let use_gpu = args.iter().any(|a| a == "--gpu");
            let mut out: Vec<u32> = Vec::new();
            print!("{prompt}");
            if use_gpu {
                let fmt = if native {
                    forge_gpu::Fmt::Native
                } else if args.iter().any(|a| a == "--q8") {
                    forge_gpu::Fmt::Q8_0
                } else {
                    forge_gpu::Fmt::F16
                };
                let gpu = forge_gpu::GpuModel::upload(&model, fmt, 0).unwrap_or_else(|e| {
                    eprintln!("{e}");
                    std::process::exit(1)
                });
                let mut sess = forge_gpu::GpuSession::new(&gpu, ids.len() + n + 8).unwrap();
                let t1 = Instant::now();
                if ids.len() >= sess.prefill_batched_min() {
                    sess.prefill_batched(&ids).unwrap();
                } else {
                    sess.prefill(&ids).unwrap();
                }
                let mut logits = sess.logits().unwrap();
                let pre = t1.elapsed().as_secs_f64();
                let mut history = ids.clone();
                let t2 = Instant::now();
                for _ in 0..n {
                    let t = sampler.sample(&logits, &history);
                    if stop.contains(&t) {
                        break;
                    }
                    out.push(t);
                    history.push(t);
                    logits = sess.step(t).unwrap();
                }
                let dt = t2.elapsed().as_secs_f64();
                println!("{}", tok.decode(&out, true));
                eprintln!("[gpu] prefill {} tokens in {:.0} ms; decode {:.1} tokens/s (with CPU sampling)", ids.len(), pre * 1e3, out.len() as f64 / dt);
            } else {
                let mut sess = Session::new(&model, ids.len() + n + 8);
                let t1 = Instant::now();
                let mut logits = sess.prefill(&ids);
                let pre = t1.elapsed().as_secs_f64();
                let mut history = ids.clone();
                let t2 = Instant::now();
                for _ in 0..n {
                    let t = sampler.sample(&logits, &history);
                    if stop.contains(&t) {
                        break;
                    }
                    out.push(t);
                    history.push(t);
                    logits = sess.step(t, true).unwrap();
                }
                let dt = t2.elapsed().as_secs_f64();
                println!("{}", tok.decode(&out, true));
                eprintln!(
                    "[cpu] prefill {} tokens in {:.0} ms; decode {:.1} tokens/s",
                    ids.len(),
                    pre * 1e3,
                    out.len() as f64 / dt
                );
            }
        }
        Some("gpu-bench") => {
            // forge gpu-bench <model> [--tokens N] [--ctx C]: decode speed of the CUDA path (greedy, device-resident loop)
            let model_path = PathBuf::from(args.get(1).unwrap_or_else(|| usage()));
            let n: usize = flag(&args, "--tokens")
                .and_then(|v| v.parse().ok())
                .unwrap_or(256);
            let ctx: usize = flag(&args, "--ctx")
                .and_then(|v| v.parse().ok())
                .unwrap_or(1024);
            let native = args.iter().any(|a| a == "--native");
            let cpu = (if native {
                Model::load_native(&model_path)
            } else {
                Model::load(&model_path)
            })
            .unwrap_or_else(|e| {
                eprintln!("{e}");
                std::process::exit(1)
            });
            let t0 = Instant::now();
            let gfmt = if native {
                forge_gpu::Fmt::Native
            } else if args.iter().any(|a| a == "--q8") {
                forge_gpu::Fmt::Q8_0
            } else {
                forge_gpu::Fmt::F16
            };
            let gpu = forge_gpu::GpuModel::upload(&cpu, gfmt, 0).unwrap_or_else(|e| {
                eprintln!("{e}");
                std::process::exit(1)
            });
            eprintln!(
                "uploaded {:.0} MB in {:.2} s",
                gpu.weight_bytes as f64 / 1e6,
                t0.elapsed().as_secs_f32()
            );
            drop(cpu);
            let mut sess = forge_gpu::GpuSession::new(&gpu, ctx).unwrap();
            let prompt: Vec<u32> = (1..=8).collect();
            let mut best = 0f64;
            for rep in 0..5 {
                sess.reset().unwrap();
                sess.prefill(&prompt).unwrap();
                let t = Instant::now();
                sess.decode_greedy(n).unwrap();
                let dt = t.elapsed().as_secs_f64();
                let tps = n as f64 / dt;
                best = best.max(tps);
                eprintln!(
                    "run {rep}: {n} tokens in {:.1} ms = {:.0} tokens/s ({:.0} us per token)",
                    dt * 1e3,
                    tps,
                    dt * 1e6 / n as f64
                );
            }
            println!("best {best:.0} tokens/s");
        }
        Some("prefill-bench") => {
            // forge prefill-bench <model> [--tokens N]: batched tensor-core prompt processing speed (fp16 weights)
            let model_path = PathBuf::from(args.get(1).unwrap_or_else(|| usage()));
            let n: usize = flag(&args, "--tokens")
                .and_then(|v| v.parse().ok())
                .unwrap_or(512);
            let native = args.iter().any(|a| a == "--native");
            let cpu = (if native {
                Model::load_native(&model_path)
            } else {
                Model::load(&model_path)
            })
            .unwrap_or_else(|e| {
                eprintln!("{e}");
                std::process::exit(1)
            });
            let pfmt = if native {
                forge_gpu::Fmt::Native
            } else if args.iter().any(|a| a == "--q8") {
                forge_gpu::Fmt::Q8_0
            } else {
                forge_gpu::Fmt::F16
            };
            let gpu = forge_gpu::GpuModel::upload(&cpu, pfmt, 0).unwrap();
            drop(cpu);
            let mut sess = forge_gpu::GpuSession::new(&gpu, (n + 192).max(1024)).unwrap();
            let prompt: Vec<u32> = (0..n as u32).map(|i| 100 + (i * 7919) % 40000).collect();
            let mut best = 0f64;
            let reps: usize = flag(&args, "--reps")
                .and_then(|v| v.parse().ok())
                .unwrap_or(6);
            for rep in 0..reps {
                sess.reset().unwrap();
                let t = Instant::now();
                sess.prefill_batched(&prompt).unwrap();
                let dt = t.elapsed().as_secs_f64();
                best = best.max(n as f64 / dt);
                eprintln!(
                    "run {rep}: {n} tokens in {:.1} ms = {:.0} tokens/s",
                    dt * 1e3,
                    n as f64 / dt
                );
            }
            println!("best {best:.0} tokens/s");
            if std::env::var("FORGE_KPROF").is_ok() {
                // the per-kernel table of all the runs above (FORGE_KPROF=1 or 2)
                println!("{}", forge_cuda::kprof_report());
            }
            // decode speed at this depth (the prompt is in the cache)
            let t = Instant::now();
            let toks = sess.decode_greedy(129).unwrap();
            let dt = t.elapsed().as_secs_f64();
            eprintln!(
                "decode at depth {n}: 128 tokens in {:.1} ms = {:.1} tokens/s (first {:?})",
                dt * 1e3,
                128.0 / dt,
                &toks[..4]
            );
        }
        Some("export-hf") => {
            // forge export-hf <checkpoint> --out DIR [--tokenizer FILE] [--f32]
            let ck = PathBuf::from(args.get(1).unwrap_or_else(|| usage()));
            let out = PathBuf::from(flag(&args, "--out").unwrap_or("export"));
            match forge_train::export_hf(&ck, &out, !args.iter().any(|a| a == "--f32")) {
                Ok(cfg) => {
                    if let Some(t) = flag(&args, "--tokenizer") {
                        std::fs::copy(t, out.join("tokenizer.json"))
                            .expect("cannot copy the tokenizer");
                    }
                    eprintln!(
                        "exported {} layers, d {} to {}",
                        cfg.n_layer,
                        cfg.n_embd,
                        out.display()
                    );
                }
                Err(e) => {
                    eprintln!("{e}");
                    std::process::exit(1);
                }
            }
        }
        Some("eval-hellaswag") => std::process::exit(eval_cmd::hellaswag(&args)),
        Some("eval-mc") => std::process::exit(eval_cmd::mc(&args)),
        Some("eval-ppl") => std::process::exit(eval_cmd::ppl(&args)),
        Some("eval-lambada") => std::process::exit(eval_lambada::lambada(&args)),
        Some("chat") => std::process::exit(chat_cmd::chat(&args)),
        Some("gen-check") => std::process::exit(gencheck_cmd::gen_check(&args)),
        Some("q8-eval") => std::process::exit(q8eval_cmd::q8_eval(&args)),
        Some("ppl") => std::process::exit(quant_cmd::ppl(&args)),
        Some("imatrix") => std::process::exit(quant_cmd::imatrix(&args)),
        Some("hessian") => std::process::exit(quant_cmd::hessian(&args)),
        Some("quantize") => std::process::exit(quant_cmd::quantize(&args)),
        Some("cmp-hessians") => std::process::exit(quant_cmd::cmp_hessians(&args)),
        Some("mtp-agree") => std::process::exit(quant_cmd::mtp_agree(&args)),
        Some("serve") => std::process::exit(serve_cmd::serve(&args)),
        Some("train") => std::process::exit(train_cmd::train(&args)),
        Some("kbench") => std::process::exit(kbench_cmd::kbench(&args)),
        Some("tok-train") => std::process::exit(data_cmds::tok_train(&args)),
        Some("tok-encode") => std::process::exit(data_cmds::tok_encode(&args)),
        Some("chat-mask") => std::process::exit(data_cmds::chat_mask(&args)),
        Some("tokenize") => {
            // forge tokenize <tokenizer.json | model.gguf> TEXT: the token ids (a GGUF file is read for its own tokenizer)
            let path = args.get(1).unwrap_or_else(|| usage());
            let tok = if path.to_lowercase().ends_with(".gguf") {
                forge_gguf::Gguf::open(path)
                    .and_then(|g| Tokenizer::from_gguf(&g))
                    .unwrap_or_else(|e| {
                        eprintln!("{e}");
                        std::process::exit(1)
                    })
            } else {
                Tokenizer::from_file(path).unwrap_or_else(|e| {
                    eprintln!("{e}");
                    std::process::exit(1)
                })
            };
            let ids = tok.encode(args.get(2).unwrap_or_else(|| usage()));
            println!("{ids:?}");
        }
        Some("gguf-stitch") => {
            // forge gguf-stitch BASE.gguf RECIPE.json OUT.gguf: a GGUF made of the tensors of BASE, except those the recipe takes from other files of the same model (no re-quantization)
            let (base, recipe, out) = (
                args.get(1).unwrap_or_else(|| usage()),
                args.get(2).unwrap_or_else(|| usage()),
                args.get(3).unwrap_or_else(|| usage()),
            );
            match forge_model::weights::stitch(Path::new(base), Path::new(recipe), Path::new(out)) {
                Ok((n, bytes)) => println!("{out}: {n} tensors from other files, {:.2} GB written", bytes as f64 / 1e9),
                Err(e) => {
                    eprintln!("gguf-stitch: {e}");
                    std::process::exit(1)
                }
            }
        }
        Some("info") => {
            let g = Gguf::open(args.get(1).unwrap_or_else(|| usage())).unwrap_or_else(|e| {
                eprintln!("{e}");
                std::process::exit(1)
            });
            println!(
                "GGUF v{}, architecture {:?}, {} metadata keys, {} tensors, data at {}",
                g.version,
                g.architecture(),
                g.metadata.len(),
                g.tensors.len(),
                g.data_offset
            );
            for (k, v) in &g.metadata {
                let s = format!("{v:?}");
                println!(
                    "  {k} = {}",
                    if s.len() > 100 {
                        format!("{}...", &s[..100])
                    } else {
                        s
                    }
                );
            }
            for t in &g.tensors {
                println!("  {:44} {:8} {:?}", t.name, t.dtype.name(), t.shape());
            }
        }
        Some("tensor") => {
            // forge tensor MODEL.gguf NAME [ROW0 NROWS COL0 NCOLS]: one weight matrix of a GGUF file (dequantized to f32), its statistics and a block of its numbers, as JSON
            let (path, name) = (
                args.get(1).unwrap_or_else(|| usage()),
                args.get(2).unwrap_or_else(|| usage()),
            );
            let num = |i: usize, default: usize| args.get(i).and_then(|v| v.parse().ok()).unwrap_or(default);
            let g = Gguf::open(path).unwrap_or_else(|e| {
                eprintln!("{e}");
                std::process::exit(1)
            });
            let t = g.tensor_info(name).unwrap_or_else(|| {
                eprintln!("no tensor named {name:?} (forge info lists them)");
                std::process::exit(1)
            });
            let shape = t.shape();
            let data = g.tensor_f32(name).unwrap_or_else(|e| {
                eprintln!("{e}");
                std::process::exit(1)
            });
            let (rows, cols) = match shape.len() {
                0 => (1, 1),
                1 => (1, shape[0]),
                _ => (shape[0], shape[1..].iter().product::<usize>()),
            };
            let n = data.len().max(1) as f64;
            let mean = data.iter().map(|&v| v as f64).sum::<f64>() / n;
            let var = data.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / n;
            let (mut lo, mut hi) = (f32::MAX, f32::MIN);
            for &v in &data {
                lo = lo.min(v);
                hi = hi.max(v);
            }
            let (r0, nr, c0, nc) = (num(3, 0), num(4, 16), num(5, 0), num(6, 16));
            let (r1, c1) = ((r0 + nr).min(rows), (c0 + nc).min(cols));
            let mut block = String::from("[");
            for r in r0.min(rows)..r1 {
                if r > r0 {
                    block.push(',');
                }
                block.push('[');
                for c in c0.min(cols)..c1 {
                    if c > c0 {
                        block.push(',');
                    }
                    block.push_str(&format!("{:.5}", data[r * cols + c]));
                }
                block.push(']');
            }
            block.push(']');
            println!(
                "{{\"name\":{name:?},\"dtype\":{:?},\"shape\":{shape:?},\"count\":{},\"min\":{lo},\"max\":{hi},\"mean\":{mean:.6},\"std\":{:.6},\"block\":{block}}}",
                t.dtype.name(),
                data.len(),
                var.sqrt()
            );
        }
        _ => usage(),
    }
}
