//! Quality measurements of (quantized) weights on the GPU.
//!
//!   forge ppl <model> --tokenizer FILE --text FILE[,FILE..] [--native | --q8] [--ctx 512] [--max-tokens 131072] [--dump-ref FILE[,..]] [--ref FILE[,..]]
//!                      [--quant-like FILE.gguf [--weighting uniform|magnitude|importance|impmag] [--imatrix FILE] [--gptq HESSIANS [--gptq-damp 0.01]] [--override "token_embd=Q5_K;ffn_down=Q4_K"] [--no-refine]]
//!
//! With `--quant-like`, a Hugging Face model is quantized in memory by our own quantizer (forge-quant) to the format llama.cpp's file `FILE.gguf` uses for every tensor, and then measured:
//! same sizes as that file, so only the quality of the encoder differs.
//!
//! The text is cut into windows of `ctx` tokens; every window is run from an empty state and the logits after every token (batched prefill path, fp16-operand GEMMs, the same numerics the
//! server uses for prompts) give the negative log-likelihood of the next token. `--dump-ref` writes the reference model's top-64 next-token distribution per position; a second run with
//! `--ref` reports how far the quantized model's distribution is from it: KL(reference || model) over the reference's top 64 tokens, the share of positions with the same top-1 token, and the
//! change of the log-likelihood.

use forge_core::Json;
use forge_gguf::{DType, Gguf};
use forge_gpu::{Fmt, GpuModel, GpuSession, EVAL_ROWS};
use forge_model::{Mat, Model, Raw};
use forge_quant::{encode_matrix, gptq_matrix, EncodeOpts, GptqOpts, Weighting};
use forge_tok::Tokenizer;
use std::collections::HashMap;
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

const TOPK: usize = 64;

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
    model
        .parent()
        .unwrap_or(Path::new("."))
        .join("tokenizer.json")
}

/// A text file; invalid UTF-8 (a cut in the middle of a multi-byte character) becomes U+FFFD instead of an error.
fn read_text(path: &str) -> String {
    String::from_utf8_lossy(
        &std::fs::read(path).unwrap_or_else(|e| panic!("cannot read {path}: {e}")),
    )
    .to_string()
}

/// log-sum-exp of a row of logits
fn logsumexp(row: &[f32]) -> f64 {
    let m = row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let s: f64 = row.iter().map(|&x| ((x - m) as f64).exp()).sum();
    m as f64 + s.ln()
}

/// the TOPK largest logits of a row: (token, logit), best first
fn top_k(row: &[f32]) -> Vec<(u32, f32)> {
    let mut idx: Vec<u32> = (0..row.len() as u32).collect();
    idx.select_nth_unstable_by(TOPK, |&a, &b| {
        row[b as usize]
            .partial_cmp(&row[a as usize])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut top: Vec<(u32, f32)> = idx[..TOPK].iter().map(|&i| (i, row[i as usize])).collect();
    top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    top
}

/// What is kept of the reference model at every position.
struct RefRow {
    ids: [u32; TOPK],
    logp: [f32; TOPK],
    /// log-probability of the actual next token
    tgt: f32,
}

fn read_ref(path: &str) -> Vec<RefRow> {
    let mut f = std::fs::File::open(path).unwrap_or_else(|e| panic!("cannot open {path}: {e}"));
    let mut head = [0u8; 16];
    f.read_exact(&mut head).unwrap();
    assert_eq!(&head[..8], b"FQREF002", "not a reference dump");
    let k = u32::from_le_bytes(head[8..12].try_into().unwrap()) as usize;
    let n = u32::from_le_bytes(head[12..16].try_into().unwrap()) as usize;
    assert_eq!(k, TOPK);
    let stride = TOPK * 8 + 4;
    let mut buf = vec![0u8; n * stride];
    f.read_exact(&mut buf).unwrap();
    (0..n)
        .map(|i| {
            let b = &buf[i * stride..(i + 1) * stride];
            let mut r = RefRow {
                ids: [0; TOPK],
                logp: [0.0; TOPK],
                tgt: f32::from_le_bytes(b[TOPK * 8..TOPK * 8 + 4].try_into().unwrap()),
            };
            for j in 0..TOPK {
                r.ids[j] = u32::from_le_bytes(b[4 * j..4 * j + 4].try_into().unwrap());
                r.logp[j] = f32::from_le_bytes(
                    b[4 * TOPK + 4 * j..4 * TOPK + 4 * j + 4]
                        .try_into()
                        .unwrap(),
                );
            }
            r
        })
        .collect()
}

/// All weight matrices of a model with the names the GGUF files give them.
fn named_matrices(model: &mut Model) -> Vec<(String, &mut Mat)> {
    let mut v: Vec<(String, &mut Mat)> = Vec::new();
    v.push(("token_embd.weight".into(), &mut model.tok_emb));
    if let Some(h) = model.lm_head.as_mut() {
        v.push(("output.weight".into(), h));
    }
    for (i, l) in model.layers.iter_mut().enumerate() {
        let b = format!("blk.{i}");
        if let Some(d) = l.delta.as_mut() {
            v.push((format!("{b}.attn_qkv.weight"), &mut d.w_qkv));
            v.push((format!("{b}.attn_gate.weight"), &mut d.w_z));
            v.push((format!("{b}.ssm_alpha.weight"), &mut d.w_a));
            v.push((format!("{b}.ssm_beta.weight"), &mut d.w_b));
            v.push((format!("{b}.ssm_out.weight"), &mut d.w_out));
        } else {
            v.push((format!("{b}.attn_q.weight"), &mut l.wq));
            v.push((format!("{b}.attn_k.weight"), &mut l.wk));
            v.push((format!("{b}.attn_v.weight"), &mut l.wv));
            v.push((format!("{b}.attn_output.weight"), &mut l.wo));
        }
        v.push((format!("{b}.ffn_gate.weight"), &mut l.w_gate));
        v.push((format!("{b}.ffn_up.weight"), &mut l.w_up));
        v.push((format!("{b}.ffn_down.weight"), &mut l.w_down));
    }
    v
}

/// The format of every tensor of a GGUF file.
fn formats_of(path: &str) -> HashMap<String, DType> {
    let g = Gguf::open(path).unwrap_or_else(|e| panic!("cannot open {path}: {e}"));
    g.tensors
        .iter()
        .map(|t| (t.name.clone(), t.dtype))
        .collect()
}

/// `--override "token_embd=Q5_K;ffn_down=Q4_K"`: change the format of every tensor whose name contains the text before `=` (after the formats of the template file were read); with a leading ^ the name must start with it.
fn apply_overrides(types: &mut HashMap<String, DType>, spec: Option<&str>) {
    let Some(spec) = spec else { return };
    for item in spec.split(';').filter(|s| !s.is_empty()) {
        let (pat, f) = item
            .split_once('=')
            .unwrap_or_else(|| panic!("bad override {item}"));
        let dt = match f {
            "Q8_0" => DType::Q8_0,
            "Q4_0" => DType::Q4_0,
            "Q5_0" => DType::Q5_0,
            "Q4_K" => DType::Q4_K,
            "Q5_K" => DType::Q5_K,
            "Q6_K" => DType::Q6_K,
            other => panic!("unknown format {other}"),
        };
        let mut n = 0;
        for (name, t) in types.iter_mut() {
            // (a pattern that starts with ^ must match the start of the name: "^output.weight" is the head, "output.weight" would also be every attn_output.weight)
            let hit = match pat.strip_prefix('^') {
                Some(prefix) => name.starts_with(prefix),
                None => name.contains(pat),
            };
            if hit {
                *t = dt;
                n += 1;
            }
        }
        eprintln!("[quant] override {pat} -> {f}: {n} tensors");
    }
}

/// Per-input-column importance of the linear layers (mean square of the activations that multiply each column), by tensor name; file written by `forge imatrix`.
fn read_importance(path: &str) -> HashMap<String, Vec<f32>> {
    let b = std::fs::read(path).unwrap_or_else(|e| panic!("cannot read {path}: {e}"));
    assert_eq!(&b[..8], b"FQIMAT01", "not an importance file");
    let n = u32::from_le_bytes(b[8..12].try_into().unwrap()) as usize;
    let mut p = 12;
    let mut m = HashMap::new();
    for _ in 0..n {
        let nl = u32::from_le_bytes(b[p..p + 4].try_into().unwrap()) as usize;
        let name = String::from_utf8_lossy(&b[p + 4..p + 4 + nl]).to_string();
        p += 4 + nl;
        let len = u32::from_le_bytes(b[p..p + 4].try_into().unwrap()) as usize;
        p += 4;
        let v: Vec<f32> = b[p..p + 4 * len]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        p += 4 * len;
        m.insert(name, v);
    }
    m
}

/// In a model with a separate output head the input embedding (token_embd) is a lookup table: the second moments `forge hessian` stores under "head" are the statistics of the inputs of the OUTPUT head,
/// the wrong objective for it (quantizing the 9B's embedding with them cost about 6 % of its KL; llama.cpp also leaves token_embd out of its importance matrix). It is rounded plainly then.
fn plain_embedding(gname: &str, untied: bool) -> bool {
    untied && gname == "token_embd.weight"
}

/// The key under which `forge hessian` stores the second moment of the inputs of the tensor `name`.
fn hessian_key(name: &str) -> String {
    if name == "token_embd.weight" || name == "output.weight" {
        return "head".to_string();
    }
    let mut it = name.splitn(3, '.');
    let (_, layer, rest) = (it.next(), it.next().unwrap_or(""), it.next().unwrap_or(""));
    let role = match rest.trim_end_matches(".weight") {
        "attn_q" | "attn_k" | "attn_v" | "attn_qkv" | "attn_gate" | "ssm_alpha" | "ssm_beta" => {
            "attn_in"
        }
        "attn_output" | "ssm_out" => "out_in",
        "ffn_gate" | "ffn_up" => "ffn_in",
        "ffn_down" => "down_in",
        _ => "none",
    };
    format!("blk.{layer}.{role}")
}

/// Second moments written by `forge hessian`, left on disk: an index of key -> (file, offset, width); a matrix is read when a layer needs it (the second moments of a 9B model are 22 GB).
/// Files are named as a list separated by commas, or with one `*` pattern (every file of the directory that starts with the text before it and ends with the text after it).
struct HessianStore {
    index: HashMap<String, (PathBuf, u64, usize)>,
    /// the last matrix read (the layers' tensors share their inputs: attention / DeltaNet projections, gate and up)
    last: Option<(String, usize, std::rc::Rc<Vec<f32>>)>,
}

impl HessianStore {
    fn open(paths: &str) -> HessianStore {
        let mut files: Vec<PathBuf> = Vec::new();
        for path in paths.split(',') {
            if let Some((pre, post)) = path.split_once('*') {
                let dir = Path::new(pre)
                    .parent()
                    .map(|p| p.to_path_buf())
                    .unwrap_or_else(|| PathBuf::from("."));
                let stem = Path::new(pre)
                    .file_name()
                    .map(|s| s.to_string_lossy().to_string())
                    .unwrap_or_default();
                let mut found: Vec<PathBuf> = std::fs::read_dir(&dir)
                    .expect("cannot list the Hessian directory")
                    .flatten()
                    .map(|e| e.path())
                    .filter(|p| {
                        p.file_name()
                            .map(|n| {
                                n.to_string_lossy().starts_with(&stem)
                                    && n.to_string_lossy().ends_with(post)
                            })
                            .unwrap_or(false)
                    })
                    .collect();
                found.sort();
                files.extend(found);
            } else {
                files.push(PathBuf::from(path));
            }
        }
        let mut index = HashMap::new();
        for p in files {
            let mut f = std::io::BufReader::new(
                std::fs::File::open(&p)
                    .unwrap_or_else(|e| panic!("cannot open {}: {e}", p.display())),
            );
            let mut head = [0u8; 12];
            f.read_exact(&mut head).unwrap();
            assert_eq!(&head[..8], b"FQHESS01", "not a Hessian file");
            let n = u32::from_le_bytes(head[8..12].try_into().unwrap()) as usize;
            let mut at = 12u64;
            for _ in 0..n {
                let mut b4 = [0u8; 4];
                f.read_exact(&mut b4).unwrap();
                let mut name = vec![0u8; u32::from_le_bytes(b4) as usize];
                f.read_exact(&mut name).unwrap();
                f.read_exact(&mut b4).unwrap();
                let dim = u32::from_le_bytes(b4) as usize;
                at += 8 + name.len() as u64;
                index.insert(
                    String::from_utf8_lossy(&name).to_string(),
                    (p.clone(), at, dim),
                );
                let bytes = (dim * dim * 4) as u64;
                f.seek_relative(bytes as i64).unwrap();
                at += bytes;
            }
        }
        eprintln!("[quant] second moments of {} inputs indexed", index.len());
        HessianStore { index, last: None }
    }

    /// (width, matrix) of an input, or None if there are no second moments for it.
    fn get(&mut self, key: &str) -> Option<(usize, std::rc::Rc<Vec<f32>>)> {
        if let Some((k, d, v)) = &self.last {
            if k == key {
                return Some((*d, v.clone()));
            }
        }
        let (path, at, dim) = self.index.get(key)?.clone();
        let mut f = std::fs::File::open(&path)
            .unwrap_or_else(|e| panic!("cannot open {}: {e}", path.display()));
        f.seek(SeekFrom::Start(at)).unwrap();
        let mut raw = vec![0u8; dim * dim * 4];
        f.read_exact(&mut raw)
            .unwrap_or_else(|e| panic!("{}: {key}: {e}", path.display()));
        let v: std::rc::Rc<Vec<f32>> = std::rc::Rc::new(
            raw.chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect(),
        );
        self.last = Some((key.to_string(), dim, v.clone()));
        Some((dim, v))
    }
}

/// Quantize every matrix whose format in `types` is one of ours, in place (the GPU upload then reads the blocks natively). Returns (matrices, weights, bytes).
fn quantize_model(
    model: &mut Model,
    types: &HashMap<String, DType>,
    imp: Option<&HashMap<String, Vec<f32>>>,
    mut hess: Option<&mut HessianStore>,
    opts: &EncodeOpts,
    gopts: &GptqOpts,
) -> (usize, usize, usize) {
    let (mut n, mut weights, mut bytes) = (0, 0, 0);
    let t0 = Instant::now();
    for (name, m) in named_matrices(model) {
        if m.rows == 0 {
            continue;
        }
        let Some(&dt) = types.get(&name) else {
            eprintln!("[quant] no format for {name}: left as it is");
            continue;
        };
        if !matches!(
            dt,
            DType::Q8_0 | DType::Q4_0 | DType::Q5_0 | DType::Q4_K | DType::Q5_K | DType::Q6_K
        ) {
            eprintln!(
                "[quant] {name}: {} is not one of ours, left as it is",
                dt.name()
            );
            continue;
        }
        let plain = plain_embedding(&name, types.contains_key("output.weight"));
        let iv = imp
            .and_then(|h| h.get(&name))
            .map(|v| v.as_slice())
            .filter(|_| !plain);
        if matches!(
            opts.weighting,
            Weighting::Importance | Weighting::ImportanceMagnitude
        ) && iv.is_none()
            && !plain
        {
            panic!("no importance for {name}");
        }
        let hv = hess
            .as_mut()
            .and_then(|h| h.get(&hessian_key(&name)))
            .filter(|(d, _)| *d == m.cols)
            .filter(|_| !plain);
        let uniform = EncodeOpts {
            weighting: Weighting::Uniform,
            refine: opts.refine,
            threads: opts.threads,
        };
        let enc = match hv {
            Some((_, h)) => gptq_matrix(dt, m.rows, m.cols, &m.w, &h, opts, gopts)
                .unwrap_or_else(|e| panic!("{name}: {e}")),
            None => encode_matrix(
                dt,
                m.rows,
                m.cols,
                &m.w,
                iv,
                if plain { &uniform } else { opts },
            )
            .unwrap_or_else(|e| panic!("{name}: {e}")),
        };
        n += 1;
        weights += m.rows * m.cols;
        bytes += enc.len();
        m.raw = Some(Raw {
            dtype: dt,
            bytes: enc.into(),
        });
    }
    eprintln!(
        "[quant] {n} matrices, {:.1} M weights -> {:.1} MB ({:.3} bits per weight) in {:.1} s",
        weights as f64 / 1e6,
        bytes as f64 / 1e6,
        bytes as f64 * 8.0 / weights as f64,
        t0.elapsed().as_secs_f32()
    );
    (n, weights, bytes)
}

pub fn ppl(args: &[String]) -> i32 {
    let model_path = PathBuf::from(args.get(1).expect("usage: forge ppl <model> --tokenizer FILE --text FILE[,FILE..] [--native|--q8] [--ctx N] [--max-tokens N] [--dump-ref FILE[,..]] [--ref FILE[,..]]"));
    let tok = Tokenizer::from_file(find_tokenizer(&model_path, flag(args, "--tokenizer")))
        .expect("cannot load the tokenizer");
    let texts: Vec<&str> = flag(args, "--text")
        .expect("--text is required")
        .split(',')
        .collect();
    let refs: Vec<Option<&str>> = match flag(args, "--ref") {
        Some(r) => r.split(',').map(Some).collect(),
        None => vec![None; texts.len()],
    };
    let dumps: Vec<Option<&str>> = match flag(args, "--dump-ref") {
        Some(r) => r.split(',').map(Some).collect(),
        None => vec![None; texts.len()],
    };
    assert!(
        refs.len() == texts.len() && dumps.len() == texts.len(),
        "--ref and --dump-ref need one file per text"
    );
    let ctx: usize = flag(args, "--ctx")
        .and_then(|v| v.parse().ok())
        .unwrap_or(512);
    let max_tokens: usize = flag(args, "--max-tokens")
        .and_then(|v| v.parse().ok())
        .unwrap_or(131072);
    let native = args.iter().any(|a| a == "--native");
    let t0 = Instant::now();
    let cpu = (if native {
        Model::load_native(&model_path)
    } else {
        Model::load(&model_path)
    })
    .unwrap_or_else(|e| {
        eprintln!("{}: {e}", model_path.display());
        std::process::exit(1)
    });
    let mut cpu = cpu;
    let mut native = native;
    if let Some(like) = flag(args, "--quant-like") {
        let weighting = match flag(args, "--weighting").unwrap_or("uniform") {
            "uniform" => Weighting::Uniform,
            "magnitude" => Weighting::Magnitude,
            "importance" => Weighting::Importance,
            "impmag" => Weighting::ImportanceMagnitude,
            other => panic!("unknown --weighting {other}"),
        };
        let opts = EncodeOpts {
            weighting,
            refine: !args.iter().any(|a| a == "--no-refine"),
            threads: 0,
        };
        let imp = flag(args, "--imatrix").map(read_importance);
        let mut hess = flag(args, "--gptq").map(HessianStore::open);
        let gopts = GptqOpts {
            damp: flag(args, "--gptq-damp")
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.01),
            threads: 0,
        };
        let mut types = formats_of(like);
        apply_overrides(&mut types, flag(args, "--override"));
        quantize_model(&mut cpu, &types, imp.as_ref(), hess.as_mut(), &opts, &gopts);
        native = true;
    }
    let fmt = if native {
        Fmt::Native
    } else if args.iter().any(|a| a == "--q8") {
        Fmt::Q8_0
    } else {
        Fmt::F16
    };
    let gpu = if fmt == Fmt::F16 {
        GpuModel::upload_eval(&cpu, 0)
    } else {
        GpuModel::upload(&cpu, fmt, 0)
    }
    .expect("cannot upload the model");
    let vocab = cpu.cfg.vocab;
    drop(cpu);
    eprintln!(
        "[ppl] {} loaded in {:.1} s ({:.0} MB on the GPU)",
        model_path.display(),
        t0.elapsed().as_secs_f32(),
        gpu.weight_bytes as f64 / 1e6
    );
    let mut sess = GpuSession::new(&gpu, ctx + 8).expect("cannot create the session");
    let model_name = model_path
        .file_name()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_default();
    for (ti, text_path) in texts.iter().enumerate() {
        let text = read_text(text_path);
        let ids = tok.encode(&text);
        let n_windows = (ids.len() / ctx).min(max_tokens / ctx);
        assert!(
            n_windows > 0,
            "{text_path}: shorter than one window ({} tokens)",
            ids.len()
        );
        let name = Path::new(text_path)
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_default();
        let reference = refs[ti].map(read_ref);
        let mut dump: Vec<RefRow> = Vec::new();
        let want_dump = dumps[ti].is_some();
        let (mut n, mut nll_sum) = (0usize, 0f64);
        let (mut kl_sum, mut same_top1, mut dnll_sum) = (0f64, 0usize, 0f64);
        let t1 = Instant::now();
        for w in 0..n_windows {
            let win = &ids[w * ctx..(w + 1) * ctx];
            sess.reset().unwrap();
            let mut start = 0;
            while start < ctx {
                let chunk = &win[start..(start + EVAL_ROWS).min(ctx)];
                let logits = sess.prefill_logits(chunk).expect("evaluation chunk failed");
                let rows: Vec<&[f32]> = logits.chunks(vocab).collect();
                let per = (rows.len() + 5) / 6;
                // per row (in parallel): log-sum-exp, the log-likelihood of the next token, the top-64 when a dump is wanted
                let results: Vec<(f64, f64, Option<Vec<(u32, f32)>>)> = std::thread::scope(|s| {
                    let handles: Vec<_> = rows
                        .chunks(per)
                        .enumerate()
                        .map(|(ci, part)| {
                            let base = start + ci * per;
                            s.spawn(move || {
                                part.iter()
                                    .enumerate()
                                    .map(|(j, row)| {
                                        let pos = base + j;
                                        let lse = logsumexp(row);
                                        let nll = if pos + 1 < ctx {
                                            lse - row[win[pos + 1] as usize] as f64
                                        } else {
                                            0.0
                                        };
                                        let top = if pos + 1 < ctx && want_dump {
                                            Some(top_k(row))
                                        } else {
                                            None
                                        };
                                        (lse, nll, top)
                                    })
                                    .collect::<Vec<_>>()
                            })
                        })
                        .collect();
                    handles
                        .into_iter()
                        .flat_map(|h| h.join().unwrap())
                        .collect()
                });
                for (j, (lse, nll, top)) in results.into_iter().enumerate() {
                    let pos = start + j;
                    if pos + 1 >= ctx {
                        continue;
                    }
                    n += 1;
                    nll_sum += nll;
                    if let Some(top) = top {
                        let mut r = RefRow {
                            ids: [0; TOPK],
                            logp: [0.0; TOPK],
                            tgt: -(nll as f32),
                        };
                        for (i, (id, l)) in top.iter().enumerate() {
                            r.ids[i] = *id;
                            r.logp[i] = (*l as f64 - lse) as f32;
                        }
                        dump.push(r);
                    }
                    if let Some(refs) = &reference {
                        let r = &refs[n - 1];
                        let row = rows[j];
                        let mut kl = 0f64;
                        for i in 0..TOPK {
                            let p = (r.logp[i] as f64).exp();
                            kl += p * (r.logp[i] as f64 - (row[r.ids[i] as usize] as f64 - lse));
                        }
                        kl_sum += kl;
                        let top1 = row
                            .iter()
                            .enumerate()
                            .fold((0usize, f32::NEG_INFINITY), |a, (i, &x)| {
                                if x > a.1 {
                                    (i, x)
                                } else {
                                    a
                                }
                            })
                            .0;
                        same_top1 += (top1 as u32 == r.ids[0]) as usize;
                        dnll_sum += nll - (-(r.tgt as f64));
                    }
                }
                start += chunk.len();
            }
        }
        eprintln!(
            "[ppl] {name}: {} windows of {ctx}, {:.0} tokens/s",
            n_windows,
            n as f64 / t1.elapsed().as_secs_f64()
        );
        let mean = nll_sum / n as f64;
        println!(
            "RESULT {model_name} text {name} tokens {n} nll {mean:.5} ppl {:.4}",
            mean.exp()
        );
        if reference.is_some() {
            println!(
                "VS-REFERENCE text {name} kl_top64 {:.5} top1_agree {:.2}% dnll {:+.5}",
                kl_sum / n as f64,
                100.0 * same_top1 as f64 / n as f64,
                dnll_sum / n as f64
            );
        }
        if let Some(path) = dumps[ti] {
            let mut f =
                BufWriter::new(std::fs::File::create(path).expect("cannot create the dump"));
            f.write_all(b"FQREF002").unwrap();
            f.write_all(&(TOPK as u32).to_le_bytes()).unwrap();
            f.write_all(&(dump.len() as u32).to_le_bytes()).unwrap();
            for r in &dump {
                for id in &r.ids {
                    f.write_all(&id.to_le_bytes()).unwrap();
                }
                for l in &r.logp {
                    f.write_all(&l.to_le_bytes()).unwrap();
                }
                f.write_all(&r.tgt.to_le_bytes()).unwrap();
            }
            eprintln!(
                "[ppl] reference written: {} positions in {path}",
                dump.len()
            );
        }
    }
    0
}

/// `forge imatrix <model> --text FILE --out FILE [--tokenizer FILE] [--native] [--ctx 512] [--max-tokens 65536]`: how much each input column of every linear layer matters, measured as the mean
/// square of the activation that multiplies it over a calibration text (the model runs unquantized, in fp16). The file feeds `forge ppl --quant-like ... --weighting importance --imatrix FILE`.
pub fn imatrix(args: &[String]) -> i32 {
    let model_path = PathBuf::from(args.get(1).expect("usage: forge imatrix <model> --text FILE --out FILE [--tokenizer FILE] [--native] [--ctx N] [--max-tokens N]"));
    let tok = Tokenizer::from_file(find_tokenizer(&model_path, flag(args, "--tokenizer")))
        .expect("cannot load the tokenizer");
    let text = read_text(flag(args, "--text").expect("--text is required"));
    let out = flag(args, "--out").expect("--out is required");
    let ctx: usize = flag(args, "--ctx")
        .and_then(|v| v.parse().ok())
        .unwrap_or(512);
    let max_tokens: usize = flag(args, "--max-tokens")
        .and_then(|v| v.parse().ok())
        .unwrap_or(65536);
    let native = args.iter().any(|a| a == "--native");
    let ids = tok.encode(&text);
    let n_windows = (ids.len() / ctx).min(max_tokens / ctx);
    assert!(n_windows > 0, "the text is shorter than one window");
    let mut cpu = (if native {
        Model::load_native(&model_path)
    } else {
        Model::load(&model_path)
    })
    .expect("cannot load the model");
    cpu.mtp = None;
    let gpu = if native {
        GpuModel::upload(&cpu, Fmt::Native, 0)
    } else if args.iter().any(|a| a == "--q8") {
        GpuModel::upload(&cpu, Fmt::Q8_0, 0)
    } else {
        GpuModel::upload_eval(&cpu, 0)
    }
    .expect("cannot upload the model");
    drop(cpu);
    let mut sess = GpuSession::new(&gpu, ctx + 8).expect("cannot create the session");
    sess.enable_stats(false).expect("statistics");
    let t0 = Instant::now();
    for w in 0..n_windows {
        let win = &ids[w * ctx..(w + 1) * ctx];
        sess.reset().unwrap();
        for chunk in win.chunks(EVAL_ROWS) {
            sess.eval_chunk(chunk).expect("calibration chunk failed");
        }
    }
    let stats = sess.take_stats().expect("statistics");
    let mut f = BufWriter::new(std::fs::File::create(out).expect("cannot create the output"));
    f.write_all(b"FQIMAT01").unwrap();
    f.write_all(&(stats.len() as u32).to_le_bytes()).unwrap();
    for (name, v) in &stats {
        f.write_all(&(name.len() as u32).to_le_bytes()).unwrap();
        f.write_all(name.as_bytes()).unwrap();
        f.write_all(&(v.len() as u32).to_le_bytes()).unwrap();
        for x in v {
            f.write_all(&x.to_le_bytes()).unwrap();
        }
    }
    eprintln!(
        "[imatrix] {} tokens in {:.1} s, {} tensors written to {out}",
        n_windows * ctx,
        t0.elapsed().as_secs_f32(),
        stats.len()
    );
    0
}

/// `forge hessian <model> --text FILE --out FILE [--tokenizer FILE] [--ctx 512] [--max-tokens 65536]`: the second moments E[x x^T] of the inputs of every linear layer (for GPTQ), measured on a
/// calibration text with the unquantized fp16 model.
pub fn hessian(args: &[String]) -> i32 {
    let model_path = PathBuf::from(args.get(1).expect("usage: forge hessian <model> --text FILE --out FILE [--tokenizer FILE] [--ctx N] [--max-tokens N]"));
    let tok = Tokenizer::from_file(find_tokenizer(&model_path, flag(args, "--tokenizer")))
        .expect("cannot load the tokenizer");
    let text = read_text(flag(args, "--text").expect("--text is required"));
    let out = flag(args, "--out").expect("--out is required");
    let ctx: usize = flag(args, "--ctx")
        .and_then(|v| v.parse().ok())
        .unwrap_or(512);
    let max_tokens: usize = flag(args, "--max-tokens")
        .and_then(|v| v.parse().ok())
        .unwrap_or(65536);
    let ids = tok.encode(&text);
    let n_windows = (ids.len() / ctx).min(max_tokens / ctx);
    assert!(n_windows > 0, "the text is shorter than one window");
    let native = args.iter().any(|a| a == "--native");
    let mut cpu = (if native {
        Model::load_native(&model_path)
    } else {
        Model::load(&model_path)
    })
    .expect("cannot load the model");
    cpu.mtp = None;
    // Q8_0 weights (--q8) for models whose fp16 copy does not fit the card (the activations hardly differ); --native: a GGUF file or a quantized directory (a GGUF has its value heads in the tiled order)
    let gpu = if native {
        GpuModel::upload(&cpu, Fmt::Native, 0)
    } else if args.iter().any(|a| a == "--q8") {
        GpuModel::upload(&cpu, Fmt::Q8_0, 0)
    } else {
        GpuModel::upload_eval(&cpu, 0)
    }
    .expect("cannot upload the model");
    let n_layers = cpu.cfg.n_layer;
    let n_embd = cpu.cfg.n_embd;
    drop(cpu);
    let mut sess = GpuSession::new(&gpu, ctx + 8).expect("cannot create the session");
    // The second moments of a big model do not fit the card at once. --layers a-b: layers a..=b only (and the head with --head); --layers-per-pass N: all layers, N at a time, in this process
    // (the files are `<out>.<a>-<b>.bin` and `<out>.head.bin`, give them all to --gptq as a list or with a *).
    // The groups run in order and hand the residual stream of every calibration window on to the next group (kept in host memory: tokens x E floats), so each layer sees the calibration text once;
    // the head is a group of its own at the end. --no-cache runs every group from the embedding instead (once per later group, the old way: a cross-check). --first-layer K: do not write the
    // groups before layer K (with --no-cache: do not run them), e.g. to redo the groups after one that failed.
    let first: usize = flag(args, "--first-layer")
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let per_pass = flag(args, "--layers-per-pass").and_then(|v| v.parse::<usize>().ok());
    let cached = per_pass.is_some() && !args.iter().any(|a| a == "--no-cache");
    let groups: Vec<(usize, usize, bool)> = if let Some(n) = per_pass {
        if cached {
            let mut g: Vec<(usize, usize, bool)> = (0..n_layers)
                .step_by(n)
                .map(|a| (a, (a + n).min(n_layers), false))
                .collect();
            g.push((n_layers, n_layers, true));
            g
        } else {
            (first..n_layers)
                .step_by(n)
                .map(|g| (g, (g + n).min(n_layers), g == 0))
                .collect()
        }
    } else if let Some(r) = flag(args, "--layers") {
        let (a, b) = r.split_once('-').expect("--layers a-b");
        vec![(
            a.parse::<usize>().unwrap(),
            (b.parse::<usize>().unwrap() + 1).min(n_layers),
            args.iter().any(|a| a == "--head"),
        )]
    } else {
        vec![(0, n_layers, true)]
    };
    let multi = groups.len() > 1 || per_pass.is_some();
    // residual stream of window w at the start of the next group: [ctx][E] floats
    let mut resid: Vec<Vec<f32>> = Vec::new();
    for (l0, l1, head) in groups {
        sess.enable_stats_for(true, l0..l1, head)
            .expect("statistics");
        if cached {
            sess.set_start_layer(l0).expect("start layer");
        }
        let t0 = Instant::now();
        for w in 0..n_windows {
            let win = &ids[w * ctx..(w + 1) * ctx];
            sess.reset().unwrap();
            if cached {
                // a group over layers takes a whole window per call, the head group the evaluation chunks (it computes logits)
                let rows = if head {
                    EVAL_ROWS
                } else {
                    ctx.min(forge_gpu::prefill_chunk_len())
                };
                let mut next: Vec<f32> = Vec::new();
                for (ci, chunk) in win.chunks(rows).enumerate() {
                    let input = if l0 > 0 {
                        Some(&resid[w][ci * rows * n_embd..(ci * rows + chunk.len()) * n_embd])
                    } else {
                        None
                    };
                    if let Some(o) = sess
                        .eval_chunk_resid(chunk, input)
                        .expect("calibration chunk failed")
                    {
                        next.extend_from_slice(&o);
                    }
                }
                if !head {
                    if w < resid.len() {
                        resid[w] = next;
                    } else {
                        resid.push(next);
                    }
                }
            } else {
                for chunk in win.chunks(EVAL_ROWS) {
                    sess.eval_chunk(chunk).expect("calibration chunk failed");
                }
            }
        }
        let hs = sess.take_hessians().expect("second moments");
        if cached && l1 > l0 && l0 < first {
            eprintln!("[hessian] layers {l0}..{l1}: run for the residual stream, not written (--first-layer {first})");
            continue;
        }
        let path = if !multi {
            out.to_string()
        } else if l0 == l1 {
            format!("{out}.head.bin")
        } else {
            format!("{out}.{l0}-{}.bin", l1 - 1)
        };
        let mut f = BufWriter::new(std::fs::File::create(&path).expect("cannot create the output"));
        f.write_all(b"FQHESS01").unwrap();
        f.write_all(&(hs.len() as u32).to_le_bytes()).unwrap();
        for (name, dim, v) in &hs {
            f.write_all(&(name.len() as u32).to_le_bytes()).unwrap();
            f.write_all(name.as_bytes()).unwrap();
            f.write_all(&(*dim as u32).to_le_bytes()).unwrap();
            for x in v {
                f.write_all(&x.to_le_bytes()).unwrap();
            }
        }
        f.flush().unwrap();
        eprintln!(
            "[hessian] layers {l0}..{l1}: {} tokens in {:.1} s, {} matrices written to {path}",
            n_windows * ctx,
            t0.elapsed().as_secs_f32(),
            hs.len()
        );
    }
    0
}

/// For a tensor indexed by value heads in blocks of `hd` entries starting at `first`: `p[i]` = the old index of the entry that moves to position `i` when Hugging Face's grouped order (value head
/// h pairs with key head h / r) becomes llama.cpp's tiled order (value head h pairs with key head h mod n_k); the identity outside the blocks.
fn tiled_perm(len: usize, first: usize, n_v: usize, hd: usize, n_k: usize) -> Vec<usize> {
    let r = n_v / n_k;
    let mut p: Vec<usize> = (0..len).collect();
    for h in 0..n_v {
        let old = (h % n_k) * r + h / n_k;
        for j in 0..hd {
            p[first + h * hd + j] = first + old * hd + j;
        }
    }
    p
}

fn permute_rows(w: &[f32], cols: usize, p: &[usize]) -> Vec<f32> {
    let mut out = vec![0f32; w.len()];
    for (i, &o) in p.iter().enumerate() {
        out[i * cols..(i + 1) * cols].copy_from_slice(&w[o * cols..(o + 1) * cols]);
    }
    out
}

fn permute_cols(w: &[f32], cols: usize, p: &[usize]) -> Vec<f32> {
    let mut out = vec![0f32; w.len()];
    for (src, dst) in w.chunks(cols).zip(out.chunks_mut(cols)) {
        for (j, &o) in p.iter().enumerate() {
            dst[j] = src[o];
        }
    }
    out
}

fn permute_sym(h: &[f32], dim: usize, p: &[usize]) -> Vec<f32> {
    let mut out = vec![0f32; h.len()];
    for (a, &pa) in p.iter().enumerate() {
        for (b, &pb) in p.iter().enumerate() {
            out[a * dim + b] = h[pa * dim + pb];
        }
    }
    out
}

fn cosine32(a: &[f32], b: &[f32]) -> f64 {
    let (mut ab, mut aa, mut bb) = (0f64, 0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        ab += *x as f64 * *y as f64;
        aa += (*x as f64).powi(2);
        bb += (*y as f64).powi(2);
    }
    ab / (aa.sqrt() * bb.sqrt()).max(1e-30)
}

fn gguf_type_id(v: &forge_gguf::Value) -> u32 {
    use forge_gguf::Value as V;
    match v {
        V::U8(_) => 0,
        V::I8(_) => 1,
        V::U16(_) => 2,
        V::I16(_) => 3,
        V::U32(_) => 4,
        V::I32(_) => 5,
        V::F32(_) => 6,
        V::Bool(_) => 7,
        V::Str(_) => 8,
        V::Arr(_) => 9,
        V::U64(_) => 10,
        V::I64(_) => 11,
        V::F64(_) => 12,
    }
}

fn gguf_put_string(out: &mut Vec<u8>, s: &str) {
    out.extend_from_slice(&(s.len() as u64).to_le_bytes());
    out.extend_from_slice(s.as_bytes());
}

fn gguf_put_value(out: &mut Vec<u8>, v: &forge_gguf::Value) {
    use forge_gguf::Value as V;
    match v {
        V::U8(x) => out.push(*x),
        V::I8(x) => out.push(*x as u8),
        V::U16(x) => out.extend_from_slice(&x.to_le_bytes()),
        V::I16(x) => out.extend_from_slice(&x.to_le_bytes()),
        V::U32(x) => out.extend_from_slice(&x.to_le_bytes()),
        V::I32(x) => out.extend_from_slice(&x.to_le_bytes()),
        V::F32(x) => out.extend_from_slice(&x.to_le_bytes()),
        V::Bool(x) => out.push(*x as u8),
        V::Str(x) => gguf_put_string(out, x),
        V::U64(x) => out.extend_from_slice(&x.to_le_bytes()),
        V::I64(x) => out.extend_from_slice(&x.to_le_bytes()),
        V::F64(x) => out.extend_from_slice(&x.to_le_bytes()),
        V::Arr(items) => {
            out.extend_from_slice(&items.first().map(gguf_type_id).unwrap_or(4).to_le_bytes());
            out.extend_from_slice(&(items.len() as u64).to_le_bytes());
            for it in items {
                gguf_put_value(out, it);
            }
        }
    }
}

/// `forge quantize <hf-dir> <out.gguf> --like TEMPLATE.gguf --gguf [--imatrix F | --gptq H] [--override ...] [--from-dir DIR] [--keep PAT,...]`: our quantization written as a GGUF file that llama.cpp loads.
/// The file is the template with the data of every weight matrix we quantize replaced (the metadata, norms and SSM parameters are the template's; the tensor table is rewritten, so an --override can give
/// a tensor another format and the file another size). Models whose value heads are tiled in the GGUF (n_v != n_k) get the matrices rows / columns reordered before quantizing. For the first four layers
/// (and the embedding) what we are about to write is compared with the template's own tensor (cosine, must be > 0.9: a wrong name mapping or head order shows up as ~0).
#[allow(clippy::too_many_arguments)]
fn quantize_gguf(
    src: &Path,
    out: &Path,
    like: &str,
    from_dir: Option<&Path>,
    keep: &[&str],
    types: &HashMap<String, DType>,
    imp: Option<&HashMap<String, Vec<f32>>>,
    hess: &mut Option<HessianStore>,
    opts: &EncodeOpts,
    gopts: &GptqOpts,
) -> i32 {
    let t0 = Instant::now();
    // --from-dir DIR: take the blocks from a directory `forge quantize` already wrote (hours of GPTQ are not repeated) when its format for the tensor is the one wanted: rows of the reordered tensors move
    // as byte runs; the output projection of the DeltaNet layers (its columns are reordered, which moves weights between blocks) is quantized again from the checkpoint, with the second moments if --gptq is given
    let qd = from_dir.map(|p| {
        forge_gguf::SafeTensorsDir::open(p)
            .unwrap_or_else(|e| panic!("cannot open the quantized directory {}: {e}", p.display()))
    });
    let tm = Gguf::open(like).unwrap_or_else(|e| panic!("cannot open {like}: {e}"));
    let cfg = forge_model::Config::from_gguf(&tm).expect("the template's configuration");
    let d = forge_gguf::SafeTensorsDir::open(src).expect("cannot open the checkpoint");
    let prefix = if d
        .tensor("model.language_model.embed_tokens.weight")
        .is_some()
    {
        "model.language_model."
    } else {
        "model."
    };
    let mut names: Vec<String> = d.names().map(|s| s.to_string()).collect();
    names.sort();
    let tiled = cfg.delta.clone().filter(|dc| dc.tiled_v);
    if tiled.is_some() {
        eprintln!("[gguf] the template stores the value heads in the tiled order: reordering the DeltaNet matrices");
    }
    // the data of the tensors we replace, by GGUF name, with the format they now have
    let mut replaced: HashMap<String, (DType, Vec<u8>)> = HashMap::new();
    let (mut done, mut weights, mut checked) = (0usize, 0usize, 0usize);
    for n in &names {
        let (st, t) = d.tensor(n).unwrap();
        let Some(g) = gguf_name_of(prefix, n).filter(|_| t.shape.len() == 2) else {
            continue;
        };
        let Some(info) = tm.tensor_info(&g) else {
            continue;
        };
        let dt = types.get(&g).copied().unwrap_or(info.dtype);
        let ours = |x: DType| {
            matches!(
                x,
                DType::Q8_0 | DType::Q4_0 | DType::Q5_0 | DType::Q4_K | DType::Q5_K | DType::Q6_K
            )
        };
        if !ours(info.dtype) || !ours(dt) {
            continue;
        }
        if keep.iter().any(|k| g.contains(k)) {
            eprintln!("[gguf] {g}: kept as the template has it");
            continue;
        }
        let (rows, cols) = (t.shape[0], t.shape[1]);
        assert_eq!(
            info.shape(),
            vec![rows, cols],
            "{g}: the template's shape differs from the checkpoint's"
        );
        // how the template orders this tensor: rows that move (value heads tiled), columns that move (the output projection of the DeltaNet layers), the first row to compare with the template's
        let (mut row_perm, mut col_perm, mut probe_first) =
            (None::<Vec<usize>>, None::<Vec<usize>>, 0usize);
        if let Some(dc) = &tiled {
            let (nk, nv, hd) = (dc.n_k, dc.n_v, dc.head_v);
            if g.ends_with(".attn_qkv.weight") {
                probe_first = 2 * dc.key_dim();
                row_perm = Some(tiled_perm(rows, probe_first, nv, hd, nk));
            } else if g.ends_with(".attn_gate.weight") {
                row_perm = Some(tiled_perm(rows, 0, nv, hd, nk));
            } else if g.ends_with(".ssm_alpha.weight") || g.ends_with(".ssm_beta.weight") {
                row_perm = Some(tiled_perm(rows, 0, nv, 1, nk));
            } else if g.ends_with(".ssm_out.weight") {
                col_perm = Some(tiled_perm(cols, 0, nv, hd, nk));
            }
        }
        let plain = plain_embedding(&g, tm.tensor_info("output.weight").is_some());
        // blocks from the quantized directory, if it has this tensor in the format wanted
        let from_dir_blocks: Option<&[u8]> = qd
            .as_ref()
            .filter(|_| col_perm.is_none() && !plain)
            .and_then(|qd| {
                let (qst, qt) = qd.tensor(n)?;
                match forge_model::quant_entry(qst, n) {
                    Some(Ok((qdt, qr, qc))) if qdt == dt && qr == rows && qc == cols => {
                        Some(qst.bytes(qt))
                    }
                    _ => None,
                }
            });
        let enc: Vec<u8> = match from_dir_blocks {
            Some(raw) => {
                assert_eq!(
                    raw.len(),
                    dt.bytes_for(rows * cols),
                    "{g}: the quantized directory's tensor has another size"
                );
                match &row_perm {
                    Some(p) => {
                        let rb = dt.bytes_for(cols);
                        let mut o = vec![0u8; raw.len()];
                        for (i, &s) in p.iter().enumerate() {
                            o[i * rb..(i + 1) * rb].copy_from_slice(&raw[s * rb..(s + 1) * rb]);
                        }
                        o
                    }
                    None => raw.to_vec(),
                }
            }
            None => {
                let mut w = st.to_f32(t).unwrap_or_else(|er| panic!("{n}: {er}"));
                let mut iv: Option<Vec<f32>> =
                    imp.and_then(|h| h.get(&g)).cloned().filter(|_| !plain);
                let mut hv: Option<(usize, std::rc::Rc<Vec<f32>>)> = hess
                    .as_mut()
                    .and_then(|h| h.get(&hessian_key(&g)))
                    .filter(|(dim, _)| *dim == cols)
                    .filter(|_| !plain);
                let uniform = EncodeOpts {
                    weighting: Weighting::Uniform,
                    refine: opts.refine,
                    threads: opts.threads,
                };
                if let Some(p) = &row_perm {
                    w = permute_rows(&w, cols, p);
                }
                if let Some(p) = &col_perm {
                    w = permute_cols(&w, cols, p);
                    iv = iv.map(|v| p.iter().map(|&o| v[o]).collect());
                    hv = hv.map(|(dim, h)| (dim, std::rc::Rc::new(permute_sym(&h, dim, p))));
                }
                match hv {
                    Some((_, h)) => gptq_matrix(dt, rows, cols, &w, &h, opts, gopts),
                    None => encode_matrix(
                        dt,
                        rows,
                        cols,
                        &w,
                        iv.as_deref(),
                        if plain { &uniform } else { opts },
                    ),
                }
                .unwrap_or_else(|er| panic!("{g}: {er}"))
            }
        };
        assert_eq!(
            enc.len(),
            dt.bytes_for(rows * cols),
            "{g}: encoded size is not the format's"
        );
        let layer: usize = g
            .strip_prefix("blk.")
            .and_then(|r| r.split('.').next())
            .and_then(|v| v.parse().ok())
            .unwrap_or(usize::MAX);
        if layer < 4 || g == "token_embd.weight" {
            // what we are about to write against the template's own weights (both are quantizations of the same checkpoint)
            let nr = 512usize.min(rows - probe_first);
            let (bm, bt) = (dt.bytes_for(cols), info.dtype.bytes_for(cols));
            let mine = forge_gguf::dequantize(
                dt,
                &enc[probe_first * bm..(probe_first + nr) * bm],
                nr * cols,
            )
            .unwrap_or_else(|e| panic!("{g}: {e}"));
            let theirs = forge_gguf::dequantize(
                info.dtype,
                &tm.tensor_bytes(info)[probe_first * bt..(probe_first + nr) * bt],
                nr * cols,
            )
            .unwrap_or_else(|e| panic!("{g}: {e}"));
            let cos = cosine32(&mine, &theirs);
            assert!(cos > 0.9, "{g}: cosine {cos:.3} between our blocks and the template's (a wrong name mapping or head order)");
            checked += 1;
        }
        replaced.insert(g, (dt, enc));
        done += 1;
        weights += rows * cols;
        if done % 20 == 0 {
            eprintln!(
                "[quantize] {done} matrices, {:.1} s",
                t0.elapsed().as_secs_f32()
            );
        }
    }
    // the output file: header with the template's metadata, the tensor table with the new formats and offsets, then the data (every tensor starts on the alignment)
    let align = tm.alignment as u64;
    let pad = |n: u64| (n + align - 1) / align * align;
    let mut table: Vec<(DType, u64, u64)> = Vec::new(); // (format, offset, bytes)
    let mut off = 0u64;
    for info in &tm.tensors {
        let dt = replaced
            .get(&info.name)
            .map(|(d, _)| *d)
            .unwrap_or(info.dtype);
        let nb = replaced
            .get(&info.name)
            .map(|(_, b)| b.len())
            .unwrap_or_else(|| info.n_bytes()) as u64;
        table.push((dt, off, nb));
        off += pad(nb);
    }
    let mut head: Vec<u8> = Vec::new();
    head.extend_from_slice(b"GGUF");
    head.extend_from_slice(&3u32.to_le_bytes());
    head.extend_from_slice(&(tm.tensors.len() as u64).to_le_bytes());
    head.extend_from_slice(&(tm.metadata.len() as u64).to_le_bytes());
    for (k, v) in &tm.metadata {
        gguf_put_string(&mut head, k);
        head.extend_from_slice(&gguf_type_id(v).to_le_bytes());
        gguf_put_value(&mut head, v);
    }
    for (info, (dt, o, _)) in tm.tensors.iter().zip(&table) {
        gguf_put_string(&mut head, &info.name);
        head.extend_from_slice(&(info.dims.len() as u32).to_le_bytes());
        for dm in &info.dims {
            head.extend_from_slice(&dm.to_le_bytes());
        }
        head.extend_from_slice(&dt.ggml_id().to_le_bytes());
        head.extend_from_slice(&o.to_le_bytes());
    }
    head.resize(pad(head.len() as u64) as usize, 0);
    let mut f = BufWriter::with_capacity(
        1 << 22,
        std::fs::File::create(out)
            .unwrap_or_else(|e| panic!("cannot create {}: {e}", out.display())),
    );
    f.write_all(&head).unwrap();
    let zeros = vec![0u8; align as usize];
    for (info, (_, _, nb)) in tm.tensors.iter().zip(&table) {
        match replaced.get(&info.name) {
            Some((_, b)) => f.write_all(b).unwrap(),
            None => f.write_all(tm.tensor_bytes(info)).unwrap(),
        }
        let padn = (pad(*nb) - *nb) as usize;
        f.write_all(&zeros[..padn]).unwrap();
    }
    f.flush().unwrap();
    eprintln!("[gguf] {}: {done} matrices replaced ({:.1} M weights, {checked} compared with the template's), {:.2} GB in {:.1} s", out.display(), weights as f64 / 1e6, (head.len() as u64 + off) as f64 / 1e9, t0.elapsed().as_secs_f32());
    0
}

/// `forge mtp-agree <model> --text FILE [--tokenizer F] [--mtp DIR|FILE] [--native] [--ctx 512] [--max-tokens 24576]`: how often the MTP head's draft equals the main model's own next token on a held-out
/// text with the text's tokens as the pending tokens (teacher forcing) = the acceptance rate speculative decoding would have for this quantization, without the noise of two free-running generations
/// that drift apart. Also printed: how often the draft and the main model hit the text's actual token.
pub fn mtp_agree(args: &[String]) -> i32 {
    let model_path = PathBuf::from(args.get(1).expect("usage: forge mtp-agree <model> --text FILE [--tokenizer F] [--mtp DIR|FILE] [--native] [--ctx N] [--max-tokens N]"));
    let tok = Tokenizer::from_file(find_tokenizer(&model_path, flag(args, "--tokenizer")))
        .expect("cannot load the tokenizer");
    let text = read_text(flag(args, "--text").expect("--text is required"));
    let ctx: usize = flag(args, "--ctx")
        .and_then(|v| v.parse().ok())
        .unwrap_or(512);
    let max_tokens: usize = flag(args, "--max-tokens")
        .and_then(|v| v.parse().ok())
        .unwrap_or(24576);
    let ids = tok.encode(&text);
    let n_windows = (ids.len() / ctx).min(max_tokens / ctx);
    assert!(n_windows > 0, "the text is shorter than one window");
    let native = args.iter().any(|a| a == "--native");
    let mut cpu = (if native {
        Model::load_native(&model_path)
    } else {
        Model::load(&model_path)
    })
    .expect("cannot load the model");
    if cpu.mtp.is_none() {
        let p = flag(args, "--mtp").expect("the model has no MTP head: --mtp DIR|FILE");
        cpu.load_mtp(
            Path::new(p),
            &forge_model::LoadOpts {
                dequantize: false,
                keep_raw: true,
            },
        )
        .expect("cannot load the MTP head");
    }
    let gpu = GpuModel::upload(&cpu, if native { Fmt::Native } else { Fmt::Q8_0 }, 0)
        .expect("cannot upload the model");
    drop(cpu);
    let mut sess = GpuSession::new(&gpu, ctx + 8).expect("cannot create the session");
    let (mut n, mut agree, mut draft_hits, mut main_hits, mut n_main) =
        (0usize, 0usize, 0usize, 0usize, 0usize);
    let t0 = Instant::now();
    for w in 0..n_windows {
        let win = &ids[w * ctx..(w + 1) * ctx];
        sess.reset().unwrap();
        // the draft made after token i is for position i + 2 (it sees the actual token i + 1 as the pending one); the main model's own prediction for that position comes after token i + 1
        let mut prev: Option<u32> = None;
        for i in 0..ctx - 1 {
            sess.advance(win[i]).unwrap();
            let main_next = sess.argmax_token().unwrap();
            n_main += 1;
            if main_next == win[i + 1] {
                main_hits += 1;
            }
            if let Some(d) = prev {
                n += 1;
                if d == main_next {
                    agree += 1;
                }
                if d == win[i + 1] {
                    draft_hits += 1;
                }
            }
            prev = sess.mtp_step(win[i + 1], sess.hidden(), i, true).unwrap();
        }
    }
    let name = model_path
        .file_name()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_default();
    eprintln!(
        "[mtp-agree] {name}: {n} positions in {:.1} s: draft == the model's own next token {:.2} %, draft == the text's token {:.2} %, the model's argmax == the text's token {:.2} %",
        t0.elapsed().as_secs_f32(),
        100.0 * agree as f64 / n as f64,
        100.0 * draft_hits as f64 / n as f64,
        100.0 * main_hits as f64 / n_main as f64
    );
    0
}

/// `forge cmp-hessians A B`: compare the second-moment files of two `forge hessian` runs (lists or `*` patterns, as for --gptq): per input, the largest difference relative to the largest entry.
/// Used to check the residual-stream cache against the plain passes; prints MATCH when every matrix agrees to 1e-4.
pub fn cmp_hessians(args: &[String]) -> i32 {
    let mut a = HessianStore::open(args.get(1).expect("usage: forge cmp-hessians A B"));
    let mut b = HessianStore::open(args.get(2).expect("usage: forge cmp-hessians A B"));
    let mut keys: Vec<String> = a
        .index
        .keys()
        .filter(|k| b.index.contains_key(*k))
        .cloned()
        .collect();
    keys.sort();
    let only_a = a.index.keys().filter(|k| !b.index.contains_key(*k)).count();
    let only_b = b.index.keys().filter(|k| !a.index.contains_key(*k)).count();
    println!(
        "{} matrices in A, {} in B, {} in common ({only_a} only in A, {only_b} only in B)",
        a.index.len(),
        b.index.len(),
        keys.len()
    );
    let mut worst = 0f64;
    for k in &keys {
        let (da, ma) = a.get(k).unwrap();
        let (db, mb) = b.get(k).unwrap();
        if da != db {
            println!("  {k}: widths differ ({da} against {db})");
            return 1;
        }
        let scale = ma.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-30) as f64;
        let diff = ma
            .iter()
            .zip(mb.iter())
            .fold(0f32, |m, (x, y)| m.max((x - y).abs())) as f64
            / scale;
        worst = worst.max(diff);
        if diff > 1e-5 {
            println!("  {k}: relative max difference {diff:.2e}");
        }
    }
    println!(
        "worst relative difference over {} matrices: {worst:.2e} -> {}",
        keys.len(),
        if worst < 1e-4 && !keys.is_empty() {
            "MATCH"
        } else {
            "DIFFERENT"
        }
    );
    0
}

fn json_text(j: &Json) -> String {
    let mut s = String::new();
    j.write(&mut s);
    s
}

/// GGUF tensor name of an HF weight matrix (inverse of `hf_name`; None for tensors that are not linear weights).
fn gguf_name_of(prefix: &str, hf: &str) -> Option<String> {
    if hf == format!("{prefix}embed_tokens.weight") {
        return Some("token_embd.weight".to_string());
    }
    if hf == "lm_head.weight" {
        return Some("output.weight".to_string());
    }
    let rest = hf.strip_prefix(prefix)?.strip_prefix("layers.")?;
    let (layer, part) = rest.split_once('.')?;
    let name = match part.trim_end_matches(".weight") {
        "self_attn.q_proj" => "attn_q",
        "self_attn.k_proj" => "attn_k",
        "self_attn.v_proj" => "attn_v",
        "self_attn.o_proj" => "attn_output",
        "linear_attn.in_proj_qkv" => "attn_qkv",
        "linear_attn.in_proj_z" => "attn_gate",
        "linear_attn.in_proj_a" => "ssm_alpha",
        "linear_attn.in_proj_b" => "ssm_beta",
        "linear_attn.out_proj" => "ssm_out",
        "mlp.gate_proj" => "ffn_gate",
        "mlp.up_proj" => "ffn_up",
        "mlp.down_proj" => "ffn_down",
        _ => return None,
    };
    Some(format!("blk.{layer}.{name}.weight"))
}

/// `forge quantize <hf-dir> <out-dir> --like FILE.gguf [--weighting ...] [--imatrix FILE] [--gptq HESSIANS [--gptq-damp X]] [--override "token_embd=Q5_K;..."]`
///
/// Writes a model directory the engine loads like the original: config and tokenizer files copied, and one `model.safetensors` in which every weight matrix is replaced by its quantized blocks
/// (U8 tensors; `__metadata__.quant` says which format each is) and all other tensors are copied unchanged. `--like` names a GGUF file whose per-tensor formats are used (e.g. unsloth's Q4_K_M).
/// One tensor at a time is held in memory (a 9B checkpoint needs a few GB, not the 36 GB of its f32 copy).
pub fn quantize(args: &[String]) -> i32 {
    let src = PathBuf::from(args.get(1).expect("usage: forge quantize <hf-dir> <out-dir> --like FILE.gguf [--weighting W] [--imatrix FILE] [--gptq HESSIANS]"));
    let out = PathBuf::from(
        args.get(2)
            .expect("usage: forge quantize <hf-dir> <out-dir> --like FILE.gguf ..."),
    );
    let like = flag(args, "--like").expect("--like FILE.gguf is required (the per-tensor formats)");
    let weighting =
        match flag(args, "--weighting").unwrap_or(if flag(args, "--imatrix").is_some() {
            "importance"
        } else {
            "uniform"
        }) {
            "uniform" => Weighting::Uniform,
            "magnitude" => Weighting::Magnitude,
            "importance" => Weighting::Importance,
            "impmag" => Weighting::ImportanceMagnitude,
            other => panic!("unknown --weighting {other}"),
        };
    let opts = EncodeOpts {
        weighting,
        refine: !args.iter().any(|a| a == "--no-refine"),
        threads: 0,
    };
    let imp = flag(args, "--imatrix").map(read_importance);
    let mut hess = flag(args, "--gptq").map(HessianStore::open);
    let gopts = GptqOpts {
        damp: flag(args, "--gptq-damp")
            .and_then(|v| v.parse().ok())
            .unwrap_or(0.01),
        threads: 0,
    };
    let t0 = Instant::now();
    let mut types = formats_of(like);
    apply_overrides(&mut types, flag(args, "--override"));
    if args.iter().any(|a| a == "--gguf") {
        let keep: Vec<&str> = flag(args, "--keep")
            .map(|v| v.split(',').filter(|x| !x.is_empty()).collect())
            .unwrap_or_default();
        return quantize_gguf(
            &src,
            &out,
            like,
            flag(args, "--from-dir").map(Path::new),
            &keep,
            &types,
            imp.as_ref(),
            &mut hess,
            &opts,
            &gopts,
        );
    }
    let d = forge_gguf::SafeTensorsDir::open(&src).expect("cannot open the checkpoint");
    // --reuse DIR: a directory `forge quantize` wrote before; a matrix that has the same format there is copied (its blocks are final) instead of quantized again, so a recipe that changes a few tensors
    // (the head to Q5_K, the embedding to another format) costs minutes, not the hour of the GPTQ of the whole model
    let reuse = flag(args, "--reuse").map(|p| {
        forge_gguf::SafeTensorsDir::open(Path::new(p))
            .unwrap_or_else(|e| panic!("cannot open the directory to reuse {p}: {e}"))
    });
    // --redo PAT[,PAT]: quantize the tensors whose GGUF name contains PAT again even if --reuse has them in the right format
    let redo: Vec<&str> = flag(args, "--redo")
        .map(|v| v.split(',').filter(|x| !x.is_empty()).collect())
        .unwrap_or_default();
    let untied = d.tensor("lm_head.weight").is_some();
    let prefix = if d
        .tensor("model.language_model.embed_tokens.weight")
        .is_some()
    {
        "model.language_model."
    } else {
        "model."
    };
    let mut names: Vec<String> = d.names().map(|s| s.to_string()).collect();
    names.sort();
    // the plan: what every tensor becomes (quantized ones get their size from the format alone)
    struct Entry {
        name: String,
        dtype: String,
        shape: Vec<usize>,
        start: usize,
        end: usize,
        quant: Option<(DType, String)>, // format and GGUF name
    }
    let mut entries: Vec<Entry> = Vec::new();
    let mut quant_meta: Vec<(String, Json)> = Vec::new();
    let mut at = 0usize;
    for n in &names {
        let (_, t) = d.tensor(n).unwrap();
        let g = gguf_name_of(prefix, n).filter(|_| t.shape.len() == 2);
        let q = g
            .as_ref()
            .and_then(|g| types.get(g).copied())
            .filter(|dt| {
                matches!(
                    dt,
                    DType::Q8_0
                        | DType::Q4_0
                        | DType::Q5_0
                        | DType::Q4_K
                        | DType::Q5_K
                        | DType::Q6_K
                )
            })
            .filter(|dt| t.shape[1] % dt.block_values() == 0);
        let (dtype, shape, len, quant) = match q {
            Some(dt) => {
                let bytes = dt.bytes_for(t.shape[0] * t.shape[1]);
                quant_meta.push((
                    n.clone(),
                    Json::Obj(vec![
                        ("dtype".to_string(), Json::Str(dt.name())),
                        ("rows".to_string(), Json::Int(t.shape[0] as i64)),
                        ("cols".to_string(), Json::Int(t.shape[1] as i64)),
                    ]),
                ));
                ("U8".to_string(), vec![bytes], bytes, Some((dt, g.unwrap())))
            }
            None => (
                if t.dtype == forge_gguf::StDType::Bool {
                    "BOOL".to_string()
                } else {
                    format!("{:?}", t.dtype)
                },
                t.shape.clone(),
                t.end - t.start,
                None,
            ),
        };
        entries.push(Entry {
            name: n.clone(),
            dtype,
            shape,
            start: at,
            end: at + len,
            quant,
        });
        at += len;
    }
    std::fs::create_dir_all(&out).expect("cannot create the output directory");
    for entry in std::fs::read_dir(&src)
        .expect("cannot list the source directory")
        .flatten()
    {
        let p = entry.path();
        let n = entry.file_name().to_string_lossy().to_string();
        if p.is_file() && !n.ends_with(".safetensors") && n != "model.safetensors.index.json" {
            std::fs::copy(&p, out.join(&n)).expect("cannot copy a side file");
        }
    }
    let mut header: Vec<(String, Json)> = Vec::new();
    let n_quant = quant_meta.len();
    header.push((
        "__metadata__".to_string(),
        Json::Obj(vec![
            ("format".to_string(), Json::Str("pt".into())),
            ("forgeinfer".to_string(), Json::Str("quantized-1".into())),
            (
                "quant".to_string(),
                Json::Str(json_text(&Json::Obj(quant_meta))),
            ),
        ]),
    ));
    for e in &entries {
        header.push((
            e.name.clone(),
            Json::Obj(vec![
                ("dtype".to_string(), Json::Str(e.dtype.clone())),
                (
                    "shape".to_string(),
                    Json::Arr(e.shape.iter().map(|&v| Json::Int(v as i64)).collect()),
                ),
                (
                    "data_offsets".to_string(),
                    Json::Arr(vec![Json::Int(e.start as i64), Json::Int(e.end as i64)]),
                ),
            ]),
        ));
    }
    let mut htext = json_text(&Json::Obj(header)).into_bytes();
    while (htext.len() + 8) % 8 != 0 {
        htext.push(b' ');
    }
    let mut f = BufWriter::new(
        std::fs::File::create(out.join("model.safetensors"))
            .expect("cannot create model.safetensors"),
    );
    f.write_all(&(htext.len() as u64).to_le_bytes()).unwrap();
    f.write_all(&htext).unwrap();
    let (mut done, mut weights, mut reused) = (0usize, 0usize, 0usize);
    for e in &entries {
        let (st, t) = d.tensor(&e.name).unwrap();
        match &e.quant {
            Some((dt, gname)) => {
                let (rows, cols) = (t.shape[0], t.shape[1]);
                let plain = plain_embedding(gname, untied);
                if let Some((rst, rt)) = reuse
                    .as_ref()
                    .filter(|_| !redo.iter().any(|k| gname.contains(k)))
                    .and_then(|r| r.tensor(&e.name))
                {
                    if matches!(forge_model::quant_entry(rst, &e.name), Some(Ok((rdt, rr, rc))) if rdt == *dt && rr == rows && rc == cols)
                    {
                        let blocks = rst.bytes(rt);
                        assert_eq!(
                            blocks.len(),
                            e.end - e.start,
                            "{}: the reused blocks have another size",
                            e.name
                        );
                        f.write_all(blocks).unwrap();
                        reused += 1;
                        done += 1;
                        weights += rows * cols;
                        continue;
                    }
                }
                let w = st.to_f32(t).unwrap_or_else(|er| panic!("{}: {er}", e.name));
                let iv = imp
                    .as_ref()
                    .and_then(|h| h.get(gname))
                    .map(|v| v.as_slice())
                    .filter(|_| !plain);
                if matches!(
                    opts.weighting,
                    Weighting::Importance | Weighting::ImportanceMagnitude
                ) && iv.is_none()
                    && hess.is_none()
                    && !plain
                {
                    panic!("no importance for {gname}");
                }
                let hv = hess
                    .as_mut()
                    .and_then(|h| h.get(&hessian_key(gname)))
                    .filter(|(dim, _)| *dim == cols)
                    .filter(|_| !plain);
                let uniform = EncodeOpts {
                    weighting: Weighting::Uniform,
                    refine: opts.refine,
                    threads: opts.threads,
                };
                let enc = match hv {
                    Some((_, h)) => gptq_matrix(*dt, rows, cols, &w, &h, &opts, &gopts),
                    None => encode_matrix(
                        *dt,
                        rows,
                        cols,
                        &w,
                        iv,
                        if plain { &uniform } else { &opts },
                    ),
                }
                .unwrap_or_else(|er| panic!("{}: {er}", e.name));
                assert_eq!(enc.len(), e.end - e.start);
                f.write_all(&enc).unwrap();
                done += 1;
                weights += rows * cols;
                if done % 20 == 0 {
                    eprintln!(
                        "[quantize] {done}/{n_quant} matrices, {:.1} s",
                        t0.elapsed().as_secs_f32()
                    );
                }
            }
            None => f.write_all(st.bytes(t)).unwrap(),
        }
    }
    f.flush().unwrap();
    eprintln!("[quantize] {} -> {}: {} tensors, {} matrices quantized ({:.1} M weights, {} of them reused), {:.2} GB in {:.1} s", src.display(), out.display(), entries.len(), done, weights as f64 / 1e6, reused, at as f64 / 1e9, t0.elapsed().as_secs_f32());
    0
}
