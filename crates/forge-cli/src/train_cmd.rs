//! `forge train`: pre-training loop on uint16 token shards. Random windows of seq_len + 1 tokens, gradient accumulation over micro-batches, AdamW with warmup + WSD (warmup, stable, linear decay) schedule,
//! dynamic loss scaling, validation on a held-out region, atomic checkpoints (params + Adam state) and resume.
//!
//!   forge train --data DIR --out DIR --layers 30 --dim 576 --heads 9 --kv 3 --ffn 1536 --seq 2048 --micro 2 --tokens-per-step 131072 --steps 50000 --lr 1e-3 [--qk-norm] [--resume]
//!   --optim muon [--muon-lr PEAK] [--muon-wd 0.05] [--muon-mu 0.95]   Muon for the hidden matrices (Moonshot scaling), AdamW (--lr, --wd) for the embedding and the norm gains
//!   Forge-1 structure options (all off = the plain Llama-style decoder): --untied (separate zero-initialised output head)  --relu2 (ReLU squared MLP, --ffn is then its hidden width)
//!   --ngram 2,3 [--ngram-bits 17] [--ngram-dim 128]   hashed n-gram memory: tables of 2^bits rows indexed by a hash of the last 2 / 3 tokens, projected into the embedding output
//!   --lr-emb-mult M --lr-head-mult M --lr-table-mult M --lr-scalar-mult M   AdamW learning-rate multipliers of the embedding, the separate head, the lookup tables (value embeddings, n-gram memory) and the scalars (default 1)
//!   --init-from CKPT   start from the parameters of a checkpoint (fresh optimizer, step 0): fine-tuning; --resume continues a run and wins over it
//!   --time-limit-s N   stop (after writing a checkpoint) at the first optimizer step past N seconds: sessions with a wall-clock limit continue with --resume
//!   --gpus N   data parallel over N devices of this machine (the gradients are summed through host memory every optimizer step)
//!   --softcap 30 (logits = 30 tanh(z / 30))  --lambdas (per-layer residual scalars with the embedding)  --skips (U-net skip connections)  --value-emb 0,1,10,11 (value embeddings in those layers)
//!   --cool-data DIR [--cool-blend-steps N] [--cool-share F] [--cool-start STEP]   cool-down data: from the first step of the learning-rate decay (or STEP) a growing share of the training windows comes from the shards of DIR
//!       (cool_NNN.u16; files named *_val.u16 are never trained on; an optional heldout/cool_val.u16 or cool_val.u16 adds a `val_cool` loss), the rest from --data (whose last shard still holds the validation tail,
//!       so `val` stays comparable with the earlier part of the run). The share rises linearly from 0 to F (default 0.6) over N steps (default 0 = a hard switch); a pure function of the step, so --resume continues it.
//!       The first decay step is the one of the learning-rate schedule (whatever --decay-shape); each window (also the shorter ones of --seq-schedule and the windows of a --tps-schedule step) draws its pool.
//!   Sample-efficiency options (all off by default; tools/ref_muon.py and tools/gen_train_goldens.py are their specifications):
//!   --muon-polar   Polar Express coefficients (one minimax polynomial per iteration, input / (1.02 |X|_F)) instead of Jordan's fixed quintic;  --muon-iters N   orthogonalisation iterations (default 5;
//!                  Polar Express 3 to 7)
//!   --normuon [--normuon-axis long|rows] [--normuon-beta2 0.95]   NorMuon: every lane of the orthogonalised update (the rows, or for a wide matrix the columns) divided by the root of a running mean
//!                  square, the matrix then rescaled to its previous norm (the running averages are stored in the checkpoint's second-moment array, which Muon does not use otherwise)
//!   --cautious-wd  cautious weight decay of the Muon matrices: decay only where the update has the sign of the weight;  --cautious-wd-adam  the same in AdamW (the embedding's decay)
//!   --ema 0.999 [--ema-start STEP] [--ema-every K]   exponential moving average of the parameters (host memory) from STEP on (default: the start of the decay phase); every evaluation also reports
//!                  val_ema (the average on the same batches) and every checkpoint writes ckpt_ema.bin (a normal checkpoint holding the average; --resume continues the average from it)
//!   --attn-gate K [--lr-gate-mult M]   gated attention: every head's output times sigmoid(w_h . the first K channels of the normalised attention input), before the output projection (K = --dim:
//!                  the head-wise gate of Qiu et al. 2025; K = 12 or 32: modded-nanogpt's sparse gate); the gate weights start at 0 (gates 0.5) and train with AdamW (multiplier M, default 1)
//!   --peri-norm    Peri-LN: an RMSNorm (own gain, initialised to 1) on the output of every attention and MLP sublayer before the residual add;  --emb-norm  also normalise the embedding output
//!   --doc-mask [--eos 0]   document masking: an end-of-text token opens a new document and a position attends only to its own document (training and validation; head_dim 64); every
//!                  evaluation also reports val_unmasked (plain causal attention on the same batches: the number comparable with runs without the mask)
//!   --decay-shape linear|sqrt|cosine   the shape of the final learning-rate decay (t = fraction of the decay phase done, f = --min-lr-frac): linear peak (1 - (1 - f) t) (default),
//!                  sqrt peak (f + (1 - f)(1 - sqrt t)) (Haegele et al. 2024), cosine peak (f + (1 - f)(1 + cos(pi t)) / 2)
//!   --tps-schedule 32768:0,65536:20000   tokens per optimizer step from the given step on (before the first entry: --tokens-per-step; each a multiple of micro x seq x gpus); the learning rate
//!                  schedule is unchanged; log.jsonl gets the training tokens so far ("tokens") on every line
//!   --seq-schedule 512:300,1024:600   training windows of 512 tokens before step 300, of 1024 before step 600, then --seq; the tokens per step stay the same (more, shorter rows); validation always
//!                  uses --seq; the window follows the step number, so --resume works across a switch
//!   --window 512 [--window-layers auto|0,1,2]   sliding-window attention in those layers (auto: three of every four, every fourth and the last layer stay dense); default off

use forge_core::Mmap;
use forge_train::{MuonOpts, NorMuonAxis, TrainConfig, Trainer};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

/// The value of `--name VALUE`. A flag given more than once must have the same value every time: the first one used to win silently, and tools/launch_f1.sh puts --steps, --decay-frac,
/// --min-lr-frac, --ffn ... BEFORE `F1_ARGS`, so an override in F1_ARGS (say --steps 90000 at a restart) was ignored without a word.
fn flag<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    asked(name);
    let mut values = args
        .iter()
        .enumerate()
        .filter(|(_, a)| *a == name)
        .map(|(i, _)| args.get(i + 1).map(|s| s.as_str()));
    let first = values.next()?;
    for v in values {
        assert!(
            v == first,
            "{name} is given more than once with different values ({} and {}): give it once",
            first.unwrap_or("nothing"),
            v.unwrap_or("nothing")
        );
    }
    first
}
/// The value of a numeric flag, or the message that says what is wrong with it.
fn parse_num<T: std::str::FromStr>(name: &str, v: &str) -> Result<T, String> {
    v.parse()
        .map_err(|_| format!("{name} takes a number, not {v:?}"))
}
/// A numeric flag: the default when it is absent; stop with a clear message when it is given but is not a number (it used to fall back to the default without a word: `--steps 12k` trained for
/// the default number of steps).
fn num<T: std::str::FromStr>(args: &[String], name: &str, default: T) -> T {
    match flag(args, name) {
        None => default,
        Some(v) => parse_num(name, v).unwrap_or_else(|e| panic!("{e}")),
    }
}
fn has(args: &[String], name: &str) -> bool {
    asked(name);
    args.iter().any(|a| a == name)
}

/// The flags that a lookup asked for (from any thread: the workers read flags too). What is left over in `args` once the configuration is read is a misspelled flag, or one that this
/// configuration does not use: `report_unused_flags` says so, because a typo used to be dropped without a word and the run went on with the default.
static ASKED: std::sync::Mutex<std::collections::BTreeSet<String>> =
    std::sync::Mutex::new(std::collections::BTreeSet::new());

fn asked(name: &str) {
    if let Ok(mut a) = ASKED.lock() {
        a.insert(name.to_string());
    }
}

/// The `--flags` of `args` that nobody asked for, in the order given, each once.
fn unused_flags(args: &[String], asked: &std::collections::BTreeSet<String>) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for a in args {
        if a.len() > 2 && a.starts_with("--") && !asked.contains(a) && !out.contains(a) {
            out.push(a.clone());
        }
    }
    out
}

/// Say on stderr which flags were not used; with FORGE_STRICT_FLAGS=1 stop with exit code 2 instead of going on.
fn report_unused_flags(args: &[String]) {
    let unused = match ASKED.lock() {
        Ok(a) => unused_flags(args, &a),
        Err(_) => return,
    };
    if unused.is_empty() {
        return;
    }
    eprintln!(
        "[train] WARNING: not used: {} (a misspelled flag, or one that this configuration does not use: it is ignored)",
        unused.join(" ")
    );
    if std::env::var("FORGE_STRICT_FLAGS")
        .map(|v| v == "1")
        .unwrap_or(false)
    {
        eprintln!("[train] FORGE_STRICT_FLAGS=1: stopping");
        std::process::exit(2);
    }
}

fn splitmix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

struct Shard {
    map: Mmap,
    /// number of tokens usable for training (the last shard keeps its tail for validation)
    train_tokens: usize,
}

struct Data {
    shards: Vec<Shard>,
    total_train: usize,
    val_start: usize, // token offset inside the last shard where the validation region starts
    val_tokens: usize,
    /// bit 15 of a u16 token marks "not a target" (chat fine-tuning masks the prompt tokens); only meaningful when the vocabulary fits in 15 bits
    flags: bool,
    /// the cool-down pool (--cool-data), if any
    cool: Option<CoolPool>,
}

/// The cool-down shards: every shard is training data (no validation tail); `held` is the optional held-out file, never trained on.
struct CoolPool {
    shards: Vec<Shard>,
    total: usize,
    held: Option<Mmap>,
}

impl Data {
    fn open(dir: &Path, val_tokens: usize, flags: bool) -> Data {
        let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
            .expect("cannot read the data directory")
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().map(|x| x == "u16").unwrap_or(false))
            .collect();
        files.sort();
        assert!(!files.is_empty(), "no .u16 shards in {}", dir.display());
        let mut shards = Vec::new();
        let n = files.len();
        let mut val_start = 0;
        for (i, f) in files.iter().enumerate() {
            let map = Mmap::open(f).expect("cannot map a shard");
            let tokens = map.len() / 2;
            let mut train_tokens = tokens;
            if i + 1 == n {
                let v = val_tokens.min(tokens / 4);
                train_tokens = tokens - v;
                val_start = train_tokens;
            }
            shards.push(Shard { map, train_tokens });
        }
        let total_train = shards.iter().map(|s| s.train_tokens).sum();
        let last_tokens = shards.last().unwrap().map.len() / 2;
        Data {
            total_train,
            val_tokens: last_tokens - val_start,
            val_start,
            shards,
            flags,
            cool: None,
        }
    }

    /// The cool-down pool of `dir`: every `*.u16` whose name does not end in `_val` (cool_000.u16 ...); the held-out file is `dir/heldout/cool_val.u16` or `dir/cool_val.u16`.
    /// `window`: the longest row the run reads (seq + 1 tokens). Every shard must hold more than one such window and the held-out file at least one: `window_cool` draws a position below
    /// total - window and moves a window that would straddle the end of a shard to the start of the next (or the first) shard, and `cool_val_window` reads whole rows, so a shorter file would
    /// make them read past its end. Checked here, at start-up, instead of failing at the first cool-down window or evaluation (the first decay step: possibly hours after a restart).
    fn open_cool(dir: &Path, window: usize) -> CoolPool {
        let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
            .expect("cannot read the cool-down directory")
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| {
                p.extension().map(|x| x == "u16").unwrap_or(false)
                    && !p
                        .file_stem()
                        .and_then(|s| s.to_str())
                        .map(|s| s.ends_with("_val"))
                        .unwrap_or(false)
            })
            .collect();
        files.sort();
        assert!(
            !files.is_empty(),
            "no cool-down shards (*.u16 files whose name does not end in _val) in {}",
            dir.display()
        );
        let shards: Vec<Shard> = files
            .iter()
            .map(|f| {
                let map = Mmap::open(f).expect("cannot map a cool-down shard");
                let train_tokens = map.len() / 2;
                assert!(train_tokens > window, "cool-down shard {} holds {train_tokens} tokens: every cool-down shard must hold more than one training window ({window} tokens)", f.display());
                Shard { map, train_tokens }
            })
            .collect();
        let total = shards.iter().map(|s| s.train_tokens).sum();
        let held = [dir.join("heldout").join("cool_val.u16"), dir.join("cool_val.u16")].into_iter().find(|p| p.exists()).map(|p| {
            let m = Mmap::open(&p).expect("cannot map the held-out cool-down file");
            assert!(m.len() / 2 >= window, "the held-out cool-down file {} holds {} tokens, less than one validation row ({window} tokens)", p.display(), m.len() / 2);
            m
        });
        CoolPool {
            shards,
            total,
            held,
        }
    }

    fn token(map: &Mmap, i: usize, flags: bool) -> u32 {
        let v = u16::from_le_bytes([map[2 * i], map[2 * i + 1]]) as u32;
        if flags {
            (v & 0x7fff) | ((v & 0x8000) << 16)
        } else {
            v
        }
    }

    /// A training window of `len` tokens chosen by the position hash `h`.
    fn window(&self, h: u64, len: usize, out: &mut Vec<u32>) {
        let mut p = (h % (self.total_train - len) as u64) as usize;
        for s in &self.shards {
            if p + len <= s.train_tokens {
                for i in 0..len {
                    out.push(Self::token(&s.map, p + i, self.flags));
                }
                return;
            }
            if p < s.train_tokens {
                // would straddle the end of the shard: take it from the start of the shard instead
                p = 0;
                continue;
            }
            p -= s.train_tokens;
        }
        // fallback: start of the first shard
        for i in 0..len {
            out.push(Self::token(&self.shards[0].map, i, self.flags));
        }
    }

    fn val_window(&self, k: usize, len: usize, out: &mut Vec<u32>) {
        let last = self.shards.last().unwrap();
        let span = self.val_tokens.saturating_sub(len).max(1);
        let p = self.val_start + (k * 7919 * 4093) % span;
        for i in 0..len {
            out.push(Self::token(&last.map, p + i, self.flags));
        }
    }

    /// Like `window`, from the cool-down pool (same rule for a window that would straddle the end of a shard).
    fn window_cool(&self, h: u64, len: usize, out: &mut Vec<u32>) {
        let c = self.cool.as_ref().expect("no cool-down pool");
        let mut p = (h % (c.total - len) as u64) as usize;
        for s in &c.shards {
            if p + len <= s.train_tokens {
                for i in 0..len {
                    out.push(Self::token(&s.map, p + i, self.flags));
                }
                return;
            }
            if p < s.train_tokens {
                p = 0;
                continue;
            }
            p -= s.train_tokens;
        }
        for i in 0..len {
            out.push(Self::token(&c.shards[0].map, i, self.flags));
        }
    }

    /// A training window from the cool-down pool if `is_cool(h, share)`, else from the main pool.
    fn window_mixed(&self, h: u64, share: f32, len: usize, out: &mut Vec<u32>) {
        if is_cool(h, share) {
            self.window_cool(h, len, out);
        } else {
            self.window(h, len, out);
        }
    }

    /// Validation window k of the held-out cool-down file (the spacing rule of `val_window`, from the start of the file); false when there is no such file.
    fn cool_val_window(&self, k: usize, len: usize, out: &mut Vec<u32>) -> bool {
        let Some(held) = self.cool.as_ref().and_then(|c| c.held.as_ref()) else {
            return false;
        };
        let span = (held.len() / 2).saturating_sub(len).max(1);
        let p = (k * 7919 * 4093) % span;
        for i in 0..len {
            out.push(Self::token(held, p + i, self.flags));
        }
        true
    }
}

/// The shape of the final learning-rate decay of the WSD schedule (t = the fraction of the decay phase done, f = min_lr_frac):
///   linear: peak (1 - (1 - f) t)      sqrt: peak (f + (1 - f)(1 - sqrt(t)))  (the 1-sqrt cooldown of Haegele et al. 2024)      cosine: peak (f + (1 - f)(1 + cos(pi t)) / 2)
#[derive(Clone, Copy, Debug, PartialEq)]
enum DecayShape {
    Linear,
    Sqrt,
    Cosine,
}

fn lr_at(
    step: usize,
    steps: usize,
    peak: f32,
    warmup: usize,
    min_frac: f32,
    decay_frac: f32,
    shape: DecayShape,
) -> f32 {
    if step < warmup {
        return peak * (step + 1) as f32 / warmup as f32;
    }
    let decay_start = decay_start(steps, decay_frac);
    if step < decay_start {
        return peak;
    }
    let t = ((step - decay_start) as f32 / (steps - decay_start).max(1) as f32).min(1.0);
    match shape {
        // (the original expression: the default stays bit for bit what it was)
        DecayShape::Linear => peak * (1.0 - (1.0 - min_frac) * t),
        DecayShape::Sqrt => peak * (min_frac + (1.0 - min_frac) * (1.0 - t.sqrt())),
        DecayShape::Cosine => {
            peak * (min_frac + (1.0 - min_frac) * 0.5 * (1.0 + (std::f32::consts::PI * t).cos()))
        }
    }
}

/// The first step of the learning-rate decay (`lr_at`'s decay_start, for every --decay-shape)
fn decay_start(steps: usize, decay_frac: f32) -> usize {
    ((1.0 - decay_frac) * steps as f32) as usize
}

/// The share of the cool-down pool at (0-based) step `step`: 0 before `start` (default: the first step of the learning-rate decay, the same `decay_start` as `lr_at`), then a linear ramp over `blend` steps up to
/// `max` (blend 0 = a hard switch). A pure function of the step: a resumed run continues the ramp where it stopped.
fn cool_share_at(
    step: usize,
    steps: usize,
    decay_frac: f32,
    blend: usize,
    max: f32,
    start: Option<usize>,
) -> f32 {
    let first = start.unwrap_or(decay_start(steps, decay_frac));
    if step < first {
        return 0.0;
    }
    if blend == 0 {
        return max;
    }
    max * ((step - first) as f32 / blend as f32).min(1.0)
}

/// Does the window with position hash `h` come from the cool-down pool? The share is a probability, the draw a hash of `h` (deterministic: the same step gives the same windows after a resume).
fn is_cool(h: u64, share: f32) -> bool {
    share > 0.0
        && ((splitmix(h ^ 0xC001_D0D0_C001_D0D0) >> 11) as f64 / (1u64 << 53) as f64) < share as f64
}

/// Micro-steps done before optimizer step `step` when step s uses accum_at(s) micro-steps: `sched` holds (micro-steps, from step) in increasing step order, `base` applies before the first entry.
/// Without a schedule this is step * base (the data positions of a run without the schedule do not change).
fn micro_steps_before(step: usize, base: usize, sched: &[(usize, usize)]) -> usize {
    let (mut total, mut at, mut cur) = (0usize, 0usize, base);
    for &(acc, from) in sched {
        if from >= step {
            break;
        }
        total += (from.max(at) - at) * cur;
        at = from.max(at);
        cur = acc;
    }
    total + (step - at) * cur
}

/// micro-steps of optimizer step `step` under the schedule
fn accum_at(step: usize, base: usize, sched: &[(usize, usize)]) -> usize {
    sched
        .iter()
        .filter(|&&(_, from)| from <= step)
        .last()
        .map(|&(acc, _)| acc)
        .unwrap_or(base)
}

pub fn train(args: &[String]) -> i32 {
    let data_dir = PathBuf::from(flag(args, "--data").expect("--data DIR is required"));
    let out_dir = PathBuf::from(flag(args, "--out").unwrap_or("runs/train"));
    std::fs::create_dir_all(&out_dir).expect("cannot create the output directory");
    let seq: usize = num(args, "--seq", 2048);
    let micro: usize = num(args, "--micro", 2);
    let tokens_per_step: usize = num(args, "--tokens-per-step", 131072);
    let steps: usize = num(args, "--steps", 1000);
    // --gpus N: data parallel over N devices of this machine (every optimizer step the gradients of the devices are summed through host memory); the micro-steps are dealt out round robin
    let nranks: usize = num(args, "--gpus", 1).max(1);
    let accum = (tokens_per_step + micro * seq - 1) / (micro * seq);
    let accum = (accum + nranks - 1) / nranks * nranks;
    let tokens_per_step = accum * micro * seq;
    let cfg = TrainConfig {
        n_layer: num(args, "--layers", 4),
        n_embd: num(args, "--dim", 256),
        n_head: num(args, "--heads", 4),
        n_kv: num(args, "--kv", 2),
        head_dim: num(args, "--head-dim", 64),
        n_ff: num(args, "--ffn", 768),
        vocab: num(args, "--vocab", 32768),
        seq_len: seq,
        batch: micro,
        qk_norm: has(args, "--qk-norm"),
        rope_theta: num(args, "--theta", 10000.0),
        eps: 1e-5,
        ce_chunk: num(args, "--ce-chunk", 2048),
        untied: has(args, "--untied"),
        relu2: has(args, "--relu2"),
        softcap: num(args, "--softcap", 0.0),
        lambdas: has(args, "--lambdas"),
        skips: has(args, "--skips"),
        // --value-emb 0,1,10,11: layers with a value embedding
        ve_layers: flag(args, "--value-emb")
            .map(|v| {
                v.split(',')
                    .filter(|x| !x.is_empty())
                    .map(|x| {
                        x.parse()
                            .expect("--value-emb takes layer numbers separated by commas")
                    })
                    .collect()
            })
            .unwrap_or_default(),
        // --ngram 2,3 --ngram-bits 17 --ngram-dim 128: hashed n-gram memory
        ngram: flag(args, "--ngram")
            .map(|v| {
                v.split(',')
                    .filter(|x| !x.is_empty())
                    .map(|x| x.parse().expect("--ngram takes orders separated by commas"))
                    .collect()
            })
            .unwrap_or_default(),
        ngram_bits: num(args, "--ngram-bits", 17),
        ngram_dim: num(args, "--ngram-dim", 128),
        // --attn-gate K: gated attention reading the first K channels of the normalised attention input (K = --dim: the head-wise gate of Qiu et al. 2025)
        attn_gate: num(args, "--attn-gate", 0),
        // --peri-norm: Peri-LN output norms on every sublayer; --emb-norm: its initial embedding normalisation
        peri_norm: has(args, "--peri-norm"),
        emb_norm: has(args, "--emb-norm"),
        // --window 512 [--window-layers auto|0,1,2,...]: sliding-window attention (default off); "auto" (the default list) = three of every four layers, every fourth and the last one dense
        window: num(args, "--window", 0),
        window_layers: match flag(args, "--window-layers") {
            None | Some("auto") => {
                if num(args, "--window", 0usize) > 0 {
                    TrainConfig::auto_window_layers(num(args, "--layers", 4))
                } else {
                    Vec::new()
                }
            }
            Some(v) => v
                .split(',')
                .filter(|x| !x.is_empty())
                .map(|x| {
                    x.parse()
                        .expect("--window-layers takes layer numbers separated by commas, or auto")
                })
                .collect(),
        },
    };
    let peak_lr: f32 = num(args, "--lr", 1e-3);
    let warmup: usize = num(args, "--warmup", 100);
    let min_frac: f32 = num(args, "--min-lr-frac", 0.1);
    let decay_frac: f32 = num(args, "--decay-frac", 0.2);
    // --decay-shape linear|sqrt|cosine: the shape of the final decay (default linear)
    let decay_shape = match flag(args, "--decay-shape").unwrap_or("linear") {
        "linear" => DecayShape::Linear,
        "sqrt" => DecayShape::Sqrt,
        "cosine" => DecayShape::Cosine,
        other => panic!("--decay-shape takes linear, sqrt or cosine, not {other}"),
    };
    // --tps-schedule 32768:0,65536:20000: tokens per optimizer step from the given step on (before the first entry: --tokens-per-step); the learning-rate schedule is unchanged
    let tps_sched: Vec<(usize, usize)> = flag(args, "--tps-schedule")
        .map(|v| {
            let mut e: Vec<(usize, usize)> = v
                .split(',')
                .filter(|x| !x.is_empty())
                .map(|x| {
                    let (t, from) = x.split_once(':').expect("--tps-schedule takes TOKENS:FROM_STEP pairs separated by commas");
                    let t: usize = t.parse().expect("--tps-schedule: tokens");
                    assert!(t > 0 && t % (micro * seq * nranks) == 0, "--tps-schedule: tokens per step must be a multiple of micro x seq x gpus ({})", micro * seq * nranks);
                    (t / (micro * seq), from.parse().expect("--tps-schedule: step"))
                })
                .collect();
            e.sort_by_key(|x| x.1);
            e
        })
        .unwrap_or_default();
    let wd: f32 = num(args, "--wd", 0.1);
    let clip: f32 = num(args, "--clip", 1.0);
    let beta2: f32 = num(args, "--beta2", 0.95);
    let use_muon = flag(args, "--optim") == Some("muon");
    let muon_peak: f32 = num(args, "--muon-lr", peak_lr);
    let muon_wd: f32 = num(args, "--muon-wd", 0.05);
    let muon_mu: f32 = num(args, "--muon-mu", 0.95);
    let muon_opts = MuonOpts {
        polar: has(args, "--muon-polar"),
        iters: num(args, "--muon-iters", 5),
        normuon: if has(args, "--normuon") {
            match flag(args, "--normuon-axis").unwrap_or("long") {
                "long" => Some(NorMuonAxis::Long),
                "rows" => Some(NorMuonAxis::Rows),
                other => panic!("--normuon-axis takes long or rows, not {other}"),
            }
        } else {
            None
        },
        normuon_beta2: num(args, "--normuon-beta2", 0.95),
        cautious_wd: has(args, "--cautious-wd"),
    };
    // --ema 0.999 [--ema-start STEP] [--ema-every K]: exponential moving average of the parameters (kept in host memory) from STEP on (default: the start of the decay phase)
    let ema_decay: f32 = num(args, "--ema", 0.0);
    let ema_every: usize = num(args, "--ema-every", 1usize).max(1);
    let ema_start: usize = num(
        args,
        "--ema-start",
        ((1.0 - decay_frac) * steps as f32) as usize,
    );
    // --seq-schedule 512:300,1024:600: training windows of 512 tokens before step 300, of 1024 before step 600, then --seq (the tokens per step stay the same: more, shorter rows)
    let seq_sched: Vec<(usize, usize)> = flag(args, "--seq-schedule")
        .map(|v| {
            v.split(',')
                .filter(|x| !x.is_empty())
                .map(|x| {
                    let (w, s) = x
                        .split_once(':')
                        .expect("--seq-schedule takes WINDOW:UNTIL_STEP pairs separated by commas");
                    (
                        w.parse().expect("--seq-schedule: window"),
                        s.parse().expect("--seq-schedule: step"),
                    )
                })
                .collect()
        })
        .unwrap_or_default();
    for &(w, _) in &seq_sched {
        assert!(
            w > 0 && w % 64 == 0 && w <= seq && (micro * seq) % w == 0,
            "--seq-schedule: every window must be a multiple of 64 that divides micro x seq ({})",
            micro * seq
        );
    }
    let seed: u64 = num(args, "--seed", 1);
    let save_every: usize = num(args, "--save-every", 500);
    let eval_every: usize = num(args, "--eval-every", 100);
    let eval_batches: usize = num(args, "--eval-batches", 8);
    let mut data = Data::open(
        &data_dir,
        num(args, "--val-tokens", 20_000_000),
        cfg.vocab <= 32768,
    );
    let (cool_blend, cool_max, cool_start): (usize, f32, Option<usize>) = (
        num(args, "--cool-blend-steps", 0),
        num(args, "--cool-share", 0.6),
        flag(args, "--cool-start").and_then(|v| v.parse().ok()),
    );
    if let Some(dir) = flag(args, "--cool-data") {
        let c = Data::open_cool(Path::new(dir), seq + 1);
        let first = cool_start.unwrap_or(decay_start(steps, decay_frac));
        eprintln!(
            "[train] cool-down data: {} shards, {:.1} M tokens, held-out file: {}; cool-down share 0 before step {}, then {}",
            c.shards.len(),
            c.total as f64 / 1e6,
            if c.held.is_some() { "yes" } else { "no" },
            first,
            if cool_blend == 0 { format!("a hard switch to {cool_max:.2}") } else { format!("a linear ramp over {cool_blend} steps up to {cool_max:.2}") }
        );
        data.cool = Some(c);
    }
    eprintln!(
        "[train] data: {} shards, {:.2} G training tokens, {:.1} M validation tokens; model {} layers, dim {}, {}/{} heads, ffn {}, vocab {}, seq {}, micro-batch {} x {}, {} micro-steps per optimizer step ({} tokens)",
        data.shards.len(),
        data.total_train as f64 / 1e9,
        data.val_tokens as f64 / 1e6,
        cfg.n_layer,
        cfg.n_embd,
        cfg.n_head,
        cfg.n_kv,
        cfg.n_ff,
        cfg.vocab,
        seq,
        micro,
        seq,
        accum,
        tokens_per_step
    );
    let p = Params {
        seq,
        micro,
        tokens_per_step,
        steps,
        accum,
        peak_lr,
        warmup,
        min_frac,
        decay_frac,
        wd,
        clip,
        beta2,
        use_muon,
        muon_peak,
        muon_wd,
        muon_mu,
        muon_opts,
        seed,
        save_every,
        eval_every,
        eval_batches,
        mults: (
            num(args, "--lr-emb-mult", 1.0),
            num(args, "--lr-head-mult", 1.0),
            num(args, "--lr-table-mult", 1.0),
            num(args, "--lr-scalar-mult", 1.0),
        ),
        gate_mult: num(args, "--lr-gate-mult", 1.0),
        time_limit_s: num(args, "--time-limit-s", 0.0),
        ema_decay,
        ema_every,
        ema_start,
        seq_sched,
        decay_shape,
        tps_sched,
        cool_blend,
        cool_max,
        cool_start,
    };
    let shared = Shared {
        barrier: std::sync::Barrier::new(nranks),
        grads: std::sync::Mutex::new(vec![Vec::new(); nranks]),
        losses: std::sync::Mutex::new(vec![0.0; nranks]),
        stop: std::sync::atomic::AtomicBool::new(false),
    };
    let (data_r, cfg_r, out_r, shared_r, p_r) = (&data, &cfg, &out_dir, &shared, &p);
    let code = std::thread::scope(|sc| {
        let handles: Vec<_> = (0..nranks)
            .map(|rank| {
                sc.spawn(move || worker(rank, nranks, shared_r, data_r, cfg_r, out_r, p_r, args))
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().unwrap_or(1))
            .fold(0, i32::max)
    });
    if std::env::var("FORGE_KPROF")
        .map(|v| v != "0")
        .unwrap_or(false)
    {
        eprintln!(
            "[kprof] kernel time by kernel (all steps):\n{}",
            forge_cuda::kprof_report()
        );
    }
    code
}

/// Everything one worker needs to know besides the data and the model configuration.
struct Params {
    seq: usize,
    micro: usize,
    tokens_per_step: usize,
    steps: usize,
    accum: usize,
    peak_lr: f32,
    warmup: usize,
    min_frac: f32,
    decay_frac: f32,
    wd: f32,
    clip: f32,
    beta2: f32,
    use_muon: bool,
    muon_peak: f32,
    muon_wd: f32,
    muon_mu: f32,
    /// Polar Express, iterations, NorMuon, cautious decay (default: the plain Muon update)
    muon_opts: MuonOpts,
    seed: u64,
    save_every: usize,
    eval_every: usize,
    eval_batches: usize,
    mults: (f32, f32, f32, f32),
    /// AdamW learning-rate multiplier of the attention gate weights
    gate_mult: f32,
    /// stop after the first optimizer step past this many seconds (a checkpoint is written first); 0 = no limit
    time_limit_s: f64,
    /// exponential moving average of the parameters: decay per optimizer step (0 = off), updated every `ema_every` steps from step `ema_start` on
    ema_decay: f32,
    ema_every: usize,
    ema_start: usize,
    /// (window, until step): shorter training windows early (constant tokens per step)
    seq_sched: Vec<(usize, usize)>,
    /// the shape of the final learning-rate decay
    decay_shape: DecayShape,
    /// (micro-steps per optimizer step, from step): a tokens-per-step schedule (accum before the first entry)
    tps_sched: Vec<(usize, usize)>,
    /// cool-down data (--cool-data): steps of the linear ramp of its share, the share after the ramp, the first step (None = the first step of the learning-rate decay)
    cool_blend: usize,
    cool_max: f32,
    cool_start: Option<usize>,
}

/// Exponential moving average of the parameters in host memory (no GPU memory): every `every` optimizer steps from `start` on, ema = d^every ema + (1 - d^every) params, seeded with the parameters at
/// its first update. Evaluated next to the raw parameters (`val_ema`) and written as ckpt_ema.bin (a normal checkpoint whose parameters are the average).
struct Ema {
    decay: f64,
    every: usize,
    start: usize,
    w: Option<Vec<f32>>,
}

impl Ema {
    /// is an update due after `done` optimizer steps?
    fn due(&self, done: usize) -> bool {
        done >= self.start && (done - self.start) % self.every == 0
    }
    fn update(&mut self, p: &[f32]) {
        match &mut self.w {
            None => self.w = Some(p.to_vec()),
            Some(w) => {
                let d = self.decay.powi(self.every as i32) as f32;
                for (a, b) in w.iter_mut().zip(p) {
                    *a = d * *a + (1.0 - d) * *b;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        accum_at, eval_json_fields, flag, lr_at, micro_steps_before, num, DecayShape, Ema,
    };

    #[test]
    fn a_flag_given_twice_with_different_values_is_refused() {
        let args = |s: &str| s.split(' ').map(String::from).collect::<Vec<String>>();
        // launch_f1.sh: --steps ${STEPS:-45000} ... then F1_ARGS: the override used to be ignored (shown with the CLI: --steps 20 ... --steps 40 trained 20 steps)
        let dup = args("train --data d --steps 45000 --decay-frac 0.1 --relu2 --steps 90000");
        assert!(
            std::panic::catch_unwind(|| num(&dup, "--steps", 1000usize)).is_err(),
            "--steps 45000 and --steps 90000 must be refused"
        );
        assert_eq!(num(&dup, "--decay-frac", 0.2f32), 0.1);
        // the same value twice, a single value, a missing flag: as before
        let same = args("train --steps 45000 --qk-norm --steps 45000");
        assert_eq!(num(&same, "--steps", 1000usize), 45000);
        assert_eq!(flag(&same, "--data"), None);
        assert_eq!(flag(&args("train --out runs/x"), "--out"), Some("runs/x"));
    }

    #[test]
    fn evaluation_lines_keep_the_unmasked_and_the_average_loss_together() {
        // --doc-mask with --ema: both fields (the --ema field used to replace val_loss_unmasked: shown with the CLI, log.jsonl had only val_loss_ema)
        let both = eval_json_fields(Some(6.9408), Some(6.9394));
        assert_eq!(
            both,
            ",\"val_loss_unmasked\":6.9408,\"val_loss_ema\":6.9394"
        );
        assert_eq!(
            eval_json_fields(Some(1.5), None),
            ",\"val_loss_unmasked\":1.5"
        );
        assert_eq!(eval_json_fields(None, Some(2.25)), ",\"val_loss_ema\":2.25");
        // neither option: the line is what it always was
        assert_eq!(eval_json_fields(None, None), "");
    }

    #[test]
    fn decay_shapes_follow_their_formulas() {
        // 100 steps, warmup 10, decay over the last 20 % (steps 80..99) to 10 %
        let at = |s: usize, sh: DecayShape| lr_at(s, 100, 1.0, 10, 0.1, 0.2, sh);
        for sh in [DecayShape::Linear, DecayShape::Sqrt, DecayShape::Cosine] {
            assert_eq!(at(4, sh), 0.5);
            assert_eq!(at(50, sh), 1.0);
            assert_eq!(at(80, sh), 1.0);
        }
        // t = 10 / 20 = 0.5
        assert!((at(90, DecayShape::Linear) - 0.55).abs() < 1e-6);
        assert!((at(90, DecayShape::Sqrt) - (0.1 + 0.9 * (1.0 - 0.5f32.sqrt()))).abs() < 1e-6);
        assert!((at(90, DecayShape::Cosine) - 0.55).abs() < 1e-6);
        // t = 5 / 20: sqrt is below linear, cosine above
        assert!(
            at(85, DecayShape::Sqrt) < at(85, DecayShape::Linear)
                && at(85, DecayShape::Cosine) > at(85, DecayShape::Linear)
        );
        // the last step: t = 19 / 20
        assert!((at(99, DecayShape::Linear) - (1.0 - 0.9 * 0.95)).abs() < 1e-6);
    }

    #[test]
    fn tokens_per_step_schedule_counts_micro_steps() {
        // 4 micro-steps per step before step 10, 8 from step 10 on, 2 from step 20 on
        let sched = [(8usize, 10usize), (2, 20)];
        assert_eq!(micro_steps_before(0, 4, &sched), 0);
        assert_eq!(micro_steps_before(10, 4, &sched), 40);
        assert_eq!(micro_steps_before(12, 4, &sched), 56);
        assert_eq!(micro_steps_before(25, 4, &sched), 40 + 80 + 10);
        assert_eq!(
            (
                accum_at(9, 4, &sched),
                accum_at(10, 4, &sched),
                accum_at(19, 4, &sched),
                accum_at(20, 4, &sched)
            ),
            (4, 8, 8, 2)
        );
        // a schedule that starts at step 0, and no schedule (the data positions of a run without one stay step x accum)
        assert_eq!(micro_steps_before(7, 4, &[(6, 0)]), 42);
        assert_eq!(micro_steps_before(7, 4, &[]), 28);
        for s in 0..30 {
            assert_eq!(
                micro_steps_before(s + 1, 4, &sched) - micro_steps_before(s, 4, &sched),
                accum_at(s, 4, &sched)
            );
        }
    }

    #[test]
    fn the_parameter_average_seeds_then_decays_every_k_steps() {
        let mut e = Ema {
            decay: 0.5,
            every: 2,
            start: 10,
            w: None,
        };
        assert!(!e.due(9) && e.due(10) && !e.due(11) && e.due(12));
        e.update(&[4.0, 8.0]);
        assert_eq!(e.w.as_deref(), Some(&[4.0f32, 8.0][..]));
        // two steps at decay 0.5: the old average keeps a weight of 0.25
        e.update(&[0.0, 0.0]);
        assert_eq!(e.w.as_deref(), Some(&[1.0f32, 2.0][..]));
    }
}

/// The optional JSON fields of an evaluation line of log.jsonl: the unmasked validation loss (--doc-mask) and the validation loss of the parameter average (--ema), each when it was measured
/// (empty without both options). Both fields together used to lose val_loss_unmasked (the --ema field replaced the string instead of extending it).
fn eval_json_fields(val_unmasked: Option<f32>, val_ema: Option<f32>) -> String {
    let mut s = String::new();
    if let Some(vu) = val_unmasked {
        s.push_str(&format!(",\"val_loss_unmasked\":{}", vu));
    }
    if let Some(ve) = val_ema {
        s.push_str(&format!(",\"val_loss_ema\":{}", ve));
    }
    s
}

/// Mean loss on the held-out cool-down file over `eval_batches` micro-batches of full-length rows (the training window is put back afterwards).
fn cool_val_loss(
    tr: &mut Trainer,
    data: &Data,
    micro: usize,
    seq: usize,
    eval_batches: usize,
    window: &mut Vec<u32>,
) -> f32 {
    let saved = tr.window();
    tr.set_window(None).unwrap();
    let mut vc = 0f32;
    for k in 0..eval_batches {
        window.clear();
        for s in 0..micro {
            data.cool_val_window(k * micro + s, seq + 1, window);
        }
        vc += tr.eval_loss(window).unwrap();
    }
    tr.set_window(saved).unwrap();
    vc / eval_batches as f32
}

/// Mean validation loss over `eval_batches` micro-batches of full-length rows (the training window, if a schedule set one, is put back afterwards).
fn val_loss(
    tr: &mut Trainer,
    data: &Data,
    micro: usize,
    seq: usize,
    eval_batches: usize,
    window: &mut Vec<u32>,
) -> f32 {
    let saved = tr.window();
    tr.set_window(None).unwrap();
    let mut vl = 0f32;
    for k in 0..eval_batches {
        window.clear();
        for s in 0..micro {
            data.val_window(k * micro + s, seq + 1, window);
        }
        vl += tr.eval_loss(window).unwrap();
    }
    tr.set_window(saved).unwrap();
    vl / eval_batches as f32
}

/// What the workers of a data-parallel run exchange: the gradients (summed after every optimizer step's micro-steps) and the losses (for the log).
struct Shared {
    barrier: std::sync::Barrier,
    grads: std::sync::Mutex<Vec<Vec<f32>>>,
    losses: std::sync::Mutex<Vec<f32>>,
    /// set by worker 0 when the time limit is reached: every worker leaves its loop at the start of the next step
    stop: std::sync::atomic::AtomicBool,
}

/// One device's training loop. With several workers the micro-steps of every optimizer step are dealt out round robin, the gradient sums are exchanged through host memory, and every worker applies the same
/// update (same gradients, same state: the parameters stay identical, nothing else is communicated); worker 0 evaluates, logs and writes the checkpoints.
#[allow(clippy::too_many_arguments)]
fn worker(
    rank: usize,
    nranks: usize,
    shared: &Shared,
    data: &Data,
    cfg: &TrainConfig,
    out_dir: &Path,
    p: &Params,
    args: &[String],
) -> i32 {
    let Params {
        seq,
        micro,
        tokens_per_step: _,
        steps,
        accum,
        peak_lr,
        warmup,
        min_frac,
        decay_frac,
        wd,
        clip,
        beta2,
        use_muon,
        muon_peak,
        muon_wd,
        muon_mu,
        muon_opts,
        seed,
        save_every,
        eval_every,
        eval_batches,
        mults,
        time_limit_s,
        ema_decay,
        ema_every,
        ema_start,
        decay_shape,
        cool_blend,
        cool_max,
        cool_start,
        ..
    } = *p;
    // FORGE_GPU_ORDINALS=0,0 maps the workers to device numbers (a test of the data-parallel path on a machine with one GPU)
    let ordinal = std::env::var("FORGE_GPU_ORDINALS")
        .ok()
        .and_then(|v| v.split(',').nth(rank).and_then(|x| x.trim().parse().ok()))
        .unwrap_or(rank);
    let mut tr = match Trainer::new(cfg.clone(), ordinal) {
        Ok(t) => t,
        Err(e) => {
            // the other workers would wait for this one at the barrier forever
            eprintln!("cannot create the trainer on device {ordinal}: {e}");
            std::process::exit(1);
        }
    };
    tr.set_lr_mults(mults.0, mults.1, mults.2, mults.3);
    tr.set_gate_lr_mult(p.gate_mult);
    tr.set_adam_cautious_wd(has(args, "--cautious-wd-adam"));
    // --doc-mask [--eos ID]: document masking (the end-of-text id is 0 in our tokenizers)
    if has(args, "--doc-mask") {
        let eos = num(args, "--eos", 0u32);
        if let Err(e) = tr.set_doc_mask(Some(eos)) {
            eprintln!("--doc-mask: {e}");
            std::process::exit(1);
        }
        if rank == 0 {
            eprintln!("[train] document masking (end-of-text id {eos}): attention runs on the masked first-generation kernels (flash_doc.cu), not on the newer kernels of the dense layers");
        }
    }
    if let Err(e) = tr.set_muon_opts(muon_opts) {
        eprintln!("bad Muon options: {e}");
        std::process::exit(1);
    }
    if rank == 0 {
        eprintln!("[train] structure: {}; AdamW learning-rate multipliers: embedding {}, head {}, lookup tables {}, scalars {}", cfg.structure(), mults.0, mults.1, mults.2, mults.3);
        eprintln!(
            "[train] {} on {}: {:.1} M parameters, optimizer {}{}",
            tr.device_info(),
            out_dir.display(),
            tr.n_params as f64 / 1e6,
            if use_muon {
                format!("Muon (lr {muon_peak}, wd {muon_wd}, momentum {muon_mu}; {}) + AdamW for embedding/gains", muon_opts.describe())
            } else {
                "AdamW".to_string()
            },
            if nranks > 1 {
                format!("; data parallel over {nranks} devices")
            } else {
                String::new()
            }
        );
    }
    let mut loss_scale: f32 = num(args, "--loss-scale", 32768.0);
    let mut start_step = 0usize;
    let ckpt = out_dir.join("ckpt_latest.bin");
    if has(args, "--resume") && ckpt.exists() {
        loss_scale = tr
            .load_checkpoint(&ckpt)
            .expect("cannot load the checkpoint");
        start_step = tr.step_count;
        if rank == 0 {
            eprintln!("[train] resumed from step {start_step}, loss scale {loss_scale}");
        }
    } else if let Some(init) = flag(args, "--init-from") {
        // fine-tuning: the parameters of an earlier run, a fresh optimizer and a step counter at 0
        tr.load_params(Path::new(init))
            .expect("cannot load the parameters to start from");
        if rank == 0 {
            eprintln!("[train] starting from the parameters of {init}");
        }
    } else {
        tr.init_random(seed).expect("init failed");
    }
    // the average of the parameters (worker 0 only; continued from ckpt_ema.bin when a run resumes)
    let ema_ckpt = out_dir.join("ckpt_ema.bin");
    let mut ema = if ema_decay > 0.0 && rank == 0 {
        let mut e = Ema {
            decay: ema_decay as f64,
            every: ema_every,
            start: ema_start,
            w: None,
        };
        if has(args, "--resume") && ema_ckpt.exists() && start_step > 0 {
            e.w = forge_train::read_checkpoint_params(&ema_ckpt)
                .ok()
                .filter(|w| w.len() == tr.n_params);
        }
        eprintln!("[train] parameter average: decay {ema_decay} per step, every {ema_every} steps from step {ema_start}{}", if e.w.is_some() { " (continued from ckpt_ema.bin)" } else { "" });
        Some(e)
    } else {
        None
    };
    // the training window of a step (None = --seq)
    let window_at = |step: usize| -> Option<usize> {
        p.seq_sched
            .iter()
            .find(|&&(_, until)| step < until)
            .map(|&(w, _)| w)
            .filter(|&w| w != seq)
    };
    let mut log = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(out_dir.join("log.jsonl"))
        .expect("cannot open the log");
    if rank == 0 {
        report_unused_flags(args); // every lookup of the configuration has been made by now
    }
    let mut clean_steps = 0usize;
    let t_start = Instant::now();
    let mut window: Vec<u32> = Vec::new();
    for step in start_step..steps {
        if nranks > 1 {
            shared.barrier.wait();
        }
        if shared.stop.load(std::sync::atomic::Ordering::SeqCst) {
            break;
        }
        let t0 = Instant::now();
        let lr = lr_at(
            step,
            steps,
            peak_lr,
            warmup,
            min_frac,
            decay_frac,
            decay_shape,
        );
        // this step's micro-steps and tokens (a tokens-per-step schedule changes them), and the micro-steps of the run so far (they index the data positions)
        let accum_base = accum;
        let accum = accum_at(step, accum_base, &p.tps_sched);
        let tokens_per_step = accum * micro * seq;
        let done_micro = micro_steps_before(step, accum_base, &p.tps_sched);
        let win = window_at(step);
        // the share of the cool-down pool at this step (0 without --cool-data)
        let share = if data.cool.is_some() {
            cool_share_at(step, steps, decay_frac, cool_blend, cool_max, cool_start)
        } else {
            0.0
        };
        tr.set_window(win).unwrap();
        tr.zero_grads().unwrap();
        let mut loss_sum = 0f32;
        for mi in 0..accum {
            if mi % nranks != rank {
                continue;
            }
            window.clear();
            match win {
                None => {
                    for s in 0..micro {
                        let h =
                            splitmix(seed ^ splitmix((done_micro * micro + mi * micro + s) as u64));
                        data.window_mixed(h, share, seq + 1, &mut window);
                    }
                }
                Some(w) => {
                    // a shorter window: more rows of w + 1 tokens (own positions: the hash also takes the window length)
                    let rows = micro * seq / w;
                    for s in 0..rows {
                        let h = splitmix(
                            seed ^ splitmix(((done_micro + mi) * rows + s) as u64)
                                ^ splitmix(0x5E9_0000 + w as u64),
                        );
                        data.window_mixed(h, share, w + 1, &mut window);
                    }
                }
            }
            loss_sum += tr
                .forward_backward(&window, loss_scale, tokens_per_step)
                .unwrap();
        }
        if nranks > 1 {
            // sum the gradients of all devices through host memory (every worker computes the same sum), and the losses
            shared.losses.lock().unwrap()[rank] = loss_sum;
            shared.grads.lock().unwrap()[rank] = tr.download_grads().unwrap();
            shared.barrier.wait();
            {
                let all = shared.grads.lock().unwrap();
                let mut sum = all[rank].clone();
                for (r, g) in all.iter().enumerate() {
                    if r != rank {
                        for (a, b) in sum.iter_mut().zip(g) {
                            *a += *b;
                        }
                    }
                }
                drop(all);
                tr.upload_grads(&sum).unwrap();
                loss_sum = shared.losses.lock().unwrap().iter().sum();
            }
            shared.barrier.wait();
            shared.grads.lock().unwrap()[rank] = Vec::new();
        }
        let loss = loss_sum / accum as f32;
        let stats = if use_muon {
            tr.step_muon(
                lr,
                lr * muon_peak / peak_lr,
                0.9,
                beta2,
                1e-8,
                wd,
                muon_wd,
                muon_mu,
                clip,
                loss_scale,
            )
            .unwrap()
        } else {
            tr.step(lr, 0.9, beta2, 1e-8, wd, clip, loss_scale).unwrap()
        };
        if stats.found_inf {
            loss_scale = (loss_scale / 2.0).max(1.0);
            clean_steps = 0;
            if rank == 0 {
                eprintln!("[train] step {step}: non-finite gradients, update skipped, loss scale -> {loss_scale}");
            }
        } else {
            clean_steps += 1;
            if clean_steps % 1000 == 0 {
                loss_scale = (loss_scale * 2.0).min(1048576.0);
            }
        }
        if rank != 0 {
            continue;
        }
        if let Some(e) = ema.as_mut() {
            if !stats.found_inf && e.due(step + 1) {
                e.update(&tr.download_params().unwrap());
            }
        }
        let dt = t0.elapsed().as_secs_f64();
        let tps = tokens_per_step as f64 / dt;
        // training tokens of the run after this step (the log carries them: runs with different tokens per step are compared at equal tokens)
        let tokens_done = micro_steps_before(step + 1, accum_base, &p.tps_sched) * micro * seq;
        let mut line = format!(
            "step {:5}/{} loss {:.4} lr {:.2e} gnorm {:.3} scale {:.0} {:.0} tok/s ({:.2} s)",
            step + 1,
            steps,
            loss,
            lr,
            stats.grad_norm,
            loss_scale,
            tps,
            dt
        );
        let mut extra = String::new(); // extra JSON fields of the cool-down data (empty without --cool-data)
        if data.cool.is_some() {
            line.push_str(&format!(" cool {share:.2}"));
            extra.push_str(&format!(",\"cool_share\":{share}"));
        }
        if (step + 1) % eval_every == 0 || step + 1 == steps {
            let vl = val_loss(&mut tr, data, micro, seq, eval_batches, &mut window);
            line.push_str(&format!("  val {:.4}", vl));
            // the parameter average, evaluated on the same batches (the raw parameters are put back afterwards)
            let (mut v_unmasked, mut v_ema) = (None, None);
            if let Some(eos) = tr.doc_mask() {
                // with document masking the validation above is masked too; the plain causal number is the one comparable with runs without the mask
                tr.set_doc_mask(None).unwrap();
                let vu = val_loss(&mut tr, data, micro, seq, eval_batches, &mut window);
                tr.set_doc_mask(Some(eos)).unwrap();
                line.push_str(&format!("  val_unmasked {:.4}", vu));
                v_unmasked = Some(vu);
            }
            if let Some(w) = ema.as_ref().and_then(|e| e.w.as_ref()) {
                let raw = tr.download_params().unwrap();
                tr.upload_params(w).unwrap();
                let ve = val_loss(&mut tr, data, micro, seq, eval_batches, &mut window);
                tr.upload_params(&raw).unwrap();
                line.push_str(&format!("  val_ema {:.4}", ve));
                v_ema = Some(ve);
            }
            let ema_json = eval_json_fields(v_unmasked, v_ema);
            if data
                .cool
                .as_ref()
                .map(|c| c.held.is_some())
                .unwrap_or(false)
            {
                // the held-out cool-down file (same windows at every evaluation, full-length rows)
                let vc = cool_val_loss(&mut tr, data, micro, seq, eval_batches, &mut window);
                line.push_str(&format!("  val_cool {:.4}", vc));
                extra.push_str(&format!(",\"val_cool\":{vc}"));
            }
            if let Some(w) = win {
                line.push_str(&format!("  window {w}"));
            }
            writeln!(log, "{{\"step\":{},\"loss\":{},\"val_loss\":{}{},\"lr\":{},\"grad_norm\":{},\"tokens_per_s\":{},\"tokens\":{}{}}}", step + 1, loss, vl, ema_json, lr, stats.grad_norm, tps, tokens_done, extra).ok();
        } else {
            writeln!(log, "{{\"step\":{},\"loss\":{},\"lr\":{},\"grad_norm\":{},\"tokens_per_s\":{},\"tokens\":{}{}}}", step + 1, loss, lr, stats.grad_norm, tps, tokens_done, extra).ok();
        }
        println!("{line}");
        std::io::stdout().flush().ok();
        let save_ema = |tr: &Trainer| {
            if let Some(w) = ema.as_ref().and_then(|e| e.w.as_ref()) {
                tr.save_checkpoint_params(&ema_ckpt, loss_scale, w)
                    .expect("checkpoint of the parameter average failed");
            }
        };
        if (step + 1) % save_every == 0 || step + 1 == steps {
            tr.save_checkpoint(&ckpt, loss_scale)
                .expect("checkpoint failed");
            save_ema(&tr);
            eprintln!(
                "[train] checkpoint at step {} ({:.0} s elapsed)",
                step + 1,
                t_start.elapsed().as_secs_f32()
            );
        } else if time_limit_s > 0.0 && t_start.elapsed().as_secs_f64() > time_limit_s {
            tr.save_checkpoint(&ckpt, loss_scale)
                .expect("checkpoint failed");
            save_ema(&tr);
            eprintln!(
                "[train] time limit of {time_limit_s:.0} s reached: checkpoint at step {} and stop",
                step + 1
            );
            shared.stop.store(true, std::sync::atomic::Ordering::SeqCst);
        }
        if time_limit_s > 0.0
            && (step + 1) % save_every == 0
            && t_start.elapsed().as_secs_f64() > time_limit_s
        {
            shared.stop.store(true, std::sync::atomic::Ordering::SeqCst);
        }
    }
    0
}

#[cfg(test)]
mod cool_tests {
    use super::*;

    /// shard files whose tokens tell where they come from: id * 100 + position % 100 (token / 100 = the shard id)
    fn write_shard(path: &Path, id: u16, n: usize) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        let mut bytes = Vec::with_capacity(2 * n);
        for i in 0..n {
            bytes.extend_from_slice(&(id * 100 + (i % 100) as u16).to_le_bytes());
        }
        std::fs::write(path, bytes).unwrap();
    }

    fn fresh_dir(name: &str) -> PathBuf {
        let d =
            std::env::temp_dir().join(format!("forge_cool_test_{}_{}", name, std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn share_schedule() {
        let first = ((1.0 - 0.1f32) * 1000.0f32) as usize; // the decay start of lr_at (about 900)
        assert_eq!(cool_share_at(0, 1000, 0.1, 100, 0.8, None), 0.0);
        assert_eq!(cool_share_at(first - 1, 1000, 0.1, 100, 0.8, None), 0.0);
        assert_eq!(cool_share_at(first, 1000, 0.1, 100, 0.8, None), 0.0); // the first decay step: the ramp starts at 0
        assert!((cool_share_at(first + 50, 1000, 0.1, 100, 0.8, None) - 0.4).abs() < 1e-6);
        assert_eq!(cool_share_at(first + 100, 1000, 0.1, 100, 0.8, None), 0.8);
        assert_eq!(cool_share_at(first + 5000, 1000, 0.1, 100, 0.8, None), 0.8);
        assert_eq!(cool_share_at(first - 1, 1000, 0.1, 0, 0.8, None), 0.0); // hard switch
        assert_eq!(cool_share_at(first, 1000, 0.1, 0, 0.8, None), 0.8);
        assert_eq!(cool_share_at(9, 1000, 0.1, 100, 1.0, Some(10)), 0.0); // absolute start
        assert!((cool_share_at(60, 1000, 0.1, 100, 1.0, Some(10)) - 0.5).abs() < 1e-6);
        // the real run: 45,000 steps, decay over the last 10 %
        let f = ((1.0 - 0.1f32) * 45000.0f32) as usize;
        assert_eq!(f, 40500);
        assert_eq!(cool_share_at(40499, 45000, 0.1, 1125, 0.8, None), 0.0);
        assert!((cool_share_at(40500 + 1125 / 2, 45000, 0.1, 1125, 0.8, None) - 0.4).abs() < 1e-3);
        assert_eq!(cool_share_at(44999, 45000, 0.1, 1125, 0.8, None), 0.8);
    }

    #[test]
    fn share_of_windows_is_the_probability() {
        for share in [0.0f32, 0.1, 0.5, 0.8, 1.0] {
            let n = 200_000u64;
            let hits = (0..n).filter(|&i| is_cool(splitmix(i), share)).count() as f64 / n as f64;
            assert!((hits - share as f64).abs() < 0.004, "share {share}: {hits}");
        }
        // deterministic: the same hash always decides the same way (a resumed run sees the same windows)
        for i in 0..1000u64 {
            assert_eq!(is_cool(splitmix(i), 0.5), is_cool(splitmix(i), 0.5));
        }
    }

    #[test]
    fn pools_validation_and_windows() {
        let main = fresh_dir("main");
        let cool = fresh_dir("cool");
        write_shard(&main.join("a_000.u16"), 1, 10_000);
        write_shard(&main.join("a_001.u16"), 2, 10_000);
        write_shard(&main.join("zzz_val.u16"), 3, 4_000);
        write_shard(&cool.join("cool_000.u16"), 11, 6_000);
        write_shard(&cool.join("cool_001.u16"), 12, 6_000);
        write_shard(&cool.join("heldout").join("cool_val.u16"), 20, 3_000); // held out
        write_shard(&cool.join("cool_extra_val.u16"), 21, 3_000); // a stray *_val file: never trained on
        let mut data = Data::open(&main, 1000, true);
        // the validation of the main pool: the last 1000 tokens of the last shard (zzz_val), exactly as without a cool-down pool
        assert_eq!(
            (data.total_train, data.val_tokens, data.val_start),
            (23_000, 1000, 3_000)
        );
        let c = Data::open_cool(&cool, 17);
        assert_eq!(c.shards.len(), 2);
        assert_eq!(c.total, 12_000);
        assert!(c.held.is_some());
        data.cool = Some(c);
        let mut w = Vec::new();
        let mut seen_main = std::collections::BTreeSet::new();
        let mut seen_cool = std::collections::BTreeSet::new();
        for i in 0..5000u64 {
            let h = splitmix(i);
            w.clear();
            data.window(h, 16, &mut w);
            let ids: std::collections::BTreeSet<u32> = w.iter().map(|t| t / 100).collect();
            assert_eq!(ids.len(), 1, "a window straddles two shards: {ids:?}");
            seen_main.extend(ids);
            w.clear();
            data.window_cool(h, 16, &mut w);
            let ids: std::collections::BTreeSet<u32> = w.iter().map(|t| t / 100).collect();
            assert_eq!(
                ids.len(),
                1,
                "a cool-down window straddles two shards: {ids:?}"
            );
            seen_cool.extend(ids);
        }
        assert_eq!(seen_main, [1, 2, 3].into_iter().collect()); // zzz_val's training part (3000 tokens) is part of the pool, its tail is not
        assert_eq!(seen_cool, [11, 12].into_iter().collect()); // neither heldout/cool_val.u16 (20) nor cool_extra_val.u16 (21)
        for k in 0..8 {
            w.clear();
            assert!(data.cool_val_window(k, 16, &mut w));
            assert!(w.iter().all(|t| t / 100 == 20));
        }
        // a cool-down directory without a held-out file
        let cool2 = fresh_dir("cool2");
        write_shard(&cool2.join("cool_000.u16"), 11, 6_000);
        let mut d2 = Data::open(&main, 1000, true);
        d2.cool = Some(Data::open_cool(&cool2, 17));
        assert!(!d2.cool_val_window(0, 16, &mut w));
        // the validation windows of the main pool do not depend on the cool-down pool
        let (mut a, mut b) = (Vec::new(), Vec::new());
        for k in 0..4 {
            data.val_window(k, 16, &mut a);
            d2.val_window(k, 16, &mut b);
        }
        assert_eq!(a, b);
        assert!(a.iter().all(|t| t / 100 == 3));
        for d in [main, cool, cool2] {
            let _ = std::fs::remove_dir_all(d);
        }
    }

    #[test]
    fn a_cool_down_pool_too_short_for_a_window_is_refused_at_start_up() {
        // a shard shorter than one training window: open_cool used to accept it, and the first cool-down window (the first decay step, possibly hours after a restart) then read past the end of
        // the file and the trainer panicked (shown with the CLI: "index out of bounds: the len is 200 but the index is 200" at the first step with a cool-down share)
        let cool = fresh_dir("short");
        write_shard(&cool.join("cool_000.u16"), 11, 100);
        assert!(
            std::panic::catch_unwind(|| Data::open_cool(&cool, 129)).is_err(),
            "a 100-token shard must be refused for windows of 129 tokens"
        );
        assert!(
            std::panic::catch_unwind(|| Data::open_cool(&cool, 100)).is_err(),
            "a shard must hold MORE than one window (window_cool draws below total - window)"
        );
        // windows that fit are served from inside the shard
        let mut data = Data::open(&cool, 10, true);
        data.cool = Some(Data::open_cool(&cool, 65));
        let mut w = Vec::new();
        for i in 0..2000u64 {
            w.clear();
            data.window_cool(splitmix(i), 65, &mut w);
            assert!(w.len() == 65 && w.iter().all(|t| t / 100 == 11));
        }
        // a held-out file shorter than one validation row is refused too (cool_val_window reads whole rows)
        let cool2 = fresh_dir("short_held");
        write_shard(&cool2.join("cool_000.u16"), 11, 6_000);
        write_shard(&cool2.join("heldout").join("cool_val.u16"), 20, 50);
        assert!(
            std::panic::catch_unwind(|| Data::open_cool(&cool2, 129)).is_err(),
            "a 50-token held-out file must be refused for rows of 129 tokens"
        );
        assert!(Data::open_cool(&cool2, 50).held.is_some());
        for d in [cool, cool2] {
            let _ = std::fs::remove_dir_all(d);
        }
    }

    #[test]
    fn share_starts_at_the_first_decay_step_of_every_decay_shape() {
        for shape in [DecayShape::Linear, DecayShape::Sqrt, DecayShape::Cosine] {
            let (steps, frac) = (2000usize, 0.15f32);
            let first = decay_start(steps, frac);
            // the learning rate is the peak up to the first decay step and below it right after; the share is 0 before it and positive one step later
            assert_eq!(lr_at(first - 1, steps, 1.0, 10, 0.05, frac, shape), 1.0);
            assert!(lr_at(first + 1, steps, 1.0, 10, 0.05, frac, shape) < 1.0);
            assert_eq!(cool_share_at(first - 1, steps, frac, 100, 0.6, None), 0.0);
            assert!(cool_share_at(first + 1, steps, frac, 100, 0.6, None) > 0.0);
        }
    }

    #[test]
    fn every_window_of_a_scheduled_step_draws_its_own_pool() {
        // the windows of a step (more, shorter rows under --seq-schedule; more micro-steps under --tps-schedule) have distinct hashes, so the share applies per window
        let share = 0.5f32;
        let (seed, w, rows) = (1u64, 512usize, 16usize);
        let mut cool = 0usize;
        for done_micro in [0usize, 40, 4000] {
            for mi in 0..8usize {
                for s in 0..rows {
                    let h = splitmix(
                        seed ^ splitmix(((done_micro + mi) * rows + s) as u64)
                            ^ splitmix(0x5E9_0000 + w as u64),
                    );
                    cool += is_cool(h, share) as usize;
                }
            }
        }
        let n = 3 * 8 * rows;
        assert!((cool as f64 / n as f64 - 0.5).abs() < 0.15, "{cool} of {n}");
    }

    fn argv(s: &str) -> Vec<String> {
        s.split_whitespace().map(|x| x.to_string()).collect()
    }

    #[test]
    fn a_misspelled_flag_is_reported_and_the_known_ones_are_not() {
        let args =
            argv("forge train --data d --stpes 100 --qk-norm --steps 10 --stpes 7 --lr 1e-3");
        let asked: std::collections::BTreeSet<String> = ["--data", "--qk-norm", "--steps", "--lr"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert_eq!(unused_flags(&args, &asked), vec!["--stpes".to_string()]);
        // a value that is not a flag, a lone "--" and negative numbers are not flags
        let mut asked = asked;
        asked.insert("--seed".to_string());
        let args = argv("forge train --seed -1 -- x");
        assert!(unused_flags(&args, &asked).is_empty());
    }

    #[test]
    fn the_lookups_record_what_was_asked() {
        let args = argv("forge train --data d --qk-norm --warmpu 5");
        assert_eq!(flag(&args, "--data"), Some("d"));
        assert!(has(&args, "--qk-norm"));
        assert_eq!(num(&args, "--warmup", 100usize), 100); // absent: the default (and the misspelled --warmpu stays unused)
        let asked = ASKED.lock().unwrap().clone();
        assert_eq!(unused_flags(&args, &asked), vec!["--warmpu".to_string()]);
    }

    #[test]
    fn a_numeric_flag_that_is_not_a_number_is_an_error_not_the_default() {
        assert_eq!(parse_num::<usize>("--steps", "12000"), Ok(12000));
        assert_eq!(parse_num::<f32>("--lr", "1e-3"), Ok(0.001));
        for bad in ["12k", "", "--seed", "1,5", "0x10"] {
            let e = parse_num::<usize>("--steps", bad).unwrap_err();
            assert!(e.contains("--steps") && e.contains("number"), "{e}");
        }
        let args = argv("forge train --steps 12k");
        let r = std::panic::catch_unwind(|| num(&args, "--steps", 1000usize));
        assert!(r.is_err());
    }
}
