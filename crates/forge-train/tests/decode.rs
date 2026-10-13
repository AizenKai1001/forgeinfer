//! KV-cache decoding (`Decoder`) against the no-cache reference (`Trainer::logits_at`, the trainer's own forward pass over the whole text): every structure option of the golden tiny models, plus a
//! randomly initialised model shaped like Forge-1 (GQA 3:1, qk-norm, ReLU^2, lambdas with non-trivial values, value embeddings on the first and last layers). The decoder reproduces the trainer's
//! arithmetic row for row, so the logits must be the same BITS at every position, for the fused single-token kernels, for the row path, for a prompt fed in pieces, and after prefix reuse.

use forge_core::Json;
use forge_cuda::Device;
use forge_train::{AttnKind, Decoder, TrainConfig, Trainer};
use std::collections::HashMap;
use std::path::PathBuf;

/// The tests of this file take turns on the GPU: a CUDA graph capture in one thread and a synchronous copy (legacy stream) in another one invalidate each other
/// (CUDA_ERROR_STREAM_CAPTURE_IMPLICIT), and the decoder captures its step as a graph.
static GPU: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn dir(variant: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/golden/train_tiny")
        .join(variant)
}

fn read_f32(p: PathBuf) -> Vec<f32> {
    std::fs::read(p)
        .unwrap()
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn golden_config(man: &Json, seq_len: usize) -> TrainConfig {
    let c = &man["config"];
    TrainConfig {
        n_layer: c["n_layer"].as_usize().unwrap(),
        n_embd: c["n_embd"].as_usize().unwrap(),
        n_head: c["n_head"].as_usize().unwrap(),
        n_kv: c["n_kv"].as_usize().unwrap(),
        head_dim: c["hd"].as_usize().unwrap(),
        n_ff: c["n_ff"].as_usize().unwrap(),
        vocab: c["vocab"].as_usize().unwrap(),
        seq_len,
        batch: 1,
        qk_norm: c["qk_norm"].as_bool().unwrap(),
        rope_theta: c["theta"].as_f64().unwrap() as f32,
        eps: 1e-5,
        ce_chunk: 64,
        untied: c["untied"].as_bool().unwrap_or(false),
        relu2: c["relu2"].as_bool().unwrap_or(false),
        softcap: c["softcap"].as_f64().unwrap_or(0.0) as f32,
        lambdas: c["lambdas"].as_bool().unwrap_or(false),
        skips: c["skips"].as_bool().unwrap_or(false),
        ve_layers: c["ve_layers"]
            .as_array()
            .map(|a| a.iter().filter_map(|v| v.as_usize()).collect())
            .unwrap_or_default(),
        ngram: c["ngram"]
            .as_array()
            .map(|a| a.iter().filter_map(|v| v.as_usize()).collect())
            .unwrap_or_default(),
        ngram_bits: c["ngram_bits"].as_usize().unwrap_or(0),
        ngram_dim: c["ngram_dim"].as_usize().unwrap_or(0),
        // options added to the trainer later keep their defaults (off)
        ..TrainConfig::default()
    }
}

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
    fn unit(&mut self) -> f32 {
        self.next() as f32 / (1u64 << 31) as f32
    }
}

/// max |a - b| and whether all values are the same bits
fn compare(a: &[f32], b: &[f32]) -> (f32, bool) {
    assert_eq!(a.len(), b.len());
    let mut worst = 0f32;
    let mut same = true;
    for (x, y) in a.iter().zip(b) {
        worst = worst.max((x - y).abs());
        same &= x.to_bits() == y.to_bits();
    }
    (worst, same)
}

/// Runs every way of filling the cache against `logits_at` on one token sequence of `len` tokens (`prompt` of them fed at once).
fn check(name: &str, tr: &mut Trainer, toks: &[u32], prompt: usize) {
    let len = toks.len();
    let reference: Vec<Vec<f32>> = (1..=len).map(|n| tr.logits_at(toks, n).unwrap()).collect();
    let mut problems: Vec<String> = Vec::new();
    let mut report = |what: &str, n: usize, got: &[f32]| {
        let (d, same) = compare(got, &reference[n - 1]);
        if !same {
            problems.push(format!("{what} at length {n}: max |diff| {d:e}"));
        }
    };
    // 1. prompt at once (row path), then one token at a time (fused kernels + graph where the structure allows)
    let mut fast = Decoder::new(tr, 32).unwrap();
    let l = fast.feed(tr, &toks[..prompt]).unwrap();
    report("prompt", prompt, &l);
    for n in prompt + 1..=len {
        let l = fast.feed(tr, &toks[n - 1..n]).unwrap();
        report(
            if fast.fast_path() {
                "fused step"
            } else {
                "row step"
            },
            n,
            &l,
        );
    }
    let snap_fast = fast.cache_snapshot(tr).unwrap();
    // 1b. the other single-token attention kernels (separate q/k/v post-processing + per-KV-head attention; flash.cu's kernel per query head)
    if fast.fast_path() {
        for kind in [
            AttnKind::Fused,
            AttnKind::Gqa,
            AttnKind::Flash,
            AttnKind::Split,
        ] {
            let mut d = Decoder::new(tr, 32).unwrap();
            d.set_attention(kind);
            d.feed(tr, &toks[..prompt]).unwrap();
            for n in prompt + 1..=len {
                let l = d.feed(tr, &toks[n - 1..n]).unwrap();
                report(
                    match kind {
                        AttnKind::Gqa => "fused step, gqa attention",
                        AttnKind::Flash => "fused step, flash attention",
                        AttnKind::Split => "fused step, split attention",
                        _ => "fused step, fused attention",
                    },
                    n,
                    &l,
                );
            }
        }
    }
    // 2. the row path for single tokens too
    let mut rows = Decoder::new(tr, 32).unwrap();
    rows.use_fast = false;
    let l = rows.feed(tr, &toks[..1]).unwrap();
    report("row path", 1, &l);
    for n in 2..=len {
        let l = rows.feed(tr, &toks[n - 1..n]).unwrap();
        report("row path", n, &l);
    }
    // 3. the whole text at once, in pieces of 48 rows: same logits, same cache
    let mut whole = Decoder::new(tr, 48).unwrap();
    let l = whole.feed(tr, toks).unwrap();
    report("whole text in pieces", len, &l);
    let snap_whole = whole.cache_snapshot(tr).unwrap();
    let cache_problem = if snap_whole != snap_fast {
        let diff = snap_whole
            .iter()
            .zip(&snap_fast)
            .filter(|(a, b)| a != b)
            .count();
        Some(format!("cache filled at once differs from the cache filled token by token in {diff} of {} entries", snap_fast.len()))
    } else {
        None
    };
    // 4. prefix reuse: keep the first 2/3, feed the rest again
    let keep = len * 2 / 3;
    fast.truncate(keep);
    let l = fast.feed(tr, &toks[keep..]).unwrap();
    report("after prefix reuse", len, &l);
    // 5. a different continuation after the shared prefix, then the original one again (the stale entries must not leak)
    fast.truncate(keep);
    let other: Vec<u32> = toks[keep..]
        .iter()
        .map(|t| (t + 1) % tr.cfg.vocab as u32)
        .collect();
    fast.feed(tr, &other).unwrap();
    fast.truncate(keep);
    for n in keep + 1..=len {
        let l = fast.feed(tr, &toks[n - 1..n]).unwrap();
        report("after a divergent branch", n, &l);
    }
    problems.extend(cache_problem);
    eprintln!(
        "{name}: {} positions, fused path {}: {}",
        len,
        fast.fast_path(),
        if problems.is_empty() {
            "bitwise equal everywhere".to_string()
        } else {
            format!("{} differences", problems.len())
        }
    );
    for p in problems.iter().take(10) {
        eprintln!("  {p}");
    }
    assert!(
        problems.is_empty(),
        "{name}: the KV-cache decoder differs from logits_at"
    );
}

#[test]
fn decoder_matches_logits_at_on_every_golden_structure() {
    let _gpu = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if Device::count().unwrap_or(0) == 0 {
        eprintln!("skipped: no CUDA device");
        return;
    }
    for variant in [
        "plain", "qknorm", "relu2", "untied", "softcap", "lambdas", "skips", "valemb", "ngram",
        "forge1",
    ] {
        let d = dir(variant);
        let man = Json::from_file(d.join("manifest.json")).unwrap();
        let mut weights: HashMap<String, Vec<f32>> = HashMap::new();
        for p in man["params"].as_array().unwrap() {
            let name = p["name"].as_str().unwrap().to_string();
            weights.insert(name.clone(), read_f32(d.join(format!("{name}.w.f32"))));
        }
        let cfg = golden_config(&man, 128);
        let vocab = cfg.vocab as u32;
        let mut tr = Trainer::new_inference(cfg, 0).unwrap();
        tr.set_params(&weights).unwrap();
        let mut rng = Lcg(7);
        let toks: Vec<u32> = (0..100).map(|_| rng.next() % vocab).collect();
        check(variant, &mut tr, &toks, 37);
    }
}

#[test]
fn decoder_matches_logits_at_on_a_forge1_shaped_model() {
    let _gpu = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if Device::count().unwrap_or(0) == 0 {
        eprintln!("skipped: no CUDA device");
        return;
    }
    // the production structure in small: 6 layers, d 192, 3 query heads on 1 KV head (head_dim 64), ReLU^2 MLP of 768, tied head, qk-norm, lambdas, value embeddings on layers 0, 1, 4, 5
    let cfg = TrainConfig {
        n_layer: 6,
        n_embd: 192,
        n_head: 3,
        n_kv: 1,
        head_dim: 64,
        n_ff: 768,
        vocab: 2048,
        seq_len: 320,
        batch: 1,
        qk_norm: true,
        rope_theta: 10000.0,
        eps: 1e-5,
        ce_chunk: 64,
        untied: false,
        relu2: true,
        softcap: 0.0,
        lambdas: true,
        skips: false,
        ve_layers: vec![0, 1, 4, 5],
        ngram: vec![],
        ngram_bits: 0,
        ngram_dim: 0,
        ..TrainConfig::default()
    };
    let vocab = cfg.vocab as u32;
    let mut tr = Trainer::new_inference(cfg, 0).unwrap();
    tr.init_random(11).unwrap();
    // scalars away from their initial (1, 0, 0) / (1, 1) so that every mixing term matters
    let mut named = tr.params_named().unwrap();
    let mut rng = Lcg(3);
    for v in named.get_mut("lam").unwrap().iter_mut() {
        *v += 0.4 * (rng.unit() - 0.5);
    }
    for v in named.get_mut("velam").unwrap().iter_mut() {
        *v += 0.4 * (rng.unit() - 0.5);
    }
    tr.set_params(&named).unwrap();
    // a long text: several 64-key tiles of attention, positions past 256
    let toks: Vec<u32> = (0..300).map(|_| rng.next() % vocab).collect();
    check("forge1-shaped", &mut tr, &toks, 130);
}

#[test]
fn decoder_matches_logits_at_with_head_dim_128_and_swiglu() {
    let _gpu = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if Device::count().unwrap_or(0) == 0 {
        eprintln!("skipped: no CUDA device");
        return;
    }
    // the other attention template (head dimension 128: tiles of 32 keys) and the SwiGLU MLP, 4 query heads on 2 KV heads, separate head, skips
    let cfg = TrainConfig {
        n_layer: 4,
        n_embd: 256,
        n_head: 4,
        n_kv: 2,
        head_dim: 128,
        n_ff: 512,
        vocab: 1024,
        seq_len: 192,
        batch: 1,
        qk_norm: true,
        rope_theta: 10000.0,
        eps: 1e-5,
        ce_chunk: 64,
        untied: true,
        relu2: false,
        softcap: 0.0,
        lambdas: true,
        skips: true,
        ve_layers: vec![0, 3],
        ngram: vec![],
        ngram_bits: 0,
        ngram_dim: 0,
        ..TrainConfig::default()
    };
    let vocab = cfg.vocab as u32;
    let mut tr = Trainer::new_inference(cfg, 0).unwrap();
    tr.init_random(5).unwrap();
    let mut named = tr.params_named().unwrap();
    let mut rng = Lcg(9);
    for v in named.get_mut("lam").unwrap().iter_mut() {
        *v += 0.4 * (rng.unit() - 0.5);
    }
    // the separate head starts at zero (all logits equal): give it values
    for v in named.get_mut("head").unwrap().iter_mut() {
        *v = 0.05 * (rng.unit() - 0.5);
    }
    tr.set_params(&named).unwrap();
    let toks: Vec<u32> = (0..150).map(|_| rng.next() % vocab).collect();
    check("hd128-swiglu-skips", &mut tr, &toks, 41);
}

// ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// Sliding-window attention (`window` / `window_layers`): the decoder's local layers attend to the keys i - window < j <= i, the rule of the training kernels (flash_train.cu).
// ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

/// A Forge-1-like model (qk-norm, ReLU^2, lambdas with non-trivial values, value embeddings on the first and last layer) with the given window setting.
fn window_cfg(
    n_layer: usize,
    heads: (usize, usize),
    seq_len: usize,
    window: usize,
    layers: Vec<usize>,
) -> TrainConfig {
    TrainConfig {
        n_layer,
        n_embd: 64 * heads.0,
        n_head: heads.0,
        n_kv: heads.1,
        head_dim: 64,
        n_ff: 256,
        vocab: 512,
        seq_len,
        batch: 1,
        qk_norm: true,
        ce_chunk: 64,
        relu2: true,
        lambdas: true,
        ve_layers: vec![0, n_layer - 1],
        window,
        window_layers: layers,
        ..TrainConfig::default()
    }
}

/// An inference trainer with random weights (same seed: same weights whatever the window), lambdas moved away from their initial values.
fn window_trainer(cfg: TrainConfig) -> Trainer {
    let mut tr = Trainer::new_inference(cfg, 0).unwrap();
    tr.init_random(21).unwrap();
    let mut named = tr.params_named().unwrap();
    let mut rng = Lcg(5);
    for k in ["lam", "velam"] {
        for v in named.get_mut(k).unwrap().iter_mut() {
            *v += 0.4 * (rng.unit() - 0.5);
        }
    }
    tr.set_params(&named).unwrap();
    tr
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut ab, mut aa, mut bb) = (0f64, 0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        ab += *x as f64 * *y as f64;
        aa += (*x as f64).powi(2);
        bb += (*y as f64).powi(2);
    }
    ab / (aa.sqrt() * bb.sqrt()).max(1e-30)
}

fn argmax(a: &[f32]) -> u32 {
    let mut best = 0;
    for (i, &x) in a.iter().enumerate() {
        if x > a[best] {
            best = i;
        }
    }
    best as u32
}

fn fnv(a: &[f32]) -> u64 {
    a.iter().fold(0xcbf29ce484222325u64, |h, x| {
        (h ^ x.to_bits() as u64).wrapping_mul(0x100000001b3)
    })
}

/// The decoder (prompt of `prompt` tokens fed at once, then one token at a time) against `logits_at` at every position of `toks` from `prompt` on: cosine, top-1, and the bits; returns the decoder's logits.
fn window_vs_reference(
    name: &str,
    tr: &mut Trainer,
    toks: &[u32],
    prompt: usize,
) -> Vec<Vec<f32>> {
    let mut dec = Decoder::new(tr, 32).unwrap();
    let mut got = vec![dec.feed(tr, &toks[..prompt]).unwrap()];
    for n in prompt + 1..=toks.len() {
        got.push(dec.feed(tr, &toks[n - 1..n]).unwrap());
    }
    let (mut worst_cos, mut bitwise) = (1f64, true);
    for (i, g) in got.iter().enumerate() {
        let n = prompt + i;
        let r = tr.logits_at(toks, n).unwrap();
        let c = cosine(g, &r);
        worst_cos = worst_cos.min(c);
        assert!(c > 0.9999, "{name}: cosine {c} at length {n}");
        assert_eq!(argmax(g), argmax(&r), "{name}: top-1 differs at length {n}");
        bitwise &= compare(g, &r).1;
    }
    eprintln!(
        "{name}: {} positions, worst cosine {worst_cos:.7}, bitwise equal: {bitwise}, fused path {}",
        got.len(),
        dec.fast_path()
    );
    got
}

/// Greedy continuation of `steps` tokens: the decoder (KV cache) against `logits_at` step by step.
fn greedy_matches(name: &str, tr: &mut Trainer, prompt: &[u32], steps: usize) {
    let mut dec = Decoder::new(tr, 32).unwrap();
    let mut logits = dec.feed(tr, prompt).unwrap();
    let mut seq = prompt.to_vec();
    for s in 0..steps {
        let want = tr.logits_at(&seq, seq.len()).unwrap();
        assert_eq!(
            argmax(&logits),
            argmax(&want),
            "{name}: greedy token {s} differs (decoder vs forward pass)"
        );
        let next = argmax(&want);
        seq.push(next);
        logits = dec.feed(tr, &[next]).unwrap();
    }
    eprintln!(
        "{name}: greedy continuation of {steps} tokens identical to the forward pass ({} -> {} tokens)",
        prompt.len(),
        seq.len()
    );
}

#[test]
fn windowed_decoder_matches_the_training_forward_and_the_window_matters() {
    let _gpu = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if Device::count().unwrap_or(0) == 0 {
        eprintln!("skipped: no CUDA device");
        return;
    }
    // 4 layers, window 16, `auto` = layers 0, 1, 2 local and layer 3 dense; and a custom list; heads 2:1 and 3:1
    let auto = TrainConfig::auto_window_layers(4);
    assert_eq!(auto, vec![0, 1, 2]);
    let mut rng = Lcg(77);
    let toks: Vec<u32> = (0..48).map(|_| rng.next() % 512).collect();
    for (name, heads, window, layers) in [
        ("window16-auto-2:1", (2, 1), 16, auto.clone()),
        ("window16-auto-3:1", (3, 1), 16, auto.clone()),
        ("window16-custom-0,2", (2, 1), 16, vec![0, 2]),
    ] {
        let mut tr = window_trainer(window_cfg(4, heads, 128, window, layers.clone()));
        // every way of filling the cache, bitwise, at every position
        check(name, &mut tr, &toks, 20);
        let got = window_vs_reference(name, &mut tr, &toks, 8);
        greedy_matches(name, &mut tr, &toks[..40], 32);
        // the window matters: the same weights without it give other logits past the window, and the same bits before it
        let mut dense = window_trainer(window_cfg(4, heads, 128, 0, Vec::new()));
        let dense_dec = window_vs_reference("dense", &mut dense, &toks, 8);
        let (mut differ, mut before_same) = (0, true);
        for (i, (w, d)) in got.iter().zip(&dense_dec).enumerate() {
            let n = 8 + i;
            if n <= window {
                before_same &= compare(w, d).1;
            } else if compare(w, d).0 > 1e-3 {
                differ += 1;
            }
        }
        eprintln!(
            "{name}: windowed logits differ from the dense ones at {differ} of the {} positions past the window; inside it they are the same bits",
            toks.len() - window
        );
        assert!(
            before_same,
            "{name}: inside the window the logits must be the dense ones"
        );
        assert!(
            differ >= 20,
            "{name}: the window changes too few logits ({differ})"
        );
        // a window as long as the context is the dense case
        let mut wide = window_trainer(window_cfg(4, heads, 128, 64, layers));
        let wide_dec = window_vs_reference("window64", &mut wide, &toks, 8);
        for (w, d) in wide_dec.iter().zip(&dense_dec) {
            assert!(
                compare(w, d).1,
                "{name}: a window longer than the text must equal dense attention"
            );
        }
    }
}

#[test]
fn windowed_decoder_matches_on_long_contexts_with_the_split_attention() {
    let _gpu = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if Device::count().unwrap_or(0) == 0 {
        eprintln!("skipped: no CUDA device");
        return;
    }
    // 6 layers (auto: 0, 1, 2, 4 local), 300 tokens: from 256 cached positions on the single-token step uses the split attention; windows of 64 (tile aligned) and 100 (not)
    let mut rng = Lcg(31);
    let toks: Vec<u32> = (0..300).map(|_| rng.next() % 512).collect();
    for window in [64usize, 100] {
        let layers = TrainConfig::auto_window_layers(6);
        assert_eq!(layers, vec![0, 1, 2, 4]);
        let mut tr = window_trainer(window_cfg(6, (3, 1), 320, window, layers));
        check(&format!("window{window}-long"), &mut tr, &toks, 130);
    }
}

#[test]
fn a_model_without_a_window_keeps_its_bits() {
    let _gpu = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if Device::count().unwrap_or(0) == 0 {
        eprintln!("skipped: no CUDA device");
        return;
    }
    // window 0 (and a window list without a window): the decoder equals logits_at bit for bit, and the fingerprint of its logits is printed so that two builds can be compared
    let mut rng = Lcg(5);
    let toks: Vec<u32> = (0..300).map(|_| rng.next() % 512).collect();
    for (name, window, layers) in [("dense", 0, Vec::new()), ("dense-list-only", 0, vec![0, 1])] {
        let mut tr = window_trainer(window_cfg(6, (3, 1), 320, window, layers));
        check(name, &mut tr, &toks, 130);
        let mut dec = Decoder::new(&tr, 32).unwrap();
        let mut h = fnv(&dec.feed(&tr, &toks[..130]).unwrap());
        for n in 131..=toks.len() {
            h ^= fnv(&dec.feed(&tr, &toks[n - 1..n]).unwrap()).rotate_left((n % 63) as u32);
        }
        eprintln!("DENSE_LOGITS_FINGERPRINT {name} {h:016x}");
    }
}
