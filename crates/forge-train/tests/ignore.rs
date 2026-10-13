//! Ignored targets (chat fine-tuning): a token with the `IGNORE_TARGET` bit is an input like any other, but the position that predicts it has no loss and no gradient.
//! Checked on a tiny model with every structure option (plain cross entropy and the soft-capped one):
//!  - the mean loss of a batch with ignored targets equals the mean of the negative log-likelihoods of the remaining positions;
//!  - the gradient is linear in the set of positions: gradient(all) = gradient(set A) + gradient(set B) when A and B partition the positions;
//!  - a batch whose targets are all ignored has no gradient at all.

use forge_core::Json;
use forge_cuda::Device;
use forge_train::{TrainConfig, Trainer, IGNORE_TARGET};
use std::collections::HashMap;
use std::path::PathBuf;

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

fn config(man: &Json) -> TrainConfig {
    let c = &man["config"];
    let t = c["T"].as_usize().unwrap();
    let b = c["B"].as_usize().unwrap();
    TrainConfig {
        n_layer: c["n_layer"].as_usize().unwrap(),
        n_embd: c["n_embd"].as_usize().unwrap(),
        n_head: c["n_head"].as_usize().unwrap(),
        n_kv: c["n_kv"].as_usize().unwrap(),
        head_dim: c["hd"].as_usize().unwrap(),
        n_ff: c["n_ff"].as_usize().unwrap(),
        vocab: c["vocab"].as_usize().unwrap(),
        seq_len: t,
        batch: b,
        qk_norm: c["qk_norm"].as_bool().unwrap(),
        rope_theta: c["theta"].as_f64().unwrap() as f32,
        eps: 1e-5,
        ce_chunk: 128,
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
        attn_gate: c["attn_gate"].as_usize().unwrap_or(0),
        peri_norm: c["peri_norm"].as_bool().unwrap_or(false),
        emb_norm: c["emb_norm"].as_bool().unwrap_or(false),
        window: 0,
        window_layers: Vec::new(),
    }
}

/// `tokens` with the ignore bit set on the targets (positions j + 1 of every row) for which `ignore(row, j)` holds.
fn with_ignored(tokens: &[u32], t: usize, ignore: impl Fn(usize, usize) -> bool) -> Vec<u32> {
    let mut out = tokens.to_vec();
    for r in 0..tokens.len() / (t + 1) {
        for j in 0..t {
            if ignore(r, j) {
                out[r * (t + 1) + j + 1] |= IGNORE_TARGET;
            }
        }
    }
    out
}

fn rel_diff(a: &[f32], b: &[f32]) -> f64 {
    let (mut dd, mut bb) = (0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        dd += ((*x - *y) as f64).powi(2);
        bb += (*y as f64).powi(2);
    }
    (dd / bb.max(1e-30)).sqrt()
}

fn check(variant: &str) {
    if Device::count().unwrap_or(0) == 0 {
        eprintln!("skipped: no CUDA device");
        return;
    }
    let d = dir(variant);
    let man = Json::from_file(d.join("manifest.json")).unwrap();
    let cfg = config(&man);
    let (t, n) = (cfg.seq_len, cfg.tokens());
    let mut weights: HashMap<String, Vec<f32>> = HashMap::new();
    for p in man["params"].as_array().unwrap() {
        let name = p["name"].as_str().unwrap().to_string();
        weights.insert(name.clone(), read_f32(d.join(format!("{name}.w.f32"))));
    }
    let tokens: Vec<u32> = std::fs::read(d.join("tokens.u32"))
        .unwrap()
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let mut tr = Trainer::new(cfg, 0).unwrap();
    tr.set_params(&weights).unwrap();

    // the positions A: about a third of them, scattered over the rows and not a prefix (the mean must sum over all rows and divide by the number of valid ones); B is the rest
    let in_a = |r: usize, j: usize| (r * 5 + j * 7 + j / 3) % 3 == 0;
    let only_a = with_ignored(&tokens, t, |r, j| !in_a(r, j));
    let only_b = with_ignored(&tokens, t, |r, j| in_a(r, j));
    let nothing = with_ignored(&tokens, t, |_, _| true);

    // 1. the mean loss over the remaining positions
    let nll = tr.token_nll(&tokens).unwrap();
    assert_eq!(nll.len(), n);
    let (mut sum_a, mut cnt_a) = (0f64, 0usize);
    for r in 0..tokens.len() / (t + 1) {
        for j in 0..t {
            if in_a(r, j) {
                sum_a += nll[r * t + j] as f64;
                cnt_a += 1;
            }
        }
    }
    let want = (sum_a / cnt_a as f64) as f32;
    let got = tr.eval_loss(&only_a).unwrap();
    eprintln!("{variant}: mean loss of set A {got:.5}, from the full forward {want:.5} ({cnt_a} of {n} positions)");
    assert!(cnt_a > 20 && cnt_a < n - 20);
    assert!(
        (got - want).abs() < 1e-4,
        "mean loss with ignored targets {got} vs {want}"
    );

    // 2. the gradient is linear in the positions; all gradients carry the same constant scale n
    let scale = 1024.0f32;
    tr.zero_grads().unwrap();
    tr.forward_backward(&tokens, scale, n).unwrap();
    let g_all = tr.download_grads().unwrap();
    tr.zero_grads().unwrap();
    tr.forward_backward(&only_a, scale, n).unwrap();
    tr.forward_backward(&only_b, scale, n).unwrap();
    let g_sum = tr.download_grads().unwrap();
    let rel = rel_diff(&g_sum, &g_all);
    eprintln!(
        "{variant}: gradient(A) + gradient(B) against gradient(all): relative difference {rel:.2e}"
    );
    assert!(
        rel < 2e-3,
        "the gradient is not linear in the positions: {rel}"
    );

    // 3. no target at all: no gradient, loss 0 (the mean over an empty set is defined as 0 here)
    tr.zero_grads().unwrap();
    let l = tr.forward_backward(&nothing, scale, n).unwrap();
    let g_none = tr.download_grads().unwrap();
    let biggest = g_none.iter().fold(0f32, |m, v| m.max(v.abs()));
    eprintln!("{variant}: all targets ignored: loss {l}, largest gradient entry {biggest}");
    assert_eq!(l, 0.0);
    assert_eq!(biggest, 0.0);
}

#[test]
fn ignored_targets_plain_cross_entropy() {
    check("relu2");
}

#[test]
fn ignored_targets_with_every_option_and_the_soft_cap() {
    check("forge1");
}
