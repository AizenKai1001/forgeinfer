//! Generation through the trainer: `logits_at` (one position's logits, the forward window shortened to the context) against the training forward pass, for a model with every Forge-1 structure option
//! (the `forge1` golden variant: ReLU squared, separate head, soft cap, per-layer scalars, skips, value embeddings, n-gram memory).

use forge_core::Json;
use forge_cuda::Device;
use forge_train::{TrainConfig, Trainer};
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

fn log_softmax_at(logits: &[f32], target: usize) -> f32 {
    let mx = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let lse = mx + logits.iter().map(|v| (v - mx).exp()).sum::<f32>().ln();
    logits[target] - lse
}

fn config(man: &Json, seq_len: usize, batch: usize) -> TrainConfig {
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
        batch,
        qk_norm: c["qk_norm"].as_bool().unwrap(),
        rope_theta: c["theta"].as_f64().unwrap() as f32,
        eps: 1e-5,
        ce_chunk: seq_len * batch,
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

#[test]
fn generation_logits_match_the_training_forward() {
    // forge1: every option (the skip connections keep one activation set per layer); lambdas and valemb: no skips, so the inference trainer shares two activation sets between the four layers
    for variant in [
        "forge1",
        "lambdas",
        "valemb",
        "relu2",
        "agate_full",
        "peri_emb",
    ] {
        check(variant);
    }
}

fn check(variant: &str) {
    if Device::count().unwrap_or(0) == 0 {
        eprintln!("skipped: no CUDA device");
        return;
    }
    eprintln!("variant {variant}");
    let d = dir(variant);
    let man = Json::from_file(d.join("manifest.json")).unwrap();
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
    let t = man["config"]["T"].as_usize().unwrap();
    let row0: Vec<u32> = tokens[..t + 1].to_vec();
    // the training forward over the same row (batch 1, window 64): per-position negative log-likelihoods
    let mut train = Trainer::new(config(&man, t, 1), 0).unwrap();
    train.set_params(&weights).unwrap();
    let nll = train.token_nll(&row0).unwrap();
    // generation with the same window (the logits path) and with a longer allocated window (the forward is shortened to 64 positions)
    let mut gen_same = Trainer::new_inference(config(&man, t, 1), 0).unwrap();
    gen_same.set_params(&weights).unwrap();
    let mut gen_long = Trainer::new_inference(config(&man, 2 * t, 1), 0).unwrap();
    gen_long.set_params(&weights).unwrap();
    let mut worst = 0f32;
    let mut worst_long = 0f32;
    for len in [1usize, 2, 7, 33, 63, 64] {
        let want = nll[len - 1];
        let target = row0[len] as usize;
        let l1 = gen_same.logits_at(&row0, len).unwrap();
        let l2 = gen_long.logits_at(&row0, len).unwrap();
        let (n1, n2) = (-log_softmax_at(&l1, target), -log_softmax_at(&l2, target));
        eprintln!(
            "len {len:2}: training nll {want:.5}, logits_at {n1:.5}, shortened window {n2:.5}"
        );
        worst = worst.max((n1 - want).abs());
        worst_long = worst_long.max((n2 - want).abs());
    }
    assert!(
        worst < 2e-3,
        "logits_at differs from the training forward by {worst}"
    );
    assert!(
        worst_long < 2e-3,
        "the shortened window differs from the training forward by {worst_long}"
    );
}
