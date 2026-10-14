//! The GQA-shared split attention (one block per key-value head, decode.cu `attn_gqa_impl`) against the one-block-per-query-head kernels on real models: a dense one (FORGE_QWEN2_GGUF, any dense GGUF with four query
//! heads per key-value head and head width 128, the Qwen3-4B of the homelab; the Qwen2.5-0.5B of the other tests by default, whose 7 heads per group run the padded kernel of 8) and a hybrid one
//! (FORGE_VARIANTS_MODEL, the Qwen3.5-0.8B), with the fp16 cache and the int8 and int4 caches, a prompt of 1,500 tokens (more than one split of the attention) and 24 greedy tokens after it. The two kernels
//! make bit-identical partial results (tests/attn_gqa.rs), so the greedy tokens AND the logits after every step must be EXACTLY equal. FORGE_ATTN_GQA_FROM=1 makes the sessions of 4,096 positions use the
//! GQA kernels. A test is skipped when its model or a CUDA device is missing.

use forge_cuda::Device;
use forge_gpu::{Fmt, GpuModel, GpuSession};
use forge_model::Model;
use std::path::PathBuf;

mod common;

fn models_dir() -> PathBuf {
    match std::env::var("FORGE_MODELS") {
        Ok(p) => PathBuf::from(p),
        Err(_) => PathBuf::from(std::env::var("HOME").unwrap_or_default()).join("forgeinfer/oracle/models"),
    }
}

fn dense_path() -> PathBuf {
    match std::env::var("FORGE_QWEN2_GGUF") {
        Ok(p) => PathBuf::from(p),
        Err(_) => models_dir().join("Qwen__Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q4_k_m.gguf"),
    }
}

fn hybrid_path() -> PathBuf {
    match std::env::var("FORGE_VARIANTS_MODEL") {
        Ok(p) => PathBuf::from(p),
        Err(_) => models_dir().join("unsloth__Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf"),
    }
}

/// Natural-text token ids (the chat prompts of the Qwen2 / Qwen3 tests, repeated)
fn prompt(n: usize) -> Vec<u32> {
    let base: Vec<u32> = vec![
        151644, 8948, 198, 2610, 525, 1207, 16948, 11, 3465, 553, 54364, 14817, 13, 1446, 525, 264, 10950, 17847, 13, 151645, 198, 151644, 872, 198, 7985, 264, 6386, 38242, 911, 279, 9396, 13, 151645, 198,
        151644, 77091, 198, 785, 9396, 374, 6884, 323, 6783, 11, 151645, 198, 151644, 872, 198, 9330, 432, 438, 264, 2805, 31365, 13, 151645, 198,
    ];
    (0..n).map(|i| base[i % base.len()]).collect()
}

/// The greedy tokens of 24 decode steps after the prompt and the logits after every one of them.
fn run(path: &PathBuf, kv: &str, gqa: bool) -> Vec<(u32, Vec<f32>)> {
    if kv == "fp16" {
        std::env::remove_var("FORGE_KV");
    } else {
        std::env::set_var("FORGE_KV", kv);
    }
    std::env::set_var("FORGE_ATTN_GQA", if gqa { "1" } else { "0" });
    std::env::set_var("FORGE_ATTN_GQA_FROM", "1");
    let model = Model::load_native(path).unwrap();
    let gpu = GpuModel::upload(&model, Fmt::Native, 0).unwrap();
    drop(model);
    let mut sess = GpuSession::new(&gpu, 4096).unwrap();
    sess.prefill_batched(&prompt(1500)).unwrap();
    let mut out = Vec::new();
    for _ in 0..24 {
        let t = sess.decode_greedy(1).unwrap();
        out.push((t[0], sess.logits().unwrap()));
    }
    std::env::remove_var("FORGE_KV");
    std::env::remove_var("FORGE_ATTN_GQA");
    std::env::remove_var("FORGE_ATTN_GQA_FROM");
    out
}

fn check(path: PathBuf, label: &str) {
    if !path.exists() || Device::count().unwrap_or(0) == 0 {
        eprintln!("skipped ({label}): {} or a CUDA device is missing", path.display());
        return;
    }
    for kv in ["fp16", "int8", "int4"] {
        let a = run(&path, kv, false);
        let b = run(&path, kv, true);
        assert_eq!(a.len(), b.len());
        for (k, (x, y)) in a.iter().zip(&b).enumerate() {
            assert_eq!(x.0, y.0, "{label}, {kv}: the greedy token of step {k} differs");
            assert!(x.1.iter().zip(&y.1).all(|(p, q)| p.to_bits() == q.to_bits()), "{label}, {kv}: the logits of step {k} are not bit for bit equal");
        }
        eprintln!("{label}, {kv} cache: 24 decode steps after 1,500 prompt tokens, tokens and logits bit for bit equal");
    }
}

#[test]
fn a_dense_model_decodes_bit_for_bit_the_same_with_the_gqa_attention() {
    let _g = common::gpu_lock();
    check(dense_path(), "dense");
}

#[test]
fn a_hybrid_model_decodes_bit_for_bit_the_same_with_the_gqa_attention() {
    let _g = common::gpu_lock();
    check(hybrid_path(), "hybrid");
}
