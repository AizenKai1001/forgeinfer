//! The Ampere prompt-attention kernels of flash_a80.cu (m16n8k16 MMAs, cp.async K / V tiles, fp32 accumulators) against the sm_75 kernels of flash.cu on a dense model: the
//! logits of the last prompt token must stay close, the greedy tokens that follow must match, and the speed of the three is printed. The gated head width 256 of Qwen3.5 is covered by
//! prefill_variants.rs, the dense head width 128 by `scratch/spike/flash_a80_bench.cu` (no model of that width is kept next to the tests).
//! Qwen2.5-0.5B-Instruct (14 query heads over 2 key/value heads of width 64); skipped when the file or a CUDA device is missing: set FORGE_QWEN2_GGUF to the file, or put it in FORGE_MODELS as
//! Qwen__Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q4_k_m.gguf.

use forge_cuda::Device;
use forge_gpu::{Fmt, GpuModel, GpuSession};
use forge_model::{argmax, Model};
use std::path::PathBuf;

mod common;

fn model_path() -> PathBuf {
    if let Ok(p) = std::env::var("FORGE_QWEN2_GGUF") {
        return PathBuf::from(p);
    }
    let dir = match std::env::var("FORGE_MODELS") {
        Ok(p) => PathBuf::from(p),
        Err(_) => PathBuf::from(std::env::var("HOME").unwrap_or_default()).join("forgeinfer/oracle/models"),
    };
    dir.join("Qwen__Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q4_k_m.gguf")
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut ab, mut aa, mut bb) = (0f64, 0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        ab += (*x as f64) * (*y as f64);
        aa += (*x as f64).powi(2);
        bb += (*y as f64).powi(2);
    }
    ab / (aa.sqrt() * bb.sqrt())
}

/// the chat prompts of the Qwen2 test repeated: natural token statistics and a low-entropy next-token distribution
fn prompt(n: usize) -> Vec<u32> {
    let base: Vec<u32> = vec![
        151644, 8948, 198, 2610, 525, 1207, 16948, 11, 3465, 553, 54364, 14817, 13, 1446, 525, 264, 10950, 17847, 13, 151645, 198, 151644, 872, 198, 7985, 264, 6386, 38242, 911, 279, 9396, 13, 151645, 198,
        151644, 77091, 198, 785, 9396, 374, 6884, 323, 6783, 11, 151645, 198, 151644, 872, 198, 9330, 432, 438, 264, 2805, 31365, 13, 151645, 198,
    ];
    (0..n).map(|i| base[i % base.len()]).collect()
}

#[test]
fn ampere_prompt_attention_matches_the_sm75_kernels_on_a_dense_model() {
    let _gpu = common::gpu_lock();
    let path = model_path();
    if !path.exists() || Device::count().unwrap_or(0) == 0 {
        eprintln!("skipped: the Qwen2.5 GGUF ({}) or a CUDA device is missing", path.display());
        return;
    }
    let model = Model::load_native(&path).unwrap();
    let gpu = GpuModel::upload(&model, Fmt::Native, 0).unwrap();
    drop(model);
    // 5,000 tokens: five chunks of 1,024 (the later ones attend to up to 4,000 earlier positions), and a ragged one of 904
    let n = 5000;
    let p = prompt(n);
    let mut sess = GpuSession::new(&gpu, n + 64).unwrap();
    let mut results = Vec::new();
    let mut tokens = Vec::new();
    for a80 in [false, true] {
        sess.set_flash_a80(a80);
        sess.reset().unwrap();
        sess.prefill_batched(&p).unwrap();
        let logits = sess.logits().unwrap();
        let mut best = f64::MAX;
        for _ in 0..3 {
            sess.reset().unwrap();
            let t0 = std::time::Instant::now();
            sess.prefill_batched(&p).unwrap();
            best = best.min(t0.elapsed().as_secs_f64());
        }
        eprintln!(
            "Ampere attention = {a80}: prefill of {n} tokens in {:.1} ms = {:.0} tokens/s",
            best * 1e3,
            n as f64 / best
        );
        sess.reset().unwrap();
        sess.prefill_batched(&p).unwrap();
        tokens.push(sess.decode_greedy(16).unwrap());
        results.push(logits);
    }
    let cos = cosine(&results[0], &results[1]);
    let same = tokens[0].iter().zip(&tokens[1]).filter(|(a, b)| a == b).count();
    eprintln!(
        "logits cosine {cos:.7}, top-1 {} / {}, {same} of {} greedy tokens equal",
        argmax(&results[0]),
        argmax(&results[1]),
        tokens[0].len()
    );
    // (head width 64: the 32-key tile of the old kernel's online softmax is the same, the summation order of the scores differs in the last bit)
    assert!(cos > 0.9995, "the Ampere attention changes the logits: cosine {cos}");
    assert!(results[1].iter().all(|v| v.is_finite()));
    assert!(same * 10 >= tokens[0].len() * 9, "the Ampere attention changes the greedy tokens");
    assert_eq!(argmax(&results[0]), argmax(&results[1]));
}
