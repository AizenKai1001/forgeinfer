//! Prefill of a quantized model with the fused dequantizing GEMMs (K-quant weights read in place) against dequantize-to-fp16-then-GEMM: same weights, same activations, so the logits must agree to
//! the rounding of fp32 sums (the two paths compute the same fp16 weights, only the accumulation order differs by tile), and the speed difference is printed. Qwen3.5-0.8B Q4_K_M (Q4_K, Q5_K and
//! Q6_K matrices). Skipped when the model or a CUDA device is missing.

use forge_core::Json;
use forge_cuda::Device;
use forge_gpu::{Fmt, GpuModel, GpuSession};
use forge_model::{argmax, Model};
use std::path::PathBuf;

fn models_dir() -> PathBuf {
    match std::env::var("FORGE_MODELS") {
        Ok(p) => PathBuf::from(p),
        Err(_) => PathBuf::from(std::env::var("HOME").unwrap_or_default())
            .join("forgeinfer/oracle/models"),
    }
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

#[test]
fn fused_dequantizing_gemms_agree_with_dequantize_then_gemm() {
    let path = std::env::var("FORGE_VARIANTS_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            models_dir().join("unsloth__Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf")
        });
    if !path.exists() || Device::count().unwrap_or(0) == 0 {
        eprintln!("skipped: model or CUDA device missing");
        return;
    }
    let gold = Json::from_file(
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/golden/fwd_qwen35.json"),
    )
    .unwrap();
    let mut ids: Vec<u32> = Vec::new();
    for p in gold["prompts"].as_array().unwrap() {
        ids.extend(
            p["ids"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as u32),
        );
    }
    ids.truncate(1000);
    let model = Model::load_native(&path).unwrap();
    let gpu = GpuModel::upload(&model, Fmt::Native, 0).unwrap();
    drop(model);
    let mut sess = GpuSession::new(&gpu, 1100).unwrap();
    // (the int8 GEMMs take these matrices first when they are on: this test compares the two fp16 paths)
    sess.set_mmq(false);
    let mut results = Vec::new();
    for fused in [false, true] {
        sess.set_fused(fused);
        sess.reset().unwrap();
        sess.prefill_batched(&ids).unwrap();
        let logits = sess.logits().unwrap();
        let mut best = f64::MAX;
        for _ in 0..3 {
            sess.reset().unwrap();
            let t0 = std::time::Instant::now();
            sess.prefill_batched(&ids).unwrap();
            best = best.min(t0.elapsed().as_secs_f64());
        }
        eprintln!(
            "fused = {fused}: prefill of {} tokens in {:.1} ms = {:.0} tokens/s",
            ids.len(),
            best * 1e3,
            ids.len() as f64 / best
        );
        results.push(logits);
    }
    let cos = cosine(&results[0], &results[1]);
    let maxdiff = results[0]
        .iter()
        .zip(&results[1])
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    eprintln!(
        "logits: cosine {cos:.7}, max |difference| {maxdiff:.4}, top-1 {} / {}",
        argmax(&results[0]),
        argmax(&results[1])
    );
    // (both paths decode with the same fp16x2 arithmetic now: they differ in the accumulation order only)
    assert!(
        cos > 0.99999,
        "the fused GEMMs change the logits: cosine {cos}"
    );
    assert_eq!(argmax(&results[0]), argmax(&results[1]));
    assert!(results[1].iter().all(|v| v.is_finite()));
}

/// The integer tensor-core GEMMs (int8 activations with one scale per 32 values, Q4_K / Q5_K weights read in place) against the fp16 GEMMs on the same weights: the logits of the last prompt token must stay
/// close (the activation quantization is the only difference; `forge ppl` measures 4e-4 KL on the 0.8B and the 9B). The speed of both is printed.
#[test]
fn int8_prompt_gemms_agree_with_the_fp16_path() {
    let path = std::env::var("FORGE_VARIANTS_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            models_dir().join("unsloth__Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf")
        });
    if !path.exists() || Device::count().unwrap_or(0) == 0 {
        eprintln!("skipped: model or CUDA device missing");
        return;
    }
    let gold = Json::from_file(
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/golden/fwd_qwen35.json"),
    )
    .unwrap();
    let mut ids: Vec<u32> = Vec::new();
    for p in gold["prompts"].as_array().unwrap() {
        ids.extend(
            p["ids"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as u32),
        );
    }
    ids.truncate(1000);
    let model = Model::load_native(&path).unwrap();
    let gpu = GpuModel::upload(&model, Fmt::Native, 0).unwrap();
    drop(model);
    let mut sess = GpuSession::new(&gpu, 1100).unwrap();
    let mut results = Vec::new();
    for mmq in [false, true] {
        sess.set_mmq(mmq);
        sess.reset().unwrap();
        sess.prefill_batched(&ids).unwrap();
        let logits = sess.logits().unwrap();
        let mut best = f64::MAX;
        for _ in 0..3 {
            sess.reset().unwrap();
            let t0 = std::time::Instant::now();
            sess.prefill_batched(&ids).unwrap();
            best = best.min(t0.elapsed().as_secs_f64());
        }
        eprintln!(
            "int8 GEMMs = {mmq}: prefill of {} tokens in {:.1} ms = {:.0} tokens/s",
            ids.len(),
            best * 1e3,
            ids.len() as f64 / best
        );
        results.push(logits);
    }
    let cos = cosine(&results[0], &results[1]);
    eprintln!(
        "logits: cosine {cos:.7}, top-1 {} / {}",
        argmax(&results[0]),
        argmax(&results[1])
    );
    assert!(cos > 0.998, "the int8 GEMMs change the logits: cosine {cos}");
    assert!(results[1].iter().all(|v| v.is_finite()));
}
