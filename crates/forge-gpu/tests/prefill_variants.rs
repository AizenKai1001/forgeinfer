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

/// The Ampere attention kernels of flash_a80.cu (warp pairs sharing S, 32-key tiles in swizzled shared memory, m16n8k16 MMAs, cp.async) against the sm_75 kernels of flash256.cu on a prompt of 2,500 tokens (three
/// chunks, so the later chunks attend to earlier ones): with fp32 accumulators in both the results are bit-identical, so the logits of the last token must be equal; the speed of both is printed. Qwen3.5-0.8B
/// (head width 256, gated).
#[test]
fn ampere_attention_kernels_agree_with_the_sm75_ones() {
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
    let mut base: Vec<u32> = Vec::new();
    for p in gold["prompts"].as_array().unwrap() {
        base.extend(
            p["ids"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as u32),
        );
    }
    let mut ids: Vec<u32> = Vec::new();
    while ids.len() < 2500 {
        ids.extend_from_slice(&base);
    }
    ids.truncate(2500);
    let model = Model::load_native(&path).unwrap();
    let gpu = GpuModel::upload(&model, Fmt::Native, 0).unwrap();
    drop(model);
    let mut sess = GpuSession::new(&gpu, 2600).unwrap();
    // (on a device below compute capability 8 the three configurations are the same kernels: the comparison is trivially exact)
    let mut results = Vec::new();
    for a80 in [false, true] {
        sess.set_flash_a80(a80);
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
            "Ampere attention = {a80}: prefill of {} tokens in {:.1} ms = {:.0} tokens/s",
            ids.len(),
            best * 1e3,
            ids.len() as f64 / best
        );
        results.push(logits);
    }
    eprintln!("logits cosine {:.7}, top-1 {} / {}", cosine(&results[0], &results[1]), argmax(&results[0]), argmax(&results[1]));
    assert!(results[0] == results[1], "the Ampere attention changes the logits (cosine {})", cosine(&results[0], &results[1]));
    assert!(results[1].iter().all(|v| v.is_finite()));
}

/// The Gated DeltaNet prompt scan with several state columns per thread (prefill_h.cu `gdn_scan_nc_*`) sums every column in the order of the one-column-per-thread kernel, so a whole prompt pass must give
/// the very same logits, bit for bit; the speed of both is printed. Qwen3.5-0.8B, 1,000 tokens.
#[test]
fn gdn_scan_with_several_columns_per_thread_is_bit_identical() {
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
    // the prompts repeated to 1,500 tokens: two chunks of the prefill (the convolution window and the state carry over) and a ragged tail of 476 (not a multiple of the 8-token tile)
    let base = ids.clone();
    while ids.len() < 1500 {
        ids.extend_from_slice(&base);
    }
    ids.truncate(1500);
    let model = Model::load_native(&path).unwrap();
    let gpu = GpuModel::upload(&model, Fmt::Native, 0).unwrap();
    drop(model);
    let mut sess = GpuSession::new(&gpu, 1600).unwrap();
    let mut results = Vec::new();
    for nc in [false, true] {
        sess.set_scan_nc(nc);
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
            "scan with several columns per thread = {nc}: prefill of {} tokens in {:.1} ms = {:.0} tokens/s",
            ids.len(),
            best * 1e3,
            ids.len() as f64 / best
        );
        results.push(logits);
    }
    assert!(
        results[0] == results[1],
        "the scan with several columns per thread changes the logits (cosine {})",
        cosine(&results[0], &results[1])
    );
    assert!(results[1].iter().all(|v| v.is_finite()));
}

/// A prompt pass with and without the fused SwiGLU + int8 quantizer of the FFN's down projection (`swiglu_quant_rows_kernel`, which rounds every activation through fp16 exactly as `swiglu_kernel` stores it):
/// the integer GEMM reads the same bytes either way, so the logits of the last token must be equal bit for bit. The speed of both is printed.
fn swiglu_q_logits(path: &PathBuf, ids: &[u32], label: &str) -> Vec<Vec<f32>> {
    let model = Model::load_native(path).unwrap();
    let gpu = GpuModel::upload(&model, Fmt::Native, 0).unwrap();
    drop(model);
    let mut sess = GpuSession::new(&gpu, ids.len() + 100).unwrap();
    let mut results = Vec::new();
    for fused in [false, true] {
        sess.set_swiglu_q(fused);
        sess.reset().unwrap();
        sess.prefill_batched(ids).unwrap();
        let logits = sess.logits().unwrap();
        let mut best = f64::MAX;
        for _ in 0..3 {
            sess.reset().unwrap();
            let t0 = std::time::Instant::now();
            sess.prefill_batched(ids).unwrap();
            best = best.min(t0.elapsed().as_secs_f64());
        }
        eprintln!(
            "{label}: fused SwiGLU + quantizer = {fused}: prefill of {} tokens in {:.1} ms = {:.0} tokens/s",
            ids.len(),
            best * 1e3,
            ids.len() as f64 / best
        );
        results.push(logits);
    }
    results
}

#[test]
fn fused_swiglu_quantizer_is_bit_identical_on_a_hybrid_model() {
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
    // 1,500 tokens: a full chunk of 1,024 and a ragged tail of 476
    let base = ids.clone();
    while ids.len() < 1500 {
        ids.extend_from_slice(&base);
    }
    ids.truncate(1500);
    let results = swiglu_q_logits(&path, &ids, "Qwen3.5-0.8B");
    assert!(
        results[0] == results[1],
        "the fused SwiGLU + quantizer changes the logits (cosine {})",
        cosine(&results[0], &results[1])
    );
    assert!(results[1].iter().all(|v| v.is_finite()));
}

/// The same on a dense model (Qwen2.5-0.5B: the other prefill path of the engine); set FORGE_QWEN2_GGUF to the file. Skipped when it is missing.
#[test]
fn fused_swiglu_quantizer_is_bit_identical_on_a_dense_model() {
    let path = match std::env::var("FORGE_QWEN2_GGUF") {
        Ok(p) => PathBuf::from(p),
        Err(_) => models_dir().join("Qwen__Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q4_k_m.gguf"),
    };
    if !path.exists() || Device::count().unwrap_or(0) == 0 {
        eprintln!("skipped: the Qwen2.5 GGUF ({}) or a CUDA device is missing", path.display());
        return;
    }
    let base: Vec<u32> = vec![
        151644, 8948, 198, 2610, 525, 1207, 16948, 11, 3465, 553, 54364, 14817, 13, 1446, 525, 264, 10950, 17847, 13, 151645, 198, 151644, 872, 198, 7985, 264, 6386, 38242, 911, 279, 9396, 13, 151645, 198,
        151644, 77091, 198, 785, 9396, 374, 6884, 323, 6783, 11, 151645, 198, 151644, 872, 198, 9330, 432, 438, 264, 2805, 31365, 13, 151645, 198,
    ];
    let ids: Vec<u32> = (0..1500).map(|i| base[i % base.len()]).collect();
    let results = swiglu_q_logits(&path, &ids, "Qwen2.5-0.5B");
    assert!(
        results[0] == results[1],
        "the fused SwiGLU + quantizer changes the logits (cosine {})",
        cosine(&results[0], &results[1])
    );
    assert!(results[1].iter().all(|v| v.is_finite()));
}
