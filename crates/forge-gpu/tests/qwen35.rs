//! The hybrid Qwen3.5-0.8B (18 Gated DeltaNet layers + 6 gated full-attention layers) on the GPU against the f32 CPU reference of the same weights. The GPU runs quantized weights
//! (Q8_0 made from the Hugging Face checkpoint, or the native blocks of the GGUF files) and 8-bit activations, so the logits differ by quantization noise only: same top token (or a
//! near tie), high cosine similarity. One sequential test (graph capture on several threads at once invalidates the captures). Skipped when the files or a CUDA device are missing.

use forge_core::Json;
use forge_cuda::Device;
use forge_gpu::{Fmt, GpuModel, GpuSession};
use forge_model::{argmax, Model, Session};
use std::path::PathBuf;

mod common;

fn models_dir() -> PathBuf {
    match std::env::var("FORGE_MODELS") {
        Ok(p) => PathBuf::from(p),
        Err(_) => PathBuf::from(std::env::var("HOME").unwrap_or_default())
            .join("forgeinfer/oracle/models"),
    }
}

fn golden_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/golden")
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

/// GPU (fmt) against the CPU reference of the same file; `prompts`: indices of the golden prompts to run.
fn check(name: &str, path: PathBuf, fmt: Fmt, min_cos: f64, prompts: &[usize]) {
    if !path.exists() || Device::count().unwrap_or(0) == 0 {
        eprintln!("skipped {name}: model or CUDA device missing");
        return;
    }
    let gold = Json::from_file(golden_dir().join("fwd_qwen35.json")).unwrap();
    let cpu = Model::load(&path).unwrap();
    let gpu_model = if fmt == Fmt::Native {
        Model::load_native(&path).unwrap()
    } else {
        Model::load(&path).unwrap()
    };
    let gpu = GpuModel::upload(&gpu_model, fmt, 0).unwrap();
    eprintln!(
        "{name}: {:.0} MB of weights on the GPU",
        gpu.weight_bytes as f64 / 1e6
    );
    drop(gpu_model);
    let mut sess = GpuSession::new(&gpu, 512).unwrap();
    for &pi in prompts {
        let p = &gold["prompts"].as_array().unwrap()[pi];
        let ids: Vec<u32> = p["ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        let mut cs = Session::new(&cpu, ids.len() + 8);
        let want = cs.prefill(&ids);
        for batched in [false, true] {
            sess.reset().unwrap();
            let t0 = std::time::Instant::now();
            if batched {
                sess.prefill_batched(&ids).unwrap();
            } else {
                sess.prefill(&ids).unwrap();
            }
            let got = sess.logits().unwrap();
            let cos = cosine(&got, &want);
            let (gt, wt) = (argmax(&got), argmax(&want));
            eprintln!(
                "{name} prompt {pi} ({} tokens, {}, {:.1} ms): cosine {cos:.5}, top-1 {}",
                ids.len(),
                if batched { "batched" } else { "token by token" },
                t0.elapsed().as_secs_f64() * 1e3,
                if gt == wt { "same" } else { "DIFFERENT" }
            );
            assert!(cos > min_cos, "{name} prompt {pi}: logits cosine {cos}");
            let mut order: Vec<usize> = (0..want.len()).collect();
            order.sort_by(|a, b| want[*b].partial_cmp(&want[*a]).unwrap());
            assert!(
                order[..3].contains(&(gt as usize)),
                "{name} prompt {pi}: GPU top-1 {gt} is not in the reference top 3"
            );
            // the recurrent state must carry on correctly: decode the next token and compare with the reference
            let next = wt;
            let got2 = sess.step(next).unwrap();
            let want2 = cs.step(next, true).unwrap();
            let cos2 = cosine(&got2, &want2);
            eprintln!(
                "{name} prompt {pi} ({}): next-token logits cosine {cos2:.5}",
                if batched { "batched" } else { "token by token" }
            );
            assert!(
                cos2 > min_cos,
                "{name} prompt {pi}: logits after one more token: cosine {cos2}"
            );
            if !batched {
                // restore the CPU session for the second round
                cs = Session::new(&cpu, ids.len() + 8);
                cs.prefill(&ids);
            }
        }
    }
}

#[test]
fn qwen35_on_the_gpu_matches_the_cpu_reference() {
    let _gpu = common::gpu_lock();
    let m = models_dir();
    // quantized from the Hugging Face weights
    check(
        "qwen35 HF -> Q8_0",
        m.join("Qwen__Qwen3.5-0.8B"),
        Fmt::Q8_0,
        0.99,
        &[0, 2, 6],
    );
    // the GGUF files in their own formats
    check(
        "qwen35 Q8_0 GGUF",
        m.join("unsloth__Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q8_0.gguf"),
        Fmt::Native,
        0.99,
        &[0, 2, 6],
    );
    check(
        "qwen35 Q4_K_M GGUF",
        m.join("unsloth__Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf"),
        Fmt::Native,
        0.99,
        &[0, 2, 6],
    );
}

/// A prompt longer than two prefill chunks: the batched path (chunks of PREFILL_CHUNK tokens, the recurrent state and the conv window carried from chunk to chunk) must agree with the
/// token-by-token path on the same weights.
#[test]
fn qwen35_long_prompts_agree_between_the_batched_and_the_token_by_token_path() {
    let _gpu = common::gpu_lock();
    let path = models_dir().join("unsloth__Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf");
    if !path.exists() || Device::count().unwrap_or(0) == 0 {
        eprintln!("skipped: model or CUDA device missing");
        return;
    }
    let gold = Json::from_file(golden_dir().join("fwd_qwen35.json")).unwrap();
    let mut ids: Vec<u32> = Vec::new();
    while ids.len() < 2500 {
        for p in gold["prompts"].as_array().unwrap() {
            ids.extend(
                p["ids"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_u64().unwrap() as u32),
            );
        }
    }
    ids.truncate(2500);
    let model = Model::load_native(&path).unwrap();
    let gpu = GpuModel::upload(&model, Fmt::Native, 0).unwrap();
    let mut sess = GpuSession::new(&gpu, 4096).unwrap();
    sess.prefill(&ids).unwrap();
    let a = sess.logits().unwrap();
    let a2 = sess.step(argmax(&a)).unwrap();
    sess.reset().unwrap();
    sess.prefill_batched(&ids).unwrap();
    let b = sess.logits().unwrap();
    let b2 = sess.step(argmax(&a)).unwrap();
    let (c1, c2) = (cosine(&a, &b), cosine(&a2, &b2));
    eprintln!("2500-token prompt: batched vs token by token: logits cosine {c1:.5}, next token {c2:.5}, top-1 {}", if argmax(&a) == argmax(&b) { "same" } else { "DIFFERENT" });
    assert!(
        c1 > 0.99 && c2 > 0.99,
        "chunked prefill disagrees: {c1} {c2}"
    );
}

/// Split-KV attention (contexts above 4096 positions) against the one-block-per-head kernel on the same 2500-token prompt, then four decoded tokens.
#[test]
fn qwen35_split_kv_attention_matches_the_single_block_kernel() {
    let _gpu = common::gpu_lock();
    let path = models_dir().join("unsloth__Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf");
    if !path.exists() || Device::count().unwrap_or(0) == 0 {
        eprintln!("skipped: model or CUDA device missing");
        return;
    }
    let gold = Json::from_file(golden_dir().join("fwd_qwen35.json")).unwrap();
    let mut ids: Vec<u32> = Vec::new();
    while ids.len() < 2500 {
        for p in gold["prompts"].as_array().unwrap() {
            ids.extend(
                p["ids"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_u64().unwrap() as u32),
            );
        }
    }
    ids.truncate(2500);
    let model = Model::load_native(&path).unwrap();
    let gpu = GpuModel::upload(&model, Fmt::Native, 0).unwrap();
    let mut run = |max_ctx: usize| -> Vec<Vec<f32>> {
        let mut sess = GpuSession::new(&gpu, max_ctx).unwrap();
        sess.prefill_batched(&ids).unwrap();
        let mut out = vec![sess.logits().unwrap()];
        let mut t = argmax(&out[0]);
        for _ in 0..4 {
            let l = sess.step(t).unwrap();
            t = argmax(&l);
            out.push(l);
        }
        out
    };
    let single = run(4096);
    let split = run(8192);
    for (i, (a, b)) in single.iter().zip(&split).enumerate() {
        let c = cosine(a, b);
        eprintln!(
            "position {}: single-block vs split-KV attention: logits cosine {c:.6}, top-1 {}",
            ids.len() + i,
            if argmax(a) == argmax(b) {
                "same"
            } else {
                "DIFFERENT"
            }
        );
        assert!(c > 0.9995, "split-KV attention disagrees at step {i}: {c}");
    }
}
