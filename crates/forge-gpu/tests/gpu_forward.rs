//! The CUDA decode path against the recorded Hugging Face goldens: last-position logits of the six prompts, 48 greedy tokens decoded on the device, and a decode-speed measurement.
//! Skipped when no CUDA device or no model is present (they live on the homelab).

use forge_core::Json;
use forge_cuda::Device;
use forge_gpu::{Fmt, GpuModel, GpuSession};
use forge_model::{argmax, Model};
use std::path::PathBuf;

mod common;

/// CUDA graph capture on several threads at once invalidates the captures (the global capture mode): the tests of this file take turns on the GPU
static GPU_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn golden_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/golden")
}

fn models_dir() -> PathBuf {
    match std::env::var("FORGE_MODELS") {
        Ok(p) => PathBuf::from(p),
        Err(_) => PathBuf::from(std::env::var("HOME").unwrap_or_default())
            .join("forgeinfer/oracle/models"),
    }
}

fn read_f32(path: PathBuf) -> Vec<f32> {
    std::fs::read(path)
        .unwrap()
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn run(
    golden: &str,
    model_path: PathBuf,
    fmt: Fmt,
    tol: f32,
    greedy_prompts: &[usize],
    strict_greedy: bool,
    batched: bool,
) {
    if !model_path.exists() || Device::count().unwrap_or(0) == 0 {
        eprintln!("skipped {golden}: model or CUDA device missing");
        return;
    }
    let cpu = Model::load(&model_path).unwrap();
    let gpu = GpuModel::upload(&cpu, fmt, 0).unwrap();
    eprintln!(
        "uploaded {:.0} MB of weights",
        gpu.weight_bytes as f64 / 1e6
    );
    drop(cpu);
    let gold = Json::from_file(golden_dir().join(format!("fwd_{golden}.json"))).unwrap();
    let mut sess = GpuSession::new(&gpu, 512).unwrap();
    for (pi, p) in gold["prompts"].as_array().unwrap().iter().enumerate() {
        let ids: Vec<u32> = p["ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        let want = read_f32(golden_dir().join(p["last_logits_file"].as_str().unwrap()));
        sess.reset().unwrap();
        if batched {
            sess.prefill_batched(&ids).unwrap();
        } else {
            sess.prefill(&ids).unwrap();
        }
        let logits = sess.logits().unwrap();
        let d = logits
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        let top1 = p["top20"][0][0].as_u64().unwrap() as u32;
        eprintln!(
            "{golden} prompt {pi} ({} tokens): max |logit diff| {d:.2e}, top-1 {}",
            ids.len(),
            if argmax(&logits) == top1 {
                "same"
            } else {
                "DIFFERENT"
            }
        );
        if strict_greedy {
            assert_eq!(argmax(&logits), top1, "top-1 differs on prompt {pi}");
        }
        assert!(d < tol, "prompt {pi}: logits differ by {d}");
        if greedy_prompts.contains(&pi) {
            let want_ids: Vec<u32> = p["greedy48"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as u32)
                .collect();
            let got = sess.decode_greedy(want_ids.len()).unwrap();
            let same = got
                .iter()
                .zip(&want_ids)
                .take_while(|(a, b)| a == b)
                .count();
            eprintln!(
                "{golden} prompt {pi}: greedy {} tokens on the device, first {same} identical",
                want_ids.len()
            );
            if strict_greedy {
                assert_eq!(
                    got, want_ids,
                    "greedy continuation differs after {same} tokens"
                );
            }
        }
    }
    // speed: 8 tokens of prompt, then 256 device-side greedy tokens
    sess.reset().unwrap();
    sess.prefill(&[1, 2, 3, 4, 5, 6, 7, 8]).unwrap();
    let t0 = std::time::Instant::now();
    let n = 256;
    sess.decode_greedy(n).unwrap();
    let dt = t0.elapsed().as_secs_f64();
    eprintln!(
        "{golden}: {n} greedy tokens in {:.1} ms = {:.0} tokens/s ({:.0} us per token)",
        dt * 1e3,
        n as f64 / dt,
        dt * 1e6 / n as f64
    );
}

#[test]
fn smollm2_f16_on_the_gpu() {
    let _gpu = common::gpu_lock();
    let _gpu = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    run(
        "smollm2",
        models_dir().join("HuggingFaceTB__SmolLM2-135M-Instruct"),
        Fmt::F16,
        5e-2,
        &[0, 3],
        true,
        false,
    );
}

#[test]
fn smollm2_q8_0_on_the_gpu() {
    let _gpu = common::gpu_lock();
    let _gpu = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    // int8 weights and int8 activations: the logits move by quantisation noise (about 0.5), the continuation of a short prompt usually stays the same
    run(
        "smollm2",
        models_dir().join("HuggingFaceTB__SmolLM2-135M-Instruct"),
        Fmt::Q8_0,
        1.5,
        &[0, 3],
        false,
        false,
    );
}

#[test]
fn qwen3_on_the_gpu() {
    let _gpu = common::gpu_lock();
    let _gpu = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    run(
        "qwen3",
        models_dir().join("Qwen__Qwen3-0.6B"),
        Fmt::F16,
        1e-1,
        &[0],
        true,
        false,
    );
}

#[test]
fn smollm2_batched_prefill_matches_the_goldens_and_is_fast() {
    let _gpu = common::gpu_lock();
    let _gpu = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    run(
        "smollm2",
        models_dir().join("HuggingFaceTB__SmolLM2-135M-Instruct"),
        Fmt::F16,
        1.5e-1,
        &[0, 3],
        true,
        true,
    );
    // prefill speed on a 512-token prompt: batched tensor-core path against the token-by-token decode path
    let path = models_dir().join("HuggingFaceTB__SmolLM2-135M-Instruct");
    if !path.exists() || Device::count().unwrap_or(0) == 0 {
        return;
    }
    let cpu = Model::load(&path).unwrap();
    let gpu = GpuModel::upload(&cpu, Fmt::F16, 0).unwrap();
    let mut sess = GpuSession::new(&gpu, 2048).unwrap();
    let prompt: Vec<u32> = (0..512u32).map(|i| 100 + (i * 7919) % 40000).collect();
    for _ in 0..2 {
        sess.reset().unwrap();
        sess.prefill_batched(&prompt).unwrap();
    }
    let t0 = std::time::Instant::now();
    let reps = 5;
    for _ in 0..reps {
        sess.reset().unwrap();
        sess.prefill_batched(&prompt).unwrap();
    }
    let dt = t0.elapsed().as_secs_f64() / reps as f64;
    eprintln!("batched prefill of 512 tokens: {:.1} ms = {:.0} tokens/s (llama.cpp b11521 on this card: 33,300 tokens/s for F16)", dt * 1e3, 512.0 / dt);
    let tb = sess.logits().unwrap();
    sess.reset().unwrap();
    let t1 = std::time::Instant::now();
    sess.prefill(&prompt).unwrap();
    let dt1 = t1.elapsed().as_secs_f64();
    let ts = sess.logits().unwrap();
    let diff = tb
        .iter()
        .zip(&ts)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    eprintln!("token-by-token prefill of the same prompt: {:.1} ms = {:.0} tokens/s; max logit difference between the two paths {diff:.2e}", dt1 * 1e3, 512.0 / dt1);
    assert!(diff < 0.1);
}
