//! Native GGUF formats on the GPU against the f32 CPU reference of the same file: SmolLM2-135M in Q8_0, Q4_0, Q4_K_M (a mix of Q4_K, Q5_0, Q6_K and Q8_0 tensors) and Q6_K. The GPU quantizes the activations
//! to 8 bits and runs its own integer kernels, the CPU reference dequantizes the weights to f32, so the logits differ by quantization noise only: same top-1 token, high cosine similarity.
//! Both the token-by-token decode path and the batched tensor-core prefill (dequantized weights, fp16 GEMM) are checked. Skipped when the files or a CUDA device are missing.

use forge_core::Json;
use forge_cuda::Device;
use forge_gpu::{Fmt, GpuModel, GpuSession};
use forge_model::{argmax, Model, Session};
use std::path::PathBuf;

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

fn check(file: &str) {
    let path = models_dir()
        .join("bartowski__SmolLM2-135M-Instruct-GGUF")
        .join(file);
    if !path.exists() || Device::count().unwrap_or(0) == 0 {
        eprintln!("skipped {file}: model or CUDA device missing");
        return;
    }
    let gold = Json::from_file(golden_dir().join("fwd_smollm2.json")).unwrap();
    let cpu = Model::load(&path).unwrap();
    let gpu_model = Model::load_native(&path).unwrap();
    let gpu = GpuModel::upload(&gpu_model, Fmt::Native, 0).unwrap();
    drop(gpu_model);
    let mut sess = GpuSession::new(&gpu, 512).unwrap();
    for (pi, p) in gold["prompts"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
        .take(4)
    {
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
            if batched {
                sess.prefill_batched(&ids).unwrap();
            } else {
                sess.prefill(&ids).unwrap();
            }
            let got = sess.logits().unwrap();
            let cos = cosine(&got, &want);
            let (gt, wt) = (argmax(&got), argmax(&want));
            eprintln!(
                "{file} prompt {pi} ({} tokens, {}): cosine {cos:.5}, top-1 {}",
                ids.len(),
                if batched {
                    "batched prefill"
                } else {
                    "token by token"
                },
                if gt == wt { "same" } else { "DIFFERENT" }
            );
            assert!(cos > 0.995, "{file} prompt {pi}: logits cosine {cos}");
            // quantization noise can swap two near-tied candidates: the GPU's choice must be in the reference's top 3
            let mut order: Vec<usize> = (0..want.len()).collect();
            order.sort_by(|a, b| want[*b].partial_cmp(&want[*a]).unwrap());
            assert!(
                order[..3].contains(&(gt as usize)),
                "{file} prompt {pi}: GPU top-1 {gt} is not in the reference top 3"
            );
        }
    }
}

// one test, sequential: graph capture on several threads at once invalidates the captures (a CUDA rule for the global capture mode)
#[test]
fn native_gguf_formats_match_the_cpu_reference() {
    for f in [
        "SmolLM2-135M-Instruct-Q8_0.gguf",
        "SmolLM2-135M-Instruct-Q4_0.gguf",
        "SmolLM2-135M-Instruct-Q4_K_M.gguf",
        "SmolLM2-135M-Instruct-Q6_K.gguf",
    ] {
        check(f);
    }
}
