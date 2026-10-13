//! Qwen3.5's MTP head on the GPU (Qwen3.5-0.8B Q4_K_M main model, the head from the Hugging Face checkpoint quantized to Q8_0): its draft for the token after the next one must be the main
//! model's own prediction most of the time (the CPU reference reaches 95 % on this text), and the KV entries written during the batched prefill must give the same drafts as the ones written
//! one token at a time. Skipped when the files or a CUDA device are missing.

use forge_core::Json;
use forge_cuda::Device;
use forge_gpu::{Fmt, GpuModel, GpuSession};
use forge_model::{argmax, LoadOpts, Model};
use std::path::PathBuf;

fn models_dir() -> PathBuf {
    match std::env::var("FORGE_MODELS") {
        Ok(p) => PathBuf::from(p),
        Err(_) => PathBuf::from(std::env::var("HOME").unwrap_or_default())
            .join("forgeinfer/oracle/models"),
    }
}

#[test]
fn mtp_drafts_on_the_gpu_agree_with_the_main_model() {
    let gguf = models_dir().join("unsloth__Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf");
    let hf = models_dir().join("Qwen__Qwen3.5-0.8B");
    if !gguf.exists() || !hf.exists() || Device::count().unwrap_or(0) == 0 {
        eprintln!("skipped: model or CUDA device missing");
        return;
    }
    let gold = Json::from_file(
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/golden/fwd_qwen35.json"),
    )
    .unwrap();
    let ids: Vec<u32> = gold["prompts"][6]["ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect();
    let mut model = Model::load_native(&gguf).unwrap();
    model
        .load_mtp(
            &hf,
            &LoadOpts {
                dequantize: false,
                keep_raw: true,
            },
        )
        .unwrap();
    assert!(model.mtp.is_some());
    let gpu = GpuModel::upload(&model, Fmt::Native, 0).unwrap();
    drop(model);
    let mut sess = GpuSession::new(&gpu, 512).unwrap();
    let (mut agree_batched, mut agree_tokenwise, mut same, mut n) = (0, 0, 0, 0);
    for k in (10..ids.len() - 1).step_by(4) {
        // batched prefill of ids[..k], the head's draft for the token after ids[k]
        sess.reset().unwrap();
        sess.prefill_batched(&ids[..k]).unwrap();
        assert!(sess.mtp_ready());
        let d1 = sess
            .mtp_step(ids[k], sess.hidden(), k - 1, true)
            .unwrap()
            .unwrap();
        sess.advance(ids[k]).unwrap();
        let main_next = sess.argmax_token().unwrap();
        // token-by-token prefill of the same tokens
        sess.reset().unwrap();
        sess.prefill(&ids[..k]).unwrap();
        assert!(sess.mtp_ready());
        let d2 = sess
            .mtp_step(ids[k], sess.hidden(), k - 1, true)
            .unwrap()
            .unwrap();
        n += 1;
        agree_batched += (d1 == main_next) as usize;
        agree_tokenwise += (d2 == main_next) as usize;
        same += (d1 == d2) as usize;
    }
    eprintln!("MTP drafts at {n} positions: equal to the main model's own next-next token: {agree_batched} (batched prefill), {agree_tokenwise} (token-by-token prefill); the two prefills give the same draft {same} times");
    assert!(
        agree_batched as f64 > 0.7 * n as f64,
        "the drafts rarely match the main model: {agree_batched} of {n}"
    );
    assert!(agree_tokenwise as f64 > 0.7 * n as f64);
    assert!(same as f64 > 0.9 * n as f64);
}
