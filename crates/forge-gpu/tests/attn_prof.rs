//! Profiling helper (ignored by default): the per-kernel time of a short chunk of tokens read after a long prompt, as the engine reads the tokens of a chat follow-up (the multi-row path).
//!     FORGE_KPROF=1 FORGE_QWEN2_GGUF=model.gguf PROF_POS=6300 PROF_ROWS=16 cargo test --release -p forge-gpu --test attn_prof -- --ignored --nocapture
//! prints the table of `forge_cuda::kprof_report` (every kernel of the prompt and of the repeated chunk, time and launches; the chunk's kernels are launched 20 times).

use forge_cuda::Device;
use forge_gpu::{Fmt, GpuModel, GpuSession};
use forge_model::Model;
use std::path::PathBuf;

mod common;

fn model_path() -> PathBuf {
    if let Ok(p) = std::env::var("FORGE_QWEN2_GGUF") {
        return PathBuf::from(p);
    }
    PathBuf::from(std::env::var("HOME").unwrap_or_default()).join("forgeinfer/oracle/models/Qwen__Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q4_k_m.gguf")
}

#[test]
#[ignore]
fn kernel_times_of_a_short_chunk_after_a_long_prompt() {
    let _g = common::gpu_lock();
    let path = model_path();
    if !path.exists() || Device::count().unwrap_or(0) == 0 {
        eprintln!("skipped: {} or a CUDA device is missing", path.display());
        return;
    }
    let env = |k: &str, d: usize| std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d);
    let (pos, rows) = (env("PROF_POS", 6300), env("PROF_ROWS", 16));
    let model = Model::load_native(&path).unwrap();
    let gpu = GpuModel::upload(&model, Fmt::Native, 0).unwrap();
    drop(model);
    let mut sess = GpuSession::new(&gpu, pos + 2048).unwrap();
    let base: Vec<u32> = vec![
        151644, 8948, 198, 2610, 525, 1207, 16948, 11, 3465, 553, 54364, 14817, 13, 1446, 525, 264, 10950, 17847, 13, 151645, 198, 151644, 872, 198, 7985, 264, 6386, 38242, 911, 279, 9396, 13, 151645, 198, 151644,
        77091, 198, 785, 9396, 374, 6884, 323, 6783, 11, 151645, 198, 151644, 872, 198, 9330, 432, 438, 264, 2805, 31365, 13, 151645, 198,
    ];
    let p: Vec<u32> = (0..pos + rows).map(|i| base[i % base.len()]).collect();
    sess.prefill_batched(&p[..pos]).unwrap();
    let mut best = f64::MAX;
    for _ in 0..20 {
        sess.truncate(pos);
        let t = std::time::Instant::now();
        sess.prefill_batched(&p[pos..pos + rows]).unwrap();
        best = best.min(t.elapsed().as_secs_f64());
    }
    eprintln!("{rows} tokens after {pos} positions: best of 20 {:.1} ms", best * 1e3);
    println!("{}", forge_cuda::kprof_report());
}
