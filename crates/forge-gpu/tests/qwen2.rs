//! Qwen2.5-0.5B-Instruct (Q4_K_M GGUF: Q5_0, Q8_0, Q4_K and Q6_K tensors, 14 query heads over 2 key/value heads, biases on the q, k and v projections) on the GPU against the f32 CPU reference of the
//! same file. The GPU runs the quantized blocks and 8-bit activations, so the logits differ by quantization noise only: same top token (or a near tie) and a high cosine similarity. Both ways of
//! reading a prompt are checked (token by token = the decode kernels, batched = the prompt kernels), and the step after it. Skipped when the file or a CUDA device is missing: set
//! FORGE_QWEN2_GGUF to the file, or put it in FORGE_MODELS as Qwen__Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q4_k_m.gguf.

use forge_cuda::Device;
use forge_gpu::{Fmt, GpuModel, GpuSession};
use forge_model::{argmax, Model, Session};
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

/// Token ids of the Qwen2.5 tokenizer (`forge tokenize`): a plain sentence, a chat prompt with the default system turn, and a longer chat prompt.
fn prompts() -> Vec<Vec<u32>> {
    vec![
        vec![9707, 11, 1879, 0, 576, 3974, 13876, 38835, 34208, 916, 279, 15678, 5562, 13],
        vec![
            151644, 8948, 198, 2610, 525, 1207, 16948, 11, 3465, 553, 54364, 14817, 13, 1446, 525, 264, 10950, 17847, 13, 151645, 198, 151644, 872, 198, 7985, 264, 6386, 38242, 911, 279, 9396, 13,
            151645, 198, 151644, 77091, 198,
        ],
        // the chat prompt above followed by a second user turn: positions beyond 32 (a prompt chunk of more than one tile)
        {
            let mut v = vec![
                151644, 8948, 198, 2610, 525, 1207, 16948, 11, 3465, 553, 54364, 14817, 13, 1446, 525, 264, 10950, 17847, 13, 151645, 198, 151644, 872, 198, 7985, 264, 6386, 38242, 911, 279, 9396, 13,
                151645, 198, 151644, 77091, 198, 785, 9396, 374, 6884, 323, 6783, 11, 151645, 198, 151644, 872, 198, 9330, 432, 438, 264, 2805, 31365, 13, 151645, 198, 151644, 77091, 198,
            ];
            v.extend((0..40u32).map(|i| 1000 + i * 37 % 5000));
            v
        },
    ]
}

#[test]
fn qwen2_on_the_gpu_matches_the_cpu_reference() {
    let _gpu = common::gpu_lock();
    let path = model_path();
    if !path.exists() || Device::count().unwrap_or(0) == 0 {
        eprintln!("skipped: the Qwen2.5 GGUF ({}) or a CUDA device is missing", path.display());
        return;
    }
    let cpu = Model::load(&path).unwrap();
    assert!(cpu.cfg.qkv_bias && !cpu.cfg.qk_norm, "a Qwen2 model has biases and no QK norm");
    assert!(cpu.layers.iter().all(|l| l.bq.is_some() && l.bk.is_some() && l.bv.is_some()));
    let native = Model::load_native(&path).unwrap();
    let gpu = GpuModel::upload(&native, Fmt::Native, 0).unwrap();
    eprintln!("Qwen2.5-0.5B: {:.0} MB of weights on the GPU", gpu.weight_bytes as f64 / 1e6);
    drop(native);
    let mut sess = GpuSession::new(&gpu, 512).unwrap();
    for (pi, ids) in prompts().iter().enumerate() {
        let mut cs = Session::new(&cpu, ids.len() + 8);
        let want = cs.prefill(ids);
        for batched in [false, true] {
            sess.reset().unwrap();
            if batched {
                sess.prefill_batched(ids).unwrap();
            } else {
                sess.prefill(ids).unwrap();
            }
            let got = sess.logits().unwrap();
            let cos = cosine(&got, &want);
            let (gt, wt) = (argmax(&got), argmax(&want));
            eprintln!(
                "prompt {pi} ({} tokens, {}): cosine {cos:.5}, top-1 {}",
                ids.len(),
                if batched { "batched" } else { "token by token" },
                if gt == wt { "same" } else { "DIFFERENT" }
            );
            assert!(cos > 0.99, "prompt {pi}: logits cosine {cos}");
            let mut order: Vec<usize> = (0..want.len()).collect();
            order.sort_by(|a, b| want[*b].partial_cmp(&want[*a]).unwrap());
            assert!(order[..3].contains(&(gt as usize)), "prompt {pi}: GPU top-1 {gt} is not in the reference top 3");
            // the cache must carry on correctly: decode the next token and compare with the reference
            let got2 = sess.step(wt).unwrap();
            let want2 = cs.step(wt, true).unwrap();
            let cos2 = cosine(&got2, &want2);
            eprintln!("prompt {pi} ({}): next-token logits cosine {cos2:.5}", if batched { "batched" } else { "token by token" });
            assert!(cos2 > 0.99, "prompt {pi}: logits after one more token: cosine {cos2}");
            if !batched {
                cs = Session::new(&cpu, ids.len() + 8);
                cs.prefill(ids);
            }
        }
    }
}
