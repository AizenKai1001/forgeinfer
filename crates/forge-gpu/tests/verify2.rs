//! The two-token step of speculative decoding (Qwen3.5-0.8B Q4_K_M): the logits of both tokens must agree with two single-token steps, the state after it must carry on correctly, and a rollback
//! must return to the state after the first token. Skipped when the model or a CUDA device is missing.

use forge_core::Json;
use forge_cuda::Device;
use forge_gpu::{Fmt, GpuModel, GpuSession};
use forge_model::{argmax, Model};
use std::path::PathBuf;

mod common;

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
fn two_token_step_matches_two_single_steps() {
    let _gpu = common::gpu_lock();
    let path = models_dir().join("unsloth__Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf");
    if !path.exists() || Device::count().unwrap_or(0) == 0 {
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
    let model = Model::load_native(&path).unwrap();
    let gpu = GpuModel::upload(&model, Fmt::Native, 0).unwrap();
    for max_ctx in [2048usize, 6144] {
        // 6144 > 4096 selects the split-KV attention kernels
        let mut sess = GpuSession::new(&gpu, max_ctx).unwrap();
        sess.prefill_batched(&ids).unwrap();
        let u0 = argmax(&sess.logits().unwrap());
        let l0 = sess.step(u0).unwrap();
        let u1 = argmax(&l0);
        let l1 = sess.step(u1).unwrap();
        let u2 = argmax(&l1);
        let l2 = sess.step(u2).unwrap();

        sess.reset().unwrap();
        sess.prefill_batched(&ids).unwrap();
        sess.verify2(u0, u1).unwrap();
        let (v0, v1) = (
            sess.verify_logits(0).unwrap(),
            sess.verify_logits(1).unwrap(),
        );
        let am = sess.verify_argmax().unwrap();
        let (c0, c1) = (cosine(&l0, &v0), cosine(&l1, &v1));
        eprintln!("max_ctx {max_ctx}: two-token step vs two single steps: logits cosine {c0:.6} / {c1:.6}, arg-max {} {}", am[0] == argmax(&l0), am[1] == argmax(&l1));
        assert!(c0 > 0.9995 && c1 > 0.9995, "logits differ: {c0} {c1}");
        assert_eq!(am[0], argmax(&v0));
        assert_eq!(am[1], argmax(&v1));
        let l2b = sess.step(u2).unwrap();
        let c2 = cosine(&l2, &l2b);
        eprintln!("max_ctx {max_ctx}: the step after it: logits cosine {c2:.6}");
        assert!(
            c2 > 0.999,
            "the state after the two-token step is wrong: {c2}"
        );

        // rollback: a wrong second token, then the right one
        sess.reset().unwrap();
        sess.prefill_batched(&ids).unwrap();
        sess.verify2(u0, (u1 + 12345) % 1000 + 10).unwrap();
        sess.rollback_verify().unwrap();
        let l1c = sess.step(u1).unwrap();
        let c3 = cosine(&l1, &l1c);
        eprintln!("max_ctx {max_ctx}: after a rollback the second token's logits: cosine {c3:.6}");
        assert!(c3 > 0.999, "the rollback did not restore the state: {c3}");
    }
}

/// Where the time of one speculation cycle goes (run with --nocapture on an idle GPU; FORGE_VERIFY_MODEL = a Qwen3.5 GGUF, FORGE_VERIFY_HEAD = its mtp.safetensors).
#[test]
fn speculative_cycle_timing() {
    let _gpu = common::gpu_lock();
    let (Ok(path), Ok(head)) = (
        std::env::var("FORGE_VERIFY_MODEL"),
        std::env::var("FORGE_VERIFY_HEAD"),
    ) else {
        eprintln!("skipped: FORGE_VERIFY_MODEL / FORGE_VERIFY_HEAD not set");
        return;
    };
    if Device::count().unwrap_or(0) == 0 {
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
    let mut model = Model::load_native(PathBuf::from(path)).unwrap();
    model
        .load_mtp(
            PathBuf::from(head),
            &forge_model::LoadOpts {
                dequantize: false,
                keep_raw: true,
            },
        )
        .unwrap();
    let gpu = GpuModel::upload(&model, Fmt::Native, 0).unwrap();
    drop(model);
    let mut sess = GpuSession::new(&gpu, 2048).unwrap();
    sess.prefill_batched(&ids).unwrap();
    let mut t = argmax(&sess.logits().unwrap());
    for _ in 0..8 {
        sess.advance(t).unwrap();
        t = sess.argmax_token().unwrap();
    }
    // FORGE_TIMING_PHASE = single | verify | draft restricts the run to one kind of step (for a clean Nsight Systems profile of it); default: all with a summary
    let phase = std::env::var("FORGE_TIMING_PHASE").unwrap_or_else(|_| "all".into());
    let n = 60;
    if phase == "single" {
        for _ in 0..n * 2 {
            sess.advance(t).unwrap();
            t = sess.argmax_token().unwrap();
        }
        return;
    }
    let time = |f: &mut dyn FnMut()| {
        for _ in 0..3 {
            f();
        }
        let t0 = std::time::Instant::now();
        for _ in 0..n {
            f();
        }
        t0.elapsed().as_secs_f64() / n as f64 * 1e3
    };
    // single-token step (device loop); the positions advance, so this is done first
    let single = if phase == "all" {
        let t0 = std::time::Instant::now();
        for _ in 0..n {
            sess.advance(t).unwrap();
            t = sess.argmax_token().unwrap();
        }
        t0.elapsed().as_secs_f64() / n as f64 * 1e3
    } else {
        0.0
    };
    let pos = sess.pos;
    let hid = sess.hidden();
    let d = sess.mtp_step(t, hid, pos - 1, true).unwrap().unwrap();
    if phase == "verify" {
        for _ in 0..n * 2 {
            sess.verify2(t, d).unwrap();
            sess.rollback_verify().unwrap();
        }
        return;
    }
    // the head's steps rewrite the same slot, so they can be repeated
    let draft = time(&mut || {
        sess.mtp_step(t, hid, pos - 1, true).unwrap();
    });
    let kv_only = time(&mut || {
        sess.mtp_step(t, hid, pos - 1, false).unwrap();
    });
    if phase == "draft" {
        return;
    }
    let verify_rb = time(&mut || {
        sess.verify2(t, d).unwrap();
        sess.rollback_verify().unwrap();
    });
    let mut f = || {
        sess.verify2(t, d).unwrap();
        let _ = sess.verify_argmax().unwrap();
        sess.rollback_verify().unwrap();
    };
    let verify_arg = time(&mut f);
    let rb_only = {
        sess.verify2(t, d).unwrap();
        let t0 = std::time::Instant::now();
        for _ in 0..n {
            sess.rollback_verify().unwrap();
            // put the position back so that the rollback can be repeated
            sess.pos += 1;
        }
        let r = t0.elapsed().as_secs_f64() / n as f64 * 1e3;
        sess.pos -= 1;
        r
    };
    let commit = {
        let t0 = std::time::Instant::now();
        for _ in 0..n {
            sess.commit_verify().unwrap();
        }
        t0.elapsed().as_secs_f64() / n as f64 * 1e3
    };
    eprintln!("single-token step {single:.3} ms | MTP draft step {draft:.3} ms | MTP KV-only step {kv_only:.3} ms | two-token step + rollback {verify_rb:.3} ms (+ arg-max read {:.3}) | rollback alone {rb_only:.3} ms | commit {commit:.3} ms", verify_arg - verify_rb);
    let accepted = draft + verify_rb - rb_only + kv_only + commit;
    eprintln!("a cycle with the draft accepted costs about {accepted:.3} ms for 2 tokens = {:.3} ms per token (single step {single:.3}); rejected: {:.3} ms for 1 token", accepted / 2.0, draft + verify_rb);
}

/// Timing: a two-token step against a single-token step (run with --nocapture on an idle GPU; the point of speculative decoding is that the first costs far less than twice the second).
#[test]
fn two_token_step_timing() {
    let _gpu = common::gpu_lock();
    let path = std::env::var("FORGE_VERIFY_MODEL")
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
    let ids: Vec<u32> = gold["prompts"][6]["ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect();
    let model = Model::load_native(&path).unwrap();
    let gpu = GpuModel::upload(&model, Fmt::Native, 0).unwrap();
    drop(model);
    let mut sess = GpuSession::new(&gpu, 2048).unwrap();
    sess.prefill_batched(&ids).unwrap();
    let mut t = argmax(&sess.logits().unwrap());
    // single steps (the device-resident loop is the fastest way to run them; step() adds a logits copy, advance() does not)
    for _ in 0..8 {
        sess.advance(t).unwrap();
        t = sess.argmax_token().unwrap();
    }
    let n = 100;
    let t0 = std::time::Instant::now();
    for _ in 0..n {
        sess.advance(t).unwrap();
        t = sess.argmax_token().unwrap();
    }
    let single = t0.elapsed().as_secs_f64() / n as f64;
    sess.verify2(t, 1000).unwrap();
    sess.rollback_verify().unwrap();
    let t0 = std::time::Instant::now();
    for _ in 0..n {
        sess.verify2(t, 1000).unwrap();
        sess.rollback_verify().unwrap();
    }
    let double = t0.elapsed().as_secs_f64() / n as f64;
    eprintln!("single-token step {:.3} ms ({:.0} tokens/s), two-token step with rollback {:.3} ms = {:.2}x of one step", single * 1e3, 1.0 / single, double * 1e3, double / single);
}
