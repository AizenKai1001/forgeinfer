//! The split-KV decode attention of dense models (decode.cu: `attn_split_kernel` + `attn_merge_*_kernel`, used from a context of 4,096 positions on; the one-block-per-head kernel keeps all scores in 48 KB of
//! shared memory and fails beyond about 11k positions) against the one-block kernel on the same context: the last logits after a few decoded tokens must agree to float rounding and the greedy tokens must
//! match. The decode graph is captured when the session is created, so each variant gets its own session (FORGE_ATTN_SPLIT=0 / 1). Also checks that a context far beyond the old limit loads and decodes.
//! Qwen2.5-0.5B-Instruct (14 query heads over 2 key/value heads of width 64); skipped when the file or a CUDA device is missing: set FORGE_QWEN2_GGUF to the file, or put it in FORGE_MODELS as
//! Qwen__Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q4_k_m.gguf.

use forge_cuda::Device;
use forge_gpu::{Fmt, GpuModel, GpuSession};
use forge_model::Model;
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

/// the chat prompts of the Qwen2 test repeated: natural token statistics and a low-entropy next-token distribution (a random-token prompt gives a flat one in which a rounding step of the 8-bit
/// activations flips the greedy token and the histories diverge: that tests the chaos of the model, not the kernel)
fn prompt(n: usize) -> Vec<u32> {
    let base: Vec<u32> = vec![
        151644, 8948, 198, 2610, 525, 1207, 16948, 11, 3465, 553, 54364, 14817, 13, 1446, 525, 264, 10950, 17847, 13, 151645, 198, 151644, 872, 198, 7985, 264, 6386, 38242, 911, 279, 9396, 13, 151645, 198,
        151644, 77091, 198, 785, 9396, 374, 6884, 323, 6783, 11, 151645, 198, 151644, 872, 198, 9330, 432, 438, 264, 2805, 31365, 13, 151645, 198,
    ];
    (0..n).map(|i| base[i % base.len()]).collect()
}

/// prefill the first `n` tokens in chunks, then a short tail (the multi-row kernels), then decode `steps` tokens: (tokens, last logits)
fn run(gpu: &GpuModel, split: Option<bool>, ctx: usize, n: usize, tail: usize, steps: usize) -> (Vec<u32>, Vec<f32>) {
    match split {
        Some(on) => std::env::set_var("FORGE_ATTN_SPLIT", if on { "1" } else { "0" }),
        None => std::env::remove_var("FORGE_ATTN_SPLIT"),
    }
    let mut sess = GpuSession::new(gpu, ctx).unwrap();
    std::env::remove_var("FORGE_ATTN_SPLIT");
    let p = prompt(n + tail);
    sess.prefill_batched(&p[..n]).unwrap();
    if tail > 0 {
        sess.prefill_batched(&p[n..]).unwrap();
    }
    let toks = sess.decode_greedy(steps).unwrap();
    (toks, sess.logits().unwrap())
}

#[test]
fn split_kv_decode_attention_matches_the_one_block_kernel() {
    let _gpu = common::gpu_lock();
    let path = model_path();
    if !path.exists() || Device::count().unwrap_or(0) == 0 {
        eprintln!("skipped: the Qwen2.5 GGUF ({}) or a CUDA device is missing", path.display());
        return;
    }
    let model = Model::load_native(&path).unwrap();
    let gpu = GpuModel::upload(&model, Fmt::Native, 0).unwrap();
    drop(model);
    // the prompt in chunks (the prompt kernels), a short tail (the multi-row kernels), then 40 decode steps through the decode graph with the attention under test. A different order of summation flips a
    // rounding step of the 8-bit activations here and there (a logit moves by 0.2 to 0.7, cosine 0.999 after 40 steps): the greedy tokens of a natural prompt must not change.
    for &(n, tail) in &[(3000usize, 0usize), (3000, 20), (1500, 5), (200, 0)] {
        let (ta, la) = run(&gpu, Some(false), 4096, n, tail, 40);
        let (tb, lb) = run(&gpu, Some(true), 4096, n, tail, 40);
        let cos = cosine(&la, &lb);
        let maxdiff = la.iter().zip(&lb).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        let same = ta.iter().zip(&tb).filter(|(a, b)| a == b).count();
        eprintln!("prompt {n} + {tail}: {same} of {} greedy tokens equal, cosine of the last logits {cos:.8}, max |difference| {maxdiff:.2e}", ta.len());
        assert_eq!(ta, tb, "the greedy tokens differ");
        assert!(cos > 0.99, "the split-KV attention changes the logits: cosine {cos}");
        assert!(lb.iter().all(|v| v.is_finite()));
    }
}

#[test]
fn a_dense_model_loads_and_decodes_beyond_the_old_context_limit() {
    let _gpu = common::gpu_lock();
    let path = model_path();
    if !path.exists() || Device::count().unwrap_or(0) == 0 {
        eprintln!("skipped: the Qwen2.5 GGUF ({}) or a CUDA device is missing", path.display());
        return;
    }
    let model = Model::load_native(&path).unwrap();
    let gpu = GpuModel::upload(&model, Fmt::Native, 0).unwrap();
    drop(model);
    // 16,500 positions: the one-block kernel needs 66 KB of shared memory for its scores and fails to launch (CUDA_ERROR_INVALID_VALUE)
    let (toks, logits) = run(&gpu, None, 16_500, 16_000, 0, 8);
    eprintln!("16,000-token prompt, 8 greedy tokens: {toks:?}");
    assert_eq!(toks.len(), 8);
    assert!(logits.iter().all(|v| v.is_finite()));
}

/// The two kernels called directly on random caches: 16 query tokens at positions 3000 .. 3015 (the multi-row layout, grid y / z = token), head widths 64 and 128. The quantized outputs agree up to rounding of
/// the last bit of a scale: at most one step of the int8 values in a tiny fraction, equal sums otherwise.
#[test]
fn split_kernels_match_the_one_block_kernel_on_random_caches() {
    use forge_core::half::f32_to_f16;
    use forge_cuda::{Args, DeviceBuffer, Module, Stream};
    use forge_model::sample::Rng;
    let _gpu = common::gpu_lock();
    if Device::count().unwrap_or(0) == 0 {
        eprintln!("skipped: no CUDA device");
        return;
    }
    let dev = Device::open(0).unwrap();
    let (image, _) = forge_kernels::image_for(dev.info.cc_major, dev.info.cc_minor).unwrap();
    let module = Module::load(image).unwrap();
    let stream = Stream::new().unwrap();
    let one = module.function("attention_q_kernel").unwrap();
    let split = module.function("attn_split_kernel").unwrap();
    let merge = module.function("attn_merge_q_kernel").unwrap();
    for &(n_head, n_kv, hd) in &[(14usize, 2usize, 64usize), (8, 2, 128)] {
        let (pos0, ntok) = (3000usize, 16usize);
        let ctx = pos0 + ntok;
        let t_max = 3100usize;
        let ns_max = (t_max + 1023) / 1024;
        let kv_dim = n_kv * hd;
        let mut rng = Rng::new(7 + hd as u64);
        let kh: Vec<u8> = (0..ctx * kv_dim).flat_map(|_| f32_to_f16(rng.next_f32() * 2.0 - 1.0).to_le_bytes()).collect();
        let vh: Vec<u8> = (0..ctx * kv_dim).flat_map(|_| f32_to_f16(rng.next_f32() * 2.0 - 1.0).to_le_bytes()).collect();
        let q: Vec<f32> = (0..ntok * n_head * hd).map(|_| (rng.next_f32() * 2.0 - 1.0) * 3.0).collect();
        let dk = DeviceBuffer::from_bytes(&kh).unwrap();
        let dv = DeviceBuffer::from_bytes(&vh).unwrap();
        let dq = DeviceBuffer::alloc(q.len() * 4).unwrap();
        dq.upload_f32(&q).unwrap();
        let mut st = vec![0u8; 16];
        st[4..8].copy_from_slice(&(pos0 as u32).to_le_bytes());
        let dst = DeviceBuffer::from_bytes(&st).unwrap();
        let groups = ntok * n_head * hd / 32;
        let run_one = || {
            let (oq, od, os) = (DeviceBuffer::alloc(ntok * n_head * hd).unwrap(), DeviceBuffer::alloc(groups * 4).unwrap(), DeviceBuffer::alloc(groups * 4).unwrap());
            let ngrp = 128 / (hd / 8);
            let smem = ((t_max + hd + ngrp * hd + 32) * 4) as u32;
            one.launch(
                (n_head as u32, ntok as u32, 1),
                (128, 1, 1),
                smem,
                &stream,
                &Args::new().buf(&dq).buf(&dk).buf(&dv).buf(&oq).buf(&od).buf(&os).buf(&dst).i32(n_head as i32).i32(n_kv as i32).i32(hd as i32).i32(t_max as i32),
            )
            .unwrap();
            stream.synchronize().unwrap();
            (oq, od, os)
        };
        let run_split = || {
            let (oq, od, os) = (DeviceBuffer::alloc(ntok * n_head * hd).unwrap(), DeviceBuffer::alloc(groups * 4).unwrap(), DeviceBuffer::alloc(groups * 4).unwrap());
            let (pm, pl, po) = (
                DeviceBuffer::alloc(ntok * n_head * ns_max * 4).unwrap(),
                DeviceBuffer::alloc(ntok * n_head * ns_max * 4).unwrap(),
                DeviceBuffer::alloc(ntok * n_head * ns_max * 256 * 4).unwrap(),
            );
            split
                .launch(
                    (n_head as u32, ns_max as u32, ntok as u32),
                    (128, 1, 1),
                    0,
                    &stream,
                    &Args::new().buf(&dq).buf(&dk).buf(&dv).buf(&pm).buf(&pl).buf(&po).buf(&dst).i32(n_head as i32).i32(n_kv as i32).i32(hd as i32).i32(ns_max as i32),
                )
                .unwrap();
            merge
                .launch(
                    (n_head as u32, ntok as u32, 1),
                    (128, 1, 1),
                    0,
                    &stream,
                    &Args::new().buf(&pm).buf(&pl).buf(&po).buf(&oq).buf(&od).buf(&os).buf(&dst).i32(ns_max as i32).i32(hd as i32),
                )
                .unwrap();
            stream.synchronize().unwrap();
            (oq, od, os)
        };
        let (a, b) = (run_one(), run_split());
        let mut qa = vec![0u8; ntok * n_head * hd];
        let mut qb = vec![0u8; ntok * n_head * hd];
        a.0.download(&mut qa).unwrap();
        b.0.download(&mut qb).unwrap();
        let (da, db) = (a.1.download_f32(groups).unwrap(), b.1.download_f32(groups).unwrap());
        let diff_q = qa.iter().zip(&qb).filter(|(x, y)| x != y).count();
        let far = qa.iter().zip(&qb).filter(|(x, y)| ((**x as i8) as i32 - (**y as i8) as i32).abs() > 1).count();
        let dscale = da.iter().zip(&db).map(|(x, y)| ((x - y) / x.abs().max(1e-12)).abs()).fold(0f32, f32::max);
        eprintln!("hd {hd}: {diff_q} of {} int8 values differ (by more than one: {far}), largest relative difference of a scale {dscale:.2e}", qa.len());
        assert!(far == 0 && diff_q * 100 <= qa.len(), "hd {hd}: the split attention differs from the one-block kernel");
        assert!(dscale < 1e-4);
    }
}
