//! The split attention kernels that read a compressed KV cache directly (decode.cu `attn_split_q_kernel`, decode_h.cu `attn_split_g_q_kernel`) against the path they replace: `kv_dequant_kernel` writes the context
//! into the fp16 scratch and the plain split kernel reads the scratch. The fused kernels take the scale out of the sum (a value is an integer times the scale of its group, the fp16 rounding of the dequantized
//! value is gone), so the partial results (maximum, sum, unnormalised output of every split) agree to a small tolerance instead of bit for bit: scores to 0.05, the sum to 5 percent, an output value to
//! 5 percent of the sum (an error in decoding, a wrong scale or nibble order, is of the order of the sum itself). Where nothing is read from the store (the first position: the keys are the fp16 rows of the step)
//! they are equal bit for bit. Int8 and int4 keys and values, several scale groups, head widths 64, 128 and 256, groups of one to three tokens and positions around the 1,024-position splits. No model is needed:
//! random caches. Skipped without a CUDA device.

use forge_cuda::{Args, Device, DeviceBuffer, Module, Stream};
use forge_model::sample::Rng;

mod common;

fn f16_bytes(rng: &mut Rng, n: usize, amp: f32) -> Vec<u8> {
    use forge_core::half::f32_to_f16;
    (0..n).flat_map(|_| f32_to_f16((rng.next_f32() * 2.0 - 1.0) * amp).to_le_bytes()).collect()
}

#[test]
fn the_fused_kernels_agree_with_the_dequantizing_path() {
    let _gpu = common::gpu_lock();
    if Device::count().unwrap_or(0) == 0 {
        eprintln!("skipped: no CUDA device");
        return;
    }
    let dev = Device::open(0).unwrap();
    let (image, _) = forge_kernels::image_for(dev.info.cc_major, dev.info.cc_minor).unwrap();
    let module = Module::load(image).unwrap();
    let stream = Stream::new().unwrap();
    let quant = module.function("kv_quant_kernel").unwrap();
    let dequant = module.function("kv_dequant_kernel").unwrap();
    let plain = module.function("attn_split_kernel").unwrap();
    let plain_g = module.function("attn_split_g_kernel").unwrap();
    let fused = module.function("attn_split_q_kernel").unwrap();
    let fused_g = module.function("attn_split_g_q_kernel").unwrap();
    let mut compared = 0usize;
    let mut worst = 0f64;
    // (heads, kv heads, head width, bits of K, bits of V, scale group, hybrid kernel)
    let configs: [(usize, usize, usize, u32, u32, usize, bool); 9] = [
        (14, 2, 64, 8, 8, 32, false),
        (14, 2, 64, 4, 4, 64, false),
        (8, 2, 128, 8, 4, 32, false),
        (32, 8, 128, 4, 8, 128, false),
        (8, 2, 128, 4, 4, 32, false),
        (16, 4, 256, 8, 8, 32, true),
        (16, 4, 256, 4, 4, 32, true),
        (16, 4, 256, 8, 4, 64, true),
        (16, 4, 256, 4, 8, 256, true),
    ];
    for &(n_head, n_kv, hd, bits_k, bits_v, group, hybrid) in &configs {
        let max_ctx = 2600usize;
        let ns_max = (max_ctx + 1023) / 1024;
        let kv_dim = n_kv * hd;
        let mut rng = Rng::new(31 + hd as u64 + bits_k as u64 * 7 + bits_v as u64);
        // the truth: fp16 keys and values of every position (a few outliers so that the scales differ)
        let kt = f16_bytes(&mut rng, max_ctx * kv_dim, 2.0);
        let vt = f16_bytes(&mut rng, max_ctx * kv_dim, 1.0);
        let d_kt = DeviceBuffer::from_bytes(&kt).unwrap();
        let d_vt = DeviceBuffer::from_bytes(&vt).unwrap();
        let rows = max_ctx * n_kv;
        let (kbytes, vbytes) = (rows * hd * bits_k as usize / 8, rows * hd * bits_v as usize / 8);
        let sbytes = rows * (hd / group) * 2;
        let (kst, vst) = (DeviceBuffer::alloc(kbytes).unwrap(), DeviceBuffer::alloc(vbytes).unwrap());
        let (ksc, vsc) = (DeviceBuffer::alloc(sbytes).unwrap(), DeviceBuffer::alloc(sbytes).unwrap());
        for &(pos0, ntok) in &[(0usize, 1usize), (1usize, 1usize), (1023, 2), (1024, 1), (1500, 3), (2047, 2), (2048, 1), (2590, 3)] {
            assert!(pos0 + ntok <= max_ctx);
            let mut st = vec![0u8; 16];
            st[4..8].copy_from_slice(&(pos0 as u32).to_le_bytes());
            let dst = DeviceBuffer::from_bytes(&st).unwrap();
            // the store of the positions before the step (kv_quant_kernel reads rows of the truth and writes the store)
            if pos0 > 0 {
                let blocks = ((pos0 * n_kv + 7) / 8).clamp(1, 2048) as u32;
                quant
                    .launch(
                        (blocks, 2, 1),
                        (256, 1, 1),
                        0,
                        &stream,
                        &Args::new().buf(&d_kt).buf(&kst).buf(&ksc).buf(&d_vt).buf(&vst).buf(&vsc).buf(&dst).i32(0).i32(pos0 as i32).i32(n_kv as i32).i32(hd as i32).i32(group as i32).i32(bits_k as i32).i32(bits_v as i32),
                    )
                    .unwrap();
            }
            // the reference scratch: the dequantized positions before the step, the rows of the step as they are
            let ref_k = DeviceBuffer::from_bytes(&kt).unwrap();
            let ref_v = DeviceBuffer::from_bytes(&vt).unwrap();
            if pos0 > 0 {
                let blocks = ((pos0 * n_kv + 7) / 8).min(2048) as u32;
                dequant
                    .launch(
                        (blocks, 2, 1),
                        (256, 1, 1),
                        0,
                        &stream,
                        &Args::new().buf(&kst).buf(&ksc).buf(&ref_k).buf(&vst).buf(&vsc).buf(&ref_v).buf(&dst).i32(pos0 as i32).i32(n_kv as i32).i32(hd as i32).i32(group as i32).i32(bits_k as i32).i32(bits_v as i32),
                    )
                    .unwrap();
            }
            // the scratch of the fused kernels: only the rows of the step are there, the rest is not a number (what is read from it by mistake shows)
            let mut fk = vec![0xFFu8; max_ctx * kv_dim * 2];
            let mut fv = vec![0xFFu8; max_ctx * kv_dim * 2];
            let (lo, hi) = (pos0 * kv_dim * 2, (pos0 + ntok) * kv_dim * 2);
            fk[lo..hi].copy_from_slice(&kt[lo..hi]);
            fv[lo..hi].copy_from_slice(&vt[lo..hi]);
            let (fused_k, fused_v) = (DeviceBuffer::from_bytes(&fk).unwrap(), DeviceBuffer::from_bytes(&fv).unwrap());
            let q: Vec<f32> = (0..ntok * n_head * hd).map(|_| (rng.next_f32() * 2.0 - 1.0) * 3.0).collect();
            let dq = DeviceBuffer::alloc(q.len() * 4).unwrap();
            dq.upload_f32(&q).unwrap();
            let n_part = ntok * n_head * ns_max;
            let blocks = (128u32 << (hybrid as u32), 1u32, 1u32);
            let run = |is_fused: bool| -> (Vec<f32>, Vec<f32>, Vec<f32>) {
                let pm = DeviceBuffer::from_bytes(&vec![0xFFu8; n_part * 4]).unwrap();
                let pl = DeviceBuffer::from_bytes(&vec![0xFFu8; n_part * 4]).unwrap();
                let po = DeviceBuffer::from_bytes(&vec![0xFFu8; n_part * 256 * 4]).unwrap();
                let grid = (n_head as u32, ns_max as u32, ntok as u32);
                if is_fused {
                    let f = if hybrid { &fused_g } else { &fused };
                    f.launch(
                        grid,
                        blocks,
                        0,
                        &stream,
                        &Args::new()
                            .buf(&dq)
                            .buf(&kst)
                            .buf(&ksc)
                            .buf(&vst)
                            .buf(&vsc)
                            .buf(&fused_k)
                            .buf(&fused_v)
                            .buf(&pm)
                            .buf(&pl)
                            .buf(&po)
                            .buf(&dst)
                            .i32(n_head as i32)
                            .i32(n_kv as i32)
                            .i32(hd as i32)
                            .i32(ns_max as i32)
                            .i32(group as i32)
                            .i32(bits_k as i32)
                            .i32(bits_v as i32),
                    )
                    .unwrap();
                } else {
                    let f = if hybrid { &plain_g } else { &plain };
                    f.launch(
                        grid,
                        blocks,
                        0,
                        &stream,
                        &Args::new().buf(&dq).buf(&ref_k).buf(&ref_v).buf(&pm).buf(&pl).buf(&po).buf(&dst).i32(n_head as i32).i32(n_kv as i32).i32(hd as i32).i32(ns_max as i32),
                    )
                    .unwrap();
                }
                stream.synchronize().unwrap();
                (pm.download_f32(n_part).unwrap(), pl.download_f32(n_part).unwrap(), po.download_f32(n_part * 256).unwrap())
            };
            let (a, b) = (run(false), run(true));
            for tt in 0..ntok {
                for hh in 0..n_head {
                    for sp in 0..ns_max {
                        let pidx = (tt * n_head + hh) * ns_max + sp;
                        let sees = sp * 1024 < pos0 + 1 + tt;
                        if !sees {
                            assert!(a.0[pidx] == f32::NEG_INFINITY && b.0[pidx] == f32::NEG_INFINITY && a.1[pidx] == 0.0 && b.1[pidx] == 0.0, "hd {hd} K{bits_k}V{bits_v} pos {pos0} row {tt}: sentinels differ");
                            continue;
                        }
                        let (ma, mb, la, lb) = (a.0[pidx], b.0[pidx], a.1[pidx], b.1[pidx]);
                        let where_ = format!("hd {hd} K{bits_k}V{bits_v} group {group} pos {pos0} row {tt} head {hh} split {sp}");
                        if pos0 == 0 {
                            // nothing in the store: the same fp16 rows, the same arithmetic
                            assert_eq!((ma.to_bits(), la.to_bits()), (mb.to_bits(), lb.to_bits()), "{where_}: maximum or sum differ");
                        } else {
                            assert!((ma - mb).abs() <= 0.05, "{where_}: the maximum score differs ({ma} / {mb})");
                            assert!((la - lb).abs() <= 0.05 * la, "{where_}: the sum differs ({la} / {lb})");
                        }
                        for d in 0..hd {
                            let (oa, ob) = (a.2[pidx * 256 + d], b.2[pidx * 256 + d]);
                            if pos0 == 0 {
                                assert_eq!(oa.to_bits(), ob.to_bits(), "{where_} dim {d}: output differs");
                            } else {
                                assert!((oa - ob).abs() <= 0.05 * la + 1e-3, "{where_} dim {d}: output differs ({oa} / {ob}, sum {la})");
                                worst = worst.max(((oa - ob).abs() / (la + 1e-3)) as f64);
                            }
                        }
                        compared += 1;
                    }
                }
            }
        }
    }
    eprintln!("{compared} partial results agree; the largest output difference is {:.2e} of the sum", worst);
    assert!(compared > 3_000);
}
