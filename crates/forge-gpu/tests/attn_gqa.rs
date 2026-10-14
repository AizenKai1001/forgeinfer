//! The GQA-shared split attention kernels (decode.cu `attn_gqa_impl`: one block per key-value head for the four query heads that share it, the keys and values of a position loaded and decoded once for the four,
//! the cross-lane sums made together by a butterfly that exchanges halves of the four partial sums) against the one-head kernels they replace (`attn_split_kernel`, `attn_split_g_kernel`, and for a compressed
//! cache `attn_split_q_kernel`, `attn_split_g_q_kernel`): the partial results (maximum, sum and unnormalised output of every head and split) must be EQUAL BIT FOR BIT, because the merge kernels, the multi-row
//! kernel and the verification of drafted tokens are built on the one-head kernels' arithmetic. Head widths 128 (dense) and 256 (hybrid), 16-bit caches and int8 / int4 keys and values with several scale
//! groups, groups of one to three tokens and positions around the 1,024-position splits. No model is needed: random caches. Skipped without a CUDA device.

use forge_cuda::{Args, Device, DeviceBuffer, Module, Stream};
use forge_model::sample::Rng;

mod common;

fn f16_bytes(rng: &mut Rng, n: usize, amp: f32) -> Vec<u8> {
    use forge_core::half::f32_to_f16;
    (0..n).flat_map(|_| f32_to_f16((rng.next_f32() * 2.0 - 1.0) * amp).to_le_bytes()).collect()
}

#[test]
fn the_gqa_kernels_equal_the_one_head_kernels_bit_for_bit() {
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
    let one = |hybrid: bool, compressed: bool| module.function(match (hybrid, compressed) {
        (false, false) => "attn_split_kernel",
        (true, false) => "attn_split_g_kernel",
        (false, true) => "attn_split_q_kernel",
        (true, true) => "attn_split_g_q_kernel",
    }).unwrap();
    // the GQA kernel of a head width and a group of query heads (padded to a power of two)
    let gqa = |hd: usize, group_heads: usize, compressed: bool| module.function(&format!("attn_gqa_{}_{hd}_{}_kernel", if compressed { "q" } else { "f16" }, group_heads.next_power_of_two())).unwrap();
    let mut compared = 0usize;
    // (heads, kv heads, head width, hybrid kernel, bits of K, bits of V, scale group); bits 16 = the plain fp16 cache
    let configs: [(usize, usize, usize, bool, u32, u32, usize); 17] = [
        (16, 4, 128, false, 16, 16, 32),
        (32, 8, 128, false, 16, 16, 32),
        (16, 4, 256, true, 16, 16, 32),
        (32, 8, 128, false, 8, 8, 32),
        (16, 4, 128, false, 4, 4, 32),
        (8, 2, 128, false, 8, 4, 64),
        (16, 4, 128, false, 4, 8, 128),
        (16, 4, 256, true, 8, 8, 32),
        (16, 4, 256, true, 4, 4, 32),
        (8, 2, 256, true, 8, 4, 64),
        (16, 4, 256, true, 4, 8, 256),
        // other groups of query heads and head widths: 7 heads per key-value head (padded to 8) at head width 64 and 128, 4 at 64, 2, 3 (padded to 4) and 5 (padded to 8) at 128
        (14, 2, 64, false, 16, 16, 32),
        (14, 2, 64, false, 8, 4, 32),
        (32, 8, 64, false, 4, 4, 32),
        (28, 4, 128, false, 8, 8, 32),
        (16, 8, 128, false, 8, 4, 64),
        (24, 8, 128, false, 4, 8, 32),
    ];
    for &(n_head, n_kv, hd, hybrid, bits_k, bits_v, group) in &configs {
        let compressed = bits_k != 16;
        let max_ctx = 2700usize;
        let ns_max = (max_ctx + 1023) / 1024;
        let kv_dim = n_kv * hd;
        let mut rng = Rng::new(97 + hd as u64 + bits_k as u64 * 7 + bits_v as u64 + n_head as u64);
        let kt = f16_bytes(&mut rng, max_ctx * kv_dim, 2.0);
        let vt = f16_bytes(&mut rng, max_ctx * kv_dim, 1.0);
        let d_kt = DeviceBuffer::from_bytes(&kt).unwrap();
        let d_vt = DeviceBuffer::from_bytes(&vt).unwrap();
        let rows = max_ctx * n_kv;
        let (kbytes, vbytes) = (rows * hd * bits_k.min(8) as usize / 8, rows * hd * bits_v.min(8) as usize / 8);
        let sbytes = rows * (hd / group) * 2;
        let (kst, vst) = (DeviceBuffer::alloc(kbytes).unwrap(), DeviceBuffer::alloc(vbytes).unwrap());
        let (ksc, vsc) = (DeviceBuffer::alloc(sbytes).unwrap(), DeviceBuffer::alloc(sbytes).unwrap());
        for &(pos0, ntok) in &[(0usize, 1usize), (1usize, 1usize), (1023, 2), (1024, 1), (1500, 3), (2047, 2), (2048, 1), (2690, 3)] {
            assert!(pos0 + ntok <= max_ctx);
            let mut st = vec![0u8; 16];
            st[4..8].copy_from_slice(&(pos0 as u32).to_le_bytes());
            let dst = DeviceBuffer::from_bytes(&st).unwrap();
            if compressed && pos0 > 0 {
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
            // the cache the kernels read: the plain cache is the truth; a compressed cache has only the rows of the step in the scratch (what is read from it by mistake shows: not a number)
            let (ck, cv) = if compressed {
                let mut fk = vec![0xFFu8; max_ctx * kv_dim * 2];
                let mut fv = vec![0xFFu8; max_ctx * kv_dim * 2];
                let (lo, hi) = (pos0 * kv_dim * 2, (pos0 + ntok) * kv_dim * 2);
                fk[lo..hi].copy_from_slice(&kt[lo..hi]);
                fv[lo..hi].copy_from_slice(&vt[lo..hi]);
                (DeviceBuffer::from_bytes(&fk).unwrap(), DeviceBuffer::from_bytes(&fv).unwrap())
            } else {
                (DeviceBuffer::from_bytes(&kt).unwrap(), DeviceBuffer::from_bytes(&vt).unwrap())
            };
            let q: Vec<f32> = (0..ntok * n_head * hd).map(|_| (rng.next_f32() * 2.0 - 1.0) * 3.0).collect();
            let dq = DeviceBuffer::alloc(q.len() * 4).unwrap();
            dq.upload_f32(&q).unwrap();
            let n_part = ntok * n_head * ns_max;
            let threads = (128u32 << (hybrid as u32), 1u32, 1u32);
            let run = |gqa_kernel: bool| -> (Vec<f32>, Vec<f32>, Vec<f32>) {
                let pm = DeviceBuffer::from_bytes(&vec![0xFFu8; n_part * 4]).unwrap();
                let pl = DeviceBuffer::from_bytes(&vec![0xFFu8; n_part * 4]).unwrap();
                let po = DeviceBuffer::from_bytes(&vec![0xFFu8; n_part * 256 * 4]).unwrap();
                let heads_per_grid = if gqa_kernel { n_kv } else { n_head };
                let grid = (heads_per_grid as u32, ns_max as u32, ntok as u32);
                let f = if gqa_kernel { gqa(hd, n_head / n_kv, compressed) } else { one(hybrid, compressed) };
                let a = Args::new().buf(&dq);
                let a = if compressed { a.buf(&kst).buf(&ksc).buf(&vst).buf(&vsc) } else { a };
                let a = a.buf(&ck).buf(&cv).buf(&pm).buf(&pl).buf(&po).buf(&dst).i32(n_head as i32).i32(n_kv as i32).i32(hd as i32).i32(ns_max as i32);
                let a = if compressed { a.i32(group as i32).i32(bits_k as i32).i32(bits_v as i32) } else { a };
                f.launch(grid, threads, 0, &stream, &a).unwrap();
                stream.synchronize().unwrap();
                (pm.download_f32(n_part).unwrap(), pl.download_f32(n_part).unwrap(), po.download_f32(n_part * 256).unwrap())
            };
            let (a, b) = (run(false), run(true));
            for tt in 0..ntok {
                for hh in 0..n_head {
                    for sp in 0..ns_max {
                        let pidx = (tt * n_head + hh) * ns_max + sp;
                        let where_ = format!("hd {hd} K{bits_k}V{bits_v} group {group} pos {pos0} row {tt} head {hh} split {sp}");
                        assert_eq!((a.0[pidx].to_bits(), a.1[pidx].to_bits()), (b.0[pidx].to_bits(), b.1[pidx].to_bits()), "{where_}: maximum or sum differ ({} {} / {} {})", a.0[pidx], a.1[pidx], b.0[pidx], b.1[pidx]);
                        if sp * 1024 < pos0 + 1 + tt {
                            for d in 0..hd {
                                assert_eq!(a.2[pidx * 256 + d].to_bits(), b.2[pidx * 256 + d].to_bits(), "{where_} dim {d}: output differs ({} / {})", a.2[pidx * 256 + d], b.2[pidx * 256 + d]);
                            }
                            compared += 1;
                        }
                    }
                }
            }
        }
    }
    eprintln!("{compared} partial results equal bit for bit");
    assert!(compared > 3_000);
}
