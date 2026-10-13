//! The integer tensor-core GEMMs for Q4_K, Q5_K and Q6_K weights (gemm_mmq.cu) against a CPU reference that quantizes the activations the same way (one scale per 32 values, llama.cpp's Q8_1 rule) and multiplies them with the
//! dequantized weights in f64: the quantizer kernel must reproduce the CPU quantization exactly, the GEMM (fp16 / fp32 / add epilogues, shapes that are not multiples of the 128 x 128 tile) must agree to
//! the rounding of its fp16 minimum operands and fp32 sums. Skipped without a CUDA device of compute capability 8 or newer.

use forge_core::half::{f16_to_f32, f32_to_f16};
use forge_cuda::{Args, Device, DeviceBuffer, Function, Module, Stream};
use forge_gguf::dequantize;
use forge_gpu::{repack, QFmt};
use forge_model::sample::Rng;

const SMEM: u32 = 81920;
const SMEM6: u32 = 98304;

struct Ctx {
    module: Module,
    stream: Stream,
}

fn setup() -> Option<Ctx> {
    if Device::count().unwrap_or(0) == 0 {
        eprintln!("skipped: no CUDA device");
        return None;
    }
    let dev = Device::open(0).unwrap();
    if dev.info.cc_major < 8 {
        eprintln!("skipped: the integer GEMMs need compute capability 8");
        return None;
    }
    let (image, _) = forge_kernels::image_for(dev.info.cc_major, dev.info.cc_minor).unwrap();
    Some(Ctx {
        module: Module::load(image).unwrap(),
        stream: Stream::new().unwrap(),
    })
}

fn mmq(c: &Ctx, name: &str) -> Function {
    let f = c.module.function(name).unwrap();
    f.set_max_dynamic_smem(if name.contains("q6k") { SMEM6 } else { SMEM } as i32).unwrap();
    f
}

/// native Q4_K (144 bytes per super-block: d, dmin, 12 scale bytes, 128 bytes of nibbles), Q5_K (176: the same plus 32 bytes of 5th bits before the nibbles) or Q6_K (210: 128 bytes of ql, 64 of qh,
/// 16 int8 scales, d) blocks of n rows of k weights with random content and small valid scales
fn random_blocks(tag: &str, n: usize, k: usize, seed: u64) -> Vec<u8> {
    let mut rng = Rng::new(seed);
    let mut out = Vec::new();
    for _ in 0..n * k / 256 {
        match tag {
            "q6k" => {
                for _ in 0..208 {
                    out.push((rng.next_f32() * 256.0) as u8);
                }
                out.extend_from_slice(&f32_to_f16(0.0004 + 0.0006 * rng.next_f32()).to_le_bytes());
            }
            _ => {
                let d = f32_to_f16(0.002 + 0.006 * rng.next_f32());
                let dmin = f32_to_f16(0.001 + 0.003 * rng.next_f32());
                out.extend_from_slice(&d.to_le_bytes());
                out.extend_from_slice(&dmin.to_le_bytes());
                for _ in 0..if tag == "q5k" { 172 } else { 140 } {
                    out.push((rng.next_f32() * 256.0) as u8);
                }
            }
        }
    }
    out
}

/// the quantization of quant_group / quant_rows_h_kernel: d = amax * (1 / 127), q = round(x / d) (ties away from zero), the stored scale is d rounded to fp16
fn quantize_rows(x: &[f32], rows: usize, k: usize) -> (Vec<i8>, Vec<f32>, Vec<i32>) {
    let g = k / 32;
    let mut q = vec![0i8; rows * k];
    let mut d = vec![0f32; rows * g];
    let mut s = vec![0i32; rows * g];
    for r in 0..rows {
        for j in 0..g {
            let v = &x[r * k + j * 32..r * k + j * 32 + 32];
            let amax = v.iter().fold(0f32, |m, a| m.max(a.abs()));
            let dd = amax * (1.0f32 / 127.0f32);
            let id = if dd != 0.0 { 1.0f32 / dd } else { 0.0 };
            let mut sum = 0i32;
            for (i, a) in v.iter().enumerate() {
                let qi = (a * id).round() as i32;
                q[r * k + j * 32 + i] = qi as i8;
                sum += qi;
            }
            d[r * g + j] = f16_to_f32(f32_to_f16(dd));
            s[r * g + j] = sum;
        }
    }
    (q, d, s)
}

#[test]
fn quantizer_and_q4k_gemm_match_the_cpu_reference() {
    let Some(c) = setup() else { return };
    let quant = c.module.function("quant_rows_h_kernel").unwrap();
    // (m, n, k): m and n are not multiples of the 128 x 128 tile, k of 256 and 1024
    let shapes = [(200usize, 384usize, 768usize), (130, 136, 256), (1024, 512, 1024), (7, 128, 512), (257, 1000, 1280)];
    for &(tag, qf) in &[("q4k", QFmt::Q4_K), ("q5k", QFmt::Q5_K), ("q6k", QFmt::Q6_K)] {
    for &(m, n, k) in &shapes {
        let native = random_blocks(tag, n, k, 11 + n as u64);
        let w = dequantize(qf.dtype(), &native, n * k).unwrap();
        let planes: Vec<DeviceBuffer> = repack(qf, &native)
            .unwrap()
            .iter()
            .map(|p| DeviceBuffer::from_bytes(p).unwrap())
            .collect();
        let mut rng = Rng::new(5 + m as u64);
        // activations: fp16 values, a few outliers (the heavy-tailed channels of real activations)
        let ah: Vec<u16> = (0..m * k)
            .map(|i| {
                let v = (rng.next_f32() * 2.0 - 1.0) * if i % 97 == 0 { 8.0 } else { 1.0 };
                f32_to_f16(v)
            })
            .collect();
        let af: Vec<f32> = ah.iter().map(|&h| f16_to_f32(h)).collect();
        let (qref, dref, sref) = quantize_rows(&af, m, k);
        let rows_pad = (m + 127) / 128 * 128;
        let da = DeviceBuffer::from_bytes(&ah.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>()).unwrap();
        let dq = DeviceBuffer::from_bytes(&vec![0u8; rows_pad * k]).unwrap();
        let dd = DeviceBuffer::from_bytes(&vec![0u8; rows_pad * k / 32 * 4]).unwrap();
        let ds = DeviceBuffer::from_bytes(&vec![0u8; rows_pad * k / 32 * 2]).unwrap();
        quant
            .launch(
                (m as u32, 1, 1),
                (256, 1, 1),
                0,
                &c.stream,
                &Args::new().buf(&da).buf(&dq).buf(&dd).buf(&ds).i32(m as i32).i32(k as i32),
            )
            .unwrap();
        c.stream.synchronize().unwrap();
        let mut qraw = vec![0u8; m * k];
        dq.download_at(0, &mut qraw).unwrap();
        let qgot: Vec<i8> = qraw.iter().map(|&b| b as i8).collect();
        // the scale and the fp16 minimum operand per 32 values come back in the GEMM's tile layout: adT[((tile * K/256 + sb) * 8 + j) * 128 + tok_pos(row % 128)], ashT[((tile * K/256 + sb) * 128 + row % 128) * 8 + j]
        let nb = k / 256;
        let adt = dd.download_f32(rows_pad * k / 32).unwrap();
        let mut sraw = vec![0u8; rows_pad * k / 32 * 2];
        ds.download_at(0, &mut sraw).unwrap();
        let asht: Vec<u16> = sraw.chunks_exact(2).map(|b| u16::from_le_bytes([b[0], b[1]])).collect();
        let tok_pos = |tl: usize| -> usize {
            let (slab, r) = (tl / 64, tl % 64);
            let (mt, rr) = (r / 16, r % 16);
            slab * 64 + (rr % 8) * 8 + mt * 2 + rr / 8
        };
        let mut dgot = vec![0f32; m * k / 32];
        for row in 0..m {
            for gi in 0..k / 32 {
                let (sb, j) = (gi / 8, gi % 8);
                let tile = row / 128;
                dgot[row * (k / 32) + gi] = adt[((tile * nb + sb) * 8 + j) * 128 + tok_pos(row % 128)];
            }
        }
        // the kernels are built with --use_fast_math (the reciprocal of the scale is the fast one, as in the decode kernels' quantizer), so a value that lies almost exactly between two integers may round the other
        // way: at most one step, and in a tiny fraction of the values. The scales (one multiplication) are exact, the minimum operands must be d * the sum of the quantized values the device wrote.
        let differing = qgot.iter().zip(&qref).filter(|(a, b)| a != b).count();
        let too_far = qgot.iter().zip(&qref).filter(|(a, b)| (**a as i32 - **b as i32).abs() > 1).count();
        eprintln!("quantizer {m}x{k}: {differing} of {} values differ from the CPU rule by one step", m * k);
        assert!(too_far == 0 && differing * 1000 <= m * k, "{m}x{k}: quantized values ({differing} differ, {too_far} by more than one)");
        assert_eq!(dgot, dref, "{m}x{k}: scales");
        let _ = &sref;
        for r in 0..m {
            for gi in 0..k / 32 {
                let sum: i32 = qgot[r * k + gi * 32..r * k + gi * 32 + 32].iter().map(|&v| v as i32).sum();
                let (sb, j) = (gi / 8, gi % 8);
                let want = f32_to_f16(dgot[r * (k / 32) + gi] * sum as f32);
                assert_eq!(asht[(((r / 128) * nb + sb) * 128 + r % 128) * 8 + j], want, "{m}x{k}: minimum operand of group {gi} of row {r}");
            }
        }
        // the reference product below uses the arrays the device produced: it tests the GEMM, not the quantizer
        let (qref, dref) = (qgot, dgot);

        // reference product: the dequantized weights times the dequantized (int8 x scale) activations, in f64
        let mut want = vec![0f64; m * n];
        for i in 0..m {
            let arow: Vec<f64> = (0..k).map(|kk| qref[i * k + kk] as f64 * dref[i * (k / 32) + kk / 32] as f64).collect();
            for j in 0..n {
                let wr = &w[j * k..(j + 1) * k];
                want[i * n + j] = wr.iter().zip(&arow).map(|(a, b)| *a as f64 * b).sum();
            }
        }
        let scale = want.iter().fold(0f64, |mx, v| mx.max(v.abs())).max(1.0);
        let grid = (((n + 127) / 128) as u32, ((m + 127) / 128) as u32, 1);
        let args = |out: &DeviceBuffer| {
            let mut a = Args::new().buf(&dq).buf(&dd).buf(&ds);
            for p in &planes {
                a = a.buf(p);
            }
            for _ in planes.len()..4 {
                a = a.ptr(0);
            }
            a.buf(out).i32(m as i32).i32(n as i32).i32(k as i32).i32(n as i32)
        };
        // fp32 store
        let dc = DeviceBuffer::alloc(m * n * 4).unwrap();
        mmq(&c, &format!("mmq_{tag}_f32")).launch(grid, (256, 1, 1), if tag == "q6k" { SMEM6 } else { SMEM }, &c.stream, &args(&dc)).unwrap();
        c.stream.synchronize().unwrap();
        let got = dc.download_f32(m * n).unwrap();
        let worst = got.iter().zip(&want).map(|(a, b)| (*a as f64 - b).abs()).fold(0f64, f64::max);
        eprintln!("mmq_{tag}_f32 {m}x{n}x{k}: max |diff| {worst:.2e} (max |value| {scale:.2}, relative {:.1e})", worst / scale);
        assert!(worst < 3e-4 * scale, "mmq_{tag}_f32 {m}x{n}x{k}: {worst} against {scale}");
        // add into C (starts at 1)
        dc.upload_f32(&vec![1f32; m * n]).unwrap();
        mmq(&c, &format!("mmq_{tag}_add")).launch(grid, (256, 1, 1), if tag == "q6k" { SMEM6 } else { SMEM }, &c.stream, &args(&dc)).unwrap();
        c.stream.synchronize().unwrap();
        let got = dc.download_f32(m * n).unwrap();
        let worst = got.iter().zip(&want).map(|(a, b)| (*a as f64 - 1.0 - b).abs()).fold(0f64, f64::max);
        assert!(worst < 3e-4 * scale, "mmq_{tag}_add {m}x{n}x{k}: {worst}");
        // fp16 store
        let dh = DeviceBuffer::alloc(m * n * 2).unwrap();
        mmq(&c, &format!("mmq_{tag}_f16")).launch(grid, (256, 1, 1), if tag == "q6k" { SMEM6 } else { SMEM }, &c.stream, &args(&dh)).unwrap();
        c.stream.synchronize().unwrap();
        let mut raw = vec![0u8; m * n * 2];
        dh.download(&mut raw).unwrap();
        let worst = raw
            .chunks_exact(2)
            .zip(&want)
            .map(|(b, w)| (f16_to_f32(u16::from_le_bytes([b[0], b[1]])) as f64 - w).abs())
            .fold(0f64, f64::max);
        assert!(worst < 2e-3 * scale, "mmq_{tag}_f16 {m}x{n}x{k}: {worst}");
    }
    }
}
