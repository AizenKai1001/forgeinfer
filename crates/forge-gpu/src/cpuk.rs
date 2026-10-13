//! CPU kernels for the weights that live in host memory (`FORGE_HOST_FFN`): quantized matrix-vector products on the planar layouts of qrepack.rs with the SAME pre-quantized activations as the GPU
//! kernels (decode_q.cu: int8 values, an fp16-rounded scale and the exact integer sum per 32 values), a thread pool, and the whole FFN of a layer (gate / up, SwiGLU, quantization, down) as one job.
//!
//! Why: a decode step reads every weight once, and a byte costs 2.4 ms per GB in VRAM (425 GB/s), 118 ms per GB when the GPU reads it from pinned host memory over PCIe (measured 8.5 GB/s) and about
//! 35-40 ms per GB when the CPU reads it from RAM (the DRAM of the homelab streams 31 GB/s). The weights that do not fit the GPU are therefore multiplied on the CPU.
//!
//! Layout of the planes (see qrepack.rs): Q8_0 p0 int8 [rows][K], p1 fp16 [rows][K/32]; Q4_K p0 qs 128 B per super-block, p1 header 16 B (d, dmin, 12 scale bytes);
//! Q6_K p0 ql 128 B, p1 qh 64 B, p2 sc' 16 B int8 (sc'[4 (2h + u) + q] = sc[8 h + 2 q + u]), p3 fp16 d.
//! Q3_K and Q2_K (host memory only): p0 = the native GGUF blocks (110 and 84 B per 256 values).
//! Numerics: the integer parts are exact; the float parts are the GPU's formulas, summed in a different order.

use crate::qrepack::QFmt;
use forge_core::half::{f16_to_f32, f32_to_f16};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;
use std::time::Duration;

// ---- activations -------------------------------------------------------------------------------------------------------------------------------------------------------------------------

/// Pre-quantized activation vectors (llama.cpp's Q8_1 blocks, planar): `n` columns of K values each; column c starts at `c * K` in xq and `c * K / 32` in xd / xs.
pub struct ActBuf {
    pub k: usize,
    pub xq: Vec<i8>,
    pub xd: Vec<f32>,
    pub xs: Vec<i32>,
}

impl ActBuf {
    pub fn new(k: usize, cols: usize) -> ActBuf {
        ActBuf {
            k,
            xq: vec![0; k * cols],
            xd: vec![0.0; k / 32 * cols],
            xs: vec![0; k / 32 * cols],
        }
    }

    pub fn cols(&self) -> usize {
        self.xq.len() / self.k
    }

    pub fn col(&self, c: usize) -> Act<'_> {
        let g = self.k / 32;
        Act {
            xq: &self.xq[c * self.k..(c + 1) * self.k],
            xd: &self.xd[c * g..(c + 1) * g],
            xs: &self.xs[c * g..(c + 1) * g],
        }
    }

    /// Quantize column c from f32 values (the rule of decode_q.cu quant_group).
    pub fn quantize_col(&mut self, c: usize, x: &[f32]) {
        let g = self.k / 32;
        quantize_groups(
            x,
            &mut self.xq[c * self.k..(c + 1) * self.k],
            &mut self.xd[c * g..(c + 1) * g],
            &mut self.xs[c * g..(c + 1) * g],
        );
    }
}

#[derive(Clone, Copy)]
pub struct Act<'a> {
    pub xq: &'a [i8],
    pub xd: &'a [f32],
    pub xs: &'a [i32],
}

/// d = amax / 127 (fp32), q = round(x / d) with the unrounded d, the stored scale rounded to fp16, the integer sum exact.
pub fn quantize_groups(x: &[f32], xq: &mut [i8], xd: &mut [f32], xs: &mut [i32]) {
    for (g, blk) in x.chunks_exact(32).enumerate() {
        let amax = blk.iter().fold(0f32, |m, v| m.max(v.abs()));
        let d = amax * (1.0 / 127.0);
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        let mut sum = 0i32;
        for (j, v) in blk.iter().enumerate() {
            let q = (v * id).round() as i32;
            xq[g * 32 + j] = q as i8;
            sum += q;
        }
        xd[g] = f16_to_f32(f32_to_f16(d));
        xs[g] = sum;
    }
}

/// Like `quantize_groups` with ONE scale per 256 values (amax / 127, kept in fp32 and repeated for the eight groups of a super-block so that every consumer still finds `xd[group]`); the integer sums stay per 32 values.
pub fn quantize_groups256(x: &[f32], xq: &mut [i8], xd: &mut [f32], xs: &mut [i32]) {
    for (sb, blk) in x.chunks_exact(256).enumerate() {
        let amax = blk.iter().fold(0f32, |m, v| m.max(v.abs()));
        let d = amax * (1.0 / 127.0);
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        for g in 0..8 {
            let mut sum = 0i32;
            for j in 0..32 {
                let q = (blk[g * 32 + j] * id).round() as i32;
                xq[sb * 256 + g * 32 + j] = q as i8;
                sum += q;
            }
            xd[sb * 8 + g] = d;
            xs[sb * 8 + g] = sum;
        }
    }
}

impl ActBuf {
    /// Column c quantized with one scale per 256 values.
    pub fn quantize_col256(&mut self, c: usize, x: &[f32]) {
        let g = self.k / 32;
        quantize_groups256(x, &mut self.xq[c * self.k..(c + 1) * self.k], &mut self.xd[c * g..(c + 1) * g], &mut self.xs[c * g..(c + 1) * g]);
    }

    /// The activation columns `acts` (one scale per 32 values, as the GPU leaves them) requantized with one scale per 256.
    pub fn from_acts256(k: usize, acts: &[Act]) -> ActBuf {
        let mut b = ActBuf::new(k, acts.len());
        let mut x = vec![0f32; k];
        for (c, a) in acts.iter().enumerate() {
            for i in 0..k {
                x[i] = a.xq[i] as f32 * a.xd[i / 32];
            }
            b.quantize_col256(c, &x);
        }
        b
    }
}

// ---- weights -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

/// A matrix in host memory in the planar layout of its format. The pointers stay valid as long as the buffers they point into (page-locked host buffers of the GpuMat) live.
#[derive(Clone, Copy)]
pub struct HostMat {
    pub fmt: QFmt,
    pub rows: usize,
    pub cols: usize,
    pub p: [*const u8; 4],
}

unsafe impl Send for HostMat {}
unsafe impl Sync for HostMat {}

impl HostMat {
    /// Formats with a CPU kernel.
    pub fn supported(f: QFmt) -> bool {
        matches!(f, QFmt::Q8_0 | QFmt::Q4_K | QFmt::Q6_K | QFmt::Q3_K | QFmt::Q2_K)
    }

    /// The matrix can be multiplied on the CPU: a format with a kernel and a row length that is a whole number of its blocks.
    pub fn usable(f: QFmt, cols: usize) -> bool {
        Self::supported(f) && cols % f.block().0 == 0
    }

    /// Bytes of one row in plane `i` of the planar layout (0: the plane is not used by the format).
    fn plane_row_bytes(&self, i: usize) -> usize {
        let nsb = self.cols / 256;
        match (self.fmt, i) {
            (QFmt::Q4_K, 0) => nsb * 128,
            (QFmt::Q4_K, 1) => nsb * 16,
            (QFmt::Q6_K, 0) => nsb * 128,
            (QFmt::Q6_K, 1) => nsb * 64,
            (QFmt::Q6_K, 2) => nsb * 16,
            (QFmt::Q6_K, 3) => nsb * 2,
            (QFmt::Q3_K, 0) => nsb * 110,
            (QFmt::Q2_K, 0) => nsb * 84,
            (QFmt::Q8_0, 0) => self.cols,
            (QFmt::Q8_0, 1) => self.cols / 32 * 2,
            _ => 0,
        }
    }

    /// Bytes of one row over all planes.
    pub fn row_bytes(&self) -> usize {
        (0..4).map(|i| self.plane_row_bytes(i)).sum()
    }

    /// Asks the memory system for rows r0..r1 (into L2 and the shared L3), one cache line at a time.
    pub fn prefetch_rows(&self, r0: usize, r1: usize) {
        let r1 = r1.min(self.rows);
        if r0 >= r1 {
            return;
        }
        #[cfg(target_arch = "x86_64")]
        for i in 0..4 {
            let rb = self.plane_row_bytes(i);
            if rb == 0 {
                continue;
            }
            let (a, b) = (r0 * rb, r1 * rb);
            let mut off = a & !63;
            while off < b {
                unsafe { std::arch::x86_64::_mm_prefetch(self.p[i].add(off) as *const i8, std::arch::x86_64::_MM_HINT_T1) };
                off += 64;
            }
        }
    }
}

#[inline]
fn h2f(p: *const u8, idx: usize) -> f32 {
    unsafe { f16_to_f32(u16::from_le_bytes([*p.add(idx * 2), *p.add(idx * 2 + 1)])) }
}

/// The six-bit scales and minimums of the eight sub-blocks of a Q4_K super-block (the 12 packed bytes at hdr + 4).
#[inline]
fn q4k_scales(hdr: *const u8) -> ([u8; 8], [u8; 8]) {
    let rd = |o: usize| unsafe { u32::from_le_bytes([*hdr.add(o), *hdr.add(o + 1), *hdr.add(o + 2), *hdr.add(o + 3)]) };
    let (u0, u1, u2) = (rd(4), rd(8), rd(12));
    let mut sc = [0u8; 8];
    let mut mn = [0u8; 8];
    for j in 0..4 {
        sc[j] = ((u0 >> (8 * j)) & 63) as u8;
        mn[j] = ((u1 >> (8 * j)) & 63) as u8;
        sc[j + 4] = (((u2 >> (8 * j)) & 0xF) | (((u0 >> (8 * j + 6)) & 3) << 4)) as u8;
        mn[j + 4] = (((u2 >> (8 * j + 4)) & 0xF) | (((u1 >> (8 * j + 6)) & 3) << 4)) as u8;
    }
    (sc, mn)
}

// ---- scalar reference dot products (one row, one column) ---------------------------------------------------------------------------------------------------------------------------------

pub fn dot_row_ref(m: &HostMat, row: usize, a: Act) -> f32 {
    match m.fmt {
        QFmt::Q8_0 => dot_q8_ref(m, row, a),
        QFmt::Q4_K => dot_q4k_ref(m, row, a),
        QFmt::Q6_K => dot_q6k_ref(m, row, a),
        QFmt::Q3_K => dot_q3k_ref(m, row, a),
        QFmt::Q2_K => dot_q2k_ref(m, row, a),
        f => panic!("no CPU kernel for {}", f.name()),
    }
}

fn dot_q8_ref(m: &HostMat, row: usize, a: Act) -> f32 {
    let nb = m.cols / 32;
    let mut acc = 0f32;
    for b in 0..nb {
        let q = unsafe { std::slice::from_raw_parts(m.p[0].add((row * nb + b) * 32) as *const i8, 32) };
        let s: i32 = q.iter().zip(&a.xq[b * 32..b * 32 + 32]).map(|(w, x)| *w as i32 * *x as i32).sum();
        acc += s as f32 * (h2f(m.p[1], row * nb + b) * a.xd[b]);
    }
    acc
}

fn dot_q4k_ref(m: &HostMat, row: usize, a: Act) -> f32 {
    let nsb = m.cols / 256;
    let mut acc = 0f32;
    for sb in 0..nsb {
        let blk = row * nsb + sb;
        let qs = unsafe { std::slice::from_raw_parts(m.p[0].add(blk * 128), 128) };
        let hdr = unsafe { m.p[1].add(blk * 16) };
        let d = h2f(hdr, 0);
        let dmin = h2f(hdr, 1);
        let (sc, mn) = q4k_scales(hdr);
        for j in 0..8 {
            let c = j / 2;
            let hi = j % 2 == 1;
            let ab = sb * 8 + j;
            let mut p = 0i32;
            for l in 0..32 {
                let byte = qs[32 * c + l];
                let q = if hi { byte >> 4 } else { byte & 15 };
                p += q as i32 * a.xq[ab * 32 + l] as i32;
            }
            acc += d * a.xd[ab] * (p * sc[j] as i32) as f32 - dmin * a.xd[ab] * (mn[j] as i32 * a.xs[ab]) as f32;
        }
    }
    acc
}

fn dot_q6k_ref(m: &HostMat, row: usize, a: Act) -> f32 {
    let nsb = m.cols / 256;
    let mut acc = 0f32;
    for sb in 0..nsb {
        let blk = row * nsb + sb;
        let ql = unsafe { std::slice::from_raw_parts(m.p[0].add(blk * 128), 128) };
        let qh = unsafe { std::slice::from_raw_parts(m.p[1].add(blk * 64), 64) };
        let sc = unsafe { std::slice::from_raw_parts(m.p[2].add(blk * 16) as *const i8, 16) };
        let d = h2f(m.p[3], blk);
        let mut sum = 0f32;
        for h in 0..2 {
            let (qlh, qhh) = (&ql[64 * h..64 * h + 64], &qh[32 * h..32 * h + 32]);
            for q in 0..4 {
                for u in 0..2 {
                    let mut t = 0i32;
                    for l in 16 * u..16 * u + 16 {
                        let lo = if q % 2 == 0 { qlh[l] } else { qlh[l + 32] };
                        let nib = if q < 2 { lo & 0xF } else { lo >> 4 };
                        let v = (nib | (((qhh[l] >> (2 * q)) & 3) << 4)) as i32 - 32;
                        t += v * a.xq[sb * 256 + 128 * h + 32 * q + l] as i32;
                    }
                    let s = sc[4 * (2 * h + u) + q] as i32;
                    sum += a.xd[sb * 8 + 4 * h + q] * (t * s) as f32;
                }
            }
        }
        acc += d * sum;
    }
    acc
}


/// The 16 scales of a Q3_K block minus 32 (the 12 packed bytes at `sc`), ggml's unpacking.
#[inline]
fn q3k_scales(sc: *const u8) -> [i32; 16] {
    let b = |o: usize| unsafe { *sc.add(o) } as u32;
    let mut out = [0i32; 16];
    for is in 0..16 {
        let (w, k) = (is >> 2, is & 3);
        let lo = (b((w & 1) * 4 + k) >> if w < 2 { 0 } else { 4 }) & 0xF;
        let hi = (b(8 + k) >> (2 * w)) & 3;
        out[is] = (lo | (hi << 4)) as i32 - 32;
    }
    out
}

fn dot_q3k_ref(m: &HostMat, row: usize, a: Act) -> f32 {
    let nsb = m.cols / 256;
    let mut acc = 0f32;
    for sb in 0..nsb {
        let b = unsafe { std::slice::from_raw_parts(m.p[0].add((row * nsb + sb) * 110), 110) };
        let sc = q3k_scales(b[96..108].as_ptr());
        let d = f16_to_f32(u16::from_le_bytes([b[108], b[109]]));
        let (hm, qs) = (&b[0..32], &b[32..96]);
        for h in 0..2 {
            for j in 0..4 {
                for part in 0..2 {
                    let mut t = 0i32;
                    for l in 0..16 {
                        let low = ((qs[32 * h + 16 * part + l] >> (2 * j)) & 3) as i32;
                        let hb = (hm[16 * part + l] >> (4 * h + j)) & 1;
                        let v = low - if hb != 0 { 0 } else { 4 };
                        t += v * a.xq[sb * 256 + 128 * h + 32 * j + 16 * part + l] as i32;
                    }
                    acc += d * a.xd[sb * 8 + 4 * h + j] * (t * sc[8 * h + 2 * j + part]) as f32;
                }
            }
        }
    }
    acc
}

fn dot_q2k_ref(m: &HostMat, row: usize, a: Act) -> f32 {
    let nsb = m.cols / 256;
    let mut acc = 0f32;
    for sb in 0..nsb {
        let b = unsafe { std::slice::from_raw_parts(m.p[0].add((row * nsb + sb) * 84), 84) };
        let (sc, qs) = (&b[0..16], &b[16..80]);
        let d = f16_to_f32(u16::from_le_bytes([b[80], b[81]]));
        let dmin = f16_to_f32(u16::from_le_bytes([b[82], b[83]]));
        for h in 0..2 {
            for j in 0..4 {
                for part in 0..2 {
                    let (mut t, mut sx) = (0i32, 0i32);
                    for l in 0..16 {
                        let q = ((qs[32 * h + 16 * part + l] >> (2 * j)) & 3) as i32;
                        let x = a.xq[sb * 256 + 128 * h + 32 * j + 16 * part + l] as i32;
                        t += q * x;
                        sx += x;
                    }
                    let s = sc[8 * h + 2 * j + part];
                    let xd = a.xd[sb * 8 + 4 * h + j];
                    acc += xd * (d * ((s & 0xF) as i32 * t) as f32 - dmin * ((s >> 4) as i32 * sx) as f32);
                }
            }
        }
    }
    acc
}

// ---- AVX2 -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
mod avx {
    use super::*;
    use std::arch::x86_64::*;

    #[inline(always)]
    unsafe fn hsum(v: __m256) -> f32 {
        let lo = _mm256_castps256_ps128(v);
        let hi = _mm256_extractf128_ps(v, 1);
        let s = _mm_add_ps(lo, hi);
        let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
        let s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
        _mm_cvtss_f32(s)
    }

    /// i16 vector with the byte `j` of `sc` (u8 x 8, in the low half of an xmm) in every lane: shuffle mask for byte j -> bytes 2i, zero in the odd bytes.
    #[inline(always)]
    unsafe fn bcast_mask(j: u8) -> __m256i {
        let mut m = [0x80u8; 32];
        for i in 0..16 {
            m[2 * i] = j;
        }
        _mm256_loadu_si256(m.as_ptr() as *const __m256i)
    }

    /// Shuffle masks of `bcast_mask` as constants in memory (held in registers, eight of them leave no room for the accumulators of three and four columns).
    static BCAST: [[u8; 32]; 8] = {
        let mut t = [[0x80u8; 32]; 8];
        let mut j = 0;
        while j < 8 {
            let mut i = 0;
            while i < 16 {
                t[j][2 * i] = j as u8;
                i += 1;
            }
            j += 1;
        }
        t
    };

    /// Two fp16 values at `p` (4 bytes) as f32 in lanes 0 and 1 (F16C: exact, no branches; the software conversion cost about a tenth of all instructions of a Q4_K row).
    #[inline(always)]
    unsafe fn h2f_2(p: *const u8) -> __m128 {
        _mm_cvtph_ps(_mm_cvtsi32_si128((p as *const u32).read_unaligned() as i32))
    }

    #[inline(always)]
    unsafe fn h2f_1(p: *const u8) -> f32 {
        _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128((p as *const u16).read_unaligned() as i32)))
    }

    /// `pair_mask(a, b)`: pshufb mask for a register that holds eight i16 in each 128-bit lane: every i16 of the low lane becomes element `a`, every i16 of the high lane element `b`.
    const fn pair_mask(a: usize, b: usize) -> [u8; 32] {
        let mut m = [0u8; 32];
        let mut i = 0;
        while i < 8 {
            m[2 * i] = (2 * a) as u8;
            m[2 * i + 1] = (2 * a + 1) as u8;
            m[16 + 2 * i] = (2 * b) as u8;
            m[16 + 2 * i + 1] = (2 * b + 1) as u8;
            i += 1;
        }
        m
    }
    /// scale pairs (2j, 2j + 1) of a group of 32 values with a scale per 16 (Q3_K, Q2_K)
    static PAIR_ADJ: [[u8; 32]; 4] = [pair_mask(0, 1), pair_mask(2, 3), pair_mask(4, 5), pair_mask(6, 7)];
    /// scale pairs (q, 4 + q) (Q6_K: the two halves of a group of 32 have the scales sc'[4 (2h + u) + q])
    static PAIR_Q6: [[u8; 32]; 4] = [pair_mask(0, 4), pair_mask(1, 5), pair_mask(2, 6), pair_mask(3, 7)];

    /// Both lanes of the result hold the 8 i16 of lane `h` (0 or 1) of `v`.
    #[inline(always)]
    unsafe fn lane_both<const H: i32>(v: __m256i) -> __m256i {
        if H == 0 {
            _mm256_permute2x128_si256(v, v, 0x00)
        } else {
            _mm256_permute2x128_si256(v, v, 0x11)
        }
    }

    /// The scales (bytes 0..8) and minimums (bytes 0..8) of the eight sub-blocks of a Q4_K super-block, unpacked in registers: `hdr` is the 16-byte header (d, dmin, 12 packed bytes), ggml's utmp unpacking as vector operations.
    #[inline(always)]
    unsafe fn q4k_scales_v(hdr: *const u8) -> (__m128i, __m128i) {
        let h = _mm_loadu_si128(hdr as *const __m128i);
        let v = _mm_srli_si128(h, 4); // u0 u1 u2 0
        let m6 = _mm_set1_epi8(0x3f);
        let m4 = _mm_set1_epi8(0x0f);
        let m2 = _mm_set1_epi8(0x03);
        let lo6 = _mm_and_si128(v, m6); // lane 0: scales 0..3, lane 1: mins 0..3
        let top = _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(v, 6), m2), 4); // (u >> 6 & 3) << 4 per lane
        let lo4 = _mm_and_si128(v, m4); // lane 2: u2 & 0xf
        let hi4 = _mm_and_si128(_mm_srli_epi16(v, 4), m4); // lane 2: u2 >> 4 & 0xf
        let sc_hi = _mm_or_si128(_mm_srli_si128(lo4, 8), top); // lane 0: scales 4..7
        let mn_hi = _mm_or_si128(_mm_srli_si128(hi4, 8), _mm_srli_si128(top, 4)); // lane 0: mins 4..7
        (_mm_unpacklo_epi32(lo6, sc_hi), _mm_unpacklo_epi32(_mm_srli_si128(lo6, 4), mn_hi))
    }

    /// The 16 scales (minus 32, as i8) of a Q3_K block from its 12 packed bytes at `sc` (ggml's unpacking as vector operations).
    #[inline(always)]
    unsafe fn q3k_scales_v(sc: *const u8) -> __m128i {
        let lo = _mm_loadl_epi64(sc as *const __m128i);
        let hi = _mm_cvtsi32_si128((sc.add(8) as *const u32).read_unaligned() as i32);
        let v = _mm_unpacklo_epi64(lo, hi); // a0 a1 tmp 0
        let a = _mm_shuffle_epi32(v, 0x44); // a0 a1 a0 a1
        let low = _mm_and_si128(_mm_srlv_epi32(a, _mm_setr_epi32(0, 0, 4, 4)), _mm_set1_epi8(0x0f));
        let t = _mm_shuffle_epi32(v, 0xAA); // tmp x 4
        let high = _mm_slli_epi32(_mm_and_si128(_mm_srlv_epi32(t, _mm_setr_epi32(0, 2, 4, 6)), _mm_set1_epi8(0x03)), 4);
        _mm_sub_epi8(_mm_or_si128(low, high), _mm_set1_epi8(32))
    }

    /// Q4_K: `NC` columns (1..=4) at once, one row. Returns the NC partial sums. The activation scale of a group of 32 is broadcast straight from memory and multiplied into a per-super-block sum, the
    /// weight scale `d` is applied once per super-block (a multiplication and a broadcast fewer per group and column than scaling every group by d * xd).
    #[target_feature(enable = "avx2,fma,f16c")]
    pub unsafe fn dot_q4k<const NC: usize, const S: bool>(m: &HostMat, row: usize, acts: &[Act; NC]) -> [f32; NC] {
        let nsb = m.cols / 256;
        let qs0 = m.p[0].add(row * nsb * 128);
        let hd0 = m.p[1].add(row * nsb * 16);
        let low = _mm256_set1_epi8(0x0F);
        let mut acc = [_mm256_setzero_ps(); NC];
        for sb in 0..nsb {
            let qs = qs0.add(sb * 128);
            let hdr = hd0.add(sb * 16);
            _mm_prefetch(qs.add(512) as *const i8, _MM_HINT_T0);
            _mm_prefetch(qs.add(576) as *const i8, _MM_HINT_T0);
            if sb % 4 == 0 {
                _mm_prefetch(hdr.add(256) as *const i8, _MM_HINT_T0);
            }
            let dd = h2f_2(hdr); // d, dmin
            let (sc8, mn8) = q4k_scales_v(hdr);
            let sc_v = _mm256_broadcastsi128_si256(sc8);
            let mn_i = _mm256_cvtepu8_epi32(mn8);
            let mn_f = _mm256_cvtepi32_ps(mn_i);
            let dminv = _mm256_broadcastss_ps(_mm_shuffle_ps(dd, dd, 1));
            let dv = _mm256_broadcastss_ps(dd);
            let (d, dmin) = (_mm_cvtss_f32(dd), _mm_cvtss_f32(_mm_shuffle_ps(dd, dd, 1)));
            let base = sb * 8;
            let mut sba = [_mm256_setzero_ps(); NC];
            let mut si = [_mm256_setzero_si256(); NC];
            for c in 0..4 {
                let q = _mm256_loadu_si256(qs.add(32 * c) as *const __m256i);
                let lo = _mm256_and_si256(q, low);
                let hi = _mm256_and_si256(_mm256_srli_epi16(q, 4), low);
                let s_lo = _mm256_shuffle_epi8(sc_v, _mm256_loadu_si256(BCAST[2 * c].as_ptr() as *const __m256i));
                let s_hi = _mm256_shuffle_epi8(sc_v, _mm256_loadu_si256(BCAST[2 * c + 1].as_ptr() as *const __m256i));
                for col in 0..NC {
                    let a = &acts[col];
                    let xq = a.xq.as_ptr().add((base + 2 * c) * 32);
                    let xdp = a.xd.as_ptr().add(base + 2 * c);
                    let il = _mm256_madd_epi16(_mm256_maddubs_epi16(lo, _mm256_loadu_si256(xq as *const __m256i)), s_lo);
                    let ih = _mm256_madd_epi16(_mm256_maddubs_epi16(hi, _mm256_loadu_si256(xq.add(32) as *const __m256i)), s_hi);
                    if S {
                        // one activation scale per 256 values: the eight groups add up as integers, one conversion per super-block
                        si[col] = _mm256_add_epi32(si[col], _mm256_add_epi32(il, ih));
                    } else {
                        sba[col] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(il), _mm256_broadcast_ss(&*xdp), sba[col]);
                        sba[col] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(ih), _mm256_broadcast_ss(&*xdp.add(1)), sba[col]);
                    }
                }
            }
            for col in 0..NC {
                let a = &acts[col];
                if S {
                    let d256 = *a.xd.get_unchecked(base);
                    let xsi = _mm256_loadu_si256(a.xs.as_ptr().add(base) as *const __m256i);
                    let minv = _mm256_mullo_epi32(mn_i, xsi);
                    acc[col] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(si[col]), _mm256_set1_ps(d * d256), acc[col]);
                    acc[col] = _mm256_fnmadd_ps(_mm256_cvtepi32_ps(minv), _mm256_set1_ps(dmin * d256), acc[col]);
                } else {
                    let xd = _mm256_loadu_ps(a.xd.as_ptr().add(base));
                    let xs = _mm256_cvtepi32_ps(_mm256_loadu_si256(a.xs.as_ptr().add(base) as *const __m256i));
                    acc[col] = _mm256_fmadd_ps(sba[col], dv, acc[col]);
                    acc[col] = _mm256_fnmadd_ps(dminv, _mm256_mul_ps(mn_f, _mm256_mul_ps(xd, xs)), acc[col]);
                }
            }
        }
        let mut out = [0f32; NC];
        for col in 0..NC {
            out[col] = hsum(acc[col]);
        }
        out
    }

    /// Q8_0: int8 weights times int8 activations: |w| (unsigned) times x with the sign of w.
    #[target_feature(enable = "avx2,fma,f16c")]
    pub unsafe fn dot_q8<const NC: usize>(m: &HostMat, row: usize, acts: &[Act; NC]) -> [f32; NC] {
        let nb = m.cols / 32;
        let q0 = m.p[0].add(row * nb * 32);
        let d0 = m.p[1].add(row * nb * 2);
        let mut acc = [_mm256_setzero_ps(); NC];
        for b in 0..nb {
            _mm_prefetch(q0.add(b * 32 + 512) as *const i8, _MM_HINT_T0);
            let w = _mm256_loadu_si256(q0.add(b * 32) as *const __m256i);
            let aw = _mm256_sign_epi8(w, w);
            // (the software conversion is faster here: one per 32 values, and the F16C one adds a cross-domain hop to every block)
            let d = f16_to_f32(u16::from_le_bytes([*d0.add(b * 2), *d0.add(b * 2 + 1)]));
            for col in 0..NC {
                let a = &acts[col];
                let x = _mm256_loadu_si256(a.xq.as_ptr().add(b * 32) as *const __m256i);
                let sx = _mm256_sign_epi8(x, w);
                let p = _mm256_madd_epi16(_mm256_maddubs_epi16(aw, sx), _mm256_set1_epi16(1));
                acc[col] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p), _mm256_set1_ps(d * *a.xd.get_unchecked(b)), acc[col]);
            }
        }
        let mut out = [0f32; NC];
        for col in 0..NC {
            out[col] = hsum(acc[col]);
        }
        out
    }

    /// Q6_K: unpack 6-bit codes to signed bytes (code - 32), multiply with the sign trick, two scales per 32 values.
    #[target_feature(enable = "avx2,fma,f16c")]
    pub unsafe fn dot_q6k<const NC: usize, const S: bool>(m: &HostMat, row: usize, acts: &[Act; NC]) -> [f32; NC] {
        let nsb = m.cols / 256;
        let ql0 = m.p[0].add(row * nsb * 128);
        let qh0 = m.p[1].add(row * nsb * 64);
        let sc0 = m.p[2].add(row * nsb * 16);
        let d0 = m.p[3].add(row * nsb * 2);
        let low = _mm256_set1_epi8(0x0F);
        let m30 = _mm256_set1_epi8(0x30);
        let c32 = _mm256_set1_epi8(32);
        let ones = _mm256_set1_epi16(1);
        let mut acc = [_mm256_setzero_ps(); NC];
        for sb in 0..nsb {
            let mut sba = [_mm256_setzero_ps(); NC];
            let mut si = [_mm256_setzero_si256(); NC];
            _mm_prefetch(ql0.add(sb * 128 + 512) as *const i8, _MM_HINT_T0);
            _mm_prefetch(ql0.add(sb * 128 + 576) as *const i8, _MM_HINT_T0);
            _mm_prefetch(qh0.add(sb * 64 + 256) as *const i8, _MM_HINT_T0);
            let d = h2f_1(d0.add(sb * 2));
            // the 16 int8 scales as i16, in both 128-bit lanes (per half h, below) for a shuffle per value of q
            let s16 = _mm256_cvtepi8_epi16(_mm_loadu_si128(sc0.add(sb * 16) as *const __m128i));
            for h in 0..2 {
                let ql = ql0.add(sb * 128 + 64 * h);
                let qh = qh0.add(sb * 64 + 32 * h);
                let sh = if h == 0 { lane_both::<0>(s16) } else { lane_both::<1>(s16) };
                let b1 = _mm256_loadu_si256(ql as *const __m256i);
                let b2 = _mm256_loadu_si256(ql.add(32) as *const __m256i);
                let hb = _mm256_loadu_si256(qh as *const __m256i);
                let h0 = _mm256_and_si256(_mm256_slli_epi16(hb, 4), m30);
                let h1 = _mm256_and_si256(_mm256_slli_epi16(hb, 2), m30);
                let h2 = _mm256_and_si256(hb, m30);
                let h3 = _mm256_and_si256(_mm256_srli_epi16(hb, 2), m30);
                // codes 0..63 of the values l + 32 q (q = 0..3)
                let codes = [
                    _mm256_or_si256(_mm256_and_si256(b1, low), h0),
                    _mm256_or_si256(_mm256_and_si256(b2, low), h1),
                    _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(b1, 4), low), h2),
                    _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(b2, 4), low), h3),
                ];
                for q in 0..4 {
                    let w = _mm256_sub_epi8(codes[q], c32);
                    let aw = _mm256_sign_epi8(w, w);
                    // two scales: values 0..15 use sc'[4 (2h + 0) + q], values 16..31 sc'[4 (2h + 1) + q]: elements q and 4 + q of the half
                    let sv = _mm256_shuffle_epi8(sh, _mm256_loadu_si256(PAIR_Q6[q].as_ptr() as *const __m256i));
                    let ab = sb * 8 + 4 * h + q;
                    for col in 0..NC {
                        let a = &acts[col];
                        let x = _mm256_loadu_si256(a.xq.as_ptr().add(ab * 32) as *const __m256i);
                        let sx = _mm256_sign_epi8(x, w);
                        let p = _mm256_madd_epi16(_mm256_maddubs_epi16(aw, sx), sv);
                        let _ = ones;
                        if S {
                            si[col] = _mm256_add_epi32(si[col], p);
                        } else {
                            sba[col] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p), _mm256_broadcast_ss(&*a.xd.as_ptr().add(ab)), sba[col]);
                        }
                    }
                }
            }
            let dv = _mm256_set1_ps(d);
            for col in 0..NC {
                if S {
                    acc[col] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(si[col]), _mm256_set1_ps(d * *acts[col].xd.get_unchecked(sb * 8)), acc[col]);
                } else {
                    acc[col] = _mm256_fmadd_ps(sba[col], dv, acc[col]);
                }
            }
        }
        let mut out = [0f32; NC];
        for col in 0..NC {
            out[col] = hsum(acc[col]);
        }
        out
    }

    /// Q3_K (native blocks): codes (2 bits + the hmask bit) become signed bytes -4..3, multiplied with the sign trick, a scale per 16 values. The 16 scales are unpacked in registers and
    /// handed out by one shuffle per group, the high bit of a code is taken from the mask with one shift and one mask.
    #[target_feature(enable = "avx2,fma,f16c")]
    pub unsafe fn dot_q3k<const NC: usize, const S: bool>(m: &HostMat, row: usize, acts: &[Act; NC]) -> [f32; NC] {
        let nsb = m.cols / 256;
        let base = m.p[0].add(row * nsb * 110);
        let c3 = _mm256_set1_epi8(3);
        let c4 = _mm256_set1_epi8(4);
        let mut acc = [_mm256_setzero_ps(); NC];
        for sb in 0..nsb {
            let b = base.add(sb * 110);
            _mm_prefetch(b.add(660) as *const i8, _MM_HINT_T0);
            _mm_prefetch(b.add(724) as *const i8, _MM_HINT_T0);
            let d = h2f_1(b.add(108));
            let s16 = _mm256_cvtepi8_epi16(q3k_scales_v(b.add(96)));
            let hmv = _mm256_loadu_si256(b as *const __m256i);
            let mut sba = [_mm256_setzero_ps(); NC];
            let mut si = [_mm256_setzero_si256(); NC];
            // bit B of the high-bit mask of every byte, moved to bit 2 (value 4)
            macro_rules! hbit {
                (0) => {
                    _mm256_and_si256(_mm256_slli_epi16(hmv, 2), c4)
                };
                (1) => {
                    _mm256_and_si256(_mm256_slli_epi16(hmv, 1), c4)
                };
                (2) => {
                    _mm256_and_si256(hmv, c4)
                };
                (3) => {
                    _mm256_and_si256(_mm256_srli_epi16(hmv, 1), c4)
                };
                (4) => {
                    _mm256_and_si256(_mm256_srli_epi16(hmv, 2), c4)
                };
                (5) => {
                    _mm256_and_si256(_mm256_srli_epi16(hmv, 3), c4)
                };
                (6) => {
                    _mm256_and_si256(_mm256_srli_epi16(hmv, 4), c4)
                };
                (7) => {
                    _mm256_and_si256(_mm256_srli_epi16(hmv, 5), c4)
                };
            }
            macro_rules! block {
                ($q:ident, $sh:ident, $h:literal, $j:literal, $bit:tt) => {{
                    let low = if $j == 0 { _mm256_and_si256($q, c3) } else { _mm256_and_si256(_mm256_srli_epi16($q, 2 * $j), c3) };
                    let v = _mm256_sub_epi8(_mm256_or_si256(low, hbit!($bit)), c4);
                    let aw = _mm256_sign_epi8(v, v);
                    let sv = _mm256_shuffle_epi8($sh, _mm256_loadu_si256(PAIR_ADJ[$j].as_ptr() as *const __m256i));
                    let ab = sb * 8 + 4 * $h + $j;
                    for col in 0..NC {
                        let a = &acts[col];
                        let x = _mm256_loadu_si256(a.xq.as_ptr().add(ab * 32) as *const __m256i);
                        let p = _mm256_madd_epi16(_mm256_maddubs_epi16(aw, _mm256_sign_epi8(x, v)), sv);
                        if S {
                            si[col] = _mm256_add_epi32(si[col], p);
                        } else {
                            sba[col] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p), _mm256_broadcast_ss(&*a.xd.as_ptr().add(ab)), sba[col]);
                        }
                    }
                }};
            }
            let q0 = _mm256_loadu_si256(b.add(32) as *const __m256i);
            let sh0 = lane_both::<0>(s16);
            block!(q0, sh0, 0, 0, 0);
            block!(q0, sh0, 0, 1, 1);
            block!(q0, sh0, 0, 2, 2);
            block!(q0, sh0, 0, 3, 3);
            let q1 = _mm256_loadu_si256(b.add(64) as *const __m256i);
            let sh1 = lane_both::<1>(s16);
            block!(q1, sh1, 1, 0, 4);
            block!(q1, sh1, 1, 1, 5);
            block!(q1, sh1, 1, 2, 6);
            block!(q1, sh1, 1, 3, 7);
            let dv = _mm256_set1_ps(d);
            for col in 0..NC {
                if S {
                    acc[col] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(si[col]), _mm256_set1_ps(d * *acts[col].xd.get_unchecked(sb * 8)), acc[col]);
                } else {
                    acc[col] = _mm256_fmadd_ps(sba[col], dv, acc[col]);
                }
            }
        }
        let mut out = [0f32; NC];
        for col in 0..NC {
            out[col] = hsum(acc[col]);
        }
        out
    }

    /// Q2_K (native blocks): 2-bit codes times the activations, minus the minimum term with the sums of the activations of each group of 16. The scales and minimums are unpacked in registers.
    #[target_feature(enable = "avx2,fma,f16c")]
    pub unsafe fn dot_q2k<const NC: usize, const S: bool>(m: &HostMat, row: usize, acts: &[Act; NC]) -> [f32; NC] {
        let nsb = m.cols / 256;
        let base = m.p[0].add(row * nsb * 84);
        let c3 = _mm256_set1_epi8(3);
        let ones8 = _mm256_set1_epi8(1);
        let m4 = _mm_set1_epi8(0x0f);
        let mut acc = [_mm256_setzero_ps(); NC];
        let mut macc = [_mm256_setzero_ps(); NC];
        for sb in 0..nsb {
            let b = base.add(sb * 84);
            _mm_prefetch(b.add(504) as *const i8, _MM_HINT_T0);
            let dd = h2f_2(b.add(80));
            let (d, dmin) = (_mm_cvtss_f32(dd), _mm_cvtss_f32(_mm_shuffle_ps(dd, dd, 1)));
            let scb = _mm_loadu_si128(b as *const __m128i); // 16 bytes: the scale (low nibble) and the minimum (high nibble) of each group of 16 values
            let s16 = _mm256_cvtepu8_epi16(_mm_and_si128(scb, m4));
            let m16 = _mm256_cvtepu8_epi16(_mm_and_si128(_mm_srli_epi16(scb, 4), m4));
            let mut sba = [_mm256_setzero_ps(); NC];
            let mut sbm = [_mm256_setzero_ps(); NC];
            let mut si = [_mm256_setzero_si256(); NC];
            let mut sm = [_mm256_setzero_si256(); NC];
            macro_rules! block {
                ($q:ident, $sh:ident, $mh:ident, $h:literal, $j:literal) => {{
                    let low = if $j == 0 { _mm256_and_si256($q, c3) } else { _mm256_and_si256(_mm256_srli_epi16($q, 2 * $j), c3) };
                    let pm = _mm256_loadu_si256(PAIR_ADJ[$j].as_ptr() as *const __m256i);
                    let sv = _mm256_shuffle_epi8($sh, pm);
                    let mv = _mm256_shuffle_epi8($mh, pm);
                    let ab = sb * 8 + 4 * $h + $j;
                    for col in 0..NC {
                        let a = &acts[col];
                        let x = _mm256_loadu_si256(a.xq.as_ptr().add(ab * 32) as *const __m256i);
                        let p = _mm256_madd_epi16(_mm256_maddubs_epi16(low, x), sv);
                        let sx = _mm256_madd_epi16(_mm256_maddubs_epi16(ones8, x), mv);
                        if S {
                            si[col] = _mm256_add_epi32(si[col], p);
                            sm[col] = _mm256_add_epi32(sm[col], sx);
                        } else {
                            let xdv = _mm256_broadcast_ss(&*a.xd.as_ptr().add(ab));
                            sba[col] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p), xdv, sba[col]);
                            sbm[col] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(sx), xdv, sbm[col]);
                        }
                    }
                }};
            }
            let q0 = _mm256_loadu_si256(b.add(16) as *const __m256i);
            let (sh0, mh0) = (lane_both::<0>(s16), lane_both::<0>(m16));
            block!(q0, sh0, mh0, 0, 0);
            block!(q0, sh0, mh0, 0, 1);
            block!(q0, sh0, mh0, 0, 2);
            block!(q0, sh0, mh0, 0, 3);
            let q1 = _mm256_loadu_si256(b.add(48) as *const __m256i);
            let (sh1, mh1) = (lane_both::<1>(s16), lane_both::<1>(m16));
            block!(q1, sh1, mh1, 1, 0);
            block!(q1, sh1, mh1, 1, 1);
            block!(q1, sh1, mh1, 1, 2);
            block!(q1, sh1, mh1, 1, 3);
            let (dv, dmv) = (_mm256_set1_ps(d), _mm256_set1_ps(dmin));
            for col in 0..NC {
                if S {
                    let d256 = *acts[col].xd.get_unchecked(sb * 8);
                    acc[col] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(si[col]), _mm256_set1_ps(d * d256), acc[col]);
                    acc[col] = _mm256_fnmadd_ps(_mm256_cvtepi32_ps(sm[col]), _mm256_set1_ps(dmin * d256), acc[col]);
                } else {
                    acc[col] = _mm256_fmadd_ps(sba[col], dv, acc[col]);
                    macc[col] = _mm256_fmadd_ps(sbm[col], dmv, macc[col]);
                }
            }
        }
        let mut out = [0f32; NC];
        for col in 0..NC {
            out[col] = hsum(acc[col]) - hsum(macc[col]);
        }
        out
    }
}

fn avx2_ok() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") && std::is_x86_feature_detected!("f16c")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// One row against up to four columns: out[c] = row . column c. AVX2 when the CPU has it, the scalar reference otherwise.
pub fn dot_row(m: &HostMat, row: usize, acts: &[Act], out: &mut [f32]) {
    dot_row_any(m, row, acts, out, false)
}

/// The same for activations with one scale per 256 values (`quantize_groups256`: the scale is repeated for the eight groups of a super-block): the K-quant kernels add the groups up as integers.
pub fn dot_row_s256(m: &HostMat, row: usize, acts: &[Act], out: &mut [f32]) {
    dot_row_any(m, row, acts, out, true)
}

fn dot_row_any(m: &HostMat, row: usize, acts: &[Act], out: &mut [f32], s256: bool) {
    debug_assert!(acts.len() == out.len() && !acts.is_empty() && acts.len() <= 4);
    #[cfg(target_arch = "x86_64")]
    if avx2_ok() {
        macro_rules! go {
            ($n:literal) => {{
                let a: [Act; $n] = std::array::from_fn(|i| acts[i]);
                let r = unsafe {
                    match (m.fmt, s256) {
                        (QFmt::Q4_K, false) => avx::dot_q4k::<$n, false>(m, row, &a),
                        (QFmt::Q4_K, true) => avx::dot_q4k::<$n, true>(m, row, &a),
                        (QFmt::Q6_K, false) => avx::dot_q6k::<$n, false>(m, row, &a),
                        (QFmt::Q6_K, true) => avx::dot_q6k::<$n, true>(m, row, &a),
                        (QFmt::Q8_0, _) => avx::dot_q8::<$n>(m, row, &a),
                        (QFmt::Q3_K, false) => avx::dot_q3k::<$n, false>(m, row, &a),
                        (QFmt::Q3_K, true) => avx::dot_q3k::<$n, true>(m, row, &a),
                        (QFmt::Q2_K, false) => avx::dot_q2k::<$n, false>(m, row, &a),
                        (QFmt::Q2_K, true) => avx::dot_q2k::<$n, true>(m, row, &a),
                        (f, _) => panic!("no CPU kernel for {}", f.name()),
                    }
                };
                out.copy_from_slice(&r);
                return;
            }};
        }
        match acts.len() {
            1 => go!(1),
            2 => go!(2),
            3 => go!(3),
            _ => go!(4),
        }
    }
    for (o, a) in out.iter_mut().zip(acts) {
        *o = dot_row_ref(m, row, *a);
    }
}

// ---- thread pool -------------------------------------------------------------------------------------------------------------------------------------------------------------------------

struct Shared {
    n: usize,
    gen: AtomicU64,
    done: AtomicUsize,
    stop: AtomicBool,
    job: AtomicUsize,
    /// `*const PostJob` for the generation `post_gen` (0: none): work the workers do right after that job, before they go back to waiting
    post: AtomicUsize,
    post_gen: AtomicU64,
    lock: Mutex<()>,
    cv: Condvar,
}

/// Work for the pool's worker threads right after a job, while the thread that called `run` is already elsewhere (the GPU's turn in a decode step): `f(data, thread, threads)`. The FFN of a layer uses it to
/// pull the first megabytes of the next layer's weights into the cache while the GPU computes attention.
pub struct PostJob {
    f: unsafe fn(*const (), usize, usize),
    data: *const (),
}

unsafe impl Send for PostJob {}
unsafe impl Sync for PostJob {}

unsafe fn post_nothing(_: *const (), _: usize, _: usize) {}

/// Bytes of the next layer's gate / up rows to prefetch (FORGE_CPU_PREFETCH_MB, default 16: decode of the 27B 9.34 -> 9.54 tok/s; 0 switches it off).
fn prefetch_bytes() -> usize {
    static B: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *B.get_or_init(|| std::env::var("FORGE_CPU_PREFETCH_MB").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(16) << 20)
}

unsafe fn prefetch_ffn(data: *const (), t: usize, n: usize) {
    let f = &*(data as *const CpuFfn);
    let per_row = f.gate.row_bytes() + f.up.row_bytes();
    if per_row == 0 {
        return;
    }
    // the rows the next call takes first (chunks of 64 rows, in order)
    let rows = (prefetch_bytes() / per_row).min(f.gate.rows) / 64 * 64;
    let (a, b) = (rows * t / n, rows * (t + 1) / n);
    f.gate.prefetch_rows(a, b);
    f.up.prefetch_rows(a, b);
}

impl PostJob {
    pub fn none() -> PostJob {
        PostJob { f: post_nothing, data: std::ptr::null() }
    }

    /// Prefetch the first rows of `next` (which must outlive every call that uses this job).
    pub fn prefetch_of(next: *const CpuFfn) -> PostJob {
        if prefetch_bytes() == 0 {
            return PostJob::none();
        }
        PostJob { f: prefetch_ffn, data: next as *const () }
    }

    pub fn is_set(&self) -> bool {
        !self.data.is_null()
    }
}

type Job<'a> = &'a (dyn Fn(usize, usize) + Sync);

fn worker(sh: Arc<Shared>, idx: usize) {
    let mut seen = 0u64;
    loop {
        let mut spins = 0u32;
        loop {
            let g = sh.gen.load(Ordering::Acquire);
            if g != seen {
                seen = g;
                break;
            }
            if sh.stop.load(Ordering::Relaxed) {
                return;
            }
            spins += 1;
            if spins < 20000 {
                std::hint::spin_loop();
            } else if spins < 20400 {
                thread::yield_now();
            } else {
                let guard = sh.lock.lock().unwrap();
                if sh.gen.load(Ordering::Acquire) == seen && !sh.stop.load(Ordering::Relaxed) {
                    let _ = sh.cv.wait_timeout(guard, Duration::from_millis(20)).unwrap();
                }
            }
        }
        if sh.stop.load(Ordering::Acquire) {
            return;
        }
        let job: Job = unsafe { *(sh.job.load(Ordering::Acquire) as *const Job) };
        job(idx, sh.n);
        sh.done.fetch_add(1, Ordering::AcqRel);
        if sh.post_gen.load(Ordering::Acquire) == seen {
            let p = sh.post.load(Ordering::Acquire);
            if p != 0 {
                let pj = unsafe { &*(p as *const PostJob) };
                unsafe { (pj.f)(pj.data, idx, sh.n) };
            }
        }
    }
}

/// A fork-join pool: the caller is thread 0 and `n - 1` workers join it. Workers spin briefly after a job (consecutive layers follow each other within a millisecond), then yield, then sleep.
pub struct Pool {
    sh: Arc<Shared>,
    handles: Vec<thread::JoinHandle<()>>,
    run_lock: Mutex<()>,
    /// `run_chunks` hands out small chunks to whichever thread is free (FORGE_CPU_DYN=0: one fixed slice per thread)
    dynamic: bool,
}

impl Pool {
    pub fn new(n: usize) -> Pool {
        let n = n.max(1);
        let sh = Arc::new(Shared {
            n,
            gen: AtomicU64::new(0),
            done: AtomicUsize::new(0),
            stop: AtomicBool::new(false),
            job: AtomicUsize::new(0),
            post: AtomicUsize::new(0),
            post_gen: AtomicU64::new(0),
            lock: Mutex::new(()),
            cv: Condvar::new(),
        });
        let handles = (1..n)
            .map(|i| {
                let s = sh.clone();
                thread::spawn(move || worker(s, i))
            })
            .collect();
        Pool {
            sh,
            handles,
            run_lock: Mutex::new(()),
            dynamic: std::env::var("FORGE_CPU_DYN").map(|v| v != "0").unwrap_or(true),
        }
    }

    pub fn threads(&self) -> usize {
        self.sh.n
    }

    /// Run `f(start, end)` over `0..total` split into chunks of `chunk` items. Every thread takes the next unclaimed chunk when it is free, so a thread that is slow or preempted (the machine also runs the desktop,
    /// Docker, the dashboards) does not hold the whole call back for its share. Every item is covered exactly once.
    pub fn run_chunks(&self, total: usize, chunk: usize, f: &(dyn Fn(usize, usize) + Sync)) {
        self.run_chunks_post(total, chunk, f, None)
    }

    /// `run_chunks` followed by `post` on the worker threads (see `run_post`).
    pub fn run_chunks_post(&self, total: usize, chunk: usize, f: &(dyn Fn(usize, usize) + Sync), post: Option<*const PostJob>) {
        if !self.dynamic || self.sh.n == 1 {
            self.run_post(
                &|t, n| {
                    let (a, b) = (total * t / n, total * (t + 1) / n);
                    if a < b {
                        f(a, b);
                    }
                },
                post,
            );
            return;
        }
        let chunk = chunk.max(1);
        let next = AtomicUsize::new(0);
        self.run_post(
            &|_, _| loop {
                let s = next.fetch_add(chunk, Ordering::Relaxed);
                if s >= total {
                    break;
                }
                f(s, (s + chunk).min(total));
            },
            post,
        );
    }

    /// Run `f(thread_index, n_threads)` on every thread and return when all are done.
    pub fn run(&self, f: &(dyn Fn(usize, usize) + Sync)) {
        self.run_post(f, None)
    }

    /// `run`, and every worker thread then does `post` (if given) before it goes back to waiting: the caller does not wait for it.
    pub fn run_post(&self, f: &(dyn Fn(usize, usize) + Sync), post: Option<*const PostJob>) {
        let _g = self.run_lock.lock().unwrap();
        let n = self.sh.n;
        if n == 1 {
            f(0, 1);
            return;
        }
        let job: Job = f;
        self.sh.job.store(&job as *const Job as usize, Ordering::Release);
        self.sh.done.store(0, Ordering::Release);
        if let Some(p) = post {
            self.sh.post.store(p as usize, Ordering::Release);
            let g = self.sh.gen.load(Ordering::Acquire) + 1;
            self.sh.post_gen.store(g, Ordering::Release);
        }
        self.sh.gen.fetch_add(1, Ordering::AcqRel);
        {
            let _l = self.sh.lock.lock().unwrap();
            self.sh.cv.notify_all();
        }
        f(0, n);
        while self.sh.done.load(Ordering::Acquire) < n - 1 {
            std::hint::spin_loop();
        }
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        self.sh.stop.store(true, Ordering::Release);
        self.sh.gen.fetch_add(1, Ordering::AcqRel);
        {
            let _l = self.sh.lock.lock().unwrap();
            self.sh.cv.notify_all();
        }
        for h in self.handles.drain(..) {
            let _ = h.join();
        }
    }
}

// ---- the FFN of one layer ---------------------------------------------------------------------------------------------------------------------------------------------------------------

/// One layer's FFN on the CPU: out[c] = down . (silu(gate . x[c]) * (up . x[c])), c < cols (1..=4), with the SwiGLU output quantized to the same int8 blocks the GPU path uses.
pub struct CpuFfn {
    pub gate: HostMat,
    pub up: HostMat,
    pub down: HostMat,
    pub e: usize,
    pub f: usize,
    /// activations with one scale per 256 values (FORGE_CPU_S256=1): the K-quant kernels then add the eight groups of a super-block as integers (about 40 percent fewer vector operations per weight and column)
    pub s256: bool,
}

/// FORGE_CPU_S256=1 switches the CPU FFN to activations with one scale per 256 values.
pub fn s256_default() -> bool {
    std::env::var("FORGE_CPU_S256").map(|v| v == "1").unwrap_or(false)
}

/// Scratch of a CPU FFN job (the quantized SwiGLU output of every column).
pub struct FfnScratch {
    act: ActBuf,
}

impl FfnScratch {
    pub fn new(f: usize, cols: usize) -> FfnScratch {
        FfnScratch {
            act: ActBuf::new(f, cols),
        }
    }
}

/// Raw pointer wrapper for disjoint writes from the pool's threads.
#[derive(Clone, Copy)]
struct SyncPtr<T>(*mut T);
unsafe impl<T> Send for SyncPtr<T> {}
unsafe impl<T> Sync for SyncPtr<T> {}
impl<T> SyncPtr<T> {
    /// (a method, so that a closure captures the whole wrapper and not its raw-pointer field)
    fn ptr(self) -> *mut T {
        self.0
    }
}

impl CpuFfn {
    /// `xs`: the normed, quantized input columns (K = e each); `out`: `cols * e` floats, overwritten.
    pub fn run(&self, pool: &Pool, xs: &[Act], scratch: &mut FfnScratch, out: &mut [f32]) {
        self.run_post(pool, xs, scratch, out, None)
    }

    /// `run`; the pool's workers do `post` (when it is set) right after the last phase, while the caller returns to the GPU.
    pub fn run_post(&self, pool: &Pool, xs: &[Act], scratch: &mut FfnScratch, out: &mut [f32], post: Option<&PostJob>) {
        let post = post.filter(|p| p.is_set()).map(|p| p as *const PostJob);
        let cols = xs.len();
        assert!(cols >= 1 && cols <= 4 && out.len() == cols * self.e);
        if self.s256 && self.e % 256 == 0 && self.f % 256 == 0 {
            return self.run_s256(pool, xs, scratch, out);
        }
        let (e, f) = (self.e, self.f);
        if scratch.act.k != f || scratch.act.cols() < cols {
            scratch.act = ActBuf::new(f, cols);
        }
        let groups = f / 32;
        let (aq, ad, asum) = (
            SyncPtr(scratch.act.xq.as_mut_ptr()),
            SyncPtr(scratch.act.xd.as_mut_ptr()),
            SyncPtr(scratch.act.xs.as_mut_ptr()),
        );
        // phase 1: gate and up of 32-row groups, SwiGLU, quantization of the group (each thread owns whole groups)
        pool.run_chunks(groups, 2, &|g0, g1| {
            let mut gv = [[0f32; 32]; 4];
            let mut uv = [[0f32; 32]; 4];
            for g in g0..g1 {
                for r in 0..32 {
                    let mut o = [0f32; 4];
                    dot_row(&self.gate, g * 32 + r, &xs, &mut o[..cols]);
                    for c in 0..cols {
                        gv[c][r] = o[c];
                    }
                    dot_row(&self.up, g * 32 + r, &xs, &mut o[..cols]);
                    for c in 0..cols {
                        uv[c][r] = o[c];
                    }
                }
                for c in 0..cols {
                    let mut a = [0f32; 32];
                    for r in 0..32 {
                        let gx = gv[c][r];
                        a[r] = (gx / (1.0 + (-gx).exp())) * uv[c][r];
                    }
                    let (q, d, s) = unsafe {
                        (
                            std::slice::from_raw_parts_mut(aq.ptr().add(c * f + g * 32), 32),
                            std::slice::from_raw_parts_mut(ad.ptr().add(c * groups + g), 1),
                            std::slice::from_raw_parts_mut(asum.ptr().add(c * groups + g), 1),
                        )
                    };
                    quantize_groups(&a, q, d, s);
                }
            }
        });
        // phase 2: the down projection, rows split across the threads
        let acts: Vec<Act> = (0..cols).map(|c| scratch.act.col(c)).collect();
        let optr = SyncPtr(out.as_mut_ptr());
        pool.run_chunks_post(
            e,
            16,
            &|r0, r1| {
                let mut o = [0f32; 4];
                for r in r0..r1 {
                    dot_row(&self.down, r, &acts, &mut o[..cols]);
                    for c in 0..cols {
                        unsafe { *optr.ptr().add(c * e + r) = o[c] };
                    }
                }
            },
            post,
        );
    }
}

impl CpuFfn {
    /// `run` with activations of one scale per 256 values: the input columns are requantized once, gate / up are computed for groups of 256 rows (each thread owns whole super-blocks of the SwiGLU output, which
    /// is quantized per 256 too), and the down projection reads them with the integer kernels.
    fn run_s256(&self, pool: &Pool, xs: &[Act], scratch: &mut FfnScratch, out: &mut [f32]) {
        let cols = xs.len();
        let (e, f) = (self.e, self.f);
        let xin = ActBuf::from_acts256(e, xs);
        let xv: Vec<Act> = (0..cols).map(|c| xin.col(c)).collect();
        if scratch.act.k != f || scratch.act.cols() < cols {
            scratch.act = ActBuf::new(f, cols);
        }
        let groups = f / 32;
        let units = f / 256;
        let (aq, ad, asum) = (
            SyncPtr(scratch.act.xq.as_mut_ptr()),
            SyncPtr(scratch.act.xd.as_mut_ptr()),
            SyncPtr(scratch.act.xs.as_mut_ptr()),
        );
        pool.run_chunks(units, 1, &|u0, u1| {
            let mut gv = [[0f32; 256]; 4];
            let mut uv = [[0f32; 256]; 4];
            for u in u0..u1 {
                for r in 0..256 {
                    let mut o = [0f32; 4];
                    dot_row_s256(&self.gate, u * 256 + r, &xv, &mut o[..cols]);
                    for c in 0..cols {
                        gv[c][r] = o[c];
                    }
                    dot_row_s256(&self.up, u * 256 + r, &xv, &mut o[..cols]);
                    for c in 0..cols {
                        uv[c][r] = o[c];
                    }
                }
                for c in 0..cols {
                    let mut a = [0f32; 256];
                    for r in 0..256 {
                        let gx = gv[c][r];
                        a[r] = (gx / (1.0 + (-gx).exp())) * uv[c][r];
                    }
                    let (q, d, sm) = unsafe {
                        (
                            std::slice::from_raw_parts_mut(aq.ptr().add(c * f + u * 256), 256),
                            std::slice::from_raw_parts_mut(ad.ptr().add(c * groups + u * 8), 8),
                            std::slice::from_raw_parts_mut(asum.ptr().add(c * groups + u * 8), 8),
                        )
                    };
                    quantize_groups256(&a, q, d, sm);
                }
            }
        });
        let acts: Vec<Act> = (0..cols).map(|c| scratch.act.col(c)).collect();
        let optr = SyncPtr(out.as_mut_ptr());
        pool.run_chunks(e, 16, &|r0, r1| {
            let mut o = [0f32; 4];
            for r in r0..r1 {
                dot_row_s256(&self.down, r, &acts, &mut o[..cols]);
                for c in 0..cols {
                    unsafe { *optr.ptr().add(c * e + r) = o[c] };
                }
            }
        });
    }
}

// ---- tests -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qrepack::repack;
    use forge_gguf::{dequantize, DType};

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn f(&mut self) -> f32 {
            (self.next() >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
        }
    }

    /// Random native blocks of a format: random bytes with valid fp16 scales.
    fn random_native(fmt: QFmt, rows: usize, cols: usize, rng: &mut Rng) -> Vec<u8> {
        let (bv, bb) = fmt.block();
        let nb = rows * cols / bv;
        let mut v = vec![0u8; nb * bb];
        for b in 0..nb {
            let blk = &mut v[b * bb..(b + 1) * bb];
            for x in blk.iter_mut() {
                *x = rng.next() as u8;
            }
            let d = (0.002 + 0.01 * (rng.next() % 100) as f32 / 100.0) * if rng.next() % 2 == 0 { 1.0 } else { -1.0 };
            let h = f32_to_f16(d).to_le_bytes();
            match fmt {
                QFmt::Q8_0 => blk[0..2].copy_from_slice(&h),
                QFmt::Q4_K => {
                    blk[0..2].copy_from_slice(&h);
                    blk[2..4].copy_from_slice(&f32_to_f16(d.abs() * 0.5).to_le_bytes());
                }
                QFmt::Q6_K => blk[208..210].copy_from_slice(&h),
                QFmt::Q3_K => blk[108..110].copy_from_slice(&h),
                QFmt::Q2_K => {
                    blk[80..82].copy_from_slice(&h);
                    blk[82..84].copy_from_slice(&f32_to_f16(d.abs() * 0.5).to_le_bytes());
                }
                _ => unreachable!(),
            }
        }
        v
    }

    fn host_mat(fmt: QFmt, rows: usize, cols: usize, native: &[u8]) -> (HostMat, Vec<Vec<u8>>) {
        let planes = repack(fmt, native).unwrap();
        let mut p = [std::ptr::null(); 4];
        for (i, pl) in planes.iter().enumerate() {
            p[i] = pl.as_ptr();
        }
        (HostMat { fmt, rows, cols, p }, planes)
    }

    fn rand_act(k: usize, cols: usize, rng: &mut Rng) -> ActBuf {
        let mut a = ActBuf::new(k, cols);
        for c in 0..cols {
            let x: Vec<f32> = (0..k).map(|i| rng.f() * if i % 97 == 0 { 8.0 } else { 1.0 }).collect();
            a.quantize_col(c, &x);
        }
        a
    }

    /// The ground truth: the golden-tested dequantizer of the native layout against the dequantized activation.
    fn truth(dt: DType, native: &[u8], rows: usize, cols: usize, a: Act, row: usize) -> f64 {
        let w = dequantize(dt, native, rows * cols).unwrap();
        let mut s = 0f64;
        for i in 0..cols {
            s += w[row * cols + i] as f64 * (a.xq[i] as f64 * a.xd[i / 32] as f64);
        }
        s
    }

    fn check(fmt: QFmt, dt: DType) {
        let mut rng = Rng(0x9E3779B97F4A7C15);
        let (rows, cols) = (24, 1024);
        let native = random_native(fmt, rows, cols, &mut rng);
        let (m, _planes) = host_mat(fmt, rows, cols, &native);
        let acts = rand_act(cols, 4, &mut rng);
        for row in [0usize, 1, 7, 23] {
            let ac: Vec<Act> = (0..4).map(|c| acts.col(c)).collect();
            for n in 1..=4 {
                let mut out = vec![0f32; n];
                dot_row(&m, row, &ac[..n], &mut out);
                for c in 0..n {
                    let t = truth(dt, &native, rows, cols, ac[c], row);
                    let r = dot_row_ref(&m, row, ac[c]) as f64;
                    let scale = t.abs().max(1e-3);
                    assert!((r - t).abs() / scale < 2e-4, "{} scalar reference row {row} col {c}: {r} vs truth {t}", fmt.name());
                    assert!(
                        (out[c] as f64 - t).abs() / scale < 2e-4,
                        "{} avx2 row {row} col {c} of {n}: {} vs truth {t}",
                        fmt.name(),
                        out[c]
                    );
                }
            }
        }
    }

    #[test]
    fn q4k_matches_truth() {
        check(QFmt::Q4_K, DType::Q4_K);
    }

    #[test]
    fn q6k_matches_truth() {
        check(QFmt::Q6_K, DType::Q6_K);
    }

    #[test]
    fn q3k_matches_truth() {
        check(QFmt::Q3_K, DType::Q3_K);
    }

    #[test]
    fn q2k_matches_truth() {
        check(QFmt::Q2_K, DType::Q2_K);
    }

    #[test]
    fn q8_matches_truth() {
        check(QFmt::Q8_0, DType::Q8_0);
    }


    /// The ground truth for activations with one scale per 256 values: the same golden dequantizer against the dequantized activation (the scale is repeated per 32 values).
    fn check256(fmt: QFmt, dt: DType) {
        let mut rng = Rng(0x9E3779B97F4A7C15);
        let (rows, cols) = (24, 1024);
        let native = random_native(fmt, rows, cols, &mut rng);
        let (m, _planes) = host_mat(fmt, rows, cols, &native);
        let mut acts = ActBuf::new(cols, 4);
        for c in 0..4 {
            let x: Vec<f32> = (0..cols).map(|i| rng.f() * if i % 97 == 0 { 8.0 } else { 1.0 }).collect();
            acts.quantize_col256(c, &x);
        }
        for row in [0usize, 1, 7, 23] {
            let ac: Vec<Act> = (0..4).map(|c| acts.col(c)).collect();
            for n in 1..=4 {
                let mut out = vec![0f32; n];
                dot_row_s256(&m, row, &ac[..n], &mut out);
                for c in 0..n {
                    let t = truth(dt, &native, rows, cols, ac[c], row);
                    let scale = t.abs().max(1e-3);
                    assert!((out[c] as f64 - t).abs() / scale < 2e-4, "{} s256 avx2 row {row} col {c} of {n}: {} vs truth {t}", fmt.name(), out[c]);
                }
            }
        }
    }

    #[test]
    fn s256_kernels_match_truth() {
        check256(QFmt::Q4_K, DType::Q4_K);
        check256(QFmt::Q6_K, DType::Q6_K);
        check256(QFmt::Q3_K, DType::Q3_K);
        check256(QFmt::Q2_K, DType::Q2_K);
        check256(QFmt::Q8_0, DType::Q8_0);
    }

    #[test]
    fn ffn_s256_matches_a_float_reference() {
        let mut rng = Rng(0x1234567);
        let (e, f) = (512usize, 1536usize);
        let (ng, nu, nd) = (random_native(QFmt::Q4_K, f, e, &mut rng), random_native(QFmt::Q4_K, f, e, &mut rng), random_native(QFmt::Q6_K, e, f, &mut rng));
        let (g, _pg) = host_mat(QFmt::Q4_K, f, e, &ng);
        let (u, _pu) = host_mat(QFmt::Q4_K, f, e, &nu);
        let (d, _pd) = host_mat(QFmt::Q6_K, e, f, &nd);
        let (ffn, ffn32) = (CpuFfn { gate: g, up: u, down: d, e, f, s256: true }, CpuFfn { gate: g, up: u, down: d, e, f, s256: false });
        let pool = Pool::new(3);
        let x = rand_act(e, 2, &mut rng);
        let xv: Vec<Act> = (0..2).map(|c| x.col(c)).collect();
        let mut scratch = FfnScratch::new(f, 2);
        let (mut out, mut out32) = (vec![0f32; 2 * e], vec![0f32; 2 * e]);
        ffn.run(&pool, &xv, &mut scratch, &mut out);
        ffn32.run(&pool, &xv, &mut scratch, &mut out32);
        // against the per-32 path (the two differ by the activation quantization only)
        let norm = out32.iter().fold(0f32, |m, v| m.max(v.abs()));
        let worst = out.iter().zip(&out32).fold(0f32, |m, (a, b)| m.max((a - b).abs()));
        assert!(worst / norm < 0.05, "ffn s256: worst difference {worst} against a maximum of {norm}");
    }

    #[test]
    fn pool_runs_every_thread_once() {
        let pool = Pool::new(4);
        for _ in 0..50 {
            let hits = AtomicUsize::new(0);
            pool.run(&|t, n| {
                assert!(t < n);
                hits.fetch_add(1 << (4 * t), Ordering::SeqCst);
            });
            assert_eq!(hits.load(Ordering::SeqCst), 0x1111);
        }
    }

    #[test]
    fn run_chunks_covers_every_item_once() {
        for threads in [1usize, 2, 5, 7] {
            let pool = Pool::new(threads);
            for (total, chunk) in [(0usize, 3usize), (1, 3), (17, 1), (64, 2), (100, 16), (5120, 16), (17408 / 32, 2)] {
                let seen: Vec<AtomicUsize> = (0..total).map(|_| AtomicUsize::new(0)).collect();
                pool.run_chunks(total, chunk, &|a, b| {
                    assert!(a < b && b <= total);
                    for i in a..b {
                        seen[i].fetch_add(1, Ordering::SeqCst);
                    }
                });
                assert!(seen.iter().all(|s| s.load(Ordering::SeqCst) == 1), "threads {threads} total {total} chunk {chunk}");
            }
        }
    }

    #[test]
    fn ffn_matches_a_float_reference() {
        let mut rng = Rng(0x1234567);
        let (e, f) = (512usize, 1536usize);
        let (ng, nu, nd) = (random_native(QFmt::Q4_K, f, e, &mut rng), random_native(QFmt::Q4_K, f, e, &mut rng), random_native(QFmt::Q6_K, e, f, &mut rng));
        let (g, _pg) = host_mat(QFmt::Q4_K, f, e, &ng);
        let (u, _pu) = host_mat(QFmt::Q4_K, f, e, &nu);
        let (d, _pd) = host_mat(QFmt::Q6_K, e, f, &nd);
        let ffn = CpuFfn { gate: g, up: u, down: d, e, f, s256: false };
        let pool = Pool::new(3);
        let x = rand_act(e, 2, &mut rng);
        let mut scratch = FfnScratch::new(f, 2);
        let mut out = vec![0f32; 2 * e];
        let xv: Vec<Act> = (0..2).map(|c| x.col(c)).collect();
        ffn.run(&pool, &xv, &mut scratch, &mut out);
        // float reference
        let (wg, wu, wd) = (
            dequantize(DType::Q4_K, &ng, f * e).unwrap(),
            dequantize(DType::Q4_K, &nu, f * e).unwrap(),
            dequantize(DType::Q6_K, &nd, e * f).unwrap(),
        );
        let mut worst = 0f64;
        let mut norm = 0f64;
        for c in 0..2 {
            let a = x.col(c);
            let xf: Vec<f32> = (0..e).map(|i| a.xq[i] as f32 * a.xd[i / 32]).collect();
            let mut act = vec![0f32; f];
            for r in 0..f {
                let gs: f32 = (0..e).map(|i| wg[r * e + i] * xf[i]).sum();
                let us: f32 = (0..e).map(|i| wu[r * e + i] * xf[i]).sum();
                act[r] = gs / (1.0 + (-gs).exp()) * us;
            }
            for r in 0..e {
                let t: f32 = (0..f).map(|i| wd[r * f + i] * act[i]).sum();
                worst = worst.max((t - out[c * e + r]).abs() as f64);
                norm = norm.max(t.abs() as f64);
            }
        }
        // the SwiGLU output is quantized to int8 on the way (relative error about 0.5 percent per value, averaging out in the dot product)
        assert!(worst / norm < 0.03, "ffn: worst error {worst} against a maximum of {norm}");
    }
    /// 2 MiB aligned copy of a plane, advised to be backed by transparent huge pages (BENCH_THP=1).
    struct ThpBuf {
        ptr: *mut u8,
        layout: std::alloc::Layout,
    }
    #[cfg(unix)]
    extern "C" {
        fn madvise(addr: *mut std::ffi::c_void, len: usize, advice: i32) -> i32;
    }
    #[cfg(not(unix))]
    unsafe fn madvise(_addr: *mut std::ffi::c_void, _len: usize, _advice: i32) -> i32 {
        0
    }
    impl ThpBuf {
        fn copy(src: &[u8]) -> ThpBuf {
            let size = (src.len() + (2 << 20) - 1) / (2 << 20) * (2 << 20);
            let layout = std::alloc::Layout::from_size_align(size, 2 << 20).unwrap();
            let ptr = unsafe { std::alloc::alloc(layout) };
            unsafe {
                madvise(ptr as *mut std::ffi::c_void, size, 14); // MADV_HUGEPAGE
                std::ptr::copy_nonoverlapping(src.as_ptr(), ptr, src.len());
            }
            ThpBuf { ptr, layout }
        }
    }
    impl Drop for ThpBuf {
        fn drop(&mut self) {
            unsafe { std::alloc::dealloc(self.ptr, self.layout) };
        }
    }

    fn host_mat_thp(fmt: QFmt, rows: usize, cols: usize, native: &[u8], keep: &mut Vec<ThpBuf>) -> HostMat {
        let planes = repack(fmt, native).unwrap();
        let mut p = [std::ptr::null(); 4];
        for (i, pl) in planes.iter().enumerate() {
            let b = ThpBuf::copy(pl);
            p[i] = b.ptr as *const u8;
            keep.push(b);
        }
        HostMat { fmt, rows, cols, p }
    }

    /// What the kernels do when memory is not the limit: every thread runs `dot_row` over its own 48 rows (K = 5120, about 140 KB of weights: L2 resident) again and again. Compare the weight rate with
    /// the DRAM rate (36 GB/s) to see how much of the compute budget a real decode step uses: cargo test --release -p forge-gpu --lib cpuk::tests::bench_compute_only -- --ignored --nocapture
    #[test]
    #[ignore]
    fn bench_compute_only() {
        let env_usize = |k: &str, d: usize| std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d);
        let threads = env_usize("BENCH_THREADS", 7);
        let (rows, k) = (48usize, 5120usize);
        // BENCH_FMT=q4_k, BENCH_ONLY_COLS=3, BENCH_S256=0|1 narrow it to one configuration (for `perf stat`)
        let only_fmt = std::env::var("BENCH_FMT").unwrap_or_default();
        let only_cols = env_usize("BENCH_ONLY_COLS", 0);
        let only_s256 = std::env::var("BENCH_S256").ok();
        for (name, fmt) in [("q4_k", QFmt::Q4_K), ("q6_k", QFmt::Q6_K), ("q3_k", QFmt::Q3_K), ("q2_k", QFmt::Q2_K), ("q8_0", QFmt::Q8_0)] {
            if !only_fmt.is_empty() && only_fmt != name {
                continue;
            }
            for s256 in [false, true] {
                if (s256 && fmt == QFmt::Q8_0) || only_s256.as_deref().map_or(false, |v| (v == "1") != s256) {
                    continue;
                }
                for cols in (1..=4usize).filter(|c| only_cols == 0 || *c == only_cols) {
                    let mut rng = Rng(5);
                    let mut mats = Vec::new();
                    let mut keep = Vec::new();
                    for _ in 0..threads {
                        let native = random_native(fmt, rows, k, &mut rng);
                        let (m, planes) = host_mat(fmt, rows, k, &native);
                        keep.push(planes);
                        mats.push(m);
                    }
                    let acts = if s256 {
                        let mut a = ActBuf::new(k, cols);
                        for c in 0..cols {
                            let x: Vec<f32> = (0..k).map(|i| rng.f() * if i % 97 == 0 { 8.0 } else { 1.0 }).collect();
                            a.quantize_col256(c, &x);
                        }
                        a
                    } else {
                        rand_act(k, cols, &mut rng)
                    };
                    let bytes_per_iter = {
                        let (bv, bb) = fmt.block();
                        rows * k / bv * bb
                    };
                    let pool = Pool::new(threads);
                    let total_iters = AtomicU64::new(0);
                    let secs = 0.6;
                    let t0 = std::time::Instant::now();
                    pool.run(&|t, _| {
                        let xv: Vec<Act> = (0..cols).map(|c| acts.col(c)).collect();
                        let mut o = [0f32; 4];
                        let mut iters = 0u64;
                        let m = &mats[t];
                        while t0.elapsed().as_secs_f64() < secs {
                            for _ in 0..20 {
                                for r in 0..rows {
                                    dot_row_any(m, r, &xv, &mut o[..cols], s256);
                                }
                                iters += 1;
                            }
                        }
                        std::hint::black_box(o);
                        total_iters.fetch_add(iters, Ordering::Relaxed);
                    });
                    let dt = t0.elapsed().as_secs_f64();
                    let gbs = total_iters.load(Ordering::Relaxed) as f64 * bytes_per_iter as f64 / dt / 1e9;
                    println!("{name} {} cols {cols} threads {threads}: {gbs:6.1} GB/s of weights from cache ({:.1} G weight-columns/s)", if s256 { "s256 " } else { "per32" }, gbs * 1e9 / (bytes_per_iter as f64 / (rows * k) as f64) * cols as f64 / 1e9);
                    drop(keep);
                }
            }
        }
    }

    /// Does the GPU's PCIe link add memory bandwidth next to the CPU's streaming? A thread copies page-locked host buffers into a device scratch buffer (the same DRAM reads a kernel on host-mapped weights
    /// would cause) while the CPU runs the FFN on 27B-sized matrices. Needs the GPU: cargo test --release -p forge-gpu --lib cpuk::tests::bench_concurrent_dma -- --ignored --nocapture
    #[test]
    #[ignore]
    fn bench_concurrent_dma() {
        use forge_cuda::{Device, DeviceBuffer, PinnedBuffer, Stream};
        let dev = Arc::new(Device::open(0).unwrap());
        dev.make_current().unwrap();
        let mut rng = Rng(99);
        let (e, f) = (5120usize, 17408usize);
        let layers = 6;
        let mut keep: Vec<Vec<Vec<u8>>> = Vec::new();
        let mut ffns = Vec::new();
        let mut bytes_per_layer = 0usize;
        for _ in 0..layers {
            let ng = random_native(QFmt::Q4_K, f, e, &mut rng);
            let nu = random_native(QFmt::Q4_K, f, e, &mut rng);
            let nd = random_native(QFmt::Q4_K, e, f, &mut rng);
            bytes_per_layer = ng.len() + nu.len() + nd.len();
            let (g, pg) = host_mat(QFmt::Q4_K, f, e, &ng);
            let (u, pu) = host_mat(QFmt::Q4_K, f, e, &nu);
            let (d, pd) = host_mat(QFmt::Q4_K, e, f, &nd);
            keep.push(pg);
            keep.push(pu);
            keep.push(pd);
            ffns.push(CpuFfn { gate: g, up: u, down: d, e, f, s256: false });
        }
        let env_usize = |k: &str, d: usize| std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d);
        let cols = env_usize("BENCH_COLS", 3);
        let threads = env_usize("BENCH_THREADS", 6);
        let acts = rand_act(e, cols, &mut rng);
        let xv: Vec<Act> = (0..cols).map(|c| acts.col(c)).collect();
        let pool = Pool::new(threads);
        let mut scratch = FfnScratch::new(f, cols);
        let mut out = vec![0f32; cols * e];
        // the DMA side: 12 buffers of 64 MiB (768 MiB, far beyond the caches) copied round and round into one device buffer
        const CHUNK: usize = 64 << 20;
        let mut srcs = Vec::new();
        for i in 0..12usize {
            let mut b = PinnedBuffer::alloc(CHUNK).unwrap();
            for (j, x) in b.as_mut_slice().iter_mut().enumerate().step_by(4096) {
                *x = (i + j) as u8;
            }
            srcs.push(b);
        }
        let dst = DeviceBuffer::alloc(CHUNK).unwrap();
        let stop = Arc::new(AtomicBool::new(false));
        let on = Arc::new(AtomicBool::new(false));
        let moved = Arc::new(AtomicU64::new(0));
        let h = {
            let (dev, stop, on, moved) = (dev.clone(), stop.clone(), on.clone(), moved.clone());
            thread::spawn(move || {
                dev.make_current().unwrap();
                let s = Stream::new().unwrap();
                let mut i = 0usize;
                while !stop.load(Ordering::Relaxed) {
                    if on.load(Ordering::Relaxed) {
                        for _ in 0..4 {
                            s.upload_async(&dst, 0, &srcs[i % srcs.len()], CHUNK).unwrap();
                            i += 1;
                        }
                        s.synchronize().unwrap();
                        moved.fetch_add(4 * CHUNK as u64, Ordering::Relaxed);
                    } else {
                        thread::sleep(Duration::from_millis(1));
                    }
                }
            })
        };
        let mut cpu_run = |secs: f64| -> f64 {
            let t0 = std::time::Instant::now();
            let mut n = 0usize;
            while t0.elapsed().as_secs_f64() < secs {
                for l in &ffns {
                    l.run(&pool, &xv, &mut scratch, &mut out);
                    n += 1;
                }
            }
            n as f64 * bytes_per_layer as f64 / t0.elapsed().as_secs_f64() / 1e9
        };
        let secs = 4.0;
        cpu_run(0.5);
        println!("cols {cols} threads {threads}: cpu alone      {:6.2} GB/s", cpu_run(secs));
        on.store(true, Ordering::Relaxed);
        thread::sleep(Duration::from_millis(500));
        let (m0, t0) = (moved.load(Ordering::Relaxed), std::time::Instant::now());
        thread::sleep(Duration::from_secs_f64(secs));
        let (m1, t1) = (moved.load(Ordering::Relaxed), std::time::Instant::now());
        println!("cols {cols} threads {threads}: dma alone      {:6.2} GB/s", (m1 - m0) as f64 / (t1 - t0).as_secs_f64() / 1e9);
        let (m0, t0) = (moved.load(Ordering::Relaxed), std::time::Instant::now());
        let cpu = cpu_run(secs);
        let (m1, t1) = (moved.load(Ordering::Relaxed), std::time::Instant::now());
        let dma = (m1 - m0) as f64 / (t1 - t0).as_secs_f64() / 1e9;
        println!("cols {cols} threads {threads}: cpu + dma      cpu {cpu:6.2} GB/s, dma {dma:6.2} GB/s, together {:6.2} GB/s", cpu + dma);
        on.store(false, Ordering::Relaxed);
        stop.store(true, Ordering::Relaxed);
        h.join().unwrap();
        drop(keep);
    }

    /// Throughput of the CPU FFN on 27B-sized matrices (hidden 5120, FFN 17408, Q4_K gate/up and Q4_K or Q6_K down) that do not fit the caches: cargo test --release -p forge-gpu --lib cpuk::tests::bench_ffn -- --ignored --nocapture
    #[test]
    #[ignore]
    fn bench_ffn() {
        let mut rng = Rng(77);
        // BENCH_E / BENCH_F / BENCH_LAYERS: small matrices that stay in the L3 cache show what the kernels do when memory is not the limit
        let env_usize = |k: &str, d: usize| std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d);
        let (e, f) = (env_usize("BENCH_E", 5120), env_usize("BENCH_F", 17408));
        let layers = env_usize("BENCH_LAYERS", 6);
        let fmt_of = |v: &str, d: QFmt| match v {
            "q2" => QFmt::Q2_K,
            "q3" => QFmt::Q3_K,
            "q4" => QFmt::Q4_K,
            "q6" => QFmt::Q6_K,
            _ => d,
        };
        let down_fmt = fmt_of(&std::env::var("BENCH_DOWN").unwrap_or_default(), QFmt::Q4_K);
        let gu_fmt = fmt_of(&std::env::var("BENCH_GU").unwrap_or_default(), QFmt::Q4_K);
        let mut keep: Vec<Vec<Vec<u8>>> = Vec::new();
        let mut thp: Vec<ThpBuf> = Vec::new();
        let mut ffns = Vec::new();
        let mut bytes_per_layer = 0usize;
        for _ in 0..layers {
            let ng = random_native(gu_fmt, f, e, &mut rng);
            let nu = random_native(gu_fmt, f, e, &mut rng);
            let nd = random_native(down_fmt, e, f, &mut rng);
            bytes_per_layer = ng.len() + nu.len() + nd.len();
            if std::env::var("BENCH_THP").is_ok() {
                let g = host_mat_thp(gu_fmt, f, e, &ng, &mut thp);
                let u = host_mat_thp(gu_fmt, f, e, &nu, &mut thp);
                let d = host_mat_thp(down_fmt, e, f, &nd, &mut thp);
                ffns.push(CpuFfn { gate: g, up: u, down: d, e, f, s256: std::env::var("BENCH_S256").is_ok() });
                continue;
            }
            let (g, pg) = host_mat(gu_fmt, f, e, &ng);
            let (u, pu) = host_mat(gu_fmt, f, e, &nu);
            let (d, pd) = host_mat(down_fmt, e, f, &nd);
            keep.push(pg);
            keep.push(pu);
            keep.push(pd);
            ffns.push(CpuFfn { gate: g, up: u, down: d, e, f, s256: std::env::var("BENCH_S256").is_ok() });
        }
        let acts = rand_act(e, 4, &mut rng);
        // BENCH_ONLY_COLS / BENCH_ONLY_THREADS: one configuration (for `perf stat`)
        let only_cols = env_usize("BENCH_ONLY_COLS", 0);
        let only_threads = env_usize("BENCH_ONLY_THREADS", 0);
        for cols in [1usize, 2, 3, 4].into_iter().filter(|c| only_cols == 0 || *c == only_cols) {
            let x = {
                let mut a = ActBuf::new(e, cols);
                for c in 0..cols {
                    a.xq[c * e..(c + 1) * e].copy_from_slice(&acts.xq[c * e..(c + 1) * e]);
                    a.xd[c * e / 32..(c + 1) * e / 32].copy_from_slice(&acts.xd[c * e / 32..(c + 1) * e / 32]);
                    a.xs[c * e / 32..(c + 1) * e / 32].copy_from_slice(&acts.xs[c * e / 32..(c + 1) * e / 32]);
                }
                a
            };
            let xv: Vec<Act> = (0..cols).map(|c| x.col(c)).collect();
            for threads in [1usize, 2, 3, 4, 5, 6, 7, 8].into_iter().filter(|t| only_threads == 0 || *t == only_threads) {
                let pool = Pool::new(threads);
                let mut scratch = FfnScratch::new(f, cols);
                let mut out = vec![0f32; cols * e];
                ffns[0].run(&pool, &xv, &mut scratch, &mut out);
                let reps = env_usize("BENCH_REPS", 8);
                let mut times = Vec::with_capacity(reps * layers);
                for _ in 0..reps {
                    for l in &ffns {
                        let t0 = std::time::Instant::now();
                        l.run(&pool, &xv, &mut scratch, &mut out);
                        times.push(t0.elapsed().as_secs_f64());
                    }
                }
                // the mean is what a decode step pays; the median shows how much of it is noise (preempted threads)
                let mean = times.iter().sum::<f64>() / times.len() as f64;
                times.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let med = times[times.len() / 2];
                println!(
                    "cols {cols} threads {threads}: {:6.2} GB/s of weights (mean), {:5.2} ms mean / {:5.2} ms median per layer FFN ({:.0} MB)",
                    bytes_per_layer as f64 / mean / 1e9,
                    mean * 1e3,
                    med * 1e3,
                    bytes_per_layer as f64 / 1e6
                );
            }
        }
        drop(keep);
        drop(thp);
    }
}
