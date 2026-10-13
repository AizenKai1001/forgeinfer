//! Tensor element types of the GGUF container and the dequantisation of the block formats we read.
//! Written from the byte layouts (tools/ref_gguf.py is the executable specification and `tests/golden/quant_*` the bit-exact references).
//!
//! Block layouts (all little endian, every block starts with an f16 scale `d` unless noted):
//!   Q8_0  34 B / 32 values : d, 32 x i8
//!   Q4_0  18 B / 32        : d, 16 B of nibbles (low nibbles = values 0..15, high nibbles = values 16..31), value = d * (q - 8)
//!   Q4_1  20 B / 32        : d, m, 16 B nibbles, value = d * q + m
//!   Q5_0  22 B / 32        : d, u32 qh (5th bit of the 32 values), 16 B nibbles, value = d * (q - 16)
//!   Q5_1  24 B / 32        : d, m, u32 qh, 16 B nibbles, value = d * q + m
//!   Q4_K 144 B / 256       : d, dmin, 12 B packed 6-bit scales and mins of 8 sub-blocks, 128 B nibbles
//!   Q5_K 176 B / 256       : d, dmin, 12 B scales/mins, 32 B qh, 128 B nibbles
//!   Q6_K 210 B / 256       : 128 B ql, 64 B qh, 16 x i8 scales, d LAST
//!   Q2_K  84 B / 256       : 16 B (scale | min << 4) per 16 values, 64 B of 2-bit codes, d, dmin; value = d * sc * q - dmin * min
//!   Q3_K 110 B / 256       : 32 B hmask (the 3rd bit), 64 B of 2-bit codes, 12 B packed 6-bit scales of 16 groups, d LAST; value = d * (scale - 32) * (q - 4 * (hmask bit clear))

use forge_core::half::{bf16_to_f32, f16_to_f32};
use forge_core::{Error, Result};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[allow(non_camel_case_types)]
pub enum DType {
    F32,
    F16,
    BF16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q4_K,
    Q5_K,
    Q6_K,
    Q2_K,
    Q3_K,
    /// A ggml type id this build cannot read.
    Other(u32),
}

impl DType {
    pub fn from_ggml(id: u32) -> DType {
        match id {
            0 => DType::F32,
            1 => DType::F16,
            2 => DType::Q4_0,
            3 => DType::Q4_1,
            6 => DType::Q5_0,
            7 => DType::Q5_1,
            8 => DType::Q8_0,
            12 => DType::Q4_K,
            13 => DType::Q5_K,
            10 => DType::Q2_K,
            11 => DType::Q3_K,
            14 => DType::Q6_K,
            30 => DType::BF16,
            other => DType::Other(other),
        }
    }

    pub fn ggml_id(self) -> u32 {
        match self {
            DType::F32 => 0,
            DType::F16 => 1,
            DType::Q4_0 => 2,
            DType::Q4_1 => 3,
            DType::Q5_0 => 6,
            DType::Q5_1 => 7,
            DType::Q8_0 => 8,
            DType::Q4_K => 12,
            DType::Q5_K => 13,
            DType::Q2_K => 10,
            DType::Q3_K => 11,
            DType::Q6_K => 14,
            DType::BF16 => 30,
            DType::Other(i) => i,
        }
    }

    /// Values per block (1 for the plain float types).
    pub fn block_values(self) -> usize {
        match self {
            DType::F32 | DType::F16 | DType::BF16 => 1,
            DType::Q4_0 | DType::Q4_1 | DType::Q5_0 | DType::Q5_1 | DType::Q8_0 => 32,
            DType::Q4_K | DType::Q5_K | DType::Q6_K | DType::Q2_K | DType::Q3_K => 256,
            DType::Other(_) => 0,
        }
    }

    /// Bytes per block.
    pub fn block_bytes(self) -> usize {
        match self {
            DType::F32 => 4,
            DType::F16 | DType::BF16 => 2,
            DType::Q4_0 => 18,
            DType::Q4_1 => 20,
            DType::Q5_0 => 22,
            DType::Q5_1 => 24,
            DType::Q8_0 => 34,
            DType::Q4_K => 144,
            DType::Q5_K => 176,
            DType::Q6_K => 210,
            DType::Q2_K => 84,
            DType::Q3_K => 110,
            DType::Other(_) => 0,
        }
    }

    pub fn name(self) -> String {
        match self {
            DType::F32 => "F32".into(),
            DType::F16 => "F16".into(),
            DType::BF16 => "BF16".into(),
            DType::Q4_0 => "Q4_0".into(),
            DType::Q4_1 => "Q4_1".into(),
            DType::Q5_0 => "Q5_0".into(),
            DType::Q5_1 => "Q5_1".into(),
            DType::Q8_0 => "Q8_0".into(),
            DType::Q4_K => "Q4_K".into(),
            DType::Q5_K => "Q5_K".into(),
            DType::Q6_K => "Q6_K".into(),
            DType::Q2_K => "Q2_K".into(),
            DType::Q3_K => "Q3_K".into(),
            DType::Other(i) => format!("ggml_type_{i}"),
        }
    }

    /// Bytes of `n` values (n must be a multiple of the block size).
    pub fn bytes_for(self, n: usize) -> usize {
        let bv = self.block_values();
        if bv == 0 {
            0
        } else {
            n / bv * self.block_bytes()
        }
    }

    pub fn is_quantized(self) -> bool {
        self.block_values() > 1
    }
}

#[inline]
fn h(b: &[u8]) -> f32 {
    f16_to_f32(u16::from_le_bytes([b[0], b[1]]))
}

/// The packed 6-bit (scale, min) pair of sub-block `j` (0..8) of a Q4_K / Q5_K block (`s` = the 12 scale bytes).
#[inline]
fn scale_min_k4(j: usize, s: &[u8]) -> (u8, u8) {
    if j < 4 {
        (s[j] & 63, s[j + 4] & 63)
    } else {
        (
            (s[j + 4] & 0x0F) | ((s[j - 4] >> 6) << 4),
            (s[j + 4] >> 4) | ((s[j] >> 6) << 4),
        )
    }
}

/// The 16 six-bit scales of a Q3_K block, minus 32 (`s` = the 12 packed bytes at offset 96): ggml's kmask unpacking.
pub fn q3k_scales(s: &[u8]) -> [i32; 16] {
    let rd = |o: usize| u32::from_le_bytes([s[o], s[o + 1], s[o + 2], s[o + 3]]);
    let (a0, a1, tmp) = (rd(0), rd(4), rd(8));
    const K1: u32 = 0x0303_0303;
    const K2: u32 = 0x0f0f_0f0f;
    let n2 = ((a0 >> 4) & K2) | (((tmp >> 4) & K1) << 4);
    let n3 = ((a1 >> 4) & K2) | (((tmp >> 6) & K1) << 4);
    let n0 = (a0 & K2) | ((tmp & K1) << 4);
    let n1 = (a1 & K2) | (((tmp >> 2) & K1) << 4);
    let mut out = [0i32; 16];
    for (w, word) in [n0, n1, n2, n3].iter().enumerate() {
        for k in 0..4 {
            out[4 * w + k] = ((word >> (8 * k)) & 0xFF) as i32 - 32;
        }
    }
    out
}

/// Dequantise `out.len()` values from `src` (the exact bytes of `out.len() / block_values` blocks).
pub fn dequantize_into(dtype: DType, src: &[u8], out: &mut [f32]) -> Result<()> {
    let n = out.len();
    let bv = dtype.block_values();
    if bv == 0 {
        return Err(Error::unsupported(format!(
            "dequantisation of {}",
            dtype.name()
        )));
    }
    if !n.is_multiple_of(bv) {
        return Err(Error::format(format!(
            "{} values are not a multiple of the {} block size {}",
            n,
            dtype.name(),
            bv
        )));
    }
    let need = dtype.bytes_for(n);
    if src.len() < need {
        return Err(Error::format(format!(
            "{}: {} bytes given, {} needed for {} values",
            dtype.name(),
            src.len(),
            need,
            n
        )));
    }
    let bb = dtype.block_bytes();
    match dtype {
        DType::F32 => {
            for (i, o) in out.iter_mut().enumerate() {
                *o = f32::from_le_bytes([
                    src[4 * i],
                    src[4 * i + 1],
                    src[4 * i + 2],
                    src[4 * i + 3],
                ]);
            }
        }
        DType::F16 => {
            for (i, o) in out.iter_mut().enumerate() {
                *o = f16_to_f32(u16::from_le_bytes([src[2 * i], src[2 * i + 1]]));
            }
        }
        DType::BF16 => {
            for (i, o) in out.iter_mut().enumerate() {
                *o = bf16_to_f32(u16::from_le_bytes([src[2 * i], src[2 * i + 1]]));
            }
        }
        DType::Q8_0 => {
            for (b, y) in src.chunks_exact(bb).zip(out.chunks_exact_mut(32)) {
                let d = h(b);
                for i in 0..32 {
                    y[i] = d * (b[2 + i] as i8) as f32;
                }
            }
        }
        DType::Q4_0 => {
            for (b, y) in src.chunks_exact(bb).zip(out.chunks_exact_mut(32)) {
                let d = h(b);
                for j in 0..16 {
                    let q = b[2 + j];
                    y[j] = d * ((q & 0xF) as f32 - 8.0);
                    y[j + 16] = d * ((q >> 4) as f32 - 8.0);
                }
            }
        }
        DType::Q4_1 => {
            for (b, y) in src.chunks_exact(bb).zip(out.chunks_exact_mut(32)) {
                let d = h(b);
                let m = h(&b[2..]);
                for j in 0..16 {
                    let q = b[4 + j];
                    y[j] = d * (q & 0xF) as f32 + m;
                    y[j + 16] = d * (q >> 4) as f32 + m;
                }
            }
        }
        DType::Q5_0 => {
            for (b, y) in src.chunks_exact(bb).zip(out.chunks_exact_mut(32)) {
                let d = h(b);
                let qh = u32::from_le_bytes([b[2], b[3], b[4], b[5]]);
                for j in 0..16 {
                    let q = b[6 + j];
                    let x0 = ((q & 0xF) as u32 | (((qh >> j) & 1) << 4)) as f32 - 16.0;
                    let x1 = ((q >> 4) as u32 | (((qh >> (j + 16)) & 1) << 4)) as f32 - 16.0;
                    y[j] = d * x0;
                    y[j + 16] = d * x1;
                }
            }
        }
        DType::Q5_1 => {
            for (b, y) in src.chunks_exact(bb).zip(out.chunks_exact_mut(32)) {
                let d = h(b);
                let m = h(&b[2..]);
                let qh = u32::from_le_bytes([b[4], b[5], b[6], b[7]]);
                for j in 0..16 {
                    let q = b[8 + j];
                    let x0 = ((q & 0xF) as u32 | (((qh >> j) & 1) << 4)) as f32;
                    let x1 = ((q >> 4) as u32 | (((qh >> (j + 16)) & 1) << 4)) as f32;
                    y[j] = d * x0 + m;
                    y[j + 16] = d * x1 + m;
                }
            }
        }
        DType::Q4_K => {
            for (b, y) in src.chunks_exact(bb).zip(out.chunks_exact_mut(256)) {
                let d = h(b);
                let dmin = h(&b[2..]);
                let sc = &b[4..16];
                let qs = &b[16..144];
                for g in 0..4 {
                    let (s0, m0) = scale_min_k4(2 * g, sc);
                    let (s1, m1) = scale_min_k4(2 * g + 1, sc);
                    let (d0, mm0) = (d * s0 as f32, dmin * m0 as f32);
                    let (d1, mm1) = (d * s1 as f32, dmin * m1 as f32);
                    for l in 0..32 {
                        let q = qs[g * 32 + l];
                        y[g * 64 + l] = d0 * (q & 0xF) as f32 - mm0;
                        y[g * 64 + 32 + l] = d1 * (q >> 4) as f32 - mm1;
                    }
                }
            }
        }
        DType::Q5_K => {
            for (b, y) in src.chunks_exact(bb).zip(out.chunks_exact_mut(256)) {
                let d = h(b);
                let dmin = h(&b[2..]);
                let sc = &b[4..16];
                let qh = &b[16..48];
                let qs = &b[48..176];
                for g in 0..4 {
                    let (s0, m0) = scale_min_k4(2 * g, sc);
                    let (s1, m1) = scale_min_k4(2 * g + 1, sc);
                    let (d0, mm0) = (d * s0 as f32, dmin * m0 as f32);
                    let (d1, mm1) = (d * s1 as f32, dmin * m1 as f32);
                    let (u0, u1) = (1u8 << (2 * g), 2u8 << (2 * g));
                    for l in 0..32 {
                        let q = qs[g * 32 + l];
                        let hi0 = if qh[l] & u0 != 0 { 16.0 } else { 0.0 };
                        let hi1 = if qh[l] & u1 != 0 { 16.0 } else { 0.0 };
                        y[g * 64 + l] = d0 * ((q & 0xF) as f32 + hi0) - mm0;
                        y[g * 64 + 32 + l] = d1 * ((q >> 4) as f32 + hi1) - mm1;
                    }
                }
            }
        }
        DType::Q6_K => {
            for (b, y) in src.chunks_exact(bb).zip(out.chunks_exact_mut(256)) {
                let ql = &b[0..128];
                let qh = &b[128..192];
                let scales = &b[192..208];
                let d = h(&b[208..]);
                for half in 0..2 {
                    let l_ = &ql[half * 64..half * 64 + 64];
                    let h_ = &qh[half * 32..half * 32 + 32];
                    let sc = &scales[half * 8..half * 8 + 8];
                    let base = half * 128;
                    for l in 0..32 {
                        let isc = l / 16;
                        let q1 = ((l_[l] & 0xF) | ((h_[l] & 3) << 4)) as i32 - 32;
                        let q2 = ((l_[l + 32] & 0xF) | (((h_[l] >> 2) & 3) << 4)) as i32 - 32;
                        let q3 = ((l_[l] >> 4) | (((h_[l] >> 4) & 3) << 4)) as i32 - 32;
                        let q4 = ((l_[l + 32] >> 4) | (((h_[l] >> 6) & 3) << 4)) as i32 - 32;
                        y[base + l] = d * (sc[isc] as i8) as f32 * q1 as f32;
                        y[base + 32 + l] = d * (sc[isc + 2] as i8) as f32 * q2 as f32;
                        y[base + 64 + l] = d * (sc[isc + 4] as i8) as f32 * q3 as f32;
                        y[base + 96 + l] = d * (sc[isc + 6] as i8) as f32 * q4 as f32;
                    }
                }
            }
        }
        DType::Q2_K => {
            for (b, y) in src.chunks_exact(bb).zip(out.chunks_exact_mut(256)) {
                let sc = &b[0..16];
                let qs = &b[16..80];
                let d = h(&b[80..]);
                let dmin = h(&b[82..]);
                let mut is = 0;
                for half in 0..2 {
                    let q = &qs[half * 32..half * 32 + 32];
                    let mut shift = 0;
                    for j in 0..4 {
                        for part in 0..2 {
                            let s = sc[is];
                            is += 1;
                            let (dl, ml) = (d * (s & 0xF) as f32, dmin * (s >> 4) as f32);
                            for l in 0..16 {
                                y[half * 128 + j * 32 + part * 16 + l] = dl * ((q[part * 16 + l] >> shift) & 3) as f32 - ml;
                            }
                        }
                        shift += 2;
                    }
                }
            }
        }
        DType::Q3_K => {
            for (b, y) in src.chunks_exact(bb).zip(out.chunks_exact_mut(256)) {
                let hm = &b[0..32];
                let qs = &b[32..96];
                let sc = q3k_scales(&b[96..108]);
                let d = h(&b[108..]);
                let mut is = 0;
                let mut m = 1u8;
                for half in 0..2 {
                    let q = &qs[half * 32..half * 32 + 32];
                    let mut shift = 0;
                    for j in 0..4 {
                        for part in 0..2 {
                            let dl = d * sc[is] as f32;
                            is += 1;
                            for l in 0..16 {
                                let low = ((q[part * 16 + l] >> shift) & 3) as i32;
                                let off = if hm[part * 16 + l] & m != 0 { 0 } else { 4 };
                                y[half * 128 + j * 32 + part * 16 + l] = dl * (low - off) as f32;
                            }
                        }
                        shift += 2;
                        m <<= 1;
                    }
                }
            }
        }
        DType::Other(_) => unreachable!(),
    }
    Ok(())
}

pub fn dequantize(dtype: DType, src: &[u8], n: usize) -> Result<Vec<f32>> {
    let mut out = vec![0f32; n];
    dequantize_into(dtype, src, &mut out)?;
    Ok(out)
}
