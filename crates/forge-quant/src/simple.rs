//! Q8_0, Q4_0 and Q5_0: one f16 scale per 32 values, llama.cpp's reference rounding rules.

use crate::EncodeOpts;
use forge_core::half::f32_to_f16;

/// Q8_0 (34 bytes / 32 values): d = amax / 127, q = round(x / d).
pub(crate) fn q8_0(x: &[f32], out: &mut [u8]) {
    for (b, o) in x.chunks_exact(32).zip(out.chunks_exact_mut(34)) {
        let amax = b.iter().fold(0f32, |m, v| m.max(v.abs()));
        let d = amax / 127.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        o[0..2].copy_from_slice(&f32_to_f16(d).to_le_bytes());
        for (i, v) in b.iter().enumerate() {
            o[2 + i] = (v * id).round() as i8 as u8;
        }
    }
}

/// The signed value of largest magnitude of a block (llama.cpp's `max`).
fn signed_max(b: &[f32]) -> f32 {
    let (mut amax, mut max) = (0f32, 0f32);
    for &v in b {
        if v.abs() > amax {
            amax = v.abs();
            max = v;
        }
    }
    max
}

/// Q4_0 (18 bytes / 32 values): d = max / -8, q = min(15, trunc(x / d + 8.5)); low nibbles hold values 0..15, high nibbles values 16..31.
pub(crate) fn q4_0(x: &[f32], _imp: Option<&[f32]>, _opts: &EncodeOpts, out: &mut [u8]) {
    for (b, o) in x.chunks_exact(32).zip(out.chunks_exact_mut(18)) {
        let d = signed_max(b) / -8.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        o[0..2].copy_from_slice(&f32_to_f16(d).to_le_bytes());
        for j in 0..16 {
            let x0 = b[j] * id;
            let x1 = b[16 + j] * id;
            let q0 = ((x0 + 8.5) as i32).clamp(0, 15) as u8;
            let q1 = ((x1 + 8.5) as i32).clamp(0, 15) as u8;
            o[2 + j] = q0 | (q1 << 4);
        }
    }
}

/// Q5_0 (22 bytes / 32 values): d = max / -16, q = min(31, trunc(x / d + 16.5)); the 5th bits are collected in a u32.
pub(crate) fn q5_0(x: &[f32], _imp: Option<&[f32]>, _opts: &EncodeOpts, out: &mut [u8]) {
    for (b, o) in x.chunks_exact(32).zip(out.chunks_exact_mut(22)) {
        let d = signed_max(b) / -16.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        o[0..2].copy_from_slice(&f32_to_f16(d).to_le_bytes());
        let mut qh = 0u32;
        for j in 0..16 {
            let q0 = ((b[j] * id + 16.5) as i32).clamp(0, 31) as u32;
            let q1 = ((b[16 + j] * id + 16.5) as i32).clamp(0, 31) as u32;
            o[6 + j] = ((q0 & 0xF) | ((q1 & 0xF) << 4)) as u8;
            qh |= ((q0 >> 4) & 1) << j;
            qh |= ((q1 >> 4) & 1) << (j + 16);
        }
        o[2..6].copy_from_slice(&qh.to_le_bytes());
    }
}
