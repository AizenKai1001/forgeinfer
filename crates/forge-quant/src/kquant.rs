//! The K-quants: super-blocks of 256 values.
//!
//! Q4_K / Q5_K: 8 sub-blocks of 32 values, each x ~ d * sc * q - dmin * mn with 6-bit sc and mn (so a sub-block is an affine grid with a non-positive offset).
//! Q6_K: 16 sub-blocks of 16 values, x ~ d * sc * (q - 32) with int8 sc.
//!
//! The scales are chosen by fitting each sub-block to minimize sum_i w_i (x_i - decoded_i)^2 (w = the weighting of the encoder options), then quantizing the scales and re-deriving the codes
//! from the decoded scales, and finally (with `refine`) trying the neighbouring scale codes of every sub-block against the true weighted error of the decoded values.

use crate::{av_x, element_weights, EncodeOpts};
use forge_core::half::{f16_to_f32, f32_to_f16};

/// Round to f16 and back (the decoder only sees the f16 value).
fn r16(x: f32) -> f32 {
    f16_to_f32(f32_to_f16(x))
}

/// Weighted squared error of `q` decoded as `scale * q + lo`.
fn affine_error(x: &[f32], w: &[f32], q: &[u8], scale: f32, lo: f32) -> f32 {
    let mut e = 0f32;
    for i in 0..x.len() {
        let d = scale * q[i] as f32 + lo - x[i];
        e += w[i] * d * d;
    }
    e
}

/// Weighted least squares for (scale, lo) given the codes: minimizes sum w (scale q + lo - x)^2 with lo <= 0. Returns (scale, lo).
fn solve_affine(x: &[f32], w: &[f32], q: &[u8]) -> (f32, f32) {
    let (mut sw, mut sl, mut sl2, mut sxl, mut sx) = (0f32, 0f32, 0f32, 0f32, 0f32);
    for i in 0..x.len() {
        let l = q[i] as f32;
        sw += w[i];
        sl += w[i] * l;
        sl2 += w[i] * l * l;
        sxl += w[i] * x[i] * l;
        sx += w[i] * x[i];
    }
    let det = sw * sl2 - sl * sl;
    if det > 1e-12 * (sw * sl2).abs().max(1e-30) {
        let scale = (sw * sxl - sx * sl) / det;
        let lo = (sl2 * sx - sl * sxl) / det;
        if lo > 0.0 {
            return (if sl2 > 0.0 { sxl / sl2 } else { 0.0 }, 0.0);
        }
        (scale, lo)
    } else {
        // all codes equal: a pure offset
        (0.0, if sw > 0.0 { (sx / sw).min(0.0) } else { 0.0 })
    }
}

/// Weighted affine fit of one sub-block: x ~ d * q - m with q in 0..=levels, d >= 0, m >= 0. Fills the codes, returns (d, m).
fn fit_affine(x: &[f32], w: &[f32], levels: u32, q: &mut [u8]) -> (f32, f32) {
    let n = x.len();
    let lv = levels as i32;
    let mut min = x[0];
    let mut max = x[0];
    for &v in x {
        min = min.min(v);
        max = max.max(v);
    }
    if min > 0.0 {
        min = 0.0;
    }
    if max <= min {
        q.iter_mut().for_each(|v| *v = 0);
        return (0.0, -min);
    }
    let mut best_err = f32::INFINITY;
    let mut best = (0f32, min);
    let mut cand = [0u8; 32];
    // a sweep of the grid spacing around range / levels (llama.cpp's make_qkx2_quants), each candidate followed by the least-squares scale and offset of its codes
    for step in 0..=20 {
        let iscale = (levels as f32 - 1.0 + 0.1 * step as f32) / (max - min);
        for i in 0..n {
            cand[i] = ((iscale * (x[i] - min)).round() as i32).clamp(0, lv) as u8;
        }
        let (scale, lo) = solve_affine(x, w, &cand[..n]);
        if scale <= 0.0 {
            continue;
        }
        let e = affine_error(x, w, &cand[..n], scale, lo);
        if e < best_err {
            best_err = e;
            best = (scale, lo);
            q[..n].copy_from_slice(&cand[..n]);
        }
    }
    if best_err == f32::INFINITY {
        q.iter_mut().for_each(|v| *v = 0);
        return (0.0, -min.min(0.0));
    }
    // alternate: codes from the fitted grid, grid from the codes
    for _ in 0..4 {
        let (scale, lo) = best;
        let mut improved = false;
        for i in 0..n {
            cand[i] = (((x[i] - lo) / scale).round() as i32).clamp(0, lv) as u8;
        }
        let (s2, l2) = solve_affine(x, w, &cand[..n]);
        if s2 > 0.0 {
            let e = affine_error(x, w, &cand[..n], s2, l2);
            if e < best_err {
                best_err = e;
                best = (s2, l2);
                q[..n].copy_from_slice(&cand[..n]);
                improved = true;
            }
        }
        if !improved {
            break;
        }
    }
    (best.0, -best.1)
}

/// Scale and minimum codes (6 bits each) of the 8 sub-blocks, packed in 12 bytes (llama.cpp's `get_scale_min_k4` layout).
fn pack_scales(ls: &[u8; 8], lm: &[u8; 8], out: &mut [u8]) {
    for j in 0..4 {
        out[j] = ls[j] & 63;
        out[j + 4] = lm[j] & 63;
    }
    for j in 4..8 {
        out[j + 4] = (ls[j] & 0xF) | ((lm[j] & 0xF) << 4);
        out[j - 4] |= (ls[j] >> 4) << 6;
        out[j] |= (lm[j] >> 4) << 6;
    }
}

/// Scales of one Q4_K / Q5_K super-block: d, dmin (f16-representable) and the 6-bit codes of the eight sub-blocks' scales and minimums.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Q4kScales {
    pub d: f32,
    pub dmin: f32,
    pub ls: [u8; 8],
    pub lm: [u8; 8],
}

/// Fit one super-block (256 values `xs` with element weights `wbuf`): the sub-block grids, their 6-bit scale codes, and (with `refine`) the neighbouring scale codes tried against the weighted
/// error; fills the codes of the values.
pub(crate) fn q4k_fit(
    xs: &[f32],
    wbuf: &[f32],
    levels: u32,
    refine: bool,
    codes: &mut [u8; 256],
) -> Q4kScales {
    let lv = levels as i32;
    let mut scale = [0f32; 8];
    let mut minv = [0f32; 8];
    for j in 0..8 {
        let r = 32 * j..32 * j + 32;
        let (d, m) = fit_affine(&xs[r.clone()], &wbuf[r.clone()], levels, &mut codes[r]);
        scale[j] = d;
        minv[j] = m;
    }
    let max_scale = scale.iter().cloned().fold(0f32, f32::max);
    let max_min = minv.iter().cloned().fold(0f32, f32::max);
    let d = r16(max_scale / 63.0);
    let dmin = r16(max_min / 63.0);
    let mut ls = [0u8; 8];
    let mut lm = [0u8; 8];
    for j in 0..8 {
        ls[j] = if d > 0.0 {
            ((scale[j] / d).round() as i32).clamp(0, 63) as u8
        } else {
            0
        };
        lm[j] = if dmin > 0.0 {
            ((minv[j] / dmin).round() as i32).clamp(0, 63) as u8
        } else {
            0
        };
    }
    // codes from the decoded scales; with `refine`, the neighbouring scale codes are tried against the true weighted error
    for j in 0..8 {
        let r = 32 * j..32 * j + 32;
        let (xj, wj) = (&xs[r.clone()], &wbuf[r.clone()]);
        let try_codes = |lsj: u8, lmj: u8, q: &mut [u8]| -> f32 {
            let sc = d * lsj as f32;
            let mn = dmin * lmj as f32;
            if sc <= 0.0 {
                q.iter_mut().for_each(|v| *v = 0);
                return affine_error(xj, wj, q, 0.0, -mn);
            }
            for i in 0..32 {
                q[i] = (((xj[i] + mn) / sc).round() as i32).clamp(0, lv) as u8;
            }
            affine_error(xj, wj, q, sc, -mn)
        };
        let mut tmp = [0u8; 32];
        let mut best_err = try_codes(ls[j], lm[j], &mut tmp);
        let mut best = (ls[j], lm[j], tmp);
        if refine {
            for dl in -1i32..=1 {
                for dm in -1i32..=1 {
                    if dl == 0 && dm == 0 {
                        continue;
                    }
                    let (a, b) = (ls[j] as i32 + dl, lm[j] as i32 + dm);
                    if !(0..=63).contains(&a) || !(0..=63).contains(&b) {
                        continue;
                    }
                    let e = try_codes(a as u8, b as u8, &mut tmp);
                    if e < best_err {
                        best_err = e;
                        best = (a as u8, b as u8, tmp);
                    }
                }
            }
        }
        ls[j] = best.0;
        lm[j] = best.1;
        codes[r].copy_from_slice(&best.2);
    }
    Q4kScales { d, dmin, ls, lm }
}

/// Write the bytes of a Q4_K (144 B) or Q5_K (176 B, `five`) super-block.
pub(crate) fn q4k_pack(s: &Q4kScales, codes: &[u8; 256], five: bool, o: &mut [u8]) {
    o.iter_mut().for_each(|b| *b = 0);
    o[0..2].copy_from_slice(&f32_to_f16(s.d).to_le_bytes());
    o[2..4].copy_from_slice(&f32_to_f16(s.dmin).to_le_bytes());
    pack_scales(&s.ls, &s.lm, &mut o[4..16]);
    let (qh_off, qs_off) = if five { (16, 48) } else { (0, 16) };
    for g in 0..4 {
        for l in 0..32 {
            let lo = codes[64 * g + l];
            let hi = codes[64 * g + 32 + l];
            o[qs_off + 32 * g + l] = (lo & 0xF) | ((hi & 0xF) << 4);
            if five {
                o[qh_off + l] |= (((lo >> 4) & 1) << (2 * g)) | (((hi >> 4) & 1) << (2 * g + 1));
            }
        }
    }
}

/// The element weights of a super-block for the fit (a block whose weights are all zero is fitted with equal weights).
pub(crate) fn block_weights(
    opts: &EncodeOpts,
    xs: &[f32],
    imps: Option<&[f32]>,
    wbuf: &mut [f32; 256],
) {
    element_weights(opts, xs, imps, av_x(xs), wbuf);
    if wbuf.iter().sum::<f32>() <= 0.0 {
        wbuf.iter_mut().for_each(|w| *w = 1.0);
    }
}

/// Q4_K (144 bytes per 256 values: d, dmin, scales[12], qs[128]) for `levels` = 15, Q5_K (176 bytes: d, dmin, scales[12], qh[32], qs[128]) for `levels` = 31.
pub(crate) fn q4_k(x: &[f32], imp: Option<&[f32]>, opts: &EncodeOpts, out: &mut [u8], levels: u32) {
    let (bb, five) = if levels == 15 {
        (144, false)
    } else {
        (176, true)
    };
    let mut wbuf = [0f32; 256];
    for (sb, (xs, o)) in x
        .chunks_exact(256)
        .zip(out.chunks_exact_mut(bb))
        .enumerate()
    {
        block_weights(
            opts,
            xs,
            imp.map(|v| &v[sb * 256..sb * 256 + 256]),
            &mut wbuf,
        );
        let mut codes = [0u8; 256];
        let sc = q4k_fit(xs, &wbuf, levels, opts.refine, &mut codes);
        q4k_pack(&sc, &codes, five, o);
    }
}

/// Weighted symmetric fit of a 16-value sub-block: x ~ s * q with q in -32..=31 (s may be negative, which uses the extra negative level for blocks dominated by a positive value).
fn fit_sym(x: &[f32], w: &[f32], q: &mut [i8]) -> f32 {
    let n = x.len();
    let (mut ext, mut ext_abs) = (0f32, 0f32);
    for &v in x {
        if v.abs() > ext_abs {
            ext_abs = v.abs();
            ext = v;
        }
    }
    if ext_abs == 0.0 {
        q.iter_mut().for_each(|v| *v = 0);
        return 0.0;
    }
    let mut best_err = f32::INFINITY;
    let mut best_s = 0f32;
    let mut cand = [0i8; 16];
    let try_scale =
        |s: f32, cand: &mut [i8; 16], best_err: &mut f32, best_s: &mut f32, q: &mut [i8]| {
            if s == 0.0 {
                return;
            }
            // codes for this scale, then the least-squares scale of those codes
            let (mut sxq, mut sq2) = (0f32, 0f32);
            for i in 0..n {
                let c = ((x[i] / s).round() as i32).clamp(-32, 31);
                cand[i] = c as i8;
                sxq += w[i] * x[i] * c as f32;
                sq2 += w[i] * (c * c) as f32;
            }
            let s2 = if sq2 > 0.0 { sxq / sq2 } else { s };
            if s2 == 0.0 {
                return;
            }
            let mut e = 0f32;
            for i in 0..n {
                let d = s2 * cand[i] as f32 - x[i];
                e += w[i] * d * d;
            }
            if e < *best_err {
                *best_err = e;
                *best_s = s2;
                q[..n].copy_from_slice(&cand[..n]);
            }
        };
    // the extreme value at -32 or at +31, and spacings around those
    for target in [-32.0f32, 31.0] {
        let s0 = ext / target;
        for k in -12i32..=12 {
            try_scale(
                s0 * (1.0 + 0.025 * k as f32),
                &mut cand,
                &mut best_err,
                &mut best_s,
                q,
            );
        }
    }
    best_s
}

/// Scales of one Q6_K super-block: d (f16-representable) and the int8 codes of the sixteen sub-block scales.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Q6kScales {
    pub d: f32,
    pub sc: [i8; 16],
}

/// Fit one Q6_K super-block (256 values with element weights); fills the codes (-32..=31).
pub(crate) fn q6k_fit(xs: &[f32], wbuf: &[f32], refine: bool, codes: &mut [i8; 256]) -> Q6kScales {
    let mut s = [0f32; 16];
    for j in 0..16 {
        let r = 16 * j..16 * j + 16;
        s[j] = fit_sym(&xs[r.clone()], &wbuf[r.clone()], &mut codes[r]);
    }
    let max_s = s.iter().fold(0f32, |m, v| m.max(v.abs()));
    let d = r16(max_s / 127.0);
    let mut sc = [0i8; 16];
    for j in 0..16 {
        sc[j] = if d > 0.0 {
            ((s[j] / d).round() as i32).clamp(-128, 127) as i8
        } else {
            0
        };
    }
    for j in 0..16 {
        let r = 16 * j..16 * j + 16;
        let (xj, wj) = (&xs[r.clone()], &wbuf[r.clone()]);
        let try_code = |c: i8, q: &mut [i8; 16]| -> f32 {
            let sj = d * c as f32;
            let mut e = 0f32;
            for i in 0..16 {
                q[i] = if sj != 0.0 {
                    ((xj[i] / sj).round() as i32).clamp(-32, 31) as i8
                } else {
                    0
                };
                let dd = sj * q[i] as f32 - xj[i];
                e += wj[i] * dd * dd;
            }
            e
        };
        let mut tmp = [0i8; 16];
        let mut best_err = try_code(sc[j], &mut tmp);
        let mut best = (sc[j], tmp);
        if refine {
            for dc in [-1i32, 1] {
                let c = sc[j] as i32 + dc;
                if !(-128..=127).contains(&c) {
                    continue;
                }
                let e = try_code(c as i8, &mut tmp);
                if e < best_err {
                    best_err = e;
                    best = (c as i8, tmp);
                }
            }
        }
        sc[j] = best.0;
        codes[r].copy_from_slice(&best.1);
    }
    Q6kScales { d, sc }
}

/// Write the 210 bytes of a Q6_K super-block: two 128-value halves, each 4 quarters of 32 values sharing the low/high nibbles of ql and the 2-bit slices of qh.
pub(crate) fn q6k_pack(s: &Q6kScales, codes: &[i8; 256], o: &mut [u8]) {
    for g in 0..2 {
        for l in 0..32 {
            let q = |k: usize| (codes[128 * g + 32 * k + l] as i32 + 32) as u8;
            let (q1, q2, q3, q4) = (q(0), q(1), q(2), q(3));
            o[64 * g + l] = (q1 & 0xF) | ((q3 & 0xF) << 4);
            o[64 * g + 32 + l] = (q2 & 0xF) | ((q4 & 0xF) << 4);
            o[128 + 32 * g + l] =
                (q1 >> 4) | ((q2 >> 4) << 2) | ((q3 >> 4) << 4) | ((q4 >> 4) << 6);
        }
    }
    for j in 0..16 {
        o[192 + j] = s.sc[j] as u8;
    }
    o[208..210].copy_from_slice(&f32_to_f16(s.d).to_le_bytes());
}

/// Q6_K (210 bytes per 256 values: ql[128], qh[64], scales[16] as i8, d).
pub(crate) fn q6_k(x: &[f32], imp: Option<&[f32]>, opts: &EncodeOpts, out: &mut [u8]) {
    let mut wbuf = [0f32; 256];
    for (sb, (xs, o)) in x
        .chunks_exact(256)
        .zip(out.chunks_exact_mut(210))
        .enumerate()
    {
        block_weights(
            opts,
            xs,
            imp.map(|v| &v[sb * 256..sb * 256 + 256]),
            &mut wbuf,
        );
        let mut codes = [0i8; 256];
        let sc = q6k_fit(xs, &wbuf, opts.refine, &mut codes);
        q6k_pack(&sc, &codes, o);
    }
}
