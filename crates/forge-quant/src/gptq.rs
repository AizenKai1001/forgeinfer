//! Error-compensating quantization (GPTQ, Frantar et al. 2022) for the K-quant formats.
//!
//! A linear layer computes y = W x. Rounding the weights of a row one column at a time, the rounding error of column c is not simply dropped: the remaining (not yet rounded) weights of the row
//! are corrected so that the layer output for the *calibration inputs* changes as little as possible. With H = E[x x^T] the second moment of the layer's inputs and U the upper Cholesky factor of
//! H^{-1} (H^{-1} = U^T U), rounding column c with error e = (w_c - q_c) / U[c][c] changes the remaining weights by  w_k -= e * U[c][k]  (k > c). The scales of every sub-block are fitted by the
//! plain weighted fit (importance = the diagonal of H) at the start of each super-block on the weights as corrected so far, and stay fixed while its 256 columns are rounded.
//!
//! The cost is rows * cols^2 / 2 multiply-adds per matrix plus two O(cols^3 / 3) factorizations of H (Cholesky of the reversed matrix and a triangular inverse), all in plain Rust on all cores.

use crate::kquant::{block_weights, q4k_fit, q4k_pack, q6k_fit, q6k_pack};
use crate::{encode_matrix, EncodeOpts, Weighting};
use forge_core::{Error, Result};
use forge_gguf::DType;

#[derive(Clone, Debug)]
pub struct GptqOpts {
    /// dampening of the Hessian: lambda = damp * mean(diag(H)) is added to the diagonal
    pub damp: f32,
    /// worker threads (0: all cores)
    pub threads: usize,
}

impl Default for GptqOpts {
    fn default() -> Self {
        GptqOpts {
            damp: 0.01,
            threads: 0,
        }
    }
}

fn n_threads(t: usize) -> usize {
    if t == 0 {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
    } else {
        t
    }
}

/// dot product of two equal-length slices with four accumulators (portable version)
#[inline]
fn dot64_scalar(a: &[f64], b: &[f64]) -> f64 {
    let (mut s0, mut s1, mut s2, mut s3) = (0f64, 0f64, 0f64, 0f64);
    let n = a.len() & !3;
    let mut i = 0;
    while i < n {
        s0 += a[i] * b[i];
        s1 += a[i + 1] * b[i + 1];
        s2 += a[i + 2] * b[i + 2];
        s3 += a[i + 3] * b[i + 3];
        i += 4;
    }
    while i < a.len() {
        s0 += a[i] * b[i];
        i += 1;
    }
    (s0 + s1) + (s2 + s3)
}

/// AVX2 + FMA version: four 4-lane accumulators.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot64_avx2(a: &[f64], b: &[f64]) -> f64 {
    use std::arch::x86_64::*;
    let n = a.len();
    let (pa, pb) = (a.as_ptr(), b.as_ptr());
    let (mut s0, mut s1, mut s2, mut s3) = (
        _mm256_setzero_pd(),
        _mm256_setzero_pd(),
        _mm256_setzero_pd(),
        _mm256_setzero_pd(),
    );
    let mut i = 0;
    while i + 16 <= n {
        s0 = _mm256_fmadd_pd(_mm256_loadu_pd(pa.add(i)), _mm256_loadu_pd(pb.add(i)), s0);
        s1 = _mm256_fmadd_pd(
            _mm256_loadu_pd(pa.add(i + 4)),
            _mm256_loadu_pd(pb.add(i + 4)),
            s1,
        );
        s2 = _mm256_fmadd_pd(
            _mm256_loadu_pd(pa.add(i + 8)),
            _mm256_loadu_pd(pb.add(i + 8)),
            s2,
        );
        s3 = _mm256_fmadd_pd(
            _mm256_loadu_pd(pa.add(i + 12)),
            _mm256_loadu_pd(pb.add(i + 12)),
            s3,
        );
        i += 16;
    }
    while i + 4 <= n {
        s0 = _mm256_fmadd_pd(_mm256_loadu_pd(pa.add(i)), _mm256_loadu_pd(pb.add(i)), s0);
        i += 4;
    }
    let s = _mm256_add_pd(_mm256_add_pd(s0, s1), _mm256_add_pd(s2, s3));
    let mut lanes = [0f64; 4];
    _mm256_storeu_pd(lanes.as_mut_ptr(), s);
    let mut t = (lanes[0] + lanes[1]) + (lanes[2] + lanes[3]);
    while i < n {
        t += *pa.add(i) * *pb.add(i);
        i += 1;
    }
    t
}

#[inline]
fn dot64(a: &[f64], b: &[f64]) -> f64 {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            // SAFETY: the CPU supports the features the function is compiled for
            return unsafe { dot64_avx2(a, b) };
        }
    }
    dot64_scalar(a, b)
}

/// In-place lower Cholesky factor (row-major n x n; the strict upper triangle is left alone). Blocked: the panel solve and the trailing update run on all threads.
fn cholesky_lower(a: &mut [f64], n: usize, threads: usize) -> Result<()> {
    const NB: usize = 64;
    let mut k = 0;
    while k < n {
        let kb = (n - k).min(NB);
        // 1. the diagonal block
        for j in k..k + kb {
            let mut d = a[j * n + j];
            for p in k..j {
                d -= a[j * n + p] * a[j * n + p];
            }
            #[allow(clippy::neg_cmp_op_on_partial_ord)] // NaN must take the error path too
            if !(d > 0.0) || !d.is_finite() {
                return Err(Error::format(format!(
                    "the Hessian is not positive definite (pivot {j}: {d})"
                )));
            }
            let l = d.sqrt();
            a[j * n + j] = l;
            for i in j + 1..k + kb {
                let mut s = a[i * n + j];
                for p in k..j {
                    s -= a[i * n + p] * a[j * n + p];
                }
                a[i * n + j] = s / l;
            }
        }
        if k + kb < n {
            // the diagonal block, copied (read-only below)
            let mut blk = vec![0f64; kb * kb];
            for i in 0..kb {
                blk[i * kb..i * kb + i + 1]
                    .copy_from_slice(&a[(k + i) * n + k..(k + i) * n + k + i + 1]);
            }
            let rows_below = n - k - kb;
            let per = rows_below.div_ceil(threads);
            // 2. the panel below the block: L[i][k..k+kb] = A[i][k..k+kb] * inv(L_blk^T)
            {
                let tail = &mut a[(k + kb) * n..];
                std::thread::scope(|s| {
                    for chunk in tail.chunks_mut(per * n) {
                        let blk = &blk;
                        s.spawn(move || {
                            for row in chunk.chunks_mut(n) {
                                for j in 0..kb {
                                    let mut v = row[k + j];
                                    for p in 0..j {
                                        v -= row[k + p] * blk[j * kb + p];
                                    }
                                    row[k + j] = v / blk[j * kb + j];
                                }
                            }
                        });
                    }
                });
            }
            // 3. the trailing update A[i][j] -= sum_p L[i][p] L[j][p] over the panel columns, j <= i
            let mut panel = vec![0f64; rows_below * kb];
            for i in 0..rows_below {
                panel[i * kb..(i + 1) * kb]
                    .copy_from_slice(&a[(k + kb + i) * n + k..(k + kb + i) * n + k + kb]);
            }
            let tail = &mut a[(k + kb) * n..];
            let panel = &panel;
            std::thread::scope(|s| {
                // rows are handed out interleaved so that the long (late) rows spread over the threads
                let mut groups: Vec<Vec<(usize, &mut [f64])>> =
                    (0..threads).map(|_| Vec::new()).collect();
                for (i, row) in tail.chunks_mut(n).enumerate() {
                    groups[i % threads].push((i, row));
                }
                for g in groups {
                    s.spawn(move || {
                        for (i, row) in g {
                            let pi = &panel[i * kb..(i + 1) * kb];
                            for j in 0..=i {
                                row[k + kb + j] -= dot64(pi, &panel[j * kb..(j + 1) * kb]);
                            }
                        }
                    });
                }
            });
        }
        k += kb;
    }
    Ok(())
}

/// U (row-major n x n, upper triangular, f32) with H^{-1} = U^T U, from the symmetric positive definite H (row-major f64, destroyed).
/// With J the reversal, J H J = M M^T (M lower), so H = A A^T with A = J M J upper, H^{-1} = (A^{-1})^T A^{-1}, and A^{-1} = J M^{-1} J.
pub fn inverse_cholesky_upper(h: &mut [f64], n: usize, threads: usize) -> Result<Vec<f32>> {
    let threads = n_threads(threads);
    // reverse rows and columns
    for i in 0..n / 2 {
        for j in 0..n {
            let (a, b) = (i * n + j, (n - 1 - i) * n + (n - 1 - j));
            h.swap(a, b);
        }
    }
    if n % 2 == 1 {
        let mid = n / 2;
        for j in 0..n / 2 {
            h.swap(mid * n + j, mid * n + (n - 1 - j));
        }
    }
    cholesky_lower(h, n, threads)?;
    // M^{-1} (lower), column by column
    let m: &[f64] = h;
    let mut minv_cols: Vec<Vec<f64>> = vec![Vec::new(); n];
    let per = n.div_ceil(threads);
    std::thread::scope(|s| {
        // interleaved columns: the cost of column j is (n - j)^2 / 2
        let mut groups: Vec<Vec<(usize, &mut Vec<f64>)>> =
            (0..threads).map(|_| Vec::new()).collect();
        for (j, c) in minv_cols.iter_mut().enumerate() {
            groups[j % threads].push((j, c));
        }
        let _ = per;
        for g in groups {
            s.spawn(move || {
                for (j, out) in g {
                    let mut y = vec![0f64; n];
                    y[j] = 1.0 / m[j * n + j];
                    for i in j + 1..n {
                        let row = &m[i * n + j..i * n + i];
                        let s = dot64(row, &y[j..i]);
                        y[i] = -s / m[i * n + i];
                    }
                    *out = y;
                }
            });
        }
    });
    // U[i][j] = Minv[n-1-i][n-1-j] for j >= i; Minv[r][c] = minv_cols[c][r]
    let mut u = vec![0f32; n * n];
    for i in 0..n {
        for j in i..n {
            u[i * n + j] = minv_cols[n - 1 - j][n - 1 - i] as f32;
        }
    }
    Ok(u)
}

/// tile[r][be..] -= sum_j errs[r][j] * u[bs + j][be..] for the rows r < nr of the tile: the rounding errors of one block of 128 columns reaching the columns after it.
/// Register-blocked: 4 rows x 16 columns of the tile stay in vector registers while the 128 rows of U stream past.
#[inline(always)]
fn propagate_impl<const FMA: bool>(
    tile: &mut [f32],
    cols: usize,
    nr: usize,
    errs: &[[f32; 128]; 8],
    u: &[f32],
    bs: usize,
    be: usize,
) {
    for r0 in (0..nr).step_by(4) {
        let rr = (nr - r0).min(4);
        let mut c0 = be;
        while c0 < cols {
            let w = (cols - c0).min(16);
            let mut acc = [[0f32; 16]; 4];
            if w == 16 {
                for j in 0..128 {
                    let ur = &u[(bs + j) * cols + c0..(bs + j) * cols + c0 + 16];
                    for r in 0..rr {
                        let e = errs[r0 + r][j];
                        for k in 0..16 {
                            acc[r][k] = if FMA {
                                e.mul_add(ur[k], acc[r][k])
                            } else {
                                acc[r][k] + e * ur[k]
                            };
                        }
                    }
                }
            } else {
                for j in 0..128 {
                    let ur = &u[(bs + j) * cols + c0..(bs + j) * cols + c0 + w];
                    for r in 0..rr {
                        let e = errs[r0 + r][j];
                        for k in 0..w {
                            acc[r][k] += e * ur[k];
                        }
                    }
                }
            }
            for r in 0..rr {
                let base = (r0 + r) * cols + c0;
                for k in 0..w {
                    tile[base + k] -= acc[r][k];
                }
            }
            c0 += 16;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn propagate_avx2(
    tile: &mut [f32],
    cols: usize,
    nr: usize,
    errs: &[[f32; 128]; 8],
    u: &[f32],
    bs: usize,
    be: usize,
) {
    propagate_impl::<true>(tile, cols, nr, errs, u, bs, be)
}

fn propagate(
    tile: &mut [f32],
    cols: usize,
    nr: usize,
    errs: &[[f32; 128]; 8],
    u: &[f32],
    bs: usize,
    be: usize,
) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            // SAFETY: the CPU supports the features the function is compiled for
            unsafe { propagate_avx2(tile, cols, nr, errs, u, bs, be) };
            return;
        }
    }
    propagate_impl::<false>(tile, cols, nr, errs, u, bs, be)
}

/// GPTQ for one matrix [rows][cols] (row-major f32) and the input second-moment matrix `hess` ([cols][cols], row-major, symmetric, E[x x^T] over the calibration tokens).
/// Q4_K, Q5_K and Q6_K are quantized with error compensation; other formats fall back to the plain encoder with the Hessian's diagonal as the importance.
pub fn gptq_matrix(
    dtype: DType,
    rows: usize,
    cols: usize,
    w: &[f32],
    hess: &[f32],
    enc: &EncodeOpts,
    g: &GptqOpts,
) -> Result<Vec<u8>> {
    if hess.len() != cols * cols {
        return Err(Error::format(format!(
            "Hessian of {} values for {cols} columns",
            hess.len()
        )));
    }
    let diag: Vec<f32> = (0..cols).map(|i| hess[i * cols + i]).collect();
    if !matches!(dtype, DType::Q4_K | DType::Q5_K | DType::Q6_K) || !cols.is_multiple_of(256) {
        let o = EncodeOpts {
            weighting: Weighting::Importance,
            ..enc.clone()
        };
        return encode_matrix(dtype, rows, cols, w, Some(&diag), &o);
    }
    let threads = n_threads(g.threads);
    // dead columns (inputs that are always zero) get a unit Hessian entry and their weights are irrelevant
    let mean_diag = diag.iter().map(|&v| v as f64).sum::<f64>() / cols as f64;
    let damp = g.damp as f64 * mean_diag.max(1e-12);
    let mut h = vec![0f64; cols * cols];
    let mut dead = vec![false; cols];
    for i in 0..cols {
        for j in 0..cols {
            h[i * cols + j] = hess[i * cols + j] as f64;
        }
        if diag[i] <= 0.0 {
            dead[i] = true;
            h[i * cols + i] = 1.0;
        }
        h[i * cols + i] += damp;
    }
    let u = inverse_cholesky_upper(&mut h, cols, threads)?;
    drop(h);
    let imp: Vec<f32> = diag.iter().map(|&v| v.max(0.0)).collect();
    let opts = EncodeOpts {
        weighting: Weighting::Importance,
        refine: false,
        threads: 1,
    };
    let rb = dtype.bytes_for(cols);
    let bb = dtype.block_bytes();
    let mut out = vec![0u8; rows * rb];
    let per = rows.div_ceil(threads);
    const R: usize = 8; // rows rounded together (they share the loads of U)
    std::thread::scope(|s| {
        for (ci, ochunk) in out.chunks_mut(per * rb).enumerate() {
            let (u, imp, dead, opts) = (&u, &imp, &dead, &opts);
            s.spawn(move || {
                let row0 = ci * per;
                let nrows = ochunk.len() / rb;
                let mut tile = vec![0f32; R * cols];
                let mut codes = vec![0i16; R * cols];
                let mut wbuf = [0f32; 256];
                for t0 in (0..nrows).step_by(R) {
                    let nr = (nrows - t0).min(R);
                    for r in 0..nr {
                        tile[r * cols..(r + 1) * cols].copy_from_slice(
                            &w[(row0 + t0 + r) * cols..(row0 + t0 + r + 1) * cols],
                        );
                        for c in 0..cols {
                            if dead[c] {
                                tile[r * cols + c] = 0.0;
                            }
                        }
                    }
                    for sb in 0..cols / 256 {
                        // 1. the grids of this super-block, fitted to the weights as corrected so far
                        let mut s4 = [None; R];
                        let mut s6 = [None; R];
                        for r in 0..nr {
                            let xs = &tile[r * cols + sb * 256..r * cols + sb * 256 + 256];
                            block_weights(
                                opts,
                                xs,
                                Some(&imp[sb * 256..sb * 256 + 256]),
                                &mut wbuf,
                            );
                            if dtype == DType::Q6_K {
                                let mut cd = [0i8; 256];
                                s6[r] = Some(q6k_fit(xs, &wbuf, false, &mut cd));
                            } else {
                                let mut cd = [0u8; 256];
                                s4[r] = Some(q4k_fit(
                                    xs,
                                    &wbuf,
                                    if dtype == DType::Q4_K { 15 } else { 31 },
                                    false,
                                    &mut cd,
                                ));
                            }
                        }
                        // 2. round the 256 columns in two blocks of 128: inside a block each rounding error is passed on to the rest of the block at once; the block's errors reach the columns
                        //    beyond it afterwards as one small matrix product (the "lazy batch update" of GPTQ), which keeps the weights of the row tile in registers instead of re-reading them per column
                        for half in 0..2 {
                            let bs = sb * 256 + half * 128;
                            let be = bs + 128;
                            let mut errs = [[0f32; 128]; R];
                            for c in bs..be {
                                let jj = c - sb * 256;
                                let ucc = u[c * cols + c];
                                let inv_ucc = if ucc.abs() > 1e-20 && ucc.is_finite() {
                                    1.0 / ucc
                                } else {
                                    1.0
                                };
                                for r in 0..nr {
                                    let wv = tile[r * cols + c];
                                    let (q, deq) = if dtype == DType::Q6_K {
                                        let sc = s6[r].as_ref().unwrap();
                                        let sj = sc.d * sc.sc[jj / 16] as f32;
                                        let q = if sj != 0.0 {
                                            ((wv / sj).round() as i32).clamp(-32, 31)
                                        } else {
                                            0
                                        };
                                        (q, sj * q as f32)
                                    } else {
                                        let sc = s4[r].as_ref().unwrap();
                                        let (sj, mn) = (
                                            sc.d * sc.ls[jj / 32] as f32,
                                            sc.dmin * sc.lm[jj / 32] as f32,
                                        );
                                        let lv = if dtype == DType::Q4_K { 15 } else { 31 };
                                        let q = if sj > 0.0 {
                                            (((wv + mn) / sj).round() as i32).clamp(0, lv)
                                        } else {
                                            0
                                        };
                                        (q, sj * q as f32 - mn)
                                    };
                                    codes[r * cols + c] = q as i16;
                                    errs[r][c - bs] = (wv - deq) * inv_ucc;
                                }
                                // the rest of this block
                                let urow = &u[c * cols + c + 1..c * cols + be];
                                for r in 0..nr {
                                    let e = errs[r][c - bs];
                                    if e != 0.0 {
                                        for (d, uk) in tile[r * cols + c + 1..r * cols + be]
                                            .iter_mut()
                                            .zip(urow)
                                        {
                                            *d -= e * uk;
                                        }
                                    }
                                }
                            }
                            if be < cols {
                                propagate(&mut tile, cols, nr, &errs, u, bs, be);
                            }
                        }
                        // 3. the bytes
                        for r in 0..nr {
                            let o =
                                &mut ochunk[(t0 + r) * rb + sb * bb..(t0 + r) * rb + (sb + 1) * bb];
                            if dtype == DType::Q6_K {
                                let mut cd = [0i8; 256];
                                for j in 0..256 {
                                    cd[j] = codes[r * cols + sb * 256 + j] as i8;
                                }
                                q6k_pack(s6[r].as_ref().unwrap(), &cd, o);
                            } else {
                                let mut cd = [0u8; 256];
                                for j in 0..256 {
                                    cd[j] = codes[r * cols + sb * 256 + j] as u8;
                                }
                                q4k_pack(s4[r].as_ref().unwrap(), &cd, dtype == DType::Q5_K, o);
                            }
                        }
                    }
                }
            });
        }
    });
    Ok(out)
}
