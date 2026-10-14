//! The one hot loop of the encoder: the dot product of two f32 rows (AVX2 + FMA when the CPU has them, eight lanes of plain code otherwise).

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let n = a.len();
    let (pa, pb) = (a.as_ptr(), b.as_ptr());
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut acc2 = _mm256_setzero_ps();
    let mut acc3 = _mm256_setzero_ps();
    let mut i = 0;
    while i + 32 <= n {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(pa.add(i)), _mm256_loadu_ps(pb.add(i)), acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(pa.add(i + 8)), _mm256_loadu_ps(pb.add(i + 8)), acc1);
        acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(pa.add(i + 16)), _mm256_loadu_ps(pb.add(i + 16)), acc2);
        acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(pa.add(i + 24)), _mm256_loadu_ps(pb.add(i + 24)), acc3);
        i += 32;
    }
    while i + 8 <= n {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(pa.add(i)), _mm256_loadu_ps(pb.add(i)), acc0);
        i += 8;
    }
    let acc = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
    let mut lanes = [0f32; 8];
    _mm256_storeu_ps(lanes.as_mut_ptr(), acc);
    let mut s = ((lanes[0] + lanes[4]) + (lanes[1] + lanes[5])) + ((lanes[2] + lanes[6]) + (lanes[3] + lanes[7]));
    while i < n {
        s += *pa.add(i) * *pb.add(i);
        i += 1;
    }
    s
}

/// Four rows of `x` against two rows of `w` (all of length k, a multiple of 8): out[r][c] = x[r] . w[c]. Every weight vector is loaded once for four tokens and every token vector once for two outputs,
/// eight FMAs per six loads, which is what it takes to get near the FMA rate of the core.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_4x2_avx2(x: &[f32], w0: &[f32], w1: &[f32], k: usize, out: &mut [[f32; 2]; 4]) {
    use std::arch::x86_64::*;
    let (px, p0, p1) = (x.as_ptr(), w0.as_ptr(), w1.as_ptr());
    let mut a00 = _mm256_setzero_ps();
    let mut a01 = _mm256_setzero_ps();
    let mut a10 = _mm256_setzero_ps();
    let mut a11 = _mm256_setzero_ps();
    let mut a20 = _mm256_setzero_ps();
    let mut a21 = _mm256_setzero_ps();
    let mut a30 = _mm256_setzero_ps();
    let mut a31 = _mm256_setzero_ps();
    let mut i = 0;
    while i < k {
        let v0 = _mm256_loadu_ps(p0.add(i));
        let v1 = _mm256_loadu_ps(p1.add(i));
        let x0 = _mm256_loadu_ps(px.add(i));
        a00 = _mm256_fmadd_ps(x0, v0, a00);
        a01 = _mm256_fmadd_ps(x0, v1, a01);
        let x1 = _mm256_loadu_ps(px.add(k + i));
        a10 = _mm256_fmadd_ps(x1, v0, a10);
        a11 = _mm256_fmadd_ps(x1, v1, a11);
        let x2 = _mm256_loadu_ps(px.add(2 * k + i));
        a20 = _mm256_fmadd_ps(x2, v0, a20);
        a21 = _mm256_fmadd_ps(x2, v1, a21);
        let x3 = _mm256_loadu_ps(px.add(3 * k + i));
        a30 = _mm256_fmadd_ps(x3, v0, a30);
        a31 = _mm256_fmadd_ps(x3, v1, a31);
        i += 8;
    }
    let hsum = |v: __m256| -> f32 {
        let mut lanes = [0f32; 8];
        _mm256_storeu_ps(lanes.as_mut_ptr(), v);
        ((lanes[0] + lanes[4]) + (lanes[1] + lanes[5])) + ((lanes[2] + lanes[6]) + (lanes[3] + lanes[7]))
    };
    out[0] = [hsum(a00), hsum(a01)];
    out[1] = [hsum(a10), hsum(a11)];
    out[2] = [hsum(a20), hsum(a21)];
    out[3] = [hsum(a30), hsum(a31)];
}

/// Rows `r0..r1` of the product of `x` (n rows of k values) with the transposed `w` (rows of k values): part[(o - r0) * n + t] = x[t] . w[o]. Uses the 4x2 kernel where it fits.
pub fn matmul_rows(x: &[f32], n: usize, k: usize, w: &[f32], r0: usize, r1: usize, avx2: bool, part: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if avx2 && k % 8 == 0 {
            let mut o = r0;
            while o + 2 <= r1 {
                let (w0, w1) = (&w[o * k..(o + 1) * k], &w[(o + 1) * k..(o + 2) * k]);
                let mut t = 0;
                while t + 4 <= n {
                    let mut out = [[0f32; 2]; 4];
                    // SAFETY: `avx2` is true only when `has_avx2()` said so; the slices hold 4 rows of k values.
                    unsafe { dot_4x2_avx2(&x[t * k..(t + 4) * k], w0, w1, k, &mut out) };
                    for (r, pair) in out.iter().enumerate() {
                        part[(o - r0) * n + t + r] = pair[0];
                        part[(o + 1 - r0) * n + t + r] = pair[1];
                    }
                    t += 4;
                }
                while t < n {
                    part[(o - r0) * n + t] = dot(w0, &x[t * k..(t + 1) * k], true);
                    part[(o + 1 - r0) * n + t] = dot(w1, &x[t * k..(t + 1) * k], true);
                    t += 1;
                }
                o += 2;
            }
            if o < r1 {
                let w0 = &w[o * k..(o + 1) * k];
                for t in 0..n {
                    part[(o - r0) * n + t] = dot(w0, &x[t * k..(t + 1) * k], true);
                }
            }
            return;
        }
    }
    for o in r0..r1 {
        let wrow = &w[o * k..(o + 1) * k];
        for t in 0..n {
            part[(o - r0) * n + t] = dot(wrow, &x[t * k..(t + 1) * k], avx2);
        }
    }
}

fn dot_plain(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len() / 8 * 8;
    let mut acc = [0f32; 8];
    for (ca, cb) in a[..n].chunks_exact(8).zip(b[..n].chunks_exact(8)) {
        for k in 0..8 {
            acc[k] += ca[k] * cb[k];
        }
    }
    let mut s = ((acc[0] + acc[4]) + (acc[1] + acc[5])) + ((acc[2] + acc[6]) + (acc[3] + acc[7]));
    for k in n..a.len() {
        s += a[k] * b[k];
    }
    s
}

/// Does this CPU run the AVX2 + FMA path?
pub fn has_avx2() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// The dot product of two rows of the same length.
#[inline]
pub fn dot(a: &[f32], b: &[f32], avx2: bool) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "x86_64")]
    {
        if avx2 {
            // SAFETY: `avx2` is true only when `has_avx2()` said so.
            return unsafe { dot_avx2(a, b) };
        }
    }
    let _ = avx2;
    dot_plain(a, b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_blocked_product_matches_the_plain_one() {
        // sizes that leave a rest in every direction: an odd number of outputs, tokens not a multiple of four, k a multiple of 8 and not
        for (n, k, rows) in [(1usize, 8usize, 1usize), (3, 16, 5), (4, 24, 2), (7, 64, 9), (9, 768, 13), (10, 33, 4)] {
            let x: Vec<f32> = (0..n * k).map(|i| ((i * 31 % 97) as f32 - 48.0) / 40.0).collect();
            let w: Vec<f32> = (0..rows * k).map(|i| ((i * 17 % 83) as f32 - 41.0) / 35.0).collect();
            let mut fast = vec![0f32; rows * n];
            matmul_rows(&x, n, k, &w, 0, rows, has_avx2(), &mut fast);
            for o in 0..rows {
                for t in 0..n {
                    let truth: f64 = (0..k).map(|i| w[o * k + i] as f64 * x[t * k + i] as f64).sum();
                    assert!((fast[o * n + t] as f64 - truth).abs() < 1e-3 * (1.0 + truth.abs()), "n {n} k {k} rows {rows}: o {o} t {t}: {} vs {truth}", fast[o * n + t]);
                }
            }
            // a sub-range of the outputs gives the same numbers
            if rows > 2 {
                let mut part = vec![0f32; 2 * n];
                matmul_rows(&x, n, k, &w, 1, 3, has_avx2(), &mut part);
                assert_eq!(&part[..], &fast[n..3 * n]);
            }
        }
    }

    #[test]
    fn both_paths_agree_with_the_plain_sum() {
        for n in [1usize, 7, 8, 9, 31, 32, 33, 64, 100, 768, 3072] {
            let a: Vec<f32> = (0..n).map(|i| ((i * 37 % 101) as f32 - 50.0) / 25.0).collect();
            let b: Vec<f32> = (0..n).map(|i| ((i * 53 % 89) as f32 - 44.0) / 30.0).collect();
            let truth: f64 = a.iter().zip(&b).map(|(x, y)| *x as f64 * *y as f64).sum();
            assert!((dot(&a, &b, false) as f64 - truth).abs() < 1e-3 * (1.0 + truth.abs()), "plain, n {n}");
            assert!((dot(&a, &b, has_avx2()) as f64 - truth).abs() < 1e-3 * (1.0 + truth.abs()), "avx2, n {n}");
        }
    }
}
