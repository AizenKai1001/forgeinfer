//! GPTQ: the factorization behind it, and that error compensation lowers the error of the layer's OUTPUT on inputs it was calibrated for.

use forge_gguf::{quant::dequantize_into, DType};
use forge_quant::{
    encode_matrix, gptq_matrix, inverse_cholesky_upper, EncodeOpts, GptqOpts, Weighting,
};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn uniform(&mut self) -> f32 {
        ((self.next() >> 11) as f64 / (1u64 << 53) as f64) as f32
    }
    fn normal(&mut self) -> f32 {
        let (u, v) = (self.uniform().max(1e-9), self.uniform());
        (-2.0 * u.ln()).sqrt() * (2.0 * std::f32::consts::PI * v).cos()
    }
}

/// inputs with structure: low-rank signal plus noise, with a few directions much louder than the rest
fn inputs(rng: &mut Rng, tokens: usize, dim: usize, mix: &[f32], rank: usize) -> Vec<f32> {
    let mut x = vec![0f32; tokens * dim];
    for t in 0..tokens {
        let z: Vec<f32> = (0..rank)
            .map(|k| rng.normal() * (1.0 + 4.0 * (k % 5 == 0) as u32 as f32))
            .collect();
        for d in 0..dim {
            let mut v = 0.1 * rng.normal();
            for k in 0..rank {
                v += z[k] * mix[k * dim + d];
            }
            x[t * dim + d] = v;
        }
    }
    x
}

fn second_moment(x: &[f32], tokens: usize, dim: usize) -> Vec<f32> {
    let mut h = vec![0f64; dim * dim];
    for t in 0..tokens {
        let r = &x[t * dim..(t + 1) * dim];
        for i in 0..dim {
            let ri = r[i] as f64;
            for j in 0..dim {
                h[i * dim + j] += ri * r[j] as f64;
            }
        }
    }
    h.iter().map(|v| (*v / tokens as f64) as f32).collect()
}

#[test]
fn inverse_cholesky_factor_inverts() {
    for n in [7usize, 64, 130] {
        let mut rng = Rng(11 + n as u64);
        // SPD matrix: A A^T + n I
        let a: Vec<f64> = (0..n * n).map(|_| rng.normal() as f64).collect();
        let mut h = vec![0f64; n * n];
        for i in 0..n {
            for j in 0..n {
                let mut s = 0f64;
                for k in 0..n {
                    s += a[i * n + k] * a[j * n + k];
                }
                h[i * n + j] = s + if i == j { n as f64 } else { 0.0 };
            }
        }
        let h0 = h.clone();
        let u = inverse_cholesky_upper(&mut h, n, 3).unwrap();
        // U^T U should be H^{-1}: (U^T U) H = I
        let mut worst = 0f64;
        for i in 0..n {
            for j in 0..n {
                let mut ut_u = 0f64; // sum_k U[k][i] U[k][j] H[?]...  compute (U^T U H)[i][j]
                for k in 0..n {
                    // (U^T U)[i][k]
                    let mut a_ik = 0f64;
                    for m in 0..=i.min(k) {
                        a_ik += u[m * n + i] as f64 * u[m * n + k] as f64;
                    }
                    ut_u += a_ik * h0[k * n + j];
                }
                let want = if i == j { 1.0 } else { 0.0 };
                worst = worst.max((ut_u - want).abs());
            }
        }
        eprintln!("n = {n}: max |U^T U H - I| = {worst:.2e}");
        assert!(worst < 2e-3, "n = {n}: {worst}");
        // upper triangular
        for i in 0..n {
            for j in 0..i {
                assert_eq!(u[i * n + j], 0.0);
            }
        }
    }
}

/// mean squared output error of the quantized matrix on inputs `x`, relative to the mean squared output
fn output_error(
    w: &[f32],
    back: &[f32],
    rows: usize,
    cols: usize,
    x: &[f32],
    tokens: usize,
) -> f64 {
    let (mut e, mut n) = (0f64, 0f64);
    for t in 0..tokens {
        let xr = &x[t * cols..(t + 1) * cols];
        for r in 0..rows {
            let (mut y, mut yq) = (0f64, 0f64);
            for c in 0..cols {
                y += w[r * cols + c] as f64 * xr[c] as f64;
                yq += back[r * cols + c] as f64 * xr[c] as f64;
            }
            e += (y - yq) * (y - yq);
            n += y * y;
        }
    }
    e / n
}

#[test]
fn error_compensation_lowers_the_output_error() {
    let (rows, cols, rank) = (48usize, 512usize, 40usize);
    let mut rng = Rng(5);
    let mix: Vec<f32> = (0..rank * cols)
        .map(|_| rng.normal() / (rank as f32).sqrt())
        .collect();
    let w: Vec<f32> = (0..rows * cols).map(|_| rng.normal() * 0.02).collect();
    let calib = inputs(&mut rng, 3000, cols, &mix, rank);
    let held_out = inputs(&mut rng, 800, cols, &mix, rank);
    let h = second_moment(&calib, 3000, cols);
    let diag: Vec<f32> = (0..cols).map(|i| h[i * cols + i]).collect();
    for dt in [DType::Q4_K, DType::Q5_K, DType::Q6_K] {
        let uni = encode_matrix(dt, rows, cols, &w, None, &EncodeOpts::default()).unwrap();
        let imp = encode_matrix(
            dt,
            rows,
            cols,
            &w,
            Some(&diag),
            &EncodeOpts {
                weighting: Weighting::Importance,
                ..Default::default()
            },
        )
        .unwrap();
        let gp = gptq_matrix(
            dt,
            rows,
            cols,
            &w,
            &h,
            &EncodeOpts::default(),
            &GptqOpts::default(),
        )
        .unwrap();
        assert_eq!(gp.len(), uni.len());
        let dec = |b: &[u8]| {
            let mut back = vec![0f32; rows * cols];
            dequantize_into(dt, b, &mut back).unwrap();
            back
        };
        let (eu, ei, eg) = (
            output_error(&w, &dec(&uni), rows, cols, &held_out, 800),
            output_error(&w, &dec(&imp), rows, cols, &held_out, 800),
            output_error(&w, &dec(&gp), rows, cols, &held_out, 800),
        );
        eprintln!("{}: output error on held-out inputs: uniform {eu:.4e}, importance {ei:.4e}, GPTQ {eg:.4e} ({:.1}x lower than uniform)", dt.name(), eu / eg);
        assert!(
            eg < ei && eg < eu,
            "{}: GPTQ {eg} is not below importance {ei} / uniform {eu}",
            dt.name()
        );
    }
}

/// Speed on a layer-sized problem (run with --ignored --nocapture).
#[test]
#[ignore]
fn gptq_speed() {
    let (rows, cols, rank) = (2048usize, 3584usize, 200usize);
    let mut rng = Rng(9);
    let mix: Vec<f32> = (0..rank * cols)
        .map(|_| rng.normal() / (rank as f32).sqrt())
        .collect();
    let w: Vec<f32> = (0..rows * cols).map(|_| rng.normal() * 0.02).collect();
    let calib = inputs(&mut rng, 600, cols, &mix, rank);
    // second moments of 600 tokens in 3584 dimensions are rank deficient: the damping makes them invertible
    let mut h = vec![0f32; cols * cols];
    for t in 0..600 {
        let r = &calib[t * cols..(t + 1) * cols];
        for i in 0..cols {
            let ri = r[i] / 600.0;
            for j in 0..cols {
                h[i * cols + j] += ri * r[j];
            }
        }
    }
    let t0 = std::time::Instant::now();
    let out = gptq_matrix(
        DType::Q4_K,
        rows,
        cols,
        &w,
        &h,
        &EncodeOpts::default(),
        &GptqOpts::default(),
    )
    .unwrap();
    let dt = t0.elapsed().as_secs_f64();
    let macs = rows as f64 * cols as f64 * cols as f64 / 2.0;
    eprintln!(
        "GPTQ {rows} x {cols}: {dt:.2} s = {:.2} GMAC/s for the rounding ({} bytes)",
        macs / dt / 1e9,
        out.len()
    );
}
