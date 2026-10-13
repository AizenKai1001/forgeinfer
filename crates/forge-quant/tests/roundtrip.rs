//! The encoders against the (golden-tested) decoders of forge-gguf: what is encoded must decode to a close approximation, the importance weighting must pay off where the weights say it
//! should, and the refinement must not make things worse.

use forge_gguf::{quant::dequantize_into, DType};
use forge_quant::{encode_matrix, relative_rmse, EncodeOpts, Weighting};

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

/// weights like a trained layer: Gaussian with a few outliers and a different scale per row
fn matrix(rows: usize, cols: usize, seed: u64) -> Vec<f32> {
    let mut r = Rng(seed);
    let mut w = Vec::with_capacity(rows * cols);
    for i in 0..rows {
        let s = 0.01 * (1.0 + (i % 7) as f32 * 0.3);
        for _ in 0..cols {
            let mut v = r.normal() * s;
            if r.uniform() < 0.002 {
                v *= 6.0;
            }
            w.push(v);
        }
    }
    w
}

#[test]
fn every_format_decodes_close_to_the_input() {
    let (rows, cols) = (16, 1024);
    let w = matrix(rows, cols, 7);
    let opts = EncodeOpts::default();
    // (format, bound on the relative rms error for this kind of data)
    for (dt, bound) in [
        (DType::Q8_0, 0.008),
        (DType::Q6_K, 0.02),
        (DType::Q5_K, 0.05),
        (DType::Q4_K, 0.10),
        (DType::Q5_0, 0.06),
        (DType::Q4_0, 0.12),
    ] {
        let enc = encode_matrix(dt, rows, cols, &w, None, &opts).unwrap();
        assert_eq!(enc.len(), dt.bytes_for(rows * cols));
        let e = relative_rmse(dt, rows, cols, &w, &enc).unwrap();
        eprintln!(
            "{:6} relative rms error {:.4} ({} bits per weight)",
            dt.name(),
            e,
            enc.len() as f64 * 8.0 / (rows * cols) as f64
        );
        assert!(e < bound, "{}: error {e} above {bound}", dt.name());
    }
}

#[test]
fn zero_and_constant_rows_roundtrip() {
    let opts = EncodeOpts::default();
    for dt in [
        DType::Q8_0,
        DType::Q6_K,
        DType::Q5_K,
        DType::Q4_K,
        DType::Q5_0,
        DType::Q4_0,
    ] {
        let cols = 512;
        let mut w = vec![0f32; 2 * cols];
        for v in w[cols..].iter_mut() {
            *v = -0.125;
        }
        let enc = encode_matrix(dt, 2, cols, &w, None, &opts).unwrap();
        let mut back = vec![1f32; 2 * cols];
        dequantize_into(dt, &enc, &mut back).unwrap();
        assert!(
            back[..cols].iter().all(|v| *v == 0.0),
            "{}: zeros",
            dt.name()
        );
        let err = back[cols..]
            .iter()
            .map(|v| (v + 0.125).abs())
            .fold(0f32, f32::max);
        assert!(err < 0.01, "{}: constant row off by {err}", dt.name());
    }
}

#[test]
fn refinement_never_hurts() {
    let (rows, cols) = (16, 1024);
    let w = matrix(rows, cols, 11);
    for dt in [DType::Q4_K, DType::Q5_K, DType::Q6_K] {
        let a = encode_matrix(
            dt,
            rows,
            cols,
            &w,
            None,
            &EncodeOpts {
                refine: false,
                ..Default::default()
            },
        )
        .unwrap();
        let b = encode_matrix(
            dt,
            rows,
            cols,
            &w,
            None,
            &EncodeOpts {
                refine: true,
                ..Default::default()
            },
        )
        .unwrap();
        let (ea, eb) = (
            relative_rmse(dt, rows, cols, &w, &a).unwrap(),
            relative_rmse(dt, rows, cols, &w, &b).unwrap(),
        );
        eprintln!(
            "{}: error without refinement {ea:.5}, with {eb:.5}",
            dt.name()
        );
        assert!(eb <= ea * 1.0001);
    }
}

/// importance-weighted squared error of an encoding
fn weighted_err(dt: DType, rows: usize, cols: usize, w: &[f32], enc: &[u8], imp: &[f32]) -> f64 {
    let mut back = vec![0f32; rows * cols];
    dequantize_into(dt, enc, &mut back).unwrap();
    let mut e = 0f64;
    for r in 0..rows {
        for c in 0..cols {
            e += imp[c] as f64 * ((w[r * cols + c] - back[r * cols + c]) as f64).powi(2);
        }
    }
    e
}

#[test]
fn importance_weighting_lowers_the_weighted_error() {
    let (rows, cols) = (32, 1024);
    let w = matrix(rows, cols, 3);
    let mut r = Rng(5);
    // a few input columns matter a hundred times more than the rest
    let imp: Vec<f32> = (0..cols)
        .map(|_| {
            if r.uniform() < 0.05 {
                100.0
            } else {
                r.uniform() + 0.05
            }
        })
        .collect();
    for dt in [DType::Q4_K, DType::Q5_K, DType::Q6_K] {
        let uni = encode_matrix(dt, rows, cols, &w, Some(&imp), &EncodeOpts::default()).unwrap();
        let wei = encode_matrix(
            dt,
            rows,
            cols,
            &w,
            Some(&imp),
            &EncodeOpts {
                weighting: Weighting::Importance,
                ..Default::default()
            },
        )
        .unwrap();
        let (eu, ew) = (
            weighted_err(dt, rows, cols, &w, &uni, &imp),
            weighted_err(dt, rows, cols, &w, &wei, &imp),
        );
        eprintln!("{}: importance-weighted error uniform {eu:.4e}, importance fit {ew:.4e} ({:.1} % lower)", dt.name(), 100.0 * (1.0 - ew / eu));
        assert!(
            ew < eu,
            "{}: the importance fit is not better ({ew} vs {eu})",
            dt.name()
        );
    }
}
