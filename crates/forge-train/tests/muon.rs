//! Muon's Newton-Schulz orthogonalisation on the GPU (fp16 operands, fp32 accumulation) against the float64 executable spec in tools/ref_muon.py, and the properties that matter for training:
//! the output is approximately orthogonal (singular values in the documented band), the tall and wide orientations agree, and a full optimizer step moves the weights.

use forge_train::{TrainConfig, Trainer};
use std::path::PathBuf;

fn golden(name: &str) -> (usize, usize, Vec<f32>, Vec<f32>) {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/golden/muon")
        .join(format!("{name}.f32"));
    let raw = std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let r = u32::from_le_bytes(raw[0..4].try_into().unwrap()) as usize;
    let c = u32::from_le_bytes(raw[4..8].try_into().unwrap()) as usize;
    let f = |from: usize| -> Vec<f32> {
        raw[from..from + r * c * 4]
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect()
    };
    (r, c, f(8), f(8 + r * c * 4))
}

fn small_trainer() -> Option<Trainer> {
    if forge_cuda::Device::count().unwrap_or(0) == 0 {
        eprintln!("skipped: no CUDA device");
        return None;
    }
    // the model's own matrices size the Muon scratch: with e = 64, ff = 192, 4 heads x 64 = 256 q dims, 1 kv head (64) the largest is 256 x 64 and 192 x 64
    let cfg = TrainConfig {
        n_layer: 1,
        n_embd: 64,
        n_head: 1,
        n_kv: 1,
        head_dim: 64,
        n_ff: 192,
        vocab: 128,
        seq_len: 64,
        batch: 1,
        qk_norm: true,
        rope_theta: 10000.0,
        eps: 1e-5,
        ce_chunk: 64,
        ..TrainConfig::default()
    };
    Some(Trainer::new(cfg, 0).unwrap())
}

/// singular values of x (rows x cols) via the eigenvalues of the Gram matrix with a plain Jacobi sweep (small sizes only)
fn singular_values(x: &[f32], rows: usize, cols: usize) -> Vec<f64> {
    let (m, n) = if rows <= cols {
        (rows, cols)
    } else {
        (cols, rows)
    };
    let at = |i: usize, j: usize| -> f64 {
        if rows <= cols {
            x[i * cols + j] as f64
        } else {
            x[j * cols + i] as f64
        }
    };
    let mut a = vec![0f64; m * m];
    for i in 0..m {
        for j in 0..m {
            a[i * m + j] = (0..n).map(|k| at(i, k) * at(j, k)).sum();
        }
    }
    for _ in 0..60 {
        let mut off = 0f64;
        for p in 0..m {
            for q in p + 1..m {
                off += a[p * m + q] * a[p * m + q];
                if a[p * m + q].abs() < 1e-14 {
                    continue;
                }
                let theta = (a[q * m + q] - a[p * m + p]) / (2.0 * a[p * m + q]);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let (c, s) = (1.0 / (t * t + 1.0).sqrt(), t / (t * t + 1.0).sqrt());
                for k in 0..m {
                    let (akp, akq) = (a[k * m + p], a[k * m + q]);
                    a[k * m + p] = c * akp - s * akq;
                    a[k * m + q] = s * akp + c * akq;
                }
                for k in 0..m {
                    let (apk, aqk) = (a[p * m + k], a[q * m + k]);
                    a[p * m + k] = c * apk - s * aqk;
                    a[q * m + k] = s * apk + c * aqk;
                }
            }
        }
        if off < 1e-24 {
            break;
        }
    }
    (0..m).map(|i| a[i * m + i].max(0.0).sqrt()).collect()
}

#[test]
fn newton_schulz_matches_the_float64_spec_and_is_nearly_orthogonal() {
    let Some(mut tr) = small_trainer() else {
        return;
    };
    for name in ["wide", "tall", "square", "kv"] {
        let (r, c, g, want) = golden(name);
        let got = tr.muon_direction(&g, r, c, 5).unwrap();
        let rms_want =
            (want.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / want.len() as f64).sqrt();
        let err = (got
            .iter()
            .zip(&want)
            .map(|(a, b)| ((a - b) as f64).powi(2))
            .sum::<f64>()
            / want.len() as f64)
            .sqrt();
        let mut sv = singular_values(&got, r, c);
        sv.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let (lo, med, hi) = (sv[0], sv[sv.len() / 2], sv[sv.len() - 1]);
        eprintln!("{name:7} {r}x{c}: rms error vs float64 spec {:.2e} (rms of the output {:.3e}); GPU output singular values: min {lo:.3} median {med:.3} max {hi:.3}", err, rms_want);
        assert!(
            err < 0.05 * rms_want,
            "{name}: GPU direction differs from the spec: {err} vs rms {rms_want}"
        );
        // five iterations cannot lift singular values below ~1e-3 of the largest (a random square matrix has some), but the bulk lands in the documented band
        assert!(
            med > 0.6 && hi < 1.4,
            "{name}: singular values (median {med}, max {hi}) outside the Newton-Schulz band"
        );
    }
}

#[test]
fn a_muon_step_changes_the_hidden_matrices_and_keeps_the_embedding_on_adamw() {
    let Some(mut tr) = small_trainer() else {
        return;
    };
    tr.init_random(3).unwrap();
    let before = tr.params_named().unwrap();
    // a gradient from one forward/backward pass on a fixed sequence
    let toks: Vec<u32> = (0..65).map(|i| (i * 7 % 128) as u32).collect();
    tr.zero_grads().unwrap();
    tr.forward_backward(&toks, 1024.0, 64).unwrap();
    let stats = tr
        .step_muon(1e-3, 1e-3, 0.9, 0.95, 1e-8, 0.1, 0.05, 0.95, 1.0, 1024.0)
        .unwrap();
    assert!(!stats.found_inf && stats.grad_norm.is_finite() && stats.grad_norm > 0.0);
    let after = tr.params_named().unwrap();
    for name in [
        "l0.wq",
        "l0.wk",
        "l0.wv",
        "l0.wo",
        "l0.w_gate",
        "l0.w_up",
        "l0.w_down",
    ] {
        let (b, a) = (&before[name], &after[name]);
        let delta = (a
            .iter()
            .zip(b)
            .map(|(x, y)| ((x - y) as f64).powi(2))
            .sum::<f64>()
            / a.len() as f64)
            .sqrt();
        // Moonshot scaling: the update RMS per element is about lr * 0.2 (plus the tiny weight decay term)
        eprintln!(
            "{name}: update rms {delta:.3e} (expected about {:.1e})",
            1e-3 * 0.2
        );
        assert!(
            delta > 0.4 * 1e-3 * 0.2 && delta < 2.5 * 1e-3 * 0.2,
            "{name}: update rms {delta}"
        );
    }
    let (b, a) = (&before["emb"], &after["emb"]);
    assert!(
        a.iter().zip(b).any(|(x, y)| x != y),
        "the embedding must be updated by AdamW"
    );
}
