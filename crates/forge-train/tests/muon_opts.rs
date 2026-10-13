//! The options of the Muon update (Polar Express, the number of iterations, NorMuon, cautious weight decay) against the float64 executable spec in tools/ref_muon.py: three full optimizer steps of
//! a tiny model's Muon matrices (tall, square and wide ones) with given gradients, compared with the spec's weights after the third step (tests/golden/muon_step). Every variant must match its own
//! spec closely and be clearly farther from the default update's spec (so a variant that silently fell back to the default would fail).

use forge_core::Json;
use forge_train::{MuonOpts, NorMuonAxis, TrainConfig, Trainer};
use std::collections::HashMap;
use std::path::PathBuf;

fn dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/golden/muon_step")
}

fn read_f32(p: PathBuf) -> Vec<f32> {
    std::fs::read(&p)
        .unwrap_or_else(|e| panic!("{}: {e}", p.display()))
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn rel(a: &[f32], b: &[f32]) -> f64 {
    let (mut dd, mut bb) = (0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        dd += ((x - y) as f64).powi(2);
        bb += (*y as f64).powi(2);
    }
    (dd / bb.max(1e-30)).sqrt()
}

fn delta(w: &[f32], w0: &[f32]) -> Vec<f32> {
    w.iter().zip(w0).map(|(a, b)| a - b).collect()
}

fn opts_of(v: &Json) -> MuonOpts {
    MuonOpts {
        polar: v["polar"].as_bool().unwrap_or(false),
        iters: v["iters"].as_usize().unwrap_or(5),
        normuon: match v["normuon"].as_str() {
            Some("long") => Some(NorMuonAxis::Long),
            Some("rows") => Some(NorMuonAxis::Rows),
            _ => None,
        },
        normuon_beta2: v["beta2"].as_f64().unwrap_or(0.95) as f32,
        cautious_wd: v["cautious"].as_bool().unwrap_or(false),
    }
}

#[test]
fn muon_options_match_the_float64_spec() {
    if forge_cuda::Device::count().unwrap_or(0) == 0 {
        eprintln!("skipped: no CUDA device");
        return;
    }
    let man = Json::from_file(dir().join("manifest.json")).unwrap();
    let hp = &man["hp"];
    let (lr, wd, mu) = (
        hp["lr"].as_f64().unwrap() as f32,
        hp["wd"].as_f64().unwrap() as f32,
        hp["mu"].as_f64().unwrap() as f32,
    );
    let mats: Vec<(String, usize, usize)> = man["mats"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| {
            (
                m["name"].as_str().unwrap().to_string(),
                m["rows"].as_usize().unwrap(),
                m["cols"].as_usize().unwrap(),
            )
        })
        .collect();
    let cfg = TrainConfig {
        n_layer: 1,
        n_embd: 64,
        n_head: 2,
        n_kv: 1,
        head_dim: 64,
        n_ff: 192,
        vocab: 128,
        seq_len: 64,
        batch: 1,
        qk_norm: true,
        ce_chunk: 64,
        ..TrainConfig::default()
    };
    let w0: HashMap<String, Vec<f32>> = mats
        .iter()
        .map(|(n, _, _)| (n.clone(), read_f32(dir().join(format!("{n}.w0.f32")))))
        .collect();
    let grads: Vec<HashMap<String, Vec<f32>>> = (1..=3)
        .map(|s| {
            mats.iter()
                .map(|(n, _, _)| (n.clone(), read_f32(dir().join(format!("{n}.g{s}.f32")))))
                .collect()
        })
        .collect();
    let default_w3: HashMap<String, Vec<f32>> = mats
        .iter()
        .map(|(n, _, _)| {
            (
                n.clone(),
                read_f32(dir().join("default").join(format!("{n}.w3.f32"))),
            )
        })
        .collect();
    let variants = man["variants"].as_object().unwrap();
    for (var, spec) in variants.iter() {
        let o = opts_of(spec);
        let mut tr = Trainer::new(cfg.clone(), 0).unwrap();
        tr.set_muon_opts(o).unwrap();
        let mut named = HashMap::new();
        for p in &tr.params {
            named.insert(
                p.name.clone(),
                w0.get(&p.name).cloned().unwrap_or_else(|| {
                    vec![if p.name.ends_with("_norm") { 1.0 } else { 0.01 }; p.len]
                }),
            );
        }
        tr.set_params(&named).unwrap();
        for g in &grads {
            // the flat gradient: the given matrices, zero elsewhere (AdamW then moves nothing but by its decay)
            let mut flat = vec![0f32; tr.n_params];
            for p in &tr.params {
                if let Some(v) = g.get(&p.name) {
                    flat[p.offset..p.offset + p.len].copy_from_slice(v);
                }
            }
            tr.upload_grads(&flat).unwrap();
            // loss scale 1 and no clipping: the update sees the gradient as given
            let s = tr
                .step_muon(1e-3, lr, 0.9, 0.95, 1e-8, 0.0, wd, mu, 0.0, 1.0)
                .unwrap();
            assert!(!s.found_inf);
        }
        let got = tr.params_named().unwrap();
        for (n, _, _) in &mats {
            let want = read_f32(dir().join(var).join(format!("{n}.w3.f32")));
            let (dg, dw, dd) = (
                delta(&got[n], &w0[n]),
                delta(&want, &w0[n]),
                delta(&default_w3[n], &w0[n]),
            );
            let err = rel(&dg, &dw);
            let sep = rel(&dd, &dw);
            eprintln!("{var:13} {n:10}: relative error of the 3-step change vs the spec {err:.4} (distance of the default update from this variant's spec {sep:.4})");
            assert!(
                err < 0.03,
                "{var} {n}: the GPU update differs from the spec by {err}"
            );
            if *var != "default" {
                assert!(err < 0.5 * sep, "{var} {n}: the GPU update is not clearly closer to its own spec ({err}) than the default update is ({sep})");
            }
        }
    }
}

#[test]
fn polar_express_tables_are_the_specs() {
    // the coefficient tables compiled into the trainer are the ones tools/ref_muon.py derives (manifest.json carries them)
    let man = Json::from_file(dir().join("manifest.json")).unwrap();
    for k in 3..=7usize {
        let want = man["polar_express"][k.to_string().as_str()]
            .as_array()
            .unwrap();
        let got = forge_train::polar_express_coeffs(k).unwrap();
        assert_eq!(got.len(), k);
        for (g, w) in got.iter().zip(want) {
            let w: Vec<f64> = w
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_f64().unwrap())
                .collect();
            for (a, b) in [g.0, g.1, g.2].iter().zip(&w) {
                assert!(
                    ((*a as f64) - b).abs() < 1e-5 * b.abs().max(1.0),
                    "{k} iterations: {a} vs {b}"
                );
            }
        }
    }
    assert!(
        forge_train::polar_express_coeffs(2).is_none()
            && forge_train::polar_express_coeffs(8).is_none()
    );
}

#[test]
fn adam_cautious_weight_decay_matches_the_host_formula() {
    // one plain AdamW step (step(), every parameter on AdamW) with --cautious-wd-adam against the elementwise formula on the host: below decay_end an element decays only where the update moves it
    // towards zero; with the option off every decayed element decays
    if forge_cuda::Device::count().unwrap_or(0) == 0 {
        eprintln!("skipped: no CUDA device");
        return;
    }
    let cfg = TrainConfig {
        n_layer: 1,
        n_embd: 64,
        n_head: 2,
        n_kv: 1,
        head_dim: 64,
        n_ff: 192,
        vocab: 128,
        seq_len: 64,
        batch: 1,
        qk_norm: true,
        ce_chunk: 64,
        ..TrainConfig::default()
    };
    let lay = forge_train::param_layout(&cfg);
    for cautious in [false, true] {
        let mut tr = Trainer::new(cfg.clone(), 0).unwrap();
        tr.init_random(4).unwrap();
        tr.set_adam_cautious_wd(cautious);
        let p0 = tr.download_params().unwrap();
        let g: Vec<f32> = (0..tr.n_params)
            .map(|i| ((i as f32 * 0.7311).sin()) * 1e-2)
            .collect();
        tr.upload_grads(&g).unwrap();
        let (lr, b1, b2, eps, wd) = (1e-2f32, 0.9f32, 0.95f32, 1e-8f32, 0.5f32);
        let s = tr.step(lr, b1, b2, eps, wd, 0.0, 1.0).unwrap();
        assert!(!s.found_inf);
        let p1 = tr.download_params().unwrap();
        let (bc1, bc2) = (1.0 - b1, 1.0 - b2);
        let mut worst = 0f32;
        let mut decayed = 0usize;
        for i in 0..tr.n_params {
            let (m, v) = ((1.0 - b1) * g[i], (1.0 - b2) * g[i] * g[i]);
            let u = (m / bc1) / ((v / bc2).sqrt() + eps);
            let mut p = p0[i];
            if i < lay.decay_end && (!cautious || u * p >= 0.0) {
                p *= 1.0 - lr * wd;
                decayed += 1;
            }
            p -= lr * u;
            worst = worst.max((p - p1[i]).abs());
        }
        eprintln!("cautious {cautious}: {decayed} of {} decayable elements decayed; worst difference to the host formula {worst:e}", lay.decay_end);
        assert!(
            worst < 1e-6,
            "cautious {cautious}: worst difference {worst}"
        );
        if cautious {
            assert!(decayed > lay.decay_end / 4 && decayed < 3 * lay.decay_end / 4);
        } else {
            assert_eq!(decayed, lay.decay_end);
        }
    }
}
