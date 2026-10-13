//! Sequence-length schedule (`Trainer::set_window`): a trainer built for seq_len 128 that trains on windows of 64 must compute exactly what a trainer built for seq_len 64 (with twice the rows)
//! computes on the same rows: same loss, same gradients (the Forge-1 recipe's structure options, which depend on the position inside the window, are on). Also: the parameter-average checkpoint
//! (`save_checkpoint_params`) round-trips through `read_checkpoint_params` and `load_params`.

use forge_train::{TrainConfig, Trainer};

fn cfg(seq: usize, batch: usize) -> TrainConfig {
    TrainConfig {
        n_layer: 2,
        n_embd: 64,
        n_head: 2,
        n_kv: 1,
        head_dim: 64,
        n_ff: 128,
        vocab: 256,
        seq_len: seq,
        batch,
        qk_norm: true,
        ce_chunk: 128,
        relu2: true,
        lambdas: true,
        ve_layers: vec![0, 1],
        ..TrainConfig::default()
    }
}

fn rel(a: &[f32], b: &[f32]) -> f64 {
    let (mut dd, mut bb) = (0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        dd += ((x - y) as f64).powi(2);
        bb += (*y as f64).powi(2);
    }
    (dd / bb.max(1e-30)).sqrt()
}

#[test]
fn a_training_window_equals_a_trainer_built_for_that_length() {
    if forge_cuda::Device::count().unwrap_or(0) == 0 {
        eprintln!("skipped: no CUDA device");
        return;
    }
    let mut long = Trainer::new(cfg(128, 2), 0).unwrap();
    let mut short = Trainer::new(cfg(64, 4), 0).unwrap();
    long.init_random(5).unwrap();
    short.init_random(5).unwrap();
    // four rows of 65 ids
    let toks: Vec<u32> = (0..4 * 65)
        .map(|i| ((i * 37 + (i / 65) * 11) % 256) as u32)
        .collect();
    long.set_window(Some(64)).unwrap();
    assert_eq!(long.window(), Some(64));
    long.zero_grads().unwrap();
    short.zero_grads().unwrap();
    let la = long.forward_backward(&toks, 1024.0, 256).unwrap();
    let lb = short.forward_backward(&toks, 1024.0, 256).unwrap();
    eprintln!("loss with a 64-token window {la} / built for 64 tokens {lb}");
    assert!((la - lb).abs() <= 1e-6 * lb.abs(), "{la} vs {lb}");
    let (ga, gb) = (
        long.grads_named(1.0 / 1024.0).unwrap(),
        short.grads_named(1.0 / 1024.0).unwrap(),
    );
    for p in &short.params {
        let r = rel(&ga[&p.name], &gb[&p.name]);
        assert!(r < 1e-5, "{}: gradients differ by {r}", p.name);
    }
    assert_eq!(
        long.eval_loss(&toks).unwrap(),
        short.eval_loss(&toks).unwrap()
    );
    // back to the full length: two rows of 129
    long.set_window(None).unwrap();
    let full: Vec<u32> = (0..2 * 129).map(|i| (i * 13 % 256) as u32).collect();
    let l = long.eval_loss(&full).unwrap();
    assert!(l.is_finite() && l > 0.0);
    // windows the trainer refuses
    assert!(
        long.set_window(Some(96)).is_err()
            && long.set_window(Some(256)).is_err()
            && long.set_window(Some(0)).is_err()
    );
}

#[test]
fn parameter_average_checkpoints_round_trip() {
    if forge_cuda::Device::count().unwrap_or(0) == 0 {
        eprintln!("skipped: no CUDA device");
        return;
    }
    let mut tr = Trainer::new(cfg(64, 1), 0).unwrap();
    tr.init_random(9).unwrap();
    let raw = tr.download_params().unwrap();
    let avg: Vec<f32> = raw
        .iter()
        .enumerate()
        .map(|(i, v)| v * 0.5 + (i % 7) as f32 * 1e-3)
        .collect();
    let dir = std::env::temp_dir().join(format!("forge_avg_test_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("ckpt_ema.bin");
    tr.save_checkpoint_params(&path, 2048.0, &avg).unwrap();
    assert_eq!(forge_train::read_checkpoint_params(&path).unwrap(), avg);
    // the trainer's own parameters are untouched, and the file loads like any checkpoint
    assert_eq!(tr.download_params().unwrap(), raw);
    let mut other = Trainer::new(cfg(64, 1), 0).unwrap();
    other.load_params(&path).unwrap();
    assert_eq!(other.download_params().unwrap(), avg);
    // upload_params / download_params
    tr.upload_params(&avg).unwrap();
    assert_eq!(tr.download_params().unwrap(), avg);
    std::fs::remove_file(&path).ok();
    std::fs::remove_dir(&dir).ok();
}

#[test]
fn document_masking_refuses_sliding_windows() {
    // the masked kernels (flash_doc.cu) have no window: the combination is refused instead of silently dropping one of the two
    if forge_cuda::Device::count().unwrap_or(0) == 0 {
        eprintln!("skipped: no CUDA device");
        return;
    }
    let mut windowed = Trainer::new(
        TrainConfig {
            window: 64,
            window_layers: vec![0],
            ..cfg(128, 2)
        },
        0,
    )
    .unwrap();
    assert!(windowed.set_doc_mask(Some(0)).is_err());
    assert!(windowed.set_doc_mask(None).is_ok());
    let mut dense = Trainer::new(cfg(128, 2), 0).unwrap();
    assert!(dense.set_doc_mask(Some(0)).is_ok());
    assert_eq!(dense.doc_mask(), Some(0));
}
