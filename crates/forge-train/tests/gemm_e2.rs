//! The staged GEMM epilogue (`*_e2`, FORGE_GEMM_E2, default on) and the column-group tile orders (`*_e2gN`, kbench only) must give BIT-IDENTICAL outputs to the original Turing kernels:
//! every kernel pair on Forge-1 shapes with a partial last row tile and a half-empty last column tile, the fallback for widths that are not a multiple of 8, the head's width, and the trainer's whole
//! forward/backward with FORGE_GEMM_E2=0 against the default. Small sizes (well under 200 MB of GPU memory, a few seconds).

use forge_core::half::f32_to_f16;
use forge_cuda::{Args, Device, DeviceBuffer, Function, Module, Stream};
use forge_train::{TrainConfig, Trainer};

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
    fn uniform(&mut self) -> f32 {
        (self.next() as f32 + 0.5) / (1u64 << 31) as f32
    }
}

fn rand_buf(n: usize, seed: u64, scale: f32) -> DeviceBuffer {
    let mut r = Lcg(seed);
    let b: Vec<u8> = (0..n)
        .flat_map(|_| f32_to_f16((r.uniform() * 2.0 - 1.0) * scale).to_le_bytes())
        .collect();
    DeviceBuffer::from_bytes(&b).unwrap()
}

fn bytes(b: &DeviceBuffer, n: usize) -> Vec<u8> {
    let mut raw = vec![0u8; n];
    b.download(&mut raw).unwrap();
    raw
}

fn setup() -> Option<(Device, Module, Stream)> {
    if Device::count().unwrap_or(0) == 0 {
        eprintln!("skipped: no CUDA device");
        return None;
    }
    let dev = Device::open(0).unwrap();
    if dev.info.cc_major != 7 {
        eprintln!("skipped: the staged epilogue is in the Turing kernels (sm_75)");
        return None;
    }
    let (image, _) = forge_kernels::image_for(dev.info.cc_major, dev.info.cc_minor).unwrap();
    let module = Module::load(image).unwrap();
    Some((dev, module, Stream::new().unwrap()))
}

/// layouts: 0 NT (A [M][K], B [N][K]), 1 NN (A [M][K], B [K][N]), 2 TN (A [K][M], B [K][N]); kinds: 0 plain, 1 fp32 residual (R [M][N] fp32), 2 ReLU^2 forward (R out, fp16), 3 ReLU^2 backward
/// (R = u in, fp16), 4 accumulate into C (fp32)
#[allow(clippy::too_many_arguments)]
fn run(
    module: &Module,
    st: &Stream,
    name: &str,
    layout: usize,
    kind: usize,
    out_bytes: usize,
    m: usize,
    n: usize,
    k: usize,
    a: &DeviceBuffer,
    b: &DeviceBuffer,
    r_init: &[u8],
) -> (Vec<u8>, Vec<u8>) {
    let func: Function = module
        .function(name)
        .unwrap_or_else(|e| panic!("{name}: {e}"));
    let (lda, ldb) = match layout {
        0 => (k, k),
        1 => (k, n),
        _ => (m, n),
    };
    let c = DeviceBuffer::alloc(m * n * out_bytes).unwrap();
    let r = DeviceBuffer::alloc(m * n * 4).unwrap();
    // C starts with a pattern (the accumulating kind adds into it), R with the given bytes
    let c_init: Vec<u8> = (0..m * n * out_bytes)
        .map(|i| (i * 7 % 251) as u8 & 0x3f)
        .collect();
    c.upload(&c_init).unwrap();
    if !r_init.is_empty() {
        r.upload(r_init).unwrap();
    }
    let mut args = Args::new()
        .buf(a)
        .buf(b)
        .buf(&c)
        .i32(m as i32)
        .i32(n as i32)
        .i32(k as i32)
        .i32(lda as i32)
        .i32(ldb as i32)
        .i32(n as i32)
        .f32(1.0);
    if kind == 1 || kind == 2 || kind == 3 {
        args = args.buf(&r).i32(n as i32).f32(0.0);
    }
    let bn = if name.ends_with("w144") { 144 } else { 128 };
    if bn == 144 {
        func.set_smem_carveout(100).unwrap();
    }
    func.launch(
        (((n + bn - 1) / bn) as u32, ((m + 127) / 128) as u32, 1),
        (256, 1, 1),
        0,
        st,
        &args,
    )
    .unwrap();
    st.synchronize().unwrap();
    (
        bytes(&c, m * n * out_bytes),
        if kind == 2 {
            bytes(&r, m * n * 2)
        } else {
            Vec::new()
        },
    )
}

#[test]
fn staged_epilogue_and_tile_orders_are_bit_identical() {
    let Some((_dev, module, st)) = setup() else {
        return;
    };
    // (old kernel, layout, kind, output bytes, variants)
    let kernels: [(&str, usize, usize, usize, &[&str]); 9] = [
        ("gemm_nt_f16", 0, 0, 2, &["e2", "e2g3", "w144"]),
        ("gemm_nt_f32", 0, 0, 4, &["e2", "e2g4"]),
        ("gemm_nt_res", 0, 1, 4, &["e2", "w144"]),
        ("gemm_nt_relu2", 0, 2, 2, &["e2", "e2g3", "e2g6"]),
        ("gemm_nn_f16", 1, 0, 2, &["e2", "w144"]),
        ("gemm_nn_f32", 1, 0, 4, &["e2", "w144"]),
        ("gemm_nn_add", 1, 4, 4, &["e2"]),
        ("gemm_nn_relu2_bwd", 1, 3, 2, &["e2", "e2g3", "e2g6"]),
        ("gemm_tn_add", 2, 4, 4, &["e2"]),
    ];
    // (M, N, K): Forge-1 widths with a partial last row tile (300, 1000) and half-empty column tiles (960, 576); N = 100 takes the fallback (not a multiple of 8); 2048 wide like a head slice
    let shapes = [
        (300usize, 960usize, 576usize),
        (512, 576, 2304),
        (1000, 2304, 576),
        (256, 100, 192),
        (128, 2048, 576),
        (384, 576, 960),
        (136, 200, 576),
    ];
    let mut checked = 0;
    for &(m, n, k) in &shapes {
        let a = rand_buf(m.max(k) * k.max(m), 3 + m as u64, 2.0);
        let b = rand_buf(n.max(k) * k.max(n), 4 + n as u64, 0.06);
        let r_f32: Vec<u8> = (0..m * n)
            .flat_map(|i| (((i % 89) as f32 - 44.0) * 0.03).to_le_bytes())
            .collect();
        let r_u: Vec<u8> = (0..m * n)
            .flat_map(|i| f32_to_f16(((i % 61) as f32 - 30.0) * 0.05).to_le_bytes())
            .collect();
        for &(old, layout, kind, ob, vars) in &kernels {
            // the k-major operands need their row width (N for NN, M and N for TN) to be a multiple of 8 (the kernels' contract)
            if (layout == 1 && n % 8 != 0) || (layout == 2 && (m % 8 != 0 || n % 8 != 0)) {
                continue;
            }
            let r_init: &[u8] = match kind {
                1 => &r_f32,
                3 => &r_u,
                _ => &[],
            };
            let want = run(&module, &st, old, layout, kind, ob, m, n, k, &a, &b, r_init);
            for sfx in vars {
                let got = run(
                    &module,
                    &st,
                    &format!("{old}_{sfx}"),
                    layout,
                    kind,
                    ob,
                    m,
                    n,
                    k,
                    &a,
                    &b,
                    r_init,
                );
                assert!(got.0 == want.0, "{old}_{sfx} {m}x{n}x{k}: output differs");
                assert!(
                    got.1 == want.1,
                    "{old}_{sfx} {m}x{n}x{k}: relu(u)^2 differs"
                );
                checked += 1;
            }
        }
    }
    eprintln!(
        "{checked} kernel / shape pairs bit-identical to the original epilogue and tile order"
    );
}

#[test]
fn trainer_with_staged_epilogue_matches_the_original_kernels() {
    if Device::count().unwrap_or(0) == 0 {
        eprintln!("skipped: no CUDA device");
        return;
    }
    if Device::open(0).unwrap().info.cc_major != 7 {
        eprintln!("skipped: FORGE_GEMM_E2 applies to the Turing path only");
        return;
    }
    // Forge-1 structure at a small size (as tests/kopt.rs)
    let cfg = TrainConfig {
        n_layer: 3,
        n_embd: 192,
        n_head: 3,
        n_kv: 1,
        head_dim: 64,
        n_ff: 768,
        vocab: 2048,
        seq_len: 512,
        batch: 2,
        qk_norm: true,
        ce_chunk: 512,
        relu2: true,
        lambdas: true,
        ve_layers: vec![0, 2],
        ..TrainConfig::default()
    };
    let n = cfg.tokens();
    let mut r = Lcg(77);
    let tokens: Vec<u32> = (0..cfg.batch * (cfg.seq_len + 1))
        .map(|_| r.next() % cfg.vocab as u32)
        .collect();
    let mut runs = Vec::new();
    for setting in ["0", "1", "0"] {
        std::env::set_var("FORGE_GEMM_E2", setting);
        let mut tr = Trainer::new(cfg.clone(), 0).unwrap();
        std::env::remove_var("FORGE_GEMM_E2");
        tr.init_random(5).unwrap();
        let l_eval = tr.eval_loss(&tokens).unwrap();
        tr.zero_grads().unwrap();
        let l_fb = tr.forward_backward(&tokens, 1024.0, n).unwrap();
        runs.push((l_eval, l_fb, tr.grads_named(1.0 / 1024.0).unwrap()));
    }
    let worst = |a: &std::collections::HashMap<String, Vec<f32>>,
                 b: &std::collections::HashMap<String, Vec<f32>>| {
        a.iter()
            .map(|(name, x)| {
                let y = &b[name];
                let (mut dd, mut xx) = (0f64, 0f64);
                for (p, q) in x.iter().zip(y) {
                    dd += ((p - q) as f64).powi(2);
                    xx += (*p as f64).powi(2);
                }
                (dd / xx.max(1e-30)).sqrt()
            })
            .fold(0f64, f64::max)
    };
    let noise = worst(&runs[0].2, &runs[2].2);
    let d = worst(&runs[0].2, &runs[1].2);
    eprintln!("loss: original epilogue {:.7} / {:.7}, staged {:.7} / {:.7}; worst relative gradient difference {d:.2e} (run-to-run noise of the original path {noise:.2e})", runs[0].0, runs[0].1, runs[1].0, runs[1].1);
    assert_eq!(
        runs[0].0.to_bits(),
        runs[1].0.to_bits(),
        "eval loss differs"
    );
    assert_eq!(
        runs[0].1.to_bits(),
        runs[1].1.to_bits(),
        "forward/backward loss differs"
    );
    assert!(
        d <= noise.max(1e-7) * 3.0,
        "gradients differ beyond the atomics noise: {d:.2e} vs {noise:.2e}"
    );
}

#[test]
fn trainer_with_144_wide_tiles_matches_the_128_wide_kernels() {
    if Device::count().unwrap_or(0) == 0 {
        eprintln!("skipped: no CUDA device");
        return;
    }
    if Device::open(0).unwrap().info.cc_major != 7 {
        eprintln!("skipped: FORGE_GEMM_W144 applies to the Turing path only");
        return;
    }
    // width 288 = 2 x 144: the attention output, the MLP down projection and the data gradients into the residual width take the 128 x 144 kernels
    let cfg = TrainConfig {
        n_layer: 2,
        n_embd: 288,
        n_head: 4,
        n_kv: 2,
        head_dim: 64,
        n_ff: 1152,
        vocab: 2048,
        seq_len: 256,
        batch: 2,
        qk_norm: true,
        ce_chunk: 512,
        relu2: true,
        lambdas: true,
        ve_layers: vec![1],
        ..TrainConfig::default()
    };
    let n = cfg.tokens();
    let mut r = Lcg(78);
    let tokens: Vec<u32> = (0..cfg.batch * (cfg.seq_len + 1))
        .map(|_| r.next() % cfg.vocab as u32)
        .collect();
    let mut runs = Vec::new();
    // "force": the 128 x 144 kernels whatever the wave count of this small model on this device
    for setting in ["0", "force", "0"] {
        std::env::set_var("FORGE_GEMM_W144", setting);
        let mut tr = Trainer::new(cfg.clone(), 0).unwrap();
        std::env::remove_var("FORGE_GEMM_W144");
        tr.init_random(6).unwrap();
        let l_eval = tr.eval_loss(&tokens).unwrap();
        tr.zero_grads().unwrap();
        let l_fb = tr.forward_backward(&tokens, 1024.0, n).unwrap();
        runs.push((l_eval, l_fb, tr.grads_named(1.0 / 1024.0).unwrap()));
    }
    let worst = |a: &std::collections::HashMap<String, Vec<f32>>,
                 b: &std::collections::HashMap<String, Vec<f32>>| {
        a.iter()
            .map(|(name, x)| {
                let y = &b[name];
                let (mut dd, mut xx) = (0f64, 0f64);
                for (p, q) in x.iter().zip(y) {
                    dd += ((p - q) as f64).powi(2);
                    xx += (*p as f64).powi(2);
                }
                (dd / xx.max(1e-30)).sqrt()
            })
            .fold(0f64, f64::max)
    };
    let noise = worst(&runs[0].2, &runs[2].2);
    let d = worst(&runs[0].2, &runs[1].2);
    eprintln!("loss: 128-wide {:.7} / {:.7}, 144-wide {:.7} / {:.7}; worst relative gradient difference {d:.2e} (run-to-run noise {noise:.2e})", runs[0].0, runs[0].1, runs[1].0, runs[1].1);
    assert_eq!(
        runs[0].0.to_bits(),
        runs[1].0.to_bits(),
        "eval loss differs"
    );
    assert_eq!(
        runs[0].1.to_bits(),
        runs[1].1.to_bits(),
        "forward/backward loss differs"
    );
    assert!(
        d <= noise.max(1e-7) * 3.0,
        "gradients differ beyond the atomics noise: {d:.2e} vs {noise:.2e}"
    );
}
