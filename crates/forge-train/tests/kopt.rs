//! Parity of the training-step kernel optimisations (branch kernel-opt) with the paths they replace, on random data:
//!  - split-K weight-gradient GEMM (Turing, atomics) against the unsplit kernel and a CPU reference, for the 576 / 960 / 2304 shapes of Forge-1,
//!  - the ReLU^2 GEMM epilogues against GEMM + separate kernel (bit-identical),
//!  - the register-resident cross-entropy kernel against the three-pass one,
//!  - a whole forward/backward of a small Forge-1-like model with every optimisation on against every one off (FORGE_SPLITK=1 FORGE_NO_FUSE=1 FORGE_CE_OLD=1 FORGE_ATTN_OLD=1 FORGE_NO_FUSE_MIX=1),
//!  - the second-generation and sliding-window attention kernels, Muon on several streams, the split-K cost model.
//! Sizes stay small (well under 400 MB of GPU memory, a few seconds each).

use forge_core::half::{f16_to_f32, f32_to_f16};
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

fn rand_f16(n: usize, seed: u64, scale: f32) -> (Vec<u16>, Vec<f32>) {
    let mut r = Lcg(seed);
    let h: Vec<u16> = (0..n)
        .map(|_| f32_to_f16((r.uniform() * 2.0 - 1.0) * scale))
        .collect();
    let f = h.iter().map(|&v| f16_to_f32(v)).collect();
    (h, f)
}

fn bytes16(h: &[u16]) -> Vec<u8> {
    h.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn download16(b: &DeviceBuffer, n: usize) -> Vec<u16> {
    let mut raw = vec![0u8; n * 2];
    b.download(&mut raw).unwrap();
    raw.chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect()
}

fn setup() -> Option<(Device, Module, Stream)> {
    if Device::count().unwrap_or(0) == 0 {
        eprintln!("skipped: no CUDA device");
        return None;
    }
    let dev = Device::open(0).unwrap();
    if dev.info.cc_major != 7 {
        eprintln!("skipped: the Turing kernels are only used on sm_75");
        return None;
    }
    let (image, _) = forge_kernels::image_for(dev.info.cc_major, dev.info.cc_minor).unwrap();
    let module = Module::load(image).unwrap();
    Some((dev, module, Stream::new().unwrap()))
}

#[allow(clippy::too_many_arguments)]
fn gemm(
    f: Function,
    st: &Stream,
    grid: (usize, usize, usize),
    a: u64,
    b: u64,
    c: u64,
    m: usize,
    n: usize,
    k: usize,
    lda: usize,
    ldb: usize,
    ldc: usize,
    extra: Option<(u64, usize)>,
) {
    let mut args = Args::new()
        .ptr(a)
        .ptr(b)
        .ptr(c)
        .i32(m as i32)
        .i32(n as i32)
        .i32(k as i32)
        .i32(lda as i32)
        .i32(ldb as i32)
        .i32(ldc as i32)
        .f32(1.0);
    if let Some((r, ldr)) = extra {
        args = args.ptr(r).i32(ldr as i32).f32(0.0);
    }
    f.launch(
        (grid.0 as u32, grid.1 as u32, grid.2 as u32),
        (256, 1, 1),
        0,
        st,
        &args,
    )
    .unwrap();
}

#[test]
fn split_k_weight_gradient_matches_the_unsplit_kernel_and_the_cpu() {
    let Some((_dev, module, st)) = setup() else {
        return;
    };
    let tn_add = module.function("gemm_tn_add").unwrap();
    // (M, N, K): C[M][N] += sum_k A[k][M] B[k][N]  (the Forge-1 shapes with a short token count; 960 and 576 have a half-empty last tile)
    for &(m, n, k) in &[
        (576usize, 576usize, 2048usize),
        (960, 576, 1024),
        (576, 2304, 768),
        (2304, 576, 512),
    ] {
        let (ah, af) = rand_f16(k * m, 1 + m as u64, 1.0);
        let (bh, bf) = rand_f16(k * n, 2 + n as u64, 1.0);
        let mut want = vec![1f32; m * n];
        for kk in 0..k {
            let arow = &af[kk * m..(kk + 1) * m];
            let brow = &bf[kk * n..(kk + 1) * n];
            for i in 0..m {
                let a = arow[i];
                let w = &mut want[i * n..(i + 1) * n];
                for j in 0..n {
                    w[j] += a * brow[j];
                }
            }
        }
        let scale = want.iter().fold(0f32, |s, v| s.max(v.abs()));
        let da = DeviceBuffer::from_bytes(&bytes16(&ah)).unwrap();
        let db = DeviceBuffer::from_bytes(&bytes16(&bh)).unwrap();
        let dc = DeviceBuffer::alloc(m * n * 4).unwrap();
        let mut unsplit: Vec<f32> = Vec::new();
        for gz in [1usize, 2, 3, 4, 8] {
            if k % (16 * gz) != 0 {
                continue;
            }
            dc.upload_f32(&vec![1f32; m * n]).unwrap();
            gemm(
                tn_add,
                &st,
                ((n + 127) / 128, (m + 127) / 128, gz),
                da.ptr(),
                db.ptr(),
                dc.ptr(),
                m,
                n,
                k,
                m,
                n,
                n,
                None,
            );
            st.synchronize().unwrap();
            let got = dc.download_f32(m * n).unwrap();
            let worst = got
                .iter()
                .zip(&want)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            if gz == 1 {
                unsplit = got.clone();
            }
            let worst_vs_unsplit = got
                .iter()
                .zip(&unsplit)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            eprintln!("TN add {m}x{n}x{k} split {gz}: max |diff| vs CPU {worst:.2e}, vs unsplit {worst_vs_unsplit:.2e} (max |value| {scale:.1})");
            assert!(worst < 2e-3 * scale.max(1.0), "split {gz}: {worst}");
            assert!(
                worst_vs_unsplit < 1e-5 * scale.max(1.0),
                "split {gz} vs unsplit: {worst_vs_unsplit}"
            );
        }
    }
}

#[test]
fn relu2_epilogues_are_bit_identical_to_gemm_plus_kernel() {
    let Some((_dev, module, st)) = setup() else {
        return;
    };
    let f = |n: &str| module.function(n).unwrap();
    let (nt_f16, nt_relu2, nn_f16, nn_relu2_bwd, relu2, relu2_bwd) = (
        f("gemm_nt_f16"),
        f("gemm_nt_relu2"),
        f("gemm_nn_f16"),
        f("gemm_nn_relu2_bwd"),
        f("relu2_kernel"),
        f("relu2_bwd_kernel"),
    );
    // forward: u = x W^T (M tokens, N = ffn, K = d); 300 rows: a partial last row tile; values large enough that some relu(u)^2 saturate
    let (m, n, k) = (300usize, 640usize, 576usize);
    let (xh, _) = rand_f16(m * k, 5, 8.0);
    let (wh, _) = rand_f16(n * k, 6, 1.0);
    let x = DeviceBuffer::from_bytes(&bytes16(&xh)).unwrap();
    let w = DeviceBuffer::from_bytes(&bytes16(&wh)).unwrap();
    let (u0, a0, u1, a1) = (
        DeviceBuffer::alloc(m * n * 2).unwrap(),
        DeviceBuffer::alloc(m * n * 2).unwrap(),
        DeviceBuffer::alloc(m * n * 2).unwrap(),
        DeviceBuffer::alloc(m * n * 2).unwrap(),
    );
    let grid = ((n + 127) / 128, (m + 127) / 128, 1);
    gemm(
        nt_f16,
        &st,
        grid,
        x.ptr(),
        w.ptr(),
        u0.ptr(),
        m,
        n,
        k,
        k,
        k,
        n,
        None,
    );
    relu2
        .launch(
            (((m * n + 255) / 256) as u32, 1, 1),
            (256, 1, 1),
            0,
            &st,
            &Args::new().buf(&u0).buf(&a0).u64((m * n) as u64),
        )
        .unwrap();
    gemm(
        nt_relu2,
        &st,
        grid,
        x.ptr(),
        w.ptr(),
        u1.ptr(),
        m,
        n,
        k,
        k,
        k,
        n,
        Some((a1.ptr(), n)),
    );
    st.synchronize().unwrap();
    let (u0h, a0h, u1h, a1h) = (
        download16(&u0, m * n),
        download16(&a0, m * n),
        download16(&u1, m * n),
        download16(&a1, m * n),
    );
    assert_eq!(u0h, u1h, "u differs");
    assert_eq!(a0h, a1h, "relu(u)^2 differs");
    let positive = u1h.iter().filter(|&&v| f16_to_f32(v) > 0.0).count();
    let saturated = a1h.iter().filter(|&&v| f16_to_f32(v) >= 64000.0).count();
    eprintln!(
        "forward: {} values identical ({positive} positive, {saturated} saturated)",
        m * n
    );
    assert!(positive > m * n / 4 && positive < 3 * m * n / 4);
    // backward: da = dy W_down (M tokens, N = ffn, K = d; W_down stored [K = d][N = ffn]); du = da * 2 relu(u) with u = u1 above
    let (dyh, _) = rand_f16(m * k, 7, 1.0);
    let (wdh, _) = rand_f16(k * n, 8, 0.2);
    let dy = DeviceBuffer::from_bytes(&bytes16(&dyh)).unwrap();
    let wd = DeviceBuffer::from_bytes(&bytes16(&wdh)).unwrap();
    let (da, du0, du1) = (
        DeviceBuffer::alloc(m * n * 2).unwrap(),
        DeviceBuffer::alloc(m * n * 2).unwrap(),
        DeviceBuffer::alloc(m * n * 2).unwrap(),
    );
    gemm(
        nn_f16,
        &st,
        grid,
        dy.ptr(),
        wd.ptr(),
        da.ptr(),
        m,
        n,
        k,
        k,
        n,
        n,
        None,
    );
    relu2_bwd
        .launch(
            (((m * n + 255) / 256) as u32, 1, 1),
            (256, 1, 1),
            0,
            &st,
            &Args::new().buf(&da).buf(&u1).buf(&du0).u64((m * n) as u64),
        )
        .unwrap();
    gemm(
        nn_relu2_bwd,
        &st,
        grid,
        dy.ptr(),
        wd.ptr(),
        du1.ptr(),
        m,
        n,
        k,
        k,
        n,
        n,
        Some((u1.ptr(), n)),
    );
    st.synchronize().unwrap();
    let (du0h, du1h) = (download16(&du0, m * n), download16(&du1, m * n));
    assert_eq!(du0h, du1h, "du differs");
    let nonzero = du1h.iter().filter(|&&v| f16_to_f32(v) != 0.0).count();
    eprintln!("backward: {} values identical ({nonzero} non-zero)", m * n);
    assert!(nonzero > m * n / 4);
}

#[test]
fn register_cross_entropy_matches_the_three_pass_kernel() {
    let Some((_dev, module, st)) = setup() else {
        return;
    };
    let (k_old, k_reg) = (
        module.function("ce_kernel").unwrap(),
        module.function("ce_reg_kernel").unwrap(),
    );
    for &(rows, v) in &[(48usize, 32768usize), (40, 1000), (16, 4096)] {
        let mut r = Lcg(31 + v as u64);
        let z: Vec<f32> = (0..rows * v).map(|_| r.uniform() * 24.0 - 12.0).collect();
        // targets spread over the vocabulary; row 3 is ignored (target outside the vocabulary)
        let tg: Vec<u32> = (0..rows)
            .map(|i| {
                if i == 3 {
                    u32::MAX
                } else {
                    ((i * 7919 + 13) % v) as u32
                }
            })
            .collect();
        let zb = DeviceBuffer::alloc(rows * v * 4).unwrap();
        zb.upload_f32(&z).unwrap();
        let tb =
            DeviceBuffer::from_bytes(&tg.iter().flat_map(|t| t.to_le_bytes()).collect::<Vec<u8>>())
                .unwrap();
        let (l0, l1) = (
            DeviceBuffer::alloc(rows * 4).unwrap(),
            DeviceBuffer::alloc(rows * 4).unwrap(),
        );
        let (d0, d1) = (
            DeviceBuffer::alloc(rows * v * 2).unwrap(),
            DeviceBuffer::alloc(rows * v * 2).unwrap(),
        );
        let scale = 0.75f32;
        k_old
            .launch(
                (rows as u32, 1, 1),
                (256, 1, 1),
                0,
                &st,
                &Args::new()
                    .buf(&zb)
                    .buf(&tb)
                    .buf(&l0)
                    .buf(&d0)
                    .i32(v as i32)
                    .f32(scale),
            )
            .unwrap();
        k_reg
            .launch(
                (rows as u32, 1, 1),
                (1024, 1, 1),
                0,
                &st,
                &Args::new()
                    .buf(&zb)
                    .buf(&tb)
                    .buf(&l1)
                    .buf(&d1)
                    .i32(v as i32)
                    .f32(scale),
            )
            .unwrap();
        st.synchronize().unwrap();
        let (lo, lr) = (
            l0.download_f32(rows).unwrap(),
            l1.download_f32(rows).unwrap(),
        );
        let worst_l = lo
            .iter()
            .zip(&lr)
            .map(|(a, b)| (a - b).abs() / a.abs().max(1.0))
            .fold(0f32, f32::max);
        assert_eq!(lr[3], 0.0);
        let (dh0, dh1) = (download16(&d0, rows * v), download16(&d1, rows * v));
        let mut worst_d = 0f32;
        let mut differ = 0usize;
        for (a, b) in dh0.iter().zip(&dh1) {
            if a != b {
                differ += 1;
                worst_d = worst_d.max((f16_to_f32(*a) - f16_to_f32(*b)).abs());
            }
        }
        let max_d = dh0.iter().fold(0f32, |s, v| s.max(f16_to_f32(*v).abs()));
        eprintln!("CE {rows}x{v}: loss max rel diff {worst_l:.2e}; dZ: {differ} of {} values differ (max |diff| {worst_d:.2e}, max |dZ| {max_d:.2e})", rows * v);
        assert!(worst_l < 2e-6, "loss {worst_l}");
        assert!(worst_d <= max_d * 2f32.powi(-10), "dZ {worst_d}");
        assert!(
            dh1[3 * v..4 * v].iter().all(|&x| x == 0),
            "ignored row must have a zero gradient"
        );
    }
}

fn small_cfg() -> TrainConfig {
    // Forge-1 structure at a small size: relu2, lambdas, value embeddings, qk-norm; 2 x 512 tokens per micro-batch (the weight-gradient GEMMs reduce over 1024 tokens: split-K is chosen)
    TrainConfig {
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
    }
}

#[test]
fn optimised_step_matches_the_reference_paths() {
    if Device::count().unwrap_or(0) == 0 {
        eprintln!("skipped: no CUDA device");
        return;
    }
    let cfg = small_cfg();
    let n = cfg.tokens();
    let mut r = Lcg(77);
    let tokens: Vec<u32> = (0..cfg.batch * (cfg.seq_len + 1))
        .map(|_| r.next() % cfg.vocab as u32)
        .collect();
    let scale = 1024.0f32;
    let mut results = Vec::new();
    for reference in [true, true, false] {
        for (k, val) in [
            ("FORGE_SPLITK", "1"),
            ("FORGE_NO_FUSE", "1"),
            ("FORGE_CE_OLD", "1"),
            ("FORGE_ATTN_OLD", "1"),
            ("FORGE_NO_FUSE_MIX", "1"),
        ] {
            if reference {
                std::env::set_var(k, val);
            } else {
                std::env::remove_var(k);
            }
        }
        let mut tr = Trainer::new(cfg.clone(), 0).unwrap();
        tr.init_random(5).unwrap();
        tr.zero_grads().unwrap();
        let loss = tr.forward_backward(&tokens, scale, n).unwrap();
        let loss2 = tr.forward_backward(&tokens, scale, n).unwrap(); // a second micro-step accumulates
        let grads = tr.grads_named(1.0 / scale).unwrap();
        results.push((loss, loss2, grads));
    }
    for (k, _) in [
        ("FORGE_SPLITK", "1"),
        ("FORGE_NO_FUSE", "1"),
        ("FORGE_CE_OLD", "1"),
        ("FORGE_ATTN_OLD", "1"),
        ("FORGE_NO_FUSE_MIX", "1"),
    ] {
        std::env::remove_var(k);
    }
    let (ref_loss, ref_loss2, ref_g) = &results[0];
    let (opt_loss, opt_loss2, opt_g) = &results[2];
    // run-to-run noise of the reference paths themselves (the embedding and value-embedding gradients use atomics)
    let rel = |a: &[f32], b: &[f32]| {
        let (mut dd, mut bb) = (0f64, 0f64);
        for (x, y) in a.iter().zip(b) {
            dd += ((x - y) as f64).powi(2);
            bb += (*x as f64).powi(2);
        }
        (dd / bb.max(1e-30)).sqrt()
    };
    let noise = ref_g
        .keys()
        .map(|k| rel(&ref_g[k], &results[1].2[k]))
        .fold(0f64, f64::max);
    eprintln!(
        "run-to-run noise of the reference paths: worst relative gradient difference {noise:.2e}"
    );
    eprintln!(
        "loss reference {ref_loss:.6} / {ref_loss2:.6}, optimised {opt_loss:.6} / {opt_loss2:.6}"
    );
    assert!((ref_loss - opt_loss).abs() < 1e-5 * ref_loss.abs().max(1.0));
    assert!((ref_loss2 - opt_loss2).abs() < 1e-5 * ref_loss2.abs().max(1.0));
    let mut names: Vec<&String> = ref_g.keys().collect();
    names.sort();
    let mut worst = 0f64;
    for name in names {
        let (a, b) = (&ref_g[name], &opt_g[name]);
        let (mut dd, mut bb) = (0f64, 0f64);
        for (x, y) in a.iter().zip(b) {
            dd += ((x - y) as f64).powi(2);
            bb += (*x as f64).powi(2);
        }
        let rel = (dd / bb.max(1e-30)).sqrt();
        worst = worst.max(rel);
        if rel > 1e-6 {
            eprintln!("  grad {name:12}: relative difference {rel:.2e}");
        }
    }
    eprintln!("worst relative gradient difference (optimised vs reference paths): {worst:.2e}");
    assert!(worst < 2e-3, "worst {worst}");
}

/// q, k, v, dO as fp16 device buffers plus their f32 values (b sequences of t tokens, nh query heads, nkv key/value heads of 64)
struct AttnData {
    q: (DeviceBuffer, Vec<f32>),
    k: (DeviceBuffer, Vec<f32>),
    v: (DeviceBuffer, Vec<f32>),
    d_o: (DeviceBuffer, Vec<f32>),
}

fn attn_data(n: usize, nh: usize, nkv: usize) -> AttnData {
    let mk = |len: usize, seed: u64, scale: f32| {
        let (h, f) = rand_f16(len, seed, scale);
        (DeviceBuffer::from_bytes(&bytes16(&h)).unwrap(), f)
    };
    AttnData {
        q: mk(n * nh * 64, 21, 2.0),
        k: mk(n * nkv * 64, 22, 2.0),
        v: mk(n * nkv * 64, 23, 1.0),
        d_o: mk(n * nh * 64, 24, 0.5),
    }
}

/// forward (O fp16, lse), D = rowsum(dO O), then dK dV and dQ with the given kernels; returns (O, lse, dq, dk, dv) on the host
#[allow(clippy::too_many_arguments)]
fn run_attention(
    module: &Module,
    st: &Stream,
    d: &AttnData,
    b: usize,
    t: usize,
    nh: usize,
    nkv: usize,
    fwd: &str,
    dkdv: &str,
    dq: &str,
    window: usize,
) -> (Vec<u16>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let n = b * t;
    let o = DeviceBuffer::alloc(n * nh * 64 * 2).unwrap();
    let lse = DeviceBuffer::alloc(n * nh * 4).unwrap();
    let dvec = DeviceBuffer::alloc(n * nh * 4).unwrap();
    let (dqb, dkb, dvb) = (
        DeviceBuffer::alloc(n * nh * 64 * 4).unwrap(),
        DeviceBuffer::alloc(n * nkv * 64 * 4).unwrap(),
        DeviceBuffer::alloc(n * nkv * 64 * 4).unwrap(),
    );
    let args = |extra: Vec<&DeviceBuffer>| {
        let mut a = Args::new().buf(&d.q.0).buf(&d.k.0).buf(&d.v.0);
        for e in extra {
            a = a.buf(e);
        }
        a.i32(t as i32)
            .i32(nh as i32)
            .i32(nkv as i32)
            .i32(window as i32)
    };
    module
        .function(fwd)
        .unwrap()
        .launch(
            ((t / 64) as u32, nh as u32, b as u32),
            (128, 1, 1),
            0,
            st,
            &args(vec![&o, &lse]),
        )
        .unwrap();
    module
        .function("attn_bwd_prep_kernel")
        .unwrap()
        .launch(
            (((n * nh + 7) / 8) as u32, 1, 1),
            (256, 1, 1),
            0,
            st,
            &Args::new()
                .buf(&o)
                .buf(&d.d_o.0)
                .buf(&dvec)
                .i32(n as i32)
                .i32(nh as i32)
                .i32(64),
        )
        .unwrap();
    module
        .function(dkdv)
        .unwrap()
        .launch(
            ((t / 64) as u32, nkv as u32, b as u32),
            (128, 1, 1),
            0,
            st,
            &args(vec![&d.d_o.0, &lse, &dvec, &dkb, &dvb]),
        )
        .unwrap();
    module
        .function(dq)
        .unwrap()
        .launch(
            ((t / 64) as u32, nh as u32, b as u32),
            (128, 1, 1),
            0,
            st,
            &args(vec![&d.d_o.0, &lse, &dvec, &dqb]),
        )
        .unwrap();
    st.synchronize().unwrap();
    (
        download16(&o, n * nh * 64),
        lse.download_f32(n * nh).unwrap(),
        dqb.download_f32(n * nh * 64).unwrap(),
        dkb.download_f32(n * nkv * 64).unwrap(),
        dvb.download_f32(n * nkv * 64).unwrap(),
    )
}

#[test]
fn new_attention_kernels_are_bit_identical_in_the_dense_case() {
    let Some((_dev, module, st)) = setup() else {
        return;
    };
    let (b, t, nh, nkv) = (2usize, 256usize, 6usize, 2usize);
    let d = attn_data(b * t, nh, nkv);
    let old = run_attention(
        &module,
        &st,
        &d,
        b,
        t,
        nh,
        nkv,
        "flash_fwd_train_64",
        "flash_bwd_dkdv_64",
        "flash_bwd_dq_64",
        0,
    );
    let new = run_attention(
        &module,
        &st,
        &d,
        b,
        t,
        nh,
        nkv,
        "flash_fwd_t64",
        "flash_bwd_dkdv_64",
        "flash_bwd_dq_t64",
        0,
    );
    // the launch-order flag of the dense forward (window != 0) changes only the order of the blocks
    let lpt = run_attention(
        &module,
        &st,
        &d,
        b,
        t,
        nh,
        nkv,
        "flash_fwd_t64",
        "flash_bwd_dkdv_64",
        "flash_bwd_dq_t64",
        1,
    );
    // a window as long as the sequence is the dense case
    let win = run_attention(
        &module,
        &st,
        &d,
        b,
        t,
        nh,
        nkv,
        "flash_fwd_t64_win",
        "flash_bwd_dkdv_64_win",
        "flash_bwd_dq_64_win",
        t,
    );
    for (name, other) in [
        ("prefetching / longest-first kernels", &new),
        ("dense forward in the whole-grid longest-first order", &lpt),
        ("window kernels with window = T", &win),
    ] {
        assert!(old.0 == other.0, "{name}: O differs");
        assert!(
            old.1
                .iter()
                .zip(&other.1)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "{name}: lse differs"
        );
        assert!(
            old.2
                .iter()
                .zip(&other.2)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "{name}: dq differs"
        );
        assert!(
            old.3
                .iter()
                .zip(&other.3)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "{name}: dk differs"
        );
        assert!(
            old.4
                .iter()
                .zip(&other.4)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "{name}: dv differs"
        );
        eprintln!("{name}: O, lse, dq, dk, dv bit-identical to the first-generation kernels");
    }
}

#[test]
fn sliding_window_attention_matches_a_dense_masked_cpu_reference() {
    let Some((_dev, module, st)) = setup() else {
        return;
    };
    let (b, t, nh, nkv) = (2usize, 256usize, 4usize, 2usize);
    let d = attn_data(b * t, nh, nkv);
    for window in [100usize, 64, 1, 200] {
        let got = run_attention(
            &module,
            &st,
            &d,
            b,
            t,
            nh,
            nkv,
            "flash_fwd_t64_win",
            "flash_bwd_dkdv_64_win",
            "flash_bwd_dq_64_win",
            window,
        );
        // CPU reference in f64: query i of a sequence sees keys j with i - window < j <= i
        let (qd, kvd, rep) = (nh * 64, nkv * 64, nh / nkv);
        let (q, k, v, d_o) = (&d.q.1, &d.k.1, &d.v.1, &d.d_o.1);
        let mut o_ref = vec![0f64; b * t * qd];
        let mut lse_ref = vec![0f64; b * t * nh];
        let mut dq_ref = vec![0f64; b * t * qd];
        let mut dk_ref = vec![0f64; b * t * kvd];
        let mut dv_ref = vec![0f64; b * t * kvd];
        for s in 0..b {
            for h in 0..nh {
                let kh = h / rep;
                for i in 0..t {
                    let qi = (s * t + i) * qd + h * 64;
                    let lo = (i + 1).saturating_sub(window);
                    let scores: Vec<f64> = (lo..=i)
                        .map(|j| {
                            (0..64)
                                .map(|x| {
                                    q[qi + x] as f64 * k[(s * t + j) * kvd + kh * 64 + x] as f64
                                })
                                .sum::<f64>()
                                * 0.125
                        })
                        .collect();
                    let mx = scores.iter().cloned().fold(f64::MIN, f64::max);
                    let sum: f64 = scores.iter().map(|z| (z - mx).exp()).sum();
                    let p: Vec<f64> = scores.iter().map(|z| (z - mx).exp() / sum).collect();
                    lse_ref[(s * t + i) * nh + h] = (mx + sum.ln()) / std::f64::consts::LN_2;
                    for (jj, j) in (lo..=i).enumerate() {
                        for x in 0..64 {
                            o_ref[qi + x] += p[jj] * v[(s * t + j) * kvd + kh * 64 + x] as f64;
                        }
                    }
                    let dvec: f64 = (0..64).map(|x| d_o[qi + x] as f64 * o_ref[qi + x]).sum();
                    for (jj, j) in (lo..=i).enumerate() {
                        let kj = (s * t + j) * kvd + kh * 64;
                        let dp: f64 = (0..64).map(|x| d_o[qi + x] as f64 * v[kj + x] as f64).sum();
                        let ds = p[jj] * (dp - dvec);
                        for x in 0..64 {
                            dq_ref[qi + x] += ds * 0.125 * k[kj + x] as f64;
                            dk_ref[kj + x] += ds * 0.125 * q[qi + x] as f64;
                            dv_ref[kj + x] += p[jj] * d_o[qi + x] as f64;
                        }
                    }
                }
            }
        }
        let rel = |a: &[f32], r: &[f64]| {
            let (mut dd, mut rr) = (0f64, 0f64);
            for (x, y) in a.iter().zip(r) {
                dd += (*x as f64 - y).powi(2);
                rr += y * y;
            }
            (dd / rr.max(1e-30)).sqrt()
        };
        let o_f: Vec<f32> = got.0.iter().map(|&h| f16_to_f32(h)).collect();
        let lse_err = got
            .1
            .iter()
            .zip(&lse_ref)
            .map(|(a, r)| (*a as f64 - r).abs())
            .fold(0f64, f64::max);
        let (eo, eq, ek, ev) = (
            rel(&o_f, &o_ref),
            rel(&got.2, &dq_ref),
            rel(&got.3, &dk_ref),
            rel(&got.4, &dv_ref),
        );
        eprintln!("window {window}: relative error O {eo:.2e}, dq {eq:.2e}, dk {ek:.2e}, dv {ev:.2e}; max |lse diff| {lse_err:.2e}");
        assert!(
            eo < 2e-3 && lse_err < 1e-3,
            "forward: O {eo}, lse {lse_err}"
        );
        // with a window of 1 a query sees only itself: p = 1 and dS = dO.v - dO.o = 0, so dq and dk are exactly zero and only rounding is left on the GPU (relative errors are meaningless)
        let tiny = |a: &[f32]| a.iter().fold(0f32, |m, v| m.max(v.abs())) < 1e-3;
        assert!(
            (eq < 1e-2 || (window == 1 && tiny(&got.2)))
                && (ek < 1e-2 || (window == 1 && tiny(&got.3)))
                && ev < 1e-2,
            "backward: dq {eq}, dk {ek}, dv {ev}"
        );
    }
}

#[test]
fn windowed_model_trains_and_its_checkpoint_keeps_the_window() {
    if Device::count().unwrap_or(0) == 0 {
        eprintln!("skipped: no CUDA device");
        return;
    }
    let mut cfg = small_cfg();
    let n = cfg.tokens();
    let mut r = Lcg(78);
    let tokens: Vec<u32> = (0..cfg.batch * (cfg.seq_len + 1))
        .map(|_| r.next() % cfg.vocab as u32)
        .collect();
    let mut dense = Trainer::new(cfg.clone(), 0).unwrap();
    dense.init_random(5).unwrap();
    let l_dense = dense.eval_loss(&tokens).unwrap();
    drop(dense);
    cfg.window = 96;
    cfg.window_layers = TrainConfig::auto_window_layers(cfg.n_layer);
    assert_eq!(cfg.window_layers, vec![0, 1]);
    let mut tr = Trainer::new(cfg.clone(), 0).unwrap();
    tr.init_random(5).unwrap();
    let l_win = tr.eval_loss(&tokens).unwrap();
    tr.zero_grads().unwrap();
    let l_fb = tr.forward_backward(&tokens, 1024.0, n).unwrap();
    let g = tr.grads_named(1.0 / 1024.0).unwrap();
    let finite = g.values().all(|v| v.iter().all(|x| x.is_finite()));
    eprintln!("loss dense {l_dense:.5}, window 96 in layers {:?}: {l_win:.5} (forward/backward {l_fb:.5}), gradients finite: {finite}", cfg.window_layers);
    assert!(finite && (l_win - l_fb).abs() < 1e-5);
    assert!(
        (l_win - l_dense).abs() > 1e-6,
        "the window must change the model"
    );
    let path = std::env::temp_dir().join(format!("kopt_window_{}.ckpt", std::process::id()));
    tr.save_checkpoint(&path, 1024.0).unwrap();
    let back = forge_train::config_from_checkpoint(&path).unwrap();
    std::fs::remove_file(&path).ok();
    assert_eq!((back.window, back.window_layers.clone()), (96, vec![0, 1]));
    assert_eq!(back.structure(), cfg.structure());
}

#[test]
fn split_k_model_picks_the_measured_best_factors() {
    // kbench on a T4 (40 SMs), job kernel-opt-job1b: the fastest split factor of each weight-gradient shape (tiles of 128 x 128, reduction over the tokens)
    for (tiles, k, best) in [
        (90usize, 4096usize, 4usize),
        (25, 8192, 8),
        (40, 4096, 2),
        (90, 8192, 4),
        (40, 8192, 2),
        (1280, 2048, 1),
    ] {
        assert_eq!(
            forge_train::split_k_turing(tiles, k, 40),
            best,
            "{tiles} tiles, k {k}"
        );
    }
    // kbench on the 2060 SUPER (34 SMs) at 2048 / 4096 tokens: the fastest factor of 576 x 2304 (90 tiles), 576 x 576 (25), 960 x 576 (40) and the head (1280)
    for (tiles, k, best) in [
        (90usize, 2048usize, 2usize),
        (25, 2048, 4),
        (40, 2048, 4),
        (90, 4096, 4),
        (25, 4096, 8),
        (40, 4096, 4),
        (1280, 2048, 1),
        (1280, 4096, 1),
    ] {
        assert_eq!(
            forge_train::split_k_turing(tiles, k, 34),
            best,
            "34 SMs: {tiles} tiles, k {k}"
        );
    }
    // the 576 x 576 shape at 4096 tokens: measured 0.192 / 0.185 / 0.181 ms for 2 / 4 / 8 slices; the model picks 2 (within 6 %)
    assert!([2, 4, 8].contains(&forge_train::split_k_turing(25, 4096, 40)));
}

#[test]
fn muon_on_several_streams_is_bit_identical_to_one_stream() {
    if Device::count().unwrap_or(0) == 0 {
        eprintln!("skipped: no CUDA device");
        return;
    }
    let cfg = small_cfg();
    let n = cfg.tokens();
    let mut r = Lcg(79);
    let tokens: Vec<u32> = (0..cfg.batch * (cfg.seq_len + 1))
        .map(|_| r.next() % cfg.vocab as u32)
        .collect();
    // the same gradients in both trainers (the backward itself has atomics): computed once, uploaded into both
    let mut grads: Option<Vec<f32>> = None;
    let mut params = Vec::new();
    for streams in ["1", "4"] {
        std::env::set_var("FORGE_MUON_STREAMS", streams);
        let mut tr = Trainer::new(cfg.clone(), 0).unwrap();
        std::env::remove_var("FORGE_MUON_STREAMS");
        tr.init_random(9).unwrap();
        if grads.is_none() {
            tr.zero_grads().unwrap();
            tr.forward_backward(&tokens, 1024.0, n).unwrap();
            grads = Some(tr.download_grads().unwrap());
        }
        for _ in 0..2 {
            tr.upload_grads(grads.as_ref().unwrap()).unwrap();
            let s = tr
                .step_muon(1e-3, 2e-2, 0.9, 0.95, 1e-8, 0.1, 0.05, 0.95, 1.0, 1024.0)
                .unwrap();
            assert!(!s.found_inf);
        }
        params.push(tr.params_named().unwrap());
    }
    let mut names: Vec<&String> = params[0].keys().collect();
    names.sort();
    let mut changed = 0;
    for name in names {
        assert!(
            params[0][name]
                .iter()
                .zip(&params[1][name])
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "{name} differs between 1 and 4 Muon streams"
        );
        changed += 1;
    }
    eprintln!(
        "{changed} parameter tensors bit-identical after two Muon steps on 1 and on 4 streams"
    );
}

#[test]
fn gradient_difference_of_each_optimisation_alone() {
    if Device::count().unwrap_or(0) == 0 {
        eprintln!("skipped: no CUDA device");
        return;
    }
    let cfg = small_cfg();
    let n = cfg.tokens();
    let mut r = Lcg(77);
    let tokens: Vec<u32> = (0..cfg.batch * (cfg.seq_len + 1))
        .map(|_| r.next() % cfg.vocab as u32)
        .collect();
    let switches = [
        ("FORGE_SPLITK", "1"),
        ("FORGE_NO_FUSE", "1"),
        ("FORGE_CE_OLD", "1"),
        ("FORGE_ATTN_OLD", "1"),
        ("FORGE_NO_FUSE_MIX", "1"),
    ];
    let run = |on: Option<usize>| {
        // every switch set (= the reference path) except the one optimisation under test
        for (i, (k, v)) in switches.iter().enumerate() {
            if Some(i) == on {
                std::env::remove_var(k);
            } else {
                std::env::set_var(k, v);
            }
        }
        let mut tr = Trainer::new(cfg.clone(), 0).unwrap();
        tr.init_random(5).unwrap();
        tr.zero_grads().unwrap();
        tr.forward_backward(&tokens, 1024.0, n).unwrap();
        tr.grads_named(1.0 / 1024.0).unwrap()
    };
    let reference = run(None);
    let mut worst_all = 0f64;
    for (i, (k, _)) in switches.iter().enumerate() {
        let g = run(Some(i));
        let mut worst = (0f64, String::new());
        for (name, a) in &reference {
            let b = &g[name];
            let (mut dd, mut aa) = (0f64, 0f64);
            for (x, y) in a.iter().zip(b) {
                dd += ((x - y) as f64).powi(2);
                aa += (*x as f64).powi(2);
            }
            let rel = (dd / aa.max(1e-30)).sqrt();
            if rel > worst.0 {
                worst = (rel, name.clone());
            }
        }
        eprintln!(
            "only the optimisation behind {k} on: worst relative gradient difference {:.2e} ({})",
            worst.0, worst.1
        );
        worst_all = worst_all.max(worst.0);
    }
    for (k, _) in switches {
        std::env::remove_var(k);
    }
    assert!(worst_all < 2e-3);
}

#[test]
fn fused_mixing_and_norm_forward_is_bit_identical() {
    let Some((_dev, module, st)) = setup() else {
        return;
    };
    let (n, e) = (300usize, 576usize);
    let mut r = Lcg(41);
    let xp: Vec<f32> = (0..n * e).map(|_| r.uniform() * 8.0 - 4.0).collect();
    let x0: Vec<f32> = (0..n * e).map(|_| r.uniform() * 2.0 - 1.0).collect();
    let w: Vec<f32> = (0..e).map(|_| 0.5 + r.uniform()).collect();
    let up = |v: &[f32]| {
        let b = DeviceBuffer::alloc(v.len() * 4).unwrap();
        b.upload_f32(v).unwrap();
        b
    };
    let (bxp, bx0, bw, bsc) = (up(&xp), up(&x0), up(&w), up(&[0.9, 0.35, 0.0]));
    let (x_a, x_b) = (
        DeviceBuffer::alloc(n * e * 4).unwrap(),
        DeviceBuffer::alloc(n * e * 4).unwrap(),
    );
    let (h_a, h_b) = (
        DeviceBuffer::alloc(n * e * 2).unwrap(),
        DeviceBuffer::alloc(n * e * 2).unwrap(),
    );
    let (r_a, r_b) = (
        DeviceBuffer::alloc(n * 4).unwrap(),
        DeviceBuffer::alloc(n * 4).unwrap(),
    );
    let ne = n * e;
    module
        .function("mix_fwd_kernel")
        .unwrap()
        .launch(
            (((ne + 255) / 256) as u32, 1, 1),
            (256, 1, 1),
            0,
            &st,
            &Args::new()
                .buf(&bxp)
                .buf(&bx0)
                .buf(&bxp)
                .buf(&x_a)
                .u64(ne as u64)
                .buf(&bsc)
                .i32(1)
                .i32(0),
        )
        .unwrap();
    module
        .function("rmsnorm_fwd_kernel")
        .unwrap()
        .launch(
            (((n + 7) / 8) as u32, 1, 1),
            (256, 1, 1),
            0,
            &st,
            &Args::new()
                .buf(&x_a)
                .buf(&bw)
                .buf(&h_a)
                .buf(&r_a)
                .i32(n as i32)
                .i32(e as i32)
                .f32(1e-5),
        )
        .unwrap();
    module
        .function("mix_rmsnorm_fwd_kernel")
        .unwrap()
        .launch(
            (((n + 7) / 8) as u32, 1, 1),
            (256, 1, 1),
            0,
            &st,
            &Args::new()
                .buf(&bxp)
                .buf(&bx0)
                .buf(&bsc)
                .buf(&x_b)
                .buf(&bw)
                .buf(&h_b)
                .buf(&r_b)
                .i32(n as i32)
                .i32(e as i32)
                .f32(1e-5),
        )
        .unwrap();
    st.synchronize().unwrap();
    let bits = |b: &DeviceBuffer, len: usize| {
        b.download_f32(len)
            .unwrap()
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<u32>>()
    };
    assert_eq!(bits(&x_a, ne), bits(&x_b, ne), "x differs");
    assert_eq!(bits(&r_a, n), bits(&r_b, n), "rstd differs");
    assert_eq!(download16(&h_a, ne), download16(&h_b, ne), "h differs");
    eprintln!("mixing + RMSNorm forward: x, h and rstd of {n} x {e} bit-identical to mix_fwd_kernel + rmsnorm_fwd_kernel");
}

/// The dS spill backward (flash_bwd_dkdv_ds_64 writes the fp16 dS^T tiles, flash_bwd_dq_ds_64 reads them) against the three-matmul dQ kernel: dq, dk and dv bit-identical
#[test]
fn dense_backward_with_the_ds_spill_is_bit_identical() {
    let Some((_dev, module, st)) = setup() else {
        return;
    };
    for (b, t, nh, nkv) in [(2usize, 256usize, 6usize, 2usize), (1, 2048, 9, 3), (2, 64, 3, 1)] {
        let d = attn_data(b * t, nh, nkv);
        let old = run_attention(
            &module,
            &st,
            &d,
            b,
            t,
            nh,
            nkv,
            "flash_fwd_t64",
            "flash_bwd_dkdv_64",
            "flash_bwd_dq_t64",
            0,
        );
        let n = b * t;
        let o = DeviceBuffer::alloc(n * nh * 64 * 2).unwrap();
        let lse = DeviceBuffer::alloc(n * nh * 4).unwrap();
        let dvec = DeviceBuffer::alloc(n * nh * 4).unwrap();
        let dqb = DeviceBuffer::alloc(n * nh * 64 * 4).unwrap();
        let dkb = DeviceBuffer::alloc(n * nkv * 64 * 4).unwrap();
        let dvb = DeviceBuffer::alloc(n * nkv * 64 * 4).unwrap();
        let nt = t / 64;
        let ds = DeviceBuffer::alloc(b * nh * (nt * (nt + 1) / 2) * 4096 * 2).unwrap();
        module
            .function("flash_fwd_t64")
            .unwrap()
            .launch(
                ((t / 64) as u32, nh as u32, b as u32),
                (128, 1, 1),
                0,
                &st,
                &Args::new()
                    .buf(&d.q.0)
                    .buf(&d.k.0)
                    .buf(&d.v.0)
                    .buf(&o)
                    .buf(&lse)
                    .i32(t as i32)
                    .i32(nh as i32)
                    .i32(nkv as i32)
                    .i32(0),
            )
            .unwrap();
        module
            .function("attn_bwd_prep_kernel")
            .unwrap()
            .launch(
                (((n * nh + 7) / 8) as u32, 1, 1),
                (256, 1, 1),
                0,
                &st,
                &Args::new()
                    .buf(&o)
                    .buf(&d.d_o.0)
                    .buf(&dvec)
                    .i32(n as i32)
                    .i32(nh as i32)
                    .i32(64),
            )
            .unwrap();
        module
            .function("flash_bwd_dkdv_ds_64")
            .unwrap()
            .launch(
                ((t / 64) as u32, nkv as u32, b as u32),
                (128, 1, 1),
                0,
                &st,
                &Args::new()
                    .buf(&d.q.0)
                    .buf(&d.k.0)
                    .buf(&d.v.0)
                    .buf(&d.d_o.0)
                    .buf(&lse)
                    .buf(&dvec)
                    .buf(&dkb)
                    .buf(&dvb)
                    .buf(&ds)
                    .i32(t as i32)
                    .i32(nh as i32)
                    .i32(nkv as i32),
            )
            .unwrap();
        module
            .function("flash_bwd_dq_ds_64")
            .unwrap()
            .launch(
                ((t / 64) as u32, nh as u32, b as u32),
                (128, 1, 1),
                0,
                &st,
                &Args::new()
                    .buf(&d.k.0)
                    .buf(&ds)
                    .buf(&dqb)
                    .i32(t as i32)
                    .i32(nh as i32)
                    .i32(nkv as i32),
            )
            .unwrap();
        st.synchronize().unwrap();
        let dq = dqb.download_f32(n * nh * 64).unwrap();
        let dk = dkb.download_f32(n * nkv * 64).unwrap();
        let dv = dvb.download_f32(n * nkv * 64).unwrap();
        let same = |a: &[f32], b: &[f32]| a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits());
        assert!(same(&old.2, &dq), "b={b} t={t}: dq differs");
        assert!(same(&old.3, &dk), "b={b} t={t}: dk differs");
        assert!(same(&old.4, &dv), "b={b} t={t}: dv differs");
        eprintln!("b={b} t={t} heads {nh}/{nkv}: dq, dk, dv bit-identical with the dS spill");
    }
}
