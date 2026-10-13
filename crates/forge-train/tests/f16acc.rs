//! fp16-accumulate forward GEMMs (`gemm_nt_{f16,res,relu2}_hP`, FORGE_F16ACC=P): accuracy against a float64 reference on the Forge-1 forward shapes (d 576, qkv 960, MLP 2304), compared with the
//! fp32-accumulate kernels and with a CPU emulation of the fp16 accumulator (k = 8 per MMA step, the result of each step rounded to fp16 to nearest or toward zero, added into fp32 every P steps); the
//! rounding of the hardware accumulator; and the trainer switch (FORGE_F16ACC unset = the fp32-accumulate path: identity check; FORGE_F16ACC=8 changes the results by about the fp16 rounding only).
//! Error unit: the RMS error of rounding the exact product to fp16 (about 2.07e-4 relative). The numerics study of the production checkpoint (research note: ForgeInfer - fp16 Accumulate on GeForce Turing)
//! puts the emulated error at P = 8 at 1.5 to 2.5 units with rounding to nearest; acceptance: at most 3 units. Real operands of the production checkpoint (written by
//! scratch/agents/f16acc/dump_operands.py) are checked too when FORGE_F16ACC_DATA points to their directory. Small sizes (512 rows; well under 100 MB of GPU memory).

use forge_core::half::{f16_to_f32, f32_to_f16};
use forge_cuda::{Args, Device, DeviceBuffer, Module, Stream};
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
        eprintln!("skipped: the fp16-accumulate kernels are Turing kernels (sm_75)");
        return None;
    }
    let (image, _) = forge_kernels::image_for(dev.info.cc_major, dev.info.cc_minor).unwrap();
    let module = Module::load(image).unwrap();
    Some((dev, module, Stream::new().unwrap()))
}

/// output kinds of the forward GEMMs: 0 fp16 C (qkv), 1 fp32 C = R + A W^T (attention output and MLP down projection, residual stream), 2 the ReLU^2 epilogue (MLP up projection: C = u, R = relu(u)^2)
const OLD: [&str; 3] = ["gemm_nt_f16", "gemm_nt_res", "gemm_nt_relu2"];

/// A [m][k] x W [n][k]^T on the GPU with kernel `name` of output kind `kind`; returns the product as f32: the fp16 output (kinds 0, 2) or C - R (kind 1, with a non-zero residual R)
fn run_gemm(
    module: &Module,
    st: &Stream,
    name: &str,
    kind: usize,
    a: &DeviceBuffer,
    w: &DeviceBuffer,
    m: usize,
    n: usize,
    k: usize,
) -> Vec<f32> {
    let func = module.function(name).unwrap();
    let grid = (((n + 127) / 128) as u32, ((m + 127) / 128) as u32, 1u32);
    let c = DeviceBuffer::alloc(m * n * 4).unwrap();
    let r = DeviceBuffer::alloc(m * n * 4).unwrap();
    let resid: Vec<f32> = (0..m * n)
        .map(|i| ((i % 89) as f32 - 44.0) * 0.03)
        .collect();
    if kind == 1 {
        r.upload_f32(&resid).unwrap();
    }
    let mut args = Args::new()
        .buf(a)
        .buf(w)
        .buf(&c)
        .i32(m as i32)
        .i32(n as i32)
        .i32(k as i32)
        .i32(k as i32)
        .i32(k as i32)
        .i32(n as i32)
        .f32(1.0);
    if kind != 0 {
        args = args.buf(&r).i32(n as i32).f32(0.0);
    }
    func.launch(grid, (256, 1, 1), 0, st, &args).unwrap();
    st.synchronize().unwrap();
    match kind {
        1 => c
            .download_f32(m * n)
            .unwrap()
            .iter()
            .zip(&resid)
            .map(|(c, r)| c - r)
            .collect(),
        _ => download16(&c, m * n)
            .iter()
            .map(|&h| f16_to_f32(h))
            .collect(),
    }
}

/// rows [0, m) split over three threads (the CPU references are the slow part of these tests)
fn par_rows<T: Send + Default + Clone>(
    m: usize,
    n: usize,
    f: impl Fn(usize, &mut [T]) + Sync,
) -> Vec<T> {
    let mut out = vec![T::default(); m * n];
    let chunk = (m + 2) / 3;
    std::thread::scope(|s| {
        for (ci, part) in out.chunks_mut(chunk * n).enumerate() {
            let f = &f;
            s.spawn(move || {
                for (r, row) in part.chunks_mut(n).enumerate() {
                    f(ci * chunk + r, row);
                }
            });
        }
    });
    out
}

/// the float64 product
fn reference(a: &[f32], w: &[f32], m: usize, n: usize, k: usize) -> Vec<f64> {
    par_rows(m, n, |i, row: &mut [f64]| {
        let ar = &a[i * k..(i + 1) * k];
        for (j, o) in row.iter_mut().enumerate() {
            *o = ar
                .iter()
                .zip(&w[j * k..(j + 1) * k])
                .map(|(x, y)| *x as f64 * *y as f64)
                .sum();
        }
    })
}

/// fp32 -> fp16 rounded toward zero
fn f16_rz(x: f32) -> u16 {
    let h = f32_to_f16(x);
    if f16_to_f32(h).abs() > x.abs() {
        h - 1
    } else {
        h
    }
}

/// CPU emulation of the fp16 accumulator: per k8 step the sum of the 8 products (exact, rounded to f32) is added to the fp16 accumulator in f32 and the result rounded to fp16 (to nearest even, or toward
/// zero); after every `p` steps (and at the end) the fp16 value is added into an f32 accumulator and the fp16 accumulator restarts from zero. Returns the f32 accumulator.
fn emulate(a: &[f32], w: &[f32], m: usize, n: usize, k: usize, p: usize, rz: bool) -> Vec<f32> {
    par_rows(m, n, |i, row: &mut [f32]| {
        let ar = &a[i * k..(i + 1) * k];
        for (j, o) in row.iter_mut().enumerate() {
            let wr = &w[j * k..(j + 1) * k];
            let (mut acc32, mut acc16) = (0f32, 0f32);
            for s in 0..k / 8 {
                let part: f64 = (0..8)
                    .map(|x| ar[8 * s + x] as f64 * wr[8 * s + x] as f64)
                    .sum();
                let v = acc16 + part as f32;
                acc16 = f16_to_f32(if rz { f16_rz(v) } else { f32_to_f16(v) });
                if (s + 1) % p == 0 {
                    acc32 += acc16;
                    acc16 = 0.0;
                }
            }
            *o = acc32 + acc16;
        }
    })
}

/// relative RMS error against the reference
fn rel(x: impl Iterator<Item = f64>, r: &[f64]) -> f64 {
    let (mut dd, mut rr) = (0f64, 0f64);
    for (v, y) in x.zip(r) {
        dd += (v - y) * (v - y);
        rr += y * y;
    }
    (dd / rr.max(1e-300)).sqrt()
}

struct Eval {
    /// errors in units of the fp16 rounding of the exact product: the fp32-accumulate kernel, the fp16-accumulate kernels at P = 2, 4, 8, 16, plain fp16 accumulation (fp16 output), the emulation at P = 8
    old: f64,
    h: [f64; 4],
    never: f64,
    emu_rn: f64,
    emu_rz: f64,
    /// RMS difference between the P = 8 and the fp32-accumulate kernel outputs, same unit
    diff8: f64,
}

/// one forward GEMM of kind `kind` with fp16 operands A [m][k], W [n][k]
fn evaluate(
    module: &Module,
    st: &Stream,
    label: &str,
    kind: usize,
    ah: &[u16],
    wh: &[u16],
    m: usize,
    n: usize,
    k: usize,
) -> Eval {
    let af: Vec<f32> = ah.iter().map(|&h| f16_to_f32(h)).collect();
    let wf: Vec<f32> = wh.iter().map(|&h| f16_to_f32(h)).collect();
    let r = reference(&af, &wf, m, n, k);
    let unit = rel(
        r.iter().map(|&v| f16_to_f32(f32_to_f16(v as f32)) as f64),
        &r,
    );
    let (da, dw) = (
        DeviceBuffer::from_bytes(&bytes16(ah)).unwrap(),
        DeviceBuffer::from_bytes(&bytes16(wh)).unwrap(),
    );
    let err = |v: &[f32]| rel(v.iter().map(|&x| x as f64), &r) / unit;
    let old = run_gemm(module, st, OLD[kind], kind, &da, &dw, m, n, k);
    let mut h = [0f64; 4];
    let mut out8 = Vec::new();
    for (i, p) in [2usize, 4, 8, 16].iter().enumerate() {
        let got = run_gemm(
            module,
            st,
            &format!("{}_h{p}", OLD[kind]),
            kind,
            &da,
            &dw,
            m,
            n,
            k,
        );
        h[i] = err(&got);
        if *p == 8 {
            out8 = got;
        }
    }
    let never = err(&run_gemm(
        module,
        st,
        "gemm_nt_f16_h0",
        0,
        &da,
        &dw,
        m,
        n,
        k,
    ));
    // the emulation in the kernel's output format (fp16 for kinds 0 and 2, f32 for kind 1)
    let fmt = |x: f32| {
        if kind == 1 {
            x as f64
        } else {
            f16_to_f32(f32_to_f16(x)) as f64
        }
    };
    let emu_rn = rel(
        emulate(&af, &wf, m, n, k, 8, false).iter().map(|&x| fmt(x)),
        &r,
    ) / unit;
    let emu_rz = rel(
        emulate(&af, &wf, m, n, k, 8, true).iter().map(|&x| fmt(x)),
        &r,
    ) / unit;
    let ofl: Vec<f64> = old.iter().map(|&x| x as f64).collect();
    let diff8 = rel(out8.iter().map(|&x| x as f64), &ofl) / unit;
    let e = Eval {
        old: err(&old),
        h,
        never,
        emu_rn,
        emu_rz,
        diff8,
    };
    eprintln!(
        "{label:16} {m}x{n}x{k} ({}): unit {unit:.2e} | fp32 acc {:.2} | fp16 acc P2 {:.2} P4 {:.2} P8 {:.2} P16 {:.2} never(fp16 out) {:.2} | emulated P8 nearest {:.2} toward-zero {:.2} | P8 vs fp32-acc kernel {:.2}",
        ["fp16 out", "fp32 out + residual", "relu2 epilogue"][kind],
        e.old,
        e.h[0],
        e.h[1],
        e.h[2],
        e.h[3],
        e.never,
        e.emu_rn,
        e.emu_rz,
        e.diff8
    );
    e
}

fn rand_f16(n: usize, seed: u64, f: impl Fn(f32) -> f32) -> Vec<u16> {
    let mut r = Lcg(seed);
    (0..n)
        .map(|_| f32_to_f16(f(r.uniform() * 2.0 - 1.0)))
        .collect()
}

#[test]
fn f16acc_forward_gemms_stay_within_three_rounding_units_at_p8() {
    let Some((_dev, module, st)) = setup() else {
        return;
    };
    let m = 512;
    // (label, kind, N, K): the four forward GEMMs of a Forge-1 layer; the MLP down projection reads relu(u)^2 (non-negative, heavy tailed)
    for (label, kind, n, k) in [
        ("qkv", 0usize, 960usize, 576usize),
        ("attn out (res)", 1, 576, 576),
        ("mlp up (relu2)", 2, 2304, 576),
        ("mlp down (res)", 1, 576, 2304),
    ] {
        let a = if label.starts_with("mlp down") {
            rand_f16(m * k, 11 + k as u64, |u| 6.0 * u.max(0.0) * u.max(0.0))
        } else {
            rand_f16(m * k, 11 + k as u64, |u| 2.5 * u)
        };
        let w = rand_f16(n * k, 12 + n as u64, |u| 0.06 * u);
        let e = evaluate(&module, &st, label, kind, &a, &w, m, n, k);
        assert!(e.h[2] <= 3.0, "{label}: P8 error {:.2} units", e.h[2]);
        assert!(
            e.diff8 <= 3.5,
            "{label}: P8 differs from the fp32-accumulate kernel by {:.2} units",
            e.diff8
        );
        // the kernel must not be worse than the emulation with rounding toward zero, nor much better than the exact-step emulation can be
        assert!(
            e.h[2] < e.emu_rz && e.h[2] > 0.5 * e.emu_rn,
            "{label}: P8 {:.2} vs emulation {:.2} / {:.2}",
            e.h[2],
            e.emu_rn,
            e.emu_rz
        );
        // the flush interval sets the error: more frequent flushes are not worse
        assert!(
            e.h[0] <= e.h[2] * 1.05 && e.h[2] <= e.h[3] * 1.05,
            "{label}: P2 {:.2} P8 {:.2} P16 {:.2}",
            e.h[0],
            e.h[2],
            e.h[3]
        );
    }
}

#[test]
fn f16acc_real_operands_of_the_production_checkpoint() {
    let Ok(dir) = std::env::var("FORGE_F16ACC_DATA") else {
        eprintln!("skipped: FORGE_F16ACC_DATA (directory of dump_operands.py) not set");
        return;
    };
    let Some((_dev, module, st)) = setup() else {
        return;
    };
    let index = std::fs::read_to_string(format!("{dir}/index.txt")).unwrap();
    let read16 = |p: String| -> Vec<u16> {
        std::fs::read(&p)
            .unwrap_or_else(|e| panic!("{p}: {e}"))
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect()
    };
    let mut worst = (0f64, String::new());
    let mut sum = [0f64; 4];
    let mut cnt = 0usize;
    for line in index.lines() {
        let f: Vec<&str> = line.split_whitespace().collect();
        let (layer, name, m, n, k) = (
            f[0],
            f[1],
            f[2].parse::<usize>().unwrap(),
            f[3].parse::<usize>().unwrap(),
            f[4].parse::<usize>().unwrap(),
        );
        let kind = match name {
            "qkv" => 0,
            "up" => 2,
            _ => 1,
        };
        let a = read16(format!("{dir}/L{layer}_{name}_A.f16"));
        let w = read16(format!("{dir}/L{layer}_{name}_W.f16"));
        let e = evaluate(
            &module,
            &st,
            &format!("L{layer} {name}"),
            kind,
            &a,
            &w,
            m,
            n,
            k,
        );
        if e.h[2] > worst.0 {
            worst = (e.h[2], format!("L{layer} {name}"));
        }
        sum[0] += e.h[2];
        sum[1] += e.emu_rn;
        sum[2] += e.emu_rz;
        sum[3] += e.h[3];
        cnt += 1;
    }
    let c = cnt as f64;
    eprintln!("{cnt} GEMMs: mean error at P8 {:.2} units (emulation: nearest {:.2}, toward zero {:.2}), P16 {:.2}; worst P8 {:.2} ({})", sum[0] / c, sum[1] / c, sum[2] / c, sum[3] / c, worst.0, worst.1);
    assert!(
        worst.0 <= 3.0,
        "worst P8 error {:.2} units ({})",
        worst.0,
        worst.1
    );
}

#[test]
fn f16acc_rounding_of_the_hardware_accumulator() {
    // C[i][0] = 1 + x_i with x_i a fraction of the fp16 ulp of 1 (2^-10), added either in a second MMA step (rows 0-3: the fp16 accumulator holds 1.0, then gets x) or inside one step (rows 4-7: products
    // 1.0 and x in the same k8 step). Kernel gemm_nt_f16_h0 (never flushed: the output is the fp16 accumulator). Rounding to nearest even gives 1 + ulp for 0.75 and 0.625, 1 for 0.5 and 0.25; toward zero 1.
    let Some((_dev, module, st)) = setup() else {
        return;
    };
    let (m, n, k) = (128usize, 128usize, 32usize);
    let fr = [0.75f32, 0.5, 0.25, 0.625];
    let mut a = vec![0u16; m * k];
    let mut w = vec![0u16; n * k];
    for (i, x) in fr.iter().enumerate() {
        a[i * k] = f32_to_f16(1.0);
        a[i * k + 8] = f32_to_f16(x / 1024.0);
        a[(4 + i) * k] = f32_to_f16(1.0);
        a[(4 + i) * k + 1] = f32_to_f16(x / 1024.0);
    }
    w[0] = f32_to_f16(1.0);
    w[1] = f32_to_f16(1.0);
    w[8] = f32_to_f16(1.0);
    let (da, dw) = (
        DeviceBuffer::from_bytes(&bytes16(&a)).unwrap(),
        DeviceBuffer::from_bytes(&bytes16(&w)).unwrap(),
    );
    let h = run_gemm(&module, &st, "gemm_nt_f16_h0", 0, &da, &dw, m, n, k);
    let f = run_gemm(&module, &st, "gemm_nt_f16", 0, &da, &dw, m, n, k);
    let ulps = |v: f32| (v - 1.0) * 1024.0;
    let mut line = String::new();
    for (i, x) in fr.iter().enumerate() {
        line += &format!(
            "  x = {x} ulp: second step -> 1 + {} ulp, same step -> 1 + {} ulp (fp32 acc: 1 + {})",
            ulps(h[i * n]),
            ulps(h[(4 + i) * n]),
            ulps(f[i * n])
        );
    }
    eprintln!("fp16 accumulator rounding:{line}");
    let up = |v: f32| ulps(v) == 1.0;
    let mode = match (up(h[0]), up(h[n]), up(h[3 * n])) {
        (true, false, true) => "round to nearest even",
        (true, true, true) => "round to nearest, ties away",
        (false, false, false) => "toward zero (truncation)",
        _ => "other",
    };
    eprintln!("hardware fp16 accumulation between MMA steps: {mode}");
    // the products are exact and the fp32 path rounds once at the output: 1 + 0.75 ulp -> 1 + 1 ulp there
    assert!(up(f[0]));
}

fn small_cfg() -> TrainConfig {
    // Forge-1 structure at a small size (as tests/kopt.rs): relu2, lambdas, value embeddings, qk-norm; 2 x 512 tokens; qkv N = 320 (a half-empty last column tile)
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
fn f16acc_switch_unset_is_the_fp32_path_and_p8_changes_little() {
    if Device::count().unwrap_or(0) == 0 {
        eprintln!("skipped: no CUDA device");
        return;
    }
    if Device::open(0).unwrap().info.cc_major != 7 {
        eprintln!("skipped: FORGE_F16ACC applies to the Turing path only");
        return;
    }
    let cfg = small_cfg();
    let n = cfg.tokens();
    let mut r = Lcg(77);
    let tokens: Vec<u32> = (0..cfg.batch * (cfg.seq_len + 1))
        .map(|_| r.next() % cfg.vocab as u32)
        .collect();
    let scale = 1024.0f32;
    let mut runs = Vec::new();
    for setting in [None, Some("0"), Some("8"), None] {
        match setting {
            Some(v) => std::env::set_var("FORGE_F16ACC", v),
            None => std::env::remove_var("FORGE_F16ACC"),
        }
        let mut tr = Trainer::new(cfg.clone(), 0).unwrap();
        std::env::remove_var("FORGE_F16ACC");
        tr.init_random(5).unwrap();
        let l_eval = tr.eval_loss(&tokens).unwrap();
        tr.zero_grads().unwrap();
        let l_fb = tr.forward_backward(&tokens, scale, n).unwrap();
        runs.push((l_eval, l_fb, tr.grads_named(1.0 / scale).unwrap()));
    }
    let grad_diff = |a: &std::collections::HashMap<String, Vec<f32>>,
                     b: &std::collections::HashMap<String, Vec<f32>>| {
        let mut worst = (0f64, String::new());
        for (name, x) in a {
            let y = &b[name];
            let (mut dd, mut xx) = (0f64, 0f64);
            for (p, q) in x.iter().zip(y) {
                dd += ((p - q) as f64).powi(2);
                xx += (*p as f64).powi(2);
            }
            let rel = (dd / xx.max(1e-30)).sqrt();
            if rel > worst.0 {
                worst = (rel, name.clone());
            }
        }
        worst
    };
    let (base, zero, p8, again) = (&runs[0], &runs[1], &runs[2], &runs[3]);
    let noise = grad_diff(&base.2, &again.2);
    let d0 = grad_diff(&base.2, &zero.2);
    let d8 = grad_diff(&base.2, &p8.2);
    eprintln!("eval loss: unset {:.7}, FORGE_F16ACC=0 {:.7}, =8 {:.7} (rel diff {:.1e}); forward/backward loss {:.7} / {:.7} / {:.7}", base.0, zero.0, p8.0, ((p8.0 - base.0) / base.0).abs(), base.1, zero.1, p8.1);
    eprintln!("worst relative gradient difference: unset vs unset (run-to-run noise) {:.2e} ({}), vs FORGE_F16ACC=0 {:.2e} ({}), vs FORGE_F16ACC=8 {:.2e} ({})", noise.0, noise.1, d0.0, d0.1, d8.0, d8.1);
    // identity: 0 and unset run the same kernels (the forward is deterministic; the backward's atomics give the noise floor)
    assert_eq!(
        base.0.to_bits(),
        zero.0.to_bits(),
        "eval loss differs between unset and FORGE_F16ACC=0"
    );
    assert_eq!(base.0.to_bits(), again.0.to_bits());
    assert!(
        d0.0 <= noise.0.max(1e-6) * 2.0,
        "gradients of FORGE_F16ACC=0 differ beyond the noise: {:.2e}",
        d0.0
    );
    // the switch is active (gradients move far beyond the noise floor; the loss of a random-init model hardly moves: 7.6516438 in all three runs when measured), and its effect is of the order of the
    // fp16 rounding (measured: worst 3.0e-3, the q-norm gain gradient of the last layer, against 4.9e-8 noise)
    assert!(
        d8.0 > 100.0 * noise.0.max(1e-9),
        "FORGE_F16ACC=8 did not change the step: {:.2e}",
        d8.0
    );
    assert!(((p8.0 - base.0) / base.0).abs() < 1e-3);
    assert!(
        d8.0 < 1e-2,
        "gradient difference with FORGE_F16ACC=8: {:.2e}",
        d8.0
    );
}
