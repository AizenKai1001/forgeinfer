//! `forge kbench`: times the kernels of one Forge-1 training step (150M shapes: d 576, 9/3 heads of 64, ReLU^2 MLP of 2304, vocabulary 32768, sequences of 2048) one by one, with their variants side by side:
//! the GEMMs of every layer shape (with and without the idle-warp skip, the weight-gradient GEMMs at every split-K factor), the fused ReLU^2 epilogues against GEMM + separate kernel, the two cross-entropy
//! kernels and the attention kernels. For A/B work on a GPU that is not shared (the numbers of a busy GPU mean nothing).
//!
//!   forge kbench [--tokens 4096,8192] [--reps 10] [--only gemm|ce|attn|relu2|f16acc|f16power|e2] [--f16p 8] [--e2v e2,e2g3]
//!   (--only f16acc: the tensor-core rate with fp32 / fp16 accumulators and the forward GEMMs against their fp16-accumulate versions, flush every P k8 steps for each P of --f16p)
//!   (--only f16power [--secs 3] [--rounds 3]: sustained blocks of one variant at a time, MMA probes and the MLP up GEMM with fp32 / fp16 accumulators, with nvidia-smi power and clock: for an idle GPU)

use forge_cuda::{Args, Device, DeviceBuffer, Event, Function, Module, Stream};

fn flag<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str())
}

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
    /// a uniform value in [-s, s) as fp16 bits
    fn half(&mut self, s: f32) -> u16 {
        let u = (self.next() as f32 / (1u64 << 31) as f32) * 2.0 - 1.0;
        forge_core::half::f32_to_f16(u * s)
    }
}

fn rand_half_buf(n: usize, s: f32, seed: u64) -> DeviceBuffer {
    let mut r = Lcg(seed);
    let bytes: Vec<u8> = (0..n).flat_map(|_| r.half(s).to_le_bytes()).collect();
    DeviceBuffer::from_bytes(&bytes).expect("alloc")
}

struct Bench {
    stream: Stream,
    e0: Event,
    e1: Event,
    reps: usize,
}

impl Bench {
    /// average milliseconds of `a` and of `b`, interleaved launch by launch (a b a b ...): the clock of a power-capped GPU (T4, 70 W) drifts by tens of percent over seconds, so only interleaved timings compare
    fn time_pair(&self, mut a: impl FnMut(), mut b: impl FnMut()) -> (f64, f64) {
        let ev: Vec<Event> = (0..3).map(|_| Event::new().unwrap()).collect();
        a();
        b();
        self.stream.synchronize().unwrap();
        let (mut ta, mut tb) = (0f64, 0f64);
        for _ in 0..self.reps {
            ev[0].record(&self.stream).unwrap();
            a();
            ev[1].record(&self.stream).unwrap();
            b();
            ev[2].record(&self.stream).unwrap();
            ev[2].synchronize().unwrap();
            ta += ev[1].elapsed_ms(&ev[0]).unwrap() as f64;
            tb += ev[2].elapsed_ms(&ev[1]).unwrap() as f64;
        }
        (ta / self.reps as f64, tb / self.reps as f64)
    }

    /// average milliseconds of `reps` launches (after two warm-up launches)
    fn time(&self, mut launch: impl FnMut()) -> f64 {
        launch();
        launch();
        self.stream.synchronize().unwrap();
        self.e0.record(&self.stream).unwrap();
        for _ in 0..self.reps {
            launch();
        }
        self.e1.record(&self.stream).unwrap();
        self.e1.synchronize().unwrap();
        self.e1.elapsed_ms(&self.e0).unwrap() as f64 / self.reps as f64
    }
}

#[allow(clippy::too_many_arguments)]
fn gemm_args(
    a: u64,
    b: u64,
    c: u64,
    m: usize,
    n: usize,
    k: usize,
    lda: usize,
    ldb: usize,
    ldc: usize,
) -> Args {
    Args::new()
        .ptr(a)
        .ptr(b)
        .ptr(c)
        .i32(m as i32)
        .i32(n as i32)
        .i32(k as i32)
        .i32(lda as i32)
        .i32(ldb as i32)
        .i32(ldc as i32)
        .f32(1.0)
}

pub fn kbench(args: &[String]) -> i32 {
    let tokens: Vec<usize> = flag(args, "--tokens")
        .unwrap_or("4096,8192")
        .split(',')
        .filter_map(|v| v.parse().ok())
        .collect();
    let reps: usize = flag(args, "--reps")
        .and_then(|v| v.parse().ok())
        .unwrap_or(10);
    let only = flag(args, "--only").unwrap_or("all");
    let dev = Device::open(0).expect("no CUDA device");
    let (image, _) = forge_kernels::image_for(dev.info.cc_major, dev.info.cc_minor)
        .expect("no kernels for this device");
    let module = Module::load(image).expect("module");
    let f = |n: &str| -> Function { module.function(n).unwrap_or_else(|e| panic!("{n}: {e}")) };
    let sm = dev.info.sm_count as usize;
    println!("[kbench] {} ({} SMs), reps {reps}", dev.info.name, sm);
    let b = Bench {
        stream: Stream::new().unwrap(),
        e0: Event::new().unwrap(),
        e1: Event::new().unwrap(),
        reps,
    };
    let st = &b.stream;
    let (e, qkv, ff, v, ce) = (576usize, 960usize, 2304usize, 32768usize, 2048usize);
    let nmax = *tokens.iter().max().unwrap_or(&8192);
    // operands: A up to max(n x 2304, 2048 x 32768) halves, B up to 32768 x 576, C fp32 up to 2048 x 32768, R up to n x 2304 fp32
    // (--only f16acc: no head GEMMs, smaller buffers: it runs next to the production training on the 2060 SUPER)
    let head = only != "f16acc" && only != "f16power" && only != "e2";
    let a_big = rand_half_buf(
        if head {
            (nmax * ff).max(ce * v)
        } else {
            nmax * ff
        },
        1.0,
        1,
    );
    let b_big = rand_half_buf(if head { v * e } else { ff * ff }, 0.05, 2);
    let c_big = DeviceBuffer::alloc(if head {
        (ce * v * 4).max(nmax * ff * 4)
    } else {
        nmax * ff * 4
    })
    .unwrap();
    let r_big = DeviceBuffer::alloc(nmax * ff * 4).unwrap();
    let x_half = rand_half_buf(nmax * ff, 1.0, 3);
    r_big.zero(st).unwrap();
    c_big.zero(st).unwrap();
    let tf = |m: usize, n: usize, k: usize, ms: f64| {
        2.0 * m as f64 * n as f64 * k as f64 / (ms * 1e-3) / 1e12
    };

    if only == "all" || only == "gemm" {
        let (
            nt_f16,
            nt_f16_ns,
            nt_res,
            nt_res_ns,
            nt_f32,
            nn_f16,
            nn_f32,
            nn_f32_ns,
            tn_add,
            tn_add_ns,
        ) = (
            f("gemm_nt_f16"),
            f("gemm_nt_f16_ns"),
            f("gemm_nt_res"),
            f("gemm_nt_res_ns"),
            f("gemm_nt_f32"),
            f("gemm_nn_f16"),
            f("gemm_nn_f32"),
            f("gemm_nn_f32_ns"),
            f("gemm_tn_add"),
            f("gemm_tn_add_ns"),
        );
        let grid = |m: usize, n: usize, z: usize| {
            (((n + 127) / 128) as u32, ((m + 127) / 128) as u32, z as u32)
        };
        for &n in &tokens {
            println!("--- GEMMs, {n} tokens ---");
            // forward NT: A = activations [n][K], B = W [N][K]; idle-warp skip against no skip, interleaved
            for (name, nn, kk) in [("F1 qkv", qkv, e), ("F3 up", ff, e)] {
                let run = |func: Function| {
                    func.launch(
                        grid(n, nn, 1),
                        (256, 1, 1),
                        0,
                        st,
                        &gemm_args(
                            x_half.ptr(),
                            b_big.ptr(),
                            c_big.ptr(),
                            n,
                            nn,
                            kk,
                            kk,
                            kk,
                            nn,
                        ),
                    )
                    .unwrap()
                };
                let (ms_s, ms_n) = b.time_pair(|| run(nt_f16), || run(nt_f16_ns));
                println!("{name:10} NT f16   {n}x{nn}x{kk}  skip {ms_s:.3} ms ({:.1} TF)  noskip {ms_n:.3} ms ({:+.1} %)", tf(n, nn, kk, ms_s), 100.0 * (ms_s / ms_n - 1.0));
            }
            for (name, kk) in [("F2 wo", e), ("F4 down", ff)] {
                let run = |func: Function| {
                    func.launch(
                        grid(n, e, 1),
                        (256, 1, 1),
                        0,
                        st,
                        &gemm_args(x_half.ptr(), b_big.ptr(), c_big.ptr(), n, e, kk, kk, kk, e)
                            .ptr(r_big.ptr())
                            .i32(e as i32)
                            .f32(0.0),
                    )
                    .unwrap()
                };
                let (ms_s, ms_n) = b.time_pair(|| run(nt_res), || run(nt_res_ns));
                println!("{name:10} NT res   {n}x{e}x{kk}  skip {ms_s:.3} ms ({:.1} TF)  noskip {ms_n:.3} ms ({:+.1} %)", tf(n, e, kk, ms_s), 100.0 * (ms_s / ms_n - 1.0));
            }
            // data gradients NN: A = dY [n][K], B = W [K][N]
            for (name, nn, kk, func, tag) in [
                ("D1 down", ff, e, nn_f16, "f16"),
                ("D3 wo", e, e, nn_f16, "f16"),
            ] {
                let ms = b.time(|| {
                    func.launch(
                        grid(n, nn, 1),
                        (256, 1, 1),
                        0,
                        st,
                        &gemm_args(
                            x_half.ptr(),
                            b_big.ptr(),
                            c_big.ptr(),
                            n,
                            nn,
                            kk,
                            kk,
                            nn,
                            nn,
                        ),
                    )
                    .unwrap()
                });
                println!(
                    "{name:10} NN {tag:5} {n}x{nn}x{kk}  {ms:.3} ms ({:.1} TF)",
                    tf(n, nn, kk, ms)
                );
            }
            for (name, kk) in [("D2 up", ff), ("D4 qkv", qkv)] {
                let run = |func: Function| {
                    func.launch(
                        grid(n, e, 1),
                        (256, 1, 1),
                        0,
                        st,
                        &gemm_args(x_half.ptr(), b_big.ptr(), c_big.ptr(), n, e, kk, kk, e, e),
                    )
                    .unwrap()
                };
                let (ms_s, ms_n) = b.time_pair(|| run(nn_f32), || run(nn_f32_ns));
                println!("{name:10} NN f32   {n}x{e}x{kk}  skip {ms_s:.3} ms ({:.1} TF)  noskip {ms_n:.3} ms ({:+.1} %)", tf(n, e, kk, ms_s), 100.0 * (ms_s / ms_n - 1.0));
            }
            // weight gradients TN: A = dY stored [K = n][M], B = X stored [K = n][N]; C += ; every split-K factor interleaved with the unsplit kernel without the skip (the old kernel)
            for (name, mm, nn) in [
                ("W1 down", e, ff),
                ("W2 up", ff, e),
                ("W3 wo", e, e),
                ("W4 qkv", qkv, e),
            ] {
                let tiles = ((mm + 127) / 128) * ((nn + 127) / 128);
                let auto = crate::kbench_cmd::split_auto(tiles, n, sm);
                let mut line = format!("{name:10} TN add   {mm}x{nn}x{n} (auto split {auto}), time relative to the old kernel:");
                let old = || {
                    tn_add_ns
                        .launch(
                            grid(mm, nn, 1),
                            (256, 1, 1),
                            0,
                            st,
                            &gemm_args(
                                x_half.ptr(),
                                a_big.ptr(),
                                c_big.ptr(),
                                mm,
                                nn,
                                n,
                                mm,
                                nn,
                                nn,
                            ),
                        )
                        .unwrap()
                };
                for gz in [1usize, 2, 3, 4, 6, 8] {
                    if n % (16 * gz) != 0 {
                        continue;
                    }
                    let (ms_old, ms) = b.time_pair(old, || {
                        tn_add
                            .launch(
                                grid(mm, nn, gz),
                                (256, 1, 1),
                                0,
                                st,
                                &gemm_args(
                                    x_half.ptr(),
                                    a_big.ptr(),
                                    c_big.ptr(),
                                    mm,
                                    nn,
                                    n,
                                    mm,
                                    nn,
                                    nn,
                                ),
                            )
                            .unwrap()
                    });
                    line += &format!(
                        "  z{gz} {:.3}/{:.3} ms ({:+.0} %)",
                        ms,
                        ms_old,
                        100.0 * (ms / ms_old - 1.0)
                    );
                }
                println!("{line}");
            }
        }
        println!("--- head (chunk of {ce} rows) ---");
        let ms = b.time(|| {
            nt_f32
                .launch(
                    grid(ce, v, 1),
                    (256, 1, 1),
                    0,
                    st,
                    &gemm_args(x_half.ptr(), b_big.ptr(), c_big.ptr(), ce, v, e, e, e, v),
                )
                .unwrap()
        });
        println!(
            "F5 head    NT f32   {ce}x{v}x{e}         {ms:8.3} ms {:6.1} TFLOPs",
            tf(ce, v, e, ms)
        );
        let ms = b.time(|| {
            nn_f32
                .launch(
                    grid(ce, e, 1),
                    (256, 1, 1),
                    0,
                    st,
                    &gemm_args(a_big.ptr(), b_big.ptr(), c_big.ptr(), ce, e, v, v, e, e),
                )
                .unwrap()
        });
        println!(
            "D5 head    NN f32   {ce}x{e}x{v}         {ms:8.3} ms {:6.1} TFLOPs",
            tf(ce, e, v, ms)
        );
        let ms = b.time(|| {
            tn_add
                .launch(
                    grid(v, e, 1),
                    (256, 1, 1),
                    0,
                    st,
                    &gemm_args(a_big.ptr(), x_half.ptr(), c_big.ptr(), v, e, ce, v, e, e),
                )
                .unwrap()
        });
        println!(
            "W5 head    TN add   {v}x{e}x{ce}         {ms:8.3} ms {:6.1} TFLOPs",
            tf(v, e, ce, ms)
        );
    }

    if only == "e2" {
        // The trainer's GEMMs (forward NT, data-gradient NN, the head) against their variants with the staged epilogue (`_e2`, FORGE_GEMM_E2) and the column-group tile orders (`_e2gN`), interleaved
        // launch by launch; every variant's output is compared bit for bit with the old kernel's. --e2v lists the suffixes (default e2); a suffix that has no kernel for a shape is skipped.
        let vars: Vec<String> = flag(args, "--e2v")
            .unwrap_or("e2")
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();
        let n = tokens[0];
        let hc = ce / 2; // head chunk rows: 1024 (the trainer's 2048 would need 268 MB more for the logits; the tiles are the same)
        let wbuf = rand_half_buf(v * e, 0.05, 21); // W of every shape (the head's is the largest)
                                                   // kinds: 0 plain, 1 fp32 residual R [M][N], 2 ReLU^2 forward (R = relu(u)^2 out, fp16), 3 ReLU^2 backward (R = u in, fp16)
        let shapes: [(&str, &str, usize, usize, usize, usize, usize, usize, usize); 9] = [
            ("F1 qkv", "gemm_nt_f16", 0, n, qkv, e, e, e, qkv),
            ("F2 wo", "gemm_nt_res", 1, n, e, e, e, e, e),
            ("F3 up", "gemm_nt_relu2", 2, n, ff, e, e, e, ff),
            ("F4 down", "gemm_nt_res", 1, n, e, ff, ff, ff, e),
            ("D1 down", "gemm_nn_relu2_bwd", 3, n, ff, e, e, ff, ff),
            ("D2 up", "gemm_nn_f32", 0, n, e, ff, ff, e, e),
            ("D3 wo", "gemm_nn_f16", 0, n, e, e, e, e, e),
            ("D4 qkv", "gemm_nn_f32", 0, n, e, qkv, qkv, e, e),
            ("F5 head", "gemm_nt_f32", 0, hc, v, e, e, e, v),
        ];
        let out_bytes = |name: &str| {
            if name.ends_with("f32") || name.ends_with("res") {
                4usize
            } else {
                2
            }
        };
        let cbuf = DeviceBuffer::alloc((hc * v).max(n * ff) * 4).unwrap();
        let rbuf = DeviceBuffer::alloc(n * ff * 4).unwrap();
        let rv: Vec<f32> = (0..n * ff)
            .map(|i| ((i % 97) as f32 - 48.0) * 0.01)
            .collect();
        let bytes_of = |b: &DeviceBuffer, len: usize| -> Vec<u8> {
            let mut raw = vec![0u8; len];
            b.download(&mut raw).unwrap();
            raw
        };
        println!("--- staged epilogue / tile order vs the trainer's GEMMs, {n} tokens (head: chunk of {hc} rows), interleaved; 'same' = bit-identical outputs ---");
        for (label, old, kind, m, nn, kk, lda, ldb, ldc) in shapes {
            // operands: x_half [m][kk] or [m][lda]; b_big holds W in either layout
            // (the 128 x 144 kernels `_w144` take a grid of 144-wide column tiles)
            let launch_bn = |func: Function, bn: usize| {
                let a = gemm_args(
                    x_half.ptr(),
                    wbuf.ptr(),
                    cbuf.ptr(),
                    m,
                    nn,
                    kk,
                    lda,
                    ldb,
                    ldc,
                );
                let a = match kind {
                    0 => a,
                    1 => a.ptr(rbuf.ptr()).i32(nn as i32).f32(0.0),
                    _ => a.ptr(rbuf.ptr()).i32(nn as i32).f32(0.0),
                };
                func.launch(
                    (((nn + bn - 1) / bn) as u32, ((m + 127) / 128) as u32, 1),
                    (256, 1, 1),
                    0,
                    st,
                    &a,
                )
                .unwrap();
            };
            let launch = |func: Function| launch_bn(func, 128);
            let reset_r = || match kind {
                1 => rbuf.upload_f32(&rv[..m * nn]).unwrap(),
                3 => {
                    // u for the ReLU^2 backward: half of it negative
                    let u: Vec<u8> = (0..m * nn)
                        .flat_map(|i| {
                            forge_core::half::f32_to_f16(((i % 61) as f32 - 30.0) * 0.05)
                                .to_le_bytes()
                        })
                        .collect();
                    rbuf.upload(&u).unwrap();
                }
                _ => {}
            };
            let fo = f(old);
            let cb = m * nn * out_bytes(old);
            reset_r();
            launch(fo);
            st.synchronize().unwrap();
            let c_old = bytes_of(&cbuf, cb);
            let r_old = if kind == 2 {
                bytes_of(&rbuf, m * nn * 2)
            } else {
                Vec::new()
            };
            let mut line = format!("{label:8} {m}x{nn}x{kk}");
            for sfx in &vars {
                let Ok(fv) = module.function(&format!("{old}_{sfx}")) else {
                    continue;
                };
                let bn = if sfx.starts_with("w144") { 144 } else { 128 };
                if bn == 144 {
                    fv.set_smem_carveout(100).unwrap();
                }
                reset_r();
                launch_bn(fv, bn);
                st.synchronize().unwrap();
                let same = bytes_of(&cbuf, cb) == c_old
                    && (kind != 2 || bytes_of(&rbuf, m * nn * 2) == r_old);
                let (ms_o, ms_v) = b.time_pair(|| launch(fo), || launch_bn(fv, bn));
                line += &format!(
                    " | {sfx}: {ms_o:.3} -> {ms_v:.3} ms ({:.1} -> {:.1} TF, {:.2}x, {})",
                    tf(m, nn, kk, ms_o),
                    tf(m, nn, kk, ms_v),
                    ms_o / ms_v,
                    if same { "same" } else { "DIFFERENT" }
                );
            }
            println!("{line}");
        }
    }

    if only == "f16power" {
        // Sustained blocks instead of interleaved launches: on a power-capped GPU each variant settles at its own clock, so a lower energy per FLOP shows up as a shorter time per launch. Every block runs
        // --secs seconds of one variant (variants alternate, --rounds times); nvidia-smi samples power and SM clock in the middle of each block. Meant for an otherwise idle GPU.
        let secs: f64 = flag(args, "--secs")
            .and_then(|v| v.parse().ok())
            .unwrap_or(3.0);
        let rounds: usize = flag(args, "--rounds")
            .and_then(|v| v.parse().ok())
            .unwrap_or(3);
        let n = tokens[0];
        let smi = || -> String {
            std::process::Command::new("nvidia-smi")
                .args([
                    "--query-gpu=power.draw,clocks.sm,temperature.gpu",
                    "--format=csv,noheader",
                ])
                .output()
                .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
                .unwrap_or_default()
        };
        let grid =
            |m: usize, nn: usize| (((nn + 127) / 128) as u32, ((m + 127) / 128) as u32, 1u32);
        let (blocks, iters) = ((sm * 4) as u32, 256i32);
        let probe_flops = 2.0 * 1024.0 * 16.0 * iters as f64 * (blocks as f64 * 8.0);
        let (pk32, pk16) = (f("mma_peak_f32"), f("mma_peak_f16"));
        let (xh, bb, cb, rb) = (&x_half, &b_big, &c_big, &r_big);
        let probe = |func: Function| -> Box<dyn Fn() + '_> {
            Box::new(move || {
                func.launch(
                    (blocks, 1, 1),
                    (256, 1, 1),
                    0,
                    st,
                    &Args::new().ptr(cb.ptr()).i32(iters),
                )
                .unwrap()
            })
        };
        let up = |name: &str| -> Box<dyn Fn() + '_> {
            let func = f(name);
            Box::new(move || {
                func.launch(
                    grid(n, ff),
                    (256, 1, 1),
                    0,
                    st,
                    &gemm_args(xh.ptr(), bb.ptr(), cb.ptr(), n, ff, e, e, e, ff)
                        .ptr(rb.ptr())
                        .i32(ff as i32)
                        .f32(0.0),
                )
                .unwrap()
            })
        };
        let up_flops = 2.0 * n as f64 * ff as f64 * e as f64;
        let groups: Vec<Vec<(String, Box<dyn Fn() + '_>, f64)>> = vec![
            vec![
                (
                    "MMA probe, fp32 accumulate".to_string(),
                    probe(pk32),
                    probe_flops,
                ),
                (
                    "MMA probe, fp16 accumulate".to_string(),
                    probe(pk16),
                    probe_flops,
                ),
            ],
            vec![
                (
                    format!("MLP up {n}x{ff}x{e}, fp32 accumulate (gemm_nt_relu2)"),
                    up("gemm_nt_relu2"),
                    up_flops,
                ),
                (
                    format!("MLP up {n}x{ff}x{e}, fp16 accumulate P2 (gemm_nt_relu2_h2)"),
                    up("gemm_nt_relu2_h2"),
                    up_flops,
                ),
                (
                    format!("MLP up {n}x{ff}x{e}, fp16 accumulate P8 (gemm_nt_relu2_h8)"),
                    up("gemm_nt_relu2_h8"),
                    up_flops,
                ),
            ],
        ];
        let (e0, e1) = (Event::new().unwrap(), Event::new().unwrap());
        for group in &groups {
            println!("--- sustained blocks of {secs} s, {rounds} rounds ---");
            for round in 0..rounds {
                for (label, launch, flops) in group {
                    launch();
                    st.synchronize().unwrap();
                    e0.record(st).unwrap();
                    launch();
                    e1.record(st).unwrap();
                    e1.synchronize().unwrap();
                    let count = ((secs * 1000.0 / e1.elapsed_ms(&e0).unwrap().max(1e-3) as f64)
                        as usize)
                        .clamp(5, 200_000);
                    let sampler = std::thread::spawn(move || {
                        std::thread::sleep(std::time::Duration::from_secs_f64(secs * 0.6));
                        smi()
                    });
                    e0.record(st).unwrap();
                    for _ in 0..count {
                        launch();
                    }
                    e1.record(st).unwrap();
                    e1.synchronize().unwrap();
                    let ms = e1.elapsed_ms(&e0).unwrap() as f64 / count as f64;
                    let sample = sampler.join().unwrap_or_default();
                    println!("round {round} {label:62} {ms:8.4} ms/launch {:6.1} TFLOPs   [power W, SM MHz, deg C mid-block: {sample}]", flops / (ms * 1e9));
                }
            }
        }
    }

    if only == "all" || only == "relu2" {
        let (nt_f16, nt_relu2, nn_f16, nn_relu2_bwd, relu2, relu2_bwd) = (
            f("gemm_nt_f16"),
            f("gemm_nt_relu2"),
            f("gemm_nn_f16"),
            f("gemm_nn_relu2_bwd"),
            f("relu2_kernel"),
            f("relu2_bwd_kernel"),
        );
        let grid = |m: usize, n: usize| (((n + 127) / 128) as u32, ((m + 127) / 128) as u32, 1u32);
        for &n in &tokens {
            let cnt = n * ff;
            let ew = (((cnt + 255) / 256).min(65535) as u32, 1, 1);
            let (ms_2, ms_f) = b.time_pair(
                || {
                    nt_f16
                        .launch(
                            grid(n, ff),
                            (256, 1, 1),
                            0,
                            st,
                            &gemm_args(x_half.ptr(), b_big.ptr(), c_big.ptr(), n, ff, e, e, e, ff),
                        )
                        .unwrap();
                    relu2
                        .launch(
                            ew,
                            (256, 1, 1),
                            0,
                            st,
                            &Args::new()
                                .ptr(c_big.ptr())
                                .ptr(r_big.ptr())
                                .u64(cnt as u64),
                        )
                        .unwrap();
                },
                || {
                    nt_relu2
                        .launch(
                            grid(n, ff),
                            (256, 1, 1),
                            0,
                            st,
                            &gemm_args(x_half.ptr(), b_big.ptr(), c_big.ptr(), n, ff, e, e, e, ff)
                                .ptr(r_big.ptr())
                                .i32(ff as i32)
                                .f32(0.0),
                        )
                        .unwrap()
                },
            );
            println!("relu2 fwd {n} tokens (interleaved): GEMM + relu2 {ms_2:.3} ms   fused {ms_f:.3} ms ({:+.1} %)", 100.0 * (ms_f / ms_2 - 1.0));
            let (ms_2, ms_f) = b.time_pair(
                || {
                    nn_f16
                        .launch(
                            grid(n, ff),
                            (256, 1, 1),
                            0,
                            st,
                            &gemm_args(x_half.ptr(), b_big.ptr(), c_big.ptr(), n, ff, e, e, ff, ff),
                        )
                        .unwrap();
                    relu2_bwd
                        .launch(
                            ew,
                            (256, 1, 1),
                            0,
                            st,
                            &Args::new()
                                .ptr(c_big.ptr())
                                .ptr(x_half.ptr())
                                .ptr(r_big.ptr())
                                .u64(cnt as u64),
                        )
                        .unwrap();
                },
                || {
                    nn_relu2_bwd
                        .launch(
                            grid(n, ff),
                            (256, 1, 1),
                            0,
                            st,
                            &gemm_args(x_half.ptr(), b_big.ptr(), c_big.ptr(), n, ff, e, e, ff, ff)
                                .ptr(x_half.ptr())
                                .i32(ff as i32)
                                .f32(0.0),
                        )
                        .unwrap()
                },
            );
            println!("relu2 bwd {n} tokens (interleaved): GEMM + relu2_bwd {ms_2:.3} ms   fused {ms_f:.3} ms ({:+.1} %)", 100.0 * (ms_f / ms_2 - 1.0));
        }
    }

    if only == "all" || only == "f16acc" {
        // fp16-accumulate forward GEMMs (FORGE_F16ACC=P in the trainer). 1) the tensor-core rate with fp32 and fp16 accumulators on register operands; 2) every forward NT GEMM against its fp16-accumulate
        // version for each flush interval P (--f16p, k8 steps), interleaved, with the relative RMS difference of the outputs (fp16 outputs: u and relu(u)^2; fp32 residual outputs: C - R)
        // the fp16 kernel runs twice the iterations: on a GPU shared with another process (time slices of milliseconds) only kernels of about equal duration compare
        let (pk32, pk16) = (f("mma_peak_f32"), f("mma_peak_f16"));
        let (blocks, iters) = ((sm * 4) as u32, 256i32);
        let (t32, t16) = b.time_pair(
            || {
                pk32.launch(
                    (blocks, 1, 1),
                    (256, 1, 1),
                    0,
                    st,
                    &Args::new().ptr(c_big.ptr()).i32(iters),
                )
                .unwrap()
            },
            || {
                pk16.launch(
                    (blocks, 1, 1),
                    (256, 1, 1),
                    0,
                    st,
                    &Args::new().ptr(c_big.ptr()).i32(2 * iters),
                )
                .unwrap()
            },
        );
        let flops = 2.0 * 1024.0 * 16.0 * iters as f64 * (blocks as f64 * 8.0);
        println!(
            "tensor-core rate (register operands, interleaved, {t32:.3} / {t16:.3} ms): fp32 accumulate {:.1} TFLOPs, fp16 accumulate {:.1} TFLOPs, fp16/fp32 speed {:.2}x",
            flops / (t32 * 1e9),
            2.0 * flops / (t16 * 1e9),
            2.0 * t32 / t16
        );
        // kernel suffixes: P (flush interval) -> `_hP`; experiments `8l`, `8b1`; `0` (never flushed) exists for the fp16-output kernel only
        let ps: Vec<String> = flag(args, "--f16p")
            .unwrap_or("8")
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();
        let grid = |m: usize, n: usize| (((n + 127) / 128) as u32, ((m + 127) / 128) as u32, 1u32);
        let out2 = DeviceBuffer::alloc(nmax * ff * 4).unwrap();
        let r2 = DeviceBuffer::alloc(nmax * ff * 2).unwrap();
        let rel = |a: &[f32], b: &[f32]| {
            let (mut dd, mut bb) = (0f64, 0f64);
            for (x, y) in a.iter().zip(b) {
                dd += ((x - y) as f64).powi(2);
                bb += (*y as f64).powi(2);
            }
            (dd / bb.max(1e-30)).sqrt()
        };
        let h2f = |b: &DeviceBuffer, n: usize| -> Vec<f32> {
            let mut raw = vec![0u8; n * 2];
            b.download(&mut raw).unwrap();
            raw.chunks_exact(2)
                .map(|c| forge_core::half::f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect()
        };
        for &n in &tokens {
            println!("--- forward GEMMs, {n} tokens: fp32 accumulate vs fp16 accumulate (flush every P k8 steps), interleaved ---");
            // kind 0: fp16 output (qkv), 1: fp32 output + fp32 residual (wo, down), 2: ReLU^2 epilogue (up)
            for (name, nn, kk, kind) in [
                ("F1 qkv", qkv, e, 0usize),
                ("F2 wo", e, e, 1),
                ("F3 up", ff, e, 2),
                ("F4 down", e, ff, 1),
            ] {
                let old = ["gemm_nt_f16", "gemm_nt_res", "gemm_nt_relu2"][kind];
                let launch = |func: Function, c: u64, r: u64| {
                    let a = gemm_args(x_half.ptr(), b_big.ptr(), c, n, nn, kk, kk, kk, nn);
                    let a = match kind {
                        0 => a,
                        1 => a.ptr(r).i32(nn as i32).f32(0.0),
                        _ => a.ptr(r).i32(nn as i32).f32(0.0),
                    };
                    func.launch(grid(n, nn), (256, 1, 1), 0, st, &a).unwrap();
                };
                // the residual: small non-zero values (C - R is compared)
                if kind == 1 {
                    let rv: Vec<f32> = (0..n * nn)
                        .map(|i| ((i % 97) as f32 - 48.0) * 0.01)
                        .collect();
                    r_big.upload_f32(&rv).unwrap();
                }
                let fo = f(old);
                launch(fo, c_big.ptr(), r_big.ptr());
                st.synchronize().unwrap();
                let base: Vec<f32> = match kind {
                    0 => h2f(&c_big, n * nn),
                    1 => {
                        let c = c_big.download_f32(n * nn).unwrap();
                        let r = r_big.download_f32(n * nn).unwrap();
                        c.iter().zip(&r).map(|(x, y)| x - y).collect()
                    }
                    _ => h2f(&c_big, n * nn),
                };
                let mut line = format!("{name:8} {n}x{nn}x{kk}");
                for p in &ps {
                    if p == "0" && kind != 0 {
                        continue;
                    }
                    let Ok(fh) = module.function(&format!("{old}_h{p}")) else {
                        continue;
                    };
                    let (rh, ch) = if kind == 2 {
                        (r2.ptr(), out2.ptr())
                    } else {
                        (r_big.ptr(), out2.ptr())
                    };
                    launch(fh, ch, rh);
                    st.synchronize().unwrap();
                    let got: Vec<f32> = match kind {
                        0 | 2 => h2f(&out2, n * nn),
                        _ => {
                            let c = out2.download_f32(n * nn).unwrap();
                            let r = r_big.download_f32(n * nn).unwrap();
                            c.iter().zip(&r).map(|(x, y)| x - y).collect()
                        }
                    };
                    let d = rel(&got, &base);
                    let (ms_o, ms_h) = b.time_pair(
                        || launch(fo, c_big.ptr(), r_big.ptr()),
                        || launch(fh, ch, rh),
                    );
                    line += &format!(" | P{p}: {ms_o:.3} -> {ms_h:.3} ms ({:.1} -> {:.1} TF, {:.2}x, rel diff {d:.1e})", tf(n, nn, kk, ms_o), tf(n, nn, kk, ms_h), ms_o / ms_h);
                }
                println!("{line}");
            }
        }
    }

    if only == "all" || only == "ce" {
        // logits of one chunk: realistic magnitudes (|z| < 8); targets spread over the vocabulary
        let z: Vec<f32> = {
            let mut r = Lcg(9);
            (0..ce * v)
                .map(|_| (r.next() as f32 / (1u64 << 31) as f32) * 16.0 - 8.0)
                .collect()
        };
        let zb = DeviceBuffer::alloc(ce * v * 4).unwrap();
        zb.upload_f32(&z).unwrap();
        let tg: Vec<u8> = (0..ce)
            .flat_map(|i| (((i * 7919) % v) as u32).to_le_bytes())
            .collect();
        let tb = DeviceBuffer::from_bytes(&tg).unwrap();
        let lr = DeviceBuffer::alloc(ce * 4).unwrap();
        let dz = DeviceBuffer::alloc(ce * v * 2).unwrap();
        let cargs = || {
            Args::new()
                .buf(&zb)
                .buf(&tb)
                .buf(&lr)
                .buf(&dz)
                .i32(v as i32)
                .f32(0.5 / 65536.0)
        };
        let (k_old, k_reg) = (f("ce_kernel"), f("ce_reg_kernel"));
        k_old
            .launch((ce as u32, 1, 1), (256, 1, 1), 0, st, &cargs())
            .unwrap();
        st.synchronize().unwrap();
        let l_old = lr.download_f32(ce).unwrap();
        k_reg
            .launch((ce as u32, 1, 1), (1024, 1, 1), 0, st, &cargs())
            .unwrap();
        st.synchronize().unwrap();
        let l_reg = lr.download_f32(ce).unwrap();
        let (ms_old, ms_reg) = b.time_pair(
            || {
                k_old
                    .launch((ce as u32, 1, 1), (256, 1, 1), 0, st, &cargs())
                    .unwrap()
            },
            || {
                k_reg
                    .launch((ce as u32, 1, 1), (1024, 1, 1), 0, st, &cargs())
                    .unwrap()
            },
        );
        let worst = l_old
            .iter()
            .zip(&l_reg)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        let gb = |ms: f64, bytes: f64| bytes / (ms * 1e-3) / 1e9;
        println!("cross entropy {ce}x{v}: three-pass {ms_old:.3} ms   register {ms_reg:.3} ms ({:.0} GB/s of logits + gradient)   max |loss diff| {worst:.2e}", gb(ms_reg, (ce * v * 6) as f64));
    }

    if only == "all" || only == "attnds" {
        // dense attention backward: dK dV + dQ (three matmuls) against the dS spill (dK dV stores the dS tiles, dQ is one matmul per tile), interleaved
        let (t, nh, nkv, hd) = (2048usize, 9usize, 3usize, 64usize);
        let (dkdv, dq2, dkdv_ds, dq_ds) = (
            f("flash_bwd_dkdv_64"),
            f("flash_bwd_dq_t64"),
            f("flash_bwd_dkdv_ds_64"),
            f("flash_bwd_dq_ds_64"),
        );
        for &n in &tokens {
            let bsz = n / t;
            let q = rand_half_buf(n * nh * hd, 1.0, 11);
            let kk = rand_half_buf(n * nkv * hd, 1.0, 12);
            let vv = rand_half_buf(n * nkv * hd, 1.0, 13);
            let dout = rand_half_buf(n * nh * hd, 1.0, 14);
            let lse = DeviceBuffer::alloc(n * nh * 4).unwrap();
            lse.zero(st).unwrap();
            let dd = DeviceBuffer::alloc(n * nh * 4).unwrap();
            dd.zero(st).unwrap();
            let dq32 = DeviceBuffer::alloc(n * nh * hd * 4).unwrap();
            let dk32 = DeviceBuffer::alloc(n * nkv * hd * 4).unwrap();
            let dv32 = DeviceBuffer::alloc(n * nkv * hd * 4).unwrap();
            let nt = t / 64;
            let ds = DeviceBuffer::alloc(bsz * nh * (nt * (nt + 1) / 2) * 4096 * 2).unwrap();
            ds.zero(st).unwrap();
            let kv_grid = ((t / 64) as u32, nkv as u32, bsz as u32);
            let q_grid = ((t / 64) as u32, nh as u32, bsz as u32);
            let old_args = |dst: Vec<&DeviceBuffer>| {
                let mut a = Args::new().buf(&q).buf(&kk).buf(&vv).buf(&dout).buf(&lse).buf(&dd);
                for x in dst {
                    a = a.buf(x);
                }
                a.i32(t as i32).i32(nh as i32).i32(nkv as i32).i32(0)
            };
            let old = || {
                dkdv.launch(kv_grid, (128, 1, 1), 0, st, &old_args(vec![&dk32, &dv32])).unwrap();
                dq2.launch(q_grid, (128, 1, 1), 0, st, &old_args(vec![&dq32])).unwrap();
            };
            let new = || {
                dkdv_ds
                    .launch(
                        kv_grid,
                        (128, 1, 1),
                        0,
                        st,
                        &Args::new()
                            .buf(&q)
                            .buf(&kk)
                            .buf(&vv)
                            .buf(&dout)
                            .buf(&lse)
                            .buf(&dd)
                            .buf(&dk32)
                            .buf(&dv32)
                            .buf(&ds)
                            .i32(t as i32)
                            .i32(nh as i32)
                            .i32(nkv as i32),
                    )
                    .unwrap();
                dq_ds
                    .launch(
                        q_grid,
                        (128, 1, 1),
                        0,
                        st,
                        &Args::new().buf(&kk).buf(&ds).buf(&dq32).i32(t as i32).i32(nh as i32).i32(nkv as i32),
                    )
                    .unwrap();
            };
            let (a, bb) = b.time_pair(old, new);
            let (kv_old, kv_new) = b.time_pair(
                || dkdv.launch(kv_grid, (128, 1, 1), 0, st, &old_args(vec![&dk32, &dv32])).unwrap(),
                || {
                    dkdv_ds
                        .launch(
                            kv_grid,
                            (128, 1, 1),
                            0,
                            st,
                            &Args::new()
                                .buf(&q)
                                .buf(&kk)
                                .buf(&vv)
                                .buf(&dout)
                                .buf(&lse)
                                .buf(&dd)
                                .buf(&dk32)
                                .buf(&dv32)
                                .buf(&ds)
                                .i32(t as i32)
                                .i32(nh as i32)
                                .i32(nkv as i32),
                        )
                        .unwrap()
                },
            );
            let (q_old, q_new) = b.time_pair(
                || dq2.launch(q_grid, (128, 1, 1), 0, st, &old_args(vec![&dq32])).unwrap(),
                || {
                    dq_ds
                        .launch(
                            q_grid,
                            (128, 1, 1),
                            0,
                            st,
                            &Args::new().buf(&kk).buf(&ds).buf(&dq32).i32(t as i32).i32(nh as i32).i32(nkv as i32),
                        )
                        .unwrap()
                },
            );
            println!("attention backward {n} tokens, interleaved A/B: dkdv {kv_old:.3} -> {kv_new:.3} ms ({:+.1} %)  dq {q_old:.3} -> {q_new:.3} ms ({:+.1} %)  both {a:.3} -> {bb:.3} ms ({:+.1} %)",
                100.0 * (kv_new / kv_old - 1.0), 100.0 * (q_new / q_old - 1.0), 100.0 * (bb / a - 1.0));
        }
    }
    if only == "all" || only == "attn" {
        let (t, nh, nkv, hd) = (2048usize, 9usize, 3usize, 64usize);
        let (fwd, dkdv, dq) = (
            f("flash_fwd_train_64"),
            f("flash_bwd_dkdv_64"),
            f("flash_bwd_dq_64"),
        );
        for &n in &tokens {
            let bsz = n / t;
            let q = rand_half_buf(n * nh * hd, 1.0, 11);
            let kk = rand_half_buf(n * nkv * hd, 1.0, 12);
            let vv = rand_half_buf(n * nkv * hd, 1.0, 13);
            let o = DeviceBuffer::alloc(n * nh * hd * 2).unwrap();
            let lse = DeviceBuffer::alloc(n * nh * 4).unwrap();
            let dd = DeviceBuffer::alloc(n * nh * 4).unwrap();
            dd.zero(st).unwrap();
            let dq32 = DeviceBuffer::alloc(n * nh * hd * 4).unwrap();
            let dk32 = DeviceBuffer::alloc(n * nkv * hd * 4).unwrap();
            let dv32 = DeviceBuffer::alloc(n * nkv * hd * 4).unwrap();
            // causal attention FLOPs of the forward (QK^T and PV over the lower triangle)
            let fl = 2.0 * 2.0 * (bsz * nh) as f64 * (t * t / 2) as f64 * hd as f64;
            let ms_f = b.time(|| {
                fwd.launch(
                    (((t + 63) / 64) as u32, nh as u32, bsz as u32),
                    (128, 1, 1),
                    0,
                    st,
                    &Args::new()
                        .buf(&q)
                        .buf(&kk)
                        .buf(&vv)
                        .buf(&o)
                        .buf(&lse)
                        .i32(t as i32)
                        .i32(nh as i32)
                        .i32(nkv as i32),
                )
                .unwrap()
            });
            let bargs = |dst: &DeviceBuffer, dst2: Option<&DeviceBuffer>| {
                let mut a = Args::new()
                    .buf(&q)
                    .buf(&kk)
                    .buf(&vv)
                    .buf(&o)
                    .buf(&lse)
                    .buf(&dd)
                    .buf(dst);
                if let Some(x) = dst2 {
                    a = a.buf(x);
                }
                a.i32(t as i32).i32(nh as i32).i32(nkv as i32)
            };
            let ms_kv = b.time(|| {
                dkdv.launch(
                    ((t / 64) as u32, nkv as u32, bsz as u32),
                    (128, 1, 1),
                    0,
                    st,
                    &bargs(&dk32, Some(&dv32)),
                )
                .unwrap()
            });
            let ms_q = b.time(|| {
                dq.launch(
                    ((t / 64) as u32, nh as u32, bsz as u32),
                    (128, 1, 1),
                    0,
                    st,
                    &bargs(&dq32, None),
                )
                .unwrap()
            });
            println!(
                "attention {n} tokens: fwd {ms_f:.3} ms ({:.1} TF)  dkdv {ms_kv:.3} ms ({:.1} TF)  dq {ms_q:.3} ms ({:.1} TF)",
                fl / (ms_f * 1e-3) / 1e12,
                2.0 * fl / (ms_kv * 1e-3) / 1e12,
                1.5 * fl / (ms_q * 1e-3) / 1e12
            );
            // second generation (flash_train.cu) and the sliding window; window = 0 / T: the dense kernels
            let (fwd2, dq2, fwdw, dkdvw, dqw) = (
                f("flash_fwd_t64"),
                f("flash_bwd_dq_t64"),
                f("flash_fwd_t64_win"),
                f("flash_bwd_dkdv_64_win"),
                f("flash_bwd_dq_64_win"),
            );
            let wargs = |dst: Vec<&DeviceBuffer>, w: usize| {
                let mut a = Args::new().buf(&q).buf(&kk).buf(&vv);
                for x in dst {
                    a = a.buf(x);
                }
                a.i32(t as i32).i32(nh as i32).i32(nkv as i32).i32(w as i32)
            };
            let fwd_grid = ((t / 64) as u32, nh as u32, bsz as u32);
            let kv_grid = ((t / 64) as u32, nkv as u32, bsz as u32);
            // interleaved A/B: first against second generation
            let (f_old, f_new) = b.time_pair(
                || {
                    fwd.launch(
                        fwd_grid,
                        (128, 1, 1),
                        0,
                        st,
                        &Args::new()
                            .buf(&q)
                            .buf(&kk)
                            .buf(&vv)
                            .buf(&o)
                            .buf(&lse)
                            .i32(t as i32)
                            .i32(nh as i32)
                            .i32(nkv as i32),
                    )
                    .unwrap()
                },
                || {
                    fwd2.launch(fwd_grid, (128, 1, 1), 0, st, &wargs(vec![&o, &lse], 0))
                        .unwrap()
                },
            );
            let (q_old, q_new) = b.time_pair(
                || {
                    dq.launch(fwd_grid, (128, 1, 1), 0, st, &bargs(&dq32, None))
                        .unwrap()
                },
                || {
                    dq2.launch(
                        fwd_grid,
                        (128, 1, 1),
                        0,
                        st,
                        &wargs(vec![&o, &lse, &dd, &dq32], 0),
                    )
                    .unwrap()
                },
            );
            println!("attention {n} tokens, interleaved A/B: fwd {f_old:.3} -> {f_new:.3} ms ({:+.1} %)  dq {q_old:.3} -> {q_new:.3} ms ({:+.1} %)", 100.0 * (f_new / f_old - 1.0), 100.0 * (q_new / q_old - 1.0));
            for w in [256usize, 512, 1024] {
                let dense = || {
                    fwd2.launch(fwd_grid, (128, 1, 1), 0, st, &wargs(vec![&o, &lse], 0))
                        .unwrap();
                    dkdv.launch(kv_grid, (128, 1, 1), 0, st, &bargs(&dk32, Some(&dv32)))
                        .unwrap();
                    dq2.launch(
                        fwd_grid,
                        (128, 1, 1),
                        0,
                        st,
                        &wargs(vec![&o, &lse, &dd, &dq32], 0),
                    )
                    .unwrap();
                };
                let windowed = || {
                    fwdw.launch(fwd_grid, (128, 1, 1), 0, st, &wargs(vec![&o, &lse], w))
                        .unwrap();
                    dkdvw
                        .launch(
                            kv_grid,
                            (128, 1, 1),
                            0,
                            st,
                            &wargs(vec![&o, &lse, &dd, &dk32, &dv32], w),
                        )
                        .unwrap();
                    dqw.launch(
                        fwd_grid,
                        (128, 1, 1),
                        0,
                        st,
                        &wargs(vec![&o, &lse, &dd, &dq32], w),
                    )
                    .unwrap();
                };
                let (md, mw) = b.time_pair(dense, windowed);
                println!("attention {n} tokens, window {w} (fwd + dkdv + dq, interleaved with dense): {mw:.3} ms vs dense {md:.3} ms = {:.0} %", 100.0 * mw / md);
            }
        }
    }
    0
}

/// the trainer's choice of the split-K factor (kept in step with forge-train: printed next to the measured factors)
pub fn split_auto(tiles: usize, k: usize, sm_count: usize) -> usize {
    forge_train::split_k_turing(tiles, k, sm_count)
}
