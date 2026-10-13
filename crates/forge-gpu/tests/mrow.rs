//! The multi-row prompt path (kernels/decode_m.cu, src/mrow.rs).
//!
//! 1. The matvec kernels against a double-precision reference (the dequantized weights times the dequantized int8 activation), for every weight format, the four row counts (4, 8, 12, 16), several
//!    launch shapes and K slicings: the result of a row must not depend on the number of rows, the grid or the block size (bit for bit), and the residual variant must add exactly.
//! 2. Whole prompts through the model (Qwen3 dense, Qwen3.5 hybrid): `prefill_batched` with the multi-row path against the GEMM path (`set_mrow_max(0)`): the logits, the greedy continuation, chunks of
//!    every length up to 40 tokens, a prompt in two pieces, every group size, and the evaluation logits of `prefill_logits`.
//!    `verify_rows` / `accept_rows` against single-token steps.
//! 3. The attention prep of the dense path (QK norm, RoPE, cache append) against a host reference, with and without norm weights (the Llama-like models).
//! Skipped when a CUDA device (or a model file) is missing.

use forge_core::half::f32_to_f16;
use forge_core::Json;
use forge_cuda::{Args, Device, DeviceBuffer, Function, Module, Stream};
use forge_gguf::dequantize;
use forge_gpu::{repack, Fmt, GpuModel, GpuSession, QFmt};
use forge_model::{argmax, Model};
use std::path::PathBuf;

mod common;

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
    fn unit(&mut self) -> f32 {
        self.next() as f32 / (1u64 << 30) as f32 - 1.0
    }
}

/// Random native blocks of `rows` x `k` weights with finite scales.
fn rand_blocks(qf: QFmt, rows: usize, k: usize, seed: u64) -> Vec<u8> {
    let (bv, bb) = qf.block();
    let nblocks = rows * k / bv;
    let mut r = Lcg(seed);
    let mut v = vec![0u8; nblocks * bb];
    for b in 0..nblocks {
        let blk = &mut v[b * bb..(b + 1) * bb];
        for x in blk.iter_mut() {
            *x = (r.next() >> 4) as u8;
        }
        let half = |r: &mut Lcg| f32_to_f16(0.002 + 0.001 * (r.next() % 8) as f32).to_le_bytes();
        match qf {
            QFmt::Q8_0 | QFmt::Q4_0 | QFmt::Q5_0 => blk[0..2].copy_from_slice(&half(&mut r)),
            QFmt::Q4_K | QFmt::Q5_K => {
                blk[0..2].copy_from_slice(&half(&mut r));
                blk[2..4].copy_from_slice(&half(&mut r));
            }
            QFmt::Q6_K => blk[208..210].copy_from_slice(&half(&mut r)),
            QFmt::Q3_K | QFmt::Q2_K => unreachable!("host-only formats: the multi-row GPU kernels do not read them"),
        }
    }
    v
}

struct Ctx {
    _dev: Device,
    module: Module,
    stream: Stream,
}

fn setup() -> Option<Ctx> {
    if Device::count().unwrap_or(0) == 0 {
        eprintln!("skipped: no CUDA device");
        return None;
    }
    let dev = Device::open(0).unwrap();
    let (image, _) = forge_kernels::image_for(dev.info.cc_major, dev.info.cc_minor).unwrap();
    let module = Module::load(image).unwrap();
    Some(Ctx {
        _dev: dev,
        module,
        stream: Stream::new().unwrap(),
    })
}

fn func(c: &Ctx, name: &str) -> Function {
    c.module
        .function(name)
        .unwrap_or_else(|e| panic!("{name}: {e}"))
}

fn planes_args(a: Args, planes: &[DeviceBuffer]) -> Args {
    let mut a = a;
    for i in 0..4 {
        a = a.ptr(planes.get(i).map(|b| b.ptr()).unwrap_or(0));
    }
    a
}

/// The fewest K slices that bring `t` rows of a slice under the 48 KB of default shared memory.
fn min_nj(qf: QFmt, t: usize, k: usize) -> usize {
    let scw = if matches!(qf, QFmt::Q8_0 | QFmt::Q6_K) { 1 } else { 2 };
    let nsb = k / 256;
    (1..=nsb)
        .find(|&nj| {
            let ks = ((nsb + nj - 1) / nj) * 256;
            t * ks + t * (ks / 32) * 4 * scw <= 46000
        })
        .unwrap()
}

const SUF: [&str; 6] = ["q8", "q40", "q50", "q4k", "q5k", "q6k"];

/// The weights of `rows` x `k` random blocks of `qf` on the device (planes), with the dequantized values.
fn weights(qf: QFmt, rows: usize, k: usize, seed: u64) -> (Vec<DeviceBuffer>, Vec<f32>) {
    let native = rand_blocks(qf, rows, k, seed);
    let want = dequantize(qf.dtype(), &native, rows * k).unwrap();
    let planes = repack(qf, &native)
        .unwrap()
        .iter()
        .map(|p| DeviceBuffer::from_bytes(p).unwrap())
        .collect();
    (planes, want)
}

/// `t` activation rows of `k` values (outliers and different scales per row), quantized by the production quantizer: (xq, xd, xs) on the device and downloaded.
struct Acts {
    xq: DeviceBuffer,
    xd: DeviceBuffer,
    xs: DeviceBuffer,
    hq: Vec<i8>,
    hd: Vec<f32>,
}

fn acts(c: &Ctx, t: usize, k: usize, seed: u64) -> Acts {
    let mut r = Lcg(seed);
    let x: Vec<f32> = (0..t * k)
        .map(|i| {
            let row = i / k;
            let mut v = r.unit() * (0.3 + 0.2 * row as f32);
            if r.next() % 23 == 0 {
                v *= 6.0;
            }
            v
        })
        .collect();
    let xin = DeviceBuffer::from_bytes(&x.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>())
        .unwrap();
    let (xq, xd, xs) = (
        DeviceBuffer::alloc(t * k).unwrap(),
        DeviceBuffer::alloc(t * k / 32 * 4).unwrap(),
        DeviceBuffer::alloc(t * k / 32 * 4).unwrap(),
    );
    let nm = [1usize, 2, 4, 8, 16, 32]
        .into_iter()
        .find(|&n| n * 512 >= k)
        .unwrap();
    for row in 0..t {
        func(c, &format!("quant_kernel_{nm}"))
            .launch(
                (1, 1, 1),
                (512, 1, 1),
                0,
                &c.stream,
                &Args::new()
                    .ptr(xin.ptr() + (row * k * 4) as u64)
                    .ptr(xq.ptr() + (row * k) as u64)
                    .ptr(xd.ptr() + (row * k / 32 * 4) as u64)
                    .ptr(xs.ptr() + (row * k / 32 * 4) as u64)
                    .i32(k as i32),
            )
            .unwrap();
    }
    c.stream.synchronize().unwrap();
    let mut bq = vec![0u8; t * k];
    xq.download(&mut bq).unwrap();
    Acts {
        hq: bq.iter().map(|&b| b as i8).collect(),
        hd: xd.download_f32(t * k / 32).unwrap(),
        xq,
        xd,
        xs,
    }
}

/// y[t][r] = sum_k w[r][k] * (xq[t][k] * xd[t][k / 32]) in double precision
fn reference(w: &[f32], a: &Acts, t: usize, rows: usize, k: usize) -> Vec<f64> {
    let mut y = vec![0f64; t * rows];
    for tok in 0..t {
        for r in 0..rows {
            let mut s = 0f64;
            for b in 0..k / 32 {
                let mut p = 0f64;
                for i in 0..32 {
                    p += w[r * k + b * 32 + i] as f64 * a.hq[tok * k + b * 32 + i] as f64;
                }
                s += p * a.hd[tok * k / 32 + b] as f64;
            }
            y[tok * rows + r] = s;
        }
    }
    y
}

fn rel_rms(a: &[f64], b: &[f64]) -> f64 {
    let num: f64 = a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum();
    let den: f64 = b.iter().map(|y| y * y).sum();
    (num / den.max(1e-30)).sqrt()
}

/// One matrix of a launch of the multi-row kernel.
struct Mat<'a> {
    planes: &'a [DeviceBuffer],
    rows: usize,
}

/// The multi-row kernel `mm_{out,res}_<fmt>_t<T>` on up to three matrices of K = `k` (one launch; `nj` K slices, then a partial-sum buffer and the reduce kernel), `gx` blocks of `nw` warps; `bases` are
/// the initial contents of the outputs for the residual variant. Returns the outputs of the matrices ([t][rows] each).
#[allow(clippy::too_many_arguments)]
fn run_mm(
    c: &Ctx,
    qf: QFmt,
    mats: &[Mat],
    a: &Acts,
    t_var: usize,
    k: usize,
    nj: usize,
    gx: u32,
    nw: u32,
    bases: Option<&[Vec<f32>]>,
) -> Vec<Vec<f32>> {
    let scw = if matches!(qf, QFmt::Q8_0 | QFmt::Q6_K) { 1 } else { 2 };
    let nsb = k / 256;
    let ks = ((nsb + nj - 1) / nj) * 256;
    let smem = t_var * ks + t_var * (ks / 32) * 4 * scw;
    assert!(smem <= 48 * 1024, "{smem} bytes of shared memory");
    let res = bases.is_some();
    let outs: Vec<DeviceBuffer> = mats
        .iter()
        .enumerate()
        .map(|(i, m)| {
            let o = DeviceBuffer::alloc(t_var * m.rows * 4).unwrap();
            match bases {
                Some(b) => o.upload_f32(&b[i]).unwrap(),
                None => o.upload_f32(&vec![-777.0f32; t_var * m.rows]).unwrap(),
            }
            o
        })
        .collect();
    let parts: Vec<DeviceBuffer> = mats
        .iter()
        .map(|m| DeviceBuffer::alloc(nj * t_var * m.rows * 4).unwrap())
        .collect();
    let f = func(
        c,
        &format!(
            "mm_{}_{}_t{t_var}_kernel",
            if res { "res" } else { "out" },
            SUF[qf as usize]
        ),
    );
    let mut args = Args::new();
    for i in 0..3 {
        args = match mats.get(i) {
            Some(m) => planes_args(args, m.planes),
            None => args.ptr(0).ptr(0).ptr(0).ptr(0),
        };
    }
    for i in 0..3 {
        args = args.ptr(match mats.get(i) {
            Some(_) => {
                if nj > 1 {
                    parts[i].ptr()
                } else {
                    outs[i].ptr()
                }
            }
            None => 0,
        });
    }
    for i in 0..3 {
        args = args.i32(mats.get(i).map(|m| m.rows as i32).unwrap_or(0));
    }
    for i in 0..3 {
        args = args.i32(mats.get(i).map(|m| m.rows as i32).unwrap_or(0));
    }
    f.launch(
        (gx, nj as u32, 1),
        (nw * 32, 1, 1),
        smem as u32,
        &c.stream,
        &args
            .i32(mats.len() as i32)
            .buf(&a.xq)
            .buf(&a.xd)
            .buf(&a.xs)
            .i32(k as i32),
    )
    .unwrap();
    if nj > 1 {
        let mut r = Args::new();
        for i in 0..3 {
            r = r.ptr(if i < mats.len() { parts[i].ptr() } else { 0 });
        }
        for i in 0..3 {
            r = r.ptr(if i < mats.len() { outs[i].ptr() } else { 0 });
        }
        for i in 0..3 {
            r = r.i32(mats.get(i).map(|m| m.rows as i32).unwrap_or(0));
        }
        for i in 0..3 {
            r = r.i32(mats.get(i).map(|m| m.rows as i32).unwrap_or(0));
        }
        let max_rows = mats.iter().map(|m| m.rows).max().unwrap();
        func(
            c,
            if res {
                "mm_reduce_res_kernel"
            } else {
                "mm_reduce_out_kernel"
            },
        )
        .launch(
            (((max_rows + 255) / 256) as u32, t_var as u32, mats.len() as u32),
            (256, 1, 1),
            0,
            &c.stream,
            &r.i32(nj as i32),
        )
        .unwrap();
    }
    c.stream.synchronize().unwrap();
    mats.iter()
        .zip(&outs)
        .map(|(m, o)| o.download_f32(t_var * m.rows).unwrap())
        .collect()
}

/// A single matrix.
#[allow(clippy::too_many_arguments)]
fn run_one(c: &Ctx, qf: QFmt, planes: &[DeviceBuffer], a: &Acts, t_var: usize, rows: usize, k: usize, nj: usize, gx: u32, nw: u32, base: Option<&[f32]>) -> Vec<f32> {
    let bases = base.map(|b| vec![b.to_vec()]);
    run_mm(c, qf, &[Mat { planes, rows }], a, t_var, k, nj, gx, nw, bases.as_deref())
        .pop()
        .unwrap()
}

#[test]
fn multi_row_matvec_matches_the_reference_and_does_not_depend_on_the_launch() {
    let _gpu = common::gpu_lock();
    let Some(c) = setup() else { return };
    for qf in [
        QFmt::Q8_0,
        QFmt::Q4_0,
        QFmt::Q5_0,
        QFmt::Q4_K,
        QFmt::Q5_K,
        QFmt::Q6_K,
    ] {
        // (rows, K): K of one slice and of six, rows that leave idle slots, rows of a single group
        for (rows, k) in [(66usize, 1536usize), (8, 256), (130, 768), (4, 2560)] {
            let (planes, w) = weights(qf, rows, k, 11 + k as u64);
            let a = acts(&c, 16, k, 5 + rows as u64);
            let want = reference(&w, &a, 16, rows, k);
            let mut first: Option<Vec<f32>> = None;
            let nj0 = min_nj(qf, 16, k);
            for t_var in [4usize, 8, 12, 16] {
                for (gx, nw) in [(1u32, 1u32), (3, 4), (2, 12)] {
                    let got = run_one(&c, qf, &planes, &a, t_var, rows, k, nj0, gx, nw, None);
                    // the rows < t_var of the reference
                    let err = rel_rms(
                        &got[..t_var * rows]
                            .iter()
                            .map(|v| *v as f64)
                            .collect::<Vec<_>>(),
                        &want[..t_var * rows],
                    );
                    assert!(
                        err < 1e-5,
                        "{} rows {rows} K {k} T {t_var} grid {gx} x {nw}: relative rms {err}",
                        qf.name()
                    );
                    // bit for bit independent of the number of rows and of the launch shape
                    match &first {
                        None => first = Some(got[..4 * rows].to_vec()),
                        Some(f) => assert_eq!(
                            &got[..4 * rows],
                            &f[..],
                            "{} rows {rows} K {k} T {t_var} grid {gx} x {nw}: rows differ from the first run",
                            qf.name()
                        ),
                    }
                }
            }
            // K slices: another summation order, same values
            let nsb = k / 256;
            for nj in [2usize, 3, 6].into_iter().filter(|&j| j <= nsb) {
                let got = run_one(&c, qf, &planes, &a, 8, rows, k, nj.max(min_nj(qf, 8, k)), 3, 4, None);
                let err = rel_rms(
                    &got.iter().map(|v| *v as f64).collect::<Vec<_>>(),
                    &want[..8 * rows],
                );
                assert!(
                    err < 1e-5,
                    "{} rows {rows} K {k}: {nj} slices: relative rms {err}",
                    qf.name()
                );
            }
            // the residual variant adds exactly
            let base: Vec<f32> = (0..8 * rows).map(|i| (i % 7) as f32 - 3.0).collect();
            let nj8 = min_nj(qf, 8, k);
            let out = run_one(&c, qf, &planes, &a, 8, rows, k, nj8, 3, 4, None);
            let add = run_one(&c, qf, &planes, &a, 8, rows, k, nj8, 3, 4, Some(&base));
            for i in 0..8 * rows {
                assert_eq!(add[i], base[i] + out[i], "{} residual row {i}", qf.name());
            }
            // up to three matrices in one launch: every output equals the one of a launch of its own, bit for bit (also with K slices)
            {
                let (p2, _) = weights(qf, rows + 6, k, 77 + k as u64);
                let (p3, _) = weights(qf, 2, k, 99 + k as u64);
                let mats = [
                    Mat { planes: &planes, rows },
                    Mat { planes: &p2, rows: rows + 6 },
                    Mat { planes: &p3, rows: 2 },
                ];
                for nj in [nj8, (nj8 + 1).min(nsb)] {
                    let fused = run_mm(&c, qf, &mats, &a, 8, k, nj, 3, 4, None);
                    for (i, m) in mats.iter().enumerate() {
                        let alone = run_one(&c, qf, m.planes, &a, 8, m.rows, k, nj, 2, 12, None);
                        assert_eq!(fused[i], alone, "{} K {k}: matrix {i} of a fused launch ({nj} slices) differs from its own launch", qf.name());
                    }
                }
            }
            eprintln!(
                "{} rows {rows} K {k}: relative rms vs the double-precision reference {:.2e}",
                qf.name(),
                rel_rms(
                    &out.iter().map(|v| *v as f64).collect::<Vec<_>>(),
                    &want[..8 * rows]
                )
            );
        }
    }
}

// ---- whole prompts -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

fn models_dir() -> PathBuf {
    match std::env::var("FORGE_MODELS") {
        Ok(p) => PathBuf::from(p),
        Err(_) => PathBuf::from(std::env::var("HOME").unwrap_or_default())
            .join("forgeinfer/oracle/models"),
    }
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut ab, mut aa, mut bb) = (0f64, 0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        ab += (*x as f64) * (*y as f64);
        aa += (*x as f64).powi(2);
        bb += (*y as f64).powi(2);
    }
    ab / (aa.sqrt() * bb.sqrt())
}

fn prompts(golden: &str) -> Vec<Vec<u32>> {
    let gold = Json::from_file(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/golden").join(golden)).unwrap();
    gold["prompts"]
        .as_array()
        .unwrap()
        .iter()
        .map(|p| {
            p["ids"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as u32)
                .collect()
        })
        .collect()
}

/// The logits after `prefill_batched` of `ids` (in pieces of `piece` tokens), then `n` more greedy tokens. `token_by_token`: the decode graphs (`prefill`: the activations are quantized to int8 as in the
/// multi-row path), the reference of identical numerics; `prefill_batched` with `mrow_max` 0 is the GEMM path (fp16 operands), the reference the multi-row path is measured against.
fn run(sess: &mut GpuSession, ids: &[u32], piece: usize, n: usize, token_by_token: bool) -> (Vec<f32>, Vec<u32>) {
    sess.reset().unwrap();
    for ch in ids.chunks(piece.max(1)) {
        if token_by_token {
            sess.prefill(ch).unwrap();
        } else {
            sess.prefill_batched(ch).unwrap();
        }
    }
    let logits = sess.logits().unwrap();
    // (single steps: after a prompt in several `prefill_batched` calls the output ring of `decode_greedy` holds the arg-max of every call's last token first)
    let mut toks = vec![argmax(&logits)];
    for _ in 1..n {
        let l = sess.step(*toks.last().unwrap()).unwrap();
        toks.push(argmax(&l));
    }
    (logits, toks)
}

fn check_model(rel: &str, golden: &str, name: &str) {
    let path = models_dir().join(rel);
    if !path.exists() || Device::count().unwrap_or(0) == 0 {
        eprintln!("skipped {name}: model or CUDA device missing");
        return;
    }
    let ps = prompts(golden);
    let model = Model::load_native(&path).unwrap();
    let gpu = GpuModel::upload(&model, Fmt::Native, 0).unwrap();
    drop(model);
    let mut sess = GpuSession::new(&gpu, 512).unwrap();
    let all: Vec<u32> = ps.iter().flatten().copied().collect();
    if std::env::var("FORGE_HOST_FFN").is_ok() {
        // the FFN on the CPU (FORGE_HOST_FFN): its kernels take up to 4 rows, so one group of 4 and chunks of 1 to 4 tokens, against the decode path
        sess.set_mrow_group(4);
        let mut worst = 1f64;
        for t in 1..=4usize {
            let ids = &all[..t];
            sess.set_mrow_max(0);
            let (la, _) = run(&mut sess, ids, t, 1, true);
            sess.set_mrow_max(4);
            let (lm, _) = run(&mut sess, ids, t, 1, false);
            worst = worst.min(cosine(&la, &lm));
        }
        eprintln!("{name}: FFN on the CPU, chunks of 1 to 4 tokens through the multi-row path: worst logits cosine against the decode path {worst:.6}");
        assert!(worst > 0.98, "{name}: FFN on the CPU: multi-row chunks: worst logits cosine {worst}");
        return;
    }
    // 1. the prompts of the golden file, every group size: against the decode graphs (same activation quantization) and against the GEMM path (fp16 operands)
    let (mut worst_a, mut worst_b, mut worst_ab) = (1f64, 1f64, 1f64);
    let (mut equal_tokens, mut total_tokens, mut same_top1, mut n_runs) = (0usize, 0usize, 0usize, 0usize);
    for p in ps.iter().filter(|p| p.len() >= 2 && p.len() <= 64) {
        sess.set_mrow_max(0);
        let (la, ta) = run(&mut sess, p, p.len(), 12, true);
        let (lb, _) = run(&mut sess, p, p.len(), 12, false);
        worst_ab = worst_ab.min(cosine(&la, &lb));
        for g in [4usize, 8, 12, 16] {
            sess.set_mrow_max(64);
            sess.set_mrow_group(g);
            let (lm, tm) = run(&mut sess, p, p.len(), 12, false);
            let (ca, cb) = (cosine(&la, &lm), cosine(&lb, &lm));
            if std::env::var("MROW_VERBOSE").is_ok() {
                eprintln!("{name}: prompt of {} tokens, group {g}: cos(decode, mrow) {ca:.7}  cos(gemm, mrow) {cb:.7}  cos(decode, gemm) {:.7}", p.len(), cosine(&la, &lb));
            }
            worst_a = worst_a.min(ca);
            worst_b = worst_b.min(cb);
            assert!(ca > 0.999, "{name}: a prompt of {} tokens, group {g}: logits cosine {ca} against the decode path", p.len());
            assert!(cb > 0.998, "{name}: a prompt of {} tokens, group {g}: logits cosine {cb} against the GEMM path", p.len());
            // int8 activations make the choice between near-tied candidates path dependent: the multi-row path's top-1 must be among the GEMM path's best three
            let mut order: Vec<usize> = (0..lb.len()).collect();
            order.sort_by(|a, b| lb[*b].partial_cmp(&lb[*a]).unwrap());
            assert!(order[..3].contains(&(argmax(&lm) as usize)), "{name}: a prompt of {} tokens, group {g}: top-1 {} is not in the GEMM path's top 3", p.len(), argmax(&lm));
            same_top1 += (argmax(&lm) == argmax(&la)) as usize;
            n_runs += 1;
            equal_tokens += ta.iter().zip(&tm).filter(|(a, b)| a == b).count();
            total_tokens += ta.len();
        }
    }
    eprintln!("{name}: golden prompts: logits cosine against the decode path >= {worst_a:.7}, against the GEMM path >= {worst_b:.6} (the decode path against the GEMM path: {worst_ab:.6}); top-1 equal to the decode path's in {same_top1} of {n_runs} runs, greedy continuation {equal_tokens} of {total_tokens} tokens equal to the decode path's");
    // 2. every chunk length from 1 to 40 (idle rows of the 4 / 8 / 12 / 16 variants, several groups), in one chunk and in two pieces (the second continues at a position inside the cache): the logits agree with the
    //    decode path's about as well as the GEMM path's do (the int8 activations make every pair of paths differ by a few 1e-4 in cosine, more for the shortest prompts)
    sess.set_mrow_group(8);
    let (mut w_gemm, mut w_mrow, mut w_pieces) = (1f64, 1f64, 1f64);
    for t in 1..=40usize {
        let ids = &all[..t];
        sess.set_mrow_max(0);
        let (la, _) = run(&mut sess, ids, t, 1, true);
        let (lg, _) = run(&mut sess, ids, t, 1, false);
        sess.set_mrow_max(64);
        let (lm, _) = run(&mut sess, ids, t, 1, false);
        w_gemm = w_gemm.min(cosine(&la, &lg));
        w_mrow = w_mrow.min(cosine(&la, &lm));
        if t >= 3 {
            let (lp, _) = run(&mut sess, ids, t / 2 + 1, 1, false);
            w_pieces = w_pieces.min(cosine(&la, &lp));
        }
    }
    eprintln!("{name}: chunks of 1 to 40 tokens, worst logits cosine against the decode path: GEMM path {w_gemm:.6}, multi-row {w_mrow:.6}, multi-row in two pieces {w_pieces:.6}");
    assert!(w_mrow > 0.98 && w_mrow > w_gemm - 0.01, "{name}: chunks: worst logits cosine {w_mrow} (GEMM path {w_gemm})");
    assert!(w_pieces > 0.98 && w_pieces > w_gemm - 0.01, "{name}: pieces: worst logits cosine {w_pieces} (GEMM path {w_gemm})");
    // 2b. the multi-row path is deterministic and does not depend on how a chunk is cut: groups of 4, 8, 12 or 16 rows and prompts in pieces give the same logits bit for bit
    sess.set_mrow_max(64);
    let ids = &all[..37.min(all.len())];
    sess.set_mrow_group(16);
    let (l16, t16) = run(&mut sess, ids, ids.len(), 8, false);
    for g in [4usize, 8, 12] {
        sess.set_mrow_group(g);
        let (lg, tg) = run(&mut sess, ids, ids.len(), 8, false);
        assert!(lg == l16 && tg == t16, "{name}: groups of {g} rows give other logits than groups of 16");
    }
    sess.set_mrow_group(16);
    for piece in [5usize, 11, 16, 20] {
        let (lp, tp) = run(&mut sess, ids, piece, 8, false);
        if std::env::var("MROW_VERBOSE").is_ok() {
            let md = lp.iter().zip(&l16).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
            let nd = lp.iter().zip(&l16).filter(|(a, b)| a != b).count();
            eprintln!("{name}: pieces of {piece}: {nd} of {} logits differ, max |diff| {md:e}, tokens equal {}", lp.len(), tp == t16);
        }
        assert!(lp == l16 && tp == t16, "{name}: pieces of {piece} tokens give other logits than one chunk");
    }
    let (again, _) = run(&mut sess, ids, ids.len(), 8, false);
    assert!(again == l16, "{name}: a second run gives other logits");
    // 3. the evaluation logits (every row) of a chunk against the GEMM path's
    let ids = &all[..40.min(all.len())];
    sess.set_mrow_max(0);
    sess.reset().unwrap();
    sess.set_mrow_eval(false);
    let l0 = sess.prefill_logits(ids).unwrap();
    sess.reset().unwrap();
    sess.set_mrow_eval(true);
    sess.set_mrow_group(16);
    let l1 = sess.prefill_logits(ids).unwrap();
    sess.set_mrow_eval(false);
    let v = l0.len() / ids.len();
    let (mut worst, mut same) = (1f64, 0usize);
    let first = cosine(&l0[..v], &l1[..v]);
    for r in 0..ids.len() {
        let (a, b) = (&l0[r * v..(r + 1) * v], &l1[r * v..(r + 1) * v]);
        // (the first token of a sequence carries the model's massive activations: the int8 activations of the decode path and of this one differ from the fp16 GEMM path's there by far more than anywhere else)
        if r > 0 {
            worst = worst.min(cosine(a, b));
        }
        same += (argmax(a) == argmax(b)) as usize;
    }
    eprintln!("{name}: evaluation logits of {} rows against the GEMM path: cosine of the first row {first:.4}, of the others >= {worst:.6}, top-1 equal in {same}", ids.len());
    assert!(first > 0.9, "{name}: evaluation logits cosine {first} of the first row");
    assert!(worst > 0.985, "{name}: evaluation logits cosine {worst}");
    assert!(same * 100 >= ids.len() * 85, "{name}: top-1 agreement {same} of {}", ids.len());
}

#[test]
fn multi_row_prefill_matches_the_gemm_path_dense() {
    let _gpu = common::gpu_lock();
    check_model(
        "Qwen__Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf",
        "fwd_qwen3.json",
        "Qwen3-0.6B Q8_0",
    );
    check_model(
        "unsloth__Qwen3-1.7B-GGUF/Qwen3-1.7B-Q4_K_M.gguf",
        "fwd_qwen3.json",
        "Qwen3-1.7B Q4_K_M",
    );
}

#[test]
fn multi_row_prefill_matches_the_gemm_path_hybrid() {
    let _gpu = common::gpu_lock();
    check_model(
        "unsloth__Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf",
        "fwd_qwen35.json",
        "Qwen3.5-0.8B Q4_K_M",
    );
}

/// `qkv_prep_multi_kernel` (QK norm, RoPE and the cache append of N rows of a dense model) against a host reference: with norm weights (Qwen3) and without (null weights: the Llama-like models), full and
/// partial rotary dimensions, rows at positions beyond the start of the cache. (Qwen3 with its norms removed is no substitute for a Llama model: it computes noise, and the three paths of the engine then
/// differ from each other by far more than any bug would.)
#[test]
fn qkv_prep_multi_matches_the_reference() {
    let _gpu = common::gpu_lock();
    let Some(c) = setup() else { return };
    let f = func(&c, "qkv_prep_multi_kernel");
    let bytes = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
    let mut r = Lcg(77);
    for &(nh, nkv, hd, rd, norms) in &[
        (8usize, 2usize, 64usize, 64usize, true),
        (8, 2, 64, 64, false),
        (4, 2, 128, 128, true),
        (4, 2, 128, 128, false),
        (6, 3, 64, 32, true),
        (6, 3, 64, 32, false),
    ] {
        let (n, pos0, eps) = (5usize, 7usize, 1e-6f32);
        let (half, kvd, tbl) = (rd / 2, nkv * hd, pos0 + 5);
        let q: Vec<f32> = (0..n * nh * hd).map(|_| r.unit() * 3.0).collect();
        let k: Vec<f32> = (0..n * kvd).map(|_| r.unit() * 3.0).collect();
        let v: Vec<f32> = (0..n * kvd).map(|_| r.unit()).collect();
        let qw: Vec<f32> = (0..hd).map(|_| 0.5 + r.unit().abs()).collect();
        let kw: Vec<f32> = (0..hd).map(|_| 0.5 + r.unit().abs()).collect();
        let cs: Vec<f32> = (0..tbl * half).map(|i| ((i % 97) as f32 * 0.07).cos()).collect();
        let sn: Vec<f32> = (0..tbl * half).map(|i| ((i % 97) as f32 * 0.07).sin()).collect();
        let up = |x: &[f32]| DeviceBuffer::from_bytes(&bytes(x)).unwrap();
        let (qb, kb, vb, qwb, kwb, cb, sb) = (up(&q), up(&k), up(&v), up(&qw), up(&kw), up(&cs), up(&sn));
        let mut st = Vec::new();
        for x in [0u32, pos0 as u32, 0, 0] {
            st.extend_from_slice(&x.to_le_bytes());
        }
        let stb = DeviceBuffer::from_bytes(&st).unwrap();
        let qo = DeviceBuffer::alloc(n * nh * hd * 4).unwrap();
        let (kc, vc) = (DeviceBuffer::alloc(tbl * kvd * 2).unwrap(), DeviceBuffer::alloc(tbl * kvd * 2).unwrap());
        f.launch(
            ((nh + 2 * nkv) as u32, n as u32, 1),
            (256, 1, 1),
            0,
            &c.stream,
            &Args::new()
                .ptr(qb.ptr())
                .ptr(kb.ptr())
                .ptr(vb.ptr())
                .ptr(if norms { qwb.ptr() } else { 0 })
                .ptr(if norms { kwb.ptr() } else { 0 })
                .ptr(cb.ptr())
                .ptr(sb.ptr())
                .ptr(stb.ptr())
                .ptr(qo.ptr())
                .ptr(kc.ptr())
                .ptr(vc.ptr())
                .i32(nh as i32)
                .i32(nkv as i32)
                .i32(hd as i32)
                .i32(rd as i32)
                .f32(eps),
        )
        .unwrap();
        c.stream.synchronize().unwrap();
        // the host reference of one head: RMS norm with the weights, then the rotation of the first `rd` dimensions
        let head_ref = |src: &[f32], w: Option<&[f32]>, pos: usize| -> Vec<f64> {
            let mut rs = 1f64;
            if w.is_some() {
                let ss: f64 = src.iter().map(|x| (*x as f64).powi(2)).sum();
                rs = 1.0 / (ss / hd as f64 + eps as f64).sqrt();
            }
            let wd = |i: usize| w.map(|w| w[i] as f64).unwrap_or(1.0);
            (0..hd)
                .map(|d| {
                    if d < rd {
                        let i = if d < half { d } else { d - half };
                        let (a, b) = (src[i] as f64 * rs * wd(i), src[i + half] as f64 * rs * wd(i + half));
                        let (cc, ss) = (cs[pos * half + i] as f64, sn[pos * half + i] as f64);
                        if d < half {
                            a * cc - b * ss
                        } else {
                            b * cc + a * ss
                        }
                    } else {
                        src[d] as f64 * rs * wd(d)
                    }
                })
                .collect()
        };
        let got_q = qo.download_f32(n * nh * hd).unwrap();
        let half_vec = |b: &DeviceBuffer| -> Vec<f32> {
            let mut raw = vec![0u8; tbl * kvd * 2];
            b.download(&mut raw).unwrap();
            raw.chunks_exact(2).map(|h| forge_core::half::f16_to_f32(u16::from_le_bytes([h[0], h[1]]))).collect()
        };
        let (got_k, got_v) = (half_vec(&kc), half_vec(&vc));
        let what = format!("{nh} q heads, {nkv} kv heads, hd {hd}, rd {rd}, {}", if norms { "QK norm" } else { "no QK norm" });
        for t in 0..n {
            let pos = pos0 + t;
            for h in 0..nh {
                let want = head_ref(&q[(t * nh + h) * hd..(t * nh + h + 1) * hd], norms.then_some(&qw[..]), pos);
                for d in 0..hd {
                    let g = got_q[(t * nh + h) * hd + d] as f64;
                    assert!((g - want[d]).abs() <= 2e-5 * want[d].abs().max(1.0), "{what}: q row {t} head {h} dim {d}: {g} against {}", want[d]);
                }
            }
            for h in 0..nkv {
                let want = head_ref(&k[(t * nkv + h) * hd..(t * nkv + h + 1) * hd], norms.then_some(&kw[..]), pos);
                for d in 0..hd {
                    let g = got_k[pos * kvd + h * hd + d] as f64;
                    assert!((g - want[d]).abs() <= 1.5e-3 * want[d].abs().max(1.0), "{what}: k row {t} head {h} dim {d}: {g} against {}", want[d]);
                    let (gv, wv) = (got_v[pos * kvd + h * hd + d] as f64, v[(t * nkv + h) * hd + d] as f64);
                    assert!((gv - wv).abs() <= 1e-3 * wv.abs().max(1e-2), "{what}: v row {t} head {h} dim {d}: {gv} against {wv}");
                }
            }
        }
    }
}

// ---- multi-row verification ------------------------------------------------------------------------------------------------------------------------------------------------------------------------

/// `verify_rows` of k tokens against k single-token steps from the same state, and the state after `accept_rows(m)` (the position, the hidden state, a hybrid model's recurrent state) against the state
/// of m single steps: the step after it must give the logits the single-step chain gives.
fn check_verify(rel: &str, golden: &str, name: &str) {
    let path = models_dir().join(rel);
    if !path.exists() || Device::count().unwrap_or(0) == 0 {
        eprintln!("skipped {name}: model or CUDA device missing");
        return;
    }
    let ps = prompts(golden);
    let ids: Vec<u32> = ps[6.min(ps.len() - 1)].clone();
    let model = Model::load_native(&path).unwrap();
    let gpu = GpuModel::upload(&model, Fmt::Native, 0).unwrap();
    drop(model);
    let mut sess = GpuSession::new(&gpu, 512).unwrap();
    // the reference chain: 18 greedy tokens by 17 single steps
    sess.reset().unwrap();
    sess.prefill_batched(&ids).unwrap();
    let mut chain = vec![argmax(&sess.logits().unwrap())];
    let mut logits: Vec<Vec<f32>> = Vec::new();
    for i in 0..17 {
        let l = sess.step(chain[i]).unwrap();
        chain.push(argmax(&l));
        logits.push(l);
    }
    let start = sess.pos - 17;
    assert_eq!(start, ids.len());
    // (weights in host memory, FORGE_HOST_FFN: the CPU kernels take at most 4 rows)
    let max_k = if std::env::var("FORGE_HOST_FFN").is_ok() { 4 } else { 16 };
    for k in [2usize, 3, 4, 7, 16].into_iter().filter(|&k| k <= max_k) {
        sess.reset().unwrap();
        sess.prefill_batched(&ids).unwrap();
        sess.verify_rows(&chain[..k]).unwrap();
        assert_eq!(sess.pos, start + k);
        let am = sess.rows_argmax().unwrap();
        for i in 0..k {
            let l = sess.rows_logits(i).unwrap();
            let cos = cosine(&l, &logits[i]);
            assert!(cos > 0.998, "{name}: {k} rows, row {i}: logits cosine {cos} against the single step");
            assert_eq!(am[i], argmax(&l), "{name}: arg-max of row {i}");
        }
        // go back to the position after row m - 1 and continue with the true next token: the logits must be those of the single-step chain
        for m in (1..=k).filter(|m| *m == 1 || *m == k || *m == k / 2) {
            sess.reset().unwrap();
            sess.prefill_batched(&ids).unwrap();
            sess.verify_rows(&chain[..k]).unwrap();
            sess.accept_rows(m).unwrap();
            assert_eq!(sess.pos, start + m);
            let l = sess.step(chain[m]).unwrap();
            let cos = cosine(&l, &logits[m]);
            assert!(cos > 0.998, "{name}: {k} rows, accept {m}: the step after it has logits cosine {cos}");
        }
    }
    eprintln!("{name}: verify_rows of 2 / 3 / 4 / 7 / 16 rows agree with single steps, accept_rows restores the state");
}

#[test]
fn verify_rows_matches_single_steps_dense() {
    let _gpu = common::gpu_lock();
    check_verify("unsloth__Qwen3-1.7B-GGUF/Qwen3-1.7B-Q4_K_M.gguf", "fwd_qwen3.json", "Qwen3-1.7B Q4_K_M");
}

#[test]
fn verify_rows_matches_single_steps_hybrid() {
    let _gpu = common::gpu_lock();
    check_verify("unsloth__Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf", "fwd_qwen35.json", "Qwen3.5-0.8B Q4_K_M");
}

/// Timing of the multi-row verification against a single-token step and the two-token step (run with --ignored --nocapture on an idle GPU; FORGE_VERIFY_MODEL = a GGUF).
#[test]
#[ignore]
fn verify_rows_timing() {
    let _gpu = common::gpu_lock();
    let Ok(path) = std::env::var("FORGE_VERIFY_MODEL") else {
        eprintln!("skipped: FORGE_VERIFY_MODEL not set");
        return;
    };
    if Device::count().unwrap_or(0) == 0 {
        return;
    }
    let model = Model::load_native(PathBuf::from(path)).unwrap();
    let gpu = GpuModel::upload(&model, Fmt::Native, 0).unwrap();
    let hybrid = model.cfg.delta.is_some();
    drop(model);
    let mut sess = GpuSession::new(&gpu, 2048).unwrap();
    let ids: Vec<u32> = (0..40u32).map(|i| 100 + (i * 7919) % 40000).collect();
    sess.prefill_batched(&ids).unwrap();
    let mut t = argmax(&sess.logits().unwrap());
    let n = 60;
    for _ in 0..8 {
        sess.advance(t).unwrap();
        t = sess.argmax_token().unwrap();
    }
    let t0 = std::time::Instant::now();
    for _ in 0..n {
        sess.advance(t).unwrap();
        t = sess.argmax_token().unwrap();
    }
    let single = t0.elapsed().as_secs_f64() / n as f64 * 1e3;
    let mut report = format!("single-token step {single:.2} ms");
    if hybrid {
        sess.verify2(t, 1000).unwrap();
        sess.rollback_verify().unwrap();
        let t0 = std::time::Instant::now();
        for _ in 0..n {
            sess.verify2(t, 1000).unwrap();
            sess.rollback_verify().unwrap();
        }
        report += &format!(" | verify2 + rollback {:.2} ms", t0.elapsed().as_secs_f64() / n as f64 * 1e3);
    }
    for k in [2usize, 3, 4, 6, 8, 12, 16] {
        let toks: Vec<u32> = (0..k as u32).map(|i| 200 + i * 31).collect();
        sess.verify_rows(&toks).unwrap();
        sess.accept_rows(1).unwrap();
        let t0 = std::time::Instant::now();
        for _ in 0..n {
            sess.verify_rows(&toks).unwrap();
            let _ = sess.rows_argmax().unwrap();
            sess.accept_rows(1).unwrap();
        }
        report += &format!(" | verify_rows({k}) + argmax + accept(1) {:.2} ms", t0.elapsed().as_secs_f64() / n as f64 * 1e3);
    }
    eprintln!("{report}");
}
