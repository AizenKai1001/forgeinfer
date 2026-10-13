//! The tensor-core GEMM kernels (forward NT, data-gradient NN, weight-gradient TN) against a CPU reference (f32 accumulation of the fp16 inputs) on awkward and large shapes, and throughput.

use forge_core::half::{f16_to_f32, f32_to_f16};
use forge_cuda::{Args, Device, DeviceBuffer, Event, Function, Module, Stream};
use forge_model::sample::Rng;

fn rand_f16(n: usize, seed: u64) -> (Vec<u16>, Vec<f32>) {
    let mut rng = Rng::new(seed);
    let h: Vec<u16> = (0..n)
        .map(|_| f32_to_f16(rng.next_f32() * 2.0 - 1.0))
        .collect();
    let f: Vec<f32> = h.iter().map(|&v| f16_to_f32(v)).collect();
    (h, f)
}

fn bytes_of(h: &[u16]) -> Vec<u8> {
    h.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// C[m][n] = sum_k A(m,k) B(k,n) with A given as [m][k] or [k][m], B as [n][k] (b_nk) or [k][n]
fn cpu_gemm(
    a: &[f32],
    a_col: bool,
    b: &[f32],
    b_col: bool,
    m: usize,
    n: usize,
    k: usize,
) -> Vec<f32> {
    let mut c = vec![0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut s = 0f32;
            for kk in 0..k {
                let av = if a_col { a[kk * m + i] } else { a[i * k + kk] };
                let bv = if b_col { b[kk * n + j] } else { b[j * k + kk] };
                s += av * bv;
            }
            c[i * n + j] = s;
        }
    }
    c
}

struct Kernels {
    nt_f16: Function,
    nt_f32: Function,
    nt_add: Function,
    nn_f32: Function,
    nn_add: Function,
    tn_f32: Function,
    tn_add: Function,
    // previous generations: padded rows with the store right behind the load, BK = 16 (two blocks per SM) and BK = 32 (one block per SM)
    nt_f32_b16: Function,
    nn_f32_b16: Function,
    tn_f32_b16: Function,
    nt_f32_b32: Function,
    nn_f32_b32: Function,
    tn_f32_b32: Function,
}

fn setup() -> Option<(Device, Module, Kernels, Stream)> {
    if Device::count().unwrap_or(0) == 0 {
        eprintln!("skipped: no CUDA device");
        return None;
    }
    let dev = Device::open(0).unwrap();
    let (image, _) = forge_kernels::image_for(dev.info.cc_major, dev.info.cc_minor).unwrap();
    let module = Module::load(image).unwrap();
    let k = Kernels {
        nt_f16: module.function("gemm_nt_f16").unwrap(),
        nt_f32: module.function("gemm_nt_f32").unwrap(),
        nt_add: module.function("gemm_nt_add").unwrap(),
        nn_f32: module.function("gemm_nn_f32").unwrap(),
        nn_add: module.function("gemm_nn_add").unwrap(),
        tn_f32: module.function("gemm_tn_f32").unwrap(),
        tn_add: module.function("gemm_tn_add").unwrap(),
        nt_f32_b16: module.function("gemm_nt_f32_b16").unwrap(),
        nn_f32_b16: module.function("gemm_nn_f32_b16").unwrap(),
        tn_f32_b16: module.function("gemm_tn_f32_b16").unwrap(),
        nt_f32_b32: module.function("gemm_nt_f32_b32").unwrap(),
        nn_f32_b32: module.function("gemm_nn_f32_b32").unwrap(),
        tn_f32_b32: module.function("gemm_tn_f32_b32").unwrap(),
    };
    Some((dev, module, k, Stream::new().unwrap()))
}

#[allow(clippy::too_many_arguments)]
fn launch(
    f: Function,
    stream: &Stream,
    a: &DeviceBuffer,
    b: &DeviceBuffer,
    c: &DeviceBuffer,
    m: usize,
    n: usize,
    k: usize,
    lda: usize,
    ldb: usize,
    ldc: usize,
) {
    let grid = (((n + 127) / 128) as u32, ((m + 127) / 128) as u32, 1);
    let args = Args::new()
        .buf(a)
        .buf(b)
        .buf(c)
        .i32(m as i32)
        .i32(n as i32)
        .i32(k as i32)
        .i32(lda as i32)
        .i32(ldb as i32)
        .i32(ldc as i32)
        .f32(1.0);
    f.launch(grid, (256, 1, 1), 0, stream, &args).unwrap();
}

#[test]
fn gemm_layouts_match_the_cpu_reference() {
    let Some((_dev, _module, k, stream)) = setup() else {
        return;
    };
    // (m, n, k): not multiples of the tile; the k-major operands need m and n multiples of 8
    for &(m, n, kk) in &[
        (130usize, 200usize, 96usize),
        (8, 64, 32),
        (257 - 1, 576, 576),
        (512, 1536, 576),
        (72, 960, 1536),
        (1024, 576, 2048),
    ] {
        for &(a_col, b_col) in &[(false, false), (false, true), (true, true)] {
            if (a_col && m % 8 != 0) || (b_col && n % 8 != 0) {
                continue;
            }
            let (ah, af) = rand_f16(m * kk, 1 + m as u64);
            let (bh, bf) = rand_f16(n * kk, 2 + n as u64);
            let want = cpu_gemm(&af, a_col, &bf, b_col, m, n, kk);
            let da = DeviceBuffer::from_bytes(&bytes_of(&ah)).unwrap();
            let db = DeviceBuffer::from_bytes(&bytes_of(&bh)).unwrap();
            let dc = DeviceBuffer::alloc(m * n * 4).unwrap();
            let lda = if a_col { m } else { kk };
            let ldb = if b_col { n } else { kk };
            let f = match (a_col, b_col) {
                (false, false) => k.nt_f32,
                (false, true) => k.nn_f32,
                _ => k.tn_f32,
            };
            launch(f, &stream, &da, &db, &dc, m, n, kk, lda, ldb, n);
            stream.synchronize().unwrap();
            let got = dc.download_f32(m * n).unwrap();
            let worst = got
                .iter()
                .zip(&want)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            let scale = want.iter().fold(0f32, |m, v| m.max(v.abs()));
            let name = match (a_col, b_col) {
                (false, false) => "NT",
                (false, true) => "NN",
                _ => "TN",
            };
            eprintln!("{name} {m}x{n}x{kk}: max |diff| {worst:.2e} (max |value| {scale:.1})");
            assert!(
                worst < 2e-3 * scale.max(1.0),
                "{name} {m}x{n}x{kk}: {worst}"
            );
            for (tag, legacy) in [
                ("BK 16 old", [k.nt_f32_b16, k.nn_f32_b16, k.tn_f32_b16]),
                ("BK 32 old", [k.nt_f32_b32, k.nn_f32_b32, k.tn_f32_b32]),
            ] {
                if tag == "BK 32 old" && kk % 32 != 0 {
                    continue;
                }
                let f = match (a_col, b_col) {
                    (false, false) => legacy[0],
                    (false, true) => legacy[1],
                    _ => legacy[2],
                };
                launch(f, &stream, &da, &db, &dc, m, n, kk, lda, ldb, n);
                stream.synchronize().unwrap();
                let got = dc.download_f32(m * n).unwrap();
                let worst = got
                    .iter()
                    .zip(&want)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f32, f32::max);
                assert!(
                    worst < 2e-3 * scale.max(1.0),
                    "{name} ({tag}) {m}x{n}x{kk}: {worst}"
                );
            }
            // accumulate variant: C starts at 1
            let add = match (a_col, b_col) {
                (false, false) => k.nt_add,
                (false, true) => k.nn_add,
                _ => k.tn_add,
            };
            dc.upload_f32(&vec![1f32; m * n]).unwrap();
            launch(add, &stream, &da, &db, &dc, m, n, kk, lda, ldb, n);
            stream.synchronize().unwrap();
            let got = dc.download_f32(m * n).unwrap();
            let worst = got
                .iter()
                .zip(&want)
                .map(|(a, b)| (a - 1.0 - b).abs())
                .fold(0f32, f32::max);
            assert!(
                worst < 2e-3 * scale.max(1.0),
                "{name} add {m}x{n}x{kk}: {worst}"
            );
        }
    }
    // fp16 output (forward layout only)
    let (m, n, kk) = (130usize, 200usize, 96usize);
    let (ah, af) = rand_f16(m * kk, 7);
    let (bh, bf) = rand_f16(n * kk, 8);
    let want = cpu_gemm(&af, false, &bf, false, m, n, kk);
    let da = DeviceBuffer::from_bytes(&bytes_of(&ah)).unwrap();
    let db = DeviceBuffer::from_bytes(&bytes_of(&bh)).unwrap();
    let dh = DeviceBuffer::alloc(m * n * 2).unwrap();
    launch(k.nt_f16, &stream, &da, &db, &dh, m, n, kk, kk, kk, n);
    stream.synchronize().unwrap();
    let mut raw = vec![0u8; m * n * 2];
    dh.download(&mut raw).unwrap();
    let worst = raw
        .chunks_exact(2)
        .zip(&want)
        .map(|(c, w)| (f16_to_f32(u16::from_le_bytes([c[0], c[1]])) - w).abs())
        .fold(0f32, f32::max);
    assert!(worst < 0.05, "fp16 output: {worst}");
}

#[test]
fn gemm_throughput() {
    let Some((_dev, _module, k, stream)) = setup() else {
        return;
    };
    let e0 = Event::new().unwrap();
    let e1 = Event::new().unwrap();
    for &(m, n, kk, what) in &[
        (8192usize, 576usize, 576usize, "8192 tokens, attention-out"),
        (8192, 960, 576, "QKV"),
        (8192, 3072, 576, "gate+up"),
        (8192, 576, 1536, "down"),
        (8192, 8192, 2048, "large"),
        (512, 576, 576, "prefill 512 attention-out"),
        (512, 3072, 576, "prefill 512 gate+up"),
        (512, 576, 1536, "prefill 512 down"),
    ] {
        let (ah, _) = rand_f16(m * kk, 5);
        let (bh, _) = rand_f16(n * kk, 6);
        let da = DeviceBuffer::from_bytes(&bytes_of(&ah)).unwrap();
        let db = DeviceBuffer::from_bytes(&bytes_of(&bh)).unwrap();
        let dc = DeviceBuffer::alloc(m * n * 4).unwrap();
        for (name, f, lda, ldb) in [
            ("NT", k.nt_f32, kk, kk),
            ("NT old16", k.nt_f32_b16, kk, kk),
            ("NT old32", k.nt_f32_b32, kk, kk),
            ("NN", k.nn_f32, kk, n),
            ("NN old16", k.nn_f32_b16, kk, n),
            ("NN old32", k.nn_f32_b32, kk, n),
        ] {
            for _ in 0..3 {
                launch(f, &stream, &da, &db, &dc, m, n, kk, lda, ldb, n);
            }
            stream.synchronize().unwrap();
            let reps = 20;
            e0.record(&stream).unwrap();
            for _ in 0..reps {
                launch(f, &stream, &da, &db, &dc, m, n, kk, lda, ldb, n);
            }
            e1.record(&stream).unwrap();
            e1.synchronize().unwrap();
            let ms = e1.elapsed_ms(&e0).unwrap() as f64 / reps as f64;
            let tf = 2.0 * m as f64 * n as f64 * kk as f64 / (ms * 1e-3) / 1e12;
            eprintln!(
                "GEMM {name} {m}x{n}x{kk} ({what}): {ms:.3} ms = {tf:.1} TFLOPs ({:.0} % of 33.5)",
                100.0 * tf / 33.5
            );
        }
        // weight gradient layout: reduction over m (tokens): C[kk][n]... use (m_out = kk, n_out = n, k_red = m)
        if m % 32 == 0 && kk % 8 == 0 && n % 8 == 0 {
            let dcw = DeviceBuffer::alloc(kk * n * 4).unwrap();
            let (bth, _) = rand_f16(m * n, 9);
            let dbt = DeviceBuffer::from_bytes(&bytes_of(&bth)).unwrap(); // the activation-gradient matrix [tokens][n]
            for (name, f) in [
                ("TN", k.tn_f32),
                ("TN old16", k.tn_f32_b16),
                ("TN old32", k.tn_f32_b32),
            ] {
                for _ in 0..3 {
                    launch(f, &stream, &da, &dbt, &dcw, kk, n, m, kk, n, n);
                }
                stream.synchronize().unwrap();
                let reps = 10;
                e0.record(&stream).unwrap();
                for _ in 0..reps {
                    launch(f, &stream, &da, &dbt, &dcw, kk, n, m, kk, n, n);
                }
                e1.record(&stream).unwrap();
                e1.synchronize().unwrap();
                let ms = e1.elapsed_ms(&e0).unwrap() as f64 / reps as f64;
                let tf = 2.0 * kk as f64 * n as f64 * m as f64 / (ms * 1e-3) / 1e12;
                eprintln!(
                    "GEMM {name} {kk}x{n} reduced over {m} ({what}): {ms:.3} ms = {tf:.1} TFLOPs"
                );
            }
        }
    }
}

/// The prompt-processing shapes of the Qwen3.5 models: M = tokens of a chunk, N = output features, K = input features. Run with --nocapture; the numbers are only meaningful on an idle GPU.
#[test]
fn gemm_inference_throughput() {
    let Some((_dev, _module, k, stream)) = setup() else {
        return;
    };
    let e0 = Event::new().unwrap();
    let e1 = Event::new().unwrap();
    for &(m, n, kk, what) in &[
        (512usize, 9216usize, 2560usize, "4B gate/up"),
        (512, 9216, 2048, "K 2048"),
        (512, 9216, 3072, "K 3072"),
        (512, 4096, 12288, "9B down"),
    ] {
        let (ah, _) = rand_f16(m * kk, 5);
        let (bh, _) = rand_f16(n * kk, 6);
        let da = DeviceBuffer::from_bytes(&bytes_of(&ah)).unwrap();
        let db = DeviceBuffer::from_bytes(&bytes_of(&bh)).unwrap();
        let dc = DeviceBuffer::alloc(m * n * 4).unwrap();
        for (name, f) in [
            ("NT", k.nt_f32),
            ("NT old16", k.nt_f32_b16),
            ("NT old32", k.nt_f32_b32),
        ] {
            for _ in 0..3 {
                launch(f, &stream, &da, &db, &dc, m, n, kk, kk, kk, n);
            }
            stream.synchronize().unwrap();
            let reps = 20;
            e0.record(&stream).unwrap();
            for _ in 0..reps {
                launch(f, &stream, &da, &db, &dc, m, n, kk, kk, kk, n);
            }
            e1.record(&stream).unwrap();
            e1.synchronize().unwrap();
            let ms = e1.elapsed_ms(&e0).unwrap() as f64 / reps as f64;
            let tf = 2.0 * m as f64 * n as f64 * kk as f64 / (ms * 1e-3) / 1e12;
            eprintln!(
                "GEMM {name} {m}x{n}x{kk} ({what}): {ms:.3} ms = {tf:.1} TFLOPs ({:.0} % of 33.5)",
                100.0 * tf / 33.5
            );
        }
    }
}
