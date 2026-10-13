//! The Ampere GEMM kernels (`*_s80`: cp.async pipeline, mma.m16n8k16, gemm_s80.cu) against a CPU reference in all three layouts and the epilogues the trainer uses, on awkward shapes, and their throughput next to the
//! Turing kernels. Below sm_80 the kernels run with an emulation of the two new instructions: the test then checks the indexing, swizzle and pipeline logic (and the speed numbers mean nothing).

use forge_core::half::{f16_to_f32, f32_to_f16};
use forge_cuda::{Args, Device, DeviceBuffer, Event, Function, Module, Stream};
use forge_model::sample::Rng;

const SMEM: u32 = 3 * 2 * 32 * 136 * 2;

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

fn setup() -> Option<(Device, Module, Stream)> {
    if Device::count().unwrap_or(0) == 0 {
        eprintln!("skipped: no CUDA device");
        return None;
    }
    let dev = Device::open(0).unwrap();
    let (image, _) = forge_kernels::image_for(dev.info.cc_major, dev.info.cc_minor).unwrap();
    let module = Module::load(image).unwrap();
    eprintln!(
        "{} (sm_{}{}), real Ampere instructions: {}",
        dev.info.name,
        dev.info.cc_major,
        dev.info.cc_minor,
        forge_kernels::has_s80(dev.info.cc_major)
    );
    Some((dev, module, Stream::new().unwrap()))
}

fn s80(module: &Module, name: &str) -> Function {
    let f = module.function(&format!("{name}_s80")).unwrap();
    f.set_max_dynamic_smem(SMEM as i32).unwrap();
    f
}

#[allow(clippy::too_many_arguments)]
fn launch(
    f: Function,
    smem: u32,
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
    f.launch(grid, (256, 1, 1), smem, stream, &args).unwrap();
}

#[test]
fn ampere_gemms_match_the_cpu_reference() {
    let Some((_dev, module, stream)) = setup() else {
        return;
    };
    // (m, n, k): not multiples of the tile; the k-major operands need m (resp. n) to be multiples of 8, K a multiple of 32
    for &(m, n, kk) in &[
        (130usize, 200usize, 96usize),
        (8, 64, 32),
        (256, 576, 576),
        (512, 1536, 576),
        (72, 960, 1536),
        (1024, 576, 2048),
        (200, 168, 64),
    ] {
        for &(a_col, b_col, f32_name, add_name, f16_name) in &[
            (false, false, "gemm_nt_f32", "gemm_nt_add", "gemm_nt_f16"),
            (false, true, "gemm_nn_f32", "gemm_nn_add", "gemm_nn_f16"),
            (true, true, "gemm_tn_f32", "gemm_tn_add", "gemm_tn_f16"),
        ] {
            if (a_col && m % 8 != 0) || (b_col && n % 8 != 0) {
                continue;
            }
            let (ah, af) = rand_f16(m * kk, 1 + m as u64);
            let (bh, bf) = rand_f16(n * kk, 2 + n as u64);
            let want = cpu_gemm(&af, a_col, &bf, b_col, m, n, kk);
            let scale = want.iter().fold(0f32, |mx, v| mx.max(v.abs())).max(1.0);
            let da = DeviceBuffer::from_bytes(&bytes_of(&ah)).unwrap();
            let db = DeviceBuffer::from_bytes(&bytes_of(&bh)).unwrap();
            let dc = DeviceBuffer::alloc(m * n * 4).unwrap();
            let dh = DeviceBuffer::alloc(m * n * 2).unwrap();
            let (lda, ldb) = (if a_col { m } else { kk }, if b_col { n } else { kk });
            // fp32 store
            launch(
                s80(&module, f32_name),
                SMEM,
                &stream,
                &da,
                &db,
                &dc,
                m,
                n,
                kk,
                lda,
                ldb,
                n,
            );
            stream.synchronize().unwrap();
            let got = dc.download_f32(m * n).unwrap();
            let worst = got
                .iter()
                .zip(&want)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            eprintln!(
                "{f32_name}_s80 {m}x{n}x{kk}: max |diff| {worst:.2e} (max |value| {scale:.1})"
            );
            assert!(worst < 2e-3 * scale, "{f32_name} {m}x{n}x{kk}: {worst}");
            // accumulate into C (starts at 1)
            dc.upload_f32(&vec![1f32; m * n]).unwrap();
            launch(
                s80(&module, add_name),
                SMEM,
                &stream,
                &da,
                &db,
                &dc,
                m,
                n,
                kk,
                lda,
                ldb,
                n,
            );
            stream.synchronize().unwrap();
            let got = dc.download_f32(m * n).unwrap();
            let worst = got
                .iter()
                .zip(&want)
                .map(|(a, b)| (a - 1.0 - b).abs())
                .fold(0f32, f32::max);
            assert!(worst < 2e-3 * scale, "{add_name} {m}x{n}x{kk}: {worst}");
            // fp16 store
            launch(
                s80(&module, f16_name),
                SMEM,
                &stream,
                &da,
                &db,
                &dh,
                m,
                n,
                kk,
                lda,
                ldb,
                n,
            );
            stream.synchronize().unwrap();
            let mut raw = vec![0u8; m * n * 2];
            dh.download(&mut raw).unwrap();
            let worst = raw
                .chunks_exact(2)
                .zip(&want)
                .map(|(c, w)| (f16_to_f32(u16::from_le_bytes([c[0], c[1]])) - w).abs())
                .fold(0f32, f32::max);
            assert!(
                worst < 2e-3 * scale + 0.01,
                "{f16_name} {m}x{n}x{kk}: {worst}"
            );
        }
    }
}

#[test]
fn ampere_split_k_weight_gradient() {
    let Some((_dev, module, stream)) = setup() else {
        return;
    };
    // C += A^T B, the reduction cut into 2, 4 and 8 slices that add with atomics (C starts at 1)
    for &(m, n, kk) in &[
        (200usize, 168usize, 512usize),
        (576, 576, 2048),
        (960, 576, 4096),
    ] {
        let (ah, af) = rand_f16(m * kk, 21);
        let (bh, bf) = rand_f16(n * kk, 22);
        let want = cpu_gemm(&af, true, &bf, true, m, n, kk);
        let scale = want.iter().fold(0f32, |mx, v| mx.max(v.abs())).max(1.0);
        let da = DeviceBuffer::from_bytes(&bytes_of(&ah)).unwrap();
        let db = DeviceBuffer::from_bytes(&bytes_of(&bh)).unwrap();
        let dc = DeviceBuffer::alloc(m * n * 4).unwrap();
        let f = s80(&module, "gemm_tn_add");
        for gz in [1u32, 2, 4, 8] {
            if kk % (32 * gz as usize) != 0 {
                continue;
            }
            dc.upload_f32(&vec![1f32; m * n]).unwrap();
            let grid = (((n + 127) / 128) as u32, ((m + 127) / 128) as u32, gz);
            let args = Args::new()
                .buf(&da)
                .buf(&db)
                .buf(&dc)
                .i32(m as i32)
                .i32(n as i32)
                .i32(kk as i32)
                .i32(m as i32)
                .i32(n as i32)
                .i32(n as i32)
                .f32(1.0);
            f.launch(grid, (256, 1, 1), SMEM, &stream, &args).unwrap();
            stream.synchronize().unwrap();
            let got = dc.download_f32(m * n).unwrap();
            let worst = got
                .iter()
                .zip(&want)
                .map(|(a, b)| (a - 1.0 - b).abs())
                .fold(0f32, f32::max);
            eprintln!("tn_add {m}x{n}x{kk} in {gz} slices: max |diff| {worst:.2e}");
            assert!(
                worst < 2e-3 * scale,
                "split-K {gz} slices {m}x{n}x{kk}: {worst}"
            );
        }
    }
}

#[test]
fn ampere_residual_epilogue() {
    let Some((_dev, module, stream)) = setup() else {
        return;
    };
    // C = A B^T + R with an fp32 residual R that aliases C (the transformer's residual stream)
    let (m, n, kk) = (200usize, 192usize, 128usize);
    let (ah, af) = rand_f16(m * kk, 11);
    let (bh, bf) = rand_f16(n * kk, 12);
    let want = cpu_gemm(&af, false, &bf, false, m, n, kk);
    let r0: Vec<f32> = (0..m * n).map(|i| (i % 7) as f32 * 0.25).collect();
    let da = DeviceBuffer::from_bytes(&bytes_of(&ah)).unwrap();
    let db = DeviceBuffer::from_bytes(&bytes_of(&bh)).unwrap();
    let dc = DeviceBuffer::alloc(m * n * 4).unwrap();
    dc.upload_f32(&r0).unwrap();
    let f = s80(&module, "gemm_nt_res");
    let grid = (((n + 127) / 128) as u32, ((m + 127) / 128) as u32, 1);
    let args = Args::new()
        .buf(&da)
        .buf(&db)
        .buf(&dc)
        .i32(m as i32)
        .i32(n as i32)
        .i32(kk as i32)
        .i32(kk as i32)
        .i32(kk as i32)
        .i32(n as i32)
        .f32(1.0)
        .buf(&dc)
        .i32(n as i32)
        .f32(0.0);
    f.launch(grid, (256, 1, 1), SMEM, &stream, &args).unwrap();
    stream.synchronize().unwrap();
    let got = dc.download_f32(m * n).unwrap();
    let worst = got
        .iter()
        .zip(want.iter().zip(&r0))
        .map(|(g, (w, r))| (g - w - r).abs())
        .fold(0f32, f32::max);
    assert!(worst < 1e-2, "residual epilogue: {worst}");
}

#[test]
fn ampere_gemm_throughput() {
    let Some((dev, module, stream)) = setup() else {
        return;
    };
    if !forge_kernels::has_s80(dev.info.cc_major) {
        eprintln!("throughput is only meaningful with the real Ampere instructions: skipped");
        return;
    }
    let (e0, e1) = (Event::new().unwrap(), Event::new().unwrap());
    // the shapes of the 125M training step with 16,384 tokens per micro-batch (NT forward, NN data gradient, TN weight gradient) and a big square one
    for &(m, n, kk, what) in &[
        (16384usize, 960usize, 576usize, "QKV forward"),
        (16384, 3072, 576, "gate+up forward"),
        (16384, 576, 1536, "down forward"),
        (16384, 32768, 576, "head forward"),
        (8192, 8192, 2048, "square"),
    ] {
        let (ah, _) = rand_f16(m * kk, 5);
        let (bh, _) = rand_f16(n * kk, 6);
        let da = DeviceBuffer::from_bytes(&bytes_of(&ah)).unwrap();
        let db = DeviceBuffer::from_bytes(&bytes_of(&bh)).unwrap();
        let dc = DeviceBuffer::alloc(m * n * 4).unwrap();
        for (name, f, smem) in [
            ("old", module.function("gemm_nt_f32").unwrap(), 0u32),
            ("s80", s80(&module, "gemm_nt_f32"), SMEM),
        ] {
            for _ in 0..2 {
                launch(f, smem, &stream, &da, &db, &dc, m, n, kk, kk, kk, n);
            }
            stream.synchronize().unwrap();
            let reps = 8;
            e0.record(&stream).unwrap();
            for _ in 0..reps {
                launch(f, smem, &stream, &da, &db, &dc, m, n, kk, kk, kk, n);
            }
            e1.record(&stream).unwrap();
            e1.synchronize().unwrap();
            let ms = e1.elapsed_ms(&e0).unwrap() as f64 / reps as f64;
            let tf = 2.0 * m as f64 * n as f64 * kk as f64 / (ms * 1e-3) / 1e12;
            eprintln!("NT {name:3} {m}x{n}x{kk} ({what}): {ms:.3} ms = {tf:.1} TFLOPs");
        }
    }
    // weight gradient (TN): C[576][960] += A^T B with the reduction over 16,384 tokens
    for &(m, n, kk, what) in &[
        (960usize, 576usize, 16384usize, "QKV weight gradient"),
        (3072, 576, 16384, "gate+up weight gradient"),
        (576, 1536, 16384, "down weight gradient"),
    ] {
        let (ah, _) = rand_f16(m * kk, 7);
        let (bh, _) = rand_f16(n * kk, 8);
        let da = DeviceBuffer::from_bytes(&bytes_of(&ah)).unwrap();
        let db = DeviceBuffer::from_bytes(&bytes_of(&bh)).unwrap();
        let dc = DeviceBuffer::alloc(m * n * 4).unwrap();
        for (name, f, smem) in [
            ("old", module.function("gemm_tn_add").unwrap(), 0u32),
            ("s80", s80(&module, "gemm_tn_add"), SMEM),
        ] {
            for _ in 0..2 {
                launch(f, smem, &stream, &da, &db, &dc, m, n, kk, m, n, n);
            }
            stream.synchronize().unwrap();
            let reps = 8;
            e0.record(&stream).unwrap();
            for _ in 0..reps {
                launch(f, smem, &stream, &da, &db, &dc, m, n, kk, m, n, n);
            }
            e1.record(&stream).unwrap();
            e1.synchronize().unwrap();
            let ms = e1.elapsed_ms(&e0).unwrap() as f64 / reps as f64;
            let tf = 2.0 * m as f64 * n as f64 * kk as f64 / (ms * 1e-3) / 1e12;
            eprintln!("TN {name:3} {m}x{n}x{kk} ({what}): {ms:.3} ms = {tf:.1} TFLOPs");
        }
    }
}

/// The fp16-accumulate NT kernels (`*_s80h`: the MMAs add 16 products in fp16, flushed into the fp32 accumulators after every k16 step; sm_80 and newer, below that the kernels run the fp32 form) against a CPU
/// reference: the error stays far below an fp16 rounding step of the output for K up to 2,048, and the epilogues (fp16 out, fp32 out, add into fp32) write the right values.
#[test]
fn fp16_accumulate_gemms_match_the_cpu_reference() {
    let Some((_dev, module, stream)) = setup() else {
        return;
    };
    let h = |name: &str| {
        let f = module.function(&format!("{name}_s80h")).unwrap();
        f.set_max_dynamic_smem(SMEM as i32).unwrap();
        f
    };
    for &(m, n, kk) in &[(130usize, 200usize, 96usize), (8, 64, 32), (256, 576, 576), (1024, 576, 2048), (200, 168, 64), (77, 4352, 1024)] {
        let (ah, af) = rand_f16(m * kk, 11 + m as u64);
        let (bh, bf) = rand_f16(n * kk, 12 + n as u64);
        let want = cpu_gemm(&af, false, &bf, false, m, n, kk);
        let scale = want.iter().fold(0f32, |mx, v| mx.max(v.abs())).max(1.0);
        let da = DeviceBuffer::from_bytes(&bytes_of(&ah)).unwrap();
        let db = DeviceBuffer::from_bytes(&bytes_of(&bh)).unwrap();
        // fp32 out
        let dc = DeviceBuffer::alloc(m * n * 4).unwrap();
        launch(h("gemm_nt_f32"), SMEM, &stream, &da, &db, &dc, m, n, kk, kk, kk, n);
        stream.synchronize().unwrap();
        let got = dc.download_f32(m * n).unwrap();
        let worst = got.iter().zip(&want).fold(0f32, |mx, (g, w)| mx.max((g - w).abs()));
        eprintln!("gemm_nt_f32_s80h {m}x{n}x{kk}: max |diff| {worst:.2e} (max |value| {scale:.1})");
        assert!(worst <= 4e-3 * scale, "gemm_nt_f32_s80h {m}x{n}x{kk}: {worst} against {scale}");
        // add: C starts as the reference and receives the product again
        let dadd = DeviceBuffer::from_bytes(&want.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>()).unwrap();
        launch(h("gemm_nt_add"), SMEM, &stream, &da, &db, &dadd, m, n, kk, kk, kk, n);
        stream.synchronize().unwrap();
        let got = dadd.download_f32(m * n).unwrap();
        let worst = got.iter().zip(&want).fold(0f32, |mx, (g, w)| mx.max((g - 2.0 * w).abs()));
        assert!(worst <= 4e-3 * scale, "gemm_nt_add_s80h {m}x{n}x{kk}: {worst} against {scale}");
        // fp16 out
        let dh = DeviceBuffer::alloc(m * n * 2).unwrap();
        launch(h("gemm_nt_f16"), SMEM, &stream, &da, &db, &dh, m, n, kk, kk, kk, n);
        stream.synchronize().unwrap();
        let mut raw = vec![0u8; m * n * 2];
        dh.download(&mut raw).unwrap();
        let worst = raw.chunks_exact(2).zip(&want).fold(0f32, |mx, (b, w)| mx.max((f16_to_f32(u16::from_le_bytes([b[0], b[1]])) - w).abs()));
        assert!(worst <= 4e-3 * scale + 1e-3 * scale, "gemm_nt_f16_s80h {m}x{n}x{kk}: {worst} against {scale}");
    }
}
