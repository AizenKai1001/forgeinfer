//! The six quantized weight formats of the decode kernels (Q8_0, Q4_0, Q5_0, Q4_K, Q5_K, Q6_K) against the executable specification tools/ref_qdot.py: the planar repacking must dequantize to the same
//! values as the native blocks, and the matvec kernel on a pre-quantized activation must reproduce the specification's integer pipeline (tests/golden/qdot, N = 64 rows, K = 1536).

use forge_cuda::{Args, Device, DeviceBuffer, Function, Module, Stream};
use forge_gguf::dequantize;
use forge_gpu::{repack, QFmt};
use std::path::PathBuf;

const ROWS: usize = 64;
const K: usize = 1536;

fn golden(name: &str) -> Vec<u8> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/golden/qdot")
        .join(name);
    std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

fn f32s(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn rel_rms(a: &[f32], b: &[f32]) -> f64 {
    let num: f64 = a.iter().zip(b).map(|(x, y)| ((x - y) as f64).powi(2)).sum();
    let den: f64 = b.iter().map(|y| (*y as f64).powi(2)).sum();
    (num / den).sqrt()
}

fn planes_args(a: Args, planes: &[DeviceBuffer]) -> Args {
    let mut a = a;
    for i in 0..4 {
        a = a.ptr(planes.get(i).map(|b| b.ptr()).unwrap_or(0));
    }
    a
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

#[test]
fn repacked_planes_dequantize_like_the_native_blocks() {
    let Some(c) = setup() else { return };
    for qf in [
        QFmt::Q8_0,
        QFmt::Q4_0,
        QFmt::Q5_0,
        QFmt::Q4_K,
        QFmt::Q5_K,
        QFmt::Q6_K,
    ] {
        let native = golden(&format!("{}.w", qf.name()));
        let want = dequantize(qf.dtype(), &native, ROWS * K).unwrap();
        let planes: Vec<DeviceBuffer> = repack(qf, &native)
            .unwrap()
            .iter()
            .map(|p| DeviceBuffer::from_bytes(p).unwrap())
            .collect();
        let out = DeviceBuffer::alloc(ROWS * K * 4).unwrap();
        let a = planes_args(Args::new(), &planes)
            .i32(qf as i32)
            .buf(&out)
            .i32(ROWS as i32)
            .i32(K as i32);
        func(&c, "dequant_test_kernel")
            .launch((64, 1, 1), (256, 1, 1), 0, &c.stream, &a)
            .unwrap();
        c.stream.synchronize().unwrap();
        let got = out.download_f32(ROWS * K).unwrap();
        let worst = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        eprintln!("{}: repacked planes dequantize to the native values, max |diff| {worst:.2e} ({} + {} bytes)", qf.name(), native.len(), 0);
        assert!(
            worst <= 1e-6 * want.iter().fold(0f32, |m, v| m.max(v.abs())).max(1.0),
            "{}: {worst}",
            qf.name()
        );
    }
}

#[test]
fn matvec_on_a_prequantized_activation_matches_the_specification() {
    let Some(c) = setup() else { return };
    let x = f32s(&golden("x.f32"));
    assert_eq!(x.len(), K);
    // quantize the activation on the GPU (llama.cpp Q8_1 rule, scale rounded to fp16, exact integer sums)
    let xd_in =
        DeviceBuffer::from_bytes(&x.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>())
            .unwrap();
    let (xq, xd, xs) = (
        DeviceBuffer::alloc(K).unwrap(),
        DeviceBuffer::alloc(K / 32 * 4).unwrap(),
        DeviceBuffer::alloc(K / 32 * 4).unwrap(),
    );
    func(&c, "quant_kernel_4")
        .launch(
            (1, 1, 1),
            (512, 1, 1),
            0,
            &c.stream,
            &Args::new()
                .buf(&xd_in)
                .buf(&xq)
                .buf(&xd)
                .buf(&xs)
                .i32(K as i32),
        )
        .unwrap();
    c.stream.synchronize().unwrap();
    for qf in [
        QFmt::Q8_0,
        QFmt::Q4_0,
        QFmt::Q5_0,
        QFmt::Q4_K,
        QFmt::Q5_K,
        QFmt::Q6_K,
    ] {
        let native = golden(&format!("{}.w", qf.name()));
        let planes: Vec<DeviceBuffer> = repack(qf, &native)
            .unwrap()
            .iter()
            .map(|p| DeviceBuffer::from_bytes(p).unwrap())
            .collect();
        let y = DeviceBuffer::alloc(ROWS * 4).unwrap();
        y.zero(&c.stream).unwrap();
        let a = planes_args(Args::new(), &planes)
            .i32(qf as i32)
            .buf(&xq)
            .buf(&xd)
            .buf(&xs)
            .buf(&y)
            .i32(ROWS as i32)
            .i32(K as i32);
        func(
            &c,
            &format!(
                "mvq_res_{}_kernel_3",
                ["q8", "q40", "q50", "q4k", "q5k", "q6k"][qf as usize]
            ),
        )
        .launch(((ROWS / 8) as u32, 1, 1), (256, 1, 1), 0, &c.stream, &a)
        .unwrap();
        c.stream.synchronize().unwrap();
        let got = y.download_f32(ROWS).unwrap();
        let spec = f32s(&golden(&format!("{}.y", qf.name())));
        let y64: Vec<f32> = golden(&format!("{}.y64", qf.name()))
            .chunks_exact(8)
            .map(|b| f64::from_le_bytes(b.try_into().unwrap()) as f32)
            .collect();
        let vs_spec = rel_rms(&got, &spec);
        let vs_float = rel_rms(&got, &y64);
        eprintln!("{}: kernel vs specification (integer pipeline) {vs_spec:.2e} relative rms; vs the float product of the dequantized weights and the unquantized x {vs_float:.2e}", qf.name());
        assert!(vs_spec < 2e-5, "{}: {vs_spec}", qf.name());
        assert!(vs_float < 3e-2, "{}: {vs_float}", qf.name());
    }
}
