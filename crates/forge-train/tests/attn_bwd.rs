//! The tensor-core flash-attention backward against the reference (CUDA-core, untiled) backward on random data with several tiles, GQA and a batch.

use forge_core::half::{f16_to_f32, f32_to_f16};
use forge_cuda::{Args, Device, DeviceBuffer, Module, Stream};

fn rand_half(n: usize, seed: u64, scale: f32) -> Vec<u8> {
    let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .flat_map(|_| {
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            let r = ((x.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 40) as f32 / (1u64 << 24) as f32)
                * 2.0
                - 1.0;
            f32_to_f16(r * scale).to_le_bytes()
        })
        .collect()
}

fn stats(a: &[f32], b: &[f32]) -> (f64, f64) {
    let (mut dd, mut bb) = (0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        dd += ((x - y) as f64).powi(2);
        bb += (*y as f64).powi(2);
    }
    (
        (dd / bb.max(1e-30)).sqrt(),
        a.iter()
            .zip(b)
            .map(|(x, y)| (*x as f64) * (*y as f64))
            .sum::<f64>()
            / ((a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>() * bb).sqrt()).max(1e-30),
    )
}

#[test]
fn flash_backward_matches_the_reference_backward() {
    if Device::count().unwrap_or(0) == 0 {
        eprintln!("skipped: no CUDA device");
        return;
    }
    let dev = Device::open(0).unwrap();
    let (image, _) = forge_kernels::image_for(dev.info.cc_major, dev.info.cc_minor).unwrap();
    let module = Module::load(image).unwrap();
    let stream = Stream::new().unwrap();
    let (b, t, nh, nkv, hd) = (2usize, 192usize, 6usize, 2usize, 64usize);
    let (qd, kvd) = (nh * hd, nkv * hd);
    let n = b * t;
    let q = DeviceBuffer::from_bytes(&rand_half(n * qd, 1, 1.0)).unwrap();
    let k = DeviceBuffer::from_bytes(&rand_half(n * kvd, 2, 1.0)).unwrap();
    let v = DeviceBuffer::from_bytes(&rand_half(n * kvd, 3, 1.0)).unwrap();
    let dout = DeviceBuffer::from_bytes(&rand_half(n * qd, 4, 0.5)).unwrap();
    let o = DeviceBuffer::alloc(n * qd * 2).unwrap();
    let lse = DeviceBuffer::alloc(n * nh * 4).unwrap();
    let dvec = DeviceBuffer::alloc(n * nh * 4).unwrap();
    let fwd = module.function("flash_fwd_train_64").unwrap();
    fwd.launch(
        ((t / 64) as u32, nh as u32, b as u32),
        (128, 1, 1),
        0,
        &stream,
        &Args::new()
            .buf(&q)
            .buf(&k)
            .buf(&v)
            .buf(&o)
            .buf(&lse)
            .i32(t as i32)
            .i32(nh as i32)
            .i32(nkv as i32),
    )
    .unwrap();
    module
        .function("attn_bwd_prep_kernel")
        .unwrap()
        .launch(
            (((n * nh + 7) / 8) as u32, 1, 1),
            (256, 1, 1),
            0,
            &stream,
            &Args::new()
                .buf(&o)
                .buf(&dout)
                .buf(&dvec)
                .i32(n as i32)
                .i32(nh as i32)
                .i32(hd as i32),
        )
        .unwrap();
    let (dq_ref, dk_ref, dv_ref) = (
        DeviceBuffer::alloc(n * qd * 4).unwrap(),
        DeviceBuffer::alloc(n * kvd * 4).unwrap(),
        DeviceBuffer::alloc(n * kvd * 4).unwrap(),
    );
    module
        .function("attn_bwd_dkdv_naive_kernel")
        .unwrap()
        .launch(
            (((b * t * nkv + 127) / 128) as u32, 1, 1),
            (128, 1, 1),
            0,
            &stream,
            &Args::new()
                .buf(&q)
                .buf(&k)
                .buf(&v)
                .buf(&dout)
                .buf(&lse)
                .buf(&dvec)
                .buf(&dk_ref)
                .buf(&dv_ref)
                .i32(b as i32)
                .i32(t as i32)
                .i32(nh as i32)
                .i32(nkv as i32)
                .i32(hd as i32),
        )
        .unwrap();
    module
        .function("attn_bwd_dq_naive_kernel")
        .unwrap()
        .launch(
            (((b * t * nh + 127) / 128) as u32, 1, 1),
            (128, 1, 1),
            0,
            &stream,
            &Args::new()
                .buf(&q)
                .buf(&k)
                .buf(&v)
                .buf(&dout)
                .buf(&lse)
                .buf(&dvec)
                .buf(&dq_ref)
                .i32(b as i32)
                .i32(t as i32)
                .i32(nh as i32)
                .i32(nkv as i32)
                .i32(hd as i32),
        )
        .unwrap();
    let (dq, dk, dv) = (
        DeviceBuffer::alloc(n * qd * 4).unwrap(),
        DeviceBuffer::alloc(n * kvd * 4).unwrap(),
        DeviceBuffer::alloc(n * kvd * 4).unwrap(),
    );
    module
        .function("flash_bwd_dkdv_64")
        .unwrap()
        .launch(
            ((t / 64) as u32, nkv as u32, b as u32),
            (128, 1, 1),
            0,
            &stream,
            &Args::new()
                .buf(&q)
                .buf(&k)
                .buf(&v)
                .buf(&dout)
                .buf(&lse)
                .buf(&dvec)
                .buf(&dk)
                .buf(&dv)
                .i32(t as i32)
                .i32(nh as i32)
                .i32(nkv as i32),
        )
        .unwrap();
    module
        .function("flash_bwd_dq_64")
        .unwrap()
        .launch(
            ((t / 64) as u32, nh as u32, b as u32),
            (128, 1, 1),
            0,
            &stream,
            &Args::new()
                .buf(&q)
                .buf(&k)
                .buf(&v)
                .buf(&dout)
                .buf(&lse)
                .buf(&dvec)
                .buf(&dq)
                .i32(t as i32)
                .i32(nh as i32)
                .i32(nkv as i32),
        )
        .unwrap();
    stream.synchronize().unwrap();
    for (name, a, r, len) in [
        ("dq", &dq, &dq_ref, n * qd),
        ("dk", &dk, &dk_ref, n * kvd),
        ("dv", &dv, &dv_ref, n * kvd),
    ] {
        let (got, want) = (a.download_f32(len).unwrap(), r.download_f32(len).unwrap());
        let (rel, cos) = stats(&got, &want);
        let mag = want.iter().fold(0f32, |m, v| m.max(v.abs()));
        eprintln!("{name}: relative error {rel:.3e}, cosine {cos:.6} (max |value| {mag:.3})");
        assert!(
            cos > 0.9995 && rel < 0.03,
            "{name}: relative {rel}, cosine {cos}"
        );
    }
    let _ = f16_to_f32(0);
}
