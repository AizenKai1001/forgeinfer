//! Feasibility spike (throwaway): AVX2 + FMA in stable Rust. Same two loops as bench/cpu_micro.c (fp32 FMA peak, int8 maddubs/madd chain) plus a streaming f32 dot product, one thread.
//! The C results on the homelab were 146.6 GFLOPs (fp32) and 710 GOPS (int8).
use std::arch::x86_64::*;
use std::hint::black_box;
use std::time::Instant;

#[target_feature(enable = "avx2,fma")]
unsafe fn fma_peak(iters: u64) -> f32 {
    let b = _mm256_set1_ps(1.000_000_1);
    let c = _mm256_set1_ps(1e-7);
    let mut a0 = _mm256_set1_ps(1.0);
    let mut a1 = _mm256_set1_ps(2.0);
    let mut a2 = _mm256_set1_ps(3.0);
    let mut a3 = _mm256_set1_ps(4.0);
    let mut a4 = _mm256_set1_ps(5.0);
    let mut a5 = _mm256_set1_ps(6.0);
    let mut a6 = _mm256_set1_ps(7.0);
    let mut a7 = _mm256_set1_ps(8.0);
    let mut a8 = _mm256_set1_ps(9.0);
    let mut a9 = _mm256_set1_ps(10.0);
    for _ in 0..iters {
        a0 = _mm256_fmadd_ps(a0, b, c);
        a1 = _mm256_fmadd_ps(a1, b, c);
        a2 = _mm256_fmadd_ps(a2, b, c);
        a3 = _mm256_fmadd_ps(a3, b, c);
        a4 = _mm256_fmadd_ps(a4, b, c);
        a5 = _mm256_fmadd_ps(a5, b, c);
        a6 = _mm256_fmadd_ps(a6, b, c);
        a7 = _mm256_fmadd_ps(a7, b, c);
        a8 = _mm256_fmadd_ps(a8, b, c);
        a9 = _mm256_fmadd_ps(a9, b, c);
    }
    let s = _mm256_add_ps(_mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3)), _mm256_add_ps(_mm256_add_ps(a4, a5), _mm256_add_ps(_mm256_add_ps(a6, a7), _mm256_add_ps(a8, a9))));
    let mut out = [0f32; 8];
    _mm256_storeu_ps(out.as_mut_ptr(), s);
    out[0] + out[7]
}

/// Unsigned-by-signed int8 dot product over L1-resident buffers (the arithmetic limit of a quantized dot kernel without VNNI): maddubs -> madd(1) -> add, 4 independent accumulators.
/// `w` holds unsigned 0..15 values (like Q4 nibbles), `x` signed values; both are 8 KiB (256 vectors of 32 bytes). Returns the checksum and the MACs done per call.
#[target_feature(enable = "avx2")]
unsafe fn int8_dot(w: &[u8], x: &[i8], reps: u64) -> i32 {
    let ones = _mm256_set1_epi16(1);
    let mut a0 = _mm256_setzero_si256();
    let mut a1 = a0;
    let mut a2 = a0;
    let mut a3 = a0;
    for _ in 0..reps {
        let mut i = 0usize;
        while i < 256 {
            let w0 = _mm256_loadu_si256(w.as_ptr().add(i * 32) as *const __m256i);
            let w1 = _mm256_loadu_si256(w.as_ptr().add(i * 32 + 32) as *const __m256i);
            let w2 = _mm256_loadu_si256(w.as_ptr().add(i * 32 + 64) as *const __m256i);
            let w3 = _mm256_loadu_si256(w.as_ptr().add(i * 32 + 96) as *const __m256i);
            let x0 = _mm256_loadu_si256(x.as_ptr().add(i * 32) as *const __m256i);
            let x1 = _mm256_loadu_si256(x.as_ptr().add(i * 32 + 32) as *const __m256i);
            let x2 = _mm256_loadu_si256(x.as_ptr().add(i * 32 + 64) as *const __m256i);
            let x3 = _mm256_loadu_si256(x.as_ptr().add(i * 32 + 96) as *const __m256i);
            a0 = _mm256_add_epi32(a0, _mm256_madd_epi16(_mm256_maddubs_epi16(w0, x0), ones));
            a1 = _mm256_add_epi32(a1, _mm256_madd_epi16(_mm256_maddubs_epi16(w1, x1), ones));
            a2 = _mm256_add_epi32(a2, _mm256_madd_epi16(_mm256_maddubs_epi16(w2, x2), ones));
            a3 = _mm256_add_epi32(a3, _mm256_madd_epi16(_mm256_maddubs_epi16(w3, x3), ones));
            i += 4;
        }
    }
    let s = _mm256_add_epi32(_mm256_add_epi32(a0, a1), _mm256_add_epi32(a2, a3));
    let mut out = [0i32; 8];
    _mm256_storeu_si256(out.as_mut_ptr() as *mut __m256i, s);
    out.iter().sum()
}

#[target_feature(enable = "avx2,fma")]
unsafe fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len();
    let mut s0 = _mm256_setzero_ps();
    let mut s1 = _mm256_setzero_ps();
    let mut s2 = _mm256_setzero_ps();
    let mut s3 = _mm256_setzero_ps();
    let mut i = 0;
    while i + 32 <= n {
        s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a.as_ptr().add(i)), _mm256_loadu_ps(b.as_ptr().add(i)), s0);
        s1 = _mm256_fmadd_ps(_mm256_loadu_ps(a.as_ptr().add(i + 8)), _mm256_loadu_ps(b.as_ptr().add(i + 8)), s1);
        s2 = _mm256_fmadd_ps(_mm256_loadu_ps(a.as_ptr().add(i + 16)), _mm256_loadu_ps(b.as_ptr().add(i + 16)), s2);
        s3 = _mm256_fmadd_ps(_mm256_loadu_ps(a.as_ptr().add(i + 24)), _mm256_loadu_ps(b.as_ptr().add(i + 24)), s3);
        i += 32;
    }
    let s = _mm256_add_ps(_mm256_add_ps(s0, s1), _mm256_add_ps(s2, s3));
    let mut out = [0f32; 8];
    _mm256_storeu_ps(out.as_mut_ptr(), s);
    out.iter().sum()
}

fn main() {
    assert!(is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma"), "needs AVX2 + FMA");
    let iters = 100_000_000u64;
    let t = Instant::now();
    let r = unsafe { fma_peak(black_box(iters)) };
    black_box(r);
    let dt = t.elapsed().as_secs_f64();
    println!("fp32 FMA (Rust, 1 thread): {:.1} GFLOPs   (C: 146.6)", 2.0 * 8.0 * 10.0 * iters as f64 / dt / 1e9);
    let w: Vec<u8> = (0..8192usize).map(|i| ((i * 7 + 3) % 16) as u8).collect();
    let x: Vec<i8> = (0..8192usize).map(|i| (((i * 13 + 5) % 255) as i32 - 127) as i8).collect();
    let reps = 300_000u64;
    let t = Instant::now();
    let r = unsafe { int8_dot(black_box(&w), black_box(&x), black_box(reps)) };
    black_box(r);
    let dt = t.elapsed().as_secs_f64();
    println!("int8 dot, L1-resident (Rust, 1 thread): {:.1} GOPS = {:.1} G int8 MACs/s   (the earlier chain benchmark was invalid: the compiler hoisted its invariant part)", 2.0 * 8192.0 * reps as f64 / dt / 1e9, 8192.0 * reps as f64 / dt / 1e9);
    // streaming dot product over 256 MiB each (DRAM bound): read bandwidth of one thread
    let n = 64usize << 20;
    let a = vec![1.0f32; n];
    let b = vec![2.0f32; n];
    let mut best = f64::MAX;
    let mut sink = 0.0;
    for _ in 0..5 {
        let t = Instant::now();
        sink += unsafe { dot_f32(black_box(&a), black_box(&b)) };
        best = best.min(t.elapsed().as_secs_f64());
    }
    black_box(sink);
    println!("f32 dot over 2 x 256 MiB (Rust, 1 thread): {:.1} GB/s read   (C single-thread read: 21.2)", 2.0 * n as f64 * 4.0 / best / 1e9);
}
