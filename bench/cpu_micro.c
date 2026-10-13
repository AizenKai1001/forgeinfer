// ForgeInfer CPU microbenchmarks for the homelab (i7-9700K: 8 cores, AVX2 + FMA, no AVX-512, no VNNI). Measures what the CPU really does so the CPU engine is designed against measured limits:
//   DRAM streaming (triad, read) with 1..8 threads, AVX2 FMA peak, and the int8 multiply-add chain (maddubs + madd) that quantized dot products are built from.
// Build: gcc -O3 -mavx2 -mfma -fopenmp -o cpu_micro bench/cpu_micro.c      Run: nice -n 15 ./cpu_micro
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <immintrin.h>
#include <omp.h>

static double now(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec + ts.tv_nsec * 1e-9; }
static volatile float sink_f;
static volatile int sink_i;

static double triad(float *a, const float *b, const float *c, size_t n, int threads) {
    double best = 1e9;
    for (int r = 0; r < 6; r++) {
        double t0 = now();
        #pragma omp parallel for num_threads(threads) schedule(static)
        for (size_t i = 0; i < n; i += 8) {
            __m256 vb = _mm256_loadu_ps(b + i), vc = _mm256_loadu_ps(c + i);
            _mm256_storeu_ps(a + i, _mm256_fmadd_ps(_mm256_set1_ps(3.0f), vc, vb));
        }
        double t = now() - t0;
        if (t < best) best = t;
    }
    return 3.0 * n * 4 / best / 1e9;                         // STREAM convention: two reads + one write
}

static double readbw(const float *b, size_t n, int threads) {
    double best = 1e9;
    for (int r = 0; r < 6; r++) {
        double t0 = now();
        float total = 0;
        #pragma omp parallel num_threads(threads) reduction(+ : total)
        {
            __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps(), s2 = _mm256_setzero_ps(), s3 = _mm256_setzero_ps();
            #pragma omp for schedule(static) nowait
            for (size_t i = 0; i < n; i += 32) {
                s0 = _mm256_add_ps(s0, _mm256_loadu_ps(b + i));
                s1 = _mm256_add_ps(s1, _mm256_loadu_ps(b + i + 8));
                s2 = _mm256_add_ps(s2, _mm256_loadu_ps(b + i + 16));
                s3 = _mm256_add_ps(s3, _mm256_loadu_ps(b + i + 24));
            }
            float tmp[8];
            _mm256_storeu_ps(tmp, _mm256_add_ps(_mm256_add_ps(s0, s1), _mm256_add_ps(s2, s3)));
            for (int k = 0; k < 8; k++) total += tmp[k];
        }
        sink_f = total;
        double t = now() - t0;
        if (t < best) best = t;
    }
    return (double)n * 4 / best / 1e9;
}

static double fma_peak(int threads, long iters) {
    double t0 = now();
    #pragma omp parallel num_threads(threads)
    {
        __m256 a0 = _mm256_set1_ps(1), a1 = _mm256_set1_ps(2), a2 = _mm256_set1_ps(3), a3 = _mm256_set1_ps(4), a4 = _mm256_set1_ps(5), a5 = _mm256_set1_ps(6), a6 = _mm256_set1_ps(7), a7 = _mm256_set1_ps(8),
               a8 = _mm256_set1_ps(9), a9 = _mm256_set1_ps(10);
        const __m256 b = _mm256_set1_ps(1.0000001f), c = _mm256_set1_ps(1e-7f);
        for (long i = 0; i < iters; i++) {
            a0 = _mm256_fmadd_ps(a0, b, c); a1 = _mm256_fmadd_ps(a1, b, c); a2 = _mm256_fmadd_ps(a2, b, c); a3 = _mm256_fmadd_ps(a3, b, c); a4 = _mm256_fmadd_ps(a4, b, c);
            a5 = _mm256_fmadd_ps(a5, b, c); a6 = _mm256_fmadd_ps(a6, b, c); a7 = _mm256_fmadd_ps(a7, b, c); a8 = _mm256_fmadd_ps(a8, b, c); a9 = _mm256_fmadd_ps(a9, b, c);
        }
        float tmp[8];
        _mm256_storeu_ps(tmp, _mm256_add_ps(_mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3)), _mm256_add_ps(_mm256_add_ps(a4, a5), _mm256_add_ps(_mm256_add_ps(a6, a7), _mm256_add_ps(a8, a9)))));
        sink_f = tmp[0] + tmp[7];
    }
    double t = now() - t0;
    return 2.0 * 8 * 10 * iters * threads / t / 1e9;
}

static double int8_peak(int threads, long iters) {
    /* unsigned-by-signed int8 dot over L1-resident buffers: maddubs -> madd(1) -> add; the operands are loaded each time so that nothing can be hoisted out of the loop */
    static uint8_t w[8192] __attribute__((aligned(32)));
    static int8_t x[8192] __attribute__((aligned(32)));
    for (int i = 0; i < 8192; i++) { w[i] = (uint8_t)((i * 7 + 3) % 16); x[i] = (int8_t)(((i * 13 + 5) % 255) - 127); }
    long reps = iters / 330;                                  /* about the same running time as the fp32 loop */
    double t0 = now();
    #pragma omp parallel num_threads(threads)
    {
        const __m256i ones = _mm256_set1_epi16(1);
        __m256i a0 = _mm256_setzero_si256(), a1 = a0, a2 = a0, a3 = a0;
        for (long r = 0; r < reps; r++) {
            for (int i = 0; i < 256; i += 4) {
                const __m256i w0 = _mm256_loadu_si256((const __m256i *)(w + i * 32)), w1 = _mm256_loadu_si256((const __m256i *)(w + i * 32 + 32));
                const __m256i w2 = _mm256_loadu_si256((const __m256i *)(w + i * 32 + 64)), w3 = _mm256_loadu_si256((const __m256i *)(w + i * 32 + 96));
                const __m256i x0 = _mm256_loadu_si256((const __m256i *)(x + i * 32)), x1 = _mm256_loadu_si256((const __m256i *)(x + i * 32 + 32));
                const __m256i x2 = _mm256_loadu_si256((const __m256i *)(x + i * 32 + 64)), x3 = _mm256_loadu_si256((const __m256i *)(x + i * 32 + 96));
                a0 = _mm256_add_epi32(a0, _mm256_madd_epi16(_mm256_maddubs_epi16(w0, x0), ones));
                a1 = _mm256_add_epi32(a1, _mm256_madd_epi16(_mm256_maddubs_epi16(w1, x1), ones));
                a2 = _mm256_add_epi32(a2, _mm256_madd_epi16(_mm256_maddubs_epi16(w2, x2), ones));
                a3 = _mm256_add_epi32(a3, _mm256_madd_epi16(_mm256_maddubs_epi16(w3, x3), ones));
            }
        }
        int tmp[8];
        _mm256_storeu_si256((__m256i *)tmp, _mm256_add_epi32(_mm256_add_epi32(a0, a1), _mm256_add_epi32(a2, a3)));
        sink_i = tmp[0] + tmp[7];
    }
    double t = now() - t0;
    return 2.0 * 8192 * reps * threads / t / 1e9;             /* GOPS: one MAC = two ops */
}

int main(void) {
    size_t n = (size_t)64 << 20;                              // 64M floats = 256 MiB per array
    float *a = aligned_alloc(64, n * 4), *b = aligned_alloc(64, n * 4), *c = aligned_alloc(64, n * 4);
    for (size_t i = 0; i < n; i++) { a[i] = 0; b[i] = 1; c[i] = 2; }
    printf("cpu threads available %d\n", omp_get_max_threads());
    int ths[] = {1, 2, 4, 6, 8};
    for (int k = 0; k < 5; k++) {
        int th = ths[k];
        printf("threads=%d  triad %7.1f GB/s   read %7.1f GB/s\n", th, triad(a, b, c, n, th), readbw(b, n, th));
        fflush(stdout);
    }
    int ths2[] = {1, 8};
    for (int k = 0; k < 2; k++) {
        int th = ths2[k];
        printf("threads=%d  fp32 FMA %7.1f GFLOPs   int8 maddubs chain %7.1f GOPS\n", th, fma_peak(th, 100000000L), int8_peak(th, 100000000L));
        fflush(stdout);
    }
    printf("done\n");
    return 0;
}
