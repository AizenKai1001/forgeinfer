// Stand-alone micro-benchmark of the multi-row quantized matvec kernels (crates/forge-kernels/kernels/decode_m.cu): random weights in the planar layouts of decode_q.cu, random activations quantized with
// the production quantizer, the kernel for T = 2 .. 16 tokens against the one-token kernel run once per token (correctness: relative difference of the outputs) and timed (weights streamed once for all T rows).
//   build (homelab, from the work tree root): tools/mrow_kbench.sh      run: ./mrow_kbench [format|all] [rows] [K] [T] ...
// Not part of the product: it exists so that a kernel variant can be compiled and measured in a minute instead of rebuilding forge-kernels (about 4 minutes).

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <string>
#include <cuda_runtime.h>

#include "decode.cu"
#include "decode_q8.cu"
#include "decode_q.cu"
#include "decode_h.cu"
#include "decode_t.cu"
#define MM_NO_INST
#include "decode_m.cu"

#define CK(x)                                                                                                                                                                                \
    do {                                                                                                                                                                                     \
        cudaError_t e = (x);                                                                                                                                                                 \
        if (e != cudaSuccess) {                                                                                                                                                              \
            fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e));                                                                                                       \
            exit(1);                                                                                                                                                                         \
        }                                                                                                                                                                                    \
    } while (0)

template <int F, int T, int R, int NW, int MINB, int EPI>
__global__ void __launch_bounds__(NW * 32, MINB) bench_mm(QP w, const int8_t* xq, const float* xd, const int* xs, float* out, int ldo, int rows, int K, size_t plane) {
    MmMats mats;
    mats.m[0] = MmMat{w, out, rows, ldo};
    mats.n = 1;
    mm_impl<F, T, R, EPI>(mats, xq, xd, xs, K);
}

static int g_budget = 46000;
static uint64_t g_rng = 88172645463325252ull;
static uint32_t rnd() {
    g_rng ^= g_rng << 13;
    g_rng ^= g_rng >> 7;
    g_rng ^= g_rng << 17;
    return (uint32_t)(g_rng >> 11);
}
static float rndf() { return (float)rnd() / 2147483648.f * 2.f - 1.f; }
static uint16_t half_bits(float f) {
    __half h = __float2half(f);
    return *reinterpret_cast<uint16_t*>(&h);
}

struct Planes {
    void* p[4] = {nullptr, nullptr, nullptr, nullptr};
    size_t bytes = 0;
};

static void* upload(const std::vector<uint8_t>& v) {
    void* d;
    CK(cudaMalloc(&d, v.size()));
    CK(cudaMemcpy(d, v.data(), v.size(), cudaMemcpyHostToDevice));
    return d;
}

static Planes make_weights(int fmt, int rows, int K) {
    Planes P;
    const size_t nb32 = (size_t)rows * (K / 32), nsb = (size_t)rows * (K / 256);
    size_t sz[4] = {0, 0, 0, 0};
    switch (fmt) {
        case FMT_Q8_0: sz[0] = (size_t)rows * K; sz[1] = nb32 * 2; break;
        case FMT_Q4_0: sz[0] = nb32 * 16; sz[1] = nb32 * 2; break;
        case FMT_Q5_0: sz[0] = nb32 * 16; sz[1] = nb32 * 4; sz[2] = nb32 * 2; break;
        case FMT_Q4_K: sz[0] = nsb * 128; sz[1] = nsb * 16; break;
        case FMT_Q5_K: sz[0] = nsb * 128; sz[1] = nsb * 32; sz[2] = nsb * 16; break;
        default: sz[0] = nsb * 128; sz[1] = nsb * 64; sz[2] = nsb * 16; sz[3] = nsb * 2; break;
    }
    for (int i = 0; i < 4; i++) {
        if (!sz[i]) continue;
        std::vector<uint8_t> v(sz[i]);
        for (auto& b : v) b = (uint8_t)rnd();
        P.bytes += sz[i];
        // fix the fp16 fields
        auto set_half = [&](size_t idx, float f) {
            uint16_t h = half_bits(f);
            v[idx * 2] = h & 0xFF;
            v[idx * 2 + 1] = h >> 8;
        };
        if ((fmt == FMT_Q8_0 && i == 1) || (fmt == FMT_Q4_0 && i == 1) || (fmt == FMT_Q5_0 && i == 2)) {
            for (size_t k = 0; k < sz[i] / 2; k++) set_half(k, 0.004f + 0.004f * (rnd() & 7));
        }
        if ((fmt == FMT_Q4_K && i == 1) || (fmt == FMT_Q5_K && i == 2)) {
            for (size_t b = 0; b < sz[i] / 16; b++) {
                set_half(b * 8, 0.0005f + 0.0005f * (rnd() & 7));
                set_half(b * 8 + 1, 0.0005f + 0.0005f * (rnd() & 7));
            }
        }
        if (fmt == FMT_Q6_K && i == 2) {
            for (auto& b : v) b = (uint8_t)(int8_t)((int)(rnd() % 41) - 20);
        }
        if (fmt == FMT_Q6_K && i == 3) {
            for (size_t k = 0; k < sz[i] / 2; k++) set_half(k, 0.0008f + 0.0004f * (rnd() & 7));
        }
        P.p[i] = upload(v);
    }
    return P;
}

static const char* FNAME[6] = {"q8_0", "q4_0", "q5_0", "q4_k", "q5_k", "q6_k"};

struct Act {
    int8_t* xq;
    float* xd;
    int* xs;
};

static Act make_act(int T, int K) {
    std::vector<float> h((size_t)T * K);
    for (auto& v : h) v = rndf() * (0.2f + 3.f * (rnd() % 7 == 0));
    float* d;
    CK(cudaMalloc(&d, h.size() * 4));
    CK(cudaMemcpy(d, h.data(), h.size() * 4, cudaMemcpyHostToDevice));
    Act a;
    CK(cudaMalloc(&a.xq, (size_t)T * K));
    CK(cudaMalloc(&a.xd, (size_t)T * K / 32 * 4));
    CK(cudaMalloc(&a.xs, (size_t)T * K / 32 * 4));
    const int need = (K + 511) / 512;
    for (int t = 0; t < T; t++) {
        const float* x = d + (size_t)t * K;
        int8_t* q = a.xq + (size_t)t * K;
        float* sd = a.xd + (size_t)t * K / 32;
        int* ss = a.xs + (size_t)t * K / 32;
        if (need <= 8) quant_kernel_8<<<1, 512>>>(x, q, sd, ss, K);
        else if (need <= 16) quant_kernel_16<<<1, 512>>>(x, q, sd, ss, K);
        else quant_kernel_32<<<1, 512>>>(x, q, sd, ss, K);
    }
    CK(cudaDeviceSynchronize());
    CK(cudaFree(d));
    return a;
}

// slices of whole super-blocks so that T tokens of a slice fit the shared-memory budget
template <int F>
static int pick_slices(int T, int K, int budget) {
    const int nsb = K / 256;
    for (int nj = 1; nj <= nsb; nj++) {
        const int sbmax = (nsb + nj - 1) / nj;
        if (mm_smem_bytes<F>(T, sbmax * 256) <= (size_t)budget) return nj;
    }
    return nsb;
}

// the one-token reference: mvt1_out_<fmt>_kernel per token
typedef void (*RefKernel)(QP, const int8_t*, const float*, const int*, float*, int, int, int);

static void ref_run(int fmt, const Planes& P, const Act& a, int T, int rows, int K, float* out, int ldo, cudaStream_t st) {
    const char* names[6] = {"mvt1_out_q8_kernel", "mvt1_out_q40_kernel", "mvt1_out_q50_kernel", "mvt1_out_q4k_kernel", "mvt1_out_q5k_kernel", "mvt1_out_q6k_kernel"};
    (void)names;
    typedef void (*K1)(const uint8_t*, const uint8_t*, const uint8_t*, const uint8_t*, const int8_t*, const float*, const int*, float*, int, int, int);
    K1 fns[6] = {mvt1_out_q8_kernel, mvt1_out_q40_kernel, mvt1_out_q50_kernel, mvt1_out_q4k_kernel, mvt1_out_q5k_kernel, mvt1_out_q6k_kernel};
    for (int t = 0; t < T; t++) {
        const int smem = K + K / 32 * 8;
        cudaFuncSetAttribute(fns[fmt], cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        fns[fmt]<<<272, 256, smem, st>>>((const uint8_t*)P.p[0], (const uint8_t*)P.p[1], (const uint8_t*)P.p[2], (const uint8_t*)P.p[3], a.xq + (size_t)t * K, a.xd + (size_t)t * K / 32,
                                         a.xs + (size_t)t * K / 32, out + (size_t)t * ldo, ldo, rows, K);
    }
}

// the SM and memory clocks of a GPU that has been idle take a few hundred milliseconds to rise: a timing loop of a few milliseconds measures the idle clocks
static void warm_gpu(float ms_target) {
    float* buf;
    CK(cudaMalloc(&buf, 256 << 20));
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    cudaEventRecord(e0);
    float ms = 0;
    while (ms < ms_target) {
        for (int i = 0; i < 20; i++) cudaMemsetAsync(buf, i, 256 << 20);
        cudaEventRecord(e1);
        cudaEventSynchronize(e1);
        cudaEventElapsedTime(&ms, e0, e1);
    }
    CK(cudaFree(buf));
}

template <int F, int T, int R, int NW, int MINB>
static void run_one(const Planes& P, const Act& a, int rows, int K, int reps, const float* ref, const char* tag) {
    typedef void (*KF)(QP, const int8_t*, const float*, const int*, float*, int, int, int, size_t);
    KF kf = bench_mm<F, T, R, NW, MINB, 0>;
    const int nj = pick_slices<F>(T, K, g_budget);
    const int sbmax = (K / 256 + nj - 1) / nj;
    const int smem = (int)mm_smem_bytes<F>(T, sbmax * 256);
    CK(cudaFuncSetAttribute(kf, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    int bps = 0;
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bps, kf, NW * 32, smem));
    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, 0));
    const int ngroups = rows / R;
    int gx = (bps * prop.multiProcessorCount) / nj;
    if (gx < 1) gx = 1;
    gx = std::min(gx, (ngroups + NW * 4 - 1) / (NW * 4));
    float *out, *part;
    CK(cudaMalloc(&out, (size_t)T * rows * 4));
    CK(cudaMalloc(&part, (size_t)nj * T * rows * 4));
    cudaFuncAttributes fa;
    CK(cudaFuncGetAttributes(&fa, kf));
    QP w{(const uint8_t*)P.p[0], (const uint8_t*)P.p[1], (const uint8_t*)P.p[2], (const uint8_t*)P.p[3]};
    auto launch = [&]() {
        if (nj == 1) kf<<<dim3(gx, 1), NW * 32, smem>>>(w, a.xq, a.xd, a.xs, out, rows, rows, K, (size_t)T * rows);
        else {
            kf<<<dim3(gx, nj), NW * 32, smem>>>(w, a.xq, a.xd, a.xs, part, rows, rows, K, (size_t)T * rows);
            mm_reduce_out_kernel<<<dim3((rows + 255) / 256, T, 1), 256>>>(part, nullptr, nullptr, out, nullptr, nullptr, rows, 0, 0, rows, 0, 0, nj);
        }
    };
    launch();
    CK(cudaDeviceSynchronize());
    CK(cudaGetLastError());
    std::vector<float> h((size_t)T * rows);
    CK(cudaMemcpy(h.data(), out, h.size() * 4, cudaMemcpyDeviceToHost));
    double num = 0, den = 0, maxabs = 0;
    int bitexact = 0;
    for (size_t i = 0; i < h.size(); i++) {
        const double d = (double)h[i] - ref[i];
        num += d * d;
        den += (double)ref[i] * ref[i];
        maxabs = std::max(maxabs, fabs(d));
        bitexact += h[i] == ref[i];
    }
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    warm_gpu(300.f);
    for (int i = 0; i < 5; i++) launch();
    cudaEventRecord(e0);
    for (int i = 0; i < reps; i++) launch();
    cudaEventRecord(e1);
    cudaEventSynchronize(e1);
    float ms;
    cudaEventElapsedTime(&ms, e0, e1);
    const double us = ms * 1000.0 / reps;
    printf("%-6s %-5s rows %6d K %5d T %2d R %d NW %d | nj %d gx %3d bps %d regs %3d smem %5d | %8.1f us  %6.1f GB/s | rel.rms %.2e max %.2e bitexact %5.1f%%\n", FNAME[F], tag, rows, K, T, R, NW, nj, gx, bps, fa.numRegs, smem, us,
           (double)P.bytes / (us * 1e3), sqrt(num / den), maxabs, 100.0 * bitexact / h.size());
    CK(cudaFree(out));
    CK(cudaFree(part));
}

int main(int argc, char** argv) {
    int rows = 9728, K = 2560, reps = 300;
    std::string only = "all";
    int onlyT = 0;
    int onlyR = getenv("MM_R") ? atoi(getenv("MM_R")) : 0, onlyNW = getenv("MM_NW") ? atoi(getenv("MM_NW")) : 0;
    if (argc > 1) only = argv[1];
    if (argc > 2) rows = atoi(argv[2]);
    if (argc > 3) K = atoi(argv[3]);
    if (argc > 4) onlyT = atoi(argv[4]);
    if (argc > 5) g_budget = atoi(argv[5]);
    const int fmts[6] = {FMT_Q8_0, FMT_Q4_0, FMT_Q5_0, FMT_Q4_K, FMT_Q5_K, FMT_Q6_K};
    for (int fi = 0; fi < 6; fi++) {
        const int fmt = fmts[fi];
        if (only != "all" && only != FNAME[fmt]) continue;
        Planes P = make_weights(fmt, rows, K);
        const int Tmax = 16;
        Act a = make_act(Tmax, K);
        float* refd;
        CK(cudaMalloc(&refd, (size_t)Tmax * rows * 4));
        ref_run(fmt, P, a, Tmax, rows, K, refd, rows, 0);
        CK(cudaDeviceSynchronize());
        CK(cudaGetLastError());
        std::vector<float> ref((size_t)Tmax * rows);
        CK(cudaMemcpy(ref.data(), refd, ref.size() * 4, cudaMemcpyDeviceToHost));
        // the one-token reference kernel's time (per token) for orientation
        {
            cudaEvent_t e0, e1;
            cudaEventCreate(&e0);
            cudaEventCreate(&e1);
            warm_gpu(300.f);
            ref_run(fmt, P, a, 1, rows, K, refd, rows, 0);
            cudaEventRecord(e0);
            for (int i = 0; i < reps; i++) ref_run(fmt, P, a, 1, rows, K, refd, rows, 0);
            cudaEventRecord(e1);
            cudaEventSynchronize(e1);
            float ms;
            cudaEventElapsedTime(&ms, e0, e1);
            printf("%-6s one-token reference kernel (mvt1): %8.1f us  %6.1f GB/s\n", FNAME[fmt], ms * 1000.0 / reps, (double)P.bytes / (ms * 1000.0 / reps * 1e3));
        }
#define RUN(FF, TT, RR, NWW, MB)                                                                                                                                                             \
    if (fmt == FF && (onlyT == 0 || onlyT == TT) && (onlyR == 0 || onlyR == RR) && (onlyNW == 0 || onlyNW == NWW)) run_one<FF, TT, RR, NWW, MB>(P, a, rows, K, reps, ref.data(), "new");
        RUN(FMT_Q4_K, 4, 2, 4, 3) RUN(FMT_Q4_K, 4, 4, 4, 3)
        RUN(FMT_Q4_K, 8, 2, 4, 3) RUN(FMT_Q4_K, 8, 2, 6, 2) RUN(FMT_Q4_K, 8, 2, 8, 2) RUN(FMT_Q4_K, 8, 2, 12, 1) RUN(FMT_Q4_K, 8, 1, 4, 4) RUN(FMT_Q4_K, 8, 4, 4, 2) RUN(FMT_Q4_K, 8, 4, 8, 1)
        RUN(FMT_Q4_K, 16, 2, 4, 3) RUN(FMT_Q4_K, 16, 2, 8, 2) RUN(FMT_Q4_K, 16, 2, 12, 1) RUN(FMT_Q4_K, 16, 1, 4, 4)
        RUN(FMT_Q6_K, 8, 2, 4, 3) RUN(FMT_Q6_K, 8, 2, 8, 2) RUN(FMT_Q6_K, 8, 1, 4, 4) RUN(FMT_Q6_K, 16, 2, 8, 2) RUN(FMT_Q6_K, 16, 1, 4, 3)
        RUN(FMT_Q8_0, 8, 2, 4, 3) RUN(FMT_Q8_0, 8, 2, 8, 2) RUN(FMT_Q5_K, 8, 2, 4, 3) RUN(FMT_Q5_K, 8, 2, 8, 2) RUN(FMT_Q4_0, 8, 2, 4, 3) RUN(FMT_Q4_0, 8, 2, 8, 2) RUN(FMT_Q5_0, 8, 2, 4, 3) RUN(FMT_Q5_0, 8, 2, 8, 2)
        for (int i = 0; i < 4; i++)
            if (P.p[i]) CK(cudaFree(P.p[i]));
        CK(cudaFree(a.xq));
        CK(cudaFree(a.xd));
        CK(cudaFree(a.xs));
        CK(cudaFree(refd));
    }
    return 0;
}
