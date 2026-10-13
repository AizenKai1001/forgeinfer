// Stand-alone check and benchmark of the Q8 matrix-vector kernels (crates/forge-kernels/kernels/decode_q8f.cu), without the Rust crates:
//   nvcc -O3 -arch=sm_75 --use_fast_math -std=c++17 -o q8dev tools/q8dev.cu -I crates/forge-kernels/kernels && ./q8dev [--shape N K] [--check-only] [--sweep]
// For every Forge-1 matrix shape (365M and 150M models) it quantizes a random fp32 matrix on the device (q8f_quantize), compares the kernel output with a double-precision reference computed from the
// dequantized weights (independent of the device layout code), and times the kernels on cold weights (copies of the matrix are cycled so that nothing stays in the L2 cache).
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <random>
#include "decode_q8f.cu"

#define CK(x)                                                                                                                                                                                    \
    do {                                                                                                                                                                                         \
        cudaError_t e_ = (x);                                                                                                                                                                    \
        if (e_ != cudaSuccess) {                                                                                                                                                                 \
            printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__);                                                                                                      \
            exit(1);                                                                                                                                                                             \
        }                                                                                                                                                                                        \
    } while (0)

// the same kernels with other batch sizes (super-chunks per batch)
Q8F_GEMV(q8f_gemv_mixnorm_f16_nb2, q8f::PRO_MIXNORM, q8f::EPI_F16, 2)
Q8F_GEMV(q8f_gemv_vec_res_nb2, q8f::PRO_VEC, q8f::EPI_RES, 2)
Q8F_GEMV(q8f_gemv_norm_relu2_nb2, q8f::PRO_NORM, q8f::EPI_RELU2, 2)
Q8F_GEMV(q8f_gemv_mixnorm_f16_nb3, q8f::PRO_MIXNORM, q8f::EPI_F16, 3)
Q8F_GEMV(q8f_gemv_vec_res_nb3, q8f::PRO_VEC, q8f::EPI_RES, 3)
Q8F_GEMV(q8f_gemv_norm_relu2_nb3, q8f::PRO_NORM, q8f::EPI_RELU2, 3)
Q8F_GEMV(q8f_gemv_mixnorm_f16_nb6, q8f::PRO_MIXNORM, q8f::EPI_F16, 6)
Q8F_GEMV(q8f_gemv_vec_res_nb6, q8f::PRO_VEC, q8f::EPI_RES, 6)
Q8F_GEMV(q8f_gemv_norm_relu2_nb6, q8f::PRO_NORM, q8f::EPI_RELU2, 6)
Q8F_GEMV(q8f_gemv_mixnorm_f16_nb8, q8f::PRO_MIXNORM, q8f::EPI_F16, 8)
Q8F_GEMV(q8f_gemv_vec_res_nb8, q8f::PRO_VEC, q8f::EPI_RES, 8)
Q8F_GEMV(q8f_gemv_norm_relu2_nb8, q8f::PRO_NORM, q8f::EPI_RELU2, 8)

static float h2f(uint16_t h) { return __half2float(*reinterpret_cast<__half*>(&h)); }
static uint16_t f2h(float f) {
    __half h = __float2half_rn(f);
    return *reinterpret_cast<uint16_t*>(&h);
}

// reference quantization (the rule of q8f_quantize): d = fp16(amax / 127), q = round(x / d)
static void ref_dequant(const std::vector<float>& w, int N, int K, std::vector<float>& deq) {
    deq.resize((size_t)N * K);
    for (int r = 0; r < N; r++)
        for (int kb = 0; kb < K / 32; kb++) {
            float amax = 0;
            for (int i = 0; i < 32; i++) amax = std::max(amax, fabsf(w[(size_t)r * K + kb * 32 + i]));
            const float d = h2f(f2h(amax * (1.f / 127.f)));
            const float id = d != 0.f ? 1.f / d : 0.f;
            for (int i = 0; i < 32; i++) {
                int q = (int)lrintf(w[(size_t)r * K + kb * 32 + i] * id);
                q = std::max(-127, std::min(127, q));
                deq[(size_t)r * K + kb * 32 + i] = q * d;
            }
        }
}

__global__ void stream_sum(const uint4* p, size_t n, float* out) {
    float s = 0;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        const uint4 v = __ldcs(p + i);
        s += (float)(v.x ^ v.y) + (float)(v.z ^ v.w);
    }
    if (s == 12345.f) out[0] = s;
}

struct Case {
    const char* name;
    int N, K;
    int pro;   // 0 vec, 1 norm, 2 mixnorm
    int epi;   // 0 f16, 1 f32, 4 res, 5 relu2
};

static size_t smem_bytes(int K, int W) { return 16 + (size_t)K * 2 + (size_t)W * 64 + 128; }

typedef void (*Kern)(const uint8_t*, int, int, int, const half*, const float*, const float*, const float*, const float*, int, int, float*, const float*, float, void*, const float*);

static int g_nb = 4;
static Kern pick(int pro, int epi) {
    if (pro == 2) return g_nb == 2 ? q8f_gemv_mixnorm_f16_nb2 : g_nb == 3 ? q8f_gemv_mixnorm_f16_nb3 : g_nb == 6 ? q8f_gemv_mixnorm_f16_nb6 : g_nb == 8 ? q8f_gemv_mixnorm_f16_nb8 : q8f_gemv_mixnorm_f16;
    if (pro == 1 && epi == 0) return q8f_gemv_norm_f16;
    if (pro == 1) return g_nb == 2 ? q8f_gemv_norm_relu2_nb2 : g_nb == 3 ? q8f_gemv_norm_relu2_nb3 : g_nb == 6 ? q8f_gemv_norm_relu2_nb6 : g_nb == 8 ? q8f_gemv_norm_relu2_nb8 : q8f_gemv_norm_relu2;
    if (epi == 4) return g_nb == 2 ? q8f_gemv_vec_res_nb2 : g_nb == 3 ? q8f_gemv_vec_res_nb3 : g_nb == 6 ? q8f_gemv_vec_res_nb6 : g_nb == 8 ? q8f_gemv_vec_res_nb8 : q8f_gemv_vec_res;
    return q8f_gemv_vec_f32;
}

int main(int argc, char** argv) {
    bool check_only = false, sweep = false, hot = false;
    int onlyN = 0, onlyK = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--check-only")) check_only = true;
        else if (!strcmp(argv[i], "--sweep")) sweep = true;
        else if (!strcmp(argv[i], "--nb") && i + 1 < argc) g_nb = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--hot")) hot = true;   // one copy of the weights: they stay in the L2 cache (the launch and prologue cost without the DRAM)
        else if (!strcmp(argv[i], "--shape") && i + 2 < argc) {
            onlyN = atoi(argv[i + 1]);
            onlyK = atoi(argv[i + 2]);
            i += 2;
        }
    }
    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, 0));
    printf("%s, %d SMs\n", prop.name, prop.multiProcessorCount);
    std::vector<Case> cases = {
        {"qkv365", 1536, 1024, 2, 0}, {"wo365", 1024, 1024, 0, 4},   {"up365", 4096, 1024, 1, 5}, {"down365", 1024, 4096, 0, 4}, {"head365", 32768, 1024, 0, 1},
        {"qkv150", 960, 576, 2, 0},   {"wo150", 576, 576, 0, 4},     {"up150", 1536, 576, 1, 5},  {"down150", 576, 1536, 0, 4},   {"head150", 32768, 576, 0, 1},
    };
    if (onlyN) cases = {{"custom", onlyN, onlyK, 2, 0}, {"custom_vec", onlyN, onlyK, 0, 4}};
    std::mt19937 rng(1234);
    std::normal_distribution<float> nd(0.f, 1.f);
    float* sink;
    CK(cudaMalloc(&sink, 16));
    // the bandwidth ceiling: a plain streaming read of 256 MB
    {
        const size_t bytes = 256u << 20;
        uint4* p;
        CK(cudaMalloc(&p, bytes));
        CK(cudaMemset(p, 1, bytes));
        cudaEvent_t a, b;
        cudaEventCreate(&a);
        cudaEventCreate(&b);
        for (int w = 0; w < 3; w++) stream_sum<<<prop.multiProcessorCount * 8, 256>>>(p, bytes / 16, sink);
        cudaEventRecord(a);
        for (int w = 0; w < 10; w++) stream_sum<<<prop.multiProcessorCount * 8, 256>>>(p, bytes / 16, sink);
        cudaEventRecord(b);
        cudaEventSynchronize(b);
        float ms;
        cudaEventElapsedTime(&ms, a, b);
        printf("plain streaming read of 256 MB: %.1f GB/s\n", 10.0 * bytes / (ms * 1e-3) / 1e9);
        cudaFree(p);
    }
    int fails = 0;
    for (const Case& cs : cases) {
        const int N = cs.N, K = cs.K;
        std::vector<float> w((size_t)N * K);
        for (auto& v : w) v = 0.04f * nd(rng) * (1.f + 3.f * (rng() % 50 == 0));
        std::vector<float> x(K), nwv(K), x0(K), xsk(K);
        for (int i = 0; i < K; i++) {
            x[i] = nd(rng);
            nwv[i] = 0.5f + (rng() % 100) / 100.f;
            x0[i] = nd(rng);
            xsk[i] = nd(rng);
        }
        const float sc[3] = {0.9f, 0.3f, -0.2f};
        const size_t wbytes = (size_t)(N / 16) * (K / 64) * q8f::SC_BYTES;
        const int copies = hot ? 1 : (int)std::max<size_t>(2, (192u << 20) / wbytes);
        float* dw;
        uint8_t* dq;
        CK(cudaMalloc(&dw, (size_t)N * K * 4));
        CK(cudaMemcpy(dw, w.data(), (size_t)N * K * 4, cudaMemcpyHostToDevice));
        CK(cudaMalloc(&dq, wbytes * copies));
        CK(cudaMemset(dq, 0, wbytes * copies));
        q8f_quantize<<<(N * (K / 32) + 7) / 8, 256>>>(dw, N, K, dq);
        CK(cudaGetLastError());
        for (int c = 1; c < copies; c++) CK(cudaMemcpy(dq + c * wbytes, dq, wbytes, cudaMemcpyDeviceToDevice));
        float *dx, *dnw, *dx0, *dxs, *dsc, *dxout, *dout, *dR;
        half* dh;
        CK(cudaMalloc(&dx, K * 4));
        CK(cudaMalloc(&dnw, K * 4));
        CK(cudaMalloc(&dx0, K * 4));
        CK(cudaMalloc(&dxs, K * 4));
        CK(cudaMalloc(&dsc, 12));
        CK(cudaMalloc(&dxout, K * 4));
        CK(cudaMalloc(&dout, (size_t)N * 4));
        CK(cudaMalloc(&dR, (size_t)N * 4));
        CK(cudaMalloc(&dh, K * 2));
        CK(cudaMemcpy(dx, x.data(), K * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dnw, nwv.data(), K * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dx0, x0.data(), K * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dxs, xsk.data(), K * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dsc, sc, 12, cudaMemcpyHostToDevice));
        std::vector<uint16_t> xh(K);
        std::vector<float> xhf(K);
        for (int i = 0; i < K; i++) {
            xh[i] = f2h(x[i]);
            xhf[i] = h2f(xh[i]);
        }
        CK(cudaMemcpy(dh, xh.data(), K * 2, cudaMemcpyHostToDevice));
        std::vector<float> rr(N);
        for (auto& v : rr) v = nd(rng);
        CK(cudaMemcpy(dR, rr.data(), (size_t)N * 4, cudaMemcpyHostToDevice));
        // the activation the kernel sees, on the host
        std::vector<float> act(K);
        if (cs.pro == 0) act = xhf;
        else {
            std::vector<float> m(K);
            for (int i = 0; i < K; i++) m[i] = cs.pro == 2 ? sc[0] * x[i] + sc[1] * x0[i] + sc[2] * xsk[i] : x[i];
            double ss = 0;
            for (int i = 0; i < K; i++) ss += (double)m[i] * m[i];
            const float r = 1.f / sqrtf((float)(ss / K) + 1e-5f);
            for (int i = 0; i < K; i++) act[i] = h2f(f2h(m[i] * r * nwv[i]));
        }
        std::vector<float> deq;
        ref_dequant(w, N, K, deq);
        {
            // q8f_dequant against the host dequantization (q * d rounded to fp16)
            half* dd;
            CK(cudaMalloc(&dd, (size_t)N * K * 2));
            q8f_dequant<<<((N / 16) * (K / 32) + 7) / 8, 256>>>(dq, N, K, dd);
            CK(cudaGetLastError());
            std::vector<uint16_t> back((size_t)N * K);
            CK(cudaMemcpy(back.data(), dd, (size_t)N * K * 2, cudaMemcpyDeviceToHost));
            size_t bad = 0;
            for (size_t i = 0; i < back.size(); i++)
                if (fabsf(h2f(back[i]) - deq[i]) > 1e-3f * fabsf(deq[i]) + 1e-6f) bad++;
            // a few elements may differ by one step where the device and the host round x / d differently
            if (bad > back.size() / 100000 + 8) {
                printf("   DEQUANT MISMATCH: %zu of %zu\n", bad, back.size());
                fails++;
            }
            cudaFree(dd);
        }
        std::vector<double> ref(N);
        for (int n = 0; n < N; n++) {
            double s = 0;
            for (int k = 0; k < K; k++) s += (double)deq[(size_t)n * K + k] * act[k];
            ref[n] = s;
        }
        Kern kern = pick(cs.pro, cs.epi);
        const int tiles = N / 16, nsc = K / 64;
        struct Cfg {
            int W, S;
        };
        std::vector<Cfg> cfgs;
        if (sweep) {
            for (int W : {4, 8})
                for (int S : {1, 2, 4, 8})
                    if (S <= W && W % S == 0 && S <= nsc) cfgs.push_back({W, S});
        } else {
            int S = 1;
            while (S < 8 && tiles * S < 256 && S * 2 <= nsc) S *= 2;
            cfgs.push_back({8, S});
        }
        printf("%-9s N %5d K %5d  weights %6.2f MB  (pro %d epi %d)\n", cs.name, N, K, wbytes / 1e6, cs.pro, cs.epi);
        for (Cfg cf : cfgs) {
            const int TB = cf.W / cf.S;
            const int blocks = (tiles + TB - 1) / TB;
            const size_t sm = smem_bytes(K, cf.W);
            auto launch = [&](int copy) {
                kern<<<blocks, cf.W * 32, sm>>>(dq + (size_t)copy * wbytes, N, K, cf.S, dh, dx, dx0, dxs, dsc, 1, 1, dxout, dnw, 1e-5f, dout, dR);
            };
            CK(cudaMemset(dout, 0, (size_t)N * 4));
            launch(0);
            CK(cudaGetLastError());
            CK(cudaDeviceSynchronize());
            std::vector<float> got(N);
            std::vector<uint16_t> goth(N);
            if (cs.epi == 0 || cs.epi == 5) {
                CK(cudaMemcpy(goth.data(), dout, (size_t)N * 2, cudaMemcpyDeviceToHost));
                for (int n = 0; n < N; n++) got[n] = h2f(goth[n]);
            } else {
                CK(cudaMemcpy(got.data(), dout, (size_t)N * 4, cudaMemcpyDeviceToHost));
            }
            double maxerr = 0, maxref = 0;
            int bad = 0;
            for (int n = 0; n < N; n++) {
                double want = ref[n];
                if (cs.epi == 4) want += rr[n];
                if (cs.epi == 5) {
                    const float u = h2f(f2h((float)ref[n]));
                    const float r0 = std::max(u, 0.f);
                    want = h2f(f2h(std::min(r0 * r0, 65000.f)));
                }
                const double err = fabs(got[n] - want);
                const double tol = (cs.epi == 0 || cs.epi == 5 ? 2e-3 * fabs(want) + 1e-3 : 1e-3 * fabs(want) + 2e-3);
                if (err > tol) bad++;
                maxerr = std::max(maxerr, err);
                maxref = std::max(maxref, fabs(want));
            }
            if (bad) fails++;
            if (check_only) {
                printf("   W %d S %d: max |err| %.3e (max |ref| %.3e), %d outside the tolerance\n", cf.W, cf.S, maxerr, maxref, bad);
                continue;
            }
            cudaEvent_t a, b;
            cudaEventCreate(&a);
            cudaEventCreate(&b);
            for (int w = 0; w < 20; w++) launch(w % copies);
            const int reps = 300;
            cudaEventRecord(a);
            for (int w = 0; w < reps; w++) launch(w % copies);
            cudaEventRecord(b);
            CK(cudaEventSynchronize(b));
            float ms;
            cudaEventElapsedTime(&ms, a, b);
            const double us = ms * 1e3 / reps;
            printf("   W %d S %d (%3d blocks x %d tiles): %7.2f us  %6.1f GB/s   max |err| %.2e, %d bad\n", cf.W, cf.S, blocks, TB, us, wbytes / (us * 1e-6) / 1e9, maxerr, bad);
        }
        cudaFree(dw);
        cudaFree(dq);
        cudaFree(dx);
        cudaFree(dnw);
        cudaFree(dx0);
        cudaFree(dxs);
        cudaFree(dsc);
        cudaFree(dxout);
        cudaFree(dout);
        cudaFree(dR);
        cudaFree(dh);
    }
    printf(fails ? "FAILED: %d configurations\n" : "all configurations match the reference (%d failures)\n", fails);
    return fails != 0;
}
