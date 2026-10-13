// Bench + check of crates/forge-kernels/kernels/flash_a80.cu (the Ampere prompt-attention kernels) against the sm_75 kernels of flash.cu / flash256.cu, compiled from the very files the engine builds: time, error against the old kernels and ragged shapes for the head widths 256 (gated), 128 and 64.
//   nvcc -O3 --use_fast_math -arch=sm_89 -o gpu_flash_a80_bench bench/gpu_flash_a80_bench.cu && ./gpu_flash_a80_bench [T pos0 [config 0|1|2]]
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include "../crates/forge-kernels/kernels/flash.cu"
#include "../crates/forge-kernels/kernels/flash256.cu"
#include "../crates/forge-kernels/kernels/flash_a80.cu"

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(1); } } while (0)

static uint64_t rng_s = 88172645463325252ull;
static uint32_t rnd32() { rng_s ^= rng_s << 13; rng_s ^= rng_s >> 7; rng_s ^= rng_s << 17; return (uint32_t)(rng_s >> 16); }
static float rndf() { return ((rnd32() & 0xFFFFFF) / 16777216.0f) * 2.f - 1.f; }

typedef void (*KernG)(const half*, const half*, const half*, half*, int, int, int, int, const half*, int, int);
typedef void (*KernD)(const half*, const half*, const half*, half*, int, int, int, int);

struct Cfg {
    const char* name;
    int hq, n_head, n_kv, z;
    bool gated;
    void* old_k;
    void* new_k;
    void* new_kh;
};

int main(int argc, char** argv) {
    const int T = argc > 1 ? atoi(argv[1]) : 1024, pos0 = argc > 2 ? atoi(argv[2]) : 14336;
    const int only = argc > 3 ? atoi(argv[3]) : -1;
    Cfg cfgs[] = {
        {"hd 256 gated (16 heads, 4 kv)", 256, 16, 4, 2, true, (void*)flash_fwd_256g, (void*)flash_fwd_256g_a80, (void*)flash_fwd_256g_a80h},
        {"hd 128 (32 heads, 8 kv)", 128, 32, 8, 1, false, (void*)flash_fwd_128, (void*)flash_fwd_128_a80, (void*)flash_fwd_128_a80h},
        {"hd 64 (14 heads, 2 kv)", 64, 14, 2, 1, false, (void*)flash_fwd_64, (void*)flash_fwd_64_a80, (void*)flash_fwd_64_a80h},
    };
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    for (int ci = 0; ci < 3; ci++) {
        if (only >= 0 && ci != only) continue;
        const Cfg& c = cfgs[ci];
        const int HQ = c.hq, n_head = c.n_head, n_kv = c.n_kv;
        const int ctx = pos0 + T, q_dim = n_head * HQ, kv_dim = n_kv * HQ;
        printf("\n== %s: T %d pos0 %d (context %d)\n", c.name, T, pos0, ctx);
        std::vector<half> q((size_t)T * q_dim), k((size_t)ctx * kv_dim), v((size_t)ctx * kv_dim), gate((size_t)T * q_dim);
        for (auto& x : q) x = __float2half(rndf() * 2.f);
        for (auto& x : k) x = __float2half(rndf() * 2.f);
        for (auto& x : v) x = __float2half(rndf());
        for (auto& x : gate) x = __float2half(rndf() * 2.f);
        half *d_q, *d_k, *d_v, *d_g, *d_o;
        CK(cudaMalloc(&d_q, q.size() * 2)); CK(cudaMalloc(&d_k, k.size() * 2)); CK(cudaMalloc(&d_v, v.size() * 2)); CK(cudaMalloc(&d_g, gate.size() * 2));
        CK(cudaMalloc(&d_o, (size_t)T * q_dim * 2));
        CK(cudaMemcpy(d_q, q.data(), q.size() * 2, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_k, k.data(), k.size() * 2, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_v, v.data(), v.size() * 2, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_g, gate.data(), gate.size() * 2, cudaMemcpyHostToDevice));
        struct Ent { const char* name; void* f; } ents[] = {{"old (sm_75 kernel)", c.old_k}, {"a80  (fp32 acc)", c.new_k}, {"a80h (fp16 acc S)", c.new_kh}};
        std::vector<half> ref((size_t)T * q_dim), got((size_t)T * q_dim);
        auto run = [&](void* f, int TT, int pp) {
            dim3 grid((TT + 63) / 64, n_head, c.z);
            if (c.gated) ((KernG)f)<<<grid, 128>>>(d_q, d_k, d_v, d_o, TT, pp, n_head, n_kv, d_g, q_dim, HQ);
            else ((KernD)f)<<<grid, 128>>>(d_q, d_k, d_v, d_o, TT, pp, n_head, n_kv);
        };
        for (int ei = 0; ei < 3; ei++) {
            cudaFuncAttributes fa;
            cudaFuncGetAttributes(&fa, (const void*)ents[ei].f);
            CK(cudaMemset(d_o, 0, (size_t)T * q_dim * 2));
            run(ents[ei].f, T, pos0);
            CK(cudaDeviceSynchronize());
            CK(cudaMemcpy(got.data(), d_o, got.size() * 2, cudaMemcpyDeviceToHost));
            if (ei == 0) ref = got;
            double worst = 0, rms = 0, rms_ref = 0;
            for (size_t i = 0; i < got.size(); i++) {
                const double a = __half2float(got[i]), b = __half2float(ref[i]);
                worst = std::max(worst, std::fabs(a - b));
                rms += (a - b) * (a - b);
                rms_ref += b * b;
            }
            float best = 1e30f;
            for (int rep = 0; rep < 5; rep++) {
                cudaEventRecord(e0);
                for (int i = 0; i < 3; i++) run(ents[ei].f, T, pos0);
                cudaEventRecord(e1);
                cudaEventSynchronize(e1);
                float ms;
                cudaEventElapsedTime(&ms, e0, e1);
                best = std::min(best, ms / 3);
            }
            const double flops = 4.0 * (double)T * (pos0 + T / 2.0) * HQ * n_head;
            printf("%-22s %3d regs, spill %3zu B: %8.3f ms  %5.1f TFLOPs (counted)   max |diff| %.2e, rms rel %.2e\n", ents[ei].name, fa.numRegs, fa.localSizeBytes, best, flops / (best * 1e-3) / 1e12, worst, std::sqrt(rms / std::max(rms_ref, 1e-30)));
        }
        // awkward shapes: ragged chunks, short contexts, a chunk that starts at 0, a single row, rows beyond a block boundary
        const int shapes[][2] = {{100, 5}, {1000, 0}, {1, 70}, {65, 63}, {257, 1900}, {16, 0}, {64, 0}, {63, 17}, {129, 64}};
        for (auto& sh : shapes) {
            const int T2 = sh[0], p2 = sh[1];
            if (p2 + T2 > ctx) continue;
            std::vector<half> ref2((size_t)T2 * q_dim), got2((size_t)T2 * q_dim);
            for (int ei = 0; ei < 3; ei++) {
                CK(cudaMemset(d_o, 0, (size_t)T2 * q_dim * 2));
                run(ents[ei].f, T2, p2);
                CK(cudaDeviceSynchronize());
                CK(cudaMemcpy(got2.data(), d_o, got2.size() * 2, cudaMemcpyDeviceToHost));
                if (ei == 0) { ref2 = got2; continue; }
                double rms = 0, rms_ref = 0, worst = 0;
                for (size_t i = 0; i < got2.size(); i++) {
                    const double a = __half2float(got2[i]), b = __half2float(ref2[i]);
                    if (std::isnan(a)) { worst = 1e30; }
                    rms += (a - b) * (a - b); rms_ref += b * b; worst = std::max(worst, std::fabs(a - b));
                }
                printf("  T %4d pos0 %4d  %-20s rms rel %.2e  max |diff| %.2e\n", T2, p2, ents[ei].name, std::sqrt(rms / std::max(rms_ref, 1e-30)), worst);
            }
        }
        cudaFree(d_q); cudaFree(d_k); cudaFree(d_v); cudaFree(d_g); cudaFree(d_o);
    }
    return 0;
}
