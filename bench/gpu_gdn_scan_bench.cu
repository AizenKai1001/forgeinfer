// Bench + check of the Gated DeltaNet prompt scan (crates/forge-kernels/kernels/prefill_h.cu): the one-column-per-thread kernels (gdn_scan_kernel_32 / 64 / 128) against variants that give every thread NC
// state columns, so that the k / q vectors a warp reads from shared memory serve 4 * NC columns instead of 4 (the one-column kernel spends most of its time in LDS.128 wavefronts: 12 per warp and token).
//   nvcc -O3 --use_fast_math -arch=sm_89 -o gpu_gdn_scan_bench bench/gpu_gdn_scan_bench.cu && ./gpu_gdn_scan_bench [T [n_v]]
// The dot products keep the engine's summation order (16 sequential FMAs per lane, then the 3-level xor tree inside the 8 lanes of a column): the outputs must be BIT-IDENTICAL (the table prints the
// number of differing words of the output, the final state and the convolution window). On the RTX 4060 (T 1024, 32 value heads): 1,100 us -> 630 us for 64 columns per block with 4 per thread.
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(1); } } while (0)

#define GDN_D 128
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float block_sum(float v, float* red) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = (blockDim.x + 31) >> 5;
    v = warp_sum(v);
    __syncthreads();
    if (lane == 0) red[warp] = v;
    __syncthreads();
    float t = 0.f;
    for (int i = 0; i < nw; i++) t += red[i];
    return t;
}
__device__ __forceinline__ float h_sigmoid(float x) { return 1.f / (1.f + __expf(-x)); }
__device__ __forceinline__ float h_silu(float x) { return x / (1.f + __expf(-x)); }
__device__ __forceinline__ float h_softplus(float x) { return x > 20.f ? x : logf(1.f + __expf(x)); }

#include "../crates/forge-kernels/kernels/prefill_h.cu"

// ---- variants -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// CPB columns per block, NC columns per thread (8 lanes per column, columns of a thread contiguous), tiles of TT tokens staged through registers into double-buffered shared memory as in the engine.
template <int CPB, int NC, int TT>
__device__ __forceinline__ void gdn_scan_nc_impl(const float* __restrict__ pq, const float* __restrict__ pk, const float* __restrict__ pv, const float* __restrict__ pg, const float* __restrict__ mixed,
                                                 float* __restrict__ conv_state, float* __restrict__ S, float* __restrict__ so, int T, int n_k, int n_v, int tiled, int conv_k, int conv_dim) {
    constexpr int NT = CPB * 8 / NC;                  // threads
    constexpr int NV4 = TT * CPB / 4;                 // float4 of v per tile
    constexpr int TOTAL = 2 * TT * 32 + NV4;          // float4 per tile: k, q, v
    constexpr int ITEMS = (TOTAL + NT - 1) / NT;
    __shared__ __align__(16) float sk[2][TT * GDN_D];
    __shared__ __align__(16) float sq[2][TT * GDN_D];
    __shared__ __align__(16) float sv[2][TT * CPB];
    __shared__ __align__(16) float sg[2][TT * 4];
    __shared__ __align__(16) float sot[2][TT * CPB];
    const int h = blockIdx.x, c0 = blockIdx.y * CPB, tid = threadIdx.x;
    const int cg = tid >> 3, l8 = tid & 7;            // column group (NC contiguous columns) within the block, lane within the column
    const int lcol0 = cg * NC;
    float4 s[NC][4];
#pragma unroll
    for (int c = 0; c < NC; c++) {
        const float4* s_ptr = reinterpret_cast<const float4*>(S + ((size_t)h * GDN_D + c0 + lcol0 + c) * GDN_D);
#pragma unroll
        for (int r = 0; r < 4; r++) s[c][r] = s_ptr[8 * r + l8];
    }
    const int ntiles = (T + TT - 1) / TT;
    float4 pre[ITEMS];
    float pre_g = 0.f;
    auto load_tile = [&](int tile) {
        const int t0 = tile * TT;
#pragma unroll
        for (int n = 0; n < ITEMS; n++) {
            const int i = tid + n * NT;
            float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
            if (i < TT * 32) {
                const int j = i >> 5, c = i & 31;
                if (t0 + j < T) v = __ldg(reinterpret_cast<const float4*>(pk + ((size_t)(t0 + j) * n_v + h) * GDN_D) + c);
            } else if (i < 2 * TT * 32) {
                const int ii = i - TT * 32, j = ii >> 5, c = ii & 31;
                if (t0 + j < T) v = __ldg(reinterpret_cast<const float4*>(pq + ((size_t)(t0 + j) * n_v + h) * GDN_D) + c);
            } else if (i < TOTAL) {
                const int ii = i - 2 * TT * 32, j = ii / (CPB / 4), c = ii % (CPB / 4);
                if (t0 + j < T) v = __ldg(reinterpret_cast<const float4*>(pv + ((size_t)(t0 + j) * n_v + h) * GDN_D + c0) + c);
            }
            pre[n] = v;
        }
        pre_g = 0.f;
        if (tid < TT * 3) {
            const int j = tid / 3, c = tid % 3;
            if (t0 + j < T) pre_g = __ldg(pg + ((size_t)(t0 + j) * n_v + h) * 3 + c);
        }
    };
    auto store_tile = [&](int buf) {
#pragma unroll
        for (int n = 0; n < ITEMS; n++) {
            const int i = tid + n * NT;
            if (i < TT * 32) reinterpret_cast<float4*>(sk[buf])[i] = pre[n];
            else if (i < 2 * TT * 32) reinterpret_cast<float4*>(sq[buf])[i - TT * 32] = pre[n];
            else if (i < TOTAL) reinterpret_cast<float4*>(sv[buf])[i - 2 * TT * 32] = pre[n];
        }
        if (tid < TT * 3) sg[buf][(tid / 3) * 4 + tid % 3] = pre_g;
    };
    load_tile(0);
    store_tile(0);
    __syncthreads();
    for (int tile = 0; tile < ntiles; tile++) {
        const int cur = tile & 1;
        const int t0 = tile * TT;
        if (tile + 1 < ntiles) load_tile(tile + 1);
        const int nt = min(TT, T - t0);
        for (int j = 0; j < nt; j++) {
            const float4* kj = reinterpret_cast<const float4*>(&sk[cur][j * GDN_D]);
            const float4* qj = reinterpret_cast<const float4*>(&sq[cur][j * GDN_D]);
            const float beta = sg[cur][j * 4], a = sg[cur][j * 4 + 1], kq = sg[cur][j * 4 + 2];
            float4 k4[4], q4[4];
#pragma unroll
            for (int r = 0; r < 4; r++) {
                k4[r] = kj[8 * r + l8];
                q4[r] = qj[8 * r + l8];
            }
            float ks[NC], qs[NC];
#pragma unroll
            for (int c = 0; c < NC; c++) {
                float kk = 0.f, qq = 0.f;
#pragma unroll
                for (int r = 0; r < 4; r++) {
                    kk = fmaf(s[c][r].x, k4[r].x, kk);
                    kk = fmaf(s[c][r].y, k4[r].y, kk);
                    kk = fmaf(s[c][r].z, k4[r].z, kk);
                    kk = fmaf(s[c][r].w, k4[r].w, kk);
                    qq = fmaf(s[c][r].x, q4[r].x, qq);
                    qq = fmaf(s[c][r].y, q4[r].y, qq);
                    qq = fmaf(s[c][r].z, q4[r].z, qq);
                    qq = fmaf(s[c][r].w, q4[r].w, qq);
                }
                ks[c] = kk;
                qs[c] = qq;
            }
#pragma unroll
            for (int m = 1; m < 8; m <<= 1) {
#pragma unroll
                for (int c = 0; c < NC; c++) {
                    ks[c] += __shfl_xor_sync(0xffffffffu, ks[c], m);
                    qs[c] += __shfl_xor_sync(0xffffffffu, qs[c], m);
                }
            }
#pragma unroll
            for (int c = 0; c < NC; c++) {
                const float u = beta * (sv[cur][j * CPB + lcol0 + c] - a * ks[c]);
                const float o = a * qs[c] + kq * u;
#pragma unroll
                for (int r = 0; r < 4; r++) {
                    s[c][r].x = fmaf(a, s[c][r].x, k4[r].x * u);
                    s[c][r].y = fmaf(a, s[c][r].y, k4[r].y * u);
                    s[c][r].z = fmaf(a, s[c][r].z, k4[r].z * u);
                    s[c][r].w = fmaf(a, s[c][r].w, k4[r].w * u);
                }
                if (l8 == 0) sot[cur][j * CPB + lcol0 + c] = o;
            }
        }
        if (tile + 1 < ntiles) store_tile(cur ^ 1);
        __syncthreads();
        for (int i = tid; i < TT * CPB; i += NT) {
            const int j = i / CPB, c = i % CPB;
            if (j < nt) so[((size_t)(t0 + j) * n_v + h) * GDN_D + c0 + c] = sot[cur][j * CPB + c];
        }
    }
#pragma unroll
    for (int c = 0; c < NC; c++) {
        float4* s_ptr = reinterpret_cast<float4*>(S + ((size_t)h * GDN_D + c0 + lcol0 + c) * GDN_D);
#pragma unroll
        for (int r = 0; r < 4; r++) s_ptr[8 * r + l8] = s[c][r];
    }
    if (blockIdx.y == 0) {
        const int kh = tiled ? (h % n_k) : (h / (n_v / n_k));
        const int key_dim = n_k * GDN_D;
        for (int tc = tid; tc < 3 * GDN_D; tc += NT) {
            const int part = tc / GDN_D, i = tc % GDN_D;
            const int ch = part == 0 ? kh * GDN_D + i : (part == 1 ? key_dim + kh * GDN_D + i : 2 * key_dim + h * GDN_D + i);
            float* st = conv_state + (size_t)h * (conv_k - 1) * 384 + tc;
            for (int j = 0; j < conv_k - 1; j++) {
                const int sidx = T - (conv_k - 1) + j;
                st[j * 384] = sidx >= 0 ? mixed[(size_t)sidx * conv_dim + ch] : st[(conv_k - 1 + sidx) * 384];
            }
        }
    }
}

#define NC_KERNEL(NAME, CPB, NC, TT, NTH)                                                                                                                                                         \
    extern "C" __global__ void __launch_bounds__(NTH) NAME(GDN_SCAN_ARGS) { gdn_scan_nc_impl<CPB, NC, TT>(pq, pk, pv, pg, mixed, conv_state, S, so, T, n_k, n_v, tiled, conv_k, conv_dim); }
NC_KERNEL(nc_c32_n1, 32, 1, 8, 256)
NC_KERNEL(nc_c32_n2, 32, 2, 8, 128)
NC_KERNEL(nc_c32_n4, 32, 4, 8, 64)
NC_KERNEL(nc_c16_n2, 16, 2, 8, 64)
NC_KERNEL(nc_c16_n1, 16, 1, 8, 128)
NC_KERNEL(nc_c64_n2, 64, 2, 8, 256)
NC_KERNEL(nc_c64_n4, 64, 4, 8, 128)
NC_KERNEL(nc_c32_n2_t16, 32, 2, 16, 128)
NC_KERNEL(nc_c32_n4_t16, 32, 4, 16, 64)
NC_KERNEL(nc_c128_n4, 128, 4, 8, 256)
NC_KERNEL(nc_c128_n2, 128, 2, 8, 512)
NC_KERNEL(nc_c64_n4_t4, 64, 4, 4, 128)
NC_KERNEL(nc_c16_n4, 16, 4, 8, 32)

// ---- host -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
static uint64_t rng_s = 88172645463325252ull;
static uint32_t rnd32() { rng_s ^= rng_s << 13; rng_s ^= rng_s >> 7; rng_s ^= rng_s << 17; return (uint32_t)(rng_s >> 16); }
static float rndf() { return ((rnd32() & 0xFFFFFF) / 16777216.0f) * 2.f - 1.f; }

typedef void (*Kern)(const float*, const float*, const float*, const float*, const float*, float*, float*, float*, int, int, int, int, int, int);

int main(int argc, char** argv) {
    const int T = argc > 1 ? atoi(argv[1]) : 1024;
    const int n_v = argc > 2 ? atoi(argv[2]) : 32, n_k = n_v / 2, conv_k = 4, tiled = 1;
    const int key_dim = n_k * GDN_D, value_dim = n_v * GDN_D, conv_dim = 2 * key_dim + value_dim;
    printf("gdn scan spike: T %d, %d value heads, %d key heads, conv_dim %d\n", T, n_v, n_k, conv_dim);
    std::vector<float> pq((size_t)T * n_v * GDN_D), pk(pq.size()), pv(pq.size()), pg((size_t)T * n_v * 3), mixed((size_t)T * conv_dim), S0((size_t)n_v * GDN_D * GDN_D), cs0((size_t)n_v * (conv_k - 1) * 384);
    for (size_t i = 0; i < (size_t)T * n_v; i++) {
        float nq = 0, nk = 0;
        for (int d = 0; d < GDN_D; d++) { pq[i * GDN_D + d] = rndf(); pk[i * GDN_D + d] = rndf(); nq += pq[i * GDN_D + d] * pq[i * GDN_D + d]; nk += pk[i * GDN_D + d] * pk[i * GDN_D + d]; }
        for (int d = 0; d < GDN_D; d++) { pq[i * GDN_D + d] *= 0.08838834764831845f / sqrtf(nq); pk[i * GDN_D + d] /= sqrtf(nk); pv[i * GDN_D + d] = rndf() * 2.f; }
        float kq = 0;
        for (int d = 0; d < GDN_D; d++) kq += pq[i * GDN_D + d] * pk[i * GDN_D + d];
        pg[i * 3 + 0] = 0.2f + 0.7f * (rndf() * 0.5f + 0.5f);
        pg[i * 3 + 1] = 0.9f + 0.099f * (rndf() * 0.5f + 0.5f);
        pg[i * 3 + 2] = kq;
    }
    for (auto& x : mixed) x = rndf();
    for (auto& x : S0) x = rndf() * 0.05f;
    for (auto& x : cs0) x = rndf();
    float *d_pq, *d_pk, *d_pv, *d_pg, *d_mixed, *d_S, *d_cs, *d_so;
    CK(cudaMalloc(&d_pq, pq.size() * 4)); CK(cudaMalloc(&d_pk, pk.size() * 4)); CK(cudaMalloc(&d_pv, pv.size() * 4)); CK(cudaMalloc(&d_pg, pg.size() * 4));
    CK(cudaMalloc(&d_mixed, mixed.size() * 4)); CK(cudaMalloc(&d_S, S0.size() * 4)); CK(cudaMalloc(&d_cs, cs0.size() * 4)); CK(cudaMalloc(&d_so, pq.size() * 4));
    CK(cudaMemcpy(d_pq, pq.data(), pq.size() * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(d_pk, pk.data(), pk.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_pv, pv.data(), pv.size() * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(d_pg, pg.data(), pg.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_mixed, mixed.data(), mixed.size() * 4, cudaMemcpyHostToDevice));
    struct Entry { const char* name; Kern f; int cpb; int threads; } entries[] = {
        {"engine _32", gdn_scan_kernel_32, 32, 256}, {"engine _64", gdn_scan_kernel_64, 64, 512}, {"engine _128", gdn_scan_kernel_128, 128, 1024},
        {"engine nc_16", gdn_scan_nc_16, 16, 128}, {"engine nc_32", gdn_scan_nc_32, 32, 128}, {"engine nc_64", gdn_scan_nc_64, 64, 128},
        {"nc c32 n1", nc_c32_n1, 32, 256}, {"nc c32 n2", nc_c32_n2, 32, 128}, {"nc c32 n4", nc_c32_n4, 32, 64}, {"nc c16 n1", nc_c16_n1, 16, 128}, {"nc c16 n2", nc_c16_n2, 16, 64},
        {"nc c64 n2", nc_c64_n2, 64, 256}, {"nc c64 n4", nc_c64_n4, 64, 128}, {"nc c32 n2 tt16", nc_c32_n2_t16, 32, 128}, {"nc c32 n4 tt16", nc_c32_n4_t16, 32, 64},
        {"nc c128 n4", nc_c128_n4, 128, 256}, {"nc c128 n2", nc_c128_n2, 128, 512}, {"nc c64 n4 tt4", nc_c64_n4_t4, 64, 128}, {"nc c16 n4", nc_c16_n4, 16, 32}};
    std::vector<float> ref_so(pq.size()), got(pq.size()), ref_S(S0.size()), got_S(S0.size()), ref_cs(cs0.size()), got_cs(cs0.size());
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    for (size_t ei = 0; ei < sizeof(entries) / sizeof(entries[0]); ei++) {
        const Entry& en = entries[ei];
        cudaFuncAttributes fa;
        cudaFuncGetAttributes(&fa, (const void*)en.f);
        dim3 grid(n_v, GDN_D / en.cpb);
        auto launch = [&]() { en.f<<<grid, en.threads>>>(d_pq, d_pk, d_pv, d_pg, d_mixed, d_cs, d_S, d_so, T, n_k, n_v, tiled, conv_k, conv_dim); };
        CK(cudaMemcpy(d_S, S0.data(), S0.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_cs, cs0.data(), cs0.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemset(d_so, 0, pq.size() * 4));
        launch();
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(got.data(), d_so, got.size() * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(got_S.data(), d_S, got_S.size() * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(got_cs.data(), d_cs, got_cs.size() * 4, cudaMemcpyDeviceToHost));
        if (ei == 0) { ref_so = got; ref_S = got_S; ref_cs = got_cs; }
        size_t bad = 0, bad_s = 0, bad_c = 0;
        double worst = 0;
        for (size_t i = 0; i < got.size(); i++) { if (memcmp(&got[i], &ref_so[i], 4)) bad++; worst = std::max(worst, (double)fabsf(got[i] - ref_so[i])); }
        for (size_t i = 0; i < got_S.size(); i++) if (memcmp(&got_S[i], &ref_S[i], 4)) bad_s++;
        for (size_t i = 0; i < got_cs.size(); i++) if (memcmp(&got_cs[i], &ref_cs[i], 4)) bad_c++;
        float best = 1e30f;
        for (int rep = 0; rep < 8; rep++) {
            cudaEventRecord(e0);
            for (int i = 0; i < 3; i++) launch();
            cudaEventRecord(e1);
            cudaEventSynchronize(e1);
            float ms;
            cudaEventElapsedTime(&ms, e0, e1);
            best = std::min(best, ms / 3);
        }
        printf("%-16s grid (%d,%d) x %4d thr, %3d regs, spill %3zu B: %7.1f us   differing words: out %zu, S %zu, conv %zu (max |diff| %.2e)\n", en.name, grid.x, grid.y, en.threads, fa.numRegs, fa.localSizeBytes, best * 1e3, bad, bad_s, bad_c, worst);
    }
    return 0;
}
