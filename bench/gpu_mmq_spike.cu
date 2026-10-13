// Spike: how fast can an int8 tensor-core GEMM loop run on this GPU in the structure of kernels/gemm_s80.cu (128 x 128 tiles, 8 warps of 64 x 32, three-stage cp.async pipeline, ldmatrix fragments)?
//
//   PLANES = 1: one int8 weight plane x int8 activations (m16n8k32, int32 accumulators): the ceiling of any int8 prefill kernel in this structure, without decoding and without epilogue.
//   PLANES = 2: the weights as two int8 planes (hi, lo) with two accumulator sets, S = 32 S_hi + S_lo: the scheme in which the 6-bit sub-block scales of a Q4_K matrix are folded into the weights so that the
//               accumulators need no float conversion inside a super-block. Two MMAs per k32 slab: the ceiling is half of PLANES = 1.
//
// The numbers are weight-MACs per second (M N K / time), to be compared with the fused fp16 kernel (about 12.5 TMAC/s = 25 TFLOP/s on the RTX 4060 for the Q4_K shapes of the 4B).
//   nvcc -O3 -arch=sm_89 -o mmq_spike bench/gpu_mmq_spike.cu && ./mmq_spike
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(1); } } while (0)

__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src) { asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst), "l"(src)); }
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }
__device__ __forceinline__ void ldsm_x4(uint32_t (&r)[4], uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}
__device__ __forceinline__ void mma_s8(int (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

constexpr int BM = 128, BN = 128, BKB = 64;   // BKB: bytes (int8 values) of K per stage and row = two k32 slabs
constexpr int TILE_BYTES = 128 * BKB;           // one operand tile of a stage: 8 KB

// C[n][m] = sum_k W[n][k] X[m][k] (W: weights [N][K], X: activations [M][K], both int8, K contiguous). PLANES 2: W = 32 W_hi + W_lo with W_hi in `W` and W_lo in `W2`.
template <int PLANES, int STAGES>
__global__ void __launch_bounds__(256, 1) mmq(const int8_t* __restrict__ W, const int8_t* __restrict__ W2, const int8_t* __restrict__ X, int* __restrict__ C, int N, int M, int K) {
    extern __shared__ __align__(16) unsigned char smem_raw[];
    constexpr int STAGE_BYTES = (PLANES + 1) * TILE_BYTES;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = (warp >> 2) * 64, wn = (warp & 3) * 32;
    const int n0 = blockIdx.x * BM, m0 = blockIdx.y * BN;

    int acc[4][4][4];
    int acc2[PLANES == 2 ? 4 : 1][PLANES == 2 ? 4 : 1][PLANES == 2 ? 4 : 1];
#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
        for (int j = 0; j < 4; j++)
#pragma unroll
            for (int e = 0; e < 4; e++) acc[i][j][e] = 0;
    if constexpr (PLANES == 2) {
#pragma unroll
        for (int i = 0; i < 4; i++)
#pragma unroll
            for (int j = 0; j < 4; j++)
#pragma unroll
                for (int e = 0; e < 4; e++) acc2[i][j][e] = 0;
    }

    // global -> shared: two 16-byte chunks per operand tile and thread (a row has four chunks, XOR-swizzled as in gemm_s80.cu)
    const int8_t* pw[2];
    const int8_t* pw2[2];
    const int8_t* px[2];
    int soff[2];
#pragma unroll
    for (int u = 0; u < 2; u++) {
        const int c = tid + 256 * u, row = c >> 2, ch = c & 3;
        pw[u] = W + (size_t)(n0 + row) * K + ch * 16;
        pw2[u] = PLANES == 2 ? W2 + (size_t)(n0 + row) * K + ch * 16 : nullptr;
        px[u] = X + (size_t)(m0 + row) * K + ch * 16;
        soff[u] = row * BKB + ((ch ^ ((row >> 1) & 3)) * 16);
    }
    const uint32_t smem0 = smem_u32(smem_raw);
    auto load_stage = [&](int stage, int kt) {
        const uint32_t s0 = smem0 + stage * STAGE_BYTES;
        const size_t k0 = (size_t)kt * BKB;
#pragma unroll
        for (int u = 0; u < 2; u++) {
            cp_async16(s0 + soff[u], pw[u] + k0);
            if constexpr (PLANES == 2) cp_async16(s0 + TILE_BYTES + soff[u], pw2[u] + k0);
            cp_async16(s0 + PLANES * TILE_BYTES + soff[u], px[u] + k0);
        }
    };

    const int q = lane >> 3, lr = lane & 7, swz = (lr >> 1) & 3;
    const uint32_t a_base = (uint32_t)((wm + lr + (q & 1) * 8) * BKB);
    const uint32_t b_base = (uint32_t)((wn + lr + (q >> 1) * 8) * BKB);
    constexpr uint32_t FRAG = 16 * BKB;   // bytes between consecutive 16-row fragments

    const int KT = K / BKB;
#pragma unroll
    for (int s = 0; s < STAGES - 1; s++) {
        if (s < KT) load_stage(s, s);
        cp_commit();
    }
    for (int kt = 0; kt < KT; kt++) {
        cp_wait<STAGES - 2>();
        __syncthreads();
        {
            const int nk = kt + STAGES - 1;
            if (nk < KT) load_stage(nk % STAGES, nk);
            cp_commit();
        }
        const uint32_t sa = smem0 + (kt % STAGES) * STAGE_BYTES;
        const uint32_t sb = sa + PLANES * TILE_BYTES;
#pragma unroll
        for (int ks = 0; ks < 2; ks++) {
            uint32_t fa[4][4], fa2[PLANES == 2 ? 4 : 1][4], fb[2][4];
#pragma unroll
            for (int i = 0; i < 4; i++) {
                const uint32_t off = a_base + (uint32_t)(((2 * ks + (q >> 1)) ^ swz) * 16) + i * FRAG;
                ldsm_x4(fa[i], sa + off);
                if constexpr (PLANES == 2) ldsm_x4(fa2[i], sa + TILE_BYTES + off);
            }
#pragma unroll
            for (int j = 0; j < 2; j++) ldsm_x4(fb[j], sb + b_base + (uint32_t)(((2 * ks + (q & 1)) ^ swz) * 16) + j * FRAG);
#pragma unroll
            for (int i = 0; i < 4; i++)
#pragma unroll
                for (int j = 0; j < 2; j++) {
                    mma_s8(acc[i][2 * j], fa[i], fb[j][0], fb[j][1]);
                    mma_s8(acc[i][2 * j + 1], fa[i], fb[j][2], fb[j][3]);
                    if constexpr (PLANES == 2) {
                        mma_s8(acc2[i][2 * j], fa2[i], fb[j][0], fb[j][1]);
                        mma_s8(acc2[i][2 * j + 1], fa2[i], fb[j][2], fb[j][3]);
                    }
                }
        }
    }
    // result: S = 32 acc + acc2 (or acc); stored as [n][m]
    const int g = lane >> 2, t = lane & 3;
#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
        for (int nt = 0; nt < 4; nt++)
#pragma unroll
            for (int h = 0; h < 2; h++) {
                const int n = n0 + wm + 16 * i + g + 8 * h, m = m0 + wn + 8 * nt + 2 * t;
                int v0 = acc[i][nt][2 * h], v1 = acc[i][nt][2 * h + 1];
                if constexpr (PLANES == 2) {
                    v0 = 32 * v0 + acc2[i][nt][2 * h];
                    v1 = 32 * v1 + acc2[i][nt][2 * h + 1];
                }
                C[(size_t)n * M + m] = v0;
                C[(size_t)n * M + m + 1] = v1;
            }
}

template <int PLANES, int STAGES>
double run(const int8_t* W, const int8_t* W2, const int8_t* X, int* C, int N, int M, int K, int reps) {
    constexpr int SMEM = STAGES * (PLANES + 1) * TILE_BYTES;
    auto kern = mmq<PLANES, STAGES>;
    CK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
    dim3 grid(N / BM, M / BN);
    kern<<<grid, 256, SMEM>>>(W, W2, X, C, N, M, K);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    cudaEventRecord(a);
    for (int r = 0; r < reps; r++) kern<<<grid, 256, SMEM>>>(W, W2, X, C, N, M, K);
    cudaEventRecord(b);
    CK(cudaEventSynchronize(b));
    float ms;
    cudaEventElapsedTime(&ms, a, b);
    return ms / reps;
}

int main(int argc, char** argv) {
    const int N = 8192, M = 2048, K = 4096;
    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, 0);
    printf("%s, %d SMs, sm_%d%d\n", p.name, p.multiProcessorCount, p.major, p.minor);
    std::vector<int8_t> hw((size_t)N * K), hw2((size_t)N * K), hx((size_t)M * K);
    uint32_t s = 12345;
    auto rnd = [&]() { s = s * 1664525u + 1013904223u; return (int)(s >> 24); };
    for (auto& v : hw) v = (int8_t)(rnd() % 61 - 30);     // "hi" plane: small values
    for (auto& v : hw2) v = (int8_t)(rnd() % 32);          // "lo" plane: 0..31
    for (auto& v : hx) v = (int8_t)(rnd() - 128);
    int8_t *dw, *dw2, *dx;
    int* dc;
    CK(cudaMalloc(&dw, hw.size()));
    CK(cudaMalloc(&dw2, hw2.size()));
    CK(cudaMalloc(&dx, hx.size()));
    CK(cudaMalloc(&dc, (size_t)N * M * 4));
    CK(cudaMemcpy(dw, hw.data(), hw.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dw2, hw2.data(), hw2.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dx, hx.data(), hx.size(), cudaMemcpyHostToDevice));

    // correctness of the fragment layout on a few output elements (PLANES = 1 first, then 2)
    for (int planes = 1; planes <= 2; planes++) {
        if (planes == 1) run<1, 3>(dw, dw2, dx, dc, N, M, K, 1);
        else run<2, 2>(dw, dw2, dx, dc, N, M, K, 1);
        std::vector<int> hc((size_t)N * M);
        CK(cudaMemcpy(hc.data(), dc, hc.size() * 4, cudaMemcpyDeviceToHost));
        int bad = 0;
        for (int tcase = 0; tcase < 64; tcase++) {
            const int n = (rnd() * 31 + tcase * 977) % N, m = (rnd() * 17 + tcase * 331) % M;
            long long ref = 0;
            for (int k = 0; k < K; k++) {
                const int w = planes == 1 ? hw[(size_t)n * K + k] : 32 * hw[(size_t)n * K + k] + hw2[(size_t)n * K + k];
                ref += (long long)w * hx[(size_t)m * K + k];
            }
            if (ref != hc[(size_t)n * M + m]) bad++;
        }
        printf("PLANES %d: %d of 64 sampled outputs differ from the reference\n", planes, bad);
    }
    const double macs = (double)M * N * K;
    for (int rep = 0; rep < 2; rep++) {
        double t1 = run<1, 3>(dw, dw2, dx, dc, N, M, K, 20);
        printf("PLANES 1, 3 stages: %.3f ms = %.1f TMAC/s of int8 (%.1f TOPS)\n", t1, macs / t1 / 1e9, 2 * macs / t1 / 1e9);
        double t2 = run<2, 2>(dw, dw2, dx, dc, N, M, K, 20);
        printf("PLANES 2, 2 stages: %.3f ms = %.1f TMAC/s of WEIGHT MACs (two MMAs each; %.1f TOPS issued)\n", t2, macs / t2 / 1e9, 4 * macs / t2 / 1e9);
    }
    printf("for comparison, the fused fp16 kernel of the engine reaches about 12.5 TMAC/s on the Q4_K shapes of the 4B\n");
    return 0;
}
