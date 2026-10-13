// KV-cache decoding of a Forge-1 TRAINING checkpoint: one new token per step (batch 1), bitwise equal to the trainer's own forward pass (crates/forge-train/src/decode.rs drives these kernels).
//
// Why bitwise: the trainer's fp16 forward is the reference (the model as trained and evaluated), so every kernel here repeats its arithmetic for one row instead of approximating it:
//  - matrix-vector products issue the SAME tensor-core chain along K as gemm.cu (mma.m16n8k8, k8 steps in increasing order, accumulator starting at 0) or, when the trainer runs the Ampere GEMMs, as
//    gemm_s80.cu (mma.m16n8k16 on sm_80+; below sm_80 that kernel's emulation is the same two k8 steps). The activation is row 0 of the A operand and rows 1-15 are zero: every output element of an mma
//    depends only on its own row of A and column of B, so row 0 comes out exactly as in the big GEMM. The epilogues are those of gemm.cu's store_acc (fp16 store, fp32 store, fp32 + residual);
//  - RMSNorm, the layer-input mixing (lambdas / skips), qk-norm + RoPE, value embeddings and ReLU^2 use the expressions of train.cu / train_f1.cu in the same order;
//  - attention repeats flash.cu's tile loop for a single query at position `pos` (keys in tiles of 64 from position 0, online softmax, P rounded to fp16, as in training): flash_fwd_impl itself, a
//    per-KV-head version with the query heads as mma rows, and a split version for long contexts (below).
// The weights are the trainer's fp16 working copy ([out][in] row-major); nothing is repacked.
// The token and position of the step live in device memory (F1Step), so the whole step (embedding, 5 kernels per layer, final norm, head) is captured once as a CUDA graph and replayed.
//
// Matrix-vector kernel: one warp computes 8 output columns (one n8 tile) over the whole K. A lane loads 16 bytes of its column's weight row per 32 k (the four lanes of a quad cover 64 contiguous bytes),
// eight 32-k blocks per batch go through shared memory (XOR-swizzled, conflict free) and come back with ldmatrix, which delivers exactly the B fragments of four consecutive k8 steps; the next batch is
// loading while one is consumed (4 KB in flight per warp): with few columns (576 outputs = 72 warps) the bandwidth comes from the depth of the prefetch, not from the number of warps.
// (Measured on a T4: loading the B fragments directly as 32-bit words, without shared memory, was 2x SLOWER: four times the L1 wavefronts.)

#include <cuda_fp16.h>
#include <stdint.h>

namespace f1d {

struct F1Step {
    uint32_t token;
    uint32_t pos;
};

__device__ __forceinline__ uint32_t smem_addr(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }

__device__ __forceinline__ void ldsm4(uint32_t (&r)[4], uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}

__device__ __forceinline__ void sts128(uint32_t addr, const uint4& v) {
    asm volatile("st.shared.v4.u32 [%0], {%1, %2, %3, %4};\n" ::"r"(addr), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w));
}

// one 16-k step of row 0: (a0: k 0-7 pair of this lane, a2: k 8-15 pair; zero for lanes of rows 1-15) x (b0, b1)
template <bool K16>
__device__ __forceinline__ void mma16x(float (&c)[4], uint32_t a0, uint32_t a2, uint32_t b0, uint32_t b1) {
    if (K16) {
        const uint32_t a[4] = {a0, 0u, a2, 0u};
        gemm80::mma16(c, a, b0, b1);
    } else {
        gemm::mma_k8(c, a0, 0u, b0);
        gemm::mma_k8(c, a2, 0u, b1);
    }
}

constexpr int GV_U = 8;   // 32-k blocks per batch: one batch is consumed while the next one is loading (4 KB in flight per warp)

// The first batch of this lane's weight row (issued before the prologue so that its latency overlaps it). A lane holds 16-byte chunk t of row n0 + g of every 32-k block.
__device__ __forceinline__ void gemv_prefetch(const half* __restrict__ W, int K, int n0, uint4 (&cur)[GV_U]) {
    const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const uint4* src = reinterpret_cast<const uint4*>(W + (size_t)(n0 + g) * K) + t;
    const int KB = K >> 5;
#pragma unroll
    for (int u = 0; u < GV_U; u++) cur[u] = u < KB ? __ldg(src + 4 * u) : make_uint4(0u, 0u, 0u, 0u);
}

// The k-chain of one n8 tile: acc (row 0 in c[0], c[1] of lanes 0-3) = sum over k of x[k] * W[n0 + col][k], mma by mma in increasing k as in the GEMM. xs32: the activation as half2 words in shared
// memory; stage: this warp's GV_U x 512 bytes; cur: the prefetched first batch. A batch of 8 x 32 tiles goes to shared memory (XOR-swizzled, conflict free) with one barrier, ldmatrix reads back the
// B fragments (four k8 steps per x4), and the chain runs over the batch without waiting for shared memory in between.
//   BATCHED = false: the first version, one tile through shared memory at a time (two barriers per 32 k).
template <bool K16, bool BATCHED>
__device__ __forceinline__ void gemv_tile(const half* __restrict__ W, int K, int n0, const uint32_t* __restrict__ xs32, uint32_t stage, uint4 (&cur)[GV_U], float (&acc)[4]) {
    const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const uint4* src = reinterpret_cast<const uint4*>(W + (size_t)(n0 + g) * K) + t;   // 16-byte chunk t of 32-k block kb is src[4 kb]
    const int KB = K >> 5;
    const uint32_t sts_at = stage + (uint32_t)(g * 64 + ((t ^ ((g >> 1) & 3)) * 16));
    const int lr = lane & 7, lq = lane >> 3;
    const uint32_t ld_at = stage + (uint32_t)(lr * 64 + ((lq ^ ((lr >> 1) & 3)) * 16));
    const uint32_t am = g == 0 ? 0xffffffffu : 0u;
    acc[0] = acc[1] = acc[2] = acc[3] = 0.f;
    for (int kb0 = 0; kb0 < KB; kb0 += GV_U) {
        uint4 nxt[GV_U];
#pragma unroll
        for (int u = 0; u < GV_U; u++) {
            const int kb = kb0 + GV_U + u;
            nxt[u] = kb < KB ? __ldg(src + 4 * kb) : make_uint4(0u, 0u, 0u, 0u);
        }
        if (BATCHED) {
#pragma unroll
            for (int u = 0; u < GV_U; u++) sts128(sts_at + u * 512, cur[u]);
            __syncwarp();
            uint32_t r[GV_U][4];
#pragma unroll
            for (int u = 0; u < GV_U; u++) ldsm4(r[u], ld_at + u * 512);
#pragma unroll
            for (int u = 0; u < GV_U; u++) {
                const int kb = kb0 + u;
                if (kb < KB) {
                    const uint32_t* xa = xs32 + kb * 16 + t;
                    const uint32_t a0 = xa[0] & am, a1 = xa[4] & am, a2 = xa[8] & am, a3 = xa[12] & am;
                    mma16x<K16>(acc, a0, a1, r[u][0], r[u][1]);
                    mma16x<K16>(acc, a2, a3, r[u][2], r[u][3]);
                }
            }
            __syncwarp();
        } else {
#pragma unroll
            for (int u = 0; u < GV_U; u++) {
                const int kb = kb0 + u;
                if (kb < KB) {
                    sts128(sts_at, cur[u]);
                    __syncwarp();
                    uint32_t r[4];
                    ldsm4(r, ld_at);
                    const uint32_t* xa = xs32 + kb * 16 + t;
                    const uint32_t a0 = xa[0] & am, a1 = xa[4] & am, a2 = xa[8] & am, a3 = xa[12] & am;
                    mma16x<K16>(acc, a0, a1, r[0], r[1]);
                    mma16x<K16>(acc, a2, a3, r[2], r[3]);
                    __syncwarp();
                }
            }
        }
#pragma unroll
        for (int u = 0; u < GV_U; u++) cur[u] = nxt[u];
    }
}

// rmsnorm_fwd_kernel (train.cu) for one row, by ONE warp: x (fp32, E floats, 16-byte aligned) -> h (fp16, shared memory)
__device__ __forceinline__ void rmsnorm_row_warp(const float* __restrict__ x, const float* __restrict__ w, half* __restrict__ h, int E, float eps, int lane) {
    const float4* xr = reinterpret_cast<const float4*>(x);
    const int E4 = E >> 2;
    float ss = 0.f;
    for (int i = lane; i < E4; i += 32) {
        const float4 v = xr[i];
        ss += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }
    ss = warp_sum(ss);
    const float r = rsqrtf(ss / (float)E + eps);
    const float4* w4 = reinterpret_cast<const float4*>(w);
    uint2* hr = reinterpret_cast<uint2*>(h);
    for (int i = lane; i < E4; i += 32) {
        const float4 v = xr[i], g = w4[i];
        const __half2 a = __floats2half2_rn(v.x * r * g.x, v.y * r * g.y), b = __floats2half2_rn(v.z * r * g.z, v.w * r * g.w);
        hr[i] = make_uint2(*reinterpret_cast<const uint32_t*>(&a), *reinterpret_cast<const uint32_t*>(&b));
    }
}

// prologues: what the A operand (row 0) is
constexpr int PRO_VEC = 0;     // an fp16 vector of K values in global memory (attention output, MLP activation)
constexpr int PRO_NORM = 1;    // RMSNorm (weights nw) of an fp32 vector x of K = E values
constexpr int PRO_MIXNORM = 2; // the layer-input mixing  x = a * xprev + b * x0 + c * xskip  (mix_fwd_kernel; block 0 stores x to xout), then RMSNorm
// epilogues (gemm.cu store_acc): fp16, fp32, fp32 + residual R, and fp16 followed by relu2_kernel
constexpr int EPI_F16 = 0, EPI_F32 = 1, EPI_RES = 4, EPI_RELU2 = 5;

template <int PRO, int EPI, bool K16, bool BATCHED>
__device__ __forceinline__ void gemv_impl(const half* __restrict__ W, int N, int K, const half* __restrict__ a16, const float* __restrict__ xin, const float* __restrict__ x0, const float* __restrict__ xskip,
                                          const float* __restrict__ sc, int use_ab, int use_c, float* __restrict__ xout, const float* __restrict__ nw, float eps, void* __restrict__ out,
                                          const float* __restrict__ R) {
    extern __shared__ __align__(16) unsigned char smem[];
    const int nwarps = blockDim.x >> 5, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    half* xs = reinterpret_cast<half*>(smem);                                          // K halves
    float* xm = reinterpret_cast<float*>(smem + (size_t)K * 2);                         // K floats (PRO_NORM / PRO_MIXNORM)
    unsigned char* stg = smem + (size_t)K * 2 + (PRO == PRO_VEC ? 0 : (size_t)K * 4);   // (BATCHED ? GV_U : 1) x 512 bytes per warp
    const int n0 = (blockIdx.x * nwarps + warp) * 8;
    uint4 cur[GV_U];
    if (n0 < N) gemv_prefetch(W, K, n0, cur);
    if (PRO == PRO_VEC) {
        const uint4* s = reinterpret_cast<const uint4*>(a16);
        for (int i = threadIdx.x; i < (K >> 3); i += blockDim.x) reinterpret_cast<uint4*>(xs)[i] = s[i];
        __syncthreads();
    } else {
        const float* xv = xin;
        if (PRO == PRO_MIXNORM) {
            const float a = use_ab ? sc[0] : 1.f, b = use_ab ? sc[1] : 0.f, c = use_c ? sc[2] : 0.f;
            for (int i = threadIdx.x; i < K; i += blockDim.x) {
                float o = a * xin[i];
                if (use_ab) o += b * x0[i];
                if (use_c) o += c * xskip[i];
                xm[i] = o;
                if (blockIdx.x == 0) xout[i] = o;
            }
            __syncthreads();
            xv = xm;
        }
        if (warp == 0) rmsnorm_row_warp(xv, nw, xs, K, eps, lane);
        __syncthreads();
    }
    if (n0 >= N) return;
    float acc[4];
    gemv_tile<K16, BATCHED>(W, K, n0, reinterpret_cast<const uint32_t*>(xs), smem_addr(stg + warp * (BATCHED ? GV_U * 512 : 512)), cur, acc);
    const int g = lane >> 2, t = lane & 3;
    if (g != 0) return;
    const int col = n0 + 2 * t;
    const float v0 = acc[0], v1 = acc[1];
    if (EPI == EPI_F16) {
        *reinterpret_cast<half2*>(reinterpret_cast<half*>(out) + col) = __floats2half2_rn(v0, v1);
    } else if (EPI == EPI_F32) {
        *reinterpret_cast<float2*>(reinterpret_cast<float*>(out) + col) = make_float2(v0, v1);
    } else if (EPI == EPI_RES) {
        const float2 r = *reinterpret_cast<const float2*>(R + col);
        *reinterpret_cast<float2*>(reinterpret_cast<float*>(out) + col) = make_float2(v0 + r.x, v1 + r.y);
    } else {
        const half2 u = __floats2half2_rn(v0, v1);
        const float r0 = fmaxf(__low2float(u), 0.f), r1 = fmaxf(__high2float(u), 0.f);
        half* o = reinterpret_cast<half*>(out) + col;
        o[0] = __float2half(fminf(r0 * r0, 65000.f));
        o[1] = __float2half(fminf(r1 * r1, 65000.f));
    }
}

// Warp-specialised variant (not the default: 5 % slower than the per-warp kernel on the T4; every block repeats the norm prologue): ONE n8 tile per block of four warps. All four warps load the weight tile, sixteen 32-k blocks (8 KB) per stage, into a double-buffered shared-memory ring while warp 0
// runs the mma chain over the previous stage. The chain is the same; what changes is how much is in flight: 8 KB per output tile instead of 4 KB per warp, and with N / 8 blocks the small matrices
// (576 outputs = 72 blocks) spread over every multiprocessor.
constexpr int WS_SB = 16;            // 32-k blocks per stage
constexpr int WS_V = WS_SB * 32 / 128; // 16-byte chunks per thread per stage

__device__ __forceinline__ void ws_load(const half* __restrict__ W, int K, int n0, int kb0, int KB, uint4 (&r)[WS_V]) {
#pragma unroll
    for (int v = 0; v < WS_V; v++) {
        const int i = threadIdx.x + 128 * v;
        const int u = i >> 5, g = (i >> 2) & 7, t = i & 3;
        const int kb = kb0 + u;
        r[v] = kb < KB ? __ldg(reinterpret_cast<const uint4*>(W + (size_t)(n0 + g) * K + (size_t)kb * 32) + t) : make_uint4(0u, 0u, 0u, 0u);
    }
}

__device__ __forceinline__ void ws_store(uint32_t buf, const uint4 (&r)[WS_V]) {
#pragma unroll
    for (int v = 0; v < WS_V; v++) {
        const int i = threadIdx.x + 128 * v;
        const int u = i >> 5, g = (i >> 2) & 7, t = i & 3;
        sts128(buf + (uint32_t)(u * 512 + g * 64 + ((t ^ ((g >> 1) & 3)) * 16)), r[v]);
    }
}

template <int PRO, int EPI, bool K16>
__device__ __forceinline__ void gemv_ws_impl(const half* __restrict__ W, int N, int K, const half* __restrict__ a16, const float* __restrict__ xin, const float* __restrict__ x0, const float* __restrict__ xskip,
                                             const float* __restrict__ sc, int use_ab, int use_c, float* __restrict__ xout, const float* __restrict__ nw, float eps, void* __restrict__ out,
                                             const float* __restrict__ R) {
    extern __shared__ __align__(16) unsigned char smem[];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    half* xs = reinterpret_cast<half*>(smem);                                          // K halves
    float* xm = reinterpret_cast<float*>(smem + (size_t)K * 2);                         // K floats (PRO_NORM / PRO_MIXNORM)
    unsigned char* stg = smem + (size_t)K * 2 + (PRO == PRO_VEC ? 0 : (size_t)K * 4);   // two stages of WS_SB x 512 bytes
    const uint32_t buf0 = smem_addr(stg), buf1 = buf0 + WS_SB * 512;
    const int n0 = blockIdx.x * 8;
    const int KB = K >> 5;
    uint4 r[WS_V];
    ws_load(W, K, n0, 0, KB, r);
    if (PRO == PRO_VEC) {
        const uint4* s4 = reinterpret_cast<const uint4*>(a16);
        for (int i = threadIdx.x; i < (K >> 3); i += blockDim.x) reinterpret_cast<uint4*>(xs)[i] = s4[i];
    } else {
        const float* xv = xin;
        if (PRO == PRO_MIXNORM) {
            const float a = use_ab ? sc[0] : 1.f, b = use_ab ? sc[1] : 0.f, c = use_c ? sc[2] : 0.f;
            for (int i = threadIdx.x; i < K; i += blockDim.x) {
                float o = a * xin[i];
                if (use_ab) o += b * x0[i];
                if (use_c) o += c * xskip[i];
                xm[i] = o;
                if (blockIdx.x == 0) xout[i] = o;
            }
            __syncthreads();
            xv = xm;
        }
        if (warp == 0) rmsnorm_row_warp(xv, nw, xs, K, eps, lane);
    }
    ws_store(buf0, r);
    __syncthreads();
    const int g = lane >> 2, t = lane & 3;
    const uint32_t am = g == 0 ? 0xffffffffu : 0u;
    const int lr = lane & 7, lq = lane >> 3;
    const uint32_t ld_off = (uint32_t)(lr * 64 + ((lq ^ ((lr >> 1) & 3)) * 16));
    const uint32_t* xs32 = reinterpret_cast<const uint32_t*>(xs);
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    const int nst = (KB + WS_SB - 1) / WS_SB;
    for (int st = 0; st < nst; st++) {
        const uint32_t cur = (st & 1) ? buf1 : buf0, nxt = (st & 1) ? buf0 : buf1;
        if (st + 1 < nst) ws_load(W, K, n0, (st + 1) * WS_SB, KB, r);
        if (warp == 0) {
#pragma unroll
            for (int u = 0; u < WS_SB; u++) {
                const int kb = st * WS_SB + u;
                if (kb < KB) {
                    uint32_t b[4];
                    ldsm4(b, cur + u * 512 + ld_off);
                    const uint32_t* xa = xs32 + kb * 16 + t;
                    const uint32_t a0 = xa[0] & am, a1 = xa[4] & am, a2 = xa[8] & am, a3 = xa[12] & am;
                    mma16x<K16>(acc, a0, a1, b[0], b[1]);
                    mma16x<K16>(acc, a2, a3, b[2], b[3]);
                }
            }
        }
        if (st + 1 < nst) ws_store(nxt, r);
        __syncthreads();
    }
    if (warp != 0 || g != 0 || n0 >= N) return;
    const int col = n0 + 2 * t;
    const float v0 = acc[0], v1 = acc[1];
    if (EPI == EPI_F16) {
        *reinterpret_cast<half2*>(reinterpret_cast<half*>(out) + col) = __floats2half2_rn(v0, v1);
    } else if (EPI == EPI_F32) {
        *reinterpret_cast<float2*>(reinterpret_cast<float*>(out) + col) = make_float2(v0, v1);
    } else if (EPI == EPI_RES) {
        const float2 rr = *reinterpret_cast<const float2*>(R + col);
        *reinterpret_cast<float2*>(reinterpret_cast<float*>(out) + col) = make_float2(v0 + rr.x, v1 + rr.y);
    } else {
        const half2 u = __floats2half2_rn(v0, v1);
        const float r0 = fmaxf(__low2float(u), 0.f), r1 = fmaxf(__high2float(u), 0.f);
        half* o = reinterpret_cast<half*>(out) + col;
        o[0] = __float2half(fminf(r0 * r0, 65000.f));
        o[1] = __float2half(fminf(r1 * r1, 65000.f));
    }
}

}  // namespace f1d

#define F1D_GEMV1(NAME, PRO, EPI, K16, BATCHED)                                                                                                                                                   \
    extern "C" __global__ void __launch_bounds__(256) NAME(const half* W, int N, int K, const half* a16, const float* xin, const float* x0, const float* xskip, const float* sc, int use_ab,          \
                                                           int use_c, float* xout, const float* nw, float eps, void* out, const float* R) {                                                       \
        f1d::gemv_impl<PRO, EPI, K16, BATCHED>(W, N, K, a16, xin, x0, xskip, sc, use_ab, use_c, xout, nw, eps, out, R);                                                                         \
    }
#define F1D_GEMV_WS(NAME, PRO, EPI, K16)                                                                                                                                                          \
    extern "C" __global__ void __launch_bounds__(128) NAME(const half* W, int N, int K, const half* a16, const float* xin, const float* x0, const float* xskip, const float* sc, int use_ab,          \
                                                           int use_c, float* xout, const float* nw, float eps, void* out, const float* R) {                                                       \
        f1d::gemv_ws_impl<PRO, EPI, K16>(W, N, K, a16, xin, x0, xskip, sc, use_ab, use_c, xout, nw, eps, out, R);                                                                                 \
    }
// every variant three times: the per-tile staging of the first version (suffix _t1, the default: FORGE_KV_GEMV=1), batched staging per warp (no suffix, =3) and warp-specialised (suffix _ws, =0).
// T4, production checkpoint, alternating runs (job kv4): 510-515, 467-469 and 487-490 tokens/s.
#define F1D_GEMV(NAME, PRO, EPI, K16) F1D_GEMV1(NAME, PRO, EPI, K16, true) F1D_GEMV1(NAME##_t1, PRO, EPI, K16, false) F1D_GEMV_WS(NAME##_ws, PRO, EPI, K16)

// layer input (mixing + attention norm) -> q | k | v projection;  norm -> projection without mixing;  attention output -> + residual;  MLP norm -> up projection + relu2;  MLP norm -> gate | up;
// MLP activation -> down projection + residual;  final norm (computed once by rmsnorm_fwd_kernel) -> head (fp32 logits)
F1D_GEMV(f1d_gemv_mixnorm_f16, f1d::PRO_MIXNORM, f1d::EPI_F16, false)
F1D_GEMV(f1d_gemv_norm_f16, f1d::PRO_NORM, f1d::EPI_F16, false)
F1D_GEMV(f1d_gemv_vec_res, f1d::PRO_VEC, f1d::EPI_RES, false)
F1D_GEMV(f1d_gemv_norm_relu2, f1d::PRO_NORM, f1d::EPI_RELU2, false)
F1D_GEMV(f1d_gemv_vec_f32, f1d::PRO_VEC, f1d::EPI_F32, false)
F1D_GEMV(f1d_gemv_mixnorm_f16_k16, f1d::PRO_MIXNORM, f1d::EPI_F16, true)
F1D_GEMV(f1d_gemv_norm_f16_k16, f1d::PRO_NORM, f1d::EPI_F16, true)
F1D_GEMV(f1d_gemv_vec_res_k16, f1d::PRO_VEC, f1d::EPI_RES, true)
F1D_GEMV(f1d_gemv_norm_relu2_k16, f1d::PRO_NORM, f1d::EPI_RELU2, true)
F1D_GEMV(f1d_gemv_vec_f32_k16, f1d::PRO_VEC, f1d::EPI_F32, true)

// embedding row of the step's token (embed_batch_kernel for one row) and the token history the prefill path reads (value embeddings, n-gram hashes)
extern "C" __global__ void f1d_embed(const half* __restrict__ emb, const f1d::F1Step* __restrict__ st, float* __restrict__ x, uint32_t* __restrict__ hist, int E) {
    const size_t row = st->token;
    for (int i = threadIdx.x; i < E; i += blockDim.x) x[i] = __half2float(emb[row * E + i]);
    if (threadIdx.x == 0) hist[st->pos] = st->token;
}

// qkv_rope_train_kernel + ve_fwd_kernel (train.cu / train_f1.cu) for one head of the single row at position pos, by one warp: head hb < n_head is a query head (optional per-head RMSNorm,
// RoPE) written to q_dst (its hd values: global memory or a row of shared memory), then n_kv key heads (-> the key cache) and n_kv value heads (optional value embedding
// v = lv * v + lve * ve[token] -> the value cache).
namespace f1d {
__device__ __forceinline__ void qkv_item(int hb, const half* __restrict__ qkv, half* q_dst, half* __restrict__ k_cache, half* __restrict__ v_cache, const float* __restrict__ cos_t,
                                         const float* __restrict__ sin_t, const float* __restrict__ qn, const float* __restrict__ kn, const half* __restrict__ ve, const float* __restrict__ ve_sc,
                                         uint32_t token, int pos, int n_head, int n_kv, int hd, float eps, int lane) {
    const int q_dim = n_head * hd, kv_dim = n_kv * hd, half_d = hd / 2;
    const half* base = qkv;
    if (hb >= n_head + n_kv) {
        const int head = hb - n_head - n_kv;
        half* vo = v_cache + (size_t)pos * kv_dim;
        if (ve != nullptr) {
            const float lv = ve_sc[0], lve = ve_sc[1];
            const half* e = ve + (size_t)token * kv_dim;
            for (int d = lane; d < hd; d += 32) {
                const int j = head * hd + d;
                const half raw = base[q_dim + kv_dim + j];
                vo[j] = __float2half(lv * __half2float(raw) + lve * __half2float(e[j]));
            }
        } else {
            for (int d = lane; d < hd; d += 32) vo[head * hd + d] = base[q_dim + kv_dim + head * hd + d];
        }
        return;
    }
    const bool is_q = hb < n_head;
    const int head = is_q ? hb : hb - n_head;
    const half* src = base + (is_q ? head * hd : q_dim + head * hd);
    const float* wn = is_q ? qn : kn;
    float a[2], b[2];
    float ss = 0.f;
#pragma unroll
    for (int p = 0; p < 2; p++) {
        if (p * 64 < hd) {
            a[p] = __half2float(src[lane + 32 * p]);
            b[p] = __half2float(src[lane + 32 * p + half_d]);
            ss += a[p] * a[p] + b[p] * b[p];
        }
    }
    float r = 1.f;
    if (wn != nullptr) {
        ss = warp_sum(ss);
        r = rsqrtf(ss / (float)hd + eps);
    }
    half* dst = is_q ? q_dst : k_cache + (size_t)pos * kv_dim + head * hd;
#pragma unroll
    for (int p = 0; p < 2; p++) {
        if (p * 64 < hd) {
            const int i = lane + 32 * p;
            float x1 = a[p], x2 = b[p];
            if (wn != nullptr) {
                x1 *= r * wn[i];
                x2 *= r * wn[i + half_d];
            }
            const float c = cos_t[(size_t)pos * half_d + i], s = sin_t[(size_t)pos * half_d + i];
            dst[i] = __float2half(x1 * c - x2 * s);
            dst[i + half_d] = __float2half(x2 * c + x1 * s);
        }
    }
}
}  // namespace f1d

// all heads of the row, one warp each
extern "C" __global__ void f1d_qkv_post(const half* __restrict__ qkv, half* __restrict__ q_out, half* __restrict__ k_cache, half* __restrict__ v_cache, const float* __restrict__ cos_t,
                                        const float* __restrict__ sin_t, const float* __restrict__ qn, const float* __restrict__ kn, const half* __restrict__ ve, const float* __restrict__ ve_sc,
                                        const f1d::F1Step* __restrict__ st, int n_head, int n_kv, int hd, float eps) {
    const int item = blockIdx.x * 8 + (threadIdx.x >> 5);
    if (item >= n_head + 2 * n_kv) return;
    f1d::qkv_item(item, qkv, q_out + (item < n_head ? item * hd : 0), k_cache, v_cache, cos_t, sin_t, qn, kn, ve, ve_sc, st->token, (int)st->pos, n_head, n_kv, hd, eps, threadIdx.x & 31);
}

// Attention of the single query at position st->pos, ONE block per KV head: the query heads that share it are the rows of one 16-row mma tile (rows are independent in every step of flash.cu's loop,
// so each row gets exactly the arithmetic of the training kernel), and the work of a key tile is split over the four warps instead of being repeated by them:
//   S = Q K^T: warp w computes the key columns 16 w .. 16 w + 15 (the same k8 chain over the head dimension as flash.cu) and writes the scaled, masked scores to shared memory;
//   softmax: every warp reads the whole tile row and repeats flash.cu's running max / correction / row sum (identical values in every warp);
//   O += P V: warp w owns its share of the output dimensions (the chain over the key chunks of a tile and over the tiles is the training kernel's).
// The next key / value tile is loaded into registers while the current one is consumed (double-buffered shared memory). HD = 64 (tiles of 64 keys) or 128 (tiles of 32 keys) as in flash.cu.
namespace f1d {

// FUSED: the block first does f1d_qkv_post's work for its own heads (its query heads into shared memory, its key and value heads into the cache at pos), then attends; the cache is then read with
// coherent loads (this block has just written row pos).
template <int HD, int KT, bool FUSED>
__device__ __forceinline__ void attn_gqa_impl(const half* __restrict__ q, half* __restrict__ k_cache, half* __restrict__ v_cache, half* __restrict__ out, const F1Step* __restrict__ st,
                                              int n_head, int n_kv, const half* __restrict__ qkv, const float* __restrict__ cos_t, const float* __restrict__ sin_t, const float* __restrict__ qn,
                                              const float* __restrict__ kn, const half* __restrict__ ve, const float* __restrict__ ve_sc, float eps, int window) {
    constexpr int LDS = HD + 8;
    constexpr int KS = HD / 8;      // k8 steps over the head dimension
    constexpr int NK = KT / 8;      // n8 key tiles of a key tile
    constexpr int ND = HD / 8;      // n8 tiles of the output
    constexpr int NDW = ND / 4;     // output n8 tiles per warp
    constexpr int CH = KT * HD / 8; // 16-byte chunks of a K (or V) tile
    constexpr int CPT = CH / 128;   // per thread
    __shared__ __align__(16) half Qs[16 * LDS];
    __shared__ __align__(16) half Ks[2][KT * LDS];
    __shared__ __align__(16) half Vs[2][KT * LDS];
    __shared__ float Ss[16][KT + 4];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int g = lane >> 2, t4 = lane & 3;
    const int kvh = blockIdx.x, rep = n_head / n_kv;
    const int kv_dim = n_kv * HD;
    const int pos = (int)st->pos;
    const int kmax = pos;
    // the query rows (heads kvh * rep .. + rep - 1), zero rows below
    if (FUSED) {
        for (int i = tid; i < 16 * HD / 8; i += blockDim.x) {
            const int r = i / (HD / 8), c = i % (HD / 8);
            if (r >= rep) *reinterpret_cast<uint4*>(&Qs[r * LDS + c * 8]) = make_uint4(0u, 0u, 0u, 0u);
        }
        for (int it = warp; it < rep + 2; it += 4) {
            const int hb = it < rep ? kvh * rep + it : (it == rep ? n_head + kvh : n_head + n_kv + kvh);
            qkv_item(hb, qkv, it < rep ? &Qs[it * LDS] : nullptr, k_cache, v_cache, cos_t, sin_t, qn, kn, ve, ve_sc, st->token, pos, n_head, n_kv, HD, eps, lane);
        }
        __syncthreads();
    } else {
        for (int i = tid; i < 16 * HD / 8; i += blockDim.x) {
            const int r = i / (HD / 8), c = i % (HD / 8);
            uint4 v = make_uint4(0u, 0u, 0u, 0u);
            if (r < rep) v = *reinterpret_cast<const uint4*>(q + (size_t)(kvh * rep + r) * HD + c * 8);
            *reinterpret_cast<uint4*>(&Qs[r * LDS + c * 8]) = v;
        }
    }
    auto ldg_tile = [&](int k0, uint4 (&kr)[CPT], uint4 (&vr)[CPT]) {
#pragma unroll
        for (int u = 0; u < CPT; u++) {
            const int i = tid + 128 * u;
            const int r = i / (HD / 8), c = i % (HD / 8);
            const int kp = k0 + r;
            kr[u] = make_uint4(0u, 0u, 0u, 0u);
            vr[u] = make_uint4(0u, 0u, 0u, 0u);
            if (kp <= kmax) {
                const uint4* kp4 = reinterpret_cast<const uint4*>(k_cache + (size_t)kp * kv_dim + kvh * HD) + c;
                const uint4* vp4 = reinterpret_cast<const uint4*>(v_cache + (size_t)kp * kv_dim + kvh * HD) + c;
                if (FUSED) {
                    kr[u] = *kp4;
                    vr[u] = *vp4;
                } else {
                    kr[u] = __ldg(kp4);
                    vr[u] = __ldg(vp4);
                }
            }
        }
    };
    auto sts_tile = [&](int buf, const uint4 (&kr)[CPT], const uint4 (&vr)[CPT]) {
#pragma unroll
        for (int u = 0; u < CPT; u++) {
            const int i = tid + 128 * u;
            const int r = i / (HD / 8), c = i % (HD / 8);
            *reinterpret_cast<uint4*>(&Ks[buf][r * LDS + c * 8]) = kr[u];
            *reinterpret_cast<uint4*>(&Vs[buf][r * LDS + c * 8]) = vr[u];
        }
    };
    // sliding window (window > 0): the query sees the keys pos - window < kp <= pos (the rule of flash_train.cu); the tiles below it are skipped, they would not change the online softmax
    const int it0 = window > 0 ? max(0, pos - window + 1) / KT : 0;
    {
        uint4 kr[CPT], vr[CPT];
        ldg_tile(it0 * KT, kr, vr);
        sts_tile(0, kr, vr);
    }
    __syncthreads();
    uint32_t qf[KS][2];
    {
        const int a_row = (lane & 7) + ((lane >> 3) & 1) * 8, a_col = (lane >> 4) * 8;
#pragma unroll
        for (int s2 = 0; s2 < HD / 16; s2++) {
            uint32_t r[4];
            flash::ldsm_x4(r, flash::smem_u32(&Qs[a_row * LDS + s2 * 16 + a_col]));
            qf[2 * s2][0] = r[0];
            qf[2 * s2][1] = r[1];
            qf[2 * s2 + 1][0] = r[2];
            qf[2 * s2 + 1][1] = r[3];
        }
    }
    float o[NDW][4];
#pragma unroll
    for (int i = 0; i < NDW; i++) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.f;
    float m[2] = {-INFINITY, -INFINITY}, l[2] = {0.f, 0.f};
    const float sl2 = rsqrtf((float)HD) * 1.4426950408889634f;
    const bool ok0 = g < rep, ok1 = g + 8 < rep;
    const int b_row = (lane & 7) + (lane >> 4) * 8, b_col = ((lane >> 3) & 1) * 8;
    const int v_row = lane & 7, v_mat = lane >> 3;
    const int ntiles = kmax / KT + 1;
    for (int it = it0; it < ntiles; it++) {
        const int buf = (it - it0) & 1, k0 = it * KT;
        uint4 kr[CPT], vr[CPT];
        if (it + 1 < ntiles) ldg_tile(k0 + KT, kr, vr);
        // S for this warp's 16 keys (warps beyond KT / 16 have none)
        if (warp < KT / 16) {
            float s[2][4];
#pragma unroll
            for (int i = 0; i < 2; i++) s[i][0] = s[i][1] = s[i][2] = s[i][3] = 0.f;
            const int j2 = warp;
#pragma unroll
            for (int s2 = 0; s2 < HD / 16; s2++) {
                uint32_t kb[4];
                flash::ldsm_x4(kb, flash::smem_u32(&Ks[buf][(j2 * 16 + b_row) * LDS + s2 * 16 + b_col]));
                flash::mma_k8(s[0], qf[2 * s2][0], qf[2 * s2][1], kb[0]);
                flash::mma_k8(s[0], qf[2 * s2 + 1][0], qf[2 * s2 + 1][1], kb[1]);
                flash::mma_k8(s[1], qf[2 * s2][0], qf[2 * s2][1], kb[2]);
                flash::mma_k8(s[1], qf[2 * s2 + 1][0], qf[2 * s2 + 1][1], kb[3]);
            }
#pragma unroll
            for (int i = 0; i < 2; i++)
#pragma unroll
                for (int e = 0; e < 4; e++) {
                    const int col = (2 * j2 + i) * 8 + 2 * t4 + (e & 1);
                    const int kp = k0 + col;
                    const bool rok = (e >> 1) ? ok1 : ok0;
                    float v = s[i][e] * sl2;
                    if (kp > pos || !rok || kp > kmax || (window > 0 && pos - kp >= window)) v = -INFINITY;
                    Ss[g + 8 * (e >> 1)][col] = v;
                }
        }
        __syncthreads();
        // flash.cu's softmax step on the whole tile (every warp, identical values)
        float sv[NK][4];
        float mx0 = -INFINITY, mx1 = -INFINITY;
#pragma unroll
        for (int nt = 0; nt < NK; nt++) {
#pragma unroll
            for (int e = 0; e < 4; e++) {
                const float v = Ss[g + 8 * (e >> 1)][nt * 8 + 2 * t4 + (e & 1)];
                sv[nt][e] = v;
                if (e >> 1) mx1 = fmaxf(mx1, v); else mx0 = fmaxf(mx0, v);
            }
        }
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 1));
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 2));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 1));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 2));
        const float mn0 = fmaxf(m[0], mx0), mn1 = fmaxf(m[1], mx1);
        const float ms0 = mn0 == -INFINITY ? 0.f : mn0, ms1 = mn1 == -INFINITY ? 0.f : mn1;
        const float corr0 = exp2f(m[0] - ms0), corr1 = exp2f(m[1] - ms1);
        m[0] = mn0;
        m[1] = mn1;
        l[0] *= corr0;
        l[1] *= corr1;
#pragma unroll
        for (int i = 0; i < NDW; i++) {
            o[i][0] *= corr0;
            o[i][1] *= corr0;
            o[i][2] *= corr1;
            o[i][3] *= corr1;
        }
        uint32_t pa[NK][2];
#pragma unroll
        for (int nt = 0; nt < NK; nt++) {
            const float p0 = exp2f(sv[nt][0] - ms0), p1 = exp2f(sv[nt][1] - ms0), p2 = exp2f(sv[nt][2] - ms1), p3 = exp2f(sv[nt][3] - ms1);
            l[0] += p0 + p1;
            l[1] += p2 + p3;
            pa[nt][0] = flash::pack_half2(p0, p1);
            pa[nt][1] = flash::pack_half2(p2, p3);
        }
        // O += P V for this warp's output tiles: the x4 ldmatrix of flash.cu that covers them, keeping its NDW matrices
        constexpr int GRP = 4 / NDW;   // warps per group of four output tiles
        const int nd4 = warp / GRP, u0 = (warp % GRP) * NDW;
#pragma unroll
        for (int j = 0; j < NK; j++) {
            uint32_t vb[4];
            flash::ldsm_x4_t(vb, flash::smem_u32(&Vs[buf][(j * 8 + v_row) * LDS + (nd4 * 4 + v_mat) * 8]));
#pragma unroll
            for (int u = 0; u < NDW; u++) flash::mma_k8(o[u], pa[j][0], pa[j][1], vb[u0 + u]);
        }
        if (it + 1 < ntiles) sts_tile(buf ^ 1, kr, vr);
        __syncthreads();
    }
    l[0] += __shfl_xor_sync(0xffffffffu, l[0], 1);
    l[0] += __shfl_xor_sync(0xffffffffu, l[0], 2);
    l[1] += __shfl_xor_sync(0xffffffffu, l[1], 1);
    l[1] += __shfl_xor_sync(0xffffffffu, l[1], 2);
    const float inv0 = l[0] > 0.f ? 1.f / l[0] : 0.f, inv1 = l[1] > 0.f ? 1.f / l[1] : 0.f;
    constexpr int GRP = 4 / NDW;
    const int nd4 = warp / GRP, u0 = (warp % GRP) * NDW;
#pragma unroll
    for (int u = 0; u < NDW; u++) {
        const int col = (nd4 * 4 + u0 + u) * 8 + 2 * t4;
        if (ok0) *reinterpret_cast<half2*>(out + (size_t)(kvh * rep + g) * HD + col) = __floats2half2_rn(o[u][0] * inv0, o[u][1] * inv0);
        if (ok1) *reinterpret_cast<half2*>(out + (size_t)(kvh * rep + g + 8) * HD + col) = __floats2half2_rn(o[u][2] * inv1, o[u][3] * inv1);
    }
}

}  // namespace f1d

extern "C" __global__ void __launch_bounds__(128) f1d_attn_gqa_64(const half* q, half* k_cache, half* v_cache, half* out, const f1d::F1Step* st, int n_head, int n_kv, int window) {
    f1d::attn_gqa_impl<64, 64, false>(q, k_cache, v_cache, out, st, n_head, n_kv, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 0.f, window);
}
extern "C" __global__ void __launch_bounds__(128) f1d_attn_gqa_128(const half* q, half* k_cache, half* v_cache, half* out, const f1d::F1Step* st, int n_head, int n_kv, int window) {
    f1d::attn_gqa_impl<128, 32, false>(q, k_cache, v_cache, out, st, n_head, n_kv, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 0.f, window);
}
// q / k / v post-processing (f1d_qkv_post) fused in front of the attention: one launch less per layer
extern "C" __global__ void __launch_bounds__(128) f1d_attn_fused_64(const half* qkv, half* k_cache, half* v_cache, half* out, const f1d::F1Step* st, int n_head, int n_kv, const float* cos_t,
                                                                    const float* sin_t, const float* qn, const float* kn, const half* ve, const float* ve_sc, float eps, int window) {
    f1d::attn_gqa_impl<64, 64, true>(nullptr, k_cache, v_cache, out, st, n_head, n_kv, qkv, cos_t, sin_t, qn, kn, ve, ve_sc, eps, window);
}
extern "C" __global__ void __launch_bounds__(128) f1d_attn_fused_128(const half* qkv, half* k_cache, half* v_cache, half* out, const f1d::F1Step* st, int n_head, int n_kv, const float* cos_t,
                                                                     const float* sin_t, const float* qn, const float* kn, const half* ve, const float* ve_sc, float eps, int window) {
    f1d::attn_gqa_impl<128, 32, true>(nullptr, k_cache, v_cache, out, st, n_head, n_kv, qkv, cos_t, sin_t, qn, kn, ve, ve_sc, eps, window);
}

// Long contexts: the attention above walks the key tiles one after the other in one block per KV head (the online softmax is a chain over the tiles). Only the running max, the row sum and the
// O += P V accumulation are a chain; the scores are not. Split version, still flash.cu's arithmetic:
//   f1d_attn_scores: one block per (key tile, KV head) computes S = Q K^T scaled and masked (the k8 chain over the head dimension of flash.cu) into a scratch row per query head;
//   f1d_attn_pv: one warp per (KV head, pair of output n8 tiles) walks the tiles: it reads the scores, repeats flash.cu's max / correction / row sum per lane, rounds P to fp16 and runs the mma chain
//   of its two output tiles over the value tiles (prefetched one tile ahead).
namespace f1d {

template <int HD, int KT>
__device__ __forceinline__ void attn_scores_impl(const half* __restrict__ q, const half* __restrict__ k_cache, float* __restrict__ S, const F1Step* __restrict__ st, int n_head, int n_kv, int s_stride,
                                                 int window) {
    constexpr int LDS = HD + 8;
    constexpr int KS = HD / 8;
    __shared__ __align__(16) half Qs[16 * LDS];
    __shared__ __align__(16) half Ks[KT * LDS];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int g = lane >> 2, t4 = lane & 3;
    const int kvh = blockIdx.y, rep = n_head / n_kv;
    const int kv_dim = n_kv * HD;
    const int pos = (int)st->pos;
    const int k0 = blockIdx.x * KT;
    if (k0 > pos) return;
    if (window > 0 && k0 + KT - 1 <= pos - window) return;   // the whole tile lies below the window: the softmax chain (f1d_attn_pv) starts after it
    for (int i = tid; i < 16 * HD / 8; i += blockDim.x) {
        const int r = i / (HD / 8), c = i % (HD / 8);
        uint4 v = make_uint4(0u, 0u, 0u, 0u);
        if (r < rep) v = *reinterpret_cast<const uint4*>(q + (size_t)(kvh * rep + r) * HD + c * 8);
        *reinterpret_cast<uint4*>(&Qs[r * LDS + c * 8]) = v;
    }
    for (int i = tid; i < KT * HD / 8; i += blockDim.x) {
        const int r = i / (HD / 8), c = i % (HD / 8);
        const int kp = k0 + r;
        uint4 v = make_uint4(0u, 0u, 0u, 0u);
        if (kp <= pos) v = __ldg(reinterpret_cast<const uint4*>(k_cache + (size_t)kp * kv_dim + kvh * HD) + c);
        *reinterpret_cast<uint4*>(&Ks[r * LDS + c * 8]) = v;
    }
    __syncthreads();
    if (warp >= KT / 16) return;
    uint32_t qf[KS][2];
    {
        const int a_row = (lane & 7) + ((lane >> 3) & 1) * 8, a_col = (lane >> 4) * 8;
#pragma unroll
        for (int s2 = 0; s2 < HD / 16; s2++) {
            uint32_t r[4];
            flash::ldsm_x4(r, flash::smem_u32(&Qs[a_row * LDS + s2 * 16 + a_col]));
            qf[2 * s2][0] = r[0];
            qf[2 * s2][1] = r[1];
            qf[2 * s2 + 1][0] = r[2];
            qf[2 * s2 + 1][1] = r[3];
        }
    }
    const float sl2 = rsqrtf((float)HD) * 1.4426950408889634f;
    const int b_row = (lane & 7) + (lane >> 4) * 8, b_col = ((lane >> 3) & 1) * 8;
    float sc[2][4];
#pragma unroll
    for (int i = 0; i < 2; i++) sc[i][0] = sc[i][1] = sc[i][2] = sc[i][3] = 0.f;
    const int j2 = warp;
#pragma unroll
    for (int s2 = 0; s2 < HD / 16; s2++) {
        uint32_t kb[4];
        flash::ldsm_x4(kb, flash::smem_u32(&Ks[(j2 * 16 + b_row) * LDS + s2 * 16 + b_col]));
        flash::mma_k8(sc[0], qf[2 * s2][0], qf[2 * s2][1], kb[0]);
        flash::mma_k8(sc[0], qf[2 * s2 + 1][0], qf[2 * s2 + 1][1], kb[1]);
        flash::mma_k8(sc[1], qf[2 * s2][0], qf[2 * s2][1], kb[2]);
        flash::mma_k8(sc[1], qf[2 * s2 + 1][0], qf[2 * s2 + 1][1], kb[3]);
    }
#pragma unroll
    for (int i = 0; i < 2; i++)
#pragma unroll
        for (int e = 0; e < 4; e++) {
            const int row = g + 8 * (e >> 1);
            if (row >= rep) continue;
            const int col = (2 * j2 + i) * 8 + 2 * t4 + (e & 1);
            const int kp = k0 + col;
            float v = sc[i][e] * sl2;
            if (kp > pos || (window > 0 && pos - kp >= window)) v = -INFINITY;
            S[(size_t)(kvh * 16 + row) * s_stride + kp] = v;
        }
}

template <int HD, int KT>
__device__ __forceinline__ void attn_pv_impl(const half* __restrict__ v_cache, const float* __restrict__ S, half* __restrict__ out, const F1Step* __restrict__ st, int n_head, int n_kv, int s_stride,
                                             int window) {
    constexpr int NK = KT / 8;
    constexpr int VLDS = 24;   // 16 dims + 8 halves of padding: 48-byte rows, conflict free for ldmatrix
    constexpr int VCH = KT * 2; // 16-byte chunks of the value tile (two per key)
    constexpr int VPL = VCH / 32;
    __shared__ __align__(16) half Vs[KT * VLDS];
    const int lane = threadIdx.x & 31;
    const int g = lane >> 2, t4 = lane & 3;
    const int kvh = blockIdx.x, part = blockIdx.y, rep = n_head / n_kv;
    const int kv_dim = n_kv * HD;
    const int pos = (int)st->pos;
    const int ntiles = pos / KT + 1;
    const int it0 = window > 0 ? max(0, pos - window + 1) / KT : 0;   // first tile with a key inside the window
    const bool ok0 = g < rep, ok1 = g + 8 < rep;
    const float* srow0 = S + (size_t)(kvh * 16 + g) * s_stride;
    const float* srow1 = S + (size_t)(kvh * 16 + g + 8) * s_stride;
    auto ld_s = [&](int k0, float (&sv)[NK][4]) {
#pragma unroll
        for (int nt = 0; nt < NK; nt++) {
            const int col = k0 + nt * 8 + 2 * t4;
            float2 a = make_float2(-INFINITY, -INFINITY), b = make_float2(-INFINITY, -INFINITY);
            if (ok0) a = *reinterpret_cast<const float2*>(srow0 + col);
            if (ok1) b = *reinterpret_cast<const float2*>(srow1 + col);
            sv[nt][0] = a.x;
            sv[nt][1] = a.y;
            sv[nt][2] = b.x;
            sv[nt][3] = b.y;
        }
    };
    auto ld_v = [&](int k0, uint4 (&vr)[VPL]) {
#pragma unroll
        for (int u = 0; u < VPL; u++) {
            const int i = lane + 32 * u;
            const int r = i >> 1, c = i & 1;
            const int kp = k0 + r;
            vr[u] = kp <= pos ? __ldg(reinterpret_cast<const uint4*>(v_cache + (size_t)kp * kv_dim + kvh * HD + part * 16) + c) : make_uint4(0u, 0u, 0u, 0u);
        }
    };
    float sv[NK][4];
    uint4 vr[VPL];
    ld_s(it0 * KT, sv);
    ld_v(it0 * KT, vr);
    float o[2][4];
#pragma unroll
    for (int i = 0; i < 2; i++) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.f;
    float m[2] = {-INFINITY, -INFINITY}, l[2] = {0.f, 0.f};
    const int v_row = (lane & 7) + ((lane >> 4) & 1) * 8, v_col = ((lane >> 3) & 1) * 8;
    for (int it = it0; it < ntiles; it++) {
        float nsv[NK][4];
        uint4 nvr[VPL];
        if (it + 1 < ntiles) {
            ld_s((it + 1) * KT, nsv);
            ld_v((it + 1) * KT, nvr);
        }
        __syncwarp();
#pragma unroll
        for (int u = 0; u < VPL; u++) {
            const int i = lane + 32 * u;
            *reinterpret_cast<uint4*>(&Vs[(i >> 1) * VLDS + (i & 1) * 8]) = vr[u];
        }
        __syncwarp();
        // flash.cu's softmax step
        float mx0 = -INFINITY, mx1 = -INFINITY;
#pragma unroll
        for (int nt = 0; nt < NK; nt++) {
            mx0 = fmaxf(mx0, sv[nt][0]);
            mx0 = fmaxf(mx0, sv[nt][1]);
            mx1 = fmaxf(mx1, sv[nt][2]);
            mx1 = fmaxf(mx1, sv[nt][3]);
        }
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 1));
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 2));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 1));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 2));
        const float mn0 = fmaxf(m[0], mx0), mn1 = fmaxf(m[1], mx1);
        const float ms0 = mn0 == -INFINITY ? 0.f : mn0, ms1 = mn1 == -INFINITY ? 0.f : mn1;
        const float corr0 = exp2f(m[0] - ms0), corr1 = exp2f(m[1] - ms1);
        m[0] = mn0;
        m[1] = mn1;
        l[0] *= corr0;
        l[1] *= corr1;
#pragma unroll
        for (int i = 0; i < 2; i++) {
            o[i][0] *= corr0;
            o[i][1] *= corr0;
            o[i][2] *= corr1;
            o[i][3] *= corr1;
        }
        uint32_t pa[NK][2];
#pragma unroll
        for (int nt = 0; nt < NK; nt++) {
            const float p0 = exp2f(sv[nt][0] - ms0), p1 = exp2f(sv[nt][1] - ms0), p2 = exp2f(sv[nt][2] - ms1), p3 = exp2f(sv[nt][3] - ms1);
            l[0] += p0 + p1;
            l[1] += p2 + p3;
            pa[nt][0] = flash::pack_half2(p0, p1);
            pa[nt][1] = flash::pack_half2(p2, p3);
        }
#pragma unroll
        for (int j = 0; j < NK; j += 2) {
            uint32_t vb[4];
            flash::ldsm_x4_t(vb, flash::smem_u32(&Vs[(j * 8 + v_row) * VLDS + v_col]));
            flash::mma_k8(o[0], pa[j][0], pa[j][1], vb[0]);
            flash::mma_k8(o[1], pa[j][0], pa[j][1], vb[1]);
            flash::mma_k8(o[0], pa[j + 1][0], pa[j + 1][1], vb[2]);
            flash::mma_k8(o[1], pa[j + 1][0], pa[j + 1][1], vb[3]);
        }
        if (it + 1 < ntiles) {
#pragma unroll
            for (int nt = 0; nt < NK; nt++)
#pragma unroll
                for (int e = 0; e < 4; e++) sv[nt][e] = nsv[nt][e];
#pragma unroll
            for (int u = 0; u < VPL; u++) vr[u] = nvr[u];
        }
    }
    l[0] += __shfl_xor_sync(0xffffffffu, l[0], 1);
    l[0] += __shfl_xor_sync(0xffffffffu, l[0], 2);
    l[1] += __shfl_xor_sync(0xffffffffu, l[1], 1);
    l[1] += __shfl_xor_sync(0xffffffffu, l[1], 2);
    const float inv0 = l[0] > 0.f ? 1.f / l[0] : 0.f, inv1 = l[1] > 0.f ? 1.f / l[1] : 0.f;
#pragma unroll
    for (int u = 0; u < 2; u++) {
        const int col = (part * 2 + u) * 8 + 2 * t4;
        if (ok0) *reinterpret_cast<half2*>(out + (size_t)(kvh * rep + g) * HD + col) = __floats2half2_rn(o[u][0] * inv0, o[u][1] * inv0);
        if (ok1) *reinterpret_cast<half2*>(out + (size_t)(kvh * rep + g + 8) * HD + col) = __floats2half2_rn(o[u][2] * inv1, o[u][3] * inv1);
    }
}

}  // namespace f1d

extern "C" __global__ void __launch_bounds__(128) f1d_attn_scores_64(const half* q, const half* k_cache, float* S, const f1d::F1Step* st, int n_head, int n_kv, int s_stride, int window) {
    f1d::attn_scores_impl<64, 64>(q, k_cache, S, st, n_head, n_kv, s_stride, window);
}
extern "C" __global__ void __launch_bounds__(128) f1d_attn_scores_128(const half* q, const half* k_cache, float* S, const f1d::F1Step* st, int n_head, int n_kv, int s_stride, int window) {
    f1d::attn_scores_impl<128, 32>(q, k_cache, S, st, n_head, n_kv, s_stride, window);
}
extern "C" __global__ void __launch_bounds__(32) f1d_attn_pv_64(const half* v_cache, const float* S, half* out, const f1d::F1Step* st, int n_head, int n_kv, int s_stride, int window) {
    f1d::attn_pv_impl<64, 64>(v_cache, S, out, st, n_head, n_kv, s_stride, window);
}
extern "C" __global__ void __launch_bounds__(32) f1d_attn_pv_128(const half* v_cache, const float* S, half* out, const f1d::F1Step* st, int n_head, int n_kv, int s_stride, int window) {
    f1d::attn_pv_impl<128, 32>(v_cache, S, out, st, n_head, n_kv, s_stride, window);
}

// attention of the single query at position st->pos over the cache: flash.cu's tile loop itself (T = 1 query row, pos0 = pos); kept as the reference variant (FORGE_KV_ATTN=flash)
extern "C" __global__ void __launch_bounds__(128) f1d_attn_64(const half* q, const half* k_cache, const half* v_cache, half* out, const f1d::F1Step* st, int n_head, int n_kv) {
    flash::flash_fwd_impl<64, 64>(q, k_cache, v_cache, out, nullptr, 1, (int)st->pos, n_head, n_kv);
}
// the same with a sliding window (window >= 1)
extern "C" __global__ void __launch_bounds__(128) f1d_attn_win_64(const half* q, const half* k_cache, const half* v_cache, half* out, const f1d::F1Step* st, int n_head, int n_kv, int window) {
    flash::flash_fwd_impl<64, 64, true>(q, k_cache, v_cache, out, nullptr, 1, (int)st->pos, n_head, n_kv, window);
}
extern "C" __global__ void __launch_bounds__(128) f1d_attn_128(const half* q, const half* k_cache, const half* v_cache, half* out, const f1d::F1Step* st, int n_head, int n_kv) {
    flash::flash_fwd_impl<128, 32>(q, k_cache, v_cache, out, nullptr, 1, (int)st->pos, n_head, n_kv);
}
