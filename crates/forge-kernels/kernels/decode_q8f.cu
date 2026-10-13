// Q8 weights for the KV-cache decoder of a Forge-1 TRAINING checkpoint (crates/forge-train/src/decode.rs, `Decoder::enable_q8`; `forge serve CKPT --q8`).
//
// One new token is memory bound: the matrix-vector products stream every weight once. Q8_0-style weights (int8 plus one fp16 scale per 32 weights, 1.0625 bytes per weight instead of 2) cut that
// traffic by 47 percent. The activations stay fp16 (as in decode_f1.cu): only the weights are rounded, the products are exact in fp32 (int8 * fp16 fits in fp16 * fp16 -> fp32 exactly) and the
// accumulation is fp32 (tensor cores within a 32-weight block, fp32 FMA with the block's scale across blocks).
//
// MATRIX LAYOUT (built on the device by q8f_quantize from the fp32 master weights, rows = outputs, K = inputs, rows a multiple of 16, K a multiple of 64):
//   16 rows form a TILE; a tile is cut along K into SUPER-CHUNKS of 64 weights per row (two 32-weight blocks) of 1088 bytes: 2 x 512 bytes of weights, then 64 bytes of scales.
//   Super-chunks of a tile follow each other along K, tiles follow each other: byte offset of (tile i, super-chunk c) = (i * (K / 64) + c) * 1088.
//   Weights of one 32-weight block (512 bytes) are stored in the order in which the lanes of a warp need them as the A operand of mma.m16n8k16 (the weight tile is the 16 x 16 A matrix, the
//   activation vector is column 0 of the 16 x 8 B matrix, so one mma covers 256 weights; the other 7 columns of B are zero):
//     lane l = 4 g + t (g = row within 8, t = k pair index) holds 16 bytes = four 32-bit words
//        word 0: row g,     k 2t, 2t+1, 2t+8, 2t+9 of the first 16 k      (a0 | a2 of step 0 as byte pairs)
//        word 1: row g + 8, the same k
//        word 2: row g,     k 16 + 2t, 16 + 2t + 1, 16 + 2t + 8, 16 + 2t + 9   (step 1)
//        word 3: row g + 8, the same k
//     every byte is q + 128 (offset binary, q in [-127, 127]), so that PRMT with 0x64 as the high byte makes the fp16 value 1024 + q + 128 and one HSUB2 gives q exactly.
//   Scales: per super-chunk 8 lanes groups g x 2 blocks x (d_row_g, d_row_g+8) as fp16 pairs: the 8 bytes of group g are at 1024 + 8 g.
//   One warp load of a block is 512 contiguous bytes (lane l reads bytes 16 l ..), no staging through shared memory and no ldmatrix are needed.
// The activation vector lives in shared memory as fp16 in the matching order: for each 32-k block 4 x 16 bytes, uint4 t = (b0 of step 0, b1 of step 0, b0 of step 1, b1 of step 1) of lane group g = 0.
// (Lanes with g != 0 read a zero line: their B column is 0 and they contribute nothing.)
//
// GEMV kernels: one CTA = W warps; S warps share a 16-row tile (split along K, partial sums reduced through shared memory), W / S tiles per CTA. The first batch of weights of every warp is requested
// before the prologue (norm of the input vector) so that the memory latency overlaps it. Prologues and epilogues are those of decode_f1.cu (f1d_gemv_*).

#include <cuda_fp16.h>
#include <stdint.h>

namespace q8f {

constexpr int SC_BYTES = 1088;   // bytes of one super-chunk of a tile (64 k)

struct Step {
    uint32_t token;
    uint32_t pos;
};

__device__ __forceinline__ float wsum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float wmax(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

__device__ __forceinline__ void mma_k8(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t b0) {
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(a1), "r"(b0));
}
// one 16 x 16 x 8 product block: A = 16 x 16 (a0: row g k 2t.., a1: row g + 8, a2: row g k 2t + 8.., a3: row g + 8), B = 16 x 8 (b0: k 2t.., b1: k 2t + 8..)
__device__ __forceinline__ void mma16(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
#if __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#else
    mma_k8(c, a0, a1, b0);
    mma_k8(c, a2, a3, b1);
#endif
}

__device__ __forceinline__ uint32_t hsub2_raw(uint32_t a, uint32_t b) {
    uint32_t d;
    asm("sub.rn.f16x2 %0, %1, %2;\n" : "=r"(d) : "r"(a), "r"(b));
    return d;
}
__device__ __forceinline__ uint32_t hmul2_raw(uint32_t a, uint32_t b) {
    uint32_t d;
    asm("mul.rn.f16x2 %0, %1, %2;\n" : "=r"(d) : "r"(a), "r"(b));
    return d;
}
// two offset-binary bytes (q + 128) -> the fp16 pair (q, q'): PRMT puts each byte under 0x64 (= 1024 + u), HSUB2 takes 1152 away
__device__ __forceinline__ uint32_t dq_lo(uint32_t w) { return hsub2_raw(__byte_perm(w, 0x64646464u, 0x5140u), 0x64806480u); }   // bytes 0, 1
__device__ __forceinline__ uint32_t dq_hi(uint32_t w) { return hsub2_raw(__byte_perm(w, 0x64646464u, 0x5342u), 0x64806480u); }   // bytes 2, 3

__device__ __forceinline__ uint4 ld_w16(const uint8_t* p) { return __ldcs(reinterpret_cast<const uint4*>(p)); }
__device__ __forceinline__ uint2 ld_w8(const uint8_t* p) { return __ldcs(reinterpret_cast<const uint2*>(p)); }

// prologues: what the activation vector (K values, fp16, the B operand) is
constexpr int PRO_VEC = 0;      // an fp16 vector in global memory (attention output, MLP activation, final-norm output)
constexpr int PRO_NORM = 1;     // RMSNorm (weights nw) of an fp32 vector xin of K values
constexpr int PRO_MIXNORM = 2;  // the layer-input mixing  x = a * xin + b * x0 + c * xskip  (block 0 stores x to xout), then RMSNorm
constexpr int EPI_F16 = 0, EPI_F32 = 1, EPI_RES = 4, EPI_RELU2 = 5;   // as in decode_f1.cu

// element j (a 32-bit word of two halves) of the activation vector -> its place in the permuted vector: block kb = j / 16, word w = j % 16 -> uint4 (4 kb + w % 4), component w / 4
__device__ __forceinline__ int perm_word(int j) {
    const int kb = j >> 4, w = j & 15;
    return ((kb * 4 + (w & 3)) << 2) + (w >> 2);
}

template <int PRO, int EPI, int NB>
__device__ __forceinline__ void gemv_impl(const uint8_t* __restrict__ Wq, int N, int K, int S, const half* __restrict__ a16, const float* __restrict__ xin, const float* __restrict__ x0,
                                          const float* __restrict__ xskip, const float* __restrict__ sc, int use_ab, int use_c, float* __restrict__ xout, const float* __restrict__ nw,
                                          float eps, void* __restrict__ out, const float* __restrict__ R) {
    extern __shared__ __align__(16) unsigned char smem[];
    const int tid = threadIdx.x, nthr = blockDim.x;
    const int warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
    const int nw_ = nthr >> 5;
    const int TB = nw_ / S;
    const int tl = warp / S, ks = warp - tl * S;
    const int tile = blockIdx.x * TB + tl;
    const int ntiles = N >> 4;
    const int nsc = K >> 6;
    const bool live = tl < TB && tile < ntiles;
    const int sc0 = ks * nsc / S, sc1 = (ks + 1) * nsc / S;

    uint32_t* xs32 = reinterpret_cast<uint32_t*>(smem + 16);                  // K / 2 words, permuted
    float* red = reinterpret_cast<float*>(smem + 16 + (size_t)K * 2);          // nw_ x 16 partial sums, then 32 floats for the norm
    float* nred = red + nw_ * 16;

    // the first batch of weights, requested before anything else
    uint4 cd0[NB], cd1[NB];
    uint2 cs[NB];
    const uint8_t* wp = Wq + ((size_t)(live ? tile : 0) * nsc + sc0) * SC_BYTES;
#pragma unroll
    for (int u = 0; u < NB; u++) {
        if (live && sc0 + u < sc1) {
            const uint8_t* p = wp + (size_t)u * SC_BYTES;
            cd0[u] = ld_w16(p + lane * 16);
            cd1[u] = ld_w16(p + 512 + lane * 16);
            cs[u] = ld_w8(p + 1024 + g * 8);
        } else {
            cd0[u] = cd1[u] = make_uint4(0u, 0u, 0u, 0u);
            cs[u] = make_uint2(0u, 0u);
        }
    }

    // ---- prologue: the activation vector into shared memory (all threads) ----
    if (tid < 4) reinterpret_cast<uint32_t*>(smem)[tid] = 0u;   // the zero line read by the lanes of g != 0
    if (PRO == PRO_VEC) {
        const uint4* s4 = reinterpret_cast<const uint4*>(a16);
        for (int i = tid; i < (K >> 3); i += nthr) {
            const uint4 v = s4[i];
            const int kb = i >> 2, m = i & 3;
            xs32[((kb * 4 + 0) << 2) + m] = v.x;
            xs32[((kb * 4 + 1) << 2) + m] = v.y;
            xs32[((kb * 4 + 2) << 2) + m] = v.z;
            xs32[((kb * 4 + 3) << 2) + m] = v.w;
        }
    } else {
        // K floats in float4 chunks, at most 4 per thread (K <= 1024: 64 threads or more)
        float4 xv[4];
        const float4* xi4 = reinterpret_cast<const float4*>(xin);
        const float4* x04 = reinterpret_cast<const float4*>(x0);
        const float4* xs4_ = reinterpret_cast<const float4*>(xskip);
        const float4* nw4 = reinterpret_cast<const float4*>(nw);
        const int K4 = K >> 2;
        float a = 1.f, b = 0.f, c = 0.f;
        if (PRO == PRO_MIXNORM) {
            a = use_ab ? sc[0] : 1.f;
            b = use_ab ? sc[1] : 0.f;
            c = use_c ? sc[2] : 0.f;
        }
        float ss = 0.f;
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const int i = tid + j * nthr;
            xv[j] = make_float4(0.f, 0.f, 0.f, 0.f);
            if (i < K4) {
                float4 v = xi4[i];
                if (PRO == PRO_MIXNORM) {
                    float4 o = make_float4(a * v.x, a * v.y, a * v.z, a * v.w);
                    if (use_ab) {
                        const float4 p = x04[i];
                        o.x += b * p.x;
                        o.y += b * p.y;
                        o.z += b * p.z;
                        o.w += b * p.w;
                    }
                    if (use_c) {
                        const float4 p = xs4_[i];
                        o.x += c * p.x;
                        o.y += c * p.y;
                        o.z += c * p.z;
                        o.w += c * p.w;
                    }
                    v = o;
                    if (blockIdx.x == 0) reinterpret_cast<float4*>(xout)[i] = v;
                }
                xv[j] = v;
                ss += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
            }
        }
        ss = wsum(ss);
        if (lane == 0) nred[warp] = ss;
        __syncthreads();
        float tot = 0.f;
        for (int i = 0; i < nw_; i++) tot += nred[i];
        const float r = rsqrtf(tot / (float)K + eps);
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const int i = tid + j * nthr;
            if (i < K4) {
                const float4 v = xv[j], gw = nw4[i];
                const __half2 h0 = __floats2half2_rn(v.x * r * gw.x, v.y * r * gw.y), h1 = __floats2half2_rn(v.z * r * gw.z, v.w * r * gw.w);
                xs32[perm_word(2 * i)] = *reinterpret_cast<const uint32_t*>(&h0);
                xs32[perm_word(2 * i + 1)] = *reinterpret_cast<const uint32_t*>(&h1);
            }
        }
    }
    __syncthreads();

    // ---- the K loop of this warp: super-chunks sc0 .. sc1 of its tile, NB per batch, the next batch loading while one is consumed ----
    float tot0 = 0.f, tot1 = 0.f;   // rows g and g + 8 (valid in the lanes with t == 0)
    const uint4* xp = g == 0 ? reinterpret_cast<const uint4*>(smem + 16) + t : reinterpret_cast<const uint4*>(smem);
    const int xstride = g == 0 ? 4 : 0;
    if (live) {
        for (int b0 = sc0; b0 < sc1; b0 += NB) {
            uint4 nd0[NB], nd1[NB];
            uint2 ns[NB];
            const uint8_t* np = Wq + ((size_t)tile * nsc + b0 + NB) * SC_BYTES;
#pragma unroll
            for (int u = 0; u < NB; u++) {
                if (b0 + NB + u < sc1) {
                    const uint8_t* p = np + (size_t)u * SC_BYTES;
                    nd0[u] = ld_w16(p + lane * 16);
                    nd1[u] = ld_w16(p + 512 + lane * 16);
                    ns[u] = ld_w8(p + 1024 + g * 8);
                } else {
                    nd0[u] = nd1[u] = make_uint4(0u, 0u, 0u, 0u);
                    ns[u] = make_uint2(0u, 0u);
                }
            }
#pragma unroll
            for (int u = 0; u < NB; u++) {
                if (b0 + u < sc1) {
                    const int kb = 2 * (b0 + u);
                    {
                        const uint4 w = cd0[u], xb = xp[kb * xstride];
                        float c[4] = {0.f, 0.f, 0.f, 0.f};
                        mma16(c, dq_lo(w.x), dq_lo(w.y), dq_hi(w.x), dq_hi(w.y), xb.x, xb.y);
                        mma16(c, dq_lo(w.z), dq_lo(w.w), dq_hi(w.z), dq_hi(w.w), xb.z, xb.w);
                        const float2 d = __half22float2(*reinterpret_cast<const half2*>(&cs[u].x));
                        tot0 = fmaf(d.x, c[0], tot0);
                        tot1 = fmaf(d.y, c[2], tot1);
                    }
                    {
                        const uint4 w = cd1[u], xb = xp[(kb + 1) * xstride];
                        float c[4] = {0.f, 0.f, 0.f, 0.f};
                        mma16(c, dq_lo(w.x), dq_lo(w.y), dq_hi(w.x), dq_hi(w.y), xb.x, xb.y);
                        mma16(c, dq_lo(w.z), dq_lo(w.w), dq_hi(w.z), dq_hi(w.w), xb.z, xb.w);
                        const float2 d = __half22float2(*reinterpret_cast<const half2*>(&cs[u].y));
                        tot0 = fmaf(d.x, c[0], tot0);
                        tot1 = fmaf(d.y, c[2], tot1);
                    }
                }
            }
#pragma unroll
            for (int u = 0; u < NB; u++) {
                cd0[u] = nd0[u];
                cd1[u] = nd1[u];
                cs[u] = ns[u];
            }
        }
    }
    // ---- reduction over the k-slices of a tile, epilogue (threads 0 .. 16 TB - 1: one output row each) ----
    if (t == 0) {
        red[warp * 16 + g] = tot0;
        red[warp * 16 + g + 8] = tot1;
    }
    __syncthreads();
    if (tid < TB * 16) {
        const int tl2 = tid >> 4, r = tid & 15;
        const int tile2 = blockIdx.x * TB + tl2;
        if (tile2 < ntiles) {
            float v = 0.f;
            for (int s = 0; s < S; s++) v += red[(tl2 * S + s) * 16 + r];
            const int row = tile2 * 16 + r;
            if (EPI == EPI_F16) {
                reinterpret_cast<half*>(out)[row] = __float2half_rn(v);
            } else if (EPI == EPI_F32) {
                reinterpret_cast<float*>(out)[row] = v;
            } else if (EPI == EPI_RES) {
                reinterpret_cast<float*>(out)[row] = v + R[row];
            } else {
                const float u = __half2float(__float2half_rn(v));
                const float r0 = fmaxf(u, 0.f);
                reinterpret_cast<half*>(out)[row] = __float2half(fminf(r0 * r0, 65000.f));
            }
        }
    }
}

}  // namespace q8f

#define Q8F_GEMV(NAME, PRO, EPI, NB)                                                                                                                                                            \
    extern "C" __global__ void __launch_bounds__(256) NAME(const uint8_t* Wq, int N, int K, int S, const half* a16, const float* xin, const float* x0, const float* xskip, const float* sc, int use_ab,   \
                                                           int use_c, float* xout, const float* nw, float eps, void* out, const float* R) {                                                   \
        q8f::gemv_impl<PRO, EPI, NB>(Wq, N, K, S, a16, xin, x0, xskip, sc, use_ab, use_c, xout, nw, eps, out, R);                                                                               \
    }

// layer input (mixing + attention norm) -> q | k | v;  norm -> projection without mixing;  attention output -> + residual;  MLP norm -> up projection + relu2;  MLP norm -> gate | up;  final-norm output -> head
Q8F_GEMV(q8f_gemv_mixnorm_f16, q8f::PRO_MIXNORM, q8f::EPI_F16, 3)
Q8F_GEMV(q8f_gemv_norm_f16, q8f::PRO_NORM, q8f::EPI_F16, 3)
Q8F_GEMV(q8f_gemv_vec_res, q8f::PRO_VEC, q8f::EPI_RES, 3)
Q8F_GEMV(q8f_gemv_norm_relu2, q8f::PRO_NORM, q8f::EPI_RELU2, 3)
Q8F_GEMV(q8f_gemv_vec_f32, q8f::PRO_VEC, q8f::EPI_F32, 3)

// ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// Quantization (fp32 matrix [rows][K] -> the layout above), dequantization (-> fp16 [rows][K], for the prefill GEMMs) and the embedding gathers.
// ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
namespace q8f {

// byte offset of element (row r, column k) inside a matrix of `nsc` super-chunks per tile, and of the scale (fp16) of its row and 32-block
__device__ __forceinline__ size_t byte_at(int r, int k, int nsc) {
    const int i = r >> 4, rr = r & 15, g = rr & 7, hi = rr >> 3;
    const int kb = k >> 5, kk = k & 31, c = kb >> 1, b = kb & 1;
    const int s = kk >> 4, kk2 = kk & 15, t = (kk2 & 7) >> 1, e = kk2 & 1, part = kk2 >> 3;
    return ((size_t)i * nsc + c) * SC_BYTES + b * 512 + (g * 4 + t) * 16 + (s * 2 + hi) * 4 + part * 2 + e;
}
__device__ __forceinline__ size_t scale_at(int r, int kb, int nsc) {
    const int i = r >> 4, rr = r & 15, g = rr & 7, hi = rr >> 3;
    return ((size_t)i * nsc + (kb >> 1)) * SC_BYTES + 1024 + g * 8 + (kb & 1) * 4 + hi * 2;
}

__device__ __forceinline__ float elem(const uint8_t* __restrict__ q, int r, int k, int nsc) {
    const int u = q[byte_at(r, k, nsc)];
    const half d = *reinterpret_cast<const half*>(q + scale_at(r, k >> 5, nsc));
    return (float)(u - 128) * __half2float(d);
}

}  // namespace q8f

// one warp per (row, 32-block): d = fp16(amax / 127), q = round(x / d) in [-127, 127] (the rule of Q8_0 with the scale rounded to fp16 first)
extern "C" __global__ void q8f_quantize(const float* __restrict__ src, int rows, int K, uint8_t* __restrict__ dst) {
    const int wid = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    const int nb = K >> 5;
    if (wid >= rows * nb) return;
    const int r = wid / nb, kb = wid - r * nb;
    const int k = kb * 32 + lane;
    const float v = src[(size_t)r * K + k];
    const float amax = q8f::wmax(fabsf(v));
    const half d16 = __float2half_rn(amax * (1.f / 127.f));
    const float df = __half2float(d16);
    const float id = df != 0.f ? 1.f / df : 0.f;
    int q = __float2int_rn(v * id);
    q = max(-127, min(127, q));
    const int nsc = K >> 6;
    dst[q8f::byte_at(r, k, nsc)] = (uint8_t)(q + 128);
    if (lane == 0) *reinterpret_cast<half*>(dst + q8f::scale_at(r, kb, nsc)) = d16;
}

// the same matrix as fp16 [rows][K] (q * d rounded to fp16): one warp per 512-byte block of a tile (16 rows x 32 k), every lane reads its 16 bytes and writes 8 half pairs
extern "C" __global__ void q8f_dequant(const uint8_t* __restrict__ src, int rows, int K, half* __restrict__ dst) {
    const int wid = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    const int nb = K >> 5, nsc = K >> 6;
    const int tile = wid / nb, kb = wid - tile * nb;
    if (tile >= (rows >> 4)) return;
    const int g = lane >> 2, t = lane & 3;
    const uint8_t* sc = src + ((size_t)tile * nsc + (kb >> 1)) * q8f::SC_BYTES;
    const uint4 w = *reinterpret_cast<const uint4*>(sc + (kb & 1) * 512 + lane * 16);
    const uint32_t d = *reinterpret_cast<const uint32_t*>(sc + 1024 + g * 8 + (kb & 1) * 4);   // (d of row g, d of row g + 8) as two halves
    const uint32_t dg = (d & 0xffffu) * 0x10001u, dg8 = (d >> 16) * 0x10001u;
    half* r0 = dst + (size_t)(tile * 16 + g) * K + kb * 32 + 2 * t;
    half* r1 = r0 + (size_t)8 * K;
    const uint32_t wv[4] = {w.x, w.y, w.z, w.w};   // row g step 0, row g + 8 step 0, row g step 1, row g + 8 step 1
#pragma unroll
    for (int i = 0; i < 4; i++) {
        const uint32_t m = i & 1 ? dg8 : dg;
        half* o = (i & 1 ? r1 : r0) + (i >> 1) * 16;
        const uint32_t lo = q8f::hmul2_raw(q8f::dq_lo(wv[i]), m), hi = q8f::hmul2_raw(q8f::dq_hi(wv[i]), m);
        *reinterpret_cast<uint32_t*>(o) = lo;
        *reinterpret_cast<uint32_t*>(o + 8) = hi;
    }
}

// embedding row of the step's token (the Q8 version of f1d_embed): x = dequantised row, hist[pos] = token
extern "C" __global__ void q8f_embed(const uint8_t* __restrict__ q, int E, const q8f::Step* __restrict__ st, float* __restrict__ x, uint32_t* __restrict__ hist) {
    const int row = (int)st->token, nsc = E >> 6;
    for (int i = threadIdx.x; i < E; i += blockDim.x) x[i] = q8f::elem(q, row, i, nsc);
    if (threadIdx.x == 0) hist[st->pos] = st->token;
}

// embedding rows of n tokens (prompt pieces): x[j][E]
extern "C" __global__ void q8f_embed_rows(const uint8_t* __restrict__ q, int E, const uint32_t* __restrict__ tok, float* __restrict__ x) {
    const int row = (int)tok[blockIdx.x], nsc = E >> 6;
    for (int i = threadIdx.x; i < E; i += blockDim.x) x[(size_t)blockIdx.x * E + i] = q8f::elem(q, row, i, nsc);
}
