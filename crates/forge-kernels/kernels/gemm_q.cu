// Tensor-core GEMM against K-quantized weights that are read in place (prompt processing of quantized models): C[M][N] (op)= A[M][K] W[N][K]^T, A fp16 with K contiguous, W in the planar layouts of
// decode_q.cu (Q4_K, Q5_K, Q6_K; K a multiple of 256).
//
// The structure is gemm_impl3's (128 x 128 tiles, 8 warps of 64 x 32, BK = 16, the registers loaded in iteration k hold tile k + 2 and go to shared memory at the start of iteration k + 1, swizzled
// 16-byte chunks, mma.m16n8k8 fed by ldmatrix); only the B operand differs. A thread owns 8 consecutive weights of one row per tile: it loads their packed bytes and the block header they need
// from global memory (LDG.64 + LDG.128, the header is read from L1 for 16 tiles in a row), and decodes them to fp16 on the way into shared memory. The separate dequantization pass that wrote the
// whole matrix as fp16 (2 bytes per weight) and read it back (20-30 % of the time of a 512 - 4096 token prompt) is gone: the weights cross the memory system at 4.5 - 6.6 bits each.
//
// Decoding is done in fp16x2 arithmetic to keep the instruction count low (the kernel is limited by issue slots, not by the tensor pipe): a byte b becomes the half2 lane 0x64b = 1024 + b by one
// PRMT with 0x64 bytes, minus 1024 is exact, then ONE HFMA2 applies scale and offset to two weights. The scale and the offset are rounded to fp16 first, so a weight can differ from the fp32
// computation of deq_elem by a rounding step of fp16 (2^-11 of the block's value range, a hundredth of the 4-bit quantization step); the prefill_variants test bounds the effect on the logits.

namespace gemm {

// Global loads as volatile inline asm: the compiler may not move them behind the (volatile) MMAs of the iteration. With plain __ldg it sank them to the end of the loop body (the registers are scarce),
// leaving ~50 instructions between a load and its use in the next iteration's decode: 20 % of the warp stalls were long-scoreboard waits (Nsight Compute), against 1 % in the plain GEMM.
__device__ __forceinline__ uint2 ldg64(const void* p) {
    uint2 v;
    asm volatile("ld.global.nc.v2.u32 {%0, %1}, [%2];" : "=r"(v.x), "=r"(v.y) : "l"(p));
    return v;
}
__device__ __forceinline__ uint4 ldg128(const void* p) {
    uint4 v;
    asm volatile("ld.global.nc.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    return v;
}
__device__ __forceinline__ uint32_t ldg32(const void* p) {
    uint32_t v;
    asm volatile("ld.global.nc.u32 %0, [%1];" : "=r"(v) : "l"(p));
    return v;
}
__device__ __forceinline__ int ldg_s8(const void* p) {
    int v;
    asm volatile("ld.global.nc.s8 %0, [%1];" : "=r"(v) : "l"(p));
    return v;
}
__device__ __forceinline__ uint32_t ldg_u16(const void* p) {
    unsigned short v;
    asm volatile("ld.global.nc.u16 %0, [%1];" : "=h"(v) : "l"(p));
    return v;
}

__device__ __forceinline__ __half2 h2u(uint32_t v) { return *reinterpret_cast<const __half2*>(&v); }
__device__ __forceinline__ uint32_t u32h2(__half2 h) { return *reinterpret_cast<const uint32_t*>(&h); }

// the four bytes b0..b3 of w (each 0 .. 255) as two half2 values scale * (b - off) + add: lo = lanes (b0, b1), hi = lanes (b2, b3)
__device__ __forceinline__ void cvt4(uint32_t w, __half2 off, __half2 scale, __half2 add, uint32_t& lo, uint32_t& hi) {
    // byte pairs with a 0x64 byte above each: the half2 (1024 + b_a, 1024 + b_b)
    lo = u32h2(__hfma2(__hsub2(h2u(__byte_perm(w, 0x64646464u, 0x4140)), off), scale, add));
    hi = u32h2(__hfma2(__hsub2(h2u(__byte_perm(w, 0x64646464u, 0x4342)), off), scale, add));
}

template <int F> struct QB;

// ---- Q4_K: 32 weights of a sub-block j are the low (j even) or high (j odd) nibbles of 32 bytes at 32 (j / 2); value = d sc code - dmin mn --------------------------------------------------------
template <> struct QB<FMT_Q4_K> {
    struct Raw {
        uint2 q;
        uint4 h;
        int j;
    };
    static __device__ __forceinline__ Raw load(const QP& p, size_t blk, int r) {
        Raw o;
        o.j = r >> 5;
        o.q = ldg64(p.p0 + blk * 128 + 32 * (o.j >> 1) + (r & 31));
        o.h = ldg128(p.p1 + blk * 16);
        return o;
    }
    static __device__ __forceinline__ uint4 decode(const Raw& o) {
        const int j = o.j;
        const uint32_t u0 = o.h.y, u1 = o.h.z, u2 = o.h.w;
        const uint32_t sw = j < 4 ? (u0 & M_K1) : ((u2 & M_LO) | (((u0 >> 6) & M_K3) << 4));
        const uint32_t mw = j < 4 ? (u1 & M_K1) : (((u2 >> 4) & M_LO) | (((u1 >> 6) & M_K3) << 4));
        const float sc = (float)((sw >> (8 * (j & 3))) & 0xFF), mn = (float)((mw >> (8 * (j & 3))) & 0xFF);
        const float d = __half2float(__ushort_as_half((unsigned short)(o.h.x & 0xFFFFu))), dmin = __half2float(__ushort_as_half((unsigned short)(o.h.x >> 16)));
        const __half2 scale = __float2half2_rn(d * sc), nbias = __float2half2_rn(-(dmin * mn)), off = __float2half2_rn(1024.f);
        const int sh = (j & 1) * 4;
        uint4 r;
        cvt4((o.q.x >> sh) & M_LO, off, scale, nbias, r.x, r.y);
        cvt4((o.q.y >> sh) & M_LO, off, scale, nbias, r.z, r.w);
        return r;
    }
};

// ---- Q5_K: as Q4_K plus the 5th bit from one transposed 32-bit word per 32-weight task -------------------------------------------------------------------------------------------------------------
template <> struct QB<FMT_Q5_K> {
    struct Raw {
        uint2 q;
        uint4 h;
        uint32_t qh;
        int j;
        int l;
    };
    static __device__ __forceinline__ Raw load(const QP& p, size_t blk, int r) {
        Raw o;
        o.j = r >> 5;
        o.l = r & 31;
        const int c = o.j >> 1;
        o.q = ldg64(p.p0 + blk * 128 + 32 * c + o.l);
        o.qh = ldg32(reinterpret_cast<const uint32_t*>(p.p1) + blk * 8 + 2 * c + (o.l >> 4));
        o.h = ldg128(p.p2 + blk * 16);
        return o;
    }
    static __device__ __forceinline__ uint4 decode(const Raw& o) {
        const int j = o.j;
        const uint32_t u0 = o.h.y, u1 = o.h.z, u2 = o.h.w;
        const uint32_t sw = j < 4 ? (u0 & M_K1) : ((u2 & M_LO) | (((u0 >> 6) & M_K3) << 4));
        const uint32_t mw = j < 4 ? (u1 & M_K1) : (((u2 >> 4) & M_LO) | (((u1 >> 6) & M_K3) << 4));
        const float sc = (float)((sw >> (8 * (j & 3))) & 0xFF), mn = (float)((mw >> (8 * (j & 3))) & 0xFF);
        const float d = __half2float(__ushort_as_half((unsigned short)(o.h.x & 0xFFFFu))), dmin = __half2float(__ushort_as_half((unsigned short)(o.h.x >> 16)));
        const __half2 scale = __float2half2_rn(d * sc), nbias = __float2half2_rn(-(dmin * mn)), off = __float2half2_rn(1024.f);
        const int sh = (j & 1) * 4;
        // the 5th bits: weight e of the 8 has bit (8 (e & 3) + pp0 + (e >> 2)) of the transposed word, i.e. one bit per byte for four weights at a time
        const int pp0 = (j & 1) * 4 + ((o.l & 15) >> 2);
        const uint32_t c0 = ((o.q.x >> sh) & M_LO) | (((o.qh >> pp0) & 0x01010101u) << 4);
        const uint32_t c1 = ((o.q.y >> sh) & M_LO) | (((o.qh >> (pp0 + 1)) & 0x01010101u) << 4);
        uint4 r;
        cvt4(c0, off, scale, nbias, r.x, r.y);
        cvt4(c1, off, scale, nbias, r.z, r.w);
        return r;
    }
};

// ---- Q6_K: a 6-bit code = 4 bits of ql (low or high nibble of byte A or B) + 2 bits of qh; value = d sc (code - 32) -----------------------------------------------------------------------------------
template <> struct QB<FMT_Q6_K> {
    struct Raw {
        uint2 a;
        uint2 hh;
        float scale;
        int g;
    };
    static __device__ __forceinline__ Raw load(const QP& p, size_t blk, int r) {
        Raw o;
        const int h = r >> 7, r2 = r & 127, l = r2 & 31, u = l >> 4;
        o.g = r2 >> 5;
        o.a = ldg64(p.p0 + blk * 128 + 64 * h + 32 * (o.g & 1) + l);
        o.hh = ldg64(p.p1 + blk * 64 + 32 * h + l);
        const int sc = ldg_s8(p.p2 + blk * 16 + 4 * (2 * h + u) + o.g);
        o.scale = __half2float(__ushort_as_half((unsigned short)ldg_u16(reinterpret_cast<const unsigned short*>(p.p3) + blk))) * (float)sc;
        return o;
    }
    static __device__ __forceinline__ uint4 decode(const Raw& o) {
        const int sh = (o.g >> 1) * 4, hsh = 2 * o.g;
        const uint32_t c0 = ((o.a.x >> sh) & M_LO) | (((o.hh.x >> hsh) & M_K3) << 4);
        const uint32_t c1 = ((o.a.y >> sh) & M_LO) | (((o.hh.y >> hsh) & M_K3) << 4);
        const __half2 scale = __float2half2_rn(o.scale), off = __float2half2_rn(1056.f), zero = __float2half2_rn(0.f);
        uint4 r;
        cvt4(c0, off, scale, zero, r.x, r.y);
        cvt4(c1, off, scale, zero, r.z, r.w);
        return r;
    }
};

template <int F, int EPI>
__device__ __forceinline__ void gemm_q_impl(const half* __restrict__ A, const QP w, void* __restrict__ Cv, int M, int N, int K, int lda, int ldc, float alpha) {
    __shared__ __align__(16) half As[2][128 * 16];
    __shared__ __align__(16) half Bs[2][128 * 16];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = (warp >> 2) * 64, wn = (warp & 3) * 32;
    // tile order: see gemm_impl3
    int m0, n0;
    if (gridDim.x > gridDim.y) {
        const int lin = blockIdx.y * gridDim.x + blockIdx.x;
        m0 = (lin % gridDim.y) * BM;
        n0 = (lin / gridDim.y) * BN;
    } else {
        m0 = blockIdx.y * BM;
        n0 = blockIdx.x * BN;
    }

    float acc[4][4][4];
#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
        for (int j = 0; j < 4; j++)
#pragma unroll
            for (int e = 0; e < 4; e++) acc[i][j][e] = 0.f;

    // this thread's chunk of the A tile and of the B tile: row `row`, 8 k values `ch * 8 ...` of the current 16
    const int row = tid >> 1, ch = tid & 1;
    const bool a_ok = m0 + row < M, b_ok = n0 + row < N;
    const half* pa = A + (size_t)(m0 + row) * lda + ch * 8;
    const size_t bblk = (size_t)(n0 + row) * (size_t)(K >> 8);
    int kcur = ch * 8;
    const int sts = row * 16 + 8 * (ch ^ ((row >> 2) & 1));
    uint4 ra = make_uint4(0u, 0u, 0u, 0u);
    typename QB<F>::Raw rq = {};
    auto ldg_tile = [&]() {
        if (a_ok) ra = ldg128(pa);
        if (b_ok) rq = QB<F>::load(w, bblk + (size_t)(kcur >> 8), kcur & 255);
        pa += 16;
        kcur += 16;
    };
    auto sts_tile = [&](int buf) {
        *reinterpret_cast<uint4*>(&As[buf][sts]) = ra;
        *reinterpret_cast<uint4*>(&Bs[buf][sts]) = b_ok ? QB<F>::decode(rq) : make_uint4(0u, 0u, 0u, 0u);
    };

    // ldmatrix lane addressing: see gemm_impl3 (row-major A and B)
    const int q = lane >> 3, lr = lane & 7, sw = (lr >> 2) & 1;
    const uint32_t a_ld = (uint32_t)(((wm + lr + (q & 1) * 8) * 16 + 8 * ((q >> 1) ^ sw)) * 2);
    const uint32_t b_ld = (uint32_t)(((wn + lr + (q >> 1) * 8) * 16 + 8 * ((q & 1) ^ sw)) * 2);
    constexpr uint32_t FRAG = 512;
    const uint32_t sa0 = smem_u32(&As[0][0]), sb0 = smem_u32(&Bs[0][0]);

    const int KT = K / 16;
    ldg_tile();
    sts_tile(0);
    if (KT > 1) ldg_tile();
    __syncthreads();
    for (int kt = 0; kt < KT; kt++) {
        const int buf = kt & 1;
        if (kt + 1 < KT) {
            sts_tile(buf ^ 1);
            if (kt + 2 < KT) ldg_tile();
        }
        const uint32_t sa = sa0 + buf * (128 * 16 * 2), sb = sb0 + buf * (128 * 16 * 2);
        uint32_t fa[4][4], fb[2][4];
#pragma unroll
        for (int i = 0; i < 4; i++) ldsm_x4(fa[i], sa + a_ld + i * FRAG);
#pragma unroll
        for (int j = 0; j < 2; j++) ldsm_x4(fb[j], sb + b_ld + j * FRAG);
#pragma unroll
        for (int s = 0; s < 2; s++)
#pragma unroll
            for (int i = 0; i < 4; i++)
#pragma unroll
                for (int j = 0; j < 2; j++) {
                    mma_k8(acc[i][2 * j], fa[i][2 * s], fa[i][2 * s + 1], fb[j][s]);
                    mma_k8(acc[i][2 * j + 1], fa[i][2 * s], fa[i][2 * s + 1], fb[j][2 + s]);
                }
        __syncthreads();
    }
    store_acc<EPI>(acc, Cv, M, N, ldc, alpha, nullptr, 0, 0.f, m0, n0, wm, wn, lane);
}

// The same decoding as a stand-alone pass: dequantize a whole matrix [rows][K] to fp16 (the prefill of chunks too long for the fused GEMMs, and formats without one use the generic dequant_f16_kernel). One
// thread writes 8 weights (16 bytes) per iteration, so the pass is limited by the memory system, not by the arithmetic of deq_elem (which decoded the block header again for every weight).
template <int F>
__device__ __forceinline__ void dequant_fast_impl(const QP w, half* __restrict__ out, int rows, int K) {
    const size_t total8 = (size_t)rows * (size_t)K / 8;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < total8; i += (size_t)gridDim.x * blockDim.x) {
        const size_t e = i * 8;
        const size_t row = e / (size_t)K;
        const int k = (int)(e - row * (size_t)K);
        reinterpret_cast<uint4*>(out)[i] = QB<F>::decode(QB<F>::load(w, row * (size_t)(K >> 8) + (size_t)(k >> 8), k & 255));
    }
}

}  // namespace gemm

#define DEQUANT_FAST_KERNEL(NAME, F)                                                                                                                                                            extern "C" __global__ void NAME(QPARAMS(w), half* out, int rows, int K) { gemm::dequant_fast_impl<F>(QMAKE(w), out, rows, K); }
DEQUANT_FAST_KERNEL(dequantk_q4k_kernel, FMT_Q4_K)
DEQUANT_FAST_KERNEL(dequantk_q5k_kernel, FMT_Q5_K)
DEQUANT_FAST_KERNEL(dequantk_q6k_kernel, FMT_Q6_K)

#define GEMM_Q_KERNEL(NAME, F, EPI, CT)                                                                                                                                                     \
    extern "C" __global__ void __launch_bounds__(256, 2) NAME(const half* A, QPARAMS(w), CT* C, int M, int N, int K, int lda, int ldc, float alpha) {                                         \
        gemm::gemm_q_impl<F, EPI>(A, QMAKE(w), C, M, N, K, lda, ldc, alpha);                                                                                                                \
    }

GEMM_Q_KERNEL(gemmq_q4k_f16, FMT_Q4_K, 0, half)
GEMM_Q_KERNEL(gemmq_q4k_f32, FMT_Q4_K, 1, float)
GEMM_Q_KERNEL(gemmq_q4k_add, FMT_Q4_K, 2, float)
GEMM_Q_KERNEL(gemmq_q5k_f16, FMT_Q5_K, 0, half)
GEMM_Q_KERNEL(gemmq_q5k_f32, FMT_Q5_K, 1, float)
GEMM_Q_KERNEL(gemmq_q5k_add, FMT_Q5_K, 2, float)
GEMM_Q_KERNEL(gemmq_q6k_f16, FMT_Q6_K, 0, half)
GEMM_Q_KERNEL(gemmq_q6k_f32, FMT_Q6_K, 1, float)
GEMM_Q_KERNEL(gemmq_q6k_add, FMT_Q6_K, 2, float)
