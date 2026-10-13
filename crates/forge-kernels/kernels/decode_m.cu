// ForgeInfer multi-row quantized matvec ("MMVQ"): a weight matrix W[rows][K] in the planar layouts of decode_q.cu times T pre-quantized activation vectors (int8 values, one scale and one integer sum per
// 32 values, token after token: xq [T][K], xd [T][K/32], xs [T][K/32]), the weights streamed from DRAM ONCE for all T tokens. The prompt processing of short chunks (a chat turn after a cached prefix,
// the tail of a hybrid model's prompt) runs its projections through these instead of the dequantizing tensor-core GEMM, whose decode work is the same for 8 or 64 rows.
//
// Structure (T = 4 / 8 / 12 / 16 rows, R = 2 rows per lane (R x T a multiple of 8), see MM_ALL at the end for the instantiations; up to three matrices of one format and one K go through one launch):
//   - a block stages the activation slice [k0, k1) of all T tokens in shared memory: the int8 chunks (16 bytes) in a per-format swizzled order, the per-32-values scale and sum turned into the
//     constants the dot needs. K is cut into `gridDim.y` slices of whole super-blocks (256 weights) so that T x slice fits the shared memory of a block (62 KB at most); gridDim.y > 1 writes
//     partial sums that mm_reduce_kernel adds up in a fixed order (the number of slices is chosen by the host from the format, K and the rows alone: the result does not depend on the grid).
//   - a warp is four SLOTS of 8 lanes; a slot owns R consecutive rows (a row group) and its lanes take the lane-tasks lane8 + 8 i of the slice (a lane-task is 32 weights of Q8_0 / Q4_0 / Q5_0, 64 of
//     Q4_K / Q5_K / Q6_K: the K-quants take both 16-byte halves of a sub-block pair, so the sub-block scales and minimums are applied once per 64 weights). The rows of K = 2560 (40 or 80
//     lane-tasks) divide evenly into 8 lanes, which is why the slots are 8 lanes wide (32 lanes on one row left 40 % of them idle at K = 2560).
//   - the weights of the R rows are loaded and UNPACKED (nibbles merged with their extra bits, scales split off and turned into floats) once per lane-task and then multiplied with every token's
//     activation slice, which the lane reads from shared memory once per (lane-task, token) for all R rows. An LDS.128 of a warp costs four shared-memory wavefronts whatever the addresses of its
//     lanes, so the shared-memory traffic per weight is 1 / R of a row-at-a-time kernel (decode_t.cu's mvt kernels) and the unpacking work 1 / T. R = 4 needs 241 registers (a block of 4 warps per SM:
//     slower), R = 2 is the best compromise; at T = 16 the kernel is limited by the shared-memory pipe (ncu: l1tex 80 %, DRAM 33 %).
//   - the loads of the next lane-task (or the first of the slot's next row group: the warps are persistent) are issued before the current one is consumed.
//   - the integer part of every dot product is exact (dp4a); its conversion to float is free: the accumulator of a dp4a chain starts at 0x4B400000, the bit pattern of 1.5 * 2^23, so the chain's result
//     IS the float 12582912 + P (|P| < 2^22) and one FADD of -12582912 yields P exactly (the I2F conversion of Turing runs at a quarter of the other instructions' rate).
//   - at the end of a row group the R x T accumulators of a slot are reduced over its 8 lanes by a reduce-scatter (xor 4, 2, 1); every lane writes V / 8 values (EPI 0: out = v, EPI 1: out += v,
//     the residual stream).
// A row's result does not depend on T, R, the number of warps, the launch grid or the slot it ran in (same lane-tasks, same order, same arithmetic per lane-task); it differs from the one-token
// kernels' (decode_q.cu) only by float rounding. Measured on one 9728 x 2560 Q4_K matrix (RTX 2060 SUPER, steady clocks): the one-token kernel 41 us, T = 4 44 us, T = 8 48 us, T = 16 100 us.
// tools/mrow_kbench.cu times every variant on its own. Included after decode_t.cu.

#define MM_BAD_TASK (1 << 30)    // a lane-task index that no row has: the loaders zero-fill it
#define MM_MAGIC_I 0x4B400000    // the float 12582912 = 1.5 * 2^23
#define MM_MAGIC_F 12582912.0f

__device__ __forceinline__ float mm_i2f(int chain) { return __int_as_float(chain) - MM_MAGIC_F; }

template <int F> struct MmG {
    static constexpr int TW = (F == FMT_Q8_0 || F == FMT_Q4_0 || F == FMT_Q5_0) ? 32 : 64;   // weights per lane-task
    static constexpr int SCW = (F == FMT_Q8_0 || F == FMT_Q6_K) ? 1 : 2;                       // floats per 32 values in the scale array
};

// the order of the 16-byte activation chunks in shared memory: the lanes of a slot read chunks 2 (legacy formats), 4 (Q4_K / Q5_K) or 2 x (task, group) apart; these swizzles (they permute inside
// aligned groups of 4 or 8 chunks) put the eight lanes of a slot on eight different 16-byte bank groups
template <int F>
__device__ __forceinline__ int mm_swz(int c) {
    if constexpr (F == FMT_Q4_K || F == FMT_Q5_K) return c ^ (((c >> 3) & 1) << 1) ^ ((c >> 4) & 1);
    else if constexpr (F == FMT_Q6_K) return c ^ ((c >> 2) & 6);
    else return c ^ ((c >> 3) & 1);
}

// ---- the weights of one lane-task of one row as loaded ---------------------------------------------------------------------------------------------------------------------------------------------
template <int F> struct MmRaw;
template <> struct MmRaw<FMT_Q8_0> { uint4 q0, q1; float d; };
template <> struct MmRaw<FMT_Q4_0> { uint4 q; float d; };
template <> struct MmRaw<FMT_Q5_0> { uint4 q; uint32_t h; float d; };
template <> struct MmRaw<FMT_Q4_K> { uint4 q0, q1, hdr; };
template <> struct MmRaw<FMT_Q5_K> { uint4 q0, q1, hdr; uint32_t h0, h1; };
template <> struct MmRaw<FMT_Q6_K> { uint4 a, b, h; uint32_t sc; float d; };

#define MM_ZERO4 make_uint4(0u, 0u, 0u, 0u)

// task: the global lane-task index within the row (MM_BAD_TASK or beyond the row: zeros)
__device__ __forceinline__ void mm_load(MmRaw<FMT_Q8_0>& r, const QP& p, size_t row, int K, int task) {
    if (task < K / 32) {
        r.q0 = ldq(p.p0, row * (K / 16) + 2 * task);
        r.q1 = ldq(p.p0, row * (K / 16) + 2 * task + 1);
        r.d = ldh(p.p1, row * (K / 32) + task);
    } else {
        r.q0 = r.q1 = MM_ZERO4;
        r.d = 0.f;
    }
}
__device__ __forceinline__ void mm_load(MmRaw<FMT_Q4_0>& r, const QP& p, size_t row, int K, int task) {
    if (task < K / 32) {
        r.q = ldq(p.p0, row * (K / 32) + task);
        r.d = ldh(p.p1, row * (K / 32) + task);
    } else {
        r.q = MM_ZERO4;
        r.d = 0.f;
    }
}
__device__ __forceinline__ void mm_load(MmRaw<FMT_Q5_0>& r, const QP& p, size_t row, int K, int task) {
    if (task < K / 32) {
        r.q = ldq(p.p0, row * (K / 32) + task);
        r.h = ldw(p.p1, row * (K / 32) + task);
        r.d = ldh(p.p2, row * (K / 32) + task);
    } else {
        r.q = MM_ZERO4;
        r.h = 0u;
        r.d = 0.f;
    }
}
template <bool Q5>
__device__ __forceinline__ void mm_load_qk(MmRaw<Q5 ? FMT_Q5_K : FMT_Q4_K>& r, const QP& p, size_t row, int K, int task) {
    if (task < K / 64) {
        const int nsb = K >> 8, sb = task >> 2, c = task & 3;
        const size_t blk = row * nsb + sb;
        r.q0 = ldq(p.p0, blk * 8 + 2 * c);
        r.q1 = ldq(p.p0, blk * 8 + 2 * c + 1);
        r.hdr = ldq(Q5 ? p.p2 : p.p1, blk);
        if constexpr (Q5) {
            r.h0 = ldw(p.p1, blk * 8 + 2 * c);
            r.h1 = ldw(p.p1, blk * 8 + 2 * c + 1);
        }
    } else {
        r.q0 = r.q1 = r.hdr = MM_ZERO4;
        if constexpr (Q5) r.h0 = r.h1 = 0u;
    }
}
__device__ __forceinline__ void mm_load(MmRaw<FMT_Q4_K>& r, const QP& p, size_t row, int K, int task) { mm_load_qk<false>(r, p, row, K, task); }
__device__ __forceinline__ void mm_load(MmRaw<FMT_Q5_K>& r, const QP& p, size_t row, int K, int task) { mm_load_qk<true>(r, p, row, K, task); }
__device__ __forceinline__ void mm_load(MmRaw<FMT_Q6_K>& r, const QP& p, size_t row, int K, int task) {
    if (task < K / 64) {
        const int nsb = K >> 8, sb = task >> 2, h = (task >> 1) & 1, u = task & 1;
        const size_t blk = row * nsb + sb;
        r.a = ldq(p.p0, blk * 8 + 4 * h + u);
        r.b = ldq(p.p0, blk * 8 + 4 * h + 2 + u);
        r.h = ldq(p.p1, blk * 4 + 2 * h + u);
        r.sc = ldw(p.p2, blk * 4 + 2 * h + u);
        r.d = ldh(p.p3, blk);
    } else {
        r.a = r.b = r.h = MM_ZERO4;
        r.sc = 0u;
        r.d = 0.f;
    }
}

// ---- unpacked ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
template <int F> struct UW;
template <> struct UW<FMT_Q8_0> { uint32_t w[8]; float d; };
template <> struct UW<FMT_Q4_0> { uint32_t w[8]; float d; };
template <> struct UW<FMT_Q5_0> { uint32_t w[8]; float d; };
template <> struct UW<FMT_Q4_K> { uint32_t lw[8], hw[8]; float al, ah, bl, bh; };   // low / high nibble operands; al, ah = d * scale of the two sub-blocks, bl, bh = dmin * minimum
template <> struct UW<FMT_Q5_K> { uint32_t lw[8], hw[8]; float al, ah, bl, bh; };
template <> struct UW<FMT_Q6_K> { uint32_t f[16]; float rho[4]; };                    // rho[g] = d / 4 * scale of group g

__device__ __forceinline__ void mm_unpack(UW<FMT_Q8_0>& u, const MmRaw<FMT_Q8_0>& r, int) {
    u.w[0] = r.q0.x; u.w[1] = r.q0.y; u.w[2] = r.q0.z; u.w[3] = r.q0.w;
    u.w[4] = r.q1.x; u.w[5] = r.q1.y; u.w[6] = r.q1.z; u.w[7] = r.q1.w;
    u.d = r.d;
}
template <int F, bool Q5>
__device__ __forceinline__ void mm_unpack_q40(UW<F>& u, const uint4 q, uint32_t qh, float d) {
    const uint32_t w0 = q.x, w1 = q.y, w2 = q.z, w3 = q.w;
    uint32_t l0 = w0 & M_LO, l1 = w1 & M_LO, l2 = w2 & M_LO, l3 = w3 & M_LO;
    uint32_t h0 = (w0 >> 4) & M_LO, h1 = (w1 >> 4) & M_LO, h2 = (w2 >> 4) & M_LO, h3 = (w3 >> 4) & M_LO;
    if (Q5) {
        l0 |= tb5(qh, 0); l1 |= tb5(qh, 1); l2 |= tb5(qh, 2); l3 |= tb5(qh, 3);
        h0 |= tb5(qh, 4); h1 |= tb5(qh, 5); h2 |= tb5(qh, 6); h3 |= tb5(qh, 7);
    }
    u.w[0] = l0; u.w[1] = l1; u.w[2] = l2; u.w[3] = l3;
    u.w[4] = h0; u.w[5] = h1; u.w[6] = h2; u.w[7] = h3;
    u.d = d;
}
__device__ __forceinline__ void mm_unpack(UW<FMT_Q4_0>& u, const MmRaw<FMT_Q4_0>& r, int) { mm_unpack_q40<FMT_Q4_0, false>(u, r.q, 0u, r.d); }
__device__ __forceinline__ void mm_unpack(UW<FMT_Q5_0>& u, const MmRaw<FMT_Q5_0>& r, int) { mm_unpack_q40<FMT_Q5_0, true>(u, r.q, r.h, r.d); }

template <int F, bool Q5>
__device__ __forceinline__ void mm_unpack_qk(UW<F>& u, const MmRaw<F>& r, int task) {
    const int c = task & 3;
    const uint4 hd = r.hdr;   // zero for a lane-task that is not there: d = dmin = 0
    const float d = __half2float(__ushort_as_half((unsigned short)(hd.x & 0xFFFFu)));
    const float dmin = __half2float(__ushort_as_half((unsigned short)(hd.x >> 16)));
    const uint32_t u0 = hd.y, u1 = hd.z, u2 = hd.w;
    const uint32_t sw = c < 2 ? (u0 & M_K1) : ((u2 & M_LO) | (((u0 >> 6) & M_K3) << 4));
    const uint32_t mw = c < 2 ? (u1 & M_K1) : (((u2 >> 4) & M_LO) | (((u1 >> 6) & M_K3) << 4));
    const int sh = 8 * ((2 * c) & 3);
    u.al = d * (float)((sw >> sh) & 0xFF);
    u.ah = d * (float)((sw >> (sh + 8)) & 0xFF);
    u.bl = dmin * (float)((mw >> sh) & 0xFF);
    u.bh = dmin * (float)((mw >> (sh + 8)) & 0xFF);
    // the two 16-byte halves: bytes 0..15 of the chunk are weights 0..15 of the sub-blocks 2c (low nibbles) and 2c + 1 (high nibbles), bytes 16..31 weights 16..31
#pragma unroll
    for (int half = 0; half < 2; half++) {
        const uint4 q = half == 0 ? r.q0 : r.q1;
        uint32_t l0 = q.x & M_LO, l1 = q.y & M_LO, l2 = q.z & M_LO, l3 = q.w & M_LO;
        uint32_t h0 = (q.x >> 4) & M_LO, h1 = (q.y >> 4) & M_LO, h2 = (q.z >> 4) & M_LO, h3 = (q.w >> 4) & M_LO;
        if constexpr (Q5) {
            const uint32_t qh = half == 0 ? r.h0 : r.h1;
            l0 |= tb5(qh, 0); l1 |= tb5(qh, 1); l2 |= tb5(qh, 2); l3 |= tb5(qh, 3);
            h0 |= tb5(qh, 4); h1 |= tb5(qh, 5); h2 |= tb5(qh, 6); h3 |= tb5(qh, 7);
        }
        u.lw[4 * half] = l0; u.lw[4 * half + 1] = l1; u.lw[4 * half + 2] = l2; u.lw[4 * half + 3] = l3;
        u.hw[4 * half] = h0; u.hw[4 * half + 1] = h1; u.hw[4 * half + 2] = h2; u.hw[4 * half + 3] = h3;
    }
}
__device__ __forceinline__ void mm_unpack(UW<FMT_Q4_K>& u, const MmRaw<FMT_Q4_K>& r, int task) { mm_unpack_qk<FMT_Q4_K, false>(u, r, task); }
__device__ __forceinline__ void mm_unpack(UW<FMT_Q5_K>& u, const MmRaw<FMT_Q5_K>& r, int task) { mm_unpack_qk<FMT_Q5_K, true>(u, r, task); }

__device__ __forceinline__ void mm_unpack(UW<FMT_Q6_K>& u, const MmRaw<FMT_Q6_K>& r, int) {
    const uint4 A = r.a, B = r.b, H = r.h;
    // 4 * (code - 32) as signed bytes, see dot_q6_group
    u.f[0] = (((A.x << 2) & 0x3C3C3C3Cu) | ((H.x << 6) & 0xC0C0C0C0u)) ^ 0x80808080u;
    u.f[1] = (((A.y << 2) & 0x3C3C3C3Cu) | ((H.y << 6) & 0xC0C0C0C0u)) ^ 0x80808080u;
    u.f[2] = (((A.z << 2) & 0x3C3C3C3Cu) | ((H.z << 6) & 0xC0C0C0C0u)) ^ 0x80808080u;
    u.f[3] = (((A.w << 2) & 0x3C3C3C3Cu) | ((H.w << 6) & 0xC0C0C0C0u)) ^ 0x80808080u;
    u.f[4] = (((B.x << 2) & 0x3C3C3C3Cu) | ((H.x << 4) & 0xC0C0C0C0u)) ^ 0x80808080u;
    u.f[5] = (((B.y << 2) & 0x3C3C3C3Cu) | ((H.y << 4) & 0xC0C0C0C0u)) ^ 0x80808080u;
    u.f[6] = (((B.z << 2) & 0x3C3C3C3Cu) | ((H.z << 4) & 0xC0C0C0C0u)) ^ 0x80808080u;
    u.f[7] = (((B.w << 2) & 0x3C3C3C3Cu) | ((H.w << 4) & 0xC0C0C0C0u)) ^ 0x80808080u;
    u.f[8] = (((A.x >> 2) & 0x3C3C3C3Cu) | ((H.x << 2) & 0xC0C0C0C0u)) ^ 0x80808080u;
    u.f[9] = (((A.y >> 2) & 0x3C3C3C3Cu) | ((H.y << 2) & 0xC0C0C0C0u)) ^ 0x80808080u;
    u.f[10] = (((A.z >> 2) & 0x3C3C3C3Cu) | ((H.z << 2) & 0xC0C0C0C0u)) ^ 0x80808080u;
    u.f[11] = (((A.w >> 2) & 0x3C3C3C3Cu) | ((H.w << 2) & 0xC0C0C0C0u)) ^ 0x80808080u;
    u.f[12] = (((B.x >> 2) & 0x3C3C3C3Cu) | (H.x & 0xC0C0C0C0u)) ^ 0x80808080u;
    u.f[13] = (((B.y >> 2) & 0x3C3C3C3Cu) | (H.y & 0xC0C0C0C0u)) ^ 0x80808080u;
    u.f[14] = (((B.z >> 2) & 0x3C3C3C3Cu) | (H.z & 0xC0C0C0C0u)) ^ 0x80808080u;
    u.f[15] = (((B.w >> 2) & 0x3C3C3C3Cu) | (H.w & 0xC0C0C0C0u)) ^ 0x80808080u;
    const uint32_t sc = r.sc;
    const float d4 = r.d * 0.25f;
#pragma unroll
    for (int g = 0; g < 4; g++) u.rho[g] = d4 * (float)(int)(signed char)((sc >> (8 * g)) & 0xFF);
}

// ---- the activation side of one lane-task of one token: the int8 slice and the scale constants. The per-32-values scale array of the shared-memory layout holds
//      Q8_0, Q6_K: xd;   Q4_0, Q5_0: (xd, MAGIC - off * xs) with off = 8 / 16;   Q4_K, Q5_K: (xd, xd * xs)
template <int F> struct MmAct;
template <> struct MmAct<FMT_Q8_0> { uint4 xl, xh; float d; };
template <> struct MmAct<FMT_Q4_0> { uint4 xl, xh; float d; int init; };
template <> struct MmAct<FMT_Q5_0> { uint4 xl, xh; float d; int init; };
template <> struct MmAct<FMT_Q4_K> { uint4 x[4]; float dl, sl, dh, sh; };
template <> struct MmAct<FMT_Q5_K> { uint4 x[4]; float dl, sl, dh, sh; };
template <> struct MmAct<FMT_Q6_K> { uint4 x[4]; float4 d; };

// the activation of the lane-task tl of the slice (local index; >= the number of tasks: zeros); xq: the token's chunks, sc: the token's scale array (both of the slice)
__device__ __forceinline__ void mm_act(MmAct<FMT_Q8_0>& a, const uint4* xq, const float* sc, int tl, int ntl) {
    if (tl < ntl) {
        const int s = (tl >> 2) & 1;
        a.xl = xq[2 * tl + s];
        a.xh = xq[2 * tl + (1 ^ s)];
        a.d = sc[tl];
    } else {
        a.xl = a.xh = MM_ZERO4;
        a.d = 0.f;
    }
}
template <int F>
__device__ __forceinline__ void mm_act_q40(MmAct<F>& a, const uint4* xq, const float* sc, int tl, int ntl) {
    if (tl < ntl) {
        const int s = (tl >> 2) & 1;
        a.xl = xq[2 * tl + s];
        a.xh = xq[2 * tl + (1 ^ s)];
        const float2 v = reinterpret_cast<const float2*>(sc)[tl];
        a.d = v.x;
        a.init = __float_as_int(v.y);
    } else {
        a.xl = a.xh = MM_ZERO4;
        a.d = 0.f;
        a.init = MM_MAGIC_I;
    }
}
__device__ __forceinline__ void mm_act(MmAct<FMT_Q4_0>& a, const uint4* xq, const float* sc, int tl, int ntl) { mm_act_q40<FMT_Q4_0>(a, xq, sc, tl, ntl); }
__device__ __forceinline__ void mm_act(MmAct<FMT_Q5_0>& a, const uint4* xq, const float* sc, int tl, int ntl) { mm_act_q40<FMT_Q5_0>(a, xq, sc, tl, ntl); }
template <int F>
__device__ __forceinline__ void mm_act_qk(MmAct<F>& a, const uint4* xq, const float* sc, int tl, int ntl) {
    if (tl < ntl) {
        const int f = (((tl >> 1) & 1) << 1) | ((tl >> 2) & 1);
#pragma unroll
        for (int k = 0; k < 4; k++) a.x[k] = xq[4 * tl + (k ^ f)];
        const float4 v = reinterpret_cast<const float4*>(sc)[tl];   // (xd, xd * xs) of the sub-blocks 2c and 2c + 1
        a.dl = v.x;
        a.sl = v.y;
        a.dh = v.z;
        a.sh = v.w;
    } else {
#pragma unroll
        for (int k = 0; k < 4; k++) a.x[k] = MM_ZERO4;
        a.dl = a.sl = a.dh = a.sh = 0.f;
    }
}
__device__ __forceinline__ void mm_act(MmAct<FMT_Q4_K>& a, const uint4* xq, const float* sc, int tl, int ntl) { mm_act_qk<FMT_Q4_K>(a, xq, sc, tl, ntl); }
__device__ __forceinline__ void mm_act(MmAct<FMT_Q5_K>& a, const uint4* xq, const float* sc, int tl, int ntl) { mm_act_qk<FMT_Q5_K>(a, xq, sc, tl, ntl); }
__device__ __forceinline__ void mm_act(MmAct<FMT_Q6_K>& a, const uint4* xq, const float* sc, int tl, int ntl) {
    if (tl < ntl) {
        const int sb = tl >> 2, h = (tl >> 1) & 1, u = tl & 1;
        const int m = (h << 1) | ((sb & 1) << 2);
#pragma unroll
        for (int g = 0; g < 4; g++) a.x[g] = xq[16 * sb + 8 * h + ((2 * g + u) ^ m)];
        a.d = reinterpret_cast<const float4*>(sc)[sb * 2 + h];   // the four groups' scales: blocks sb * 8 + 4h + g
    } else {
#pragma unroll
        for (int g = 0; g < 4; g++) a.x[g] = MM_ZERO4;
        a.d = make_float4(0.f, 0.f, 0.f, 0.f);
    }
}

// acc += the contribution of one lane-task of one row to the dot product with one token
__device__ __forceinline__ void mm_dot(float& acc, const UW<FMT_Q8_0>& u, const MmAct<FMT_Q8_0>& a) {
    int t = dp4(u.w[0], a.xl.x, MM_MAGIC_I);
    t = dp4(u.w[1], a.xl.y, t);
    t = dp4(u.w[2], a.xl.z, t);
    t = dp4(u.w[3], a.xl.w, t);
    t = dp4(u.w[4], a.xh.x, t);
    t = dp4(u.w[5], a.xh.y, t);
    t = dp4(u.w[6], a.xh.z, t);
    t = dp4(u.w[7], a.xh.w, t);
    acc = fmaf(mm_i2f(t), u.d * a.d, acc);
}
template <int F>
__device__ __forceinline__ void mm_dot_q40(float& acc, const UW<F>& u, const MmAct<F>& a) {
    int t = dp4(u.w[0], a.xl.x, a.init);   // init = MAGIC - offset * (sum of the activation block)
    t = dp4(u.w[1], a.xl.y, t);
    t = dp4(u.w[2], a.xl.z, t);
    t = dp4(u.w[3], a.xl.w, t);
    t = dp4(u.w[4], a.xh.x, t);
    t = dp4(u.w[5], a.xh.y, t);
    t = dp4(u.w[6], a.xh.z, t);
    t = dp4(u.w[7], a.xh.w, t);
    acc = fmaf(mm_i2f(t), u.d * a.d, acc);
}
__device__ __forceinline__ void mm_dot(float& acc, const UW<FMT_Q4_0>& u, const MmAct<FMT_Q4_0>& a) { mm_dot_q40<FMT_Q4_0>(acc, u, a); }
__device__ __forceinline__ void mm_dot(float& acc, const UW<FMT_Q5_0>& u, const MmAct<FMT_Q5_0>& a) { mm_dot_q40<FMT_Q5_0>(acc, u, a); }
template <int F>
__device__ __forceinline__ void mm_dot_qk(float& acc, const UW<F>& u, const MmAct<F>& a) {
    // sub-block 2c (low nibbles): positions 0..15 against chunk 0, 16..31 against chunk 1; sub-block 2c + 1 (high nibbles): chunks 2, 3
    int pl = dp4(u.lw[0], a.x[0].x, MM_MAGIC_I);
    int ph = dp4(u.hw[0], a.x[2].x, MM_MAGIC_I);
    pl = dp4(u.lw[1], a.x[0].y, pl);
    ph = dp4(u.hw[1], a.x[2].y, ph);
    pl = dp4(u.lw[2], a.x[0].z, pl);
    ph = dp4(u.hw[2], a.x[2].z, ph);
    pl = dp4(u.lw[3], a.x[0].w, pl);
    ph = dp4(u.hw[3], a.x[2].w, ph);
    pl = dp4(u.lw[4], a.x[1].x, pl);
    ph = dp4(u.hw[4], a.x[3].x, ph);
    pl = dp4(u.lw[5], a.x[1].y, pl);
    ph = dp4(u.hw[5], a.x[3].y, ph);
    pl = dp4(u.lw[6], a.x[1].z, pl);
    ph = dp4(u.hw[6], a.x[3].z, ph);
    pl = dp4(u.lw[7], a.x[1].w, pl);
    ph = dp4(u.hw[7], a.x[3].w, ph);
    acc = fmaf(mm_i2f(pl), u.al * a.dl, acc);
    acc = fmaf(mm_i2f(ph), u.ah * a.dh, acc);
    acc = fmaf(-u.bl, a.sl, acc);
    acc = fmaf(-u.bh, a.sh, acc);
}
__device__ __forceinline__ void mm_dot(float& acc, const UW<FMT_Q4_K>& u, const MmAct<FMT_Q4_K>& a) { mm_dot_qk<FMT_Q4_K>(acc, u, a); }
__device__ __forceinline__ void mm_dot(float& acc, const UW<FMT_Q5_K>& u, const MmAct<FMT_Q5_K>& a) { mm_dot_qk<FMT_Q5_K>(acc, u, a); }
__device__ __forceinline__ void mm_dot(float& acc, const UW<FMT_Q6_K>& u, const MmAct<FMT_Q6_K>& a) {
    const float dd[4] = {a.d.x, a.d.y, a.d.z, a.d.w};
#pragma unroll
    for (int g = 0; g < 4; g++) {
        const uint4 x = a.x[g];
        int t = dp4(u.f[4 * g], x.x, MM_MAGIC_I);
        t = dp4(u.f[4 * g + 1], x.y, t);
        t = dp4(u.f[4 * g + 2], x.z, t);
        t = dp4(u.f[4 * g + 3], x.w, t);
        acc = fmaf(mm_i2f(t), u.rho[g] * dd[g], acc);
    }
}

// the scale constants of a 32-value block for the shared-memory scale array: xd and xs of the global activation arrays
template <int F>
__device__ __forceinline__ void mm_stage_sc(float* sc, int i, float xd, int xs) {
    if constexpr (F == FMT_Q8_0 || F == FMT_Q6_K) {
        sc[i] = xd;
    } else if constexpr (F == FMT_Q4_0) {
        sc[2 * i] = xd;
        sc[2 * i + 1] = __int_as_float(MM_MAGIC_I - 8 * xs);
    } else if constexpr (F == FMT_Q5_0) {
        sc[2 * i] = xd;
        sc[2 * i + 1] = __int_as_float(MM_MAGIC_I - 16 * xs);
    } else {
        sc[2 * i] = xd;
        sc[2 * i + 1] = xd * (float)xs;
    }
}

// ==============================================================================================================================================================================================
// Reduce V values (a multiple of 8) of every lane of an 8-lane slot over the slot: lane8 ends up with the sums of the values [lane8 V / 8, (lane8 + 1) V / 8).
// ==============================================================================================================================================================================================
template <int V>
__device__ __forceinline__ void mm_rs8(const float (&a)[V], float (&o)[V / 8], int lane) {
    static_assert(V >= 8 && V % 8 == 0, "R x T accumulators: a multiple of 8");
    const bool u4 = (lane & 4) != 0, u2 = (lane & 2) != 0, u1 = (lane & 1) != 0;
    float b[V / 2];
#pragma unroll
    for (int i = 0; i < V / 2; i++) {
        const float keep = u4 ? a[i + V / 2] : a[i];
        const float send = u4 ? a[i] : a[i + V / 2];
        b[i] = keep + __shfl_xor_sync(0xffffffffu, send, 4);
    }
    float c[V / 4];
#pragma unroll
    for (int i = 0; i < V / 4; i++) {
        const float keep = u2 ? b[i + V / 4] : b[i];
        const float send = u2 ? b[i] : b[i + V / 4];
        c[i] = keep + __shfl_xor_sync(0xffffffffu, send, 2);
    }
#pragma unroll
    for (int i = 0; i < V / 8; i++) {
        const float keep = u1 ? c[i + V / 8] : c[i];
        const float send = u1 ? c[i] : c[i + V / 8];
        o[i] = keep + __shfl_xor_sync(0xffffffffu, send, 1);
    }
}

// ==============================================================================================================================================================================================
// The kernel. Up to three matrices of one format and one K (q / k / v, gate / up) go through one launch, their row groups one after the other: matrix i gives out_i[t * ldo_i + row] (EPI 0) or += (EPI 1)
// for the tokens t < T. With gridDim.y > 1 the K slices write their partial sums to out_i[j * (T * ldo_i) + t * ldo_i + row] instead (EPI is ignored; mm_reduce_kernel adds them up).
// Dynamic shared memory: mm_smem_bytes<F>(T, slice size).
// ==============================================================================================================================================================================================
struct MmMat {
    QP w;
    float* out;
    int rows;
    int ldo;
};
struct MmMats {
    MmMat m[3];
    int n;
};

template <int F, int T, int R, int EPI>
__device__ __forceinline__ void mm_impl(const MmMats mats, const int8_t* __restrict__ xq_g, const float* __restrict__ xd_g, const int* __restrict__ xs_g, int K) {
    constexpr int TW = MmG<F>::TW;
    constexpr int SCW = MmG<F>::SCW;
    constexpr int V = T * R;
    static_assert(V % 8 == 0 && V <= 32, "R x T accumulators");
    extern __shared__ __align__(16) unsigned char smem_raw[];
    const int nsb = K >> 8, j = blockIdx.y, nj = gridDim.y;
    const int sb0 = (nsb * j) / nj, sb1 = (nsb * (j + 1)) / nj;
    const int k0 = sb0 << 8, KS = (sb1 - sb0) << 8;
    const int nch = KS >> 4, nb32 = KS >> 5;
    uint4* sxq = reinterpret_cast<uint4*>(smem_raw);
    float* ssc = reinterpret_cast<float*>(smem_raw + (size_t)T * KS);
    // the activation slice of every token: slice chunk c at position mm_swz(c)
    for (int i = threadIdx.x; i < T * nch; i += blockDim.x) {
        const int t = i / nch, c = i - t * nch;
        sxq[(size_t)t * nch + mm_swz<F>(c)] = __ldg(reinterpret_cast<const uint4*>(xq_g + (size_t)t * K + k0) + c);
    }
    for (int i = threadIdx.x; i < T * nb32; i += blockDim.x) {
        const int t = i / nb32, b = i - t * nb32;
        mm_stage_sc<F>(ssc + (size_t)t * nb32 * SCW, b, __ldg(xd_g + (size_t)t * (K >> 5) + (k0 >> 5) + b), __ldg(xs_g + (size_t)t * (K >> 5) + (k0 >> 5) + b));
    }
    __syncthreads();
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
    const int lane8 = lane & 7, slot = lane >> 3;
    const int G0 = mats.m[0].rows / R;
    const int G1 = G0 + (mats.n > 1 ? mats.m[1].rows / R : 0);
    const int ngroups = G1 + (mats.n > 2 ? mats.m[2].rows / R : 0);
    const int gstride = gridDim.x * nw * 4;
    int g0 = (blockIdx.x * nw + warp) * 4;      // the row group of slot 0; slot s takes g0 + s
    if (g0 >= ngroups) return;
    const int ntl = KS / TW;                    // lane-tasks of the slice
    const int nit = (ntl + 7) >> 3;
    const int task_base = k0 / TW;

    float acc[T][R];
#pragma unroll
    for (int t = 0; t < T; t++)
#pragma unroll
        for (int r = 0; r < R; r++) acc[t][r] = 0.f;
    MmRaw<F> raw[R];
    UW<F> cur[R];
    int it = 0;
    {
        const int g = g0 + slot;
        const int tl = lane8;
        const int arg = (g < ngroups && tl < ntl) ? task_base + tl : MM_BAD_TASK;
        const int mi = g < G0 ? 0 : (g < G1 ? 1 : 2);
        const size_t row0 = (size_t)(g - (mi == 0 ? 0 : (mi == 1 ? G0 : G1))) * R;
#pragma unroll
        for (int r = 0; r < R; r++) mm_load(raw[r], mats.m[mi].w, row0 + r, K, arg);
#pragma unroll
        for (int r = 0; r < R; r++) mm_unpack(cur[r], raw[r], arg);
    }
    while (true) {
        int gn0 = g0, itn = it + 1;
        if (itn == nit) {
            itn = 0;
            gn0 += gstride;
        }
        const bool more = gn0 < ngroups;
        int argn = MM_BAD_TASK;
        if (more) {
            const int gn = gn0 + slot;
            const int tl = lane8 + 8 * itn;
            argn = (gn < ngroups && tl < ntl) ? task_base + tl : MM_BAD_TASK;
            const int mi = gn < G0 ? 0 : (gn < G1 ? 1 : 2);
            const size_t row0 = (size_t)(gn - (mi == 0 ? 0 : (mi == 1 ? G0 : G1))) * R;
#pragma unroll
            for (int r = 0; r < R; r++) mm_load(raw[r], mats.m[mi].w, row0 + r, K, argn);
        }
        {
            const int tl = lane8 + 8 * it;
#pragma unroll
            for (int t = 0; t < T; t++) {
                MmAct<F> a;
                mm_act(a, sxq + (size_t)t * nch, ssc + (size_t)t * nb32 * SCW, tl, ntl);
#pragma unroll
                for (int r = 0; r < R; r++) mm_dot(acc[t][r], cur[r], a);
            }
        }
        if (it + 1 == nit) {
            // the row group of the slot is complete: reduce over its 8 lanes, write
            float v[V];
#pragma unroll
            for (int t = 0; t < T; t++)
#pragma unroll
                for (int r = 0; r < R; r++) {
                    v[t * R + r] = acc[t][r];
                    acc[t][r] = 0.f;
                }
            float s[V / 8];
            mm_rs8<V>(v, s, lane);
            const int g = g0 + slot;
            if (g < ngroups) {
                const int mi = g < G0 ? 0 : (g < G1 ? 1 : 2);
                float* out = mats.m[mi].out;
                const int ldo = mats.m[mi].ldo;
                const size_t row0 = (size_t)(g - (mi == 0 ? 0 : (mi == 1 ? G0 : G1))) * R;
#pragma unroll
                for (int e = 0; e < V / 8; e++) {
                    const int idx = lane8 * (V / 8) + e;
                    const int t = idx / R, r = idx % R;
                    const size_t row = row0 + r;
                    if (nj > 1) out[(size_t)j * ((size_t)T * ldo) + (size_t)t * ldo + row] = s[e];
                    else if (EPI == 0) out[(size_t)t * ldo + row] = s[e];
                    else out[(size_t)t * ldo + row] += s[e];
                }
            }
        }
        if (!more) break;
#pragma unroll
        for (int r = 0; r < R; r++) mm_unpack(cur[r], raw[r], argn);
        g0 = gn0;
        it = itn;
    }
}

// shared memory of a block: the int8 slice (T x KS bytes) and T x KS / 32 scale entries
template <int F>
__host__ __device__ constexpr size_t mm_smem_bytes(int T, int KS) { return (size_t)T * KS + (size_t)T * (KS / 32) * 4 * MmG<F>::SCW; }

// out_i[t * ldo_i + r] (+)= sum over the K slices j of part_i[j * (T * ldo_i) + t * ldo_i + r], r < rows_i, t < T = gridDim.y, matrix i = blockIdx.z; grid (ceil(max rows / 256), T, number of matrices)
template <int EPI>
__device__ __forceinline__ void mm_reduce_impl(const float* __restrict__ part, int nj, float* __restrict__ out, int ldo, int rows) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (r >= rows) return;
    const size_t plane = (size_t)gridDim.y * ldo;
    float s = part[(size_t)t * ldo + r];
    for (int j = 1; j < nj; j++) s += part[(size_t)j * plane + (size_t)t * ldo + r];
    if (EPI == 0) out[(size_t)t * ldo + r] = s;
    else out[(size_t)t * ldo + r] += s;
}
#define MM_REDUCE_KERNEL(NAME, EPI) \
    extern "C" __global__ void NAME(const float* pa, const float* pb, const float* pc, float* oa, float* ob, float* oc, int ra, int rb, int rc, int la, int lb, int lc, int nj) { \
        if (blockIdx.z == 0) mm_reduce_impl<EPI>(pa, nj, oa, la, ra); \
        else if (blockIdx.z == 1) mm_reduce_impl<EPI>(pb, nj, ob, lb, rb); \
        else mm_reduce_impl<EPI>(pc, nj, oc, lc, rc); \
    }
MM_REDUCE_KERNEL(mm_reduce_out_kernel, 0)
MM_REDUCE_KERNEL(mm_reduce_res_kernel, 1)

#ifndef MM_NO_INST   // (tools/mrow_kbench.cu instantiates the template itself)
// The instantiations: mm_out_<fmt>_t<T>_kernel (out = v) and mm_res_<fmt>_t<T>_kernel (out += v); any block size up to 384 threads (the number of warps is read at run time).
#define MM_KERNEL(F, SUF, T, R, EPI, NAME) \
    extern "C" __global__ void __launch_bounds__(384, 1) mm_##NAME##_##SUF##_t##T##_kernel(QPARAMS(a), QPARAMS(b), QPARAMS(c), float* oa, float* ob, float* oc, int ra, int rb, int rc, int la, int lb, int lc, \
                                                                                           int nm, const int8_t* xq, const float* xd, const int* xs, int K) { \
        MmMats mats; \
        mats.m[0] = MmMat{QMAKE(a), oa, ra, la}; \
        mats.m[1] = MmMat{QMAKE(b), ob, rb, lb}; \
        mats.m[2] = MmMat{QMAKE(c), oc, rc, lc}; \
        mats.n = nm; \
        mm_impl<F, T, R, EPI>(mats, xq, xd, xs, K); \
    }
#define MM_FMT(F, SUF, T, R) MM_KERNEL(F, SUF, T, R, 0, out) MM_KERNEL(F, SUF, T, R, 1, res)
#define MM_ALL(T, R) MM_FMT(0, q8, T, R) MM_FMT(1, q40, T, R) MM_FMT(2, q50, T, R) MM_FMT(3, q4k, T, R) MM_FMT(4, q5k, T, R) MM_FMT(5, q6k, T, R)
MM_ALL(4, 2) MM_ALL(8, 2) MM_ALL(12, 2) MM_ALL(16, 2)
#endif

// ==============================================================================================================================================================================================
// The rest of a layer for the multi-row prompt path (the other stages are decode_t.cu's / decode_h.cu's multi-token kernels: embed_q_multi_kernel, norm_quant_kernel_*, attention_q_kernel,
// attention_g_kernel, qkv35_prep_kernel, swiglu_quant_t_kernel, ...; the tokens are rows, the position of row t is st->pos + t).
// ==============================================================================================================================================================================================

// Attention of a dense model (Qwen3), after the q / k / v projections of N tokens (fp32 rows): q and k are RMS-normalised per head (weights qn_w / kn_w; null for models without QK norm), rotated on the
// first rd dimensions (pairs (i, i + rd / 2)), q goes to q_out (fp32 [N][n_head * hd]), k and v to the fp16 cache at position st->pos + token. One block per q head, then per k head, then per v head;
// grid (n_head + 2 n_kv, N).
extern "C" __global__ void __launch_bounds__(256) qkv_prep_multi_kernel(const float* __restrict__ q_raw, const float* __restrict__ k_raw, const float* __restrict__ v_raw, const float* __restrict__ qn_w,
                                                                       const float* __restrict__ kn_w, const float* __restrict__ cos_t, const float* __restrict__ sin_t, const StepState* __restrict__ st,
                                                                       float* __restrict__ q_out, half* __restrict__ k_cache, half* __restrict__ v_cache, int n_head, int n_kv, int hd, int rd, float eps) {
    __shared__ float red[32];
    const int b = blockIdx.x, tt = blockIdx.y;
    const uint32_t pos = st->pos + tt;
    const size_t kv_dim = (size_t)n_kv * hd;
    if (b >= n_head + n_kv) {
        const int head = b - n_head - n_kv;
        for (int d = threadIdx.x; d < hd; d += blockDim.x) v_cache[pos * kv_dim + head * hd + d] = __float2half(v_raw[tt * kv_dim + head * hd + d]);
        return;
    }
    const bool is_q = b < n_head;
    const int head = is_q ? b : b - n_head;
    const float* src = is_q ? q_raw + (size_t)tt * n_head * hd + (size_t)head * hd : k_raw + tt * kv_dim + (size_t)head * hd;
    const float* wn = is_q ? qn_w : kn_w;
    float r = 1.f;
    if (wn != nullptr) {
        float ss = 0.f;
        for (int d = threadIdx.x; d < hd; d += blockDim.x) ss += src[d] * src[d];
        ss = block_sum(ss, red);
        r = rsqrtf(ss / (float)hd + eps);
    }
    const int half_r = rd / 2;
    for (int d = threadIdx.x; d < hd; d += blockDim.x) {
        float o;
        if (d < rd) {
            const int i = d < half_r ? d : d - half_r;
            const float a = src[i] * r * (wn ? wn[i] : 1.f), bb = src[i + half_r] * r * (wn ? wn[i + half_r] : 1.f);
            const float c = cos_t[(size_t)pos * half_r + i], s = sin_t[(size_t)pos * half_r + i];
            o = d < half_r ? a * c - bb * s : bb * c + a * s;
        } else {
            o = src[d] * r * (wn ? wn[d] : 1.f);
        }
        if (is_q) q_out[(size_t)tt * n_head * hd + head * hd + d] = o;
        else k_cache[pos * kv_dim + head * hd + d] = __float2half(o);
    }
}

// The Gated DeltaNet step for ntok <= 16 tokens (gdn_step_impl, token after token, the state kept in registers; no state is saved on the side)
extern "C" __global__ void __launch_bounds__(1024) gdn_stepn_kernel(const float* __restrict__ mixed, const float* __restrict__ zv, const float* __restrict__ ab, const float* __restrict__ conv_w,
                                                                   const float* __restrict__ a_neg, const float* __restrict__ dt_bias, const float* __restrict__ norm_w, float* __restrict__ conv_state,
                                                                   float* __restrict__ S, int8_t* __restrict__ oq, float* __restrict__ od, int* __restrict__ os, int n_k, int n_v, int tiled, int conv_k,
                                                                   float eps, int ntok) {
    gdn_step_impl<16, 0>(mixed, zv, ab, conv_w, a_neg, dt_bias, norm_w, conv_state, S, nullptr, nullptr, oq, od, os, n_k, n_v, tiled, conv_k, eps, ntok);
}

// the same, saving the state after every token but the last (multi-row verification): after token t to S_side + t * s_stride, conv_side + t * c_stride (floats)
extern "C" __global__ void __launch_bounds__(1024) gdn_stepn_side_kernel(const float* __restrict__ mixed, const float* __restrict__ zv, const float* __restrict__ ab, const float* __restrict__ conv_w,
                                                                        const float* __restrict__ a_neg, const float* __restrict__ dt_bias, const float* __restrict__ norm_w,
                                                                        float* __restrict__ conv_state, float* __restrict__ S, float* __restrict__ conv_side, float* __restrict__ S_side,
                                                                        int8_t* __restrict__ oq, float* __restrict__ od, int* __restrict__ os, int n_k, int n_v, int tiled, int conv_k, float eps, int ntok,
                                                                        size_t s_stride, size_t c_stride) {
    gdn_step_impl<16, 2>(mixed, zv, ab, conv_w, a_neg, dt_bias, norm_w, conv_state, S, conv_side, S_side, oq, od, os, n_k, n_v, tiled, conv_k, eps, ntok, s_stride, c_stride);
}
