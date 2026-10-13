// ForgeInfer decode, quantized weights with PRE-QUANTIZED activations (decode path "v3"): Q8_0, Q4_0, Q5_0, Q4_K, Q5_K, Q6_K.
//
// The first quantized decode kernels (decode_q8.cu) fused RMSNorm and the activation quantization into every matvec block: each of the 128-256 blocks of a stage loaded the same activation vector,
// normalised and quantized it again (K = 3072 elements: ~800 instructions per warp, 70 % of a stage's instructions), and the grid needed two waves. Here the activation is quantized ONCE, by whoever
// produces it, and lives in global memory as three small arrays (llama.cpp's Q8_1 blocks, planar): int8 values, the fp16-rounded scale per 32 values, and the exact integer sum per 32 values
// (the offset / minimum terms of Q4_0, Q5_0, Q4_K, Q5_K need it):
//
//   norm_quant      x (fp32 residual stream) -> RMSNorm -> xq/xd/xs                  (one block)
//   qkv             Wq/Wk/Wv matvecs on (xq, xd, xs) -> RoPE -> q (fp32), k/v appended to the fp16 KV cache
//   attention_q     (decode.cu) softmax(q K^T) V per head, the output quantized in the epilogue
//   mv_res          x += W v on a pre-quantized v (Wo and the down projection)
//   gateup          gate / up matvecs -> silu(gate) * up -> quantized in the epilogue (a block of 16 warps produces exactly one 32-value group)
//   head            logits = W h on a pre-quantized h (persistent blocks)
//
// Every matvec kernel is a pure stream: the weight loads of a row and the activation loads of the lane are issued together, one DRAM round trip, no prologue, no __syncthreads.
// The weight format is a runtime argument (warp-uniform): the six format code paths sit in one kernel per row length, so a Q4_K_M model whose attn_v is Q6_K needs no special case.
//
// Weight layouts ("planar", the total bytes per block equal the native GGUF block; tools/ref_qdot.py is the executable specification, research note "Quantized Dot Products and GPU Layouts"):
//   Q8_0  p0 int8 [rows][K]                           p1 fp16 [rows][K/32]
//   Q4_0  p0 16 B/block (byte j = weight j | weight j+16 << 4)      p1 fp16 d
//   Q5_0  p0 16 B/block   p1 u32 5th-bit word per block, transposed: bit (8 i + p) = 5th bit of weight 4 p + i      p2 fp16 d
//   Q4_K  p0 128 B/super-block (chunk c = bytes 32 c..: low nibbles sub-block 2c, high nibbles 2c+1)       p1 16 B header (d, dmin fp16, 12 packed scale bytes)
//   Q5_K  p0 128 B/super-block   p1 8 x u32 per super-block, one transposed 5th-bit word per 32-weight task      p2 16 B header
//   Q6_K  p0 ql 128 B   p1 qh 64 B   p2 sc' 16 B int8 (sc'[4 (2h + u) + g] = sc[8h + u + 2g])   p3 fp16 d
// A lane-task is the unit one lane processes per iteration: 16 weights (Q8_0), one 32-weight block (Q4_0, Q5_0), 32 weights = 16 bytes of qs (Q4_K, Q5_K), 64 weights (Q6_K).

#define FMT_Q8_0 0
#define FMT_Q4_0 1
#define FMT_Q5_0 2
#define FMT_Q4_K 3
#define FMT_Q5_K 4
#define FMT_Q6_K 5
#define FMT_Q3_K 6   // host-resident FFN only (native blocks in p0): dequantized for the prefill GEMMs
#define FMT_Q2_K 7

struct QP {   // the planes of one weight tensor
    const uint8_t* p0;
    const uint8_t* p1;
    const uint8_t* p2;
    const uint8_t* p3;
};

struct ActP {   // a pre-quantized activation vector
    const int8_t* xq;
    const float* xd;
    const int* xs;
};

#define M_LO 0x0F0F0F0Fu
#define M_K1 0x3F3F3F3Fu
#define M_K3 0x03030303u

__device__ __forceinline__ float round_half(float v) { return __half2float(__float2half(v)); }

__device__ __forceinline__ uint4 ldq(const uint8_t* p, size_t idx16) { return __ldg(reinterpret_cast<const uint4*>(p) + idx16); }
__device__ __forceinline__ uint32_t ldw(const uint8_t* p, size_t idx32) { return __ldg(reinterpret_cast<const uint32_t*>(p) + idx32); }
__device__ __forceinline__ float ldh(const uint8_t* p, size_t idx16) { return __half2float(__ushort_as_half(__ldg(reinterpret_cast<const unsigned short*>(p) + idx16))); }

// 5th-bit operand p (0..7) of a transposed word: bit (8 i + p) -> bit 4 of byte i
__device__ __forceinline__ uint32_t tb5(uint32_t qh, int p) { return p <= 4 ? ((qh << (4 - p)) & 0x10101010u) : ((qh >> (p - 4)) & 0x10101010u); }

// the activation slice of a lane for 16 positions: the 16-byte vector number e16
// plain loads: the vector may be in shared memory. SW: the vector sits in shared memory with its 16-byte chunks swizzled by swz() (the long-row kernels): the lanes of a warp read chunks
// 2c + u / 4c + hf apart, which all land in the same 4 banks of a plain layout (2- to 4-way conflicts on every 128-bit load).
__device__ __forceinline__ int swz(int e16) { return e16 ^ ((e16 >> 2) & 6); }
template <bool SW = false>
__device__ __forceinline__ uint4 ldx(const ActP& a, int e16) { return reinterpret_cast<const uint4*>(a.xq)[SW ? swz(e16) : e16]; }

// ==============================================================================================================================================================================================
// Row registers per format. NI = lane-tasks per lane (ceil(tasks per row / 32)); a lane handles tasks lane + 32 i. Each load_* fills the registers (zeros beyond the row), each dot_* returns the lane's partial sum.
// ==============================================================================================================================================================================================

template <int NI> struct R8 { uint4 q[NI]; float d[NI]; };

template <int NI>
__device__ __forceinline__ void load_q8(R8<NI>& r, const QP& p, size_t row, int K, int lane) {
    const int nt = K / 16;
#pragma unroll
    for (int i = 0; i < NI; i++) {
        const int c = lane + 32 * i;
        if (c < nt) {
            r.q[i] = ldq(p.p0, row * nt + c);
            r.d[i] = ldh(p.p1, row * (K / 32) + (c >> 1));
        } else {
            r.q[i] = make_uint4(0u, 0u, 0u, 0u);
            r.d[i] = 0.f;
        }
    }
}

template <int NI, bool SW = false>
__device__ __forceinline__ float dot_q8(const R8<NI>& r, const ActP& a, int K, int lane) {
    const int nt = K / 16;
    float acc = 0.f;
#pragma unroll
    for (int i = 0; i < NI; i++) {
        const int c = lane + 32 * i;
        if (c < nt) {
            const uint4 x = ldx<SW>(a, c);
            int t = dp4(r.q[i].x, x.x, 0);
            t = dp4(r.q[i].y, x.y, t);
            t = dp4(r.q[i].z, x.z, t);
            t = dp4(r.q[i].w, x.w, t);
            acc += (float)t * (r.d[i] * a.xd[(c >> 1)]);
        }
    }
    return acc;
}

// ---- Q4_0 / Q5_0: one 32-weight block per task ---------------------------------------------------------------------------------------------------------------------------------------------------
template <int NI> struct R4 { uint4 q[NI]; uint32_t h[NI]; float d[NI]; };

template <int NI, bool Q5>
__device__ __forceinline__ void load_q40(R4<NI>& r, const QP& p, size_t row, int K, int lane) {
    const int nb = K / 32;
#pragma unroll
    for (int i = 0; i < NI; i++) {
        const int b = lane + 32 * i;
        if (b < nb) {
            r.q[i] = ldq(p.p0, row * nb + b);
            if (Q5) {
                r.h[i] = ldw(p.p1, row * nb + b);
                r.d[i] = ldh(p.p2, row * nb + b);
            } else {
                r.h[i] = 0u;
                r.d[i] = ldh(p.p1, row * nb + b);
            }
        } else {
            r.q[i] = make_uint4(0u, 0u, 0u, 0u);
            r.h[i] = 0u;
            r.d[i] = 0.f;
        }
    }
}

template <int NI, bool Q5, bool SW = false>
__device__ __forceinline__ float dot_q40(const R4<NI>& r, const ActP& a, int K, int lane) {
    const int nb = K / 32;
    float acc = 0.f;
#pragma unroll
    for (int i = 0; i < NI; i++) {
        const int b = lane + 32 * i;
        if (b < nb) {
            const uint4 xl = ldx<SW>(a, 2 * b), xh = ldx<SW>(a, 2 * b + 1);
            const uint32_t w0 = r.q[i].x, w1 = r.q[i].y, w2 = r.q[i].z, w3 = r.q[i].w;
            uint32_t l0 = w0 & M_LO, l1 = w1 & M_LO, l2 = w2 & M_LO, l3 = w3 & M_LO;
            uint32_t h0 = (w0 >> 4) & M_LO, h1 = (w1 >> 4) & M_LO, h2 = (w2 >> 4) & M_LO, h3 = (w3 >> 4) & M_LO;
            if (Q5) {
                const uint32_t qh = r.h[i];
                l0 |= tb5(qh, 0); l1 |= tb5(qh, 1); l2 |= tb5(qh, 2); l3 |= tb5(qh, 3);
                h0 |= tb5(qh, 4); h1 |= tb5(qh, 5); h2 |= tb5(qh, 6); h3 |= tb5(qh, 7);
            }
            int t = dp4(l0, xl.x, 0);
            t = dp4(l1, xl.y, t);
            t = dp4(l2, xl.z, t);
            t = dp4(l3, xl.w, t);
            t = dp4(h0, xh.x, t);
            t = dp4(h1, xh.y, t);
            t = dp4(h2, xh.z, t);
            t = dp4(h3, xh.w, t);
            t -= (Q5 ? 16 : 8) * a.xs[b];
            acc += (float)t * (r.d[i] * a.xd[b]);
        }
    }
    return acc;
}

// ---- Q4_K / Q5_K: task = 16 bytes of qs (positions 16 half.. of the sub-blocks 2c and 2c + 1) --------------------------------------------------------------------------------------------------------
template <int NI> struct RK { uint4 q[NI]; uint4 hdr[NI]; uint32_t h[NI]; };

template <int NI, bool Q5>
__device__ __forceinline__ void load_qk(RK<NI>& r, const QP& p, size_t row, int K, int lane) {
    const int nt = K / 32, nsb = K / 256;
#pragma unroll
    for (int i = 0; i < NI; i++) {
        const int t = lane + 32 * i;
        if (t < nt) {
            r.q[i] = ldq(p.p0, row * nt + t);
            if (Q5) {
                r.h[i] = ldw(p.p1, row * nt + t);
                r.hdr[i] = ldq(p.p2, row * nsb + (t >> 3));
            } else {
                r.h[i] = 0u;
                r.hdr[i] = ldq(p.p1, row * nsb + (t >> 3));
            }
        } else {
            r.q[i] = make_uint4(0u, 0u, 0u, 0u);
            r.h[i] = 0u;
            r.hdr[i] = make_uint4(0u, 0u, 0u, 0u);
        }
    }
}

template <int NI, bool Q5, bool SW = false>
__device__ __forceinline__ float dot_qk(const RK<NI>& r, const ActP& a, int K, int lane) {
    const int nt = K / 32;
    float acc = 0.f;
#pragma unroll
    for (int i = 0; i < NI; i++) {
        const int t = lane + 32 * i;
        if (t < nt) {
            const int sb = t >> 3, tt = t & 7, c = tt >> 1, hf = tt & 1;
            const int ab = sb * 8 + 2 * c;   // activation block of sub-block 2c; 2c + 1 follows
            const uint4 xl = ldx<SW>(a, ab * 2 + hf), xh = ldx<SW>(a, (ab + 1) * 2 + hf);
            const float dxl = a.xd[ab], dxh = a.xd[ab + 1];
            const uint4 hd = r.hdr[i];
            const float d = __half2float(__ushort_as_half((unsigned short)(hd.x & 0xFFFFu))), dmin = __half2float(__ushort_as_half((unsigned short)(hd.x >> 16)));
            const uint32_t u0 = hd.y, u1 = hd.z, u2 = hd.w;
            // six-bit scales and minimums of sub-blocks 2c, 2c + 1 (sub-blocks 0..3 in u0 / u1, 4..7 split between u2 and the top bits of u0 / u1)
            const uint32_t sw = c < 2 ? (u0 & M_K1) : ((u2 & M_LO) | (((u0 >> 6) & M_K3) << 4));
            const uint32_t mw = c < 2 ? (u1 & M_K1) : (((u2 >> 4) & M_LO) | (((u1 >> 6) & M_K3) << 4));
            const int sh = 8 * ((2 * c) & 3);
            const int scl = (sw >> sh) & 0xFF, sch = (sw >> (sh + 8)) & 0xFF;
            uint32_t l0 = r.q[i].x & M_LO, l1 = r.q[i].y & M_LO, l2 = r.q[i].z & M_LO, l3 = r.q[i].w & M_LO;
            uint32_t h0 = (r.q[i].x >> 4) & M_LO, h1 = (r.q[i].y >> 4) & M_LO, h2 = (r.q[i].z >> 4) & M_LO, h3 = (r.q[i].w >> 4) & M_LO;
            if (Q5) {
                const uint32_t qh = r.h[i];
                l0 |= tb5(qh, 0); l1 |= tb5(qh, 1); l2 |= tb5(qh, 2); l3 |= tb5(qh, 3);
                h0 |= tb5(qh, 4); h1 |= tb5(qh, 5); h2 |= tb5(qh, 6); h3 |= tb5(qh, 7);
            }
            int pl = dp4(l0, xl.x, 0);
            pl = dp4(l1, xl.y, pl);
            pl = dp4(l2, xl.z, pl);
            pl = dp4(l3, xl.w, pl);
            int ph = dp4(h0, xh.x, 0);
            ph = dp4(h1, xh.y, ph);
            ph = dp4(h2, xh.z, ph);
            ph = dp4(h3, xh.w, ph);
            float term = d * (dxl * (float)(pl * scl) + dxh * (float)(ph * sch));
            if (hf == 0) {   // the minimum terms use the exact integer sums of the whole 32-value activation blocks: only one of the two tasks of a sub-block adds them
                const int mnl = (mw >> sh) & 0xFF, mnh = (mw >> (sh + 8)) & 0xFF;
                term -= dmin * (dxl * (float)(mnl * a.xs[ab]) + dxh * (float)(mnh * a.xs[ab + 1]));
            }
            acc += term;
        }
    }
    return acc;
}

// ---- Q6_K: task = (super-block, half, u): 16 B ql_a, 16 B ql_b, 16 B qh, one word of scales -> 4 groups x 16 positions ---------------------------------------------------------------------------------
template <int NI> struct R6 { uint4 a[NI]; uint4 b[NI]; uint4 h[NI]; uint32_t sc[NI]; float d[NI]; };

template <int NI>
__device__ __forceinline__ void load_q6(R6<NI>& r, const QP& p, size_t row, int K, int lane) {
    const int nt = K / 64, nsb = K / 256;
#pragma unroll
    for (int i = 0; i < NI; i++) {
        const int t = lane + 32 * i;
        if (t < nt) {
            const int sb = t >> 2, h = (t >> 1) & 1, u = t & 1;
            const size_t blk = row * nsb + sb;
            r.a[i] = ldq(p.p0, blk * 8 + 4 * h + u);
            r.b[i] = ldq(p.p0, blk * 8 + 4 * h + 2 + u);
            r.h[i] = ldq(p.p1, blk * 4 + 2 * h + u);
            r.sc[i] = ldw(p.p2, blk * 4 + 2 * h + u);
            r.d[i] = ldh(p.p3, blk);
        } else {
            r.a[i] = r.b[i] = r.h[i] = make_uint4(0u, 0u, 0u, 0u);
            r.sc[i] = 0u;
            r.d[i] = 0.f;
        }
    }
}

__device__ __forceinline__ int dot_q6_group(const uint4& A, const uint4& B, const uint4& H, const uint4& x, int g) {
    // 4 * (code - 32) as signed bytes: (code << 2) ^ 0x80, no byte-subtraction intrinsic needed; the factor 4 leaves with the epilogue
    uint32_t f0, f1, f2, f3;
    if (g == 0) {
        f0 = (((A.x << 2) & 0x3C3C3C3Cu) | ((H.x << 6) & 0xC0C0C0C0u)) ^ 0x80808080u;
        f1 = (((A.y << 2) & 0x3C3C3C3Cu) | ((H.y << 6) & 0xC0C0C0C0u)) ^ 0x80808080u;
        f2 = (((A.z << 2) & 0x3C3C3C3Cu) | ((H.z << 6) & 0xC0C0C0C0u)) ^ 0x80808080u;
        f3 = (((A.w << 2) & 0x3C3C3C3Cu) | ((H.w << 6) & 0xC0C0C0C0u)) ^ 0x80808080u;
    } else if (g == 1) {
        f0 = (((B.x << 2) & 0x3C3C3C3Cu) | ((H.x << 4) & 0xC0C0C0C0u)) ^ 0x80808080u;
        f1 = (((B.y << 2) & 0x3C3C3C3Cu) | ((H.y << 4) & 0xC0C0C0C0u)) ^ 0x80808080u;
        f2 = (((B.z << 2) & 0x3C3C3C3Cu) | ((H.z << 4) & 0xC0C0C0C0u)) ^ 0x80808080u;
        f3 = (((B.w << 2) & 0x3C3C3C3Cu) | ((H.w << 4) & 0xC0C0C0C0u)) ^ 0x80808080u;
    } else if (g == 2) {
        f0 = (((A.x >> 2) & 0x3C3C3C3Cu) | ((H.x << 2) & 0xC0C0C0C0u)) ^ 0x80808080u;
        f1 = (((A.y >> 2) & 0x3C3C3C3Cu) | ((H.y << 2) & 0xC0C0C0C0u)) ^ 0x80808080u;
        f2 = (((A.z >> 2) & 0x3C3C3C3Cu) | ((H.z << 2) & 0xC0C0C0C0u)) ^ 0x80808080u;
        f3 = (((A.w >> 2) & 0x3C3C3C3Cu) | ((H.w << 2) & 0xC0C0C0C0u)) ^ 0x80808080u;
    } else {
        f0 = (((B.x >> 2) & 0x3C3C3C3Cu) | (H.x & 0xC0C0C0C0u)) ^ 0x80808080u;
        f1 = (((B.y >> 2) & 0x3C3C3C3Cu) | (H.y & 0xC0C0C0C0u)) ^ 0x80808080u;
        f2 = (((B.z >> 2) & 0x3C3C3C3Cu) | (H.z & 0xC0C0C0C0u)) ^ 0x80808080u;
        f3 = (((B.w >> 2) & 0x3C3C3C3Cu) | (H.w & 0xC0C0C0C0u)) ^ 0x80808080u;
    }
    int t = dp4(f0, x.x, 0);
    t = dp4(f1, x.y, t);
    t = dp4(f2, x.z, t);
    return dp4(f3, x.w, t);
}

template <int NI, bool SW = false>
__device__ __forceinline__ float dot_q6(const R6<NI>& r, const ActP& a, int K, int lane) {
    const int nt = K / 64;
    float acc = 0.f;
#pragma unroll
    for (int i = 0; i < NI; i++) {
        const int t = lane + 32 * i;
        if (t < nt) {
            const int sb = t >> 2, h = (t >> 1) & 1, u = t & 1;
            float sum = 0.f;
#pragma unroll
            for (int g = 0; g < 4; g++) {
                const uint4 x = ldx<SW>(a, sb * 16 + 8 * h + 2 * g + u);
                const int dot = dot_q6_group(r.a[i], r.b[i], r.h[i], x, g);
                const int scg = (int)(signed char)((r.sc[i] >> (8 * g)) & 0xFF);
                sum += a.xd[sb * 8 + 4 * h + g] * ((float)dot * (float)scg);
            }
            acc += r.d[i] * 0.25f * sum;
        }
    }
    return acc;
}

// ==============================================================================================================================================================================================
// One row dot product with a runtime format: load (hoisted by the caller before anything else is waited for) and dot are two calls so that several rows can be in flight.
// ==============================================================================================================================================================================================
template <int NI, int F> struct RowRegsF;
template <int NI> struct RowRegsF<NI, FMT_Q8_0> { R8<NI> v; };
template <int NI> struct RowRegsF<NI, FMT_Q4_0> { R4<(NI + 1) / 2> v; };
template <int NI> struct RowRegsF<NI, FMT_Q5_0> { R4<(NI + 1) / 2> v; };
template <int NI> struct RowRegsF<NI, FMT_Q4_K> { RK<(NI + 1) / 2> v; };
template <int NI> struct RowRegsF<NI, FMT_Q5_K> { RK<(NI + 1) / 2> v; };
template <int NI> struct RowRegsF<NI, FMT_Q6_K> { R6<(NI + 3) / 4> v; };
template <int NI> struct RowRegsF<NI, -1> {   // every format: the registers of the formats overlap (only one is live for a given warp-uniform format)
    R8<NI> r8;
    R4<(NI + 1) / 2> r4;
    RK<(NI + 1) / 2> rk;
    R6<(NI + 3) / 4> r6;
};

// F >= 0: the format is a compile-time constant (one code path, the registers of that format only); F = -1: `fmt` selects at run time
template <int NI, int F>
__device__ __forceinline__ void load_row(RowRegsF<NI, F>& r, int fmt, const QP& p, size_t row, int K, int lane) {
    if constexpr (F == FMT_Q8_0) {
        load_q8<NI>(r.v, p, row, K, lane);
    } else if constexpr (F == FMT_Q4_0) {
        load_q40<(NI + 1) / 2, false>(r.v, p, row, K, lane);
    } else if constexpr (F == FMT_Q5_0) {
        load_q40<(NI + 1) / 2, true>(r.v, p, row, K, lane);
    } else if constexpr (F == FMT_Q4_K) {
        load_qk<(NI + 1) / 2, false>(r.v, p, row, K, lane);
    } else if constexpr (F == FMT_Q5_K) {
        load_qk<(NI + 1) / 2, true>(r.v, p, row, K, lane);
    } else if constexpr (F == FMT_Q6_K) {
        load_q6<(NI + 3) / 4>(r.v, p, row, K, lane);
    } else {
        switch (fmt) {
            case FMT_Q8_0: load_q8<NI>(r.r8, p, row, K, lane); break;
            case FMT_Q4_0: load_q40<(NI + 1) / 2, false>(r.r4, p, row, K, lane); break;
            case FMT_Q5_0: load_q40<(NI + 1) / 2, true>(r.r4, p, row, K, lane); break;
            case FMT_Q4_K: load_qk<(NI + 1) / 2, false>(r.rk, p, row, K, lane); break;
            case FMT_Q5_K: load_qk<(NI + 1) / 2, true>(r.rk, p, row, K, lane); break;
            default: load_q6<(NI + 3) / 4>(r.r6, p, row, K, lane); break;
        }
    }
}

template <int NI, int F>
__device__ __forceinline__ float dot_row(const RowRegsF<NI, F>& r, int fmt, const ActP& a, int K, int lane) {
    float v;
    if constexpr (F == FMT_Q8_0) {
        v = dot_q8<NI>(r.v, a, K, lane);
    } else if constexpr (F == FMT_Q4_0) {
        v = dot_q40<(NI + 1) / 2, false>(r.v, a, K, lane);
    } else if constexpr (F == FMT_Q5_0) {
        v = dot_q40<(NI + 1) / 2, true>(r.v, a, K, lane);
    } else if constexpr (F == FMT_Q4_K) {
        v = dot_qk<(NI + 1) / 2, false>(r.v, a, K, lane);
    } else if constexpr (F == FMT_Q5_K) {
        v = dot_qk<(NI + 1) / 2, true>(r.v, a, K, lane);
    } else if constexpr (F == FMT_Q6_K) {
        v = dot_q6<(NI + 3) / 4>(r.v, a, K, lane);
    } else {
        switch (fmt) {
            case FMT_Q8_0: v = dot_q8<NI>(r.r8, a, K, lane); break;
            case FMT_Q4_0: v = dot_q40<(NI + 1) / 2, false>(r.r4, a, K, lane); break;
            case FMT_Q5_0: v = dot_q40<(NI + 1) / 2, true>(r.r4, a, K, lane); break;
            case FMT_Q4_K: v = dot_qk<(NI + 1) / 2, false>(r.rk, a, K, lane); break;
            case FMT_Q5_K: v = dot_qk<(NI + 1) / 2, true>(r.rk, a, K, lane); break;
            default: v = dot_q6<(NI + 3) / 4>(r.r6, a, K, lane); break;
        }
    }
    return warp_sum(v);
}

// ==============================================================================================================================================================================================
// Activation registers: the lane's slice of the activation for its tasks, loaded ONCE per warp and reused for every row the warp handles (the dot_* above reload it on every call). The pipelined stage
// kernels below use these for the compile-time formats.
// ==============================================================================================================================================================================================
template <int NI> struct AQ8 { uint4 x[NI]; float d[NI]; };

template <int NI, bool SW = false>
__device__ __forceinline__ void load_act_q8(AQ8<NI>& r, const ActP& a, int K, int lane) {
    const int nt = K / 16;
#pragma unroll
    for (int i = 0; i < NI; i++) {
        const int c = lane + 32 * i;
        if (c < nt) {
            r.x[i] = ldx<SW>(a, c);
            r.d[i] = a.xd[(c >> 1)];
        } else {
            r.x[i] = make_uint4(0u, 0u, 0u, 0u);
            r.d[i] = 0.f;
        }
    }
}

template <int NI>
__device__ __forceinline__ float dot_q8r(const R8<NI>& r, const AQ8<NI>& ar) {
    float acc = 0.f;
#pragma unroll
    for (int i = 0; i < NI; i++) {
        int t = dp4(r.q[i].x, ar.x[i].x, 0);
        t = dp4(r.q[i].y, ar.x[i].y, t);
        t = dp4(r.q[i].z, ar.x[i].z, t);
        t = dp4(r.q[i].w, ar.x[i].w, t);
        acc += (float)t * (r.d[i] * ar.d[i]);   // padded tasks: weight scale 0
    }
    return acc;
}

template <int NI> struct AQ4 { uint4 xl[NI]; uint4 xh[NI]; float d[NI]; int s[NI]; };

template <int NI, bool SW = false>
__device__ __forceinline__ void load_act_q40(AQ4<NI>& r, const ActP& a, int K, int lane) {
    const int nb = K / 32;
#pragma unroll
    for (int i = 0; i < NI; i++) {
        const int b = lane + 32 * i;
        if (b < nb) {
            r.xl[i] = ldx<SW>(a, 2 * b);
            r.xh[i] = ldx<SW>(a, 2 * b + 1);
            r.d[i] = a.xd[b];
            r.s[i] = a.xs[b];
        } else {
            r.xl[i] = r.xh[i] = make_uint4(0u, 0u, 0u, 0u);
            r.d[i] = 0.f;
            r.s[i] = 0;
        }
    }
}

template <int NI, bool Q5>
__device__ __forceinline__ float dot_q40r(const R4<NI>& r, const AQ4<NI>& ar) {
    float acc = 0.f;
#pragma unroll
    for (int i = 0; i < NI; i++) {
        const uint32_t w0 = r.q[i].x, w1 = r.q[i].y, w2 = r.q[i].z, w3 = r.q[i].w;
        uint32_t l0 = w0 & M_LO, l1 = w1 & M_LO, l2 = w2 & M_LO, l3 = w3 & M_LO;
        uint32_t h0 = (w0 >> 4) & M_LO, h1 = (w1 >> 4) & M_LO, h2 = (w2 >> 4) & M_LO, h3 = (w3 >> 4) & M_LO;
        if (Q5) {
            const uint32_t qh = r.h[i];
            l0 |= tb5(qh, 0); l1 |= tb5(qh, 1); l2 |= tb5(qh, 2); l3 |= tb5(qh, 3);
            h0 |= tb5(qh, 4); h1 |= tb5(qh, 5); h2 |= tb5(qh, 6); h3 |= tb5(qh, 7);
        }
        int t = dp4(l0, ar.xl[i].x, 0);
        t = dp4(l1, ar.xl[i].y, t);
        t = dp4(l2, ar.xl[i].z, t);
        t = dp4(l3, ar.xl[i].w, t);
        t = dp4(h0, ar.xh[i].x, t);
        t = dp4(h1, ar.xh[i].y, t);
        t = dp4(h2, ar.xh[i].z, t);
        t = dp4(h3, ar.xh[i].w, t);
        t -= (Q5 ? 16 : 8) * ar.s[i];
        acc += (float)t * (r.d[i] * ar.d[i]);
    }
    return acc;
}

template <int NI> struct AQK { uint4 xl[NI]; uint4 xh[NI]; float dl[NI]; float dh[NI]; int sl[NI]; int sh[NI]; };

template <int NI, bool SW = false>
__device__ __forceinline__ void load_act_qk(AQK<NI>& r, const ActP& a, int K, int lane) {
    const int nt = K / 32;
#pragma unroll
    for (int i = 0; i < NI; i++) {
        const int t = lane + 32 * i;
        if (t < nt) {
            const int sb = t >> 3, tt = t & 7, c = tt >> 1, hf = tt & 1;
            const int ab = sb * 8 + 2 * c;
            r.xl[i] = ldx<SW>(a, ab * 2 + hf);
            r.xh[i] = ldx<SW>(a, (ab + 1) * 2 + hf);
            r.dl[i] = a.xd[ab];
            r.dh[i] = a.xd[ab + 1];
            r.sl[i] = a.xs[ab];
            r.sh[i] = a.xs[ab + 1];
        } else {
            r.xl[i] = r.xh[i] = make_uint4(0u, 0u, 0u, 0u);
            r.dl[i] = r.dh[i] = 0.f;
            r.sl[i] = r.sh[i] = 0;
        }
    }
}

template <int NI, bool Q5>
__device__ __forceinline__ float dot_qkr(const RK<NI>& r, const AQK<NI>& ar, int lane) {
    float acc = 0.f;
#pragma unroll
    for (int i = 0; i < NI; i++) {
        const int t = lane + 32 * i;
        const int tt = t & 7, c = tt >> 1, hf = tt & 1;
        const uint4 hd = r.hdr[i];   // zero for padded tasks: d = dmin = 0
        const float d = __half2float(__ushort_as_half((unsigned short)(hd.x & 0xFFFFu))), dmin = __half2float(__ushort_as_half((unsigned short)(hd.x >> 16)));
        const uint32_t u0 = hd.y, u1 = hd.z, u2 = hd.w;
        const uint32_t sw = c < 2 ? (u0 & M_K1) : ((u2 & M_LO) | (((u0 >> 6) & M_K3) << 4));
        const uint32_t mw = c < 2 ? (u1 & M_K1) : (((u2 >> 4) & M_LO) | (((u1 >> 6) & M_K3) << 4));
        const int sh = 8 * ((2 * c) & 3);
        const int scl = (sw >> sh) & 0xFF, sch = (sw >> (sh + 8)) & 0xFF;
        uint32_t l0 = r.q[i].x & M_LO, l1 = r.q[i].y & M_LO, l2 = r.q[i].z & M_LO, l3 = r.q[i].w & M_LO;
        uint32_t h0 = (r.q[i].x >> 4) & M_LO, h1 = (r.q[i].y >> 4) & M_LO, h2 = (r.q[i].z >> 4) & M_LO, h3 = (r.q[i].w >> 4) & M_LO;
        if (Q5) {
            const uint32_t qh = r.h[i];
            l0 |= tb5(qh, 0); l1 |= tb5(qh, 1); l2 |= tb5(qh, 2); l3 |= tb5(qh, 3);
            h0 |= tb5(qh, 4); h1 |= tb5(qh, 5); h2 |= tb5(qh, 6); h3 |= tb5(qh, 7);
        }
        int pl = dp4(l0, ar.xl[i].x, 0);
        pl = dp4(l1, ar.xl[i].y, pl);
        pl = dp4(l2, ar.xl[i].z, pl);
        pl = dp4(l3, ar.xl[i].w, pl);
        int ph = dp4(h0, ar.xh[i].x, 0);
        ph = dp4(h1, ar.xh[i].y, ph);
        ph = dp4(h2, ar.xh[i].z, ph);
        ph = dp4(h3, ar.xh[i].w, ph);
        float term = d * (ar.dl[i] * (float)(pl * scl) + ar.dh[i] * (float)(ph * sch));
        if (hf == 0) {
            const int mnl = (mw >> sh) & 0xFF, mnh = (mw >> (sh + 8)) & 0xFF;
            term -= dmin * (ar.dl[i] * (float)(mnl * ar.sl[i]) + ar.dh[i] * (float)(mnh * ar.sh[i]));
        }
        acc += term;
    }
    return acc;
}

template <int NI> struct AQ6 { uint4 x[NI][4]; float d[NI][4]; };

template <int NI, bool SW = false>
__device__ __forceinline__ void load_act_q6(AQ6<NI>& r, const ActP& a, int K, int lane) {
    const int nt = K / 64;
#pragma unroll
    for (int i = 0; i < NI; i++) {
        const int t = lane + 32 * i;
#pragma unroll
        for (int g = 0; g < 4; g++) {
            if (t < nt) {
                const int sb = t >> 2, h = (t >> 1) & 1, u = t & 1;
                r.x[i][g] = ldx<SW>(a, sb * 16 + 8 * h + 2 * g + u);
                r.d[i][g] = a.xd[sb * 8 + 4 * h + g];
            } else {
                r.x[i][g] = make_uint4(0u, 0u, 0u, 0u);
                r.d[i][g] = 0.f;
            }
        }
    }
}

template <int NI>
__device__ __forceinline__ float dot_q6r(const R6<NI>& r, const AQ6<NI>& ar) {
    float acc = 0.f;
#pragma unroll
    for (int i = 0; i < NI; i++) {
        float sum = 0.f;
#pragma unroll
        for (int g = 0; g < 4; g++) {
            const int dot = dot_q6_group(r.a[i], r.b[i], r.h[i], ar.x[i][g], g);
            const int scg = (int)(signed char)((r.sc[i] >> (8 * g)) & 0xFF);
            sum += ar.d[i][g] * ((float)dot * (float)scg);
        }
        acc += r.d[i] * 0.25f * sum;
    }
    return acc;
}

template <int NI, int F> struct ActRegsT;
template <int NI> struct ActRegsT<NI, FMT_Q8_0> { AQ8<NI> v; };
template <int NI> struct ActRegsT<NI, FMT_Q4_0> { AQ4<(NI + 1) / 2> v; };
template <int NI> struct ActRegsT<NI, FMT_Q5_0> { AQ4<(NI + 1) / 2> v; };
template <int NI> struct ActRegsT<NI, FMT_Q4_K> { AQK<(NI + 1) / 2> v; };
template <int NI> struct ActRegsT<NI, FMT_Q5_K> { AQK<(NI + 1) / 2> v; };
template <int NI> struct ActRegsT<NI, FMT_Q6_K> { AQ6<(NI + 3) / 4> v; };

template <int NI, int F, bool SW = false>
__device__ __forceinline__ void load_act(ActRegsT<NI, F>& ar, const ActP& a, int K, int lane) {
    if constexpr (F == FMT_Q8_0) load_act_q8<NI, SW>(ar.v, a, K, lane);
    else if constexpr (F == FMT_Q4_0 || F == FMT_Q5_0) load_act_q40<(NI + 1) / 2, SW>(ar.v, a, K, lane);
    else if constexpr (F == FMT_Q4_K || F == FMT_Q5_K) load_act_qk<(NI + 1) / 2, SW>(ar.v, a, K, lane);
    else load_act_q6<(NI + 3) / 4, SW>(ar.v, a, K, lane);
}

// dot of a row held in registers with the activation held in registers, reduced over the warp (every lane gets the sum)
template <int NI, int F>
__device__ __forceinline__ float dot_rowr(const RowRegsF<NI, F>& r, const ActRegsT<NI, F>& ar, int lane) {
    float v;
    if constexpr (F == FMT_Q8_0) v = dot_q8r<NI>(r.v, ar.v);
    else if constexpr (F == FMT_Q4_0) v = dot_q40r<(NI + 1) / 2, false>(r.v, ar.v);
    else if constexpr (F == FMT_Q5_0) v = dot_q40r<(NI + 1) / 2, true>(r.v, ar.v);
    else if constexpr (F == FMT_Q4_K) v = dot_qkr<(NI + 1) / 2, false>(r.v, ar.v, lane);
    else if constexpr (F == FMT_Q5_K) v = dot_qkr<(NI + 1) / 2, true>(r.v, ar.v, lane);
    else v = dot_q6r<(NI + 3) / 4>(r.v, ar.v);
    return warp_sum(v);
}

// registers per row for the pipelining decision (a pipelined gate/up keeps four rows plus the activation live)
template <int NI, int F>
struct PipeOk {
    static constexpr int NIF = F == FMT_Q8_0 ? NI : (F == FMT_Q6_K ? (NI + 3) / 4 : (NI + 1) / 2);
    static constexpr int ROW = F == FMT_Q8_0 ? 5 * NI : (F <= FMT_Q5_0 ? 6 * NIF : (F <= FMT_Q5_K ? 9 * NIF : 14 * NIF));
    static constexpr int ACT = F == FMT_Q8_0 ? 5 * NI : (F <= FMT_Q5_0 ? 10 * NIF : (F <= FMT_Q5_K ? 12 * NIF : 20 * NIF));
    static constexpr bool value = F >= 0 && 4 * ROW + ACT + 25 <= 128;   // four rows and the activation live in at most 128 registers (one block of 512 threads per SM)
};

// ==============================================================================================================================================================================================
// Quantizers. Scale rule of llama.cpp's Q8_1: d = amax / 127 (fp32), q = round(x / d) with the UNROUNDED d, the stored scale is rounded to fp16; the integer sum is exact.
// ==============================================================================================================================================================================================
__device__ __forceinline__ void quant_group(float v, int lane, int8_t* xq, float* xd, int* xs, int i, int g) {
    const float amax = warp_max(fabsf(v));
    const float d = amax * (1.f / 127.f);
    const float id = d != 0.f ? 1.f / d : 0.f;
    const int q = (int)roundf(v * id);
    int sum = q;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
    xq[i] = (int8_t)q;
    if (lane == 0) {
        xd[g] = round_half(d);
        xs[g] = sum;
    }
}

// One block of 512 threads; element i = threadIdx.x + k * blockDim.x, so register k of a warp holds exactly one 32-value group. E must be a multiple of 32 and at most NMAX * 512.
template <int NMAX, bool NORM>
__device__ __forceinline__ void norm_quant_impl(const float* __restrict__ x, const float* __restrict__ w, int8_t* __restrict__ xq, float* __restrict__ xd, int* __restrict__ xs, int E, float eps) {
    __shared__ float red[32];
    // blockIdx.x selects the vector: several vectors of one multi-token step are normalised by one launch with one block each
    x += (size_t)blockIdx.x * E;
    xq += (size_t)blockIdx.x * E;
    xd += (size_t)blockIdx.x * (E / 32);
    xs += (size_t)blockIdx.x * (E / 32);
    float hv[NMAX], wv[NMAX];
    float ss = 0.f;
#pragma unroll
    for (int k = 0; k < NMAX; k++) {   // the norm weights are loaded together with x: they are needed right after the reduction
        const int i = threadIdx.x + k * blockDim.x;
        hv[k] = i < E ? x[i] : 0.f;
        wv[k] = (NORM && i < E) ? w[i] : 1.f;
        ss += hv[k] * hv[k];
    }
    float rs = 1.f;
    if (NORM) {
        ss = block_sum(ss, red);
        rs = rsqrtf(ss / (float)E + eps);
    }
    const int lane = threadIdx.x & 31;
#pragma unroll
    for (int k = 0; k < NMAX; k++) {
        const int i = threadIdx.x + k * blockDim.x;
        if (i < E) quant_group(hv[k] * rs * wv[k], lane, xq, xd, xs, i, i >> 5);
    }
}

#define INST_NQ(NM)                                                                                                                                                                          \
    extern "C" __global__ void __launch_bounds__(512) norm_quant_kernel_##NM(const float* x, const float* w, int8_t* xq, float* xd, int* xs, int E, float eps) {                              \
        norm_quant_impl<NM, true>(x, w, xq, xd, xs, E, eps);                                                                                                                                 \
    }                                                                                                                                                                                        \
    extern "C" __global__ void __launch_bounds__(512) quant_kernel_##NM(const float* x, int8_t* xq, float* xd, int* xs, int E) { norm_quant_impl<NM, false>(x, nullptr, xq, xd, xs, E, 0.f); }
INST_NQ(1) INST_NQ(2) INST_NQ(4) INST_NQ(8) INST_NQ(16) INST_NQ(32)

// ==============================================================================================================================================================================================
// Stages. The kernels are persistent: a fixed grid (about two blocks per SM) and every warp strides over the rows, with the NEXT row's weight loads issued before the current row is consumed (two
// register sets). Kernels of one-block-per-row-group design had wave boundaries (the second wave's loads only started when the first wave's blocks finished: ~30 % idle DRAM time); here the memory
// system always has a row in flight per warp until the last rows. For compile-time formats the lane's activation slice is held in registers and reused by every row.
// ==============================================================================================================================================================================================

#define QPARAMS(n) const uint8_t* n##0, const uint8_t* n##1, const uint8_t* n##2, const uint8_t* n##3
#define QMAKE(n) QP{n##0, n##1, n##2, n##3}

// ---- Q/K/V: one unit = a rotation pair of rows (i, i + hd / 2) of a head; units of q, then k, then v. rotate = 0 leaves q and k raw (Qwen3 normalises them first: qk_norm_rope_kernel). -----------------
struct QkvUnit {
    int which, head, i;
    size_t row0, row1;
};

__device__ __forceinline__ QkvUnit qkv_unit(int u, int nq, int nk, int half_d, int hd) {
    QkvUnit r;
    int idx;
    if (u < nq) { r.which = 0; idx = u; }
    else if (u < nq + nk) { r.which = 1; idx = u - nq; }
    else { r.which = 2; idx = u - nq - nk; }
    r.head = idx / half_d;
    r.i = idx % half_d;
    r.row0 = (size_t)r.head * hd + r.i;
    r.row1 = r.row0 + half_d;
    return r;
}

__device__ __forceinline__ void qkv_finish(const QkvUnit& un, float av, float bv, float* __restrict__ q_out, float* __restrict__ k_raw, half* __restrict__ k_cache, half* __restrict__ v_cache,
                                           const float* __restrict__ cos_t, const float* __restrict__ sin_t, uint32_t pos, int n_kv, int hd, int rotate) {
    const int half_d = hd / 2, kv_dim = n_kv * hd, head = un.head, i = un.i;
    if (un.which == 2) {
        v_cache[(size_t)pos * kv_dim + head * hd + i] = __float2half(av);
        v_cache[(size_t)pos * kv_dim + head * hd + i + half_d] = __float2half(bv);
        return;
    }
    float o0 = av, o1 = bv;
    if (rotate) {
        const float c = cos_t[(size_t)pos * half_d + i], s = sin_t[(size_t)pos * half_d + i];
        o0 = av * c - bv * s;
        o1 = bv * c + av * s;
    }
    if (un.which == 0) {
        q_out[head * hd + i] = o0;
        q_out[head * hd + i + half_d] = o1;
    } else if (rotate) {
        k_cache[(size_t)pos * kv_dim + head * hd + i] = __float2half(o0);
        k_cache[(size_t)pos * kv_dim + head * hd + i + half_d] = __float2half(o1);
    } else {
        k_raw[head * hd + i] = o0;
        k_raw[head * hd + i + half_d] = o1;
    }
}

template <int NI, int F>
__device__ __forceinline__ void qkv_impl(QP wq, QP wk, QP wv, int fq, int fk, int fv, ActP a, float* __restrict__ q_out, float* __restrict__ k_raw, half* __restrict__ k_cache, half* __restrict__ v_cache,
                                         const float* __restrict__ cos_t, const float* __restrict__ sin_t, const StepState* __restrict__ st, int E, int n_head, int n_kv, int hd, int rotate, int u_lo, int u_hi) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
    const int half_d = hd / 2;
    const int nq = n_head * half_d, nk = n_kv * half_d;   // units: nq of q, nk of k, then nk of v; this launch handles [u_lo, u_hi)
    const int total = u_hi;
    const int stride = gridDim.x * nw;
    int u = u_lo + blockIdx.x * nw + warp;
    if (u >= total) return;
    const uint32_t pos = st->pos;
    if constexpr (F >= 0) {
        // every tensor of the launch has format F
        auto pick = [&](int which) { return which == 0 ? wq : (which == 1 ? wk : wv); };
        ActRegsT<NI, F> ar;
        load_act<NI, F>(ar, a, E, lane);
        RowRegsF<NI, F> a0, a1, b0, b1;
        QkvUnit cu = qkv_unit(u, nq, nk, half_d, hd);
        load_row<NI, F>(a0, F, pick(cu.which), cu.row0, E, lane);
        load_row<NI, F>(a1, F, pick(cu.which), cu.row1, E, lane);
        while (true) {
            const int u1 = u + stride;
            QkvUnit nu = cu;
            if (u1 < total) {
                nu = qkv_unit(u1, nq, nk, half_d, hd);
                load_row<NI, F>(b0, F, pick(nu.which), nu.row0, E, lane);
                load_row<NI, F>(b1, F, pick(nu.which), nu.row1, E, lane);
            }
            const float av = dot_rowr<NI, F>(a0, ar, lane), bv = dot_rowr<NI, F>(a1, ar, lane);
            if (lane == 0) qkv_finish(cu, av, bv, q_out, k_raw, k_cache, v_cache, cos_t, sin_t, pos, n_kv, hd, rotate);
            if (u1 >= total) break;
            const int u2 = u1 + stride;
            QkvUnit nn = nu;
            if (u2 < total) {
                nn = qkv_unit(u2, nq, nk, half_d, hd);
                load_row<NI, F>(a0, F, pick(nn.which), nn.row0, E, lane);
                load_row<NI, F>(a1, F, pick(nn.which), nn.row1, E, lane);
            }
            const float cv = dot_rowr<NI, F>(b0, ar, lane), dv = dot_rowr<NI, F>(b1, ar, lane);
            if (lane == 0) qkv_finish(nu, cv, dv, q_out, k_raw, k_cache, v_cache, cos_t, sin_t, pos, n_kv, hd, rotate);
            if (u2 >= total) break;
            u = u2;
            cu = nn;
        }
    } else {
        // tensors of different formats: runtime switch, no pipelining
        for (; u < total; u += stride) {
            const QkvUnit un = qkv_unit(u, nq, nk, half_d, hd);
            const QP w = un.which == 0 ? wq : (un.which == 1 ? wk : wv);
            const int fmt = un.which == 0 ? fq : (un.which == 1 ? fk : fv);
            RowRegsF<NI, -1> r0, r1;
            load_row<NI, -1>(r0, fmt, w, un.row0, E, lane);
            load_row<NI, -1>(r1, fmt, w, un.row1, E, lane);
            const float av = dot_row<NI, -1>(r0, fmt, a, E, lane), bv = dot_row<NI, -1>(r1, fmt, a, E, lane);
            if (lane == 0) qkv_finish(un, av, bv, q_out, k_raw, k_cache, v_cache, cos_t, sin_t, pos, n_kv, hd, rotate);
        }
    }
}

#define INST_QKV_F(NI, F, SUF)                                                                                                                                                               \
    extern "C" __global__ void qkvq_##SUF##_kernel_##NI(QPARAMS(wq), QPARAMS(wk), QPARAMS(wv), int fq, int fk, int fv, const int8_t* xq, const float* xd, const int* xs, float* q_out,              \
                                                       float* k_raw, half* k_cache, half* v_cache, const float* cos_t, const float* sin_t, const StepState* st, int E, int n_head, int n_kv,  \
                                                       int hd, int rotate, int u_lo, int u_hi) {                                                                                                                \
        qkv_impl<NI, F>(QMAKE(wq), QMAKE(wk), QMAKE(wv), fq, fk, fv, ActP{xq, xd, xs}, q_out, k_raw, k_cache, v_cache, cos_t, sin_t, st, E, n_head, n_kv, hd, rotate, u_lo, u_hi);                       \
    }
#define INST_QKV(NI) INST_QKV_F(NI, 0, q8) INST_QKV_F(NI, 1, q40) INST_QKV_F(NI, 2, q50) INST_QKV_F(NI, 3, q4k) INST_QKV_F(NI, 4, q5k) INST_QKV_F(NI, 5, q6k) INST_QKV_F(NI, -1, x)
INST_QKV(1) INST_QKV(2) INST_QKV(3) INST_QKV(4) INST_QKV(6) INST_QKV(8) INST_QKV(12) INST_QKV(16) INST_QKV(24)

// ---- x += W v (Wo and the down projection) ---------------------------------------------------------------------------------------------------------------------------------------------------------
template <int NI, int F>
__device__ __forceinline__ void mv_res_impl(QP w, int fmt, ActP a, float* __restrict__ x, int rows, int K) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
    const int stride = gridDim.x * nw;
    int r = blockIdx.x * nw + warp;
    if (r >= rows) return;
    ActRegsT<NI, F> ar;
    load_act<NI, F>(ar, a, K, lane);
    RowRegsF<NI, F> c0, c1;
    load_row<NI, F>(c0, fmt, w, r, K, lane);
    while (true) {
        const int r1 = r + stride;
        if (r1 < rows) load_row<NI, F>(c1, fmt, w, r1, K, lane);
        const float acc0 = dot_rowr<NI, F>(c0, ar, lane);
        if (lane == 0) x[r] += acc0;
        if (r1 >= rows) break;
        const int r2 = r1 + stride;
        if (r2 < rows) load_row<NI, F>(c0, fmt, w, r2, K, lane);
        const float acc1 = dot_rowr<NI, F>(c1, ar, lane);
        if (lane == 0) x[r1] += acc1;
        if (r2 >= rows) break;
        r = r2;
    }
}

#define INST_MV_F(NI, F, SUF)                                                                                                                                                                \
    extern "C" __global__ void mvq_res_##SUF##_kernel_##NI(QPARAMS(w), int fmt, const int8_t* xq, const float* xd, const int* xs, float* x, int rows, int K) {                              \
        mv_res_impl<NI, F>(QMAKE(w), fmt, ActP{xq, xd, xs}, x, rows, K);                                                                                                                     \
    }
#define INST_MV(NI) INST_MV_F(NI, 0, q8) INST_MV_F(NI, 1, q40) INST_MV_F(NI, 2, q50) INST_MV_F(NI, 3, q4k) INST_MV_F(NI, 4, q5k) INST_MV_F(NI, 5, q6k)
INST_MV(1) INST_MV(2) INST_MV(3) INST_MV(4) INST_MV(6) INST_MV(8) INST_MV(12) INST_MV(16) INST_MV(24)

// ---- gate / up + SwiGLU + quantize ------------------------------------------------------------------------------------------------------------------------------------------------------------------
// 16 warps (512 threads); a group is 32 consecutive outputs = two iterations of 16 outputs (warp w: output 16 it + w, i.e. its gate row and its up row), then warp 0 quantizes the group. Persistent: the block
// strides over groups; for short rows (PipeOk) the next iteration's four rows are loaded before the current ones are consumed. F must be a multiple of 32.
template <int NI, int F>
__device__ __forceinline__ void gateup_impl(QP wg, QP wu, int fmt, ActP a, int8_t* __restrict__ oq, float* __restrict__ od, int* __restrict__ os, int E, int ngroups) {
    __shared__ float actv[32];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int g = blockIdx.x;
    if (g >= ngroups) return;
    if constexpr (PipeOk<NI, F>::value) {
        ActRegsT<NI, F> ar;
        load_act<NI, F>(ar, a, E, lane);
        RowRegsF<NI, F> ga, ua, gb, ub;   // iteration 0 in (ga, ua), iteration 1 in (gb, ub)
        const size_t f0 = (size_t)g * 32 + warp;
        load_row<NI, F>(ga, fmt, wg, f0, E, lane);
        load_row<NI, F>(ua, fmt, wu, f0, E, lane);
        while (true) {
            const size_t f1 = (size_t)g * 32 + 16 + warp;
            load_row<NI, F>(gb, fmt, wg, f1, E, lane);
            load_row<NI, F>(ub, fmt, wu, f1, E, lane);
            const float x0 = dot_rowr<NI, F>(ga, ar, lane), y0 = dot_rowr<NI, F>(ua, ar, lane);
            if (lane == 0) actv[warp] = (x0 / (1.f + __expf(-x0))) * y0;
            const int g2 = g + gridDim.x;
            if (g2 < ngroups) {
                const size_t f2 = (size_t)g2 * 32 + warp;
                load_row<NI, F>(ga, fmt, wg, f2, E, lane);
                load_row<NI, F>(ua, fmt, wu, f2, E, lane);
            }
            const float x1 = dot_rowr<NI, F>(gb, ar, lane), y1 = dot_rowr<NI, F>(ub, ar, lane);
            if (lane == 0) actv[16 + warp] = (x1 / (1.f + __expf(-x1))) * y1;
            __syncthreads();
            if (warp == 0) quant_group(actv[lane], lane, oq, od, os, g * 32 + lane, g);
            __syncthreads();
            if (g2 >= ngroups) break;
            g = g2;
        }
    } else {
        // rows too long for the pipelined path (the row registers of four rows plus the activation do not fit): the two sub-rows of a group one after the other, two rows (gate, up) in flight
        // per warp; four rows at once spilled to local memory (the 9B's K = 4096: 55 % of the bandwidth)
        for (; g < ngroups; g += gridDim.x) {
            const size_t fa = (size_t)g * 32 + warp, fb = fa + 16;
            {
                RowRegsF<NI, F> r0, r1;
                load_row<NI, F>(r0, fmt, wg, fa, E, lane);
                load_row<NI, F>(r1, fmt, wu, fa, E, lane);
                const float x0 = dot_row<NI, F>(r0, fmt, a, E, lane), y0 = dot_row<NI, F>(r1, fmt, a, E, lane);
                if (lane == 0) actv[warp] = (x0 / (1.f + __expf(-x0))) * y0;
            }
            {
                RowRegsF<NI, F> r2, r3;
                load_row<NI, F>(r2, fmt, wg, fb, E, lane);
                load_row<NI, F>(r3, fmt, wu, fb, E, lane);
                const float x1 = dot_row<NI, F>(r2, fmt, a, E, lane), y1 = dot_row<NI, F>(r3, fmt, a, E, lane);
                if (lane == 0) actv[16 + warp] = (x1 / (1.f + __expf(-x1))) * y1;
            }
            __syncthreads();
            if (warp == 0) quant_group(actv[lane], lane, oq, od, os, g * 32 + lane, g);
            __syncthreads();
        }
    }
}

#define INST_GU_F(NI, F, SUF)                                                                                                                                                                \
    extern "C" __global__ void __launch_bounds__(512) gateupq_##SUF##_kernel_##NI(QPARAMS(wg), QPARAMS(wu), int fmt, const int8_t* xq, const float* xd, const int* xs, int8_t* oq, float* od,       \
                                                                                  int* os, int E, int ngroups) {                                                                            \
        gateup_impl<NI, F>(QMAKE(wg), QMAKE(wu), fmt, ActP{xq, xd, xs}, oq, od, os, E, ngroups);                                                                                             \
    }
#define INST_GU(NI) INST_GU_F(NI, 0, q8) INST_GU_F(NI, 1, q40) INST_GU_F(NI, 2, q50) INST_GU_F(NI, 3, q4k) INST_GU_F(NI, 4, q5k) INST_GU_F(NI, 5, q6k)
INST_GU(1) INST_GU(2) INST_GU(3) INST_GU(4) INST_GU(6) INST_GU(8) INST_GU(12) INST_GU(16) INST_GU(24)

// ---- head: logits[r] = W[r] . h; the same pipelined row loop as the matvec stages ----------------------------------------------------------------------------------------------------------------------
template <int NI, int F>
__device__ __forceinline__ void head_impl(QP w, int fmt, ActP a, float* __restrict__ logits, int rows, int K) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
    const int stride = gridDim.x * nw;
    int r = blockIdx.x * nw + warp;
    if (r >= rows) return;
    ActRegsT<NI, F> ar;
    load_act<NI, F>(ar, a, K, lane);
    RowRegsF<NI, F> c0, c1;
    load_row<NI, F>(c0, fmt, w, r, K, lane);
    while (true) {
        const int r1 = r + stride;
        if (r1 < rows) load_row<NI, F>(c1, fmt, w, r1, K, lane);
        const float acc0 = dot_rowr<NI, F>(c0, ar, lane);
        if (lane == 0) logits[r] = acc0;
        if (r1 >= rows) break;
        const int r2 = r1 + stride;
        if (r2 < rows) load_row<NI, F>(c0, fmt, w, r2, K, lane);
        const float acc1 = dot_rowr<NI, F>(c1, ar, lane);
        if (lane == 0) logits[r1] = acc1;
        if (r2 >= rows) break;
        r = r2;
    }
}

#define INST_HEAD_F(NI, F, SUF)                                                                                                                                                              \
    extern "C" __global__ void headq_##SUF##_kernel_##NI(QPARAMS(w), int fmt, const int8_t* xq, const float* xd, const int* xs, float* logits, int rows, int K) {                            \
        head_impl<NI, F>(QMAKE(w), fmt, ActP{xq, xd, xs}, logits, rows, K);                                                                                                                  \
    }
#define INST_HEAD(NI) INST_HEAD_F(NI, 0, q8) INST_HEAD_F(NI, 1, q40) INST_HEAD_F(NI, 2, q50) INST_HEAD_F(NI, 3, q4k) INST_HEAD_F(NI, 4, q5k) INST_HEAD_F(NI, 5, q6k)
INST_HEAD(1) INST_HEAD(2) INST_HEAD(3) INST_HEAD(4) INST_HEAD(6) INST_HEAD(8) INST_HEAD(12) INST_HEAD(16) INST_HEAD(24)

// ==============================================================================================================================================================================================
// Long rows (the down projection of 4B+ models: K = 9728 ..): a row does not fit the register file next to its double buffer. The activation vector goes to shared memory once per block, and every warp
// streams each row in chunks of C tasks per lane with the NEXT chunk (or the first chunk of the next row) already in flight while the current one is consumed.
// ==============================================================================================================================================================================================
template <int F> struct BigChunk { static constexpr int C = F == FMT_Q8_0 ? 6 : (F <= FMT_Q5_0 ? 4 : (F <= FMT_Q5_K ? 3 : 2)); };

template <int NI, int F, bool SW = false>
__device__ __forceinline__ float dot_part(const RowRegsF<NI, F>& r, const ActP& a, int K, int lane) {
    if constexpr (F == FMT_Q8_0) return dot_q8<NI, SW>(r.v, a, K, lane);
    else if constexpr (F == FMT_Q4_0) return dot_q40<(NI + 1) / 2, false, SW>(r.v, a, K, lane);
    else if constexpr (F == FMT_Q5_0) return dot_q40<(NI + 1) / 2, true, SW>(r.v, a, K, lane);
    else if constexpr (F == FMT_Q4_K) return dot_qk<(NI + 1) / 2, false, SW>(r.v, a, K, lane);
    else if constexpr (F == FMT_Q5_K) return dot_qk<(NI + 1) / 2, true, SW>(r.v, a, K, lane);
    else return dot_q6<(NI + 3) / 4, SW>(r.v, a, K, lane);
}

// tasks per row (the unit `lane + 32 i` indexes)
template <int F> __device__ __forceinline__ int tasks_per_row(int K) { return F == FMT_Q8_0 ? K / 16 : (F == FMT_Q6_K ? K / 64 : K / 32); }

template <int F>
__device__ __forceinline__ void mv_res_big_impl(QP w, ActP a, float* __restrict__ x, int rows, int K) {
    constexpr int C = BigChunk<F>::C;
    // RowRegsF<NI, F> holds NIF tasks with NIF = f(NI): NI is chosen so that NIF == C for this format
    constexpr int NIC = F == FMT_Q8_0 ? C : (F == FMT_Q6_K ? C * 4 : C * 2);
    extern __shared__ __align__(16) unsigned char smem_raw[];
    int8_t* sxq = reinterpret_cast<int8_t*>(smem_raw);
    float* sxd = reinterpret_cast<float*>(sxq + K);
    int* sxs = reinterpret_cast<int*>(sxd + K / 32);
    for (int i = threadIdx.x; i < K / 16; i += blockDim.x) reinterpret_cast<uint4*>(sxq)[swz(i)] = __ldg(reinterpret_cast<const uint4*>(a.xq) + i);
    for (int i = threadIdx.x; i < K / 32; i += blockDim.x) {
        sxd[i] = __ldg(a.xd + i);
        sxs[i] = __ldg(a.xs + i);
    }
    __syncthreads();
    const ActP sa{sxq, sxd, sxs};
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
    const int stride = gridDim.x * nw;
    const int nt = tasks_per_row<F>(K);
    const int nchunks = (nt + 32 * C - 1) / (32 * C);
    int r = blockIdx.x * nw + warp;
    if (r >= rows) return;
    RowRegsF<NIC, F> cur, nxt;
    load_row<NIC, F>(cur, F, w, r, K, lane);
    int j = 0;
    float acc = 0.f;
    while (true) {
        int jn = j + 1, rn = r;
        if (jn == nchunks) { jn = 0; rn = r + stride; }
        const bool more = rn < rows;
        if (more) load_row<NIC, F>(nxt, F, w, rn, K, lane + 32 * C * jn);
        acc += dot_part<NIC, F, true>(cur, sa, K, lane + 32 * C * j);
        if (jn == 0) {
            const float total = warp_sum(acc);
            if (lane == 0) x[r] += total;
            acc = 0.f;
        }
        if (!more) break;
        cur = nxt;
        r = rn;
        j = jn;
    }
}

#define INST_MVB_F(F, SUF)                                                                                                                                                                   \
    extern "C" __global__ void mvqb_res_##SUF##_kernel(QPARAMS(w), int fmt, const int8_t* xq, const float* xd, const int* xs, float* x, int rows, int K) {                                   \
        mv_res_big_impl<F>(QMAKE(w), ActP{xq, xd, xs}, x, rows, K);                                                                                                                          \
    }
INST_MVB_F(0, q8) INST_MVB_F(1, q40) INST_MVB_F(2, q50) INST_MVB_F(3, q4k) INST_MVB_F(4, q5k) INST_MVB_F(5, q6k)

// ==============================================================================================================================================================================================
// Embedding gather for a quantized table: x[i] = dequantized weight (token, i)
// ==============================================================================================================================================================================================
__device__ __forceinline__ float deq_elem(const QP& p, int fmt, size_t row, int K, int i) {
    switch (fmt) {
        case FMT_Q8_0: return (float)((const int8_t*)p.p0)[row * K + i] * __half2float(((const half*)p.p1)[row * (K / 32) + (i >> 5)]);
        case FMT_Q4_0:
        case FMT_Q5_0: {
            const size_t b = row * (K / 32) + (i >> 5);
            const int j = i & 31;
            const uint8_t byte = p.p0[b * 16 + (j & 15)];
            int code = j < 16 ? (byte & 15) : (byte >> 4);
            if (fmt == FMT_Q5_0) {
                const uint32_t qh = ((const uint32_t*)p.p1)[b];
                code |= ((qh >> (8 * (j & 3) + (j >> 2))) & 1) << 4;
                return __half2float(((const half*)p.p2)[b]) * (float)(code - 16);
            }
            return __half2float(((const half*)p.p1)[b]) * (float)(code - 8);
        }
        case FMT_Q4_K:
        case FMT_Q5_K: {
            const int sb = i >> 8, r = i & 255, j = r >> 5, l = r & 31, c = j >> 1;
            const size_t blk = row * (K / 256) + sb;
            const uint8_t byte = p.p0[blk * 128 + 32 * c + l];
            int code = (j & 1) ? (byte >> 4) : (byte & 15);
            const uint8_t* hdr = (fmt == FMT_Q5_K ? p.p2 : p.p1) + blk * 16;
            const uint32_t d01 = *(const uint32_t*)hdr, u0 = *(const uint32_t*)(hdr + 4), u1 = *(const uint32_t*)(hdr + 8), u2 = *(const uint32_t*)(hdr + 12);
            if (fmt == FMT_Q5_K) {
                const int t = 2 * c + (l >> 4), pp = (j & 1) * 4 + ((l & 15) >> 2), ii = l & 3;
                const uint32_t qh = ((const uint32_t*)p.p1)[blk * 8 + t];
                code |= ((qh >> (8 * ii + pp)) & 1) << 4;
            }
            const uint32_t sw = j < 4 ? (u0 & M_K1) : ((u2 & M_LO) | (((u0 >> 6) & M_K3) << 4));
            const uint32_t mw = j < 4 ? (u1 & M_K1) : (((u2 >> 4) & M_LO) | (((u1 >> 6) & M_K3) << 4));
            const int sc = (sw >> (8 * (j & 3))) & 0xFF, mn = (mw >> (8 * (j & 3))) & 0xFF;
            const float d = __half2float(__ushort_as_half((unsigned short)(d01 & 0xFFFFu))), dmin = __half2float(__ushort_as_half((unsigned short)(d01 >> 16)));
            return d * (float)sc * (float)code - dmin * (float)mn;
        }
        case FMT_Q3_K: {   // 110 B: hmask[32], qs[64], 12 packed scales, d
            const int sb = i >> 8, r = i & 255, h = r >> 7, r2 = r & 127, j = r2 >> 5, rr = r2 & 31, part = rr >> 4, l = rr & 15;
            const uint8_t* b = p.p0 + (row * (size_t)(K / 256) + sb) * 110;
            const int low = (b[32 + 32 * h + 16 * part + l] >> (2 * j)) & 3;
            const int hb = (b[16 * part + l] >> (4 * h + j)) & 1;
            const int is = 8 * h + 2 * j + part, w = is >> 2, kk = is & 3;
            const int lo = (b[96 + (w & 1) * 4 + kk] >> (w < 2 ? 0 : 4)) & 0xF;
            const int hi = (b[96 + 8 + kk] >> (2 * w)) & 3;
            const float d = __half2float(__ushort_as_half((unsigned short)(b[108] | (b[109] << 8))));
            return d * (float)((lo | (hi << 4)) - 32) * (float)(low - (hb ? 0 : 4));
        }
        case FMT_Q2_K: {   // 84 B: scales[16] (scale | min << 4), qs[64], d, dmin
            const int sb = i >> 8, r = i & 255, h = r >> 7, r2 = r & 127, j = r2 >> 5, rr = r2 & 31, part = rr >> 4, l = rr & 15;
            const uint8_t* b = p.p0 + (row * (size_t)(K / 256) + sb) * 84;
            const int q = (b[16 + 32 * h + 16 * part + l] >> (2 * j)) & 3;
            const int sc = b[8 * h + 2 * j + part];
            const float d = __half2float(__ushort_as_half((unsigned short)(b[80] | (b[81] << 8)))), dmin = __half2float(__ushort_as_half((unsigned short)(b[82] | (b[83] << 8))));
            return d * (float)(sc & 15) * (float)q - dmin * (float)(sc >> 4);
        }
        default: {
            const int sb = i >> 8, r = i & 255, h = r >> 7, r2 = r & 127, g = r2 >> 5, l = r2 & 31, u = l >> 4;
            const size_t blk = row * (K / 256) + sb;
            const uint8_t A = p.p0[blk * 128 + 64 * h + l], B = p.p0[blk * 128 + 64 * h + 32 + l], H = p.p1[blk * 64 + 32 * h + l];
            int code;
            if (g == 0) code = (A & 15) | ((H & 3) << 4);
            else if (g == 1) code = (B & 15) | (((H >> 2) & 3) << 4);
            else if (g == 2) code = (A >> 4) | (((H >> 4) & 3) << 4);
            else code = (B >> 4) | (((H >> 6) & 3) << 4);
            const int sc = (int)(signed char)p.p2[blk * 16 + 4 * (2 * h + u) + g];
            return __half2float(((const half*)p.p3)[blk]) * (float)sc * (float)(code - 32);
        }
    }
}

extern "C" __global__ void embed_q_kernel(QPARAMS(w), int fmt, const StepState* __restrict__ st, float* __restrict__ x, int E) {
    const QP p = QMAKE(w);
    const size_t row = st->token;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < E; i += gridDim.x * blockDim.x) x[i] = deq_elem(p, fmt, row, E, i);
}

// test hook: dequantize a whole tensor [rows][K] to fp32 (used by the unit tests to check the repacking against the reference dequantizers)
extern "C" __global__ void dequant_test_kernel(QPARAMS(w), int fmt, float* __restrict__ out, int rows, int K) {
    const QP p = QMAKE(w);
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < (size_t)rows * K; i += (size_t)gridDim.x * blockDim.x) out[i] = deq_elem(p, fmt, i / K, K, (int)(i % K));
}

// ==============================================================================================================================================================================================
// Prefill support for quantized weights: the tensor-core GEMM path wants fp16 weights, so a matrix is dequantized into an fp16 scratch buffer right before its GEMM (one matrix at a time).
// ==============================================================================================================================================================================================
extern "C" __global__ void dequant_f16_kernel(QPARAMS(w), int fmt, half* __restrict__ out, int rows, int K) {
    const QP p = QMAKE(w);
    const size_t total8 = (size_t)rows * K / 8;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < total8; i += (size_t)gridDim.x * blockDim.x) {
        const size_t e = i * 8;
        const size_t row = e / K;
        const int c0 = (int)(e % K);
        uint4 o;
        __half2* oh = reinterpret_cast<__half2*>(&o);
#pragma unroll
        for (int j = 0; j < 4; j++) oh[j] = __floats2half2_rn(deq_elem(p, fmt, row, K, c0 + 2 * j), deq_elem(p, fmt, row, K, c0 + 2 * j + 1));
        reinterpret_cast<uint4*>(out)[i] = o;
    }
}

// like dequant_f16_kernel for the rows [row0, row0 + rows) of the matrix: the evaluation path dequantizes the LM head in slices
extern "C" __global__ void dequant_rows_f16_kernel(QPARAMS(w), int fmt, half* __restrict__ out, int rows, int K, int row0) {
    const QP p = QMAKE(w);
    const size_t total8 = (size_t)rows * K / 8;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < total8; i += (size_t)gridDim.x * blockDim.x) {
        const size_t e = i * 8;
        const size_t row = (size_t)row0 + e / K;
        const int c0 = (int)(e % K);
        uint4 o;
        __half2* oh = reinterpret_cast<__half2*>(&o);
#pragma unroll
        for (int j = 0; j < 4; j++) oh[j] = __floats2half2_rn(deq_elem(p, fmt, row, K, c0 + 2 * j), deq_elem(p, fmt, row, K, c0 + 2 * j + 1));
        reinterpret_cast<uint4*>(out)[i] = o;
    }
}

extern "C" __global__ void embed_q_batch_kernel(QPARAMS(w), int fmt, const uint32_t* __restrict__ tokens, float* __restrict__ x, int T, int E) {
    const QP p = QMAKE(w);
    const int t = blockIdx.x;
    const size_t row = tokens[t];
    for (int i = threadIdx.x; i < E; i += blockDim.x) x[(size_t)t * E + i] = deq_elem(p, fmt, row, E, i);
}

// x += y (the output of an FFN computed on the CPU, copied back to the device)
extern "C" __global__ void resid_add_f32_kernel(float* __restrict__ x, const float* __restrict__ y, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += y[i];
}
