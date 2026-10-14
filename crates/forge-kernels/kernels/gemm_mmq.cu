// Integer tensor-core GEMM against K-quant weights ("MMQ", the scheme of llama.cpp's mul_mat_q kernels, built here from scratch for the planar layouts of decode_q.cu): C[M][N] (op)= A W^T where A (the
// prompt's activations) has been quantized to int8 with one scale per 32 values (quant_rows_h_kernel: the Q8_1 rule of the decode kernels' quantizer) and W stays 4 / 5 bits per weight in global memory.
// The products are int8 x uint8 (0 .. 15 or 0 .. 31) sums, so the tensor cores run at their int8 rate (125 TOPS = 62 TMAC/s peak on an RTX 4060, measured, against 30 TMAC/s for fp16 with fp16
// accumulation and 15 for fp16 with fp32 accumulation) and the weights cross the memory system at 4.5 / 5.5 bits.
//
// Q4_K and Q5_K: a weight is d sc q - dmin m with the 6-bit scale sc and minimum m of its 32-weight sub-block j, so for one output and one sub-block
//     dot = a_d ( d sc S - dmin m T )     S = sum q a_q (the integer MMA), T = sum a_q (the quantizer's sum of the 32 values)
// The first term is the MMA's int32 result times d sc a_d. The second is a tiny product: A2[token][j] = a_d T_j (fp16) times W2[row][j] = -dmin m_j (fp16), summed over the 8 sub-blocks of a super-block
// by ONE fp16 MMA (k = 8 of 16) per output tile, whose fp32 accumulator is the output accumulator itself.
//
// The integer sum S of a sub-block is brought to float without a conversion: the MMA's accumulator input is the constant 0x4B400000 (the float 1.5 * 2^23), so its output is the bit pattern of the float
// 1.5 * 2^23 + S (exact: |S| <= 32 * 127 * 31 < 2^22) and  d sc S  is ONE fma(bits, d sc, -1.5 * 2^23 d sc) (the same product with a single rounding); a second fma applies the token's scale and adds
// into the fp32 accumulators.
//
// Structure: 128 x 128 block tile, 8 warps of 64 x 32, K stages of 128 (four sub-blocks, half a super-block), two stages, ONE barrier per stage. Everything arrives by cp.async, nothing is staged through
// registers: the activation tile in 144-byte rows, the weights of a stage PACKED as they are in memory (64 bytes per row: two 32-byte chunks, the low nibbles are sub-block 2c, the high nibbles 2c + 1 of chunk
// c; Q5_K appends its four 5th-bit words), and the per-token data the quantizer prepared in the shared-memory layout (scale per sub-block, fp16 minimum operand). One ldmatrix fetches the packed
// fragments of both sub-blocks of a chunk; the nibbles are masked out per sub-block (one LOP3 per register: the high nibbles of Q4_K stay unshifted, i.e. u8 weights 16 q, and the factor 1/16 is
// in the scale; Q5_K ORs the 5th bit in). The weight headers (d, dmin, the 12 scale / minimum bytes) are decoded once per super-block by two threads per column into d * sc (fp32) and the fp16
// minimum operand. Q6_K (scales per 16 weights, no minimum) is the same with k16 MMAs and a three-FFMA epilogue (mmq_q6k_impl). Needs sm_80 (cp.async, m16n8k32 int8 MMA, fp16 MMA) and 81,920 bytes of dynamic shared memory (Q6_K: 98,304; the host sets the attribute).

namespace mmq {

constexpr int BM = 128, BN = 128, BK = 128, NT = 256;
constexpr int ROW = BK + 16;                    // bytes per row of the activation tile (9 x 16: the ldmatrix reads are conflict-free)
constexpr int BROW = 80;                        // bytes per row of the packed weights of a stage: 64 + 16 (Q5_K: the 5th-bit words; five 16-byte pieces: conflict-free ldmatrix)
constexpr int WM = 64, WN = 32, MT = WM / 16, NTL = WN / 8;

struct __align__(16) Smem {
    int8_t a[2][BM][ROW];                       // activations of a stage
    uint8_t bp[2][BN][BROW];                    // packed weights of a stage
    float dsc[2][8][BN];                        // d * sc_j per sub-block j and (permuted) output column (Q4_K: /16 for the odd j)
    __half wmh[2][BN][8];                       // -dmin * m_j per output column and sub-block (the minimum operand)
    float adp[2][8][BM];                        // activation scale per sub-block and (permuted) token
    __half ash[2][BM][8];                       // a_d * (sum of the quantized activations) per token and sub-block (the other minimum operand)
};
constexpr int SMEM_BYTES = (int)sizeof(Smem);   // 81,920

// Q6_K (scales per 16 weights, no minimum): a stage is half a super-block, per row 64 bytes of ql and 32 bytes of qh
constexpr int BROW6 = 112;                      // 64 + 32 + 16 (seven 16-byte pieces: conflict-free ldmatrix)
struct __align__(16) Smem6 {
    int8_t a[2][BM][ROW];                       // activations of a stage
    uint8_t bp[2][BN][BROW6];                   // ql, qh of a stage
    float dsc[2][16][BN];                       // d * sc[8 h + 4 u + g]: half h of the super-block, half u of the sub-block, sub-block g of the half; (permuted) output column
    float ndsc[2][8][BN];                       // -1.5 2^23 (d sc[8 h + g] + d sc[8 h + 4 + g])
    float adp[2][8][BM];                        // activation scale per sub-block and (permuted) token
};
constexpr int SMEM6_BYTES = (int)sizeof(Smem6);  // 98,304

// The thread (g, t) of an MMA owns rows g, g + 8 of every 16-row tile and columns 2t, 2t + 1 of every 8-column tile; the permuted position of a token / column makes its 8 tokens / 8 columns contiguous
__device__ __forceinline__ int tok_pos(int tl) { const int slab = tl / WM, r = tl % WM, mt = r / 16, rr = r % 16; return slab * WM + (rr % 8) * (2 * MT) + mt * 2 + (rr / 8); }
__device__ __forceinline__ int col_pos(int n) { const int slab = n / WN, r = n % WN, nt = r / 8, rr = r % 8; return slab * WN + (rr / 2) * (2 * NTL) + nt * 2 + (rr % 2); }
__device__ __forceinline__ __half to_half_sat(float v) { return __float2half_rn(fminf(fmaxf(v, -65504.f), 65504.f)); }

#if __CUDA_ARCH__ >= 800

__device__ __forceinline__ void ldsm4(uint32_t (&r)[4], uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}
// D = A B + c with A int8, B uint8 and the accumulator input the constant `c` in all four registers (see the header comment: c = 0x4B400000 gives the float bits of 1.5 * 2^23 + S)
__device__ __forceinline__ void mma_s8u8_c(int (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1, int c) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.u8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};\n"
                 : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(c));
}
// fp16 inputs, fp32 accumulator in place: only k = 0 .. 7 are non-zero (a2 = a3 = b1 = 0)
__device__ __forceinline__ void mma_f16_k8(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t b0) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(a1), "r"(0u), "r"(0u), "r"(b0), "r"(0u));
}
__device__ __forceinline__ void cp16(uint32_t dst, const void* src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst), "l"(src));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
__device__ __forceinline__ void cp_wait0() { asm volatile("cp.async.wait_group 0;\n" ::); }

// aq: int8 [Mpad][K]; adT: fp32 [Mpad / 128][K / 256][8][128] (token position tok_pos); ashT: fp16 [Mpad / 128][K / 256][128][8] (see quant_rows_h_kernel); Mpad = M rounded up to 128
template <int F, int EPI, typename CT>
__device__ __forceinline__ void mmq_impl(const int8_t* __restrict__ aq, const float* __restrict__ adT, const __half* __restrict__ ashT, const QP w, CT* __restrict__ C, int M, int N, int K, int ldc) {
    static_assert(F == FMT_Q4_K || F == FMT_Q5_K, "Q4_K and Q5_K");
    constexpr bool Q5 = F == FMT_Q5_K;
    extern __shared__ __align__(16) unsigned char smem_raw[];
    Smem& sm = *reinterpret_cast<Smem*>(smem_raw);
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = (warp >> 2) * WM, wn = (warp & 3) * WN;
    const int g = lane >> 2, t = lane & 3;
    const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
    const int nb = K >> 8, S = 2 * nb;                   // super-blocks per row, stages
    float acc[MT][NTL][4];
#pragma unroll
    for (int i = 0; i < MT; i++)
#pragma unroll
        for (int j = 0; j < NTL; j++)
#pragma unroll
            for (int e = 0; e < 4; e++) acc[i][j][e] = 0.f;

    // A: 16-byte pieces by cp.async (rows beyond M are inside the padded buffer: their results are never stored)
    auto load_a = [&](int s, int buf) {
        const size_t kcol = (size_t)s * BK;
#pragma unroll
        for (int u = 0; u < BM * 8 / NT; u++) {
            const int c = tid + NT * u;
            const int r = c >> 3, cc = c & 7;
            cp16(gemm::smem_u32(&sm.a[buf][r][cc * 16]), aq + (size_t)(m0 + r) * K + kcol + cc * 16);
        }
    };
    // the weights of a stage as they are stored: 64 bytes of nibbles per row (two 16-byte pieces per thread), Q5_K: the stage's four 5th-bit words after them (threads 0 .. 127)
    auto load_b = [&](int st, int buf) {
#pragma unroll
        for (int u = 0; u < BN * 4 / NT; u++) {
            const int idx = tid + NT * u;
            const int r = idx >> 2, q = idx & 3;
            const size_t blk = (size_t)min(n0 + r, N - 1) * nb + (st >> 1);
            cp16(gemm::smem_u32(&sm.bp[buf][r][q * 16]), w.p0 + blk * 128 + (st & 1) * 64 + q * 16);
        }
        if constexpr (Q5) {
            if (tid < BN) {
                const size_t blk = (size_t)min(n0 + tid, N - 1) * nb + (st >> 1);
                cp16(gemm::smem_u32(&sm.bp[buf][tid][64]), w.p1 + blk * 32 + (st & 1) * 16);
            }
        }
    };
    // the token scales and minimum operands of super-block sb, already in the shared-memory layout (one 16-byte piece per thread, the minimum operands by the first half of the block)
    auto load_tok = [&](int sb, int buf) {
        const size_t tile = (size_t)blockIdx.y * nb + sb;
        cp16(gemm::smem_u32(&sm.adp[buf][tid >> 5][(tid & 31) * 4]), adT + (tile * 8 + (tid >> 5)) * BM + (tid & 31) * 4);
        if (tid < BM) cp16(gemm::smem_u32(&sm.ash[buf][tid][0]), ashT + (tile * BM + tid) * 8);
    };
    // the weight headers: thread (col, half) decodes sub-blocks 4 half .. 4 half + 3 of output column col
    uint4 hreg;
    const int hcol = tid & (BN - 1), hhalf = tid >> 7;
    auto ldg_hdr = [&](int sb) { hreg = gemm::ldg128((Q5 ? w.p2 : w.p1) + ((size_t)min(n0 + hcol, N - 1) * nb + sb) * 16); };
    auto store_hdr = [&](int buf) {
        const uint4 h = hreg;
        const uint32_t u0 = h.y, u1 = h.z, u2 = h.w;
        uint32_t sw, mw;
        if (hhalf == 0) {
            sw = u0 & M_K1;
            mw = u1 & M_K1;
        } else {
            sw = (u2 & M_LO) | (((u0 >> 6) & M_K3) << 4);
            mw = ((u2 >> 4) & M_LO) | (((u1 >> 6) & M_K3) << 4);
        }
        const __half dh = __ushort_as_half((unsigned short)(h.x & 0xFFFFu)), dmh = __ushort_as_half((unsigned short)(h.x >> 16));
        const float d = __half2float(dh);
        const int pc = col_pos(hcol);
#pragma unroll
        for (int e = 0; e < 4; e++) {
            // Q4_K: the odd sub-blocks are the high nibbles, used unshifted (weights 16 q): their scale carries 1/16
            const float de = (!Q5 && (e & 1)) ? d * 0.0625f : d;
            const float scf = __int_as_float((int)__byte_perm(sw, 0x4B000000u, 0x7440 + e));     // 2^23 + sc
            sm.dsc[buf][4 * hhalf + e][pc] = fmaf(scf, de, -8388608.f * de);                      // d sc
        }
        // -dmin m of two sub-blocks as one HFMA2: (1024 + m) (-dmin) + 1024 dmin
        const __half2 ndm = __half2half2(__hneg(dmh)), kdm = __half2half2(__hmul(dmh, __float2half(1024.f)));
        uint32_t w0 = __byte_perm(mw, 0x64646464u, 0x4140), w1 = __byte_perm(mw, 0x64646464u, 0x4342);
        const __half2 r0 = __hfma2(*reinterpret_cast<__half2*>(&w0), ndm, kdm), r1 = __hfma2(*reinterpret_cast<__half2*>(&w1), ndm, kdm);
        uint2 pk;
        pk.x = *reinterpret_cast<const uint32_t*>(&r0);
        pk.y = *reinterpret_cast<const uint32_t*>(&r1);
        *reinterpret_cast<uint2*>(&sm.wmh[buf][hcol][4 * hhalf]) = pk;
    };

    load_a(0, 0);
    load_b(0, 0);
    load_tok(0, 0);
    cp_commit();
    ldg_hdr(0);
    cp_wait0();
    store_hdr(0);
    __syncthreads();

    for (int s = 0; s < S; s++) {
        const int cur = s & 1, nxt = s + 1;
        if (nxt < S) {
            load_a(nxt, nxt & 1);
            load_b(nxt, nxt & 1);
            if ((nxt & 1) == 0) load_tok(nxt >> 1, (nxt >> 1) & 1);
            cp_commit();
            if ((nxt & 1) == 0) ldg_hdr(nxt >> 1);
        }
        const int sbuf = (s >> 1) & 1;
#pragma unroll 1
        for (int cl = 0; cl < 2; cl++) {
            const int q = lane >> 3, lr = lane & 7;
            // the packed fragments of the chunk (both of its sub-blocks), and for Q5_K the 5th-bit words of the thread's four columns (the two 16-byte halves of the chunk)
            uint32_t fbp[NTL / 2][4], qh[NTL][2];
#pragma unroll
            for (int nj = 0; nj < NTL / 2; nj++)
                ldsm4(fbp[nj], gemm::smem_u32(&sm.bp[cur][wn + nj * 16 + lr + (q >> 1) * 8][cl * 32 + (q & 1) * 16]));
            if constexpr (Q5) {
#pragma unroll
                for (int nt = 0; nt < NTL; nt++) {
                    const uint2 v = *reinterpret_cast<const uint2*>(&sm.bp[cur][wn + nt * 8 + g][64 + cl * 8]);
                    qh[nt][0] = v.x;
                    qh[nt][1] = v.y;
                }
            }
#pragma unroll
            for (int nib = 0; nib < 2; nib++) {
                const int jl = cl * 2 + nib;
                const int j = (s & 1) * 4 + jl;
                uint32_t fa[MT][4], fb[NTL / 2][4];
#pragma unroll
                for (int mt = 0; mt < MT; mt++)
                    ldsm4(fa[mt], gemm::smem_u32(&sm.a[cur][wm + mt * 16 + lr + (q & 1) * 8][jl * 32 + (q >> 1) * 16]));
                // the weights of the sub-block: bytes 0 .. 15 (q 0 .. 31 for Q5_K) of the packed fragments' nibble nib
#pragma unroll
                for (int nj = 0; nj < NTL / 2; nj++)
#pragma unroll
                    for (int e = 0; e < 4; e++) {
                        if constexpr (!Q5) {
                            fb[nj][e] = fbp[nj][e] & (nib ? 0xF0F0F0F0u : 0x0F0F0F0Fu);
                        } else {
                            // bit (8 i + 4 nib + t) of the 5th-bit word is the 5th bit of byte i of this thread's 4 bytes
                            const uint32_t w5 = qh[2 * nj + (e >> 1)][e & 1];
                            const uint32_t b5 = nib ? ((w5 >> t) & 0x10101010u) : ((w5 << (4 - t)) & 0x10101010u);
                            fb[nj][e] = ((nib ? (fbp[nj][e] >> 4) : fbp[nj][e]) & 0x0F0F0F0Fu) | b5;
                        }
                    }
                float adv[MT * 2], ds[NTL * 2], nds[NTL * 2];
#pragma unroll
                for (int i = 0; i < MT * 2 / 4; i++) *reinterpret_cast<float4*>(&adv[4 * i]) = *reinterpret_cast<const float4*>(&sm.adp[sbuf][j][wm + g * (2 * MT) + 4 * i]);
#pragma unroll
                for (int i = 0; i < NTL * 2 / 4; i++) *reinterpret_cast<float4*>(&ds[4 * i]) = *reinterpret_cast<const float4*>(&sm.dsc[sbuf][j][wn + t * (2 * NTL) + 4 * i]);
#pragma unroll
                for (int i = 0; i < NTL * 2; i++) nds[i] = ds[i] * -12582912.f;
#pragma unroll
                for (int mt = 0; mt < MT; mt++) {
#pragma unroll
                    for (int nt = 0; nt < NTL; nt++) {
                        int sr[4];
                        mma_s8u8_c(sr, fa[mt], fb[nt >> 1][(nt & 1) * 2], fb[nt >> 1][(nt & 1) * 2 + 1], 0x4B400000);
                        acc[mt][nt][0] = fmaf(adv[mt * 2], fmaf(__int_as_float(sr[0]), ds[nt * 2], nds[nt * 2]), acc[mt][nt][0]);
                        acc[mt][nt][1] = fmaf(adv[mt * 2], fmaf(__int_as_float(sr[1]), ds[nt * 2 + 1], nds[nt * 2 + 1]), acc[mt][nt][1]);
                        acc[mt][nt][2] = fmaf(adv[mt * 2 + 1], fmaf(__int_as_float(sr[2]), ds[nt * 2], nds[nt * 2]), acc[mt][nt][2]);
                        acc[mt][nt][3] = fmaf(adv[mt * 2 + 1], fmaf(__int_as_float(sr[3]), ds[nt * 2 + 1], nds[nt * 2 + 1]), acc[mt][nt][3]);
                    }
                }
            }
        }
        if (s & 1) {
            // the end of the super-block: the minimum terms of its 8 sub-blocks as one fp16 MMA per tile
#pragma unroll
            for (int mt = 0; mt < MT; mt++) {
                const uint32_t a0 = *reinterpret_cast<const uint32_t*>(&sm.ash[sbuf][wm + mt * 16 + g][2 * t]);
                const uint32_t a1 = *reinterpret_cast<const uint32_t*>(&sm.ash[sbuf][wm + mt * 16 + g + 8][2 * t]);
#pragma unroll
                for (int nt = 0; nt < NTL; nt++) {
                    const uint32_t b0 = *reinterpret_cast<const uint32_t*>(&sm.wmh[sbuf][wn + nt * 8 + g][2 * t]);
                    mma_f16_k8(acc[mt][nt], a0, a1, b0);
                }
            }
        }
        if (nxt < S) {
            cp_wait0();
            if ((nxt & 1) == 0) store_hdr((nxt >> 1) & 1);
        }
        __syncthreads();
    }
    // epilogue: acc[mt][nt][e]: row = m0 + wm + 16 mt + g + 8 (e >> 1), col = n0 + wn + 8 nt + 2 t + (e & 1)
#pragma unroll
    for (int mt = 0; mt < MT; mt++)
#pragma unroll
        for (int nt = 0; nt < NTL; nt++)
#pragma unroll
            for (int h = 0; h < 2; h++) {
                const int row = m0 + wm + mt * 16 + g + 8 * h, col = n0 + wn + nt * 8 + 2 * t;
                if (row < M && col < N) {
                    const float v0 = acc[mt][nt][2 * h], v1 = acc[mt][nt][2 * h + 1];
                    if (EPI == 0) {
                        *reinterpret_cast<__half2*>(reinterpret_cast<__half*>(C) + (size_t)row * ldc + col) = __floats2half2_rn(v0, v1);
                    } else if (EPI == 1) {
                        *reinterpret_cast<float2*>(reinterpret_cast<float*>(C) + (size_t)row * ldc + col) = make_float2(v0, v1);
                    } else {
                        float2* p = reinterpret_cast<float2*>(reinterpret_cast<float*>(C) + (size_t)row * ldc + col);
                        float2 o = *p;
                        o.x += v0;
                        o.y += v1;
                        *p = o;
                    }
                }
            }
}

// D = A B + c, k = 16, int8 x int8 (the accumulator input is the constant c in all four registers)
__device__ __forceinline__ void mma_k16s8_c(int (&d)[4], uint32_t a0, uint32_t a1, uint32_t b0, int c) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%7,%7,%7};\n"
                 : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
                 : "r"(a0), "r"(a1), "r"(b0), "r"(c));
}

// Q6_K: a weight is d sc (q6 - 32) with sc per 16 weights (int8), so a 32-weight sub-block takes two k16 MMAs with their own scale: x = fma(bits1, d sc1, fma(bits0, d sc0, -1.5 2^23 (d sc0 + d sc1))). The
// weights are made signed bytewise by (v + 0x60606060) ^ 0x80808080 (v in 0 .. 63). A stage is half a super-block (sub-blocks g = 0 .. 3 of 32): per row 64 bytes of ql (chunk c: sub-block c in the low
// nibbles, c + 2 in the high ones) and 32 bytes of qh (bits 2g, 2g + 1 of every byte: the upper two bits of sub-block g), one ldmatrix fetches the fragments of a chunk or of the qh bytes.
template <int EPI, typename CT>
__device__ __forceinline__ void mmq_q6k_impl(const int8_t* __restrict__ aq, const float* __restrict__ adT, const QP w, CT* __restrict__ C, int M, int N, int K, int ldc) {
    extern __shared__ __align__(16) unsigned char smem_raw[];
    Smem6& sm = *reinterpret_cast<Smem6*>(smem_raw);
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = (warp >> 2) * WM, wn = (warp & 3) * WN;
    const int g = lane >> 2, t = lane & 3;
    const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
    const int nb = K >> 8, S = 2 * nb;
    float acc[MT][NTL][4];
#pragma unroll
    for (int i = 0; i < MT; i++)
#pragma unroll
        for (int j = 0; j < NTL; j++)
#pragma unroll
            for (int e = 0; e < 4; e++) acc[i][j][e] = 0.f;

    auto load_a = [&](int s, int buf) {
        const size_t kcol = (size_t)s * BK;
#pragma unroll
        for (int u = 0; u < BM * 8 / NT; u++) {
            const int c = tid + NT * u;
            const int r = c >> 3, cc = c & 7;
            cp16(gemm::smem_u32(&sm.a[buf][r][cc * 16]), aq + (size_t)(m0 + r) * K + kcol + cc * 16);
        }
    };
    auto load_tok = [&](int sb, int buf) {
        const size_t tile = (size_t)blockIdx.y * nb + sb;
        cp16(gemm::smem_u32(&sm.adp[buf][tid >> 5][(tid & 31) * 4]), adT + (tile * 8 + (tid >> 5)) * BM + (tid & 31) * 4);
    };
    // a stage: per row the 64 bytes of ql and the 32 bytes of qh of half h of the super-block, six 16-byte pieces
    auto load_b = [&](int st, int buf) {
        const int h = st & 1;
#pragma unroll
        for (int u = 0; u < BN * 6 / NT; u++) {
            const int idx = tid + NT * u;
            const int r = idx / 6, q = idx % 6;
            const size_t blk = (size_t)min(n0 + r, N - 1) * nb + (st >> 1);
            const uint8_t* src = q < 4 ? w.p0 + blk * 128 + 64 * h + 16 * q : w.p1 + blk * 64 + 32 * h + 16 * (q - 4);
            cp16(gemm::smem_u32(&sm.bp[buf][r][q * 16]), src);
        }
    };
    // the scales of a super-block: thread (col, half) decodes the 8 scale bytes of half h = half
    uint4 hreg;
    unsigned short dreg;
    const int hcol = tid & (BN - 1), hhalf = tid >> 7;
    auto ldg_hdr = [&](int sb) {
        const size_t blk = (size_t)min(n0 + hcol, N - 1) * nb + sb;
        hreg = gemm::ldg128(w.p2 + blk * 16);
        dreg = (unsigned short)gemm::ldg_u16(reinterpret_cast<const unsigned short*>(w.p3) + blk);
    };
    auto store_hdr = [&](int buf) {
        const float d = __half2float(__ushort_as_half(dreg));
        const float nd1 = -8388608.f * d, nd2 = -128.f * d;      // exact; so is every step below (d has 11 significant bits, sc + 128 has 8)
        const uint32_t ww[4] = {hreg.x, hreg.y, hreg.z, hreg.w};
        const int pc = col_pos(hcol);
        float ds0[4], ds1[4];
#pragma unroll
        for (int u = 0; u < 2; u++) {
            const uint32_t x = ww[2 * hhalf + u] ^ 0x80808080u;                                   // sc + 128 per byte
#pragma unroll
            for (int gg = 0; gg < 4; gg++) {
                const float scf = __int_as_float((int)__byte_perm(x, 0x4B000000u, 0x7440 + gg));   // 2^23 + sc + 128
                const float v = fmaf(scf, d, nd1) + nd2;                                          // d sc
                sm.dsc[buf][8 * hhalf + 4 * u + gg][pc] = v;
                if (u) ds1[gg] = v; else ds0[gg] = v;
            }
        }
#pragma unroll
        for (int gg = 0; gg < 4; gg++) sm.ndsc[buf][4 * hhalf + gg][pc] = -12582912.f * (ds0[gg] + ds1[gg]);
    };

    load_a(0, 0);
    load_b(0, 0);
    load_tok(0, 0);
    cp_commit();
    ldg_hdr(0);
    cp_wait0();
    store_hdr(0);
    __syncthreads();

    for (int s = 0; s < S; s++) {
        const int cur = s & 1, nxt = s + 1, h = s & 1;
        if (nxt < S) {
            load_a(nxt, nxt & 1);
            load_b(nxt, nxt & 1);
            if ((nxt & 1) == 0) load_tok(nxt >> 1, (nxt >> 1) & 1);
            cp_commit();
            if ((nxt & 1) == 0) ldg_hdr(nxt >> 1);
        }
        const int sbuf = (s >> 1) & 1;
        const int q = lane >> 3, lr = lane & 7;
        uint32_t fqh[NTL / 2][4];
#pragma unroll
        for (int nj = 0; nj < NTL / 2; nj++) ldsm4(fqh[nj], gemm::smem_u32(&sm.bp[cur][wn + nj * 16 + lr + (q >> 1) * 8][64 + (q & 1) * 16]));
#pragma unroll 1
        for (int cl = 0; cl < 2; cl++) {
            uint32_t fql[NTL / 2][4];
#pragma unroll
            for (int nj = 0; nj < NTL / 2; nj++) ldsm4(fql[nj], gemm::smem_u32(&sm.bp[cur][wn + nj * 16 + lr + (q >> 1) * 8][cl * 32 + (q & 1) * 16]));
#pragma unroll
            for (int nib = 0; nib < 2; nib++) {
                const int gs = 2 * nib + cl;                      // the sub-block of the half
                const int j = 4 * h + gs;                         // ... of the super-block
                uint32_t fa[MT][4], fb[NTL / 2][4];
#pragma unroll
                for (int mt = 0; mt < MT; mt++) ldsm4(fa[mt], gemm::smem_u32(&sm.a[cur][wm + mt * 16 + lr + (q & 1) * 8][gs * 32 + (q >> 1) * 16]));
#pragma unroll
                for (int nj = 0; nj < NTL / 2; nj++)
#pragma unroll
                    for (int e = 0; e < 4; e++) {
                        const uint32_t lo = nib ? ((fql[nj][e] >> 4) & 0x0F0F0F0Fu) : (fql[nj][e] & 0x0F0F0F0Fu);
                        const uint32_t hi = (gs == 3) ? ((fqh[nj][e] >> 2) & 0x30303030u) : ((fqh[nj][e] << (4 - 2 * gs)) & 0x30303030u);
                        fb[nj][e] = (lo + hi + 0x60606060u) ^ 0x80808080u;     // (q6 - 32) as int8
                    }
                float adv[MT * 2], d0[NTL * 2], d1[NTL * 2], nd[NTL * 2];
#pragma unroll
                for (int i = 0; i < MT * 2 / 4; i++) *reinterpret_cast<float4*>(&adv[4 * i]) = *reinterpret_cast<const float4*>(&sm.adp[sbuf][j][wm + g * (2 * MT) + 4 * i]);
#pragma unroll
                for (int i = 0; i < NTL * 2 / 4; i++) {
                    *reinterpret_cast<float4*>(&d0[4 * i]) = *reinterpret_cast<const float4*>(&sm.dsc[sbuf][8 * h + gs][wn + t * (2 * NTL) + 4 * i]);
                    *reinterpret_cast<float4*>(&d1[4 * i]) = *reinterpret_cast<const float4*>(&sm.dsc[sbuf][8 * h + 4 + gs][wn + t * (2 * NTL) + 4 * i]);
                    *reinterpret_cast<float4*>(&nd[4 * i]) = *reinterpret_cast<const float4*>(&sm.ndsc[sbuf][4 * h + gs][wn + t * (2 * NTL) + 4 * i]);
                }
#pragma unroll
                for (int mt = 0; mt < MT; mt++) {
#pragma unroll
                    for (int nt = 0; nt < NTL; nt++) {
                        int s0[4], s1[4];
                        mma_k16s8_c(s0, fa[mt][0], fa[mt][1], fb[nt >> 1][(nt & 1) * 2], 0x4B400000);
                        mma_k16s8_c(s1, fa[mt][2], fa[mt][3], fb[nt >> 1][(nt & 1) * 2 + 1], 0x4B400000);
#pragma unroll
                        for (int e = 0; e < 4; e++) {
                            const int ci = nt * 2 + (e & 1);
                            const float x = fmaf(__int_as_float(s1[e]), d1[ci], fmaf(__int_as_float(s0[e]), d0[ci], nd[ci]));
                            acc[mt][nt][e] = fmaf(adv[mt * 2 + (e >> 1)], x, acc[mt][nt][e]);
                        }
                    }
                }
            }
        }
        if (nxt < S) {
            cp_wait0();
            if ((nxt & 1) == 0) store_hdr((nxt >> 1) & 1);
        }
        __syncthreads();
    }
#pragma unroll
    for (int mt = 0; mt < MT; mt++)
#pragma unroll
        for (int nt = 0; nt < NTL; nt++)
#pragma unroll
            for (int h2 = 0; h2 < 2; h2++) {
                const int row = m0 + wm + mt * 16 + g + 8 * h2, col = n0 + wn + nt * 8 + 2 * t;
                if (row < M && col < N) {
                    const float v0 = acc[mt][nt][2 * h2], v1 = acc[mt][nt][2 * h2 + 1];
                    if (EPI == 0) {
                        *reinterpret_cast<__half2*>(reinterpret_cast<__half*>(C) + (size_t)row * ldc + col) = __floats2half2_rn(v0, v1);
                    } else if (EPI == 1) {
                        *reinterpret_cast<float2*>(reinterpret_cast<float*>(C) + (size_t)row * ldc + col) = make_float2(v0, v1);
                    } else {
                        float2* p = reinterpret_cast<float2*>(reinterpret_cast<float*>(C) + (size_t)row * ldc + col);
                        float2 o = *p;
                        o.x += v0;
                        o.y += v1;
                        *p = o;
                    }
                }
            }
}

#endif  // __CUDA_ARCH__ >= 800

}  // namespace mmq

// EPI: 0 fp16 out, 1 fp32 out, 2 add into fp32 (as the other prompt GEMMs). K a multiple of 256, N a multiple of 8.
#if __CUDA_ARCH__ >= 800
#define MMQ_BODY(F, EPI, CT) mmq::mmq_impl<F, EPI, CT>(aq, adT, ashT, QMAKE(w), C, M, N, K, ldc);
#else
#define MMQ_BODY(F, EPI, CT)
#endif
#define MMQ_KERNEL(NAME, F, EPI, CT)                                                                                                                                                             \
    extern "C" __global__ void __launch_bounds__(256, 1) NAME(const int8_t* aq, const float* adT, const __half* ashT, QPARAMS(w), CT* C, int M, int N, int K, int ldc) { MMQ_BODY(F, EPI, CT) }
MMQ_KERNEL(mmq_q4k_f16, FMT_Q4_K, 0, half)
MMQ_KERNEL(mmq_q4k_f32, FMT_Q4_K, 1, float)
MMQ_KERNEL(mmq_q4k_add, FMT_Q4_K, 2, float)
MMQ_KERNEL(mmq_q5k_f16, FMT_Q5_K, 0, half)
MMQ_KERNEL(mmq_q5k_f32, FMT_Q5_K, 1, float)
MMQ_KERNEL(mmq_q5k_add, FMT_Q5_K, 2, float)

// Q6_K: 98,304 bytes of dynamic shared memory (mmq::Smem6); no minimum operands, so ashT is not read
#if __CUDA_ARCH__ >= 800
#define MMQ6_BODY(EPI, CT) mmq::mmq_q6k_impl<EPI, CT>(aq, adT, QMAKE(w), C, M, N, K, ldc);
#else
#define MMQ6_BODY(EPI, CT)
#endif
#define MMQ6_KERNEL(NAME, EPI, CT)                                                                                                                                                              \
    extern "C" __global__ void __launch_bounds__(256, 1) NAME(const int8_t* aq, const float* adT, const __half* ashT, QPARAMS(w), CT* C, int M, int N, int K, int ldc) { MMQ6_BODY(EPI, CT) }
MMQ6_KERNEL(mmq_q6k_f16, 0, half)
MMQ6_KERNEL(mmq_q6k_f32, 1, float)
MMQ6_KERNEL(mmq_q6k_add, 2, float)

// fp16 [M][K] -> the arrays of the integer GEMMs: int8 q [M][K] and, per 32 values, the scale d (the quantizer's fp32 d = amax / 127 rounded to fp16, as the decode kernels store it) and the minimum
// operand fp16(d * sum of the 32 quantized values), both in the layout the GEMM copies into shared memory (128-row tiles, 8 sub-blocks per super-block):
//   adT[((tile * K / 256 + sb) * 8 + j) * 128 + tok_pos(row % 128)] = d        ashT[((tile * K / 256 + sb) * 128 + row % 128) * 8 + j] = fp16(d * sum)
// The rule is quant_group's (llama.cpp's Q8_1: d = amax / 127 unrounded for the division, the stored scale rounded to fp16). One block of 8 warps per row; a warp takes 8 groups per step (4 lanes of 8
// values per group). K a multiple of 256.
extern "C" __global__ void __launch_bounds__(256) quant_rows_h_kernel(const half* __restrict__ x, int8_t* __restrict__ xq, float* __restrict__ adT, __half* __restrict__ ashT, int M, int K) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const half* xr = x + (size_t)row * K;
    int8_t* qr = xq + (size_t)row * K;
    const int nb = K >> 8;
    const size_t tile = (size_t)(row >> 7) * nb;
    const int r = row & 127, pos = mmq::tok_pos(r);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (int base = warp * 256; base < K; base += 8 * 256) {
        const uint4 raw = *reinterpret_cast<const uint4*>(xr + base + lane * 8);
        float v[8];
        float amax = 0.f;
        {
            const __half2* h2 = reinterpret_cast<const __half2*>(&raw);
#pragma unroll
            for (int i = 0; i < 4; i++) {
                const float2 f = __half22float2(h2[i]);
                v[2 * i] = f.x;
                v[2 * i + 1] = f.y;
                amax = fmaxf(amax, fmaxf(fabsf(f.x), fabsf(f.y)));
            }
        }
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
        const float d = amax * (1.f / 127.f);
        const float id = d != 0.f ? 1.f / d : 0.f;
        int q[8], sum = 0;
#pragma unroll
        for (int i = 0; i < 8; i++) {
            q[i] = (int)roundf(v[i] * id);
            sum += q[i];
        }
        sum += __shfl_xor_sync(0xffffffffu, sum, 1);
        sum += __shfl_xor_sync(0xffffffffu, sum, 2);
        uint2 pk;
        pk.x = (uint32_t)(q[0] & 0xFF) | ((uint32_t)(q[1] & 0xFF) << 8) | ((uint32_t)(q[2] & 0xFF) << 16) | ((uint32_t)(q[3] & 0xFF) << 24);
        pk.y = (uint32_t)(q[4] & 0xFF) | ((uint32_t)(q[5] & 0xFF) << 8) | ((uint32_t)(q[6] & 0xFF) << 16) | ((uint32_t)(q[7] & 0xFF) << 24);
        *reinterpret_cast<uint2*>(qr + base + lane * 8) = pk;
        if ((lane & 3) == 0) {
            const int gi = (base >> 5) + (lane >> 2);        // the 32-value group of the row
            const int sb = gi >> 3, j = gi & 7;
            const float dr = round_half(d);
            adT[((tile + sb) * 8 + j) * 128 + pos] = dr;
            ashT[((tile + sb) * 128 + r) * 8 + j] = mmq::to_half_sat(dr * (float)sum);
        }
    }
}

// QUALITY STUDY (FORGE_FAKEQ): quantize and dequantize the fp16 activations x [M][K] in place, one thread per group of G values (G <= 128, K a multiple of G), so that the fp16 GEMMs see what an int8-activation GEMM would see.
// mode 0: the engine's rule (d = amax / 127, q = round(x / d), the scale stored as fp16); 1: the scale kept in fp32; 2: asymmetric (min / max, 255 steps); 3: the largest value of the group kept exact, the scale from the
// second largest; 4: two int8 passes (the residual of the first pass quantized again with its own scale)
extern "C" __global__ void __launch_bounds__(128) fakeq_rows_kernel(half* __restrict__ x, int M, int K, int G, int mode) {
    const int row = blockIdx.x;
    if (row >= M) return;
    half* xr = x + (size_t)row * K;
    for (int g0 = threadIdx.x * G; g0 < K; g0 += blockDim.x * G) {
        float v[128];
        float amax = 0.f, vmin = 3.0e38f, vmax = -3.0e38f;
        int imax = 0;
        for (int i = 0; i < G; i++) {
            v[i] = __half2float(xr[g0 + i]);
            const float a = fabsf(v[i]);
            if (a > amax) {
                amax = a;
                imax = i;
            }
            vmin = fminf(vmin, v[i]);
            vmax = fmaxf(vmax, v[i]);
        }
        if (mode == 0 || mode == 1) {
            const float d = amax * (1.f / 127.f);
            const float id = d != 0.f ? 1.f / d : 0.f;
            const float dr = mode == 0 ? round_half(d) : d;
            for (int i = 0; i < G; i++) xr[g0 + i] = __float2half(roundf(v[i] * id) * dr);
        } else if (mode == 2) {
            const float sc = (vmax - vmin) * (1.f / 255.f);
            const float isc = sc != 0.f ? 1.f / sc : 0.f;
            for (int i = 0; i < G; i++) xr[g0 + i] = __float2half(roundf((v[i] - vmin) * isc) * sc + vmin);
        } else if (mode == 3) {
            float a2 = 0.f;
            for (int i = 0; i < G; i++)
                if (i != imax) a2 = fmaxf(a2, fabsf(v[i]));
            const float d = a2 * (1.f / 127.f);
            const float id = d != 0.f ? 1.f / d : 0.f;
            const float dr = round_half(d);
            for (int i = 0; i < G; i++) xr[g0 + i] = i == imax ? __float2half(v[i]) : __float2half(fminf(fmaxf(roundf(v[i] * id), -127.f), 127.f) * dr);
        } else {
            const float d = amax * (1.f / 127.f);
            const float id = d != 0.f ? 1.f / d : 0.f;
            const float dr = round_half(d);
            float rmax = 0.f;
            for (int i = 0; i < G; i++) {
                v[i] -= roundf(v[i] * id) * dr;
                rmax = fmaxf(rmax, fabsf(v[i]));
            }
            const float d2 = rmax * (1.f / 127.f);
            const float id2 = d2 != 0.f ? 1.f / d2 : 0.f;
            const float dr2 = round_half(d2);
            for (int i = 0; i < G; i++) {
                const float first = __half2float(xr[g0 + i]) - v[i];  // (the first pass's value, exact in fp32)
                xr[g0 + i] = __float2half(first + roundf(v[i] * id2) * dr2);
            }
        }
    }
}

// a = silu(g) * u (g, u: fp16 [M][K]) and its int8 form for the integer GEMMs (the arrays of quant_rows_h_kernel) in ONE pass: the fp16 `a` of swiglu_kernel is never written or read back; the values are rounded through
// fp16 exactly as swiglu_kernel stores them, so the result is bit-identical to swiglu_kernel followed by quant_rows_h_kernel
extern "C" __global__ void __launch_bounds__(256) swiglu_quant_rows_kernel(const half* __restrict__ gx, const half* __restrict__ ux, int8_t* __restrict__ xq, float* __restrict__ adT, __half* __restrict__ ashT, int M, int K) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const half* gr = gx + (size_t)row * K;
    const half* ur = ux + (size_t)row * K;
    int8_t* qr = xq + (size_t)row * K;
    const int nb = K >> 8;
    const size_t tile = (size_t)(row >> 7) * nb;
    const int r = row & 127, pos = mmq::tok_pos(r);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (int base = warp * 256; base < K; base += 8 * 256) {
        const uint4 rawg = *reinterpret_cast<const uint4*>(gr + base + lane * 8), rawu = *reinterpret_cast<const uint4*>(ur + base + lane * 8);
        float v[8];
        float amax = 0.f;
        {
            const __half2* g2 = reinterpret_cast<const __half2*>(&rawg);
            const __half2* u2 = reinterpret_cast<const __half2*>(&rawu);
#pragma unroll
            for (int i = 0; i < 4; i++) {
                const float2 gf = __half22float2(g2[i]), uf = __half22float2(u2[i]);
                // the fp16 values swiglu_kernel would store (the quantizer then reads them back)
                const float a0 = __half2float(__float2half((gf.x / (1.f + __expf(-gf.x))) * uf.x));
                const float a1 = __half2float(__float2half((gf.y / (1.f + __expf(-gf.y))) * uf.y));
                v[2 * i] = a0;
                v[2 * i + 1] = a1;
                amax = fmaxf(amax, fmaxf(fabsf(a0), fabsf(a1)));
            }
        }
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
        const float d = amax * (1.f / 127.f);
        const float id = d != 0.f ? 1.f / d : 0.f;
        int q[8], sum = 0;
#pragma unroll
        for (int i = 0; i < 8; i++) {
            q[i] = (int)roundf(v[i] * id);
            sum += q[i];
        }
        sum += __shfl_xor_sync(0xffffffffu, sum, 1);
        sum += __shfl_xor_sync(0xffffffffu, sum, 2);
        uint2 pk;
        pk.x = (uint32_t)(q[0] & 0xFF) | ((uint32_t)(q[1] & 0xFF) << 8) | ((uint32_t)(q[2] & 0xFF) << 16) | ((uint32_t)(q[3] & 0xFF) << 24);
        pk.y = (uint32_t)(q[4] & 0xFF) | ((uint32_t)(q[5] & 0xFF) << 8) | ((uint32_t)(q[6] & 0xFF) << 16) | ((uint32_t)(q[7] & 0xFF) << 24);
        *reinterpret_cast<uint2*>(qr + base + lane * 8) = pk;
        if ((lane & 3) == 0) {
            const int gi = (base >> 5) + (lane >> 2);        // the 32-value group of the row
            const int sb = gi >> 3, j = gi & 7;
            const float dr = round_half(d);
            adT[((tile + sb) * 8 + j) * 128 + pos] = dr;
            ashT[((tile + sb) * 128 + r) * 8 + j] = mmq::to_half_sat(dr * (float)sum);
        }
    }
}
