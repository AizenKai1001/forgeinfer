// Causal flash-attention forward on the Ampere tensor-core instructions (sm_80 and newer: GeForce RTX 30 / 40, A100, H100): the kernels of flash.cu and flash256.cu rebuilt the way gemm_s80.cu rebuilds the GEMMs.
//  - m16n8k16 MMAs (the k8 form of the Turing kernels runs at half the tensor-core rate on Ampere);
//  - the K and V tiles stream through a double-buffered cp.async pipeline (global -> shared without passing through registers; the next tile is in flight while the current one is consumed);
//  - HACC: the S = Q K^T products of one k16 step accumulate in fp16 inside the tensor core (twice the rate of the fp32 accumulators on GeForce cards) and are added into the fp32 scores after every step, so
//    a partial sum holds 16 products (the rule of the gemm_nt_*_s80h kernels); P V keeps fp32 accumulators. Without HACC everything accumulates in fp32.
// Same arguments, grid ((T + 63) / 64, n_head, HQ / HO) and result (up to rounding) as flash_fwd_64 / flash_fwd_128 / flash_fwd_256g. The blocks run from the last query rows to the first: they have the most keys to
// read, so the light ones fill the tail of the grid. Below sm_80 the kernels are empty (the host picks them only on devices of compute capability 8 and up).

#include <cuda_fp16.h>
#include <stdint.h>

namespace flash {

#if __CUDA_ARCH__ >= 800

namespace a80 {
__device__ __forceinline__ void cp16z(uint32_t dst, const void* src, bool valid) {
    const int sz = valid ? 16 : 0;   // src-size 0: nothing is read and the 16 bytes are filled with zeros
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(sz));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_wait() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}
__device__ __forceinline__ void mma_k16(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
// fp16 result, fp16 accumulator starting at zero: c = (row g, cols 2t and 2t + 1) and (row g + 8, ...) as half2
__device__ __forceinline__ void mma_k16h0(uint32_t (&c)[2], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%8,%8};\n"
                 : "=r"(c[0]), "=r"(c[1])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(0u));
}
}  // namespace a80

// HQ: head width; HO: output dimensions of one block (HQ, or HQ / 2 when blockIdx.z picks the half: S is then computed by both halves); KT: keys per tile; GATE: sigmoid gate on the output
// (gate[row * gate_tok + head * gate_head + dim], the gated attention of Qwen3.5)
template <int HQ, int HO, int KT, bool GATE, bool HACC>
__device__ __forceinline__ void flash_fwd_a80_impl(const half* __restrict__ q, const half* __restrict__ k_cache, const half* __restrict__ v_cache, half* __restrict__ out, int T, int pos0, int n_head,
                                                   int n_kv, const half* __restrict__ gate, int gate_tok, int gate_head) {
    constexpr int BQ = 64;
    constexpr int LDQ = HQ + 8;            // halves per smem row of Q and K (stride 16 bytes off a multiple of 128: ldmatrix reads are conflict free)
    constexpr int LDV = HO + 8;            // halves per smem row of the V slice
    constexpr int NK = KT / 8;             // n8 tiles of S
    constexpr int ND = HO / 8;             // n8 tiles of O
    constexpr int KV_ELEMS = KT * LDQ + KT * LDV;
    constexpr int SM_ELEMS = BQ * LDQ > 2 * KV_ELEMS ? BQ * LDQ : 2 * KV_ELEMS;   // Q only until its fragments are in registers, then two K / V buffers
    using namespace a80;
    __shared__ __align__(16) half sm[SM_ELEMS];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int g = lane >> 2, t4 = lane & 3;
    const int hh = blockIdx.y, kvh = hh / (n_head / n_kv);
    const int vo = blockIdx.z * HO;        // first output dimension of this block
    const int q_dim = n_head * HQ, kv_dim = n_kv * HQ;
    const int q0 = (gridDim.x - 1 - blockIdx.x) * BQ;
    for (int i = tid; i < BQ * HQ / 8; i += 128) {
        const int r = i / (HQ / 8), c = i % (HQ / 8);
        uint4 v = make_uint4(0u, 0u, 0u, 0u);
        if (q0 + r < T) v = __ldg(reinterpret_cast<const uint4*>(q + (size_t)(q0 + r) * q_dim + hh * HQ) + c);
        *reinterpret_cast<uint4*>(&sm[r * LDQ + c * 8]) = v;
    }
    __syncthreads();
    uint32_t qf[HQ / 16][4];
    {
        const int a_row = (lane & 7) + ((lane >> 3) & 1) * 8, a_col = (lane >> 4) * 8;
#pragma unroll
        for (int s2 = 0; s2 < HQ / 16; s2++) ldsm_x4(qf[s2], smem_u32(&sm[(warp * 16 + a_row) * LDQ + s2 * 16 + a_col]));
    }
    __syncthreads();
    float o[ND][4];
#pragma unroll
    for (int i = 0; i < ND; i++) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.f;
    float m[2] = {-INFINITY, -INFINITY}, l[2] = {0.f, 0.f};
    const float sl2 = rsqrtf((float)HQ) * 1.4426950408889634f;
    const int row0 = q0 + warp * 16 + g, row1 = row0 + 8;
    const int qpos0 = pos0 + row0, qpos1 = pos0 + row1;
    const int last_row = min(T - 1, q0 + BQ - 1);
    const int kmax = pos0 + last_row;
    const int warp_last = pos0 + min(T - 1, q0 + warp * 16 + 15);
    const int b_row = (lane & 7) + (lane >> 4) * 8, b_col = ((lane >> 3) & 1) * 8;
    const int v_row = lane & 7, v_mat = lane >> 3;
    const int ntiles = kmax / KT + 1;
    // tiles that end at or before the first query position of the block, in a block of rows that all exist, need no mask at all
    const int free_below = (q0 + BQ <= T) ? pos0 + q0 : -1;

    auto load_tile = [&](int tile, int buf) {
        half* Ks = sm + buf * KV_ELEMS;
        half* Vs = Ks + KT * LDQ;
        const int k0 = tile * KT;
        for (int i = tid; i < KT * HQ / 8; i += 128) {
            const int r = i / (HQ / 8), c = i % (HQ / 8);
            const int kp = k0 + r;
            const bool ok = kp <= kmax;
            cp16z(smem_u32(&Ks[r * LDQ + c * 8]), k_cache + (size_t)(ok ? kp : 0) * kv_dim + kvh * HQ + c * 8, ok);
        }
        for (int i = tid; i < KT * HO / 8; i += 128) {
            const int r = i / (HO / 8), c = i % (HO / 8);
            const int kp = k0 + r;
            const bool ok = kp <= kmax;
            cp16z(smem_u32(&Vs[r * LDV + c * 8]), v_cache + (size_t)(ok ? kp : 0) * kv_dim + kvh * HQ + vo + c * 8, ok);
        }
    };
    load_tile(0, 0);
    cp_commit();
    for (int tile = 0; tile < ntiles; tile++) {
        const int k0 = tile * KT;
        if (tile + 1 < ntiles) load_tile(tile + 1, (tile + 1) & 1);
        cp_commit();
        cp_wait<1>();
        __syncthreads();
        const half* Ks = sm + (tile & 1) * KV_ELEMS;
        const half* Vs = Ks + KT * LDQ;
        if (k0 <= warp_last) {
            float s[NK][4];
#pragma unroll
            for (int i = 0; i < NK; i++) s[i][0] = s[i][1] = s[i][2] = s[i][3] = 0.f;
#pragma unroll
            for (int j2 = 0; j2 < KT / 16; j2++) {
#pragma unroll
                for (int s2 = 0; s2 < HQ / 16; s2++) {
                    uint32_t kb[4];
                    ldsm_x4(kb, smem_u32(&Ks[(j2 * 16 + b_row) * LDQ + s2 * 16 + b_col]));
                    if (HACC) {
                        uint32_t h0[2], h1[2];
                        mma_k16h0(h0, qf[s2], kb[0], kb[1]);
                        mma_k16h0(h1, qf[s2], kb[2], kb[3]);
                        const float2 a0 = __half22float2(*reinterpret_cast<half2*>(&h0[0])), a1 = __half22float2(*reinterpret_cast<half2*>(&h0[1]));
                        const float2 b0 = __half22float2(*reinterpret_cast<half2*>(&h1[0])), b1 = __half22float2(*reinterpret_cast<half2*>(&h1[1]));
                        s[2 * j2][0] += a0.x;
                        s[2 * j2][1] += a0.y;
                        s[2 * j2][2] += a1.x;
                        s[2 * j2][3] += a1.y;
                        s[2 * j2 + 1][0] += b0.x;
                        s[2 * j2 + 1][1] += b0.y;
                        s[2 * j2 + 1][2] += b1.x;
                        s[2 * j2 + 1][3] += b1.y;
                    } else {
                        mma_k16(s[2 * j2], qf[s2], kb[0], kb[1]);
                        mma_k16(s[2 * j2 + 1], qf[s2], kb[2], kb[3]);
                    }
                }
            }
            float mx0 = -INFINITY, mx1 = -INFINITY;
            if (k0 + KT - 1 <= free_below) {
#pragma unroll
                for (int nt = 0; nt < NK; nt++) {
                    s[nt][0] *= sl2;
                    s[nt][1] *= sl2;
                    s[nt][2] *= sl2;
                    s[nt][3] *= sl2;
                    mx0 = fmaxf(mx0, fmaxf(s[nt][0], s[nt][1]));
                    mx1 = fmaxf(mx1, fmaxf(s[nt][2], s[nt][3]));
                }
            } else {
#pragma unroll
                for (int nt = 0; nt < NK; nt++) {
#pragma unroll
                    for (int e = 0; e < 4; e++) {
                        const int kp = k0 + nt * 8 + 2 * t4 + (e & 1);
                        const int qp = (e >> 1) ? qpos1 : qpos0;
                        const int rowv = (e >> 1) ? row1 : row0;
                        float v = s[nt][e] * sl2;
                        if (kp > qp || rowv >= T || kp > kmax) v = -INFINITY;
                        s[nt][e] = v;
                        if (e >> 1) mx1 = fmaxf(mx1, v);
                        else mx0 = fmaxf(mx0, v);
                    }
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
            for (int i = 0; i < ND; i++) {
                o[i][0] *= corr0;
                o[i][1] *= corr0;
                o[i][2] *= corr1;
                o[i][3] *= corr1;
            }
            uint32_t pa[NK][2];
#pragma unroll
            for (int nt = 0; nt < NK; nt++) {
                const float p0 = exp2f(s[nt][0] - ms0), p1 = exp2f(s[nt][1] - ms0), p2 = exp2f(s[nt][2] - ms1), p3 = exp2f(s[nt][3] - ms1);
                l[0] += p0 + p1;
                l[1] += p2 + p3;
                pa[nt][0] = pack_half2(p0, p1);
                pa[nt][1] = pack_half2(p2, p3);
            }
#pragma unroll
            for (int j = 0; j < KT / 16; j++) {
                const uint32_t pk[4] = {pa[2 * j][0], pa[2 * j][1], pa[2 * j + 1][0], pa[2 * j + 1][1]};
#pragma unroll
                for (int nd4 = 0; nd4 < ND / 4; nd4++) {
                    uint32_t vl[4], vh[4];
                    ldsm_x4_t(vl, smem_u32(&Vs[(j * 16 + v_row) * LDV + (nd4 * 4 + v_mat) * 8]));
                    ldsm_x4_t(vh, smem_u32(&Vs[(j * 16 + 8 + v_row) * LDV + (nd4 * 4 + v_mat) * 8]));
#pragma unroll
                    for (int u = 0; u < 4; u++) mma_k16(o[nd4 * 4 + u], pk, vl[u], vh[u]);
                }
            }
        }
        __syncthreads();   // the buffer of this tile is overwritten by the copy issued at the top of the next iteration
    }
    l[0] += __shfl_xor_sync(0xffffffffu, l[0], 1);
    l[0] += __shfl_xor_sync(0xffffffffu, l[0], 2);
    l[1] += __shfl_xor_sync(0xffffffffu, l[1], 1);
    l[1] += __shfl_xor_sync(0xffffffffu, l[1], 2);
    const float inv0 = l[0] > 0.f ? 1.f / l[0] : 0.f, inv1 = l[1] > 0.f ? 1.f / l[1] : 0.f;
#pragma unroll
    for (int nd = 0; nd < ND; nd++) {
        const int d = vo + nd * 8 + 2 * t4;
        float v00 = o[nd][0] * inv0, v01 = o[nd][1] * inv0, v10 = o[nd][2] * inv1, v11 = o[nd][3] * inv1;
        if (GATE) {
            if (row0 < T) {
                const half* gp = gate + (size_t)row0 * gate_tok + (size_t)hh * gate_head + d;
                v00 *= 1.f / (1.f + __expf(-__half2float(gp[0])));
                v01 *= 1.f / (1.f + __expf(-__half2float(gp[1])));
            }
            if (row1 < T) {
                const half* gp = gate + (size_t)row1 * gate_tok + (size_t)hh * gate_head + d;
                v10 *= 1.f / (1.f + __expf(-__half2float(gp[0])));
                v11 *= 1.f / (1.f + __expf(-__half2float(gp[1])));
            }
        }
        const int col = hh * HQ + d;
        if (row0 < T) *reinterpret_cast<half2*>(out + (size_t)row0 * q_dim + col) = __floats2half2_rn(v00, v01);
        if (row1 < T) *reinterpret_cast<half2*>(out + (size_t)row1 * q_dim + col) = __floats2half2_rn(v10, v11);
    }
}

#endif  // __CUDA_ARCH__ >= 800

}  // namespace flash

#if __CUDA_ARCH__ >= 800
#define FA80_BODY(HQ, HO, KT, GATE, HACC) flash::flash_fwd_a80_impl<HQ, HO, KT, GATE, HACC>(q, k_cache, v_cache, out, T, pos0, n_head, n_kv, gate, gate_tok, gate_head);
#else
#define FA80_BODY(HQ, HO, KT, GATE, HACC)
#endif
// Qwen3.5 gated attention (head width 256, one block per half of the output dimensions): grid (ceil(T / 64), n_head, 2), 128 threads; _h: fp16 accumulators for S
extern "C" __global__ void __launch_bounds__(128) flash_fwd_256g_a80(const half* q, const half* k_cache, const half* v_cache, half* out, int T, int pos0, int n_head, int n_kv, const half* gate, int gate_tok,
                                                                    int gate_head) {
    FA80_BODY(256, 128, 16, true, false)
}
extern "C" __global__ void __launch_bounds__(128) flash_fwd_256g_a80h(const half* q, const half* k_cache, const half* v_cache, half* out, int T, int pos0, int n_head, int n_kv, const half* gate, int gate_tok,
                                                                     int gate_head) {
    FA80_BODY(256, 128, 16, true, true)
}
// dense models: the arguments and the grid (ceil(T / 64), n_head, 1) of flash_fwd_64 / flash_fwd_128
#define FA80_DENSE(NAME, HQ, KT, HACC)                                                                                                                                                              \
    extern "C" __global__ void __launch_bounds__(128) NAME(const half* q, const half* k_cache, const half* v_cache, half* out, int T, int pos0, int n_head, int n_kv) {                              \
        const half* gate = nullptr;                                                                                                                                                                 \
        const int gate_tok = 0, gate_head = 0;                                                                                                                                                      \
        FA80_BODY(HQ, HQ, KT, false, HACC)                                                                                                                                                          \
    }
FA80_DENSE(flash_fwd_64_a80, 64, 32, false)
FA80_DENSE(flash_fwd_64_a80h, 64, 32, true)
FA80_DENSE(flash_fwd_128_a80, 128, 32, false)
FA80_DENSE(flash_fwd_128_a80h, 128, 32, true)
