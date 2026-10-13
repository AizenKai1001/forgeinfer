// Causal flash-attention forward for sm_75 on tensor cores (FlashAttention-2 structure: online softmax, the accumulator layout of one mma is the A layout of the next).
// A block = 64 queries of one head (4 warps x 16 rows). Q stays in registers as mma A fragments; K and V tiles stream through shared memory (row stride HD + 8 halves, conflict free for ldmatrix).
//   S = Q K^T      mma.m16n8k8 with K as the "col" B operand (K is stored [key][dim] = [n][k]: plain ldmatrix)
//   P = exp2(S * scale * log2e - m)  rescaled online;  O += P V   with V as the B operand through ldmatrix.trans.
// Queries are the rows of a chunk (absolute position pos0 + row); keys are the KV cache positions 0 .. pos0 + T - 1 (already written by qkv_rope_kernel).

#include <cuda_fp16.h>
#include <stdint.h>

namespace flash {

__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }

__device__ __forceinline__ void ldsm_x4(uint32_t (&r)[4], uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}
__device__ __forceinline__ void ldsm_x4_t(uint32_t (&r)[4], uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}
__device__ __forceinline__ void mma_k8(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t b0) {
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(a1), "r"(b0));
}
__device__ __forceinline__ uint32_t pack_half2(float lo, float hi) {
    half2 h = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<uint32_t*>(&h);
}

// WIN = true: sliding-window attention, the rule of flash_train.cu: the query at absolute position p sees the keys j with p - window < j <= p. Key tiles (absolute multiples of KT, as in the
// training kernel) that no row of the block / warp can see are skipped; a tile that is fully masked for a row changes nothing in its online softmax, so the result of a row is the training kernel's.
// WIN = false: the dense causal kernel, `window` is ignored.
template <int HD, int KT, bool WIN = false>
__device__ __forceinline__ void flash_fwd_impl(const half* __restrict__ q, const half* __restrict__ k_cache, const half* __restrict__ v_cache, half* __restrict__ out, float* __restrict__ lse, int T,
                                               int pos0, int n_head, int n_kv, int window = 0) {
    constexpr int LDS = HD + 8;            // halves per smem row
    constexpr int BQ = 64;
    constexpr int KS = HD / 8;             // k8 steps over the head dimension
    constexpr int NK = KT / 8;             // n8 tiles of S (keys per tile / 8)
    constexpr int ND = HD / 8;             // n8 tiles of O
    __shared__ __align__(16) half Qs[BQ * LDS];
    __shared__ __align__(16) half Ks[KT * LDS];
    __shared__ __align__(16) half Vs[KT * LDS];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int g = lane >> 2, t4 = lane & 3;
    const int hh = blockIdx.y, kvh = hh / (n_head / n_kv);
    const int q_dim = n_head * HD, kv_dim = n_kv * HD;
    const int q0 = blockIdx.x * BQ;
    // Q tile -> shared memory (zero rows beyond T)
    for (int i = tid; i < BQ * HD / 8; i += blockDim.x) {
        const int r = i / (HD / 8), c = i % (HD / 8);
        uint4 v = make_uint4(0u, 0u, 0u, 0u);
        if (q0 + r < T) v = __ldg(reinterpret_cast<const uint4*>(q + (size_t)(q0 + r) * q_dim + hh * HD) + c);
        *reinterpret_cast<uint4*>(&Qs[r * LDS + c * 8]) = v;
    }
    __syncthreads();
    // A fragments of this warp's 16 query rows for every k8 step
    uint32_t qf[KS][2];
    {
        const int a_row = (lane & 7) + ((lane >> 3) & 1) * 8, a_col = (lane >> 4) * 8;
#pragma unroll
        for (int s2 = 0; s2 < HD / 16; s2++) {
            uint32_t r[4];
            ldsm_x4(r, smem_u32(&Qs[(warp * 16 + a_row) * LDS + s2 * 16 + a_col]));
            qf[2 * s2][0] = r[0];
            qf[2 * s2][1] = r[1];
            qf[2 * s2 + 1][0] = r[2];
            qf[2 * s2 + 1][1] = r[3];
        }
    }
    float o[ND][4];
#pragma unroll
    for (int i = 0; i < ND; i++) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.f;
    float m[2] = {-INFINITY, -INFINITY}, l[2] = {0.f, 0.f};
    const float sl2 = rsqrtf((float)HD) * 1.4426950408889634f;
    const int row0 = q0 + warp * 16 + g, row1 = row0 + 8;          // chunk rows handled by this thread
    const int qpos0 = pos0 + row0, qpos1 = pos0 + row1;
    const int last_row = min(T - 1, q0 + BQ - 1);
    const int kmax = pos0 + last_row;
    const int warp_last = pos0 + min(T - 1, q0 + warp * 16 + 15);  // last absolute query position of this warp
    const int b_row = (lane & 7) + (lane >> 4) * 8, b_col = ((lane >> 3) & 1) * 8;
    const int v_row = lane & 7, v_mat = lane >> 3;

    const int warp_first = pos0 + q0 + warp * 16;                  // first absolute query position of this warp
    // first key tile any query of the block can see
    const int kstart = WIN ? max(0, pos0 + q0 - window + 1) / KT * KT : 0;

    for (int k0 = kstart; k0 <= kmax; k0 += KT) {
        __syncthreads();
        for (int i = tid; i < KT * HD / 8; i += blockDim.x) {
            const int r = i / (HD / 8), c = i % (HD / 8);
            const int kp = k0 + r;
            uint4 kv = make_uint4(0u, 0u, 0u, 0u), vv = make_uint4(0u, 0u, 0u, 0u);
            if (kp <= kmax) {
                kv = __ldg(reinterpret_cast<const uint4*>(k_cache + (size_t)kp * kv_dim + kvh * HD) + c);
                vv = __ldg(reinterpret_cast<const uint4*>(v_cache + (size_t)kp * kv_dim + kvh * HD) + c);
            }
            *reinterpret_cast<uint4*>(&Ks[r * LDS + c * 8]) = kv;
            *reinterpret_cast<uint4*>(&Vs[r * LDS + c * 8]) = vv;
        }
        __syncthreads();
        if (k0 > warp_last) continue;      // this whole tile lies above the warp's last query: skip (the barrier above/below are reached by every warp)
        if (WIN && k0 + KT - 1 <= warp_first - window) continue;   // the whole tile lies below the window of the warp's first query
        // S = Q K^T for 16 queries x KT keys
        float s[NK][4];
#pragma unroll
        for (int i = 0; i < NK; i++) s[i][0] = s[i][1] = s[i][2] = s[i][3] = 0.f;
#pragma unroll
        for (int j2 = 0; j2 < KT / 16; j2++) {
#pragma unroll
            for (int s2 = 0; s2 < HD / 16; s2++) {
                uint32_t kb[4];
                ldsm_x4(kb, smem_u32(&Ks[(j2 * 16 + b_row) * LDS + s2 * 16 + b_col]));
                mma_k8(s[2 * j2], qf[2 * s2][0], qf[2 * s2][1], kb[0]);
                mma_k8(s[2 * j2], qf[2 * s2 + 1][0], qf[2 * s2 + 1][1], kb[1]);
                mma_k8(s[2 * j2 + 1], qf[2 * s2][0], qf[2 * s2][1], kb[2]);
                mma_k8(s[2 * j2 + 1], qf[2 * s2 + 1][0], qf[2 * s2 + 1][1], kb[3]);
            }
        }
        // scale, causal mask, row maxima (rows g and g + 8; the four lanes of a quad hold the same rows)
        float mx0 = -INFINITY, mx1 = -INFINITY;
#pragma unroll
        for (int nt = 0; nt < NK; nt++) {
#pragma unroll
            for (int e = 0; e < 4; e++) {
                const int kp = k0 + nt * 8 + 2 * t4 + (e & 1);
                const int qp = (e >> 1) ? qpos1 : qpos0;
                const int rowv = (e >> 1) ? row1 : row0;
                float v = s[nt][e] * sl2;
                if (kp > qp || rowv >= T || kp > kmax || (WIN && qp - kp >= window)) v = -INFINITY;
                s[nt][e] = v;
                if (e >> 1) mx1 = fmaxf(mx1, v); else mx0 = fmaxf(mx0, v);
            }
        }
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 1));
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 2));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 1));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 2));
        const float mn0 = fmaxf(m[0], mx0), mn1 = fmaxf(m[1], mx1);
        const float ms0 = mn0 == -INFINITY ? 0.f : mn0, ms1 = mn1 == -INFINITY ? 0.f : mn1;
        const float corr0 = exp2f(m[0] - ms0), corr1 = exp2f(m[1] - ms1);   // exp2(-inf) = 0 on the first tile
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
        // P = exp2(S - m), row sums, A fragments for P V
        uint32_t pa[NK][2];
#pragma unroll
        for (int nt = 0; nt < NK; nt++) {
            const float p0 = exp2f(s[nt][0] - ms0), p1 = exp2f(s[nt][1] - ms0), p2 = exp2f(s[nt][2] - ms1), p3 = exp2f(s[nt][3] - ms1);
            l[0] += p0 + p1;
            l[1] += p2 + p3;
            pa[nt][0] = pack_half2(p0, p1);
            pa[nt][1] = pack_half2(p2, p3);
        }
        // O += P V : for key chunk j (8 keys) and dim tiles nd (8 dims): B operand from V through ldmatrix.trans
#pragma unroll
        for (int j = 0; j < NK; j++) {
#pragma unroll
            for (int nd4 = 0; nd4 < ND / 4; nd4++) {
                uint32_t vb[4];
                ldsm_x4_t(vb, smem_u32(&Vs[(j * 8 + v_row) * LDS + (nd4 * 4 + v_mat) * 8]));
#pragma unroll
                for (int u = 0; u < 4; u++) mma_k8(o[nd4 * 4 + u], pa[j][0], pa[j][1], vb[u]);
            }
        }
    }
    // normalise and store: l is a per-thread partial sum over its own columns; add the quad
    l[0] += __shfl_xor_sync(0xffffffffu, l[0], 1);
    l[0] += __shfl_xor_sync(0xffffffffu, l[0], 2);
    l[1] += __shfl_xor_sync(0xffffffffu, l[1], 1);
    l[1] += __shfl_xor_sync(0xffffffffu, l[1], 2);
    const float inv0 = l[0] > 0.f ? 1.f / l[0] : 0.f, inv1 = l[1] > 0.f ? 1.f / l[1] : 0.f;
    if (lse != nullptr && t4 == 0) {   // row log-sum-exp in the log2 domain of the scaled scores (used by the backward pass)
        if (row0 < T) lse[(size_t)row0 * n_head + hh] = l[0] > 0.f ? m[0] + log2f(l[0]) : -INFINITY;
        if (row1 < T) lse[(size_t)row1 * n_head + hh] = l[1] > 0.f ? m[1] + log2f(l[1]) : -INFINITY;
    }
#pragma unroll
    for (int nd = 0; nd < ND; nd++) {
        const int col = hh * HD + nd * 8 + 2 * t4;
        if (row0 < T) *reinterpret_cast<half2*>(out + (size_t)row0 * q_dim + col) = __floats2half2_rn(o[nd][0] * inv0, o[nd][1] * inv0);
        if (row1 < T) *reinterpret_cast<half2*>(out + (size_t)row1 * q_dim + col) = __floats2half2_rn(o[nd][2] * inv1, o[nd][3] * inv1);
    }
}

}  // namespace flash

extern "C" __global__ void __launch_bounds__(128) flash_fwd_64(const half* q, const half* k_cache, const half* v_cache, half* out, int T, int pos0, int n_head, int n_kv) {
    flash::flash_fwd_impl<64, 64>(q, k_cache, v_cache, out, nullptr, T, pos0, n_head, n_kv);
}
// with a sliding window (head dimension 64 only, as in training): the rows of a chunk over the cache, query at absolute position p sees p - window < j <= p
extern "C" __global__ void __launch_bounds__(128) flash_fwd_64_win(const half* q, const half* k_cache, const half* v_cache, half* out, int T, int pos0, int n_head, int n_kv, int window) {
    flash::flash_fwd_impl<64, 64, true>(q, k_cache, v_cache, out, nullptr, T, pos0, n_head, n_kv, window);
}
extern "C" __global__ void __launch_bounds__(128) flash_fwd_128(const half* q, const half* k_cache, const half* v_cache, half* out, int T, int pos0, int n_head, int n_kv) {
    flash::flash_fwd_impl<128, 32>(q, k_cache, v_cache, out, nullptr, T, pos0, n_head, n_kv);
}
// training: gridDim.z sequences of T tokens each, laid out one after the other ([B*T][dim]); positions start at 0 in every sequence; also writes the log-sum-exp
extern "C" __global__ void __launch_bounds__(128) flash_fwd_train_64(const half* q, const half* k, const half* v, half* out, float* lse, int T, int n_head, int n_kv) {
    const size_t z = blockIdx.z;
    flash::flash_fwd_impl<64, 64>(q + z * T * n_head * 64, k + z * T * n_kv * 64, v + z * T * n_kv * 64, out + z * T * n_head * 64, lse + z * T * n_head, T, 0, n_head, n_kv);
}
extern "C" __global__ void __launch_bounds__(128) flash_fwd_train_128(const half* q, const half* k, const half* v, half* out, float* lse, int T, int n_head, int n_kv) {
    const size_t z = blockIdx.z;
    flash::flash_fwd_impl<128, 32>(q + z * T * n_head * 128, k + z * T * n_kv * 128, v + z * T * n_kv * 128, out + z * T * n_head * 128, lse + z * T * n_head, T, 0, n_head, n_kv);
}
