// Causal flash-attention forward for head width 256 (Qwen3.5's gated attention) on sm_75 tensor cores; uses the helpers of flash.cu.
// The accumulators of a 64 x 256 output tile plus the Q fragments do not fit the register file next to each other, so each block computes HALF of the output dimensions (blockIdx.z selects the
// half): S = Q K^T over all 256 dimensions (recomputed by the other half's block: 1.5x the tensor work of one pass), P V for 128 output dimensions.
// Q lives in shared memory only until its fragments are in registers; the K / V tiles reuse that space. Optional sigmoid gate on the output (gate[row * gate_tok + head * gate_head + dim]).

#include <cuda_fp16.h>
#include <stdint.h>

namespace flash {

template <int HQ, int HO, int KT, bool GATE>
__device__ __forceinline__ void flash_fwd_split_impl(const half* __restrict__ q, const half* __restrict__ k_cache, const half* __restrict__ v_cache, half* __restrict__ out, int T, int pos0, int n_head,
                                                     int n_kv, const half* __restrict__ gate, int gate_tok, int gate_head) {
    constexpr int LDQ = HQ + 8;            // halves per smem row of Q and K
    constexpr int LDV = HO + 8;            // halves per smem row of the V slice
    constexpr int BQ = 64;
    constexpr int KS = HQ / 8;             // k8 steps over the head dimension
    constexpr int NK = KT / 8;             // n8 tiles of S
    constexpr int ND = HO / 8;             // n8 tiles of O
    constexpr int Q_ELEMS = BQ * LDQ;
    constexpr int KV_ELEMS = KT * LDQ + KT * LDV;
    __shared__ __align__(16) half sm[Q_ELEMS > KV_ELEMS ? Q_ELEMS : KV_ELEMS];
    half* Qs = sm;
    half* Ks = sm;
    half* Vs = sm + KT * LDQ;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int g = lane >> 2, t4 = lane & 3;
    const int hh = blockIdx.y, kvh = hh / (n_head / n_kv);
    const int vo = blockIdx.z * HO;        // first output dimension of this block
    const int q_dim = n_head * HQ, kv_dim = n_kv * HQ;
    const int q0 = blockIdx.x * BQ;
    for (int i = tid; i < BQ * HQ / 8; i += blockDim.x) {
        const int r = i / (HQ / 8), c = i % (HQ / 8);
        uint4 v = make_uint4(0u, 0u, 0u, 0u);
        if (q0 + r < T) v = __ldg(reinterpret_cast<const uint4*>(q + (size_t)(q0 + r) * q_dim + hh * HQ) + c);
        *reinterpret_cast<uint4*>(&Qs[r * LDQ + c * 8]) = v;
    }
    __syncthreads();
    uint32_t qf[KS][2];
    {
        const int a_row = (lane & 7) + ((lane >> 3) & 1) * 8, a_col = (lane >> 4) * 8;
#pragma unroll
        for (int s2 = 0; s2 < HQ / 16; s2++) {
            uint32_t r[4];
            ldsm_x4(r, smem_u32(&Qs[(warp * 16 + a_row) * LDQ + s2 * 16 + a_col]));
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
    const float sl2 = rsqrtf((float)HQ) * 1.4426950408889634f;
    const int row0 = q0 + warp * 16 + g, row1 = row0 + 8;
    const int qpos0 = pos0 + row0, qpos1 = pos0 + row1;
    const int last_row = min(T - 1, q0 + BQ - 1);
    const int kmax = pos0 + last_row;
    const int warp_last = pos0 + min(T - 1, q0 + warp * 16 + 15);
    const int b_row = (lane & 7) + (lane >> 4) * 8, b_col = ((lane >> 3) & 1) * 8;
    const int v_row = lane & 7, v_mat = lane >> 3;
    for (int k0 = 0; k0 <= kmax; k0 += KT) {
        __syncthreads();
        for (int i = tid; i < KT * HQ / 8; i += blockDim.x) {
            const int r = i / (HQ / 8), c = i % (HQ / 8);
            const int kp = k0 + r;
            uint4 kv = make_uint4(0u, 0u, 0u, 0u);
            if (kp <= kmax) kv = __ldg(reinterpret_cast<const uint4*>(k_cache + (size_t)kp * kv_dim + kvh * HQ) + c);
            *reinterpret_cast<uint4*>(&Ks[r * LDQ + c * 8]) = kv;
        }
        for (int i = tid; i < KT * HO / 8; i += blockDim.x) {
            const int r = i / (HO / 8), c = i % (HO / 8);
            const int kp = k0 + r;
            uint4 vv = make_uint4(0u, 0u, 0u, 0u);
            if (kp <= kmax) vv = __ldg(reinterpret_cast<const uint4*>(v_cache + (size_t)kp * kv_dim + kvh * HQ + vo) + c);
            *reinterpret_cast<uint4*>(&Vs[r * LDV + c * 8]) = vv;
        }
        __syncthreads();
        if (k0 > warp_last) continue;
        float s[NK][4];
#pragma unroll
        for (int i = 0; i < NK; i++) s[i][0] = s[i][1] = s[i][2] = s[i][3] = 0.f;
#pragma unroll
        for (int j2 = 0; j2 < KT / 16; j2++) {
#pragma unroll
            for (int s2 = 0; s2 < HQ / 16; s2++) {
                uint32_t kb[4];
                ldsm_x4(kb, smem_u32(&Ks[(j2 * 16 + b_row) * LDQ + s2 * 16 + b_col]));
                mma_k8(s[2 * j2], qf[2 * s2][0], qf[2 * s2][1], kb[0]);
                mma_k8(s[2 * j2], qf[2 * s2 + 1][0], qf[2 * s2 + 1][1], kb[1]);
                mma_k8(s[2 * j2 + 1], qf[2 * s2][0], qf[2 * s2][1], kb[2]);
                mma_k8(s[2 * j2 + 1], qf[2 * s2 + 1][0], qf[2 * s2 + 1][1], kb[3]);
            }
        }
        float mx0 = -INFINITY, mx1 = -INFINITY;
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
        for (int j = 0; j < NK; j++) {
#pragma unroll
            for (int nd4 = 0; nd4 < ND / 4; nd4++) {
                uint32_t vb[4];
                ldsm_x4_t(vb, smem_u32(&Vs[(j * 8 + v_row) * LDV + (nd4 * 4 + v_mat) * 8]));
#pragma unroll
                for (int u = 0; u < 4; u++) mma_k8(o[nd4 * 4 + u], pa[j][0], pa[j][1], vb[u]);
            }
        }
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

}  // namespace flash

// Qwen3.5 gated attention prefill: grid (ceil(T / 64), n_head, 2), 128 threads
extern "C" __global__ void __launch_bounds__(128) flash_fwd_256g(const half* q, const half* k_cache, const half* v_cache, half* out, int T, int pos0, int n_head, int n_kv, const half* gate, int gate_tok,
                                                                int gate_head) {
    flash::flash_fwd_split_impl<256, 128, 32, true>(q, k_cache, v_cache, out, T, pos0, n_head, n_kv, gate, gate_tok, gate_head);
}
