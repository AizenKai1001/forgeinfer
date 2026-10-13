// Flash-attention backward for sm_75 (head dimension 64, causal, GQA), two kernels so that the result is deterministic (no atomics):
//   flash_bwd_dkdv_64: one block per (64-key tile, kv head, sequence); the 4 warps own 16 keys each; loops over the query heads of the group and over the query tiles from the diagonal on.
//                      S^T = K Q^T and dP^T = V dO^T give P^T = exp2(S^T * scale * log2e - lse) and dS^T = P^T * (dP^T - D); then dV += P^T dO and dK += dS^T Q * scale. The accumulator layout of
//                      one mma is the A layout of the next, so P^T and dS^T feed the next products straight from registers.
//   flash_bwd_dq_64:   one block per (64-query tile, head, sequence); loops over the key tiles up to the diagonal; dQ += dS K * scale.
// Inputs are the fp16 q, k, v, dO, the forward's output statistics lse (log2 domain) and D = rowsum(dO * O). T must be a multiple of 64.

#include <cuda_fp16.h>
#include <stdint.h>

namespace flashb {

constexpr int LDS = 72;   // smem row stride in halves (64 + 8)

using flash::ldsm_x4;
using flash::ldsm_x4_t;
using flash::mma_k8;
using flash::pack_half2;
using flash::smem_u32;

__device__ __forceinline__ void load_tile(const half* __restrict__ src, int ld, half* dst, int tid) {
    for (int i = tid; i < 64 * 8; i += 128) {
        const int r = i >> 3, c = i & 7;
        *reinterpret_cast<uint4*>(&dst[r * LDS + c * 8]) = __ldg(reinterpret_cast<const uint4*>(src + (size_t)r * ld) + c);
    }
}

// A fragments (16 rows x 64 dims) of the warp's rows of a smem tile
__device__ __forceinline__ void load_a_frags(const half* tile, int warp, int lane, uint32_t (&f)[8][2]) {
    const int a_row = (lane & 7) + ((lane >> 3) & 1) * 8, a_col = (lane >> 4) * 8;
#pragma unroll
    for (int s2 = 0; s2 < 4; s2++) {
        uint32_t r[4];
        ldsm_x4(r, smem_u32(&tile[(warp * 16 + a_row) * LDS + s2 * 16 + a_col]));
        f[2 * s2][0] = r[0];
        f[2 * s2][1] = r[1];
        f[2 * s2 + 1][0] = r[2];
        f[2 * s2 + 1][1] = r[3];
    }
}

// C[16 x 64] (8 n-tiles) += A(16 x 64 dims, registers) * B^T where B is a smem tile [64 rows][64 dims] ("col" operand: rows are the n index)
__device__ __forceinline__ void mma_a_bt(float (&c)[8][4], const uint32_t (&a)[8][2], const half* tile, int lane) {
    const int b_row = (lane & 7) + (lane >> 4) * 8, b_col = ((lane >> 3) & 1) * 8;
#pragma unroll
    for (int j2 = 0; j2 < 4; j2++) {
#pragma unroll
        for (int s2 = 0; s2 < 4; s2++) {
            uint32_t b[4];
            ldsm_x4(b, smem_u32(&tile[(j2 * 16 + b_row) * LDS + s2 * 16 + b_col]));
            mma_k8(c[2 * j2], a[2 * s2][0], a[2 * s2][1], b[0]);
            mma_k8(c[2 * j2], a[2 * s2 + 1][0], a[2 * s2 + 1][1], b[1]);
            mma_k8(c[2 * j2 + 1], a[2 * s2][0], a[2 * s2][1], b[2]);
            mma_k8(c[2 * j2 + 1], a[2 * s2 + 1][0], a[2 * s2 + 1][1], b[3]);
        }
    }
}

// acc[16 x 64 dims] += Afrag(C layout of a [16 x 64 cols] tile, converted to fp16) * tile[64 rows (the reduction index)][64 dims]  (tile read with ldmatrix.trans)
__device__ __forceinline__ void mma_c_b(float (&acc)[8][4], const float (&c)[8][4], const half* tile, int lane) {
    const int v_row = lane & 7, v_mat = lane >> 3;
#pragma unroll
    for (int nt = 0; nt < 8; nt++) {
        const uint32_t a0 = pack_half2(c[nt][0], c[nt][1]), a1 = pack_half2(c[nt][2], c[nt][3]);
#pragma unroll
        for (int nd4 = 0; nd4 < 2; nd4++) {
            uint32_t b[4];
            ldsm_x4_t(b, smem_u32(&tile[(nt * 8 + v_row) * LDS + (nd4 * 4 + v_mat) * 8]));
#pragma unroll
            for (int u = 0; u < 4; u++) mma_k8(acc[nd4 * 4 + u], a0, a1, b[u]);
        }
    }
}

}  // namespace flashb

extern "C" __global__ void __launch_bounds__(128, 2) flash_bwd_dkdv_64(const half* q, const half* k, const half* v, const half* dO, const float* lse, const float* Dv, float* dk, float* dv, int T, int n_head,
                                                                       int n_kv) {
    using namespace flashb;
    // the K / V tiles are only read once, into register fragments; the Q / dO tiles of the loop reuse their shared memory (two blocks per SM)
    __shared__ __align__(16) half tiles[2 * 64 * LDS];
    __shared__ float Ls[64], Ds[64];
    half* Ks = tiles;
    half* Vs = tiles + 64 * LDS;
    half* Qs = tiles;
    half* dOs = tiles + 64 * LDS;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t4 = lane & 3;
    const size_t b = blockIdx.z;
    const int kvh = blockIdx.y, kt = blockIdx.x, k0 = kt * 64;
    const int q_dim = n_head * 64, kv_dim = n_kv * 64, rep = n_head / n_kv;
    const half* kb = k + (b * T + k0) * kv_dim + kvh * 64;
    const half* vb = v + (b * T + k0) * kv_dim + kvh * 64;
    load_tile(kb, kv_dim, Ks, tid);
    load_tile(vb, kv_dim, Vs, tid);
    __syncthreads();
    uint32_t kf[8][2], vf[8][2];
    load_a_frags(Ks, warp, lane, kf);
    load_a_frags(Vs, warp, lane, vf);
    float dkacc[8][4], dvacc[8][4];
#pragma unroll
    for (int i = 0; i < 8; i++)
#pragma unroll
        for (int e = 0; e < 4; e++) dkacc[i][e] = dvacc[i][e] = 0.f;
    const float scale = 0.125f;                       // 1 / sqrt(64)
    const float sl2 = scale * 1.4426950408889634f;
    for (int hr = 0; hr < rep; hr++) {
        const int h = kvh * rep + hr;
        for (int qt = kt; qt < T / 64; qt++) {
            const int q0 = qt * 64;
            __syncthreads();
            load_tile(q + (b * T + q0) * q_dim + h * 64, q_dim, Qs, tid);
            load_tile(dO + (b * T + q0) * q_dim + h * 64, q_dim, dOs, tid);
            if (tid < 64) {
                Ls[tid] = lse[(b * T + q0 + tid) * n_head + h];
                Ds[tid] = Dv[(b * T + q0 + tid) * n_head + h];
            }
            __syncthreads();
            float s[8][4], dp[8][4];
#pragma unroll
            for (int i = 0; i < 8; i++)
#pragma unroll
                for (int e = 0; e < 4; e++) s[i][e] = dp[i][e] = 0.f;
            mma_a_bt(s, kf, Qs, lane);      // S^T = K Q^T   (16 keys x 64 queries)
            mma_a_bt(dp, vf, dOs, lane);    // dP^T = V dO^T
#pragma unroll
            for (int nt = 0; nt < 8; nt++) {
#pragma unroll
                for (int e = 0; e < 4; e++) {
                    const int col = nt * 8 + 2 * t4 + (e & 1);
                    const int kpos = k0 + warp * 16 + g + 8 * (e >> 1), qpos = q0 + col;
                    const float p = kpos <= qpos ? exp2f(s[nt][e] * sl2 - Ls[col]) : 0.f;
                    s[nt][e] = p;                                  // P^T
                    dp[nt][e] = p * (dp[nt][e] - Ds[col]);         // dS^T
                }
            }
            mma_c_b(dvacc, s, dOs, lane);   // dV += P^T dO
            mma_c_b(dkacc, dp, Qs, lane);   // dK += dS^T Q
        }
    }
    // store: rows g and g + 8 of the warp's 16 keys, columns nd * 8 + 2 t4
#pragma unroll
    for (int nd = 0; nd < 8; nd++) {
#pragma unroll
        for (int hrow = 0; hrow < 2; hrow++) {
            const size_t row = b * T + k0 + warp * 16 + g + 8 * hrow;
            const int col = kvh * 64 + nd * 8 + 2 * t4;
            *reinterpret_cast<float2*>(dk + row * kv_dim + col) = make_float2(dkacc[nd][2 * hrow] * scale, dkacc[nd][2 * hrow + 1] * scale);
            *reinterpret_cast<float2*>(dv + row * kv_dim + col) = make_float2(dvacc[nd][2 * hrow], dvacc[nd][2 * hrow + 1]);
        }
    }
}

extern "C" __global__ void __launch_bounds__(128, 2) flash_bwd_dq_64(const half* q, const half* k, const half* v, const half* dO, const float* lse, const float* Dv, float* dq, int T, int n_head, int n_kv) {
    using namespace flashb;
    // the Q / dO tiles are only read once, into register fragments; the K / V tiles of the loop reuse their shared memory (two blocks per SM)
    __shared__ __align__(16) half tiles[2 * 64 * LDS];
    half* Qs = tiles;
    half* dOs = tiles + 64 * LDS;
    half* Ks = tiles;
    half* Vs = tiles + 64 * LDS;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t4 = lane & 3;
    const size_t b = blockIdx.z;
    const int h = blockIdx.y, qt = blockIdx.x, q0 = qt * 64;
    const int q_dim = n_head * 64, kv_dim = n_kv * 64, kvh = h / (n_head / n_kv);
    load_tile(q + (b * T + q0) * q_dim + h * 64, q_dim, Qs, tid);
    load_tile(dO + (b * T + q0) * q_dim + h * 64, q_dim, dOs, tid);
    __syncthreads();
    uint32_t qf[8][2], dof[8][2];
    load_a_frags(Qs, warp, lane, qf);
    load_a_frags(dOs, warp, lane, dof);
    const int r0 = warp * 16 + g, r1 = r0 + 8;
    const float L0 = lse[(b * T + q0 + r0) * n_head + h], L1 = lse[(b * T + q0 + r1) * n_head + h];
    const float D0 = Dv[(b * T + q0 + r0) * n_head + h], D1 = Dv[(b * T + q0 + r1) * n_head + h];
    float dqacc[8][4];
#pragma unroll
    for (int i = 0; i < 8; i++)
#pragma unroll
        for (int e = 0; e < 4; e++) dqacc[i][e] = 0.f;
    const float scale = 0.125f, sl2 = scale * 1.4426950408889634f;
    for (int kt = 0; kt <= qt; kt++) {
        const int k0 = kt * 64;
        __syncthreads();
        load_tile(k + (b * T + k0) * kv_dim + kvh * 64, kv_dim, Ks, tid);
        load_tile(v + (b * T + k0) * kv_dim + kvh * 64, kv_dim, Vs, tid);
        __syncthreads();
        float s[8][4], dp[8][4];
#pragma unroll
        for (int i = 0; i < 8; i++)
#pragma unroll
            for (int e = 0; e < 4; e++) s[i][e] = dp[i][e] = 0.f;
        mma_a_bt(s, qf, Ks, lane);       // S = Q K^T   (16 queries x 64 keys)
        mma_a_bt(dp, dof, Vs, lane);     // dP = dO V^T
#pragma unroll
        for (int nt = 0; nt < 8; nt++) {
#pragma unroll
            for (int e = 0; e < 4; e++) {
                const int col = nt * 8 + 2 * t4 + (e & 1);
                const int kpos = k0 + col, qpos = q0 + warp * 16 + g + 8 * (e >> 1);
                const float Lr = (e >> 1) ? L1 : L0, Dr = (e >> 1) ? D1 : D0;
                const float p = kpos <= qpos ? exp2f(s[nt][e] * sl2 - Lr) : 0.f;
                dp[nt][e] = p * (dp[nt][e] - Dr);      // dS
            }
        }
        mma_c_b(dqacc, dp, Ks, lane);    // dQ += dS K
    }
#pragma unroll
    for (int nd = 0; nd < 8; nd++) {
#pragma unroll
        for (int hrow = 0; hrow < 2; hrow++) {
            const size_t row = b * T + q0 + warp * 16 + g + 8 * hrow;
            *reinterpret_cast<float2*>(dq + row * q_dim + h * 64 + nd * 8 + 2 * t4) = make_float2(dqacc[nd][2 * hrow] * scale, dqacc[nd][2 * hrow + 1] * scale);
        }
    }
}
