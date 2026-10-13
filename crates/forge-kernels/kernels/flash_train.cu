// Training attention kernels for head dimension 64 (Forge-1), second generation. Same mathematics and the same per-element order of operations as flash_fwd_train_64 / flash_bwd_dkdv_64 / flash_bwd_dq_64
// (the outputs of the dense causal case are bit-identical), with:
//  - forward: the next K / V tile is loaded into registers while the current one is used (the old loop paid the global-memory latency at every tile), the Q tile is staged through the K buffer
//    (18 KB of shared memory instead of 27: three blocks per SM), and the query tiles run from the last (the longest under the causal mask) to the first, so that the short ones fill the tail of the grid;
//  - dQ: longest first, and the next K / V tile is loaded into registers while the current one is used;
//  - an optional SLIDING WINDOW (WIN = true): query i sees the keys j with i - window < j <= i. Tiles entirely outside the window are skipped (forward and both backward kernels), the boundary tiles are
//    masked. window >= T is the dense causal case.
// Grids as before: forward and dQ (T / 64, n_head, batch), dK dV (T / 64, n_kv, batch); 128 threads; T a multiple of 64.

#include <cuda_fp16.h>
#include <stdint.h>

namespace flasht {

constexpr int LDS = 72;   // smem row stride in halves (64 + 8)

using flash::ldsm_x4;
using flash::ldsm_x4_t;
using flash::mma_k8;
using flash::pack_half2;
using flash::smem_u32;

template <bool WIN>
__device__ __forceinline__ void fwd(const half* __restrict__ q, const half* __restrict__ k, const half* __restrict__ v, half* __restrict__ out, float* __restrict__ lse, int T, int n_head, int n_kv,
                                    int window, int bx, int by) {
    constexpr int HD = 64, KT = 64, BQ = 64, NK = KT / 8, ND = HD / 8;
    __shared__ __align__(16) half Ks[KT * LDS];
    __shared__ __align__(16) half Vs[KT * LDS];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int g = lane >> 2, t4 = lane & 3;
    const int hh = by, kvh = hh / (n_head / n_kv);
    const int q_dim = n_head * HD, kv_dim = n_kv * HD;
    const int q0 = (gridDim.x - 1 - bx) * BQ;   // longest query tiles first
    // Q tile -> the K buffer -> A fragments
    for (int i = tid; i < BQ * HD / 8; i += blockDim.x) {
        const int r = i >> 3, c = i & 7;
        uint4 val = make_uint4(0u, 0u, 0u, 0u);
        if (q0 + r < T) val = __ldg(reinterpret_cast<const uint4*>(q + (size_t)(q0 + r) * q_dim + hh * HD) + c);
        *reinterpret_cast<uint4*>(&Ks[r * LDS + c * 8]) = val;
    }
    __syncthreads();
    uint32_t qf[HD / 8][2];
    {
        const int a_row = (lane & 7) + ((lane >> 3) & 1) * 8, a_col = (lane >> 4) * 8;
#pragma unroll
        for (int s2 = 0; s2 < HD / 16; s2++) {
            uint32_t r[4];
            ldsm_x4(r, smem_u32(&Ks[(warp * 16 + a_row) * LDS + s2 * 16 + a_col]));
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
    const int row0 = q0 + warp * 16 + g, row1 = row0 + 8;
    const int last_row = min(T - 1, q0 + BQ - 1);
    const int kmax = last_row;
    const int warp_first = q0 + warp * 16;
    const int warp_last = min(T - 1, q0 + warp * 16 + 15);
    const int b_row = (lane & 7) + (lane >> 4) * 8, b_col = ((lane >> 3) & 1) * 8;
    const int v_row = lane & 7, v_mat = lane >> 3;
    // first key tile any query of the block can see
    const int kstart = WIN ? max(0, q0 - window + 1) / KT * KT : 0;
    // register prefetch of one K and one V tile: 512 chunks of 16 bytes each, 4 per thread
    uint4 kr[4], vr[4];
    auto ldg_kv = [&](int k0) {
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const int i = tid + 128 * j, r = i >> 3, c = i & 7, kp = k0 + r;
            kr[j] = make_uint4(0u, 0u, 0u, 0u);
            vr[j] = make_uint4(0u, 0u, 0u, 0u);
            if (kp <= kmax) {
                kr[j] = __ldg(reinterpret_cast<const uint4*>(k + (size_t)kp * kv_dim + kvh * HD) + c);
                vr[j] = __ldg(reinterpret_cast<const uint4*>(v + (size_t)kp * kv_dim + kvh * HD) + c);
            }
        }
    };
    ldg_kv(kstart);
    for (int k0 = kstart; k0 <= kmax; k0 += KT) {
        __syncthreads();   // everybody is done with the previous tile (and with the Q staging)
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const int i = tid + 128 * j, r = i >> 3, c = i & 7;
            *reinterpret_cast<uint4*>(&Ks[r * LDS + c * 8]) = kr[j];
            *reinterpret_cast<uint4*>(&Vs[r * LDS + c * 8]) = vr[j];
        }
        __syncthreads();
        if (k0 + KT <= kmax) ldg_kv(k0 + KT);   // in flight during this tile's MMAs
        if (k0 > warp_last) continue;          // the whole tile lies above the warp's last query
        if (WIN && k0 + KT - 1 <= warp_first - window) continue;   // the whole tile lies below the window of the warp's first query
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
        float mx0 = -INFINITY, mx1 = -INFINITY;
#pragma unroll
        for (int nt = 0; nt < NK; nt++) {
#pragma unroll
            for (int e = 0; e < 4; e++) {
                const int kp = k0 + nt * 8 + 2 * t4 + (e & 1);
                const int qp = (e >> 1) ? row1 : row0;
                float val = s[nt][e] * sl2;
                if (kp > qp || qp >= T || kp > kmax || (WIN && qp - kp >= window)) val = -INFINITY;
                s[nt][e] = val;
                if (e >> 1) mx1 = fmaxf(mx1, val); else mx0 = fmaxf(mx0, val);
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
                ldsm_x4_t(vb, smem_u32(&Vs[(j * 8 + v_row) * LDS + (nd4 * 4 + v_mat) * 8]));
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
    if (t4 == 0) {
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

// dK, dV (see flash_bwd.cu); with the window, the query tiles stop where the key tile's last key leaves every window
template <bool WIN>
__device__ __forceinline__ void dkdv(const half* q, const half* k, const half* v, const half* dO, const float* lse, const float* Dv, float* dk, float* dv, int T, int n_head, int n_kv, int window) {
    using namespace flashb;
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
    const float scale = 0.125f;
    const float sl2 = scale * 1.4426950408889634f;
    // last query tile that sees a key of this tile: the query k0 + 63 + window - 1
    const int qt_end = WIN ? min(T / 64, (k0 + 63 + window - 1) / 64 + 1) : T / 64;
    for (int hr = 0; hr < rep; hr++) {
        const int h = kvh * rep + hr;
        for (int qt = kt; qt < qt_end; qt++) {
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
            mma_a_bt(s, kf, Qs, lane);
            mma_a_bt(dp, vf, dOs, lane);
#pragma unroll
            for (int nt = 0; nt < 8; nt++) {
#pragma unroll
                for (int e = 0; e < 4; e++) {
                    const int col = nt * 8 + 2 * t4 + (e & 1);
                    const int kpos = k0 + warp * 16 + g + 8 * (e >> 1), qpos = q0 + col;
                    const bool vis = kpos <= qpos && (!WIN || qpos - kpos < window);
                    const float p = vis ? exp2f(s[nt][e] * sl2 - Ls[col]) : 0.f;
                    s[nt][e] = p;
                    dp[nt][e] = p * (dp[nt][e] - Ds[col]);
                }
            }
            mma_c_b(dvacc, s, dOs, lane);
            mma_c_b(dkacc, dp, Qs, lane);
        }
    }
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

// dQ (see flash_bwd.cu): longest query tiles first; with the window, the key tiles start at the first one inside the window of the tile's first query
template <bool WIN>
__device__ __forceinline__ void dq(const half* q, const half* k, const half* v, const half* dO, const float* lse, const float* Dv, float* dqo, int T, int n_head, int n_kv, int window) {
    using namespace flashb;
    __shared__ __align__(16) half tiles[2 * 64 * LDS];
    half* Qs = tiles;
    half* dOs = tiles + 64 * LDS;
    half* Ks = tiles;
    half* Vs = tiles + 64 * LDS;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t4 = lane & 3;
    const size_t b = blockIdx.z;
    const int h = blockIdx.y, qt = gridDim.x - 1 - blockIdx.x, q0 = qt * 64;
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
    const int kt_start = WIN ? max(0, q0 - window + 1) / 64 : 0;
    // register prefetch of the next K and V tiles (4 + 4 chunks of 16 bytes per thread), loaded while the current tiles are used
    uint4 kr[4], vr[4];
    auto ldg_kv = [&](int k0) {
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const int i = tid + 128 * j, r = i >> 3, c = i & 7;
            kr[j] = __ldg(reinterpret_cast<const uint4*>(k + (b * T + k0 + r) * kv_dim + kvh * 64) + c);
            vr[j] = __ldg(reinterpret_cast<const uint4*>(v + (b * T + k0 + r) * kv_dim + kvh * 64) + c);
        }
    };
    ldg_kv(kt_start * 64);
    for (int kt = kt_start; kt <= qt; kt++) {
        __syncthreads();   // everybody is done with the previous tiles (and with the Q / dO staging)
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const int i = tid + 128 * j, r = i >> 3, c = i & 7;
            *reinterpret_cast<uint4*>(&Ks[r * LDS + c * 8]) = kr[j];
            *reinterpret_cast<uint4*>(&Vs[r * LDS + c * 8]) = vr[j];
        }
        __syncthreads();
        if (kt < qt) ldg_kv((kt + 1) * 64);
        const int k0 = kt * 64;
        float s[8][4], dp[8][4];
#pragma unroll
        for (int i = 0; i < 8; i++)
#pragma unroll
            for (int e = 0; e < 4; e++) s[i][e] = dp[i][e] = 0.f;
        mma_a_bt(s, qf, Ks, lane);
        mma_a_bt(dp, dof, Vs, lane);
#pragma unroll
        for (int nt = 0; nt < 8; nt++) {
#pragma unroll
            for (int e = 0; e < 4; e++) {
                const int col = nt * 8 + 2 * t4 + (e & 1);
                const int kpos = k0 + col, qpos = q0 + warp * 16 + g + 8 * (e >> 1);
                const float Lr = (e >> 1) ? L1 : L0, Dr = (e >> 1) ? D1 : D0;
                const bool vis = kpos <= qpos && (!WIN || qpos - kpos < window);
                const float p = vis ? exp2f(s[nt][e] * sl2 - Lr) : 0.f;
                dp[nt][e] = p * (dp[nt][e] - Dr);
            }
        }
        mma_c_b(dqacc, dp, Ks, lane);
    }
#pragma unroll
    for (int nd = 0; nd < 8; nd++) {
#pragma unroll
        for (int hrow = 0; hrow < 2; hrow++) {
            const size_t row = b * T + q0 + warp * 16 + g + 8 * hrow;
            *reinterpret_cast<float2*>(dqo + row * q_dim + h * 64 + nd * 8 + 2 * t4) = make_float2(dqacc[nd][2 * hrow] * scale, dqacc[nd][2 * hrow + 1] * scale);
        }
    }
}


// ---- dS spill (dense causal only): dK dV writes the fp16 dS^T tiles it already has in registers, dQ is then ONE matmul per tile (dS K) instead of three (S, dP and dS K) ----------------------------------
// Layout of the spill buffer: per (sequence, query head) the 64 x 64 tiles (qt, kt), kt <= qt, in the order qt * (qt + 1) / 2 + kt; a tile is 64 rows (keys) x 64 columns (queries) of fp16, row-major
// [key][query] (8 KB). The values are the ones dK dV converts for its own dK product (dS^T = P^T * (dP^T - D), rounded to fp16), the same dS the dQ kernel above rounds: dQ is bit-identical.
__device__ __forceinline__ void dkdv_ds(const half* q, const half* k, const half* v, const half* dO, const float* lse, const float* Dv, float* dk, float* dv, half* dsb, int T, int n_head, int n_kv) {
    using namespace flashb;
    __shared__ __align__(16) half tiles[2 * 64 * LDS];
    __shared__ __align__(16) half stage[64 * LDS];
    __shared__ float Ls[64], Ds[64];
    half* Ks = tiles;
    half* Vs = tiles + 64 * LDS;
    half* Qs = tiles;
    half* dOs = tiles + 64 * LDS;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t4 = lane & 3;
    // launch order = longest blocks first over the whole grid (all sequences and kv heads of key tile 0, then of tile 1, ...): with the grid's own order the second sequence's longest blocks started last
    // (one sequence: the grid's own order, key tile fastest, measured faster there)
    const int per = gridDim.y * gridDim.z;
    const int lin = blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z);
    const bool multi = gridDim.z > 1;
    const int kt = multi ? lin / per : blockIdx.x, rest = lin - (lin / per) * per;
    const int kvh = multi ? rest % gridDim.y : blockIdx.y, k0 = kt * 64;
    const size_t b = multi ? rest / gridDim.y : blockIdx.z;
    const int q_dim = n_head * 64, kv_dim = n_kv * 64, rep = n_head / n_kv;
    const int nt_all = T / 64;
    const size_t ntri = (size_t)nt_all * (nt_all + 1) / 2;
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
    const float scale = 0.125f;
    const float sl2 = scale * 1.4426950408889634f;
    for (int hr = 0; hr < rep; hr++) {
        const int h = kvh * rep + hr;
        half* tile_h = dsb + ((b * n_head + h) * ntri) * 4096;
        for (int qt = kt; qt < nt_all; qt++) {
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
            mma_a_bt(s, kf, Qs, lane);
            mma_a_bt(dp, vf, dOs, lane);
#pragma unroll
            for (int nt = 0; nt < 8; nt++) {
#pragma unroll
                for (int e = 0; e < 4; e++) {
                    const int col = nt * 8 + 2 * t4 + (e & 1);
                    const int kpos = k0 + warp * 16 + g + 8 * (e >> 1), qpos = q0 + col;
                    const bool vis = kpos <= qpos;
                    const float p = vis ? exp2f(s[nt][e] * sl2 - Ls[col]) : 0.f;
                    s[nt][e] = p;
                    dp[nt][e] = p * (dp[nt][e] - Ds[col]);
                }
            }
            // spill dS^T: the warp's 16 key rows through its own part of the staging tile, then 16-byte stores of whole rows
#pragma unroll
            for (int nt = 0; nt < 8; nt++) {
                *reinterpret_cast<uint32_t*>(&stage[(warp * 16 + g) * LDS + nt * 8 + 2 * t4]) = pack_half2(dp[nt][0], dp[nt][1]);
                *reinterpret_cast<uint32_t*>(&stage[(warp * 16 + g + 8) * LDS + nt * 8 + 2 * t4]) = pack_half2(dp[nt][2], dp[nt][3]);
            }
            mma_c_b(dvacc, s, dOs, lane);
            mma_c_b(dkacc, dp, Qs, lane);
            __syncwarp();
            half* dst = tile_h + ((size_t)qt * (qt + 1) / 2 + kt) * 4096 + warp * 16 * 64;
#pragma unroll
            for (int j = 0; j < 4; j++) {
                const int c = lane + 32 * j, r = c >> 3, ch = c & 7;
                *reinterpret_cast<uint4*>(dst + r * 64 + ch * 8) = *reinterpret_cast<const uint4*>(&stage[(warp * 16 + r) * LDS + ch * 8]);
            }
            __syncwarp();
        }
    }
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

// dQ from the spilled dS: one block per (64-query tile, head, sequence), longest first; dQ += dS K * scale, the key tiles in ascending order like dq() above
__device__ __forceinline__ void dq_ds(const half* k, const half* dsb, float* dqo, int T, int n_head, int n_kv) {
    using namespace flashb;
    __shared__ __align__(16) half Ks[64 * LDS];
    __shared__ __align__(16) half Ss[64 * LDS];   // dS^T tile [key][query]
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    // launch order = longest blocks first over the whole grid (see dkdv_ds)
    const int per = gridDim.y * gridDim.z;
    const int lin = blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z);
    const int qt = gridDim.x - 1 - lin / per, rest = lin % per;
    const int h = rest % gridDim.y, q0 = qt * 64;
    const size_t b = rest / gridDim.y;
    const int q_dim = n_head * 64, kv_dim = n_kv * 64, kvh = h / (n_head / n_kv);
    const int nt_all = T / 64;
    const size_t ntri = (size_t)nt_all * (nt_all + 1) / 2;
    const half* tiles_q = dsb + ((b * n_head + h) * ntri + (size_t)qt * (qt + 1) / 2) * 4096;
    float dqacc[8][4];
#pragma unroll
    for (int i = 0; i < 8; i++)
#pragma unroll
        for (int e = 0; e < 4; e++) dqacc[i][e] = 0.f;
    const float scale = 0.125f;
    // register prefetch of the next K and dS tiles (4 + 4 chunks of 16 bytes per thread)
    uint4 kr[4], sr[4];
    auto ldg_tiles = [&](int kt) {
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const int i = tid + 128 * j, r = i >> 3, c = i & 7;
            kr[j] = __ldg(reinterpret_cast<const uint4*>(k + (b * T + kt * 64 + r) * kv_dim + kvh * 64) + c);
            sr[j] = __ldg(reinterpret_cast<const uint4*>(tiles_q + (size_t)kt * 4096 + r * 64) + c);
        }
    };
    ldg_tiles(0);
    // A fragments of dS (rows = the warp's 16 queries, k = keys) from the [key][query] tile with ldmatrix.trans: matrix m of the x4 load = (key block 2 s2 + (m >> 1), query block m & 1)
    const int am = lane >> 3, ar = lane & 7;
    const int a_off = (((am >> 1) * 8 + ar) * LDS) + warp * 16 + (am & 1) * 8;
    const int v_row = lane & 7, v_mat = lane >> 3;
    for (int kt = 0; kt <= qt; kt++) {
        __syncthreads();
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const int i = tid + 128 * j, r = i >> 3, c = i & 7;
            *reinterpret_cast<uint4*>(&Ks[r * LDS + c * 8]) = kr[j];
            *reinterpret_cast<uint4*>(&Ss[r * LDS + c * 8]) = sr[j];
        }
        __syncthreads();
        if (kt < qt) ldg_tiles(kt + 1);
#pragma unroll
        for (int s2 = 0; s2 < 4; s2++) {
            uint32_t a[4];
            ldsm_x4_t(a, smem_u32(&Ss[s2 * 16 * LDS + a_off]));
#pragma unroll
            for (int hh = 0; hh < 2; hh++) {
                const int nt = 2 * s2 + hh;
#pragma unroll
                for (int nd4 = 0; nd4 < 2; nd4++) {
                    uint32_t bb[4];
                    ldsm_x4_t(bb, smem_u32(&Ks[(nt * 8 + v_row) * LDS + (nd4 * 4 + v_mat) * 8]));
#pragma unroll
                    for (int u = 0; u < 4; u++) mma_k8(dqacc[nd4 * 4 + u], a[2 * hh], a[2 * hh + 1], bb[u]);
                }
            }
        }
    }
    const int g = lane >> 2, t4 = lane & 3;
#pragma unroll
    for (int nd = 0; nd < 8; nd++) {
#pragma unroll
        for (int hrow = 0; hrow < 2; hrow++) {
            const size_t row = b * T + q0 + warp * 16 + g + 8 * hrow;
            *reinterpret_cast<float2*>(dqo + row * q_dim + h * 64 + nd * 8 + 2 * t4) = make_float2(dqacc[nd][2 * hrow] * scale, dqacc[nd][2 * hrow + 1] * scale);
        }
    }
}

}  // namespace flasht

// forward: (T / 64, n_head, batch) blocks of 128 threads; `window` is ignored by the dense kernel
extern "C" __global__ void __launch_bounds__(128, 3) flash_fwd_t64(const half* q, const half* k, const half* v, half* out, float* lse, int T, int n_head, int n_kv, int window) {
    // window != 0 (the dense kernel has no window): launch order = longest query tiles first over the whole grid (all heads and sequences of the last tile, then of the one before, ...);
    // 0: the grid's own order
    const int per = gridDim.y * gridDim.z;
    const int lin = blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z);
    const bool lpt = window != 0;
    const int bx = lpt ? lin / per : blockIdx.x, rest = lin - (lin / per) * per;
    const int by = lpt ? rest % gridDim.y : blockIdx.y;
    const size_t z = lpt ? rest / gridDim.y : blockIdx.z;
    flasht::fwd<false>(q + z * T * n_head * 64, k + z * T * n_kv * 64, v + z * T * n_kv * 64, out + z * T * n_head * 64, lse + z * T * n_head, T, n_head, n_kv, window, bx, by);
}
extern "C" __global__ void __launch_bounds__(128, 3) flash_fwd_t64_win(const half* q, const half* k, const half* v, half* out, float* lse, int T, int n_head, int n_kv, int window) {
    const size_t z = blockIdx.z;
    flasht::fwd<true>(q + z * T * n_head * 64, k + z * T * n_kv * 64, v + z * T * n_kv * 64, out + z * T * n_head * 64, lse + z * T * n_head, T, n_head, n_kv, window, blockIdx.x, blockIdx.y);
}
// backward with a window (the dense case keeps flash_bwd_dkdv_64; dQ in longest-first order: flash_bwd_dq_t64)
extern "C" __global__ void __launch_bounds__(128, 2) flash_bwd_dkdv_64_win(const half* q, const half* k, const half* v, const half* dO, const float* lse, const float* Dv, float* dk, float* dv, int T, int n_head,
                                                                           int n_kv, int window) {
    flasht::dkdv<true>(q, k, v, dO, lse, Dv, dk, dv, T, n_head, n_kv, window);
}
extern "C" __global__ void __launch_bounds__(128, 2) flash_bwd_dq_t64(const half* q, const half* k, const half* v, const half* dO, const float* lse, const float* Dv, float* dq, int T, int n_head, int n_kv, int window) {
    flasht::dq<false>(q, k, v, dO, lse, Dv, dq, T, n_head, n_kv, window);
}
extern "C" __global__ void __launch_bounds__(128, 2) flash_bwd_dq_64_win(const half* q, const half* k, const half* v, const half* dO, const float* lse, const float* Dv, float* dq, int T, int n_head, int n_kv, int window) {
    flasht::dq<true>(q, k, v, dO, lse, Dv, dq, T, n_head, n_kv, window);
}

// dense causal backward with the dS spill: dK dV writes the dS^T tiles, dQ reads them (FORGE_ATTN_DS in the trainer)
extern "C" __global__ void __launch_bounds__(128, 2) flash_bwd_dkdv_ds_64(const half* q, const half* k, const half* v, const half* dO, const float* lse, const float* Dv, float* dk, float* dv, half* dsb, int T,
                                                                           int n_head, int n_kv) {
    flasht::dkdv_ds(q, k, v, dO, lse, Dv, dk, dv, dsb, T, n_head, n_kv);
}
extern "C" __global__ void __launch_bounds__(128, 3) flash_bwd_dq_ds_64(const half* k, const half* dsb, float* dq, int T, int n_head, int n_kv) {
    flasht::dq_ds(k, dsb, dq, T, n_head, n_kv);
}
