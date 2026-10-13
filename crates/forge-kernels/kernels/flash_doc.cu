// Document masking for the training attention (recipe-opt): copies of the causal flash-attention forward (flash.cu) and backward (flash_bwd.cu) kernels in which a query also ignores every key before
// the start of its own document. ds[row] (int32, one per row of the [B*T] batch, position inside the sequence) is the first position of the row's document; it never decreases along a sequence
// (the host computes it from the end-of-text tokens: an end-of-text token opens a new document). Key tiles that lie entirely before a block's (or warp's) documents are skipped.
// The original kernels are untouched (the kernel-opt branch works on them); the mask is two comparisons, to be ported into whatever kernels win.

#include <cuda_fp16.h>
#include <stdint.h>

namespace flash_doc {

using flash::ldsm_x4;
using flash::ldsm_x4_t;
using flash::mma_k8;
using flash::pack_half2;
using flash::smem_u32;

// flash::flash_fwd_impl with pos0 = 0 and the document mask
template <int HD, int KT>
__device__ __forceinline__ void fwd_impl(const half* __restrict__ q, const half* __restrict__ k_cache, const half* __restrict__ v_cache, half* __restrict__ out, float* __restrict__ lse,
                                         const int* __restrict__ ds, int T, int n_head, int n_kv) {
    constexpr int LDS = HD + 8;
    constexpr int BQ = 64;
    constexpr int KS = HD / 8;
    constexpr int NK = KT / 8;
    constexpr int ND = HD / 8;
    __shared__ __align__(16) half Qs[BQ * LDS];
    __shared__ __align__(16) half Ks[KT * LDS];
    __shared__ __align__(16) half Vs[KT * LDS];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int g = lane >> 2, t4 = lane & 3;
    const int hh = blockIdx.y, kvh = hh / (n_head / n_kv);
    const int q_dim = n_head * HD, kv_dim = n_kv * HD;
    const int q0 = blockIdx.x * BQ;
    for (int i = tid; i < BQ * HD / 8; i += blockDim.x) {
        const int r = i / (HD / 8), c = i % (HD / 8);
        uint4 v = make_uint4(0u, 0u, 0u, 0u);
        if (q0 + r < T) v = __ldg(reinterpret_cast<const uint4*>(q + (size_t)(q0 + r) * q_dim + hh * HD) + c);
        *reinterpret_cast<uint4*>(&Qs[r * LDS + c * 8]) = v;
    }
    __syncthreads();
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
    const int row0 = q0 + warp * 16 + g, row1 = row0 + 8;
    const int last_row = min(T - 1, q0 + BQ - 1);
    const int kmax = last_row;
    const int warp_first = min(T - 1, q0 + warp * 16), warp_last = min(T - 1, q0 + warp * 16 + 15);
    // document starts of this thread's rows, of the warp's first row and of the block's first row (the smallest ones: ds never decreases)
    const int ds0 = ds[min(row0, T - 1)], ds1 = ds[min(row1, T - 1)];
    const int warp_ds = ds[warp_first];
    const int k_begin = (ds[q0] / KT) * KT;
    const int b_row = (lane & 7) + (lane >> 4) * 8, b_col = ((lane >> 3) & 1) * 8;
    const int v_row = lane & 7, v_mat = lane >> 3;

    for (int k0 = k_begin; k0 <= kmax; k0 += KT) {
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
        // the tile lies above the warp's last query or before its first document: skip (every warp still reaches the barriers)
        if (k0 > warp_last || k0 + KT <= warp_ds) continue;
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
                const int dq = (e >> 1) ? ds1 : ds0;
                float v = s[nt][e] * sl2;
                if (kp > qp || kp < dq || qp >= T || kp > kmax) v = -INFINITY;
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

}  // namespace flash_doc

// training forward with the document mask: gridDim.z sequences of T tokens ([B*T][dim]); ds: [B*T] document starts (positions inside each sequence)
extern "C" __global__ void __launch_bounds__(128) flash_fwd_train_doc_64(const half* q, const half* k, const half* v, half* out, float* lse, const int* ds, int T, int n_head, int n_kv) {
    const size_t z = blockIdx.z;
    flash_doc::fwd_impl<64, 64>(q + z * T * n_head * 64, k + z * T * n_kv * 64, v + z * T * n_kv * 64, out + z * T * n_head * 64, lse + z * T * n_head, ds + z * T, T, n_head, n_kv);
}
extern "C" __global__ void __launch_bounds__(128) flash_fwd_train_doc_128(const half* q, const half* k, const half* v, half* out, float* lse, const int* ds, int T, int n_head, int n_kv) {
    const size_t z = blockIdx.z;
    flash_doc::fwd_impl<128, 32>(q + z * T * n_head * 128, k + z * T * n_kv * 128, v + z * T * n_kv * 128, out + z * T * n_head * 128, lse + z * T * n_head, ds + z * T, T, n_head, n_kv);
}

// flash_bwd_dkdv_64 with the document mask: a query sees a key only if the key is not after it and not before the query's document start; the query tiles stop at the first one whose first query starts
// its document after this key tile.
extern "C" __global__ void __launch_bounds__(128, 2) flash_bwd_dkdv_doc_64(const half* q, const half* k, const half* v, const half* dO, const float* lse, const float* Dv, const int* ds, float* dk, float* dv,
                                                                           int T, int n_head, int n_kv) {
    using namespace flashb;
    __shared__ __align__(16) half tiles[2 * 64 * LDS];
    __shared__ float Ls[64], Ds[64];
    __shared__ int Ss[64];
    half* Ks = tiles;
    half* Vs = tiles + 64 * LDS;
    half* Qs = tiles;
    half* dOs = tiles + 64 * LDS;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t4 = lane & 3;
    const size_t b = blockIdx.z;
    const int kvh = blockIdx.y, kt = blockIdx.x, k0 = kt * 64;
    const int q_dim = n_head * 64, kv_dim = n_kv * 64, rep = n_head / n_kv;
    const int* dsb = ds + b * T;
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
        for (int qt = kt; qt < T / 64; qt++) {
            const int q0 = qt * 64;
            if (dsb[q0] > k0 + 63) break;   // every query from here on starts its document after this key tile (the same value for the whole block)
            __syncthreads();
            load_tile(q + (b * T + q0) * q_dim + h * 64, q_dim, Qs, tid);
            load_tile(dO + (b * T + q0) * q_dim + h * 64, q_dim, dOs, tid);
            if (tid < 64) {
                Ls[tid] = lse[(b * T + q0 + tid) * n_head + h];
                Ds[tid] = Dv[(b * T + q0 + tid) * n_head + h];
                Ss[tid] = dsb[q0 + tid];
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
                    const float p = (kpos <= qpos && kpos >= Ss[col]) ? exp2f(s[nt][e] * sl2 - Ls[col]) : 0.f;
                    s[nt][e] = p;
                    dp[nt][e] = p * (dp[nt][e] - Ds[col]);
                }
            }
            mma_c_b(dvacc, s, dOs, lane);
            mma_c_b(dkacc, dp, Qs, lane);
        }
        __syncthreads();   // the next head reloads the shared tiles
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

// flash_bwd_dq_64 with the document mask: the key tiles start at the tile of the block's first document start.
extern "C" __global__ void __launch_bounds__(128, 2) flash_bwd_dq_doc_64(const half* q, const half* k, const half* v, const half* dO, const float* lse, const float* Dv, const int* ds, float* dq, int T, int n_head,
                                                                         int n_kv) {
    using namespace flashb;
    __shared__ __align__(16) half tiles[2 * 64 * LDS];
    half* Qs = tiles;
    half* dOs = tiles + 64 * LDS;
    half* Ks = tiles;
    half* Vs = tiles + 64 * LDS;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t4 = lane & 3;
    const size_t b = blockIdx.z;
    const int h = blockIdx.y, qt = blockIdx.x, q0 = qt * 64;
    const int q_dim = n_head * 64, kv_dim = n_kv * 64, kvh = h / (n_head / n_kv);
    const int* dsb = ds + b * T;
    load_tile(q + (b * T + q0) * q_dim + h * 64, q_dim, Qs, tid);
    load_tile(dO + (b * T + q0) * q_dim + h * 64, q_dim, dOs, tid);
    __syncthreads();
    uint32_t qf[8][2], dof[8][2];
    load_a_frags(Qs, warp, lane, qf);
    load_a_frags(dOs, warp, lane, dof);
    const int r0 = warp * 16 + g, r1 = r0 + 8;
    const float L0 = lse[(b * T + q0 + r0) * n_head + h], L1 = lse[(b * T + q0 + r1) * n_head + h];
    const float D0 = Dv[(b * T + q0 + r0) * n_head + h], D1 = Dv[(b * T + q0 + r1) * n_head + h];
    const int S0 = dsb[q0 + r0], S1 = dsb[q0 + r1];
    float dqacc[8][4];
#pragma unroll
    for (int i = 0; i < 8; i++)
#pragma unroll
        for (int e = 0; e < 4; e++) dqacc[i][e] = 0.f;
    const float scale = 0.125f, sl2 = scale * 1.4426950408889634f;
    for (int kt = dsb[q0] / 64; kt <= qt; kt++) {
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
        mma_a_bt(s, qf, Ks, lane);
        mma_a_bt(dp, dof, Vs, lane);
#pragma unroll
        for (int nt = 0; nt < 8; nt++) {
#pragma unroll
            for (int e = 0; e < 4; e++) {
                const int col = nt * 8 + 2 * t4 + (e & 1);
                const int kpos = k0 + col, qpos = q0 + warp * 16 + g + 8 * (e >> 1);
                const float Lr = (e >> 1) ? L1 : L0, Dr = (e >> 1) ? D1 : D0;
                const int Sr = (e >> 1) ? S1 : S0;
                const float p = (kpos <= qpos && kpos >= Sr) ? exp2f(s[nt][e] * sl2 - Lr) : 0.f;
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
            *reinterpret_cast<float2*>(dq + row * q_dim + h * 64 + nd * 8 + 2 * t4) = make_float2(dqacc[nd][2 * hrow] * scale, dqacc[nd][2 * hrow + 1] * scale);
        }
    }
}
