// Causal flash-attention forward on the Ampere tensor-core instructions (sm_80 and newer: GeForce RTX 30 / 40, A100, H100): the kernels of flash.cu and flash256.cu rebuilt the way gemm_s80.cu rebuilds the GEMMs.
//  - m16n8k16 MMAs (the k8 form of the Turing kernels runs at half the tensor-core rate on Ampere);
//  - the K and V tiles stream through a double-buffered cp.async pipeline (global -> shared without passing through registers; the next tile is in flight while the current one is consumed);
//  - fp32 accumulators throughout: an fp16-accumulate form of the scores S = Q K^T (flushed into fp32 after every k16 step, twice the MMA rate on GeForce cards) was built and measured and dropped: 3 to 4 percent
//    more prompt speed for a KL of 4e-4 (Qwen3.5) to 1.2e-3 (a dense 28-layer model) against the old kernels, while the fp32 forms are bit-identical to them.
// Same arguments and result (up to rounding) as flash_fwd_64 / flash_fwd_128 (grid ((T + 63) / 64, n_head, 1), 128 threads) and flash_fwd_256g (see below). The blocks run from the last query rows to the first: they have
// the most keys to read, so the light ones fill the tail of the grid. Below sm_80 the kernels are empty (the host picks them only on devices of compute capability 8 and up).
// Qwen3.5's gated head width 256 (flash_fwd_256g_a80): the accumulators of 64 rows x 256 output dimensions do not fit one warp's registers, and the old kernel gave each half of the dimensions its own block, each of
// which computed the whole S = Q K^T (1.5x the tensor work). Here a block of 8 warps is four warp PAIRS: a pair shares 16 query rows, each warp computes S for 16 of the 32 keys of a tile, the halves meet in shared
// memory behind a 64-thread named barrier, and each warp then runs the softmax over all 32 keys and P V for its own 128 output dimensions. The K and V tiles (32 keys x 256 dimensions, 16 KB each, two buffers)
// are not padded but XOR-swizzled (the 16-byte chunk c of row r sits at c ^ (r & 7)): ldmatrix and cp.async stay conflict free, and 32-key tiles are as large as the 64 KB of dynamic shared memory allow. With fp32
// accumulators the result is bit-identical to flash_fwd_256g (same online-softmax tile of 32 keys, same summation order).

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
}  // namespace a80

// HQ: head width; HO: output dimensions of one block (HQ, or HQ / 2 when blockIdx.z picks the half: S is then computed by both halves); KT: keys per tile; GATE: sigmoid gate on the output
// (gate[row * gate_tok + head * gate_head + dim], the gated attention of Qwen3.5)
template <int HQ, int HO, int KT, bool GATE>
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
                    mma_k16(s[2 * j2], qf[s2], kb[0], kb[1]);
                    mma_k16(s[2 * j2 + 1], qf[s2], kb[2], kb[3]);
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

// ---- Qwen3.5's gated attention (head width 256): warp pairs, no recomputation of S ------------------------------------------------------------------------------------------------------------------------------
__device__ __forceinline__ void bar_pair(int id) { asm volatile("bar.sync %0, 64;\n" ::"r"(id) : "memory"); }
// halves of row r, 16-byte chunk c (0 .. 31) of a 512-byte row: the chunk is permuted by XOR with (r & 7)
__device__ __forceinline__ int swz256(int r, int c) { return r * 256 + ((c ^ (r & 7)) << 3); }

// grid ((T + 63) / 64, n_head, 1), 256 threads, 65,536 bytes of dynamic shared memory; gate[row * gate_tok + head * gate_head + dim]
__device__ __forceinline__ void flash_fwd_pair256_impl(const half* __restrict__ q, const half* __restrict__ k_cache, const half* __restrict__ v_cache, half* __restrict__ out, int T, int pos0, int n_head,
                                                       int n_kv, const half* __restrict__ gate, int gate_tok, int gate_head) {
    using namespace a80;
    constexpr int HQ = 256, HO = 128, BQ = 64, NT = 256, KT = 32;
    constexpr int ND = HO / 8, NK = KT / 8;
    constexpr int KV_ELEMS = KT * 256 + KT * 256;                    // the K and the V tile of a buffer, 256 halves per row
    extern __shared__ __align__(16) half sm[];
    __shared__ float xs[4][16][KT + 1];                              // S of a pair for the 32 keys of a tile: [pair][row][key]
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int pr = warp >> 1, half_ = warp & 1;                      // the pair, and which half of the keys of a tile / of the output dimensions
    const int g = lane >> 2, t4 = lane & 3;
    const int hh = blockIdx.y, kvh = hh / (n_head / n_kv);
    const int vo = half_ * HO;
    const int q_dim = n_head * HQ, kv_dim = n_kv * HQ;
    const int q0 = (gridDim.x - 1 - blockIdx.x) * BQ;
    for (int i = tid; i < BQ * HQ / 8; i += NT) {
        const int r = i / (HQ / 8), c = i % (HQ / 8);
        uint4 v = make_uint4(0u, 0u, 0u, 0u);
        if (q0 + r < T) v = __ldg(reinterpret_cast<const uint4*>(q + (size_t)(q0 + r) * q_dim + hh * HQ) + c);
        *reinterpret_cast<uint4*>(&sm[swz256(r, c)]) = v;
    }
    __syncthreads();
    uint32_t qf[HQ / 16][4];
    {
        const int a_row = (lane & 7) + ((lane >> 3) & 1) * 8, a_chunk = lane >> 4;
#pragma unroll
        for (int s2 = 0; s2 < HQ / 16; s2++) ldsm_x4(qf[s2], smem_u32(&sm[swz256(pr * 16 + a_row, 2 * s2 + a_chunk)]));
    }
    __syncthreads();
    float o[ND][4];
#pragma unroll
    for (int i = 0; i < ND; i++) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.f;
    float m[2] = {-INFINITY, -INFINITY}, l[2] = {0.f, 0.f};
    const float sl2 = rsqrtf((float)HQ) * 1.4426950408889634f;
    const int row0 = q0 + pr * 16 + g, row1 = row0 + 8;
    const int qpos0 = pos0 + row0, qpos1 = pos0 + row1;
    const int last_row = min(T - 1, q0 + BQ - 1);
    const int kmax = pos0 + last_row;
    const int pair_last = pos0 + min(T - 1, q0 + pr * 16 + 15);
    const int b_row = (lane & 7) + (lane >> 4) * 8, b_chunk = (lane >> 3) & 1;
    const int v_row = lane & 7, v_mat = lane >> 3;
    const int ntiles = kmax / KT + 1;
    const int free_below = (q0 + BQ <= T) ? pos0 + q0 : -1;   // tiles that end at or before the first query position of a block of existing rows need no mask
    auto load_tile = [&](int tile, int buf) {
        half* Ks = sm + buf * KV_ELEMS;
        half* Vs = Ks + KT * 256;
        const int k0 = tile * KT;
        for (int i = tid; i < KT * 32; i += NT) {
            const int r = i >> 5, c = i & 31;
            const int kp = k0 + r;
            const bool ok = kp <= kmax;
            const size_t off = (size_t)(ok ? kp : 0) * kv_dim + kvh * HQ + c * 8;
            cp16z(smem_u32(&Ks[swz256(r, c)]), k_cache + off, ok);
            cp16z(smem_u32(&Vs[swz256(r, c)]), v_cache + off, ok);
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
        const half* Vs = Ks + KT * 256;
        if (k0 <= pair_last) {
            // S of the keys k0 + 16 half_ .. + 15 (two n8 tiles) for the 16 rows of the pair
            float s[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
#pragma unroll
            for (int s2 = 0; s2 < HQ / 16; s2++) {
                uint32_t kb[4];
                ldsm_x4(kb, smem_u32(&Ks[swz256(half_ * 16 + b_row, 2 * s2 + b_chunk)]));
                mma_k16(s[0], qf[s2], kb[0], kb[1]);
                mma_k16(s[1], qf[s2], kb[2], kb[3]);
            }
            // thread (g, t4) holds (row g, keys 2 t4, 2 t4 + 1) and (row g + 8, ...) of each n8 tile of its half
#pragma unroll
            for (int nt = 0; nt < 2; nt++)
#pragma unroll
                for (int e = 0; e < 4; e++) xs[pr][g + 8 * (e >> 1)][half_ * 16 + nt * 8 + 2 * t4 + (e & 1)] = s[nt][e];
            bar_pair(1 + pr);
            float sv[NK][4];
#pragma unroll
            for (int nt = 0; nt < NK; nt++)
#pragma unroll
                for (int e = 0; e < 4; e++) sv[nt][e] = ((nt >> 1) == half_) ? s[nt & 1][e] : xs[pr][g + 8 * (e >> 1)][nt * 8 + 2 * t4 + (e & 1)];
            float mx0 = -INFINITY, mx1 = -INFINITY;
            if (k0 + KT - 1 <= free_below) {
#pragma unroll
                for (int nt = 0; nt < NK; nt++) {
                    sv[nt][0] *= sl2;
                    sv[nt][1] *= sl2;
                    sv[nt][2] *= sl2;
                    sv[nt][3] *= sl2;
                    mx0 = fmaxf(mx0, fmaxf(sv[nt][0], sv[nt][1]));
                    mx1 = fmaxf(mx1, fmaxf(sv[nt][2], sv[nt][3]));
                }
            } else {
#pragma unroll
                for (int nt = 0; nt < NK; nt++) {
#pragma unroll
                    for (int e = 0; e < 4; e++) {
                        const int kp = k0 + nt * 8 + 2 * t4 + (e & 1);
                        const int qp = (e >> 1) ? qpos1 : qpos0;
                        const int rowv = (e >> 1) ? row1 : row0;
                        float v = sv[nt][e] * sl2;
                        if (kp > qp || rowv >= T || kp > kmax) v = -INFINITY;
                        sv[nt][e] = v;
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
                const float p0 = exp2f(sv[nt][0] - ms0), p1 = exp2f(sv[nt][1] - ms0), p2 = exp2f(sv[nt][2] - ms1), p3 = exp2f(sv[nt][3] - ms1);
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
                    ldsm_x4_t(vl, smem_u32(&Vs[swz256(j * 16 + v_row, (vo >> 3) + nd4 * 4 + v_mat)]));
                    ldsm_x4_t(vh, smem_u32(&Vs[swz256(j * 16 + 8 + v_row, (vo >> 3) + nd4 * 4 + v_mat)]));
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
        const int col = hh * HQ + d;
        if (row0 < T) *reinterpret_cast<half2*>(out + (size_t)row0 * q_dim + col) = __floats2half2_rn(v00, v01);
        if (row1 < T) *reinterpret_cast<half2*>(out + (size_t)row1 * q_dim + col) = __floats2half2_rn(v10, v11);
    }
}

#endif  // __CUDA_ARCH__ >= 800

}  // namespace flash

#if __CUDA_ARCH__ >= 800
#define FA80_BODY(HQ, HO, KT, GATE) flash::flash_fwd_a80_impl<HQ, HO, KT, GATE>(q, k_cache, v_cache, out, T, pos0, n_head, n_kv, gate, gate_tok, gate_head);
#else
#define FA80_BODY(HQ, HO, KT, GATE)
#endif
// Qwen3.5 gated attention (head width 256): grid ((T + 63) / 64, n_head, 1), 256 threads, 65,536 bytes of dynamic shared memory (the host sets the attribute)
#if __CUDA_ARCH__ >= 800
#define FA80P_BODY flash::flash_fwd_pair256_impl(q, k_cache, v_cache, out, T, pos0, n_head, n_kv, gate, gate_tok, gate_head);
#else
#define FA80P_BODY
#endif
extern "C" __global__ void __launch_bounds__(256) flash_fwd_256g_a80(const half* q, const half* k_cache, const half* v_cache, half* out, int T, int pos0, int n_head, int n_kv, const half* gate, int gate_tok,
                                                                    int gate_head) {
    FA80P_BODY
}
// dense models: the arguments and the grid (ceil(T / 64), n_head, 1) of flash_fwd_64 / flash_fwd_128
#define FA80_DENSE(NAME, HQ, KT)                                                                                                                                                              \
    extern "C" __global__ void __launch_bounds__(128) NAME(const half* q, const half* k_cache, const half* v_cache, half* out, int T, int pos0, int n_head, int n_kv) {                              \
        const half* gate = nullptr;                                                                                                                                                                 \
        const int gate_tok = 0, gate_head = 0;                                                                                                                                                      \
        FA80_BODY(HQ, HQ, KT, false)                                                                                                                                                          \
    }
FA80_DENSE(flash_fwd_64_a80, 64, 32)
FA80_DENSE(flash_fwd_128_a80, 128, 32)
