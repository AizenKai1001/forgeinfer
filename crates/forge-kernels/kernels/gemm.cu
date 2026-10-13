// Tensor-core GEMM for sm_75 (Turing): C[M][N] (op)= alpha * sum_k A(m,k) * B(k,n) with fp16 inputs and fp32 accumulation, in the three layouts a linear layer needs:
//
//   NT  A stored [M][K] (K contiguous), B stored [N][K]   forward  y = x W^T          (W is [out][in])
//   NN  A stored [M][K],                B stored [K][N]   data gradient  dx = dy W
//   TN  A stored [K][M],                B stored [K][N]   weight gradient  dW = dy^T x  (reduction over the tokens)
//
//  - block tile 128 x 128, 8 warps (2 x 4), warp tile 64 x 32, raw PTX mma.sync.m16n8k8 fed by ldmatrix; operands stored with the reduction index as the row are read with ldmatrix.trans.
//  - there is no cp.async on Turing: tiles go global -> registers (LDG.128) -> shared memory (STS.128).
//  - DEFAULT kernels (gemm3, BK = 16): the registers loaded in iteration k hold tile k + 2; they are stored to shared memory at the START of iteration k + 1, a whole iteration of MMAs after the
//    load was issued (the compiler used to place the store right behind the load it depends on, so every iteration paid the full global-memory latency). Load pointers advance by a constant,
//    the zero-fill predicates are hoisted, row-major tiles use a 32-byte row with an XOR swizzle of the two 16-byte chunks (conflict-free for both STS.128 and ldmatrix, no padding: 4 KB per
//    tile), two shared buffers, one __syncthreads per iteration. 16 KB of shared memory and <= 128 registers per thread: two blocks share an SM, so one block's barrier / prologue /
//    epilogue stalls are filled by the other block's MMAs.
//  - `_b16` (the first two-blocks-per-SM kernel, padded 48-byte rows, store right behind the load) and `_b32` (BK = 32, one block per SM) are kept for A/B timing.
//  - M and N arbitrary for the row-major operands (zero-filled loads, predicated stores); the k-major operands need M (resp. N) to be a multiple of 8; K must be a multiple of the BK in use.
// Epilogues (EPI): 0 store fp16, 1 store fp32, 2 add into fp32 (residual stream / gradient accumulation), 3 store fp16 of alpha * acc + beta * R (R an fp16 matrix with row stride ldr: the
// Newton-Schulz steps of the Muon optimizer, B = b A + c A A and X' = a X + B X), 4 store fp32 of alpha * acc + R with R an fp32 matrix (the residual stream: C = R + A B, no copy + add),
// 5 the ReLU^2 MLP's up projection: C = u = fp16(alpha * acc) and D = relu(u)^2 (fp16, saturating at 65000: bit-identical to gemm EPI 0 followed by relu2_kernel; D has row stride ldr),
// 6 its backward: C = fp16(fp16(alpha * acc) * 2 relu(u)) with u the fp16 matrix R (bit-identical to EPI 0 followed by relu2_bwd_kernel).
// Warps whose whole 64 x 32 warp tile lies outside the matrix (the half-empty last tile of a 576-wide dimension) skip their shared-memory reads and MMAs: exact (their results are never stored).
// Split-K (EPI 2 only): with gridDim.z > 1 block z reduces the slice [z K / gridDim.z, (z + 1) K / gridDim.z) (K / gridDim.z a multiple of 16) and adds its tile into C with atomics: the weight-gradient
// GEMMs have few output tiles (a 576 x 576 matrix is 25 tiles for 80 block slots on a T4) and a long reduction over the tokens. The summation order of the slices is then not fixed (fp32 rounding only).

#include <cuda_fp16.h>
#include <stdint.h>

namespace gemm {

constexpr int BM = 128, BN = 128;
constexpr int LDM = 136;   // row stride (halves) of tiles whose rows run along K and whose columns are M/N: BK x 136 (272 B: conflict free for ldmatrix.trans)

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

// epilogue: acc[i][nt][e]: row = m0 + wm + 16 i + g + 8 (e >> 1), col = n0 + wn + 8 nt + 2 t + (e & 1)
template <int EPI>
__device__ __forceinline__ void store_acc(const float (&acc)[4][4][4], void* Cv, int M, int N, int ldc, float alpha, const half* R, int ldr, float beta, int m0, int n0, int wm, int wn, int lane) {
    const int g = lane >> 2, t = lane & 3;
#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
        for (int nt = 0; nt < 4; nt++)
#pragma unroll
            for (int h = 0; h < 2; h++) {
                const int row = m0 + wm + 16 * i + g + 8 * h;
                const int col = n0 + wn + 8 * nt + 2 * t;
                if (row < M && col < N) {
                    const float v0 = acc[i][nt][2 * h] * alpha, v1 = acc[i][nt][2 * h + 1] * alpha;
                    if (EPI == 0) {
                        *reinterpret_cast<half2*>(reinterpret_cast<half*>(Cv) + (size_t)row * ldc + col) = __floats2half2_rn(v0, v1);
                    } else if (EPI == 5) {
                        // u = fp16(acc); a = fp16(min(relu(u)^2, 65000)) exactly as relu2_kernel computes it from the stored u
                        const half2 u = __floats2half2_rn(v0, v1);
                        *reinterpret_cast<half2*>(reinterpret_cast<half*>(Cv) + (size_t)row * ldc + col) = u;
                        const float r0 = fmaxf(__low2float(u), 0.f), r1 = fmaxf(__high2float(u), 0.f);
                        *reinterpret_cast<half2*>(const_cast<half*>(R) + (size_t)row * ldr + col) = __halves2half2(__float2half(fminf(r0 * r0, 65000.f)), __float2half(fminf(r1 * r1, 65000.f)));
                    } else if (EPI == 6) {
                        // da = fp16(acc); du = fp16(da * 2 relu(u)) exactly as relu2_bwd_kernel computes it from the stored da
                        const half2 d = __floats2half2_rn(v0, v1);
                        const half2 u = *reinterpret_cast<const half2*>(R + (size_t)row * ldr + col);
                        const float r0 = fmaxf(__low2float(u), 0.f), r1 = fmaxf(__high2float(u), 0.f);
                        *reinterpret_cast<half2*>(reinterpret_cast<half*>(Cv) + (size_t)row * ldc + col) = __halves2half2(__float2half(__low2float(d) * 2.f * r0), __float2half(__high2float(d) * 2.f * r1));
                    } else if (EPI == 3) {
                        const half2 r = *reinterpret_cast<const half2*>(R + (size_t)row * ldr + col);
                        *reinterpret_cast<half2*>(reinterpret_cast<half*>(Cv) + (size_t)row * ldc + col) = __floats2half2_rn(v0 + beta * __low2float(r), v1 + beta * __high2float(r));
                    } else if (EPI == 4) {
                        const float2 r = *reinterpret_cast<const float2*>(reinterpret_cast<const float*>(R) + (size_t)row * ldr + col);
                        *reinterpret_cast<float2*>(reinterpret_cast<float*>(Cv) + (size_t)row * ldc + col) = make_float2(v0 + r.x, v1 + r.y);
                    } else if (EPI == 1) {
                        *reinterpret_cast<float2*>(reinterpret_cast<float*>(Cv) + (size_t)row * ldc + col) = make_float2(v0, v1);
                    } else {
                        float2* p = reinterpret_cast<float2*>(reinterpret_cast<float*>(Cv) + (size_t)row * ldc + col);
                        float2 o = *p;
                        o.x += v0;
                        o.y += v1;
                        *p = o;
                    }
                }
            }
}

// split-K epilogue: C += alpha * acc with atomics (several blocks add their slice of the reduction into the same tile)
__device__ __forceinline__ void store_add_atomic(const float (&acc)[4][4][4], float* C, int M, int N, int ldc, float alpha, int m0, int n0, int wm, int wn, int lane) {
    const int g = lane >> 2, t = lane & 3;
#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
        for (int nt = 0; nt < 4; nt++)
#pragma unroll
            for (int h = 0; h < 2; h++) {
                const int row = m0 + wm + 16 * i + g + 8 * h;
                const int col = n0 + wn + 8 * nt + 2 * t;
                if (row < M && col < N) {
                    atomicAdd(C + (size_t)row * ldc + col, acc[i][nt][2 * h] * alpha);
                    atomicAdd(C + (size_t)row * ldc + col + 1, acc[i][nt][2 * h + 1] * alpha);
                }
            }
}

// Epilogue staged through shared memory (E2; same values as store_acc, bit-identical outputs). Each warp writes its 64 x 32 tile into its own 1280-byte part of the (finished) stage buffers, 16 rows
// at a time for fp16 outputs (80-byte rows: conflict-free half2 writes) and 8 rows for fp32 outputs (160-byte rows), and reads it back as 16-byte chunks: the global stores are 128-bit, a warp writes
// 64 contiguous bytes per row (fp16, two whole sectors) or 128 (fp32, a whole line), a quarter (fp16) or half (fp32) of store_acc's store instructions, and no 32-byte sector is written in two halves
// (store_acc's half2 stores cover 16 bytes of a sector each). Residual / u / C reads (EPI 4, 6, 2) are 128-bit too. Needs N, ldc (and ldr where R is used) to be multiples of 8 halves / 4 floats
// and 16-byte aligned pointers; otherwise (and for EPI 3) store_acc runs.
template <int EPI>
__device__ __forceinline__ void store_acc_e2(const float (&acc)[4][4][4], void* Cv, int M, int N, int ldc, float alpha, const half* R, int ldr, float beta, int m0, int n0, int wm, int wn, int lane,
                                             int warp, char* sa, char* sb) {
    constexpr bool F16OUT = (EPI == 0 || EPI == 5 || EPI == 6);
    constexpr int VM = F16OUT ? 7 : 3;
    const bool r_ok = (EPI == 4 || EPI == 5 || EPI == 6) ? ((ldr & VM) == 0 && (reinterpret_cast<uintptr_t>(R) & 15) == 0) : true;
    if (EPI == 3 || ((N | ldc) & VM) != 0 || (reinterpret_cast<uintptr_t>(Cv) & 15) != 0 || !r_ok) {
        store_acc<EPI>(acc, Cv, M, N, ldc, alpha, R, ldr, beta, m0, n0, wm, wn, lane);
        return;
    }
    const int g = lane >> 2, t = lane & 3;
    char* wb = (warp < 4 ? sa : sb) + (warp & 3) * 1280;
    if (F16OUT) {
        half* buf = reinterpret_cast<half*>(wb);
#pragma unroll
        for (int i = 0; i < 4; i++) {
#pragma unroll
            for (int nt = 0; nt < 4; nt++)
#pragma unroll
                for (int h = 0; h < 2; h++)
                    *reinterpret_cast<half2*>(buf + (g + 8 * h) * 40 + 8 * nt + 2 * t) = __floats2half2_rn(acc[i][nt][2 * h] * alpha, acc[i][nt][2 * h + 1] * alpha);
            __syncwarp();
#pragma unroll
            for (int k = 0; k < 2; k++) {
                const int r = (lane >> 2) + 8 * k, c = lane & 3;
                const int row = m0 + wm + 16 * i + r, col = n0 + wn + 8 * c;
                const uint4 v = *reinterpret_cast<const uint4*>(buf + r * 40 + 8 * c);
                if (row < M && col < N) {
                    uint4* cp = reinterpret_cast<uint4*>(reinterpret_cast<half*>(Cv) + (size_t)row * ldc + col);
                    if (EPI == 0) {
                        *cp = v;
                    } else if (EPI == 5) {
                        // u, and relu(u)^2 from the fp16 u exactly as store_acc / relu2_kernel compute it
                        *cp = v;
                        const half2* u = reinterpret_cast<const half2*>(&v);
                        uint4 o;
                        half2* op = reinterpret_cast<half2*>(&o);
#pragma unroll
                        for (int q = 0; q < 4; q++) {
                            const float r0 = fmaxf(__low2float(u[q]), 0.f), r1 = fmaxf(__high2float(u[q]), 0.f);
                            op[q] = __halves2half2(__float2half(fminf(r0 * r0, 65000.f)), __float2half(fminf(r1 * r1, 65000.f)));
                        }
                        *reinterpret_cast<uint4*>(const_cast<half*>(R) + (size_t)row * ldr + col) = o;
                    } else {
                        // du = fp16(da * 2 relu(u)) with da the fp16 product and u from R, as store_acc / relu2_bwd_kernel
                        const uint4 uu = *reinterpret_cast<const uint4*>(R + (size_t)row * ldr + col);
                        const half2* d = reinterpret_cast<const half2*>(&v);
                        const half2* u = reinterpret_cast<const half2*>(&uu);
                        uint4 o;
                        half2* op = reinterpret_cast<half2*>(&o);
#pragma unroll
                        for (int q = 0; q < 4; q++) {
                            const float r0 = fmaxf(__low2float(u[q]), 0.f), r1 = fmaxf(__high2float(u[q]), 0.f);
                            op[q] = __halves2half2(__float2half(__low2float(d[q]) * 2.f * r0), __float2half(__high2float(d[q]) * 2.f * r1));
                        }
                        *cp = o;
                    }
                }
            }
            __syncwarp();
        }
    } else {
        float* buf = reinterpret_cast<float*>(wb);
#pragma unroll
        for (int i = 0; i < 4; i++)
#pragma unroll
            for (int h = 0; h < 2; h++) {
#pragma unroll
                for (int nt = 0; nt < 4; nt++) *reinterpret_cast<float2*>(buf + g * 40 + 8 * nt + 2 * t) = make_float2(acc[i][nt][2 * h] * alpha, acc[i][nt][2 * h + 1] * alpha);
                __syncwarp();
#pragma unroll
                for (int k = 0; k < 2; k++) {
                    const int r = (lane >> 3) + 4 * k, c = lane & 7;
                    const int row = m0 + wm + 16 * i + 8 * h + r, col = n0 + wn + 4 * c;
                    const float4 v = *reinterpret_cast<const float4*>(buf + r * 40 + 4 * c);
                    if (row < M && col < N) {
                        float4* cp = reinterpret_cast<float4*>(reinterpret_cast<float*>(Cv) + (size_t)row * ldc + col);
                        if (EPI == 1) {
                            *cp = v;
                        } else if (EPI == 4) {
                            const float4 rr = *reinterpret_cast<const float4*>(reinterpret_cast<const float*>(R) + (size_t)row * ldr + col);
                            *cp = make_float4(v.x + rr.x, v.y + rr.y, v.z + rr.z, v.w + rr.w);
                        } else {
                            float4 o = *cp;
                            o.x += v.x;
                            o.y += v.y;
                            o.z += v.z;
                            o.w += v.w;
                            *cp = o;
                        }
                    }
                }
                __syncwarp();
            }
    }
}

// ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// gemm3: BK = 16, load-ahead of two tiles, swizzled row-major tiles. A_COL: A stored [K][M]; B_COL: B stored [K][N]
// Options (defaults = the original kernels, byte-identical SASS): E2 the staged epilogue store_acc_e2; GC > 0 column groups of GC tiles: the GC column tiles of a row tile run together and the row
// tiles one after another, group after group (a band of GC weight tiles stays in L2 while all row tiles pass; the activations are read once per group).
template <bool A_COL, bool B_COL, int EPI, bool SKIP = true, bool E2 = false, int GC = 0>
__device__ __forceinline__ void gemm_impl3(const half* __restrict__ A, const half* __restrict__ B, void* __restrict__ Cv, int M, int N, int K, int lda, int ldb, int ldc, float alpha, const half* __restrict__ R,
                                           int ldr, float beta) {
    constexpr int A_ELEMS = A_COL ? 16 * LDM : 128 * 16;
    constexpr int B_ELEMS = B_COL ? 16 * LDM : 128 * 16;
    __shared__ __align__(16) half As[2][A_ELEMS];
    __shared__ __align__(16) half Bs[2][B_ELEMS];
    if (EPI == 2 && gridDim.z > 1) {   // split-K: this block's slice of the reduction
        K /= gridDim.z;
        A += A_COL ? (size_t)blockIdx.z * K * lda : (size_t)blockIdx.z * K;
        B += B_COL ? (size_t)blockIdx.z * K * ldb : (size_t)blockIdx.z * K;
    }
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = (warp >> 2) * 64, wn = (warp & 3) * 32;
    // Tile order. The grid is (column tiles, row tiles). Consecutive blocks should share the operand that has MORE tiles (the larger one, streamed from DRAM), so that the L2 cache serves the
    // other three-quarters of its reads and the smaller operand stays resident: with few row tiles (prompt chunks of 512 tokens against big weight matrices) the row tiles of a column tile run
    // together (+67 % on the 4B's gate/up GEMM); with many row tiles (training: 8192 tokens against small weights) the old order, column tiles fastest, shares the activations.
    int m0, n0;
    if (GC > 0) {
        const int lin = blockIdx.y * gridDim.x + blockIdx.x;
        const int per = GC * gridDim.y, grp = lin / per, rem = lin - grp * per;
        const int gc0 = grp * GC, gw = min(GC, (int)gridDim.x - gc0);
        m0 = (rem / gw) * BM;
        n0 = (gc0 + rem % gw) * BN;
    } else if (gridDim.x > gridDim.y) {
        const int lin = blockIdx.y * gridDim.x + blockIdx.x;
        m0 = (lin % gridDim.y) * BM;
        n0 = (lin / gridDim.y) * BN;
    } else {
        m0 = blockIdx.y * BM;
        n0 = blockIdx.x * BN;
    }

    float acc[4][4][4];
#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
        for (int j = 0; j < 4; j++)
#pragma unroll
            for (int e = 0; e < 4; e++) acc[i][j][e] = 0.f;

    // global -> shared mapping of this thread: one 16-byte chunk of the A tile and one of the B tile per iteration
    const half* pa;
    const half* pb;
    size_t a_step, b_step;
    bool a_ok, b_ok;
    int a_sts, b_sts;
    if (!A_COL) {
        const int row = tid >> 1, ch = tid & 1;   // tile row m, chunk of 8 k values
        a_ok = m0 + row < M;
        pa = A + (size_t)(m0 + row) * lda + ch * 8;
        a_step = 16;
        a_sts = row * 16 + 8 * (ch ^ ((row >> 2) & 1));
    } else {
        const int row = tid >> 4, ch = tid & 15;  // tile row k, chunk of 8 m values
        a_ok = m0 + ch * 8 < M;
        pa = A + (size_t)row * lda + m0 + ch * 8;
        a_step = (size_t)16 * lda;
        a_sts = row * LDM + ch * 8;
    }
    if (!B_COL) {
        const int row = tid >> 1, ch = tid & 1;
        b_ok = n0 + row < N;
        pb = B + (size_t)(n0 + row) * ldb + ch * 8;
        b_step = 16;
        b_sts = row * 16 + 8 * (ch ^ ((row >> 2) & 1));
    } else {
        const int row = tid >> 4, ch = tid & 15;
        b_ok = n0 + ch * 8 < N;
        pb = B + (size_t)row * ldb + n0 + ch * 8;
        b_step = (size_t)16 * ldb;
        b_sts = row * LDM + ch * 8;
    }
    uint4 ra = make_uint4(0u, 0u, 0u, 0u), rb = make_uint4(0u, 0u, 0u, 0u);   // stay zero for threads whose chunk lies outside the matrix
    auto ldg_tile = [&]() {
        if (a_ok) ra = __ldg(reinterpret_cast<const uint4*>(pa));
        if (b_ok) rb = __ldg(reinterpret_cast<const uint4*>(pb));
        pa += a_step;
        pb += b_step;
    };
    auto sts_tile = [&](int buf) {
        *reinterpret_cast<uint4*>(&As[buf][a_sts]) = ra;
        *reinterpret_cast<uint4*>(&Bs[buf][b_sts]) = rb;
    };

    // ldmatrix lane addressing (byte offsets inside one stage buffer, for fragment 0; fragment i / j adds a constant)
    //   row-major A: matrices q = 0..3 are (rows 0-7, k 0-7) (rows 8-15, k 0-7) (rows 0-7, k 8-15) (rows 8-15, k 8-15);  row-major B: (n 0-7, k 0-7) (n 0-7, k 8-15) (n 8-15, k 0-7) (n 8-15, k 8-15)
    //   k-major operands are read with .trans: A: (k chunk, m chunk) = (q >> 1, q & 1); B: (q & 1, q >> 1)
    const int q = lane >> 3, lr = lane & 7, sw = (lr >> 2) & 1;
    uint32_t a_ld, b_ld;
    if (!A_COL) a_ld = (uint32_t)(((wm + lr + (q & 1) * 8) * 16 + 8 * ((q >> 1) ^ sw)) * 2);
    else a_ld = (uint32_t)((((q >> 1) * 8 + lr) * LDM + wm + (q & 1) * 8) * 2);
    if (!B_COL) b_ld = (uint32_t)(((wn + lr + (q >> 1) * 8) * 16 + 8 * ((q & 1) ^ sw)) * 2);
    else b_ld = (uint32_t)(((((q & 1) * 8 + lr) * LDM) + wn + (q >> 1) * 8) * 2);
    constexpr uint32_t A_FRAG = A_COL ? 32 : 512, B_FRAG = B_COL ? 32 : 512;   // bytes between consecutive 16-row / 16-column fragments
    const uint32_t sa0 = smem_u32(&As[0][0]), sb0 = smem_u32(&Bs[0][0]);

    // a warp whose whole tile is outside the matrix only helps with the copies (warp-uniform: no divergence)
    const bool wact = !SKIP || ((m0 + wm < M) && (n0 + wn < N));
    const int KT = K / 16;
    ldg_tile();
    sts_tile(0);
    if (KT > 1) ldg_tile();
    __syncthreads();
    // One k tile: the copy of the next tiles and, when `mma` (this warp's tile lies inside the matrix), the ldmatrix reads and the MMAs. A block whose tile lies entirely inside the matrix runs the
    // loop without the test: a branch inside the loop cost a few per cent of the MMA scheduling on full tiles (T4 kbench, job kernel-opt-job3).
    auto ktile = [&](int kt, bool mma) {
        const int buf = kt & 1;
        if (kt + 1 < KT) {
            sts_tile(buf ^ 1);          // tile kt + 1, loaded one iteration ago
            if (kt + 2 < KT) ldg_tile();  // tile kt + 2: its latency is hidden behind this iteration's MMAs
        }
        if (mma) {
            const uint32_t sa = sa0 + buf * (A_ELEMS * 2), sb = sb0 + buf * (B_ELEMS * 2);
            uint32_t fa[4][4], fb[2][4];
#pragma unroll
            for (int i = 0; i < 4; i++) {
                if (!A_COL) ldsm_x4(fa[i], sa + a_ld + i * A_FRAG);
                else ldsm_x4_t(fa[i], sa + a_ld + i * A_FRAG);
            }
#pragma unroll
            for (int j = 0; j < 2; j++) {
                if (!B_COL) ldsm_x4(fb[j], sb + b_ld + j * B_FRAG);
                else ldsm_x4_t(fb[j], sb + b_ld + j * B_FRAG);
            }
#pragma unroll
            for (int s = 0; s < 2; s++)
#pragma unroll
                for (int i = 0; i < 4; i++)
#pragma unroll
                    for (int j = 0; j < 2; j++) {
                        mma_k8(acc[i][2 * j], fa[i][2 * s], fa[i][2 * s + 1], fb[j][s]);
                        mma_k8(acc[i][2 * j + 1], fa[i][2 * s], fa[i][2 * s + 1], fb[j][2 + s]);
                    }
        }
        __syncthreads();
    };
    if (!SKIP || (m0 + BM <= M && n0 + BN <= N)) {
        for (int kt = 0; kt < KT; kt++) ktile(kt, true);
    } else {
        for (int kt = 0; kt < KT; kt++) ktile(kt, wact);
    }
    if (!wact) return;
    if (EPI == 2 && gridDim.z > 1) store_add_atomic(acc, reinterpret_cast<float*>(Cv), M, N, ldc, alpha, m0, n0, wm, wn, lane);
    else if (E2) store_acc_e2<EPI>(acc, Cv, M, N, ldc, alpha, R, ldr, beta, m0, n0, wm, wn, lane, warp, reinterpret_cast<char*>(&As[0][0]), reinterpret_cast<char*>(&Bs[0][0]));
    else store_acc<EPI>(acc, Cv, M, N, ldc, alpha, R, ldr, beta, m0, n0, wm, wn, lane);
}

// ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// previous generation (kept for A/B timing): padded rows, store right behind the load, BK = 16 (two blocks per SM) or 32 (one block per SM)
template <bool A_COL, bool B_COL, int EPI, int BK>
__device__ __forceinline__ void gemm_impl(const half* __restrict__ A, const half* __restrict__ B, void* __restrict__ Cv, int M, int N, int K, int lda, int ldb, int ldc, float alpha, const half* __restrict__ R,
                                          int ldr, float beta) {
    constexpr int LDK = BK + 8;   // row stride (halves) of tiles whose rows run along M/N and whose columns are K: BK = 16 -> 48 B, BK = 32 -> 80 B (both ldmatrix conflict free)
    constexpr int NL = BK / 16;   // 16-byte chunks per thread per operand per tile
    constexpr int CPR = BK / 8;   // chunks per row of a row-major tile
    constexpr int RPP = 256 / CPR;  // rows covered by one pass of the 256 threads
    constexpr int A_ELEMS = A_COL ? BK * LDM : BM * LDK;
    constexpr int B_ELEMS = B_COL ? BK * LDM : BN * LDK;
    __shared__ __align__(16) half As[2][A_ELEMS];
    __shared__ __align__(16) half Bs[2][B_ELEMS];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = (warp >> 2) * 64, wn = (warp & 3) * 32;
    // Tile order. The grid is (column tiles, row tiles). Consecutive blocks should share the operand that has MORE tiles (the larger one, streamed from DRAM), so that the L2 cache serves the
    // other three-quarters of its reads and the smaller operand stays resident: with few row tiles (prompt chunks of 512 tokens against big weight matrices) the row tiles of a column tile run
    // together (+67 % on the 4B's gate/up GEMM); with many row tiles (training: 8192 tokens against small weights) the old order, column tiles fastest, shares the activations.
    int m0, n0;
    if (gridDim.x > gridDim.y) {
        const int lin = blockIdx.y * gridDim.x + blockIdx.x;
        m0 = (lin % gridDim.y) * BM;
        n0 = (lin / gridDim.y) * BN;
    } else {
        m0 = blockIdx.y * BM;
        n0 = blockIdx.x * BN;
    }

    float acc[4][4][4];
#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
        for (int j = 0; j < 4; j++)
#pragma unroll
            for (int e = 0; e < 4; e++) acc[i][j][e] = 0.f;

    uint4 ra[NL], rb[NL];
    auto load_tile = [&](int k0) {
#pragma unroll
        for (int i = 0; i < NL; i++) {
            if (!A_COL) {
                const int row = tid / CPR + RPP * i, ch = tid % CPR, gm = m0 + row;
                ra[i] = gm < M ? __ldg(reinterpret_cast<const uint4*>(A + (size_t)gm * lda + k0) + ch) : make_uint4(0u, 0u, 0u, 0u);
            } else {
                const int row = (tid >> 4) + 16 * i, ch = tid & 15, gm = m0 + ch * 8;
                ra[i] = gm < M ? __ldg(reinterpret_cast<const uint4*>(A + (size_t)(k0 + row) * lda + gm)) : make_uint4(0u, 0u, 0u, 0u);
            }
            if (!B_COL) {
                const int row = tid / CPR + RPP * i, ch = tid % CPR, gn = n0 + row;
                rb[i] = gn < N ? __ldg(reinterpret_cast<const uint4*>(B + (size_t)gn * ldb + k0) + ch) : make_uint4(0u, 0u, 0u, 0u);
            } else {
                const int row = (tid >> 4) + 16 * i, ch = tid & 15, gn = n0 + ch * 8;
                rb[i] = gn < N ? __ldg(reinterpret_cast<const uint4*>(B + (size_t)(k0 + row) * ldb + gn)) : make_uint4(0u, 0u, 0u, 0u);
            }
        }
    };
    auto store_tile = [&](int buf) {
#pragma unroll
        for (int i = 0; i < NL; i++) {
            if (!A_COL) *reinterpret_cast<uint4*>(&As[buf][(tid / CPR + RPP * i) * LDK + (tid % CPR) * 8]) = ra[i];
            else *reinterpret_cast<uint4*>(&As[buf][((tid >> 4) + 16 * i) * LDM + (tid & 15) * 8]) = ra[i];
            if (!B_COL) *reinterpret_cast<uint4*>(&Bs[buf][(tid / CPR + RPP * i) * LDK + (tid % CPR) * 8]) = rb[i];
            else *reinterpret_cast<uint4*>(&Bs[buf][((tid >> 4) + 16 * i) * LDM + (tid & 15) * 8]) = rb[i];
        }
    };

    const int KT = K / BK;
    load_tile(0);
    store_tile(0);
    __syncthreads();

    const int q = lane >> 3, lr = lane & 7;
    for (int kt = 0; kt < KT; kt++) {
        const int buf = kt & 1;
        if (kt + 1 < KT) load_tile((kt + 1) * BK);
        const uint32_t sa = smem_u32(&As[buf][0]), sb = smem_u32(&Bs[buf][0]);
#pragma unroll
        for (int ks = 0; ks < BK / 16; ks++) {   // k16 steps per tile
            const int k0 = ks * 16;
            uint32_t fa[4][4], fb[2][4];
#pragma unroll
            for (int i = 0; i < 4; i++) {
                if (!A_COL) ldsm_x4(fa[i], sa + ((wm + 16 * i + lr + (q & 1) * 8) * LDK + k0 + (q >> 1) * 8) * 2);
                else ldsm_x4_t(fa[i], sa + ((k0 + (q >> 1) * 8 + lr) * LDM + wm + 16 * i + (q & 1) * 8) * 2);
            }
#pragma unroll
            for (int j = 0; j < 2; j++) {
                if (!B_COL) ldsm_x4(fb[j], sb + ((wn + 16 * j + lr + (q >> 1) * 8) * LDK + k0 + (q & 1) * 8) * 2);
                else ldsm_x4_t(fb[j], sb + ((k0 + (q & 1) * 8 + lr) * LDM + wn + 16 * j + (q >> 1) * 8) * 2);
            }
#pragma unroll
            for (int s = 0; s < 2; s++)
#pragma unroll
                for (int i = 0; i < 4; i++)
#pragma unroll
                    for (int j = 0; j < 2; j++) {
                        mma_k8(acc[i][2 * j], fa[i][2 * s], fa[i][2 * s + 1], fb[j][s]);
                        mma_k8(acc[i][2 * j + 1], fa[i][2 * s], fa[i][2 * s + 1], fb[j][2 + s]);
                    }
        }
        if (kt + 1 < KT) store_tile(buf ^ 1);
        __syncthreads();
    }
    store_acc<EPI>(acc, Cv, M, N, ldc, alpha, R, ldr, beta, m0, n0, wm, wn, lane);
}

// ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// Epilogue with STREAMING stores (st.global.cs: evict-first in L1 and L2) and streaming residual loads, same values as store_acc (EPI 0, 4, 5 only). The outputs of the forward GEMMs (19-38 MB per GEMM at
// 4096 tokens) otherwise push the operand tiles out of the 4 MB L2 of a Turing card (T4 Nsight Compute, MLP up GEMM: 128 MB of DRAM reads for 7.4 MB of operands).
__device__ __forceinline__ void st_cs_b32(void* p, uint32_t v) { asm volatile("st.global.cs.b32 [%0], %1;\n" ::"l"(p), "r"(v) : "memory"); }
__device__ __forceinline__ void st_cs_f2(void* p, float a, float b) { asm volatile("st.global.cs.v2.f32 [%0], {%1, %2};\n" ::"l"(p), "f"(a), "f"(b) : "memory"); }
__device__ __forceinline__ float2 ld_cs_f2(const void* p) {
    float2 r;
    asm volatile("ld.global.cs.v2.f32 {%0, %1}, [%2];\n" : "=f"(r.x), "=f"(r.y) : "l"(p));
    return r;
}
__device__ __forceinline__ uint32_t h2_bits(half2 h) { return *reinterpret_cast<uint32_t*>(&h); }

template <int EPI>
__device__ __forceinline__ void store_acc_cs(const float (&acc)[4][4][4], void* Cv, int M, int N, int ldc, float alpha, const half* R, int ldr, int m0, int n0, int wm, int wn, int lane) {
    static_assert(EPI == 0 || EPI == 4 || EPI == 5, "streaming epilogue: the forward kernels' epilogues only");
    const int g = lane >> 2, t = lane & 3;
#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
        for (int nt = 0; nt < 4; nt++)
#pragma unroll
            for (int h = 0; h < 2; h++) {
                const int row = m0 + wm + 16 * i + g + 8 * h;
                const int col = n0 + wn + 8 * nt + 2 * t;
                if (row < M && col < N) {
                    const float v0 = acc[i][nt][2 * h] * alpha, v1 = acc[i][nt][2 * h + 1] * alpha;
                    if (EPI == 0) {
                        st_cs_b32(reinterpret_cast<half*>(Cv) + (size_t)row * ldc + col, h2_bits(__floats2half2_rn(v0, v1)));
                    } else if (EPI == 5) {
                        const half2 u = __floats2half2_rn(v0, v1);
                        st_cs_b32(reinterpret_cast<half*>(Cv) + (size_t)row * ldc + col, h2_bits(u));
                        const float r0 = fmaxf(__low2float(u), 0.f), r1 = fmaxf(__high2float(u), 0.f);
                        st_cs_b32(const_cast<half*>(R) + (size_t)row * ldr + col, h2_bits(__halves2half2(__float2half(fminf(r0 * r0, 65000.f)), __float2half(fminf(r1 * r1, 65000.f)))));
                    } else {
                        const float2 r = ld_cs_f2(reinterpret_cast<const float*>(R) + (size_t)row * ldr + col);
                        st_cs_f2(reinterpret_cast<float*>(Cv) + (size_t)row * ldc + col, v0 + r.x, v1 + r.y);
                    }
                }
            }
}

// ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// fp16-ACCUMULATE variant of gemm3 for the forward NT GEMMs (FORGE_F16ACC=P in the trainer; the kernels above are untouched). GeForce Turing runs mma with fp32 accumulators at half the rate of fp16
// accumulators (RTX 2060 SUPER: ~29 vs ~57 TFLOPS); the T4 / Titan RTX / Quadro run both at full rate. The MMAs accumulate into fp16 registers (mma ... .f16.f16.f16.f16, 2 registers per 16 x 8 tile instead
// of 4) and every P k8 steps (P / 2 k tiles: P even, `FK` = P / 2; FK = 0: never, one flush at the end) the fp16 tile is added into the fp32 accumulators, which the epilogues read as before.
// The fp16 accumulator restarts from zero after a flush (the first MMA of an interval takes C = 0), so its value spans at most P x 8 products: the error is set by the flush length, not by K.
// Fragments are read per k8 step (ldmatrix.x4 of the matrices of one k half: A rows 0-7/8-15 of two 16-row fragments, B the four 8-column fragments), 12 live registers instead of 24, to make room for the 32
// fp16 accumulator registers (same number of ldmatrix.x4 per k tile). Row-major A and B only (A [M][K], B [N][K]).
__device__ __forceinline__ void mma_k8h(uint32_t (&d)[2], uint32_t a0, uint32_t a1, uint32_t b0) {
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3}, {%4}, {%0,%1};\n" : "+r"(d[0]), "+r"(d[1]) : "r"(a0), "r"(a1), "r"(b0));
}
// d = A B (C = 0): the first MMA of a flush interval
__device__ __forceinline__ void mma_k8h_z(uint32_t (&d)[2], uint32_t a0, uint32_t a1, uint32_t b0) {
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3}, {%4}, {%5,%5};\n" : "=r"(d[0]), "=r"(d[1]) : "r"(a0), "r"(a1), "r"(b0), "r"(0u));
}

// Experiment parameters (kbench only; the trainer's kernels use the defaults): UNROLL = false one k tile per loop iteration; NOMMA the timing skeleton without MMAs; NS register stages of the global ->
// shared copy (1: the load of tile kt + 2 is issued in iteration kt, as gemm3; 2: tile kt + 3, twice the bytes in flight, 8 more registers); F32 the same code with fp32 accumulators (control).
template <int EPI, int FK, bool SKIP = true, bool UNROLL = true, bool NOMMA = false, int NS = 1, bool F32 = false, bool CS = false>
__device__ __forceinline__ void gemm_nt_h(const half* __restrict__ A, const half* __restrict__ B, void* __restrict__ Cv, int M, int N, int K, int lda, int ldb, int ldc, float alpha, const half* __restrict__ R, int ldr,
                                          float beta) {
    static_assert(NS == 1 || (UNROLL && FK > 0 && FK % NS == 0), "deeper register staging needs the unrolled flush interval (compile-time register slots)");
    __shared__ __align__(16) half As[2][128 * 16];
    __shared__ __align__(16) half Bs[2][128 * 16];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = (warp >> 2) * 64, wn = (warp & 3) * 32;
    int m0, n0;
    if (gridDim.x > gridDim.y) {
        const int lin = blockIdx.y * gridDim.x + blockIdx.x;
        m0 = (lin % gridDim.y) * BM;
        n0 = (lin / gridDim.y) * BN;
    } else {
        m0 = blockIdx.y * BM;
        n0 = blockIdx.x * BN;
    }

    float acc[4][4][4];
#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
        for (int j = 0; j < 4; j++)
#pragma unroll
            for (int e = 0; e < 4; e++) acc[i][j][e] = 0.f;
    uint32_t hacc[4][4][2];   // fp16x2 accumulators: [i][nt][0] = row g (cols 2t, 2t + 1), [1] = row g + 8 (the fp32 layout's elements 0-1 and 2-3)
    if (NOMMA) {
#pragma unroll
        for (int i = 0; i < 4; i++)
#pragma unroll
            for (int nt = 0; nt < 4; nt++) hacc[i][nt][0] = hacc[i][nt][1] = 0u;
    }

    // global -> shared: one 16-byte chunk of the A tile and one of the B tile per thread and k tile (swizzled 32-byte rows, as gemm3)
    const int row = tid >> 1, ch = tid & 1;
    const bool a_ok = m0 + row < M, b_ok = n0 + row < N;
    const half* pa = A + (size_t)(m0 + row) * lda + ch * 8;
    const half* pb = B + (size_t)(n0 + row) * ldb + ch * 8;
    const int sts = row * 16 + 8 * (ch ^ ((row >> 2) & 1));
    uint4 ra[NS], rb[NS];   // register stages (tile t in slot t % NS); stay zero for threads whose chunk lies outside the matrix
#pragma unroll
    for (int z = 0; z < NS; z++) ra[z] = rb[z] = make_uint4(0u, 0u, 0u, 0u);
    auto ldg_tile = [&](int slot) {
        if (a_ok) ra[slot] = __ldg(reinterpret_cast<const uint4*>(pa));
        if (b_ok) rb[slot] = __ldg(reinterpret_cast<const uint4*>(pb));
        pa += 16;
        pb += 16;
    };
    auto sts_tile = [&](int buf, int slot) {
        *reinterpret_cast<uint4*>(&As[buf][sts]) = ra[slot];
        *reinterpret_cast<uint4*>(&Bs[buf][sts]) = rb[slot];
    };
    // ldmatrix per k8 step s (byte offsets in a stage buffer; the k chunk is bit 4 of the offset, so step 1 = step 0 XOR 16):
    //   A, fragment pair p (i = 2p, 2p + 1): matrices (i rows 0-7) (i rows 8-15) (i+1 rows 0-7) (i+1 rows 8-15) -> row wm + 32p + 8q + lr;  B: matrix q = columns wn + 8q .. + 7 -> row wn + 8q + lr
    const int q = lane >> 3, lr = lane & 7, sw = (lr >> 2) & 1;
    const uint32_t a_ld = (uint32_t)((wm + 8 * q + lr) * 32 + 16 * sw), b_ld = (uint32_t)((wn + 8 * q + lr) * 32 + 16 * sw);
    const uint32_t sa0 = smem_u32(&As[0][0]), sb0 = smem_u32(&Bs[0][0]);

    const bool wact = !SKIP || ((m0 + wm < M) && (n0 + wn < N));
    const int KT = K / 16;
    // prologue: tile 0 into shared memory, tiles 1 .. NS in flight in the registers
    ldg_tile(0);
    sts_tile(0, 0);
#pragma unroll
    for (int t = 1; t <= NS; t++)
        if (t < KT) ldg_tile(t % NS);
    __syncthreads();
    // one k tile (two k8 steps) into the fp16 accumulators: tile kt + 1 (register slot `slot` = (kt + 1) % NS, loaded NS iterations ago) goes to shared memory and tile kt + 1 + NS is loaded into the
    // freed slot, then the MMAs on tile kt; `fresh`: the first k tile of a flush interval (its first MMAs start from zero)
    auto ktile = [&](int kt, int slot, bool mma, bool fresh) {
        const int buf = kt & 1;
        if (kt + 1 < KT) {
            sts_tile(buf ^ 1, slot);
            if (kt + 1 + NS < KT) ldg_tile(slot);
        }
        if (mma) {
            const uint32_t sa = sa0 + buf * (128 * 16 * 2), sb = sb0 + buf * (128 * 16 * 2);
#pragma unroll
            for (int s = 0; s < 2; s++) {
                uint32_t fa[2][4], fb[4];
                ldsm_x4(fa[0], sa + (a_ld ^ (s << 4)));
                ldsm_x4(fa[1], sa + (a_ld ^ (s << 4)) + 32 * 32);
                ldsm_x4(fb, sb + (b_ld ^ (s << 4)));
#pragma unroll
                for (int i = 0; i < 4; i++)
#pragma unroll
                    for (int nt = 0; nt < 4; nt++) {
                        const uint32_t a0 = fa[i >> 1][2 * (i & 1)], a1 = fa[i >> 1][2 * (i & 1) + 1];
                        if (NOMMA) hacc[i][nt][0] ^= a0 ^ a1 ^ fb[nt];   // timing skeleton: the same loads, no MMA (the result is garbage)
                        else if (F32) mma_k8(acc[i][nt], a0, a1, fb[nt]);
                        else if (fresh && s == 0) mma_k8h_z(hacc[i][nt], a0, a1, fb[nt]);
                        else mma_k8h(hacc[i][nt], a0, a1, fb[nt]);
                    }
            }
        }
        __syncthreads();
    };
    // acc += fp32(hacc)
    auto flush = [&]() {
        if (F32) return;
#pragma unroll
        for (int i = 0; i < 4; i++)
#pragma unroll
            for (int nt = 0; nt < 4; nt++) {
                const float2 lo = __half22float2(*reinterpret_cast<const half2*>(&hacc[i][nt][0]));
                const float2 hi = __half22float2(*reinterpret_cast<const half2*>(&hacc[i][nt][1]));
                acc[i][nt][0] += lo.x;
                acc[i][nt][1] += lo.y;
                acc[i][nt][2] += hi.x;
                acc[i][nt][3] += hi.y;
            }
    };
    // FK k tiles per flush interval, unrolled (the `fresh` MMAs are resolved at compile time); a remainder of KT % FK tiles (FK = 0: all of them) runs as one more interval from zeroed accumulators
    const int NI = FK > 0 ? KT / FK : 0;
    const int K0 = NI * FK;
    auto run = [&](bool mma) {
        if (!UNROLL) {
            // experiment (kbench): one k tile per loop iteration (the stage buffer stays a run-time value, as with FK = 1), the fp16 accumulators zeroed after each flush
#pragma unroll
            for (int i = 0; i < 4; i++)
#pragma unroll
                for (int nt = 0; nt < 4; nt++) hacc[i][nt][0] = hacc[i][nt][1] = 0u;
#pragma unroll 1
            for (int kt = 0; kt < KT; kt++) {
                ktile(kt, 0, mma, false);
                if (mma && (kt % FK == FK - 1 || kt == KT - 1)) {
                    flush();
#pragma unroll
                    for (int i = 0; i < 4; i++)
#pragma unroll
                        for (int nt = 0; nt < 4; nt++) hacc[i][nt][0] = hacc[i][nt][1] = 0u;
                }
            }
            return;
        }
        for (int it = 0; it < NI; it++) {
#pragma unroll
            for (int u = 0; u < (FK > 0 ? FK : 1); u++) ktile(it * FK + u, (u + 1) % NS, mma, u == 0);
            if (mma) flush();
        }
        if (K0 < KT) {
            if (FK > 0) {
                // the last KT % FK tiles: one shorter interval, unrolled with guards (compile-time register slots)
#pragma unroll
                for (int u = 0; u < (FK > 0 ? FK : 1); u++)
                    if (K0 + u < KT) ktile(K0 + u, (u + 1) % NS, mma, u == 0);
            } else {
#pragma unroll
                for (int i = 0; i < 4; i++)
#pragma unroll
                    for (int nt = 0; nt < 4; nt++) hacc[i][nt][0] = hacc[i][nt][1] = 0u;
                for (int kt = 0; kt < KT; kt++) ktile(kt, 0, mma, false);
            }
            if (mma) flush();
        }
    };
    if (!SKIP || (m0 + BM <= M && n0 + BN <= N)) run(true);
    else run(wact);
    if (!wact) return;
    if (EPI == 7) {
        // timing experiment without the epilogue's memory traffic: the accumulators stay live through a sum that is written only if it takes a value it never takes
        float sum = 0.f;
#pragma unroll
        for (int i = 0; i < 4; i++)
#pragma unroll
            for (int j = 0; j < 4; j++)
#pragma unroll
                for (int e = 0; e < 4; e++) sum += acc[i][j][e];
        if (sum == -12345.f) reinterpret_cast<float*>(Cv)[tid] = sum;
        return;
    }
    if (CS) store_acc_cs<(CS ? EPI : 0)>(acc, Cv, M, N, ldc, alpha, R, ldr, m0, n0, wm, wn, lane);
    else store_acc<EPI>(acc, Cv, M, N, ldc, alpha, R, ldr, beta, m0, n0, wm, wn, lane);
}

}  // namespace gemm

#define GEMM_KERNEL(NAME, ACOL, BCOL, EPI, CT)                                                                                                                                              \
    extern "C" __global__ void __launch_bounds__(256, 2) NAME(const half* A, const half* B, CT* C, int M, int N, int K, int lda, int ldb, int ldc, float alpha) {                           \
        gemm::gemm_impl3<ACOL, BCOL, EPI>(A, B, C, M, N, K, lda, ldb, ldc, alpha, nullptr, 0, 0.f);                                                                                         \
    }
#define GEMM_KERNEL_OLD(NAME, ACOL, BCOL, EPI, CT, BKV, MINB)                                                                                                                               \
    extern "C" __global__ void __launch_bounds__(256, MINB) NAME(const half* A, const half* B, CT* C, int M, int N, int K, int lda, int ldb, int ldc, float alpha) {                       \
        gemm::gemm_impl<ACOL, BCOL, EPI, BKV>(A, B, C, M, N, K, lda, ldb, ldc, alpha, nullptr, 0, 0.f);                                                                                     \
    }

GEMM_KERNEL(gemm_nt_f16, false, false, 0, half)
GEMM_KERNEL(gemm_nt_f32, false, false, 1, float)
GEMM_KERNEL(gemm_nt_add, false, false, 2, float)
GEMM_KERNEL(gemm_nn_f16, false, true, 0, half)
GEMM_KERNEL(gemm_nn_f32, false, true, 1, float)
GEMM_KERNEL(gemm_nn_add, false, true, 2, float)
GEMM_KERNEL(gemm_tn_f32, true, true, 1, float)
GEMM_KERNEL(gemm_tn_add, true, true, 2, float)
// fp16 output for the weight-gradient layout (Gram matrix of a tall matrix in the Muon iteration)
GEMM_KERNEL(gemm_tn_f16, true, true, 0, half)

// previous configurations for comparison
GEMM_KERNEL_OLD(gemm_nt_f32_b16, false, false, 1, float, 16, 2)
GEMM_KERNEL_OLD(gemm_nn_f32_b16, false, true, 1, float, 16, 2)
GEMM_KERNEL_OLD(gemm_tn_f32_b16, true, true, 1, float, 16, 2)
GEMM_KERNEL_OLD(gemm_nt_f32_b32, false, false, 1, float, 32, 1)
GEMM_KERNEL_OLD(gemm_nn_f32_b32, false, true, 1, float, 32, 1)
GEMM_KERNEL_OLD(gemm_tn_f32_b32, true, true, 1, float, 32, 1)

// axpy epilogue (EPI 3), used by the Muon Newton-Schulz iteration: C = alpha * (A B) + beta * R, all fp16
#define GEMM_AXPY_KERNEL(NAME, ACOL, BCOL)                                                                                                                                                  \
    extern "C" __global__ void __launch_bounds__(256, 2) NAME(const half* A, const half* B, half* C, int M, int N, int K, int lda, int ldb, int ldc, float alpha, const half* R, int ldr, float beta) { \
        gemm::gemm_impl3<ACOL, BCOL, 3>(A, B, C, M, N, K, lda, ldb, ldc, alpha, R, ldr, beta);                                                                                              \
    }

GEMM_AXPY_KERNEL(gemm_nt_axpy, false, false)
GEMM_AXPY_KERNEL(gemm_nn_axpy, false, true)

// fp32 output with a separate fp32 residual (EPI 4): C = alpha * (A B) + R. R and C may alias exactly (same pointer and stride) or be different buffers.
#define GEMM_RES_KERNEL(NAME, ACOL, BCOL)                                                                                                                                                   \
    extern "C" __global__ void __launch_bounds__(256, 2) NAME(const half* A, const half* B, float* C, int M, int N, int K, int lda, int ldb, int ldc, float alpha, const float* R, int ldr, float beta) { \
        gemm::gemm_impl3<ACOL, BCOL, 4>(A, B, C, M, N, K, lda, ldb, ldc, alpha, reinterpret_cast<const half*>(R), ldr, beta);                                                               \
    }

GEMM_RES_KERNEL(gemm_nt_res, false, false)

// the same kernels WITHOUT the idle-warp skip (A/B timing in `forge kbench` only)
extern "C" __global__ void __launch_bounds__(256, 2) gemm_nt_f16_ns(const half* A, const half* B, half* C, int M, int N, int K, int lda, int ldb, int ldc, float alpha) {
    gemm::gemm_impl3<false, false, 0, false>(A, B, C, M, N, K, lda, ldb, ldc, alpha, nullptr, 0, 0.f);
}
extern "C" __global__ void __launch_bounds__(256, 2) gemm_nn_f32_ns(const half* A, const half* B, float* C, int M, int N, int K, int lda, int ldb, int ldc, float alpha) {
    gemm::gemm_impl3<false, true, 1, false>(A, B, C, M, N, K, lda, ldb, ldc, alpha, nullptr, 0, 0.f);
}
extern "C" __global__ void __launch_bounds__(256, 2) gemm_tn_add_ns(const half* A, const half* B, float* C, int M, int N, int K, int lda, int ldb, int ldc, float alpha) {
    gemm::gemm_impl3<true, true, 2, false>(A, B, C, M, N, K, lda, ldb, ldc, alpha, nullptr, 0, 0.f);
}
extern "C" __global__ void __launch_bounds__(256, 2) gemm_nt_res_ns(const half* A, const half* B, float* C, int M, int N, int K, int lda, int ldb, int ldc, float alpha, const float* R, int ldr, float beta) {
    gemm::gemm_impl3<false, false, 4, false>(A, B, C, M, N, K, lda, ldb, ldc, alpha, reinterpret_cast<const half*>(R), ldr, beta);
}

// ReLU^2 MLP epilogues (same arguments as the axpy kernels; beta unused):
//   gemm_nt_relu2      forward up projection: C = u (fp16), R = the activation output relu(u)^2 (fp16, row stride ldr)
//   gemm_nn_relu2_bwd  data gradient of the down projection: C = du = da * 2 relu(u) (fp16) with u = R (fp16, row stride ldr)
extern "C" __global__ void __launch_bounds__(256, 2) gemm_nt_relu2(const half* A, const half* B, half* C, int M, int N, int K, int lda, int ldb, int ldc, float alpha, half* R, int ldr, float beta) {
    gemm::gemm_impl3<false, false, 5>(A, B, C, M, N, K, lda, ldb, ldc, alpha, R, ldr, beta);
}
extern "C" __global__ void __launch_bounds__(256, 2) gemm_nn_relu2_bwd(const half* A, const half* B, half* C, int M, int N, int K, int lda, int ldb, int ldc, float alpha, const half* R, int ldr, float beta) {
    gemm::gemm_impl3<false, true, 6>(A, B, C, M, N, K, lda, ldb, ldc, alpha, R, ldr, beta);
}

// fp16-accumulate forward kernels (same arguments as gemm_nt_f16 / gemm_nt_res / gemm_nt_relu2), flushed into fp32 every P k8 steps: gemm_nt_f16_hP, gemm_nt_res_hP, gemm_nt_relu2_hP for P = 2, 4, 8, 16;
// gemm_nt_f16_h0 never flushes (one fp16 accumulator over the whole K: numerics and timing reference only)
#define GEMM_H_KERNELS(P)                                                                                                                                                                         \
    extern "C" __global__ void __launch_bounds__(256, 2) gemm_nt_f16_h##P(const half* A, const half* B, half* C, int M, int N, int K, int lda, int ldb, int ldc, float alpha) {                     \
        gemm::gemm_nt_h<0, P / 2>(A, B, C, M, N, K, lda, ldb, ldc, alpha, nullptr, 0, 0.f);                                                                                                       \
    }                                                                                                                                                                                             \
    extern "C" __global__ void __launch_bounds__(256, 2) gemm_nt_res_h##P(const half* A, const half* B, float* C, int M, int N, int K, int lda, int ldb, int ldc, float alpha, const float* R, int ldr, \
                                                                          float beta) {                                                                                                           \
        gemm::gemm_nt_h<4, P / 2>(A, B, C, M, N, K, lda, ldb, ldc, alpha, reinterpret_cast<const half*>(R), ldr, beta);                                                                           \
    }                                                                                                                                                                                             \
    extern "C" __global__ void __launch_bounds__(256, 2) gemm_nt_relu2_h##P(const half* A, const half* B, half* C, int M, int N, int K, int lda, int ldb, int ldc, float alpha, half* R, int ldr,    \
                                                                            float beta) {                                                                                                         \
        gemm::gemm_nt_h<5, P / 2>(A, B, C, M, N, K, lda, ldb, ldc, alpha, R, ldr, beta);                                                                                                          \
    }

GEMM_H_KERNELS(2)
GEMM_H_KERNELS(4)
GEMM_H_KERNELS(8)
GEMM_H_KERNELS(16)
extern "C" __global__ void __launch_bounds__(256, 2) gemm_nt_f16_h0(const half* A, const half* B, half* C, int M, int N, int K, int lda, int ldb, int ldc, float alpha) {
    gemm::gemm_nt_h<0, 0>(A, B, C, M, N, K, lda, ldb, ldc, alpha, nullptr, 0, 0.f);
}
// Diagnostics for `forge kbench --only f16acc --f16p ...` (names gemm_nt_{f16,res,relu2}_h<suffix>; the trainer uses only the _hP kernels above):
//   8skel   the P = 8 kernel WITHOUT the MMAs (same loads, ldmatrix, barriers, flushes, epilogue; garbage output): the time floor of everything but the tensor cores
//   8f32    this code with fp32 accumulators (control; its results are bit-identical to gemm3's)
//   8noepi / 8f32noepi / 8skelnoepi   without the epilogue's loads and stores (EPI 7: nothing is written)
// Tried and measured, not instantiated any more (research note: ForgeInfer - fp16 Accumulate GEMM Kernels (2026-10)); re-enable with one line each:
//   GEMM_H_VARIANT(8l, 2, false, false, 1, false, false, 4)   one k tile per loop iteration          GEMM_H_VARIANT(8b1, 1, true, false, 1, false, false, 4)   one block per SM
//   GEMM_H_VARIANT(8d, 2, true, false, 2, false, false, 4)    two register stages (NS = 2)           GEMM_H_VARIANT(8f32d, 2, true, false, 2, true, false, 4)  the same, fp32 accumulators
//   GEMM_H_VARIANT(8cs, 2, true, false, 1, false, true, 4)    streaming epilogue stores (CS)        GEMM_H_VARIANT(8f32cs, 2, true, false, 1, true, true, 4)  the same, fp32 accumulators
#define GEMM_H_VARIANT(SUF, MINB, UNR, NOMMA, NS, F32, CS, FKV)                                                                                                                                   \
    extern "C" __global__ void __launch_bounds__(256, MINB) gemm_nt_f16_h##SUF(const half* A, const half* B, half* C, int M, int N, int K, int lda, int ldb, int ldc, float alpha) {                 \
        gemm::gemm_nt_h<0, FKV, true, UNR, NOMMA, NS, F32, CS>(A, B, C, M, N, K, lda, ldb, ldc, alpha, nullptr, 0, 0.f);                                                                          \
    }                                                                                                                                                                                             \
    extern "C" __global__ void __launch_bounds__(256, MINB) gemm_nt_res_h##SUF(const half* A, const half* B, float* C, int M, int N, int K, int lda, int ldb, int ldc, float alpha, const float* R,    \
                                                                               int ldr, float beta) {                                                                                             \
        gemm::gemm_nt_h<4, FKV, true, UNR, NOMMA, NS, F32, CS>(A, B, C, M, N, K, lda, ldb, ldc, alpha, reinterpret_cast<const half*>(R), ldr, beta);                                              \
    }                                                                                                                                                                                             \
    extern "C" __global__ void __launch_bounds__(256, MINB) gemm_nt_relu2_h##SUF(const half* A, const half* B, half* C, int M, int N, int K, int lda, int ldb, int ldc, float alpha, half* R,         \
                                                                                 int ldr, float beta) {                                                                                           \
        gemm::gemm_nt_h<5, FKV, true, UNR, NOMMA, NS, F32, CS>(A, B, C, M, N, K, lda, ldb, ldc, alpha, R, ldr, beta);                                                                             \
    }
GEMM_H_VARIANT(8skel, 2, true, true, 1, false, false, 4)
GEMM_H_VARIANT(8f32, 2, true, false, 1, true, false, 4)
#define GEMM_H_NOEPI(SUF, NOMMA, F32)                                                                                                                                                             \
    extern "C" __global__ void __launch_bounds__(256, 2) gemm_nt_f16_h8##SUF(const half* A, const half* B, half* C, int M, int N, int K, int lda, int ldb, int ldc, float alpha) {               \
        gemm::gemm_nt_h<7, 4, true, true, NOMMA, 1, F32>(A, B, C, M, N, K, lda, ldb, ldc, alpha, nullptr, 0, 0.f);                                                                                \
    }                                                                                                                                                                                             \
    extern "C" __global__ void __launch_bounds__(256, 2) gemm_nt_res_h8##SUF(const half* A, const half* B, float* C, int M, int N, int K, int lda, int ldb, int ldc, float alpha) {              \
        gemm::gemm_nt_h<7, 4, true, true, NOMMA, 1, F32>(A, B, C, M, N, K, lda, ldb, ldc, alpha, nullptr, 0, 0.f);                                                                                \
    }                                                                                                                                                                                             \
    extern "C" __global__ void __launch_bounds__(256, 2) gemm_nt_relu2_h8##SUF(const half* A, const half* B, half* C, int M, int N, int K, int lda, int ldb, int ldc, float alpha) {             \
        gemm::gemm_nt_h<7, 4, true, true, NOMMA, 1, F32>(A, B, C, M, N, K, lda, ldb, ldc, alpha, nullptr, 0, 0.f);                                                                                \
    }
GEMM_H_NOEPI(noepi, false, false)
GEMM_H_NOEPI(f32noepi, false, true)
GEMM_H_NOEPI(skelnoepi, true, false)

// Tensor-core rate probe (`forge kbench --only f16acc`): every warp runs `iters` rounds of 16 independent m16n8k8 MMAs on register operands (no memory traffic) with fp32 or fp16 accumulators.
// The sum of the accumulators is written only when it equals a value it never takes, so the MMAs stay live without stores.
extern "C" __global__ void __launch_bounds__(256) mma_peak_f32(float* out, int iters) {
    const uint32_t a = 0x1c001c00u, b = 0x1c001c00u;   // half2(0.0039, 0.0039)
    float c[16][4];
#pragma unroll
    for (int j = 0; j < 16; j++) c[j][0] = c[j][1] = c[j][2] = c[j][3] = 0.f;
    for (int it = 0; it < iters; it++)
#pragma unroll
        for (int j = 0; j < 16; j++) gemm::mma_k8(c[j], a, a ^ (uint32_t)j, b);
    float s = 0.f;
#pragma unroll
    for (int j = 0; j < 16; j++) s += c[j][0] + c[j][1] + c[j][2] + c[j][3];
    if (s == -1.f) out[threadIdx.x] = s;
}
extern "C" __global__ void __launch_bounds__(256) mma_peak_f16(float* out, int iters) {
    const uint32_t a = 0x1c001c00u, b = 0x1c001c00u;
    uint32_t c[16][2];
#pragma unroll
    for (int j = 0; j < 16; j++) c[j][0] = c[j][1] = 0u;
    for (int it = 0; it < iters; it++)
#pragma unroll
        for (int j = 0; j < 16; j++) gemm::mma_k8h(c[j], a, a ^ (uint32_t)j, b);
    float s = 0.f;
#pragma unroll
    for (int j = 0; j < 16; j++) s += __low2float(*reinterpret_cast<const half2*>(&c[j][0])) + __high2float(*reinterpret_cast<const half2*>(&c[j][1]));
    if (s == -1.f) out[threadIdx.x] = s;
}

// ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// gemm3 with the staged epilogue (`_e2`, FORGE_GEMM_E2 in the trainer; bit-identical outputs) and, for kbench, with column groups of the tile order (`_e2gN`). Same arguments as the kernels they replace.
#define GEMM_E2_KERNEL(NAME, ACOL, BCOL, EPI, CT, GC)                                                                                                                                             \
    extern "C" __global__ void __launch_bounds__(256, 2) NAME(const half* A, const half* B, CT* C, int M, int N, int K, int lda, int ldb, int ldc, float alpha) {                               \
        gemm::gemm_impl3<ACOL, BCOL, EPI, true, true, GC>(A, B, C, M, N, K, lda, ldb, ldc, alpha, nullptr, 0, 0.f);                                                                             \
    }
#define GEMM_E2_RKERNEL(NAME, ACOL, BCOL, EPI, CT, RT, GC)                                                                                                                                        \
    extern "C" __global__ void __launch_bounds__(256, 2) NAME(const half* A, const half* B, CT* C, int M, int N, int K, int lda, int ldb, int ldc, float alpha, RT* R, int ldr, float beta) {     \
        gemm::gemm_impl3<ACOL, BCOL, EPI, true, true, GC>(A, B, C, M, N, K, lda, ldb, ldc, alpha, reinterpret_cast<const half*>(R), ldr, beta);                                                  \
    }
GEMM_E2_KERNEL(gemm_nt_f16_e2, false, false, 0, half, 0)
GEMM_E2_KERNEL(gemm_nt_f32_e2, false, false, 1, float, 0)
GEMM_E2_KERNEL(gemm_nn_f16_e2, false, true, 0, half, 0)
GEMM_E2_KERNEL(gemm_nn_f32_e2, false, true, 1, float, 0)
GEMM_E2_KERNEL(gemm_nn_add_e2, false, true, 2, float, 0)
GEMM_E2_KERNEL(gemm_tn_add_e2, true, true, 2, float, 0)
GEMM_E2_RKERNEL(gemm_nt_res_e2, false, false, 4, float, const float, 0)
GEMM_E2_RKERNEL(gemm_nt_relu2_e2, false, false, 5, half, half, 0)
GEMM_E2_RKERNEL(gemm_nn_relu2_bwd_e2, false, true, 6, half, const half, 0)
// tile-order experiments (kbench): column groups of 3 / 6 tiles on the wide forward and data-gradient GEMMs
GEMM_E2_KERNEL(gemm_nt_f16_e2g3, false, false, 0, half, 3)
GEMM_E2_RKERNEL(gemm_nt_relu2_e2g3, false, false, 5, half, half, 3)
GEMM_E2_RKERNEL(gemm_nt_relu2_e2g6, false, false, 5, half, half, 6)
GEMM_E2_RKERNEL(gemm_nn_relu2_bwd_e2g3, false, true, 6, half, const half, 3)
GEMM_E2_RKERNEL(gemm_nn_relu2_bwd_e2g6, false, true, 6, half, const half, 6)
GEMM_E2_KERNEL(gemm_nt_f32_e2g4, false, false, 1, float, 4)

namespace gemm {
// ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// gemm_w144: block tile 128 x 144 for widths that are a multiple of 144, i.e. Forge-1's model width N = 576 = 4 x 144: a 4096 x 576 GEMM is 128 blocks, 1.9 waves of the 68 block slots of a
// 2060 SUPER (34 SMs x 2), instead of gemm3's 160 blocks of 128 x 128 (2.35 waves, the third one a third full, plus a half-empty last column of tiles). 8 warps as 4 (M) x 2 (N), warp tile 32 x 72
// (2 x 9 MMA tiles, 72 fp32 accumulators); the k loop is gemm3's (BK = 16, register-staged copies two tiles ahead, two shared stages, swizzled row-major tiles; the 288 chunks of a B tile are copied by
// all threads plus warp 0). Every output element gets the same MMAs in the same k order as in gemm3: bit-identical results. A row-major [M][K]; B_COL = false: B [N][K] (forward), true: B [K][N]
// (data gradient, read with ldmatrix.trans from 304-byte rows). Epilogues EPI 0, 1, 4 with store_acc's arithmetic.
constexpr int W2N = 144, W2LD = 152;

template <int EPI>
__device__ __forceinline__ void store_acc_w144(const float (&acc)[2][9][4], void* Cv, int M, int N, int ldc, float alpha, const half* R, int ldr, int m0, int n0, int wm, int wn, int lane) {
    const int g = lane >> 2, t = lane & 3;
#pragma unroll
    for (int i = 0; i < 2; i++)
#pragma unroll
        for (int nt = 0; nt < 9; nt++)
#pragma unroll
            for (int h = 0; h < 2; h++) {
                const int row = m0 + wm + 16 * i + g + 8 * h;
                const int col = n0 + wn + 8 * nt + 2 * t;
                if (row < M && col < N) {
                    const float v0 = acc[i][nt][2 * h] * alpha, v1 = acc[i][nt][2 * h + 1] * alpha;
                    if (EPI == 0) {
                        *reinterpret_cast<half2*>(reinterpret_cast<half*>(Cv) + (size_t)row * ldc + col) = __floats2half2_rn(v0, v1);
                    } else if (EPI == 4) {
                        const float2 r = *reinterpret_cast<const float2*>(reinterpret_cast<const float*>(R) + (size_t)row * ldr + col);
                        *reinterpret_cast<float2*>(reinterpret_cast<float*>(Cv) + (size_t)row * ldc + col) = make_float2(v0 + r.x, v1 + r.y);
                    } else {
                        *reinterpret_cast<float2*>(reinterpret_cast<float*>(Cv) + (size_t)row * ldc + col) = make_float2(v0, v1);
                    }
                }
            }
}

template <bool B_COL, int EPI>
__device__ __forceinline__ void gemm_impl_w144(const half* __restrict__ A, const half* __restrict__ B, void* __restrict__ Cv, int M, int N, int K, int lda, int ldb, int ldc, float alpha,
                                               const half* __restrict__ R, int ldr) {
    constexpr int B_ELEMS = B_COL ? 16 * W2LD : W2N * 16;
    __shared__ __align__(16) half As[2][128 * 16];
    __shared__ __align__(16) half Bs[2][B_ELEMS];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = (warp >> 1) * 32, wn = (warp & 1) * 72;
    int m0, n0;
    if (gridDim.x > gridDim.y) {
        const int lin = blockIdx.y * gridDim.x + blockIdx.x;
        m0 = (lin % gridDim.y) * BM;
        n0 = (lin / gridDim.y) * W2N;
    } else {
        m0 = blockIdx.y * BM;
        n0 = blockIdx.x * W2N;
    }
    float acc[2][9][4];
#pragma unroll
    for (int i = 0; i < 2; i++)
#pragma unroll
        for (int j = 0; j < 9; j++)
#pragma unroll
            for (int e = 0; e < 4; e++) acc[i][j][e] = 0.f;

    // copies: A as gemm3 (row tid >> 1, chunk tid & 1); B chunks c = tid and, for warp 0, c = 256 + lane (288 chunks of 16 bytes per k tile)
    const int arow = tid >> 1, ach = tid & 1;
    const bool a_ok = m0 + arow < M;
    const half* pa = A + (size_t)(m0 + arow) * lda + ach * 8;
    const int a_sts = arow * 16 + 8 * (ach ^ ((arow >> 2) & 1));
    const bool w0 = warp == 0;
    bool b_ok, b2_ok;
    const half *pb, *pb2;
    int b_sts, b2_sts;
    size_t b_step;
    auto bmap = [&](int c, bool& ok, const half*& p, int& sts) {
        if (!B_COL) {
            const int row = c >> 1, ch = c & 1;
            ok = n0 + row < N;
            p = B + (size_t)(n0 + row) * ldb + ch * 8;
            sts = row * 16 + 8 * (ch ^ ((row >> 2) & 1));
        } else {
            const int row = c / 18, ch = c - 18 * (c / 18);
            ok = n0 + ch * 8 < N;
            p = B + (size_t)row * ldb + n0 + ch * 8;
            sts = row * W2LD + ch * 8;
        }
    };
    bmap(tid, b_ok, pb, b_sts);
    bmap(256 + lane, b2_ok, pb2, b2_sts);
    b_step = B_COL ? (size_t)16 * ldb : 16;
    uint4 ra = make_uint4(0u, 0u, 0u, 0u), rb = make_uint4(0u, 0u, 0u, 0u), rb2 = make_uint4(0u, 0u, 0u, 0u);
    auto ldg_tile = [&]() {
        if (a_ok) ra = __ldg(reinterpret_cast<const uint4*>(pa));
        if (b_ok) rb = __ldg(reinterpret_cast<const uint4*>(pb));
        if (w0 && b2_ok) rb2 = __ldg(reinterpret_cast<const uint4*>(pb2));
        pa += 16;
        pb += b_step;
        pb2 += b_step;
    };
    auto sts_tile = [&](int buf) {
        *reinterpret_cast<uint4*>(&As[buf][a_sts]) = ra;
        *reinterpret_cast<uint4*>(&Bs[buf][b_sts]) = rb;
        if (w0) *reinterpret_cast<uint4*>(&Bs[buf][b2_sts]) = rb2;
    };
    // ldmatrix: A as gemm3 (fragment i at rows wm + 16 i: +512 bytes); B row-major as gemm3's (16-column fragment j: +512 bytes) or k-major .trans (+32 bytes), and the last 8 columns of the warp
    // (n 64-71) with an x2 load: matrices (k 0-7) (k 8-15)
    const int q = lane >> 3, lr = lane & 7, sw = (lr >> 2) & 1;
    const uint32_t a_ld = (uint32_t)(((wm + lr + (q & 1) * 8) * 16 + 8 * ((q >> 1) ^ sw)) * 2);
    uint32_t b_ld, b_ld8;
    if (!B_COL) {
        b_ld = (uint32_t)(((wn + lr + (q >> 1) * 8) * 16 + 8 * ((q & 1) ^ sw)) * 2);
        b_ld8 = (uint32_t)(((wn + 64 + lr) * 16 + 8 * ((q & 1) ^ sw)) * 2);
    } else {
        b_ld = (uint32_t)(((((q & 1) * 8 + lr) * W2LD) + wn + (q >> 1) * 8) * 2);
        b_ld8 = (uint32_t)(((((q & 1) * 8 + lr) * W2LD) + wn + 64) * 2);
    }
    constexpr uint32_t B_FRAG = B_COL ? 32 : 512;
    const uint32_t sa0 = smem_u32(&As[0][0]), sb0 = smem_u32(&Bs[0][0]);

    const bool wact = (m0 + wm < M) && (n0 + wn < N);
    const int KT = K / 16;
    ldg_tile();
    sts_tile(0);
    if (KT > 1) ldg_tile();
    __syncthreads();
    auto ktile = [&](int kt, bool mma) {
        const int buf = kt & 1;
        if (kt + 1 < KT) {
            sts_tile(buf ^ 1);
            if (kt + 2 < KT) ldg_tile();
        }
        if (mma) {
            const uint32_t sa = sa0 + buf * (128 * 16 * 2), sb = sb0 + buf * (B_ELEMS * 2);
            uint32_t fa[2][4], fb[4][4], f8[2];
#pragma unroll
            for (int i = 0; i < 2; i++) ldsm_x4(fa[i], sa + a_ld + i * 512);
#pragma unroll
            for (int j = 0; j < 4; j++) {
                if (!B_COL) ldsm_x4(fb[j], sb + b_ld + j * B_FRAG);
                else ldsm_x4_t(fb[j], sb + b_ld + j * B_FRAG);
            }
            if (!B_COL) asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n" : "=r"(f8[0]), "=r"(f8[1]) : "r"(sb + b_ld8));
            else asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n" : "=r"(f8[0]), "=r"(f8[1]) : "r"(sb + b_ld8));
#pragma unroll
            for (int s = 0; s < 2; s++)
#pragma unroll
                for (int i = 0; i < 2; i++) {
#pragma unroll
                    for (int j = 0; j < 4; j++) {
                        mma_k8(acc[i][2 * j], fa[i][2 * s], fa[i][2 * s + 1], fb[j][s]);
                        mma_k8(acc[i][2 * j + 1], fa[i][2 * s], fa[i][2 * s + 1], fb[j][2 + s]);
                    }
                    mma_k8(acc[i][8], fa[i][2 * s], fa[i][2 * s + 1], f8[s]);
                }
        }
        __syncthreads();
    };
    if (m0 + BM <= M && n0 + W2N <= N) {
        for (int kt = 0; kt < KT; kt++) ktile(kt, true);
    } else {
        for (int kt = 0; kt < KT; kt++) ktile(kt, wact);
    }
    if (!wact) return;
    store_acc_w144<EPI>(acc, Cv, M, N, ldc, alpha, R, ldr, m0, n0, wm, wn, lane);
}
}  // namespace gemm

// 128 x 144 tiles (`_w144`, FORGE_GEMM_W144 in the trainer for N a multiple of 144; bit-identical to the 128 x 128 kernels). Same arguments; the grid is (ceil(N / 144), ceil(M / 128)).
extern "C" __global__ void __launch_bounds__(256, 2) gemm_nt_res_w144(const half* A, const half* B, float* C, int M, int N, int K, int lda, int ldb, int ldc, float alpha, const float* R, int ldr, float beta) {
    gemm::gemm_impl_w144<false, 4>(A, B, C, M, N, K, lda, ldb, ldc, alpha, reinterpret_cast<const half*>(R), ldr);
}
extern "C" __global__ void __launch_bounds__(256, 2) gemm_nt_f16_w144(const half* A, const half* B, half* C, int M, int N, int K, int lda, int ldb, int ldc, float alpha) {
    gemm::gemm_impl_w144<false, 0>(A, B, C, M, N, K, lda, ldb, ldc, alpha, nullptr, 0);
}
extern "C" __global__ void __launch_bounds__(256, 2) gemm_nn_f16_w144(const half* A, const half* B, half* C, int M, int N, int K, int lda, int ldb, int ldc, float alpha) {
    gemm::gemm_impl_w144<true, 0>(A, B, C, M, N, K, lda, ldb, ldc, alpha, nullptr, 0);
}
extern "C" __global__ void __launch_bounds__(256, 2) gemm_nn_f32_w144(const half* A, const half* B, float* C, int M, int N, int K, int lda, int ldb, int ldc, float alpha) {
    gemm::gemm_impl_w144<true, 1>(A, B, C, M, N, K, lda, ldb, ldc, alpha, nullptr, 0);
}
