// Tensor-core GEMM for sm_80+ (Ampere, Ada, Hopper): the same three layouts, epilogues and tile order as gemm.cu (kernel names with the suffix _s80, same arguments), built the way an Ampere GEMM is:
//  - block tile 128 x 128 x 32, 8 warps (2 x 4), warp tile 64 x 32, a THREE-STAGE pipeline of cp.async copies (global -> shared without passing through registers, the next two tiles are in flight while one is
//    consumed), one __syncthreads per k tile of 32, dynamic shared memory (52,224 bytes: the host sets the attribute and passes the size),
//  - raw PTX mma.sync.m16n8k16 (one instruction per 16 k instead of two m16n8k8: the k8 form runs at half the tensor-core rate on Ampere), fed by ldmatrix,
//  - row-major tiles are 64 bytes per row with an XOR swizzle of the four 16-byte chunks (chunk ^ ((row >> 1) & 3): conflict-free for the cp.async stores and the ldmatrix reads), k-major tiles keep the
//    136-halves row stride of gemm.cu (ldmatrix.trans).
//  - K must be a multiple of 32; M and N as in gemm.cu (arbitrary for row-major operands, multiples of 8 for the k-major ones).
// On devices below sm_80 the same source compiles with an EMULATION of the two new instructions (a synchronous copy, two k8 MMAs): slow, but it runs the identical indexing, swizzle and pipeline logic, so the
// kernels can be checked on a Turing GPU (the trainer uses them there only with FORGE_S80=1, for tests).

#include <cuda_fp16.h>
#include <stdint.h>

namespace gemm80 {

constexpr int BM = 128, BN = 128, BK = 32, STAGES = 3, LDM = 136;
constexpr int ROWMAJ = 128 * 32;       // halves of a row-major tile [128][32]
constexpr int KMAJ = 32 * LDM;         // halves of a k-major tile [32][136]
constexpr int SMEM_BYTES = STAGES * 2 * KMAJ * 2;   // the largest variant (both operands k-major): 52,224 bytes

#if __CUDA_ARCH__ >= 800
__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src, bool valid) {
    const int sz = valid ? 16 : 0;   // src-size 0: nothing is read and the 16 bytes are filled with zeros
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(sz));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_wait() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}
__device__ __forceinline__ void mma16(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
// The same product with an fp16 result and an fp16 accumulator that starts at zero: c = (row g, cols 2t and 2t+1) and (row g + 8, ...) as half2. On GeForce cards the tensor cores run this form at twice the
// rate of the fp32-accumulate one (measured on the RTX 4060: 61.4 against 30.3 TFLOPs); the caller adds the half2 pair into its fp32 accumulators after every MMA, so a partial sum holds 16 products.
__device__ __forceinline__ void mma16h0(uint32_t (&c)[2], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%8,%8};\n"
                 : "=r"(c[0]), "=r"(c[1])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(0u));
}
#else
__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src, bool valid) {
    const uint4 v = valid ? *reinterpret_cast<const uint4*>(src) : make_uint4(0u, 0u, 0u, 0u);
    asm volatile("st.shared.v4.u32 [%0], {%1, %2, %3, %4};\n" ::"r"(dst), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w));
}
__device__ __forceinline__ void cp_commit() {}
template <int N>
__device__ __forceinline__ void cp_wait() {}
__device__ __forceinline__ void mma16(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    gemm::mma_k8(c, a[0], a[1], b0);
    gemm::mma_k8(c, a[2], a[3], b1);
}
#endif

// split-K epilogue of the weight-gradient kernels: C += alpha * acc with atomics (several blocks add their slice of the reduction into the same tile)
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

// A_COL: A stored [K][M] (k-major tile); B_COL: B stored [K][N]
// Split-K: with gridDim.z > 1 (EPI 2 only) block z reduces the slice [z K / gridDim.z, (z + 1) K / gridDim.z) (each slice a multiple of 32) and adds its result with atomics.
// HACC (sm_80 and newer; older devices run the fp32 form): the MMAs accumulate in fp16, one k16 step (16 products) per partial sum, and the partial sums are added into the fp32 accumulators after every MMA. The
// fp16-accumulate form runs at twice the tensor-core rate on GeForce Ampere and Ada (41 against 31 TFLOPs on the RTX 4060 for these tiles); the result differs from the fp32 one by 1.8e-4 of the rms (below half
// an fp16 rounding step of the output).
template <bool A_COL, bool B_COL, int EPI, bool HACC = false>
__device__ __forceinline__ void gemm_impl80(const half* __restrict__ A, const half* __restrict__ B, void* __restrict__ Cv, int M, int N, int K, int lda, int ldb, int ldc, float alpha, const half* __restrict__ R,
                                            int ldr, float beta) {
    constexpr int A_ELEMS = A_COL ? KMAJ : ROWMAJ;
    constexpr int B_ELEMS = B_COL ? KMAJ : ROWMAJ;
    constexpr int STAGE = A_ELEMS + B_ELEMS;   // halves per pipeline stage of this variant
    extern __shared__ __align__(16) unsigned char smem_raw[];
    half* sm = reinterpret_cast<half*>(smem_raw);
    const int kz = gridDim.z;
    if (kz > 1) {
        K /= kz;
        A += A_COL ? (size_t)blockIdx.z * K * lda : (size_t)blockIdx.z * K;
        B += B_COL ? (size_t)blockIdx.z * K * ldb : (size_t)blockIdx.z * K;
    }
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = (warp >> 2) * 64, wn = (warp & 3) * 32;
    int m0, n0;
    if (gridDim.x > gridDim.y) {   // tile order: see gemm.cu
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

    // global -> shared mapping: two 16-byte chunks per operand per stage and thread
    const half* pa[2];
    const half* pb[2];
    int sa_off[2], sb_off[2];
    bool a_ok[2], b_ok[2];
#pragma unroll
    for (int u = 0; u < 2; u++) {
        const int c = tid + 256 * u;
        if (!A_COL) {
            const int row = c >> 2, ch = c & 3;
            a_ok[u] = m0 + row < M;
            pa[u] = A + (size_t)(m0 + row) * lda + ch * 8;
            sa_off[u] = row * 32 + ((ch ^ ((row >> 1) & 3)) * 8);
        } else {
            const int row = c >> 4, ch = c & 15;
            a_ok[u] = m0 + ch * 8 < M;
            pa[u] = A + (size_t)row * lda + m0 + ch * 8;
            sa_off[u] = row * LDM + ch * 8;
        }
        if (!B_COL) {
            const int row = c >> 2, ch = c & 3;
            b_ok[u] = n0 + row < N;
            pb[u] = B + (size_t)(n0 + row) * ldb + ch * 8;
            sb_off[u] = row * 32 + ((ch ^ ((row >> 1) & 3)) * 8);
        } else {
            const int row = c >> 4, ch = c & 15;
            b_ok[u] = n0 + ch * 8 < N;
            pb[u] = B + (size_t)row * ldb + n0 + ch * 8;
            sb_off[u] = row * LDM + ch * 8;
        }
    }
    const uint32_t smem0 = gemm::smem_u32(sm);
    auto load_stage = [&](int stage, int kt) {
        const uint32_t sa = smem0 + (uint32_t)(stage * STAGE) * 2u, sb = sa + (uint32_t)A_ELEMS * 2u;
        const size_t k0 = (size_t)kt * BK;
#pragma unroll
        for (int u = 0; u < 2; u++) {
            cp_async16(sa + (uint32_t)sa_off[u] * 2u, A_COL ? pa[u] + k0 * lda : pa[u] + k0, a_ok[u]);
            cp_async16(sb + (uint32_t)sb_off[u] * 2u, B_COL ? pb[u] + k0 * ldb : pb[u] + k0, b_ok[u]);
        }
    };

    // ldmatrix lane addressing (bytes, inside one stage). q selects the 8x8 matrix of the x4 load, lr the row inside it.
    const int q = lane >> 3, lr = lane & 7, swz = (lr >> 1) & 3;
    uint32_t a_base, b_base;   // row-major: byte offset of the lane's row; k-major: byte offset of the lane's (k row, column) for ks = 0
    if (!A_COL) a_base = (uint32_t)((wm + lr + (q & 1) * 8) * 32) * 2u;
    else a_base = (uint32_t)((((q >> 1) * 8 + lr) * LDM + wm + (q & 1) * 8) * 2);
    if (!B_COL) b_base = (uint32_t)((wn + lr + (q >> 1) * 8) * 32) * 2u;
    else b_base = (uint32_t)(((((q & 1) * 8 + lr) * LDM) + wn + (q >> 1) * 8) * 2);
    constexpr uint32_t A_FRAG = A_COL ? 32 : 1024, B_FRAG = B_COL ? 32 : 1024;   // bytes between consecutive 16-row / 16-column fragments

    const int KT = K / BK;
#pragma unroll
    for (int s = 0; s < STAGES - 1; s++) {
        if (s < KT) load_stage(s, s);
        cp_commit();
    }
    for (int kt = 0; kt < KT; kt++) {
        cp_wait<STAGES - 2>();   // tile kt has arrived
        __syncthreads();         // ... for every thread, and everybody has finished with the stage that the next load overwrites
        {
            const int nk = kt + STAGES - 1;
            if (nk < KT) load_stage(nk % STAGES, nk);
            cp_commit();
        }
        const int stage = kt % STAGES;
        const uint32_t sa = smem0 + (uint32_t)(stage * STAGE) * 2u, sb = sa + (uint32_t)A_ELEMS * 2u;
#pragma unroll
        for (int ks = 0; ks < 2; ks++) {
            uint32_t fa[4][4], fb[2][4];
#pragma unroll
            for (int i = 0; i < 4; i++) {
                if (!A_COL) gemm::ldsm_x4(fa[i], sa + a_base + (uint32_t)(((2 * ks + (q >> 1)) ^ swz) * 16) + i * A_FRAG);
                else gemm::ldsm_x4_t(fa[i], sa + a_base + (uint32_t)(ks * 16 * LDM * 2) + i * A_FRAG);
            }
#pragma unroll
            for (int j = 0; j < 2; j++) {
                if (!B_COL) gemm::ldsm_x4(fb[j], sb + b_base + (uint32_t)(((2 * ks + (q & 1)) ^ swz) * 16) + j * B_FRAG);
                else gemm::ldsm_x4_t(fb[j], sb + b_base + (uint32_t)(ks * 16 * LDM * 2) + j * B_FRAG);
            }
#pragma unroll
            for (int i = 0; i < 4; i++)
#pragma unroll
                for (int j = 0; j < 2; j++) {
#if __CUDA_ARCH__ >= 800
                    if constexpr (HACC) {
                        uint32_t h0[2], h1[2];
                        mma16h0(h0, fa[i], fb[j][0], fb[j][1]);
                        mma16h0(h1, fa[i], fb[j][2], fb[j][3]);
                        const float2 p0 = __half22float2(*reinterpret_cast<const __half2*>(&h0[0])), p1 = __half22float2(*reinterpret_cast<const __half2*>(&h0[1]));
                        const float2 p2 = __half22float2(*reinterpret_cast<const __half2*>(&h1[0])), p3 = __half22float2(*reinterpret_cast<const __half2*>(&h1[1]));
                        acc[i][2 * j][0] += p0.x;
                        acc[i][2 * j][1] += p0.y;
                        acc[i][2 * j][2] += p1.x;
                        acc[i][2 * j][3] += p1.y;
                        acc[i][2 * j + 1][0] += p2.x;
                        acc[i][2 * j + 1][1] += p2.y;
                        acc[i][2 * j + 1][2] += p3.x;
                        acc[i][2 * j + 1][3] += p3.y;
                    } else
#endif
                    {
                        mma16(acc[i][2 * j], fa[i], fb[j][0], fb[j][1]);
                        mma16(acc[i][2 * j + 1], fa[i], fb[j][2], fb[j][3]);
                    }
                }
        }
    }
    if (EPI == 2 && kz > 1) store_add_atomic(acc, reinterpret_cast<float*>(Cv), M, N, ldc, alpha, m0, n0, wm, wn, lane);
    else gemm::store_acc<EPI>(acc, Cv, M, N, ldc, alpha, R, ldr, beta, m0, n0, wm, wn, lane);
}

}  // namespace gemm80

#define GEMM80_KERNEL(NAME, ACOL, BCOL, EPI, CT)                                                                                                                                                         \
    extern "C" __global__ void __launch_bounds__(256, 2) NAME##_s80(const half* A, const half* B, CT* C, int M, int N, int K, int lda, int ldb, int ldc, float alpha) {                                \
        gemm80::gemm_impl80<ACOL, BCOL, EPI>(A, B, C, M, N, K, lda, ldb, ldc, alpha, nullptr, 0, 0.f);                                                                                              \
    }

GEMM80_KERNEL(gemm_nt_f16, false, false, 0, half)
GEMM80_KERNEL(gemm_nt_f32, false, false, 1, float)
GEMM80_KERNEL(gemm_nt_add, false, false, 2, float)
GEMM80_KERNEL(gemm_nn_f16, false, true, 0, half)
GEMM80_KERNEL(gemm_nn_f32, false, true, 1, float)
GEMM80_KERNEL(gemm_nn_add, false, true, 2, float)
GEMM80_KERNEL(gemm_tn_f32, true, true, 1, float)
GEMM80_KERNEL(gemm_tn_add, true, true, 2, float)
GEMM80_KERNEL(gemm_tn_f16, true, true, 0, half)

// The inference engine's prompt GEMMs after a weight matrix has been dequantized to fp16 (NT layout: A [M][K] activations, B [N][K] weights, both contiguous): fp16 accumulation, see HACC above
#define GEMM80H_KERNEL(NAME, EPI, CT)                                                                                                                                                                     \
    extern "C" __global__ void __launch_bounds__(256, 2) NAME##_s80h(const half* A, const half* B, CT* C, int M, int N, int K, int lda, int ldb, int ldc, float alpha) {                               \
        gemm80::gemm_impl80<false, false, EPI, true>(A, B, C, M, N, K, lda, ldb, ldc, alpha, nullptr, 0, 0.f);                                                                                      \
    }
GEMM80H_KERNEL(gemm_nt_f16, 0, half)
GEMM80H_KERNEL(gemm_nt_f32, 1, float)
GEMM80H_KERNEL(gemm_nt_add, 2, float)

// axpy epilogue (EPI 3): C = alpha * (A B) + beta * R, all fp16
#define GEMM80_AXPY_KERNEL(NAME, ACOL, BCOL)                                                                                                                                                             \
    extern "C" __global__ void __launch_bounds__(256, 2) NAME##_s80(const half* A, const half* B, half* C, int M, int N, int K, int lda, int ldb, int ldc, float alpha, const half* R, int ldr, float beta) {   \
        gemm80::gemm_impl80<ACOL, BCOL, 3>(A, B, C, M, N, K, lda, ldb, ldc, alpha, R, ldr, beta);                                                                                                   \
    }
GEMM80_AXPY_KERNEL(gemm_nt_axpy, false, false)
GEMM80_AXPY_KERNEL(gemm_nn_axpy, false, true)

// fp32 output with a separate fp32 residual (EPI 4): C = alpha * (A B) + R
#define GEMM80_RES_KERNEL(NAME, ACOL, BCOL)                                                                                                                                                              \
    extern "C" __global__ void __launch_bounds__(256, 2) NAME##_s80(const half* A, const half* B, float* C, int M, int N, int K, int lda, int ldb, int ldc, float alpha, const float* R, int ldr, float beta) { \
        gemm80::gemm_impl80<ACOL, BCOL, 4>(A, B, C, M, N, K, lda, ldb, ldc, alpha, reinterpret_cast<const half*>(R), ldr, beta);                                                                    \
    }
GEMM80_RES_KERNEL(gemm_nt_res, false, false)
