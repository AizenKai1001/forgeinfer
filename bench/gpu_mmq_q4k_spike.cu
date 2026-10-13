// Spike v9: v8 with the weights kept PACKED (4 bits) in shared memory: they arrive by cp.async like the activations (no register staging, no unpack-and-store phase), the nibbles are masked out of the
// ldmatrix fragments (one LOP3 per register; the high nibbles unshifted: u8 weights 16 q, the factor 1/16 is in the scale), one ldmatrix serves the two sub-blocks of a 32-byte chunk.
// Spike v8 (Q4_K, one activation scale per 32): v7 with the per-super-block scale work taken off the critical path.
//  - the token side (scale per 32 and the fp16 minimum operand a_d * sum) is prepared by the quantizer in the exact shared-memory layout and copied by cp.async: no ALU work in the GEMM,
//  - the weight side decodes the header with two threads per column (4 sub-blocks each), integer-to-float by PRMT onto 0x4B000000 and one FFMA, the minimum operand by one HFMA2 per two sub-blocks.
//   nvcc -O3 -arch=sm_89 -Xptxas -v -o gpu_mmq_q4k_spike gpu_mmq_q4k_spike.cu && ./gpu_mmq_q4k_spike [M N K]
// Stand-alone copy of the Q4_K kernel of crates/forge-kernels/kernels/gemm_mmq.cu (random weights, a host reference over 64 random outputs, timing of 10 launches x 5): 1.95 ms for 1024 x 12288 x 4096 = 26.4 TMAC/s on an RTX 4060.
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(1); } } while (0)

constexpr int BM = 128, BN = 128, BK = 128, NT = 256, ROW = BK + 16, WM = 64, WN = 32, MT = WM / 16, NTL = WN / 8, SB = 256;
#define M_LO 0x0F0F0F0Fu
#define M_K1 0x3F3F3F3Fu
#define M_K3 0x03030303u

__host__ __device__ __forceinline__ int tok_pos(int tl) { const int slab = tl / WM, r = tl % WM, mt = r / 16, rr = r % 16; return slab * WM + (rr % 8) * (2 * MT) + mt * 2 + (rr / 8); }
__host__ __device__ __forceinline__ int col_pos(int n) { const int slab = n / WN, r = n % WN, nt = r / 8, rr = r % 8; return slab * WN + (rr / 2) * (2 * NTL) + nt * 2 + (rr % 2); }

__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void ldsm4(uint32_t (&r)[4], uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}
__device__ __forceinline__ void mma_s8_c(int (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1, int c) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.u8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};\n"
                 : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(c));
}
__device__ __forceinline__ void mma_f16_k8(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t b0) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(a1), "r"(0u), "r"(0u), "r"(b0), "r"(0u));
}
__device__ __forceinline__ void cp16(uint32_t dst, const void* src) { asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst), "l"(src)); }
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
__device__ __forceinline__ void cp_wait0() { asm volatile("cp.async.wait_group 0;\n" ::); }

constexpr int BROW = 80;       // bytes per row of the packed weights of a stage: 64 + 16 (an odd number of 16-byte pieces: conflict-free ldmatrix)
struct __align__(16) Smem {
    int8_t a[2][BM][ROW];
    uint8_t bp[2][BN][BROW];
    float dsc[2][8][BN];
    __half wmh[2][BN][8];
    float adp[2][8][BM];
    __half ash[2][BM][8];
};

// aq int8 [M][K]; adT float [M / 128][K / 256][8][128] (token position tok_pos); ashT half [M / 128][K / 256][128][8]; p0 [N][K / 256][128]; p1 [N][K / 256][16]
__global__ void __launch_bounds__(NT, 1) mmq_q4k_spike(const int8_t* __restrict__ aq, const float* __restrict__ adT, const __half* __restrict__ ashT, const uint8_t* __restrict__ p0, const uint8_t* __restrict__ p1,
                                                float* __restrict__ C, int M, int N, int K) {
    extern __shared__ __align__(16) unsigned char smem_raw[];
    Smem& sm = *reinterpret_cast<Smem*>(smem_raw);
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
            cp16(smem_u32(&sm.a[buf][r][cc * 16]), aq + (size_t)(m0 + r) * K + kcol + cc * 16);
        }
    };
    // the token scales and minimum operands of super-block sb (one cp.async of 16 bytes per thread for the scales, one per thread of the first half for the minimum operands)
    auto load_tok = [&](int sb, int buf) {
        const size_t tile = (size_t)blockIdx.y * nb + sb;
        cp16(smem_u32(&sm.adp[buf][tid >> 5][(tid & 31) * 4]), adT + (tile * 8 + (tid >> 5)) * BM + (tid & 31) * 4);
        if (tid < BM) cp16(smem_u32(&sm.ash[buf][tid][0]), ashT + (tile * BM + tid) * 8);
    };
    // the packed weights of a stage: 64 bytes (two 32-byte chunks) of each of the 128 rows, two 16-byte pieces per thread
    auto load_b = [&](int st, int buf) {
#pragma unroll
        for (int u = 0; u < BN * 4 / NT; u++) {
            const int idx = tid + NT * u;
            const int r = idx >> 2, q = idx & 3;
            const size_t blk = (size_t)min(n0 + r, N - 1) * nb + (st >> 1);
            cp16(smem_u32(&sm.bp[buf][r][q * 16]), p0 + blk * 128 + (st & 1) * 64 + q * 16);
        }
    };
    // the weight headers: thread (col, half) decodes sub-blocks 4 half .. 4 half + 3 of output column col
    uint4 hreg;
    const int hcol = tid & (BN - 1), hhalf = tid >> 7;
    auto ldg_hdr = [&](int sb) { hreg = *reinterpret_cast<const uint4*>(p1 + ((size_t)min(n0 + hcol, N - 1) * nb + sb) * 16); };
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
            const float de = (e & 1) ? d * 0.0625f : d;                                          // the odd sub-blocks are the high nibbles: weights 16 q
            const float scf = __int_as_float((int)__byte_perm(sw, 0x4B000000u, 0x7440 + e));     // 2^23 + sc
            sm.dsc[buf][4 * hhalf + e][pc] = fmaf(scf, de, -8388608.f * de);                      // d sc (/ 16)
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
            uint32_t fbp[NTL / 2][4];
#pragma unroll
            for (int nj = 0; nj < NTL / 2; nj++)
                ldsm4(fbp[nj], smem_u32(&sm.bp[cur][wn + nj * 16 + lr + (q >> 1) * 8][cl * 32 + (q & 1) * 16]));
#pragma unroll
            for (int nib = 0; nib < 2; nib++) {
                const int jl = cl * 2 + nib;
                const int j = (s & 1) * 4 + jl;
                uint32_t fa[MT][4], fb[NTL / 2][4];
#pragma unroll
                for (int mt = 0; mt < MT; mt++)
                    ldsm4(fa[mt], smem_u32(&sm.a[cur][wm + mt * 16 + lr + (q & 1) * 8][jl * 32 + (q >> 1) * 16]));
#pragma unroll
                for (int nj = 0; nj < NTL / 2; nj++)
#pragma unroll
                    for (int e = 0; e < 4; e++) fb[nj][e] = fbp[nj][e] & (nib ? 0xF0F0F0F0u : 0x0F0F0F0Fu);
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
                        mma_s8_c(sr, fa[mt], fb[nt >> 1][(nt & 1) * 2], fb[nt >> 1][(nt & 1) * 2 + 1], 0x4B400000);
                        acc[mt][nt][0] = fmaf(adv[mt * 2], fmaf(__int_as_float(sr[0]), ds[nt * 2], nds[nt * 2]), acc[mt][nt][0]);
                        acc[mt][nt][1] = fmaf(adv[mt * 2], fmaf(__int_as_float(sr[1]), ds[nt * 2 + 1], nds[nt * 2 + 1]), acc[mt][nt][1]);
                        acc[mt][nt][2] = fmaf(adv[mt * 2 + 1], fmaf(__int_as_float(sr[2]), ds[nt * 2], nds[nt * 2]), acc[mt][nt][2]);
                        acc[mt][nt][3] = fmaf(adv[mt * 2 + 1], fmaf(__int_as_float(sr[3]), ds[nt * 2 + 1], nds[nt * 2 + 1]), acc[mt][nt][3]);
                    }
                }
            }
        }
        if (s & 1) {
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
#pragma unroll
    for (int mt = 0; mt < MT; mt++)
#pragma unroll
        for (int nt = 0; nt < NTL; nt++) {
            const int row = m0 + wm + mt * 16 + g, col = n0 + wn + nt * 8 + 2 * t;
            if (row < M && col < N) *reinterpret_cast<float2*>(C + (size_t)row * N + col) = make_float2(acc[mt][nt][0], acc[mt][nt][1]);
            if (row + 8 < M && col < N) *reinterpret_cast<float2*>(C + (size_t)(row + 8) * N + col) = make_float2(acc[mt][nt][2], acc[mt][nt][3]);
        }
}

// ---- host ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
static uint64_t rng_s = 88172645463325252ull;
static uint32_t rnd32() { rng_s ^= rng_s << 13; rng_s ^= rng_s >> 7; rng_s ^= rng_s << 17; return (uint32_t)(rng_s >> 16); }
static float rndf() { return ((rnd32() & 0xFFFFFF) / 16777216.0f) * 2.f - 1.f; }

static double weight_ref(const std::vector<uint8_t>& p0, const std::vector<uint8_t>& p1, int nb, int n, int k) {
    const size_t blk = (size_t)n * nb + k / SB;
    const int kk = k % SB, j = kk / 32, cidx = j / 2;
    const uint8_t byte = p0[blk * 128 + cidx * 32 + (kk % 32)];
    const int q = (j & 1) ? (byte >> 4) : (byte & 15);
    const uint8_t* h = &p1[blk * 16];
    __half dh, dmh;
    memcpy(&dh, h, 2); memcpy(&dmh, h + 2, 2);
    uint32_t u0, u1, u2;
    memcpy(&u0, h + 4, 4); memcpy(&u1, h + 8, 4); memcpy(&u2, h + 12, 4);
    const uint32_t sw = j < 4 ? (u0 & M_K1) : ((u2 & M_LO) | (((u0 >> 6) & M_K3) << 4));
    const uint32_t mw = j < 4 ? (u1 & M_K1) : (((u2 >> 4) & M_LO) | (((u1 >> 6) & M_K3) << 4));
    const double sc = (sw >> (8 * (j & 3))) & 0xFF, mn = (mw >> (8 * (j & 3))) & 0xFF;
    return (double)__half2float(dh) * sc * q - (double)__half2float(dmh) * mn;
}

int main(int argc, char** argv) {
    const int M = argc > 1 ? atoi(argv[1]) : 1024, N = argc > 2 ? atoi(argv[2]) : 12288, K = argc > 3 ? atoi(argv[3]) : 4096;
    const int nb = K / SB, tiles = M / BM;
    printf("gpu_mmq_q4k_spike: M %d N %d K %d\n", M, N, K);
    std::vector<uint8_t> p0((size_t)N * nb * 128), p1((size_t)N * nb * 16);
    for (auto& x : p0) x = (uint8_t)rnd32();
    for (size_t blk = 0; blk < (size_t)N * nb; blk++) {
        uint8_t* h = &p1[blk * 16];
        const __half d = __float2half(0.004f * (1.f + rndf() * 0.5f)), dm = __float2half(0.002f * (1.f + rndf() * 0.5f));
        memcpy(h, &d, 2);
        memcpy(h + 2, &dm, 2);
        for (int i = 4; i < 16; i++) h[i] = (uint8_t)rnd32();
    }
    std::vector<float> af((size_t)M * K);
    for (auto& x : af) x = rndf() * (1.f + 3.f * (rnd32() % 97 == 0));
    std::vector<int8_t> aq((size_t)M * K);
    std::vector<float> adv((size_t)M * K / 32);                   // reference layout [m][K / 32]
    std::vector<float> adT((size_t)M * K / 32);
    std::vector<__half> ashT((size_t)M * K / 32);
    for (int m = 0; m < M; m++)
        for (int b = 0; b < nb; b++)
            for (int j = 0; j < 8; j++) {
                const size_t base = (size_t)m * K + b * SB + j * 32;
                float amax = 0.f;
                for (int k = 0; k < 32; k++) amax = std::max(amax, std::fabs(af[base + k]));
                const float d = amax / 127.f, id = d > 0 ? 1.f / d : 0.f;
                const float dr = __half2float(__float2half(d));                // the engine stores the fp16-rounded scale
                int sum = 0;
                for (int k = 0; k < 32; k++) {
                    const int q = (int)lrintf(af[base + k] * id);
                    aq[base + k] = (int8_t)q;
                    sum += q;
                }
                adv[(size_t)m * (K / 32) + b * 8 + j] = dr;
                const int tile = m / BM, r = m % BM;
                adT[(((size_t)tile * nb + b) * 8 + j) * BM + tok_pos(r)] = dr;
                ashT[(((size_t)tile * nb + b) * BM + r) * 8 + j] = __float2half(dr * (float)sum);
            }
    int8_t* d_aq; float *d_adT, *d_c; __half* d_ash; uint8_t *d_p0, *d_p1;
    CK(cudaMalloc(&d_aq, aq.size())); CK(cudaMalloc(&d_adT, adT.size() * 4)); CK(cudaMalloc(&d_ash, ashT.size() * 2));
    CK(cudaMalloc(&d_p0, p0.size())); CK(cudaMalloc(&d_p1, p1.size())); CK(cudaMalloc(&d_c, (size_t)M * N * 4));
    CK(cudaMemcpy(d_aq, aq.data(), aq.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_adT, adT.data(), adT.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_ash, ashT.data(), ashT.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_p0, p0.data(), p0.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_p1, p1.data(), p1.size(), cudaMemcpyHostToDevice));
    const size_t smem = sizeof(Smem);
    CK(cudaFuncSetAttribute(mmq_q4k_spike, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    dim3 grid((N + BN - 1) / BN, tiles);
    auto run = [&]() { mmq_q4k_spike<<<grid, NT, smem>>>(d_aq, d_adT, d_ash, d_p0, d_p1, d_c, M, N, K); };
    run();
    CK(cudaDeviceSynchronize());
    std::vector<float> c((size_t)M * N);
    CK(cudaMemcpy(c.data(), d_c, c.size() * 4, cudaMemcpyDeviceToHost));
    double worst = 0, scale = 0;
    for (int trial = 0; trial < 64; trial++) {
        const int m = rnd32() % M, n = rnd32() % N;
        double ref = 0;
        for (int k = 0; k < K; k++) ref += weight_ref(p0, p1, nb, n, k) * ((double)aq[(size_t)m * K + k] * (double)adv[(size_t)m * (K / 32) + k / 32]);
        const double got = c[(size_t)m * N + n];
        worst = std::max(worst, std::fabs(got - ref));
        scale = std::max(scale, std::fabs(ref));
    }
    printf("max |diff| over 64 random outputs %.3e (max |value| %.3f, relative %.2e)\n", worst, scale, worst / scale);
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    for (int pass = 0; pass < 3; pass++) {
        float best = 1e30f;
        for (int rep = 0; rep < 5; rep++) {
            cudaEventRecord(e0);
            for (int i = 0; i < 10; i++) run();
            cudaEventRecord(e1);
            cudaEventSynchronize(e1);
            float ms;
            cudaEventElapsedTime(&ms, e0, e1);
            best = std::min(best, ms / 10);
        }
        const double macs = (double)M * N * K;
        cudaFuncAttributes fa;
        cudaFuncGetAttributes(&fa, mmq_q4k_spike);
        printf("kernel: smem %zu B, %d regs, spill %zu B: %.3f ms  %.2f TMAC/s (%.1f TOPS)\n", smem, fa.numRegs, fa.localSizeBytes, best, macs / (best * 1e-3) / 1e12, 2 * macs / (best * 1e-3) / 1e12);
    }
    return 0;
}
