// Experiment R2/M4: what fraction of the sustained tensor-core peak (33.5 TFLOPs fp32 accumulate, 66.4 fp16 accumulate) does a REAL inner loop reach when the fragments come from shared memory
// through ldmatrix (instead of the register-only WMMA loop of bench/gpu_sustain.cu)? Raw PTX mma.sync.m16n8k8, warp tiles of 64x64 .. 32x32, row stride 80 B (K=32 + 8 halves of padding,
// conflict free for ldmatrix). No global memory in the loop: this is the ceiling of the mainloop (issue + shared-memory bandwidth), a real GEMM adds the global->shared staging.
// Build: nvcc -O3 -arch=sm_75 -std=c++17 -o gpu_mma_ldmatrix bench/gpu_mma_ldmatrix.cu      Run: ./gpu_mma_ldmatrix
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

__device__ __forceinline__ void ldsm_x4(unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3, const void* p) {
    unsigned a = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a));
}
__device__ __forceinline__ void mma_f32(float (&c)[4], unsigned a0, unsigned a1, unsigned b0) {
    asm("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n" : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a0), "r"(a1), "r"(b0));
}
__device__ __forceinline__ void mma_f16(unsigned (&c)[2], unsigned a0, unsigned a1, unsigned b0) {
    asm("mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3}, {%4}, {%0,%1};\n" : "+r"(c[0]), "+r"(c[1]) : "r"(a0), "r"(a1), "r"(b0));
}

constexpr int LD = 40;                                           // row stride in halves (80 bytes)

template <int WM, int WN, bool ACC32>
__global__ void __launch_bounds__(128) mma_loop(float* out, int iters) {
    __shared__ __align__(16) half As[2][WM * LD];                // two copies so that the compiler cannot hoist the loads out of the loop
    __shared__ __align__(16) half Bs[2][WN * LD];
    const int lane = threadIdx.x & 31;
    for (int c = 0; c < 2; c++) {
        for (int i = threadIdx.x; i < WM * LD; i += blockDim.x) As[c][i] = __float2half(((i * 7 + c) % 13 - 6) * 0.01f);
        for (int i = threadIdx.x; i < WN * LD; i += blockDim.x) Bs[c][i] = __float2half(((i * 5 + c) % 11 - 5) * 0.01f);
    }
    __syncthreads();
    constexpr int MT = WM / 16, NT = WN / 8;
    float c32[MT][NT][4];
    unsigned c16[MT][NT][2];
#pragma unroll
    for (int m = 0; m < MT; m++)
#pragma unroll
        for (int n = 0; n < NT; n++) {
            for (int k = 0; k < 4; k++) c32[m][n][k] = 0.f;
            c16[m][n][0] = c16[m][n][1] = 0u;
        }
    // lane -> ldmatrix row address pieces
    const int q = lane >> 3, r = lane & 7;
    for (int it = 0; it < iters; it++) {
        const half* A = As[it & 1];
        const half* B = Bs[it & 1];
#pragma unroll
        for (int ks = 0; ks < 4; ks++) {                          // four k8 steps = K 32
            const int k0 = ks * 8;
            unsigned a[MT][2], b[NT];
#pragma unroll
            for (int m = 0; m < MT; m += 2) {                     // x4: two m16 tiles per instruction
                const int tile = m + (q >> 1);
                const int row = tile * 16 + (q & 1) * 8 + r;
                ldsm_x4(a[m][0], a[m][1], a[m + 1][0], a[m + 1][1], A + row * LD + k0);
            }
#pragma unroll
            for (int n = 0; n < NT; n += 4) {                     // x4: four n8 tiles per instruction
                const int row = (n + q) * 8 + r;
                ldsm_x4(b[n], b[n + 1], b[n + 2], b[n + 3], B + row * LD + k0);
            }
#pragma unroll
            for (int m = 0; m < MT; m++)
#pragma unroll
                for (int n = 0; n < NT; n++) {
                    if constexpr (ACC32) mma_f32(c32[m][n], a[m][0], a[m][1], b[n]);
                    else mma_f16(c16[m][n], a[m][0], a[m][1], b[n]);
                }
        }
    }
    float s = 0.f;
#pragma unroll
    for (int m = 0; m < MT; m++)
#pragma unroll
        for (int n = 0; n < NT; n++) {
            if constexpr (ACC32) s += c32[m][n][0] + c32[m][n][1] + c32[m][n][2] + c32[m][n][3];
            else {
                union U { unsigned u; half2 h; } u0, u1;
                u0.u = c16[m][n][0];
                u1.u = c16[m][n][1];
                s += __low2float(u0.h) + __high2float(u0.h) + __low2float(u1.h) + __high2float(u1.h);
            }
        }
    out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}

template <int WM, int WN, bool ACC32>
static void bench(int sms) {
    float* out;
    CK(cudaMalloc(&out, 4 * 128 * sms * 8));
    int maxb = 0;
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxb, mma_loop<WM, WN, ACC32>, 128, 0));
    cudaFuncAttributes fa;
    CK(cudaFuncGetAttributes(&fa, mma_loop<WM, WN, ACC32>));
    printf("warp tile %2dx%-2d %s acc  (regs %3d, max %d blocks/SM of 4 warps):", WM, WN, ACC32 ? "fp32" : "fp16", fa.numRegs, maxb);
    const int iters = 6000;
    for (int bps = 1; bps <= maxb && bps <= 4; bps++) {
        int blocks = sms * bps;
        for (int w = 0; w < 3; w++) mma_loop<WM, WN, ACC32><<<blocks, 128>>>(out, iters);
        CK(cudaDeviceSynchronize());
        cudaEvent_t a, b;
        cudaEventCreate(&a); cudaEventCreate(&b);
        cudaEventRecord(a);
        const int reps = 5;
        for (int i = 0; i < reps; i++) mma_loop<WM, WN, ACC32><<<blocks, 128>>>(out, iters);
        cudaEventRecord(b);
        cudaEventSynchronize(b);
        float ms;
        cudaEventElapsedTime(&ms, a, b);
        double flops = (double)reps * blocks * 4 /*warps*/ * iters * 4 /*k8 steps*/ * (WM / 16) * (WN / 8) * 1024.0 * 2.0;
        double tf = flops / (ms * 1e-3) / 1e12;
        printf("  %d/SM: %5.1f TFLOPs (%4.1f%% of %s)", bps, tf, 100.0 * tf / (ACC32 ? 33.5 : 66.4), ACC32 ? "33.5" : "66.4");
        cudaEventDestroy(a); cudaEventDestroy(b);
    }
    printf("\n");
    cudaFree(out);
}


// ---- INT8: mma.sync.m8n8k16 (s8 x s8 -> s32), fragments by ldmatrix (an 8x8 b16 tile = 8 rows x 16 bytes = k16 of int8) -------------------------------------------------------------
__device__ __forceinline__ void imma(int (&c)[2], unsigned a, unsigned b) {
    asm("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0,%1}, {%2}, {%3}, {%0,%1};\n" : "+r"(c[0]), "+r"(c[1]) : "r"(a), "r"(b));
}
constexpr int LD8 = 80;                                          // row stride in bytes: 64 int8 (K per iteration) + 16 padding

template <int WM, int WN>
__global__ void __launch_bounds__(128) imma_loop(int* out, int iters) {
    __shared__ __align__(16) signed char As[2][WM * LD8];
    __shared__ __align__(16) signed char Bs[2][WN * LD8];
    const int lane = threadIdx.x & 31;
    for (int c = 0; c < 2; c++) {
        for (int i = threadIdx.x; i < WM * LD8; i += blockDim.x) As[c][i] = (signed char)(((i * 7 + c) % 13) - 6);
        for (int i = threadIdx.x; i < WN * LD8; i += blockDim.x) Bs[c][i] = (signed char)(((i * 5 + c) % 11) - 5);
    }
    __syncthreads();
    constexpr int MT = WM / 8, NT = WN / 8;
    int acc[MT][NT][2];
#pragma unroll
    for (int m = 0; m < MT; m++)
#pragma unroll
        for (int n = 0; n < NT; n++) acc[m][n][0] = acc[m][n][1] = 0;
    const int q = lane >> 3, r = lane & 7;
    for (int it = 0; it < iters; it++) {
        const signed char* A = As[it & 1];
        const signed char* B = Bs[it & 1];
#pragma unroll
        for (int ks = 0; ks < 4; ks++) {                          // four k16 steps = K 64
            const int k0 = ks * 16;
            unsigned a[MT], b[NT];
#pragma unroll
            for (int m = 0; m < MT; m += 4) ldsm_x4(a[m], a[m + 1], a[m + 2], a[m + 3], A + ((m + q) * 8 + r) * LD8 + k0);
#pragma unroll
            for (int n = 0; n < NT; n += 4) ldsm_x4(b[n], b[n + 1], b[n + 2], b[n + 3], B + ((n + q) * 8 + r) * LD8 + k0);
#pragma unroll
            for (int m = 0; m < MT; m++)
#pragma unroll
                for (int n = 0; n < NT; n++) imma(acc[m][n], a[m], b[n]);
        }
    }
    int s = 0;
#pragma unroll
    for (int m = 0; m < MT; m++)
#pragma unroll
        for (int n = 0; n < NT; n++) s += acc[m][n][0] + acc[m][n][1];
    out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}

template <int WM, int WN>
static void bench_i8(int sms) {
    int* out;
    CK(cudaMalloc(&out, 4 * 128 * sms * 8));
    int maxb = 0;
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxb, imma_loop<WM, WN>, 128, 0));
    cudaFuncAttributes fa;
    CK(cudaFuncGetAttributes(&fa, imma_loop<WM, WN>));
    printf("warp tile %2dx%-2d int8 acc32 (regs %3d, max %d blocks/SM of 4 warps):", WM, WN, fa.numRegs, maxb);
    const int iters = 3000;
    for (int bps = 1; bps <= maxb && bps <= 4; bps++) {
        int blocks = sms * bps;
        for (int w = 0; w < 3; w++) imma_loop<WM, WN><<<blocks, 128>>>(out, iters);
        CK(cudaDeviceSynchronize());
        cudaEvent_t a, b;
        cudaEventCreate(&a); cudaEventCreate(&b);
        cudaEventRecord(a);
        const int reps = 5;
        for (int i = 0; i < reps; i++) imma_loop<WM, WN><<<blocks, 128>>>(out, iters);
        cudaEventRecord(b);
        cudaEventSynchronize(b);
        float ms;
        cudaEventElapsedTime(&ms, a, b);
        double ops = (double)reps * blocks * 4 * iters * 4 /*k16 steps*/ * (WM / 8) * (WN / 8) * 1024.0 * 2.0;
        double tops = ops / (ms * 1e-3) / 1e12;
        printf("  %d/SM: %6.1f TOPS (%4.1f%% of 135)", bps, tops, 100.0 * tops / 135.0);
        cudaEventDestroy(a); cudaEventDestroy(b);
    }
    printf("\n");
    cudaFree(out);
}

int main() {
    cudaDeviceProp p;
    CK(cudaGetDeviceProperties(&p, 0));
    int sms = p.multiProcessorCount;
    printf("sustained peaks measured earlier with the register-only WMMA loop: 33.5 TFLOPs (fp32 accumulate), 66.4 (fp16 accumulate)\n");
    bench<64, 64, true>(sms);
    bench<64, 32, true>(sms);
    bench<32, 64, true>(sms);
    bench<32, 32, true>(sms);
    bench<64, 64, false>(sms);
    bench<64, 32, false>(sms);
    bench<32, 64, false>(sms);
    bench<32, 32, false>(sms);
    printf("INT8 tensor peak measured earlier with WMMA: 135 TOPS\n");
    bench_i8<64, 64>(sms);
    bench_i8<64, 32>(sms);
    bench_i8<32, 64>(sms);
    bench_i8<32, 32>(sms);
    return 0;
}
