// Experiment: what does a quantized matrix-vector stage cost on the RTX 2060 SUPER, compared with the fp16 stage (bench/gpu_chain.cu)? Weights are stored as separate, aligned arrays
// (int8 values row-major + one f16 scale per 32 values: the Q8_0 information in a layout made for 128-bit loads, "SoA"); the activation vector is quantized to int8 blocks INSIDE the kernel
// (every block quantizes the small input vector into shared memory itself, so no extra kernel / dependent stage is needed); products are accumulated with dp4a.
// Measured as a chain of dependent stages with different weights per stage, captured in a CUDA graph.
// Build: nvcc -O3 -arch=sm_75 -o gpu_q8_matvec bench/gpu_q8_matvec.cu      Run: ./gpu_q8_matvec
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// y[row] = sum_k W[row][k] * x[k];  K multiple of 32; WARPS warps per block, RPW rows per warp.
template <int WARPS, int RPW>
__global__ void mv_q8(const int8_t* __restrict__ Wq, const half* __restrict__ Ws, const half* __restrict__ x, half* __restrict__ y, int rows, int K) {
    extern __shared__ __align__(16) unsigned char smem[];
    int8_t* xq = reinterpret_cast<int8_t*>(smem);                  // K bytes
    float* xs = reinterpret_cast<float*>(smem + ((K + 15) / 16) * 16);   // K/32 floats
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int nb = K >> 5;
    for (int b = warp; b < nb; b += WARPS) {                       // quantize the input vector: one warp per 32-value block
        float v = __half2float(x[b * 32 + lane]);
        float amax = warp_max(fabsf(v));
        float d = amax / 127.f;
        float inv = d > 0.f ? 1.f / d : 0.f;
        xq[b * 32 + lane] = (int8_t)__float2int_rn(v * inv);
        if (lane == 0) xs[b] = d;
    }
    __syncthreads();
    const int row0 = (blockIdx.x * WARPS + warp) * RPW;
    float acc[RPW];
#pragma unroll
    for (int r = 0; r < RPW; r++) acc[r] = 0.f;
    for (int b = lane; b < nb; b += 32) {
        const int4 x0 = *reinterpret_cast<const int4*>(xq + b * 32), x1 = *reinterpret_cast<const int4*>(xq + b * 32 + 16);
        const float dx = xs[b];
        int4 w0[RPW], w1[RPW];
#pragma unroll
        for (int r = 0; r < RPW; r++) {                            // issue all loads of all rows first (memory-level parallelism), then compute
            int row = row0 + r;
            if (row < rows) {
                const int4* wp = reinterpret_cast<const int4*>(Wq + (size_t)row * K + b * 32);
                w0[r] = __ldg(wp);
                w1[r] = __ldg(wp + 1);
            }
        }
#pragma unroll
        for (int r = 0; r < RPW; r++) {
            int row = row0 + r;
            if (row < rows) {
                int s = 0;
                s = __dp4a(w0[r].x, x0.x, s); s = __dp4a(w0[r].y, x0.y, s); s = __dp4a(w0[r].z, x0.z, s); s = __dp4a(w0[r].w, x0.w, s);
                s = __dp4a(w1[r].x, x1.x, s); s = __dp4a(w1[r].y, x1.y, s); s = __dp4a(w1[r].z, x1.z, s); s = __dp4a(w1[r].w, x1.w, s);
                acc[r] += (float)s * dx * __half2float(Ws[(size_t)row * nb + b]);
            }
        }
    }
#pragma unroll
    for (int r = 0; r < RPW; r++) {
        float t = warp_sum(acc[r]);
        if (lane == 0 && row0 + r < rows) y[row0 + r] = __float2half(t * 0.01f);
    }
}

// the fp16 baseline of bench/gpu_chain.cu
__global__ void mv_half(const half* __restrict__ W, const half* __restrict__ x, half* __restrict__ y, int rows, int cols) {
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int row = blockIdx.x * (blockDim.x >> 5) + warp;
    if (row >= rows) return;
    const uint4* w4 = reinterpret_cast<const uint4*>(W + (size_t)row * cols);
    const uint4* x4 = reinterpret_cast<const uint4*>(x);
    float acc = 0.f;
    for (int i = lane; i < cols / 8; i += 32) {
        uint4 wv = __ldg(w4 + i), xv = x4[i];
        const half2* wh = reinterpret_cast<const half2*>(&wv);
        const half2* xh = reinterpret_cast<const half2*>(&xv);
#pragma unroll
        for (int k = 0; k < 4; k++) { float2 a = __half22float2(wh[k]), b = __half22float2(xh[k]); acc += a.x * b.x + a.y * b.y; }
    }
    acc = warp_sum(acc);
    if (lane == 0) y[row] = __float2half(acc * 0.01f);
}

struct Timer {
    cudaEvent_t a, b;
    Timer() { cudaEventCreate(&a); cudaEventCreate(&b); }
    void start(cudaStream_t s) { cudaEventRecord(a, s); }
    float stop(cudaStream_t s) { cudaEventRecord(b, s); cudaEventSynchronize(b); float ms; cudaEventElapsedTime(&ms, a, b); return ms; }
};

template <typename Launch>
static double run_chain(int stages, Launch launch) {
    cudaStream_t s; CK(cudaStreamCreate(&s));
    cudaGraph_t g; cudaGraphExec_t ge;
    CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
    for (int i = 0; i < stages; i++) launch(i, s);
    CK(cudaStreamEndCapture(s, &g));
    CK(cudaGraphInstantiate(&ge, g, 0));
    for (int i = 0; i < 5; i++) CK(cudaGraphLaunch(ge, s));
    CK(cudaStreamSynchronize(s));
    Timer t;
    const int reps = 100;
    t.start(s);
    for (int i = 0; i < reps; i++) CK(cudaGraphLaunch(ge, s));
    float ms = t.stop(s);
    cudaGraphExecDestroy(ge); cudaGraphDestroy(g); cudaStreamDestroy(s);
    return ms * 1e3 / (reps * (double)stages);
}

template <int WARPS, int RPW>
static double bench_q8(int rows, int K, int stages, int8_t* Wq, half* Ws, half* vec) {
    int blocks = (rows + WARPS * RPW - 1) / (WARPS * RPW);
    size_t smem = ((K + 15) / 16) * 16 + (K / 32) * 4;
    int vmax = rows > K ? rows : K;
    return run_chain(stages, [&](int i, cudaStream_t s) {
        mv_q8<WARPS, RPW><<<blocks, WARPS * 32, smem, s>>>(Wq + (size_t)i * rows * K, Ws + (size_t)i * rows * (K / 32), vec + (i & 1) * (size_t)vmax, vec + ((i + 1) & 1) * (size_t)vmax, rows, K);
    });
}

int main() {
    struct Shape { int rows, K; const char* what; int stages; };
    Shape shapes[] = {{576, 576, "135M attn-out", 180}, {768, 576, "135M fused QKV", 180}, {1536, 576, "135M gate", 180}, {3072, 576, "135M gate+up", 180}, {576, 1536, "135M down", 180},
                      {2048, 2048, "1.5B-class 2048x2048", 90}, {8192, 2048, "1.5B-class gate", 60}, {2048, 8192, "1.5B-class down", 60}};
    printf("%-22s %11s  %9s | %9s | %-52s | best vs fp16\n", "stage", "int8 bytes", "bw time", "fp16", "q8 dp4a (us per dependent stage): 2w1r 4w1r 4w2r 8w2r 4w4r");
    for (const Shape& sh : shapes) {
        size_t per = (size_t)sh.rows * sh.K;
        int8_t* Wq; half* Ws; half* W16; half* vec;
        CK(cudaMalloc(&Wq, per * sh.stages)); CK(cudaMalloc(&Ws, per / 32 * 2 * sh.stages)); CK(cudaMalloc(&W16, per * 2 * sh.stages));
        int vmax = sh.rows > sh.K ? sh.rows : sh.K;
        CK(cudaMalloc(&vec, 2 * (size_t)vmax * sizeof(half)));
        CK(cudaMemset(Wq, 1, per * sh.stages)); CK(cudaMemset(Ws, 0x3c, per / 32 * 2 * sh.stages)); CK(cudaMemset(W16, 0x3c, per * 2 * sh.stages)); CK(cudaMemset(vec, 0x3c, 2 * (size_t)vmax * sizeof(half)));
        double t16 = run_chain(sh.stages, [&](int i, cudaStream_t s) {
            mv_half<<<(sh.rows + 3) / 4, 128, 0, s>>>(W16 + (size_t)i * per, vec + (i & 1) * (size_t)vmax, vec + ((i + 1) & 1) * (size_t)vmax, sh.rows, sh.K);
        });
        double a = bench_q8<2, 1>(sh.rows, sh.K, sh.stages, Wq, Ws, vec);
        double b = bench_q8<4, 1>(sh.rows, sh.K, sh.stages, Wq, Ws, vec);
        double c = bench_q8<4, 2>(sh.rows, sh.K, sh.stages, Wq, Ws, vec);
        double d = bench_q8<8, 2>(sh.rows, sh.K, sh.stages, Wq, Ws, vec);
        double e = bench_q8<4, 4>(sh.rows, sh.K, sh.stages, Wq, Ws, vec);
        double best = a;
        for (double v : {b, c, d, e}) if (v < best) best = v;
        double bytes = (double)per + per / 32.0 * 2;
        printf("%-22s %8.2f MB  %7.2f us | %7.2f us | %10.2f %5.2f %5.2f %5.2f %5.2f                  | %.2fx faster, %.0f%% of bandwidth\n", sh.what, bytes / 1e6, bytes / 425e3, t16, a, b, c, d, e, t16 / best,
               bytes / 425e3 / best * 100.0);
        fflush(stdout);
        cudaFree(Wq); cudaFree(Ws); cudaFree(W16); cudaFree(vec);
    }
    return 0;
}
