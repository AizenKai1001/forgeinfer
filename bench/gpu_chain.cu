// How long does ONE dependent stage of a decode step really take on the homelab GPU (RTX 2060 SUPER)? This decides between "fused kernels + CUDA graph" and a persistent megakernel.
//  1. a chain of N dependent fp16 matrix-vector products with DIFFERENT weights per stage (so every stage streams its weights from DRAM, as a real model does), captured in a CUDA graph
//     -> microseconds per stage, compared with the pure bandwidth time of the weights;
//  2. the cost of a grid-wide barrier in one cooperative persistent kernel (cooperative groups grid.sync()), for several grid sizes.
// Build: nvcc -O3 -arch=sm_75 -rdc=true -o gpu_chain bench/gpu_chain.cu      Run: ./gpu_chain
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

// y[row] = dot(W[row, :], x); one warp per row, 128-bit loads. cols must be a multiple of 8.
__global__ void matvec_half(const half* __restrict__ W, const half* __restrict__ x, half* __restrict__ y, int rows, int cols) {
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
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    if (lane == 0) y[row] = __float2half(acc * 0.01f);       // scaled so that values stay bounded along the chain
}

__global__ void barrier_loop(int iters, int* sink) {
    cg::grid_group g = cg::this_grid();
    for (int i = 0; i < iters; i++) g.sync();
    if (blockIdx.x == 0 && threadIdx.x == 0) sink[0] = iters;
}

struct Timer {
    cudaEvent_t a, b;
    Timer() { cudaEventCreate(&a); cudaEventCreate(&b); }
    void start(cudaStream_t s = 0) { cudaEventRecord(a, s); }
    float stop(cudaStream_t s = 0) { cudaEventRecord(b, s); cudaEventSynchronize(b); float ms; cudaEventElapsedTime(&ms, a, b); return ms; }
};

static void chain(int rows, int cols, int stages, int warps_per_block, const char* label) {
    size_t per = (size_t)rows * cols;
    half *W, *vec;
    CK(cudaMalloc(&W, per * stages * sizeof(half)));
    CK(cudaMalloc(&vec, 2 * (size_t)(rows > cols ? rows : cols) * sizeof(half)));
    CK(cudaMemset(W, 0x3c, per * stages * sizeof(half)));          // 0x3c3c as half = about 1.0
    CK(cudaMemset(vec, 0x3c, 2 * (size_t)(rows > cols ? rows : cols) * sizeof(half)));
    int vmax = rows > cols ? rows : cols;
    cudaStream_t s; CK(cudaStreamCreate(&s));
    int blocks = (rows + warps_per_block - 1) / warps_per_block;
    int threads = warps_per_block * 32;
    // a square chain would feed y into x; for non-square shapes alternate the two stage shapes so dimensions match: here we simply keep reading the first `cols` entries of the vector
    cudaGraph_t g; cudaGraphExec_t ge;
    CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
    for (int i = 0; i < stages; i++) {
        half* in = vec + (i & 1) * (size_t)vmax;
        half* out = vec + ((i + 1) & 1) * (size_t)vmax;
        matvec_half<<<blocks, threads, 0, s>>>(W + (size_t)i * per, in, out, rows, cols);
    }
    CK(cudaStreamEndCapture(s, &g));
    CK(cudaGraphInstantiate(&ge, g, 0));
    for (int i = 0; i < 5; i++) CK(cudaGraphLaunch(ge, s));
    CK(cudaStreamSynchronize(s));
    Timer t;
    const int reps = 100;
    t.start(s);
    for (int i = 0; i < reps; i++) CK(cudaGraphLaunch(ge, s));
    float ms = t.stop(s);
    double us_stage = ms * 1e3 / (reps * (double)stages);
    double mb = per * 2 / 1e6;
    printf("%-34s %4dx%-5d weights/stage %6.2f MB  blocks %4d x %3d thr  -> %6.2f us per dependent stage  (pure bandwidth time at 425 GB/s: %5.2f us)  stage efficiency %4.1f%%\n", label, rows, cols, mb, blocks, threads,
           us_stage, mb / 425e3 * 1e6 * 1.0 / 1.0 * 1e-0 * 1.0, mb / 425e3 * 1e6 / us_stage * 100.0);
    fflush(stdout);
    cudaGraphExecDestroy(ge); cudaGraphDestroy(g); cudaStreamDestroy(s);
    cudaFree(W); cudaFree(vec);
}

int main() {
    cudaDeviceProp p;
    CK(cudaGetDeviceProperties(&p, 0));
    printf("device %s, %d SMs\n", p.name, p.multiProcessorCount);
    // Shapes of SmolLM2-135M (hidden 576, intermediate 1536, kv 192) and a bigger model (hidden 2048, intermediate 8192)
    int stages = 180;
    chain(576, 576, stages, 4, "135M attn-out (576x576)");
    chain(576, 576, stages, 8, "135M attn-out, 8 warps/block");
    chain(768, 576, stages, 4, "135M qkv fused (576+2*96 rows)");
    chain(1536, 576, stages, 4, "135M gate (1536x576)");
    chain(3072, 576, stages, 4, "135M gate+up fused (3072x576)");
    chain(576, 1536, stages, 4, "135M down (576x1536)");
    chain(2048, 2048, 90, 4, "1.5B-class 2048x2048");
    chain(8192, 2048, 60, 4, "1.5B-class gate 8192x2048");
    chain(2048, 8192, 60, 4, "1.5B-class down 2048x8192");

    // grid barrier cost
    int* sink; CK(cudaMalloc(&sink, 4));
    for (int bps : {1, 2, 4}) {
        int blocks = p.multiProcessorCount * bps;
        int threads = 256;
        int maxb = 0;
        CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxb, barrier_loop, threads, 0));
        if (bps > maxb) { printf("grid barrier: %d blocks/SM does not fit (max %d)\n", bps, maxb); continue; }
        int iters = 2000;
        void* args[] = {&iters, &sink};
        CK(cudaLaunchCooperativeKernel((void*)barrier_loop, blocks, threads, args, 0, 0));
        CK(cudaDeviceSynchronize());
        Timer t;
        t.start();
        CK(cudaLaunchCooperativeKernel((void*)barrier_loop, blocks, threads, args, 0, 0));
        float ms = t.stop();
        printf("grid.sync(): %3d blocks x %d threads -> %.2f us per barrier\n", blocks, threads, ms * 1e3 / iters);
        fflush(stdout);
    }
    printf("done\n");
    return 0;
}
