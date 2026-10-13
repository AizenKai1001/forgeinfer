// Experiment R1: how fast can a PERSISTENT kernel run the chain of dependent matrix-vector stages of a small model's decode step (stages separated by grid-wide barriers), compared with a
// CUDA graph of separate single-stage kernels that run THE SAME stage code (apples to apples)? Variants of the persistent kernel:
//   mode 0: plain barrier chain
//   mode 1: prefetch.global.L2 of the next stage's weights before the barrier
//   mode 2: the next stage's weights (first row of every warp, it does not depend on this stage's output) are loaded into REGISTERS before the barrier, so the DRAM latency of the next stage
//           is hidden behind the current stage's compute and the barrier (a warp owns at most ~1-3 rows in a 135M model)
// Build: nvcc -O3 -arch=sm_75 -rdc=true -o gpu_mega_chain bench/gpu_mega_chain.cu      Run: ./gpu_mega_chain
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)


__device__ __forceinline__ void prefetch_l2(const void* p) { asm volatile("prefetch.global.L2 [%0];" ::"l"(p)); }

__device__ __forceinline__ float dot8(const uint4& wv, const uint4& xv) {
    const half2* wh = reinterpret_cast<const half2*>(&wv);
    const half2* xh = reinterpret_cast<const half2*>(&xv);
    float acc = 0.f;
#pragma unroll
    for (int k = 0; k < 4; k++) { float2 a = __half22float2(wh[k]), b = __half22float2(xh[k]); acc += a.x * b.x + a.y * b.y; }
    return acc;
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// one row from memory: y[row] = scale * dot(W[row, :], x)
__device__ __forceinline__ void mv_row(const half* __restrict__ W, const half* __restrict__ x, half* __restrict__ y, int row, int cols, int lane, float scale) {
    const uint4* w4 = reinterpret_cast<const uint4*>(W + (size_t)row * cols);
    const uint4* x4 = reinterpret_cast<const uint4*>(x);
    float acc = 0.f;
    for (int i = lane; i < cols / 8; i += 32) { uint4 wv = __ldg(w4 + i), xv = x4[i]; acc += dot8(wv, xv); }
    acc = warp_sum(acc);
    if (lane == 0) y[row] = __float2half(acc * scale);
}

// the rows a warp owns: gwarp, gwarp + nwt, ...
__device__ __forceinline__ void mv_stage(const half* __restrict__ W, const half* __restrict__ x, half* __restrict__ y, int rows, int cols, int gwarp, int nwt, int lane, float scale) {
    for (int row = gwarp; row < rows; row += nwt) mv_row(W, x, y, row, cols, lane, scale);
}

template <int NV>
__device__ __forceinline__ void load_first_row(const half* __restrict__ Wst, uint4 (&dst)[NV], int rows, int cols, int gwarp, int lane) {
    if (gwarp < rows) {
        const uint4* w4 = reinterpret_cast<const uint4*>(Wst + (size_t)gwarp * cols);
#pragma unroll
        for (int j = 0; j < NV; j++) { int i = lane + j * 32; if (i < cols / 8) dst[j] = __ldg(w4 + i); }
    }
}

// stage whose first row is already in registers
template <int NV>
__device__ __forceinline__ void mv_stage_regs(const half* __restrict__ Wst, const uint4 (&wr)[NV], const half* __restrict__ x, half* __restrict__ y, int rows, int cols, int gwarp, int nwt, int lane, float scale) {
    if (gwarp < rows) {
        const uint4* x4 = reinterpret_cast<const uint4*>(x);
        float acc = 0.f;
#pragma unroll
        for (int j = 0; j < NV; j++) { int i = lane + j * 32; if (i < cols / 8) acc += dot8(wr[j], x4[i]); }
        acc = warp_sum(acc);
        if (lane == 0) y[gwarp] = __float2half(acc * scale);
    }
    for (int row = gwarp + nwt; row < rows; row += nwt) mv_row(Wst, x, y, row, cols, lane, scale);
}

// NV = uint4 (8 halves) per lane held in registers (mode 2): cols <= NV*256
template <int MODE, int NV>
__global__ void mega(const half* __restrict__ W, half* vec, int rows, int cols, int stages, int vmax, float scale) {
    cg::grid_group grid = cg::this_grid();
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
    const int gwarp = blockIdx.x * nw + warp, nwt = gridDim.x * nw;
    const size_t per = (size_t)rows * cols;
    if (MODE == 2) {
        uint4 A[NV], B[NV];
        load_first_row(W, A, rows, cols, gwarp, lane);
        for (int s = 0; s < stages; s += 2) {                    // stages is even; A holds stage s, B stage s+1
            load_first_row(W + (size_t)(s + 1) * per, B, rows, cols, gwarp, lane);
            mv_stage_regs(W + (size_t)s * per, A, vec, vec + vmax, rows, cols, gwarp, nwt, lane, scale);
            grid.sync();
            if (s + 2 < stages) load_first_row(W + (size_t)(s + 2) * per, A, rows, cols, gwarp, lane);
            mv_stage_regs(W + (size_t)(s + 1) * per, B, vec + vmax, vec, rows, cols, gwarp, nwt, lane, scale);
            grid.sync();
        }
    } else {
        for (int s = 0; s < stages; s++) {
            if (MODE == 1 && s + 1 < stages) {                    // pull the next stage's rows into L2 while this stage computes
                const char* nxt = reinterpret_cast<const char*>(W + (size_t)(s + 1) * per);
                for (int row = gwarp; row < rows; row += nwt)
                    for (int off = lane * 128; off < cols * 2; off += 32 * 128) prefetch_l2(nxt + (size_t)row * cols * 2 + off);
            }
            mv_stage(W + (size_t)s * per, vec + (s & 1) * (size_t)vmax, vec + ((s + 1) & 1) * (size_t)vmax, rows, cols, gwarp, nwt, lane, scale);
            grid.sync();
        }
    }
}

// the same stage as a stand-alone kernel (for the graph of kernels)
__global__ void stage_kernel(const half* __restrict__ Wst, const half* __restrict__ x, half* __restrict__ y, int rows, int cols, float scale) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
    const int gwarp = blockIdx.x * nw + warp, nwt = gridDim.x * nw;
    mv_stage(Wst, x, y, rows, cols, gwarp, nwt, lane, scale);
}

struct Timer {
    cudaEvent_t a, b;
    Timer() { cudaEventCreate(&a); cudaEventCreate(&b); }
    void start() { cudaEventRecord(a); }
    float stop() { cudaEventRecord(b); cudaEventSynchronize(b); float ms; cudaEventElapsedTime(&ms, a, b); return ms; }
};

static float checksum(const half* vec, int n) {
    half* h = (half*)malloc(n * 2);
    CK(cudaMemcpy(h, vec, n * 2, cudaMemcpyDeviceToHost));
    double s = 0;
    for (int i = 0; i < n; i++) s += __half2float(h[i]);
    free(h);
    return (float)s;
}

template <int MODE, int NV>
static double run(const half* W, half* vec, int rows, int cols, int stages, int blocks, int threads, float* sum) {
    int vmax = rows > cols ? rows : cols;
    float scale = 0.94f / cols;
    void* args[] = {(void*)&W, (void*)&vec, &rows, &cols, &stages, &vmax, &scale};
    CK(cudaMemset(vec, 0x3c, 2 * (size_t)vmax * 2));
    for (int i = 0; i < 3; i++) CK(cudaLaunchCooperativeKernel((void*)mega<MODE, NV>, blocks, threads, args, 0, 0));
    CK(cudaDeviceSynchronize());
    CK(cudaMemset(vec, 0x3c, 2 * (size_t)vmax * 2));
    CK(cudaLaunchCooperativeKernel((void*)mega<MODE, NV>, blocks, threads, args, 0, 0));
    CK(cudaDeviceSynchronize());
    *sum = checksum(vec, 2 * vmax);                              // result after exactly one chain from the same start vector
    Timer t;
    const int reps = 50;
    t.start();
    for (int i = 0; i < reps; i++) CK(cudaLaunchCooperativeKernel((void*)mega<MODE, NV>, blocks, threads, args, 0, 0));
    float ms = t.stop();
    return ms * 1e3 / (reps * (double)stages);
}

static double run_graph(const half* W, half* vec, int rows, int cols, int stages, int blocks, int threads, float* sum) {
    int vmax = rows > cols ? rows : cols;
    float scale = 0.94f / cols;
    size_t per = (size_t)rows * cols;
    cudaStream_t st;
    CK(cudaStreamCreate(&st));
    cudaGraph_t g;
    cudaGraphExec_t ge;
    CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal));
    for (int s = 0; s < stages; s++) stage_kernel<<<blocks, threads, 0, st>>>(W + (size_t)s * per, vec + (s & 1) * (size_t)vmax, vec + ((s + 1) & 1) * (size_t)vmax, rows, cols, scale);
    CK(cudaStreamEndCapture(st, &g));
    CK(cudaGraphInstantiate(&ge, g, nullptr, nullptr, 0));
    CK(cudaMemset(vec, 0x3c, 2 * (size_t)vmax * 2));
    for (int i = 0; i < 3; i++) CK(cudaGraphLaunch(ge, st));
    CK(cudaStreamSynchronize(st));
    CK(cudaMemset(vec, 0x3c, 2 * (size_t)vmax * 2));
    CK(cudaGraphLaunch(ge, st));
    CK(cudaStreamSynchronize(st));
    *sum = checksum(vec, 2 * vmax);
    Timer t;
    const int reps = 50;
    t.start();
    for (int i = 0; i < reps; i++) CK(cudaGraphLaunch(ge, st));
    CK(cudaStreamSynchronize(st));
    float ms = t.stop();
    CK(cudaGraphExecDestroy(ge));
    CK(cudaGraphDestroy(g));
    CK(cudaStreamDestroy(st));
    return ms * 1e3 / (reps * (double)stages);
}

// prints the three blocks-per-SM columns of one megakernel mode
template <int MODE, int NV>
static void mega_columns(const half* W, half* vec, int rows, int cols, int stages, int sms, float ref, bool& same) {
    int maxb = 0;
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxb, mega<MODE, NV>, 256, 0));
    printf(" |");
    for (int bps : {1, 2, 4}) {
        if (bps > maxb) { printf("    n/a"); continue; }
        float s1 = 0;
        double t = run<MODE, NV>(W, vec, rows, cols, stages, sms * bps, 256, &s1);
        printf(" %6.2f", t);
        same &= (s1 == ref);
    }
}

int main() {
    cudaDeviceProp p;
    CK(cudaGetDeviceProperties(&p, 0));
    int sms = p.multiProcessorCount;
    struct Shape { int rows, cols; const char* what; };
    Shape shapes[] = {{576, 576, "135M attn-out"}, {960, 576, "135M fused QKV"}, {3072, 576, "135M gate+up"}, {576, 1536, "135M down"}, {2048, 2048, "1.5B-class 2048x2048"}};
    const int stages = 180;
    printf("us per dependent stage (%d stages chained). graph = CUDA graph of single-stage kernels with the same code.\n", stages);
    printf("%-22s %8s | %-27s | %-33s | %-33s\n", "stage", "bw time", "graph (1/2/4 blocks per SM)", "mega mode0 plain (1/2/4 per SM)", "mega mode2 regs  (1/2/4 per SM)");
    for (const Shape& sh : shapes) {
        size_t per = (size_t)sh.rows * sh.cols;
        half *W, *vec;
        CK(cudaMalloc(&W, per * stages * 2));
        int vmax = sh.rows > sh.cols ? sh.rows : sh.cols;
        CK(cudaMalloc(&vec, 2 * (size_t)vmax * 2));
        CK(cudaMemset(W, 0x3c, per * stages * 2));
        printf("%-22s %6.2f us |", sh.what, per * 2 / 425e3);
        float ref = 0, s1 = 0;
        bool same = true;
        for (int bps : {1, 2, 4}) { double t = run_graph(W, vec, sh.rows, sh.cols, stages, sms * bps, 256, &s1); printf(" %6.2f", t); if (bps == 1) ref = s1; else same &= (s1 == ref); }
        printf(" |");
        int nv = (sh.cols / 8 + 31) / 32;                          // uint4 per lane needed to hold one row of this shape
        if (nv <= 3) mega_columns<0, 3>(W, vec, sh.rows, sh.cols, stages, sms, ref, same); else if (nv <= 6) mega_columns<0, 6>(W, vec, sh.rows, sh.cols, stages, sms, ref, same); else mega_columns<0, 8>(W, vec, sh.rows, sh.cols, stages, sms, ref, same);
        if (nv <= 3) mega_columns<2, 3>(W, vec, sh.rows, sh.cols, stages, sms, ref, same); else if (nv <= 6) mega_columns<2, 6>(W, vec, sh.rows, sh.cols, stages, sms, ref, same); else mega_columns<2, 8>(W, vec, sh.rows, sh.cols, stages, sms, ref, same);
        printf(" results %s (sum %.4f)\n", same ? "identical" : "DIFFER", ref);
        fflush(stdout);
        cudaFree(W); cudaFree(vec);
    }
    return 0;
}
