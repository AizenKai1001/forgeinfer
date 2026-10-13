// ForgeInfer hardware microbenchmarks for the homelab GPU (RTX 2060 SUPER, Turing sm_75). Measures what the card REALLY does so that the engine is designed against measured limits:
//   DRAM read / copy bandwidth, FP32 FMA, FP16 HFMA2, INT8 dp4a, tensor-core (WMMA fp16, fp32 and fp16 accumulate), shared-memory bandwidth, kernel launch cost (stream and graph), PCIe copies.
// Build: nvcc -O3 -arch=sm_75 -o gpu_micro bench/gpu_micro.cu      Run: ./gpu_micro   (about 30-60 s; prints one "key value" line per measurement)
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
using namespace nvcuda;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

struct Timer {
    cudaEvent_t a, b;
    Timer() { cudaEventCreate(&a); cudaEventCreate(&b); }
    void start() { cudaEventRecord(a); }
    float stop() { cudaEventRecord(b); cudaEventSynchronize(b); float ms; cudaEventElapsedTime(&ms, a, b); return ms; }
};

// ---------------------------------------------------------------- memory bandwidth
__global__ void read_sum(const float4* __restrict__ p, size_t n, float* out) {
    float4 acc = make_float4(0, 0, 0, 0);
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        float4 v = __ldg(p + i);
        acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w;
    }
    float s = acc.x + acc.y + acc.z + acc.w;
    if (s == 1.2345e30f) out[0] = s;                       // never true, but the compiler cannot know it: keeps the loads alive
}

__global__ void copy4(const float4* __restrict__ a, float4* __restrict__ b, size_t n) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) b[i] = a[i];
}

// ---------------------------------------------------------------- arithmetic throughput
__global__ void fma32(float* out, int iters) {
    float a0 = threadIdx.x * 1e-3f, a1 = a0 + 1, a2 = a0 + 2, a3 = a0 + 3, a4 = a0 + 4, a5 = a0 + 5, a6 = a0 + 6, a7 = a0 + 7;
    const float b = 1.0000001f, c = 1e-7f;
    for (int i = 0; i < iters; i++) {
        a0 = fmaf(a0, b, c); a1 = fmaf(a1, b, c); a2 = fmaf(a2, b, c); a3 = fmaf(a3, b, c);
        a4 = fmaf(a4, b, c); a5 = fmaf(a5, b, c); a6 = fmaf(a6, b, c); a7 = fmaf(a7, b, c);
    }
    out[blockIdx.x * blockDim.x + threadIdx.x] = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
}

__global__ void fma16(__half2* out, int iters) {
    __half2 a0 = __floats2half2_rn(threadIdx.x * 1e-3f, 1.f), a1 = __hadd2(a0, a0), a2 = __hadd2(a1, a0), a3 = __hadd2(a2, a0), a4 = __hadd2(a3, a0), a5 = __hadd2(a4, a0), a6 = __hadd2(a5, a0), a7 = __hadd2(a6, a0);
    const __half2 b = __floats2half2_rn(1.0001f, 1.0001f), c = __floats2half2_rn(1e-4f, 1e-4f);
    for (int i = 0; i < iters; i++) {
        a0 = __hfma2(a0, b, c); a1 = __hfma2(a1, b, c); a2 = __hfma2(a2, b, c); a3 = __hfma2(a3, b, c);
        a4 = __hfma2(a4, b, c); a5 = __hfma2(a5, b, c); a6 = __hfma2(a6, b, c); a7 = __hfma2(a7, b, c);
    }
    out[blockIdx.x * blockDim.x + threadIdx.x] = __hadd2(__hadd2(__hadd2(a0, a1), __hadd2(a2, a3)), __hadd2(__hadd2(a4, a5), __hadd2(a6, a7)));
}

__global__ void dp4a_peak(int* out, int iters) {
    int x = 0x01020304 + threadIdx.x, y = 0x05060708;
    int a0 = 0, a1 = 1, a2 = 2, a3 = 3, a4 = 4, a5 = 5, a6 = 6, a7 = 7;
    for (int i = 0; i < iters; i++) {
        a0 = __dp4a(x, y, a0); a1 = __dp4a(x, y, a1); a2 = __dp4a(x, y, a2); a3 = __dp4a(x, y, a3);
        a4 = __dp4a(x, y, a4); a5 = __dp4a(x, y, a5); a6 = __dp4a(x, y, a6); a7 = __dp4a(x, y, a7);
        x += 0x01010101;
    }
    out[blockIdx.x * blockDim.x + threadIdx.x] = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
}

template <typename Acc>
__global__ void wmma_peak(float* out, int iters) {
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b;
    wmma::fragment<wmma::accumulator, 16, 16, 16, Acc> c0, c1, c2, c3;
    wmma::fill_fragment(a, __float2half(0.01f));
    wmma::fill_fragment(b, __float2half(0.01f));
    wmma::fill_fragment(c0, (Acc)0); wmma::fill_fragment(c1, (Acc)0); wmma::fill_fragment(c2, (Acc)0); wmma::fill_fragment(c3, (Acc)0);
    for (int i = 0; i < iters; i++) {
        wmma::mma_sync(c0, a, b, c0); wmma::mma_sync(c1, a, b, c1); wmma::mma_sync(c2, a, b, c2); wmma::mma_sync(c3, a, b, c3);
    }
    float s = 0;
    for (int i = 0; i < c0.num_elements; i++) s += (float)c0.x[i] + (float)c1.x[i] + (float)c2.x[i] + (float)c3.x[i];
    out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}

// ---------------------------------------------------------------- shared memory bandwidth (conflict-free 128-bit reads)
__global__ void smem_bw(float* out, int iters) {
    __shared__ float4 s[1024];
    for (int i = threadIdx.x; i < 1024; i += blockDim.x) s[i] = make_float4(i, i + 1, i + 2, i + 3);
    __syncthreads();
    float4 acc = make_float4(0, 0, 0, 0);
    int idx = threadIdx.x;
    for (int i = 0; i < iters; i++) {
        float4 v0 = s[idx & 1023], v1 = s[(idx + 32) & 1023], v2 = s[(idx + 64) & 1023], v3 = s[(idx + 96) & 1023];
        acc.x += v0.x + v1.x + v2.x + v3.x; acc.y += v0.y + v1.y + v2.y + v3.y; acc.z += v0.z + v1.z + v2.z + v3.z; acc.w += v0.w + v1.w + v2.w + v3.w;
        idx += 128;
    }
    out[blockIdx.x * blockDim.x + threadIdx.x] = acc.x + acc.y + acc.z + acc.w;
}

__global__ void empty_kernel() {}

static void report(const char* key, double v, const char* unit) { printf("%-34s %12.3f %s\n", key, v, unit); fflush(stdout); }

int main() {
    cudaDeviceProp p;
    CK(cudaGetDeviceProperties(&p, 0));
    printf("device %s | cc %d.%d | SMs %d | global %.2f GiB | L2 %.1f MiB | smem/block %zu KiB | smem/SM %zu KiB | regs/SM %d | bus %d bit\n", p.name, p.major, p.minor, p.multiProcessorCount,
           p.totalGlobalMem / 1073741824.0, p.l2CacheSize / 1048576.0, p.sharedMemPerBlock / 1024, p.sharedMemPerMultiprocessor / 1024, p.regsPerMultiprocessor, p.memoryBusWidth);
    const int sms = p.multiProcessorCount;
    Timer t;

    // DRAM bandwidth: 1 GiB read, 512 MiB copy (both far larger than the 4 MiB L2)
    const size_t bytes = 1ull << 30;
    float4 *buf, *buf2;
    float* sink;
    CK(cudaMalloc(&buf, bytes)); CK(cudaMalloc(&buf2, bytes)); CK(cudaMalloc(&sink, 1 << 20));
    CK(cudaMemset(buf, 0, bytes)); CK(cudaMemset(buf2, 0, bytes));
    size_t n4 = bytes / 16;
    for (int blocks : {sms * 4, sms * 8, sms * 16}) {
        for (int w = 0; w < 3; w++) read_sum<<<blocks, 256>>>(buf, n4, sink);
        t.start();
        const int reps = 10;
        for (int r = 0; r < reps; r++) read_sum<<<blocks, 256>>>(buf, n4, sink);
        float ms = t.stop() / reps;
        char key[64]; snprintf(key, sizeof key, "dram_read_gbps(blocks=%d)", blocks);
        report(key, bytes / (ms * 1e-3) / 1e9, "GB/s");
    }
    {
        size_t half4 = n4 / 2;
        for (int w = 0; w < 3; w++) copy4<<<sms * 8, 256>>>(buf, buf2, half4);
        t.start();
        const int reps = 10;
        for (int r = 0; r < reps; r++) copy4<<<sms * 8, 256>>>(buf, buf2, half4);
        float ms = t.stop() / reps;
        report("dram_copy_gbps(read+write)", 2.0 * half4 * 16 / (ms * 1e-3) / 1e9, "GB/s");
    }
    // L2-resident read (2 MiB working set, repeated)
    {
        size_t small4 = (2u << 20) / 16;
        t.start();
        for (int r = 0; r < 200; r++) read_sum<<<sms * 8, 256>>>(buf, small4, sink);
        float ms = t.stop() / 200;
        report("l2_read_gbps(2MiB set)", (2u << 20) / (ms * 1e-3) / 1e9, "GB/s");
    }
    cudaFree(buf); cudaFree(buf2);

    // arithmetic throughput
    const int threads = 256, blocks = sms * 8, iters = 20000;
    float* fo; __half2* ho; int* io;
    CK(cudaMalloc(&fo, (size_t)blocks * threads * 4)); CK(cudaMalloc(&ho, (size_t)blocks * threads * 4)); CK(cudaMalloc(&io, (size_t)blocks * threads * 4));
    double nthreads = (double)blocks * threads;
    fma32<<<blocks, threads>>>(fo, 100); CK(cudaDeviceSynchronize());
    t.start(); fma32<<<blocks, threads>>>(fo, iters); float ms = t.stop();
    double tflops32 = 2.0 * 8 * iters * nthreads / (ms * 1e-3) / 1e12;
    report("fp32_fma_tflops", tflops32, "TFLOPs");
    report("inferred_sm_clock_mhz(fp32)", tflops32 * 1e12 / (2.0 * 64 * sms) / 1e6, "MHz  (assumes 64 FP32 lanes per SM)");
    fma16<<<blocks, threads>>>(ho, 100); CK(cudaDeviceSynchronize());
    t.start(); fma16<<<blocks, threads>>>(ho, iters); ms = t.stop();
    report("fp16_hfma2_tflops", 4.0 * 8 * iters * nthreads / (ms * 1e-3) / 1e12, "TFLOPs");
    dp4a_peak<<<blocks, threads>>>(io, 100); CK(cudaDeviceSynchronize());
    t.start(); dp4a_peak<<<blocks, threads>>>(io, iters); ms = t.stop();
    report("int8_dp4a_tops", 8.0 * 8 * iters * nthreads / (ms * 1e-3) / 1e12, "TOPS (4 mul + 4 add per dp4a)");

    // tensor cores: one warp per 32 threads; 4 independent 16x16x16 accumulators per warp
    const int witers = 20000, wthreads = 256, wblocks = sms * 4;
    double warps = (double)wblocks * wthreads / 32;
    wmma_peak<float><<<wblocks, wthreads>>>(fo, 100); CK(cudaDeviceSynchronize());
    t.start(); wmma_peak<float><<<wblocks, wthreads>>>(fo, witers); ms = t.stop();
    report("tensor_fp16_acc32_tflops", 4.0 * 8192 * witers * warps / (ms * 1e-3) / 1e12, "TFLOPs (WMMA 16x16x16)");
    wmma_peak<half><<<wblocks, wthreads>>>(fo, 100); CK(cudaDeviceSynchronize());
    t.start(); wmma_peak<half><<<wblocks, wthreads>>>(fo, witers); ms = t.stop();
    report("tensor_fp16_acc16_tflops", 4.0 * 8192 * witers * warps / (ms * 1e-3) / 1e12, "TFLOPs (WMMA 16x16x16)");

    // shared memory
    smem_bw<<<blocks, threads>>>(fo, 100); CK(cudaDeviceSynchronize());
    t.start(); smem_bw<<<blocks, threads>>>(fo, 20000); ms = t.stop();
    report("smem_read_gbps", 4.0 * 16 * 20000 * nthreads / (ms * 1e-3) / 1e9, "GB/s (all SMs together)");

    // launch overhead
    {
        cudaStream_t s; CK(cudaStreamCreate(&s));
        for (int i = 0; i < 1000; i++) empty_kernel<<<1, 32, 0, s>>>();
        CK(cudaStreamSynchronize(s));
        t.start();
        const int N = 20000;
        for (int i = 0; i < N; i++) empty_kernel<<<1, 32, 0, s>>>();
        ms = t.stop();
        report("launch_stream_us_per_kernel", ms * 1e3 / N, "us");
        cudaGraph_t g; cudaGraphExec_t ge;
        CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
        for (int i = 0; i < 200; i++) empty_kernel<<<1, 32, 0, s>>>();
        CK(cudaStreamEndCapture(s, &g));
        CK(cudaGraphInstantiate(&ge, g, 0));
        for (int i = 0; i < 5; i++) CK(cudaGraphLaunch(ge, s));
        CK(cudaStreamSynchronize(s));
        t.start();
        const int G = 200;
        for (int i = 0; i < G; i++) CK(cudaGraphLaunch(ge, s));
        ms = t.stop();
        report("launch_graph_us_per_kernel", ms * 1e3 / (G * 200.0), "us (200-kernel graph)");
    }

    // PCIe
    {
        const size_t pb = 256u << 20;
        void *h, *d;
        CK(cudaMallocHost(&h, pb)); CK(cudaMalloc(&d, pb));
        memset(h, 1, pb);
        CK(cudaMemcpy(d, h, pb, cudaMemcpyHostToDevice));
        t.start(); for (int i = 0; i < 4; i++) CK(cudaMemcpy(d, h, pb, cudaMemcpyHostToDevice)); ms = t.stop();
        report("pcie_h2d_pinned_gbps", 4.0 * pb / (ms * 1e-3) / 1e9, "GB/s");
        t.start(); for (int i = 0; i < 4; i++) CK(cudaMemcpy(h, d, pb, cudaMemcpyDeviceToHost)); ms = t.stop();
        report("pcie_d2h_pinned_gbps", 4.0 * pb / (ms * 1e-3) / 1e9, "GB/s");
    }
    printf("done\n");
    return 0;
}
