// Sustained-load behaviour of the homelab GPU: runs the tensor-core (WMMA fp16, fp32 accumulate) loop and the DRAM read loop alternately for a given number of seconds and prints the achieved
// rate once per second together with the elapsed time, so that clock, power and thermal throttling over minutes show up as a falling curve. Sample nvidia-smi next to it for clocks/watts.
// Build: nvcc -O3 -arch=sm_75 -o gpu_sustain bench/gpu_sustain.cu      Run: ./gpu_sustain [seconds=60] [mode=tensor|dram|fp16acc]
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
using namespace nvcuda;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

template <typename Acc>
__global__ void wmma_loop(float* out, int iters) {
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


__global__ void wmma_int8_loop(int* out, int iters) {
    wmma::fragment<wmma::matrix_a, 16, 16, 16, signed char, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, signed char, wmma::col_major> b;
    wmma::fragment<wmma::accumulator, 16, 16, 16, int> c0, c1, c2, c3;
    wmma::fill_fragment(a, (signed char)3);
    wmma::fill_fragment(b, (signed char)5);
    wmma::fill_fragment(c0, 0); wmma::fill_fragment(c1, 0); wmma::fill_fragment(c2, 0); wmma::fill_fragment(c3, 0);
    for (int i = 0; i < iters; i++) {
        wmma::mma_sync(c0, a, b, c0); wmma::mma_sync(c1, a, b, c1); wmma::mma_sync(c2, a, b, c2); wmma::mma_sync(c3, a, b, c3);
    }
    int s = 0;
    for (int i = 0; i < c0.num_elements; i++) s += c0.x[i] + c1.x[i] + c2.x[i] + c3.x[i];
    out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}

__global__ void read_sum(const float4* __restrict__ p, size_t n, float* out) {
    float4 acc = make_float4(0, 0, 0, 0);
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        float4 v = __ldg(p + i);
        acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w;
    }
    float s = acc.x + acc.y + acc.z + acc.w;
    if (s == 1.2345e30f) out[0] = s;
}

int main(int argc, char** argv) {
    int seconds = argc > 1 ? atoi(argv[1]) : 60;
    const char* mode = argc > 2 ? argv[2] : "tensor";
    cudaDeviceProp p;
    CK(cudaGetDeviceProperties(&p, 0));
    int sms = p.multiProcessorCount;
    float* out;
    CK(cudaMalloc(&out, 1 << 22));
    size_t bytes = 1ull << 30;
    float4* buf;
    CK(cudaMalloc(&buf, bytes));
    CK(cudaMemset(buf, 0, bytes));
    cudaEvent_t a, b;
    cudaEventCreate(&a); cudaEventCreate(&b);
    const int wblocks = sms * 4, wthreads = 256, witers = 40000;
    double warps = (double)wblocks * wthreads / 32;
    double flops_per_launch = 4.0 * 8192 * witers * warps;
    printf("mode %s, %d s\nsecond  rate\n", mode, seconds);
    for (int s = 0; s < seconds; s++) {
        double t_acc = 0, work = 0;
        while (t_acc < 1000.0) {                                   // about one second of launches
            cudaEventRecord(a);
            if (!strcmp(mode, "dram")) { read_sum<<<sms * 8, 256>>>(buf, bytes / 16, out); work += (double)bytes; }
            else if (!strcmp(mode, "int8")) { wmma_int8_loop<<<wblocks, wthreads>>>((int*)out, witers); work += flops_per_launch; }
            else if (!strcmp(mode, "fp16acc")) { wmma_loop<half><<<wblocks, wthreads>>>(out, witers); work += flops_per_launch; }
            else { wmma_loop<float><<<wblocks, wthreads>>>(out, witers); work += flops_per_launch; }
            cudaEventRecord(b); cudaEventSynchronize(b);
            float ms; cudaEventElapsedTime(&ms, a, b);
            t_acc += ms;
        }
        if (!strcmp(mode, "dram")) printf("%6d  %8.1f GB/s\n", s + 1, work / (t_acc * 1e-3) / 1e9);
        else printf("%6d  %8.2f TFLOPs\n", s + 1, work / (t_acc * 1e-3) / 1e12);
        fflush(stdout);
    }
    return 0;
}
