// Experiment (decode kernels, recipe 3 of the Turing hardware note): DRAM read bandwidth as a function of the bytes in flight per SM.
// Each lane issues U independent 128-bit loads per iteration (grid-stride over a 512 MiB buffer, perfectly coalesced), W warps are resident per SM: in flight per SM = W * 32 * U * 16 bytes.
// Little's law: bandwidth per SM = bytes_in_flight / latency, the knee shows how many loads a matvec lane must keep outstanding. Build: nvcc -O3 -arch=sm_75 -o gpu_inflight bench/gpu_inflight.cu
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

template <int U>
__global__ void stream_read(const uint4* __restrict__ p, size_t n_vec, unsigned* sink) {
    const size_t tid = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    unsigned acc = 0;
    for (size_t base = tid; base + (size_t)(U - 1) * stride < n_vec; base += (size_t)U * stride) {
        uint4 v[U];
#pragma unroll
        for (int u = 0; u < U; u++) v[u] = __ldg(p + base + (size_t)u * stride);
#pragma unroll
        for (int u = 0; u < U; u++) acc ^= v[u].x ^ v[u].y ^ v[u].z ^ v[u].w;
    }
    if (acc == 0x12345678u) sink[0] = acc;
}

template <int U>
static double run(const uint4* p, size_t n_vec, unsigned* sink, int blocks, int threads) {
    for (int i = 0; i < 3; i++) stream_read<U><<<blocks, threads>>>(p, n_vec, sink);
    CK(cudaDeviceSynchronize());
    cudaEvent_t a, b;
    cudaEventCreate(&a); cudaEventCreate(&b);
    const int reps = 8;
    cudaEventRecord(a);
    for (int i = 0; i < reps; i++) stream_read<U><<<blocks, threads>>>(p, n_vec, sink);
    cudaEventRecord(b);
    cudaEventSynchronize(b);
    float ms;
    cudaEventElapsedTime(&ms, a, b);
    cudaEventDestroy(a); cudaEventDestroy(b);
    return (double)reps * n_vec * 16.0 / (ms * 1e-3) / 1e9;
}

int main() {
    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, 0));
    const int sms = prop.multiProcessorCount;
    const size_t bytes = 512ull << 20;
    const size_t n_vec = bytes / 16;
    uint4* buf;
    unsigned* sink;
    CK(cudaMalloc(&buf, bytes));
    CK(cudaMalloc(&sink, 4));
    CK(cudaMemset(buf, 1, bytes));
    printf("DRAM read GB/s over a %zu MiB buffer, rows = resident warps per SM (256-thread blocks), columns = independent 16-byte loads per lane (bytes in flight per SM in KiB)\n", bytes >> 20);
    printf("%-14s", "warps/SM");
    const int us[] = {1, 2, 4, 8};
    for (int u : us) printf("   U=%d", u);
    printf("\n");
    for (int bps = 1; bps <= 4; bps++) {
        int warps = bps * 8;
        printf("%2d (%2d blk/SM) ", warps, bps);
        double g[4];
        g[0] = run<1>(buf, n_vec, sink, sms * bps, 256);
        g[1] = run<2>(buf, n_vec, sink, sms * bps, 256);
        g[2] = run<4>(buf, n_vec, sink, sms * bps, 256);
        g[3] = run<8>(buf, n_vec, sink, sms * bps, 256);
        for (int i = 0; i < 4; i++) printf(" %5.0f (%3d)", g[i], warps * 32 * us[i] * 16 / 1024);
        printf("\n");
        fflush(stdout);
    }
    return 0;
}
