// Peak throughput of the int8 tensor-core MMA (m16n8k32 s8 -> s32) on this GPU: warps issue independent MMAs back to back (no memory, no epilogue).
//   nvcc -O3 -arch=sm_89 -o gpu_imma_rate gpu_imma_rate.cu && ./gpu_imma_rate
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>

template <int NACC>
__global__ void __launch_bounds__(256) bench(int* out, int iters, long long* cyc) {
    uint32_t a0 = threadIdx.x * 0x01010101u, a1 = a0 + 1, a2 = a0 + 2, a3 = a0 + 3, b0 = a0 ^ 0x5A5A5A5Au, b1 = a0 ^ 0xA5A5A5A5u;
    int c[NACC][4];
#pragma unroll
    for (int i = 0; i < NACC; i++) c[i][0] = c[i][1] = c[i][2] = c[i][3] = i;
    long long t0 = clock64();
    for (int it = 0; it < iters; it++) {
#pragma unroll
        for (int i = 0; i < NACC; i++)
            asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                         : "+r"(c[i][0]), "+r"(c[i][1]), "+r"(c[i][2]), "+r"(c[i][3])
                         : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    }
    long long t1 = clock64();
    int s = 0;
#pragma unroll
    for (int i = 0; i < NACC; i++) s += c[i][0] + c[i][1] + c[i][2] + c[i][3];
    out[blockIdx.x * blockDim.x + threadIdx.x] = s;
    if (threadIdx.x == 0) cyc[blockIdx.x] = t1 - t0;
}

template <int NACC>
void run(int blocks, int threads, int iters) {
    int* d; long long* dc;
    cudaMalloc(&d, blocks * threads * 4);
    cudaMalloc(&dc, blocks * 8);
    bench<NACC><<<blocks, threads>>>(d, 100, dc);
    cudaDeviceSynchronize();
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventRecord(e0);
    bench<NACC><<<blocks, threads>>>(d, iters, dc);
    cudaEventRecord(e1);
    cudaEventSynchronize(e1);
    float ms;
    cudaEventElapsedTime(&ms, e0, e1);
    long long cyc;
    cudaMemcpy(&cyc, dc, 8, cudaMemcpyDeviceToHost);
    int sms;
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
    const double macs = (double)blocks * (threads / 32) * iters * NACC * 4096.0;
    const double clk_hz = (double)cyc / (ms * 1e-3);
    printf("%2d accumulators, %2d blocks x %3d threads: %.3f ms = %.2f TMAC/s (%.1f TOPS), clock %.0f MHz, %.0f MAC/clk/SM\n", NACC, blocks, threads, ms, macs / (ms * 1e-3) / 1e12, 2 * macs / (ms * 1e-3) / 1e12, clk_hz / 1e6, macs / (ms * 1e-3) / clk_hz / sms);
    cudaFree(d); cudaFree(dc);
}

int main() {
    int sms;
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
    for (int rep = 0; rep < 2; rep++) {
        run<8>(sms, 256, 20000);
        run<16>(sms, 256, 10000);
        run<16>(sms * 2, 256, 10000);
        run<32>(sms, 256, 5000);
    }
    return 0;
}
