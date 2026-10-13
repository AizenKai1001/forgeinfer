// Feasibility spike (throwaway): does a CUDA graph WHILE node (conditional graph node, CUDA 12.4+) work on this Turing GPU (sm_75) with driver 595? A device-side loop that decides by itself
// when to stop lets a looped / halting model run all its iterations without a host round trip. The body kernel increments a counter and sets the loop condition (counter < limit);
// the graph is launched once and must leave the counter at `limit`. Also measures the cost per loop iteration.
// Build: nvcc -O3 -arch=sm_75 -o graph_while spikes/graph_while/graph_while.cu     Run: ./graph_while
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "CUDA error %s at %s:%d (%s)\n", cudaGetErrorString(e_), __FILE__, __LINE__, #x); exit(1); } } while (0)

__global__ void loop_body(int* counter, cudaGraphConditionalHandle handle, const int* limit) {
    int c = *counter + 1;
    *counter = c;
    cudaGraphSetConditional(handle, c < *limit ? 1u : 0u);                         // the limit lives in device memory, so one instantiated graph serves any trip count
}

int main() {
    int dev = 0;
    cudaDeviceProp p;
    CK(cudaGetDeviceProperties(&p, dev));
    int drv = 0, rt = 0;
    cudaDriverGetVersion(&drv);
    cudaRuntimeGetVersion(&rt);
    printf("device %s (cc %d.%d), driver CUDA %d, runtime %d\n", p.name, p.major, p.minor, drv, rt);

    int* counter;
    CK(cudaMalloc(&counter, sizeof(int)));
    cudaGraph_t graph;
    CK(cudaGraphCreate(&graph, 0));
    cudaGraphConditionalHandle handle;
    CK(cudaGraphConditionalHandleCreate(&handle, graph, 1u, cudaGraphCondAssignDefault));
    cudaGraphNodeParams cparams = {};
    cparams.type = cudaGraphNodeTypeConditional;
    cparams.conditional.handle = handle;
    cparams.conditional.type = cudaGraphCondTypeWhile;
    cparams.conditional.size = 1;
    cudaGraphNode_t whileNode;
    CK(cudaGraphAddNode(&whileNode, graph, nullptr, nullptr, 0, &cparams));
    cudaGraph_t body = cparams.conditional.phGraph_out[0];

    int* dlimit;
    CK(cudaMalloc(&dlimit, sizeof(int)));
    int limit = 1000;
    CK(cudaMemcpy(dlimit, &limit, sizeof(int), cudaMemcpyHostToDevice));
    void* args[] = {&counter, &handle, &dlimit};
    cudaKernelNodeParams kp = {};
    kp.func = (void*)loop_body;
    kp.gridDim = dim3(1);
    kp.blockDim = dim3(1);
    kp.kernelParams = args;
    cudaGraphNode_t kn;
    CK(cudaGraphAddKernelNode(&kn, body, nullptr, 0, &kp));

    cudaGraphExec_t exec;
    cudaError_t e = cudaGraphInstantiate(&exec, graph, 0);
    if (e != cudaSuccess) {
        printf("graph instantiate FAILED: %s -> conditional WHILE nodes are NOT usable here\n", cudaGetErrorString(e));
        return 1;
    }
    cudaStream_t s;
    CK(cudaStreamCreate(&s));
    CK(cudaMemsetAsync(counter, 0, sizeof(int), s));
    CK(cudaGraphLaunch(exec, s));
    CK(cudaStreamSynchronize(s));
    int got = 0;
    CK(cudaMemcpy(&got, counter, sizeof(int), cudaMemcpyDeviceToHost));
    printf("WHILE node: counter = %d (expected %d) -> %s\n", got, limit, got == limit ? "OK" : "WRONG");

    // cost per iteration: the same instantiated graph, trip counts changed through device memory; one warm-up launch first
    cudaEvent_t a, b;
    cudaEventCreate(&a); cudaEventCreate(&b);
    for (int trips : {1000, 1000, 10000, 100000}) {
        CK(cudaMemcpy(dlimit, &trips, sizeof(int), cudaMemcpyHostToDevice));
        CK(cudaMemsetAsync(counter, 0, sizeof(int), s));
        cudaEventRecord(a, s);
        CK(cudaGraphLaunch(exec, s));
        cudaEventRecord(b, s);
        CK(cudaEventSynchronize(b));
        float ms;
        cudaEventElapsedTime(&ms, a, b);
        CK(cudaMemcpy(&got, counter, sizeof(int), cudaMemcpyDeviceToHost));
        printf("%6d trips: counter %6d, %.2f ms = %.2f us per iteration\n", trips, got, ms, ms * 1e3 / trips);
    }
    return 0;
}
