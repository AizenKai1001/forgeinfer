// Logit post-processing for sampling on the device, so that the host never has to look at (or even copy) a vocabulary-sized logits vector:
//   penalty_kernel     repetition penalty on the logits of the (distinct) recent tokens, in place
//   logit_max_kernel   per-block maxima of the logits
//   collect_kernel     the candidate set {logit >= max - cutoff}: (index, logit) pairs appended with warp-aggregated atomics. With cutoff = 30 temperatures this is the support of the sampler chain
//                      (probability ratio below 1e-13 dropped), a few hundred to a few thousand tokens; with cutoff = 0 it is the arg-max set (greedy decoding under a repetition penalty).
// Output buffer `out` (uint32): [0] candidate count (may exceed cap: overflow), [1] the maximum logit (float bits), [2 .. 2+cap) indices, [2+cap .. 2+2cap) logits (float bits).

extern "C" __global__ void penalty_kernel(float* __restrict__ logits, const int* __restrict__ ids, int n, float pen) {
    const int i = threadIdx.x;
    if (i < n) {
        const float v = logits[ids[i]];
        logits[ids[i]] = v > 0.f ? v / pen : v * pen;
    }
}

// grid = number of blocks (<= 256), any block size; also zeroes the candidate count for the collect kernel that follows
extern "C" __global__ void logit_max_kernel(const float* __restrict__ logits, int V, float* __restrict__ partial, uint32_t* __restrict__ out) {
    __shared__ float red[32];
    float m = -INFINITY;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < V; i += gridDim.x * blockDim.x) m = fmaxf(m, logits[i]);
    m = block_max(m, red);
    if (threadIdx.x == 0) {
        partial[blockIdx.x] = m;
        if (blockIdx.x == 0) out[0] = 0u;
    }
}

extern "C" __global__ void collect_kernel(const float* __restrict__ logits, int V, const float* __restrict__ partial, int nb, float cutoff, int cap, uint32_t* __restrict__ out) {
    float m = -INFINITY;
    for (int b = 0; b < nb; b++) m = fmaxf(m, partial[b]);
    if (blockIdx.x == 0 && threadIdx.x == 0) out[1] = __float_as_uint(m);
    const float theta = m - cutoff;
    const int lane = threadIdx.x & 31;
    for (int i0 = blockIdx.x * blockDim.x; i0 < V; i0 += gridDim.x * blockDim.x) {
        const int i = i0 + threadIdx.x;
        const float x = i < V ? logits[i] : -INFINITY;
        const bool hit = x >= theta;
        const unsigned mask = __ballot_sync(0xffffffffu, hit);
        if (mask != 0u) {
            uint32_t base = 0;
            const int leader = __ffs(mask) - 1;
            if (lane == leader) base = atomicAdd(&out[0], (uint32_t)__popc(mask));
            base = __shfl_sync(0xffffffffu, base, leader);
            if (hit) {
                const uint32_t slot = base + __popc(mask & ((1u << lane) - 1u));
                if (slot < (uint32_t)cap) {
                    out[2 + slot] = (uint32_t)i;
                    out[2 + cap + slot] = __float_as_uint(x);
                }
            }
        }
    }
}
