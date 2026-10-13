// ForgeInfer decode kernels, version 0 (correctness first, fp16 weights): the fused five-stage layer of the architecture plan for batch 1.
//
//   embed        x[E] (fp32 residual stream) = embedding row of the current token
//   per layer:   S1 norm_qkv_rope   RMSNorm(x) -> Wq/Wk/Wv matvecs -> RoPE -> q (fp32 scratch), k and v appended to the fp16 KV cache
//                S1b qk_norm_rope   (Qwen3 only) per-head RMSNorm of q and k, then RoPE
//                S2 attention       softmax(q.K^T / sqrt(d)) V for one query head per block
//                S3 oproj_residual  x += Wo o
//                S4 norm_gateup     RMSNorm(x) -> gate and up matvecs -> silu(gate) * up
//                S5 down_residual   x += Wd act
//   final:       final_norm, head (logits fp32), argmax (writes the next token into the device-resident step state)
//
// All per-token variables (token id, position) live in a small device-resident state block, so the whole step can be captured once as a CUDA graph and replayed.
// Weights are fp16 [rows][cols] row-major; one warp computes one output row with 128-bit loads; activations are fp32 in shared memory.

#include <cuda_fp16.h>
#include <stdint.h>

struct StepState {
    uint32_t token;
    uint32_t pos;
    uint32_t steps;   // number of tokens produced by argmax so far (index into the output ring)
    uint32_t pad;
};

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

// sum over the block; `red` is shared memory of at least 32 floats; every thread gets the result
__device__ __forceinline__ float block_sum(float v, float* red) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = (blockDim.x + 31) >> 5;
    v = warp_sum(v);
    __syncthreads();
    if (lane == 0) red[warp] = v;
    __syncthreads();
    float t = 0.f;
    for (int i = 0; i < nw; i++) t += red[i];
    return t;
}

__device__ __forceinline__ float block_max(float v, float* red) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = (blockDim.x + 31) >> 5;
    v = warp_max(v);
    __syncthreads();
    if (lane == 0) red[warp] = v;
    __syncthreads();
    float t = -INFINITY;
    for (int i = 0; i < nw; i++) t = fmaxf(t, red[i]);
    return t;
}

__device__ __forceinline__ void unpack8(const uint4& v, float* f) {
    const half2* h = reinterpret_cast<const half2*>(&v);
#pragma unroll
    for (int k = 0; k < 4; k++) {
        float2 t = __half22float2(h[k]);
        f[2 * k] = t.x;
        f[2 * k + 1] = t.y;
    }
}

// dot product of one fp16 weight row (K halves, 16-byte aligned) with an fp32 vector in shared memory; the whole warp takes part, every lane returns the full sum
__device__ __forceinline__ float row_dot(const half* __restrict__ w, const float* __restrict__ x, int K, int lane) {
    const uint4* w4 = reinterpret_cast<const uint4*>(w);
    float acc = 0.f;
    for (int c = lane; c < K / 8; c += 32) {
        uint4 wv = __ldg(w4 + c);
        float f[8];
        unpack8(wv, f);
        const float4 a0 = *reinterpret_cast<const float4*>(x + c * 8);
        const float4 a1 = *reinterpret_cast<const float4*>(x + c * 8 + 4);
        acc += f[0] * a0.x + f[1] * a0.y + f[2] * a0.z + f[3] * a0.w + f[4] * a1.x + f[5] * a1.y + f[6] * a1.z + f[7] * a1.w;
    }
    return warp_sum(acc);
}

// two rows against the same vector (halves the shared-memory traffic)
__device__ __forceinline__ void row_dot2(const half* __restrict__ w0, const half* __restrict__ w1, const float* __restrict__ x, int K, int lane, float& r0, float& r1) {
    const uint4* p0 = reinterpret_cast<const uint4*>(w0);
    const uint4* p1 = reinterpret_cast<const uint4*>(w1);
    float a0 = 0.f, a1 = 0.f;
    for (int c = lane; c < K / 8; c += 32) {
        uint4 v0 = __ldg(p0 + c), v1 = __ldg(p1 + c);
        float f0[8], f1[8];
        unpack8(v0, f0);
        unpack8(v1, f1);
        const float4 x0 = *reinterpret_cast<const float4*>(x + c * 8);
        const float4 x1 = *reinterpret_cast<const float4*>(x + c * 8 + 4);
        a0 += f0[0] * x0.x + f0[1] * x0.y + f0[2] * x0.z + f0[3] * x0.w + f0[4] * x1.x + f0[5] * x1.y + f0[6] * x1.z + f0[7] * x1.w;
        a1 += f1[0] * x0.x + f1[1] * x0.y + f1[2] * x0.z + f1[3] * x0.w + f1[4] * x1.x + f1[5] * x1.y + f1[6] * x1.z + f1[7] * x1.w;
    }
    r0 = warp_sum(a0);
    r1 = warp_sum(a1);
}

// ---- register-hoisted rows: issue ALL loads of a row first (one DRAM round trip), consume afterwards. NI = 16-byte chunks per lane, K/8 <= 32 * NI. -----------------------------------------------
template <int NI>
__device__ __forceinline__ void load_row_regs(const half* __restrict__ w, int K8, int lane, uint4 (&r)[NI]) {
    const uint4* p = reinterpret_cast<const uint4*>(w);
#pragma unroll
    for (int i = 0; i < NI; i++) {
        const int c = lane + 32 * i;
        r[i] = c < K8 ? __ldg(p + c) : make_uint4(0u, 0u, 0u, 0u);
    }
}

template <int NI>
__device__ __forceinline__ float dot_regs(const uint4 (&r)[NI], const float* __restrict__ x, int K8, int lane) {
    float acc = 0.f;
#pragma unroll
    for (int i = 0; i < NI; i++) {
        const int c = lane + 32 * i;
        if (c < K8) {
            float f[8];
            unpack8(r[i], f);
            const float4 a0 = *reinterpret_cast<const float4*>(x + c * 8);
            const float4 a1 = *reinterpret_cast<const float4*>(x + c * 8 + 4);
            acc += f[0] * a0.x + f[1] * a0.y + f[2] * a0.z + f[3] * a0.w + f[4] * a1.x + f[5] * a1.y + f[6] * a1.z + f[7] * a1.w;
        }
    }
    return warp_sum(acc);
}

// h[i] = x[i] * rsqrt(mean(x^2) + eps) * w[i] into shared memory (every block of a consumer stage recomputes it: the vector is only E floats)
__device__ __forceinline__ void rmsnorm_to_smem(const float* __restrict__ x, const float* __restrict__ w, int E, float eps, float* h, float* red) {
    float ss = 0.f;
    for (int i = threadIdx.x; i < E; i += blockDim.x) {
        float v = x[i];
        ss += v * v;
    }
    ss = block_sum(ss, red);
    const float r = rsqrtf(ss / (float)E + eps);
    for (int i = threadIdx.x; i < E; i += blockDim.x) h[i] = x[i] * r * w[i];
    __syncthreads();
}

extern "C" __global__ void embed_kernel(const half* __restrict__ emb, const StepState* __restrict__ st, float* __restrict__ x, int E) {
    const size_t base = (size_t)st->token * E;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < E; i += gridDim.x * blockDim.x) x[i] = __half2float(emb[base + i]);
}

// ---- S1: RMSNorm + Wq/Wk/Wv matvecs (+ RoPE) -----------------------------------------------------------------------------------------------------------------------------------------------
// A "unit" is one RoPE pair of one head: rows i and i + hd/2 of the matrix. Units 0..Nq are query pairs, then key pairs, then value pairs (no rotation). One warp per unit, 4 warps per block.
// The weight rows are loaded into registers BEFORE the norm is computed (they do not depend on it), so the DRAM latency overlaps with the norm.
// rotate != 0: apply RoPE here and write q (fp32 scratch) and k (fp16 cache); rotate == 0 (Qwen3): write raw q and k as fp32 scratch for qk_norm_rope_kernel. v always goes to the cache.
template <int NI>
__device__ __forceinline__ void norm_qkv_rope_impl(const float* __restrict__ x, const float* __restrict__ norm_w, const half* __restrict__ wq, const half* __restrict__ wk, const half* __restrict__ wv,
                                                   float* __restrict__ q_out, float* __restrict__ k_raw, half* __restrict__ k_cache, half* __restrict__ v_cache,
                                                   const float* __restrict__ cos_t, const float* __restrict__ sin_t, const StepState* __restrict__ st,
                                                   int E, int n_head, int n_kv, int hd, float eps, int rotate) {
    constexpr int NX = 2 * NI;    // activation elements per thread: E <= 128 * NX
    extern __shared__ float smem[];
    float* h = smem;              // E floats
    float* red = smem + E;        // 32 floats
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int half_d = hd / 2;
    const int nq = n_head * half_d, nk = n_kv * half_d, nv = n_kv * half_d;
    const int u = blockIdx.x * (blockDim.x >> 5) + warp;
    const int K8 = E / 8;
    // 1. issue the loads of the (small, L2-resident) activation vector and of this warp's two weight rows; nothing waits yet
    float xv[NX], wn[NX];
#pragma unroll
    for (int k = 0; k < NX; k++) {
        const int i = threadIdx.x + k * blockDim.x;
        xv[k] = i < E ? x[i] : 0.f;
        wn[k] = i < E ? norm_w[i] : 0.f;
    }
    int which = 0, idx = 0, head = 0, i = 0;
    uint4 r0[NI], r1[NI];
    const bool active = u < nq + nk + nv;
    if (active) {
        if (u < nq) { which = 0; idx = u; }
        else if (u < nq + nk) { which = 1; idx = u - nq; }
        else { which = 2; idx = u - nq - nk; }
        head = idx / half_d;
        i = idx % half_d;
        const half* w = which == 0 ? wq : (which == 1 ? wk : wv);
        load_row_regs<NI>(w + (size_t)(head * hd + i) * E, K8, lane, r0);
        load_row_regs<NI>(w + (size_t)(head * hd + i + half_d) * E, K8, lane, r1);
    }
    // 2. RMSNorm of the activation while the weights are in flight
    float ss = 0.f;
#pragma unroll
    for (int k = 0; k < NX; k++) ss += xv[k] * xv[k];
    ss = block_sum(ss, red);
    const float rs = rsqrtf(ss / (float)E + eps);
#pragma unroll
    for (int k = 0; k < NX; k++) {
        const int j = threadIdx.x + k * blockDim.x;
        if (j < E) h[j] = xv[k] * rs * wn[k];
    }
    __syncthreads();
    if (!active) return;
    const float a = dot_regs<NI>(r0, h, K8, lane);
    const float b = dot_regs<NI>(r1, h, K8, lane);
    if (lane != 0) return;
    const uint32_t pos = st->pos;
    const int kv_dim = n_kv * hd;
    if (which == 2) {
        v_cache[(size_t)pos * kv_dim + head * hd + i] = __float2half(a);
        v_cache[(size_t)pos * kv_dim + head * hd + i + half_d] = __float2half(b);
        return;
    }
    float o0 = a, o1 = b;
    if (rotate) {
        const float c = cos_t[(size_t)pos * half_d + i], s = sin_t[(size_t)pos * half_d + i];
        o0 = a * c - b * s;
        o1 = b * c + a * s;
    }
    if (which == 0) {
        q_out[head * hd + i] = o0;
        q_out[head * hd + i + half_d] = o1;
    } else if (rotate) {
        k_cache[(size_t)pos * kv_dim + head * hd + i] = __float2half(o0);
        k_cache[(size_t)pos * kv_dim + head * hd + i + half_d] = __float2half(o1);
    } else {
        k_raw[head * hd + i] = o0;
        k_raw[head * hd + i + half_d] = o1;
    }
}

#define INST_QKV(NI)                                                                                                                                                                         \
    extern "C" __global__ void norm_qkv_rope_kernel_##NI(const float* x, const float* norm_w, const half* wq, const half* wk, const half* wv, float* q_out, float* k_raw, half* k_cache,  \
                                                         half* v_cache, const float* cos_t, const float* sin_t, const StepState* st, int E, int n_head, int n_kv, int hd, float eps,   \
                                                         int rotate) {                                                                                                                 \
        norm_qkv_rope_impl<NI>(x, norm_w, wq, wk, wv, q_out, k_raw, k_cache, v_cache, cos_t, sin_t, st, E, n_head, n_kv, hd, eps, rotate);                                                \
    }
INST_QKV(2) INST_QKV(3) INST_QKV(4) INST_QKV(6) INST_QKV(8) INST_QKV(12)

// Qwen3: RMSNorm over every query/key head (weight [hd]), then RoPE; k goes to the cache. One block per head.
extern "C" __global__ void qk_norm_rope_kernel(float* __restrict__ q, const float* __restrict__ k_raw, half* __restrict__ k_cache, const float* __restrict__ qn, const float* __restrict__ kn,
                                               const float* __restrict__ cos_t, const float* __restrict__ sin_t, const StepState* __restrict__ st, int n_head, int n_kv, int hd, float eps) {
    __shared__ float red[32];
    const int bh = blockIdx.x;
    const bool is_q = bh < n_head;
    const int head = is_q ? bh : bh - n_head;
    const float* src = is_q ? q + head * hd : k_raw + head * hd;
    const float* wn = is_q ? qn : kn;
    const int half_d = hd / 2;
    float ss = 0.f;
    for (int d = threadIdx.x; d < hd; d += blockDim.x) ss += src[d] * src[d];
    ss = block_sum(ss, red);
    const float r = rsqrtf(ss / (float)hd + eps);
    const uint32_t pos = st->pos;
    for (int i = threadIdx.x; i < half_d; i += blockDim.x) {
        const float a = src[i] * r * wn[i], b = src[i + half_d] * r * wn[i + half_d];
        const float c = cos_t[(size_t)pos * half_d + i], s = sin_t[(size_t)pos * half_d + i];
        const float o0 = a * c - b * s, o1 = b * c + a * s;
        if (is_q) {
            q[head * hd + i] = o0;
            q[head * hd + i + half_d] = o1;
        } else {
            const int kv_dim = n_kv * hd;
            k_cache[(size_t)pos * kv_dim + head * hd + i] = __float2half(o0);
            k_cache[(size_t)pos * kv_dim + head * hd + i + half_d] = __float2half(o1);
        }
    }
}

// ---- S2: attention for one query head per block ---------------------------------------------------------------------------------------------------------------------------------------------
// 128 threads. Keys/values are read with 16-byte loads: LPP = hd/8 lanes cover one position row, so a warp handles 32/LPP positions at once. Scores live in shared memory (t_max floats).
// dynamic shared memory: scores[t_max] + q[hd] + part[NGRP * hd] + red[32]
// QOUT: the head's output is written as a quantized activation instead (int8 per 32 values with a scale and the integer sum: the input of the next quantized matvec), see decode_q.cu
// GATE: the output of head hh is multiplied by sigmoid(gate[hh * gate_stride + d]) (Qwen3.5's gated attention: the gate halves sit inside the q projection output)
template <bool QOUT, bool GATE>
__device__ __forceinline__ void attention_impl(const float* __restrict__ q, const half* __restrict__ k_cache, const half* __restrict__ v_cache, float* __restrict__ out, int8_t* __restrict__ oq,
                                               float* __restrict__ od, int* __restrict__ os, const StepState* __restrict__ st, int n_head, int n_kv, int hd, int t_max, const float* __restrict__ gate,
                                               int gate_stride) {
    extern __shared__ float sm[];
    const int LPP = hd / 8;                    // lanes per position
    const int NGRP = (blockDim.x / LPP);       // position groups in the block
    float* scores = sm;
    float* qs = sm + t_max;
    float* part = qs + hd;
    float* red = part + NGRP * hd;
    const int hh = blockIdx.x;
    const int kvh = hh / (n_head / n_kv);
    const int kv_dim = n_kv * hd;
    const int tt = blockIdx.y;                 // token of a multi-token step (speculative verification): query tt sits at position st->pos + tt
    const int T = st->pos + 1 + tt;
    const int tid = threadIdx.x;
    const int sub = tid % LPP, grp = tid / LPP;
    for (int d = tid; d < hd; d += blockDim.x) qs[d] = q[(size_t)tt * n_head * hd + hh * hd + d];
    __syncthreads();
    const float scale = rsqrtf((float)hd);
    float qv[8];
#pragma unroll
    for (int j = 0; j < 8; j++) qv[j] = qs[sub * 8 + j];
    // phase A: scores. Position t = grp + NGRP * it; 16-byte key loads, reduction across the LPP lanes of a position (they are adjacent lanes of one warp)
    // every thread runs the same number of iterations (T rounded up to a multiple of NGRP), so the full-warp shuffles below are always reached by the whole warp
    const int T_up = (T + NGRP - 1) / NGRP * NGRP;
#pragma unroll 4
    for (int t0 = 0; t0 < T_up; t0 += NGRP) {
        const int t = t0 + grp;
        const bool valid = t < T;
        uint4 kraw = make_uint4(0u, 0u, 0u, 0u);
        if (valid) kraw = __ldg(reinterpret_cast<const uint4*>(k_cache + (size_t)t * kv_dim + kvh * hd) + sub);
        float kf[8];
        unpack8(kraw, kf);
        float acc = 0.f;
#pragma unroll
        for (int j = 0; j < 8; j++) acc += qv[j] * kf[j];
        for (int o = LPP >> 1; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
        if (valid && sub == 0) scores[t] = acc * scale;
    }
    __syncthreads();
    // phase B: softmax
    float mx = -INFINITY;
    for (int t = tid; t < T; t += blockDim.x) mx = fmaxf(mx, scores[t]);
    mx = block_max(mx, red);
    float sum = 0.f;
    for (int t = tid; t < T; t += blockDim.x) {
        const float e = __expf(scores[t] - mx);
        scores[t] = e;
        sum += e;
    }
    sum = block_sum(sum, red);
    const float inv = 1.f / sum;
    // phase C: thread (grp, sub) accumulates 8 output dims over the positions t = grp, grp + NGRP, ...
    float o[8];
#pragma unroll
    for (int j = 0; j < 8; j++) o[j] = 0.f;
#pragma unroll 4
    for (int t = grp; t < T; t += NGRP) {
        const uint4 vraw = __ldg(reinterpret_cast<const uint4*>(v_cache + (size_t)t * kv_dim + kvh * hd) + sub);
        float vf[8];
        unpack8(vraw, vf);
        const float p = scores[t];
#pragma unroll
        for (int j = 0; j < 8; j++) o[j] += p * vf[j];
    }
#pragma unroll
    for (int j = 0; j < 8; j++) part[grp * hd + sub * 8 + j] = o[j];
    __syncthreads();
    if (tid < hd) {
        float s = 0.f;
        for (int g = 0; g < NGRP; g++) s += part[g * hd + tid];
        float v = s * inv;
        if (GATE) v *= 1.f / (1.f + __expf(-gate[((size_t)tt * n_head + hh) * gate_stride + tid]));
        if (QOUT) {
            // hd is a multiple of 32, so a warp holds whole 32-value groups: scale = amax / 127, q = round(v / scale), integer sum
            const float amax = warp_max(fabsf(v));
            const float d = amax * (1.f / 127.f);
            const float id = d != 0.f ? 1.f / d : 0.f;
            const int qi = (int)roundf(v * id);
            int sum = qi;
#pragma unroll
            for (int o2 = 16; o2 > 0; o2 >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o2);
            oq[(size_t)tt * n_head * hd + hh * hd + tid] = (int8_t)qi;
            if ((tid & 31) == 0) {
                od[(size_t)tt * (n_head * hd / 32) + (hh * hd + tid) / 32] = d;
                os[(size_t)tt * (n_head * hd / 32) + (hh * hd + tid) / 32] = sum;
            }
        } else {
            out[(size_t)tt * n_head * hd + hh * hd + tid] = v;
        }
    }
}

extern "C" __global__ void attention_kernel(const float* __restrict__ q, const half* __restrict__ k_cache, const half* __restrict__ v_cache, float* __restrict__ out, const StepState* __restrict__ st,
                                            int n_head, int n_kv, int hd, int t_max) {
    attention_impl<false, false>(q, k_cache, v_cache, out, nullptr, nullptr, nullptr, st, n_head, n_kv, hd, t_max, nullptr, 0);
}

extern "C" __global__ void attention_q_kernel(const float* __restrict__ q, const half* __restrict__ k_cache, const half* __restrict__ v_cache, int8_t* __restrict__ oq, float* __restrict__ od,
                                              int* __restrict__ os, const StepState* __restrict__ st, int n_head, int n_kv, int hd, int t_max) {
    attention_impl<true, false>(q, k_cache, v_cache, nullptr, oq, od, os, st, n_head, n_kv, hd, t_max, nullptr, 0);
}

// gated attention with a quantized output; 256 threads for head width 256 (one thread per output dimension)
extern "C" __global__ void attention_g_kernel(const float* __restrict__ q, const half* __restrict__ k_cache, const half* __restrict__ v_cache, const float* __restrict__ gate, int8_t* __restrict__ oq,
                                              float* __restrict__ od, int* __restrict__ os, const StepState* __restrict__ st, int n_head, int n_kv, int hd, int t_max, int gate_stride) {
    attention_impl<true, true>(q, k_cache, v_cache, nullptr, oq, od, os, st, n_head, n_kv, hd, t_max, gate, gate_stride);
}

// ---- S3 / S5: x += W v  (one warp per output row, weights hoisted into registers) ---------------------------------------------------------------------------------------------------------
// dynamic shared memory: K floats for the input vector
template <int NI>
__device__ __forceinline__ void matvec_residual_impl(const half* __restrict__ w, const float* __restrict__ vin, float* __restrict__ x, int rows, int K) {
    constexpr int NV = 2 * NI;
    extern __shared__ float vs[];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
    const int r = blockIdx.x * nw + warp;
    const int K8 = K / 8;
    // the input vector first (L2 hits come back early), then the weight row, then stage the vector into shared memory
    float vv[NV];
#pragma unroll
    for (int k = 0; k < NV; k++) {
        const int i = threadIdx.x + k * blockDim.x;
        vv[k] = i < K ? vin[i] : 0.f;
    }
    uint4 rg[NI];
    if (r < rows) load_row_regs<NI>(w + (size_t)r * K, K8, lane, rg);
#pragma unroll
    for (int k = 0; k < NV; k++) {
        const int i = threadIdx.x + k * blockDim.x;
        if (i < K) vs[i] = vv[k];
    }
    __syncthreads();
    if (r >= rows) return;
    const float acc = dot_regs<NI>(rg, vs, K8, lane);
    if (lane == 0) x[r] += acc;
}

#define INST_MV(NI)                                                                                                                                                  \
    extern "C" __global__ void matvec_residual_kernel_##NI(const half* w, const float* vin, float* x, int rows, int K) { matvec_residual_impl<NI>(w, vin, x, rows, K); }
INST_MV(2) INST_MV(3) INST_MV(4) INST_MV(6) INST_MV(8) INST_MV(12)

// ---- S4: RMSNorm + gate/up matvecs + SwiGLU (weights hoisted) -------------------------------------------------------------------------------------------------------------------------------
// RPW gate/up pairs per warp (their loads are all issued up front)
template <int NI, int RPW>
__device__ __forceinline__ void norm_gateup_impl(const float* __restrict__ x, const float* __restrict__ norm_w, const half* __restrict__ wg, const half* __restrict__ wu, float* __restrict__ act,
                                                 int E, int F, float eps) {
    constexpr int NX = 2 * NI;
    extern __shared__ float smem[];
    float* h = smem;
    float* red = smem + E;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
    const int f0 = (blockIdx.x * nw + warp) * RPW;
    const int K8 = E / 8;
    float xv[NX], wn[NX];
#pragma unroll
    for (int k = 0; k < NX; k++) {
        const int i = threadIdx.x + k * blockDim.x;
        xv[k] = i < E ? x[i] : 0.f;
        wn[k] = i < E ? norm_w[i] : 0.f;
    }
    uint4 g[RPW][NI], u[RPW][NI];
#pragma unroll
    for (int j = 0; j < RPW; j++) {
        if (f0 + j < F) {
            load_row_regs<NI>(wg + (size_t)(f0 + j) * E, K8, lane, g[j]);
            load_row_regs<NI>(wu + (size_t)(f0 + j) * E, K8, lane, u[j]);
        }
    }
    float ss = 0.f;
#pragma unroll
    for (int k = 0; k < NX; k++) ss += xv[k] * xv[k];
    ss = block_sum(ss, red);
    const float rs = rsqrtf(ss / (float)E + eps);
#pragma unroll
    for (int k = 0; k < NX; k++) {
        const int j = threadIdx.x + k * blockDim.x;
        if (j < E) h[j] = xv[k] * rs * wn[k];
    }
    __syncthreads();
#pragma unroll
    for (int j = 0; j < RPW; j++) {
        if (f0 + j < F) {
            const float a = dot_regs<NI>(g[j], h, K8, lane);
            const float b = dot_regs<NI>(u[j], h, K8, lane);
            if (lane == 0) act[f0 + j] = (a / (1.f + __expf(-a))) * b;
        }
    }
}

#define INST_GU(NI)                                                                                                                                                                       \
    extern "C" __global__ void norm_gateup_kernel_##NI(const float* x, const float* norm_w, const half* wg, const half* wu, float* act, int E, int F, float eps) {                        \
        norm_gateup_impl<NI, 2>(x, norm_w, wg, wu, act, E, F, eps);                                                                                                                          \
    }
INST_GU(2) INST_GU(3) INST_GU(4) INST_GU(6) INST_GU(8) INST_GU(12)

// ---- final norm, LM head, argmax --------------------------------------------------------------------------------------------------------------------------------------------------------------
extern "C" __global__ void final_norm_kernel(const float* __restrict__ x, const float* __restrict__ w, float* __restrict__ h, int E, float eps) {
    __shared__ float red[32];
    float ss = 0.f;
    for (int i = threadIdx.x; i < E; i += blockDim.x) ss += x[i] * x[i];
    ss = block_sum(ss, red);
    const float r = rsqrtf(ss / (float)E + eps);
    for (int i = threadIdx.x; i < E; i += blockDim.x) h[i] = x[i] * r * w[i];
}

// logits[r] = head[r] . h ; rows are distributed grid-stride over warps
extern "C" __global__ void head_kernel(const half* __restrict__ head, const float* __restrict__ h, float* __restrict__ logits, int rows, int E) {
    extern __shared__ float vs[];
    for (int i = threadIdx.x; i < E; i += blockDim.x) vs[i] = h[i];
    __syncthreads();
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
    for (int r = blockIdx.x * nw + warp; r < rows; r += gridDim.x * nw) {
        float acc = row_dot(head + (size_t)r * E, vs, E, lane);
        if (lane == 0) logits[r] = acc;
    }
}

// pass 1: every block scans a contiguous chunk and writes its (value, index) maximum (lowest index wins ties)
extern "C" __global__ void argmax_partial_kernel(const float* __restrict__ logits, int n, float* __restrict__ pv, int* __restrict__ pi) {
    __shared__ float sv[32];
    __shared__ int si[32];
    const int chunk = (n + gridDim.x - 1) / gridDim.x;
    const int lo = blockIdx.x * chunk, hi = min(n, lo + chunk);
    float bv = -INFINITY;
    int bi = 0x7fffffff;
    for (int i = lo + threadIdx.x; i < hi; i += blockDim.x) {
        float v = logits[i];
        if (v > bv || (v == bv && i < bi)) { bv = v; bi = i; }
    }
    for (int o = 16; o > 0; o >>= 1) {
        float ov = __shfl_xor_sync(0xffffffffu, bv, o);
        int oi = __shfl_xor_sync(0xffffffffu, bi, o);
        if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
    }
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) { sv[warp] = bv; si[warp] = bi; }
    __syncthreads();
    if (threadIdx.x == 0) {
        float v = sv[0];
        int i = si[0];
        for (int w = 1; w < (blockDim.x >> 5); w++)
            if (sv[w] > v || (sv[w] == v && si[w] < i)) { v = sv[w]; i = si[w]; }
        pv[blockIdx.x] = v;
        pi[blockIdx.x] = i;
    }
}

// pass 2: one block merges the partial maxima, writes the next token and advances the position; the sampled id is also appended to the output ring
extern "C" __global__ void argmax_final_kernel(const float* __restrict__ pv, const int* __restrict__ pi, int nparts, StepState* __restrict__ st, uint32_t* __restrict__ ring, int ring_len) {
    if (threadIdx.x != 0) return;
    float v = pv[0];
    int i = pi[0];
    for (int p = 1; p < nparts; p++)
        if (pv[p] > v || (pv[p] == v && pi[p] < i)) { v = pv[p]; i = pi[p]; }
    ring[st->steps % ring_len] = (uint32_t)i;
    st->steps += 1;
    st->token = (uint32_t)i;
    st->pos += 1;
}
