// Batched (prompt processing) kernels around the tensor-core GEMM of gemm.cu: embedding gather, RMSNorm to fp16, RoPE + KV-cache write (with the optional Qwen3 per-head norm),
// causal attention for a chunk of queries, SwiGLU. A chunk of T tokens starting at absolute position pos0 is processed layer by layer:
//   x (fp32 [T][E]) -> rmsnorm -> h (fp16) -> GEMM q/k/v -> qkv_rope -> attention -> GEMM wo (+= into x) -> rmsnorm -> GEMM gate, up -> swiglu -> GEMM down (+= into x)

#include <cuda_fp16.h>
#include <stdint.h>

// x[t][:] = embedding row of tokens[t]
extern "C" __global__ void embed_batch_kernel(const half* __restrict__ emb, const uint32_t* __restrict__ tokens, float* __restrict__ x, int T, int E) {
    const int t = blockIdx.x;
    const size_t row = tokens[t];
    for (int i = threadIdx.x; i < E; i += blockDim.x) x[(size_t)t * E + i] = __half2float(emb[row * E + i]);
}

// one block per row: h = x * rsqrt(mean(x^2) + eps) * w  (fp16 output)
extern "C" __global__ void rmsnorm_rows_kernel(const float* __restrict__ x, const float* __restrict__ w, half* __restrict__ h, int E, float eps) {
    __shared__ float red[32];
    const float* xr = x + (size_t)blockIdx.x * E;
    float ss = 0.f;
    for (int i = threadIdx.x; i < E; i += blockDim.x) ss += xr[i] * xr[i];
    ss = block_sum(ss, red);
    const float r = rsqrtf(ss / (float)E + eps);
    for (int i = threadIdx.x; i < E; i += blockDim.x) h[(size_t)blockIdx.x * E + i] = __float2half(xr[i] * r * w[i]);
}

// qkv: fp16 [T][q_dim + 2 kv_dim] (q | k | v per token). Writes the rotated q to q_out (fp16 [T][q_dim]) and the rotated k and the v of every token into the layer's KV cache at pos0 + t.
// One block per (token, head); heads 0..n_head-1 are query heads, then n_kv key heads, then n_kv value heads. qn / kn (may be null) are the Qwen3 per-head RMSNorm weights; bias_q / bias_k / bias_v
// (may be null) the biases of the projections (Qwen2), added in fp32 to the fp16 GEMM output before the norm and the rotation.
extern "C" __global__ void qkv_rope_kernel(const half* __restrict__ qkv, half* __restrict__ q_out, half* __restrict__ k_cache, half* __restrict__ v_cache, const float* __restrict__ cos_t,
                                           const float* __restrict__ sin_t, const float* __restrict__ qn, const float* __restrict__ kn, int pos0, int n_head, int n_kv, int hd, float eps,
                                           const float* __restrict__ bias_q, const float* __restrict__ bias_k, const float* __restrict__ bias_v) {
    __shared__ float red[32];
    __shared__ float hv[256];
    const int t = blockIdx.x, hb = blockIdx.y;
    const int q_dim = n_head * hd, kv_dim = n_kv * hd, stride = q_dim + 2 * kv_dim;
    const int pos = pos0 + t;
    const half* src;
    int kind;   // 0 q, 1 k, 2 v
    int head;
    if (hb < n_head) { kind = 0; head = hb; src = qkv + (size_t)t * stride + head * hd; }
    else if (hb < n_head + n_kv) { kind = 1; head = hb - n_head; src = qkv + (size_t)t * stride + q_dim + head * hd; }
    else { kind = 2; head = hb - n_head - n_kv; src = qkv + (size_t)t * stride + q_dim + kv_dim + head * hd; }
    const int half_d = hd / 2;
    const float* bh = kind == 0 ? bias_q : (kind == 1 ? bias_k : bias_v);
    if (bh != nullptr) bh += head * hd;
    if (kind == 2) {
        half* dst = v_cache + (size_t)pos * kv_dim + head * hd;
        if (bh == nullptr) {
            for (int d = threadIdx.x; d < hd; d += blockDim.x) dst[d] = src[d];
        } else {
            for (int d = threadIdx.x; d < hd; d += blockDim.x) dst[d] = __float2half(__half2float(src[d]) + bh[d]);
        }
        return;
    }
    float ss = 0.f;
    for (int d = threadIdx.x; d < hd; d += blockDim.x) {
        const float v = __half2float(src[d]) + (bh ? bh[d] : 0.f);
        hv[d] = v;
        ss += v * v;
    }
    float scale = 1.f;
    const float* wn = kind == 0 ? qn : kn;
    if (wn != nullptr) {
        ss = block_sum(ss, red);
        scale = rsqrtf(ss / (float)hd + eps);
    }
    __syncthreads();
    half* dst = kind == 0 ? q_out + (size_t)t * q_dim + head * hd : k_cache + (size_t)pos * kv_dim + head * hd;
    for (int i = threadIdx.x; i < half_d; i += blockDim.x) {
        float a = hv[i], b = hv[i + half_d];
        if (wn != nullptr) {
            a *= scale * wn[i];
            b *= scale * wn[i + half_d];
        }
        const float c = cos_t[(size_t)pos * half_d + i], s = sin_t[(size_t)pos * half_d + i];
        dst[i] = __float2half(a * c - b * s);
        dst[i + half_d] = __float2half(b * c + a * s);
    }
}

// Causal attention for a chunk: out[t] = softmax(q[t] . K[0..pos0+t] / sqrt(hd)) V[0..pos0+t], one thread (TPQ = 1) or a pair of lanes (TPQ = 2, hd = 128) per query and head.
// A block handles 128 / TPQ queries of one head; K and V are streamed through shared memory in tiles of 32 keys; online softmax in fp32.
// GATE: the output is multiplied by sigmoid(gate[qi * gate_tok + hh * gate_head + dim]) (Qwen3.5's gated attention; the gate halves sit inside the q projection output)
template <int HD, int TPQ, bool GATE>
__device__ __forceinline__ void attn_prefill_impl(const half* __restrict__ q, const half* __restrict__ k_cache, const half* __restrict__ v_cache, half* __restrict__ out, int T, int pos0, int n_head,
                                                  int n_kv, const half* __restrict__ gate, int gate_tok, int gate_head) {
    constexpr int QB = 128 / TPQ;      // queries per block
    constexpr int DPT = HD / TPQ;      // dims per thread
    constexpr int KT = 32;             // keys per tile
    __shared__ __align__(16) half ks[KT * HD];
    __shared__ __align__(16) half vs[KT * HD];
    const int hh = blockIdx.y, kvh = hh / (n_head / n_kv);
    const int q_dim = n_head * HD, kv_dim = n_kv * HD;
    const int qi = blockIdx.x * QB + threadIdx.x / TPQ;     // query row within the chunk
    const int part = threadIdx.x % TPQ;
    const bool valid = qi < T;
    const int qpos = pos0 + qi;
    float qr[DPT], acc[DPT];
    if (valid) {
#pragma unroll
        for (int d = 0; d < DPT; d++) qr[d] = __half2float(q[(size_t)qi * q_dim + hh * HD + part * DPT + d]);
    } else {
#pragma unroll
        for (int d = 0; d < DPT; d++) qr[d] = 0.f;
    }
#pragma unroll
    for (int d = 0; d < DPT; d++) acc[d] = 0.f;
    float m = -INFINITY, l = 0.f;
    const float scale = rsqrtf((float)HD);
    const int last_q = min(T - 1, (blockIdx.x + 1) * QB - 1);
    const int kmax = pos0 + last_q;              // last key position any query of this block needs
    for (int k0 = 0; k0 <= kmax; k0 += KT) {
        __syncthreads();
        // cooperative tile load: KT * HD halves = KT * HD / 8 uint4
        for (int i = threadIdx.x; i < KT * HD / 8; i += blockDim.x) {
            const int j = i / (HD / 8), c = i % (HD / 8);
            const int kp = k0 + j;
            uint4 kv = make_uint4(0u, 0u, 0u, 0u), vv = make_uint4(0u, 0u, 0u, 0u);
            if (kp <= kmax) {
                kv = __ldg(reinterpret_cast<const uint4*>(k_cache + (size_t)kp * kv_dim + kvh * HD) + c);
                vv = __ldg(reinterpret_cast<const uint4*>(v_cache + (size_t)kp * kv_dim + kvh * HD) + c);
            }
            reinterpret_cast<uint4*>(ks)[i] = kv;
            reinterpret_cast<uint4*>(vs)[i] = vv;
        }
        __syncthreads();
        const int nk = min(KT, kmax + 1 - k0);
        for (int j = 0; j < nk; j++) {
            const int kp = k0 + j;
            float s = 0.f;
#pragma unroll
            for (int d = 0; d < DPT; d++) s += qr[d] * __half2float(ks[j * HD + part * DPT + d]);
#pragma unroll
            for (int o = TPQ >> 1; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
            s *= scale;
            if (kp > qpos) continue;     // causal mask (only matters on the diagonal tile); the shuffle above was executed by all lanes
            const float mn = fmaxf(m, s);
            const float p = __expf(s - mn), corr = __expf(m - mn);
            l = l * corr + p;
#pragma unroll
            for (int d = 0; d < DPT; d++) acc[d] = acc[d] * corr + p * __half2float(vs[j * HD + part * DPT + d]);
            m = mn;
        }
    }
    if (valid) {
        const float inv = 1.f / l;
#pragma unroll
        for (int d = 0; d < DPT; d++) {
            float v = acc[d] * inv;
            if (GATE) v *= 1.f / (1.f + __expf(-__half2float(gate[(size_t)qi * gate_tok + (size_t)hh * gate_head + part * DPT + d])));
            out[(size_t)qi * q_dim + hh * HD + part * DPT + d] = __float2half(v);
        }
    }
}

extern "C" __global__ void __launch_bounds__(128) attn_prefill_64(const half* q, const half* k_cache, const half* v_cache, half* out, int T, int pos0, int n_head, int n_kv) {
    attn_prefill_impl<64, 1, false>(q, k_cache, v_cache, out, T, pos0, n_head, n_kv, nullptr, 0, 0);
}
extern "C" __global__ void __launch_bounds__(128) attn_prefill_128(const half* q, const half* k_cache, const half* v_cache, half* out, int T, int pos0, int n_head, int n_kv) {
    attn_prefill_impl<128, 2, false>(q, k_cache, v_cache, out, T, pos0, n_head, n_kv, nullptr, 0, 0);
}
// head width 256 with the output gate (Qwen3.5): four lanes per query, 64 dimensions each
extern "C" __global__ void __launch_bounds__(128) attn_prefill_256g(const half* q, const half* k_cache, const half* v_cache, half* out, int T, int pos0, int n_head, int n_kv, const half* gate,
                                                                  int gate_tok, int gate_head) {
    attn_prefill_impl<256, 4, true>(q, k_cache, v_cache, out, T, pos0, n_head, n_kv, gate, gate_tok, gate_head);
}

// a[t][f] = silu(g[t][f]) * u[t][f]   (fp16 in, fp16 out, computed in fp32)
extern "C" __global__ void swiglu_kernel(const half* __restrict__ g, const half* __restrict__ u, half* __restrict__ a, size_t n) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        const float gv = __half2float(g[i]);
        a[i] = __float2half((gv / (1.f + __expf(-gv))) * __half2float(u[i]));
    }
}

// copy one fp32 row (the last token's residual stream) to the decode scratch
extern "C" __global__ void copy_row_kernel(const float* __restrict__ src, float* __restrict__ dst, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) dst[i] = src[i];
}

// state.pos := value (device-resident step state)
extern "C" __global__ void set_pos_kernel(StepState* __restrict__ st, uint32_t pos) {
    if (threadIdx.x == 0 && blockIdx.x == 0) st->pos = pos;
}

// acc[d] += sum over the T rows of x[t][d]^2: activation statistics for the importance of the columns of the matrices that multiply x (quantization calibration)
extern "C" __global__ void colsq_accum_kernel(const half* __restrict__ x, int T, int D, float* __restrict__ acc) {
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= D) return;
    float s = 0.f;
    for (int t = 0; t < T; t++) {
        const float v = __half2float(x[(size_t)t * D + d]);
        s += v * v;
    }
    acc[d] += s;
}
