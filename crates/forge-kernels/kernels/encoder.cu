// Kernels of the BERT-style embedding encoder (crates/forge-embed-gpu): the embedding gather (with the learned positions of `bert`), LayerNorm of rows (with the bias of the matrix before it), the rotary
// embedding of q and k (`nomic-bert`), attention over the tokens of one text (no mask), SwiGLU (`nomic-bert`) or the bias and GELU of `bert`, and the pooling of the token vectors. The linear layers of the
// encoder are the gemm_nt_f16 and gemm_nt_res kernels of gemm.cu (y = x W^T with fp16 operands and fp32 accumulation).
// The residual stream is fp32 (`x`), every GEMM input and output between the layers' steps is fp16; the arithmetic of the norms, the softmax and the pooling is fp32.
//
// Several texts go through the encoder in one pass without padding: their tokens are packed one after the other into `rows` rows. `starts` [S + 1] holds the first row of every text (and `rows` at the end),
// `seq_of` [rows] the text of every row. Attention of a row reads the rows of its own text only, the rotary position of a row is its place inside its text, pooling is done per text.

#include <cuda_fp16.h>

namespace enc {

// sum over the 256 threads of a block (every thread gets the total); `sh` holds 8 floats
__device__ __forceinline__ float block_sum(float v, float* sh) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    __syncthreads();
    if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = v;
    __syncthreads();
    float s = 0.f;
    for (int i = 0; i < (int)(blockDim.x >> 5); i++) s += sh[i];
    return s;
}

__device__ __forceinline__ float block_max(float v, float* sh) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    __syncthreads();
    if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = v;
    __syncthreads();
    float s = -3.0e38f;
    for (int i = 0; i < (int)(blockDim.x >> 5); i++) s = fmaxf(s, sh[i]);
    return s;
}

// eight halves of a 16-byte aligned address as floats
__device__ __forceinline__ void load8(const half* p, float* f) {
    const uint4 u = *reinterpret_cast<const uint4*>(p);
    const half2* h = reinterpret_cast<const half2*>(&u);
#pragma unroll
    for (int i = 0; i < 4; i++) {
        const float2 v = __half22float2(h[i]);
        f[2 * i] = v.x;
        f[2 * i + 1] = v.y;
    }
}

}  // namespace enc

// x[t] = token_embd[ids[t]] + token_types[0] for row t = blockIdx.x (fp32 residual stream); `type_row` is the fp16 row of the token type 0
extern "C" __global__ void __launch_bounds__(256) enc_embed(const unsigned* __restrict__ ids, const half* __restrict__ tok_emb, const half* __restrict__ type_row, float* __restrict__ x, int e) {
    const unsigned id = ids[blockIdx.x];
    for (int i = threadIdx.x; i < e; i += blockDim.x) x[(size_t)blockIdx.x * e + i] = __half2float(tok_emb[(size_t)id * e + i]) + __half2float(type_row[i]);
}

// the same for `bert`: x[t] = token_embd[ids[t]] + token_types[0] + position_embd[place of the row inside its text]; `pos_emb` [ctx][e] is fp16
extern "C" __global__ void __launch_bounds__(256) enc_embed_pos(const unsigned* __restrict__ ids, const half* __restrict__ tok_emb, const half* __restrict__ type_row, const half* __restrict__ pos_emb, const int* __restrict__ starts, const int* __restrict__ seq_of, float* __restrict__ x, int e) {
    const unsigned id = ids[blockIdx.x];
    const int pos = (int)blockIdx.x - starts[seq_of[blockIdx.x]];
    for (int i = threadIdx.x; i < e; i += blockDim.x) x[(size_t)blockIdx.x * e + i] = __half2float(tok_emb[(size_t)id * e + i]) + __half2float(type_row[i]) + __half2float(pos_emb[(size_t)pos * e + i]);
}

// LayerNorm of the row blockIdx.x of x, in place, and its fp16 copy in x16 (the input of the next GEMM): y = (x - mean) / sqrt(var + eps) * w + b
// `add` (null: nothing) is the bias of the matrix whose product was just added to x: x[i] += add[i] first (`bert` has a bias on every matrix; `nomic-bert` none)
extern "C" __global__ void __launch_bounds__(256) enc_ln(float* __restrict__ x, const float* __restrict__ w, const float* __restrict__ b, half* __restrict__ x16, int e, float eps, const float* __restrict__ add) {
    __shared__ float sh[8];
    float* row = x + (size_t)blockIdx.x * e;
    float sum = 0.f;
    // every thread visits the same indices in every pass of this kernel, so the bias can be added where the row is first read
    if (add) {
        for (int i = threadIdx.x; i < e; i += blockDim.x) row[i] += add[i];
    }
    for (int i = threadIdx.x; i < e; i += blockDim.x) sum += row[i];
    const float mean = enc::block_sum(sum, sh) / (float)e;
    float sq = 0.f;
    for (int i = threadIdx.x; i < e; i += blockDim.x) {
        const float d = row[i] - mean;
        sq += d * d;
    }
    const float inv = rsqrtf(enc::block_sum(sq, sh) / (float)e + eps);
    for (int i = threadIdx.x; i < e; i += blockDim.x) {
        const float y = (row[i] - mean) * inv * w[i] + b[i];
        row[i] = y;
        x16[(size_t)blockIdx.x * e + i] = __float2half(y);
    }
}

// rotary embedding (NeoX layout: dimension i of a head turns together with i + hd/2) of the q and k heads of qkv [rows][3 e] (fp16, in place); cos_t / sin_t [ctx][hd/2] are indexed by the place of the row inside its text
extern "C" __global__ void __launch_bounds__(256) enc_rope(half* __restrict__ qkv, const float* __restrict__ cos_t, const float* __restrict__ sin_t, const int* __restrict__ starts, const int* __restrict__ seq_of, int rows, int heads, int hd, int e) {
    const long long total = (long long)rows * 2 * heads * (hd / 2);
    const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const int half_d = hd / 2;
    const int i = (int)(idx % half_d);
    long long r = idx / half_d;
    const int h = (int)(r % heads);
    r /= heads;
    const int which = (int)(r % 2);
    const int t = (int)(r / 2);
    const int pos = t - starts[seq_of[t]];
    half* p = qkv + (size_t)t * 3 * e + which * e + h * hd;
    const float c = cos_t[pos * half_d + i], s = sin_t[pos * half_d + i];
    const float a = __half2float(p[i]), b = __half2float(p[i + half_d]);
    p[i] = __float2half(a * c - b * s);
    p[i + half_d] = __float2half(b * c + a * s);
}

// attention of row t = blockIdx.x, head h = blockIdx.y over all rows of its text: ctx[t][h] = softmax(q k^T * scale) v (hd and e are multiples of 8, the rows 16-byte aligned).
// Dynamic shared memory: hd + n_max floats (the query and the scores; n_max = the longest text of the pass)
extern "C" __global__ void __launch_bounds__(128) enc_attn(const half* __restrict__ qkv, half* __restrict__ ctx, const int* __restrict__ starts, const int* __restrict__ seq_of, int heads, int hd, int e, float scale) {
    extern __shared__ float sm[];
    __shared__ float sh[8];
    float* q = sm;
    float* sc = sm + hd;
    const int t = blockIdx.x, h = blockIdx.y, stride = 3 * e;
    const int s0 = starts[seq_of[t]], n = starts[seq_of[t] + 1] - s0;
    for (int d = threadIdx.x; d < hd; d += blockDim.x) q[d] = __half2float(qkv[(size_t)t * stride + h * hd + d]);
    __syncthreads();
    float mx = -3.0e38f;
    for (int s = threadIdx.x; s < n; s += blockDim.x) {
        const half* k = qkv + (size_t)(s0 + s) * stride + e + h * hd;
        float acc = 0.f;
        for (int d = 0; d < hd; d += 8) {
            float kf[8];
            enc::load8(k + d, kf);
#pragma unroll
            for (int j = 0; j < 8; j++) acc += q[d + j] * kf[j];
        }
        acc *= scale;
        sc[s] = acc;
        mx = fmaxf(mx, acc);
    }
    mx = enc::block_max(mx, sh);
    float sum = 0.f;
    for (int s = threadIdx.x; s < n; s += blockDim.x) {
        const float p = __expf(sc[s] - mx);
        sc[s] = p;
        sum += p;
    }
    const float inv = 1.f / enc::block_sum(sum, sh);
    __syncthreads();
    // the output: thread (d8, part) sums eight consecutive dimensions over every `parts`-th key; the parts are added up through shared memory
    const int groups = hd / 8;
    const int parts = blockDim.x / groups;
    const int g = threadIdx.x % groups, part = threadIdx.x / groups;
    float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    if (part < parts) {
        for (int s = part; s < n; s += parts) {
            float vf[8];
            enc::load8(qkv + (size_t)(s0 + s) * stride + 2 * e + h * hd + g * 8, vf);
            const float p = sc[s];
#pragma unroll
            for (int j = 0; j < 8; j++) acc[j] += p * vf[j];
        }
    }
    // the partial sums of the parts: one row of eight floats per thread
    __shared__ float part_sum[128 * 8];
    if (part < parts) {
#pragma unroll
        for (int j = 0; j < 8; j++) part_sum[threadIdx.x * 8 + j] = acc[j];
    }
    __syncthreads();
    for (int d = threadIdx.x; d < hd; d += blockDim.x) {
        const int dg = d / 8, dj = d % 8;
        float total = 0.f;
        for (int p = 0; p < parts; p++) total += part_sum[(p * groups + dg) * 8 + dj];
        ctx[(size_t)t * e + h * hd + d] = __float2half(total * inv);
    }
}

// out = silu(gate) * up, fp16
extern "C" __global__ void __launch_bounds__(256) enc_swiglu(const half* __restrict__ gate, const half* __restrict__ up, half* __restrict__ out, long long count) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const float g = __half2float(gate[i]);
    out[i] = __float2half(g / (1.f + __expf(-g)) * __half2float(up[i]));
}

// y[i] += bias[i % cols] for a matrix of fp16 values with `cols` columns (the q, k, v projections of `bert`)
extern "C" __global__ void __launch_bounds__(256) enc_bias16(half* __restrict__ y, const float* __restrict__ bias, long long count, int cols) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    y[i] = __float2half(__half2float(y[i]) + bias[i % cols]);
}

// y[i] = gelu(y[i] + bias[i % cols]) with the exact gelu, x * Phi(x) = x / 2 * (1 + erf(x / sqrt(2))) (the feed-forward layer of `bert`)
extern "C" __global__ void __launch_bounds__(256) enc_bias_gelu16(half* __restrict__ y, const float* __restrict__ bias, long long count, int cols) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const float v = __half2float(y[i]) + bias[i % cols];
    y[i] = __float2half(0.5f * v * (1.f + erff(v * 0.70710678118654752f)));
}

// the vector of text s = blockIdx.x: the mean of its token rows of x (mean != 0) or its first row, normalised to length one; blocks of 256 threads, out [S][e]
extern "C" __global__ void __launch_bounds__(256) enc_pool(const float* __restrict__ x, float* __restrict__ out, const int* __restrict__ starts, int e, int mean) {
    __shared__ float sh[8];
    const int s0 = starts[blockIdx.x], n = starts[blockIdx.x + 1] - s0;
    float* o = out + (size_t)blockIdx.x * e;
    float sq = 0.f;
    for (int i = threadIdx.x; i < e; i += blockDim.x) {
        float v;
        if (mean) {
            v = 0.f;
            for (int t = 0; t < n; t++) v += x[(size_t)(s0 + t) * e + i];
            v /= (float)n;
        } else {
            v = x[(size_t)s0 * e + i];
        }
        o[i] = v;
        sq += v * v;
    }
    const float norm = fmaxf(sqrtf(enc::block_sum(sq, sh)), 1e-12f);
    __syncthreads();
    for (int i = threadIdx.x; i < e; i += blockDim.x) o[i] /= norm;
}
