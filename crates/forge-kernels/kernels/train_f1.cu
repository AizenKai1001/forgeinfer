// Forge-1 structure options of the trainer (each one is switched on by a flag of the trainer; train.cu stays as it is). Executable specification: tools/gen_train_goldens.py (PyTorch).
// Conventions as in train.cu: the residual stream and everything that is summed is fp32, GEMM operands and saved activations fp16.

#include <cuda_fp16.h>
#include <stdint.h>

// ReLU squared MLP activation: a = relu(u)^2 (fp16 in and out, fp32 math, saturating at a large finite fp16 value so that an outlier cannot turn into infinity).
extern "C" __global__ void relu2_kernel(const half* __restrict__ u, half* __restrict__ a, size_t n) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        const float r = fmaxf(__half2float(u[i]), 0.f);
        a[i] = __float2half(fminf(r * r, 65000.f));
    }
}

// backward of the above: du = da * 2 relu(u)
extern "C" __global__ void relu2_bwd_kernel(const half* __restrict__ da, const half* __restrict__ u, half* __restrict__ du, size_t n) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        const float r = fmaxf(__half2float(u[i]), 0.f);
        du[i] = __float2half(__half2float(da[i]) * 2.f * r);
    }
}

// Cross entropy over a chunk of logits with a soft cap: the model's logits are cap * tanh(z / cap). loss_rows[row] = logsumexp(zc) - zc[target];  dZ = (softmax(zc) - onehot) * (1 - (zc / cap)^2) * dz_scale.
extern "C" __global__ void ce_cap_kernel(const float* __restrict__ Z, const uint32_t* __restrict__ targets, float* __restrict__ loss_rows, half* __restrict__ dZ, int V, float dz_scale, float cap) {
    __shared__ float red[32];
    const float* z = Z + (size_t)blockIdx.x * V;
    const float inv = 1.f / cap;
    const uint32_t y = targets[blockIdx.x];
    if (y >= (uint32_t)V) {
        // an ignored position (masked target token): no loss, no gradient
        if (threadIdx.x == 0) loss_rows[blockIdx.x] = 0.f;
        for (int j = threadIdx.x; j < V; j += blockDim.x) dZ[(size_t)blockIdx.x * V + j] = __float2half(0.f);
        return;
    }
    float mx = -INFINITY;
    for (int j = threadIdx.x; j < V; j += blockDim.x) mx = fmaxf(mx, cap * tanhf(z[j] * inv));
    mx = block_max(mx, red);
    float sum = 0.f;
    for (int j = threadIdx.x; j < V; j += blockDim.x) sum += __expf(cap * tanhf(z[j] * inv) - mx);
    sum = block_sum(sum, red);
    const float lse = mx + logf(sum);
    if (threadIdx.x == 0) loss_rows[blockIdx.x] = lse - cap * tanhf(z[y] * inv);
    for (int j = threadIdx.x; j < V; j += blockDim.x) {
        const float zc = cap * tanhf(z[j] * inv);
        const float p = __expf(zc - lse);
        const float t = zc * inv;
        dZ[(size_t)blockIdx.x * V + j] = __float2half((p - (j == (int)y ? 1.f : 0.f)) * (1.f - t * t) * dz_scale);
    }
}

// Layer input mixing (per-layer residual scalars and U-net skip connections):  out = a * xprev + b * x0 + c * xskip   (fp32 [n]; sc points at the layer's (a, b, c) in the master parameters).
// use_ab = 0: a = 1, b = 0 (the scalars are not trained); use_c = 0: no skip term (xskip may be null).
extern "C" __global__ void mix_fwd_kernel(const float* __restrict__ xprev, const float* __restrict__ x0, const float* __restrict__ xskip, float* __restrict__ out, size_t n, const float* __restrict__ sc, int use_ab, int use_c) {
    const float a = use_ab ? sc[0] : 1.f, b = use_ab ? sc[1] : 0.f, c = use_c ? sc[2] : 0.f;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        float o = a * xprev[i];
        if (use_ab) o += b * x0[i];
        if (use_c) o += c * xskip[i];
        out[i] = o;
    }
}

// Backward of the mixing. dx holds the gradient of the mixed input on entry and a * (that gradient) on exit (the gradient of xprev; its fp16 copy goes to dy16 if given); dx0 += b * dxin; dskip = c * dxin;
// partial[block][3] = per-block sums of dxin * xprev, dxin * x0, dxin * xskip (the gradients of a, b, c; reduced in fixed order by reduce_partials_kernel).
extern "C" __global__ void mix_bwd_kernel(float* __restrict__ dx, const float* __restrict__ xprev, const float* __restrict__ x0, const float* __restrict__ xskip, float* __restrict__ dx0, float* __restrict__ dskip,
                                          half* __restrict__ dy16, float* __restrict__ partial, size_t n, const float* __restrict__ sc, int use_ab, int use_c) {
    __shared__ float red[32];
    const float a = use_ab ? sc[0] : 1.f, b = use_ab ? sc[1] : 0.f, c = use_c ? sc[2] : 0.f;
    float sa = 0.f, sb = 0.f, ss = 0.f;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        const float d = dx[i];
        if (use_ab) {
            sa += d * xprev[i];
            sb += d * x0[i];
            dx0[i] += b * d;
        }
        if (use_c) {
            ss += d * xskip[i];
            dskip[i] = c * d;
        }
        const float o = a * d;
        dx[i] = o;
        if (dy16 != nullptr) dy16[i] = __float2half(o);
    }
    sa = block_sum(sa, red);
    __syncthreads();
    sb = block_sum(sb, red);
    __syncthreads();
    ss = block_sum(ss, red);
    if (threadIdx.x == 0) {
        partial[(size_t)blockIdx.x * 3 + 0] = sa;
        partial[(size_t)blockIdx.x * 3 + 1] = sb;
        partial[(size_t)blockIdx.x * 3 + 2] = ss;
    }
}

// dx += src (fp32) and the fp16 copy of the sum: the gradient of a skip source's output is the sum of the stream's gradient and the gradient that came back through the skip connection.
extern "C" __global__ void add_to_f32_f16_kernel(float* __restrict__ dx, const float* __restrict__ src, half* __restrict__ dy16, size_t n) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        const float o = dx[i] + src[i];
        dx[i] = o;
        if (dy16 != nullptr) dy16[i] = __float2half(o);
    }
}

// dst += src (fp32)
extern "C" __global__ void add_f32_kernel(float* __restrict__ dst, const float* __restrict__ src, size_t n) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) dst[i] += src[i];
}

// Value embeddings: v[row] = lv * v[row] + lve * ve[token[row]]   (fp16 v [N][kvd], fp16 table [V][kvd], (lv, lve) from the master parameters). One block per row.
extern "C" __global__ void ve_fwd_kernel(half* __restrict__ v, const half* __restrict__ ve, const uint32_t* __restrict__ tokens, const float* __restrict__ sc, int kvd) {
    const size_t row = blockIdx.x;
    const float lv = sc[0], lve = sc[1];
    const half* e = ve + (size_t)tokens[row] * kvd;
    for (int j = threadIdx.x; j < kvd; j += blockDim.x) v[row * kvd + j] = __float2half(lv * __half2float(v[row * kvd + j]) + lve * __half2float(e[j]));
}

// Backward: dv (fp32 [N][kvd]) is the gradient of the mixed v. dve[token] += lve * dv (atomics); partial[block][2] = sums of dv * v_raw and dv * ve[token] (gradients of lv and lve); dv *= lv in place
// (the gradient of the raw projection output). v_raw is read from the saved fp16 qkv projection output (row stride `stride`, v starts at `v_off`).
extern "C" __global__ void ve_bwd_kernel(float* __restrict__ dv, const half* __restrict__ qkv, int stride, int v_off, const half* __restrict__ ve, const uint32_t* __restrict__ tokens, const float* __restrict__ sc,
                                         float* __restrict__ dve, float* __restrict__ partial, int N, int kvd) {
    __shared__ float red[32];
    const float lv = sc[0], lve = sc[1];
    float s1 = 0.f, s2 = 0.f;
    for (int row = blockIdx.x; row < N; row += gridDim.x) {
        const size_t t = tokens[row];
        for (int j = threadIdx.x; j < kvd; j += blockDim.x) {
            const float d = dv[(size_t)row * kvd + j];
            s1 += d * __half2float(qkv[(size_t)row * stride + v_off + j]);
            s2 += d * __half2float(ve[t * kvd + j]);
            atomicAdd(dve + t * kvd + j, lve * d);
            dv[(size_t)row * kvd + j] = lv * d;
        }
    }
    s1 = block_sum(s1, red);
    __syncthreads();
    s2 = block_sum(s2, red);
    if (threadIdx.x == 0) {
        partial[(size_t)blockIdx.x * 2 + 0] = s1;
        partial[(size_t)blockIdx.x * 2 + 1] = s2;
    }
}

// ---- hashed n-gram memory -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// The row of every position in the table of order n is a hash of its last n tokens (tokens before the start of the window count as "none"). Executable specification: ngram_index() in tools/gen_train_goldens.py.
__device__ __forceinline__ uint32_t ngram_hash(const uint32_t* tokens, int r, int p, int order, int bits) {
    uint64_t h = 0x9E3779B97F4A7C15ull * (uint64_t)order;
    for (int k = 0; k < order; k++) {
        const uint64_t t = (p - k >= 0) ? (uint64_t)tokens[r - k] + 1ull : 0ull;
        h = (h ^ t) * 0xff51afd7ed558ccdull;
        h ^= h >> 33;
    }
    h *= 0xc4ceb9fe1a85ec53ull;
    h ^= h >> 33;
    return (uint32_t)(h >> (64 - bits));
}

// idx[row] for every row (N = batch x T rows; the position inside the window is row % T)
extern "C" __global__ void ngram_idx_kernel(const uint32_t* __restrict__ tokens, uint32_t* __restrict__ idx, int N, int T, int order, int bits) {
    for (int r = blockIdx.x * blockDim.x + threadIdx.x; r < N; r += gridDim.x * blockDim.x) idx[r] = ngram_hash(tokens, r, r % T, order, bits);
}

// m[row] = sum over the orders of table_n[idx_n[row]]   (fp16 tables [rows][dm], up to three orders; unused ones are null). One block per row.
extern "C" __global__ void ngram_gather_kernel(const half* __restrict__ t0, const uint32_t* __restrict__ i0, const half* __restrict__ t1, const uint32_t* __restrict__ i1, const half* __restrict__ t2,
                                               const uint32_t* __restrict__ i2, half* __restrict__ m, int dm) {
    const size_t row = blockIdx.x;
    for (int j = threadIdx.x; j < dm; j += blockDim.x) {
        float s = __half2float(t0[(size_t)i0[row] * dm + j]);
        if (t1 != nullptr) s += __half2float(t1[(size_t)i1[row] * dm + j]);
        if (t2 != nullptr) s += __half2float(t2[(size_t)i2[row] * dm + j]);
        m[row * dm + j] = __float2half(s);
    }
}

// dtable_n[idx_n[row]] += dm[row] for every order (fp32 gradients of the tables, atomics)
extern "C" __global__ void ngram_scatter_kernel(const half* __restrict__ dm16, const uint32_t* __restrict__ i0, float* __restrict__ g0, const uint32_t* __restrict__ i1, float* __restrict__ g1,
                                                const uint32_t* __restrict__ i2, float* __restrict__ g2, int dm) {
    const size_t row = blockIdx.x;
    for (int j = threadIdx.x; j < dm; j += blockDim.x) {
        const float d = __half2float(dm16[row * dm + j]);
        atomicAdd(g0 + (size_t)i0[row] * dm + j, d);
        if (g1 != nullptr) atomicAdd(g1 + (size_t)i1[row] * dm + j, d);
        if (g2 != nullptr) atomicAdd(g2 + (size_t)i2[row] * dm + j, d);
    }
}

// The RMSNorm backward of a layer's attention norm FUSED with the backward of the layer-input mixing that precedes it (lambdas without skip connections): rmsnorm_bwd_kernel computes the gradient d of the
// mixed input, and in the same pass  dx0 += b d,  the gradient of the previous layer's output a d (fp32 dx_out and its fp16 copy) and the per-block sums of d xprev and d x0 (the gradients of a and b;
// mix_partial[block][3], the third slot 0) are produced. Same arithmetic per element as rmsnorm_bwd_kernel followed by mix_bwd_kernel (dx_out, dx16, dx0 bit-identical; the scalar sums are added in
// another order). One warp per row (eight per block, rows strided over all warps), E <= 1024, dynamic shared memory 8 E floats.
extern "C" __global__ void rmsnorm_mix_bwd_kernel(const float* __restrict__ dy, const float* __restrict__ x, const float* __restrict__ w, const float* __restrict__ rstd, const float* __restrict__ dres,
                                                  float* __restrict__ dx_out, half* __restrict__ dx16, float* __restrict__ dw_partial, int N, int E, const float* __restrict__ xprev,
                                                  const float* __restrict__ x0, float* __restrict__ dx0, const float* __restrict__ sc, float* __restrict__ mix_partial) {
    extern __shared__ float sm[];
    __shared__ float red[32];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, E4 = E >> 2;
    const float ma = sc[0], mb = sc[1];
    float4 acc[8];
#pragma unroll
    for (int j = 0; j < 8; j++) acc[j] = make_float4(0.f, 0.f, 0.f, 0.f);
    float sa = 0.f, sb = 0.f;
    const float4* w4 = reinterpret_cast<const float4*>(w);
    for (int row = blockIdx.x * 8 + warp; row < N; row += gridDim.x * 8) {
        const float r = rstd[row];
        const float4* dyr = reinterpret_cast<const float4*>(dy + (size_t)row * E);
        const float4* xr = reinterpret_cast<const float4*>(x + (size_t)row * E);
        float4 dv[8], xv[8];
        float dot = 0.f;
#pragma unroll
        for (int j = 0; j < 8; j++) {
            const int i = lane + 32 * j;
            if (i < E4) {
                dv[j] = dyr[i];
                xv[j] = xr[i];
                const float4 g = w4[i];
                dot += dv[j].x * g.x * xv[j].x + dv[j].y * g.y * xv[j].y + dv[j].z * g.z * xv[j].z + dv[j].w * g.w * xv[j].w;
            }
        }
        dot = warp_sum(dot);
        const float coef = r * r * r * dot / (float)E;
#pragma unroll
        for (int j = 0; j < 8; j++) {
            const int i = lane + 32 * j;
            if (i < E4) {
                const float4 g = w4[i];
                const size_t off = (size_t)row * E4 + i;
                float4 o = dres ? reinterpret_cast<const float4*>(dres)[off] : make_float4(0.f, 0.f, 0.f, 0.f);
                o.x += r * (dv[j].x * g.x) - coef * xv[j].x;
                o.y += r * (dv[j].y * g.y) - coef * xv[j].y;
                o.z += r * (dv[j].z * g.z) - coef * xv[j].z;
                o.w += r * (dv[j].w * g.w) - coef * xv[j].w;
                acc[j].x += dv[j].x * xv[j].x * r;
                acc[j].y += dv[j].y * xv[j].y * r;
                acc[j].z += dv[j].z * xv[j].z * r;
                acc[j].w += dv[j].w * xv[j].w * r;
                // the mixing backward of this element (as mix_bwd_kernel)
                const float4 xp = reinterpret_cast<const float4*>(xprev)[off], xz = reinterpret_cast<const float4*>(x0)[off];
                sa += o.x * xp.x;
                sa += o.y * xp.y;
                sa += o.z * xp.z;
                sa += o.w * xp.w;
                sb += o.x * xz.x;
                sb += o.y * xz.y;
                sb += o.z * xz.z;
                sb += o.w * xz.w;
                float4 d0 = reinterpret_cast<float4*>(dx0)[off];
                d0.x += mb * o.x;
                d0.y += mb * o.y;
                d0.z += mb * o.z;
                d0.w += mb * o.w;
                reinterpret_cast<float4*>(dx0)[off] = d0;
                const float4 q = make_float4(ma * o.x, ma * o.y, ma * o.z, ma * o.w);
                reinterpret_cast<float4*>(dx_out)[off] = q;
                const __half2 h0 = __floats2half2_rn(q.x, q.y), h1 = __floats2half2_rn(q.z, q.w);
                reinterpret_cast<uint2*>(dx16)[off] = make_uint2(*reinterpret_cast<const uint32_t*>(&h0), *reinterpret_cast<const uint32_t*>(&h1));
            }
        }
    }
#pragma unroll
    for (int j = 0; j < 8; j++) {
        const int i = lane + 32 * j;
        if (i < E4) reinterpret_cast<float4*>(sm + (size_t)warp * E)[i] = acc[j];
    }
    __syncthreads();
    for (int t = threadIdx.x; t < E; t += blockDim.x) {
        float s = 0.f;
#pragma unroll
        for (int wi = 0; wi < 8; wi++) s += sm[(size_t)wi * E + t];
        dw_partial[(size_t)blockIdx.x * E + t] = s;
    }
    sa = block_sum(sa, red);
    __syncthreads();
    sb = block_sum(sb, red);
    if (threadIdx.x == 0) {
        mix_partial[(size_t)blockIdx.x * 3 + 0] = sa;
        mix_partial[(size_t)blockIdx.x * 3 + 1] = sb;
        mix_partial[(size_t)blockIdx.x * 3 + 2] = 0.f;
    }
}

// The layer-input mixing (lambdas without skip connections) FUSED with the RMSNorm of the attention block:  x = a xprev + b x0  (fp32, saved for the backward), h = x rstd w (fp16), rstd per row.
// Same arithmetic per element as mix_fwd_kernel followed by rmsnorm_fwd_kernel (x, h and rstd bit-identical); x is not read back. One warp per row, eight rows per block, E a multiple of 4 and <= 1024.
extern "C" __global__ void mix_rmsnorm_fwd_kernel(const float* __restrict__ xprev, const float* __restrict__ x0, const float* __restrict__ sc, float* __restrict__ x, const float* __restrict__ w,
                                                  half* __restrict__ h, float* __restrict__ rstd, int N, int E, float eps) {
    const int row = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (row >= N) return;
    const float a = sc[0], b = sc[1];
    const int E4 = E >> 2;
    const float4* xp = reinterpret_cast<const float4*>(xprev + (size_t)row * E);
    const float4* xz = reinterpret_cast<const float4*>(x0 + (size_t)row * E);
    float4* xo = reinterpret_cast<float4*>(x + (size_t)row * E);
    float4 v[8];
    float ss = 0.f;
#pragma unroll
    for (int j = 0; j < 8; j++) {
        const int i = lane + 32 * j;
        if (i < E4) {
            const float4 p = xp[i], z = xz[i];
            float4 o;
            o.x = a * p.x;
            o.x += b * z.x;
            o.y = a * p.y;
            o.y += b * z.y;
            o.z = a * p.z;
            o.z += b * z.z;
            o.w = a * p.w;
            o.w += b * z.w;
            v[j] = o;
            xo[i] = o;
            ss += o.x * o.x + o.y * o.y + o.z * o.z + o.w * o.w;
        }
    }
    ss = warp_sum(ss);
    const float r = rsqrtf(ss / (float)E + eps);
    if (lane == 0) rstd[row] = r;
    const float4* w4 = reinterpret_cast<const float4*>(w);
    uint2* hr = reinterpret_cast<uint2*>(h + (size_t)row * E);
#pragma unroll
    for (int j = 0; j < 8; j++) {
        const int i = lane + 32 * j;
        if (i < E4) {
            const float4 g = w4[i];
            const __half2 p = __floats2half2_rn(v[j].x * r * g.x, v[j].y * r * g.y), q = __floats2half2_rn(v[j].z * r * g.z, v[j].w * r * g.w);
            hr[i] = make_uint2(*reinterpret_cast<const uint32_t*>(&p), *reinterpret_cast<const uint32_t*>(&q));
        }
    }
}
