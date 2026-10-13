// Training kernels (forward helpers, backward passes, loss, optimizer). Conventions: the residual stream and all gradients that are summed are fp32, GEMM operands fp16, weights fp32 masters with an fp16
// working copy. tools/ref_backward_numpy.py (finite-difference checked) and tools/gen_train_goldens.py (PyTorch) are the specifications.

#include <cuda_fp16.h>
#include <stdint.h>

// ---- forward ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

// h = x * rstd * w (fp16), rstd saved per row. One warp per row (E a multiple of 4), eight rows per block.
extern "C" __global__ void rmsnorm_fwd_kernel(const float* __restrict__ x, const float* __restrict__ w, half* __restrict__ h, float* __restrict__ rstd, int N, int E, float eps) {
    const int row = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (row >= N) return;
    const float4* xr = reinterpret_cast<const float4*>(x + (size_t)row * E);
    const int E4 = E >> 2;
    float ss = 0.f;
    for (int i = lane; i < E4; i += 32) {
        const float4 v = xr[i];
        ss += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }
    ss = warp_sum(ss);
    const float r = rsqrtf(ss / (float)E + eps);
    if (lane == 0) rstd[row] = r;
    const float4* w4 = reinterpret_cast<const float4*>(w);
    uint2* hr = reinterpret_cast<uint2*>(h + (size_t)row * E);
    for (int i = lane; i < E4; i += 32) {
        const float4 v = xr[i], g = w4[i];
        const __half2 a = __floats2half2_rn(v.x * r * g.x, v.y * r * g.y), b = __floats2half2_rn(v.z * r * g.z, v.w * r * g.w);
        hr[i] = make_uint2(*reinterpret_cast<const uint32_t*>(&a), *reinterpret_cast<const uint32_t*>(&b));
    }
}

// qkv (raw GEMM output, fp16 [N][qd + 2 kvd]) -> rotated q, k and v as separate fp16 buffers; optional Qwen3 per-head RMSNorm of q and k before the rotation (weights qn, kn; the rstd of every head is saved).
// One warp per (token, head); position = row % T; the rotation pairs (i, i + hd / 2) live in the same lane. Heads: 0..n_head-1 query, then n_kv key, then n_kv value. hd = 64 or 128.
extern "C" __global__ void qkv_rope_train_kernel(const half* __restrict__ qkv, half* __restrict__ q_out, half* __restrict__ k_out, half* __restrict__ v_out, const float* __restrict__ cos_t,
                                                 const float* __restrict__ sin_t, const float* __restrict__ qn, const float* __restrict__ kn, float* __restrict__ rstd_q, float* __restrict__ rstd_k,
                                                 int N, int T, int n_head, int n_kv, int hd, float eps) {
    const int H = n_head + 2 * n_kv;
    const long item = (long)blockIdx.x * 8 + (threadIdx.x >> 5);
    if (item >= (long)N * H) return;
    const int lane = threadIdx.x & 31, row = (int)(item / H), hb = (int)(item % H);
    const int q_dim = n_head * hd, kv_dim = n_kv * hd, stride = q_dim + 2 * kv_dim;
    const int pos = row % T, half_d = hd / 2;
    const half* base = qkv + (size_t)row * stride;
    if (hb >= n_head + n_kv) {
        const int head = hb - n_head - n_kv;
        for (int d = lane; d < hd; d += 32) v_out[(size_t)row * kv_dim + head * hd + d] = base[q_dim + kv_dim + head * hd + d];
        return;
    }
    const bool is_q = hb < n_head;
    const int head = is_q ? hb : hb - n_head;
    const half* src = base + (is_q ? head * hd : q_dim + head * hd);
    const float* wn = is_q ? qn : kn;
    float a[2], b[2];
    float ss = 0.f;
#pragma unroll
    for (int p = 0; p < 2; p++) {
        if (p * 64 < hd) {
            a[p] = __half2float(src[lane + 32 * p]);
            b[p] = __half2float(src[lane + 32 * p + half_d]);
            ss += a[p] * a[p] + b[p] * b[p];
        }
    }
    float r = 1.f;
    if (wn != nullptr) {
        ss = warp_sum(ss);
        r = rsqrtf(ss / (float)hd + eps);
        if (lane == 0) (is_q ? rstd_q + (size_t)row * n_head + head : rstd_k + (size_t)row * n_kv + head)[0] = r;
    }
    half* dst = is_q ? q_out + (size_t)row * q_dim + head * hd : k_out + (size_t)row * kv_dim + head * hd;
#pragma unroll
    for (int p = 0; p < 2; p++) {
        if (p * 64 < hd) {
            const int i = lane + 32 * p;
            float x1 = a[p], x2 = b[p];
            if (wn != nullptr) {
                x1 *= r * wn[i];
                x2 *= r * wn[i + half_d];
            }
            const float c = cos_t[(size_t)pos * half_d + i], s = sin_t[(size_t)pos * half_d + i];
            dst[i] = __float2half(x1 * c - x2 * s);
            dst[i + half_d] = __float2half(x2 * c + x1 * s);
        }
    }
}

// ---- cross entropy over a chunk of logits ---------------------------------------------------------------------------------------------------------------------------------------------------
// One block per row: loss_rows[row] = logsumexp(z) - z[target]; dz = (softmax(z) - onehot) * dz_scale as fp16 (dz_scale already contains the loss scale and 1 / tokens)
extern "C" __global__ void ce_kernel(const float* __restrict__ Z, const uint32_t* __restrict__ targets, float* __restrict__ loss_rows, half* __restrict__ dZ, int V, float dz_scale) {
    __shared__ float red[32];
    const float* z = Z + (size_t)blockIdx.x * V;
    const uint32_t y = targets[blockIdx.x];
    if (y >= (uint32_t)V) {
        // an ignored position (its target token is masked, for example a prompt token in chat fine-tuning): no loss, no gradient (the whole block takes this branch)
        if (threadIdx.x == 0) loss_rows[blockIdx.x] = 0.f;
        for (int j = threadIdx.x; j < V; j += blockDim.x) dZ[(size_t)blockIdx.x * V + j] = __float2half(0.f);
        return;
    }
    float mx = -INFINITY;
    for (int j = threadIdx.x; j < V; j += blockDim.x) mx = fmaxf(mx, z[j]);
    mx = block_max(mx, red);
    float sum = 0.f;
    for (int j = threadIdx.x; j < V; j += blockDim.x) sum += __expf(z[j] - mx);
    sum = block_sum(sum, red);
    const float lse = mx + logf(sum);
    if (threadIdx.x == 0) loss_rows[blockIdx.x] = lse - z[y];
    for (int j = threadIdx.x; j < V; j += blockDim.x) {
        const float p = __expf(z[j] - lse);
        dZ[(size_t)blockIdx.x * V + j] = __float2half((p - (j == (int)y ? 1.f : 0.f)) * dz_scale);
    }
}

// The same with ONE read of the logits: 1024 threads per row hold the row in registers (up to 8 float4 = 32 values each: V <= 32768, V a multiple of 4), so the row is read once instead of three times
// (max, sum, gradient) and the fp32 logits of a chunk cross the memory bus once. Same formulas; only the summation order of the softmax denominator differs (fp32 rounding).
extern "C" __global__ void __launch_bounds__(1024, 1) ce_reg_kernel(const float* __restrict__ Z, const uint32_t* __restrict__ targets, float* __restrict__ loss_rows, half* __restrict__ dZ, int V, float dz_scale) {
    __shared__ float red[32];
    const float* z = Z + (size_t)blockIdx.x * V;
    half* dz = dZ + (size_t)blockIdx.x * V;
    const uint32_t y = targets[blockIdx.x];
    const int V4 = V >> 2;
    if (y >= (uint32_t)V) {
        if (threadIdx.x == 0) loss_rows[blockIdx.x] = 0.f;
        for (int j = threadIdx.x; j < V4; j += blockDim.x) reinterpret_cast<uint2*>(dz)[j] = make_uint2(0u, 0u);
        return;
    }
    float4 v[8];
    float mx = -INFINITY;
#pragma unroll
    for (int j = 0; j < 8; j++) {
        const int i = threadIdx.x + 1024 * j;
        if (i < V4) {
            v[j] = __ldg(reinterpret_cast<const float4*>(z) + i);
            mx = fmaxf(mx, fmaxf(fmaxf(v[j].x, v[j].y), fmaxf(v[j].z, v[j].w)));
        }
    }
    mx = block_max(mx, red);
    __syncthreads();
    float sum = 0.f;
#pragma unroll
    for (int j = 0; j < 8; j++) {
        const int i = threadIdx.x + 1024 * j;
        if (i < V4) sum += __expf(v[j].x - mx) + __expf(v[j].y - mx) + __expf(v[j].z - mx) + __expf(v[j].w - mx);
    }
    sum = block_sum(sum, red);
    const float lse = mx + logf(sum);
    if (threadIdx.x == 0) loss_rows[blockIdx.x] = lse - z[y];
#pragma unroll
    for (int j = 0; j < 8; j++) {
        const int i = threadIdx.x + 1024 * j;
        if (i < V4) {
            const int c = 4 * i;
            const float p0 = __expf(v[j].x - lse) - (c == (int)y ? 1.f : 0.f), p1 = __expf(v[j].y - lse) - (c + 1 == (int)y ? 1.f : 0.f);
            const float p2 = __expf(v[j].z - lse) - (c + 2 == (int)y ? 1.f : 0.f), p3 = __expf(v[j].w - lse) - (c + 3 == (int)y ? 1.f : 0.f);
            const __half2 a = __floats2half2_rn(p0 * dz_scale, p1 * dz_scale), b = __floats2half2_rn(p2 * dz_scale, p3 * dz_scale);
            reinterpret_cast<uint2*>(dz)[i] = make_uint2(*reinterpret_cast<const uint32_t*>(&a), *reinterpret_cast<const uint32_t*>(&b));
        }
    }
}

// loss = sum of the n values of loss_rows divided by d (single block, fixed order); d is n unless some targets are ignored (their rows hold 0)
extern "C" __global__ void mean_kernel(const float* __restrict__ v, int n, int d, float* __restrict__ out) {
    __shared__ float red[32];
    float s = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) s += v[i];
    s = block_sum(s, red);
    if (threadIdx.x == 0) out[0] = s / (float)d;
}

// ---- backward -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

// dx_out = dres + rmsnorm_backward(dy, x, w, rstd) and, if dx16 is given, its fp16 copy; dw_partial[block][E] = sum over the block's rows of dy * x * rstd (reduced in fixed order by reduce_partials_kernel).
// y = x r w, g = dy w:  dx = r g - (r^3 / E) x sum(g x).  One warp per row (eight per block, rows strided over all warps); every lane keeps the dw sums of its own float4 slots in registers. E <= 1024.
// Dynamic shared memory: 8 * E floats.
extern "C" __global__ void rmsnorm_bwd_kernel(const float* __restrict__ dy, const float* __restrict__ x, const float* __restrict__ w, const float* __restrict__ rstd, const float* __restrict__ dres,
                                              float* __restrict__ dx_out, half* __restrict__ dx16, float* __restrict__ dw_partial, int N, int E) {
    extern __shared__ float sm[];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, E4 = E >> 2;
    float4 acc[8];
#pragma unroll
    for (int j = 0; j < 8; j++) acc[j] = make_float4(0.f, 0.f, 0.f, 0.f);
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
                float4 o = dres ? reinterpret_cast<const float4*>(dres + (size_t)row * E)[i] : make_float4(0.f, 0.f, 0.f, 0.f);
                o.x += r * (dv[j].x * g.x) - coef * xv[j].x;
                o.y += r * (dv[j].y * g.y) - coef * xv[j].y;
                o.z += r * (dv[j].z * g.z) - coef * xv[j].z;
                o.w += r * (dv[j].w * g.w) - coef * xv[j].w;
                reinterpret_cast<float4*>(dx_out + (size_t)row * E)[i] = o;
                if (dx16 != nullptr) {
                    const __half2 a = __floats2half2_rn(o.x, o.y), b = __floats2half2_rn(o.z, o.w);
                    reinterpret_cast<uint2*>(dx16 + (size_t)row * E)[i] = make_uint2(*reinterpret_cast<const uint32_t*>(&a), *reinterpret_cast<const uint32_t*>(&b));
                }
                acc[j].x += dv[j].x * xv[j].x * r;
                acc[j].y += dv[j].y * xv[j].y * r;
                acc[j].z += dv[j].z * xv[j].z * r;
                acc[j].w += dv[j].w * xv[j].w * r;
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
}

// grad[i] += sum over blocks of partial[b][i], fixed order
extern "C" __global__ void reduce_partials_kernel(const float* __restrict__ partial, int nblocks, int E, float* __restrict__ grad) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < E; i += gridDim.x * blockDim.x) {
        float s = 0.f;
        for (int b = 0; b < nblocks; b++) s += partial[(size_t)b * E + i];
        grad[i] += s;
    }
}

// SwiGLU backward: a = silu(g) * u.   dg = da u sg (1 + g (1 - sg)),  du = da silu(g)   (fp16 in/out, fp32 math)
extern "C" __global__ void swiglu_bwd_kernel(const half* __restrict__ da, const half* __restrict__ g, const half* __restrict__ u, half* __restrict__ dg, half* __restrict__ du, size_t n) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        const float gv = __half2float(g[i]), uv = __half2float(u[i]), d = __half2float(da[i]);
        const float sg = 1.f / (1.f + __expf(-gv));
        dg[i] = __float2half(d * uv * sg * (1.f + gv * (1.f - sg)));
        du[i] = __float2half(d * gv * sg);
    }
}


// SwiGLU on a combined projection output gu[row][0 .. 2F) = [gate | up]:  a[row][j] = silu(gate[j]) * up[j]. Eight elements per thread (F a multiple of 8).
__device__ __forceinline__ float silu_f(float g) { return g / (1.f + __expf(-g)); }
extern "C" __global__ void swiglu2_kernel(const half* __restrict__ gu, half* __restrict__ a, int rows, int F) {
    const int F8 = F >> 3;
    const size_t total = (size_t)rows * F8;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < total; i += (size_t)gridDim.x * blockDim.x) {
        const size_t r = i / F8;
        const int j = (int)(i % F8) * 8;
        const uint4 g4 = *reinterpret_cast<const uint4*>(gu + r * 2 * F + j), u4 = *reinterpret_cast<const uint4*>(gu + r * 2 * F + F + j);
        const __half2* gh = reinterpret_cast<const __half2*>(&g4);
        const __half2* uh = reinterpret_cast<const __half2*>(&u4);
        uint4 o4;
        __half2* oh = reinterpret_cast<__half2*>(&o4);
#pragma unroll
        for (int k = 0; k < 4; k++) {
            const float2 gf = __half22float2(gh[k]), uf = __half22float2(uh[k]);
            oh[k] = __floats2half2_rn(silu_f(gf.x) * uf.x, silu_f(gf.y) * uf.y);
        }
        *reinterpret_cast<uint4*>(a + r * F + j) = o4;
    }
}

// backward of the above: dgu[row] = [dg | du] from da[row] and gu[row]:  dg = da u sg (1 + g (1 - sg)),  du = da silu(g)
extern "C" __global__ void swiglu2_bwd_kernel(const half* __restrict__ da, const half* __restrict__ gu, half* __restrict__ dgu, int rows, int F) {
    const int F8 = F >> 3;
    const size_t total = (size_t)rows * F8;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < total; i += (size_t)gridDim.x * blockDim.x) {
        const size_t r = i / F8;
        const int j = (int)(i % F8) * 8;
        const uint4 g4 = *reinterpret_cast<const uint4*>(gu + r * 2 * F + j), u4 = *reinterpret_cast<const uint4*>(gu + r * 2 * F + F + j), d4 = *reinterpret_cast<const uint4*>(da + r * F + j);
        const __half2* gh = reinterpret_cast<const __half2*>(&g4);
        const __half2* uh = reinterpret_cast<const __half2*>(&u4);
        const __half2* dh = reinterpret_cast<const __half2*>(&d4);
        uint4 og4, ou4;
        __half2* ogh = reinterpret_cast<__half2*>(&og4);
        __half2* ouh = reinterpret_cast<__half2*>(&ou4);
#pragma unroll
        for (int k = 0; k < 4; k++) {
            const float2 gf = __half22float2(gh[k]), uf = __half22float2(uh[k]), df = __half22float2(dh[k]);
            const float sx = 1.f / (1.f + __expf(-gf.x)), sy = 1.f / (1.f + __expf(-gf.y));
            ogh[k] = __floats2half2_rn(df.x * uf.x * sx * (1.f + gf.x * (1.f - sx)), df.y * uf.y * sy * (1.f + gf.y * (1.f - sy)));
            ouh[k] = __floats2half2_rn(df.x * gf.x * sx, df.y * gf.y * sy);
        }
        *reinterpret_cast<uint4*>(dgu + r * 2 * F + j) = og4;
        *reinterpret_cast<uint4*>(dgu + r * 2 * F + F + j) = ou4;
    }
}


extern "C" __global__ void f32_to_f16_kernel(const float* __restrict__ src, half* __restrict__ dst, size_t n) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) dst[i] = __float2half(src[i]);
}

extern "C" __global__ void fill_f32_kernel(float* __restrict__ dst, size_t n, float v) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) dst[i] = v;
}

// embedding gradient: demb[token[row]][:] += dx[row][:]  (atomics; summation order is not fixed)
extern "C" __global__ void embed_bwd_kernel(const float* __restrict__ dx, const uint32_t* __restrict__ tokens, float* __restrict__ demb, int E) {
    const size_t row = blockIdx.x;
    float* dst = demb + (size_t)tokens[row] * E;
    for (int i = threadIdx.x; i < E; i += blockDim.x) atomicAdd(dst + i, dx[row * E + i]);
}

// D[row][h] = sum_d dO[row][h hd + d] * O[row][h hd + d]. One warp per (row, head); N = tokens.
extern "C" __global__ void attn_bwd_prep_kernel(const half* __restrict__ o, const half* __restrict__ dO, float* __restrict__ D, int N, int n_head, int hd) {
    const long item = (long)blockIdx.x * 8 + (threadIdx.x >> 5);
    if (item >= (long)N * n_head) return;
    const int lane = threadIdx.x & 31;
    const size_t off = (size_t)item * hd;
    float s = 0.f;
    for (int d = lane * 2; d < hd; d += 64) {
        const float2 a = __half22float2(*reinterpret_cast<const __half2*>(o + off + d)), b = __half22float2(*reinterpret_cast<const __half2*>(dO + off + d));
        s += a.x * b.x + a.y * b.y;
    }
    s = warp_sum(s);
    if (lane == 0) D[item] = s;
}

// Attention backward, reference version (CUDA cores, no tiling; used for validation and small shapes). lse is in the log2 domain (see flash_fwd): p = exp2(s * scale * log2e - lse).
// dK and dV: one thread per (sequence, key, kv head), loops over the queries of the group's heads.
extern "C" __global__ void attn_bwd_dkdv_naive_kernel(const half* __restrict__ q, const half* __restrict__ k, const half* __restrict__ v, const half* __restrict__ dO, const float* __restrict__ lse,
                                                      const float* __restrict__ D, float* __restrict__ dk, float* __restrict__ dv, int B, int T, int n_head, int n_kv, int hd) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * T * n_kv) return;
    const int kh = idx % n_kv, j = (idx / n_kv) % T, b = idx / (n_kv * T);
    const int q_dim = n_head * hd, kv_dim = n_kv * hd, rep = n_head / n_kv;
    const size_t krow = (size_t)(b * T + j);
    float kj[128], vj[128], dkj[128], dvj[128];
    for (int d = 0; d < hd; d++) {
        kj[d] = __half2float(k[krow * kv_dim + kh * hd + d]);
        vj[d] = __half2float(v[krow * kv_dim + kh * hd + d]);
        dkj[d] = 0.f;
        dvj[d] = 0.f;
    }
    const float scale = rsqrtf((float)hd), sl2 = scale * 1.4426950408889634f;
    for (int hr = 0; hr < rep; hr++) {
        const int h = kh * rep + hr;
        for (int i = j; i < T; i++) {
            const size_t qrow = (size_t)(b * T + i);
            float s = 0.f, dp = 0.f;
            for (int d = 0; d < hd; d++) {
                s += __half2float(q[qrow * q_dim + h * hd + d]) * kj[d];
                dp += __half2float(dO[qrow * q_dim + h * hd + d]) * vj[d];
            }
            const float p = exp2f(s * sl2 - lse[qrow * n_head + h]);
            const float ds = p * (dp - D[qrow * n_head + h]);
            for (int d = 0; d < hd; d++) {
                dvj[d] += p * __half2float(dO[qrow * q_dim + h * hd + d]);
                dkj[d] += ds * scale * __half2float(q[qrow * q_dim + h * hd + d]);
            }
        }
    }
    for (int d = 0; d < hd; d++) {
        dk[krow * kv_dim + kh * hd + d] = dkj[d];
        dv[krow * kv_dim + kh * hd + d] = dvj[d];
    }
}

// dQ: one thread per (sequence, query, head)
extern "C" __global__ void attn_bwd_dq_naive_kernel(const half* __restrict__ q, const half* __restrict__ k, const half* __restrict__ v, const half* __restrict__ dO, const float* __restrict__ lse,
                                                    const float* __restrict__ D, float* __restrict__ dq, int B, int T, int n_head, int n_kv, int hd) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * T * n_head) return;
    const int h = idx % n_head, i = (idx / n_head) % T, b = idx / (n_head * T);
    const int q_dim = n_head * hd, kv_dim = n_kv * hd, kh = h / (n_head / n_kv);
    const size_t qrow = (size_t)(b * T + i);
    float qi[128], doi[128], dqi[128];
    for (int d = 0; d < hd; d++) {
        qi[d] = __half2float(q[qrow * q_dim + h * hd + d]);
        doi[d] = __half2float(dO[qrow * q_dim + h * hd + d]);
        dqi[d] = 0.f;
    }
    const float scale = rsqrtf((float)hd), sl2 = scale * 1.4426950408889634f;
    const float L = lse[qrow * n_head + h], Dv = D[qrow * n_head + h];
    for (int j = 0; j <= i; j++) {
        const size_t krow = (size_t)(b * T + j);
        float s = 0.f, dp = 0.f;
        for (int d = 0; d < hd; d++) {
            s += qi[d] * __half2float(k[krow * kv_dim + kh * hd + d]);
            dp += doi[d] * __half2float(v[krow * kv_dim + kh * hd + d]);
        }
        const float p = exp2f(s * sl2 - L);
        const float ds = p * (dp - Dv);
        for (int d = 0; d < hd; d++) dqi[d] += ds * scale * __half2float(k[krow * kv_dim + kh * hd + d]);
    }
    for (int d = 0; d < hd; d++) dq[qrow * q_dim + h * hd + d] = dqi[d];
}

// Backward through the inverse rotation (and the Qwen3 q/k RMSNorm) of every head; assembles the fp16 gradient of the raw qkv projection output [dq | dk | dv] per token.
// dq, dk, dv: fp32 gradients of the rotated q / k and of v. raw_qkv: the saved fp16 projection output (pre-norm values). One warp per (token, head); a block handles `tok_per_block` consecutive tokens and
// writes its sums of the q-norm / k-norm weight gradients to partial[kind][block][hd] (reduced in fixed order by reduce_partials_kernel; not written when qn is null). hd = 64 or 128.
extern "C" __global__ void qkv_bwd_kernel(const float* __restrict__ dq, const float* __restrict__ dk, const float* __restrict__ dv, const half* __restrict__ raw_qkv, const float* __restrict__ rstd_q,
                                          const float* __restrict__ rstd_k, const float* __restrict__ qn, const float* __restrict__ kn, const float* __restrict__ cos_t, const float* __restrict__ sin_t,
                                          half* __restrict__ dqkv, float* __restrict__ partial, int N, int T, int n_head, int n_kv, int hd, int tok_per_block) {
    __shared__ float red[8][2][128];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int H = n_head + 2 * n_kv;
    const int q_dim = n_head * hd, kv_dim = n_kv * hd, stride = q_dim + 2 * kv_dim, half_d = hd / 2;
    float accq[4] = {0.f, 0.f, 0.f, 0.f}, acck[4] = {0.f, 0.f, 0.f, 0.f};
    const long item0 = (long)blockIdx.x * tok_per_block * H;
    const long item1 = min(item0 + (long)tok_per_block * H, (long)N * H);
    for (long item = item0 + warp; item < item1; item += 8) {
        const int row = (int)(item / H), hb = (int)(item % H), pos = row % T;
        half* out = dqkv + (size_t)row * stride;
        if (hb >= n_head + n_kv) {
            const int head = hb - n_head - n_kv;
            for (int d = lane; d < hd; d += 32) out[q_dim + kv_dim + head * hd + d] = __float2half(dv[(size_t)row * kv_dim + head * hd + d]);
            continue;
        }
        const bool is_q = hb < n_head;
        const int head = is_q ? hb : hb - n_head;
        const float* dy = is_q ? dq + (size_t)row * q_dim + head * hd : dk + (size_t)row * kv_dim + head * hd;
        const half* xs = raw_qkv + (size_t)row * stride + (is_q ? head * hd : q_dim + head * hd);
        const float* wn = is_q ? qn : kn;
        half* dst = out + (is_q ? head * hd : q_dim + head * hd);
        float dyp1[2], dyp2[2];
#pragma unroll
        for (int p = 0; p < 2; p++) {
            if (p * 64 < hd) {
                const int i = lane + 32 * p;
                const float c = cos_t[(size_t)pos * half_d + i], s = sin_t[(size_t)pos * half_d + i];
                const float d1 = dy[i], d2 = dy[i + half_d];
                dyp1[p] = d1 * c + d2 * s;
                dyp2[p] = -d1 * s + d2 * c;
            }
        }
        if (wn == nullptr) {
#pragma unroll
            for (int p = 0; p < 2; p++) {
                if (p * 64 < hd) {
                    dst[lane + 32 * p] = __float2half(dyp1[p]);
                    dst[lane + 32 * p + half_d] = __float2half(dyp2[p]);
                }
            }
            continue;
        }
        const float r = is_q ? rstd_q[(size_t)row * n_head + head] : rstd_k[(size_t)row * n_kv + head];
        float x1[2], x2[2];
        float dot = 0.f;
#pragma unroll
        for (int p = 0; p < 2; p++) {
            if (p * 64 < hd) {
                const int i = lane + 32 * p;
                x1[p] = __half2float(xs[i]);
                x2[p] = __half2float(xs[i + half_d]);
                dot += dyp1[p] * wn[i] * x1[p] + dyp2[p] * wn[i + half_d] * x2[p];
            }
        }
        dot = warp_sum(dot);
        const float coef = r * r * r * dot / (float)hd;
#pragma unroll
        for (int p = 0; p < 2; p++) {
            if (p * 64 < hd) {
                const int i = lane + 32 * p;
                dst[i] = __float2half(r * dyp1[p] * wn[i] - coef * x1[p]);
                dst[i + half_d] = __float2half(r * dyp2[p] * wn[i + half_d] - coef * x2[p]);
                if (is_q) {
                    accq[2 * p] += dyp1[p] * x1[p] * r;
                    accq[2 * p + 1] += dyp2[p] * x2[p] * r;
                } else {
                    acck[2 * p] += dyp1[p] * x1[p] * r;
                    acck[2 * p + 1] += dyp2[p] * x2[p] * r;
                }
            }
        }
    }
    if (qn == nullptr) return;
#pragma unroll
    for (int p = 0; p < 2; p++) {
        if (p * 64 < hd) {
            const int i = lane + 32 * p;
            red[warp][0][i] = accq[2 * p];
            red[warp][0][i + half_d] = accq[2 * p + 1];
            red[warp][1][i] = acck[2 * p];
            red[warp][1][i + half_d] = acck[2 * p + 1];
        }
    }
    __syncthreads();
    for (int t = threadIdx.x; t < 2 * hd; t += blockDim.x) {
        const int kind = t / hd, d = t % hd;
        float s = 0.f;
#pragma unroll
        for (int wi = 0; wi < 8; wi++) s += red[wi][kind][d];
        partial[((size_t)kind * gridDim.x + blockIdx.x) * hd + d] = s;
    }
}

// ---- optimizer ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

// per-block partial sums of squares of the (scaled) gradients and a non-finite flag
extern "C" __global__ void grad_sumsq_kernel(const float* __restrict__ g, size_t n, float* __restrict__ partial, float* __restrict__ nonfinite_partial) {
    __shared__ float red[32];
    float s = 0.f, bad = 0.f;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        const float v = g[i];
        if (!isfinite(v)) bad = 1.f; else s += v * v;
    }
    s = block_sum(s, red);
    bad = block_max(bad, red);
    if (threadIdx.x == 0) {
        partial[blockIdx.x] = s;
        nonfinite_partial[blockIdx.x] = bad;
    }
}

// ts[0] loss scale, ts[1] found_inf, ts[2] gradient norm (unscaled), ts[3] multiplier applied to the gradients in the update (clip / loss_scale)
extern "C" __global__ void grad_finalize_kernel(const float* __restrict__ partial, const float* __restrict__ nonfinite_partial, int nblocks, float max_norm, float* __restrict__ ts) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    float s = 0.f, bad = 0.f;
    for (int b = 0; b < nblocks; b++) {
        s += partial[b];
        bad = fmaxf(bad, nonfinite_partial[b]);
    }
    const float scale = ts[0];
    const float norm = sqrtf(s) / scale;
    ts[1] = bad;
    ts[2] = norm;
    const float clip = (max_norm > 0.f && norm > max_norm) ? max_norm / norm : 1.f;
    ts[3] = clip / scale;
}

// AdamW on flat arrays: elements below decay_end get decoupled weight decay. Skipped when the step's gradients were not finite. bc1 / bc2 are the bias corrections 1 - beta^t.
extern "C" __global__ void adamw_kernel(float* __restrict__ p, const float* __restrict__ g, float* __restrict__ m, float* __restrict__ v, half* __restrict__ p16, size_t n, size_t decay_end, float lr,
                                        float beta1, float beta2, float eps, float wd, float bc1, float bc2, const float* __restrict__ ts, size_t skip_lo, size_t skip_hi) {
    if (ts[1] != 0.f) return;
    const float gs = ts[3];
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        if (i >= skip_lo && i < skip_hi) continue;   // elements owned by another optimizer (the Muon matrices)
        const float gi = g[i] * gs;
        const float mi = beta1 * m[i] + (1.f - beta1) * gi;
        const float vi = beta2 * v[i] + (1.f - beta2) * gi * gi;
        m[i] = mi;
        v[i] = vi;
        float pi = p[i];
        if (i < decay_end) pi *= 1.f - lr * wd;
        pi -= lr * (mi / bc1) / (sqrtf(vi / bc2) + eps);
        p[i] = pi;
        p16[i] = __float2half(pi);
    }
}


// ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// Muon (momentum orthogonalised by Newton-Schulz). One matrix at a time: g (fp32 gradient, scaled by loss scale / clip through ts[3]), mom (fp32 momentum, updated in place).
//   mom = mu * mom + g;  u = g + mu * mom (Nesterov);  x = u / (|u|_F + 1e-7) in fp16;  five Newton-Schulz iterations on x (GEMMs); w -= lr * scale * x + lr * wd * w.
// Skipped when the step's gradients were not finite (ts[1] != 0).

constexpr int MUON_BLOCKS = 64;

// pass 1: momentum + Nesterov update, per-block sums of u^2; u goes to u32 (scratch), partial sums to partial[MUON_BLOCKS]
extern "C" __global__ void muon_prep_kernel(const float* __restrict__ g, float* __restrict__ mom, float* __restrict__ u32, size_t n, float mu, const float* __restrict__ ts, float* __restrict__ partial) {
    __shared__ float red[32];
    float s = 0.f;
    if (ts[1] == 0.f) {
        const float gs = ts[3];
        for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
            const float gi = g[i] * gs;
            const float mi = mu * mom[i] + gi;
            mom[i] = mi;
            const float ui = gi + mu * mi;
            u32[i] = ui;
            s += ui * ui;
        }
    }
    s = block_sum(s, red);
    if (threadIdx.x == 0) partial[blockIdx.x] = s;
}

// pass 2: x = u / (|u|_F + eps) as fp16
extern "C" __global__ void muon_normalize_kernel(const float* __restrict__ u32, const float* __restrict__ partial, half* __restrict__ x, size_t n, const float* __restrict__ ts) {
    if (ts[1] != 0.f) return;
    float s = 0.f;
    for (int b = 0; b < MUON_BLOCKS; b++) s += partial[b];
    const float inv = 1.f / (sqrtf(s) + 1e-7f);
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) x[i] = __float2half(u32[i] * inv);
}

// weight update with the orthogonalised direction: p = p (1 - lr wd) - lr scale x ; also refreshes the fp16 working copy
extern "C" __global__ void muon_apply_kernel(float* __restrict__ p, half* __restrict__ p16, const half* __restrict__ x, size_t n, float lr, float wd, float scale, const float* __restrict__ ts) {
    if (ts[1] != 0.f) return;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        const float pi = p[i] * (1.f - lr * wd) - lr * scale * __half2float(x[i]);
        p[i] = pi;
        p16[i] = __float2half(pi);
    }
}
