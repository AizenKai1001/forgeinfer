// Sample-efficiency options of the trainer (each one switched on by a flag; with every option off the trainer launches only the kernels of train.cu / train_f1.cu, as before).
// Executable specifications: tools/ref_muon.py (the Muon variants) and tools/gen_train_goldens.py (the model options). Conventions as in train.cu.

#include <cuda_fp16.h>
#include <stdint.h>

// ---- Muon variants -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

// Normalisation of the Muon input with a margin: x = u / (|u|_F * margin + eps) as fp16 (Polar Express: margin 1.02, eps 1e-6, so that every singular value is safely below 1).
// partial[nparts]: the per-block sums of u^2 written by muon_prep_kernel. Skipped when the step's gradients were not finite (ts[1] != 0).
extern "C" __global__ void muon_normalize_m_kernel(const float* __restrict__ u32, const float* __restrict__ partial, int nparts, half* __restrict__ x, size_t n, const float* __restrict__ ts, float margin, float eps) {
    if (ts[1] != 0.f) return;
    float s = 0.f;
    for (int b = 0; b < nparts; b++) s += partial[b];
    const float inv = 1.f / (sqrtf(s) * margin + eps);
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) x[i] = __float2half(u32[i] * inv);
}

// NorMuon (Li et al. 2025, arXiv 2510.05491; modded-nanogpt record 41): neuron-wise normalisation of the orthogonalised update x (rows x cols, fp16). Per lane (a row, or a column) the mean of x^2 over the lane,
// its running average v (in place: v = beta2 v + (1 - beta2) mean, no bias correction) and the gain rsqrt(max(v, 1e-10)). by_row = 1: the lanes are the rows (mean over the columns), 0: the columns.
// lane_pow[lanes] and gain[lanes] are scratch for normuon_ratio_kernel. Skipped when the step's gradients were not finite: v keeps its value.
// Launch: by_row: one warp per row (blockDim a multiple of 32); columns: one thread per column (consecutive threads read consecutive columns of a row).
extern "C" __global__ void normuon_lanes_kernel(const half* __restrict__ x, float* __restrict__ v, float* __restrict__ lane_pow, float* __restrict__ gain, int rows, int cols, int by_row, float beta2,
                                                const float* __restrict__ ts) {
    if (ts[1] != 0.f) return;
    if (by_row) {
        const int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), lane = threadIdx.x & 31;
        if (row >= rows) return;
        const half* xr = x + (size_t)row * cols;
        float s = 0.f;
        for (int c = lane; c < cols; c += 32) {
            const float t = __half2float(xr[c]);
            s += t * t;
        }
        s = warp_sum(s);
        if (lane == 0) {
            const float m = s / (float)cols;
            const float vv = beta2 * v[row] + (1.f - beta2) * m;
            v[row] = vv;
            lane_pow[row] = m;
            gain[row] = rsqrtf(fmaxf(vv, 1e-10f));
        }
    } else {
        const int col = blockIdx.x * blockDim.x + threadIdx.x;
        if (col >= cols) return;
        float s = 0.f;
        for (int r = 0; r < rows; r++) {
            const float t = __half2float(x[(size_t)r * cols + col]);
            s += t * t;
        }
        const float m = s / (float)rows;
        const float vv = beta2 * v[col] + (1.f - beta2) * m;
        v[col] = vv;
        lane_pow[col] = m;
        gain[col] = rsqrtf(fmaxf(vv, 1e-10f));
    }
}

// The factor that gives the equalised update the Frobenius norm of the orthogonalised one: out[0] = sqrt(len sum lane_pow) / max(sqrt(len sum lane_pow gain^2), 1e-10)  (len = the length of a lane).
// One block, fixed summation order (deterministic). Skipped when the step's gradients were not finite.
extern "C" __global__ void normuon_ratio_kernel(const float* __restrict__ lane_pow, const float* __restrict__ gain, int lanes, int len, float* __restrict__ out, const float* __restrict__ ts) {
    __shared__ float red[32];
    if (ts[1] != 0.f) return;
    float a = 0.f, b = 0.f;
    for (int i = threadIdx.x; i < lanes; i += blockDim.x) {
        const float p = lane_pow[i], g = gain[i];
        a += p;
        b += p * g * g;
    }
    a = block_sum(a, red);
    __syncthreads();
    b = block_sum(b, red);
    if (threadIdx.x == 0) out[0] = sqrtf(a * (float)len) / fmaxf(sqrtf(b * (float)len), 1e-10f);
}

// Muon weight update with the options:  u = scale * x (* gain[lane] * ratio[0] with NorMuon);  p = p - lr wd m p - lr u  with m = 1, or with cautious weight decay (Chen et al. 2025, arXiv 2510.12402)
// m = (u p >= 0): only the entries the update moves towards zero decay. lane_mode 0: no NorMuon; 1: lanes are rows (gain[i / cols]); 2: lanes are columns (gain[i % cols]). Refreshes the fp16 working copy.
// Skipped when the step's gradients were not finite (ts[1] != 0).
extern "C" __global__ void muon_apply_opt_kernel(float* __restrict__ p, half* __restrict__ p16, const half* __restrict__ x, size_t n, int cols, float lr, float wd, float scale, const float* __restrict__ ts,
                                                 const float* __restrict__ gain, const float* __restrict__ ratio, int lane_mode, int cautious) {
    if (ts[1] != 0.f) return;
    const float r = lane_mode ? ratio[0] : 1.f;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        float u = scale * __half2float(x[i]);
        if (lane_mode == 1) u *= gain[i / (size_t)cols] * r;
        else if (lane_mode == 2) u *= gain[i % (size_t)cols] * r;
        const float pi = p[i];
        const float d = (cautious && u * pi < 0.f) ? 0.f : lr * wd * pi;
        const float pn = pi - d - lr * u;
        p[i] = pn;
        p16[i] = __float2half(pn);
    }
}

// ---- gated attention ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// Head-wise output gate (Qiu et al. 2025, arXiv 2505.06708, position G1, head-wise; with K < E the sparse gate of modded-nanogpt record 28): the output of head h at token t is multiplied by
// g[t][h] = sigmoid(sum_{k < K} h1[t][k] W[h][k]) before the output projection; h1 is the normalised attention input (fp16 [N][E]), W the gate weights [H][K] (fp32 masters).

// g[t][h] (saved for the backward) and og = o * g (fp16 [N][H hd], the input of the output projection). One warp per token, eight per block; H <= 32.
extern "C" __global__ void attn_gate_fwd_kernel(const half* __restrict__ h1, const float* __restrict__ w, const half* __restrict__ o, float* __restrict__ g, half* __restrict__ og, int N, int E, int K, int H,
                                                int hd) {
    __shared__ float gs[8][32];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int t = blockIdx.x * 8 + warp;
    if (t >= N) return;
    const half* x = h1 + (size_t)t * E;
    for (int h = 0; h < H; h++) {
        float s = 0.f;
        for (int k = lane; k < K; k += 32) s += __half2float(x[k]) * w[h * K + k];
        s = warp_sum(s);
        if (lane == 0) gs[warp][h] = 1.f / (1.f + __expf(-s));
    }
    __syncwarp();
    if (lane < H) g[(size_t)t * H + lane] = gs[warp][lane];
    const int qd = H * hd;
    const half* orow = o + (size_t)t * qd;
    half* ogrow = og + (size_t)t * qd;
    for (int i = lane * 2; i < qd; i += 64) {
        const float gv = gs[warp][i / hd];
        const float2 v = __half22float2(*reinterpret_cast<const __half2*>(orow + i));
        *reinterpret_cast<__half2*>(ogrow + i) = __floats2half2_rn(v.x * gv, v.y * gv);
    }
}

// og = o * g again (the backward recomputes the input of the output projection instead of keeping it per layer)
extern "C" __global__ void attn_gate_mul_kernel(const half* __restrict__ o, const float* __restrict__ g, half* __restrict__ og, int N, int H, int hd) {
    const size_t pairs = (size_t)N * H * hd / 2;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < pairs; i += (size_t)gridDim.x * blockDim.x) {
        const float gv = g[(2 * i) / hd];
        const float2 v = __half22float2(reinterpret_cast<const __half2*>(o)[i]);
        reinterpret_cast<__half2*>(og)[i] = __floats2half2_rn(v.x * gv, v.y * gv);
    }
}

// Backward through the gate. dog (fp16 [N][H hd]) holds the gradient of og on entry and the gradient of o (dog * g) on exit; dpre[t][h] = (sum_d dog o) g (1 - g), the gradient of the gate's logit.
// One warp per (token, head).
extern "C" __global__ void attn_gate_bwd_kernel(half* __restrict__ dog, const half* __restrict__ o, const float* __restrict__ g, float* __restrict__ dpre, int N, int H, int hd) {
    const long item = (long)blockIdx.x * 8 + (threadIdx.x >> 5);
    if (item >= (long)N * H) return;
    const int lane = threadIdx.x & 31;
    const size_t off = (size_t)item * hd;
    const float gv = g[item];
    float s = 0.f;
    for (int d = lane * 2; d < hd; d += 64) {
        __half2* dp = reinterpret_cast<__half2*>(dog + off + d);
        const float2 a = __half22float2(*dp), b = __half22float2(*reinterpret_cast<const __half2*>(o + off + d));
        s += a.x * b.x + a.y * b.y;
        *dp = __floats2half2_rn(a.x * gv, a.y * gv);
    }
    s = warp_sum(s);
    if (lane == 0) dpre[item] = s * gv * (1.f - gv);
}

// Gate weight gradient, per block: partial[block][h K + k] = sum over the block's tokens (t = block, block + gridDim, ...) of dpre[t][h] h1[t][k]; reduced in fixed order by reduce_partials_kernel.
extern "C" __global__ void attn_gate_dw_kernel(const float* __restrict__ dpre, const half* __restrict__ h1, float* __restrict__ partial, int N, int E, int K, int H) {
    const int HK = H * K;
    for (int idx = threadIdx.x; idx < HK; idx += blockDim.x) {
        const int h = idx / K, k = idx % K;
        float s = 0.f;
        for (int t = blockIdx.x; t < N; t += gridDim.x) s += dpre[(size_t)t * H + h] * __half2float(h1[(size_t)t * E + k]);
        partial[(size_t)blockIdx.x * HK + idx] = s;
    }
}

// The gate's share of the gradient of the normalised attention input: dx[t][k] += sum_h dpre[t][h] W[h][k]  (k < K; dx fp32 [N][E])
extern "C" __global__ void attn_gate_dx_kernel(const float* __restrict__ dpre, const float* __restrict__ w, float* __restrict__ dx, int N, int E, int K, int H) {
    const size_t total = (size_t)N * K;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < total; i += (size_t)gridDim.x * blockDim.x) {
        const size_t t = i / K;
        const int k = (int)(i % K);
        float s = 0.f;
        for (int h = 0; h < H; h++) s += dpre[t * H + h] * w[h * K + k];
        dx[t * E + k] += s;
    }
}

// ---- Peri-LN (Kim et al. 2025, arXiv 2502.02732): an RMSNorm on the output of every sublayer before the residual add (and optionally on the embedding output) -----------------------------------------------
// out = res + y * rstd * w  (res may be null: out = y * rstd * w), rstd saved per row; y fp16 (the sublayer output) or fp32 (the embedding output). One warp per row, eight rows per block; E a multiple of 4.
template <typename T>
__device__ __forceinline__ float4 load4(const T* p, int i);
template <>
__device__ __forceinline__ float4 load4<float>(const float* p, int i) {
    return reinterpret_cast<const float4*>(p)[i];
}
template <>
__device__ __forceinline__ float4 load4<half>(const half* p, int i) {
    const uint2 u = reinterpret_cast<const uint2*>(p)[i];
    const float2 a = __half22float2(*reinterpret_cast<const __half2*>(&u.x)), b = __half22float2(*reinterpret_cast<const __half2*>(&u.y));
    return make_float4(a.x, a.y, b.x, b.y);
}

template <typename T>
__device__ __forceinline__ void rmsnorm_res_fwd(const T* __restrict__ y, const float* __restrict__ w, const float* __restrict__ res, float* __restrict__ out, float* __restrict__ rstd, int N, int E, float eps) {
    const int row = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (row >= N) return;
    const T* yr = y + (size_t)row * E;
    const int E4 = E >> 2;
    float ss = 0.f;
    for (int i = lane; i < E4; i += 32) {
        const float4 v = load4<T>(yr, i);
        ss += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }
    ss = warp_sum(ss);
    const float r = rsqrtf(ss / (float)E + eps);
    if (lane == 0) rstd[row] = r;
    const float4* w4 = reinterpret_cast<const float4*>(w);
    float4* o4 = reinterpret_cast<float4*>(out + (size_t)row * E);
    for (int i = lane; i < E4; i += 32) {
        const float4 v = load4<T>(yr, i), g = w4[i];
        float4 o = res ? reinterpret_cast<const float4*>(res + (size_t)row * E)[i] : make_float4(0.f, 0.f, 0.f, 0.f);
        o.x += v.x * r * g.x;
        o.y += v.y * r * g.y;
        o.z += v.z * r * g.z;
        o.w += v.w * r * g.w;
        o4[i] = o;
    }
}

extern "C" __global__ void rmsnorm_res_fwd_h_kernel(const half* __restrict__ y, const float* __restrict__ w, const float* __restrict__ res, float* __restrict__ out, float* __restrict__ rstd, int N, int E, float eps) {
    rmsnorm_res_fwd<half>(y, w, res, out, rstd, N, E, eps);
}

extern "C" __global__ void rmsnorm_res_fwd_f_kernel(const float* __restrict__ y, const float* __restrict__ w, const float* __restrict__ res, float* __restrict__ out, float* __restrict__ rstd, int N, int E, float eps) {
    rmsnorm_res_fwd<float>(y, w, res, out, rstd, N, E, eps);
}

// rmsnorm_bwd_kernel of train.cu with an fp16 norm input x (the saved sublayer output): dx_out = dres + rmsnorm_backward(dy, x) and its fp16 copy dx16 (if given); dw_partial[block][E] = sum over the block's
// rows of dy * x * rstd (reduced in fixed order by reduce_partials_kernel). One warp per row (eight per block, rows strided over all warps); E <= 1024. Dynamic shared memory: 8 * E floats.
extern "C" __global__ void rmsnorm_bwd_h_kernel(const float* __restrict__ dy, const half* __restrict__ x, const float* __restrict__ w, const float* __restrict__ rstd, const float* __restrict__ dres,
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
        const half* xr = x + (size_t)row * E;
        float4 dv[8], xv[8];
        float dot = 0.f;
#pragma unroll
        for (int j = 0; j < 8; j++) {
            const int i = lane + 32 * j;
            if (i < E4) {
                dv[j] = dyr[i];
                xv[j] = load4<half>(xr, i);
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

// ---- cautious weight decay for AdamW (modded-nanogpt record 50) -------------------------------------------------------------------------------------------------------------------------------------------
// adamw_kernel of train.cu, except that an element below decay_end decays only where the update moves it towards zero (update * p >= 0; the update has the sign of the first moment).
extern "C" __global__ void adamw_cwd_kernel(float* __restrict__ p, const float* __restrict__ g, float* __restrict__ m, float* __restrict__ v, half* __restrict__ p16, size_t n, size_t decay_end, float lr,
                                            float beta1, float beta2, float eps, float wd, float bc1, float bc2, const float* __restrict__ ts, size_t skip_lo, size_t skip_hi) {
    if (ts[1] != 0.f) return;
    const float gs = ts[3];
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        if (i >= skip_lo && i < skip_hi) continue;
        const float gi = g[i] * gs;
        const float mi = beta1 * m[i] + (1.f - beta1) * gi;
        const float vi = beta2 * v[i] + (1.f - beta2) * gi * gi;
        m[i] = mi;
        v[i] = vi;
        float pi = p[i];
        const float u = (mi / bc1) / (sqrtf(vi / bc2) + eps);
        if (i < decay_end && u * pi >= 0.f) pi *= 1.f - lr * wd;
        pi -= lr * u;
        p[i] = pi;
        p16[i] = __float2half(pi);
    }
}
