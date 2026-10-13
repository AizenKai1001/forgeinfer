// ForgeInfer decode kernels for Q8_0 weights (int8 + one fp16 scale per 32 weights, planar layout, repacked at load).
// Same stage structure as decode.cu; the activation vector is quantised to int8 per 32 values (scale d = amax / 127, q = round(x / d), as llama.cpp's Q8_1) in shared memory and the dot
// products use dp4a: sum += (float)(sum_int8(w * x)) * (d_w * d_x) per 32-weight block. A lane owns 16-weight chunks (one uint4 of weights); two adjacent lanes share a block scale.
//
// Layouts: weights  int8  [rows][K]      (K a multiple of 32, rows 16-byte aligned)
//          scales   half  [rows][K / 32]
// Dynamic shared memory of every kernel here: h fp32 [K] | xq int8 [K] | xd fp32 [K/32] | red fp32 [32]    (size: q8_smem_bytes(K) on the host)

__device__ __forceinline__ int dp4(uint32_t a, uint32_t b, int c) { return __dp4a((int)a, (int)b, c); }

// quantise h[0..K) (shared, fp32) to xq / xd; every warp takes blocks of 32
__device__ __forceinline__ void quantize_act_q8(const float* __restrict__ h, int K, int8_t* __restrict__ xq, float* __restrict__ xd) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
    for (int b = warp; b < K / 32; b += nw) {
        const float v = h[b * 32 + lane];
        const float amax = warp_max(fabsf(v));
        const float d = amax * (1.f / 127.f);
        const float id = d != 0.f ? 1.f / d : 0.f;
        xq[b * 32 + lane] = (int8_t)(int)roundf(v * id);
        if (lane == 0) xd[b] = d;
    }
}

template <int NI>
__device__ __forceinline__ void load_row_q8(const int8_t* __restrict__ q, const half* __restrict__ sc, int nchunk, int lane, uint4 (&r)[NI], unsigned short (&s)[NI]) {
    const uint4* p = reinterpret_cast<const uint4*>(q);
    const unsigned short* sp = reinterpret_cast<const unsigned short*>(sc);
#pragma unroll
    for (int i = 0; i < NI; i++) {
        const int c = lane + 32 * i;
        if (c < nchunk) {
            r[i] = __ldg(p + c);
            s[i] = __ldg(sp + (c >> 1));
        } else {
            r[i] = make_uint4(0u, 0u, 0u, 0u);
            s[i] = 0;
        }
    }
}

template <int NI>
__device__ __forceinline__ float dot_q8(const uint4 (&r)[NI], const unsigned short (&s)[NI], const int8_t* __restrict__ xq, const float* __restrict__ xd, int nchunk, int lane) {
    float acc = 0.f;
#pragma unroll
    for (int i = 0; i < NI; i++) {
        const int c = lane + 32 * i;
        if (c < nchunk) {
            const uint4 a = *reinterpret_cast<const uint4*>(xq + c * 16);
            int t = dp4(r[i].x, a.x, 0);
            t = dp4(r[i].y, a.y, t);
            t = dp4(r[i].z, a.z, t);
            t = dp4(r[i].w, a.w, t);
            acc += (float)t * (__half2float(__ushort_as_half(s[i])) * xd[c >> 1]);
        }
    }
    return warp_sum(acc);
}

// ---- S1 --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
template <int NI>
__device__ __forceinline__ void norm_qkv_rope_q8_impl(const float* __restrict__ x, const float* __restrict__ norm_w, const int8_t* __restrict__ wq, const half* __restrict__ sq, const int8_t* __restrict__ wk,
                                                      const half* __restrict__ sk, const int8_t* __restrict__ wv, const half* __restrict__ sv, float* __restrict__ q_out, float* __restrict__ k_raw,
                                                      half* __restrict__ k_cache, half* __restrict__ v_cache, const float* __restrict__ cos_t, const float* __restrict__ sin_t,
                                                      const StepState* __restrict__ st, int E, int n_head, int n_kv, int hd, float eps, int rotate) {
    constexpr int NX = 2 * NI * 2;   // E <= 256 * NI * ... : activation elements per thread (128 threads): E <= 128 * NX
    extern __shared__ float smem[];
    float* h = smem;
    int8_t* xq = reinterpret_cast<int8_t*>(smem + E);
    float* xd = reinterpret_cast<float*>(xq + E);
    float* red = xd + E / 32;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int half_d = hd / 2;
    const int nq = n_head * half_d, nk = n_kv * half_d, nv = n_kv * half_d;
    const int u = blockIdx.x * (blockDim.x >> 5) + warp;
    const int nchunk = E / 16;
    float xv[NX], wn[NX];
#pragma unroll
    for (int k = 0; k < NX; k++) {
        const int i = threadIdx.x + k * blockDim.x;
        xv[k] = i < E ? x[i] : 0.f;
        wn[k] = i < E ? norm_w[i] : 0.f;
    }
    int which = 0, idx = 0, head = 0, i = 0;
    uint4 r0[NI], r1[NI];
    unsigned short s0[NI], s1[NI];
    const bool active = u < nq + nk + nv;
    if (active) {
        if (u < nq) { which = 0; idx = u; }
        else if (u < nq + nk) { which = 1; idx = u - nq; }
        else { which = 2; idx = u - nq - nk; }
        head = idx / half_d;
        i = idx % half_d;
        const int8_t* w = which == 0 ? wq : (which == 1 ? wk : wv);
        const half* sc = which == 0 ? sq : (which == 1 ? sk : sv);
        const int row0 = head * hd + i, row1 = head * hd + i + half_d;
        load_row_q8<NI>(w + (size_t)row0 * E, sc + (size_t)row0 * (E / 32), nchunk, lane, r0, s0);
        load_row_q8<NI>(w + (size_t)row1 * E, sc + (size_t)row1 * (E / 32), nchunk, lane, r1, s1);
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
    quantize_act_q8(h, E, xq, xd);
    __syncthreads();
    if (!active) return;
    const float a = dot_q8<NI>(r0, s0, xq, xd, nchunk, lane);
    const float b = dot_q8<NI>(r1, s1, xq, xd, nchunk, lane);
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

#define INST_QKV_Q8(NI)                                                                                                                                                                     \
    extern "C" __global__ void norm_qkv_rope_q8_kernel_##NI(const float* x, const float* norm_w, const int8_t* wq, const half* sq, const int8_t* wk, const half* sk, const int8_t* wv,         \
                                                            const half* sv, float* q_out, float* k_raw, half* k_cache, half* v_cache, const float* cos_t, const float* sin_t,            \
                                                            const StepState* st, int E, int n_head, int n_kv, int hd, float eps, int rotate) {                                            \
        norm_qkv_rope_q8_impl<NI>(x, norm_w, wq, sq, wk, sk, wv, sv, q_out, k_raw, k_cache, v_cache, cos_t, sin_t, st, E, n_head, n_kv, hd, eps, rotate);                                 \
    }
INST_QKV_Q8(2) INST_QKV_Q8(3) INST_QKV_Q8(4) INST_QKV_Q8(6) INST_QKV_Q8(8)

// ---- S3 / S5 ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
template <int NI>
__device__ __forceinline__ void matvec_residual_q8_impl(const int8_t* __restrict__ w, const half* __restrict__ sc, const float* __restrict__ vin, float* __restrict__ x, int rows, int K) {
    constexpr int NV = 4 * NI;
    extern __shared__ float smem[];
    float* vs = smem;
    int8_t* xq = reinterpret_cast<int8_t*>(smem + K);
    float* xd = reinterpret_cast<float*>(xq + K);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
    const int r = blockIdx.x * nw + warp;
    const int nchunk = K / 16;
    float vv[NV];
#pragma unroll
    for (int k = 0; k < NV; k++) {
        const int i = threadIdx.x + k * blockDim.x;
        vv[k] = i < K ? vin[i] : 0.f;
    }
    uint4 rg[NI];
    unsigned short sg[NI];
    if (r < rows) load_row_q8<NI>(w + (size_t)r * K, sc + (size_t)r * (K / 32), nchunk, lane, rg, sg);
#pragma unroll
    for (int k = 0; k < NV; k++) {
        const int i = threadIdx.x + k * blockDim.x;
        if (i < K) vs[i] = vv[k];
    }
    __syncthreads();
    quantize_act_q8(vs, K, xq, xd);
    __syncthreads();
    if (r >= rows) return;
    const float acc = dot_q8<NI>(rg, sg, xq, xd, nchunk, lane);
    if (lane == 0) x[r] += acc;
}

#define INST_MV_Q8(NI)                                                                                                                                                                       \
    extern "C" __global__ void matvec_residual_q8_kernel_##NI(const int8_t* w, const half* sc, const float* vin, float* x, int rows, int K) { matvec_residual_q8_impl<NI>(w, sc, vin, x, rows, K); }
INST_MV_Q8(2) INST_MV_Q8(3) INST_MV_Q8(4) INST_MV_Q8(6) INST_MV_Q8(8)

// ---- S4 --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
template <int NI, int RPW>
__device__ __forceinline__ void norm_gateup_q8_impl(const float* __restrict__ x, const float* __restrict__ norm_w, const int8_t* __restrict__ wg, const half* __restrict__ sg_, const int8_t* __restrict__ wu,
                                                    const half* __restrict__ su_, float* __restrict__ act, int E, int F, float eps) {
    constexpr int NX = 4 * NI;
    extern __shared__ float smem[];
    float* h = smem;
    int8_t* xq = reinterpret_cast<int8_t*>(smem + E);
    float* xd = reinterpret_cast<float*>(xq + E);
    float* red = xd + E / 32;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
    const int f0 = (blockIdx.x * nw + warp) * RPW;
    const int nchunk = E / 16;
    float xv[NX], wn[NX];
#pragma unroll
    for (int k = 0; k < NX; k++) {
        const int i = threadIdx.x + k * blockDim.x;
        xv[k] = i < E ? x[i] : 0.f;
        wn[k] = i < E ? norm_w[i] : 0.f;
    }
    uint4 g[RPW][NI], u[RPW][NI];
    unsigned short gs[RPW][NI], us[RPW][NI];
#pragma unroll
    for (int j = 0; j < RPW; j++) {
        if (f0 + j < F) {
            load_row_q8<NI>(wg + (size_t)(f0 + j) * E, sg_ + (size_t)(f0 + j) * (E / 32), nchunk, lane, g[j], gs[j]);
            load_row_q8<NI>(wu + (size_t)(f0 + j) * E, su_ + (size_t)(f0 + j) * (E / 32), nchunk, lane, u[j], us[j]);
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
    quantize_act_q8(h, E, xq, xd);
    __syncthreads();
#pragma unroll
    for (int j = 0; j < RPW; j++) {
        if (f0 + j < F) {
            const float a = dot_q8<NI>(g[j], gs[j], xq, xd, nchunk, lane);
            const float b = dot_q8<NI>(u[j], us[j], xq, xd, nchunk, lane);
            if (lane == 0) act[f0 + j] = (a / (1.f + __expf(-a))) * b;
        }
    }
}

#define INST_GU_Q8(NI)                                                                                                                                                                       \
    extern "C" __global__ void norm_gateup_q8_kernel_##NI(const float* x, const float* norm_w, const int8_t* wg, const half* sg, const int8_t* wu, const half* su, float* act, int E, int F,  \
                                                          float eps) {                                                                                                                       \
        norm_gateup_q8_impl<NI, 2>(x, norm_w, wg, sg, wu, su, act, E, F, eps);                                                                                                               \
    }
INST_GU_Q8(2) INST_GU_Q8(3) INST_GU_Q8(4) INST_GU_Q8(6) INST_GU_Q8(8)

// ---- head: logits[r] = head[r] . h, four rows per warp iteration (all their loads in flight) --------------------------------------------------------------------------------------
template <int NI>
__device__ __forceinline__ void head_q8_impl(const int8_t* __restrict__ w, const half* __restrict__ sc, const float* __restrict__ hin, float* __restrict__ logits, int rows, int K) {
    extern __shared__ float smem[];
    float* vs = smem;
    int8_t* xq = reinterpret_cast<int8_t*>(smem + K);
    float* xd = reinterpret_cast<float*>(xq + K);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
    const int nchunk = K / 16;
    for (int i = threadIdx.x; i < K; i += blockDim.x) vs[i] = hin[i];
    __syncthreads();
    quantize_act_q8(vs, K, xq, xd);
    __syncthreads();
    constexpr int R = 4;
    for (int r0 = (blockIdx.x * nw + warp) * R; r0 < rows; r0 += gridDim.x * nw * R) {
        uint4 rg[R][NI];
        unsigned short sg[R][NI];
#pragma unroll
        for (int j = 0; j < R; j++) {
            if (r0 + j < rows) load_row_q8<NI>(w + (size_t)(r0 + j) * K, sc + (size_t)(r0 + j) * (K / 32), nchunk, lane, rg[j], sg[j]);
        }
#pragma unroll
        for (int j = 0; j < R; j++) {
            if (r0 + j < rows) {
                const float acc = dot_q8<NI>(rg[j], sg[j], xq, xd, nchunk, lane);
                if (lane == 0) logits[r0 + j] = acc;
            }
        }
    }
}

#define INST_HEAD_Q8(NI)                                                                                                                                                                     \
    extern "C" __global__ void head_q8_kernel_##NI(const int8_t* w, const half* sc, const float* hin, float* logits, int rows, int K) { head_q8_impl<NI>(w, sc, hin, logits, rows, K); }
INST_HEAD_Q8(2) INST_HEAD_Q8(3) INST_HEAD_Q8(4) INST_HEAD_Q8(6) INST_HEAD_Q8(8)

// embedding gather for Q8_0 tables: x[i] = scale * q
extern "C" __global__ void embed_q8_kernel(const int8_t* __restrict__ q, const half* __restrict__ sc, const StepState* __restrict__ st, float* __restrict__ x, int E) {
    const size_t row = st->token;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < E; i += gridDim.x * blockDim.x) x[i] = (float)q[row * E + i] * __half2float(sc[row * (E / 32) + i / 32]);
}
