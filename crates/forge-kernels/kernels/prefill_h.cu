// Batched (prompt processing) kernels for hybrid models (Qwen3.5), around the tensor-core GEMMs: a chunk of T tokens goes through a Gated DeltaNet layer as
//   h = rmsnorm(x) -> GEMMs (qkv, z, alpha|beta; fp32) -> gdn_prep (conv + SiLU + L2 norms + gates for every token and head in parallel) -> gdn_scan (the delta rule, token-serial per
//   value head, the state S in registers for the whole chunk) -> gdn_post (gated RMSNorm, fp16) -> GEMM W_out (+= into x)
// and through a gated attention layer as
//   h -> GEMM (q|gate per head, k, v) -> qkv35_rope_batch (norm + partial RoPE + KV write) -> attn_prefill_256g (output gate in the epilogue) -> GEMM W_o (+= into x).
// The math is the decode step of decode_h.cu applied token after token, so the two paths agree with each other and with the CPU reference.

// grid (T, n_v), block 384: thread = channel of the head's 128 q | 128 k | 128 v convolution channels. Outputs per (token, head): L2-normalised q (already scaled by 1/sqrt(128)) and k, the
// convolved v, and gates {beta, decay, k.q}. The old conv_state is read-only here (gdn_scan_kernel writes the new window after this kernel has finished).
extern "C" __global__ void __launch_bounds__(384) gdn_prep_kernel(const float* __restrict__ mixed, const float* __restrict__ ab, const float* __restrict__ conv_w, const float* __restrict__ conv_state,
                                                                 const float* __restrict__ a_neg, const float* __restrict__ dt_bias, float* __restrict__ pq, float* __restrict__ pk,
                                                                 float* __restrict__ pv, float* __restrict__ pg, int n_k, int n_v, int tiled, int conv_k, int conv_dim) {
    __shared__ float qn[GDN_D], kn[GDN_D], red[16];
    const int t = blockIdx.x, h = blockIdx.y, tid = threadIdx.x;
    const int kh = tiled ? (h % n_k) : (h / (n_v / n_k));
    const int key_dim = n_k * GDN_D;
    const int part = tid / GDN_D, i = tid % GDN_D;
    const int ch = part == 0 ? kh * GDN_D + i : (part == 1 ? key_dim + kh * GDN_D + i : 2 * key_dim + h * GDN_D + i);
    const float* w = conv_w + (size_t)ch * conv_k;
    float acc = 0.f;
    for (int j = 0; j < conv_k; j++) {
        const int s = t - (conv_k - 1) + j;   // input column; negative = the carried window
        const float x = s >= 0 ? mixed[(size_t)s * conv_dim + ch] : conv_state[((size_t)h * (conv_k - 1) + (conv_k - 1 + s)) * 384 + tid];
        acc += w[j] * x;
    }
    const float c = h_silu(acc);
    const size_t base = ((size_t)t * n_v + h) * GDN_D;
    if (part == 2) pv[base + i] = c;
    if (part == 0) qn[i] = c;
    else if (part == 1) kn[i] = c;
    __syncthreads();
    const int w8 = tid >> 5, lane = tid & 31;
    if (w8 < 8) {
        const float v = (w8 < 4 ? qn : kn)[(w8 & 3) * 32 + lane];
        const float s2 = warp_sum(v * v);
        if (lane == 0) red[w8] = s2;
    }
    __syncthreads();
    const float rq = rsqrtf(red[0] + red[1] + red[2] + red[3] + 1e-6f) * 0.08838834764831845f;
    const float rk = rsqrtf(red[4] + red[5] + red[6] + red[7] + 1e-6f);
    float qv = 0.f, kv = 0.f;
    if (part == 0) { qv = qn[i] * rq; pq[base + i] = qv; qn[i] = qv; }
    else if (part == 1) { kv = kn[i] * rk; pk[base + i] = kv; kn[i] = kv; }
    __syncthreads();
    if (w8 < 4) {
        const float s2 = warp_sum(kn[w8 * 32 + lane] * qn[w8 * 32 + lane]);
        if (lane == 0) red[8 + w8] = s2;
    }
    __syncthreads();
    if (tid == 0) {
        const float kq = red[8] + red[9] + red[10] + red[11];
        const float* abt = ab + (size_t)t * 2 * n_v;
        const float beta = h_sigmoid(abt[n_v + h]);
        const float a = __expf(a_neg[h] * h_softplus(abt[h] + dt_bias[h]));
        float* g = pg + ((size_t)t * n_v + h) * 3;
        g[0] = beta;
        g[1] = a;
        g[2] = kq;
    }
}

// grid (n_v, 128 / CPB), block CPB * 8: the thread mapping of gdn_step_kernel for CPB of the head's 128 state columns (the columns of the recurrence are independent given k, q and the gates, so a head
// can be spread over several blocks and SMs). S stays in registers over the whole chunk; so[t][h][col] receives the raw output of the recurrence.
// The per-token inputs (k, q, v, gates) are staged through shared memory in tiles of GS_TT tokens, the NEXT tile's global loads are in flight while the current tile is computed: the recurrence
// itself is a chain of a few hundred cycles per token, the exposed DRAM latency of a one-token prefetch was ten times that. The outputs of a tile leave as one coalesced store.
// The block with blockIdx.y == 0 finally replaces the head's convolution window by the last conv_k - 1 input columns of the chunk.
#define GS_TT 8
template <int CPB>
__device__ __forceinline__ void gdn_scan_impl(const float* __restrict__ pq, const float* __restrict__ pk, const float* __restrict__ pv, const float* __restrict__ pg, const float* __restrict__ mixed,
                                              float* __restrict__ conv_state, float* __restrict__ S, float* __restrict__ so, int T, int n_k, int n_v, int tiled, int conv_k, int conv_dim) {
    constexpr int NT = CPB * 8;                       // threads
    constexpr int NV4 = GS_TT * CPB / 4;              // float4 of v per tile
    constexpr int TOTAL = 2 * GS_TT * 32 + NV4;       // float4 per tile: k, q, v
    constexpr int ITEMS = (TOTAL + NT - 1) / NT;
    __shared__ __align__(16) float sk[2][GS_TT * GDN_D];
    __shared__ __align__(16) float sq[2][GS_TT * GDN_D];
    __shared__ __align__(16) float sv[2][GS_TT * CPB];
    __shared__ __align__(16) float sg[2][GS_TT * 4];
    __shared__ __align__(16) float sot[2][GS_TT * CPB];
    const int h = blockIdx.x, c0 = blockIdx.y * CPB, tid = threadIdx.x;
    const int lcol = tid >> 3, l8 = tid & 7;          // column within the block, lane within the column
    const int col = c0 + lcol;
    float4* s_ptr = reinterpret_cast<float4*>(S + ((size_t)h * GDN_D + col) * GDN_D);
    float4 s[4];
#pragma unroll
    for (int r = 0; r < 4; r++) s[r] = s_ptr[8 * r + l8];

    const int ntiles = (T + GS_TT - 1) / GS_TT;
    float4 pre[ITEMS];
    float pre_g = 0.f;
    auto load_tile = [&](int tile) {
        const int t0 = tile * GS_TT;
#pragma unroll
        for (int n = 0; n < ITEMS; n++) {
            const int i = tid + n * NT;
            float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
            if (i < GS_TT * 32) {                                   // k: token i / 32, float4 i % 32 of the head's 128 values
                const int j = i >> 5, c = i & 31;
                if (t0 + j < T) v = __ldg(reinterpret_cast<const float4*>(pk + ((size_t)(t0 + j) * n_v + h) * GDN_D) + c);
            } else if (i < 2 * GS_TT * 32) {                        // q
                const int ii = i - GS_TT * 32, j = ii >> 5, c = ii & 31;
                if (t0 + j < T) v = __ldg(reinterpret_cast<const float4*>(pq + ((size_t)(t0 + j) * n_v + h) * GDN_D) + c);
            } else if (i < TOTAL) {                                 // v: this block's CPB columns
                const int ii = i - 2 * GS_TT * 32, j = ii / (CPB / 4), c = ii % (CPB / 4);
                if (t0 + j < T) v = __ldg(reinterpret_cast<const float4*>(pv + ((size_t)(t0 + j) * n_v + h) * GDN_D + c0) + c);
            }
            pre[n] = v;
        }
        pre_g = 0.f;
        if (tid < GS_TT * 3) {
            const int j = tid / 3, c = tid % 3;
            if (t0 + j < T) pre_g = __ldg(pg + ((size_t)(t0 + j) * n_v + h) * 3 + c);
        }
    };
    auto store_tile = [&](int buf) {
#pragma unroll
        for (int n = 0; n < ITEMS; n++) {
            const int i = tid + n * NT;
            if (i < GS_TT * 32) reinterpret_cast<float4*>(sk[buf])[i] = pre[n];
            else if (i < 2 * GS_TT * 32) reinterpret_cast<float4*>(sq[buf])[i - GS_TT * 32] = pre[n];
            else if (i < TOTAL) reinterpret_cast<float4*>(sv[buf])[i - 2 * GS_TT * 32] = pre[n];
        }
        if (tid < GS_TT * 3) sg[buf][(tid / 3) * 4 + tid % 3] = pre_g;
    };
    load_tile(0);
    store_tile(0);
    __syncthreads();
    for (int tile = 0; tile < ntiles; tile++) {
        const int cur = tile & 1;
        const int t0 = tile * GS_TT;
        if (tile + 1 < ntiles) load_tile(tile + 1);
        const int nt = min(GS_TT, T - t0);
        for (int j = 0; j < nt; j++) {
            const float4* kj = reinterpret_cast<const float4*>(&sk[cur][j * GDN_D]);
            const float4* qj = reinterpret_cast<const float4*>(&sq[cur][j * GDN_D]);
            const float beta = sg[cur][j * 4], a = sg[cur][j * 4 + 1], kq = sg[cur][j * 4 + 2];
            float ks = 0.f, qs = 0.f;
#pragma unroll
            for (int r = 0; r < 4; r++) {
                const float4 k4 = kj[8 * r + l8], q4 = qj[8 * r + l8];
                ks = fmaf(s[r].x, k4.x, ks);
                ks = fmaf(s[r].y, k4.y, ks);
                ks = fmaf(s[r].z, k4.z, ks);
                ks = fmaf(s[r].w, k4.w, ks);
                qs = fmaf(s[r].x, q4.x, qs);
                qs = fmaf(s[r].y, q4.y, qs);
                qs = fmaf(s[r].z, q4.z, qs);
                qs = fmaf(s[r].w, q4.w, qs);
            }
#pragma unroll
            for (int m = 1; m < 8; m <<= 1) {
                ks += __shfl_xor_sync(0xffffffffu, ks, m);
                qs += __shfl_xor_sync(0xffffffffu, qs, m);
            }
            const float u = beta * (sv[cur][j * CPB + lcol] - a * ks);
            const float o = a * qs + kq * u;
#pragma unroll
            for (int r = 0; r < 4; r++) {
                const float4 k4 = kj[8 * r + l8];
                s[r].x = fmaf(a, s[r].x, k4.x * u);
                s[r].y = fmaf(a, s[r].y, k4.y * u);
                s[r].z = fmaf(a, s[r].z, k4.z * u);
                s[r].w = fmaf(a, s[r].w, k4.w * u);
            }
            if (l8 == 0) sot[cur][j * CPB + lcol] = o;
        }
        if (tile + 1 < ntiles) store_tile(cur ^ 1);
        __syncthreads();
        // coalesced store of the tile's outputs: tokens t0 .. t0 + nt - 1, this block's CPB columns each
        for (int i = tid; i < GS_TT * CPB; i += NT) {
            const int j = i / CPB, c = i % CPB;
            if (j < nt) so[((size_t)(t0 + j) * n_v + h) * GDN_D + c0 + c] = sot[cur][j * CPB + c];
        }
    }
#pragma unroll
    for (int r = 0; r < 4; r++) s_ptr[8 * r + l8] = s[r];
    // the new convolution window: input columns T-(conv_k-1) .. T-1, from this chunk or (for a chunk shorter than the window) the old window
    if (blockIdx.y == 0) {
        const int kh = tiled ? (h % n_k) : (h / (n_v / n_k));
        const int key_dim = n_k * GDN_D;
        for (int tc = tid; tc < 3 * GDN_D; tc += NT) {
            const int part = tc / GDN_D, i = tc % GDN_D;
            const int ch = part == 0 ? kh * GDN_D + i : (part == 1 ? key_dim + kh * GDN_D + i : 2 * key_dim + h * GDN_D + i);
            float* st = conv_state + (size_t)h * (conv_k - 1) * 384 + tc;
            for (int j = 0; j < conv_k - 1; j++) {   // ascending: the old entry read for column j sits at index T + j > j, which has not been overwritten yet
                const int sidx = T - (conv_k - 1) + j;
                st[j * 384] = sidx >= 0 ? mixed[(size_t)sidx * conv_dim + ch] : st[(conv_k - 1 + sidx) * 384];
            }
        }
    }
}

#define GDN_SCAN_ARGS const float* __restrict__ pq, const float* __restrict__ pk, const float* __restrict__ pv, const float* __restrict__ pg, const float* __restrict__ mixed, \
                      float* __restrict__ conv_state, float* __restrict__ S, float* __restrict__ so, int T, int n_k, int n_v, int tiled, int conv_k, int conv_dim
extern "C" __global__ void __launch_bounds__(256) gdn_scan_kernel_32(GDN_SCAN_ARGS) { gdn_scan_impl<32>(pq, pk, pv, pg, mixed, conv_state, S, so, T, n_k, n_v, tiled, conv_k, conv_dim); }
extern "C" __global__ void __launch_bounds__(512) gdn_scan_kernel_64(GDN_SCAN_ARGS) { gdn_scan_impl<64>(pq, pk, pv, pg, mixed, conv_state, S, so, T, n_k, n_v, tiled, conv_k, conv_dim); }
extern "C" __global__ void __launch_bounds__(1024) gdn_scan_kernel_128(GDN_SCAN_ARGS) { gdn_scan_impl<128>(pq, pk, pv, pg, mixed, conv_state, S, so, T, n_k, n_v, tiled, conv_k, conv_dim); }

// grid (T, n_v), block 128: gated RMSNorm of the head's output (rms(o) * w * silu(z)), fp16 for the output projection GEMM
extern "C" __global__ void __launch_bounds__(128) gdn_post_kernel(const float* __restrict__ so, const float* __restrict__ zv, const float* __restrict__ norm_w, half* __restrict__ yh, int n_v, float eps) {
    __shared__ float red[32];
    const int t = blockIdx.x, h = blockIdx.y, i = threadIdx.x;
    const size_t idx = ((size_t)t * n_v + h) * GDN_D + i;
    const float v = so[idx];
    const float ss = block_sum(v * v, red);
    const float r = rsqrtf(ss * (1.f / GDN_D) + eps);
    yh[idx] = __float2half(v * r * norm_w[i] * h_silu(zv[idx]));
}

// qkv: fp16 [T][q_rows + 2 kv_dim] with q_rows = n_head * 2 * hd (per head: hd query values, then hd gate values), then k, then v. Writes the normalised, rotated q to q_out (fp16 [T][q_dim])
// and the rotated k and the v of every token into the layer's KV cache at pos0 + t. One block per (token, head); heads: query heads, key heads, value heads. The gates stay in qkv.
extern "C" __global__ void __launch_bounds__(256) qkv35_rope_batch_kernel(const half* __restrict__ qkv, half* __restrict__ q_out, half* __restrict__ k_cache, half* __restrict__ v_cache,
                                                                         const float* __restrict__ cos_t, const float* __restrict__ sin_t, const float* __restrict__ qn_w,
                                                                         const float* __restrict__ kn_w, int pos0, int n_head, int n_kv, int hd, int rd, float eps, int head0) {
    __shared__ float red[32];
    __shared__ float hv[256];
    const int t = blockIdx.x, hb = blockIdx.y + head0;   // head0 > 0 skips the query heads (the MTP head only needs its keys and values)
    const size_t q_rows = (size_t)n_head * 2 * hd, q_dim = (size_t)n_head * hd, kv_dim = (size_t)n_kv * hd, stride = q_rows + 2 * kv_dim;
    const size_t pos = pos0 + t;
    if (hb >= n_head + n_kv) {
        const int head = hb - n_head - n_kv;
        const half* src = qkv + (size_t)t * stride + q_rows + kv_dim + (size_t)head * hd;
        half* dst = v_cache + pos * kv_dim + (size_t)head * hd;
        for (int d = threadIdx.x; d < hd; d += blockDim.x) dst[d] = src[d];
        return;
    }
    const bool is_q = hb < n_head;
    const int head = is_q ? hb : hb - n_head;
    const half* src = is_q ? qkv + (size_t)t * stride + (size_t)head * 2 * hd : qkv + (size_t)t * stride + q_rows + (size_t)head * hd;
    const float* wn = is_q ? qn_w : kn_w;
    float ss = 0.f;
    for (int d = threadIdx.x; d < hd; d += blockDim.x) {
        const float v = __half2float(src[d]);
        hv[d] = v;
        ss += v * v;
    }
    ss = block_sum(ss, red);
    const float sc = rsqrtf(ss / (float)hd + eps);
    __syncthreads();
    half* dst = is_q ? q_out + (size_t)t * q_dim + (size_t)head * hd : k_cache + pos * kv_dim + (size_t)head * hd;
    const int half_r = rd / 2;
    for (int d = threadIdx.x; d < hd; d += blockDim.x) {
        float o;
        if (d < rd) {
            const int i = d < half_r ? d : d - half_r;
            const float a = hv[i] * sc * wn[i], b = hv[i + half_r] * sc * wn[i + half_r];
            const float c = cos_t[pos * half_r + i], s = sin_t[pos * half_r + i];
            o = d < half_r ? a * c - b * s : b * c + a * s;
        } else {
            o = hv[d] * sc * wn[d];
        }
        dst[d] = __float2half(o);
    }
}
