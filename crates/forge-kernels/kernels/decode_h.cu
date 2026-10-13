// ForgeInfer decode kernels for hybrid models (Qwen3.5): the Gated DeltaNet step and the glue of the gated full-attention layers. Included after decode.cu and decode_q.cu (helpers: warp_sum,
// block_sum, quant_group, StepState).
//
// A Gated DeltaNet layer keeps a recurrent state S[n_v][128 value columns][128 key rows] (fp32, column-contiguous) and, per value head, the last 3 columns of the convolution input. One token:
//     x -norm_quant-> y = [W_qkv x | W_z x | W_a x | W_b x]  (generic quantized matvecs, fp32 results)
//     gdn_step (one block of 1024 threads per value head): causal conv + SiLU of the head's q / k / v channels, L2 norms, gates, the delta rule on S, gated RMSNorm, quantized output
//     x += W_out y                                                                                                                  (mvq_res)
// The delta rule reads the UNDECAYED state once for both dot products: with ks = S^T k, qs = S^T q (one pass), a = exp(g):
//     u = beta (v - a ks)     o = a qs + (k.q) u     S <- a S + k (x) u        which equals  S <- a S; S <- S + beta k (x) (v - S^T k); o = S^T q
// (verified against the recurrence in the CPU reference: crates/forge-model/src/cpu.rs, delta_step).

#define GDN_D 128

__device__ __forceinline__ float h_sigmoid(float x) { return 1.f / (1.f + __expf(-x)); }
__device__ __forceinline__ float h_silu(float x) { return x / (1.f + __expf(-x)); }
__device__ __forceinline__ float h_softplus(float x) { return x > 20.f ? x : logf(1.f + __expf(x)); }

// grid = n_v (value heads), block = 1024 threads: thread t owns column t / 8 of the head's state, 16 of its 128 rows (lane8 = t % 8 holds rows 32 r + 4 lane8 + e, r < 4, e < 4).
// conv_state: [n_v][conv_k - 1][384] (q | k | v channels of THIS head, oldest column first): per-head windows, so blocks that share a key head never race on it.
// T tokens of one sequence in a row (T = 2: the verification pass of speculative decoding; T = 4 .. 16: the multi-row prompt path, SIDE 0): the state stays in registers between them. With SIDE 1
// (the default for T > 1) the state and the convolution window after the FIRST token are also written to S_side / conv_side (same layouts): if the second token is rejected, copying them back restores
// the state at the first token's position. With SIDE 2 (the multi-row verification) the state after EVERY token but the last is saved: after token t to S_side + t * s_side_stride (conv_side +
// t * c_side_stride, strides in floats), so that the sequence can return to the position after any of its rows.
// Per-token inputs and outputs are stored token after token: mixed [T][conv_dim], zv [T][value_dim], ab [T][2 n_v], oq [T][value_dim] with od / os per 32 values.
template <int T, int SIDE = (T > 1 ? 1 : 0)>
__device__ __forceinline__ void gdn_step_impl(const float* __restrict__ mixed, const float* __restrict__ zv, const float* __restrict__ ab, const float* __restrict__ conv_w,
                                              const float* __restrict__ a_neg, const float* __restrict__ dt_bias, const float* __restrict__ norm_w, float* __restrict__ conv_state,
                                              float* __restrict__ S, float* __restrict__ conv_side, float* __restrict__ S_side, int8_t* __restrict__ oq, float* __restrict__ od,
                                              int* __restrict__ os, int n_k, int n_v, int tiled, int conv_k, float eps, int ntok = T, size_t s_side_stride = 0, size_t c_side_stride = 0) {
    __shared__ float qn[GDN_D], kn[GDN_D], vv[GDN_D], osm[GDN_D], red[16];
    const int h = blockIdx.x;
    const int tid = threadIdx.x;
    const int col = tid >> 3, l8 = tid & 7;
    const int kh = tiled ? (h % n_k) : (h / (n_v / n_k));
    const int key_dim = n_k * GDN_D;
    const int value_dim = n_v * GDN_D;
    const int conv_dim = 2 * key_dim + value_dim;

    // the state loads do not depend on anything below: issue them first
    float4* s_ptr = reinterpret_cast<float4*>(S + ((size_t)h * GDN_D + col) * GDN_D);
    float4 s[4];
#pragma unroll
    for (int r = 0; r < 4; r++) s[r] = s_ptr[8 * r + l8];
    const int w8 = tid >> 5, lane = tid & 31;

#pragma unroll (T <= 2 ? T : 1)
    for (int t = 0; t < ntok; t++) {
        const float* mixed_t = mixed + (size_t)t * conv_dim;
        const float* z_t = zv + (size_t)t * value_dim;
        const float* ab_t = ab + (size_t)t * 2 * n_v;
        // causal depthwise convolution (w[conv_k - 1] multiplies the current column) + SiLU over this head's 128 q, 128 k and 128 v channels
        if (tid < 3 * GDN_D) {
            const int part = tid / GDN_D, i = tid % GDN_D;
            const int ch = part == 0 ? kh * GDN_D + i : (part == 1 ? key_dim + kh * GDN_D + i : 2 * key_dim + h * GDN_D + i);
            const float x = mixed_t[ch];
            float* st = conv_state + (size_t)h * (conv_k - 1) * 384 + tid;
            const float* w = conv_w + (size_t)ch * conv_k;
            float prev[3];
#pragma unroll
            for (int j = 0; j < 3; j++) prev[j] = j < conv_k - 1 ? st[j * 384] : 0.f;
            float acc = w[conv_k - 1] * x;
#pragma unroll
            for (int j = 0; j < 3; j++)
                if (j < conv_k - 1) acc += w[j] * prev[j];
#pragma unroll
            for (int j = 0; j < 2; j++)
                if (j < conv_k - 2) st[j * 384] = prev[j + 1];
            st[(conv_k - 2) * 384] = x;
            const float c = h_silu(acc);
            if (part == 0) qn[i] = c;
            else if (part == 1) kn[i] = c;
            else vv[i] = c;
        }
        __syncthreads();
        // L2 normalisation (epsilon inside the root); q is also scaled by 1/sqrt(128)
        if (w8 < 8) {
            const float v = (w8 < 4 ? qn : kn)[(w8 & 3) * 32 + lane];
            const float t2 = warp_sum(v * v);
            if (lane == 0) red[w8] = t2;
        }
        __syncthreads();
        const float rq = rsqrtf(red[0] + red[1] + red[2] + red[3] + 1e-6f) * 0.08838834764831845f;
        const float rk = rsqrtf(red[4] + red[5] + red[6] + red[7] + 1e-6f);
        if (tid < GDN_D) qn[tid] *= rq;
        else if (tid < 2 * GDN_D) kn[tid - GDN_D] *= rk;
        __syncthreads();
        if (w8 < 4) {
            const float t2 = warp_sum(kn[w8 * 32 + lane] * qn[w8 * 32 + lane]);
            if (lane == 0) red[8 + w8] = t2;
        }
        __syncthreads();
        const float kq = red[8] + red[9] + red[10] + red[11];
        // gates: beta = sigmoid(b), decay a = exp(g), g = -exp(A_log) * softplus(alpha + dt_bias) (a_neg holds -exp(A_log))
        const float beta = h_sigmoid(ab_t[n_v + h]);
        const float a = __expf(a_neg[h] * h_softplus(ab_t[h] + dt_bias[h]));

        // the delta rule on this thread's 16 rows of column `col`
        float4 k4[4], q4[4];
#pragma unroll
        for (int r = 0; r < 4; r++) {
            k4[r] = *reinterpret_cast<const float4*>(&kn[32 * r + 4 * l8]);
            q4[r] = *reinterpret_cast<const float4*>(&qn[32 * r + 4 * l8]);
        }
        float ks = 0.f, qs = 0.f;
#pragma unroll
        for (int r = 0; r < 4; r++) {
            ks = fmaf(s[r].x, k4[r].x, ks);
            ks = fmaf(s[r].y, k4[r].y, ks);
            ks = fmaf(s[r].z, k4[r].z, ks);
            ks = fmaf(s[r].w, k4[r].w, ks);
            qs = fmaf(s[r].x, q4[r].x, qs);
            qs = fmaf(s[r].y, q4[r].y, qs);
            qs = fmaf(s[r].z, q4[r].z, qs);
            qs = fmaf(s[r].w, q4[r].w, qs);
        }
#pragma unroll
        for (int m = 1; m < 8; m <<= 1) {   // xor 1, 2, 4 stays inside the 8 lanes of one column
            ks += __shfl_xor_sync(0xffffffffu, ks, m);
            qs += __shfl_xor_sync(0xffffffffu, qs, m);
        }
        const float u = beta * (vv[col] - a * ks);
        const float o = a * qs + kq * u;
#pragma unroll
        for (int r = 0; r < 4; r++) {
            s[r].x = fmaf(a, s[r].x, k4[r].x * u);
            s[r].y = fmaf(a, s[r].y, k4[r].y * u);
            s[r].z = fmaf(a, s[r].z, k4[r].z * u);
            s[r].w = fmaf(a, s[r].w, k4[r].w * u);
        }
        if (l8 == 0) osm[col] = o;
        if ((SIDE == 1 && t == 0) || (SIDE == 2 && t + 1 < ntok)) {
            // the state after this token, for a rollback of the ones after it
            float4* sd = reinterpret_cast<float4*>(S_side + (SIDE == 2 ? (size_t)t * s_side_stride : (size_t)0) + ((size_t)h * GDN_D + col) * GDN_D);
#pragma unroll
            for (int r = 0; r < 4; r++) sd[8 * r + l8] = s[r];
            if (tid < 3 * GDN_D) {
                const float* st = conv_state + (size_t)h * (conv_k - 1) * 384 + tid;
                float* cd = conv_side + (SIDE == 2 ? (size_t)t * c_side_stride : (size_t)0) + (size_t)h * (conv_k - 1) * 384 + tid;
                for (int j = 0; j < conv_k - 1; j++) cd[j * 384] = st[j * 384];
            }
        }
        __syncthreads();
        // gated RMSNorm per head, rms(o) * w * silu(z), then the quantized activation that feeds the output projection
        if (w8 < 4) {
            const float v = osm[w8 * 32 + lane];
            const float t2 = warp_sum(v * v);
            if (lane == 0) red[12 + w8] = t2;
        }
        __syncthreads();
        const float rr = rsqrtf((red[12] + red[13] + red[14] + red[15]) * (1.f / GDN_D) + eps);
        if (w8 < 4) {
            const int c = w8 * 32 + lane;
            const float y = osm[c] * rr * norm_w[c] * h_silu(z_t[h * GDN_D + c]);
            quant_group(y, lane, oq + (size_t)t * value_dim, od + (size_t)t * (value_dim / 32), os + (size_t)t * (value_dim / 32), h * GDN_D + c, h * 4 + w8);
        }
        __syncthreads();   // the shared arrays are reused by the next token
    }
#pragma unroll
    for (int r = 0; r < 4; r++) s_ptr[8 * r + l8] = s[r];
}

extern "C" __global__ void __launch_bounds__(1024) gdn_step_kernel(const float* __restrict__ mixed, const float* __restrict__ zv, const float* __restrict__ ab, const float* __restrict__ conv_w,
                                                                  const float* __restrict__ a_neg, const float* __restrict__ dt_bias, const float* __restrict__ norm_w,
                                                                  float* __restrict__ conv_state, float* __restrict__ S, int8_t* __restrict__ oq, float* __restrict__ od, int* __restrict__ os,
                                                                  int n_k, int n_v, int tiled, int conv_k, float eps) {
    gdn_step_impl<1>(mixed, zv, ab, conv_w, a_neg, dt_bias, norm_w, conv_state, S, nullptr, nullptr, oq, od, os, n_k, n_v, tiled, conv_k, eps);
}

// Gated full attention, step 1: q (first half of every head's block in `qg`) and k are RMS-normalised per head, rotated on the first `rd` dimensions (pairs (i, i + rd/2)), k and v go to the
// fp16 cache at the current position. One block per q head, then per k head, then per v head. The gate halves of `qg` stay where they are: attention_g_kernel reads them from there.
extern "C" __global__ void __launch_bounds__(256) qkv35_prep_kernel(const float* __restrict__ qg, const float* __restrict__ k_raw, const float* __restrict__ v_raw, const float* __restrict__ qn_w,
                                                                   const float* __restrict__ kn_w, const float* __restrict__ cos_t, const float* __restrict__ sin_t,
                                                                   const StepState* __restrict__ st, float* __restrict__ q_out, half* __restrict__ k_cache, half* __restrict__ v_cache, int n_head,
                                                                   int n_kv, int hd, int rd, float eps) {
    __shared__ float red[32];
    const int b = blockIdx.x;
    const int tt = blockIdx.y;                 // token of a multi-token step: position st->pos + tt, rows tt of qg / k_raw / v_raw / q_out
    const uint32_t pos = st->pos + tt;
    const size_t kv_dim = (size_t)n_kv * hd;
    if (b >= n_head + n_kv) {
        const int head = b - n_head - n_kv;
        for (int d = threadIdx.x; d < hd; d += blockDim.x) v_cache[pos * kv_dim + head * hd + d] = __float2half(v_raw[tt * kv_dim + head * hd + d]);
        return;
    }
    const bool is_q = b < n_head;
    const int head = is_q ? b : b - n_head;
    const float* src = is_q ? qg + (size_t)tt * n_head * 2 * hd + (size_t)head * 2 * hd : k_raw + tt * kv_dim + (size_t)head * hd;
    const float* wn = is_q ? qn_w : kn_w;
    float ss = 0.f;
    for (int d = threadIdx.x; d < hd; d += blockDim.x) ss += src[d] * src[d];
    ss = block_sum(ss, red);
    const float r = rsqrtf(ss / (float)hd + eps);
    const int half_r = rd / 2;
    for (int d = threadIdx.x; d < hd; d += blockDim.x) {
        float o;
        if (d < rd) {
            const int i = d < half_r ? d : d - half_r;
            const float a = src[i] * r * wn[i], bb = src[i + half_r] * r * wn[i + half_r];
            const float c = cos_t[(size_t)pos * half_r + i], s = sin_t[(size_t)pos * half_r + i];
            o = d < half_r ? a * c - bb * s : bb * c + a * s;
        } else {
            o = src[d] * r * wn[d];
        }
        if (is_q) q_out[(size_t)tt * n_head * hd + head * hd + d] = o;
        else k_cache[pos * kv_dim + head * hd + d] = __float2half(o);
    }
}

// ---- gated attention for long contexts: split-KV decode ("flash decoding") --------------------------------------------------------------------------------------------------------------------
// The one-block-per-head kernel keeps all scores in shared memory (positions x 4 bytes: 48 KB = 12k positions). Here every (head, split) block handles AS_CH positions with its own softmax
// statistics and writes (max, sum, unnormalised output); attn_merge_g_kernel combines the splits of a head exactly (rescaling by exp(m_s - M)), applies the output gate and writes the
// quantized activation. head width 256 (LPP = 32 lanes per key row, 8 positions per round), 256 threads.
#define AS_CH 1024
extern "C" __global__ void __launch_bounds__(256) attn_split_g_kernel(const float* __restrict__ q, const half* __restrict__ k_cache, const half* __restrict__ v_cache, float* __restrict__ pm,
                                                                     float* __restrict__ pl, float* __restrict__ po, const StepState* __restrict__ st, int n_head, int n_kv, int hd, int ns_max) {
    __shared__ float scores[AS_CH];
    __shared__ float qs[256];
    __shared__ float part[8 * 256];
    __shared__ float red[32];
    const int hh = blockIdx.x, split = blockIdx.y, tid = threadIdx.x;
    const int tt = blockIdx.z;                 // token of a multi-token step
    const int T = st->pos + 1 + tt;
    const int t0 = split * AS_CH;
    const size_t pidx = ((size_t)tt * n_head + hh) * ns_max + split;
    float* o_out = po + pidx * 256;
    if (t0 >= T) {
        if (tid == 0) {
            pm[pidx] = -INFINITY;
            pl[pidx] = 0.f;
        }
        return;
    }
    const int tn = min(AS_CH, T - t0);
    const int kvh = hh / (n_head / n_kv);
    const size_t kv_dim = (size_t)n_kv * hd;
    const int sub = tid & 31, grp = tid >> 5;   // 32 lanes per key row, 8 rows per round
    qs[tid] = q[(size_t)tt * n_head * hd + hh * hd + tid];
    __syncthreads();
    const float scale = rsqrtf((float)hd);
    float qv[8];
#pragma unroll
    for (int j = 0; j < 8; j++) qv[j] = qs[sub * 8 + j];
    for (int i0 = 0; i0 < tn; i0 += 8) {   // uniform trip count across the block: the shuffles below involve whole warps
        const int i = i0 + grp;
        const bool valid = i < tn;
        uint4 kraw = make_uint4(0u, 0u, 0u, 0u);
        if (valid) kraw = __ldg(reinterpret_cast<const uint4*>(k_cache + (size_t)(t0 + i) * kv_dim + kvh * hd) + sub);
        float kf[8];
        unpack8(kraw, kf);
        float acc = 0.f;
#pragma unroll
        for (int j = 0; j < 8; j++) acc += qv[j] * kf[j];
        acc = warp_sum(acc);
        if (valid && sub == 0) scores[i] = acc * scale;
    }
    __syncthreads();
    float mx = -INFINITY;
    for (int i = tid; i < tn; i += 256) mx = fmaxf(mx, scores[i]);
    mx = block_max(mx, red);
    float sum = 0.f;
    for (int i = tid; i < tn; i += 256) {
        const float e = __expf(scores[i] - mx);
        scores[i] = e;
        sum += e;
    }
    sum = block_sum(sum, red);
    float o[8];
#pragma unroll
    for (int j = 0; j < 8; j++) o[j] = 0.f;
    for (int i = grp; i < tn; i += 8) {
        const uint4 vraw = __ldg(reinterpret_cast<const uint4*>(v_cache + (size_t)(t0 + i) * kv_dim + kvh * hd) + sub);
        float vf[8];
        unpack8(vraw, vf);
        const float p = scores[i];
#pragma unroll
        for (int j = 0; j < 8; j++) o[j] += p * vf[j];
    }
#pragma unroll
    for (int j = 0; j < 8; j++) part[grp * 256 + sub * 8 + j] = o[j];
    __syncthreads();
    float s = 0.f;
#pragma unroll
    for (int g = 0; g < 8; g++) s += part[g * 256 + tid];
    o_out[tid] = s;
    if (tid == 0) {
        pm[pidx] = mx;
        pl[pidx] = sum;
    }
}

// grid n_head, block 256 (one thread per output dimension): exact combination of the splits, the sigmoid gate (gate + hh * gate_stride), int8 quantization per 32 dimensions
extern "C" __global__ void __launch_bounds__(256) attn_merge_g_kernel(const float* __restrict__ pm, const float* __restrict__ pl, const float* __restrict__ po, const float* __restrict__ gate, int8_t* __restrict__ oq,
                                                                     float* __restrict__ od, int* __restrict__ os, const StepState* __restrict__ st, int ns_max, int gate_stride) {
    const int hh = blockIdx.x, tid = threadIdx.x;
    const int tt = blockIdx.y;                 // token of a multi-token step
    const int n_head = gridDim.x;
    const int ns = (st->pos + 1 + tt + AS_CH - 1) / AS_CH;
    const size_t pbase = ((size_t)tt * n_head + hh) * ns_max;
    float M = -INFINITY;
    for (int s = 0; s < ns; s++) M = fmaxf(M, pm[pbase + s]);
    float L = 0.f, acc = 0.f;
    for (int s = 0; s < ns; s++) {
        const float w = __expf(pm[pbase + s] - M);
        L += pl[pbase + s] * w;
        acc += po[(pbase + s) * 256 + tid] * w;
    }
    float v = acc / L;
    v *= 1.f / (1.f + __expf(-gate[((size_t)tt * n_head + hh) * gate_stride + tid]));
    const float amax = warp_max(fabsf(v));
    const float d = amax * (1.f / 127.f);
    const float id = d != 0.f ? 1.f / d : 0.f;
    const int qi = (int)roundf(v * id);
    int sum = qi;
#pragma unroll
    for (int o2 = 16; o2 > 0; o2 >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o2);
    const size_t obase = (size_t)tt * n_head * 256 + hh * 256;
    oq[obase + tid] = (int8_t)qi;
    if ((tid & 31) == 0) {
        od[(obase + tid) / 32] = round_half(d);
        os[(obase + tid) / 32] = sum;
    }
}
