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

// The same arithmetic as gdn_prep_kernel for GDN_PT consecutive tokens per block: grid (ceil(T / GDN_PT), n_v), block 384. The partial sums of the L2 norms (one warp_sum per 32 channels and token) and of
// k . q are exchanged through shared memory for the whole tile, so a tile needs three barriers where the one-token kernel needs four per token (bit-identical results; 535 -> 427 us for 1,024 tokens of 32
// heads on the RTX 4060, which is the memory-traffic bound of the three fp32 outputs: 84 MB).
#define GDN_PT 8
extern "C" __global__ void __launch_bounds__(384) gdn_prep_tile_kernel(const float* __restrict__ mixed, const float* __restrict__ ab, const float* __restrict__ conv_w, const float* __restrict__ conv_state,
                                                                      const float* __restrict__ a_neg, const float* __restrict__ dt_bias, float* __restrict__ pq, float* __restrict__ pk,
                                                                      float* __restrict__ pv, float* __restrict__ pg, int T, int n_k, int n_v, int tiled, int conv_k, int conv_dim) {
    __shared__ float sred[GDN_PT][8], sred2[GDN_PT][4];
    __shared__ __align__(16) float skn[GDN_PT][GDN_D];
    const int t0 = blockIdx.x * GDN_PT, h = blockIdx.y, tid = threadIdx.x;
    const int kh = tiled ? (h % n_k) : (h / (n_v / n_k));
    const int key_dim = n_k * GDN_D;
    const int part = tid / GDN_D, i = tid % GDN_D;
    const int w8 = tid >> 5, lane = tid & 31;
    const int ch = part == 0 ? kh * GDN_D + i : (part == 1 ? key_dim + kh * GDN_D + i : 2 * key_dim + h * GDN_D + i);
    const float* w = conv_w + (size_t)ch * conv_k;
    float cval[GDN_PT];
#pragma unroll
    for (int j = 0; j < GDN_PT; j++) {
        const int t = t0 + j;
        float acc = 0.f;
        if (t < T) {
            for (int jj = 0; jj < conv_k; jj++) {
                const int sidx = t - (conv_k - 1) + jj;   // input column; negative = the carried window
                const float x = sidx >= 0 ? mixed[(size_t)sidx * conv_dim + ch] : conv_state[((size_t)h * (conv_k - 1) + (conv_k - 1 + sidx)) * 384 + tid];
                acc += w[jj] * x;
            }
        }
        cval[j] = h_silu(acc);
        if (part == 2 && t < T) pv[((size_t)t * n_v + h) * GDN_D + i] = cval[j];
    }
    if (w8 < 8) {
#pragma unroll
        for (int j = 0; j < GDN_PT; j++) {
            const float s2 = warp_sum(cval[j] * cval[j]);
            if (lane == 0) sred[j][w8] = s2;
        }
    }
    __syncthreads();
    float qv[GDN_PT];
#pragma unroll
    for (int j = 0; j < GDN_PT; j++) {
        const int t = t0 + j;
        qv[j] = 0.f;
        if (part == 0) {
            const float rq = rsqrtf(sred[j][0] + sred[j][1] + sred[j][2] + sred[j][3] + 1e-6f) * 0.08838834764831845f;
            qv[j] = cval[j] * rq;
            if (t < T) pq[((size_t)t * n_v + h) * GDN_D + i] = qv[j];
        } else if (part == 1) {
            const float rk = rsqrtf(sred[j][4] + sred[j][5] + sred[j][6] + sred[j][7] + 1e-6f);
            const float kv = cval[j] * rk;
            if (t < T) pk[((size_t)t * n_v + h) * GDN_D + i] = kv;
            skn[j][i] = kv;
        }
    }
    __syncthreads();
    if (w8 < 4) {
#pragma unroll
        for (int j = 0; j < GDN_PT; j++) {
            const float s2 = warp_sum(skn[j][w8 * 32 + lane] * qv[j]);
            if (lane == 0) sred2[j][w8] = s2;
        }
    }
    __syncthreads();
    if (tid < GDN_PT && t0 + tid < T) {
        const int t = t0 + tid;
        const float kq = sred2[tid][0] + sred2[tid][1] + sred2[tid][2] + sred2[tid][3];
        const float* abt = ab + (size_t)t * 2 * n_v;
        const float beta = h_sigmoid(abt[n_v + h]);
        const float a = __expf(a_neg[h] * h_softplus(abt[h] + dt_bias[h]));
        float* g = pg + ((size_t)t * n_v + h) * 3;
        g[0] = beta;
        g[1] = a;
        g[2] = kq;
    }
}

// grid (n_v, 128 / CPB), block CPB * 8 / NC: the thread mapping of gdn_step_kernel for CPB of the head's 128 state columns (the columns of the recurrence are independent given k, q and the gates, so a head
// can be spread over several blocks and SMs); a column is 8 lanes of 16 state rows each, and with NC > 1 a thread owns NC consecutive columns: the k / q vectors a warp reads from shared memory (the
// LDS wavefronts were the bottleneck: 12 per warp and token for 4 columns) then serve 4 * NC columns, and the lanes that hold a column's rows still reduce in the same order, so the result is bit-identical
// for every NC. S stays in registers over the whole chunk; so[t][h][col] receives the raw output of the recurrence.
// The per-token inputs (k, q, v, gates) are staged through shared memory in tiles of GS_TT tokens, the NEXT tile's global loads are in flight while the current tile is computed: the recurrence
// itself is a chain of a few hundred cycles per token, the exposed DRAM latency of a one-token prefetch was ten times that. The outputs of a tile leave as one coalesced store.
// The block with blockIdx.y == 0 finally replaces the head's convolution window by the last conv_k - 1 input columns of the chunk.
#define GS_TT 8
template <int CPB, int NC = 1>
__device__ __forceinline__ void gdn_scan_impl(const float* __restrict__ pq, const float* __restrict__ pk, const float* __restrict__ pv, const float* __restrict__ pg, const float* __restrict__ mixed,
                                              float* __restrict__ conv_state, float* __restrict__ S, float* __restrict__ so, int T, int n_k, int n_v, int tiled, int conv_k, int conv_dim) {
    constexpr int NT = CPB * 8 / NC;                  // threads
    constexpr int NV4 = GS_TT * CPB / 4;              // float4 of v per tile
    constexpr int TOTAL = 2 * GS_TT * 32 + NV4;       // float4 per tile: k, q, v
    constexpr int ITEMS = (TOTAL + NT - 1) / NT;
    __shared__ __align__(16) float sk[2][GS_TT * GDN_D];
    __shared__ __align__(16) float sq[2][GS_TT * GDN_D];
    __shared__ __align__(16) float sv[2][GS_TT * CPB];
    __shared__ __align__(16) float sg[2][GS_TT * 4];
    __shared__ __align__(16) float sot[2][GS_TT * CPB];
    const int h = blockIdx.x, c0 = blockIdx.y * CPB, tid = threadIdx.x;
    const int l8 = tid & 7;                           // lane within the column
    const int lcol0 = (tid >> 3) * NC;                // first of this thread's NC columns (within the block)
    float4 s[NC][4];
#pragma unroll
    for (int c = 0; c < NC; c++) {
        const float4* s_ptr = reinterpret_cast<const float4*>(S + ((size_t)h * GDN_D + c0 + lcol0 + c) * GDN_D);
#pragma unroll
        for (int r = 0; r < 4; r++) s[c][r] = s_ptr[8 * r + l8];
    }

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
            float4 k4[4], q4[4];
#pragma unroll
            for (int r = 0; r < 4; r++) {
                k4[r] = kj[8 * r + l8];
                q4[r] = qj[8 * r + l8];
            }
            float ks[NC], qs[NC];
#pragma unroll
            for (int c = 0; c < NC; c++) {
                float kk = 0.f, qq = 0.f;
#pragma unroll
                for (int r = 0; r < 4; r++) {
                    kk = fmaf(s[c][r].x, k4[r].x, kk);
                    kk = fmaf(s[c][r].y, k4[r].y, kk);
                    kk = fmaf(s[c][r].z, k4[r].z, kk);
                    kk = fmaf(s[c][r].w, k4[r].w, kk);
                    qq = fmaf(s[c][r].x, q4[r].x, qq);
                    qq = fmaf(s[c][r].y, q4[r].y, qq);
                    qq = fmaf(s[c][r].z, q4[r].z, qq);
                    qq = fmaf(s[c][r].w, q4[r].w, qq);
                }
                ks[c] = kk;
                qs[c] = qq;
            }
#pragma unroll
            for (int m = 1; m < 8; m <<= 1) {
#pragma unroll
                for (int c = 0; c < NC; c++) {
                    ks[c] += __shfl_xor_sync(0xffffffffu, ks[c], m);
                    qs[c] += __shfl_xor_sync(0xffffffffu, qs[c], m);
                }
            }
#pragma unroll
            for (int c = 0; c < NC; c++) {
                const float u = beta * (sv[cur][j * CPB + lcol0 + c] - a * ks[c]);
                const float o = a * qs[c] + kq * u;
#pragma unroll
                for (int r = 0; r < 4; r++) {
                    s[c][r].x = fmaf(a, s[c][r].x, k4[r].x * u);
                    s[c][r].y = fmaf(a, s[c][r].y, k4[r].y * u);
                    s[c][r].z = fmaf(a, s[c][r].z, k4[r].z * u);
                    s[c][r].w = fmaf(a, s[c][r].w, k4[r].w * u);
                }
                if (l8 == 0) sot[cur][j * CPB + lcol0 + c] = o;
            }
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
    for (int c = 0; c < NC; c++) {
        float4* s_ptr = reinterpret_cast<float4*>(S + ((size_t)h * GDN_D + c0 + lcol0 + c) * GDN_D);
#pragma unroll
        for (int r = 0; r < 4; r++) s_ptr[8 * r + l8] = s[c][r];
    }
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
// blocks of 128 threads whatever CPB: 16 / 32 / 64 columns with 1 / 2 / 4 columns per thread (fewer, fatter blocks: 2.7 blocks per SM of the RTX 4060 is the fastest, 1.75x the 256-thread kernel)
extern "C" __global__ void __launch_bounds__(128) gdn_scan_nc_16(GDN_SCAN_ARGS) { gdn_scan_impl<16, 1>(pq, pk, pv, pg, mixed, conv_state, S, so, T, n_k, n_v, tiled, conv_k, conv_dim); }
extern "C" __global__ void __launch_bounds__(128) gdn_scan_nc_32(GDN_SCAN_ARGS) { gdn_scan_impl<32, 2>(pq, pk, pv, pg, mixed, conv_state, S, so, T, n_k, n_v, tiled, conv_k, conv_dim); }
extern "C" __global__ void __launch_bounds__(128) gdn_scan_nc_64(GDN_SCAN_ARGS) { gdn_scan_impl<64, 4>(pq, pk, pv, pg, mixed, conv_state, S, so, T, n_k, n_v, tiled, conv_k, conv_dim); }

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
