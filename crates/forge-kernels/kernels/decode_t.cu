// Multi-token decode (T = 2 tokens of ONE sequence at consecutive positions): the verification pass of speculative decoding. The weights are streamed once and used for both tokens; every kernel
// here is the T-token counterpart of a stage of decode_q.cu / decode_h.cu. Activations are stored token after token: int8 values [T][K], fp32 scales [T][K/32], integer sums [T][K/32].
// Included after decode_h.cu.

// ==============================================================================================================================================================================================
// Streaming matvec for T activation vectors: the activations of all tokens sit in shared memory, every warp streams its rows in chunks with the next chunk (or the next row) already in flight
// (the structure of mv_res_big_impl) and accumulates one dot product per token.
//   EPI 0: out[t * ldo + r] = dot            (projections, logits)
//   EPI 1: out[t * ldo + r] += dot           (residual stream)
// ==============================================================================================================================================================================================
template <int F, int T, int EPI>
__device__ __forceinline__ void mv_big_t_impl(QP w, const int8_t* __restrict__ xq_g, const float* __restrict__ xd_g, const int* __restrict__ xs_g, float* __restrict__ out, int ldo, int rows, int K) {
    constexpr int C = BigChunk<F>::C;
    constexpr int NIC = F == FMT_Q8_0 ? C : (F == FMT_Q6_K ? C * 4 : C * 2);
    extern __shared__ __align__(16) unsigned char smem_raw[];
    int8_t* sxq = reinterpret_cast<int8_t*>(smem_raw);
    float* sxd = reinterpret_cast<float*>(sxq + T * K);
    int* sxs = reinterpret_cast<int*>(sxd + T * (K / 32));
    // the 16-byte chunks of every token's vector are stored swizzled (swz): conflict-free reads by the lanes of a warp
    for (int i = threadIdx.x; i < T * K / 16; i += blockDim.x) reinterpret_cast<uint4*>(sxq)[(i / (K / 16)) * (K / 16) + swz(i % (K / 16))] = __ldg(reinterpret_cast<const uint4*>(xq_g) + i);
    for (int i = threadIdx.x; i < T * (K / 32); i += blockDim.x) {
        sxd[i] = __ldg(xd_g + i);
        sxs[i] = __ldg(xs_g + i);
    }
    __syncthreads();
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
    const int stride = gridDim.x * nw;
    const int nt = tasks_per_row<F>(K);
    const int nchunks = (nt + 32 * C - 1) / (32 * C);
    int r = blockIdx.x * nw + warp;
    if (r >= rows) return;
    RowRegsF<NIC, F> cur, nxt;
    load_row<NIC, F>(cur, F, w, r, K, lane);
    int j = 0;
    float acc[T];
#pragma unroll
    for (int t = 0; t < T; t++) acc[t] = 0.f;
    while (true) {
        int jn = j + 1, rn = r;
        if (jn == nchunks) {
            jn = 0;
            rn = r + stride;
        }
        const bool more = rn < rows;
        if (more) load_row<NIC, F>(nxt, F, w, rn, K, lane + 32 * C * jn);
#pragma unroll
        for (int t = 0; t < T; t++) {
            const ActP sa{sxq + t * K, sxd + t * (K / 32), sxs + t * (K / 32)};
            acc[t] += dot_part<NIC, F, true>(cur, sa, K, lane + 32 * C * j);
        }
        if (jn == 0) {
#pragma unroll
            for (int t = 0; t < T; t++) {
                const float total = warp_sum(acc[t]);
                if (lane == 0) {
                    if (EPI == 0) out[(size_t)t * ldo + r] = total;
                    else out[(size_t)t * ldo + r] += total;
                }
                acc[t] = 0.f;
            }
        }
        if (!more) break;
        cur = nxt;
        r = rn;
        j = jn;
    }
}

#define INST_MVT(F, SUF, T)                                                                                                                                                                  \
    extern "C" __global__ void mvt##T##_out_##SUF##_kernel(QPARAMS(w), const int8_t* xq, const float* xd, const int* xs, float* out, int ldo, int rows, int K) {                              \
        mv_big_t_impl<F, T, 0>(QMAKE(w), xq, xd, xs, out, ldo, rows, K);                                                                                                                     \
    }                                                                                                                                                                                        \
    extern "C" __global__ void mvt##T##_res_##SUF##_kernel(QPARAMS(w), const int8_t* xq, const float* xd, const int* xs, float* out, int ldo, int rows, int K) {                              \
        mv_big_t_impl<F, T, 1>(QMAKE(w), xq, xd, xs, out, ldo, rows, K);                                                                                                                     \
    }
INST_MVT(0, q8, 1) INST_MVT(1, q40, 1) INST_MVT(2, q50, 1) INST_MVT(3, q4k, 1) INST_MVT(4, q5k, 1) INST_MVT(5, q6k, 1)
INST_MVT(0, q8, 2) INST_MVT(1, q40, 2) INST_MVT(2, q50, 2) INST_MVT(3, q4k, 2) INST_MVT(4, q5k, 2) INST_MVT(5, q6k, 2)

// ==============================================================================================================================================================================================
// The other stages of a two-token step
// ==============================================================================================================================================================================================

// x[t] = dequantized embedding row of tokens[t]; grid (blocks, T)
extern "C" __global__ void embed_q_multi_kernel(QPARAMS(w), int fmt, const uint32_t* __restrict__ tokens, float* __restrict__ x, int E) {
    const int t = blockIdx.y;
    const size_t row = tokens[t];
    const QP p = QMAKE(w);
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < E; i += gridDim.x * blockDim.x) x[(size_t)t * E + i] = deq_elem(p, fmt, row, E, i);
}

// the Gated DeltaNet step for two tokens (decode_h.cu): the state after the first token goes to S_side / conv_side
extern "C" __global__ void __launch_bounds__(1024) gdn_step2_kernel(const float* __restrict__ mixed, const float* __restrict__ zv, const float* __restrict__ ab, const float* __restrict__ conv_w,
                                                                   const float* __restrict__ a_neg, const float* __restrict__ dt_bias, const float* __restrict__ norm_w,
                                                                   float* __restrict__ conv_state, float* __restrict__ S, float* __restrict__ conv_side, float* __restrict__ S_side,
                                                                   int8_t* __restrict__ oq, float* __restrict__ od, int* __restrict__ os, int n_k, int n_v, int tiled, int conv_k, float eps) {
    gdn_step_impl<2>(mixed, zv, ab, conv_w, a_neg, dt_bias, norm_w, conv_state, S, conv_side, S_side, oq, od, os, n_k, n_v, tiled, conv_k, eps);
}

// act[t] = silu(g[t]) * u[t], quantized per 32 values; g, u: fp32 [T][F]; grid (ceil(F / 32 / 8), T), 256 threads (one warp per group of 32)
extern "C" __global__ void swiglu_quant_t_kernel(const float* __restrict__ g, const float* __restrict__ u, int8_t* __restrict__ oq, float* __restrict__ od, int* __restrict__ os, int F) {
    const int t = blockIdx.y, lane = threadIdx.x & 31;
    const int group = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (group * 32 >= F) return;
    const int i = group * 32 + lane;
    const float gv = g[(size_t)t * F + i];
    const float v = (gv / (1.f + __expf(-gv))) * u[(size_t)t * F + i];
    quant_group(v, lane, oq + (size_t)t * F, od + (size_t)t * (F / 32), os + (size_t)t * (F / 32), i, group);
}

// out[t] = index of the largest of logits[t][0 .. V) (the lowest index among equal values); grid T, 256 threads
extern "C" __global__ void argmax_rows_kernel(const float* __restrict__ logits, int V, uint32_t* __restrict__ out) {
    __shared__ float rv[32];
    __shared__ int ri[32];
    const float* l = logits + (size_t)blockIdx.x * V;
    float best = -INFINITY;
    int bi = 0;
    for (int i = threadIdx.x; i < V; i += blockDim.x) {
        const float x = l[i];
        if (x > best) {
            best = x;
            bi = i;
        }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, best, o);
        const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
        if (ov > best || (ov == best && oi < bi)) {
            best = ov;
            bi = oi;
        }
    }
    const int w = threadIdx.x >> 5, lane = threadIdx.x & 31, nw = blockDim.x >> 5;
    if (lane == 0) {
        rv[w] = best;
        ri[w] = bi;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float b = rv[0];
        int i0 = ri[0];
        for (int k = 1; k < nw; k++)
            if (rv[k] > b || (rv[k] == b && ri[k] < i0)) {
                b = rv[k];
                i0 = ri[k];
            }
        out[blockIdx.x] = (uint32_t)i0;
    }
}

// ==============================================================================================================================================================================================
// The MTP head (Qwen3.5 multi-token prediction): input of its fc projection
//   cat = [ rms(embedding(next token)) * norm_e  |  rms( rms(x) * out_norm ) * norm_h ]       x = the main model's residual stream (before its final norm) at the last processed position
// ==============================================================================================================================================================================================
// One vector, quantized (the matvec input): one block of 512 threads, E <= 5120 (the 27B; a wider model would need dynamic shared memory). The token id is read from device memory (tok[0]).
extern "C" __global__ void __launch_bounds__(512) mtp_prep_kernel(QPARAMS(w), int fmt, const uint32_t* __restrict__ tok, const float* __restrict__ x, const float* __restrict__ out_norm,
                                                                 const float* __restrict__ norm_e, const float* __restrict__ norm_h, int8_t* __restrict__ xq, float* __restrict__ xd,
                                                                 int* __restrict__ xs, int E, float eps) {
    __shared__ float red[32];
    __shared__ float hs[5120];
    __shared__ float es[5120];
    const QP p = QMAKE(w);
    const size_t row = tok[0];
    float s1 = 0.f;
    for (int i = threadIdx.x; i < E; i += blockDim.x) s1 += x[i] * x[i];
    s1 = block_sum(s1, red);
    const float r1 = rsqrtf(s1 / (float)E + eps);
    float s2 = 0.f, s3 = 0.f;
    for (int i = threadIdx.x; i < E; i += blockDim.x) {
        const float hn = x[i] * r1 * out_norm[i];
        hs[i] = hn;
        s2 += hn * hn;
        const float ev = deq_elem(p, fmt, row, E, i);
        es[i] = ev;
        s3 += ev * ev;
    }
    s2 = block_sum(s2, red);
    s3 = block_sum(s3, red);
    const float r2 = rsqrtf(s2 / (float)E + eps), r3 = rsqrtf(s3 / (float)E + eps);
    const int lane = threadIdx.x & 31;
    for (int idx = threadIdx.x; idx < 2 * E; idx += blockDim.x) {
        const float v = idx < E ? es[idx] * r3 * norm_e[idx] : hs[idx - E] * r2 * norm_h[idx - E];
        quant_group(v, lane, xq, xd, xs, idx, idx >> 5);
    }
}

// The same for a chunk of positions (prefill), fp16 output for the tensor-core GEMM: cat[t] from the residual stream row t of x (fp32 [T][E]) and the token next_tok[t]. Grid T, 256 threads.
extern "C" __global__ void __launch_bounds__(256) mtp_prep_batch_kernel(QPARAMS(w), int fmt, const uint32_t* __restrict__ next_tok, const float* __restrict__ x, const float* __restrict__ out_norm,
                                                                       const float* __restrict__ norm_e, const float* __restrict__ norm_h, half* __restrict__ cat, int E, float eps) {
    __shared__ float red[32];
    const QP p = QMAKE(w);
    const int t = blockIdx.x;
    const float* xr = x + (size_t)t * E;
    const size_t row = next_tok[t];
    float s1 = 0.f;
    for (int i = threadIdx.x; i < E; i += blockDim.x) s1 += xr[i] * xr[i];
    s1 = block_sum(s1, red);
    const float r1 = rsqrtf(s1 / (float)E + eps);
    float s2 = 0.f, s3 = 0.f;
    for (int i = threadIdx.x; i < E; i += blockDim.x) {
        const float hn = xr[i] * r1 * out_norm[i];
        s2 += hn * hn;
        const float ev = deq_elem(p, fmt, row, E, i);
        s3 += ev * ev;
    }
    s2 = block_sum(s2, red);
    s3 = block_sum(s3, red);
    const float r2 = rsqrtf(s2 / (float)E + eps), r3 = rsqrtf(s3 / (float)E + eps);
    half* out = cat + (size_t)t * 2 * E;
    for (int i = threadIdx.x; i < E; i += blockDim.x) {
        out[i] = __float2half(deq_elem(p, fmt, row, E, i) * r3 * norm_e[i]);
        out[E + i] = __float2half(xr[i] * r1 * out_norm[i] * r2 * norm_h[i]);
    }
}

// Two-pass arg-max of several rows (the draft head and the verification step read 248k logits per row): pass 1 grid (nb, rows), every block scans a contiguous chunk of its row and writes its
// (value, index) maximum (lowest index wins ties); pass 2 grid rows, merges the nb partial maxima of a row.
extern "C" __global__ void argmax_rows_partial_kernel(const float* __restrict__ logits, int V, float* __restrict__ pv, int* __restrict__ pi) {
    __shared__ float sv[32];
    __shared__ int si[32];
    const float* l = logits + (size_t)blockIdx.y * V;
    const int nb = gridDim.x;
    const int chunk = (V + nb - 1) / nb;
    const int lo = blockIdx.x * chunk, hi = min(V, lo + chunk);
    float bv = -INFINITY;
    int bi = 0x7fffffff;
    for (int i = lo + threadIdx.x; i < hi; i += blockDim.x) {
        const float v = l[i];
        if (v > bv || (v == bv && i < bi)) {
            bv = v;
            bi = i;
        }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
        const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
        if (ov > bv || (ov == bv && oi < bi)) {
            bv = ov;
            bi = oi;
        }
    }
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) {
        sv[warp] = bv;
        si[warp] = bi;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float v = sv[0];
        int i = si[0];
        for (int w = 1; w < (blockDim.x >> 5); w++)
            if (sv[w] > v || (sv[w] == v && si[w] < i)) {
                v = sv[w];
                i = si[w];
            }
        pv[(size_t)blockIdx.y * nb + blockIdx.x] = v;
        pi[(size_t)blockIdx.y * nb + blockIdx.x] = i;
    }
}

extern "C" __global__ void argmax_rows_final_kernel(const float* __restrict__ pv, const int* __restrict__ pi, int nb, uint32_t* __restrict__ out) {
    if (threadIdx.x != 0) return;
    const float* v = pv + (size_t)blockIdx.x * nb;
    const int* ix = pi + (size_t)blockIdx.x * nb;
    float bv = v[0];
    int bi = ix[0];
    for (int p = 1; p < nb; p++)
        if (v[p] > bv || (v[p] == bv && ix[p] < bi)) {
            bv = v[p];
            bi = ix[p];
        }
    out[blockIdx.x] = (uint32_t)bi;
}
