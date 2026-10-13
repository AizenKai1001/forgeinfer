// KV cache compression (FORGE_KV=k8v8 | k8v4 | k4v4 ..., FORGE_KV_GROUP=32|64|128|256): the cache of every attention layer is stored as integers with one fp16 scale per group of `group` values of a
// (position, kv head) row; the attention kernels keep reading an fp16 cache, which is a one-layer SCRATCH. Per layer and step: kv_dequant fills the scratch with the positions before the current ones,
// the QKV kernels write the new K / V rows into the scratch as before, the attention runs unchanged, and kv_quant compresses the new rows into the store.
//
//   store     rows r = position * n_kv + head, hd values per row, `bits` per value: 8 = int8, 4 = two nibbles per byte (value + 8; element 2j in the low half). Scales: fp16 [r][hd / group].
//   kv_dequant   store -> scratch fp16 [T][n_kv * hd]; T = t_fixed, or the position of the step state when t_fixed < 0.      grid (blocks, 2): y = 0: K, y = 1: V
//   kv_quant     scratch rows [pos0, pos0 + n_new) -> the store; pos0 = pos0_fixed, or the step state's position when negative.   grid (blocks, 2)
// One warp per row; a lane owns hd / 32 (2, 4 or 8) consecutive values, a scale group is `group / (hd / 32)` lanes. hd is 64, 128 or 256, group a power of two between 32 and hd.

__device__ __forceinline__ float kv_round_half(float v) { return __half2float(__float2half(v)); }
__device__ __forceinline__ unsigned kv_pack2(float a, float b) {
    half2 h = __floats2half2_rn(a, b);
    return *reinterpret_cast<unsigned*>(&h);
}

extern "C" __global__ void __launch_bounds__(256) kv_dequant_kernel(const uint8_t* __restrict__ qk, const half* __restrict__ sk, half* __restrict__ ok, const uint8_t* __restrict__ qv, const half* __restrict__ sv,
                                                                    half* __restrict__ ov, const StepState* __restrict__ st, int t_fixed, int n_kv, int hd, int group, int bits_k, int bits_v) {
    const bool isv = blockIdx.y != 0;
    const uint8_t* q = isv ? qv : qk;
    const half* sc = isv ? sv : sk;
    half* out = isv ? ov : ok;
    const int bits = isv ? bits_v : bits_k;
    const int lane = threadIdx.x & 31;
    const int epl = hd >> 5;           // values per lane
    const int lpg = group / epl;       // lanes per scale group
    const int ng = hd / group;
    const int T = t_fixed >= 0 ? t_fixed : (int)st->pos;
    const long rows = (long)T * n_kv;
    const long nwarps = ((long)gridDim.x * blockDim.x) >> 5;
    const long rbytes = (long)hd * bits / 8;
    for (long r = (((long)blockIdx.x * blockDim.x + threadIdx.x) >> 5); r < rows; r += nwarps) {
        const float s = __half2float(sc[r * ng + lane / lpg]);
        float f[8];
        unsigned long long raw = 0;
        if (bits == 8) {
            const uint8_t* src = q + r * rbytes + lane * epl;
            raw = epl == 8 ? *reinterpret_cast<const unsigned long long*>(src) : epl == 4 ? (unsigned long long)*reinterpret_cast<const unsigned*>(src) : (unsigned long long)*reinterpret_cast<const unsigned short*>(src);
#pragma unroll
            for (int e = 0; e < 8; ++e) f[e] = e < epl ? (float)(int8_t)((raw >> (8 * e)) & 0xFF) * s : 0.f;
        } else {
            const uint8_t* src = q + r * rbytes + lane * (epl >> 1);
            raw = epl == 8 ? (unsigned long long)*reinterpret_cast<const unsigned*>(src) : epl == 4 ? (unsigned long long)*reinterpret_cast<const unsigned short*>(src) : (unsigned long long)*src;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int b = (int)((raw >> (8 * j)) & 0xFF);
                f[2 * j] = 2 * j < epl ? (float)((b & 15) - 8) * s : 0.f;
                f[2 * j + 1] = 2 * j + 1 < epl ? (float)((b >> 4) - 8) * s : 0.f;
            }
        }
        half* dst = out + r * hd + lane * epl;
        if (epl == 8) {
            uint4 v;
            v.x = kv_pack2(f[0], f[1]);
            v.y = kv_pack2(f[2], f[3]);
            v.z = kv_pack2(f[4], f[5]);
            v.w = kv_pack2(f[6], f[7]);
            *reinterpret_cast<uint4*>(dst) = v;
        } else if (epl == 4) {
            uint2 v;
            v.x = kv_pack2(f[0], f[1]);
            v.y = kv_pack2(f[2], f[3]);
            *reinterpret_cast<uint2*>(dst) = v;
        } else {
            *reinterpret_cast<unsigned*>(dst) = kv_pack2(f[0], f[1]);
        }
    }
}

extern "C" __global__ void __launch_bounds__(256) kv_quant_kernel(const half* __restrict__ ik, uint8_t* __restrict__ qk, half* __restrict__ sk, const half* __restrict__ iv, uint8_t* __restrict__ qv,
                                                                  half* __restrict__ sv, const StepState* __restrict__ st, int pos0_fixed, int n_new, int n_kv, int hd, int group, int bits_k, int bits_v) {
    const bool isv = blockIdx.y != 0;
    const half* in = isv ? iv : ik;
    uint8_t* q = isv ? qv : qk;
    half* sc = isv ? sv : sk;
    const int bits = isv ? bits_v : bits_k;
    const int lane = threadIdx.x & 31;
    const int epl = hd >> 5;
    const int lpg = group / epl;
    const int ng = hd / group;
    const long pos0 = pos0_fixed >= 0 ? pos0_fixed : (long)st->pos;
    const long rows = (long)n_new * n_kv;
    const long nwarps = ((long)gridDim.x * blockDim.x) >> 5;
    const long rbytes = (long)hd * bits / 8;
    const float qmax = bits == 8 ? 127.f : 7.f;
    for (long rr = (((long)blockIdx.x * blockDim.x + threadIdx.x) >> 5); rr < rows; rr += nwarps) {
        const long r = pos0 * n_kv + rr;   // row of the store: position * n_kv + head
        const half* src = in + r * hd + lane * epl;
        float f[8];
        if (epl == 8) {
            const uint4 v = *reinterpret_cast<const uint4*>(src);
            const unsigned w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const half2 h = *reinterpret_cast<const half2*>(&w[j]);
                f[2 * j] = __low2float(h);
                f[2 * j + 1] = __high2float(h);
            }
        } else if (epl == 4) {
            const uint2 v = *reinterpret_cast<const uint2*>(src);
            const unsigned w[2] = {v.x, v.y};
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                const half2 h = *reinterpret_cast<const half2*>(&w[j]);
                f[2 * j] = __low2float(h);
                f[2 * j + 1] = __high2float(h);
            }
#pragma unroll
            for (int e = 4; e < 8; ++e) f[e] = 0.f;
        } else {
            const half2 h = *reinterpret_cast<const half2*>(src);
            f[0] = __low2float(h);
            f[1] = __high2float(h);
#pragma unroll
            for (int e = 2; e < 8; ++e) f[e] = 0.f;
        }
        float amax = 0.f;
#pragma unroll
        for (int e = 0; e < 8; ++e) amax = fmaxf(amax, fabsf(f[e]));
        for (int o = lpg >> 1; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
        const float ds = kv_round_half(amax / qmax);
        const float id = ds != 0.f ? 1.f / ds : 0.f;
        int qi[8];
#pragma unroll
        for (int e = 0; e < 8; ++e) qi[e] = max(-(int)qmax, min((int)qmax, __float2int_rn(f[e] * id)));
        if (bits == 8) {
            unsigned long long raw = 0;
#pragma unroll
            for (int e = 0; e < 8; ++e) raw |= (unsigned long long)(unsigned)(qi[e] & 0xFF) << (8 * e);
            uint8_t* dst = q + r * rbytes + lane * epl;
            if (epl == 8) *reinterpret_cast<unsigned long long*>(dst) = raw;
            else if (epl == 4) *reinterpret_cast<unsigned*>(dst) = (unsigned)raw;
            else *reinterpret_cast<unsigned short*>(dst) = (unsigned short)raw;
        } else {
            unsigned raw = 0;
#pragma unroll
            for (int e = 0; e < 8; ++e) raw |= (unsigned)((qi[e] + 8) & 15) << (4 * e);
            uint8_t* dst = q + r * rbytes + lane * (epl >> 1);
            if (epl == 8) *reinterpret_cast<unsigned*>(dst) = raw;
            else if (epl == 4) *reinterpret_cast<unsigned short*>(dst) = (unsigned short)raw;
            else *dst = (uint8_t)raw;
        }
        if ((lane % lpg) == 0) sc[r * ng + lane / lpg] = __float2half(ds);
    }
}
