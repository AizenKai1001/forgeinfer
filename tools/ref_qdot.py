"""Executable SPECIFICATION of batch-1 quantized dot products as a dp4a kernel computes them (numpy only, written from scratch).

    python3 tools/ref_qdot.py              golden checks, layout identities, integer ranges, error tables
    python3 tools/ref_qdot.py --dump DIR   also write native-layout test vectors (weights, Q8_1 activation, expected outputs) for kernel unit tests

For F16, Q8_0, Q4_0, Q4_1, Q5_0, Q4_K, Q5_K, Q6_K it holds: our own quantiser (llama.cpp style: round to nearest, d = signed max / -2^(bits-1) or absmax / 127, least-squares scale search for K-quants);
a WORD-PARALLEL unpacker (the 32-bit AND / shift / OR sequence one GPU lane runs on little-endian words); the dequantiser built on it (bit-exact against tests/golden); and the integer dot product
against the Q8_1 activation: int32 per 32-weight block by chained dp4a, one float scale per block, fp32 accumulation in the order of a warp-per-row kernel.
Research note: "ForgeInfer Research - Quantized Dot Products and GPU Layouts" (every formula below is derived there).
"""
import json
import os
import sys

import numpy as np

F32, I32 = np.float32, np.int32
GOLD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests", "golden")
K1, K2, K3 = 0x3F3F3F3F, 0x0F0F0F0F, 0x03030303                                                    # byte masks of the word-parallel unpackers


def f16(x): return np.asarray(x, F32).astype(np.float16).astype(F32)                               # round to binary16 and back
def f64(x): return np.asarray(x, np.float64)
def rnd(x): x = f64(x); return (np.sign(x) * np.floor(np.abs(x) + 0.5)).astype(F32)               # C roundf (ties away from zero); float64 so x + 0.5 cannot round up wrongly
def inv(d): return np.where(d != 0, F32(1) / np.where(d != 0, d, 1), 0).astype(F32)               # 1/d, or 0
def Wd(a): return np.ascontiguousarray(a).view("<u4")                                              # bytes -> 32-bit words: the unit a lane works on
def By(w): return np.ascontiguousarray(np.asarray(w, "<u4")).view(np.uint8)                        # words -> bytes
def h2b(x): return np.ascontiguousarray(np.asarray(x, F32).astype("<f2")[..., None]).view(np.uint8)   # f16 -> 2 bytes
def b2h(b): return np.ascontiguousarray(b).view("<f2")[..., 0].astype(F32)                         # 2 bytes -> f32
def cat(*p): return np.concatenate(p, -1)
def rel(y, ref): return float(np.linalg.norm(f64(y) - f64(ref)) / np.linalg.norm(f64(ref)))        # relative RMS error
def jload(name):
    with open(os.path.join(GOLD, name)) as fh: return json.load(fh)


def dp4a(a, b, c):   # __dp4a: c + sum of four byte products; a, b hold the (sign- or zero-extended) byte lanes; wraps modulo 2^32, no saturating form exists
    return (np.asarray(c, I32) + (a.astype(I32) * b.astype(I32)).sum(-1, dtype=I32)).astype(I32)

def dot32(c, q, n=32):   # n weights = n/4 chained dp4a on 4-byte words; c [..., n] weight codes, q [..., n] int8 activations -> int32 [...]
    acc = np.zeros(np.broadcast_shapes(c.shape[:-1], q.shape[:-1]), I32)
    for m in range(n // 4): acc = dp4a(c[..., 4 * m:4 * m + 4], q[..., 4 * m:4 * m + 4], acc)
    return acc

def lane_sum(t, lanes=32):   # float32 terms [N, T]: lane l adds t[:, l::lanes] in order, then the xor butterfly 16,8,4,2,1 of a warp-per-row kernel; lane 0's result
    n, T = t.shape
    t = cat(t, np.zeros((n, (-T) % lanes), F32)).reshape(n, -1, lanes)
    p = np.zeros((n, lanes), F32)
    for i in range(t.shape[1]): p = p + t[:, i]
    o = lanes // 2
    while o: p, o = p + p[:, np.arange(lanes) ^ o], o // 2
    return p[:, 0]

def quant_act(x, bs=32, half=True):   # Q8_1: blocks of 32, d = amax/127 (fp32 division), q = roundf(x/d) with the UNROUNDED d; the dot then uses the half-rounded d.
    b = np.asarray(x, F32).reshape(-1, bs)                                                   # s_cuda = ds.y of llama.cpp's CUDA kernel (half of the FLOAT sum); s_cpu = CPU reference / candle (half of d * integer sum)
    amax = np.abs(b).max(-1)
    d = amax / F32(127)
    q = np.where(amax[:, None] == 0, 0, rnd(b / np.where(d == 0, F32(1), d)[:, None])).astype(np.int8)
    sq = q.sum(-1, dtype=I32)
    return dict(q=q, d=d, d16=f16(d) if half else d, sq=sq, s_f32=b.sum(-1, dtype=F32), s_cuda=f16(b.sum(-1, dtype=F32)), s_cpu=f16(d * sq.astype(F32)))


# ---------------------------------------------------------------- quantisers (own code; they emit raw bytes in the GGUF block layout) -----------------------------------------------------------------------
def nib(L): return (L[..., :16] | (L[..., 16:] << 4)).astype(np.uint8)                              # byte j = code j | code j+16 << 4

def q_q8_0(w):
    b = w.reshape(w.shape[0], -1, 32); d = np.abs(b).max(-1) / F32(127)
    return cat(h2b(d), rnd(b * inv(d)[..., None]).astype(np.int8).view(np.uint8))

def q_signed_max(w, nmax, hi):   # Q4_0 / Q5_0: d = (element of largest magnitude, WITH its sign) / -nmax, code = min(hi, int(x/d + nmax + 0.5))
    b = w.reshape(w.shape[0], -1, 32)
    d = np.take_along_axis(b, np.abs(b).argmax(-1)[..., None], -1)[..., 0] / F32(-nmax)
    return d, np.minimum(hi, np.floor(b * inv(d)[..., None] + F32(nmax + 0.5))).astype(np.uint8)

def q_q4_0(w): d, L = q_signed_max(w, 8, 15); return cat(h2b(d), nib(L))

def q_q5_0(w):
    d, L = q_signed_max(w, 16, 31)
    qh = (((L >> 4) & 1).astype(np.uint32) << np.arange(32, dtype=np.uint32)).sum(-1, dtype=np.uint32)   # bit j of qh = 5th bit of weight j
    return cat(h2b(d), By(qh[..., None]), nib(L & 15))

def q_q4_1(w):
    b = w.reshape(w.shape[0], -1, 32); mn = b.min(-1); d = (b.max(-1) - mn) / F32(15)
    return cat(h2b(d), h2b(mn), nib(np.minimum(15, np.floor((b - mn[..., None]) * inv(d)[..., None] + F32(0.5))).astype(np.uint8)))

def fit_affine(x, nmax, rmin, rdelta, nstep):   # make_qkx2_quants vectorised over rows x [S,32]: x ~ sc*L - mn, L in 0..nmax, mn >= 0; error weights = rms + |x|
    w = np.sqrt((x * x).mean(-1, keepdims=True)) + np.abs(x)
    lo, hi = np.minimum(x.min(-1), 0), x.max(-1)
    rng, sw, sx = np.where(hi > lo, hi - lo, 1), w.sum(-1), (w * x).sum(-1)
    def lv(isc): return np.clip(np.rint((x - lo[:, None]) * isc[:, None]), 0, nmax)
    def err(sc, mn, L): return (w * (sc[:, None] * L + mn[:, None] - x) ** 2).sum(-1)
    L, sc, mn = lv(nmax / rng), rng / nmax, lo
    best = err(sc, mn, L)
    for i in range(nstep + 1):
        La = lv((rmin + rdelta * i + nmax) / rng)
        sl, sl2, sxl = (w * La).sum(-1), (w * La * La).sum(-1), (w * x * La).sum(-1)
        D = sw * sl2 - sl * sl
        ds = np.where(D > 0, D, 1)
        ts, tm = (sw * sxl - sx * sl) / ds, (sl2 * sx - sl * sxl) / ds
        pos = tm > 0
        ts, tm = np.where(pos, sxl / np.where(sl2 > 0, sl2, 1), ts), np.where(pos, 0, tm)
        e = err(ts, tm, La); up = (D > 0) & (e < best)
        L, sc, mn, best = np.where(up[:, None], La, L), np.where(up, ts, sc), np.where(up, tm, mn), np.where(up, e, best)
    flat = hi <= lo
    return np.where(flat, 0, sc), -np.where(flat, lo, mn), np.where(flat[:, None], 0, L)

def q_kquant(w, nmax):   # Q4_K (nmax 15) / Q5_K (31): 8 sub-blocks of 32, 6-bit scale and 6-bit min per sub-block, fp16 d and dmin per 256
    n = w.shape[0]; x = w.reshape(n, -1, 8, 32); ns = x.shape[1]
    sc, mn, _ = fit_affine(x.reshape(-1, 32), nmax, *((-1.0, 0.1, 20) if nmax == 15 else (-0.5, 0.1, 15)))
    sc, mn = sc.reshape(n, ns, 8), mn.reshape(n, ns, 8)
    ms, mm = sc.max(-1, keepdims=True), mn.max(-1, keepdims=True)
    ls = np.minimum(63, np.rint(np.where(ms > 0, 63 / np.where(ms > 0, ms, 1), 0) * sc)).astype(np.uint8)
    lm = np.minimum(63, np.rint(np.where(mm > 0, 63 / np.where(mm > 0, mm, 1), 0) * mn)).astype(np.uint8)
    d, dm = f16(ms / 63), f16(mm / 63)
    de, me = d * ls, dm * lm                                                                      # effective scale and min of every sub-block; the codes are recomputed against them
    L = (np.clip(np.rint((x + me[..., None]) / np.where(de > 0, de, 1)[..., None]), 0, nmax) * (de > 0)[..., None]).astype(np.uint8)
    s12 = np.zeros((n, ns, 12), np.uint8)
    s12[..., 0:4], s12[..., 4:8] = ls[..., :4] | ((ls[..., 4:] >> 4) << 6), lm[..., :4] | ((lm[..., 4:] >> 4) << 6)
    s12[..., 8:12] = (ls[..., 4:] & 15) | ((lm[..., 4:] & 15) << 4)
    qs = ((L & 15)[:, :, 0::2] | ((L & 15)[:, :, 1::2] << 4)).reshape(n, ns, 128)                   # chunk c: byte l = code(sub-block 2c)[l] | code(2c+1)[l] << 4
    qh = (((L >> 4) & 1) << np.arange(8, dtype=np.uint8)[None, None, :, None]).sum(2, dtype=np.uint8)   # bit j of qh[l] = 5th bit of sub-block j, position l
    return cat(h2b(d[..., 0]), h2b(dm[..., 0]), s12, *((qs,) if nmax == 15 else (qh, qs)))

def fit_sym(x, nmax=32):   # make_qx_quants (rmse_type 1) vectorised over groups x [S,16]: x ~ sc*l, l in -nmax..nmax-1, objective weights x^2; returns sc
    mx = np.take_along_axis(x, np.abs(x).argmax(-1)[:, None], -1)[:, 0]; w = x * x
    ok = np.abs(mx) > 1e-15; mxs = np.where(ok, mx, 1)
    def tr(k): l = np.clip(np.rint(-(nmax + 0.1 * k) / mxs[:, None] * x), -nmax, nmax - 1); return (w * x * l).sum(-1), (w * l * l).sum(-1)
    a, b = tr(0); sc = a / np.where(b > 0, b, 1); best = sc * a
    for k in list(range(-9, 0)) + list(range(1, 10)):
        a, b = tr(k); up = (b > 0) & (a * a > best * b)
        sc = np.where(up, a / np.where(b > 0, b, 1), sc); best = np.where(up, sc * a, best)
    return np.where(ok, sc, 0)

def q_q6_k(w):   # 16 groups of 16 with int8 scales; d (fp16, may be negative) is stored LAST; codes are recomputed against the quantised scales
    n = w.shape[0]; x = w.reshape(n, -1, 16, 16); ns = x.shape[1]
    sc = fit_sym(x.reshape(-1, 16)).reshape(n, ns, 16)
    ms = np.take_along_axis(sc, np.abs(sc).argmax(-1)[..., None], -1)
    isc = np.where(np.abs(ms) > 1e-15, -128 / np.where(ms != 0, ms, 1), 0)
    d = f16(inv(isc.astype(F32)))
    s8 = np.minimum(127, np.rint(isc * sc)).astype(np.int8)
    dj = d * s8
    l = np.clip(np.rint(x / np.where(dj != 0, dj, 1)[..., None]), -32, 31) * (dj != 0)[..., None]
    u = (l + 32).astype(np.uint8).reshape(n, ns, 2, 4, 32)                                          # [half, group of 32, l]
    ql = cat((u[:, :, :, 0] & 15) | ((u[:, :, :, 2] & 15) << 4), (u[:, :, :, 1] & 15) | ((u[:, :, :, 3] & 15) << 4)).reshape(n, ns, 128)
    qh = ((u[:, :, :, 0] >> 4) | ((u[:, :, :, 1] >> 4) << 2) | ((u[:, :, :, 2] >> 4) << 4) | ((u[:, :, :, 3] >> 4) << 6)).reshape(n, ns, 64)
    return cat(ql, qh, s8.view(np.uint8), h2b(d[..., 0]))


# ---------------------------------------------------------------- word-parallel unpackers: raw [N, nb, bytes] -> codes in LOGICAL weight order + scale fields ----------------------------------------------
def k4_scales(s12):   # 12 packed bytes -> eight 6-bit scales, eight 6-bit mins; three words, about ten integer ops for all sixteen values
    u0, u1, u2 = (Wd(s12)[..., i] for i in range(3))
    sc = cat(By((u0 & K1)[..., None]), By(((u2 & K2) | (((u0 >> 6) & K3) << 4))[..., None]))
    mn = cat(By((u1 & K1)[..., None]), By((((u2 >> 4) & K2) | (((u1 >> 6) & K3) << 4))[..., None]))
    return sc, mn

def unpack(fmt, raw):
    if fmt == "F16": return dict(fmt=fmt, w=raw.astype(F32))
    U, ns = dict(fmt=fmt), raw.shape[:2]
    if fmt in ("Q8_0", "Q4_0", "Q4_1", "Q5_0"):
        U["d"] = b2h(raw[..., 0:2])
        if fmt == "Q8_0": U["c"] = raw[..., 2:34].view(np.int8)
        elif fmt == "Q5_0":
            qh, w = Wd(raw[..., 2:6]), Wd(raw[..., 6:22]); vh = qh >> (4 * np.arange(4, dtype=np.uint32))   # word m of the nibbles sees qh bits 4m..4m+3 (low) and 16+4m.. (high)
            lo = (w & K2) | ((vh << 4) & 0x10) | ((vh << 11) & 0x1000) | ((vh << 18) & 0x100000) | ((vh << 25) & 0x10000000)
            hi = ((w >> 4) & K2) | ((vh >> 12) & 0x10) | ((vh >> 5) & 0x1000) | ((vh << 2) & 0x100000) | ((vh << 9) & 0x10000000)
            U["c"] = cat(By(lo), By(hi))
        else:
            if fmt == "Q4_1": U["m"] = b2h(raw[..., 2:4])
            w = Wd(raw[..., 4 if fmt == "Q4_1" else 2:][..., :16]); U["c"] = cat(By(w & K2), By((w >> 4) & K2))
    elif fmt in ("Q4_K", "Q5_K"):
        q5 = fmt == "Q5_K"
        U["d"], U["dm"] = b2h(raw[..., 0:2]), b2h(raw[..., 2:4])
        sc, mn = k4_scales(raw[..., 4:16]); U["sc"], U["mn"] = sc.astype(I32), mn.astype(I32)
        w = Wd(raw[..., 48 if q5 else 16:][..., :128]).reshape(*ns, 4, 8)                            # [N, sb, chunk of 64 weights, word]
        lo, hi = w & K2, (w >> 4) & K2
        if q5:
            qh = Wd(raw[..., 16:48])[:, :, None, :]; sh = (2 * np.arange(4, dtype=np.uint32))[None, None, :, None]
            lo, hi = lo | (((qh >> sh) & 0x01010101) << 4), hi | (((qh >> (sh + 1)) & 0x01010101) << 4)
        U["c"] = np.stack([By(lo), By(hi)], 3).reshape(*ns, 8, 32)                                   # sub-block 2c = low nibbles of chunk c, 2c+1 = high nibbles
    else:  # Q6_K
        ql, qh = Wd(raw[..., 0:128]).reshape(*ns, 2, 16), Wd(raw[..., 128:192]).reshape(*ns, 2, 8)   # [N, sb, half, word]
        A, B2, H = ql[..., :8], ql[..., 8:], qh
        g = [(A & K2) | ((H & K3) << 4), (B2 & K2) | (((H >> 2) & K3) << 4), ((A >> 4) & K2) | (((H >> 4) & K3) << 4), ((B2 >> 4) & K2) | (((H >> 6) & K3) << 4)]
        U["c"] = (np.stack([By(x) for x in g], 3).reshape(*ns, 16, 16).astype(np.int16) - 32).astype(np.int8)   # the -32 of vsub4; logical groups of 16
        U["sc"], U["d"] = raw[..., 192:208].view(np.int8).astype(I32), b2h(raw[..., 208:210])
    return U

def deq(U):   # float32 weights [N, K]; expression order identical to the reference dequantisers that produced tests/golden
    f, c = U["fmt"], U.get("c")
    if f == "F16": return U["w"]
    if f == "Q8_0": v = U["d"][..., None] * c
    elif f == "Q4_0": v = U["d"][..., None] * (c.astype(F32) - 8.0)
    elif f == "Q5_0": v = U["d"][..., None] * (c.astype(F32) - 16.0)
    elif f == "Q4_1": v = U["d"][..., None] * c + U["m"][..., None]
    elif f in ("Q4_K", "Q5_K"): v = (U["d"][..., None] * U["sc"])[..., None] * c - (U["dm"][..., None] * U["mn"])[..., None]
    else: v = (U["d"][..., None] * U["sc"])[..., None] * c
    return v.reshape(c.shape[0], -1).astype(F32)


# ---------------------------------------------------------------- the dot product as a dp4a kernel forms it: int32 sums per block, one float scale per block ---------------------------------------------
def terms(U, A, svar="cuda"):   # float32 contribution of every block [N, blocks]; svar = which block sum the offset formats subtract (see quant_act): cuda | float | cpu | int
    f, dx = U["fmt"], A["d16"]
    if f in ("Q8_0", "Q4_0", "Q4_1", "Q5_0"):
        sumi, d = dot32(U["c"], A["q"]), U["d"]                                                    # 8 dp4a per 32-weight block (nibble codes unsigned, activations signed)
        S = {"cuda": A["s_cuda"], "float": A["s_f32"], "cpu": A["s_cpu"], "int": dx * A["sq"].astype(F32)}[svar]
        if f == "Q8_0": return (d * dx) * sumi.astype(F32)
        if f == "Q4_1": return sumi.astype(F32) * (d * dx) + U["m"] * S                            # sum(d*c + m) * x = d*dx*sumi + m*S
        z = 8 if f == "Q4_0" else 16
        if svar == "int": return (d * dx) * (sumi - z * A["sq"]).astype(F32)                       # offset folded in the integer domain: exact, one rounding
        return d * (sumi.astype(F32) * dx - F32(z) * S)                                             # llama.cpp CUDA: d * (sumi*dx - z*S); its two threads per block each subtract z/2 * S
    if f in ("Q4_K", "Q5_K"):
        q3, d8 = A["q"].reshape(-1, 8, 32), dx.reshape(-1, 8)
        dot1, dot2 = dot32(U["c"], q3), dot32(np.ones(32, np.int8), q3)                            # dot2 = dp4a against 0x01010101 = sum of the activations of the sub-block (== A['sq'])
        sd, sm = np.zeros(dot1.shape[:2], F32), np.zeros(dot1.shape[:2], F32)
        for j in range(8):
            sd = sd + d8[:, j] * (dot1[..., j] * U["sc"][..., j]).astype(F32)
            sm = sm + d8[:, j] * (dot2[:, j] * U["mn"][..., j]).astype(F32)
        return U["d"] * sd - U["dm"] * sm
    dot = dot32(U["c"], A["q"].reshape(-1, 16, 16), 16)                                            # Q6_K: 16 groups of 16 (signed codes after the -32), two groups per activation block
    d8, sf = np.repeat(dx, 2).reshape(-1, 16), np.zeros(dot.shape[:2], F32)
    for g in range(16): sf = sf + d8[:, g] * (dot[..., g] * U["sc"][..., g]).astype(F32)
    return U["d"] * sf

def matvec(U, x, svar="cuda", lanes=32, half=True):
    if U["fmt"] == "F16": return lane_sum(U["w"] * np.asarray(x, F32)[None, :], lanes)
    return lane_sum(terms(U, quant_act(x, 32, half), svar), lanes)


QUANT = dict(F16=lambda w: w.astype(np.float16), Q8_0=q_q8_0, Q4_0=q_q4_0, Q4_1=q_q4_1, Q5_0=q_q5_0, Q4_K=lambda w: q_kquant(w, 15), Q5_K=lambda w: q_kquant(w, 31), Q6_K=q_q6_k)
BPW = dict(F16=16, Q8_0=8.5, Q4_0=4.5, Q4_1=5.0, Q5_0=5.5, Q4_K=4.5, Q5_K=5.5, Q6_K=6.5625)
# repacked GPU planes: fmt -> (weights per block, [(plane, bytes per block, load width in bytes)]); every plane is [rows][blocks_per_row * bytes]; total bytes per block == the native GGUF block, no padding anywhere
PLANES = {"Q8_0": (32, [("q", 32, 16), ("d", 2, 2)]), "Q4_0": (32, [("q", 16, 16), ("d", 2, 2)]), "Q4_1": (32, [("q", 16, 16), ("dm", 4, 4)]), "Q5_0": (32, [("q", 16, 16), ("qh", 4, 4), ("d", 2, 2)]),
          "Q4_K": (256, [("q", 128, 16), ("hdr", 16, 16)]), "Q5_K": (256, [("q", 128, 16), ("qh", 32, 4), ("hdr", 16, 16)]), "Q6_K": (256, [("ql", 128, 16), ("qh", 64, 16), ("sc", 16, 4), ("d", 2, 2)])}


# ---------------------------------------------------------------- checks and tables ---------------------------------------------------------------------------------------------------------------------
def check_golden():
    ok = True
    for name, meta in jload("quant_index.json").items():
        raw = np.fromfile(os.path.join(GOLD, f"quant_{name}.in"), dtype=np.uint8); want = np.fromfile(os.path.join(GOLD, f"quant_{name}.f32"), dtype="<f4")
        raw = raw.view("<f2")[None, :] if name == "F16" else raw.reshape(1, meta["blocks"], meta["block_bytes"])
        err = float(np.abs(deq(unpack(name, raw)).reshape(-1) - want).max()); ok &= err == 0
        print(f"  {name:5s} {len(want):6d} values, max |diff| {err:.1e} {'bit-exact' if err == 0 else 'MISMATCH'}")
    return ok


def check_roundtrip(seed=4):   # our packers + word-parallel unpackers against an independent decoder (ref_gguf.py, itself verified on the goldens) and a textbook Q5_K loop
    ok, w = True, (np.random.default_rng(seed).standard_normal((4, 1024)) * 0.05).astype(F32)
    try: sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import ref_gguf
    except Exception: ref_gguf = None
    for f in ("Q8_0", "Q4_0", "Q4_1", "Q5_0", "Q4_K", "Q5_K", "Q6_K"):
        raw = QUANT[f](w); got = deq(unpack(f, raw)); ref = None
        if f == "Q5_K":   # textbook order of the reference dequantiser: the mask walks up two bits per 64-weight chunk
            ref = np.zeros(got.shape, F32)
            for r, b in np.ndindex(4, raw.shape[1]):
                blk = raw[r, b]; d, dm = b2h(blk[0:2]), b2h(blk[2:4]); sc, mn = k4_scales(blk[4:16]); qh, qs = blk[16:48], blk[48:176]
                for j in range(8):
                    q = ((qs[32 * (j // 2):32 * (j // 2) + 32] >> (4 * (j % 2))) & 15).astype(I32) + 16 * ((qh >> j) & 1).astype(I32)
                    ref[r, b * 256 + 32 * j:b * 256 + 32 * j + 32] = (d * sc[j]) * q - dm * mn[j]
        elif ref_gguf is not None:
            t = next(k for k, v in ref_gguf.TYPES.items() if v[0] == f); ref = ref_gguf.dequant(t, raw.tobytes(), got.size).reshape(got.shape)
        same = "n/a" if ref is None else ("identical" if np.array_equal(ref, got) else "DIFFERENT"); ok &= same != "DIFFERENT"
        print(f"  {f:5s} weight error {rel(got, w):.4f} (relative RMS)   word-parallel unpack vs independent decoder: {same}")
    return ok


def check_layouts(seed=5):   # identities behind the repacked layouts (note, section 5); every line must say exact
    rng = np.random.default_rng(seed); r32 = lambda *s: rng.integers(0, 2 ** 32, s, dtype=np.uint64).astype(np.uint32)
    w4, q = r32(64, 4), rng.integers(-127, 128, (64, 32)).astype(np.int8)
    lo, hi = By(w4 & K2).reshape(64, 16), By((w4 >> 4) & K2).reshape(64, 16)
    ref = dot32(lo.astype(I32) - 8, q[:, :16], 16) + dot32(hi.astype(I32) - 8, q[:, 16:], 16)         # sum (c-8) x with the native nibble order (weights j low, j+16 high)
    fh, fl = By((w4 & 0xF0F0F0F0) ^ 0x80808080).view(np.int8).reshape(64, 16), By(((w4 << 4) & 0xF0F0F0F0) ^ 0x80808080).view(np.int8).reshape(64, 16)
    res = [("Q4_0 sign fold: int8 = ((c<<4) ^ 0x80) = 16*(c-8), dp4a needs no offset correction", np.array_equal(dot32(fl, q[:, :16], 16) + dot32(fh, q[:, 16:], 16), 16 * ref)),
           ("Q4_0 shift-free high nibbles: dp4a.u32.s32(w & 0xF0F0F0F0, x) = 16 * dot", np.array_equal(dot32(By(w4 & 0xF0F0F0F0).reshape(64, 16), q[:, 16:], 16), 16 * dot32(hi, q[:, 16:], 16)))]
    def tword(bit): return sum(int(bit(p, i)) << (8 * i + p) for p in range(8) for i in range(4))     # 5th-bit word of one 32-weight task: bit (8i+p) = 5th bit of operand p (p<4: low nibbles of word p, else high nibbles of word p-4), byte i
    def tbits(w, p): return (((w << (4 - p)) if p <= 4 else (w >> (p - 4))) & 0x10101010) >> 4          # one shift + one AND (llama.cpp: 4 shifts, 4 ANDs, 4 ORs): bit 0 of byte i
    w5 = rng.standard_normal((2, 512)).astype(F32); r0, r1 = QUANT["Q5_0"](w5), QUANT["Q5_K"](w5); c0, c1 = unpack("Q5_0", r0)["c"], unpack("Q5_K", r1)["c"]; good = True
    for r, b, p in np.ndindex(2, 16, 8):                                                                  # Q5_0: task = block, operand p byte i = weight 4p+i
        qh = int(Wd(r0[r, b, 2:6])[0]); wd = tword(lambda p_, i: (qh >> (4 * p_ + i)) & 1)
        good &= tbits(wd, p) == sum(int((c0[r, b, 4 * p + i] >> 4) & 1) << (8 * i) for i in range(4))
    for r, s, t, p in np.ndindex(2, 2, 8, 8):                                                             # Q5_K: task t = (chunk c, half); operand p<4 = sub-block 2c, p>=4 = 2c+1, positions 16*half + 4*(p%4) + i
        c, hf, qb = *divmod(t, 2), r1[r, s, 16:48]; wd = tword(lambda p_, i: (int(qb[16 * hf + 4 * (p_ % 4) + i]) >> (2 * c + p_ // 4)) & 1)
        good &= tbits(wd, p) == sum(int((c1[r, s, 2 * c + p // 4, 16 * hf + 4 * (p % 4) + i] >> 4) & 1) << (8 * i) for i in range(4))
    res.append(("Q5_0 / Q5_K repacked 5th-bit words (transposed, per 32-weight task) reproduce the decoded 5th bits", good))
    good = True   # lane-task decomposition: Q4_K/Q5_K task t = (chunk c, half) = 16 bytes of qs at 16*t -> positions 16*half.. of sub-blocks 2c (low nibbles) and 2c+1 (high nibbles)
    for r, s, t in np.ndindex(2, 2, 8):
        ch, c, hf = r1[r, s, 48 + 16 * t:64 + 16 * t], t // 2, t % 2
        good &= np.array_equal(ch & 15, c1[r, s, 2 * c, 16 * hf:16 * hf + 16] & 15) and np.array_equal(ch >> 4, c1[r, s, 2 * c + 1, 16 * hf:16 * hf + 16] & 15)
    res.append(("Q4_K/Q5_K task = 16 B of qs -> positions 16*half.. of sub-blocks 2c and 2c+1", good))
    r6 = QUANT["Q6_K"](w5); u6 = unpack("Q6_K", r6); good = True   # Q6_K task (sb, h, u): ql[64h+16u..], ql[64h+32+16u..], qh[32h+16u..] -> 4 groups x 16 positions; sc'[4(2h+u)+g] = sc[8h+u+2g]
    for r, s, h, u in np.ndindex(2, 2, 2, 2):
        b = r6[r, s].astype(I32); qa, qb, qh = b[64 * h + 16 * u:][:16], b[64 * h + 32 + 16 * u:][:16], b[128 + 32 * h + 16 * u:][:16]
        for g, v in enumerate([(qa & 15) | ((qh & 3) << 4), (qb & 15) | (((qh >> 2) & 3) << 4), (qa >> 4) | (((qh >> 4) & 3) << 4), (qb >> 4) | (((qh >> 6) & 3) << 4)]):
            good &= np.array_equal(v - 32, u6["c"].reshape(2, 2, 256)[r, s, 128 * h + 32 * g + 16 * u:][:16])
    res.append(("Q6_K task = 3 x 16 B (ql_a, ql_b, qh) -> 4 groups x 16 positions; group scale index 8h+2g+u", good))
    A, B2, H = r32(64), r32(64), r32(64)
    std = [(A & K2) | ((H & K3) << 4), (B2 & K2) | (((H >> 2) & K3) << 4), ((A >> 4) & K2) | (((H >> 4) & K3) << 4), ((B2 >> 4) & K2) | (((H >> 6) & K3) << 4)]
    fold = [(((A << 2) & 0x3C3C3C3C) | ((H << 6) & 0xC0C0C0C0)) ^ 0x80808080, (((B2 << 2) & 0x3C3C3C3C) | ((H << 4) & 0xC0C0C0C0)) ^ 0x80808080,
            (((A >> 2) & 0x3C3C3C3C) | ((H << 2) & 0xC0C0C0C0)) ^ 0x80808080, (((B2 >> 2) & 0x3C3C3C3C) | (H & 0xC0C0C0C0)) ^ 0x80808080]
    res.append(("Q6_K sign fold: int8 = 4*(u-32) from (u<<2) ^ 0x80, no vsub4 emulation", all(np.array_equal(By(a).view(np.int8).astype(I32), 4 * (By(s).astype(I32) - 32)) for a, s in zip(fold, std))))
    for name, ok in res: print(f"  {'exact' if ok else 'FAIL '} {name}")
    return all(ok for _, ok in res)

def check_ranges():
    print("  worst-case int32 magnitudes (activation codes |q| <= 127, weight codes at their extremes):")
    for name, n, cm, s in [("Q8_0 block (|c| <= 128 if a file contains -128)", 32, 128, 1), ("Q4_0 / Q4_1 block, unsigned nibbles", 32, 15, 1), ("Q5_0 block, unsigned 5-bit", 32, 31, 1),
                           ("Q4_K sub-block dot * 6-bit scale", 32, 15, 63), ("Q5_K sub-block dot * 6-bit scale", 32, 31, 63), ("Q6_K group of 16 * int8 scale (|sc| <= 128)", 16, 32, 128),
                           ("Q6_K whole super-block summed in int32", 256, 32, 128)]:
        v = n * cm * 127 * s; print(f"    {name:48s} {v:12,d} = {v / 2 ** 31:6.4f} of 2^31  {'exact in fp32' if v < 2 ** 24 else 'above 2^24: fp32 conversion rounds'}")
    print(f"    one int32 over a whole row of Q8_0 products would overflow at K > {2 ** 31 // (128 * 127):,d}; no real model comes close")

def layout():
    print("  repacked planes, bytes per row (plane stride = bytes, rows are 16-byte aligned for 128-bit planes) and 16-byte quant loads per row:")
    for f, (bw, pl) in PLANES.items():
        out = []
        for K in (576, 1536):
            if K % bw: out.append(f"K={K}: not allowed"); continue
            tot = sum(K // bw * b for _, b, _ in pl); assert all((K // bw * b) % wd == 0 for _, b, wd in pl)
            nat = dict(Q8_0=34, Q4_0=18, Q4_1=20, Q5_0=22, Q4_K=144, Q5_K=176, Q6_K=210)[f]; assert tot == K // bw * nat   # repacked bytes == native bytes
            out.append(f"K={K}: " + "+".join(f"{n} {K // bw * b}" for n, b, _ in pl) + f" = {tot} B (native {K // bw * nat}); 128-bit loads {sum(K // bw * b // 16 for _, b, wd in pl if wd == 16)}")
        print(f"    {f:5s} " + "   ".join(out))

def gaussian_table(seed=2, N=256, K=1536, nvec=16):
    print(f"  W[{N}x{K}] ~ N(0,1), {nvec} activation vectors ~ N(0,1); relative RMS errors.  'W' = the weights alone (Frobenius); 'order' = same pipeline summed with 2 lanes instead of 32;")
    print("  'exact' = integer pipeline (integer fold) versus the float64 product of the dequantised weights and the dequantised quantised activation: must be < 1e-5 (fp32 rounding only)")
    print("    fmt    bpw        W            y: weights only   y: weights + Q8_1   activation increment   order    exact")
    rng = np.random.default_rng(seed); W = rng.standard_normal((N, K)).astype(F32); X = rng.standard_normal((nvec, K)).astype(F32); ok = True
    for f in QUANT:
        U = unpack(f, QUANT[f](W)); Wq = f64(deq(U)); y0, yq = f64(W) @ f64(X).T, Wq @ f64(X).T
        yi = np.stack([matvec(U, x) for x in X], 1); yo = np.stack([matvec(U, x, lanes=2) for x in X], 1)
        XH = f64(X) if f == "F16" else f64([(a["d16"][:, None] * a["q"]).reshape(-1) for a in map(quant_act, X)])
        ex = rel(np.stack([matvec(U, x, "int") for x in X], 1), Wq @ XH.T); ok &= ex < 1e-5 and rel(yi, yq) < (2.2e-2 if f in ("Q4_0", "Q4_1", "Q5_0") else 8e-3)   # also bounds the activation quantiser
        print(f"    {f:5s} {BPW[f]:7.4f}   {rel(Wq, W):.3e}    {rel(yq, y0):.3e}         {rel(yi, y0):.3e}            {rel(yi, yq):.3e}           {rel(yo, yi):.1e}  {ex:.1e}")
    return ok

def svariant_table(seed=3, N=256, K=1536, nvec=16):
    print("  offset formats, activation increment (vs float matvec of the same dequantised weights) by the block sum that is subtracted:")
    print("    fmt    half(sum x) [CUDA]   float sum x   half(d*sum q) [CPU, candle]   integer sum q (exact)")
    rng = np.random.default_rng(seed); W = rng.standard_normal((N, K)).astype(F32); X = rng.standard_normal((nvec, K)).astype(F32)
    for f in ("Q4_0", "Q4_1", "Q5_0"):
        U = unpack(f, QUANT[f](W)); yq = f64(deq(U)) @ f64(X).T; ys = {sv: np.stack([matvec(U, x, sv) for x in X], 1) for sv in ("cuda", "float", "cpu", "int")}
        print(f"    {f:5s}  " + "      ".join(f"{rel(ys[sv], yq):.3e}" for sv in ys) + f"      outputs of the CUDA and the integer fold differ by {rel(ys['cuda'], ys['int']):.1e}")
    U = unpack("Q8_0", QUANT["Q8_0"](W)); ya, yb = (np.stack([matvec(U, x, half=h) for x in X], 1) for h in (True, False))
    print(f"  parity with a Q8_1 oracle needs the half-rounded dx: using fp32 dx changes Q8_0 outputs by {rel(yb, ya):.1e} (accuracy is the same)")


def real_layers():
    d = os.path.join(GOLD, "layer_smollm2")
    if not os.path.isdir(d): print("  (tests/golden/layer_smollm2 missing: skipped)"); return
    ld = lambda L, n: np.fromfile(os.path.join(d, f"L{L}_{n}.f32"), dtype="<f4")
    acts = {576: [ld(L, n) for L in (0, 7, 15, 29) for n in ("h", "o", "h2")], 1536: [ld(L, "a") for L in (0, 7, 15, 29)]}
    def qerr(v, bs, half): a = quant_act(v, bs, half); return rel((a["d16"][:, None] * a["q"]).reshape(-1), v)
    print("  Q8_1 quantisation error of REAL SmolLM2 matvec inputs (13-token prompt, last token, layers 0/7/15/29), relative RMS of the vector, mean (max):")
    for K, vs in acts.items():
        row = [f"blocks of 32 (Q8_1): {np.mean([qerr(v, 32, True) for v in vs]):.2e} ({np.max([qerr(v, 32, True) for v in vs]):.2e})"]
        if K % 256 == 0: row.append(f"256 (Q8_K): {np.mean([qerr(v, 256, False) for v in vs]):.2e} ({np.max([qerr(v, 256, False) for v in vs]):.2e})")
        row.append(f"one scale per vector: {np.mean([qerr(v, K, False) for v in vs]):.2e} ({np.max([qerr(v, K, False) for v in vs]):.2e})")
        mr, ku = [np.abs(v).max() / np.sqrt((v ** 2).mean()) for v in vs], [((v - v.mean()) ** 4).mean() / v.var() ** 2 for v in vs]
        print(f"    K={K}: " + ";  ".join(row) + f";  [max/rms {min(mr):.0f}..{max(mr):.0f}, kurtosis {min(ku):.0f}..{max(ku):.0f} (Gaussian: 3.0)]")
    idx = jload("quant_index.json")
    print("  integer pipeline on REAL quantised weight rows (first complete rows of the golden blocks) x real activations; increment over the float matvec of the same dequantised weights, mean (max):")
    for f in ("Q8_0", "Q4_0", "Q5_0", "Q4_1", "Q4_K", "Q6_K"):
        bb, bv = idx[f]["block_bytes"], idx[f]["block_values"]; K = 576 if f in ("Q8_0", "Q4_0", "Q5_0") else 1536; nbr = K // bv; rows = idx[f]["blocks"] // nbr
        U = unpack(f, np.fromfile(os.path.join(GOLD, f"quant_{f}.in"), dtype=np.uint8)[:rows * nbr * bb].reshape(rows, nbr, bb)); Wq = f64(deq(U))
        r = {sv: [rel(matvec(U, v, sv), Wq @ f64(v)) for v in acts[K]] for sv in (("cuda", "int") if f in ("Q4_0", "Q4_1", "Q5_0") else ("cuda",))}
        print(f"    {f:5s} {rows:2d} rows x K={K}: " + "   ".join(f"[{sv}] {np.mean(e):.2e} ({np.max(e):.2e})" for sv, e in r.items()))


def dump(dirn, N=64, K=1536):   # native-layout test vectors: <fmt>.w = GGUF blocks, x.f32, x.q8_1 = 36-byte blocks (half d, half s_cuda, int8 q[32]), <fmt>.y = fp32 pipeline (offset formats: integer block sum), <fmt>.y64 = float64 dequantised reference
    rng = np.random.default_rng(1); os.makedirs(dirn, exist_ok=True)
    W = rng.standard_normal((N, K)).astype(F32); x = rng.standard_normal(K).astype(F32); A = quant_act(x)
    cat(h2b(A["d"]), h2b(A["s_cuda"]), A["q"].view(np.uint8)).tofile(os.path.join(dirn, "x.q8_1")); x.tofile(os.path.join(dirn, "x.f32"))
    for f in QUANT:
        raw = QUANT[f](W); U = unpack(f, raw); np.ascontiguousarray(raw).tofile(os.path.join(dirn, f"{f}.w"))
        matvec(U, x, "int").astype("<f4").tofile(os.path.join(dirn, f"{f}.y")); (f64(deq(U)) @ f64(x)).tofile(os.path.join(dirn, f"{f}.y64"))
    print(f"  wrote {len(QUANT)} formats (N={N}, K={K}, seed 1) to {dirn}")


def main():
    print("golden dequantization (word-parallel unpack):"); ok = check_golden()
    print("packers + unpackers against independent decoders:"); ok &= check_roundtrip()
    print("layout identities:"); ok &= check_layouts()
    print("integer ranges:"); check_ranges()
    print("repacked layouts:"); layout()
    print("error table, random Gaussian:"); ok &= gaussian_table(); svariant_table()
    print("real layers:"); real_layers()
    if "--dump" in sys.argv: dump(sys.argv[sys.argv.index("--dump") + 1])
    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
