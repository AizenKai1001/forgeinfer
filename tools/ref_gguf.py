"""An executable SPECIFICATION of the GGUF container and of the quantization block formats we must read, written from scratch in numpy and checked against the recorded golden dequantizations
and against the Hugging Face logits. The Rust `forge-gguf` is a port of this file.

    python3 tools/ref_gguf.py              (on the homelab: needs numpy; uses the oracle GGUF files and tests/golden)

GGUF v3 layout:  "GGUF" | u32 version | u64 tensor_count | u64 kv_count | kv_count x (string key, u32 value_type, value) | tensor_count x (string name, u32 n_dims, u64 dims[n_dims], u32 type,
u64 offset) | padding to `general.alignment` (default 32) | tensor data (every tensor offset is relative to the data start and itself aligned). Strings are u64 length + bytes.
Tensor dims are listed fastest-varying first: a weight [out][in] row-major shows up as dims [in, out].
"""
import json
import os
import struct
import sys

import numpy as np

ROOT = os.path.expanduser("~/forgeinfer/oracle/models")
GOLD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests", "golden")

# value types
U8, I8, U16, I16, U32, I32, F32, BOOL, STR, ARR, U64, I64, F64 = range(13)
SCALAR = {U8: "<B", I8: "<b", U16: "<H", I16: "<h", U32: "<I", I32: "<i", F32: "<f", BOOL: "<?", U64: "<Q", I64: "<q", F64: "<d"}

# tensor types: id -> (name, values per block, bytes per block)
TYPES = {0: ("F32", 1, 4), 1: ("F16", 1, 2), 2: ("Q4_0", 32, 18), 3: ("Q4_1", 32, 20), 6: ("Q5_0", 32, 22), 7: ("Q5_1", 32, 24), 8: ("Q8_0", 32, 34), 12: ("Q4_K", 256, 144),
         13: ("Q5_K", 256, 176), 14: ("Q6_K", 256, 210), 30: ("BF16", 1, 2)}


class Reader:
    def __init__(self, buf):
        self.b, self.p = buf, 0

    def take(self, n):
        out = self.b[self.p:self.p + n]
        self.p += n
        return out

    def unpack(self, fmt):
        v = struct.unpack_from(fmt, self.b, self.p)[0]
        self.p += struct.calcsize(fmt)
        return v

    def string(self):
        n = self.unpack("<Q")
        return bytes(self.take(n)).decode("utf-8")

    def value(self, t):
        if t in SCALAR:
            return self.unpack(SCALAR[t])
        if t == STR:
            return self.string()
        if t == ARR:
            et, n = self.unpack("<I"), self.unpack("<Q")
            return [self.value(et) for _ in range(n)]
        raise ValueError(f"unknown GGUF value type {t}")


def read_gguf(path):
    buf = memoryview(open(path, "rb").read())
    r = Reader(buf)
    assert bytes(r.take(4)) == b"GGUF"
    version = r.unpack("<I")
    assert version == 3, version
    n_tensors, n_kv = r.unpack("<Q"), r.unpack("<Q")
    kv = {}
    for _ in range(n_kv):
        k = r.string()
        kv[k] = r.value(r.unpack("<I"))
    infos = []
    for _ in range(n_tensors):
        name = r.string()
        nd = r.unpack("<I")
        dims = [r.unpack("<Q") for _ in range(nd)]
        t, off = r.unpack("<I"), r.unpack("<Q")
        infos.append((name, dims, t, off))
    align = kv.get("general.alignment", 32)
    data0 = (r.p + align - 1) // align * align
    return kv, infos, buf, data0


def f16(b):
    return np.frombuffer(b, dtype="<f2").astype(np.float32)


def dequant(t, raw, n_values):
    """raw: bytes of n_values/values_per_block blocks -> float32 [n_values]"""
    name, bv, bb = TYPES[t]
    nb = n_values // bv
    blk = np.frombuffer(raw, dtype=np.uint8, count=nb * bb).reshape(nb, bb)
    if name == "F32":
        return np.frombuffer(raw, dtype="<f4", count=n_values).copy()
    if name == "F16":
        return np.frombuffer(raw, dtype="<f2", count=n_values).astype(np.float32)
    if name == "BF16":
        return (np.frombuffer(raw, dtype="<u2", count=n_values).astype(np.uint32) << 16).view(np.float32)
    d = blk[:, 0:2].copy().view("<f2").astype(np.float32)                      # every block starts with an f16 scale
    if name == "Q8_0":                                                          # d, int8 x 32
        return (d * blk[:, 2:34].view(np.int8).astype(np.float32)).reshape(-1)
    if name in ("Q4_0", "Q4_1"):                                                # d [, m], 16 bytes of nibbles: low nibbles are values 0..15, high nibbles 16..31
        off = 2 if name == "Q4_0" else 4
        qs = blk[:, off:off + 16]
        lo, hi = (qs & 0xF).astype(np.float32), (qs >> 4).astype(np.float32)
        q = np.concatenate([lo, hi], axis=1)
        if name == "Q4_0":
            return (d * (q - 8.0)).reshape(-1)
        m = blk[:, 2:4].copy().view("<f2").astype(np.float32)
        return (d * q + m).reshape(-1)
    if name == "Q5_0":                                                          # d, qh (u32: 5th bit of each of the 32 values), 16 bytes of nibbles
        qh = blk[:, 2:6].copy().view("<u4").astype(np.uint32)
        qs = blk[:, 6:22]
        j = np.arange(16)
        h0 = ((qh >> j) & 1) << 4
        h1 = ((qh >> (j + 16)) & 1) << 4
        lo = ((qs & 0xF).astype(np.uint32) | h0).astype(np.float32) - 16.0
        hi = ((qs >> 4).astype(np.uint32) | h1).astype(np.float32) - 16.0
        return (d * np.concatenate([lo, hi], axis=1)).reshape(-1)
    if name == "Q4_K":                                                          # d, dmin, 12 bytes of packed 6-bit scales/mins for 8 sub-blocks of 32, 128 bytes of nibbles
        dmin = blk[:, 2:4].copy().view("<f2").astype(np.float32)
        sc = blk[:, 4:16].astype(np.uint32)
        qs = blk[:, 16:144]
        scales = np.zeros((nb, 8), np.float32)
        mins = np.zeros((nb, 8), np.float32)
        for j in range(8):
            if j < 4:
                scales[:, j] = sc[:, j] & 63
                mins[:, j] = sc[:, j + 4] & 63
            else:
                scales[:, j] = (sc[:, j + 4] & 0xF) | ((sc[:, j - 4] >> 6) << 4)
                mins[:, j] = (sc[:, j + 4] >> 4) | ((sc[:, j] >> 6) << 4)
        out = np.zeros((nb, 256), np.float32)
        for g in range(4):                                                      # 4 groups of 64 values: low nibbles -> sub-block 2g, high nibbles -> sub-block 2g+1
            q = qs[:, g * 32:(g + 1) * 32]
            out[:, g * 64:g * 64 + 32] = (d * scales[:, 2 * g:2 * g + 1]) * (q & 0xF) - dmin * mins[:, 2 * g:2 * g + 1]
            out[:, g * 64 + 32:g * 64 + 64] = (d * scales[:, 2 * g + 1:2 * g + 2]) * (q >> 4) - dmin * mins[:, 2 * g + 1:2 * g + 2]
        return out.reshape(-1)
    if name == "Q5_K":                                                          # d, dmin, 12 bytes of packed scales/mins (as Q4_K), qh[32] fifth bits (bit 2g and 2g+1 for the two halves of group g), qs[128] nibbles
        dmin = blk[:, 2:4].copy().view("<f2").astype(np.float32)
        sc = blk[:, 4:16].astype(np.uint32)
        qh = blk[:, 16:48].astype(np.uint32)
        qs = blk[:, 48:176]
        scales = np.zeros((nb, 8), np.float32)
        mins = np.zeros((nb, 8), np.float32)
        for j in range(8):
            if j < 4:
                scales[:, j] = sc[:, j] & 63
                mins[:, j] = sc[:, j + 4] & 63
            else:
                scales[:, j] = (sc[:, j + 4] & 0xF) | ((sc[:, j - 4] >> 6) << 4)
                mins[:, j] = (sc[:, j + 4] >> 4) | ((sc[:, j] >> 6) << 4)
        out = np.zeros((nb, 256), np.float32)
        for g in range(4):
            q = qs[:, g * 32:(g + 1) * 32]
            h1 = (((qh >> (2 * g)) & 1) << 4).astype(np.float32)
            h2 = (((qh >> (2 * g + 1)) & 1) << 4).astype(np.float32)
            out[:, g * 64:g * 64 + 32] = (d * scales[:, 2 * g:2 * g + 1]) * ((q & 0xF).astype(np.float32) + h1) - dmin * mins[:, 2 * g:2 * g + 1]
            out[:, g * 64 + 32:g * 64 + 64] = (d * scales[:, 2 * g + 1:2 * g + 2]) * ((q >> 4).astype(np.float32) + h2) - dmin * mins[:, 2 * g + 1:2 * g + 2]
        return out.reshape(-1)
    if name == "Q6_K":                                                          # ql[128] low 4 bits, qh[64] high 2 bits, scales[16] int8, d at the END of the block
        ql = blk[:, 0:128]
        qh = blk[:, 128:192]
        scales = blk[:, 192:208].view(np.int8).astype(np.float32)
        d = blk[:, 208:210].copy().view("<f2").astype(np.float32)
        out = np.zeros((nb, 256), np.float32)
        for half in range(2):                                                   # two halves of 128 values
            l_ = ql[:, half * 64:(half + 1) * 64]
            h_ = qh[:, half * 32:(half + 1) * 32]
            sc = scales[:, half * 8:(half + 1) * 8]
            l = np.arange(32)
            isc = l // 16
            q1 = ((l_[:, 0:32] & 0xF) | (((h_ >> 0) & 3) << 4)).astype(np.int32) - 32
            q2 = ((l_[:, 32:64] & 0xF) | (((h_ >> 2) & 3) << 4)).astype(np.int32) - 32
            q3 = ((l_[:, 0:32] >> 4) | (((h_ >> 4) & 3) << 4)).astype(np.int32) - 32
            q4 = ((l_[:, 32:64] >> 4) | (((h_ >> 6) & 3) << 4)).astype(np.int32) - 32
            base = half * 128
            out[:, base + 0:base + 32] = d * sc[:, isc + 0] * q1
            out[:, base + 32:base + 64] = d * sc[:, isc + 2] * q2
            out[:, base + 64:base + 96] = d * sc[:, isc + 4] * q3
            out[:, base + 96:base + 128] = d * sc[:, isc + 6] * q4
        return out.reshape(-1)
    raise NotImplementedError(name)


def tensor(kv, infos, buf, data0, name):
    for n, dims, t, off in infos:
        if n == name:
            nvals = int(np.prod(dims))
            nbytes = nvals // TYPES[t][1] * TYPES[t][2]
            raw = bytes(buf[data0 + off:data0 + off + nbytes])
            return dequant(t, raw, nvals).reshape(list(reversed(dims)))       # numpy shape [out, in]
    raise KeyError(name)


def check_golden():
    index = json.load(open(os.path.join(GOLD, "quant_index.json")))
    ok = True
    for name, meta in index.items():
        t = next(k for k, v in TYPES.items() if v[0] == name)
        raw = open(os.path.join(GOLD, f"quant_{name}.in"), "rb").read()
        want = np.fromfile(os.path.join(GOLD, f"quant_{name}.f32"), dtype="<f4")
        got = dequant(t, raw, len(want))
        err = float(np.abs(got - want).max())
        print(f"  dequant {name:5s}: {len(want)} values, max |diff| {err:.2e} {'OK' if err == 0.0 else ('~' if err < 1e-6 else 'MISMATCH')}")
        ok &= err < 1e-6
    return ok


def main():
    print("golden dequantization vectors:")
    ok = check_golden()
    # the whole file: tensor listing and a forward pass through the Llama block using weights read from the F16 and the Q4_K_M files
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import ref_forward_numpy as F
    gold = json.load(open(os.path.join(GOLD, "fwd_smollm2.json")))
    cfg = json.load(open(os.path.join(ROOT, F.MODELS["smollm2"], "config.json")))
    for fn in ("SmolLM2-135M-Instruct-f16.gguf", "SmolLM2-135M-Instruct-Q8_0.gguf", "SmolLM2-135M-Instruct-Q4_K_M.gguf"):
        kv, infos, buf, data0 = read_gguf(os.path.join(ROOT, "bartowski__SmolLM2-135M-Instruct-GGUF", fn))
        print(f"{fn}: arch {kv['general.architecture']}, {len(infos)} tensors, data at {data0}, alignment {kv.get('general.alignment', 32)}")
        w = {"model.embed_tokens.weight": tensor(kv, infos, buf, data0, "token_embd.weight"), "model.norm.weight": tensor(kv, infos, buf, data0, "output_norm.weight")}
        for l in range(cfg["num_hidden_layers"]):
            for ours, hf in (("attn_norm", "input_layernorm"), ("ffn_norm", "post_attention_layernorm")):
                w[f"model.layers.{l}.{hf}.weight"] = tensor(kv, infos, buf, data0, f"blk.{l}.{ours}.weight")
            for ours, hf in (("attn_q", "self_attn.q_proj"), ("attn_k", "self_attn.k_proj"), ("attn_v", "self_attn.v_proj"), ("attn_output", "self_attn.o_proj"),
                             ("ffn_gate", "mlp.gate_proj"), ("ffn_up", "mlp.up_proj"), ("ffn_down", "mlp.down_proj")):
                w[f"model.layers.{l}.{hf}.weight"] = tensor(kv, infos, buf, data0, f"blk.{l}.{ours}.weight")
        # llama.cpp stores Q and K permuted so that adjacent-pair RoPE equals Hugging Face's rotate-half RoPE: undo the permutation to reuse the HF-convention forward
        nh, nkv, hd = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["hidden_size"] // cfg["num_attention_heads"]

        def unpermute(wt, heads):
            return wt.reshape(heads, hd // 2, 2, wt.shape[1]).swapaxes(1, 2).reshape(wt.shape)
        for l in range(cfg["num_hidden_layers"]):
            p = f"model.layers.{l}.self_attn."
            w[p + "q_proj.weight"] = unpermute(w[p + "q_proj.weight"], nh)
            w[p + "k_proj.weight"] = unpermute(w[p + "k_proj.weight"], nkv)
        worst = 0.0
        flips = 0
        for k, pr in enumerate(gold["prompts"]):
            logits, _, _ = F.forward(cfg, w, pr["ids"], False)
            ref = np.fromfile(os.path.join(GOLD, pr["last_logits_file"]), dtype="<f4")
            worst = max(worst, float(np.abs(logits[-1] - ref).max()))
            flips += int(np.argmax(logits[-1]) != pr["top20"][0][0])
        print(f"   forward from this file: worst last-position logit diff vs HF {worst:.3f}, top-1 differing in {flips}/{len(gold['prompts'])} prompts")
    print("ALL GOLDEN VECTORS EXACT" if ok else "SOME GOLDEN VECTORS DIFFER")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
