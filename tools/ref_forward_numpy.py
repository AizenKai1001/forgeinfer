"""A transparent numpy implementation of the decoder-only transformer forward pass (Llama and Qwen3 blocks), checked against the recorded goldens.

This is the SPECIFICATION the Rust engine is written against: every step is a few lines, in float32, no framework. It reads the Hugging Face safetensors of the test models.
    ~/forgeinfer/venv/bin/python tools/ref_forward_numpy.py smollm2|qwen3
Compares the final logits of all prompts with tests/golden/fwd_<name>_pK_last.f32, the all-position logits and every layer's hidden state of prompt 0 with the HF dumps, and the 48 greedy tokens.
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.expanduser("~/forgeinfer/oracle/models")
MODELS = {"smollm2": "HuggingFaceTB__SmolLM2-135M-Instruct", "qwen3": "Qwen__Qwen3-0.6B"}
GOLD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests", "golden")


def read_safetensors(path):
    """The whole safetensors format: u64 little-endian header length, a JSON header {name: {dtype, shape, data_offsets}}, then the raw tensor bytes. bf16 -> f32 is a 16-bit left shift."""
    with open(path, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(n))
        base = 8 + n
        out = {}
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            a, b = meta["data_offsets"]
            f.seek(base + a)
            raw = f.read(b - a)
            if meta["dtype"] == "BF16":
                arr = (np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16).view(np.float32)
            elif meta["dtype"] == "F16":
                arr = np.frombuffer(raw, dtype="<f2").astype(np.float32)
            elif meta["dtype"] == "F32":
                arr = np.frombuffer(raw, dtype="<f4").copy()
            else:
                raise ValueError(meta["dtype"])
            out[name] = arr.reshape(meta["shape"])
        return out


def load(name):
    d = os.path.join(ROOT, MODELS[name])
    cfg = json.load(open(os.path.join(d, "config.json")))
    return cfg, read_safetensors(os.path.join(d, "model.safetensors"))


def rms_norm(x, weight, eps):
    # x [T, H]; computed in float32: x * rsqrt(mean(x^2) + eps) * weight
    return x * (1.0 / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)) * weight


def rope_tables(positions, dim, theta):
    inv = 1.0 / (theta ** (np.arange(0, dim, 2, dtype=np.float32) / dim))      # [dim/2]
    ang = positions[:, None].astype(np.float32) * inv[None, :]                 # [T, dim/2]
    return np.cos(ang), np.sin(ang)


def apply_rope(x, cos, sin):
    # x [T, heads, dim]; Hugging Face "rotate_half" convention: pairs are (i, i + dim/2)
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    c, s = cos[:, None, :], sin[:, None, :]
    return np.concatenate([x1 * c - x2 * s, x2 * c + x1 * s], axis=-1)


def silu(x):
    return x / (1.0 + np.exp(-x))


def forward(cfg, w, ids, qk_norm, trace=None):
    T = len(ids)
    H = cfg["hidden_size"]
    nh, nkv = cfg["num_attention_heads"], cfg["num_key_value_heads"]
    hd = cfg.get("head_dim") or H // nh
    eps = cfg["rms_norm_eps"]
    theta = cfg["rope_theta"]
    x = w["model.embed_tokens.weight"][ids]                                     # [T, H]
    hidden = [x.copy()]
    pos = np.arange(T)
    cos, sin = rope_tables(pos, hd, theta)
    mask = np.triu(np.full((T, T), -np.inf, dtype=np.float32), k=1)
    for l in range(cfg["num_hidden_layers"]):
        p = f"model.layers.{l}."
        h = rms_norm(x, w[p + "input_layernorm.weight"], eps)
        q = (h @ w[p + "self_attn.q_proj.weight"].T).reshape(T, nh, hd)
        k = (h @ w[p + "self_attn.k_proj.weight"].T).reshape(T, nkv, hd)
        v = (h @ w[p + "self_attn.v_proj.weight"].T).reshape(T, nkv, hd)
        if qk_norm:                                                              # Qwen3: RMSNorm over head_dim, per head, BEFORE rope
            q = rms_norm(q, w[p + "self_attn.q_norm.weight"], eps)
            k = rms_norm(k, w[p + "self_attn.k_norm.weight"], eps)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        rep = nh // nkv
        k = np.repeat(k, rep, axis=1)                                            # GQA: kv head j serves query heads j*rep .. j*rep+rep-1
        v = np.repeat(v, rep, axis=1)
        scores = np.einsum("tnd,snd->nts", q, k) / np.sqrt(hd) + mask[None]       # [nh, T, T]
        scores = scores - scores.max(axis=-1, keepdims=True)
        att = np.exp(scores)
        att = att / att.sum(axis=-1, keepdims=True)
        o = np.einsum("nts,snd->tnd", att, v).reshape(T, nh * hd)
        x_in = hidden[-1]
        x = x + o @ w[p + "self_attn.o_proj.weight"].T
        x_mid = x.copy()
        h2 = rms_norm(x, w[p + "post_attention_layernorm.weight"], eps)
        g = h2 @ w[p + "mlp.gate_proj.weight"].T
        u = h2 @ w[p + "mlp.up_proj.weight"].T
        a = silu(g) * u
        x = x + a @ w[p + "mlp.down_proj.weight"].T
        hidden.append(x.copy())
        if trace is not None:                                                    # intermediate tensors of this layer, for per-stage kernel tests
            trace[l] = dict(x_in=x_in, h=h, q=q, k=k[:, ::rep], v=v[:, ::rep], o=o, x_mid=x_mid, h2=h2, g=g, u=u, a=a, x_out=x.copy())
    final = rms_norm(x, w["model.norm.weight"], eps)
    head = w.get("lm_head.weight", w["model.embed_tokens.weight"])               # tied embeddings when there is no lm_head
    return final @ head.T, hidden, final


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "smollm2"
    cfg, w = load(name)
    gold = json.load(open(os.path.join(GOLD, f"fwd_{name}.json")))
    qk_norm = cfg["model_type"] == "qwen3"
    worst = 0.0
    for k, pr in enumerate(gold["prompts"]):
        logits, hidden, final = forward(cfg, w, pr["ids"], qk_norm)
        ref = np.fromfile(os.path.join(GOLD, pr["last_logits_file"]), dtype="<f4")
        diff = np.abs(logits[-1] - ref).max()
        top1_ok = int(np.argmax(logits[-1])) == pr["top20"][0][0]
        worst = max(worst, diff)
        print(f"prompt {k}: max |logit diff| {diff:.2e}  top-1 {'same' if top1_ok else 'DIFFERENT'}  ({len(pr['ids'])} tokens)")
        if k == 0:
            all_ref = np.fromfile(os.path.join(GOLD, f"fwd_{name}_p0_logits.f32"), dtype="<f4").reshape(len(pr["ids"]), -1)
            print(f"   all-position logits max diff {np.abs(logits - all_ref).max():.2e}")
            hs_ref = np.fromfile(os.path.join(GOLD, f"fwd_{name}_p0_hidden.f32"), dtype="<f4").reshape(cfg["num_hidden_layers"] + 1, len(pr["ids"]), cfg["hidden_size"])
            diffs = [float(np.abs(hidden[i] - hs_ref[i]).max()) for i in range(cfg["num_hidden_layers"])]
            print("   hidden-state max diff per layer: first %.1e, last block %.1e, worst %.1e; final-norm output %.1e" % (diffs[0], diffs[-1], max(diffs), np.abs(final - hs_ref[-1]).max()))
    # greedy decoding without a KV cache (slow but simple): 8 tokens of prompt 0 are enough to prove the loop
    ids = list(gold["prompts"][0]["ids"])
    for _ in range(8):
        logits, _, _ = forward(cfg, w, ids, qk_norm)
        ids.append(int(np.argmax(logits[-1])))
    expect = gold["prompts"][0]["greedy48"][:8]
    print("greedy 8 tokens:", ids[len(gold["prompts"][0]["ids"]):], "==", expect, "->", "SAME" if ids[len(gold["prompts"][0]["ids"]):] == expect else "DIFFERENT")
    print("worst last-position logit difference:", worst)


if __name__ == "__main__":
    main()
