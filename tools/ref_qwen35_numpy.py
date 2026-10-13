"""An executable SPECIFICATION of the Qwen3.5 text decoder (hybrid: 18 Gated DeltaNet layers + 6 gated full-attention layers in the 0.8B model, pattern 3:1), in float32 numpy with our own
safetensors reader, checked against the recorded Hugging Face goldens. The Rust/CUDA support for the model family the user wants to run (Qwen3.5-9B) is a port of this file.
    ~/forgeinfer/venv/bin/python tools/ref_qwen35_numpy.py          (homelab; reads ~/forgeinfer/oracle/models/Qwen__Qwen3.5-0.8B)

What differs from the Llama/Qwen3 block (all verified below):
  * RMSNorm is ZERO-CENTERED: y = x * rsqrt(mean(x^2) + eps) * (1 + weight)   (stored weights are around 0, not around 1); the gated norm inside the DeltaNet layer is the plain kind.
  * full-attention layer: q_proj is 2x wide (per head: 256 query values then 256 gate values), q/k RMSNorm per head, PARTIAL RoPE (only the first 25 % = 64 of 256 head dims rotate, rotate-half
    pairs inside that slice, theta 1e7), GQA 8/2, softmax scale head_dim^-0.5, output multiplied by sigmoid(gate) before o_proj.
  * linear-attention layer (Gated DeltaNet): qkv = in_proj_qkv(h) -> causal depthwise conv1d (kernel 4, no bias) + SiLU over the 6144 channels -> split into q (16x128), k (16x128), v (16x128);
    beta = sigmoid(in_proj_b(h)); g = -exp(A_log) * softplus(in_proj_a(h) + dt_bias) (log decay per value head); q, k L2-normalized, q scaled by 1/sqrt(d_k);
    state S[d_k, d_v] per head:  S <- S * exp(g);  S <- S + k (x) (beta * (v - S^T k));  o = S^T q;
    output: RMSNorm(o) * weight (plain) * silu(z) with z = in_proj_z(h), then out_proj.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ref_forward_numpy as F

ROOT = os.path.expanduser("~/forgeinfer/oracle/models/Qwen__Qwen3.5-0.8B")
GOLD = F.GOLD


def silu(x):
    return x / (1.0 + np.exp(-x))


def softplus(x):
    return np.where(x > 20, x, np.log1p(np.exp(np.minimum(x, 20))))


def rms0(x, w, eps):                                       # zero-centered RMSNorm
    return x * (1.0 / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)) * (1.0 + w)


def rms_plain(x, w, eps):
    return x * (1.0 / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)) * w


def l2norm(x, eps=1e-6):
    return x * (1.0 / np.sqrt((x * x).sum(-1, keepdims=True) + eps))


def load():
    cfg = json.load(open(os.path.join(ROOT, "config.json")))["text_config"]
    w = F.read_safetensors(os.path.join(ROOT, "model.safetensors-00001-of-00001.safetensors"))
    return cfg, {k[len("model.language_model."):]: v for k, v in w.items() if k.startswith("model.language_model.")}


def gated_delta_layer(h, p, cfg):
    T = h.shape[0]
    nk, nv, dk, dv = cfg["linear_num_key_heads"], cfg["linear_num_value_heads"], cfg["linear_key_head_dim"], cfg["linear_value_head_dim"]
    K = cfg["linear_conv_kernel_dim"]
    key_dim, val_dim = nk * dk, nv * dv
    mixed = h @ p["in_proj_qkv.weight"].T                                      # [T, 2*key_dim + val_dim]
    z = (h @ p["in_proj_z.weight"].T).reshape(T, nv, dv)
    b = h @ p["in_proj_b.weight"].T
    a = h @ p["in_proj_a.weight"].T
    cw = p["conv1d.weight"][:, 0, :]                                           # [channels, K]
    padded = np.concatenate([np.zeros((K - 1, mixed.shape[1]), np.float32), mixed], axis=0)
    conv = sum(cw[:, j][None, :] * padded[j:j + T] for j in range(K))           # causal depthwise convolution
    conv = silu(conv)
    q = conv[:, :key_dim].reshape(T, nk, dk)
    k = conv[:, key_dim:2 * key_dim].reshape(T, nk, dk)
    v = conv[:, 2 * key_dim:].reshape(T, nv, dv)
    beta = 1.0 / (1.0 + np.exp(-b))                                            # [T, nv]
    g = -np.exp(p["A_log"])[None, :] * softplus(a + p["dt_bias"][None, :])      # [T, nv] log decay
    rep = nv // nk
    if rep > 1:
        q, k = np.repeat(q, rep, axis=1), np.repeat(k, rep, axis=1)
    q, k = l2norm(q), l2norm(k)
    q = q / np.sqrt(dk)
    S = np.zeros((nv, dk, dv), np.float32)
    out = np.zeros((T, nv, dv), np.float32)
    for t in range(T):
        S = S * np.exp(g[t])[:, None, None]
        kv = np.einsum("hkv,hk->hv", S, k[t])                                  # what the state currently predicts for key k_t
        delta = (v[t] - kv) * beta[t][:, None]
        S = S + k[t][:, :, None] * delta[:, None, :]
        out[t] = np.einsum("hkv,hk->hv", S, q[t])
    out = rms_plain(out, p["norm.weight"], cfg["rms_norm_eps"]) * silu(z)       # gated RMSNorm per head
    return out.reshape(T, val_dim) @ p["out_proj.weight"].T


def full_attention_layer(h, p, cfg):
    T = h.shape[0]
    nh, nkv, hd = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]
    eps = cfg["rms_norm_eps"]
    qg = (h @ p["q_proj.weight"].T).reshape(T, nh, 2 * hd)
    q, gate = qg[..., :hd], qg[..., hd:].reshape(T, nh * hd)
    q = rms0(q, p["q_norm.weight"], eps)
    k = rms0((h @ p["k_proj.weight"].T).reshape(T, nkv, hd), p["k_norm.weight"], eps)
    v = (h @ p["v_proj.weight"].T).reshape(T, nkv, hd)
    rd = int(hd * cfg["rope_parameters"]["partial_rotary_factor"])
    inv = 1.0 / (cfg["rope_parameters"]["rope_theta"] ** (np.arange(0, rd, 2, dtype=np.float32) / rd))
    ang = np.arange(T)[:, None].astype(np.float32) * inv[None, :]
    cos, sin = np.cos(ang), np.sin(ang)

    def rope(x):
        r, rest = x[..., :rd], x[..., rd:]
        return np.concatenate([F.apply_rope(r, cos, sin), rest], axis=-1)
    q, k = rope(q), rope(k)
    rep = nh // nkv
    k, v = np.repeat(k, rep, axis=1), np.repeat(v, rep, axis=1)
    s = np.einsum("tnd,snd->nts", q, k) / np.sqrt(hd) + np.triu(np.full((T, T), -np.inf, np.float32), 1)[None]
    s = s - s.max(-1, keepdims=True)
    pr = np.exp(s)
    pr /= pr.sum(-1, keepdims=True)
    o = np.einsum("nts,snd->tnd", pr, v).reshape(T, nh * hd) * (1.0 / (1.0 + np.exp(-gate)))
    return o @ p["o_proj.weight"].T


def forward(cfg, w, ids):
    eps = cfg["rms_norm_eps"]
    x = w["embed_tokens.weight"][ids]
    hidden = [x.copy()]
    for l in range(cfg["num_hidden_layers"]):
        p = {k[len(f"layers.{l}."):]: v for k, v in w.items() if k.startswith(f"layers.{l}.")}
        h = rms0(x, p["input_layernorm.weight"], eps)
        if cfg["layer_types"][l] == "linear_attention":
            mix = gated_delta_layer(h, {k[len("linear_attn."):]: v for k, v in p.items() if k.startswith("linear_attn.")}, cfg)
        else:
            mix = full_attention_layer(h, {k[len("self_attn."):]: v for k, v in p.items() if k.startswith("self_attn.")}, cfg)
        x = x + mix
        h = rms0(x, p["post_attention_layernorm.weight"], eps)
        x = x + (silu(h @ p["mlp.gate_proj.weight"].T) * (h @ p["mlp.up_proj.weight"].T)) @ p["mlp.down_proj.weight"].T
        hidden.append(x.copy())
    final = rms0(x, w["norm.weight"], eps)
    return final @ w["embed_tokens.weight"].T, hidden, final


def main():
    cfg, w = load()
    gold = json.load(open(os.path.join(GOLD, "fwd_qwen35.json")))
    worst, flips = 0.0, 0
    for k, pr in enumerate(gold["prompts"]):
        logits, hidden, final = forward(cfg, w, pr["ids"])
        ref = np.fromfile(os.path.join(GOLD, pr["last_logits_file"]), dtype="<f4")
        d = float(np.abs(logits[-1] - ref).max())
        worst = max(worst, d)
        same = int(np.argmax(logits[-1])) == pr["top20"][0][0]
        flips += 0 if same else 1
        print(f"prompt {k}: {len(pr['ids']):3d} tokens, max |logit diff| {d:.2e}, top-1 {'same' if same else 'DIFFERENT'}")
        if k == 0:
            hs = np.fromfile(os.path.join(GOLD, "fwd_qwen35_p0_hidden.f32"), dtype="<f4").reshape(cfg["num_hidden_layers"] + 1, len(pr["ids"]), cfg["hidden_size"])
            diffs = [float(np.abs(hidden[i] - hs[i]).max()) for i in range(cfg["num_hidden_layers"])]
            print("   hidden-state max diff: layer1 %.1e, layer4 %.1e, layer24 %.1e, worst %.1e; final %.1e" % (diffs[0], diffs[3], diffs[-1], max(diffs), np.abs(final - hs[-1]).max()))
    print(f"worst last-position logit difference {worst:.2e}; top-1 differs in {flips} prompts")
    return 0 if flips == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
