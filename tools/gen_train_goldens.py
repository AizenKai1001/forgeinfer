"""Golden vectors for the trainer: a tiny Llama-style decoder (optionally with Qwen3-style QK-norm and the Forge-1 structure options) in PyTorch (float32, CPU, an ORACLE only) with fixed random weights and one batch.
Writes tests/golden/train_tiny/<variant>/: the weights (f32), the tokens, the loss, the gradient of every parameter, and the weights/loss after a few AdamW steps, so that the Rust/CUDA trainer can be checked.
    ~/forgeinfer/venv/bin/python tools/gen_train_goldens.py
Model (matches forge-train): RMSNorm (eps 1e-5), GQA attention with RoPE (rotate half, theta 10000), SwiGLU MLP, tied embedding / output head, no biases, mean cross-entropy over all positions.
Forge-1 options (each a flag of the trainer, see TrainConfig):
  untied     separate output head `head` [V, E]
  relu2      MLP = relu(h Wup^T)^2 Wdown^T (no gate matrix)
  softcap    logits = cap * tanh(z / cap)
  lambdas    the input of layer l is a_l * (output of layer l-1) + b_l * x0 (x0 = embedding output; layer 0: x0 itself), (a, b, c) = lam[l]
  skips      layers l >= L - L//2 also get + c_l * (output of layer L-1-l)
  ve_layers  value embeddings: v = lv * v + lve * ve_l[token]  ((lv, lve) = velam[k] for the k-th listed layer)
  ngram      hashed n-gram memory: x0 = emb[token] + (sum over the orders n of ng_n[ngram_index(last n tokens)]) ngw^T   (ngram_index below mirrors ngram_hash in train_f1.cu)
  attn_gate  gated attention (K channels): the output of head h is multiplied by sigmoid(h1[..., :K] . agate[l, h]) before the output projection (h1 = the normalised attention input)
  doc_eos    document masking: an end-of-text input token (this id) opens a new document; a position attends only to its own document (the generator plants end-of-text tokens in the batch)
  peri_norm  Peri-LN: x = x + rms(sublayer output, pnorm[l, 0 or 1]) instead of x + sublayer output;  emb_norm: x0 = rms(embedding output, enorm)
    python tools/gen_train_goldens.py agate agate_full      writes only the named variants (the others keep their committed files)
"""
import json
import os
import re

import numpy as np
import torch

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests", "golden", "train_tiny")
EPS = 1e-5


class Cfg:
    def __init__(self, name, qk_norm, n_layer=2, untied=False, relu2=False, softcap=0.0, lambdas=False, skips=False, ve_layers=(), small=False, ngram=(), ngram_bits=7, ngram_dim=32, attn_gate=0, peri_norm=False, emb_norm=False, doc_eos=None, T=64):
        self.name = name
        # the first two variants keep the original size; the Forge-1 variants are smaller (the golden files are committed)
        self.n_layer, self.n_embd, self.n_head, self.n_kv, self.hd, self.n_ff = n_layer, (64 if small else 128), 2, 1, 64, (128 if small else 256)
        self.vocab, self.T, self.B = (256 if small else 512), T, 2
        self.theta = 10000.0
        self.qk_norm = qk_norm
        self.untied, self.relu2, self.softcap, self.lambdas, self.skips = untied, relu2, softcap, lambdas, skips
        self.ve_layers = list(ve_layers)
        self.ngram, self.ngram_bits, self.ngram_dim = list(ngram), (ngram_bits if ngram else 0), (ngram_dim if ngram else 0)
        self.attn_gate = attn_gate
        self.peri_norm, self.emb_norm = peri_norm, emb_norm
        self.doc_eos = doc_eos

    @property
    def mixing(self):
        return self.lambdas or self.skips


def ngram_index(row_tokens, p, order, bits):
    """Row of the n-gram table for position p of a window: a hash of the last `order` tokens (tokens before the start of the window count as none)."""
    M = (1 << 64) - 1
    h = (0x9E3779B97F4A7C15 * order) & M
    for k in range(order):
        t = int(row_tokens[p - k]) + 1 if p - k >= 0 else 0
        h = ((h ^ t) * 0xFF51AFD7ED558CCD) & M
        h ^= h >> 33
    h = (h * 0xC4CEB9FE1A85EC53) & M
    h ^= h >> 33
    return h >> (64 - bits)


def doc_mask(tokens, eos):
    """[B, T, T] True where query i must not see key j because j lies before the start of i's document (an end-of-text input token opens a new document)."""
    B, T = tokens.shape
    ds = torch.zeros(B, T, dtype=torch.long)
    for b in range(B):
        start = 0
        for i in range(T):
            if int(tokens[b, i]) == eos:
                start = i
            ds[b, i] = start
    j = torch.arange(T)
    return j[None, None, :] < ds[:, :, None]


def rms(x, w):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * w


def rope(x, cos, sin):  # x [B, T, H, hd]
    h = x.shape[-1] // 2
    x1, x2 = x[..., :h], x[..., h:]
    c, s = cos[None, :, None, :], sin[None, :, None, :]
    return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1)


def forward(cfg, P, tokens):
    B, T = tokens.shape
    pos = torch.arange(T, dtype=torch.float64)
    inv = cfg.theta ** (-torch.arange(0, cfg.hd, 2, dtype=torch.float64) / cfg.hd)
    ang = pos[:, None] * inv[None, :]
    cos, sin = torch.cos(ang).float(), torch.sin(ang).float()
    x0 = P["emb"][tokens]
    if cfg.ngram:
        toks = tokens.tolist()
        mem = 0
        for o in cfg.ngram:
            idx = torch.tensor([[ngram_index(toks[b], p, o, cfg.ngram_bits) for p in range(T)] for b in range(B)])
            mem = mem + P[f"ng{o}"][idx]
        x0 = x0 + mem @ P["ngw"].T
    if cfg.emb_norm:
        x0 = rms(x0, P["enorm"])
    outs = []
    n_enc = cfg.n_layer // 2
    for l in range(cfg.n_layer):
        xprev = x0 if l == 0 else outs[l - 1]
        if cfg.mixing:
            a_, b_, c_ = P["lam"][l, 0], P["lam"][l, 1], P["lam"][l, 2]
            x = (a_ if cfg.lambdas else 1.0) * xprev
            if cfg.lambdas:
                x = x + b_ * x0
            if cfg.skips and l >= cfg.n_layer - n_enc:
                x = x + c_ * outs[cfg.n_layer - 1 - l]
        else:
            x = xprev
        h = rms(x, P[f"l{l}.attn_norm"])
        q = (h @ P[f"l{l}.wq"].T).view(B, T, cfg.n_head, cfg.hd)
        k = (h @ P[f"l{l}.wk"].T).view(B, T, cfg.n_kv, cfg.hd)
        v = (h @ P[f"l{l}.wv"].T).view(B, T, cfg.n_kv, cfg.hd)
        if cfg.qk_norm:
            q = rms(q, P[f"l{l}.q_norm"])
            k = rms(k, P[f"l{l}.k_norm"])
        q, k = rope(q, cos, sin), rope(k, cos, sin)
        if l in cfg.ve_layers:
            lv, lve = P["velam"][cfg.ve_layers.index(l), 0], P["velam"][cfg.ve_layers.index(l), 1]
            v = lv * v + lve * P[f"ve{l}"][tokens].view(B, T, cfg.n_kv, cfg.hd)
        rep = cfg.n_head // cfg.n_kv
        k = k.repeat_interleave(rep, dim=2)
        v = v.repeat_interleave(rep, dim=2)
        att = torch.einsum("bthd,bshd->bhts", q, k) / np.sqrt(cfg.hd)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool), 1)
        if cfg.doc_eos is not None:
            att = att.masked_fill(doc_mask(tokens, cfg.doc_eos)[:, None], float("-inf"))
        att = att.masked_fill(mask, float("-inf")).softmax(-1)
        o = torch.einsum("bhts,bshd->bthd", att, v)
        if cfg.attn_gate:
            o = o * torch.sigmoid(h[..., :cfg.attn_gate] @ P["agate"][l].T)[..., None]
        o = o.reshape(B, T, cfg.n_head * cfg.hd)
        y = o @ P[f"l{l}.wo"].T
        x = x + (rms(y, P["pnorm"][l, 0]) if cfg.peri_norm else y)
        h2 = rms(x, P[f"l{l}.ffn_norm"])
        if cfg.relu2:
            act = torch.relu(h2 @ P[f"l{l}.w_up"].T).pow(2)
        else:
            g = h2 @ P[f"l{l}.w_gate"].T
            u = h2 @ P[f"l{l}.w_up"].T
            act = torch.nn.functional.silu(g) * u
        y = act @ P[f"l{l}.w_down"].T
        x = x + (rms(y, P["pnorm"][l, 1]) if cfg.peri_norm else y)
        outs.append(x)
    hf = rms(outs[-1], P["out_norm"])
    logits = hf @ (P["head"] if cfg.untied else P["emb"]).T
    if cfg.softcap > 0:
        logits = cfg.softcap * torch.tanh(logits / cfg.softcap)
    return logits


def make(cfg, seed):
    g = torch.Generator().manual_seed(seed)
    r = lambda *s, std=0.05: torch.randn(*s, generator=g) * std
    P = {"emb": r(cfg.vocab, cfg.n_embd, std=0.1), "out_norm": 1.0 + r(cfg.n_embd, std=0.1)}
    qd, kvd = cfg.n_head * cfg.hd, cfg.n_kv * cfg.hd
    for l in range(cfg.n_layer):
        P[f"l{l}.attn_norm"] = 1.0 + r(cfg.n_embd, std=0.1)
        P[f"l{l}.wq"] = r(qd, cfg.n_embd)
        P[f"l{l}.wk"] = r(kvd, cfg.n_embd)
        P[f"l{l}.wv"] = r(kvd, cfg.n_embd)
        P[f"l{l}.wo"] = r(cfg.n_embd, qd)
        if cfg.qk_norm:
            P[f"l{l}.q_norm"] = 1.0 + r(cfg.hd, std=0.1)
            P[f"l{l}.k_norm"] = 1.0 + r(cfg.hd, std=0.1)
        P[f"l{l}.ffn_norm"] = 1.0 + r(cfg.n_embd, std=0.1)
        if not cfg.relu2:
            P[f"l{l}.w_gate"] = r(cfg.n_ff, cfg.n_embd)
        P[f"l{l}.w_up"] = r(cfg.n_ff, cfg.n_embd)
        P[f"l{l}.w_down"] = r(cfg.n_embd, cfg.n_ff)
    # Forge-1 parameters are drawn last, so that the variants without them keep their old random numbers
    if cfg.untied:
        P["head"] = r(cfg.vocab, cfg.n_embd, std=0.1)
    for l in cfg.ve_layers:
        P[f"ve{l}"] = r(cfg.vocab, kvd, std=0.1)
    for o in cfg.ngram:
        P[f"ng{o}"] = r(1 << cfg.ngram_bits, cfg.ngram_dim, std=0.1)
    if cfg.ngram:
        P["ngw"] = r(cfg.n_embd, cfg.ngram_dim, std=0.1)
    if cfg.mixing:
        lam = torch.zeros(cfg.n_layer, 3)
        lam[:, 0] = 1.0 + r(cfg.n_layer, std=0.1)
        lam[:, 1] = r(cfg.n_layer, std=0.3)
        lam[:, 2] = r(cfg.n_layer, std=0.3)
        P["lam"] = lam
    if cfg.ve_layers:
        P["velam"] = 1.0 + r(len(cfg.ve_layers), 2, std=0.2)
    if cfg.attn_gate:
        P["agate"] = r(cfg.n_layer, cfg.n_head, cfg.attn_gate, std=0.5)
    if cfg.peri_norm:
        P["pnorm"] = 1.0 + r(cfg.n_layer, 2, cfg.n_embd, std=0.1)
    if cfg.emb_norm:
        P["enorm"] = 1.0 + r(cfg.n_embd, std=0.1)
    return {k: v.clone().requires_grad_(True) for k, v in P.items()}


def decays(name):
    """Weight decay applies to the embedding and the layer matrices (the first block of the flat parameter layout), not to the head, the tables of value embeddings or the scalars."""
    return name == "emb" or re.fullmatch(r"l\d+\.(wq|wk|wv|wo|w_gate|w_up|w_down)", name) is not None


VARIANTS = [
    Cfg("plain", False),
    Cfg("qknorm", True),
    Cfg("untied", True, n_layer=2, untied=True, small=True),
    Cfg("relu2", True, n_layer=2, relu2=True, small=True),
    Cfg("softcap", True, n_layer=2, softcap=1.5, small=True),
    Cfg("lambdas", True, n_layer=4, lambdas=True, small=True),
    Cfg("skips", True, n_layer=4, skips=True, small=True),
    Cfg("valemb", True, n_layer=4, ve_layers=(0, 3), small=True),
    Cfg("ngram", True, n_layer=2, ngram=(2, 3), small=True),
    Cfg("forge1", True, n_layer=4, untied=True, relu2=True, softcap=1.5, lambdas=True, skips=True, ve_layers=(0, 2), ngram=(2, 3), small=True),
    # recipe-opt: gated attention, the sparse form (12 channels) and the full head-wise form (all 64 channels) with the Forge-1 recipe's structure
    Cfg("agate", True, n_layer=2, attn_gate=12, small=True),
    Cfg("agate_full", True, n_layer=2, attn_gate=64, relu2=True, lambdas=True, ve_layers=(0, 1), small=True),
    # recipe-opt: Peri-LN output norms alone, and with the embedding norm, the n-gram memory and the Forge-1 recipe's structure
    Cfg("peri", True, n_layer=2, peri_norm=True, small=True),
    Cfg("peri_emb", True, n_layer=2, peri_norm=True, emb_norm=True, relu2=True, lambdas=True, ve_layers=(0, 1), ngram=(2,), small=True),
    # recipe-opt: document masking (end-of-text id 0, planted at fixed positions of the batch) with and without the recipe's structure
    Cfg("docmask", True, n_layer=2, doc_eos=0, small=True),
    Cfg("docmask_recipe", True, n_layer=2, doc_eos=0, relu2=True, lambdas=True, ve_layers=(0, 1), small=True),
    # four 64-position tiles: documents that span tiles, start inside them and start exactly at a tile boundary (the masked kernels skip whole tiles)
    Cfg("docmask_long", True, n_layer=2, doc_eos=0, small=True, T=256),
]


def main():
    import sys
    wanted = set(sys.argv[1:])
    for cfg in VARIANTS:
        if wanted and cfg.name not in wanted:
            continue
        d = os.path.join(OUT, cfg.name)
        os.makedirs(d, exist_ok=True)
        P = make(cfg, 1234)
        g = torch.Generator().manual_seed(99)
        tokens = torch.randint(0, cfg.vocab, (cfg.B, cfg.T + 1), generator=g)
        if cfg.doc_eos is not None:
            # documents of different lengths, two end-of-text tokens in a row, one at the start of a row, and the window boundaries
            tokens[tokens == cfg.doc_eos] = 1
            plant = [(0, (10, 30, 31, 50)), (1, (0, 40))] if cfg.T == 64 else [(0, (10, 70, 71, 130, 200, 255)), (1, (0, 64, 191))]
            for b, ps in plant:
                for p_ in ps:
                    tokens[b, p_] = cfg.doc_eos
        x_in, y = tokens[:, :-1], tokens[:, 1:]
        logits = forward(cfg, P, x_in)
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, cfg.vocab), y.reshape(-1))
        loss.backward()
        config = {k: getattr(cfg, k) for k in ("n_layer", "n_embd", "n_head", "n_kv", "hd", "n_ff", "vocab", "T", "B", "theta", "qk_norm")}
        config.update({k: getattr(cfg, k) for k in ("untied", "relu2", "softcap", "lambdas", "skips", "ve_layers", "ngram", "ngram_bits", "ngram_dim", "attn_gate", "peri_norm", "emb_norm", "doc_eos")})
        manifest = {"config": config, "loss": float(loss), "params": []}
        for name, p in P.items():
            grad = p.grad if p.grad is not None else torch.zeros_like(p)
            p.detach().numpy().astype("<f4").tofile(os.path.join(d, f"{name}.w.f32"))
            grad.numpy().astype("<f4").tofile(os.path.join(d, f"{name}.g.f32"))
            manifest["params"].append({"name": name, "shape": list(p.shape)})
        tokens.numpy().astype("<u4").tofile(os.path.join(d, "tokens.u32"))
        logits.detach().numpy().astype("<f4").tofile(os.path.join(d, "logits.f32"))
        # 5 AdamW steps on the same batch: lr 1e-2, betas (0.9, 0.95), eps 1e-8, weight decay 0.1 on the embedding and the layer matrices only, no clipping
        P2 = {k: v.detach().clone().requires_grad_(True) for k, v in P.items()}
        decay = [p for k, p in P2.items() if decays(k)]
        nodecay = [p for k, p in P2.items() if not decays(k)]
        opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1}, {"params": nodecay, "weight_decay": 0.0}], lr=1e-2, betas=(0.9, 0.95), eps=1e-8)
        losses = []
        for step in range(5):
            opt.zero_grad()
            lg = forward(cfg, P2, x_in)
            l2 = torch.nn.functional.cross_entropy(lg.reshape(-1, cfg.vocab), y.reshape(-1))
            l2.backward()
            losses.append(float(l2))
            opt.step()
        manifest["adamw_losses_before_each_step"] = losses
        for name, p in P2.items():
            p.detach().numpy().astype("<f4").tofile(os.path.join(d, f"{name}.w5.f32"))
        manifest["adamw"] = {"lr": 1e-2, "betas": [0.9, 0.95], "eps": 1e-8, "weight_decay": 0.1, "steps": 5}
        with open(os.path.join(d, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=1)
        print(f"{cfg.name}: loss {float(loss):.6f}, adamw losses {[round(v, 5) for v in losses]}, {len(P)} tensors")


if __name__ == "__main__":
    main()
