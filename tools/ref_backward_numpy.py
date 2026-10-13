"""An executable SPECIFICATION of the training math: forward AND hand-derived backward of the decoder-only transformer block (RMSNorm, GQA attention with RoPE and optional QK-norm, SwiGLU,
tied embeddings, cross-entropy), in float64 numpy, verified against central finite differences on a tiny model. The CUDA training kernels are written against these formulas.
    python3 tools/ref_backward_numpy.py [seed]
Forward tensors are kept in a cache; `backward` walks the layers in reverse. Shapes: T tokens, H hidden, nh query heads, nkv key/value heads, hd head dim, F FFN width, V vocabulary.
"""
import sys

import numpy as np

EPS = 1e-6


def rms_fwd(x, w):
    r = 1.0 / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + EPS)
    return x * r * w, (x, w, r)


def rms_bwd(dy, cache):
    x, w, r = cache
    g = dy * w
    H = x.shape[-1]
    dx = r * g - (r ** 3 / H) * x * np.sum(g * x, axis=-1, keepdims=True)
    dw = np.sum((dy * x * r).reshape(-1, H), axis=0)
    return dx, dw


def rope_fwd(x, cos, sin):                      # x [T, heads, hd], rotate-half pairs (i, i + hd/2)
    h = x.shape[-1] // 2
    x1, x2 = x[..., :h], x[..., h:]
    c, s = cos[:, None, :], sin[:, None, :]
    return np.concatenate([x1 * c - x2 * s, x2 * c + x1 * s], axis=-1)


def rope_bwd(dy, cos, sin):                     # the inverse rotation
    h = dy.shape[-1] // 2
    d1, d2 = dy[..., :h], dy[..., h:]
    c, s = cos[:, None, :], sin[:, None, :]
    return np.concatenate([d1 * c + d2 * s, -d1 * s + d2 * c], axis=-1)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class Model:
    def __init__(self, V=48, H=32, nh=4, nkv=2, hd=8, F=64, L=2, qk_norm=False, seed=0, theta=10000.0):
        rng = np.random.default_rng(seed)
        self.V, self.H, self.nh, self.nkv, self.hd, self.F, self.L, self.qk_norm, self.theta = V, H, nh, nkv, hd, F, L, qk_norm, theta
        n = lambda *s: rng.normal(0, 0.3, s)
        self.p = {"embed": n(V, H), "final_norm": 1 + n(H) * 0.3}
        for l in range(L):
            for name, shape in (("attn_norm", (H,)), ("mlp_norm", (H,)), ("q_norm", (hd,)), ("k_norm", (hd,))):
                self.p[f"{l}.{name}"] = 1 + n(*shape) * 0.3
            self.p[f"{l}.wq"], self.p[f"{l}.wk"], self.p[f"{l}.wv"] = n(nh * hd, H), n(nkv * hd, H), n(nkv * hd, H)
            self.p[f"{l}.wo"] = n(H, nh * hd)
            self.p[f"{l}.wg"], self.p[f"{l}.wu"], self.p[f"{l}.wd"] = n(F, H), n(F, H), n(H, F)

    def forward(self, ids, targets):
        p, T = self.p, len(ids)
        nh, nkv, hd = self.nh, self.nkv, self.hd
        rep = nh // nkv
        inv = 1.0 / (self.theta ** (np.arange(0, hd, 2) / hd))
        ang = np.arange(T)[:, None] * inv[None, :]
        cos, sin = np.cos(ang), np.sin(ang)
        mask = np.triu(np.full((T, T), -np.inf), k=1)
        x = p["embed"][ids]
        caches = []
        for l in range(self.L):
            c = {}
            h, c["n1"] = rms_fwd(x, p[f"{l}.attn_norm"])
            q = (h @ p[f"{l}.wq"].T).reshape(T, nh, hd)
            k = (h @ p[f"{l}.wk"].T).reshape(T, nkv, hd)
            v = (h @ p[f"{l}.wv"].T).reshape(T, nkv, hd)
            c["h"], c["q0"], c["k0"] = h, q, k
            if self.qk_norm:
                q, c["qn"] = rms_fwd(q, p[f"{l}.q_norm"])
                k, c["kn"] = rms_fwd(k, p[f"{l}.k_norm"])
            q, k = rope_fwd(q, cos, sin), rope_fwd(k, cos, sin)
            ke, ve = np.repeat(k, rep, axis=1), np.repeat(v, rep, axis=1)
            s = np.einsum("tnd,snd->nts", q, ke) / np.sqrt(hd) + mask[None]
            s = s - s.max(-1, keepdims=True)
            pr = np.exp(s)
            pr /= pr.sum(-1, keepdims=True)
            o = np.einsum("nts,snd->tnd", pr, ve).reshape(T, nh * hd)
            c.update(q=q, k=k, v=v, ke=ke, ve=ve, pr=pr, o=o, x_in=x)
            x = x + o @ p[f"{l}.wo"].T
            h2, c["n2"] = rms_fwd(x, p[f"{l}.mlp_norm"])
            g, u = h2 @ p[f"{l}.wg"].T, h2 @ p[f"{l}.wu"].T
            a = g * sigmoid(g) * u
            c.update(h2=h2, g=g, u=u, a=a, x_mid=x)
            x = x + a @ p[f"{l}.wd"].T
            caches.append(c)
        f, fc = rms_fwd(x, p["final_norm"])
        logits = f @ p["embed"].T
        z = logits - logits.max(-1, keepdims=True)
        prob = np.exp(z)
        prob /= prob.sum(-1, keepdims=True)
        loss = -np.mean(np.log(prob[np.arange(T), targets]))
        self.state = dict(ids=ids, targets=targets, caches=caches, fc=fc, f=f, prob=prob, cos=cos, sin=sin)
        return loss

    def backward(self):
        p, S = self.p, self.state
        T = len(S["ids"])
        nh, nkv, hd = self.nh, self.nkv, self.hd
        rep = nh // nkv
        g = {k: np.zeros_like(v) for k, v in p.items()}
        dlogits = S["prob"].copy()
        dlogits[np.arange(T), S["targets"]] -= 1.0
        dlogits /= T
        g["embed"] += dlogits.T @ S["f"]                                         # tied output head
        df = dlogits @ p["embed"]
        dx, g["final_norm"] = rms_bwd(df, S["fc"])
        for l in reversed(range(self.L)):
            c = S["caches"][l]
            # MLP
            da = dx @ p[f"{l}.wd"]
            g[f"{l}.wd"] += dx.T @ c["a"]
            sg = sigmoid(c["g"])
            silu = c["g"] * sg
            dg = da * c["u"] * sg * (1 + c["g"] * (1 - sg))
            du = da * silu
            g[f"{l}.wg"] += dg.T @ c["h2"]
            g[f"{l}.wu"] += du.T @ c["h2"]
            dh2 = dg @ p[f"{l}.wg"] + du @ p[f"{l}.wu"]
            dxm, g[f"{l}.mlp_norm"] = rms_bwd(dh2, c["n2"])
            dx = dx + dxm                                                        # residual
            # attention
            do = (dx @ p[f"{l}.wo"]).reshape(T, nh, hd)
            g[f"{l}.wo"] += dx.T @ c["o"]
            dve = np.einsum("nts,tnd->snd", c["pr"], do)
            dp = np.einsum("tnd,snd->nts", do, c["ve"])
            ds = c["pr"] * (dp - np.sum(dp * c["pr"], axis=-1, keepdims=True))
            dq = np.einsum("nts,snd->tnd", ds, c["ke"]) / np.sqrt(hd)
            dke = np.einsum("nts,tnd->snd", ds, c["q"]) / np.sqrt(hd)
            dk = dke.reshape(T, nkv, rep, hd).sum(2)                             # GQA: sum over the query heads that share a kv head
            dv = dve.reshape(T, nkv, rep, hd).sum(2)
            dq, dk = rope_bwd(dq, S["cos"], S["sin"]), rope_bwd(dk, S["cos"], S["sin"])
            if self.qk_norm:
                dq, g[f"{l}.q_norm"] = rms_bwd(dq, c["qn"])
                dk, g[f"{l}.k_norm"] = rms_bwd(dk, c["kn"])
            dqf, dkf, dvf = dq.reshape(T, nh * hd), dk.reshape(T, nkv * hd), dv.reshape(T, nkv * hd)
            g[f"{l}.wq"] += dqf.T @ c["h"]
            g[f"{l}.wk"] += dkf.T @ c["h"]
            g[f"{l}.wv"] += dvf.T @ c["h"]
            dh = dqf @ p[f"{l}.wq"] + dkf @ p[f"{l}.wk"] + dvf @ p[f"{l}.wv"]
            dxa, g[f"{l}.attn_norm"] = rms_bwd(dh, c["n1"])
            dx = dx + dxa
        np.add.at(g["embed"], S["ids"], dx)                                      # embedding lookup
        return g


def check(qk_norm, seed):
    m = Model(qk_norm=qk_norm, seed=seed)
    rng = np.random.default_rng(seed + 1)
    T = 9
    ids = rng.integers(0, m.V, T)
    targets = rng.integers(0, m.V, T)
    loss = m.forward(ids, targets)
    grads = m.backward()
    worst = 0.0
    for name in sorted(m.p):
        if not qk_norm and name.endswith(("q_norm", "k_norm")):
            continue
        arr = m.p[name]
        for _ in range(6):                                                       # six random coordinates per parameter tensor
            idx = tuple(rng.integers(0, s) for s in arr.shape)
            old = arr[idx]
            h = 1e-6
            arr[idx] = old + h
            lp = m.forward(ids, targets)
            arr[idx] = old - h
            lm = m.forward(ids, targets)
            arr[idx] = old
            num = (lp - lm) / (2 * h)
            ana = grads[name][idx]
            rel = abs(num - ana) / max(1e-8, abs(num) + abs(ana))
            worst = max(worst, rel)
            if rel > 1e-5:
                print(f"   MISMATCH {name}{idx}: numeric {num:.8f} analytic {ana:.8f}")
    print(f"qk_norm={qk_norm}: loss {loss:.4f}, worst relative gradient error over {len([n for n in m.p if qk_norm or not n.endswith(('q_norm', 'k_norm'))]) * 6} sampled coordinates: {worst:.2e}  {'OK' if worst < 1e-5 else 'FAIL'}")
    return worst < 1e-5


if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    ok = check(False, seed) & check(True, seed)
    sys.exit(0 if ok else 1)
