"""What does sliding-window attention cost a model that was trained with full attention? A CPU diagnostic with the JAX model on a checkpoint of either trainer.

    python window_diag.py CKPT --data DIR_WITH_zzz_val [--windows 8] [--window 512] [--layers auto|0,1,2]

For the same validation windows it computes (1) the dense validation loss, (2) the loss when the chosen layers attend only to the last WINDOW keys (no training, no adaptation: the damage of switching a running
model), and (3) per layer the share of attention probability that lies at key distances of WINDOW or more (queries at positions >= WINDOW only): how much a layer uses long-range context. float32 on the CPU,
eager mode (jit off), about a minute per window and variant for the 150M model. `auto` = the layers of `--window-layers auto` of the Rust trainer: all but every fourth (3, 7, 11, ...) and the last.
"""
import argparse
import math
import os
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax
import jax.numpy as jnp
import numpy as np

import forge_jax as fj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--data", required=True)
    ap.add_argument("--windows", type=int, default=8)
    ap.add_argument("--window", type=int, default=512)
    ap.add_argument("--layers", default="auto")
    ap.add_argument("--val-tokens", type=int, default=20_000_000)
    a = ap.parse_args()
    c = fj.cfg_from_checkpoint(a.ckpt)
    params, _, _, step = fj.load_checkpoint(a.ckpt, c, with_state=False)
    n = c.n_layer
    win_layers = {l for l in range(n) if l % 4 != 3 and l != n - 1} if a.layers == "auto" else {int(x) for x in a.layers.split(",") if x}
    data = fj.Data(a.data, a.val_tokens, flags=c.vocab <= 32768)
    cdt = jnp.float32
    state = {"calls": 0, "mode": "dense", "far": np.zeros(n), "far_n": np.zeros(n)}
    W = a.window

    def impl(cfg, q, k, v, cdt_):
        l = state["calls"] % n
        state["calls"] += 1
        b, t = q.shape[:2]
        rep = cfg.n_head // cfg.n_kv
        qq = q.reshape(b, t, cfg.n_kv, rep, cfg.hd).astype(cdt_)
        att = jnp.einsum("btgrd,bsgd->bgrts", qq, k.astype(cdt_), preferred_element_type=jnp.float32) / math.sqrt(cfg.hd)
        ti = jnp.arange(t)[:, None]
        si = jnp.arange(t)[None, :]
        mask = si <= ti
        if state["mode"] == "window" and l in win_layers:
            mask = mask & (si > ti - W)
        att = jnp.where(mask[None, None, None], att, -jnp.inf)
        p = jax.nn.softmax(att, axis=-1)
        if state["mode"] == "dense":
            far = (si <= ti - W)[None, None, None]
            rows = (jnp.arange(t) >= W)
            share = jnp.where(far, p, 0.0).sum(-1)  # [b, g, r, t]
            sel = share[..., rows]
            state["far"][l] += float(sel.mean())
            state["far_n"][l] += 1
        return jnp.einsum("bgrts,bsgd->btgrd", p.astype(cdt_), v.astype(cdt_), preferred_element_type=jnp.float32).reshape(b, t, cfg.q_dim)

    fj.ATTN["impl"] = impl
    res = {}
    t0 = time.time()
    with jax.disable_jit():
        for mode in ("dense", "window"):
            state["mode"], state["calls"] = mode, 0
            tot = 0.0
            for k in range(a.windows):
                tk, y, ig = data.val_batch(k, 1, c.seq)
                tot += float(fj.loss_fn(c, params, tk, y, ig, cdt, 256, remat=False))
                print(f"{mode}: window {k + 1}/{a.windows}: running mean {tot / (k + 1):.4f} ({time.time() - t0:.0f} s)", flush=True)
            res[mode] = tot / a.windows
    print(f"\ncheckpoint step {step}: {n} layers, windows {a.windows} of {c.seq} tokens, window {W} in layers {sorted(win_layers)}")
    print(f"dense validation loss      {res['dense']:.4f}")
    print(f"window without adaptation  {res['window']:.4f}   (damage {res['window'] - res['dense']:+.4f})")
    far = state["far"] / np.maximum(state["far_n"], 1)
    print("share of attention probability at distances >= %d (queries at positions >= %d), per layer:" % (W, W))
    print(" ".join(f"{l}{'w' if l in win_layers else 'd'}:{100 * far[l]:.1f}%" for l in range(n)))
    wl = [far[l] for l in range(n) if l in win_layers]
    dl = [far[l] for l in range(n) if l not in win_layers]
    print(f"mean over the layers to be windowed {100 * float(np.mean(wl)):.1f} %, over the dense layers {100 * float(np.mean(dl)):.1f} %")


if __name__ == "__main__":
    main()
