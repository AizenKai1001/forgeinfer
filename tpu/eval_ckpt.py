"""Validation loss of a checkpoint (a forgeckpt1 file of either trainer) with the JAX model, on the validation windows of the CUDA trainer: the cross-check of the two implementations on a real model.

    python eval_ckpt.py CKPT --data DIR_WITH_SHARDS [--batches 8] [--dtype f32|bf16]
"""
import argparse
import os
import sys
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
    ap.add_argument("--batches", type=int, default=8)
    ap.add_argument("--dtype", default="f32")
    ap.add_argument("--val-tokens", type=int, default=20_000_000)
    a = ap.parse_args()
    c = fj.cfg_from_checkpoint(a.ckpt)
    params, _, _, step = fj.load_checkpoint(a.ckpt, c, with_state=False)
    data = fj.Data(a.data, a.val_tokens, flags=c.vocab <= 32768)
    cdt = jnp.float32 if a.dtype == "f32" else jnp.bfloat16
    ev = fj.build_eval(c, cdt, 256)
    t0 = time.time()
    tot = 0.0
    for k in range(a.batches):
        t, y, ig = data.val_batch(k, 2, c.seq)
        tot += float(ev(params, t, y, ig))
        print(f"batch {k}: {tot / (k + 1):.4f} ({time.time() - t0:.0f} s)", flush=True)
    print(f"checkpoint step {step}: JAX validation loss {tot / a.batches:.4f} ({a.dtype}, {a.batches} batches of 2 windows, {c.n_layer} layers, dim {c.n_embd}, structure relu2={c.relu2} lambdas={c.lambdas} ve={list(c.ve_layers)})")


if __name__ == "__main__":
    sys.exit(main())
