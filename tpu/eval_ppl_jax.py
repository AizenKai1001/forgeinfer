"""The JAX twin of `forge eval-ppl`: mean negative log-likelihood per token of a forgeckpt1 checkpoint on token files, on exactly the windows the Rust command uses (`windows` windows of seq + 1 tokens, window k starts at
span * k // windows, every position scored). Used to check that a checkpoint written by one trainer is read correctly by the other:

    python eval_ppl_jax.py CKPT --data FILE.u16 [FILE.u16 ...] [--windows 16] [--seq 512] [--batch 4] [--dtype f32|bf16]
"""
import argparse
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax.numpy as jnp
import numpy as np

import forge_jax as fj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--windows", type=int, default=64)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--dtype", default="f32")
    a = ap.parse_args()
    c = fj.cfg_from_checkpoint(a.ckpt)
    c = fj.Cfg(**{**c.__dict__, "seq": a.seq})
    params, _, _, step = fj.load_checkpoint(a.ckpt, c, with_state=False)
    cdt = jnp.float32 if a.dtype == "f32" else jnp.bfloat16
    ev = fj.build_eval(c, cdt, min(256, a.seq))
    for f in a.data:
        toks = np.fromfile(f, dtype="<u2").astype(np.int64)
        span = len(toks) - (a.seq + 1)
        total, count = 0.0, 0
        n_batches = (a.windows + a.batch - 1) // a.batch
        for bi in range(n_batches):
            rows = []
            for r in range(a.batch):
                k = bi * a.batch + r
                if k >= a.windows:
                    break
                start = span * (k % a.windows) // a.windows
                rows.append(toks[start : start + a.seq + 1])
            win = np.stack(rows).astype(np.int32)
            ids = win & 0x7FFF if c.vocab <= 32768 else win
            ignore = np.zeros(ids[:, 1:].shape, bool)
            loss = float(ev(params, ids[:, :-1], ids[:, 1:], ignore))  # the mean over the positions of this batch
            total += loss * ids[:, 1:].size
            count += ids[:, 1:].size
        mean = total / count
        print(f"{os.path.splitext(os.path.basename(f))[0]}: {mean:.4f} nats per token (perplexity {np.exp(mean):.1f}) over {count} tokens  [JAX, checkpoint step {step}]")


if __name__ == "__main__":
    main()
