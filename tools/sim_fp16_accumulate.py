"""Simulation (numpy, CPU) of the numerical error of fp16 accumulation in tensor-core GEMMs, to decide the design of research item R2 BEFORE writing the kernel.
    ~/forgeinfer/venv/bin/python tools/sim_fp16_accumulate.py
Model of the hardware: a tensor-core MMA with k=8 forms the exact (fp32) sum of 8 products and adds it to the accumulator, which is rounded to the accumulator type after every MMA
(fp16: 11 significant bits; fp32: 24). Variants compared for dot products of length K with realistic data (activations ~ N(0,1), weights ~ N(0, 0.02)):
  fp32 accumulate (reference), fp16 accumulate, fp16 accumulate with a flush into an fp32 register every C elements ("chunked").
Reports the relative RMS error of the output against a float64 reference and the worst element, per K and per C.
"""
import numpy as np


def mma_chain(prod, acc_dtype, flush_every=0):
    """prod: [rows, K] products; K multiple of 8. Sum groups of 8 exactly (float64 here), accumulate rounding to acc_dtype after each MMA, optionally flushing chunks into fp32."""
    rows, K = prod.shape
    groups = prod.reshape(rows, K // 8, 8).sum(-1)                 # exact 8-term sums (the tensor core adds them with higher internal precision)
    acc = np.zeros(rows, dtype=acc_dtype)
    total = np.zeros(rows, dtype=np.float32)
    per = (flush_every // 8) if flush_every else 0
    for g in range(groups.shape[1]):
        acc = (acc.astype(np.float64) + groups[:, g]).astype(acc_dtype)
        if per and (g + 1) % per == 0:
            total = (total.astype(np.float64) + acc.astype(np.float64)).astype(np.float32)
            acc = np.zeros(rows, dtype=acc_dtype)
    return (total.astype(np.float64) + acc.astype(np.float64)) if per else acc.astype(np.float64)


def main():
    rng = np.random.default_rng(1)
    rows = 4000
    print(f"{'K':>6} {'mode':>22} {'rel RMS error':>14} {'worst rel error':>16}")
    for K in (576, 1536, 2560, 4096, 8192):
        x = rng.normal(0, 1, K)
        w = rng.normal(0, 0.02, (rows, K))
        x16, w16 = x.astype(np.float16).astype(np.float64), w.astype(np.float16).astype(np.float64)   # inputs are fp16 on the tensor core
        prod = w16 * x16[None, :]
        ref = prod.sum(-1)                                       # float64
        scale = np.sqrt(np.mean(ref ** 2))
        for name, acc, chunk in (("fp32 accumulate", np.float32, 0), ("fp16 accumulate", np.float16, 0), ("fp16, flush every 256", np.float16, 256), ("fp16, flush every 128", np.float16, 128),
                                 ("fp16, flush every 64", np.float16, 64), ("fp16, flush every 32", np.float16, 32)):
            out = mma_chain(prod, acc, chunk)
            err = np.abs(out - ref)
            print(f"{K:>6} {name:>22} {np.sqrt(np.mean(err ** 2)) / scale:>14.2e} {err.max() / scale:>16.2e}")
        print()
    print("reference points: fp16 unit roundoff 4.9e-4, fp32 unit roundoff 6.0e-8; bf16-class activations errors in LLMs are usually >= 4e-3 per layer")


if __name__ == "__main__":
    main()
