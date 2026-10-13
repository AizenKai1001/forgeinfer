"""Compare two training logs (log.jsonl of the Rust or the JAX trainer) at equal steps: validation loss and the smoothed training loss, and the difference B - A.

    python tools/f1_compare.py A.jsonl B.jsonl [--names v1,v2] [--every 500] [--window 50]

Validation losses are compared at the steps both logs have (every 100 steps); the training loss is the mean over `window` steps before the step (a smoother signal; both runs draw different random windows,
so differences below ~0.01 are noise). The last row is the latest step the shorter log reaches.
"""
import argparse
import json
from pathlib import Path


def load(path):
    rows = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except ValueError:
            continue
        rows[r["step"]] = r
    return rows


def smooth(rows, step, window):
    vals = [rows[s]["loss"] for s in range(step - window + 1, step + 1) if s in rows]
    return sum(vals) / len(vals) if len(vals) >= max(3, window // 4) else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--names", default="A,B")
    ap.add_argument("--every", type=int, default=500)
    ap.add_argument("--window", type=int, default=50)
    args = ap.parse_args()
    na, nb = args.names.split(",")[:2]
    a, b = load(args.a), load(args.b)
    val_a = {s: r["val_loss"] for s, r in a.items() if "val_loss" in r}
    val_b = {s: r["val_loss"] for s, r in b.items() if "val_loss" in r}
    common = sorted(set(val_a) & set(val_b))
    if not common:
        print("no common validation steps")
        return
    last = common[-1]
    steps = [s for s in common if s % args.every == 0]
    if last not in steps:
        steps.append(last)
    print(f"{'step':>7} | {'val ' + na:>9} | {'val ' + nb:>9} | {'diff':>8} | {'train ' + na:>10} | {'train ' + nb:>10} | {'diff':>8}")
    for s in steps:
        ta, tb = smooth(a, s, args.window), smooth(b, s, args.window)
        print(f"{s:7d} | {val_a[s]:9.4f} | {val_b[s]:9.4f} | {val_b[s] - val_a[s]:+8.4f} | {ta:10.4f} | {tb:10.4f} | {tb - ta:+8.4f}")
    k = [s for s in common if s >= last - 400]
    if len(k) >= 3:
        d = sum(val_b[s] - val_a[s] for s in k) / len(k)
        print(f"mean validation difference over the last {len(k)} common evaluations (steps {k[0]}-{k[-1]}): {d:+.4f}  ({nb} minus {na}; negative = {nb} is better)")


if __name__ == "__main__":
    main()
