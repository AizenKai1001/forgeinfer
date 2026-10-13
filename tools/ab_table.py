"""Table of the arms of one or more Kaggle proxy rounds (downloaded outputs with summary.jsonl): final validation loss, the difference to a reference arm, the validation loss at equal TOKENS
(the evaluation points are every 100 steps; rounds with a smaller batch have more points, so the loss at the last common token count is compared), tokens per second and the run time.

    python tools/ab_table.py DIR [DIR ...] [--ref NAME] [--tokens-per-step-default 65536]

DIR = <dev>/kaggle/out/forge-ab10 (a folder with summary.jsonl and <arm>.log). The tokens per step of an arm are read from its command line (--tokens-per-step N, default 65,536); `val` of the
last evaluation is compared. The noise floor of the 2000-step proxy is about 0.003 (seeds of the same recipe).
"""
import argparse
import json
import re
from pathlib import Path


def tps_of(args, default):
    m = re.search(r"--tokens-per-step\s+(\d+)", args or "")
    return int(m.group(1)) if m else default


def steps_of(args):
    m = re.search(r"--steps\s+(\d+)", args or "")
    return int(m.group(1)) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--ref", default="")
    ap.add_argument("--tokens-per-step-default", type=int, default=65536)
    a = ap.parse_args()
    rows = []
    for d in a.dirs:
        p = Path(d) / "summary.jsonl"
        if not p.exists():
            print(f"(no summary.jsonl in {d})")
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
            except ValueError:
                continue
            vals = r.get("val") or []
            tps = tps_of(r.get("args"), a.tokens_per_step_default)
            steps = steps_of(r.get("args"))
            rows.append({"round": Path(d).name, "name": r["name"], "rc": r.get("returncode"), "val": vals[-1] if vals else float("nan"), "n_val": len(vals), "tps": tps,
                         "steps": steps, "toks_M": (steps * tps / 1e6) if steps else float("nan"), "tok_s": r.get("tok_s"), "sec": r.get("seconds")})
    if not rows:
        return
    ref = next((r for r in rows if r["name"] == a.ref), None)
    print(f"{'round':12s} {'arm':18s} {'tokens M':>9s} {'val(final)':>10s} {'vs ref':>8s} {'tok/s':>8s} {'minutes':>8s} rc")
    for r in sorted(rows, key=lambda x: (x["round"], x["name"])):
        diff = f"{r['val'] - ref['val']:+.4f}" if ref else "-"
        print(f"{r['round']:12s} {r['name']:18s} {r['toks_M']:9.0f} {r['val']:10.4f} {diff:>8s} {(r['tok_s'] or 0):8.0f} {(r['sec'] or 0) / 60:8.1f} {r['rc']}")


if __name__ == "__main__":
    main()
