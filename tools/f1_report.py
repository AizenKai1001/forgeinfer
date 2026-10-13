"""Merge the proxy A/B results of the homelab (runs/f1_ab_summary.txt) and of Kaggle sweeps (downloaded outputs: summary.jsonl + <arm>.log per arm) into one markdown table with the difference to the baseline.

    python tools/f1_report.py --home FILE_OR_- [--kaggle DIR ...] [--base base,base_s2,base_s3] [--out FILE.md]

Validation loss of the last evaluation (step 500 of the proxy: 33 M tokens); the baseline value is the mean of the --base arms (their spread is the noise floor). Rows are sorted by the difference.
"""
import argparse
import json
import re
import statistics
from pathlib import Path


def home_rows(path):
    rows = {}
    if not path or path == "-":
        return rows
    for line in Path(path).read_text().splitlines():
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 5 or not parts[1].startswith("val@"):
            continue
        name = parts[0]
        val = float(parts[1].split()[1])
        tps = re.search(r"(\d+) tok/s", parts[3])
        params = re.search(r"([\d.]+) M", parts[4])
        rows[name] = {"val": val, "tok_s": int(tps.group(1)) if tps else None, "params": float(params.group(1)) if params else None, "args": parts[5] if len(parts) > 5 else "", "where": "homelab"}
    return rows


def kaggle_rows(directory):
    rows = {}
    d = Path(directory)
    summary = d / "summary.jsonl"
    if not summary.exists():
        return rows
    for line in summary.read_text().splitlines():
        r = json.loads(line)
        if r.get("returncode") != 0 or not r.get("val"):
            continue
        log = d / f"{r['name']}.log"
        params = None
        if log.exists():
            m = re.search(r"([\d.]+) M parameters", log.read_text())
            params = float(m.group(1)) if m else None
        extra = r["args"].split("--qk-norm", 1)[-1].strip()
        rows[r["name"]] = {"val": r["val"][-1], "tok_s": int(r["tok_s"]) if r.get("tok_s") else None, "params": params, "args": extra, "where": f"kaggle {d.name}"}
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--home", default="-")
    ap.add_argument("--kaggle", nargs="*", default=[])
    ap.add_argument("--base", default="base,base_s2,base_s3")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    rows = home_rows(a.home)
    for k in a.kaggle:
        rows.update(kaggle_rows(k))
    bases = [rows[b]["val"] for b in a.base.split(",") if b in rows]
    base = statistics.mean(bases) if bases else float("nan")
    noise = (max(bases) - min(bases)) if len(bases) > 1 else float("nan")
    lines = [f"Baseline (mean of {len(bases)} runs): {base:.4f}, spread {noise:.4f}", "", "| arm | val loss | difference | parameters (M) | tokens/s | where | flags |", "|---|---|---|---|---|---|---|"]
    for name, r in sorted(rows.items(), key=lambda kv: kv[1]["val"]):
        lines.append(f"| {name} | {r['val']:.4f} | {r['val'] - base:+.4f} | {r['params'] or ''} | {r['tok_s'] or ''} | {r['where']} | {r['args']} |")
    text = "\n".join(lines)
    print(text)
    if a.out:
        Path(a.out).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
