#!/usr/bin/env python3
"""sft_fn_scan.py [--max-df K] [--write]: HumanEval / MBPP contamination by FUNCTION NAME.

The 13-word rule misses rewritten problems (Evol-CodeAlpaca contains instruction-evolved versions of HumanEval items: "Increase the complexity of this coding test: def will_it_fly(q, w) ..." with a new docstring).
This scan takes the entry points of HumanEval (164) and the function names of MBPP (974, from the reference code), counts in how many pool documents each name is DEFINED (`def name(`), and treats a name as
distinctive when it is defined in at most K documents of all the pools together (the generic ones, is_prime, median, remove_duplicates ... appear in dozens or hundreds of unrelated documents).
Every pool document that defines a distinctive benchmark name is written to sft_work/pool_blacklist_fnames.txt (ids), which sft_assemble.py drops.
"""
import argparse
import collections
import glob
import os
import re
import sys

import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sft_lib as L  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--max-df", type=int, default=12)
ap.add_argument("--write", action="store_true")
a = ap.parse_args()
DEF = re.compile(r"\bdef\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(")
names = {}
for p in glob.glob(os.path.join(L.RAW, "openai__openai_humaneval", "*", "*.parquet")):
    for r in pq.ParquetFile(p).read().to_pylist():
        names[r["entry_point"]] = "humaneval"
for p in glob.glob(os.path.join(L.RAW, "google-research-datasets__mbpp", "full", "*.parquet")):
    for r in pq.ParquetFile(p).read().to_pylist():
        for n in DEF.findall(r.get("code") or ""):
            names.setdefault(n, "mbpp")
print(f"benchmark function names: {len(names)} (humaneval {sum(v == 'humaneval' for v in names.values())})")
df = collections.Counter()
where = collections.defaultdict(list)  # name -> [(source, id)]
for p in sorted(glob.glob(os.path.join(L.WORK, "pool", "*.jsonl"))):
    src = os.path.basename(p)[:-6]
    for rec in L.read_jsonl(p):
        text = "\n".join(c for _, c in rec["msgs"])
        if "def " not in text:
            continue
        for n in set(DEF.findall(text)):
            if n in names:
                df[n] += 1
                where[n].append((src, rec["id"]))
hist = collections.Counter(min(df[n], 100) for n in df)
he_names = {n for n, v in names.items() if v == "humaneval"}
# Evol-CodeAlpaca: instruction-evolved copies of HumanEval problems ("Increase the complexity of this coding test: def f(...)") keep generic entry points such as is_palindrome: drop every evolcode
# document whose PROMPT defines a HumanEval entry point, whatever its document frequency
evol_ids = set()
for rec in L.read_jsonl(os.path.join(L.WORK, "pool", "evolcode.jsonl")):
    user = "\n".join(c for r, c in rec["msgs"] if r == "user")
    if any(n in he_names for n in set(DEF.findall(user))):
        evol_ids.add(rec["id"])
print(f"evolcode documents whose prompt defines a HumanEval entry point: {len(evol_ids)}")
print("document frequency of the benchmark names that occur at all:", len(df), "names; df <= 3:", sum(1 for n in df if df[n] <= 3), "; <= 12:", sum(1 for n in df if df[n] <= 12), "; > 12:", sum(1 for n in df if df[n] > 12))
rare = {n for n in df if df[n] <= a.max_df}
by_src = collections.Counter()
ids = set()
for n in rare:
    for src, i in where[n]:
        ids.add(i)
        by_src[src] += 1
ids |= evol_ids
print(f"distinctive names (df <= {a.max_df}): {len(rare)}; documents to drop (with the evolcode rule): {len(ids)}; by source: {dict(by_src.most_common())}")
print("most frequent ones kept (generic):", ", ".join(f"{n}:{df[n]}" for n, _ in df.most_common(25) if df[n] > a.max_df))
print("examples dropped:", ", ".join(sorted(rare)[:40]))
if a.write:
    with open(os.path.join(L.WORK, "pool_blacklist_fnames.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(sorted(ids)) + "\n")
    print("written", len(ids))
