#!/usr/bin/env python3
"""human_audit.py pools [SOURCE ...] | fnames [--max-df K] | build [BUILD]: decontamination scans of the human-written-only mix (see sft_audit.py and sft_fn_scan.py of the first build).

  pools    every pool record against the exact files of the evaluation ladder (/mnt/data/forgeinfer-data/eval/{mc,hellaswag,lambada}: contexts, every continuation, endings, MMLU option lists): a verbatim
           run of 13 or more words (four consecutive matching 10-grams, mostly-numeric 10-grams ignored) -> the id goes to sft_work_human/blacklist_eval.txt (the assembler drops it).
  fnames   HumanEval entry points and the function names of the MBPP TEST problems (task_id 11-510): the pool documents that DEFINE a distinctive name (defined in at most K = 12 documents of all pools together)
           -> sft_work_human/blacklist_fnames.txt.  (MBPP train / validation problems are training data here, so their names do not count.)
  build    the final audit of the corpus files BUILD/<shard>.txt (documents separated by NUL) against the human benchmark index AND the evaluation files; exit code 1 when a run is found.
"""
import collections
import glob
import json
import os
import re
import sys

import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import human_cfg as C  # noqa: E402
import human_pools as HP  # noqa: E402
import sft_audit as A  # noqa: E402
import sft_lib as L  # noqa: E402
from sft_pools import RUN_MIN  # noqa: E402

DEF = re.compile(r"\bdef\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(")


def pools_mode(sources):
    eh = A.eval_hashes()
    names = sources or sorted(os.path.basename(p)[:-6] for p in glob.glob(os.path.join(C.POOL, "*.jsonl")))
    out = []
    for name in names:
        n = bad = 0
        for rec in L.read_jsonl(os.path.join(C.POOL, name + ".jsonl")):
            n += 1
            r, _ = A.runs("\n".join(c for _, c in rec["msgs"]), eh)
            if r >= RUN_MIN:
                out.append(rec["id"])
                bad += 1
        print(f"[audit pools] {name}: {n} records, {bad} with a verbatim run against the eval files", flush=True)
    with open(os.path.join(C.WORK, "blacklist_eval.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(out) + ("\n" if out else ""))
    print(f"[audit pools] {len(out)} ids written to blacklist_eval.txt", flush=True)


def fnames_mode(max_df):
    names = {}
    for p in glob.glob(os.path.join(L.RAW, "openai__openai_humaneval", "*", "*.parquet")):
        for r in pq.ParquetFile(p).read().to_pylist():
            names[r["entry_point"]] = "humaneval"
    for p in glob.glob(os.path.join(L.RAW, "google-research-datasets__mbpp", "full", "test-*.parquet")):
        for r in pq.ParquetFile(p).read().to_pylist():
            for n in DEF.findall(r.get("code") or ""):
                names.setdefault(n, "mbpp_test")
    print(f"benchmark function names: {len(names)} (humaneval {sum(v == 'humaneval' for v in names.values())})")
    df = collections.Counter()
    where = collections.defaultdict(list)
    for p in sorted(glob.glob(os.path.join(C.POOL, "*.jsonl"))):
        src = os.path.basename(p)[:-6]
        for rec in L.read_jsonl(p):
            text = "\n".join(c for _, c in rec["msgs"])
            if "def " not in text:
                continue
            for n in set(DEF.findall(text)):
                if n in names:
                    df[n] += 1
                    where[n].append((src, rec["id"]))
    rare = {n for n in df if df[n] <= max_df}
    by_src = collections.Counter()
    ids = set()
    for n in rare:
        for src, i in where[n]:
            ids.add(i)
            by_src[src] += 1
    print(f"names that occur: {len(df)}; distinctive (df <= {max_df}): {len(rare)}; documents to drop: {len(ids)}; by source: {dict(by_src.most_common())}")
    print("most frequent ones kept (generic):", ", ".join(f"{n}:{df[n]}" for n, _ in df.most_common(20) if df[n] > max_df))
    print("examples dropped:", ", ".join(sorted(rare)[:40]))
    with open(os.path.join(C.WORK, "blacklist_fnames.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(sorted(ids)) + ("\n" if ids else ""))


def build_mode(build):
    dc = HP.HumanDecontam()
    eh = A.eval_hashes()
    docs_meta = {}
    with open(os.path.join(build, "docs.tsv"), encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\n").split("\t")
            docs_meta[(p[0], int(p[1]))] = p[2] + ":" + p[8]
    bad = 0
    total = collections.Counter()
    offenders_all = []
    for path in sorted(glob.glob(os.path.join(build, "*.txt"))):
        name = os.path.basename(path)[:-4]
        data = open(path, "rb").read().split(b"\x00")
        seen = {}
        c = collections.Counter()
        for k, raw in enumerate(data):
            if not raw:
                continue
            text = raw.decode("utf-8")
            c["docs"] += 1
            if raw in seen:
                v = seen[raw]
            else:
                hit = dc.check(text)
                r_eval, n_eval = A.runs(text, eh)
                v = (hit, r_eval, n_eval)
                if len(seen) < 200000:
                    seen[raw] = v
            hit, r_eval, n_eval = v
            if hit:
                c["bench_run"] += 1
                offenders_all.append((name, k, docs_meta.get((name, k), "?"), "bench:" + hit))
            if r_eval >= RUN_MIN:
                c["eval_run"] += 1
                offenders_all.append((name, k, docs_meta.get((name, k), "?"), f"eval run {r_eval}"))
            if n_eval:
                c["eval_any_10gram"] += 1
        bad += c["bench_run"] + c["eval_run"]
        total.update(c)
        print(f"[audit] {name}: {dict(c)}", flush=True)
    for o in offenders_all[:30]:
        print("   offender", o)
    print("[audit] TOTAL", dict(total))
    print("[audit] RESULT:", "CLEAN (no verbatim run of 13+ words shared with any benchmark file)" if not bad else f"{bad} documents with a verbatim overlap")
    json.dump({"total": dict(total), "offenders": offenders_all}, open(os.path.join(build, "audit.json"), "w"), indent=1)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "pools":
        pools_mode(sys.argv[2:])
    elif mode == "fnames":
        k = 12
        if "--max-df" in sys.argv:
            k = int(sys.argv[sys.argv.index("--max-df") + 1])
        fnames_mode(k)
    elif mode == "build":
        build_mode(sys.argv[2] if len(sys.argv) > 2 else C.BUILD)
    else:
        print(__doc__)
