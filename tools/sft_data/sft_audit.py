#!/usr/bin/env python3
"""sft_audit.py [BUILD]: final decontamination audit of the corpus files of the SFT mix (BUILD/<shard>.txt, documents separated by NUL), independent of the pool filters.

Two indexes of word 10-grams are checked, with the same rule as the pools (a verbatim run of >= 4 consecutive matching 10-grams = >= 13 words; n-grams that are mostly numbers ignored):
  bench  sft_work/bench_index.npz   the full test / validation sets of HellaSwag, ARC, PIQA, WinoGrande, MMLU, GSM8K, MATH, HumanEval, MBPP, LAMBADA, SciQ, OpenBookQA, TruthfulQA (IFEval: 16-grams)
  eval   /mnt/data/forgeinfer-data/eval/{mc,hellaswag,lambada}/*.jsonl   the exact files of the evaluation ladder (context + every continuation, endings, passages)
Reports per file the number of documents with such a run, the number with ANY shared 10-gram (short overlaps: common phrases, kept) and prints the first offenders.  Exit code 1 when a run is found.
"""
import collections
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sft_lib as L  # noqa: E402
from sft_pools import Decontam, RUN_MIN  # noqa: E402

POOLS_MODE = len(sys.argv) > 1 and sys.argv[1] == "pools"
BUILD = sys.argv[1] if len(sys.argv) > 1 and not POOLS_MODE else os.environ.get("SFT_BUILD", "/tmp/sft_mix2_build")
EVAL = "/mnt/data/forgeinfer-data/eval"
N = 10


def eval_hashes():
    hs = []
    items = 0
    for p in sorted(glob.glob(f"{EVAL}/mc/*.jsonl")) + [f"{EVAL}/hellaswag/hellaswag_val.jsonl", f"{EVAL}/lambada/lambada_openai.jsonl"]:
        if not os.path.exists(p):
            continue
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            items += 1
            strings = []
            if "choices" in r:
                strings += [c["ctx"] + c["cont"] for c in r["choices"]]
                strings += [r["choices"][0]["ctx"]]
            elif "endings" in r:
                strings += [r["ctx"]] + [r["ctx"] + " " + e for e in r["endings"]]
            else:
                strings += [r["ctx"] + r["cont"]]
            for s in strings:
                ids = L.word_ids(s)
                if len(ids) >= N:
                    hs.append(L.ngram_hashes(ids, N))
    h = np.unique(np.concatenate(hs))
    print(f"[audit] eval index: {items} items, {len(h)} unique 10-gram hashes", flush=True)
    return h


def runs(text, hashes):
    """(longest run of consecutive matching 10-grams, number of matching 10-grams), numeric-dominated 10-grams ignored."""
    ids = L.word_ids(text)
    h = L.ngram_hashes(ids, N)
    if not len(h):
        return 0, 0
    pos = np.searchsorted(hashes, h)
    pos[pos >= len(hashes)] = 0
    ok = hashes[pos] == h
    if not ok.any():
        return 0, 0
    isnum = np.fromiter((w.isdigit() for w in L.words_of(text)), dtype=np.int32, count=len(ids))
    ok &= ~(np.convolve(isnum, np.ones(N, dtype=np.int32), "valid") >= int(0.7 * N))
    if not ok.any():
        return 0, 0
    idx = np.flatnonzero(ok)
    breaks = np.flatnonzero(np.diff(idx) != 1)
    starts = np.concatenate(([0], breaks + 1))
    ends = np.concatenate((breaks + 1, [len(idx)]))
    return int((ends - starts).max()), int(len(idx))


def main():
    dc = Decontam()
    eh = eval_hashes()
    docs_meta = {}
    with open(os.path.join(BUILD, "docs.tsv"), encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\n").split("\t")
            docs_meta[(p[0], int(p[1]))] = p[2] + ":" + p[8]
    bad = 0
    total = collections.Counter()
    for path in sorted(glob.glob(os.path.join(BUILD, "*.txt"))):
        name = os.path.basename(path)[:-4]
        data = open(path, "rb").read().split(b"\x00")
        seen = {}
        c = collections.Counter()
        offenders = []
        for k, raw in enumerate(data):
            if not raw:
                continue
            text = raw.decode("utf-8")
            c["docs"] += 1
            if raw in seen:  # oversampled copies: same verdict
                v = seen[raw]
            else:
                hit = dc.check(text)
                r_eval, n_eval = runs(text, eh)
                v = (hit, r_eval, n_eval)
                if len(seen) < 200000:
                    seen[raw] = v
            hit, r_eval, n_eval = v
            if hit:
                c["bench_run"] += 1
                offenders.append((k, "bench:" + hit))
            if r_eval >= RUN_MIN:
                c["eval_run"] += 1
                offenders.append((k, f"eval run {r_eval}"))
            if n_eval:
                c["eval_any_10gram"] += 1
        bad += c["bench_run"] + c["eval_run"]
        total.update(c)
        print(f"[audit] {name}: {dict(c)}", flush=True)
        for k, why in offenders[:8]:
            print(f"   offender doc {k} {docs_meta.get((name, k), '?')}: {why}")
    print("[audit] TOTAL", dict(total))
    print("[audit] RESULT:", "CLEAN (no verbatim run of 13+ words shared with any benchmark file)" if not bad else f"{bad} documents with a verbatim overlap")
    sys.exit(1 if bad else 0)


def scan_pools(sources):
    """pools mode: scan the pool files against the eval-file index and write the ids with a verbatim run of >= 13 words to sft_work/pool_blacklist_<sources>.txt (the assembler drops them)."""
    eh = eval_hashes()
    pool = os.path.join(L.WORK, "pool")
    names = sources or sorted(os.path.basename(p)[:-6] for p in glob.glob(os.path.join(pool, "*.jsonl")))
    out = []
    for name in names:
        n = bad = 0
        for rec in L.read_jsonl(os.path.join(pool, name + ".jsonl")):
            n += 1
            r, _ = runs("\n".join(c for _, c in rec["msgs"]), eh)
            if r >= RUN_MIN:
                out.append(rec["id"])
                bad += 1
        print(f"[audit pools] {name}: {n} records, {bad} with a verbatim run against the eval files", flush=True)
    tag = "_".join(sources) if sources else "all"
    with open(os.path.join(L.WORK, f"pool_blacklist_{tag}.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(out) + ("\n" if out else ""))
    print(f"[audit pools] {len(out)} ids written", flush=True)


if __name__ == "__main__":
    if POOLS_MODE:
        scan_pools(sys.argv[2:])
    else:
        main()
