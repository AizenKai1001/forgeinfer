#!/usr/bin/env python3
"""human_bench_index.py: the 10-gram benchmark index for the human-written-only mix: the same index as sft_bench_index.py (sft_work/bench_index.npz) with ONE difference: of MBPP only the TEST problems
(task_id 11-510) are indexed, so that the train / validation problems of MBPP can be used as training data ("GSM8K / MATH / MBPP train splits are fine, their test splits are not").
Everything else is unchanged (HellaSwag val + test, ARC test + validation, PIQA, WinoGrande, MMLU all splits, GSM8K test, MATH test, HumanEval, LAMBADA test, SciQ, OpenBookQA, IFEval, TruthfulQA).
Output: sft_work_human/bench_index_human.npz (+ stats json).  The original index is NOT touched.  Run with the homelab python under nice; the raw files are those of the first build (sft_raw)."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sft_bench_index as B  # noqa: E402  (its helpers; its main() is not called)
from sft_lib import ngram_hashes, word_ids  # noqa: E402

OUT = "/mnt/data/forgeinfer-data/sft_work_human"
N = B.N


def items():
    for it in B.items():
        if it[0] == "mbpp":
            continue
        yield it
    for r in B.rows_parquet("google-research-datasets__mbpp/full/test-*.parquet"):
        yield "mbpp", [B.s(r.get("text")), B.s(r.get("text")) + " " + B.s(r.get("code")), " ".join(r.get("test_list") or [])], []


def main():
    names, h_all, o_all = [], [], []
    counts, hash_counts = {}, {}
    for bench, fields, pairs in items():
        if bench not in names:
            names.append(bench)
        bi = names.index(bench)
        counts[bench] = counts.get(bench, 0) + 1
        strings = list(fields)
        strings.append(" ".join(f for f in fields if f))
        for q, c in pairs:
            strings.append(q + " " + c)
        seen = set()
        for t in strings:
            if not t or t in seen:
                continue
            seen.add(t)
            ids = word_ids(t)
            if len(ids) < N:
                continue
            h = ngram_hashes(ids, N)
            h_all.append(h)
            o_all.append(np.full(len(h), bi, dtype=np.int16))
            hash_counts[bench] = hash_counts.get(bench, 0) + len(h)
    h = np.concatenate(h_all)
    o = np.concatenate(o_all)
    order = np.argsort(h, kind="stable")
    h, o = h[order], o[order]
    keep = np.concatenate([[True], h[1:] != h[:-1]])
    h, o = h[keep], o[keep]
    np.savez(os.path.join(OUT, "bench_index_human.npz"), n=np.int64(N), hashes=h, owner=o, names=np.array(names))
    stats = {"ngram": N, "items": counts, "ngrams_before_dedup": hash_counts, "unique_hashes": int(len(h)), "note": "like sft_work/bench_index.npz, but MBPP = test problems 11-510 only"}
    json.dump(stats, open(os.path.join(OUT, "bench_index_human_stats.json"), "w"), indent=1)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
