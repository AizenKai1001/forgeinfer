#!/usr/bin/env python3
"""sft_bench_index.py: build the n-gram index (default n = 10 normalised words) of the benchmark test sets we evaluate (and a few we track):
HellaSwag (validation + test contexts), ARC-Easy/Challenge (test + validation), PIQA, WinoGrande (xl validation + test), MMLU (test, validation, dev), GSM8K test, MATH test, HumanEval, MBPP (all 974 problems),
LAMBADA test, SciQ, OpenBookQA, IFEval, TruthfulQA.  Output: sft_work/bench_index.npz (sorted unique hashes, owner id per hash, names) and sft_work/bench_index_stats.json.
Every item contributes the n-grams of each long field, of the concatenation of its fields and of "question + choice" for multiple-choice items (so spans across the boundary are covered)."""
import glob
import json
import os
import sys

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sft_lib import RAW, WORK, ngram_hashes, word_ids  # noqa: E402

N = int(os.environ.get("BENCH_NGRAM", "10"))


def rows_parquet(pattern):
    for p in sorted(glob.glob(os.path.join(RAW, pattern))):
        pf = pq.ParquetFile(p)
        for b in pf.iter_batches(batch_size=2000):
            yield from b.to_pylist()


def rows_jsonl(pattern):
    for p in sorted(glob.glob(os.path.join(RAW, pattern))):
        with open(p, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)


def s(x):
    return x if isinstance(x, str) else ""


def items():
    """yields (benchmark, [strings of the item], [(question, choice), ...])"""
    for r in rows_parquet("Rowan__hellaswag/data/*.parquet"):
        yield "hellaswag", [s(r.get("ctx")), s(r.get("ctx_a")) + " " + s(r.get("ctx_b"))], [(s(r.get("ctx")), e) for e in (r.get("endings") or [])]
    for r in rows_parquet("allenai__ai2_arc/ARC-Easy/*.parquet"):
        ch = (r.get("choices") or {}).get("text") or []
        yield "arc_easy", [s(r.get("question"))], [(s(r.get("question")), c) for c in ch]
    for r in rows_parquet("allenai__ai2_arc/ARC-Challenge/*.parquet"):
        ch = (r.get("choices") or {}).get("text") or []
        yield "arc_challenge", [s(r.get("question"))], [(s(r.get("question")), c) for c in ch]
    for r in rows_parquet("baber__piqa/*.parquet"):
        yield "piqa", [s(r.get("goal"))], [(s(r.get("goal")), s(r.get("sol1"))), (s(r.get("goal")), s(r.get("sol2")))]
    for r in rows_parquet("allenai__winogrande/winogrande_xl/*.parquet"):
        sent = s(r.get("sentence"))
        yield "winogrande", [sent.replace("_", s(r.get("option1"))), sent.replace("_", s(r.get("option2")))], []
    for r in rows_parquet("cais__mmlu/all/*.parquet"):
        ch = r.get("choices") or []
        yield "mmlu", [s(r.get("question"))], [(s(r.get("question")), c) for c in ch]
    for r in rows_parquet("openai__gsm8k/main/test-*.parquet"):
        yield "gsm8k_test", [s(r.get("question")), s(r.get("question")) + " " + s(r.get("answer"))], []
    for r in rows_parquet("EleutherAI__hendrycks_math/*/test-*.parquet"):
        yield "math_test", [s(r.get("problem")), s(r.get("problem")) + " " + s(r.get("solution"))], []
    for r in rows_parquet("openai__openai_humaneval/*/*.parquet"):
        yield "humaneval", [s(r.get("prompt")), s(r.get("prompt")) + " " + s(r.get("canonical_solution")), s(r.get("test"))], []
    for r in rows_parquet("google-research-datasets__mbpp/full/*.parquet"):
        yield "mbpp", [s(r.get("text")), s(r.get("text")) + " " + s(r.get("code")), " ".join(r.get("test_list") or [])], []
    for r in rows_jsonl("EleutherAI__lambada_openai/data/lambada_test.jsonl"):
        yield "lambada", [s(r.get("text"))], []
    for r in rows_parquet("allenai__sciq/data/*.parquet"):
        q = s(r.get("question"))
        yield "sciq", [q, s(r.get("support"))], [(q, s(r.get("correct_answer")))]
    for r in rows_parquet("allenai__openbookqa/main/*.parquet"):
        ch = (r.get("choices") or {}).get("text") or []
        q = s(r.get("question_stem"))
        yield "openbookqa", [q], [(q, c) for c in ch]
    for r in rows_jsonl("google__IFEval/ifeval_input_data.jsonl"):
        yield "ifeval", [s(r.get("prompt"))], []
    for r in rows_parquet("truthfulqa__truthful_qa/generation/*.parquet"):
        yield "truthfulqa", [s(r.get("question")), s(r.get("question")) + " " + s(r.get("best_answer"))], []


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
    np.savez(os.path.join(WORK, "bench_index.npz"), n=np.int64(N), hashes=h, owner=o, names=np.array(names))
    stats = {"ngram": N, "items": counts, "ngrams_before_dedup": hash_counts, "unique_hashes": int(len(h))}
    json.dump(stats, open(os.path.join(WORK, "bench_index_stats.json"), "w"), indent=1)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
