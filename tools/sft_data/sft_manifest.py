#!/usr/bin/env python3
"""sft_manifest.py [--build BUILD] [--out DIR]: write MANIFEST.txt (and manifest.json) of the published SFT mix next to its shards.
Reads the plan (plan.json, docs.tsv of the build directory), the pool statistics (sft_work/pool/*.stats.json) and the published .u16 files (token and mask counts are counted from the files themselves)."""
import argparse
import collections
import datetime
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sft_lib as L  # noqa: E402
import sft_assemble as A  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--build", default=A.BUILD)
ap.add_argument("--out", default=L.OUT)
a = ap.parse_args()

plan = json.load(open(os.path.join(a.build, "plan.json")))
T = plan["validation_tokens_T"]
n_tail = plan["val_file_tail_train_docs"]
rows = []  # (file, idx, src, stratum, ntok, nsup, turns, has_system, is_val)
with open(os.path.join(a.build, "docs.tsv"), encoding="utf-8") as f:
    next(f)
    for line in f:
        p = line.rstrip("\n").split("\t")
        is_val = p[0] == "zzz_val" and int(p[1]) >= n_tail
        rows.append((p[0], int(p[1]), p[2], p[3], int(p[4]), int(p[5]), int(p[6]), int(p[7]), is_val))

pool_stats = {}
for s in A.TARGETS_M:
    p = os.path.join(A.POOL, s + ".stats.json")
    if os.path.exists(p):
        pool_stats[s] = json.load(open(p))

# counts from the published files themselves
files = sorted(f for f in os.listdir(a.out) if f.endswith(".u16"))
fcount = {}
for f in files:
    arr = np.fromfile(os.path.join(a.out, f), dtype="<u2")
    h = hashlib.sha256(arr.tobytes()).hexdigest()[:16]
    fcount[f] = dict(tokens=int(len(arr)), supervised=int(((arr & 0x8000) == 0).sum()), bytes=int(len(arr) * 2), sha256_16=h)

by_src = collections.OrderedDict()
for s in A.TARGETS_M:
    by_src[s] = dict(train_docs=0, train_tokens=0, train_sup=0, val_docs=0, val_tokens=0, val_sup=0)
for r in rows:
    d = by_src[r[2]]
    k = "val" if r[8] else "train"
    d[k + "_docs"] += 1
    d[k + "_tokens"] += r[4]
    d[k + "_sup"] += r[5]
tot_train = sum(d["train_tokens"] for d in by_src.values())
tot_val = sum(d["val_tokens"] for d in by_src.values())
tot_sup = sum(d["train_sup"] for d in by_src.values())
val_sup = sum(d["val_sup"] for d in by_src.values())

edges = A.LEN_EDGES


def hist(vals):
    c = [0] * (len(edges) + 1)
    for v in vals:
        k = 0
        while k < len(edges) and v >= edges[k]:
            k += 1
        c[k] += 1
    return c


tr_rows = [r for r in rows if not r[8]]
len_hist = hist([r[4] for r in tr_rows])
tok_hist = [0] * (len(edges) + 1)
for r in tr_rows:
    k = 0
    while k < len(edges) and r[4] >= edges[k]:
        k += 1
    tok_hist[k] += r[4]
turns = collections.Counter(min(r[6], 4) for r in tr_rows)
n_sys = sum(r[7] for r in tr_rows)
group = collections.Counter()
prov = collections.Counter()
for s, d in by_src.items():
    group[A.META[s][3]] += d["train_tokens"]
    pv = A.META[s][2]
    prov["human-written" if pv.startswith("human") else "LLM-generated"] += d["train_tokens"]

lines = []
W = lines.append
W("MANIFEST of tokens_sft_mix2 (Forge-1 chat fine-tune mix v2)")
W("=" * 78)
W(f"written {datetime.datetime.now().strftime('%Y-%m-%d %I:%M %p')} by tools/sft_data/sft_manifest.py; description, licenses, recipe and weaknesses: research note 'ForgeInfer - SFT Data Mix'")
W("")
W("FILES  (uint16 little-endian tokens of the 32,768-entry tokenizer_f1.json; bit 15 = 'not a target': prompt, headers, separators are masked, the assistant text and its <|im_end|> train)")
for f in files:
    c = fcount[f]
    W(f"  {f:14s} {c['tokens']:>12,d} tokens  {c['bytes'] / 1e6:8.1f} MB  supervised {c['supervised']:>12,d} ({100 * c['supervised'] / c['tokens']:.1f} %)  sha256[:16] {c['sha256_16']}")
tot_tokens = sum(c["tokens"] for c in fcount.values())
W(f"  total {tot_tokens:,d} tokens in {len(files)} files; training documents {tot_train:,d} tokens (incl. oversampled copies), held-out validation {tot_val:,d} tokens")
W(f"  heldout/sft_val_masked.u16  {T:,d} tokens: the last T tokens of zzz_val.u16 = the held-out conversations alone, WITH the 'not a target' bit (for code that understands it: the trainers' Data class)")
W("  heldout/sft_val_plain.u16   the same tokens with bit 15 cleared: for `forge eval-ppl` (it reads the ids raw: NEVER give it a masked file, ids >= 32768 fault on the GPU); it scores every token of the chat text, prompts included")
W("  (both in a sub-folder on purpose: forge train reads every *.u16 of the directory it is given)")
W("")
W("VALIDATION LAYOUT (important)")
W("  The trainer takes the LAST .u16 file (alphabetical) and reserves min(--val-tokens, tokens / 4) of its tail as the validation region; the rest of that file is TRAINING data.")
W(f"  zzz_val.u16 = [{plan['val_file_pad_tokens']} masked <|endoftext|> padding tokens][{n_tail:,d} training documents, {plan['val_file_tail_train_tokens']:,d} tokens][{plan['val_docs']:,d} held-out documents, T = {T:,d} tokens],")
W(f"  length 4 x T = {4 * T:,d}, so tokens / 4 = T and the validation region is exactly the held-out set for ANY --val-tokens >= {T} (the code default is 20,000,000, the smoke script uses 4,000,000).")
W("  Do NOT put a replay shard whose name sorts after 'sft_*' into the directory (the validation tail is taken from the last file).")
W(f"  Validation loss = mean over the assistant tokens of the held-out conversations ({100 * val_sup / max(1, tot_val):.1f} % of the region is supervised); windows start at random token offsets of the region.")
W("")
W("SOURCES  (tokens counted with the tokenizer, ChatML framing and the closing <|endoftext|> included; 'x' = oversampling of the training copy)")
W(f"  {'source':12s} {'group':8s} {'x':>2s} {'train docs':>10s} {'train tokens':>13s} {'share':>6s} {'sup %':>6s} {'val docs':>8s} {'val tokens':>11s}  license | provenance | origin")
for s, d in by_src.items():
    m = A.META[s]
    x = A.OVERSAMPLE.get(s, 1)
    sup = 100 * d["train_sup"] / max(1, d["train_tokens"])
    W(f"  {s:12s} {m[3]:8s} {x:>2d} {d['train_docs']:>10,d} {d['train_tokens']:>13,d} {100 * d['train_tokens'] / tot_train:>5.1f}% {sup:>6.1f} {d['val_docs']:>8,d} {d['val_tokens']:>11,d}  {m[1]} | {m[2]} | {m[0]}")
W(f"  {'TOTAL':12s} {'':8s} {'':>2s} {sum(d['train_docs'] for d in by_src.values()):>10,d} {tot_train:>13,d} {'100.0%':>6s} {100 * tot_sup / tot_train:>6.1f} {sum(d['val_docs'] for d in by_src.values()):>8,d} {tot_val:>11,d}")
W("")
W("BY GROUP (share of the training tokens): " + ", ".join(f"{g} {100 * t / tot_train:.1f} %" for g, t in group.most_common()))
W("BY PROVENANCE: " + ", ".join(f"{g} {100 * t / tot_train:.1f} %" for g, t in prov.most_common()) + "  (human-written = OASST2, Dolly, GSM8K, MATH and the hand-written sets; everything else was generated by LLMs, named per source above)")
W("")
W("DOCUMENT LENGTH (tokens per conversation incl. framing, training documents)")
W("  bucket      " + "  ".join(f"{('<' + str(edges[0])) if i == 0 else (str(edges[i - 1]) + '-' + str(edges[i]) if i < len(edges) else '>=' + str(edges[-1])):>9s}" for i in range(len(edges) + 1)))
W("  documents   " + "  ".join(f"{c:>9,d}" for c in len_hist))
W("  tokens      " + "  ".join(f"{c:>9,d}" for c in tok_hist))
W(f"  conversations with 1 / 2 / 3 / 4+ assistant turns: {turns[1]:,d} / {turns[2]:,d} / {turns[3]:,d} / {turns[4]:,d};  with a system prompt: {n_sys:,d} ({100 * n_sys / len(tr_rows):.1f} %)")
W("")
W("FILTERS  (sft_lib.py, sft_pools.py, sft_assemble.py; per-source drop counts in sft_work/pool/<source>.stats.json)")
W("  cleaning: CRLF -> LF, control and zero-width characters removed, trailing blanks, runs of 4+ blank lines -> 2; messages stripped (a leading blank would change the role header's tokens)")
W("  length: conversations cut to their first 1-3 exchanges that fit 700-1300 tokens (per source), minimum 24 tokens, assistant answer >= 12 characters (3-4 for greetings / dolly)")
W("  language: >= 90 % ASCII and a stop-word ratio >= 12 % of the prose words (code blocks excluded)")
W("  other vendors: any mention of ChatGPT / OpenAI / GPT-3/4 / Anthropic / Claude anywhere; first-person identity statements naming Google, Meta, Microsoft, Qwen, Llama, Mistral ... ('I am', 'I was trained by ...'); 'as an AI language model'; knowledge cut-off statements; 'as an AI / as a chatbot ...' anywhere in an answer (kept only in the CoCoNot / WildGuard refusal sources and the hand-written sets); Evol-CodeAlpaca items with an A/B/C/D option list (MMLU-style questions) and the 12 items that matched the evaluation files; every document that defines a distinctive HumanEval / MBPP function name (728, mostly Evol-CodeAlpaca copies of HumanEval problems)")
W("  markup: special-token strings, [INST] / <<SYS>> / ### Instruction artifacts, HTML tags and entities (tags allowed in code sources), mojibake, unbalanced code fences, role leaks ('Human:' in an answer), escaped newlines, degenerate repetition, answers cut off mid-sentence")
W("  PII and links: e-mail addresses (except example domains), phone numbers, URLs in an answer that appear in no earlier message of the conversation (not checked for code sources)")
W("  explicit words: profanity / sexual vocabulary dropped from the general sources (not from the safety sources)")
W("  decontamination: a verbatim run of >= 13 words (10-gram index, 3 further consecutive 10-grams; runs of mostly numbers ignored) shared with the test / validation sets of HellaSwag, ARC-Easy/Challenge, PIQA, WinoGrande, MMLU, GSM8K test, MATH test, HumanEval, MBPP, LAMBADA, SciQ, OpenBookQA, TruthfulQA; IFEval: any shared 16-gram")
W("  duplicates: exact (same conversation or same user side) and near (MinHash of word 3-grams of the first user message, Jaccard >= 0.8 for short, >= 0.7 for long prompts) ACROSS sources, human-written sources kept first")
d_bench = collections.Counter()
for s, st in pool_stats.items():
    for b, n in st.get("bench_hits", {}).items():
        d_bench[b] += n
W("  decontamination drops in the pools (conversations with a >= 13-word verbatim overlap): " + ", ".join(f"{b} {n}" for b, n in d_bench.most_common()) + f"  (total {sum(d_bench.values())} of {sum(st['processed'] for st in pool_stats.values())} candidates examined)")
W("  assembly: dedup drops " + json.dumps(plan["dedup_drops"]) + "; late filter drops " + json.dumps(plan.get("late_filter_drops", {})))
W("")
W("SEEDS AND DETERMINISM")
W(f"  base seed {L.SEED}; every random choice uses random.Random(stable_seed(<purpose>, <source>)) = (SEED + crc32(parts)) & 0x7fffffff: pools ('pool', source), assembly ('assemble-shuffle', source), ('assemble-val'), ('assemble-train'), hand-written text ('handwritten').")
W(f"  validation share: every source gives its token share of {plan['val_target']:,d} validation tokens (hand-written sets: 8 % of their conversations); oversampled sources are held out BEFORE oversampling.")
W("")
W("COMMANDS (homelab, cwd /mnt/data/forgeinfer-data/sft_work, everything under nice -n 19 ionice -c2 -n7, RAYON_NUM_THREADS=1..2; the scripts are tools/sft_data/*.py of the repository)")
W("  python sft_download.py bench small smoltalk tulu ultrachat           # raw sources and benchmark test sets -> /mnt/data/forgeinfer-data/sft_raw (done by the first run)")
W("  python sft_bench_index.py                                             # sft_work/bench_index.npz: 5.27 M 10-gram hashes of the benchmark test sets")
W("  python fetch_cards.py <repo ...>                                      # dataset cards and hub metadata -> sft_work/cards/")
W("  python sft_sources.py extract_tulu                                    # the six Tulu-3 parquet files -> sft_work/extract/tulu__<source>.jsonl")
W("  python sft_handwritten.py [--creator TEXT]                            # pool/identity.jsonl, pool/safety_hw.jsonl")
W("  python sft_pools.py magpie ultrachat capybara systemchats summarize rewrite constraints oasst2 dolly everyday gsm8k hendrycks metamath orcamath tulu_gsm tulu_grade selfoss magicoder evolcode tulu_code coconot wildguard   # pool/<source>.jsonl + stats")
W("  python sft_audit.py pools [source ...]                                # scan the pools against the eval/*.jsonl files of the evaluation ladder -> pool_blacklist_*.txt (ids dropped by the assembler)")
W("  python sft_fn_scan.py --max-df 12 --write                             # documents that define a distinctive HumanEval / MBPP function name (instruction-evolved copies of HumanEval items) -> pool_blacklist_fnames.txt")
W("  python sft_assemble.py plan                                           # late filters, dedup, thinning, validation split, oversampling, shuffle -> /tmp/sft_mix2_build/*.txt + docs.tsv + plan.json")
W("  python sft_assemble.py encode                                         # /tmp/forge_sft tok-encode TOK X.txt --out X.enc.u16 --threads 2; /tmp/forge_sft chat-mask TOK X.enc.u16 --out X.u16 (counts checked against the plan)")
W("  python sft_assemble.py publish                                        # throttled copy (40 MB/s, 20 s pause) into tokens_sft_mix2/")
W("  python sft_audit.py /tmp/sft_mix2_build                               # final decontamination audit of the corpus files (bench index + eval files)")
W("  python sft_verify.py /mnt/data/forgeinfer-data/tokens_sft_mix2 --n 3000   # independent round-trip, mask and layout checks")
W("  python sft_manifest.py                                                # this file")
W("  /tmp/forge_sft = copy of ~/forgeinfer/target-dev3/release/forge (the dev build; the v1 binary does not understand bit 15)")
W("")
W("PLAN SUMMARY (plan.json): " + json.dumps({k: v for k, v in plan.items() if k not in ("dedup_drops", "late_filter_drops", "targets_m")}))
text = "\n".join(lines) + "\n"
os.makedirs(a.out, exist_ok=True)
open(os.path.join(a.out, "MANIFEST.txt"), "w", encoding="utf-8").write(text)
json.dump({"files": fcount, "sources": by_src, "plan": plan, "len_hist": len_hist, "len_edges": edges}, open(os.path.join(a.out, "manifest.json"), "w"), indent=1)
print(text)
