#!/usr/bin/env python3
"""contamination_check.py - GPT-3 style n-gram overlap audit: do the texts of an evaluation set occur in a token corpus?

What it does
    1. `index`:  every item of the evaluation sets (the Forge-1 ladder layout: EVAL/mc/*.jsonl, EVAL/hellaswag/hellaswag_val.jsonl, EVAL/lambada/*.jsonl) is turned into texts: the question or context, each
       answer alone, question + answer joined, the harness prompt as the evaluator scores it, the HellaSwag text with and without its [header]/[title] markers, the LAMBADA passage, the SciQ support
       passage ...  Each text is tokenised with the corpus tokenizer in up to four variants (as it is, with a leading space, and both again with typographic characters mapped to ASCII: a byte-level BPE
       tokenises the first word differently after a space), and EVERY window of N consecutive tokens that lies inside one text is hashed.  Hash = polynomial rolling hash mod 2^64 with a random odd
       multiplier B, window hash(i) = sum_k (x[i+k] + 1) * B^k, finished by a xor-shift multiply.  The sorted unique hashes are the index (N = 16 is the primary window, N = 10 the looser one).
       Texts that are shorter than N tokens are tested separately: the first 8 tokens (the "anchor") are hashed and a hit is extended to the whole text by an exact comparison ("short" index).
    2. `scan`:   corpus shards (uint16 little-endian token files; bit 15 can be cleared with --mask15) are read in chunks of --chunk tokens plus an overlap of 15 tokens (a window across a chunk
       boundary is seen exactly once), hashed with the SAME function (prefix sums: O(1) numpy passes per chunk whatever N is), pre-filtered by a bitmap, looked up with a binary search and
       VERIFIED against the exact token sequence of the eval window: a hash collision can never count as a hit.  Per shard the result is (window group, number of occurrences, first position).
       Reads use O_DIRECT where the OS has it (the page cache of a machine that is training is not disturbed).
    3. `report`: per benchmark and per corpus source: items with at least one verified hit (N = 16 and 10), items with a text of which at least half of the windows occur ("strong"), items too short
       for the window size and how many of those occur as a whole sequence, hits on the correct choice versus the wrong choices, the documents that hold the strong hits (a document with many items of one
       benchmark is a dump of it), the most revealing hits decoded to text, JSON + markdown reports, exclusion_spans.json (the documents with a strong hit, as token spans of the training shards) and
       clean / dirty subsets of the evaluation files (same layout and manifest, so `tools/f1_eval.sh CKPT --data DIR` can score them).

Usage (homelab; niced, one thread; ~/forgeinfer/venv/bin/python has numpy and tokenizers)
    nice -n 15 ionice -c3 python tools/contamination_check.py index  --eval-dir /mnt/data/forgeinfer-data/eval --tokenizer ~/forgeinfer/data/tokenizer_f1.json --out OUT
    nice -n 15 ionice -c3 python tools/contamination_check.py scan   --out OUT --corpus 'main=/mnt/data/forgeinfer-data/tokens_f1/*.u16' --corpus 'sft=.../tokens_sft_mix2/*.u16' --mask15 sft
    python tools/contamination_check.py report --out OUT --eval-dir EVAL --tokenizer TOK --report-dir REPORT --subset-dir SUBSETS
    python tools/contamination_check.py all    ...   (index + scan + report in one process, same options)
  `scan --part 0/2` and `--part 1/2` split the shards between two processes (balanced by size).  Results are written per shard (OUT/scan/*.json + *.npz hits + *.eos.npz document separators) and a
  finished shard is skipped on a rerun.  A corpus is LABEL=PATH where PATH is a .u16 file, a directory (its *.u16) or a glob; the source label of a shard is LABEL/<file stem without the trailing _NNN>
  (main/fwedu, cool/cool, sft/sft ...).  Speed: about 13 M tokens/s per process on the homelab i7-9700K (7.2 B tokens in 13 CPU-minutes), 112 s for the index.
  On a machine that is training from the same disk: one or two niced processes, and `--max-mb-s 20` (measured on the homelab: two processes reading 26 MB/s each raised the step time of the
  training run from 3.90 s to 4.1-4.3 s, one process to 4.0-4.1 s, a pass that read 150 MB/s to 4.7-5.0 s).  `scan --eos-only` is such a pass: it needs no index and only records document separators.
  Self test: python tools/test_contamination.py   (numpy only; uses the real tokenizer too when it finds one).

Honest limits: token-level matching sees only verbatim copies up to the typography variants above (a paraphrase, a translation, another whitespace layout or HTML escapes are invisible);
the first/last token of a text can differ in the corpus because of BPE merges across the text boundary (at most one window per text is affected); a hit says "this token sequence occurs", the
decoded context tells whether it is the benchmark item, a quote, boilerplate or a textbook that the benchmark was written from.
"""
import argparse
import glob
import gzip
import hashlib
import itertools
import json
import math
import mmap
import os
import re
import sys
import time

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("RAYON_NUM_THREADS", "1")

import numpy as np  # noqa: E402

TOOL_VERSION = 1
DEFAULT_SEED = 20261010
MASK64 = (1 << 64) - 1
MIX_C = 0xD6E8FEB86659FD93
U64 = np.uint64
U16 = np.uint16
PAGE = 4096
SHORT_MAX = 15  # the "short" index covers texts of ANCHOR .. SHORT_MAX tokens (shorter than the primary window)
DEFAULT_EVAL = "/mnt/data/forgeinfer-data/eval"
DEFAULT_TOKENIZER = os.path.expanduser("~/forgeinfer/data/tokenizer_f1.json")
KINDS = ["ctx", "end", "join", "full", "passage", "support", "raw_ctx", "raw_end", "raw_join", "blank", "block"]
KIND_ID = {k: i for i, k in enumerate(KINDS)}
# texts that get the leading-space variant (they are natural text that can follow a space in a document); "full" (harness prompt with its template) and "block" (MMLU letter block) do not
SPACE_KINDS = {"ctx", "end", "join", "passage", "support", "raw_ctx", "raw_end", "raw_join", "blank"}
CHOICE_KINDS = {"end", "join", "full", "raw_end", "raw_join"}
BENCH_ORDER = ["hellaswag", "arc_easy", "arc_challenge", "piqa", "winogrande", "openbookqa", "sciq", "sciq_support", "lambada_openai", "mmlu_cloze", "mmlu_letter"]
TYPO = str.maketrans({"‘": "'", "’": "'", "‚": "'", "‛": "'", "ʼ": "'", "“": '"', "”": '"', "„": '"', "–": "-", "—": "-", "−": "-",
                      "…": "...", " ": " ", "​": "", "﻿": ""})
RARE_MAX = 10  # a window that occurs at most this often in the whole corpus counts as "specific" (not boilerplate)
STRONG_MIN_WINDOWS = 3  # a text needs at least this many windows to be called "strong" (half of its windows found)
CONTEXT_KINDS = ("ctx", "raw_ctx", "passage", "block", "support")  # the question / context / passage texts (not an answer alone, not question + answer)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def say(msg):
    print(time.strftime("%H:%M:%S ") + msg, flush=True)


# ============================================================================== hashing
def _powers(b, n):
    a = np.empty(n, dtype=np.uint64)
    a[0] = 1
    if n > 1:
        a[1:] = np.cumprod(np.full(n - 1, b, dtype=np.uint64))  # unsigned overflow wraps: exact mod 2^64
    return a


class Hasher:
    """polynomial rolling hash of token windows mod 2^64.  hash(window) = finish(sum_k (x[k] + 1) * B^k).  `windows(prefix(x), N, m)` gives the hashes of all windows of a chunk with a few vectorised
    passes (prefix sums, then normalising the position out with the inverse powers of B); `naive` computes the same value window by window (used for the index and as the test oracle)."""

    def __init__(self, seed=DEFAULT_SEED, weak_bits=64):
        rng = np.random.default_rng(seed)
        self.seed = seed
        self.B = int(rng.integers(0, 1 << 63)) * 2 + 1  # odd: invertible mod 2^64
        self.Binv = pow(self.B, -1, 1 << 64)
        self.weak_bits = weak_bits  # tests only: keep just the top `weak_bits` bits of every hash, to force collisions
        self._n = 0
        self.pw = None
        self.ipw = None

    def ensure(self, n):
        if self._n < n:
            self.pw = _powers(self.B, n)
            self.ipw = _powers(self.Binv, n)
            self._n = n

    def finish(self, w):
        w ^= w >> U64(32)
        w *= U64(MIX_C)
        if self.weak_bits < 64:
            w &= U64((MASK64 << (64 - self.weak_bits)) & MASK64)
        return w

    def prefix(self, x):
        """cumulative sums of (x + 1) * B^i over a chunk (uint64, wraps mod 2^64)"""
        n = len(x)
        self.ensure(n)
        xs = x.astype(np.uint64)
        xs += U64(1)
        xs *= self.pw[:n]
        np.cumsum(xs, out=xs)
        return xs

    def windows(self, cs, N, m):
        """hashes of the windows x[i:i+N] for i in 0..m-1 (needs len(cs) >= N + m - 1)"""
        w = np.empty(m, dtype=np.uint64)
        w[0] = cs[N - 1]
        if m > 1:
            np.subtract(cs[N:N + m - 1], cs[:m - 1], out=w[1:])
        w *= self.ipw[:m]
        return self.finish(w)

    def naive(self, tok, starts, N):
        """(hash, contains_token_0) of the windows tok[s:s+N] for s in starts: the same function, computed per window"""
        h = np.zeros(len(starts), dtype=np.uint64)
        has0 = np.zeros(len(starts), dtype=bool)
        pwk = 1
        for k in range(N):
            t = tok[starts + k]
            h += (t.astype(np.uint64) + U64(1)) * U64(pwk)
            has0 |= t == 0
            pwk = (pwk * self.B) & MASK64
        return self.finish(h), has0


# ============================================================================== tokenizer wrappers
class HFTok:
    """the corpus tokenizer (HuggingFace tokenizers JSON); texts are encoded without special tokens"""

    def __init__(self, path):
        from tokenizers import Tokenizer

        self.path = path
        self.tok = Tokenizer.from_file(path)
        self.sha256 = sha256_file(path)

    def encode_many(self, strings):
        """(flat uint16 array of all ids, int64 array of lengths)"""
        lens = np.zeros(len(strings), dtype=np.int64)
        parts = []
        B = 20000
        for i in range(0, len(strings), B):
            encs = self.tok.encode_batch(strings[i:i + B], add_special_tokens=False)
            for j, e in enumerate(encs):
                lens[i + j] = len(e.ids)
            n = int(lens[i:i + B].sum())
            parts.append(np.fromiter(itertools.chain.from_iterable(e.ids for e in encs), dtype=np.uint16, count=n))
        return (np.concatenate(parts) if parts else np.zeros(0, np.uint16)), lens

    def decode(self, ids):
        return self.tok.decode([int(i) for i in ids], skip_special_tokens=False)


# ============================================================================== eval items -> texts
def hs_preprocess(text):
    """the lm-evaluation-harness HellaSwag preprocessing (as tools/f1_eval_data.py): ' [title]' -> '. ', bracketed markers removed, double spaces collapsed"""
    t = text.strip().replace(" [title]", ". ")
    out, depth = [], 0
    for c in t:
        if c == "[":
            depth += 1
        elif c == "]" and depth > 0:
            depth -= 1
        elif depth == 0:
            out.append(c)
    return "".join(out).replace("  ", " ")


def strip_prompt(ctx):
    """a harness prompt without its template: the MMLU header, the leading 'Question:' and the trailing 'Answer:'"""
    s = re.sub(r"^The following are (?:multiple choice )?questions (?:\(with answers\) )?about [^\n]*\.\n+", "", ctx)
    if s.startswith("Question:"):
        s = s[len("Question:"):]
    s = re.sub(r"\n?Answer:\s*$", "", s)
    return s.strip()


def _dedupe(texts):
    seen, out = set(), []
    for kind, ch, s in texts:
        s = s.replace("\x00", " ").strip()
        key = (ch if kind in CHOICE_KINDS else -1, s)  # the same string for two different choices is two texts (the choice matters for the correct-versus-wrong analysis)
        if s and key not in seen:
            seen.add(key)
            out.append((kind, ch, s))
    return out


def build_mc(bench, idx, rec):
    """items (bench, idx, label, n_choices, texts) of one record of an eval-mc file; SciQ gives two items (question + answer, and the support passage)"""
    ch = rec["choices"]
    label = int(rec["label"])
    ctx0 = ch[0]["ctx"]
    if bench == "mmlu_letter":
        return [(bench, idx, label, len(ch), _dedupe([("block", -1, strip_prompt(ctx0))]))]
    if bench == "winogrande":
        ctxs = [c["ctx"] for c in ch]
        pre = os.path.commonprefix(ctxs)
        pre = pre[:pre.rfind(" ") + 1] if " " in pre else ""  # whole words: the sentence up to the blank
        texts = [("ctx", -1, pre)]
        for c, x in enumerate(ch):
            texts.append(("join", c, x["ctx"] + x["cont"]))  # the filled-in sentence, natural text
            texts.append(("end", c, x["cont"]))
        if pre.strip():
            texts.append(("blank", -1, pre + "_" + ch[0]["cont"]))  # the dataset's own form of the sentence (with the underscore)
        return [(bench, idx, label, len(ch), _dedupe(texts))]
    out = []
    qctx = ctx0
    if bench == "sciq" and "\nQuestion:" in ctx0:
        support, q = ctx0.rsplit("\nQuestion:", 1)
        qctx = "Question:" + q
        if support.strip():
            out.append(("sciq_support", idx, label, 1, _dedupe([("support", -1, support)])))
    body = strip_prompt(qctx)
    texts = [("ctx", -1, body)]
    for c, x in enumerate(ch):
        cont = x["cont"].strip()
        qc = x["ctx"] if bench != "sciq" or "\nQuestion:" not in x["ctx"] else "Question:" + x["ctx"].rsplit("\nQuestion:", 1)[1]
        texts.append(("end", c, cont))
        texts.append(("join", c, body + " " + cont))
        texts.append(("full", c, qc + x["cont"]))  # the prompt as the evaluator scores it (identical to "join" for prompts without a template: dropped by the de-duplication)
    return [(bench, idx, label, len(ch), _dedupe(texts))] + out


def build_hellaswag(idx, rec):
    label = int(rec["label"])
    ca, cb = rec.get("ctx_a", ""), rec.get("ctx_b", "")
    pctx = hs_preprocess(f"{ca} {cb[:1].upper() + cb[1:].lower()}")
    raw_ctx = rec.get("ctx") or (ca + " " + cb)
    texts = [("ctx", -1, pctx), ("raw_ctx", -1, raw_ctx)]
    for c, e in enumerate(rec["endings"]):
        pe = hs_preprocess(e)
        texts += [("end", c, pe), ("join", c, pctx + " " + pe), ("raw_end", c, e), ("raw_join", c, raw_ctx + " " + e)]
    return [("hellaswag", idx, label, len(rec["endings"]), _dedupe(texts))]


def _jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(ln) for ln in f if ln.strip()]


def load_eval(eval_dir, only=None, log=say):
    """-> items: list of (bench, idx, label, n_choices, [(kind, choice, text)]) in the order BENCH_ORDER; idx = index of the record in its jsonl file (blank lines ignored)"""
    items = []
    mc_dir = os.path.join(eval_dir, "mc")
    if os.path.isdir(mc_dir):
        for f in sorted(os.listdir(mc_dir)):
            if f.endswith(".jsonl") and (not only or f[:-6] in only):
                recs = _jsonl(os.path.join(mc_dir, f))
                for i, r in enumerate(recs):
                    items += build_mc(f[:-6], i, r)
                log(f"  {f[:-6]}: {len(recs)} records")
    hs = os.path.join(eval_dir, "hellaswag", "hellaswag_val.jsonl")
    if os.path.isfile(hs) and (not only or "hellaswag" in only):
        recs = _jsonl(hs)
        for i, r in enumerate(recs):
            items += build_hellaswag(i, r)
        log(f"  hellaswag: {len(recs)} records")
    lam_dir = os.path.join(eval_dir, "lambada")
    if os.path.isdir(lam_dir):
        for f in sorted(os.listdir(lam_dir)):
            if f.endswith(".jsonl") and (not only or f[:-6] in only):
                recs = _jsonl(os.path.join(lam_dir, f))
                for i, r in enumerate(recs):
                    items.append((f[:-6], i, -1, 1, _dedupe([("passage", -1, r["ctx"] + r["cont"])])))
                log(f"  {f[:-6]}: {len(recs)} records")
    order = {b: i for i, b in enumerate(BENCH_ORDER)}
    items.sort(key=lambda it: (order.get(it[0], 99), it[0], it[1]))  # stable: sciq before sciq_support is irrelevant, the key is (bench, idx)
    return items


def make_variants(kind, s):
    """[(variant type, string)]: 0 as is, 1 with a leading space, 2 typography mapped to ASCII, 3 both"""
    out = [(0, s)]
    if kind in SPACE_KINDS:
        out.append((1, " " + s))
    a = s.translate(TYPO)
    if a != s:
        out.append((2, a))
        if kind in SPACE_KINDS:
            out.append((3, " " + a))
    return out


# ============================================================================== index
def _group_hashes(h):
    """sorted unique hashes, group id of every element, index of the first element of each group"""
    if len(h) == 0:
        return np.zeros(0, np.uint64), np.zeros(0, np.int64), np.zeros(0, np.int64)
    order = np.argsort(h, kind="stable")
    hs = h[order]
    new = np.empty(len(hs), dtype=bool)
    new[0] = True
    np.not_equal(hs[1:], hs[:-1], out=new[1:])
    gid_sorted = np.cumsum(new) - 1
    grp = np.empty(len(h), dtype=np.int64)
    grp[order] = gid_sorted
    return hs[new], grp, order[new]


def _enumerate_windows(v_start, v_len, N):
    ok = np.flatnonzero(v_len >= N)
    cnt = v_len[ok] - N + 1
    tot = int(cnt.sum())
    if tot == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    rep_v = np.repeat(ok, cnt)
    offs = np.arange(tot, dtype=np.int64) - np.repeat(np.cumsum(cnt) - cnt, cnt)
    return v_start[rep_v] + offs, rep_v


def build_index(eval_dir, tok, out_dir, Ns=(16, 10), anchor=8, hasher=None, only=None, log=say):
    t0 = time.time()
    hasher = hasher or Hasher()
    idx_dir = os.path.join(out_dir, "index")
    os.makedirs(idx_dir, exist_ok=True)
    log(f"index: reading the evaluation sets of {eval_dir}")
    items = load_eval(eval_dir, only=only, log=log)
    t_item, t_kind, t_choice, t_str = [], [], [], []
    for ii, (bench, idx, label, nch, texts) in enumerate(items):
        for kind, ch, s in texts:
            t_item.append(ii)
            t_kind.append(KIND_ID[kind])
            t_choice.append(ch)
            t_str.append(s)
    T = len(t_str)
    vstr, vtext, vtype = [], [], []
    for ti in range(T):
        for vt, vs in make_variants(KINDS[t_kind[ti]], t_str[ti]):
            vstr.append(vs)
            vtext.append(ti)
            vtype.append(vt)
    log(f"index: {len(items)} items, {T} texts, {len(vstr)} text variants; tokenising")
    tok_flat, v_len = tok.encode_many(vstr)
    v_start = np.concatenate([[0], np.cumsum(v_len)[:-1]]).astype(np.int64)
    v_text = np.asarray(vtext, dtype=np.int32)
    v_type = np.asarray(vtype, dtype=np.int8)
    t_item = np.asarray(t_item, dtype=np.int32)
    t_kind = np.asarray(t_kind, dtype=np.int8)
    t_choice = np.asarray(t_choice, dtype=np.int8)
    log(f"index: {len(tok_flat)} tokens in the variants")
    # longest variant per text (for the "too short" statistics)
    t_maxlen = np.zeros(T, dtype=np.int32)
    np.maximum.at(t_maxlen, v_text, v_len.astype(np.int32))
    arrays = {}
    meta_n = {}
    for N in Ns:
        starts, rep_v = _enumerate_windows(v_start, v_len, N)
        h, has0 = hasher.naive(tok_flat, starts, N)
        keep = ~has0
        starts, rep_v, h = starts[keep], rep_v[keep], h[keep]
        uniq, grp, first = _group_hashes(h)
        G = len(uniq)
        rep_start = starts[first]
        # the members of a group must be identical token sequences (a collision between two different eval windows would corrupt the index): check in blocks
        ar = np.arange(N)
        bad = 0
        for a in range(0, len(starts), 2_000_000):
            sl = slice(a, a + 2_000_000)
            w = tok_flat[starts[sl][:, None] + ar]
            r = tok_flat[rep_start[grp[sl]][:, None] + ar]
            bad += int((w != r).any(axis=1).sum())
        if bad:
            raise SystemExit(f"index N={N}: {bad} eval windows share a hash with a different token sequence: use another --seed")
        keys = v_text[rep_v].astype(np.int64) * G + grp
        uk = np.unique(keys)
        tt, gg = uk // G, uk % G
        t_ptr = np.searchsorted(tt, np.arange(T + 1)).astype(np.int64)
        arrays[f"n{N}_uniq"] = uniq
        arrays[f"n{N}_rep"] = rep_start.astype(np.int64)
        arrays[f"n{N}_tptr"] = t_ptr
        arrays[f"n{N}_tgrp"] = gg.astype(np.int32)
        meta_n[str(N)] = {"windows": int(len(starts)), "windows_with_token0_dropped": int(has0.sum()), "groups": int(G), "text_window_pairs": int(len(uk)),
                          "texts_with_windows": int((np.diff(t_ptr) > 0).sum())}
        log(f"index N={N}: {len(starts)} windows, {G} distinct, {meta_n[str(N)]['texts_with_windows']} of {T} texts have windows")
        del starts, rep_v, h, grp, keys, uk, tt, gg
    # short index: the first `anchor` tokens of every variant that is shorter than the primary window
    smax = max(Ns) - 1
    elig = np.flatnonzero((v_len >= anchor) & (v_len <= smax))
    h, has0 = hasher.naive(tok_flat, v_start[elig], anchor)
    uniq, grp, first = _group_hashes(h)
    order = np.argsort(grp, kind="stable")
    sg_ptr = np.searchsorted(grp[order], np.arange(len(uniq) + 1)).astype(np.int64)
    arrays["short_uniq"] = uniq
    arrays["short_rep"] = v_start[elig][first].astype(np.int64)
    arrays["short_gptr"] = sg_ptr
    arrays["short_gvar"] = order.astype(np.int32)  # index into short_vid
    arrays["short_vid"] = elig.astype(np.int64)  # variant ids
    meta_short = {"anchor": anchor, "max_len": smax, "variants": int(len(elig)), "groups": int(len(uniq))}
    log(f"index short: {len(elig)} variants of {anchor}..{smax} tokens, {len(uniq)} distinct anchors")
    bench_names = sorted({it[0] for it in items}, key=lambda b: (BENCH_ORDER.index(b) if b in BENCH_ORDER else 99, b))
    arrays["tokens"] = tok_flat
    arrays.update(v_start=v_start, v_len=v_len.astype(np.int32), v_text=v_text, v_type=v_type, t_item=t_item, t_kind=t_kind, t_choice=t_choice, t_maxlen=t_maxlen,
                  item_bench=np.asarray([bench_names.index(it[0]) for it in items], dtype=np.int16),
                  item_idx=np.asarray([it[1] for it in items], dtype=np.int32), item_label=np.asarray([it[2] for it in items], dtype=np.int16),
                  item_nch=np.asarray([it[3] for it in items], dtype=np.int16))
    np.savez(os.path.join(idx_dir, "index.npz"), **arrays)
    with gzip.open(os.path.join(idx_dir, "texts.jsonl.gz"), "wt", encoding="utf-8") as f:
        for s in t_str:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    meta = {"tool_version": TOOL_VERSION, "index_id": hashlib.sha256(f"{time.time()}{os.getpid()}{hasher.seed}".encode()).hexdigest()[:16], "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "eval_dir": os.path.abspath(eval_dir), "tokenizer": getattr(tok, "path", None), "tokenizer_sha256": getattr(tok, "sha256", None), "seed": hasher.seed, "weak_bits": hasher.weak_bits,
            "Ns": list(Ns), "anchor": anchor, "items": len(items), "texts": T, "variants": len(vstr), "tokens": int(len(tok_flat)), "benchmarks": bench_names,
            "items_per_bench": {b: sum(1 for it in items if it[0] == b) for b in bench_names}, "n": meta_n, "short": meta_short, "seconds": round(time.time() - t0, 1)}
    with open(os.path.join(idx_dir, "index.json"), "w") as f:
        json.dump(meta, f, indent=1)
    log(f"index: done in {time.time() - t0:.0f} s")
    return meta


class NGramIndex:
    """what the scan needs of one window length: sorted hashes, the exact tokens of one representative window per hash, a bitmap pre-filter"""

    def __init__(self, name, N, uniq, rep, bm_bits):
        self.name, self.N, self.span = name, N, N
        self.uniq, self.rep = uniq, rep
        self.bm_bits = bm_bits
        self.shift = 64 - bm_bits
        self.bm = np.zeros(1 << bm_bits, dtype=np.uint8)
        self.bm[(uniq >> U64(self.shift)).astype(np.intp)] = 1


def load_index(out_dir, bitmap_bits=None, log=say):
    idx_dir = os.path.join(out_dir, "index")
    meta = json.load(open(os.path.join(idx_dir, "index.json")))
    z = np.load(os.path.join(idx_dir, "index.npz"))
    tok = z["tokens"]
    hasher = Hasher(meta["seed"], meta.get("weak_bits", 64))
    ixs = []
    for N in meta["Ns"]:
        uniq = z[f"n{N}_uniq"]
        rs = z[f"n{N}_rep"]
        rep = np.empty((len(uniq), N), dtype=np.uint16)
        ar = np.arange(N)
        for a in range(0, len(uniq), 1_000_000):
            rep[a:a + 1_000_000] = tok[rs[a:a + 1_000_000][:, None] + ar]
        bits = bitmap_bits or min(28, max(18, int(math.ceil(math.log2(max(2, len(uniq))))) + 5))
        ixs.append(NGramIndex(f"n{N}", N, uniq, rep, bits))
    anchor = meta["short"]["anchor"]
    uniq = z["short_uniq"]
    rs = z["short_rep"]
    ar = np.arange(anchor)
    rep = tok[rs[:, None] + ar] if len(uniq) else np.zeros((0, anchor), np.uint16)
    sx = NGramIndex("short", anchor, uniq, rep, min(24, max(16, int(math.ceil(math.log2(max(2, len(uniq))))) + 5)))
    sx.span = meta["short"]["max_len"] + 1
    vid = z["short_vid"]
    vs, vl = z["v_start"][vid], z["v_len"][vid]
    pad = np.full((len(vid), sx.span), 0xFFFF, dtype=np.uint16)
    for k in range(sx.span):
        m = vl > k
        pad[m, k] = tok[vs[m] + k]
    sx.var_tok, sx.var_len, sx.gptr, sx.gvar, sx.vid = pad, vl.astype(np.int64), z["short_gptr"], z["short_gvar"], vid
    ixs.append(sx)
    log("index loaded: " + ", ".join(f"{ix.name}: {len(ix.uniq)} windows (bitmap 2^{ix.bm_bits})" for ix in ixs))
    return meta, hasher, ixs


# ============================================================================== corpus reading and scanning
class Reader:
    """reads token chunks of a uint16 file; O_DIRECT on Linux (no page-cache pollution), plain reads otherwise"""

    def __init__(self, path, max_tokens, direct=True):
        self.path = path
        self.size = os.path.getsize(path) // 2
        self.direct = False
        self.dfd = None  # the O_DIRECT descriptor (reads must be page aligned)
        self.pfd = None  # a plain descriptor, opened when needed
        if direct and hasattr(os, "O_DIRECT") and hasattr(os, "preadv"):
            try:
                self.dfd = os.open(path, os.O_RDONLY | os.O_DIRECT)
                self.buf = mmap.mmap(-1, ((max_tokens * 2 + PAGE - 1) // PAGE) * PAGE + PAGE)
                self.direct = True
            except OSError:
                if self.dfd is not None:
                    os.close(self.dfd)
                self.dfd = None

    def _plain(self):
        if self.pfd is None:
            self.pfd = os.open(self.path, os.O_RDONLY | getattr(os, "O_BINARY", 0))
        return self.pfd

    def read(self, start, n):
        n = max(0, min(n, self.size - start))
        if n == 0:
            return np.zeros(0, np.uint16)
        off = start * 2
        if self.direct and off % PAGE == 0:
            want = ((n * 2 + PAGE - 1) // PAGE) * PAGE
            view = memoryview(self.buf)
            got = 0
            try:
                while got < want and off + got < self.size * 2:
                    r = os.preadv(self.dfd, [view[got:want]], off + got)
                    if r <= 0:
                        break
                    got += r
                return np.frombuffer(self.buf, dtype="<u2", count=min(got // 2, n))
            except OSError:  # this file system does not like O_DIRECT after all: use buffered reads from now on
                self.direct = False
        if hasattr(os, "pread"):
            fd = self._plain()
            parts, got = [], 0
            while got < n * 2:
                d = os.pread(fd, n * 2 - got, off + got)
                if not d:
                    break
                parts.append(d)
                got += len(d)
            if hasattr(os, "posix_fadvise"):
                os.posix_fadvise(fd, off, n * 2, os.POSIX_FADV_DONTNEED)  # only pages that nobody maps: the page cache of the other readers of the file is left alone
            return np.frombuffer(b"".join(parts), dtype="<u2")
        return np.fromfile(self.path, dtype="<u2", count=n, offset=off)

    def close(self):
        for a in ("dfd", "pfd"):
            if getattr(self, a) is not None:
                os.close(getattr(self, a))
                setattr(self, a, None)


class _Acc:
    """per-file accumulator of (group, occurrences, first position)"""

    def __init__(self):
        self.g, self.p = [], []

    def add(self, g, pos):
        if len(g):
            self.g.append(g)
            self.p.append(pos)

    def result(self):
        if not self.g:
            z = np.zeros(0, np.int64)
            return z, z, z
        g = np.concatenate(self.g)
        p = np.concatenate(self.p)
        order = np.argsort(g, kind="stable")  # positions are increasing: the first of a run is the first occurrence
        g, p = g[order], p[order]
        new = np.empty(len(g), dtype=bool)
        new[0] = True
        np.not_equal(g[1:], g[:-1], out=new[1:])
        starts = np.flatnonzero(new)
        cnt = np.diff(np.append(starts, len(g)))
        return g[starts].astype(np.int64), cnt.astype(np.int64), p[starts].astype(np.int64)


def scan_array_chunk(x, base, hasher, cs, ixs, chunk, accs, stats, ar):
    """one chunk: x = tokens[base : base + len(x)] (len(x) <= chunk + overlap); windows starting in [base, base + chunk) are examined"""
    L = len(x)
    for ix in ixs:
        N = ix.N
        m = min(chunk, L - N + 1)
        if m <= 0 or len(ix.uniq) == 0:
            continue
        W = hasher.windows(cs, N, m)
        key = (W >> U64(ix.shift)).view(np.intp)
        cand = np.flatnonzero(ix.bm.take(key, mode="clip"))
        st = stats[ix.name]
        st["bitmap"] += len(cand)
        if len(cand) == 0:
            continue
        hc = W[cand]
        j = np.searchsorted(ix.uniq, hc)
        np.minimum(j, len(ix.uniq) - 1, out=j)
        ok = ix.uniq[j] == hc
        cand, j = cand[ok], j[ok]
        st["hash_match"] += len(cand)
        if len(cand) == 0:
            continue
        win = x[cand[:, None] + ar[:N]]
        eq = (win == ix.rep[j]).all(axis=1)
        st["collisions"] += int((~eq).sum())
        cand, j = cand[eq], j[eq]
        if ix.name != "short":
            st["verified"] += len(cand)
            accs[ix.name].add(j, base + cand)
            continue
        # short texts: extend the verified anchor to the whole text (exact comparison, padding = 0xFFFF never equals a token)
        ptr0, ptr1 = ix.gptr[j], ix.gptr[j + 1]
        nv = ptr1 - ptr0
        if nv.sum() == 0:
            continue
        pos = np.repeat(cand, nv)
        vi = ix.gvar[np.repeat(ptr0, nv) + (np.arange(int(nv.sum())) - np.repeat(np.cumsum(nv) - nv, nv))]
        span = ix.span
        take = pos[:, None] + ar[:span]
        w = np.full(take.shape, 0xFFFF, dtype=np.uint16)
        inside = take < L
        w[inside] = x[take[inside]]
        lens = ix.var_len[vi]
        eq = ((w == ix.var_tok[vi]) | (ar[:span][None, :] >= lens[:, None])).all(axis=1)
        st["verified"] += int(eq.sum())
        accs["short"].add(ix.vid[vi[eq]], base + pos[eq])  # global variant ids


def scan_file(path, ixs, hasher, chunk=1 << 24, mask15=False, direct=True, max_mb_s=0.0, log=None):
    """-> ({index name: (groups, counts, first positions)}, stats, positions of the document separator token 0).  With no index (`ixs` empty) only the separators are collected (fast)."""
    span = max([ix.span for ix in ixs] or [1])
    overlap = span - 1
    if ixs:
        hasher.ensure(chunk + overlap + 1)
    size = os.path.getsize(path) // 2
    accs = {ix.name: _Acc() for ix in ixs}
    stats = {ix.name: {"bitmap": 0, "hash_match": 0, "collisions": 0, "verified": 0} for ix in ixs}
    ar = np.arange(span)
    rd = Reader(path, chunk + overlap, direct=direct)
    t0 = time.time()
    maxtok = 0
    eos = []
    try:
        for base in range(0, size, chunk):
            tc = time.time()
            x = rd.read(base, chunk + overlap)
            if len(x):
                maxtok = max(maxtok, int(x.max()))
            if mask15:
                x = x & U16(0x7FFF)
            eos.append(np.flatnonzero(x[:chunk] == 0) + base)
            if ixs:
                cs = hasher.prefix(x)
                scan_array_chunk(x, base, hasher, cs, ixs, chunk, accs, stats, ar)
            if max_mb_s > 0:
                need = len(x) * 2 / 1e6 / max_mb_s - (time.time() - tc)
                if need > 0:
                    time.sleep(need)
            if log and (base // chunk) % 8 == 7:
                log(f"    {os.path.basename(path)}: {min(base + chunk, size) / size * 100:.0f} %  ({(min(base + chunk, size)) / max(1e-9, time.time() - t0) / 1e6:.2f} M tokens/s)")
    finally:
        rd.close()
    return ({n: a.result() for n, a in accs.items()}, {"tokens": int(size), "seconds": round(time.time() - t0, 2), "max_token": maxtok, "ngrams": stats},
            (np.concatenate(eos) if eos else np.zeros(0, np.int64)).astype(np.int64))


def source_of(label, path):
    stem = os.path.splitext(os.path.basename(path))[0]
    return f"{label}/{re.sub(r'_?[0-9]+$', '', stem) or stem}"


def expand_corpora(specs):
    """['LABEL=PATH', ...] -> [(label, file)] (PATH: a .u16 file, a directory or a glob)"""
    out = []
    for s in specs:
        label, _, p = s.partition("=")
        if not p:
            raise SystemExit(f"--corpus {s}: expected LABEL=PATH")
        paths = sorted(glob.glob(os.path.expanduser(p))) if any(c in p for c in "*?[") else [os.path.expanduser(p)]
        for q in paths:
            if os.path.isdir(q):
                out += [(label, f) for f in sorted(glob.glob(os.path.join(q, "*.u16")))]
            elif os.path.isfile(q):
                out.append((label, q))
            else:
                raise SystemExit(f"--corpus {s}: {q} does not exist")
    return out


def assign_part(files, part):
    """greedy balance by size: files sorted by size (largest first) go to the part with the least load"""
    i, m = part
    loads = [0] * m
    mine = []
    for lab, f in sorted(files, key=lambda lf: -os.path.getsize(lf[1])):
        k = loads.index(min(loads))
        loads[k] += os.path.getsize(f)
        if k == i:
            mine.append((lab, f))
    return sorted(mine, key=lambda lf: lf[1])


def _eos_valid(epath, size):
    try:
        with np.load(epath) as z:
            return int(z["size"]) == size
    except Exception:  # noqa: BLE001
        return False


def run_scan(out_dir, corpora, part=(0, 1), chunk=1 << 24, mask15_labels=(), direct=True, max_mb_s=0.0, bitmap_bits=None, eos_only=False, log=say):
    """scan the corpus files of this part; per file: OUT/scan/LABEL__NAME.json + .npz (hits) + .eos.npz (document separator positions).  eos_only: no hashing, only the separators (a fast pass for
    results that were made without them)"""
    if eos_only:
        meta, hasher, ixs = {"index_id": None}, Hasher(), []
    else:
        meta, hasher, ixs = load_index(out_dir, bitmap_bits, log)
    files = expand_corpora(corpora)
    mine = assign_part(files, part)
    sdir = os.path.join(out_dir, "scan")
    os.makedirs(sdir, exist_ok=True)
    log(f"scan: part {part[0]}/{part[1]}: {len(mine)} of {len(files)} files, {sum(os.path.getsize(f) for _, f in mine) / 1e9:.2f} GB" + (" (document separators only)" if eos_only else ""))
    t0 = time.time()
    for lab, f in mine:
        src = source_of(lab, f)
        key = f"{lab}__{os.path.basename(f)}"
        jpath = os.path.join(sdir, key + ".json")
        epath = os.path.join(sdir, key + ".eos.npz")
        ntok = os.path.getsize(f) // 2
        have_hits = False
        if os.path.exists(jpath) and not eos_only:
            try:
                old = json.load(open(jpath))
                have_hits = old.get("index_id") == meta["index_id"] and old.get("bytes") == os.path.getsize(f) and old.get("chunk") == chunk
            except Exception:  # noqa: BLE001
                pass
        have_eos = _eos_valid(epath, ntok)
        if (eos_only and have_eos) or (have_hits and have_eos):
            log(f"scan: {key} already done, skipped")
            continue
        if have_hits or eos_only:  # only the separators are missing
            _, st, eos = scan_file(f, [], hasher, chunk=chunk, mask15=lab in mask15_labels, direct=direct, max_mb_s=max_mb_s, log=log)
            np.savez(epath, eos=eos, size=ntok)
            log(f"scan: {key}: {len(eos)} document separators in {st['seconds']:.0f} s")
            continue
        res, st, eos = scan_file(f, ixs, hasher, chunk=chunk, mask15=lab in mask15_labels, direct=direct, max_mb_s=max_mb_s, log=log)
        z = {}
        for name, (g, c, p) in res.items():
            z[f"{name}_g"], z[f"{name}_c"], z[f"{name}_p"] = g, c, p
        np.savez(os.path.join(sdir, key + ".npz"), **z)
        np.savez(epath, eos=eos, size=ntok)
        st.update({"file": f, "label": lab, "source": src, "bytes": os.path.getsize(f), "index_id": meta["index_id"], "chunk": chunk, "mask15": lab in mask15_labels, "documents": int(len(eos)),
                   "distinct_hit_windows": {n: int(len(r[0])) for n, r in res.items()}})
        with open(jpath, "w") as fh:
            json.dump(st, fh, indent=1)
        log(f"scan: {key}: {st['tokens'] / 1e6:.1f} M tokens in {st['seconds']:.0f} s, hit windows " + ", ".join(f"{n} {len(r[0])}" for n, r in res.items()) + ", collisions rejected " +
            ", ".join(f"{n} {st['ngrams'][n]['collisions']}" for n in st["ngrams"]) + f", max token {st['max_token']}")
    log(f"scan: part {part[0]}/{part[1]} finished in {time.time() - t0:.0f} s")


# ============================================================================== report
def _fmt_pct(a, b):
    return f"{a} ({100.0 * a / b:.2f} %)" if b else f"{a}"


class Results:
    """the aggregated scan results of one index: per N, occurrences per window group and which source contains it"""

    def __init__(self, out_dir, log=say):
        self.out_dir = out_dir
        self.meta = json.load(open(os.path.join(out_dir, "index", "index.json")))
        z = np.load(os.path.join(out_dir, "index", "index.npz"))
        self.z = {k: z[k] for k in z.files if k != "tokens"}
        self.tokens = z["tokens"]
        with gzip.open(os.path.join(out_dir, "index", "texts.jsonl.gz"), "rt", encoding="utf-8") as f:
            self.t_str = [json.loads(ln) for ln in f]
        self.files = []
        for jp in sorted(glob.glob(os.path.join(out_dir, "scan", "*.json"))):
            st = json.load(open(jp))
            if st.get("index_id") != self.meta["index_id"]:
                log(f"report: ignoring {jp} (made with another index)")
                continue
            st["_npz"] = jp[:-5] + ".npz"
            ep = jp[:-5] + ".eos.npz"
            st["_eos"] = None
            if os.path.exists(ep) and _eos_valid(ep, st["tokens"]):
                with np.load(ep) as ez:
                    st["_eos"] = ez["eos"]
            self.files.append(st)
        self.sources = sorted({st["source"] for st in self.files})
        self.names = [f"n{N}" for N in self.meta["Ns"]]
        self.cnt, self.found_src, self.loc_file, self.loc_pos = {}, {}, {}, {}
        for n in self.names:
            G = len(self.z[f"{n}_uniq"])
            self.cnt[n] = np.zeros(G, dtype=np.int64)
            self.found_src[n] = {s: np.zeros(G, dtype=bool) for s in self.sources}
            self.loc_file[n] = np.full(G, -1, dtype=np.int32)
            self.loc_pos[n] = np.zeros(G, dtype=np.int64)
        V = len(self.z["v_len"])
        self.short_cnt = np.zeros(V, dtype=np.int64)
        self.short_src = {s: np.zeros(V, dtype=bool) for s in self.sources}
        self.short_file = np.full(V, -1, dtype=np.int32)
        self.short_pos = np.zeros(V, dtype=np.int64)
        for fi, st in enumerate(self.files):
            with np.load(st["_npz"]) as r:
                r = {k: r[k] for k in r.files}
            for n in self.names:
                g, c, p = r[f"{n}_g"], r[f"{n}_c"], r[f"{n}_p"]
                self.cnt[n][g] += c
                self.found_src[n][st["source"]][g] = True
                new = self.loc_file[n][g] < 0
                self.loc_file[n][g[new]] = fi
                self.loc_pos[n][g[new]] = p[new]
            g, c, p = r["short_g"], r["short_c"], r["short_p"]
            self.short_cnt[g] += c
            self.short_src[st["source"]][g] = True
            new = self.short_file[g] < 0
            self.short_file[g[new]] = fi
            self.short_pos[g[new]] = p[new]
        self.n_items = len(self.z["item_idx"])
        self.n_texts = len(self.t_str)


def _range_sum(a, ptr):
    cs = np.concatenate([[0], np.cumsum(a, dtype=np.int64)])
    return cs[ptr[1:]] - cs[ptr[:-1]]


def analyse(R, log=say):
    z = R.z
    I, T = R.n_items, R.n_texts
    t_item, t_kind = z["t_item"], z["t_kind"]
    it_ptr = np.searchsorted(t_item, np.arange(I + 1))  # texts are generated item by item
    item_bench = z["item_bench"]
    out = {"per_N": {}}
    natural = t_kind != KIND_ID["full"]
    for n in R.names:
        N = int(n[1:])
        tptr, tgrp = z[f"{n}_tptr"], z[f"{n}_tgrp"].astype(np.int64)
        nwin = np.diff(tptr)
        cnt = R.cnt[n]
        found = cnt > 0
        rare = found & (cnt <= RARE_MAX)
        nf = _range_sum(found[tgrp], tptr)
        nf_rare = _range_sum(rare[tgrp], tptr)
        nf_src = {s: _range_sum(R.found_src[n][s][tgrp], tptr) for s in R.sources}
        cov = np.where(nwin > 0, nf / np.maximum(nwin, 1), 0.0)
        strong_t = (nwin >= STRONG_MIN_WINDOWS) & (nf * 2 >= nwin)
        # union coverage per item (the item's distinct windows over all of its texts)
        pair_item = np.repeat(t_item, nwin).astype(np.int64)
        G = len(cnt)
        uk = np.unique(pair_item * G + tgrp)
        ui, ug = uk // G, uk % G
        n_union = np.bincount(ui, minlength=I)
        f_union = np.bincount(ui, weights=found[ug].astype(np.float64), minlength=I)
        strict = (n_union >= STRONG_MIN_WINDOWS) & (f_union * 2 >= n_union)
        # item-specific windows: a window that occurs in the texts of exactly one item of the same benchmark (a template or preamble shared by many items is not specific to any of them)
        k2 = item_bench[ui].astype(np.int64) * G + ug
        uk2, c2 = np.unique(k2, return_counts=True)
        text_bench = item_bench[t_item].astype(np.int64)
        kt = np.repeat(text_bench, nwin) * G + tgrp
        spec_pair = c2[np.searchsorted(uk2, kt)] == 1
        nf_spec = _range_sum(found[tgrp] & spec_pair, tptr)
        out["per_N"][n] = dict(N=N, nwin=nwin, nf=nf, nf_rare=nf_rare, nf_spec=nf_spec, spec_pair=spec_pair, nf_src=nf_src, cov=cov, strong_t=strong_t, n_union=n_union, f_union=f_union, strict=strict,
                               it_ptr=it_ptr, natural_testable=_range_sum((nwin > 0) & natural, it_ptr) > 0)
    # short index: whole-text matches of texts shorter than the window
    sv = R.z["short_vid"]
    vtext = z["v_text"]
    sh_found_t = np.zeros(T, dtype=bool)
    sh_found_t[vtext[sv[R.short_cnt[sv] > 0]]] = True
    sh_rare_t = np.zeros(T, dtype=bool)  # a whole text that occurs at most RARE_MAX times in all variants together
    tot = np.zeros(T, dtype=np.int64)
    np.add.at(tot, vtext[sv], R.short_cnt[sv])
    sh_rare_t[(tot > 0) & (tot <= RARE_MAX)] = True
    sh_elig_t = np.zeros(T, dtype=bool)
    sh_elig_t[vtext[sv]] = True
    sh_src = {}
    for s in R.sources:
        a = np.zeros(T, dtype=bool)
        a[vtext[sv[R.short_src[s][sv]]]] = True
        sh_src[s] = a
    out["short"] = dict(found_t=sh_found_t, found_rare_t=sh_rare_t, elig_t=sh_elig_t, src=sh_src)
    out["it_ptr"] = it_ptr
    out["natural"] = natural
    return out


def choice_analysis(R, A, n):
    """for MC items and HellaSwag: windows that belong to one choice only (not to the question, not to another choice): found for the correct choice, for wrong choices, or both"""
    z = R.z
    t_choice = z["t_choice"]
    tptr, tgrp = z[f"{n}_tptr"], z[f"{n}_tgrp"]
    found = R.cnt[n] > 0
    it_ptr = A["it_ptr"]
    names = R.meta["benchmarks"]
    res = {}
    for ii in range(R.n_items):
        b = names[int(z["item_bench"][ii])]
        nch = int(z["item_nch"][ii])
        if nch < 2:
            continue
        t0, t1 = int(it_ptr[ii]), int(it_ptr[ii + 1])
        base, per = set(), {}
        for t in range(t0, t1):
            s = tgrp[tptr[t]:tptr[t + 1]]
            c = int(t_choice[t])
            if c < 0:
                base.update(s.tolist())
            else:
                per.setdefault(c, set()).update(s.tolist())
        if len(per) < 2:
            continue
        label = int(z["item_label"][ii])
        spec = {}
        for c, gs in per.items():
            others = set().union(*[v for k, v in per.items() if k != c])
            spec[c] = [g for g in gs if g not in base and g not in others]
        r = res.setdefault(b, {"items": 0, "with_specific_windows": 0, "correct_hit": 0, "wrong_hit": 0, "wrong_choices": 0, "only_correct": 0, "only_wrong": 0, "both": 0, "any": 0})
        r["items"] += 1
        hits = {c: bool(len(g) and found[np.asarray(g)].any()) for c, g in spec.items()}
        if label in hits:
            r["with_specific_windows"] += 1
            r["correct_hit"] += int(hits[label])
            wr = [c for c in hits if c != label]
            r["wrong_choices"] += len(wr)
            r["wrong_hit"] += sum(hits[c] for c in wr)
            anyw = any(hits[c] for c in wr)
            if hits[label] or anyw:
                r["any"] += 1
                r["only_correct"] += int(hits[label] and not anyw)
                r["only_wrong"] += int((not hits[label]) and anyw)
                r["both"] += int(hits[label] and anyw)
    return res


def doc_span(path, pos, size, block=1 << 16, limit=1 << 25):
    """(start, end) token positions of the document (separator token 0 ends a document) that contains the token at pos, found by reading blocks around it; end = position of the separator"""
    lo = pos
    start = 0
    while lo > 0:
        a = max(0, lo - block)
        x = np.fromfile(path, dtype="<u2", count=lo - a, offset=2 * a) & U16(0x7FFF)
        z = np.flatnonzero(x == 0)
        if len(z):
            start = a + int(z[-1]) + 1
            break
        lo = a
        if pos - lo > limit:
            start = lo
            break
    hi = pos
    end = size
    while hi < size:
        b = min(size, hi + block)
        x = np.fromfile(path, dtype="<u2", count=b - hi, offset=2 * hi) & U16(0x7FFF)
        z = np.flatnonzero(x == 0)
        if len(z):
            end = hi + int(z[0])
            break
        hi = b
        if hi - pos > limit:
            end = hi
            break
    return start, end


def doc_of(st, pos):
    """the document of a file (a result record of Results.files) that contains the token at pos: from the recorded separator positions when there are some, otherwise by reading around pos"""
    eos = st.get("_eos")
    if eos is not None:
        d = int(np.searchsorted(eos, pos, side="right"))
        return (int(eos[d - 1]) + 1 if d > 0 else 0), (int(eos[d]) if d < len(eos) else int(st["tokens"]))
    return doc_span(st["file"], pos, int(st["tokens"]))


def doc_analysis(R, A, n, tok=None, min_items=3, top=5):
    """the documents that hold the strong hits (a text of which at least half of the windows occur): per benchmark the number of distinct documents and the documents that hold many different items of
    that benchmark (a benchmark dump, an exam page, a quiz aggregator).  Those are the clearest cases of contamination; a single item per document is an isolated copy of one question or passage.
    One window per strong text is located (the rarest one that occurs), so the cost is a few hundred small reads even when the separator positions were not recorded."""
    z = R.z
    a = A["per_N"][n]
    tptr, tgrp = z[f"{n}_tptr"], z[f"{n}_tgrp"].astype(np.int64)
    cnt = R.cnt[n]
    names = R.meta["benchmarks"]
    t_item, item_bench, item_idx = z["t_item"], z["item_bench"], z["item_idx"]
    per = {}
    for t in np.flatnonzero(a["strong_t"]):
        gs = tgrp[tptr[t]:tptr[t + 1]]
        fg = gs[cnt[gs] > 0]
        if len(fg) == 0:
            continue
        g = int(fg[np.argmin(cnt[fg])])
        fi, pos = int(R.loc_file[n][g]), int(R.loc_pos[n][g])
        start, end = doc_of(R.files[fi], pos)
        it = int(t_item[t])
        d = per.setdefault(names[int(item_bench[it])], {}).setdefault((fi, start), {"end": end, "items": set()})
        d["items"].add(int(item_idx[it]))
    res = {"per_benchmark": {}, "all_documents": []}
    for b, docs in per.items():
        for (fi, start), d in sorted(docs.items()):  # every located document: the spans to blank out of a training pool if one wants the strong hits gone
            st = R.files[fi]
            res["all_documents"].append({"benchmark": b, "source": st["source"], "file": st["file"], "start": start, "end": d["end"], "tokens": d["end"] - start, "items": sorted(d["items"])})
        ordered = sorted(docs.items(), key=lambda kv: -len(kv[1]["items"]))
        in_big = set()
        for (fi, start), d in docs.items():
            if len(d["items"]) >= 5:
                in_big |= d["items"]
        entry = {"documents_with_strong_items": len(docs), "documents_with_min_items": sum(1 for d in docs.values() if len(d["items"]) >= min_items),
                 "documents_with_10_items": sum(1 for d in docs.values() if len(d["items"]) >= 10), "items_in_documents_with_5_or_more_items": len(in_big), "top": []}
        for (fi, start), d in ordered[:top]:
            if len(d["items"]) < min_items:
                break
            st = R.files[fi]
            snippet = ""
            if tok is not None:
                x = np.fromfile(st["file"], dtype="<u2", count=min(80, d["end"] - start), offset=2 * start) & U16(0x7FFF)
                snippet = tok.decode(x)
            e = {"source": st["source"], "file": os.path.basename(st["file"]), "start": start, "tokens": d["end"] - start, "items": len(d["items"]), "start_of_document": snippet}
            if len(d["items"]) >= 10 and b in ("hellaswag", "arc_easy", "arc_challenge", "piqa", "winogrande", "openbookqa", "sciq", "mmlu_cloze") and len(entry["top"]) < 2:
                e["exact"] = doc_exact(R, st, start, d["end"], b)  # a dump: count its items exactly, with their answers
            entry["top"].append(e)
        res["per_benchmark"][b] = entry
    return res


def present_flags(R, A):
    """items of which a text is (half) present: `context` = the question / context / passage text is strong at 16 or 10 tokens or occurs whole as a short text that is rare; `any` = any natural text strong at 16 or 10;
    `present` = their union without the whole-text matches of short ANSWER strings (a 10-token answer such as a list of planets occurs everywhere)"""
    z = R.z
    it_ptr = A["it_ptr"]
    t_kind = z["t_kind"]
    isctx = np.isin(t_kind, [KIND_ID[k] for k in CONTEXT_KINDS])
    nat = A["natural"]
    sh = A["short"]
    ctx = _range_sum(sh["found_rare_t"] & nat & isctx, it_ptr) > 0
    anyf = np.zeros(len(z["item_idx"]), dtype=bool)
    for n in R.names:
        s_ = A["per_N"][n]["strong_t"]
        ctx |= _range_sum(s_ & isctx, it_ptr) > 0
        anyf |= _range_sum(s_ & nat, it_ptr) > 0
    return {"context": ctx, "any": anyf, "present": ctx | anyf}


def doc_exact(R, st, start, end, bench):
    """exact count (no hashing) of the items of `bench` whose question text occurs in the document tokens[start:end] of file st, and of those how many have the correct answer line / all choice lines within 90 tokens after it"""
    z = R.z
    names = R.meta["benchmarks"]
    doc = np.fromfile(st["file"], dtype="<u2", count=end - start, offset=2 * start) & U16(0x7FFF)
    db = doc.tobytes()
    vs, vl, vt = z["v_start"], z["v_len"], z["v_text"]
    order = np.argsort(vt, kind="stable")
    vptr = np.searchsorted(vt[order], np.arange(len(R.t_str) + 1))
    t_item, t_kind, t_choice = z["t_item"], z["t_kind"], z["t_choice"]
    it_ptr = np.searchsorted(t_item, np.arange(len(z["item_idx"]) + 1))
    NL = np.uint16(26).tobytes()  # the newline token

    def variants(t):
        return [R.tokens[vs[i]:vs[i] + vl[i]].tobytes() for i in order[vptr[t]:vptr[t + 1]]]

    ctx_kinds = [KIND_ID[k] for k in ("ctx", "passage", "block", "support")]
    bi = names.index(bench)
    n_q = n_ans = n_all = 0
    for it in np.flatnonzero(z["item_bench"] == bi):
        ts = list(range(int(it_ptr[it]), int(it_ptr[it + 1])))
        ctx = [t for t in ts if t_kind[t] in ctx_kinds]
        if not ctx:
            continue
        hits = []
        for vb in variants(ctx[0]):
            i = db.find(vb)
            while i >= 0:
                if i % 2 == 0:
                    hits.append((i // 2, len(vb) // 2))
                i = db.find(vb, i + 1)
        if not hits:
            continue
        n_q += 1
        qp, ql = min(hits)
        ends = {int(t_choice[t]): t for t in ts if t_kind[t] == KIND_ID["end"]}
        after = db[2 * (qp + ql):2 * (qp + ql + 90)]
        present = {c: any((vb + NL) in after for vb in variants(t)) for c, t in ends.items()}
        label = int(z["item_label"][it])
        n_ans += int(bool(present.get(label)))
        n_all += int(bool(present) and all(present.values()))
    return {"questions_in_document": n_q, "correct_answer_line_after": n_ans, "all_choice_lines_after": n_all}


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def make_report(out_dir, tok=None, eval_dir=None, report_dir=None, subset_dir=None, top=10, log=say):
    R = Results(out_dir, log)
    meta = R.meta
    log(f"report: {len(R.files)} scanned files, sources {R.sources}")
    A = analyse(R, log)
    z = R.z
    benches = meta["benchmarks"]
    item_bench = z["item_bench"]
    item_names = np.array([benches[int(i)] for i in item_bench])
    t_item, t_kind = z["t_item"], z["t_kind"]
    text_bench = item_names[t_item]
    it_ptr = A["it_ptr"]
    rep = {"tool_version": TOOL_VERSION, "created": time.strftime("%Y-%m-%d %H:%M:%S"), "index": {k: meta[k] for k in ("index_id", "created", "eval_dir", "tokenizer", "tokenizer_sha256", "seed", "Ns", "anchor", "items", "texts",
           "variants", "tokens", "items_per_bench", "n", "short", "seconds")}, "scan": {"files": [], "sources": R.sources}, "benchmarks": {}, "examples": {}, "choice_analysis": {}}
    tot_s, tot_t = 0.0, 0
    for st in R.files:
        rep["scan"]["files"].append({k: st[k] for k in ("source", "file", "tokens", "seconds", "max_token", "mask15", "distinct_hit_windows", "ngrams")})
        tot_s += st["seconds"]
        tot_t += st["tokens"]
    rep["scan"].update(total_tokens=tot_t, total_seconds=round(tot_s, 1), collisions_rejected={n: sum(st["ngrams"][n]["collisions"] for st in R.files) for n in R.names + ["short"]},
                       verified_hits={n: sum(st["ngrams"][n]["verified"] for st in R.files) for n in R.names + ["short"]})
    tokens_by_source = {s: sum(st["tokens"] for st in R.files if st["source"] == s) for s in R.sources}
    rep["scan"]["tokens_by_source"] = tokens_by_source
    flags = {}  # (bench, N) -> boolean array over the items of the bench: has a verified hit
    log("report: correct-versus-wrong choice analysis")
    ca_all = {n: choice_analysis(R, A, n) for n in R.names}
    pres = present_flags(R, A)
    for b in benches:
        isb = item_names == b
        n_items = int(isb.sum())
        tb = text_bench == b
        rb = {"items": n_items, "present": {k: int(v[isb].sum()) for k, v in pres.items()}}
        flags[(b, "present")] = pres["present"][isb]
        for n in R.names:
            a = A["per_N"][n]
            N = a["N"]
            item_hit = _range_sum(a["nf"] > 0, it_ptr) > 0
            item_rare = _range_sum(a["nf_rare"] > 0, it_ptr) > 0
            item_spec = _range_sum(a["nf_spec"] > 0, it_ptr) > 0
            item_strong_t = _range_sum(a["strong_t"], it_ptr) > 0
            item_ctx_present = _range_sum(a["strong_t"] & np.isin(t_kind, [KIND_ID[k] for k in CONTEXT_KINDS]), it_ptr) > 0
            testable = a["natural_testable"]
            flags[(b, n)] = item_hit[isb]
            flags[(b, n + "spec")] = item_spec[isb]
            flags[(b, n + "strong")] = item_strong_t[isb]
            kinds = {}
            for k in KINDS:
                km = tb & (t_kind == KIND_ID[k])
                if not km.any():
                    continue
                th = km & (a["nf"] > 0)
                ik = np.zeros(R.n_items, dtype=bool)
                ik[t_item[th]] = True
                isk = np.zeros(R.n_items, dtype=bool)
                isk[t_item[km & (a["strong_t"])]] = True
                kinds[k] = {"texts": int(km.sum()), "texts_with_windows": int((km & (a["nwin"] > 0)).sum()), "texts_hit": int(th.sum()), "items_hit": int(ik[isb].sum()), "items_strong": int(isk[isb].sum())}
            by_src = {}
            for s in R.sources:
                ih = _range_sum(a["nf_src"][s] > 0, it_ptr) > 0
                ist = _range_sum((a["nwin"] >= STRONG_MIN_WINDOWS) & (a["nf_src"][s] * 2 >= a["nwin"]), it_ptr) > 0
                by_src[s] = {"items_hit": int(ih[isb].sum()), "items_strong": int(ist[isb].sum())}
            rb[n] = {"N": N, "testable_items": int(testable[isb].sum()), "too_short_items": int((~testable[isb]).sum()), "items_hit": int(item_hit[isb].sum()), "items_hit_specific": int(item_spec[isb].sum()),
                     "items_hit_rare": int(item_rare[isb].sum()), "items_strong_text": int(item_strong_t[isb].sum()), "items_context_present": int(item_ctx_present[isb].sum()),
                     "items_strong_union": int(a["strict"][isb].sum()), "kinds": kinds, "by_source": by_src}
            if b in ca_all[n]:
                rb[n]["choices"] = ca_all[n][b]
        # short exact matches among the items that are too short for N = max(Ns)
        nmax = R.names[0]
        sh = A["short"]
        too_short = ~A["per_N"][nmax]["natural_testable"]
        tested_t = sh["elig_t"] & A["natural"]
        item_tested = _range_sum(tested_t, it_ptr) > 0
        item_found = _range_sum(sh["found_t"] & A["natural"], it_ptr) > 0
        item_found_rare = _range_sum(sh["found_rare_t"] & A["natural"], it_ptr) > 0
        rb["short"] = {"for_N": A["per_N"][nmax]["N"], "items_too_short": int(too_short[isb].sum()), "items_tested": int((too_short & item_tested)[isb].sum()), "items_with_whole_text_hit": int((too_short & item_found)[isb].sum()),
                       "items_with_whole_text_hit_rare": int((too_short & item_found_rare)[isb].sum()), "all_items_with_whole_text_hit": int(item_found[isb].sum()),
                       "texts_tested": int((tb & tested_t).sum()), "texts_hit": int((tb & sh["found_t"] & A["natural"]).sum())}
        rep["benchmarks"][b] = rb
    # ------------------------------------------------------------------ examples: the most revealing hits, decoded
    n16 = R.names[0]
    a = A["per_N"][n16]
    tptr, tgrp = z[f"{n16}_tptr"], z[f"{n16}_tgrp"]
    cnt = R.cnt[n16]
    rng = np.random.default_rng(7)

    def snippet(name, g, N):
        fi = int(R.loc_file[name][g])
        pos = int(R.loc_pos[name][g])
        st = R.files[fi]
        before, after = 40, 60
        a0 = max(0, pos - before)
        x = np.fromfile(st["file"], dtype="<u2", count=before + N + after, offset=2 * a0) & U16(0x7FFF)
        pre = tok.decode(x[:pos - a0]) if tok else ""
        mid = tok.decode(x[pos - a0:pos - a0 + N]) if tok else ""
        post = tok.decode(x[pos - a0 + N:]) if tok else ""
        return st, pos, f"{pre}[[{mid}]]{post}"

    for b in benches:
        isb = item_names == b
        ts = np.flatnonzero((text_bench == b) & (a["nf"] > 0))
        if len(ts) == 0:
            rep["examples"][b] = {"top": [], "weak_sample": []}
            continue
        # most revealing first: texts with an item-specific window that was found, then highest coverage, then the longest text
        order = ts[np.lexsort((-a["nwin"][ts], -a["cov"][ts], -(a["nf_spec"][ts] > 0).astype(np.int8)))]
        chosen, seen_items = [], set()
        for t in order:
            it = int(t_item[t])
            if it in seen_items:
                continue
            seen_items.add(it)
            chosen.append(int(t))
            if len(chosen) >= top:
                break
        weak = [int(t) for t in ts if a["nf"][t] == 1 and a["nwin"][t] >= STRONG_MIN_WINDOWS]
        weak = [weak[i] for i in rng.permutation(len(weak))[:5]] if weak else []

        def describe(t):
            gs = tgrp[tptr[t]:tptr[t + 1]]
            fg = gs[cnt[gs] > 0]
            g = int(fg[np.argmin(cnt[fg])])
            st, pos, snip = snippet(n16, g, 16)
            it = int(t_item[t])
            region = ""
            if st["source"] == "main/zzz_val":
                region = "training part of zzz_val" if pos < 26_090_802 else "validation tail of zzz_val"
            return {"item_index": int(z["item_idx"][it]), "kind": KINDS[int(t_kind[t])], "choice": int(z["t_choice"][t]), "label": int(z["item_label"][it]), "windows": int(a["nwin"][t]), "windows_found": int(a["nf"][t]),
                    "coverage": round(float(a["cov"][t]), 3), "text": R.t_str[t][:600], "corpus_occurrences_of_rarest_found_window": int(cnt[g]), "source": st["source"], "file": os.path.basename(st["file"]), "position": pos,
                    "region": region, "context": snip}

        rep["examples"][b] = {"top": [describe(t) for t in chosen], "weak_sample": [describe(t) for t in weak]}
    log("report: documents that hold many items")
    rep["documents"] = {n: doc_analysis(R, A, n, tok=tok) for n in R.names}
    # ------------------------------------------------------------------ clean / dirty subsets
    subsets = {}
    for b in benches:
        subsets[b] = {n: flags[(b, n)] for n in R.names}
    for b in benches:
        subsets[b].update({n + "spec": flags[(b, n + "spec")] for n in R.names})
        subsets[b][R.names[0] + "strong"] = flags[(b, R.names[0] + "strong")]
        subsets[b]["present"] = flags[(b, "present")]
    rep["flags"] = {b: {n: {"dirty": int(v.sum()), "clean": int((~v).sum())} for n, v in d.items()} for b, d in subsets.items()}
    if subset_dir and eval_dir:
        rep["subsets"] = write_subsets(eval_dir, subset_dir, meta, subsets, log)
    # ------------------------------------------------------------------ files
    if report_dir:
        os.makedirs(report_dir, exist_ok=True)
        with open(os.path.join(report_dir, "contamination_results.json"), "w") as f:
            json.dump(rep, f, indent=1)
        with open(os.path.join(report_dir, "contamination_report.md"), "w", encoding="utf-8") as f:
            f.write(render_markdown(rep, R))
        np.savez(os.path.join(report_dir, "item_flags.npz"), **{f"{b}__{n}": v for b, d in subsets.items() for n, v in d.items()})
        for n, dd in rep["documents"].items():  # documents of the training shards that hold a strong hit (at 16 tokens: clear cases; at 10 tokens: also the short questions)
            spans = [d for d in dd.get("all_documents", []) if d["source"] not in ("cool/cool_val", "sft/zzz_val")]
            with open(os.path.join(report_dir, "exclusion_spans.json" if n == R.names[0] else f"exclusion_spans_{n}.json"), "w") as f:
                json.dump(spans, f, indent=1)
        log(f"report: written to {report_dir}")
    return rep


def write_subsets(eval_dir, subset_dir, meta, subsets, log):
    """clean/dirty copies of the eval files (same layout; MANIFEST.json adjusted) for N = each window size"""
    man_path = os.path.join(eval_dir, "MANIFEST.json")
    manifest = json.load(open(man_path)) if os.path.isfile(man_path) else None
    files = {}
    for sub in ("mc", "hellaswag", "lambada"):
        d = os.path.join(eval_dir, sub)
        if os.path.isdir(d):
            for f in sorted(os.listdir(d)):
                if f.endswith(".jsonl"):
                    files["hellaswag" if (sub == "hellaswag" and f == "hellaswag_val.jsonl") else f[:-6]] = (sub, f)
    info = {}
    keys = [f"n{N}" for N in meta["Ns"]] + [f"n{meta['Ns'][0]}strong", "present"]
    for n in keys:
        for part in ("clean", "dirty"):
            name = part + (n if n == "present" else n[1:])
            root = os.path.join(subset_dir, name)
            os.makedirs(root, exist_ok=True)
            man = json.loads(json.dumps(manifest)) if manifest else None
            counts = {}
            for b, (sub, f) in files.items():
                if b not in subsets or b == "mmlu_letter":  # the lettered MMLU format is not part of the default ladder (it would only add 32 MB per directory)
                    if man and b in man["tasks"]:
                        del man["tasks"][b]
                    continue
                dirty = subsets[b][n]
                lines = [ln for ln in open(os.path.join(eval_dir, sub, f), encoding="utf-8") if ln.strip()]
                if len(lines) != len(dirty):
                    log(f"  subsets: {b}: {len(lines)} records but {len(dirty)} flags: skipped")
                    continue
                keep = [i for i in range(len(lines)) if bool(dirty[i]) == (part == "dirty")]
                if b == "sciq":  # a SciQ question is dirty when its question/answer texts hit; the support passage is a separate benchmark (sciq_support) and does not make a question dirty
                    pass
                counts[b] = len(keep)
                os.makedirs(os.path.join(root, sub), exist_ok=True)
                body = "".join(lines[i].rstrip("\n") + "\n" for i in keep)
                if keep:
                    with open(os.path.join(root, sub, f), "w", encoding="utf-8", newline="\n") as fh:
                        fh.write(body)
                if man and b in man["tasks"]:
                    t = man["tasks"][b]
                    if not keep:
                        del man["tasks"][b]
                        continue
                    t["examples_in_full_split"] = t.get("examples")
                    t["examples"] = len(keep)
                    t["bytes"] = len(body.encode("utf-8"))
                    t["sha256"] = hashlib.sha256(body.encode("utf-8")).hexdigest()
                    t["quick"] = min(t.get("quick", len(keep)), len(keep))
                    t["full_limit"] = min(t.get("full_limit", len(keep)), len(keep))
                    if t.get("kind") == "mc":
                        ks = [len(json.loads(lines[i])["choices"]) for i in keep]
                        t["chance"] = round(100.0 * sum(1.0 / k for k in ks) / len(ks), 2)
                    N_ = n[1:].replace("strong", "")
                    if n == "present":
                        t["contamination_subset"] = f"{part} 'present': items {'with' if part == 'dirty' else 'without'} a text of which at least half of the 16- or 10-token windows occur, or a rare whole short question, in the scanned corpora"
                    else:
                        t["contamination_subset"] = (f"{part} at N={N_}: items {'with' if part == 'dirty' else 'without'} a text of which at least half of the {N_}-token windows occur in the scanned corpora" if n.endswith("strong")
                                                     else f"{part} at N={N_}: items {'with' if part == 'dirty' else 'without'} a verified {N_}-token window of any of their texts in the scanned corpora")
            if man:
                man["order"] = [t for t in man.get("order", []) if t in man["tasks"]]
                man["generator"] = f"tools/contamination_check.py ({part}{n[1:]} subset of {man.get('generator')})"
                with open(os.path.join(root, "MANIFEST.json"), "w") as fh:
                    json.dump(man, fh, indent=1)
            for extra in ("tokenizer_f1.json",):
                src = os.path.join(eval_dir, extra)
                if os.path.isfile(src) and not os.path.exists(os.path.join(root, extra)):
                    with open(src, "rb") as a_, open(os.path.join(root, extra), "wb") as b_:
                        b_.write(a_.read())
            ppl = os.path.join(eval_dir, "ppl")
            if os.path.isdir(ppl) and not os.path.exists(os.path.join(root, "ppl")) and hasattr(os, "symlink"):
                try:
                    os.symlink(os.path.abspath(ppl), os.path.join(root, "ppl"))
                except OSError:
                    pass
            info[name] = {"dir": root, "examples": counts}
    return info


def render_markdown(rep, R):
    L = []
    idx = rep["index"]
    L.append("# Forge-1 benchmark contamination check")
    L.append("")
    L.append(f"Generated {rep['created']} by `tools/contamination_check.py` (version {rep['tool_version']}); eval sets {idx['eval_dir']}, tokenizer sha256 {str(idx['tokenizer_sha256'])[:16]}, "
             f"windows N = {', '.join(map(str, idx['Ns']))} tokens (+ whole-text match of texts of {idx['short']['anchor']}..{idx['short']['max_len']} tokens), hash seed {idx['seed']}.")
    L.append("")
    sc = rep["scan"]
    L.append(f"Scanned {sc['total_tokens'] / 1e9:.3f} B tokens in {len(sc['files'])} files ({sc['total_seconds'] / 60:.1f} CPU-minutes of scanning); every hit was verified against the exact tokens; "
             f"hash candidates rejected as collisions: " + ", ".join(f"{k} {v}" for k, v in sc["collisions_rejected"].items()) + "; verified window occurrences: " + ", ".join(f"{k} {v}" for k, v in sc["verified_hits"].items()) + ".")
    L.append("")
    L.append(md_table(["source", "tokens (M)"], [[s, f"{v / 1e6:.1f}"] for s, v in sc["tokens_by_source"].items()]))
    L.append("")
    for n in R.names:
        N = n[1:]
        L.append(f"## Window N = {N}")
        L.append("")
        rows = []
        for b, rb in rep["benchmarks"].items():
            r = rb[n]
            rows.append([b, rb["items"], r["too_short_items"], _fmt_pct(r["items_hit"], rb["items"]), _fmt_pct(r["items_hit_specific"], rb["items"]), _fmt_pct(r["items_hit_rare"], rb["items"]), r["items_strong_text"],
                         r["items_context_present"], r["items_strong_union"]])
        L.append(md_table(["benchmark", "items", f"too short for {N}", "items with a verified hit", "... on a window that no other item has", f"... on a window that occurs <= {RARE_MAX} times in the corpora", "a text with >= 50 % of its windows",
                           "... of which the question / context / passage", "item windows >= 50 % (union)"], rows))
        L.append("")
        srcs = R.sources
        rows = [[b] + [f"{rb[n]['by_source'][s]['items_hit']} / {rb[n]['by_source'][s]['items_strong']}" for s in srcs] for b, rb in rep["benchmarks"].items()]
        L.append("Items with a hit / items with a text at least half present, per corpus source:")
        L.append("")
        L.append(md_table(["benchmark"] + srcs, rows))
        L.append("")
    L.append("## Items too short for the window: whole-text match")
    L.append("")
    rows = []
    for b, rb in rep["benchmarks"].items():
        s = rb["short"]
        rows.append([b, s["for_N"], s["items_too_short"], s["items_tested"], s["items_with_whole_text_hit"], s["items_with_whole_text_hit_rare"], s["all_items_with_whole_text_hit"]])
    L.append(md_table(["benchmark", "N", "items too short", "of these with a testable text (anchor..15 tokens)", "of these with a whole-text hit", f"... occurring <= {RARE_MAX} times", "all items with a whole-text hit"], rows))
    L.append("")
    L.append("## Items that are present (any window size)")
    L.append("")
    L.append("`question present` = the question / context / passage text is strong at 16 or at 10 tokens (at least 3 windows, half of them found) or occurs whole as a short text that is rare in the corpora; `any text` = any natural text of the item "
             "(also an answer or an ending) is strong at 16 or 10 tokens; `present` = the union. The 16-token windows alone miss items with a short question (see the positive control in the research note).")
    L.append("")
    rows = [[b, rb["items"], f"{rb['present']['context']} ({100.0 * rb['present']['context'] / rb['items']:.1f} %)", rb["present"]["any"], f"{rb['present']['present']} ({100.0 * rb['present']['present'] / rb['items']:.1f} %)"]
            for b, rb in rep["benchmarks"].items()]
    L.append(md_table(["benchmark", "items", "question present", "any text strong", "present (union)"], rows))
    L.append("")
    for n, docs in rep.get("documents", {}).items():
        if not docs.get("per_benchmark"):
            continue
        L.append(f"## Documents that hold the strong hits (N = {n[1:]})")
        L.append("")
        L.append("A document that holds many different items of one benchmark is a copy of the benchmark (or of an exam / quiz page it was made from); a single item per document is an isolated copy of one question or passage. "
                 "Only strong texts (at least half of their windows occur) are located. `exact` = exact count in the biggest dump documents: questions found, correct answer line within 90 tokens after the question, all choice lines after it.")
        L.append("")
        rows = [[b, e["documents_with_strong_items"], e["documents_with_min_items"], e["documents_with_10_items"], e["items_in_documents_with_5_or_more_items"]] for b, e in docs["per_benchmark"].items()]
        L.append(md_table(["benchmark", "documents with a strong item", "documents with >= 3 strong items", "documents with >= 10 strong items", "strong items that sit in a document with >= 5 of them"], rows))
        L.append("")
        rows = []
        for b, e in docs["per_benchmark"].items():
            for d in e["top"]:
                ex = d.get("exact")
                rows.append([b, d["items"], d["source"], d["file"], d["start"], d["tokens"], (f"{ex['questions_in_document']} / {ex['correct_answer_line_after']} / {ex['all_choice_lines_after']}" if ex else ""), repr(d["start_of_document"][:100])])
        L.append(md_table(["benchmark", "items in the document", "source", "file", "document starts at token", "document tokens", "exact: questions / correct line / all lines", "beginning of the document"], rows))
        L.append("")
    L.append("## Hit by text kind (N = " + R.names[0][1:] + ")")
    L.append("")
    rows = []
    for b, rb in rep["benchmarks"].items():
        for k, v in rb[R.names[0]]["kinds"].items():
            rows.append([b, k, v["texts"], v["texts_with_windows"], v["texts_hit"], v["items_hit"], v["items_strong"]])
    L.append(md_table(["benchmark", "kind", "texts", "texts with windows", "texts hit", "items hit", "items strong"], rows))
    L.append("")
    L.append("## Correct versus wrong choices (windows that belong to one choice only), N = " + R.names[0][1:])
    L.append("")
    rows = []
    for b, rb in rep["benchmarks"].items():
        c = rb[R.names[0]].get("choices")
        if c and c["with_specific_windows"]:
            rows.append([b, c["with_specific_windows"], c["correct_hit"], f"{c['wrong_hit']} of {c['wrong_choices']}", c["only_correct"], c["only_wrong"], c["both"]])
    L.append(md_table(["benchmark", "items", "correct choice hit", "wrong choices hit", "items: only correct", "only wrong", "both"], rows))
    L.append("")
    L.append("## Examples (the most complete hits, then a few single-window hits), N = " + R.names[0][1:])
    for b, ex in rep["examples"].items():
        L.append("")
        L.append(f"### {b}")
        for tag, lst in (("top", ex["top"]), ("single-window sample", ex["weak_sample"])):
            for e in lst:
                L.append(f"- [{tag}] item {e['item_index']} ({e['kind']}, choice {e['choice']}, label {e['label']}): {e['windows_found']}/{e['windows']} windows found, rarest window occurs {e['corpus_occurrences_of_rarest_found_window']} times; "
                         f"{e['source']} {e['file']} @ {e['position']} {e['region']}")
                L.append(f"  - eval text: `{e['text'][:300]!r}`")
                L.append(f"  - corpus: `{e['context'][:420]!r}`")
    return "\n".join(L) + "\n"


# ============================================================================== CLI
def parse_part(s):
    i, _, m = s.partition("/")
    return int(i), int(m or 1)


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, index=False, scan=False, report=False):
        p.add_argument("--out", required=True, help="working directory (index, scan results)")
        if index:
            p.add_argument("--eval-dir", default=DEFAULT_EVAL)
            p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
            p.add_argument("--n", default="16,10", help="window lengths in tokens, the first is the primary one")
            p.add_argument("--anchor", type=int, default=8, help="anchor length of the whole-text match for texts shorter than the windows")
            p.add_argument("--seed", type=int, default=DEFAULT_SEED)
            p.add_argument("--bench", default="", help="only these benchmarks (comma separated file names without .jsonl)")
        if scan:
            p.add_argument("--corpus", action="append", default=[], help="LABEL=PATH (file, directory or glob of .u16 files); repeat")
            p.add_argument("--mask15", default="", help="corpus labels whose tokens carry a loss-mask flag in bit 15 (cleared before hashing)")
            p.add_argument("--chunk", type=int, default=1 << 24, help="tokens per chunk (a multiple of 2048)")
            p.add_argument("--part", default="0/1", help="I/M: scan the I-th of M size-balanced file groups")
            p.add_argument("--no-direct", action="store_true", help="do not use O_DIRECT")
            p.add_argument("--max-mb-s", type=float, default=0.0, help="read at most this many MB/s (0 = as fast as the CPU goes)")
            p.add_argument("--bitmap-bits", type=int, default=0)
            p.add_argument("--eos-only", action="store_true", help="only record the document separator positions (needs no index; a fast pass for scan results made without them)")
        if report:
            p.add_argument("--eval-dir", default=DEFAULT_EVAL)
            p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
            p.add_argument("--report-dir", default="")
            p.add_argument("--subset-dir", default="", help="write clean/dirty copies of the eval data here")
            p.add_argument("--top", type=int, default=10)

    common(sub.add_parser("index", help="build the index of the evaluation windows"), index=True)
    common(sub.add_parser("scan", help="scan corpus shards"), scan=True)
    common(sub.add_parser("report", help="aggregate, decode examples, write reports and subsets"), report=True)
    p = sub.add_parser("all", help="index + scan + report")
    common(p, index=True, scan=True)
    p.add_argument("--report-dir", default="")
    p.add_argument("--subset-dir", default="")
    p.add_argument("--top", type=int, default=10)
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    tok = None
    if a.cmd in ("index", "all"):
        tok = HFTok(os.path.expanduser(a.tokenizer))
        Ns = tuple(int(x) for x in a.n.split(","))
        build_index(os.path.expanduser(a.eval_dir), tok, a.out, Ns=Ns, anchor=a.anchor, hasher=Hasher(a.seed), only=set(filter(None, a.bench.split(","))) or None)
    if a.cmd in ("scan", "all"):
        run_scan(a.out, a.corpus, part=parse_part(a.part), chunk=a.chunk, mask15_labels=set(filter(None, a.mask15.split(","))), direct=not a.no_direct, max_mb_s=a.max_mb_s,
                 bitmap_bits=a.bitmap_bits or None, eos_only=getattr(a, "eos_only", False))
    if a.cmd in ("report", "all"):
        if tok is None:
            tok = HFTok(os.path.expanduser(a.tokenizer))
        make_report(a.out, tok=tok, eval_dir=os.path.expanduser(a.eval_dir) if hasattr(a, "eval_dir") else None, report_dir=a.report_dir or os.path.join(a.out, "report"),
                    subset_dir=a.subset_dir or None, top=a.top)
    return 0


if __name__ == "__main__":
    sys.exit(main())
