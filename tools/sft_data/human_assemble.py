#!/usr/bin/env python3
"""human_assemble.py plan | encode | publish | replay: assemble the human-written-only Forge-1 chat mix from the filtered pools (see ForgeInfer - SFT Data Mix (human-written only) in the research notes).
Same layout, masks and validation-tail rule as sft_assemble.py (tokens_sft_mix2); only the sources, the late filters and the paths differ.

  plan      load the pools (sft_work_human/pool/*.jsonl, plus the identity pool of the first build), apply the late filters (blacklists, HTML entities, 'as an AI', length requests, Stack Exchange
            question-answer alignment), remove exact and near duplicates ACROSS sources (MinHash of the first user message), thin every source to its token target, hold out the validation conversations,
            oversample the identity set, shuffle (seed), cut into training shards and the validation shard, and write the NUL-separated ChatML corpus files + docs.tsv + plan.json to the build directory.
  encode    for every corpus file: `forge tok-encode` -> `forge chat-mask` (bit 15 = not a target) and check the token counts against the plan.
  publish   copy the shards to /mnt/data/forgeinfer-data/tokens_sft_human (throttled to 40 MB/s, 20 s pause between files); heldout/ with the held-out region alone (masked and plain).
  replay    tokens_sft_human_replay: a_replay.u16 (the first bytes of tokens_cool/replay/replay_000.u16, 12 % of the total) + symlinks to the SFT files (the way tokens_sft_mix2_replay was built).

The validation shard zzz_val.u16 has the layout [padding][training documents][validation documents] with  length = 4 x (validation tokens):  the trainer takes the LAST file, reserves min(--val-tokens, tokens / 4)
of its tail as the validation region and trains on the rest of that file, so the region is exactly the held-out documents when --val-tokens >= T.  No file may sort after sft_* / zzz_val.
"""
import argparse
import collections
import glob
import json
import math
import os
import re
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import human_cfg as C  # noqa: E402
import human_lib as H  # noqa: E402
import sft_lib as L  # noqa: E402

POOL = C.POOL
BUILD = C.BUILD
FORGE = os.environ.get("FORGE_SFT_BIN", "/tmp/forge_sft")
VAL_TARGET = 2_000_000
PER_SHARD = 28_000_000  # about the size of the shards of mix2

# source -> final tokens in millions (before oversampling); the ORDER is the priority of the cross-source deduplication (hand-written and human chat first)
# OASST2 is NOT in the mix by default: 55 % of its best-ranked answers carry chatbot fingerprints (2.8 % in the human control Dolly, 2 % in Stack Exchange) and many of the others read like ChatGPT as well, so
# "every answer was written by a human" cannot be claimed for it.  HUMAN_INCLUDE_OASST2=1 adds the pool (after the style filter below) for a build that accepts that risk.
INCLUDE_OASST2 = os.environ.get("HUMAN_INCLUDE_OASST2", "0") == "1"
TARGETS_M = collections.OrderedDict([
    ("identity", 99), ("dolly", 99), ("no_robots", 99), ("aya", 99),
    ("gsm8k", 99), ("hendrycks", 99), ("aqua", 99),
    ("se", 16.3), ("shp", 99),
    ("squad", 3.0), ("coqa", 99), ("nq", 0.6), ("trivia", 0.6),
    ("cnn", 4.0), ("dialogsum", 2.0), ("wp", 99),
    ("csn", 99), ("apps", 99), ("rosetta", 99), ("mbpp", 99),
])
if INCLUDE_OASST2:
    TARGETS_M = collections.OrderedDict(list(TARGETS_M.items())[:1] + [("oasst2", 99)] + list(TARGETS_M.items())[1:])
OVERSAMPLE = {"identity": 5}
NODEDUP = {"identity"}
LENGTH_CHECKED = {"oasst2", "dolly", "no_robots", "aya"}
ENTITY_CHECKED_OFF = {"se", "csn", "apps", "rosetta", "mbpp", "identity"}  # sources where HTML-looking strings are legitimate (code)
ALIGN_MIN = 4.0  # Stack Exchange: minimal IDF-weighted word overlap between question and answer (the HuggingFaceH4 dump pairs a few percent of the questions with answers of another question)
LEN_EDGES = [64, 128, 256, 384, 512, 768, 1024, 1536, 2048]

LICENSE_NOTE = {}
_ENTITY = re.compile(r"&(?:#\d{2,5}|#x[0-9a-fA-F]{2,4}|quot|amp|lt|gt|nbsp|apos);")
LATE_DROPS = collections.Counter()
_WORDS_REQ = re.compile(r"(\d[\d,]*)\s*(?:-|to|and)?\s*(\d[\d,]*)?\s*[- ]?words?\b", re.I)
_AS_AI = re.compile(r"\bas an? (?:ai|a\.i\.|artificial intelligence|language model|text-based|virtual assistant|digital assistant|chatbot)\b", re.I)


def length_mismatch(msgs):
    user = next((c for r, c in msgs if r == "user"), "")
    need = 0
    for m in _WORDS_REQ.finditer(user):
        a = int(m.group(1).replace(",", ""))
        b = int(m.group(2).replace(",", "")) if m.group(2) else a
        need = max(need, a, b)
    if need < 300:
        return False
    ans = next((c for r, c in msgs if r == "assistant"), "")
    return len(ans.split()) < 0.6 * need


def has_as_ai(msgs):
    return bool(_AS_AI.search("\n".join(c for r, c in msgs if r != "user")))


# LLM fingerprints in the answers of the datasets collected in 2023 or later (OASST2, Dolly, No Robots, Aya): volunteers and annotators sometimes pasted chatbot output although they were asked not to.  The patterns
# have a rate of 0.0-0.3 % in Dolly (human control, collected in March 2023 under a no-generative-AI instruction) and of 20-55 % in OASST2 (see the research note): an answer with one STRONG pattern or two WEAK ones is dropped.
_LLM_STRONG = [re.compile(p, re.I | re.M) for p in (
    r"\bhope (?:this|that|it) helps\b", r"\blet me know if (?:you|there)", r"\bfeel free to (?:ask|reach|let|contact|use|modify|adjust|customize|experiment)\b", r"^\s*(?:certainly|sure|absolutely|of course|great question)\b[!,.]",
    r"\bi(?:'m| am) sorry,? (?:but|for|if)\b|\bi apologi[sz]e\b", r"\bas an? (?:ai|language model|artificial)\b|\bi(?:'m| am) an? (?:ai|language model|assistant|chatbot)\b",
    r"\bit(?:'s| is) (?:important|worth|essential|crucial) to (?:note|remember|keep|mention|consider|understand)\b", r"\b(?:my|the) previous (?:response|answer|message|reply)\b|\bimproved version\b|\bin my previous\b",
    r"^\s*(?:overall|in conclusion|in summary|to summarize|in short),", r"\bhowever, it(?:'s| is) (?:important|worth)\b", r"^\s*\d+\.\s+\*\*[^*]+\*\*:", r"\bopen[- ]?assistant\b|\bopenai\b|\bchatgpt\b|\bgpt-?\d\b|\bllm\b|\blarge language model")]
_LLM_WEAK = [re.compile(p, re.I | re.M) for p in (
    r"\bhere(?:'s| is| are) (?:a|an|the|some|my|your|one|two|three|four|five|\d+)\b[^\n]{0,80}:\s*$", r"\b(?:additionally|furthermore|moreover),",
    r"\bdelve\b|\btapestry\b|\bvibrant\b|\blandscape of\b|\bplays a (?:crucial|vital|significant) role\b|\bin the realm of\b|\bnavigate the\b")]
LLM_STYLE_CHECKED = {"oasst2", "dolly", "no_robots", "aya"}


def llm_style(msgs):
    a = "\n".join(c for r, c in msgs if r == "assistant")
    if any(rx.search(a) for rx in _LLM_STRONG):
        return True
    return sum(1 for rx in _LLM_WEAK if rx.search(a)) >= 2


def _load_blacklist():
    """id -> reason: every sft_work_human/blacklist_*.txt (one id per line; the name says why: blacklist_fnames = HumanEval / MBPP function names, blacklist_eval = verbatim run against the evaluation
    files, blacklist_contam = hits of tools/contamination_check.py on the finished token files)."""
    ids = {}
    for p in sorted(glob.glob(os.path.join(C.WORK, "blacklist_*.txt"))):
        why = os.path.basename(p)[len("blacklist_"):-4]
        if why == "contam":
            continue  # handled as DEAD below
        for x in open(p, encoding="utf-8"):
            if x.strip():
                ids.setdefault(x.strip(), why)
    return ids


BLACKLIST = _load_blacklist()
# the conversations found by tools/contamination_check.py in the finished files stay in the pool during the DEDUPLICATION (as 'dead' records): their near duplicates, which would otherwise take their place and carry the
# same sentence into the mix again, are dropped as near duplicates of them; they are removed right after the deduplication
DEAD = set()
_pc = os.path.join(C.WORK, "blacklist_contam.txt")
if os.path.exists(_pc):
    DEAD = {x.strip() for x in open(_pc, encoding="utf-8") if x.strip()}


# ---- Stack Exchange alignment -------------------------------------------------------------------------------------------------------------------------------------------------------------
_STOP = L._STOP | {"with", "from", "have", "what", "when", "which", "would", "could", "should", "there", "their", "about", "these", "those", "will", "your", "than", "then", "them", "they", "been", "were", "also",
                   "does", "just", "like", "some", "more", "into", "only", "other", "such", "very", "much", "many", "most", "over", "here", "that", "this", "using", "used", "thanks", "question", "can", "get", "how",
                   "why", "not", "but", "has", "had", "its", "any", "all", "who", "you", "was", "may", "one", "two"}
_W = re.compile(r"[a-z][a-z0-9_]{2,}")


def _words(s):
    return {w for w in _W.findall(s.lower()) if w not in _STOP}


def align_filter(recs):
    """Drop the Stack Exchange conversations whose answer shares (almost) no content words with the question: the IDF-weighted overlap of the two texts, idf over the pool itself.
    On a 8,543-conversation test pool the overlap of real pairs had a median of 22.6 and of pairs shuffled within a site 3.1."""
    W = [(_words(r["msgs"][0][1]), _words("\n".join(c for ro, c in r["msgs"] if ro == "assistant"))) for r in recs]
    df = collections.Counter()
    for q, a in W:
        for w in q | a:
            df[w] += 1
    n = len(recs)
    idf = {w: math.log(n / c) for w, c in df.items()}
    out = []
    for r, (q, a) in zip(recs, W):
        if sum(idf[w] for w in q & a) < ALIGN_MIN:
            LATE_DROPS[(r["src"], "question_answer_not_aligned")] += 1
        else:
            out.append(r)
    return out


def load_pool(name):
    p = os.path.join(C.OLD_WORK, "pool", "identity.jsonl") if name == "identity" else os.path.join(POOL, name + ".jsonl")
    if not os.path.exists(p):
        return []
    out = []
    for r in L.read_jsonl(p):
        if r["id"] in BLACKLIST:
            LATE_DROPS[(name, BLACKLIST[r["id"]])] += 1
            continue
        if name not in ENTITY_CHECKED_OFF and _ENTITY.search("\n".join(c for _, c in r["msgs"])):
            LATE_DROPS[(name, "html_entity")] += 1
            continue
        if name != "identity" and has_as_ai(r["msgs"]):
            LATE_DROPS[(name, "as_an_ai_boilerplate")] += 1
            continue
        if name in LENGTH_CHECKED and length_mismatch(r["msgs"]):
            LATE_DROPS[(name, "length_request_not_met")] += 1
            continue
        if name in LLM_STYLE_CHECKED and llm_style(r["msgs"]):
            LATE_DROPS[(name, "llm_style_text")] += 1
            continue
        if r["id"] in DEAD:
            r["dead"] = True
        out.append(r)
    if name == "se":
        out = align_filter(out)
    return out


def first_user(rec):
    for r, c in rec["msgs"]:
        if r == "user":
            return c
    return ""


def dedup_across(recs_by_src):
    """Cross-source duplicates. Exact: same conversation text or same user side. Near: MinHash (60 permutations, 12 bands of 5) of the word 3-grams of the first user message, verified by the exact
    Jaccard similarity (>= 0.8 for prompts shorter than 40 words, >= 0.7 for longer ones). The record that comes first in the priority order stays."""
    order = [(s, i) for s in recs_by_src for i in range(len(recs_by_src[s]))]
    n = len(order)
    drop = np.zeros(n, dtype=bool)
    reason = {}
    seen_c, seen_u = {}, {}
    t0 = time.time()
    for k, (s, i) in enumerate(order):
        if s in NODEDUP:
            continue
        rec = recs_by_src[s][i]
        hc, hu = L.conv_hash(rec["msgs"]), L.prompt_hash(rec["msgs"])
        if hc in seen_c or hu in seen_u:
            drop[k] = True
            reason[k] = "exact"
            continue
        seen_c[hc] = k
        seen_u[hu] = k
    print(f"[dedup] exact pass {time.time() - t0:.0f} s, {int(drop.sum())} dropped", flush=True)
    sig_k, sigs = [], []
    for k, (s, i) in enumerate(order):
        if drop[k] or s in NODEDUP:
            continue
        fu = first_user(recs_by_src[s][i])
        if len(fu.split()) < 8:
            continue
        sg = L.minhash(fu)
        if sg is None:
            continue
        sig_k.append(k)
        sigs.append(sg)
    print(f"[dedup] {len(sig_k)} signatures in {time.time() - t0:.0f} s", flush=True)
    S = np.stack(sigs)
    sig_k = np.array(sig_k)
    near = 0
    mult = np.array([0x9E3779B97F4A7C15, 0xC2B2AE3D27D4EB4F, 0x165667B19E3779F9, 0x27D4EB2F165667C5, 0x85EBCA77C2B2AE63], dtype=np.uint64)
    shingle_cache = {}

    def shingles(k):
        if k not in shingle_cache:
            s, i = order[k]
            shingle_cache[k] = L.shingle_set(first_user(recs_by_src[s][i]))
        return shingle_cache[k]

    for b in range(12):
        with np.errstate(over="ignore"):
            keys = (S[:, 5 * b : 5 * b + 5] * mult).sum(axis=1)
        idx = np.argsort(keys, kind="stable")
        ks = keys[idx]
        brk = np.flatnonzero(ks[1:] != ks[:-1]) + 1
        starts = np.concatenate(([0], brk))
        ends = np.concatenate((brk, [len(ks)]))
        for g in np.flatnonzero(ends - starts > 1):
            members = sorted(int(sig_k[j]) for j in idx[starts[g] : ends[g]])
            leaders = []
            for m in members:
                if drop[m]:
                    continue
                sm = shingles(m)
                thr = 0.8 if len(sm) < 38 else 0.7
                if any(L.jaccard(sm, shingles(ld)) >= thr for ld in leaders):
                    drop[m] = True
                    reason[m] = "near"
                    near += 1
                else:
                    leaders.append(m)
        if len(shingle_cache) > 300000:
            shingle_cache.clear()
    print(f"[dedup] near duplicates dropped: {near}, total {int(drop.sum())} of {n}, {time.time() - t0:.0f} s", flush=True)
    out = {s: [] for s in recs_by_src}
    stats = collections.Counter()
    for k, (s, i) in enumerate(order):
        if drop[k]:
            stats[(s, reason[k])] += 1
        else:
            out[s].append(recs_by_src[s][i])
    return out, stats


def cmd_plan(a):
    os.makedirs(BUILD, exist_ok=True)
    t0 = time.time()
    recs = collections.OrderedDict()
    for s in TARGETS_M:
        r = load_pool(s)
        if not r:
            print(f"[plan] WARNING: no pool for {s}")
            continue
        rng = L.rng_for("human-assemble-shuffle", s)
        rng.shuffle(r)
        recs[s] = r
        print(f"[plan] {s}: {len(r)} pool records after the late filters, {sum(x['ntok'] for x in r) / 1e6:.2f} M tokens", flush=True)
    recs, dstats = dedup_across(recs)
    for s in list(recs):
        live = [x for x in recs[s] if not x.get("dead")]
        LATE_DROPS[(s, "contamination_check")] += len(recs[s]) - len(live)
        recs[s] = live
    selected = collections.OrderedDict()
    for s, r in recs.items():
        target = TARGETS_M[s] * 1e6
        tok, keep = 0, []
        for x in r:
            if tok >= target:
                break
            keep.append(x)
            tok += x["ntok"]
        selected[s] = keep
        print(f"[plan] {s}: selected {len(keep)} / {len(r)} records, {tok / 1e6:.2f} M tokens", flush=True)
    total = sum(x["ntok"] for r in selected.values() for x in r)
    vrng = L.rng_for("human-assemble-val")
    val, train = [], []
    for s, r in selected.items():
        if s == "identity":
            nval = max(2, int(round(0.08 * len(r))))
        else:
            nval = int(round(VAL_TARGET * sum(x["ntok"] for x in r) / total / max(1.0, np.mean([x["ntok"] for x in r]))))
        idx = list(range(len(r)))
        vrng.shuffle(idx)
        vset = set(idx[:nval])
        for j, x in enumerate(r):
            (val if j in vset else train).append(x)
    vrng.shuffle(val)
    vtok, v2 = 0, []
    for x in val:
        if vtok >= VAL_TARGET:
            train.append(x)
        else:
            v2.append(x)
            vtok += x["ntok"]
    val = v2
    T = sum(x["ntok"] for x in val)
    train_all = []
    for x in train:
        train_all.extend([x] * OVERSAMPLE.get(x["src"], 1))
    trng = L.rng_for("human-assemble-train")
    trng.shuffle(train_all)
    tail, tail_tok = [], 0
    while train_all and tail_tok + train_all[-1]["ntok"] <= 3 * T:
        x = train_all.pop()
        tail.append(x)
        tail_tok += x["ntok"]
    pad = 3 * T - tail_tok
    tail.reverse()
    total_train = sum(x["ntok"] for x in train_all)
    nshards = max(1, int(round(total_train / PER_SHARD)))
    shards, cur, cur_tok, per = [], [], 0, total_train / nshards
    for x in train_all:
        cur.append(x)
        cur_tok += x["ntok"]
        if cur_tok >= per and len(shards) < nshards - 1:
            shards.append(cur)
            cur, cur_tok = [], 0
    shards.append(cur)
    docs = open(os.path.join(BUILD, "docs.tsv"), "w", encoding="utf-8")
    docs.write("file\tindex\tsrc\tstratum\tntok\tnsup\tturns\thas_system\tid\n")
    files = [(f"sft_{k:03d}", sh) for k, sh in enumerate(shards)]
    files.append(("zzz_val", tail + val))
    sizes = {}
    for name, sh in files:
        with open(os.path.join(BUILD, name + ".txt"), "wb") as f:
            for j, x in enumerate(sh):
                txt = L.render([tuple(m) for m in x["msgs"]])
                assert "\x00" not in txt
                f.write(txt.encode("utf-8"))
                f.write(b"\x00")
                turns = sum(1 for r, _ in x["msgs"] if r == "assistant")
                docs.write(f"{name}\t{j}\t{x['src']}\t{x['stratum']}\t{x['ntok']}\t{x['nsup']}\t{turns}\t{int(x['msgs'][0][0] == 'system')}\t{x['id']}\n")
        sizes[name] = sum(x["ntok"] for x in sh)
    docs.close()
    plan = {"validation_tokens_T": T, "val_docs": len(val), "val_file_tail_train_docs": len(tail), "val_file_tail_train_tokens": tail_tok, "val_file_pad_tokens": pad, "val_file_tokens": 4 * T,
            "shard_tokens": sizes, "train_tokens_total": total_train + tail_tok, "dedup_drops": {f"{s}:{r}": n for (s, r), n in sorted(dstats.items())},
            "late_filter_drops": {f"{s}:{r}": n for (s, r), n in sorted(LATE_DROPS.items()) if n}, "oversample": OVERSAMPLE, "targets_m": TARGETS_M, "val_target": VAL_TARGET, "seconds": round(time.time() - t0),
            "blacklist_ids": len(BLACKLIST), "align_min": ALIGN_MIN}
    json.dump(plan, open(os.path.join(BUILD, "plan.json"), "w"), indent=1)
    print(json.dumps(plan, indent=1))


def run(cmd, **kw):
    print("$", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    print(r.stdout[-600:], r.stderr[-900:], flush=True)
    if r.returncode != 0:
        raise SystemExit(f"command failed: {cmd}")
    return r


def read_docs():
    rows = collections.defaultdict(list)
    with open(os.path.join(BUILD, "docs.tsv"), encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\n").split("\t")
            rows[p[0]].append((p[2], int(p[4]), int(p[5])))
    return rows


def cmd_encode(a):
    plan = json.load(open(os.path.join(BUILD, "plan.json")))
    rows = read_docs()
    for name in rows:
        enc = os.path.join(BUILD, name + ".enc.u16")
        out = os.path.join(BUILD, name + ".u16")
        run([FORGE, "tok-encode", L.TOKENIZER, os.path.join(BUILD, name + ".txt"), "--out", enc, "--threads", "1"])
        run([FORGE, "chat-mask", L.TOKENIZER, enc, "--out", out])
        arr = np.fromfile(out, dtype="<u2")
        want_tok = sum(n for _, n, _ in rows[name])
        want_sup = sum(s for _, _, s in rows[name])
        unmasked = int(((arr & 0x8000) == 0).sum())
        ok = len(arr) == want_tok and unmasked == want_sup
        print(f"[encode] {name}: {len(arr)} tokens (plan {want_tok}), unmasked {unmasked} (plan {want_sup}) -> {'OK' if ok else 'MISMATCH'}", flush=True)
        if not ok:
            raise SystemExit(f"accounting mismatch in {name}")
        if name == "zzz_val":
            T = plan["validation_tokens_T"]
            pad = plan["val_file_pad_tokens"]
            if pad:
                arr = np.concatenate([np.full(pad, 0x8000, dtype="<u2"), arr])
            assert len(arr) == 4 * T, (len(arr), 4 * T)
            arr.tofile(out)
            print(f"[encode] zzz_val: padded with {pad} masked eos tokens -> {len(arr)} = 4 x {T}", flush=True)
        os.remove(enc)


def throttled_copy(src, dst, mbps=40.0, chunk=4 << 20):
    t0, done = time.time(), 0
    with open(src, "rb") as f, open(dst, "wb") as g:
        while True:
            b = f.read(chunk)
            if not b:
                break
            g.write(b)
            done += len(b)
            lag = done / (mbps * 1e6) - (time.time() - t0)
            if lag > 0:
                time.sleep(lag)
        g.flush()
        os.fsync(g.fileno())
        try:
            os.posix_fadvise(g.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        except OSError:
            pass
    return done


def cmd_publish(a):
    plan = json.load(open(os.path.join(BUILD, "plan.json")))
    os.makedirs(C.OUT, exist_ok=True)
    names = [n for n in sorted(plan["shard_tokens"]) if n != "zzz_val"] + ["zzz_val"]
    for k, n in enumerate(names):
        t0 = time.time()
        sz = throttled_copy(os.path.join(BUILD, n + ".u16"), os.path.join(C.OUT, n + ".u16"))
        print(f"[publish] {n}.u16 {sz / 1e6:.1f} MB in {time.time() - t0:.0f} s", flush=True)
        if k + 1 < len(names):
            time.sleep(20)
    T = plan["validation_tokens_T"]
    arr = np.fromfile(os.path.join(C.OUT, "zzz_val.u16"), dtype="<u2")
    assert len(arr) == 4 * T
    os.makedirs(os.path.join(C.OUT, "heldout"), exist_ok=True)
    arr[-T:].tofile(os.path.join(C.OUT, "heldout", "sft_val_masked.u16"))
    (arr[-T:] & 0x7FFF).astype("<u2").tofile(os.path.join(C.OUT, "heldout", "sft_val_plain.u16"))
    print(f"[publish] heldout/sft_val_masked.u16 and sft_val_plain.u16: the last {T} tokens of zzz_val.u16")
    print("[publish] done; now write the manifest with human_manifest.py")


def cmd_replay(a):
    """tokens_sft_human_replay = a_replay.u16 + symlinks to the SFT files; the replay part is 12 % of the total (replay / (replay + SFT files) = 0.12)."""
    files = sorted(glob.glob(os.path.join(C.OUT, "*.u16")))
    sft_tokens = sum(os.path.getsize(f) // 2 for f in files)
    rep_tokens = int(round(sft_tokens * 0.12 / 0.88))
    rep_tokens -= rep_tokens % 1024
    src = "/mnt/data/forgeinfer-data/tokens_cool/replay/replay_000.u16"
    os.makedirs(C.OUT_REPLAY, exist_ok=True)
    nbytes = rep_tokens * 2
    with open(src, "rb") as f, open(os.path.join(C.OUT_REPLAY, "a_replay.u16"), "wb") as g:
        left = nbytes
        while left > 0:
            b = f.read(min(left, 8 << 20))
            if not b:
                raise SystemExit("replay source too short")
            g.write(b)
            left -= len(b)
    for f in files:
        link = os.path.join(C.OUT_REPLAY, os.path.basename(f))
        if os.path.lexists(link):
            os.remove(link)
        os.symlink(f, link)
    print(f"[replay] a_replay.u16 {rep_tokens:,d} tokens ({nbytes / 1e6:.1f} MB) + {len(files)} symlinks; replay share {100 * rep_tokens / (rep_tokens + sft_tokens):.1f} % of {rep_tokens + sft_tokens:,d} tokens")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["plan", "encode", "publish", "replay"])
    a = ap.parse_args()
    {"plan": cmd_plan, "encode": cmd_encode, "publish": cmd_publish, "replay": cmd_replay}[a.cmd](a)
