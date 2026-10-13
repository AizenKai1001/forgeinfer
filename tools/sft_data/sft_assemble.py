#!/usr/bin/env python3
"""sft_assemble.py plan | encode | publish: assemble the Forge-1 SFT mix v2 from the filtered pools (see ForgeInfer - SFT Data Mix in the research notes).

  plan      load the pools (sft_work/pool/*.jsonl), remove exact and near duplicates ACROSS sources (MinHash of the first user message, Jaccard >= 0.8 / 0.7), thin every source to its token target,
            hold out the validation conversations, oversample the small hand-written sets, shuffle (seed), cut into NSHARDS training shards and the validation shard, and write the NUL-separated
            ChatML corpus files + docs.tsv (one line per conversation) to the build directory.
  encode    for every corpus file: `forge tok-encode` -> `forge chat-mask` (bit 15 = not a target) and check the token counts against the plan (every document and every supervised token is accounted for).
  publish   copy the shards to /mnt/data/forgeinfer-data/tokens_sft_mix2 one at a time (throttled to 40 MB/s, 20 s pause between files) and write MANIFEST.txt.

The validation shard zzz_val.u16 has the layout [padding][training documents][validation documents] with  length = 4 x (validation tokens):  the trainer (crates/forge-cli/src/train_cmd.rs, Data::open) takes
the LAST file, reserves min(--val-tokens, tokens / 4) of its tail as the validation region and trains on the rest of that file, so the validation region is exactly the held-out documents when --val-tokens >= T.
"""
import argparse
import collections
import glob
import json
import os
import re
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sft_lib as L  # noqa: E402

POOL = os.path.join(L.WORK, "pool")
BUILD = os.environ.get("SFT_BUILD", "/tmp/sft_mix2_build")
FORGE = os.environ.get("FORGE_SFT_BIN", "/tmp/forge_sft")
NSHARDS = 4
VAL_TARGET = 2_000_000

# source -> final tokens in millions (before oversampling); the order is the priority of the cross-source deduplication (human-written first)
TARGETS_M = collections.OrderedDict([
    ("identity", 99), ("safety_hw", 99), ("oasst2", 9), ("dolly", 9), ("gsm8k", 9), ("hendrycks", 9), ("everyday", 9),
    ("magpie", 38), ("ultrachat", 11), ("capybara", 7), ("systemchats", 8), ("summarize", 9), ("rewrite", 6), ("constraints", 6),
    ("metamath", 5.0), ("orcamath", 3.0), ("tulu_gsm", 2.5), ("tulu_grade", 2.5),
    ("selfoss", 6.5), ("magicoder", 6.0), ("evolcode", 4.0), ("tulu_code", 2.5),
    ("coconot", 1.0), ("wildguard", 1.0),
])
OVERSAMPLE = {"identity": 5, "safety_hw": 2, "everyday": 2}
NODEDUP = {"identity", "safety_hw"}  # hand-written: the same prompt with different answer wordings is intended

# license / provenance table of the sources (kept in the manifest and in the research note)
META = {
    "identity": ("hand-written for Forge-1 (sft_handwritten.py)", "own work", "human (templated variants)", "identity"),
    "safety_hw": ("hand-written refusals and harmless look-alikes (sft_handwritten.py)", "own work", "human (templated variants)", "safety"),
    "oasst2": ("OpenAssistant/oasst2, English, best-ranked path per tree", "Apache-2.0", "human volunteers", "chat"),
    "dolly": ("databricks/databricks-dolly-15k", "CC-BY-SA-3.0 (share-alike)", "human (Databricks employees)", "chat"),
    "gsm8k": ("openai/gsm8k train (main)", "MIT", "human", "math"),
    "hendrycks": ("EleutherAI/hendrycks_math train, levels 1-3", "MIT", "human", "math"),
    "everyday": ("HuggingFaceTB/smoltalk everyday-conversations (=everyday-conversations-llama3.1-2k)", "Apache-2.0", "LLM (Llama-3.1-70B-Instruct)", "chat"),
    "magpie": ("HuggingFaceTB/smoltalk smol-magpie-ultra, quality good/excellent, difficulty <= medium", "Apache-2.0 (SmolTalk card, new subsets)", "LLM (Llama-3.1-405B-Instruct, Magpie)", "chat"),
    "ultrachat": ("HuggingFaceH4/ultrachat_200k train_sft", "MIT", "LLM (ChatGPT)", "chat"),
    "capybara": ("LDJnr/Capybara CapybaraPure_Decontaminated", "Apache-2.0", "LLM (Amplify-Instruct synthetic multi-turn)", "chat"),
    "systemchats": ("HuggingFaceTB/smoltalk systemchats-30k (SystemChat-2.0)", "Apache-2.0", "LLM (synthetic)", "chat"),
    "summarize": ("HuggingFaceTB/smoltalk smol-summarize", "Apache-2.0 (SmolTalk card, new subsets)", "LLM (distilabel; generator not stated on the card)", "writing"),
    "rewrite": ("HuggingFaceTB/smoltalk smol-rewrite", "Apache-2.0 (SmolTalk card, new subsets)", "LLM (distilabel; generator not stated on the card)", "writing"),
    "constraints": ("HuggingFaceTB/smoltalk smol-constraints (IFEval-decontaminated by its authors)", "Apache-2.0 (SmolTalk card, new subsets)", "LLM (distilabel; generator not stated on the card)", "writing"),
    "metamath": ("meta-math/MetaMathQA, GSM types + some MATH_AnsAug", "MIT", "LLM (GPT-3.5-Turbo rewrites of GSM8K / MATH TRAIN questions, per the MetaMath paper)", "math"),
    "orcamath": ("microsoft/orca-math-word-problems-200k", "MIT", "LLM (Azure GPT-4-Turbo)", "math"),
    "tulu_gsm": ("allenai/tulu-3-sft-mixture source tulu_v3.9_open_math_2_gsm8k_50k (a sample of nvidia/OpenMathInstruct-2, inferred from the name)", "CC-BY-4.0 (OpenMathInstruct-2 card; not listed in the Tulu card)", "LLM (Llama-3.1-405B-Instruct)", "math"),
    "tulu_grade": ("allenai/tulu-3-sft-personas-math-grade (via the Tulu-3 mixture)", "ODC-BY-1.0 (Tulu-3 card)", "LLM (synthetic, persona-driven)", "math"),
    "selfoss": ("HuggingFaceTB/smoltalk self-oss-instruct (= bigcode/self-oss-instruct-sc2-exec-filter-50k, Python, execution-filtered)", "ODC-BY (bigcode card)", "LLM (StarCoder2, self-instruct)", "code"),
    "magicoder": ("ise-uiuc/Magicoder-OSS-Instruct-75K (decontaminated file), Python + some shell/TypeScript", "MIT", "LLM (gpt-3.5-turbo-1106, card)", "code"),
    "evolcode": ("allenai/tulu-3-sft-mixture source evol_codealpaca_heval_decontaminated (theblackcat102/evol-codealpaca-v1), Python items", "Apache-2.0 (its card and the Tulu-3 card)", "LLM (GPT-4, card)", "code"),
    "tulu_code": ("allenai/tulu-3-sft-personas-code (via the Tulu-3 mixture)", "ODC-BY-1.0 (Tulu-3 card and its own card)", "LLM (synthetic, persona-driven)", "code"),
    "coconot": ("allenai/coconot (via the Tulu-3 mixture): requests a model should decline or answer with care", "ODC-BY-1.0 (Tulu-3 card; the CoCoNot card also links the Ai2 ImpACT low-risk license: ambiguous)", "LLM (GPT-4, card)", "safety"),
    "wildguard": ("allenai/wildguardmix train (via the Tulu-3 mixture; the original is gated and was NOT accessed), prompts <= 45 words", "Apache-2.0 per the Tulu-3 card (original card: ODC-BY)", "LLM (GPT-4 responses)", "safety"),
}
LEN_EDGES = [64, 128, 256, 384, 512, 768, 1024, 1536, 2048]


_ENTITY = re.compile(r"&(?:#\d{2,5}|#x[0-9a-fA-F]{2,4}|quot|amp|lt|gt|nbsp|apos);")  # HTML entities in the text: scraped-markup junk (3 % of Evol-CodeAlpaca)
LATE_DROPS = collections.Counter()


_WORDS_REQ = re.compile(r"(\d[\d,]*)\s*(?:-|to|and)?\s*(\d[\d,]*)?\s*[- ]?words?\b", re.I)
LENGTH_CHECKED = {"magpie", "ultrachat", "capybara", "oasst2", "systemchats", "dolly"}  # the sources that are not built around exact length constraints


def length_mismatch(msgs):
    """The first user message asks for N >= 300 words and the first answer has fewer than 60 % of them: such pairs teach a small model to ignore length requests (typical of UltraChat)."""
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


_MC_OPTION = re.compile(r"(?:^|\n|\s)\(?([A-D])[\).:]\s")
MC_FILTERED = {"evolcode"}  # Evol-CodeAlpaca contains MMLU-style multiple-choice questions (their option lists matched the MMLU test items): drop every item with an A/B/C/D option list


_AS_AI = re.compile(r"\bas an? (?:ai|a\.i\.|artificial intelligence|language model|text-based|virtual assistant|digital assistant|chatbot)\b", re.I)
AS_AI_ALLOWED = {"coconot", "wildguard", "identity", "safety_hw"}  # 'as an AI ...' is the most overused boilerplate of chat data (19 % of SystemChat): kept only where it is the point (the refusal / limits sources)


def has_as_ai(msgs):
    return bool(_AS_AI.search("\n".join(c for r, c in msgs if r != "user")))


def has_mc_options(msgs):
    user = "\n".join(c for r, c in msgs if r == "user")
    return len({m.group(1) for m in _MC_OPTION.finditer(user)}) >= 3


def _load_blacklist():
    """id -> reason: pool_blacklist_fnames.txt (sft_fn_scan.py: the document defines a distinctive HumanEval / MBPP function name) or pool_blacklist_<sources>.txt (sft_audit.py pools: a verbatim run against the evaluation files)."""
    ids = {}
    for p in sorted(glob.glob(os.path.join(L.WORK, "pool_blacklist_*.txt"))):
        why = "benchmark_function_name" if p.endswith("_fnames.txt") else "eval_file_overlap"
        for x in open(p, encoding="utf-8"):
            if x.strip():
                ids.setdefault(x.strip(), why)
    return ids


BLACKLIST = _load_blacklist()  # ids to drop, with the reason


def load_pool(name):
    p = os.path.join(POOL, name + ".jsonl")
    if not os.path.exists(p):
        return []
    out = []
    for r in L.read_jsonl(p):
        if r["id"] in BLACKLIST:
            LATE_DROPS[(name, BLACKLIST[r["id"]])] += 1
            continue
        if name not in NODEDUP and _ENTITY.search("\n".join(c for _, c in r["msgs"])):
            LATE_DROPS[(name, "html_entity")] += 1
            continue
        if name not in AS_AI_ALLOWED and has_as_ai(r["msgs"]):
            LATE_DROPS[(name, "as_an_ai_boilerplate")] += 1
            continue
        if name in MC_FILTERED and has_mc_options(r["msgs"]):
            LATE_DROPS[(name, "multiple_choice_items")] += 1
            continue
        if name in LENGTH_CHECKED and length_mismatch(r["msgs"]):
            LATE_DROPS[(name, "length_request_not_met")] += 1
            continue
        out.append(r)
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
    # near duplicates
    sig_k, sigs, texts = [], [], {}
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
    S = np.stack(sigs)  # N x 64
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
        rng = L.rng_for("assemble-shuffle", s)
        rng.shuffle(r)
        recs[s] = r
        print(f"[plan] {s}: {len(r)} pool records, {sum(x['ntok'] for x in r) / 1e6:.2f} M tokens", flush=True)
    recs, dstats = dedup_across(recs)
    # thin every source to its target
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
    # validation split: every source gives the share of its tokens in the mix (hand-written sets: 8 % of their conversations)
    vrng = L.rng_for("assemble-val")
    val, train = [], []
    for s, r in selected.items():
        if s in OVERSAMPLE and s in ("identity", "safety_hw"):
            share = 0.08
            nval = max(2, int(round(share * len(r))))
        else:
            nval = int(round(VAL_TARGET * sum(x["ntok"] for x in r) / total / max(1.0, np.mean([x["ntok"] for x in r]))))
        idx = list(range(len(r)))
        vrng.shuffle(idx)
        vset = set(idx[:nval])
        for j, x in enumerate(r):
            (val if j in vset else train).append(x)
    # trim the validation set to about VAL_TARGET tokens by moving surplus documents back to the training side
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
    # oversampling and shuffling of the training side
    train_all = []
    for x in train:
        train_all.extend([x] * OVERSAMPLE.get(x["src"], 1))
    trng = L.rng_for("assemble-train")
    trng.shuffle(train_all)
    # the training part of the validation file: the last documents of the shuffled list, 3 T tokens minus the padding
    tail, tail_tok = [], 0
    while train_all and tail_tok + train_all[-1]["ntok"] <= 3 * T:
        x = train_all.pop()
        tail.append(x)
        tail_tok += x["ntok"]
    pad = 3 * T - tail_tok
    tail.reverse()
    total_train = sum(x["ntok"] for x in train_all)
    # shards of about equal size
    shards, cur, cur_tok, per = [], [], 0, total_train / NSHARDS
    for x in train_all:
        cur.append(x)
        cur_tok += x["ntok"]
        if cur_tok >= per and len(shards) < NSHARDS - 1:
            shards.append(cur)
            cur, cur_tok = [], 0
    shards.append(cur)
    # write corpus files and docs.tsv
    docs = open(os.path.join(BUILD, "docs.tsv"), "w", encoding="utf-8")
    docs.write("file\tindex\tsrc\tstratum\tntok\tnsup\tturns\thas_system\tid\n")
    files = []
    for k, sh in enumerate(shards):
        files.append((f"sft_{k:03d}", sh))
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
            "shard_tokens": sizes, "train_tokens_total": total_train + tail_tok, "dedup_drops": {f"{s}:{r}": n for (s, r), n in sorted(dstats.items())}, "late_filter_drops": {f"{s}:{r}": n for (s, r), n in sorted(LATE_DROPS.items())}, "oversample": OVERSAMPLE, "targets_m": TARGETS_M,
            "val_target": VAL_TARGET, "seconds": round(time.time() - t0)}
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
        run([FORGE, "tok-encode", L.TOKENIZER, os.path.join(BUILD, name + ".txt"), "--out", enc, "--threads", "2"])
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
                arr = np.concatenate([np.full(pad, 0x8000, dtype="<u2"), arr])  # masked <|endoftext|> (id 0) tokens in front
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
    return done


def cmd_publish(a):
    plan = json.load(open(os.path.join(BUILD, "plan.json")))
    os.makedirs(L.OUT, exist_ok=True)
    names = [n for n in sorted(plan["shard_tokens"]) if n != "zzz_val"] + ["zzz_val"]
    for k, n in enumerate(names):
        t0 = time.time()
        sz = throttled_copy(os.path.join(BUILD, n + ".u16"), os.path.join(L.OUT, n + ".u16"))
        print(f"[publish] {n}.u16 {sz / 1e6:.1f} MB in {time.time() - t0:.0f} s", flush=True)
        if k + 1 < len(names):
            time.sleep(20)
    # the held-out region alone, in a sub-folder because the trainer reads every *.u16 of the directory it is given:
    #   sft_val_masked.u16  with the 'not a target' bit (for code that understands it: the Rust and JAX trainers' Data class)
    #   sft_val_plain.u16   bit 15 cleared (for `forge eval-ppl`, which reads the ids raw and would fault on ids >= 32768: it scores every token of the chat text, prompts included)
    T = plan["validation_tokens_T"]
    arr = np.fromfile(os.path.join(L.OUT, "zzz_val.u16"), dtype="<u2")
    assert len(arr) == 4 * T
    os.makedirs(os.path.join(L.OUT, "heldout"), exist_ok=True)
    arr[-T:].tofile(os.path.join(L.OUT, "heldout", "sft_val_masked.u16"))
    (arr[-T:] & 0x7FFF).astype("<u2").tofile(os.path.join(L.OUT, "heldout", "sft_val_plain.u16"))
    print(f"[publish] heldout/sft_val_masked.u16 and sft_val_plain.u16: the last {T} tokens of zzz_val.u16")
    print("[publish] done; now write the manifest with sft_manifest.py")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["plan", "encode", "publish"])
    a = ap.parse_args()
    {"plan": cmd_plan, "encode": cmd_encode, "publish": cmd_publish}[a.cmd](a)
