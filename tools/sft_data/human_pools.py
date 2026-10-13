#!/usr/bin/env python3
"""human_pools.py SOURCE [SOURCE ...] [--quota-m X] [--debug]: build the FILTERED POOL of one human-written source (adapters: human_sources.py; same filters and record format as sft_pools.py).

For every candidate conversation of the adapter (seeded random order, streamed):
  token budget (cut to the first exchanges that fit) -> quality filters (sft_lib.basic_checks: vendors / identity boilerplate, markup, fabricated links, PII, broken fences, truncation, profanity ...; English
  on the prose) -> forum rules for Stack Exchange and Reddit answers (human_lib: no 'the OP', 'in the comments', 'my wife', years of experience ...) -> decontamination against the benchmark test sets
  (10-gram index of human_bench_index.py: a verbatim run of 13 or more words, IFEval 16-grams) -> exact duplicates -> stratum quota -> accept, until the token quota of the source or of every stratum is reached.
Output: sft_work_human/pool/<source>.jsonl (id, src, msgs, ntok, nsup, stratum) and <source>.stats.json (drop reasons, benchmark hits, histograms).  Deterministic: the seed is derived from the source name.
"""
import collections
import itertools
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import human_cfg as C  # noqa: E402
import human_lib as H  # noqa: E402
import human_sources as HS  # noqa: E402
import sft_lib as L  # noqa: E402
from sft_pools import Decontam, histogram, roles_problem  # noqa: E402

# group = the part of the mix a source feeds; Stack Exchange sites and subreddits get their group from the tables below (see group_of)
CFG = {
    "oasst2": dict(quota_m=9, max_tok=1300, max_ex=4, group="chat", ban=True),
    "dolly": dict(quota_m=4, max_tok=1100, max_ex=1, group="chat", ban=True, min_answer_chars=4),
    "no_robots": dict(quota_m=6, max_tok=1100, max_ex=4, group="chat", ban=True, min_answer_chars=4),
    "aya": dict(quota_m=2, max_tok=900, max_ex=1, group="chat", ban=True, min_answer_chars=4),
    "se": dict(quota_m=40, max_tok=760, max_ex=1, group="forum", ban=True, forum="se", allow_html=True, strata={s: w for s, (w, _) in C.SE_SITES.items()}),
    "shp": dict(quota_m=16, max_tok=760, max_ex=1, group="forum", ban=True, forum="reddit", strata=dict(C.SHP_SUBS)),
    "gsm8k": dict(quota_m=2.3, max_tok=700, max_ex=1, group="math", ban=False),
    "hendrycks": dict(quota_m=4, max_tok=800, max_ex=1, group="math", ban=False),
    "aqua": dict(quota_m=6, max_tok=600, max_ex=1, group="math", ban=False),
    "apps": dict(quota_m=4, max_tok=1100, max_ex=1, group="code", allow_html=True, check_urls=False, ban=False),
    "mbpp": dict(quota_m=0.3, max_tok=700, max_ex=1, group="code", allow_html=True, check_urls=False, ban=False),
    "rosetta": dict(quota_m=1.5, max_tok=900, max_ex=1, group="code", allow_html=True, check_urls=False, ban=False),
    "csn": dict(quota_m=9, max_tok=700, max_ex=1, group="code", allow_html=True, check_urls=False, ban=False),
    "squad": dict(quota_m=5, max_tok=1000, max_ex=4, group="qa", ban=False, min_answer_chars=1),
    "coqa": dict(quota_m=6, max_tok=1000, max_ex=8, group="qa", ban=False, min_answer_chars=1),
    "nq": dict(quota_m=1.2, max_tok=200, max_ex=1, group="qa", ban=False, min_answer_chars=1, min_tok=12),
    "trivia": dict(quota_m=1.5, max_tok=200, max_ex=1, group="qa", ban=False, min_answer_chars=1, min_tok=12),
    "cnn": dict(quota_m=6, max_tok=1000, max_ex=1, group="writing", ban=False),
    "dialogsum": dict(quota_m=3, max_tok=800, max_ex=1, group="writing", ban=False),
    "wp": dict(quota_m=9, max_tok=1000, max_ex=1, group="writing", ban=True),
}

SE_GROUP = {}
for _s in ("unix", "superuser", "askubuntu", "serverfault", "softwareengineering", "codereview", "security", "crypto", "apple", "android", "webmasters", "networkengineering", "dba", "raspberrypi",
           "arduino", "blender", "gamedev", "tex", "retrocomputing", "vi", "emacs", "Stackoverflow"):
    SE_GROUP[_s] = "code"
for _s in ("cooking", "diy", "gardening", "travel", "money", "workplace", "academia", "lifehacks", "outdoors", "bicycles", "fitness", "pets", "woodworking", "crafts", "coffee", "beer", "homebrew",
           "martialarts", "sports", "3dprinting", "freelancing", "expatriates", "sustainability", "interpersonal", "law", "photo", "graphicdesign", "ux", "pm", "opensource", "softwarerecs", "hardwarerecs"):
    SE_GROUP[_s] = "howto"
SE_GROUP["math"] = "math"
SE_GROUP["matheducators"] = "explain"
SE_GROUP["writers"] = "writing"
SHP_GROUP = {"askculinary": "howto", "askbaking": "howto", "askcarguys": "howto", "askhr": "howto"}


def group_of(src, stratum):
    """The category of a conversation for the manifest: chat, writing, code, math, explain, qa, howto, identity."""
    if src == "se":
        return SE_GROUP.get(stratum.split(".")[0], "explain")
    if src == "shp":
        return SHP_GROUP.get(stratum, "explain")
    if src == "identity":
        return "identity"
    return CFG[src]["group"]


class HumanDecontam(Decontam):
    """Decontam with the human index (MBPP = test problems only); the IFEval 16-gram set comes from the first build's raw file."""

    def __init__(self):
        z = np.load(os.path.join(C.WORK, "bench_index_human.npz"), allow_pickle=True)
        self.n = int(z["n"])
        self.hashes = z["hashes"]
        self.owner = z["owner"]
        self.names = [str(x) for x in z["names"]]
        self.ifeval_id = self.names.index("ifeval")
        self.if_set = set()
        for r in HS.S.jsonl_rows(os.path.join(L.RAW, "google__IFEval", "ifeval_input_data.jsonl")):
            self.if_set.update(L.ngram_hashes(L.word_ids(r["prompt"]), 16).tolist())


class Ctx:
    def __init__(self, name, quota_s):
        self.rng = L.rng_for("human-pool", name)
        self.tok_s = collections.Counter()
        self.quota_s = quota_s

    def full(self, stratum):
        return self.quota_s is not None and stratum in self.quota_s and self.tok_s[stratum] >= self.quota_s[stratum]


def build(name, args, decon):
    cfg = dict(CFG[name])
    quota = (args.quota_m if args.quota_m is not None else cfg["quota_m"]) * 1e6
    strata = cfg.get("strata")
    wsum = sum(strata.values()) if strata else 0
    quota_s = {k: v / wsum * quota for k, v in strata.items()} if strata else None
    ctx = Ctx(name, quota_s)
    drops = collections.Counter()
    bench_hits = collections.Counter()
    soft_hits = collections.Counter()
    seen_conv, seen_prompt = set(), set()
    accepted, doc_lens, turns, sup_total, tok_total = [], [], collections.Counter(), 0, 0
    n_in = 0
    t0 = time.time()
    os.makedirs(args.out, exist_ok=True)
    gen = HS.ADAPTERS[name](ctx)
    CH = 1500
    done = False
    min_tok = cfg.get("min_tok", 24)
    while not done:
        chunk = list(itertools.islice(gen, CH))
        if not chunk:
            break
        prepared = []
        for cd in chunk:
            n_in += 1
            msgs = []
            for r, c in cd["msgs"]:
                c = L.clean(c)
                if r == "system" and not c:
                    continue
                msgs.append((r, c))
            if not msgs or any(not c for _, c in msgs):
                drops["empty_message"] += 1
                continue
            if sum(len(c) for _, c in msgs) > 60000:
                drops["too_long_chars"] += 1
                continue
            if msgs[-1][0] == "user":
                msgs = msgs[:-1]
            why = roles_problem([r for r, _ in msgs])
            if why:
                drops[why] += 1
                continue
            prepared.append((cd, msgs))
        if not prepared:
            continue
        counts = L.content_token_counts([m for _, m in prepared])
        for (cd, msgs), cnt in zip(prepared, counts):
            fit = L.fit_prefix(msgs, cnt, cfg["max_tok"], cfg["max_ex"])
            if fit is None:
                drops["too_long_tokens"] += 1
                continue
            msgs, cnt = fit
            ntok = L.doc_tokens(msgs, cnt)
            if ntok < min_tok:
                drops["too_short"] += 1
                continue
            stratum = cd["meta"].get("stratum", "")
            if quota_s is not None:
                if stratum not in quota_s:
                    drops["stratum_not_used"] += 1
                    continue
                if ctx.tok_s[stratum] >= quota_s[stratum]:
                    drops["stratum_full"] += 1
                    continue
            full = "\n".join(c for _, c in msgs)
            why = L.basic_checks([list(m) for m in msgs], allow_html=cfg.get("allow_html", False), min_answer_chars=cfg.get("min_answer_chars", 12), ban_words=cfg.get("ban", False),
                                 check_urls=cfg.get("check_urls", True), check_vendor=cfg.get("check_vendor", True), check_pii=True, need_english=False,
                                 ident_groups=cfg.get("ident_groups", ("vendor", "boiler", "cutoff")))
            if why:
                drops[why] += 1
                continue
            if not H.english_ok_human(full):
                drops["not_english"] += 1
                continue
            if cfg.get("forum"):
                strict = cfg["forum"] == "reddit"
                why = H.forum_question_problem(msgs[0][1]) or H.forum_answer_problem(msgs[-1][1], strict=strict)
                if why:
                    drops[why] += 1
                    continue
            hit = decon.check(full, soft_hits)
            if hit:
                drops["decontam"] += 1
                bench_hits[hit] += 1
                continue
            h1 = L.conv_hash(msgs)
            h2 = L.prompt_hash(msgs)
            if h1 in seen_conv or h2 in seen_prompt:
                drops["duplicate"] += 1
                continue
            seen_conv.add(h1)
            seen_prompt.add(h2)
            nsup = L.supervised_tokens(msgs, cnt)
            rec = {"id": cd["id"], "src": name, "msgs": [list(m) for m in msgs], "ntok": ntok, "nsup": nsup, "stratum": stratum}
            accepted.append(rec)
            tok_total += ntok
            sup_total += nsup
            ctx.tok_s[stratum] += ntok
            doc_lens.append(ntok)
            turns[sum(1 for r, _ in msgs if r == "assistant")] += 1
            if tok_total >= quota or (quota_s is not None and all(ctx.tok_s[k] >= v for k, v in quota_s.items())):
                done = True
                break
        if (n_in // CH) % 5 == 0:
            print(f"[{name}] {n_in} processed, {len(accepted)} accepted, {tok_total / 1e6:.1f} M tokens, {time.time() - t0:.0f} s", flush=True)
    out = os.path.join(args.out, f"{name}.jsonl")
    L.write_jsonl(out, accepted)
    stats = {
        "source": name, "group": cfg["group"], "processed": n_in, "accepted": len(accepted), "tokens": tok_total, "supervised_tokens": sup_total,
        "supervised_fraction": round(sup_total / max(1, tok_total), 4), "mean_doc_tokens": round(tok_total / max(1, len(accepted)), 1),
        "len_edges": [64, 128, 256, 384, 512, 768, 1024, 1536], "len_hist": histogram(doc_lens, [64, 128, 256, 384, 512, 768, 1024, 1536]), "assistant_turns": dict(sorted(turns.items())),
        "drops": dict(drops.most_common()), "bench_hits": dict(bench_hits.most_common()), "short_overlaps_kept": dict(soft_hits.most_common()), "strata_tokens": dict(ctx.tok_s), "quota_tokens": quota,
        "seconds": round(time.time() - t0), "config": {k: v for k, v in cfg.items() if k != "strata"},
    }
    json.dump(stats, open(os.path.join(args.out, f"{name}.stats.json"), "w"), indent=1)
    print(f"[{name}] DONE accepted {len(accepted)} / {n_in} processed, {tok_total / 1e6:.2f} M tokens (supervised {100 * sup_total / max(1, tok_total):.0f} %), mean {tok_total / max(1, len(accepted)):.0f} tokens/doc, {time.time() - t0:.0f} s")
    print(f"[{name}] drops: {dict(drops.most_common())}")
    if bench_hits:
        print(f"[{name}] benchmark hits: {dict(bench_hits.most_common())}")
    if args.debug:
        import random

        r2 = random.Random(5)
        for rec in r2.sample(accepted, min(4, len(accepted))):
            print("---- sample", rec["id"], rec["ntok"], "tokens")
            print(L.render(rec["msgs"])[:1600])


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("sources", nargs="+")
    ap.add_argument("--quota-m", type=float, default=None)
    ap.add_argument("--out", default=C.POOL)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()
    decon = HumanDecontam()
    names = list(CFG) if args.sources == ["all"] else args.sources
    for n in names:
        build(n, args, decon)


if __name__ == "__main__":
    main()
