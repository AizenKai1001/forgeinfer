#!/usr/bin/env python3
"""sft_pools.py SOURCE [SOURCE ...] [--quota-m X] [--frac F] [--out DIR] [--debug]: build the FILTERED POOL of one raw source (see ForgeInfer - SFT Data Mix in the research notes).

For every candidate conversation of the source adapter (sft_sources.py), in a seeded random order:
  clean the text -> cut to the first exchanges that fit the token budget -> quality filters (language, other vendors' identity, markup, fabricated links, PII, broken code fences ...)
  -> decontamination against the benchmark test sets (10-gram index, IFEval: 16-gram) -> exact duplicate of a conversation or of a first user prompt -> stratum quota -> accept
until the token quota of the source (or of every stratum) is reached.  Output: sft_work/pool/<source>.jsonl (one JSON per line: id, src, msgs, ntok, nsup, stratum) and <source>.stats.json
(drop reasons, benchmark hits, token / turn / length histograms).  Deterministic: the seed is derived from the source name.
"""
import collections
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sft_lib as L  # noqa: E402
import sft_sources as S  # noqa: E402

POOL = os.path.join(L.WORK, "pool")

# magpie category weights: the categories 'coding' and 'math' are served by the dedicated code / math sources, 'data-analysis' is long numeric text that a 150M model cannot use
MAGPIE_STRATA = {"creative-writing": 0.16, "editing": 0.14, "role-playing": 0.08, "reasoning": 0.13, "brainstorming": 0.10, "planning": 0.08, "advice-seeking": 0.12, "information-seeking": 0.08, "coding": 0.06, "math": 0.03, "data-analysis": 0.02}

# group: the part of the mix this source feeds.  quota_m = pool size in million tokens (the assembler thins the pools down to the final proportions).
CFG = {
    "magpie": dict(frac=0.55, quota_m=46, max_tok=800, max_ex=2, strata=MAGPIE_STRATA, allow_html=True, group="chat", ban=True),
    "ultrachat": dict(frac=0.22, quota_m=15, max_tok=800, max_ex=2, group="chat", ban=True),
    "oasst2": dict(frac=1.0, quota_m=9, max_tok=1300, max_ex=4, group="chat", ban=True),
    "dolly": dict(frac=1.0, quota_m=4, max_tok=1100, max_ex=1, group="chat", ban=True, min_answer_chars=4),
    "capybara": dict(frac=1.0, quota_m=9, max_tok=1000, max_ex=3, group="chat", ban=True),
    "everyday": dict(frac=1.0, quota_m=2, max_tok=900, max_ex=6, group="chat", ban=True, min_answer_chars=3),
    "systemchats": dict(frac=0.45, quota_m=11, max_tok=1000, max_ex=3, group="chat", ban=True),
    "summarize": dict(frac=0.22, quota_m=12, max_tok=1100, max_ex=1, group="writing", ban=True),
    "rewrite": dict(frac=0.35, quota_m=8, max_tok=1100, max_ex=1, group="writing", ban=True),
    "constraints": dict(frac=0.5, quota_m=8, max_tok=1000, max_ex=1, group="writing", ban=True),
    "gsm8k": dict(frac=1.0, quota_m=2.3, max_tok=700, max_ex=1, group="math", ban=False),
    "hendrycks": dict(frac=1.0, quota_m=1.3, max_tok=900, max_ex=1, group="math", ban=False),
    "metamath": dict(frac=0.30, quota_m=8, max_tok=800, max_ex=1, group="math", ban=False),
    "orcamath": dict(frac=0.12, quota_m=5, max_tok=800, max_ex=1, group="math", ban=False),
    "tulu_gsm": dict(frac=0.35, quota_m=4, max_tok=800, max_ex=1, group="math", ban=False),
    "tulu_grade": dict(frac=0.25, quota_m=4, max_tok=1000, max_ex=1, group="math", ban=False),
    "selfoss": dict(frac=0.5, quota_m=11, max_tok=1000, max_ex=1, group="code", allow_html=True, check_urls=False, ban=False),
    "magicoder": dict(frac=0.40, quota_m=11, max_tok=1100, max_ex=1, group="code", allow_html=True, check_urls=False, ban=False),
    "evolcode": dict(frac=0.30, quota_m=5.5, max_tok=1100, max_ex=1, group="code", allow_html=True, check_urls=False, ban=False),
    "tulu_code": dict(frac=0.45, quota_m=5, max_tok=1100, max_ex=1, group="code", allow_html=True, check_urls=False, ban=False),
    "coconot": dict(frac=1.0, quota_m=1.5, max_tok=700, max_ex=1, group="safety", ban=False, ident_groups=("vendor", "cutoff")),
    "wildguard": dict(frac=0.5, quota_m=3, max_tok=500, max_ex=1, group="safety", ban=False, ident_groups=("vendor", "cutoff"), max_user_words=45),
}

IFEVAL_N = 16
RUN_MIN = 4  # consecutive matching 10-grams = a verbatim overlap of 10 + 3 = 13 words


class Decontam:
    """The 10-gram benchmark index (every benchmark but IFEval) and a 16-gram set of the IFEval prompts (their standard instruction sentences would otherwise delete the whole constraints source)."""

    def __init__(self):
        z = np.load(os.path.join(L.WORK, "bench_index.npz"), allow_pickle=True)
        self.n = int(z["n"])
        self.hashes = z["hashes"]
        self.owner = z["owner"]
        self.names = [str(x) for x in z["names"]]
        self.ifeval_id = self.names.index("ifeval")
        self.if_set = set()
        for r in S.jsonl_rows(os.path.join(L.RAW, "google__IFEval", "ifeval_input_data.jsonl")):
            self.if_set.update(L.ngram_hashes(L.word_ids(r["prompt"]), IFEVAL_N).tolist())

    def check(self, text, soft=None):
        """Name of the first benchmark with which the text shares a verbatim run of at least RUN_MIN consecutive matching 10-grams (= 13 or more words in a row, the GPT-3 rule), or None.
        A single shared 10-gram is mostly a common phrase ('one of the most important things you can do is', '1 2 3 4 5 6 7 8 9 10'): such short overlaps are only counted in `soft`."""
        ids = L.word_ids(text)
        h = L.ngram_hashes(ids, self.n)
        if len(h):
            pos = np.searchsorted(self.hashes, h)
            pos[pos >= len(self.hashes)] = 0
            own = self.owner[pos]
            ok = (self.hashes[pos] == h) & (own != self.ifeval_id)
            if ok.any():
                # 10-grams that are mostly numbers (1 2 3 4 5 6 7 8 9 10 ..., a list of primes, a table) are generic, not a copy of a benchmark item
                isnum = np.fromiter((w.isdigit() for w in L.words_of(text)), dtype=np.int32, count=len(ids))
                numdom = np.convolve(isnum, np.ones(self.n, dtype=np.int32), "valid") >= int(0.7 * self.n)
                ok &= ~numdom
            if ok.any():
                idx = np.flatnonzero(ok)
                breaks = np.flatnonzero(np.diff(idx) != 1)
                starts = np.concatenate(([0], breaks + 1))
                ends = np.concatenate((breaks + 1, [len(idx)]))
                lens = ends - starts
                k = int(np.argmax(lens))
                if lens[k] >= RUN_MIN:
                    return self.names[own[idx[starts[k]]]]
                if soft is not None:
                    soft[self.names[own[idx[0]]]] += 1
        h16 = L.ngram_hashes(ids, IFEVAL_N)
        if len(h16) and any(x in self.if_set for x in h16.tolist()):
            return "ifeval"
        return None


def roles_problem(roles):
    """None when the roles are: an optional system message, then user, assistant, user, assistant ... ending with the assistant (at least one exchange)."""
    k = 1 if roles and roles[0] == "system" else 0
    rest = roles[k:]
    if len(rest) < 2:
        return "no_exchange"
    if len(rest) % 2 or any(r != ("user" if i % 2 == 0 else "assistant") for i, r in enumerate(rest)):
        return "bad_role_order"
    return None


def histogram(values, edges):
    cnt = [0] * (len(edges) + 1)
    for v in values:
        k = 0
        while k < len(edges) and v >= edges[k]:
            k += 1
        cnt[k] += 1
    return cnt


LEN_EDGES = [64, 128, 256, 384, 512, 768, 1024, 1536]


def build(name, args, decon):
    cfg = dict(CFG[name])
    quota = (args.quota_m if args.quota_m is not None else cfg["quota_m"]) * 1e6
    frac = args.frac if args.frac is not None else cfg["frac"]
    rng = L.rng_for("pool", name)
    t0 = time.time()
    print(f"[{name}] loading candidates (frac {frac}) ...", flush=True)
    cands = list(S.ADAPTERS[name](rng, frac))
    rng.shuffle(cands)
    print(f"[{name}] {len(cands)} candidates in {time.time() - t0:.0f} s", flush=True)
    strata = cfg.get("strata")
    quota_s = {k: v * quota for k, v in strata.items()} if strata else None
    tok_s = collections.Counter()
    drops = collections.Counter()
    bench_hits = collections.Counter()
    soft_hits = collections.Counter()
    seen_conv, seen_prompt = set(), set()
    accepted, doc_lens, turns, sup_total, tok_total = [], [], collections.Counter(), 0, 0
    n_in = 0
    os.makedirs(POOL, exist_ok=True)
    CH = 1500
    done = False
    for c0 in range(0, len(cands), CH):
        if done:
            break
        chunk = cands[c0 : c0 + CH]
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
                msgs = msgs[:-1]  # a conversation ends with the assistant
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
            if ntok < 24:
                drops["too_short"] += 1
                continue
            stratum = cd["meta"].get("stratum", "")
            if cfg.get("stratum_from") == "category":
                stratum = ""
            if quota_s is not None:
                if stratum not in quota_s:
                    drops["stratum_not_used"] += 1
                    continue
                if tok_s[stratum] >= quota_s[stratum]:
                    drops["stratum_full"] += 1
                    continue
            if cfg.get("max_user_words") and len(msgs[0 if msgs[0][0] == "user" else 1][1].split()) > cfg["max_user_words"]:
                drops["long_prompt"] += 1
                continue
            why = L.basic_checks([list(m) for m in msgs], allow_html=cfg.get("allow_html", False), min_answer_chars=cfg.get("min_answer_chars", 12), ban_words=cfg.get("ban", False),
                                 check_urls=cfg.get("check_urls", True), check_vendor=cfg.get("check_vendor", True), check_pii=True, ident_groups=cfg.get("ident_groups", ("vendor", "boiler", "cutoff")))
            if why:
                drops[why] += 1
                continue
            text = "\n".join(c for _, c in msgs)
            hit = decon.check(text, soft_hits)
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
            tok_s[stratum] += ntok
            doc_lens.append(ntok)
            turns[sum(1 for r, _ in msgs if r == "assistant")] += 1
            if tok_total >= quota or (quota_s is not None and all(tok_s[k] >= v for k, v in quota_s.items())):
                done = True
                break
        if (c0 // CH) % 5 == 0:
            print(f"[{name}] {n_in} processed, {len(accepted)} accepted, {tok_total / 1e6:.1f} M tokens, {time.time() - t0:.0f} s", flush=True)
    out = os.path.join(args.out, f"{name}.jsonl")
    L.write_jsonl(out, accepted)
    stats = {
        "source": name, "group": cfg["group"], "candidates": len(cands), "processed": n_in, "accepted": len(accepted), "tokens": tok_total, "supervised_tokens": sup_total,
        "supervised_fraction": round(sup_total / max(1, tok_total), 4), "mean_doc_tokens": round(tok_total / max(1, len(accepted)), 1),
        "len_edges": LEN_EDGES, "len_hist": histogram(doc_lens, LEN_EDGES), "assistant_turns": dict(sorted(turns.items())), "drops": dict(drops.most_common()),
        "bench_hits": dict(bench_hits.most_common()), "short_overlaps_kept": dict(soft_hits.most_common()), "strata_tokens": dict(tok_s), "quota_tokens": quota, "seconds": round(time.time() - t0), "frac": frac, "config": {k: v for k, v in cfg.items() if k != "strata"},
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
            print(L.render(rec["msgs"])[:1400])


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("sources", nargs="+")
    ap.add_argument("--quota-m", type=float, default=None)
    ap.add_argument("--frac", type=float, default=None)
    ap.add_argument("--out", default=POOL)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    decon = Decontam()
    names = list(CFG) if args.sources == ["all"] else args.sources
    for n in names:
        build(n, args, decon)


if __name__ == "__main__":
    main()
