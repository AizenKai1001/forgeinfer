#!/usr/bin/env python3
"""human_note_tables.py [--out DIR]: the numbers and tables of the research note 'ForgeInfer - SFT Data Mix (human-written only)', computed from the build (docs.tsv, plan.json), the pool statistics, the manifest of
the published files and contam_human.json, so that no number of the note is typed by hand.  With --out DIR it writes DIR/values.json (flat key -> number / string, used by the note template) and the markdown tables
DIR/*.md; without it only prints the tables."""
import collections
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import human_assemble as A  # noqa: E402
import human_cfg as C  # noqa: E402
import human_meta as M  # noqa: E402  (META and GROUP_NAMES)
import human_pools as HP  # noqa: E402

OUT = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else None
plan = json.load(open(os.path.join(C.BUILD, "plan.json")))
man = json.load(open(os.path.join(C.OUT, "manifest.json")))
n_tail = plan["val_file_tail_train_docs"]
rows = []
with open(os.path.join(C.BUILD, "docs.tsv"), encoding="utf-8") as f:
    next(f)
    for line in f:
        p = line.rstrip("\n").split("\t")
        rows.append((p[0], int(p[1]), p[2], p[3], int(p[4]), int(p[5]), int(p[6]), int(p[7]), p[0] == "zzz_val" and int(p[1]) >= n_tail, p[8]))
train = [r for r in rows if not r[8]]
val = [r for r in rows if r[8]]
tot = sum(r[4] for r in train)
sup = sum(r[5] for r in train)
uniq = {}
for r in train:
    uniq.setdefault(r[9], r)
V = {}
V["n_conv_all"] = len(rows)
V["n_conv_train_copies"] = len(train)
V["n_conv_train_unique"] = len(uniq)
V["n_conv_val"] = len(val)
V["tok_train"] = tot
V["tok_train_unique"] = sum(r[4] for r in uniq.values())
V["tok_val"] = sum(r[4] for r in val)
V["tok_all"] = tot + V["tok_val"]
V["tok_files"] = sum(c["tokens"] for c in man["files"].values())
V["sup_tok"] = sup
V["sup_pct"] = 100 * sup / tot
V["val_sup_pct"] = 100 * sum(r[5] for r in val) / V["tok_val"]
lens = np.array([r[4] for r in train])
w = np.sort(lens)
cs = np.cumsum(w)
V["len_median_conv"] = int(np.median(lens))
V["len_median_tok"] = int(w[np.searchsorted(cs, cs[-1] / 2)])
V["len_p5"], V["len_p25"], V["len_p75"], V["len_p95"], V["len_p99"] = (int(np.quantile(lens, q)) for q in (0.05, 0.25, 0.75, 0.95, 0.99))
V["len_max"] = int(lens.max())
turns = collections.Counter(min(r[6], 4) for r in train)
V["turn1"], V["turn2"], V["turn3"], V["turn4"] = turns[1], turns[2], turns[3], turns[4]
V["multi_conv_pct"] = 100 * (len(train) - turns[1]) / len(train)
V["multi_tok_pct"] = 100 * sum(r[4] for r in train if r[6] >= 2) / tot
V["sys_n"] = sum(r[7] for r in train)
V["sys_pct"] = 100 * V["sys_n"] / len(train)
V["T"] = plan["validation_tokens_T"]
V["val_pad"] = plan["val_file_pad_tokens"]
V["val_tail_docs"] = plan["val_file_tail_train_docs"]
V["val_tail_tokens"] = plan["val_file_tail_train_tokens"]
V["val_file_tokens"] = plan["val_file_tokens"]
for fn, c in man["files"].items():
    V["mb_" + fn.replace(".u16", "")] = c["bytes"] / 1e6
    V["tok_" + fn.replace(".u16", "")] = c["tokens"]
by = collections.OrderedDict()
for s in A.TARGETS_M:
    rs = [r for r in train if r[2] == s]
    by[s] = (sum(r[4] for r in rs), len(rs), sum(r[5] for r in rs))
    V["share_" + s] = 100 * by[s][0] / tot
    V["tokM_" + s] = by[s][0] / 1e6
    V["conv_" + s] = by[s][1]
grp = collections.Counter()
gsup = collections.Counter()
gn = collections.Counter()
for r in train:
    g = HP.group_of(r[2], r[3])
    grp[g] += r[4]
    gsup[g] += r[5]
    gn[g] += 1
for g in M.GROUP_NAMES:
    V["cat_" + g] = 100 * grp[g] / tot
    V["catM_" + g] = grp[g] / 1e6
    V["catsup_" + g] = 100 * gsup[g] / max(1, grp[g])
se = collections.Counter()
sites = collections.Counter()
for r in train:
    if r[2] == "se":
        se[HP.group_of("se", r[3])] += r[4]
        sites[r[3]] += r[4]
V["se_sites"] = len(sites)
for g in ("explain", "code", "howto", "math", "writing"):
    V["se_" + g + "M"] = se[g] / 1e6
V["se_top"] = ", ".join(f"{s.replace('.stackexchange.com', '').replace('.com', '')} {t / 1e6:.2f}" for s, t in sites.most_common(10))
shp = collections.Counter()
for r in train:
    if r[2] == "shp":
        shp[r[3]] += r[4]
V["shp_top"] = ", ".join(f"{s} {t / 1e6:.2f}" for s, t in shp.most_common(6))
lic = collections.Counter()
for s, d in man["sources"].items():
    L_ = M.META[s][1]
    key = "nc" if "NON-COMMERCIAL" in L_ else ("sa" if ("share-alike" in L_ or "SA-" in L_ or "GFDL" in L_) else ("none" if ("none stated" in L_ or "unknown" in L_) else "perm"))
    lic[key] += d["train_tokens"]
for k in ("nc", "sa", "none", "perm"):
    V["lic_" + k] = 100 * lic[k] / tot
cj = os.path.join(C.WORK, "contam_human.json")
if os.path.exists(cj):
    cd = json.load(open(cj))
    V["contam_rounds"] = len(cd["rounds"])
    V["contam_removed"] = cd["final"]["blacklist_total"]
    V["contam_r1_windows"] = cd["rounds"][0]["hit_windows_16_tokens"]
    V["contam_final_windows"] = cd["final"]["hit_windows_16_tokens"]
    V["contam_final_short"] = cd["final"]["conversations_flagged_this_round"]
    V["contam_10gram_windows"] = cd["final"]["hit_windows_10_tokens (weak, not removed)"]
sj = os.path.join(C.WORK, "contam_human_spans.json")
if os.path.exists(sj):
    hist = json.load(open(sj))
    live = [h for h in hist if not h.get("dry")]
    V["contam_spans_last"] = live[-1]["new_ids"]
    V["contam_spans_docs"] = hist[-1]["strong_10_token_documents"]  # the last (dry) count: strong 10-token documents in the FINAL files
    V["contam_spans_total"] = sum(h["new_ids"] for h in live)
    V["contam_spans_series"] = ", ".join(str(h["new_ids"]) for h in live)
V["blacklist_eval"] = sum(1 for _ in open(os.path.join(C.WORK, "blacklist_eval.txt")))
V["blacklist_fnames"] = sum(1 for _ in open(os.path.join(C.WORK, "blacklist_fnames.txt")))
V["blacklist_contam"] = sum(1 for _ in open(os.path.join(C.WORK, "blacklist_contam.txt")))
V["contam_blacklisted"] = V["blacklist_contam"]
ld = plan["late_filter_drops"]
for k in ("llm_style_text", "question_answer_not_aligned", "contamination_check", "eval", "fnames", "as_an_ai_boilerplate", "length_request_not_met"):
    V["late_" + k] = sum(n for key, n in ld.items() if key.endswith(":" + k))
V["dedup_exact"] = sum(n for key, n in plan["dedup_drops"].items() if key.endswith(":exact"))
V["dedup_near"] = sum(n for key, n in plan["dedup_drops"].items() if key.endswith(":near"))
ps = {s: json.load(open(os.path.join(C.POOL, s + ".stats.json"))) for s in A.TARGETS_M if s != "identity"}
V["pool_candidates"] = sum(p["processed"] for p in ps.values())
V["pool_accepted"] = sum(p["accepted"] for p in ps.values())
V["pool_tokens_M"] = sum(p["tokens"] for p in ps.values()) / 1e6
bench = collections.Counter()
for p in ps.values():
    for b, n in p.get("bench_hits", {}).items():
        bench[b] += n
V["pool_decontam"] = sum(bench.values())
V["pool_decontam_detail"] = ", ".join(f"{b} {n}" for b, n in bench.most_common())
files = sorted(os.listdir(C.OUT))
rep_tokens = os.path.getsize(os.path.join(C.OUT_REPLAY, "a_replay.u16")) // 2
V["replay_tokens"] = rep_tokens
V["replay_total"] = rep_tokens + V["tok_files"]
V["replay_pct"] = 100 * rep_tokens / V["replay_total"]
V["steps_2_epochs"] = int(round(2 * V["replay_total"] / 65536 / 50.0) * 50)
V["epochs_at_steps"] = V["steps_2_epochs"] * 65536 / V["replay_total"]
V["hours_2060"] = V["steps_2_epochs"] * 65536 / 14400 / 3600
V["minutes_a100"] = V["steps_2_epochs"] * 65536 / 78900 / 60

lines = []
P = lines.append
P("| source | tokens (M) | share | conversations | mean tokens | targets % |")
P("|---|---|---|---|---|---|")
for s, (t, n, sp) in by.items():
    P(f"| `{s}` | {t / 1e6:.2f} | {100 * t / tot:.1f} % | {n:,d} | {t / max(1, n):.0f} | {100 * sp / max(1, t):.0f} |")
src_table = "\n".join(lines)
lines = []
P = lines.append
P("| category | tokens (M) | share | conversations | targets % |")
P("|---|---|---|---|---|")
for g, name in M.GROUP_NAMES.items():
    if grp[g]:
        P(f"| {name} | {grp[g] / 1e6:.2f} | {100 * grp[g] / tot:.1f} % | {gn[g]:,d} | {100 * gsup[g] / grp[g]:.0f} |")
cat_table = "\n".join(lines)
lines = []
P = lines.append
P("| source | candidates examined | pool accepted | pool tokens (M) | in the mix (M) |")
P("|---|---|---|---|---|")
for s in A.TARGETS_M:
    if s == "identity":
        continue
    st = ps[s]
    P(f"| `{s}` | {st['processed']:,d} | {st['accepted']:,d} | {st['tokens'] / 1e6:.2f} | {by[s][0] / 1e6:.2f} |")
funnel_table = "\n".join(lines)
print(f"conversations {V['n_conv_all']:,d} (training copies {V['n_conv_train_copies']:,d}, unique {V['n_conv_train_unique']:,d}, held out {V['n_conv_val']:,d}); tokens train {tot:,d}, held out {V['tok_val']:,d}, files {V['tok_files']:,d}; targets {sup:,d} = {V['sup_pct']:.1f} %")
print(src_table)
print(cat_table)
print(funnel_table)
print(json.dumps(V, indent=1, default=str))
if OUT:
    os.makedirs(OUT, exist_ok=True)
    json.dump(V, open(os.path.join(OUT, "values.json"), "w"), indent=1, default=str)
    for name, t in (("sources", src_table), ("categories", cat_table), ("funnel", funnel_table)):
        open(os.path.join(OUT, name + ".md"), "w", encoding="utf-8").write(t + "\n")
    print("written", OUT)
