#!/usr/bin/env python3
"""human_manifest.py [--build BUILD] [--out DIR]: write MANIFEST.txt (and manifest.json) of the published human-written-only SFT mix next to its shards.
Reads the plan (plan.json, docs.tsv of the build directory), the pool statistics (sft_work_human/pool/*.stats.json), the published .u16 files (token and mask counts are counted from the files themselves)
and, when present, contam_human.json (the result of tools/contamination_check.py on the finished token files, written by human_contam.py)."""
import argparse
import collections
import datetime
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import human_assemble as A  # noqa: E402
import human_cfg as C  # noqa: E402
import human_pools as HP  # noqa: E402
import sft_lib as L  # noqa: E402

from human_meta import EXCLUDED, GROUP_NAMES, META  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--build", default=A.BUILD)
ap.add_argument("--out", default=C.OUT)
a = ap.parse_args()

plan = json.load(open(os.path.join(a.build, "plan.json")))
T = plan["validation_tokens_T"]
n_tail = plan["val_file_tail_train_docs"]
rows = []  # (file, idx, src, stratum, ntok, nsup, turns, has_system, is_val, id)
with open(os.path.join(a.build, "docs.tsv"), encoding="utf-8") as f:
    next(f)
    for line in f:
        p = line.rstrip("\n").split("\t")
        is_val = p[0] == "zzz_val" and int(p[1]) >= n_tail
        rows.append((p[0], int(p[1]), p[2], p[3], int(p[4]), int(p[5]), int(p[6]), int(p[7]), is_val, p[8]))

pool_stats = {}
for s in A.TARGETS_M:
    p = os.path.join(C.POOL, s + ".stats.json")
    if os.path.exists(p):
        pool_stats[s] = json.load(open(p))

files = sorted(f for f in os.listdir(a.out) if f.endswith(".u16"))
fcount = {}
for f in files:
    arr = np.fromfile(os.path.join(a.out, f), dtype="<u2")
    h = hashlib.sha256(arr.tobytes()).hexdigest()[:16]
    fcount[f] = dict(tokens=int(len(arr)), supervised=int(((arr & 0x8000) == 0).sum()), bytes=int(len(arr) * 2), sha256_16=h)

by_src = collections.OrderedDict((s, dict(train_docs=0, train_tokens=0, train_sup=0, val_docs=0, val_tokens=0, val_sup=0)) for s in A.TARGETS_M)
by_group = collections.Counter()
by_group_sup = collections.Counter()
by_stratum = collections.defaultdict(lambda: collections.Counter())
for r in rows:
    d = by_src[r[2]]
    k = "val" if r[8] else "train"
    d[k + "_docs"] += 1
    d[k + "_tokens"] += r[4]
    d[k + "_sup"] += r[5]
    if not r[8]:
        g = HP.group_of(r[2], r[3])
        by_group[g] += r[4]
        by_group_sup[g] += r[5]
        by_stratum[r[2]][r[3]] += r[4]
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
lens = np.array([r[4] for r in tr_rows])
turns = collections.Counter(min(r[6], 4) for r in tr_rows)
n_sys = sum(r[7] for r in tr_rows)
prov = collections.Counter()
lic = collections.Counter()
for s, d in by_src.items():
    kind = META[s][4]
    prov["project-authored (identity)" if kind == "project" else "third-party human-written"] += d["train_tokens"]
    L_ = META[s][1]
    key = "non-commercial (CC-BY-NC / NC-SA)" if "NON-COMMERCIAL" in L_ else ("share-alike (CC-BY-SA, GFDL)" if ("share-alike" in L_ or "SA-" in L_ or "GFDL" in L_) else ("no license stated" if ("none stated" in L_ or "unknown" in L_) else "permissive (MIT, Apache, CC-BY, other)"))
    lic[key] += d["train_tokens"]

lines = []
W = lines.append
W("MANIFEST of tokens_sft_human (Forge-1 chat fine-tune mix, HUMAN-WRITTEN ANSWERS ONLY)")
W("=" * 90)
W(f"written {datetime.datetime.now().strftime('%Y-%m-%d %I:%M %p')} by tools/sft_data/human_manifest.py; description, weaknesses and the comparison with tokens_sft_mix2: research note 'ForgeInfer - SFT Data Mix (human-written only)'")
W(f"RULE: every assistant answer in this mix was written by a human (third-party text from the cited datasets), except the identity set ({100 * by_src['identity']['train_tokens'] / tot_train:.1f} % of the tokens), which the project authored. No answer was generated, rewritten or translated by a language model;")
W("      the transformations are mechanical (HTML to markdown, detokenisation, templated user-side wording, truncation to complete exchanges, the closing 'The answer is: X' line of the math sources). The USER side of the templated")
W("      sources (summaries, code tasks, reading comprehension, stories) has wording written by the project's templates.")
W("")
W("FILES  (uint16 little-endian tokens of the 32,768-entry tokenizer_f1.json; bit 15 = 'not a target': prompt, headers, separators are masked, the assistant text and its <|im_end|> train)")
for f in files:
    c = fcount[f]
    W(f"  {f:14s} {c['tokens']:>12,d} tokens  {c['bytes'] / 1e6:8.1f} MB  supervised {c['supervised']:>12,d} ({100 * c['supervised'] / c['tokens']:.1f} %)  sha256[:16] {c['sha256_16']}")
tot_tokens = sum(c["tokens"] for c in fcount.values())
W(f"  total {tot_tokens:,d} tokens in {len(files)} files; training documents {tot_train:,d} tokens (incl. oversampled copies), held-out validation {tot_val:,d} tokens; conversations {len(rows):,d} ({len(tr_rows):,d} training copies, {len(rows) - len(tr_rows):,d} held out)")
W(f"  targets (tokens that carry a loss): {tot_sup:,d} of {tot_train:,d} training tokens = {100 * tot_sup / tot_train:.1f} % (held-out region {100 * val_sup / max(1, tot_val):.1f} %)")
W(f"  heldout/sft_val_masked.u16  {T:,d} tokens: the last T tokens of zzz_val.u16 = the held-out conversations alone, WITH the 'not a target' bit (for code that understands it: the trainers' Data class)")
W("  heldout/sft_val_plain.u16   the same tokens with bit 15 cleared: for `forge eval-ppl` (it reads the ids raw: NEVER give it a masked file, ids >= 32768 fault on the GPU); it scores every token of the chat text, prompts included")
W("  (both in a sub-folder on purpose: forge train reads every *.u16 of the directory it is given)")
W("  replay set: /mnt/data/forgeinfer-data/tokens_sft_human_replay = a_replay.u16 (12 % pre-training replay, the first bytes of tokens_cool/replay/replay_000.u16) + symlinks to the files above")
W("")
W("VALIDATION LAYOUT (important)")
W("  The trainer takes the LAST .u16 file (alphabetical) and reserves min(--val-tokens, tokens / 4) of its tail as the validation region; the rest of that file is TRAINING data.")
W(f"  zzz_val.u16 = [{plan['val_file_pad_tokens']} masked <|endoftext|> padding tokens][{n_tail:,d} training documents, {plan['val_file_tail_train_tokens']:,d} tokens][{plan['val_docs']:,d} held-out documents, T = {T:,d} tokens],")
W(f"  length 4 x T = {4 * T:,d}, so tokens / 4 = T and the validation region is exactly the held-out set for ANY --val-tokens >= {T} (the code default is 20,000,000, the smoke script uses 4,000,000).")
W("  Do NOT put a replay shard whose name sorts after 'sft_*' into the directory (the validation tail is taken from the last file); no file here sorts after zzz_val.u16.")
W(f"  Validation loss = mean over the assistant tokens of the held-out conversations ({100 * val_sup / max(1, tot_val):.1f} % of the region is supervised); windows start at random token offsets of the region.")
W("")
W("SOURCES  (tokens counted with the tokenizer, ChatML framing and the closing <|endoftext|> included; 'x' = oversampling of the training copy; share = share of the training tokens)")
W(f"  {'source':10s} {'group':8s} {'x':>2s} {'train docs':>10s} {'train tokens':>13s} {'share':>6s} {'sup %':>6s} {'val docs':>8s} {'val tokens':>11s}  license | who wrote the answers | what")
for s, d in by_src.items():
    m = META[s]
    x = A.OVERSAMPLE.get(s, 1)
    sup = 100 * d["train_sup"] / max(1, d["train_tokens"])
    grp = collections.Counter()
    for st, t in by_stratum[s].items():
        grp[HP.group_of(s, st)] += t
    g = "mixed" if len(grp) > 1 and grp.most_common(1)[0][1] < 0.9 * sum(grp.values()) else (grp.most_common(1)[0][0] if grp else "")
    W(f"  {s:10s} {g:8s} {x:>2d} {d['train_docs']:>10,d} {d['train_tokens']:>13,d} {100 * d['train_tokens'] / tot_train:>5.1f}% {sup:>6.1f} {d['val_docs']:>8,d} {d['val_tokens']:>11,d}  {m[1]} | {m[2]} | {m[0]}")
W(f"  {'TOTAL':10s} {'':8s} {'':>2s} {sum(d['train_docs'] for d in by_src.values()):>10,d} {tot_train:>13,d} {'100.0%':>6s} {100 * tot_sup / tot_train:>6.1f} {sum(d['val_docs'] for d in by_src.values()):>8,d} {tot_val:>11,d}")
W("")
W("PROVENANCE  (who wrote the assistant side, with a quoted line from the dataset card; the cards were fetched on 2026-10-10 and saved in sft_work_human/cards/)")
for s in by_src:
    W(f"  {s}: {META[s][2]}")
    W(f"      {META[s][3]}")
W("")
W("LLM FINGERPRINTS (measured 2026-10-10 with scratch/agents/sft-human/llm_isms.py on the pools: share of the conversations whose assistant text hits at least one chatbot pattern of human_assemble.py: 'Certainly/Sure' openers, 'hope this helps', 'let me know if',")
W("  'it is important to note', 'Overall,', 'Here is a ...:' ...; controls = text that cannot be model-written)")
W("  Stack Exchange 4.2 % (before mid-2022), Dolly 2.8 %, SHP 2.4 %, WritingPrompts 2.5 %, CoQA 0.0 %   <- human controls: the false-positive rate of the patterns on human text")
W("  Aya 8.5 %, No Robots 12.7 % (3.9 % of its answers start with 'Certainly/Sure'), OASST2 55.3 % (16.9 % start with 'Certainly/Sure', 27.6 % contain 'Here is a ...:')")
W("  rule: an answer with one STRONG pattern or two WEAK ones is dropped from Dolly, No Robots and Aya (they were collected in 2023): retained 99.2 % / 94.3 % / 98.4 %; the same rule keeps only 60.4 % of OASST2, and what it keeps still contains answers that read like ChatGPT,")
W("  so OASST2 is NOT part of this mix (HUMAN_INCLUDE_OASST2=1 adds it).  On the finished mix the same patterns hit 0.7 % of the conversations (tokens_sft_mix2: 19.6 %; tools/sft_data/compare_mixes.py, 25 % sample of each corpus).")
W("")
W("EXCLUDED SOURCES (looked at, not used; the reason)")
for name, why in EXCLUDED:
    W(f"  {name}: {why}")
W("")
W("BY CATEGORY (share of the training tokens; the category of a Stack Exchange site or subreddit is in tools/sft_data/human_pools.py SE_GROUP / SHP_GROUP): " + ", ".join(f"{GROUP_NAMES[g]} {100 * by_group[g] / tot_train:.1f} % (targets {100 * by_group_sup[g] / max(1, by_group[g]):.0f} %)" for g in GROUP_NAMES if by_group[g]))
W("BY PROVENANCE: " + ", ".join(f"{g} {100 * t / tot_train:.1f} %" for g, t in prov.most_common()) + "  (no token of any other kind: no LLM-generated text)")
W("BY LICENSE CLASS (what the cards say; read the licenses before any release): " + ", ".join(f"{g} {100 * t / tot_train:.1f} %" for g, t in lic.most_common()))
big = sorted(((d["train_tokens"], s) for s, d in by_src.items()), reverse=True)[:5]
W("LARGEST SOURCES: " + ", ".join(f"{s} {100 * t / tot_train:.1f} %" for t, s in big) + "  (cap: no single dataset above 25 %; Stack Exchange counts as one dataset)")
se_tot = sum(by_stratum["se"].values())
W(f"STACK EXCHANGE: {len(by_stratum['se'])} sites, {se_tot:,d} tokens; " + ", ".join(f"{st} {100 * t / se_tot:.1f} %" for st, t in sorted(by_stratum["se"].items(), key=lambda kv: -kv[1])[:30]) + " ...")
shp_tot = sum(by_stratum["shp"].values())
W(f"SHP: {shp_tot:,d} tokens; " + ", ".join(f"{st} {100 * t / shp_tot:.1f} %" for st, t in sorted(by_stratum["shp"].items(), key=lambda kv: -kv[1])))
W("")
W("DOCUMENT LENGTH (tokens per conversation incl. framing, training documents)")
W("  bucket      " + "  ".join(f"{('<' + str(edges[0])) if i == 0 else (str(edges[i - 1]) + '-' + str(edges[i]) if i < len(edges) else '>=' + str(edges[-1])):>9s}" for i in range(len(edges) + 1)))
W("  documents   " + "  ".join(f"{c:>9,d}" for c in len_hist))
W("  tokens      " + "  ".join(f"{c:>9,d}" for c in tok_hist))
W("  quantiles (tokens): " + ", ".join(f"p{int(q * 100)} {int(np.quantile(lens, q))}" for q in (0.05, 0.25, 0.5, 0.75, 0.95, 0.99)))
W(f"  conversations with 1 / 2 / 3 / 4+ assistant turns: {turns[1]:,d} / {turns[2]:,d} / {turns[3]:,d} / {turns[4]:,d};  with a system prompt: {n_sys:,d} ({100 * n_sys / len(tr_rows):.1f} %)")
W("")
W("FILTERS  (sft_lib.py, human_lib.py, human_pools.py, human_assemble.py; per-source drop counts in sft_work_human/pool/<source>.stats.json)")
W("  cleaning: CRLF -> LF, control and zero-width characters removed, trailing blanks, runs of 4+ blank lines -> 2; messages stripped (a leading blank would change the role header's tokens)")
W("  length: conversations cut to their first complete exchanges that fit 600-1300 tokens (per source), minimum 24 tokens (12 for the short-answer QA), assistant answer >= 12 characters (1-4 for QA / Dolly / aya / no_robots); no answer is cut inside a text")
W("  language: >= 90 % ASCII and a stop-word ratio >= 12 % of the prose words (code and mathematics excluded)")
W("  other vendors: any mention of ChatGPT / OpenAI / GPT-3/4 / Anthropic / Claude anywhere; first-person identity statements naming other models; 'as an AI language model'; knowledge cut-off statements; 'as an AI ...' anywhere in an answer")
W("  markup: special-token strings, [INST] / <<SYS>> artifacts, HTML tags and entities (tags allowed in the code sources and Stack Exchange), mojibake, unbalanced code fences, role leaks, escaped newlines, degenerate repetition, answers cut off mid-sentence")
W("  PII and links: e-mail addresses (except example domains), phone numbers, URLs in an answer that appear in no earlier message (not for the code sources); Stack Exchange: links keep their text and lose the URL, an answer with 2+ links or one short answer with a link is dropped")
W("  explicit words: profanity / sexual vocabulary dropped from every source that is user-written prose")
W("  forum rules (Stack Exchange and Reddit answers; regular expressions in human_lib.py): no '@user', 'the OP', 'in the comments', 'my answer', 'other answers', 'this site', 'edit:' markers, 'thanks for the gold'; no life story (my wife, my job, as a physicist I ..., years of experience); first-person density <= 5 % (Reddit 3.5 %); questions that refer to the title, an image, a screenshot, a bounty or a duplicate are dropped")
W(f"  Stack Exchange alignment: the HuggingFaceH4 dump pairs a few percent of the questions with the answers of another question (answer id below the question id: 3.2 % of all answers; a math.SE question about floor sums with answers about dice); conversations whose IDF-weighted word overlap between question and answer is below {plan.get('align_min', 4.0)} are dropped (real pairs: median 22.6; pairs shuffled within a site: 3.1)")
W("  dates: Stack Exchange questions dated before 2022-06-30, Reddit comments written before 2022-09-01 (ChatGPT was released on 2022-11-30); the other datasets were collected before 2023 or by named annotators")
W("  decontamination: a verbatim run of >= 13 words (10-gram index of human_bench_index.py, 3 further consecutive 10-grams; runs of mostly numbers ignored) shared with the test / validation sets of HellaSwag, ARC-Easy/Challenge, PIQA, WinoGrande, MMLU, GSM8K test, MATH test, HumanEval, MBPP TEST (problems 11-510 only: MBPP train / validation are training data here), LAMBADA, SciQ, OpenBookQA, TruthfulQA; IFEval: any shared 16-gram;")
W("                   plus the exact files of the evaluation ladder (human_audit.py pools), the function names of HumanEval and of the MBPP test problems (human_audit.py fnames), and the token-level audit of tools/contamination_check.py on the finished files (below)")
W("  duplicates: exact (same conversation or same user side) and near (MinHash of word 3-grams of the first user message, Jaccard >= 0.8 for short, >= 0.7 for long prompts) ACROSS sources, hand-written and human chat sources kept first")
d_bench = collections.Counter()
for s, st in pool_stats.items():
    for b, n in st.get("bench_hits", {}).items():
        d_bench[b] += n
W("  decontamination drops in the pools (conversations with a >= 13-word verbatim overlap): " + ", ".join(f"{b} {n}" for b, n in d_bench.most_common()) + f"  (total {sum(d_bench.values())} of {sum(st['processed'] for st in pool_stats.values())} candidates examined)")
W("  decontamination drops per source: " + ", ".join(f"{s} {st.get('drops', {}).get('decontam', 0)}" for s, st in pool_stats.items() if st.get("drops", {}).get("decontam", 0)))
W("  pool drops by reason (all sources): " + ", ".join(f"{r} {n}" for r, n in sorted(sum((collections.Counter(st.get('drops', {})) for st in pool_stats.values()), collections.Counter()).items(), key=lambda kv: -kv[1])[:24]))
W("  assembly: dedup drops " + json.dumps(plan["dedup_drops"]) + "; late filter drops " + json.dumps(plan.get("late_filter_drops", {})))
cj = os.path.join(C.WORK, "contam_human.json")
if os.path.exists(cj):
    cd = json.load(open(cj))
    W("  tools/contamination_check.py on the finished token files (index of the ladder's files, contam/run1; per round: scan of the published files, then every conversation with a verified 16-token window or a whole sentence-like short text of 8-15 tokens is removed (all")
    W("  occurrences of the window, not only the first position the tool reports) and the files are rebuilt; the removed conversations stay in the pool as 'dead' records during the deduplication so that their near duplicates do not take their place; pure numbers like '800,000' are no evidence):")
    for i, r in enumerate(cd["rounds"], 1):
        W(f"    round {i}: verified 16-token hit windows {r['hit_windows_16_tokens']} ({r['occurrences_16_tokens']} occurrences), short whole-text hits {r['hit_texts_short']} ({r['short_numeric_hits_ignored']} numeric ignored), 10-token windows {r['hit_windows_10_tokens (weak, not removed)']} (weak, kept);"
          f" conversations removed {r['conversations_flagged_this_round']} " + json.dumps(r["by_source_this_round"]))
    sjp = os.path.join(C.WORK, "contam_human_spans.json")
    if os.path.exists(sjp):
        sh_ = json.load(open(sjp))
        live_ = [h for h in sh_ if not h.get("dry")]
        W(f"  documents with a STRONG hit at 10 tokens (report/exclusion_spans_n10.json; a text of which half of the 10-token windows occur, mostly number-heavy word problems and passages) removed per round: {', '.join(str(h['new_ids']) for h in live_)} (total {sum(h['new_ids'] for h in live_)});")
        W(f"  the last scan of the final files still lists {sh_[-1]['strong_10_token_documents']} such documents ({json.dumps(sh_[-1]['by_source'])}): the criterion keeps firing on generic number text, so the loop was stopped (they are NOT removed)")
    W(f"  total blacklisted by the audit: {cd['final']['blacklist_total']} conversations; final scan of the published files: {cd['final']['hit_windows_16_tokens']} verified 16-token hit windows and {cd['final']['conversations_flagged_this_round']} sentence-like short hits (report: contam_human/report/contamination_report.md)")
W("")
W("SEEDS AND DETERMINISM")
W(f"  base seed {L.SEED}; every random choice uses random.Random(stable_seed(<purpose>, <source>)) = (SEED + crc32(parts)) & 0x7fffffff: pools ('human-pool', source), assembly ('human-assemble-shuffle', source), ('human-assemble-val'), ('human-assemble-train').")
W(f"  validation share: every source gives its token share of {plan['val_target']:,d} validation tokens (identity: 8 % of its conversations); oversampled sources are held out BEFORE oversampling.")
W("")
W("COMMANDS (homelab, cwd /mnt/data/forgeinfer-data/sft_work_human, every job under nice -n 15 ionice -c3 with OMP_NUM_THREADS=1, started through scripts/run_job_human.sh and watched by scripts/guard_human.sh; the scripts are tools/sft_data/*.py of the repository)")
W("  python fetch_human.py small shp se                                    # raw files -> sft_work_human/raw (HF hub, public files only, 4.2 GB of Stack Exchange parquet + the others)")
W("  python human_bench_index.py                                           # bench_index_human.npz: the benchmark 10-gram index with MBPP test problems only")
W("  python human_pools.py oasst2 dolly no_robots aya gsm8k hendrycks aqua apps mbpp rosetta squad coqa nq trivia cnn dialogsum wp csn shp   # pool/<source>.jsonl + stats")
W("  python human_pools.py se                                              # the Stack Exchange pool (per site quotas)")
W("  python human_audit.py pools && python human_audit.py fnames --max-df 12   # blacklist_eval.txt, blacklist_fnames.txt")
W("  python human_assemble.py plan                                         # late filters, alignment check, dedup, thinning, validation split, oversampling, shuffle -> build/*.txt + docs.tsv + plan.json")
W("  python human_assemble.py encode                                       # /tmp/forge_sft tok-encode TOK X.txt --out X.enc.u16 --threads 1; /tmp/forge_sft chat-mask TOK X.enc.u16 --out X.u16 (counts checked against the plan)")
W("  python human_audit.py build BUILD && python sft_verify.py BUILD --build BUILD --n 0   # decontamination audit of the corpus text; round trip, masks, layout of every document")
W("  python human_assemble.py publish && python human_assemble.py replay   # tokens_sft_human + heldout/ and tokens_sft_human_replay")
W("  bash contam_loop_human.sh 6                                           # plan, encode, publish, replay, `contamination_check.py scan` (index of contam/run1, --mask15 human), human_contam.py collect -> blacklist_contam.txt; until a scan finds nothing")
W("  python human_contam.py report                                         # contamination_check.py report on the final files")
W("  python sft_verify.py /mnt/data/forgeinfer-data/tokens_sft_human --build BUILD --n 0 && python human_check_published.py && python sft_loader_sim.py DIR --val-tokens 4000000   # checks of the published files")
W("  python human_samples.py 60 && python human_manifest.py                # SAMPLES.txt and this file")
W("  /tmp/forge_sft = copy of ~/forgeinfer/target-dev3/release/forge (the dev build; the v1 binary does not understand bit 15)")
W("")
W("PLAN SUMMARY (plan.json): " + json.dumps({k: v for k, v in plan.items() if k not in ("dedup_drops", "late_filter_drops", "targets_m")}))
W("TARGETS (million tokens per source before oversampling; 99 = everything the pool holds): " + json.dumps(dict(plan.get("targets_m", {}))))
text = "\n".join(lines) + "\n"
os.makedirs(a.out, exist_ok=True)
open(os.path.join(a.out, "MANIFEST.txt"), "w", encoding="utf-8").write(text)
json.dump({"files": fcount, "sources": by_src, "plan": plan, "len_hist": len_hist, "len_edges": edges, "groups": dict(by_group), "stratum_tokens": {s: dict(v) for s, v in by_stratum.items() if s in ("se", "shp")}},
          open(os.path.join(a.out, "manifest.json"), "w"), indent=1)
print(text)
