#!/usr/bin/env python3
"""human_contam.py prepare | scan | collect | report: run tools/contamination_check.py (the token-level n-gram audit of the benchmark ladder) on the FINISHED token files of tokens_sft_human and turn its hits
into a blacklist of conversations (sft_work_human/blacklist_contam.txt, read by human_assemble.py plan).

  prepare   work directory sft_work_human/contam_human with a symlink index/ -> /mnt/data/forgeinfer-data/contam/run1/index (the index of the ladder's files, built by the contam helper; read only)
  scan      contamination_check.py scan --corpus human=<OUT>/*.u16 --mask15 human  (one process, niced, --max-mb-s 30): per file the verified 16-token and 10-token windows that occur in the ladder's texts
  collect   map the first position of every hit window (16 tokens, and the whole short texts) of every file to its conversation with the document separators and docs.tsv -> blacklist_contam.txt and
            contam_human.json (counts); windows that occur several times in a file only report the first position, so the loop plan -> encode -> publish -> scan -> collect is repeated until no hit is left
  report    contamination_check.py report: the per-benchmark tables for the research note (hit / strong / present per benchmark)
"""
import collections
import glob
import json
import os
import re
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import human_cfg as C  # noqa: E402

TOOL = os.path.expanduser("~/forgeinfer/src/tools/contamination_check.py")
HOME = os.path.join(C.WORK, "contam_human")
RUN1 = "/mnt/data/forgeinfer-data/contam/run1"
PY = os.path.expanduser("~/forgeinfer/venv/bin/python")


def sh(cmd):
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def prepare():
    os.makedirs(HOME, exist_ok=True)
    link = os.path.join(HOME, "index")
    if not os.path.lexists(link):
        os.symlink(os.path.join(RUN1, "index"), link)
    print("ready:", HOME, "->", os.readlink(link))


def scan():
    prepare()
    sh([PY, TOOL, "scan", "--out", HOME, "--corpus", f"human={C.OUT}/*.u16", "--mask15", "human", "--max-mb-s", "30"])


def occurrences(arr, w):
    """All start positions of the token sequence w in arr (numpy: filter by the first token, then verify the rest)."""
    cand = np.flatnonzero(arr[: len(arr) - len(w) + 1] == w[0])
    for k in range(1, len(w)):
        if len(cand) == 0:
            break
        cand = cand[arr[cand + k] == w[k]]
    return cand


def collect():
    plan = json.load(open(os.path.join(C.BUILD, "plan.json")))
    n_pad = plan["val_file_pad_tokens"]
    ids = collections.defaultdict(list)  # (file, doc index) -> [id, src]
    with open(os.path.join(C.BUILD, "docs.tsv"), encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\n").split("\t")
            ids[(p[0], int(p[1]))] = [p[8], p[2]]
    out = set()
    stat = collections.Counter()
    by_src = collections.Counter()
    zi = np.load(os.path.join(HOME, "index", "index.npz"))
    IDX = {k: zi[k] for k in ("v_start", "v_len", "tokens")}
    from tokenizers import Tokenizer

    TOK = Tokenizer.from_file(os.path.expanduser("~/forgeinfer/data/tokenizer_f1.json"))
    for jp in sorted(glob.glob(os.path.join(HOME, "scan", "human__*.json"))):
        base = jp[:-5]
        name = os.path.basename(base)[len("human__"):-len(".u16")]
        st = json.load(open(jp))
        with np.load(base + ".eos.npz") as ez:
            eos = ez["eos"]
        toks = np.fromfile(os.path.join(C.OUT, name + ".u16"), dtype="<u2") & 0x7FFF  # the scan keeps only the FIRST position of a hit window: find every occurrence of it in the file
        with np.load(base + ".npz") as z:
            for key in ("n16", "short", "n10"):
                if f"{key}_p" not in z.files:
                    continue
                pos = z[f"{key}_p"]
                stat[f"{key}_hit_windows"] += int(len(pos))
                if key == "n10":
                    continue
                if key == "short":  # whole texts of 8-15 tokens: only the rare ones (at most 10 occurrences in the file) count; a common short string is no evidence
                    cnt = z["short_c"]
                    stat["short_rare_hit_texts"] += int((cnt <= 10).sum())
                    pos = pos[cnt <= 10]
                allpos = []
                if key == "n16":
                    for p in pos.tolist():
                        allpos.extend(occurrences(toks, toks[p : p + 16]).tolist())
                else:  # short texts: short_g is the id of the matched text variant (index into v_start / v_len of the index); the whole variant is searched in the whole file
                    grp = z["short_g"][z["short_c"] <= 10]
                    for g, p in zip(grp.tolist(), pos.tolist()):
                        s0, ln = int(IDX["v_start"][g]), int(IDX["v_len"][g])
                        w = IDX["tokens"][s0 : s0 + ln]
                        if len(re.findall(r"[A-Za-z]{2,}", TOK.decode(w.tolist()))) < 4:  # '800,000', '1,2,3,4', '10,000 years': numbers and short answer options are no evidence (every math solution has them); a sentence or a question is
                            stat["short_numeric_ignored"] += 1
                            continue
                        if p + ln <= len(toks) and np.array_equal(toks[p : p + ln], w):
                            allpos.extend(occurrences(toks, w).tolist())
                        else:
                            allpos.append(p)
                stat[f"{key}_occurrences"] += len(allpos)
                for p in allpos:
                    k = int(np.searchsorted(eos, p, side="left"))
                    j = k - n_pad if name == "zzz_val" else k
                    if j < 0:
                        continue
                    rec = ids.get((name, j))
                    if rec is None:
                        continue
                    if rec[0] not in out:
                        by_src[rec[1]] += 1
                    out.add(rec[0])
        stat["files"] += 1
    path = os.path.join(C.WORK, "blacklist_contam.txt")
    prev = set(x.strip() for x in open(path)) if os.path.exists(path) else set()
    allids = sorted(prev | out)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(allids) + ("\n" if allids else ""))
    res = {"files_scanned": stat["files"], "hit_windows_16_tokens": stat["n16_hit_windows"], "hit_texts_short": stat["short_hit_windows"], "hit_texts_short_rare": stat["short_rare_hit_texts"],
           "hit_windows_10_tokens (weak, not removed)": stat["n10_hit_windows"], "occurrences_16_tokens": stat["n16_occurrences"], "occurrences_short_texts": stat["short_occurrences"], "short_numeric_hits_ignored": stat["short_numeric_ignored"],
           "conversations_flagged_this_round": len(out), "by_source_this_round": dict(by_src), "blacklist_total": len(allids)}
    jpath = os.path.join(C.WORK, "contam_human.json")
    rounds = json.load(open(jpath)).get("rounds", []) if os.path.exists(jpath) else []
    rounds.append(res)  # every round = one scan of the files as published at that time; the last one is the state of the final files
    json.dump({"rounds": rounds, "final": res}, open(jpath, "w"), indent=1)
    print(json.dumps(res, indent=1))


def collect_spans():
    """Add to the blacklist the conversations that hold a STRONG hit at 10 tokens (a text of which at least half of the 10-token windows occur: a question of a benchmark that sits in the mix with changed numbers or
    a changed ending): the documents of report/exclusion_spans_n10.json (and exclusion_spans.json), located through the document separators of the scan."""
    plan = json.load(open(os.path.join(C.BUILD, "plan.json")))
    n_pad = plan["val_file_pad_tokens"]
    ids = {}
    with open(os.path.join(C.BUILD, "docs.tsv"), encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\n").split("\t")
            ids[(p[0], int(p[1]))] = [p[8], p[2]]
    out, by_src = set(), collections.Counter()
    for fn in ("exclusion_spans.json", "exclusion_spans_n10.json"):
        path = os.path.join(HOME, "report", fn)
        if not os.path.exists(path):
            continue
        for d in json.load(open(path)):
            name = os.path.basename(d["file"])[:-4]
            with np.load(os.path.join(HOME, "scan", f"human__{name}.u16.eos.npz")) as ez:
                eos = ez["eos"]
            k = int(np.searchsorted(eos, int(d["start"]), side="left"))
            j = k - n_pad if name == "zzz_val" else k
            rec = ids.get((name, j))
            if rec is not None:
                out.add(rec[0])
                by_src[rec[1]] += 1
    bl = os.path.join(C.WORK, "blacklist_contam.txt")
    prev = set(x.strip() for x in open(bl)) if os.path.exists(bl) else set()
    new = out - prev
    dry = "--dry" in sys.argv  # only count (the last scan of the final files): the blacklist stays as it is
    if not dry:
        with open(bl, "w", encoding="utf-8") as f:
            f.write("\n".join(sorted(prev | out)) + "\n")
    res = {"strong_10_token_documents": len(out), "new_ids": len(new), "by_source": dict(by_src), "blacklist_total": len(prev | out), "dry": dry}
    jpath = os.path.join(C.WORK, "contam_human_spans.json")
    hist = json.load(open(jpath)) if os.path.exists(jpath) else []
    hist.append(res)
    json.dump(hist, open(jpath, "w"), indent=1)
    print(json.dumps(res, indent=1))


def report():
    rep = os.path.join(HOME, "report")
    sh([PY, TOOL, "report", "--out", HOME, "--eval-dir", "/mnt/data/forgeinfer-data/eval", "--tokenizer", os.path.expanduser("~/forgeinfer/data/tokenizer_f1.json"), "--report-dir", rep])


if __name__ == "__main__":
    {"prepare": prepare, "scan": scan, "collect": collect, "collect_spans": collect_spans, "report": report}[sys.argv[1]]()
