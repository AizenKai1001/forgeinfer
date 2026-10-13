#!/usr/bin/env python3
"""compare_mixes.py BUILD_A BUILD_B [--frac F]: style statistics of the ASSISTANT side of two SFT mixes, from their corpus text files (BUILD/*.txt, documents separated by NUL), on a sample of the documents.
Used for the comparison of tokens_sft_mix2 (LLM-generated) and tokens_sft_human (human-written) in the research note: answer length, markdown structure, chatbot boilerplate, first-person experience, lexical diversity."""
import collections
import glob
import os
import random
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import human_assemble as A  # noqa: E402  (the LLM-fingerprint patterns)
import human_lib as H  # noqa: E402

FRAC = 0.25
argv = list(sys.argv[1:])
if "--frac" in argv:
    k = argv.index("--frac")
    FRAC = float(argv[k + 1])
    del argv[k : k + 2]
args = argv
MSG = re.compile(r"<\|im_start\|>(system|user|assistant)\n(.*?)<\|im_end\|>\n", re.S)
LIST = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+\S", re.M)
BOLD = re.compile(r"\*\*[^*\n]+\*\*")
HEAD = re.compile(r"^#{1,4}\s+\S", re.M)
FENCE = re.compile(r"```")
OPENER = re.compile(r"^\s*(?:certainly|sure|absolutely|of course|great|thank you|thanks|i'm sorry|sorry|as an ai|hello|hi)\b", re.I)
CLOSER = re.compile(r"(?:hope (?:this|that|it) helps|let me know|feel free|happy to help|glad to help)", re.I)
WORD = re.compile(r"[A-Za-z']+")


def stats(build):
    rng = random.Random(1)
    n_docs = n_sys = n_multi = 0
    ans_words, usr_words = [], []
    c = collections.Counter()
    tri = collections.Counter()
    nwords = 0
    llm = 0
    narr = 0
    for f in sorted(glob.glob(os.path.join(build, "sft_*.txt"))):
        data = open(f, "rb").read().split(b"\x00")
        for raw in data:
            if not raw or rng.random() > FRAC:
                continue
            msgs = MSG.findall(raw.decode("utf-8"))
            n_docs += 1
            n_sys += msgs[0][0] == "system"
            asst = [t for r, t in msgs if r == "assistant"]
            n_multi += len(asst) > 1
            for r, t in msgs:
                if r == "user":
                    usr_words.append(len(t.split()))
            text_all = "\n".join(asst)
            if any(rx.search(text_all) for rx in A._LLM_STRONG) or sum(1 for rx in A._LLM_WEAK if rx.search(text_all)) >= 2:
                llm += 1
            if H._NARRATIVE.search(H.prose(text_all)):
                narr += 1
            for t in asst:
                w = t.split()
                ans_words.append(len(w))
                c["answers"] += 1
                c["list"] += bool(LIST.search(t))
                c["bold"] += bool(BOLD.search(t))
                c["head"] += bool(HEAD.search(t))
                c["fence"] += bool(FENCE.search(t))
                c["opener"] += bool(OPENER.match(t))
                c["closer"] += bool(CLOSER.search(t[-300:]))
                c["question_end"] += t.rstrip().endswith("?")
                if nwords < 3_000_000:
                    ws = [x.lower() for x in WORD.findall(t)]
                    nwords += len(ws)
                    for i in range(len(ws) - 2):
                        tri[(ws[i], ws[i + 1], ws[i + 2])] += 1
    a = np.array(ans_words)
    tr = sum(tri.values())
    return {
        "docs sampled": n_docs, "assistant answers": c["answers"], "with a system prompt %": 100 * n_sys / n_docs, "2+ assistant turns %": 100 * n_multi / n_docs,
        "answer words mean": float(a.mean()), "answer words median": float(np.median(a)), "answer words p90": float(np.quantile(a, 0.9)), "user words mean": float(np.mean(usr_words)),
        "answers with a list %": 100 * c["list"] / c["answers"], "with **bold** %": 100 * c["bold"] / c["answers"], "with a # heading %": 100 * c["head"] / c["answers"], "with a code fence %": 100 * c["fence"] / c["answers"],
        "answers that start with Certainly/Sure/Thanks/Sorry/Hello ... %": 100 * c["opener"] / c["answers"], "answers that end with 'hope this helps / let me know / feel free' %": 100 * c["closer"] / c["answers"],
        "answers that end with a question %": 100 * c["question_end"] / c["answers"],
        "conversations with a chatbot fingerprint (strong / 2 weak patterns) %": 100 * llm / n_docs, "conversations with a first-person life-story pattern %": 100 * narr / n_docs,
        "distinct word 3-grams per 100 (first 3 M words)": 100 * len(tri) / max(1, tr),
    }


res = {}
for b in args:
    res[b] = stats(b)
keys = list(next(iter(res.values())).keys())
print(f"{'metric':78s}" + "".join(f"{os.path.basename(b.rstrip('/')):>22s}" for b in args))
for k in keys:
    print(f"{k:78s}" + "".join(f"{res[b][k]:22.1f}" for b in args))
