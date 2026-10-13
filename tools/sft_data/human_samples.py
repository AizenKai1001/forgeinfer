#!/usr/bin/env python3
"""human_samples.py [N]: write SAMPLES.txt next to the shards of tokens_sft_human: N random training conversations (default 60) of the final build, a few of every source, as the model sees them
(the ChatML text; the label line says the source, the stratum and the id).  The trainer reads only *.u16, so the file does not disturb it."""
import collections
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import human_cfg as C  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 60
meta = [l.rstrip("\n").split("\t") for l in open(os.path.join(C.BUILD, "docs.tsv"), encoding="utf-8")][1:]
by = collections.defaultdict(list)
for i, m in enumerate(meta):
    if m[0].startswith("sft_"):
        by[m[2]].append((m[0], int(m[1]), m))
rng = random.Random(7)
picks = []
srcs = sorted(by)
per = max(1, N // len(srcs))
for s in srcs:
    picks += rng.sample(by[s], min(per, len(by[s])))
rng.shuffle(picks)
cache = {}
out = []
for f, j, m in picks[:N]:
    if f not in cache:
        cache[f] = open(os.path.join(C.BUILD, f + ".txt"), "rb").read().split(b"\x00")
    out.append(f"##### source {m[2]} | stratum {m[3]} | {m[4]} tokens | id {m[8]}\n" + cache[f][j].decode("utf-8") + "\n")
open(os.path.join(C.OUT, "SAMPLES.txt"), "w", encoding="utf-8").write("\n".join(out))
print("wrote", os.path.join(C.OUT, "SAMPLES.txt"), len(out), "conversations")
