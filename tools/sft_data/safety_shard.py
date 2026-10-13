#!/usr/bin/env python3
"""safety_shard.py --in A.jsonl [--in B.jsonl ...] --out corpus.txt [--repeat N] [--seed S]

Renders conversation records (the pool format: id, msgs [[role, content], ...], ntok, nsup) as ChatML documents ending in NUL, the input of `forge tok-encode`, then `forge chat-mask`
(see ForgeInfer - SFT Data Mix). The set is written N times (default 1), each repeat a full pass in its own seeded random order, so no document is seen more often than another.
Prints the document count and the token estimate (the sum of the records' ntok x repeats). Standard library only.
"""
import argparse
import json
import random


def render(msgs):
    return "".join(f"<|im_start|>{r}\n{c}<|im_end|>\n" for r, c in msgs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inputs", action="append", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    recs = []
    for path in a.inputs:
        with open(path, encoding="utf-8") as f:
            recs += [json.loads(line) for line in f if line.strip()]
    rng = random.Random(a.seed)
    docs = []
    for _ in range(a.repeat):
        order = list(range(len(recs)))
        rng.shuffle(order)
        docs += [render(recs[i]["msgs"]) for i in order]
    with open(a.out, "wb") as f:
        for d in docs:
            f.write(d.encode("utf-8") + b"\x00")
    toks = sum(r.get("ntok", 0) for r in recs) * a.repeat
    sup = sum(r.get("nsup", 0) for r in recs) * a.repeat
    print(f"{len(recs)} records x {a.repeat} = {len(docs)} documents, about {toks} tokens ({sup} supervised) -> {a.out}")


if __name__ == "__main__":
    main()
