#!/usr/bin/env python3
"""add_system.py --in X.jsonl --out Y.jsonl [--p 0.7] [--seed 1] [--exact 0.8]

Gives a share P of the conversations that have no system message one: with probability EXACT the text of the chat page ("You are a helpful assistant."), otherwise one of five other generic system prompts.
The chat page of the homelab sends the system prompt with every request, and a chat fine-tune refuses harmful requests far more reliably with it (measured: patch 3 refuses 57 percent of the held-out harmful
requests without it, 82 percent with it), so the training data should show the behaviour under the condition it is used in. Conversations that already have a system message are left alone.
"""
import argparse
import json
import random

PAGE = "You are a helpful assistant."
OTHERS = ["You are a helpful, honest and concise assistant.", "You are a friendly assistant. Keep your answers short.",
          "You are Forge-1, a small language model built from scratch. Answer clearly and be honest about what you don't know.", "You are a helpful AI assistant.", "Answer as briefly as you can."]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--p", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--exact", type=float, default=0.8)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    n = k = 0
    with open(a.src, encoding="utf-8") as f, open(a.out, "w", encoding="utf-8", newline="\n") as g:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            n += 1
            if r["msgs"][0][0] != "system" and rng.random() < a.p:
                r["msgs"] = [["system", PAGE if rng.random() < a.exact else rng.choice(OTHERS)]] + r["msgs"]
                k += 1
            g.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{n} conversations, {k} got a system prompt -> {a.out}")


if __name__ == "__main__":
    main()
