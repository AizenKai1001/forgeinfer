#!/usr/bin/env python3
"""tok_pieces.py TOKENIZER.json "text" ["text" ...]: the token ids that `forge tokenize` produces (our own byte-level BPE implementation) and the piece of text each id stands for.
   Prints one JSON object per text: {"text", "ids", "pieces"}. FORGE=path/to/forge overrides the binary (default ./target/release/forge)."""
import json
import os
import subprocess
import sys


def bytes_to_unicode():
    # GPT-2's table: every byte has a printable stand-in character (a space is "Ġ", a newline "Ċ"); a byte-level BPE vocabulary is written in those characters
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


tok = json.load(open(sys.argv[1], encoding="utf-8"))
vocab = tok["model"]["vocab"]
inv = {i: t for t, i in vocab.items()}
for a in tok.get("added_tokens", []):
    inv[a["id"]] = a["content"]
u2b = {v: k for k, v in bytes_to_unicode().items()}


def piece(i):
    t = inv[i]
    try:
        return bytes(u2b[c] for c in t).decode("utf-8", "replace")
    except KeyError:
        return t  # a special token such as <|im_start|>


forge = os.environ.get("FORGE", "./target/release/forge")
for text in sys.argv[2:]:
    out = subprocess.run([forge, "tokenize", sys.argv[1], text], capture_output=True, text=True).stdout.strip()
    ids = json.loads(out)
    print(json.dumps({"text": text, "ids": ids, "pieces": [piece(i) for i in ids], "vocab": len(inv)}, ensure_ascii=False))
