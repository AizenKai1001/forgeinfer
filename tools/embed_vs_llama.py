#!/usr/bin/env python3
"""Compare the embeddings of a running llama.cpp server (`llama-server -m nomic-embed-text-v1.5.*.gguf --embedding --pooling mean -c 2048 --port N`) with the golden vectors of forge-embed
(token ids from Hugging Face's tokenizer, vectors from the numpy reference tools/ref_nomic_bert.py): cosine per text, token counts, and how far the similarity scores between the texts move.

    python tools/embed_vs_llama.py http://127.0.0.1:18970 [crates/forge-embed/tests/golden/nomic_v15_q4km.json]

Read the result with the file's format in mind: on a K-quant file llama.cpp's CPU backend rounds every activation vector to 8 bits for the integer dot products, which moves the vectors by about
0.5 percent in cosine (0.9935 median on nomic-embed-text-v1.5 Q4_K_M); on the same weights converted to F32 (`llama-quantize --allow-requantize in.gguf out.gguf F32`) it agrees with the reference
to five digits, except where its WordPiece tokenizer splits a text differently from Hugging Face's (Hangul and kana).
"""
import json
import sys
import urllib.request

import numpy as np

base = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "http://127.0.0.1:18970"
golden = sys.argv[2] if len(sys.argv) > 2 else "crates/forge-embed/tests/golden/nomic_v15_q4km.json"
cases = json.load(open(golden, encoding="utf-8"))["cases"]


def llama(text):
    body = json.dumps({"input": text, "model": "x"}).encode()
    req = urllib.request.Request(base + "/v1/embeddings", data=body, headers={"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=120))
    v = np.array(r["data"][0]["embedding"], dtype=np.float32)
    return v / np.linalg.norm(v), r["usage"]["prompt_tokens"]


mine, theirs, cs, differ = [], [], [], 0
for c in cases:
    if "vector" not in c:
        continue
    v = np.array(c["vector"], dtype=np.float32)
    t, n = llama(c["text"])
    mine.append(v / np.linalg.norm(v))
    theirs.append(t)
    cs.append(float(mine[-1] @ t))
    if n != len(c["ids"]):
        differ += 1
        print("token count differs:", n, "against", len(c["ids"]), ascii(c["text"][:50]))
cs = np.array(cs)
print(f"{len(cs)} texts: cosine to llama.cpp min {cs.min():.5f} median {np.median(cs):.5f} max {cs.max():.5f}; token counts differ on {differ}")
M, T = np.array(mine), np.array(theirs)
sm, st = M @ M.T, T @ T.T
iu = np.triu_indices(len(M), 1)
d = np.abs(sm[iu] - st[iu])
print(f"similarity between texts, {len(d)} pairs: mean |difference| {d.mean():.5f}, max {d.max():.5f}")
same = 0
for i in range(len(M)):
    a, b = sm[i].copy(), st[i].copy()
    a[i] = b[i] = -2
    same += int(a.argmax() == b.argmax())
print(f"nearest neighbour identical for {same} of {len(M)} texts")
for i in np.argsort(cs)[:3]:
    print(f"  lowest: {cs[i]:.5f} {ascii(cases[i]['text'][:70])}")
