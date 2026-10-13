#!/usr/bin/env python3
"""sft_verify.py DIR [--build BUILD] [--n N]: independent checks of the masked shards of the SFT mix (see ForgeInfer - SFT Data Mix).

For every shard of DIR (*.u16): the tokens are read as uint16 (bit 15 = not a target) and cut into documents at <|endoftext|>; then
  1. structure   every document starts with a masked <|im_start|>, ends with a masked <|endoftext|>, the unmasked tokens lie in assistant spans only and every such span ends with <|im_end|>
  2. text        the ids without the mark decode to exactly the ChatML text of the corpus file BUILD/<shard>.txt (documents compared one by one, all of them or N sampled ones)
  3. targets     decoding only the unmasked tokens of a document gives  <assistant text><|im_end|>  for every assistant turn, in order (so the mask hides exactly the prompt)
  4. encoder     the Hugging Face tokenizers library, applied to the corpus text, gives the same ids as `forge tok-encode` (the shard)
  5. layout      the last shard (zzz_val) has 4 x T tokens and its tail of T tokens starts at a document start: what the trainer reserves for validation (min(--val-tokens, N/4)) is exactly the held-out set
Prints statistics and a few documents with the supervised spans in [[ ... ]].
"""
import argparse
import collections
import glob
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sft_lib as L  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("dir")
ap.add_argument("--build", default=os.environ.get("SFT_BUILD", "/tmp/sft_mix2_build"))
ap.add_argument("--n", type=int, default=0, help="documents per shard to compare in detail (0 = all)")
ap.add_argument("--show", type=int, default=4)
a = ap.parse_args()

tok = L.tokenizer()
EOS, IM_START, IM_END = tok.token_to_id("<|endoftext|>"), tok.token_to_id("<|im_start|>"), tok.token_to_id("<|im_end|>")
ASSISTANT_HDR = tok.encode("assistant\n").ids
print("ids: eos", EOS, "im_start", IM_START, "im_end", IM_END, "assistant header", ASSISTANT_HDR)
rng = random.Random(7)
errors = collections.Counter()
examples = []


def docs_of(arr):
    """(start, end) of every document: end is exclusive and includes the closing <|endoftext|>."""
    ids = arr & 0x7FFF
    ends = np.flatnonzero(ids == EOS) + 1
    starts = np.concatenate(([0], ends[:-1]))
    return list(zip(starts.tolist(), ends.tolist())), len(arr) == (int(ends[-1]) if len(ends) else 0)


def supervised_spans(ids, mask):
    spans, i, n = [], 0, len(ids)
    while i < n:
        if not mask[i]:
            j = i
            while j < n and not mask[j]:
                j += 1
            spans.append((i, j))
            i = j
        else:
            i += 1
    return spans


files = sorted(glob.glob(os.path.join(a.dir, "*.u16")))
print("shards:", [os.path.basename(f) for f in files])
tot = collections.Counter()
lens = []
for f in files:
    name = os.path.basename(f)[:-4]
    arr = np.fromfile(f, dtype="<u2")
    docs, closed = docs_of(arr)
    n_pad = sum(1 for s0, e0 in docs if e0 - s0 == 1)  # zzz_val starts with masked <|endoftext|> padding tokens (one-token 'documents')
    docs = [(s0, e0) for s0, e0 in docs if e0 - s0 > 1]
    if n_pad:
        print(f"  {n_pad} padding tokens at the start of {os.path.basename(f)} (masked <|endoftext|>)")
    mask = (arr & 0x8000) != 0
    ids = (arr & 0x7FFF).astype(np.int64)
    print(f"[{name}] {len(arr)} tokens, {len(docs)} documents, {int((~mask).sum())} supervised ({100 * (~mask).sum() / len(arr):.1f} %), starts/ends at a document boundary: {closed}")
    if not closed:
        errors["shard does not end with eos"] += 1
    text_docs = None
    tpath = os.path.join(a.build, name + ".txt")
    if os.path.exists(tpath):
        text_docs = [d for d in open(tpath, "rb").read().split(b"\x00") if d]
        if len(text_docs) != len(docs):
            errors["document count differs from the corpus file"] += 1
            print(f"  ERROR: {len(docs)} documents in the shard, {len(text_docs)} in {tpath}")
    sample = list(range(len(docs)))
    if a.n and a.n < len(docs):
        sample = sorted(rng.sample(sample, a.n))
    n_cmp = 0
    for k in sample:
        s, e = docs[k]
        d_ids, d_mask = ids[s:e], mask[s:e]
        lens.append(e - s)
        # 1. structure
        if d_ids[0] != IM_START or not d_mask[0]:
            errors["document does not start with a masked <|im_start|>"] += 1
        if d_ids[-1] != EOS or not d_mask[-1]:
            errors["document does not end with a masked <|endoftext|>"] += 1
        for (i, j) in supervised_spans(d_ids, d_mask):
            if d_ids[j - 1] != IM_END:
                errors["supervised span does not end with <|im_end|>"] += 1
            hdr = [IM_START] + ASSISTANT_HDR
            if i < len(hdr) or list(d_ids[i - len(hdr) : i]) != hdr:
                errors["supervised span does not follow an assistant header"] += 1
        if text_docs is None:
            continue
        txt = text_docs[k].decode("utf-8")
        # 2. text round trip
        dec = tok.decode(d_ids[:-1].tolist(), skip_special_tokens=False)
        if dec != txt:
            errors["decoded text differs from the corpus text"] += 1
            if errors["decoded text differs from the corpus text"] <= 2:
                print("  MISMATCH text", k, repr(dec[:120]), repr(txt[:120]))
        # 3. targets: assistant texts + <|im_end|>
        want = []
        for part in txt.split("<|im_start|>")[1:]:
            role, _, rest = part.partition("\n")
            if role == "assistant":
                want.append(rest[: rest.rindex("<|im_end|>") + len("<|im_end|>")])
        got = [tok.decode(d_ids[i:j].tolist(), skip_special_tokens=False) for i, j in supervised_spans(d_ids, d_mask)]
        if got != want:
            errors["supervised text differs from the assistant turns"] += 1
            if errors["supervised text differs from the assistant turns"] <= 2:
                print("  MISMATCH targets", k, [g[:60] for g in got][:2], [w[:60] for w in want][:2])
        # 4. encoder parity
        enc = tok.encode(txt).ids
        if enc != d_ids[:-1].tolist():
            errors["tokenizers library and forge tok-encode disagree"] += 1
        n_cmp += 1
        if len(examples) < a.show and rng.random() < 0.02:
            examples.append((name, k, d_ids.tolist(), d_mask.tolist()))
    tot["docs"] += len(docs)
    tot["tokens"] += len(arr)
    tot["supervised"] += int((~mask).sum())
    print(f"  compared {n_cmp} documents in detail")
    if name == "zzz_val":
        N = len(arr)
        v = N // 4
        start = N - v
        print(f"[layout] zzz_val: N = {N}, N / 4 = {v}, the validation region starts at token {start}")
        if N % 4 not in (0, 1, 2, 3):
            errors["layout"] += 1
        starts = {s for s, _ in docs}
        if start not in starts:
            errors["the validation region does not start at a document start"] += 1
            print("  ERROR: the region does not start at a document boundary")
        else:
            nd = sum(1 for s, _ in docs if s >= start)
            print(f"  OK: it starts at a document boundary; {nd} held-out documents, {v} tokens ({100 * (~mask[start:]).sum() / v:.1f} % supervised)")

lens = np.array(lens)
print("document length (tokens, with eos) quantiles:", {q: int(np.quantile(lens, q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95, 0.99)} if len(lens) else None)
print("TOTAL", dict(tot))
for name, k, d_ids, d_mask in examples:
    out, inside = [], False
    for t, m in zip(d_ids, d_mask):
        if not m and not inside:
            out.append("[[")
            inside = True
        if m and inside:
            out.append("]]")
            inside = False
        out.append(tok.decode([t], skip_special_tokens=False))
    print(f"---- {name} doc {k}")
    print("".join(out)[:1500])
print("ERRORS:", dict(errors) if errors else "none")
sys.exit(1 if errors else 0)
