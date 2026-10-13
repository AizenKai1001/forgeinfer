#!/usr/bin/env python3
"""mix_text.py OUT name=MB[,name=MB...] [--dir DIR]: a text corpus for tokenizer training: from each source's shard files (DIR/<name>_*.txt, documents separated by NUL) whole documents up to the given number of
megabytes, in the order of the arguments, documents of the sources interleaved in proportion to their quotas, so every part of the output is a mixture.   mix_text.py /tmp/tok_mix.txt fwedu=450,cosmo=130,pycode=120"""
import glob
import os
import sys

out = sys.argv[1]
spec = [(a.split("=")[0], int(a.split("=")[1]) * 1_000_000) for a in sys.argv[2].split(",")]
d = "/mnt/data/forgeinfer-data/text"
if "--dir" in sys.argv:
    d = sys.argv[sys.argv.index("--dir") + 1]


def docs(name):
    for path in sorted(glob.glob(os.path.join(d, name + "_*.txt"))):
        with open(path, "rb") as f:
            buf = b""
            while True:
                chunk = f.read(1 << 22)
                if not chunk:
                    break
                buf += chunk
                parts = buf.split(b"\x00")
                buf = parts.pop()
                for p in parts:
                    yield p
            if buf:
                yield buf


gens = {n: docs(n) for n, _ in spec}
left = dict(spec)
# every source gets a block of its quota / 200 per round, so all sources run out together and every part of the output (its last quarter too) has the mixture's proportions
blocks = {n: max(q // 200, 20_000) for n, q in spec}
total = 0
with open(out, "wb") as f:
    while any(v > 0 for v in left.values()):
        for n, _ in spec:
            block = 0
            while left[n] > 0 and block < blocks[n]:
                try:
                    p = next(gens[n])
                except StopIteration:
                    left[n] = 0
                    break
                f.write(p)
                f.write(b"\x00")
                left[n] -= len(p) + 1
                block += len(p) + 1
                total += len(p) + 1
print(f"{out}: {total / 1e6:.0f} MB")
