#!/usr/bin/env python3
"""Coverage of the next-token distribution by a restricted vocabulary (the draft head of speculative decoding only has to find the token the main model will choose):
the share of tokens of held-out texts that fall into (a) the first R ids and (b) the R most frequent tokens of a calibration text.
   token_coverage.py <tokenizer.json> <calib.txt> <eval.txt> [<eval2.txt> ...]"""
import sys
from collections import Counter
from tokenizers import Tokenizer

tok = Tokenizer.from_file(sys.argv[1])
calib = Counter(tok.encode(open(sys.argv[2], encoding="utf-8").read()).ids)
print("calibration:", sum(calib.values()), "tokens,", len(calib), "distinct")
ranked = [t for t, _ in calib.most_common()]
for path in sys.argv[3:]:
    ids = tok.encode(open(path, encoding="utf-8").read()).ids
    n = len(ids)
    print(path, n, "tokens, distinct", len(set(ids)))
    for r in (4096, 8192, 16384, 32768, 65536):
        first = sum(1 for t in ids if t < r) / n
        top = set(ranked[:r])
        freq = sum(1 for t in ids if t in top) / n
        print(f"  R={r:6d}: ids < R cover {100 * first:6.2f} %   the R most frequent calibration tokens cover {100 * freq:6.2f} %")
