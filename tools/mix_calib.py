#!/usr/bin/env python3
"""A calibration text: the calibration texts of every available domain, a fixed number of characters each (times a scale), cut on character boundaries.
   mix_calib.py [dir [scale [output file name]]]     default: ~/forgeinfer/data/eval, 1.0, calib_multi.txt"""
import os
import sys

E = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/forgeinfer/data/eval")
scale = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
name = sys.argv[3] if len(sys.argv) > 3 else "calib_multi.txt"
parts = []
for d, n in (("en", 110000), ("code", 110000), ("zh", 60000), ("de", 90000), ("math", 100000), ("chat", 100000)):
    p = os.path.join(E, "calib_" + d + ".txt")
    if os.path.exists(p):
        parts.append(open(p, encoding="utf-8", errors="ignore").read()[: int(n * scale)])
open(os.path.join(E, name), "w", encoding="utf-8").write("\n\n".join(parts))
print(name + ":", sum(len(x) for x in parts), "characters from", len(parts), "domains")
