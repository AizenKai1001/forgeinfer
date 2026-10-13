#!/usr/bin/env python3
"""human_check_published.py: last checks of tokens_sft_human and tokens_sft_human_replay (byte equality with the build, the held-out files, the file-name rule of the trainer's validation tail, the replay links).
Prints one line per check and exits 1 when a check fails."""
import filecmp
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import human_cfg as C  # noqa: E402

bad = []


def check(name, ok, detail=""):
    print(("ok   " if ok else "FAIL ") + name + (f": {detail}" if detail else ""))
    if not ok:
        bad.append(name)


plan = json.load(open(os.path.join(C.BUILD, "plan.json")))
T = plan["validation_tokens_T"]
names = sorted(plan["shard_tokens"])
for n in names:
    check(f"{n}.u16 identical to the build file", filecmp.cmp(os.path.join(C.BUILD, n + ".u16"), os.path.join(C.OUT, n + ".u16"), shallow=False))
files = sorted(os.listdir(C.OUT))
u16 = [f for f in files if f.endswith(".u16")]
check("the last .u16 file (alphabetical) is zzz_val.u16", u16[-1] == "zzz_val.u16", str(u16))
check("no file in the data directory sorts after zzz_val.u16", all(f <= "zzz_val.u16" for f in files if not f.startswith(".")), str(files))
z = np.fromfile(os.path.join(C.OUT, "zzz_val.u16"), dtype="<u2")
check("zzz_val.u16 has exactly 4 x T tokens", len(z) == 4 * T, f"{len(z)} = 4 x {T}")
m = np.fromfile(os.path.join(C.OUT, "heldout", "sft_val_masked.u16"), dtype="<u2")
p = np.fromfile(os.path.join(C.OUT, "heldout", "sft_val_plain.u16"), dtype="<u2")
check("heldout masked == the last T tokens of zzz_val.u16", len(m) == T and np.array_equal(m, z[-T:]))
check("heldout plain == masked with bit 15 cleared", np.array_equal(p, m & 0x7FFF))
check("heldout plain has no id >= 32768 (forge eval-ppl would fault)", int(p.max()) < 32768, f"max id {int(p.max())}")
check("the held-out region starts at a document start (<|im_start|> after <|endoftext|>)", int(z[-T] & 0x7FFF) == 1 and int(z[-T - 1] & 0x7FFF) == 0)
rep = sorted(os.listdir(C.OUT_REPLAY))
check("replay dir: a_replay.u16 first, zzz_val.u16 last, nothing after it", rep[0] == "a_replay.u16" and rep[-1] == "zzz_val.u16", str(rep))
ok = True
for f in rep:
    if f != "a_replay.u16":
        t = os.path.join(C.OUT_REPLAY, f)
        ok = ok and os.path.islink(t) and os.path.samefile(t, os.path.join(C.OUT, f))
check("replay symlinks point at the SFT files", ok)
a = np.fromfile(os.path.join(C.OUT_REPLAY, "a_replay.u16"), dtype="<u2")
check("replay tokens are plain ids (no bit 15)", int(a.max()) < 32768)
src = np.fromfile("/mnt/data/forgeinfer-data/tokens_cool/replay/replay_000.u16", dtype="<u2", count=len(a))
check("replay = the first tokens of tokens_cool/replay/replay_000.u16", np.array_equal(a, src))
sft_tokens = sum(os.path.getsize(os.path.join(C.OUT, f)) // 2 for f in u16)
check("replay share is 12 %", abs(len(a) / (len(a) + sft_tokens) - 0.12) < 0.001, f"{100 * len(a) / (len(a) + sft_tokens):.2f} %")
old = "/mnt/data/forgeinfer-data/tokens_sft_mix2"
check("the old mix tokens_sft_mix2 is untouched (files and sizes as in its manifest)", all(os.path.exists(os.path.join(old, f)) for f in ("sft_000.u16", "sft_001.u16", "sft_002.u16", "sft_003.u16", "zzz_val.u16", "MANIFEST.txt")))
print("FAILED:", bad if bad else "none")
sys.exit(1 if bad else 0)
