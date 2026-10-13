#!/usr/bin/env python3
"""ckpt_params_only.py IN.bin OUT.bin [--check]      cut a checkpoint to its weights
ckpt_params_only.py FILE.bin --verify               exit 0 only when the file is complete (header + 12 bytes per parameter, or header + 4 bytes per parameter for a weights-only file)

Cut a training checkpoint (forgeckpt1: one JSON header line, n_params float32 parameters, then the two Adam moment arrays: 12 bytes per parameter in all) down to the header line plus the parameters (4 bytes per parameter).
Everything that only needs the weights reads exactly that prefix and nothing after it (`forge train --init-from` = `load_params`, `read_checkpoint_params`, `export_hf`, the chat engine, the ladder and the probes), so a fine-tune or an
evaluation of a finished model can start from a file three times smaller: the 365M model is 4.4 GB with moments and 1.46 GB without, and the upload to the Modal volume is the slow step. NOT for resuming a pre-training run (`--resume` needs the moments).
The input is only read; the output is written to a temporary name and renamed when complete. --check reads the written file back and verifies header, length and that the parameters equal the prefix of the input (byte for byte).
"""
import json
import os
import sys


def verify(path):
    """Complete = the size the header promises: header + 12 n bytes (parameters and both Adam moments) or header + 4 n bytes (weights only). A download that broke off (the Kaggle client lost the connection at 3.2 of 4.4 GB once) fails here."""
    with open(path, "rb") as f:
        line = f.readline()
    try:
        h = json.loads(line)
    except ValueError:
        sys.exit(f"{path}: no JSON header line")
    if not isinstance(h, dict) or h.get("magic") != "forgeckpt1":
        sys.exit(f"{path}: not a forgeckpt1 checkpoint")
    n = int(h["n_params"])
    size = os.path.getsize(path)
    if size == len(line) + 12 * n:
        print(f"{path}: complete, {size} bytes, {n} parameters with both moments")
    elif size == len(line) + 4 * n:
        print(f"{path}: complete, {size} bytes, {n} parameters, weights only")
    else:
        sys.exit(f"{path}: INCOMPLETE: {size} bytes, but the header promises {len(line) + 4 * n} (weights only) or {len(line) + 12 * n} (with moments)")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    check = "--check" in sys.argv[1:]
    if "--verify" in sys.argv[1:]:
        if len(args) != 1:
            sys.exit(__doc__)
        verify(args[0])
        return
    if len(args) != 2:
        sys.exit(__doc__)
    src, dst = args
    with open(src, "rb") as f:
        line = f.readline()
        h = json.loads(line)
        if h.get("magic") != "forgeckpt1":
            sys.exit(f"{src}: not a forgeckpt1 checkpoint")
        n = int(h["n_params"])
        want = len(line) + n * 4
        size = os.path.getsize(src)
        if size < want:
            sys.exit(f"{src}: file has {size} bytes but header + {n} parameters need {want}")
        tmp = dst + ".part"
        left = n * 4
        with open(tmp, "wb") as o:
            o.write(line)
            while left:
                b = f.read(min(left, 64 << 20))
                if not b:
                    sys.exit("unexpected end of file")
                o.write(b)
                left -= len(b)
    os.replace(tmp, dst)
    print(f"{dst}: {os.path.getsize(dst)} bytes ({n} parameters; {src} has {size} bytes: moments {'present' if size >= want + n * 8 else 'absent or partial'})")
    if check:
        with open(src, "rb") as a, open(dst, "rb") as b:
            if a.readline() != b.readline():
                sys.exit("CHECK FAILED: header differs")
            left = n * 4
            while left:
                x, y = a.read(min(left, 64 << 20)), b.read(min(left, 64 << 20))
                if x != y or not x:
                    sys.exit("CHECK FAILED: parameters differ")
                left -= len(x)
            if b.read(1):
                sys.exit("CHECK FAILED: extra bytes after the parameters")
        print("check ok: header and all parameters identical to the input prefix")


if __name__ == "__main__":
    main()
