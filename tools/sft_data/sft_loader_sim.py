"""loader_sim.py DIR [--windows N]: re-implementation of the data loader of `forge train` (crates/forge-cli/src/train_cmd.rs, Data::open / window / val_window) in numpy, to check the validation region and the share of targets
of the training windows without a GPU: sorted *.u16 files, the LAST one gives min(val_tokens, tokens / 4) of its tail to validation, training windows of seq + 1 tokens start at a uniform random position of the training tokens."""
import argparse, glob, os
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("dir")
ap.add_argument("--val-tokens", type=int, default=20_000_000)
ap.add_argument("--seq", type=int, default=2048)
ap.add_argument("--windows", type=int, default=20000)
a = ap.parse_args()
files = sorted(glob.glob(os.path.join(a.dir, "*.u16")))
maps = [np.memmap(f, dtype="<u2", mode="r") for f in files]
n = len(maps)
train_tokens = []
val_start = 0
for i, m in enumerate(maps):
    t = len(m)
    tt = t
    if i + 1 == n:
        v = min(a.val_tokens, t // 4)
        tt = t - v
        val_start = tt
    train_tokens.append(tt)
total_train = sum(train_tokens)
val_tokens = len(maps[-1]) - val_start
print(f"[train] data: {n} shards ({[os.path.basename(f) for f in files]}), {total_train / 1e9:.2f} G training tokens ({total_train:,}), {val_tokens / 1e6:.1f} M validation tokens ({val_tokens:,})")
L = a.seq + 1
rng = np.random.default_rng(1)
tot_t = tot_valid = 0
from_file = np.zeros(n)
for _ in range(a.windows):
    p = int(rng.integers(0, total_train - L))
    for i, m in enumerate(maps):
        if p + L <= train_tokens[i]:
            w = np.asarray(m[p : p + L])
            break
        if p < train_tokens[i]:
            p = 0
            continue
        p -= train_tokens[i]
    from_file[i] += 1
    tgt = w[1:]
    tot_t += len(tgt)
    tot_valid += int(((tgt & 0x8000) == 0).sum())
print(f"windows: {a.windows}; share of windows per file: " + ", ".join(f"{os.path.basename(f)} {100 * c / a.windows:.1f}%" for f, c in zip(files, from_file)))
print(f"share of positions with a target (bit 15 clear): {100 * tot_valid / tot_t:.1f} %  -> {tot_valid / a.windows * 32:.0f} targets per 32 windows (one 65,536-token step)")
# validation windows: p = val_start + (k * 7919 * 4093) % span, span = max(val_tokens - L, 1)
last = maps[-1]
span = max(val_tokens - L, 1)
ok = 0
vt = vv = 0
ks = range(96)
for k in ks:
    p = val_start + (k * 7919 * 4093) % span
    assert p + L <= len(last), (k, p)
    ok += 1
    w = np.asarray(last[p : p + L])
    t = w[1:]
    vt += len(t)
    vv += int(((t & 0x8000) == 0).sum())
print(f"validation windows (--eval-batches 48 x micro 2 = 96): all {ok} inside the file; start at {val_start:,}; positions with a target {100 * vv / vt:.1f} %")
# is the validation region a set of whole held-out documents? (first token of the region must follow an <|endoftext|>)
first = int(last[val_start]) & 0x7FFF
prev = int(last[val_start - 1]) & 0x7FFF
print(f"first token of the validation region = id {first} (<|im_start|> is 1), the token before it = id {prev} (<|endoftext|> is 0)")
