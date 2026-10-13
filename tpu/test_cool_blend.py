"""CPU tests of the cool-down blend patch of forge_jax.py (--cool-data, --cool-blend-steps, --cool-share, --cool-start).

    <repo>/tpu/venv/Scripts/python.exe test_cool_blend.py          (run from this folder: it imports forge_jax.py next to it and compares with the copy of the trainer from before the patch, tests_data/forge_jax_precool.py)

Checks
  1. cool_share: 0 before the first decay step, linear ramp, plateau, hard switch (blend 0), absolute start, same decay start as lr_at
  2. CoolPool: files whose name ends in _val are never trained on, heldout/cool_val.u16 is found, windows never straddle a shard
  3. the data stream is bit-identical to the unpatched trainer when the cool-down share is 0 (with and without a cool-down pool), and the validation windows are identical
  4. the share of cool-down rows in a batch is exact (stochastic rounding) and the cool rows are spread over the batch
  5. end to end on one CPU device: the producer thread uses the share of the batch it prepares, val_cool is logged, a resumed run continues the ramp at the right step, a run without --cool-data is unchanged
  6. end to end on 8 fake CPU devices (data parallel path)
"""
import importlib.util
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import forge_jax as fj  # the patched copy

spec = importlib.util.spec_from_file_location("forge_jax_orig", os.path.join(HERE, "tests_data", "forge_jax_precool.py"))  # the trainer as it was before the cool-down patch
orig = importlib.util.module_from_spec(spec)
spec.loader.exec_module(orig)

SEQ = 15  # windows of 16 tokens in the unit tests


def write_shard(path, shard_id, n):
    """tokens that tell where they come from: shard_id * 100 + (position mod 100); token // 100 is the shard id"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    (shard_id * 100 + (np.arange(n) % 100)).astype("<u2").tofile(path)


def make_dirs(tmp):
    main, cool = os.path.join(tmp, "main"), os.path.join(tmp, "cool")
    write_shard(os.path.join(main, "a_000.u16"), 1, 10_000)
    write_shard(os.path.join(main, "a_001.u16"), 2, 10_000)
    write_shard(os.path.join(main, "zzz_val.u16"), 3, 4_000)  # the validation tail of the main pool: the last 1000 tokens with --val-tokens 1000
    write_shard(os.path.join(cool, "cool_000.u16"), 11, 6_000)
    write_shard(os.path.join(cool, "cool_001.u16"), 12, 6_000)
    write_shard(os.path.join(cool, "heldout", "cool_val.u16"), 20, 3_000)  # held out: never trained on
    write_shard(os.path.join(cool, "cool_extra_val.u16"), 21, 3_000)  # a stray *_val file in the directory: must be ignored as training data
    return main, cool


def check_schedule():
    cs = fj.cool_share
    assert cs(0, 1000, 0.1, 100, 0.8) == 0.0 and cs(899, 1000, 0.1, 100, 0.8) == 0.0
    assert cs(900, 1000, 0.1, 100, 0.8) == 0.0  # the first decay step: the ramp starts at 0
    assert abs(cs(925, 1000, 0.1, 100, 0.8) - 0.2) < 1e-12 and abs(cs(950, 1000, 0.1, 100, 0.8) - 0.4) < 1e-12
    assert cs(1000, 1000, 0.1, 100, 0.8) == 0.8 and cs(5000, 1000, 0.1, 100, 0.8) == 0.8
    assert cs(899, 1000, 0.1, 0, 0.8) == 0.0 and cs(900, 1000, 0.1, 0, 0.8) == 0.8  # hard switch
    assert cs(9, 1000, 0.1, 100, 1.0, start=10) == 0.0 and cs(60, 1000, 0.1, 100, 1.0, start=10) == 0.5  # absolute start
    # the same first decay step as the learning-rate schedule (the step where lr_at starts to fall)
    for steps, frac in ((45000, 0.1), (90000, 0.1), (2000, 0.2), (777, 0.13)):
        first = int((1.0 - frac) * steps)
        assert fj.lr_at(first, steps, 1.0, 0, 0.05, frac) == 1.0 and fj.lr_at(first + 1, steps, 1.0, 0, 0.05, frac) < 1.0
        assert fj.cool_share(first - 1, steps, frac, 100, 0.8) == 0.0 and fj.cool_share(first + 100, steps, frac, 100, 0.8) == 0.8
    print("1. schedule: ok")


def check_pool_and_validation(tmp):
    main, cool = make_dirs(tmp)
    d = fj.Data(main, val_tokens=1000, flags=True, cool_directory=cool)
    assert [f.name for f in d.cool.files] == ["cool_000.u16", "cool_001.u16"], d.cool.files  # no cool_extra_val.u16, no heldout/
    assert d.cool.total_train == 12_000 and d.cool.heldout is not None and len(d.cool.heldout) == 3_000
    # the validation of the main pool is unchanged (identical to the untouched trainer)
    o = orig.Data(main, val_tokens=1000, flags=True)
    assert (d.total_train, d.val_start, d.val_tokens) == (o.total_train, o.val_start, o.val_tokens)
    for k in range(4):
        for x, y in zip(d.val_batch(k, 2, SEQ), o.val_batch(k, 2, SEQ)):
            assert np.array_equal(x, y)
    # held-out cool-down windows come from the held-out file only (shard id 20)
    ids = d.cool_val_batch(0, 4, SEQ)[0]
    assert set((ids // 100).ravel().tolist()) == {20}
    # training windows of both pools never contain held-out tokens and never straddle a shard (every window has one shard id)
    rng = np.random.default_rng(1)
    seen = set()
    for _ in range(300):
        win, _, _ = d.train_batch(rng, 32, SEQ, 0.5)
        shard_ids = win // 100
        assert (shard_ids == shard_ids[:, :1]).all(), "a window straddles two shards"
        seen |= set(shard_ids[:, 0].tolist())
    assert seen <= {1, 2, 3, 11, 12}, seen
    assert {1, 2, 3, 11, 12} <= seen, seen
    # a cool-down directory without a held-out file works (no val_cool line)
    cool2 = os.path.join(tmp, "cool2")
    write_shard(os.path.join(cool2, "cool_000.u16"), 11, 6_000)
    assert fj.Data(main, val_tokens=1000, cool_directory=cool2).cool.heldout is None
    print("2. pool, held-out file, windows: ok")


def check_stream_compat(tmp):
    main, cool = os.path.join(tmp, "main"), os.path.join(tmp, "cool")
    for seed in (0, 1, 2):
        ref, plain, with_pool, with_pool_zero = (np.random.default_rng(seed) for _ in range(4))
        d_ref = orig.Data(main, val_tokens=1000)
        d_plain = fj.Data(main, val_tokens=1000)
        d_pool = fj.Data(main, val_tokens=1000, cool_directory=cool)
        for _ in range(6):
            r = d_ref.train_batch(ref, 8, SEQ)
            for got in (d_plain.train_batch(plain, 8, SEQ), d_pool.train_batch(with_pool, 8, SEQ), d_pool.train_batch(with_pool_zero, 8, SEQ, 0.0)):
                for x, y in zip(got, r):
                    assert np.array_equal(x, y)
    print("3. stream with share 0 identical to the unpatched trainer: ok")


def check_shares(tmp):
    main, cool = os.path.join(tmp, "main"), os.path.join(tmp, "cool")
    d = fj.Data(main, val_tokens=1000, cool_directory=cool)
    rng = np.random.default_rng(3)
    first_row_cool = 0
    for share in (0.0, 0.25, 0.3, 0.5, 0.8, 1.0):
        counts = []
        for _ in range(600):
            win, _, _ = d.train_batch(rng, 32, SEQ, share)
            is_cool = (win[:, 0] // 100) >= 10
            counts.append(int(is_cool.sum()))
            if share == 0.5:
                first_row_cool += int(is_cool[0])
        mean = np.mean(counts) / 32
        assert abs(mean - share) < 0.004, (share, mean)
        if share in (0.0, 0.25, 0.5, 1.0):
            assert set(counts) == {int(share * 32)}, (share, set(counts))  # a whole number of rows: no randomness in the count
        else:
            assert set(counts) <= {int(share * 32), int(share * 32) + 1}, set(counts)
    assert 0.4 < first_row_cool / 600 < 0.6, first_row_cool  # the cool rows are not always the first ones
    print("4. shares of cool-down rows (stochastic rounding, mean exact, rows spread): ok")


class Spy:
    """records, for every batch the producer thread prepares, the share it was asked for and the number of cool-down rows it got"""

    def __init__(self):
        self.calls, self.instances = [], []
        self.orig_train, self.orig_init = fj.Data.train_batch, fj.Data.__init__
        spy = self

        def train_batch(this, rng, batch, seq, cool=0.0):
            out = spy.orig_train(this, rng, batch, seq, cool)
            spy.calls.append((id(this), cool, int(((out[0][:, 0] // 100) >= 10).sum())))
            return out

        def init(this, *a, **k):
            spy.orig_init(this, *a, **k)
            spy.instances.append(this)

        fj.Data.train_batch, fj.Data.__init__ = train_batch, init

    def of_last_run(self):
        me = id(self.instances[-1])
        return [(s, n) for i, s, n in self.calls if i == me]

    def close(self):
        fj.Data.train_batch, fj.Data.__init__ = self.orig_train, self.orig_init


def tiny_args(main, cool, out, steps, extra):
    return ["train", "--data", main, "--out", out, "--layers", "2", "--dim", "32", "--heads", "2", "--kv", "1", "--head-dim", "16", "--ffn", "64", "--vocab", "4400", "--seq", "16", "--tokens-per-step", "64", "--micro", "2",
            "--steps", str(steps), "--lr", "3e-3", "--optim", "muon", "--muon-lr", "3e-3", "--warmup", "2", "--eval-every", "5", "--eval-batches", "2", "--val-tokens", "1000", "--dtype", "f32", "--ce-chunk", "16",
            "--no-warm-data"] + (["--cool-data", cool] if cool else []) + extra


def read_log(out):
    return [json.loads(l) for l in open(os.path.join(out, "log.jsonl"))]


def check_end_to_end(tmp):
    main, cool = os.path.join(tmp, "main"), os.path.join(tmp, "cool")
    spy = Spy()
    try:
        # (a) default start = the first decay step: 24 steps, decay over the last half -> first = 12; ramp over 8 steps up to 0.75; 4 sequences per step
        out = os.path.join(tmp, "run_a")
        assert fj.main(tiny_args(main, cool, out, 24, ["--decay-frac", "0.5", "--cool-blend-steps", "8", "--cool-share", "0.75", "--save-every", "100"])) == 0
        calls = spy.of_last_run()[:24]
        assert len(calls) == 24
        for j, (share, n) in enumerate(calls):
            want = fj.cool_share(j, 24, 0.5, 8, 0.75)
            assert abs(share - want) < 1e-12, (j, share, want)  # the producer asks for the share of ITS batch index
            x = 4 * want
            assert (n in (int(x), int(x) + 1)) if x != int(x) else (n == int(x)), (j, n, x)
        assert [n for _, n in calls[:13]] == [0] * 13  # nothing before the first decay step (and the ramp is at 0 there)
        assert [n for _, n in calls[20:]] == [3] * 4  # the plateau: 0.75 of 4 sequences
        log = read_log(out)
        assert [r["cool_share"] for r in log] == [fj.cool_share(j, 24, 0.5, 8, 0.75) for j in range(24)]
        evals = [r for r in log if "val_loss" in r]
        assert [r["step"] for r in evals] == [5, 10, 15, 20, 24] and all("val_cool" in r and np.isfinite(r["val_cool"]) for r in evals), evals
        # (b) a resumed run continues the ramp: absolute start 6, ramp over 8 steps, first run stops at step 12 (checkpoint at the end), second run goes on to 20
        out = os.path.join(tmp, "run_b")
        extra = ["--cool-start", "6", "--cool-blend-steps", "8", "--cool-share", "0.75", "--save-every", "100"]
        assert fj.main(tiny_args(main, cool, out, 12, extra)) == 0
        first = spy.of_last_run()[:12]
        assert [round(s, 6) for s, _ in first] == [round(fj.cool_share(j, 12, 0.2, 8, 0.75, 6), 6) for j in range(12)]
        assert fj.main(tiny_args(main, cool, out, 20, extra + ["--resume"])) == 0
        second = spy.of_last_run()[:8]
        assert [round(s, 6) for s, _ in second] == [round(fj.cool_share(12 + j, 20, 0.2, 8, 0.75, 6), 6) for j in range(8)], second  # steps 12..19
        assert second[0][0] == 0.75 * 6 / 8
        log = read_log(out)
        assert [r["step"] for r in log] == list(range(1, 21)) and abs(log[12]["cool_share"] - fj.cool_share(12, 20, 0.2, 8, 0.75, 6)) < 1e-12
        # (c) without --cool-data nothing changes: no cool_share, no val_cool, no extra random numbers (the stream is that of the unpatched trainer)
        out = os.path.join(tmp, "run_c")
        assert fj.main(tiny_args(main, None, out, 6, ["--save-every", "100"])) == 0
        assert all(("cool_share" not in r and "val_cool" not in r) for r in read_log(out))
        assert all(s == 0.0 for s, _ in spy.of_last_run()[:6])
    finally:
        spy.close()
    print("5. end to end on one device (producer thread share, val_cool, resume, no cool-data): ok")


def check_end_to_end_8_devices(tmp):
    main, cool = os.path.join(tmp, "main"), os.path.join(tmp, "cool")
    out = os.path.join(tmp, "run_d")
    args = tiny_args(main, cool, out, 12, ["--decay-frac", "0.5", "--cool-blend-steps", "3", "--cool-share", "0.5", "--save-every", "6"])
    args[args.index("--tokens-per-step") + 1] = "1024"  # 64 sequences of 16 tokens: a multiple of 8 devices x micro 2
    args.remove("--no-warm-data")  # this run also starts the page-cache warm-up thread (it must accept the cool-down shards too)
    env = dict(os.environ, XLA_FLAGS="--xla_force_host_platform_device_count=8", JAX_PLATFORMS="cpu", PYTHONUTF8="1")
    r = subprocess.run([sys.executable, os.path.join(HERE, "forge_jax.py")] + args, capture_output=True, text=True, env=env, timeout=900)
    assert r.returncode == 0, r.stderr[-2000:]
    assert "cpu x 8" in r.stdout and "cool-down data: 2 shards, 0 M tokens, held-out file: yes; cool-down share 0 before step 6" in r.stdout, r.stdout[:1500]
    log = read_log(out)
    assert [r_["cool_share"] for r_ in log] == [fj.cool_share(j, 12, 0.5, 3, 0.5) for j in range(12)]
    assert all("val_cool" in r_ for r_ in log if "val_loss" in r_)
    assert "val_cool" in r.stdout and " cool 0.50" in r.stdout
    print("6. end to end on 8 fake devices: ok")


def check_kaggle_layout(tmp):
    """The Kaggle session keeps the cool-down shards next to the script (dataset forge-jax: forge_jax.py, cool_NNN.cool, cool_heldout.val): --cool-data @code --cool-glob "*.cool" --cool-blend-frac F; a missing pool must not stop the run."""
    import contextlib
    import io
    import shutil

    # the length of the ramp follows the number of steps when it is given as a fraction of the cool-down phase
    assert fj.cool_blend_len(1000, 900, 0, 0.0) == 0 and fj.cool_blend_len(1000, 900, 40, 0.0) == 40
    assert fj.cool_blend_len(1000, 900, 40, 0.25) == 25 and fj.cool_blend_len(2000, 1800, 40, 0.25) == 50 and fj.cool_blend_len(10, 20, 0, 0.5) == 0
    main, _ = make_dirs(os.path.join(tmp, "k"))  # a folder of its own: the shards of the earlier checks are still memory-mapped here, and Windows refuses to rewrite a mapped file
    code = os.path.join(tmp, "code")
    os.makedirs(code)
    shutil.copy(os.path.join(HERE, "forge_jax.py"), os.path.join(code, "forge_jax.py"))
    write_shard(os.path.join(code, "cool_000.cool"), 31, 6_000)
    write_shard(os.path.join(code, "cool_001.cool"), 32, 6_000)
    write_shard(os.path.join(code, "cool_heldout.val"), 40, 3_000)
    # a pattern of its own: the pool is the *.cool files, the held-out file is found by its name
    pool = fj.CoolPool(code, True, "*.cool")
    assert [f.name for f in pool.files] == ["cool_000.cool", "cool_001.cool"] and pool.heldout is not None and len(pool.heldout) == 3_000
    # a directory with main shards AND cool shards: the main pool must not see the .cool files, the cool pool must not see the .u16 files
    mixed = os.path.join(tmp, "mixed")
    shutil.copytree(main, mixed)
    for f in os.listdir(code):
        if f.endswith((".cool", ".val")):
            shutil.copy(os.path.join(code, f), os.path.join(mixed, f))
    d = fj.Data(mixed, val_tokens=1000, cool_directory=mixed, cool_pattern="*.cool")
    assert [f.name for f in d.files] == ["a_000.u16", "a_001.u16", "zzz_val.u16"] and [f.name for f in d.cool.files] == ["cool_000.cool", "cool_001.cool"]
    # fail-soft: no pool for an empty setting, a missing directory or a directory without shards (the last two print a warning)
    assert fj.open_cool_pool("") is None
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        assert fj.open_cool_pool(os.path.join(tmp, "nowhere")) is None
        assert fj.open_cool_pool(code, "*.nothing") is None
    assert buf.getvalue().count("WARNING: no cool-down pool") == 2, buf.getvalue()
    assert fj.open_cool_pool(code, "*.cool").total_train == 12_000
    # end to end, as the Kaggle runner calls it: a copy of the script in the 'code' folder, @code, a ramp as a fraction (24 steps, decay over the last 12 -> first 12; ramp 0.5 x 12 = 6 steps)
    env = dict(os.environ, JAX_PLATFORMS="cpu", PYTHONUTF8="1")
    out = os.path.join(tmp, "run_k")
    args = tiny_args(main, None, out, 24, ["--decay-frac", "0.5", "--cool-data", "@code", "--cool-glob", "*.cool", "--cool-blend-frac", "0.5", "--cool-share", "0.75", "--save-every", "100", "--time-limit-s", "600"])
    r = subprocess.run([sys.executable, os.path.join(code, "forge_jax.py")] + args, capture_output=True, text=True, env=env, timeout=900)
    assert r.returncode == 0, r.stderr[-2000:]
    assert "cool-down data: 2 shards" in r.stdout and "a linear ramp over 6 steps up to 0.75" in r.stdout and "held-out file: yes" in r.stdout, r.stdout[:2000]
    log = read_log(out)
    assert [r_["cool_share"] for r_ in log] == [fj.cool_share(j, 24, 0.5, 6, 0.75) for j in range(24)], [r_["cool_share"] for r_ in log]
    assert all("val_cool" in r_ for r_ in log if "val_loss" in r_)
    # the same run with a pool that does not exist: a warning, the run completes without cool-down fields
    out = os.path.join(tmp, "run_k2")
    args = tiny_args(main, None, out, 8, ["--cool-data", os.path.join(tmp, "nowhere"), "--save-every", "100"])
    r = subprocess.run([sys.executable, os.path.join(code, "forge_jax.py")] + args, capture_output=True, text=True, env=env, timeout=900)
    assert r.returncode == 0, r.stderr[-2000:]
    assert "WARNING: no cool-down pool" in r.stdout and "cool-down data:" not in r.stdout
    assert all(("cool_share" not in r_ and "val_cool" not in r_) for r_ in read_log(out))
    print("7. kaggle layout (@code, *.cool, ramp fraction, fail-soft pool): ok")


def main():
    check_schedule()
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:  # (memory-mapped files stay open on Windows until the process ends)
        check_pool_and_validation(tmp)
        check_stream_compat(tmp)
        check_shares(tmp)
        check_end_to_end(tmp)
        check_end_to_end_8_devices(tmp)
        check_kaggle_layout(tmp)
    print("all cool-down blend checks passed")


if __name__ == "__main__":
    main()
