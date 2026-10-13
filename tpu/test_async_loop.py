"""Tests of the host side of the JAX trainer's training loop (tpu/forge_jax.py main): --async-steps (the pipelined loop), --async-io (log and checkpoints written by background threads), --attn-pick (a splash kernel
without the self-test), --eval-early-until, the device scalars of the two learning rates and the evaluation that is dispatched in one go. All of it on the CPU: tiny models on 2 fake devices, a few minutes.

    JAX_PLATFORMS=cpu python test_async_loop.py          (from this folder)

The point of most checks: the options change WHEN the host does something, never what is computed: the checkpoints are byte-identical, the log records and the printed lines are the same.
FORGE_TEST_OLD_TRAINER=/path/to/an/older/forge_jax.py adds a comparison of the default behaviour with that trainer (the proof that the defaults did not move; run once when the loop changes: the files of an older
commit can be made with `git show COMMIT:tpu/forge_jax.py`).
"""
import contextlib
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import time

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=2")  # the trainers that the checks start see two devices
import jax
import jax.numpy as jnp

import forge_jax as fj
import test_forge_jax as tf

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "forge_jax.py")


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:16] if os.path.exists(path) else None


def run_tiny(tmp, tag, extra, script=SCRIPT, **env_extra):
    """A tiny training run in tmp/tag: returns the subprocess result with .out (the run folder) added."""
    out, args, env = tf._tiny_run_args(os.path.join(tmp, tag), extra)
    r = subprocess.run([sys.executable, script] + args, capture_output=True, text=True, env=dict(env, **env_extra), timeout=900)
    r.out = out
    return r


def records(out):
    """log.jsonl without the time-dependent field."""
    recs = [json.loads(l) for l in open(os.path.join(out, "log.jsonl"))]
    for rec in recs:
        rec.pop("tokens_per_s", None)
    return recs


def printed(stdout, ckpt_lines=True):
    """The lines of the trainer's output with every number that depends on the clock replaced (step times, elapsed seconds, write times); ckpt_lines=False leaves the lines about checkpoints out (a write in the
    background is reported in other words than a write in the foreground)."""
    keep = []
    for line in stdout.splitlines():
        if line.startswith(("step ", "[train] time plan", "[attn]", "[train] attention") + (("[train] checkpoint", "[train] pre-decay") if ckpt_lines else ())):
            line = re.sub(r"\d+ tok/s \([\d.]+ s\)( data wait [\d.]+ s)?", "TIME", line)
            line = re.sub(r"\([^)]*s elapsed[^)]*\)", "(TIME)", line)
            line = re.sub(r"[\d.]+ s per step[^-]*", "TIME", line)
            keep.append(line)
    return keep


def check_scalars():
    """The learning rates come as replicated device scalars that are made again only when the value changes: float32 rounding as jnp.float32 does it, and the same compiled program and the same numbers as before."""
    c = fj.Cfg(n_layer=2, n_embd=64, n_head=2, n_kv=1, hd=32, n_ff=128, vocab=512, seq=64, relu2=True, lambdas=True, ve_layers=(0, 1))
    h = fj.Hyper(peak_lr=3e-3, muon_lr=3e-3, mult_table=20.0, mult_scalar=10.0, use_muon=True)
    mesh, batch_sh, eval_sh, repl_sh = fj.make_mesh()
    ds = fj.DeviceScalars(repl_sh)
    a = ds.get("lr", 3e-3)
    assert ds.get("lr", 3e-3) is a, "the same value must give the same device array (no transfer)"
    assert ds.get("lr_muon", 3e-3) is not a, "another key is another scalar"
    b = ds.get("lr", 2.5e-3)
    assert b is not a and ds.get("lr", 2.5e-3) is b
    for x in (3e-3, 1e-5, 0.1, 1 / 3, 2.99999e-4, 1.0):
        got = np.asarray(ds.get("probe", x)).view(np.uint32)
        assert got == np.asarray(jnp.float32(x)).view(np.uint32), x  # the same float32 as jnp.float32(x)
    assert a.dtype == jnp.float32 and a.shape == () and a.sharding.is_fully_replicated and a.committed and not a.weak_type
    # the program: the step compiled with the old scalars (uncommitted, on device 0) and with the new ones (committed, replicated) is the same program, and the same numbers come out
    step = fj.build_train_step(c, h, jnp.float32, 32, False)

    def fresh():
        p = fj.init_params(c, 1)
        m = {n: jnp.zeros_like(x) for n, x in p.items()}
        v = {n: jnp.zeros_like(x) for n, x in p.items()}
        p, m, v = (jax.device_put(t, repl_sh) for t in (p, m, v))
        base = np.random.default_rng(0).integers(0, 512, size=(4, 65)).astype(np.int32)
        return (p, m, v, jax.device_put(jnp.asarray(0, jnp.int32), repl_sh), jax.device_put(base[None, :, :-1], batch_sh), jax.device_put(base[None, :, 1:], batch_sh),
                jax.device_put(np.zeros((1, 4, 64), bool), batch_sh))

    loc = re.compile(r"\{file_name_id=\d+ function_name_id=\d+ line=\d+ end_line=\d+ column=\d+ end_column=\d+\}")  # source positions in the metadata: they differ between two call sites
    args_old = fresh() + (jnp.float32(3e-3), jnp.float32(3e-3))
    args_new = fresh() + (ds.get("lr", 3e-3), ds.get("lr_muon", 3e-3))
    t_old = loc.sub("{loc}", step.lower(*args_old).compile().as_text())
    t_new = loc.sub("{loc}", step.lower(*args_new).compile().as_text())
    assert t_old == t_new, "the compiled step differs between the two kinds of scalars"
    o_old, o_new = step(*args_old), step(*args_new)
    bits = lambda x: np.asarray(x).view(np.uint32)
    assert float(o_old[4]) == float(o_new[4]) and float(o_old[5]) == float(o_new[5]) and all(np.array_equal(bits(o_old[0][k]), bits(o_new[0][k])) for k in o_old[0])
    print("device scalars: the same value gives the same array, float32 as jnp.float32, the same compiled program and the same step result: ok")


def check_parse_and_pick():
    """--attn-pick: the spelling, the refusal of a bad one, the kernel built without the self-test, the fall back to the normal selection, the failure of a picked kernel in the first step."""
    assert fj.parse_attn_pick("512,1") == (512, True) and fj.parse_attn_pick("256,0") == (256, False) and fj.parse_attn_pick("1024,1") == (1024, True)
    for bad in ("", "512", "512,2", "a,b", "100,1", "0,1", "512,1,1", "-512,1"):
        try:
            fj.parse_attn_pick(bad)
            raise AssertionError(f"{bad!r} was accepted")
        except ValueError:
            pass
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        try:
            fj.parse_args(["train", "--data", "d", "--out", "o", "--attn-pick", "512"])
            raise AssertionError("a bad --attn-pick was accepted")
        except SystemExit as e:
            assert e.code == 2 and "BLOCK,FUSED" in err.getvalue(), err.getvalue()
        try:
            fj.parse_args(["train", "--data", "d", "--out", "o", "--async-steps", "17"])
            raise AssertionError("--async-steps 17 was accepted")
        except SystemExit as e:
            assert e.code == 2
    a = fj.parse_args(["train", "--data", "d", "--out", "o"])
    assert a.async_steps == 0 and not a.async_io and a.attn_pick == "" and a.eval_early_until == 0  # everything off by default
    a = fj.parse_args(["train", "--data", "d", "--out", "o", "--attn-pick", "512,1", "--async-steps", "2", "--async-io", "--eval-early-until", "1000"])
    assert (a.attn_pick, a.async_steps, a.async_io, a.eval_early_until) == ("512,1", 2, True, 1000)
    # the helper with a stand-in for the kernel builder
    c = fj.Cfg(n_layer=2, n_embd=64, n_head=2, n_kv=1, hd=32, n_ff=128, vocab=512, seq=256)
    calls, logs = [], []
    real = fj.make_splash_attention
    try:
        fj.make_splash_attention = lambda c_, mesh, blk, fused, **kw: calls.append((blk, fused, kw)) or "KERNEL"
        got = fj.attention_from_pick(c, None, "512,1", log=logs.append, residual_name="res", qkv_name=None)
        assert got == "KERNEL" and calls == [(512, True, {"residual_name": "res", "qkv_name": None})] and "without the self-test" in logs[-1], (got, calls, logs)
        assert fj.attention_from_pick(c, None, "garbage", log=logs.append) is None and "cannot be used" in logs[-1] and len(calls) == 1
        fj.make_splash_attention = lambda *a_, **k_: (_ for _ in ()).throw(AssertionError("block does not divide the sequence"))
        assert fj.attention_from_pick(c, None, "512,0", log=logs.append) is None and "cannot be used (AssertionError" in logs[-1]
    finally:
        fj.make_splash_attention = real
    print("attn pick: spelling, refusal of a bad one, built without the self-test, an unusable pick gives None (normal selection): ok")


def _main_as_tpu(tmp, tag, extra, stand_in, select_stub):
    """main() in this process with a jax that says it is a TPU, the splash builder replaced by `stand_in` and select_attention by `select_stub`; returns (exit code, stdout)."""
    out, args, env = tf._tiny_run_args(os.path.join(tmp, tag), extra + ["--watchdog-s", "0"])
    saved = (jax.default_backend, fj.make_splash_attention, fj.select_attention, dict(fj.ATTN))
    jax.default_backend = lambda: "tpu"
    fj.make_splash_attention, fj.select_attention = stand_in, select_stub
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            rc = fj.main(args)
    finally:
        jax.default_backend, fj.make_splash_attention, fj.select_attention = saved[:3]
        fj.ATTN.clear()
        fj.ATTN.update(saved[3])
    return rc, buf.getvalue()


def check_pick_in_main():
    """main() on a (pretended) TPU: a pick builds that kernel and never runs the selection; a pick that cannot be built runs the selection; a picked kernel that fails in the first step falls back to XLA's attention."""
    def xla_like(c_, q, k, v_, cdt_, local=False):
        return fj.attention_xla(c_, q, k, v_, cdt_, c_.window if local else None)

    def no_selection(*a_, **k_):
        raise AssertionError("the selection must not run when the pick works")

    seen = []
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        rc, text = _main_as_tpu(tmp, "ok", ["--steps", "6", "--attn-pick", "512,1"], lambda c_, mesh, blk, fused, **kw: seen.append((blk, fused)) or xla_like, no_selection)
        assert rc == 0 and seen == [(512, True)] and "[attn] --attn-pick 512,1: splash block 512 fused backward without the self-test" in text and "[train] attention: splash kernel" in text, text[-1500:]
        assert "step     6/6" in text
        ran = []
        rc, text = _main_as_tpu(tmp, "unusable", ["--steps", "4", "--attn-pick", "512,1"], lambda *a_, **k_: (_ for _ in ()).throw(ValueError("no such kernel")), lambda *a_, **k_: ran.append(1) or None)
        assert rc == 0 and ran == [1] and "cannot be used (ValueError: no such kernel): the normal selection" in text and "[train] attention: XLA ops" in text, text[-1200:]

        def failing(c_, q, k, v_, cdt_, local=False):
            raise RuntimeError("the picked kernel does not compile in the full step")

        rc, text = _main_as_tpu(tmp, "fails", ["--steps", "6", "--attn-pick", "512,1", "--async-steps", "2"], lambda *a_, **k_: failing, no_selection)
        assert rc == 0 and "the splash kernel failed in the full training step" in text and "step     6/6" in text, text[-1500:]
        rc, text = _main_as_tpu(tmp, "xla", ["--steps", "3", "--attn", "xla", "--attn-pick", "512,1"], no_selection, no_selection)
        assert rc == 0 and "--attn-pick 512,1 ignored (--attn xla)" in text, text[-600:]
    # off a TPU (the CPU of the tests) the pick is ignored with a note
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        r = run_tiny(tmp, "cpu", ["--steps", "4", "--attn-pick", "512,1"])
        assert r.returncode == 0 and "--attn-pick 512,1 ignored (not a TPU)" in r.stdout, r.stdout[-800:] + r.stderr[-500:]
    print("attn pick in main(): used without the selection, an unusable pick selects normally, a failing picked kernel falls back to XLA, ignored with --attn xla and off a TPU: ok")


def check_lag_changes_nothing():
    """--async-steps 0, 1, 2 (and with --async-io): the same checkpoints byte for byte (the last one, the pre-decay one), the same log records, the same printed lines, with evaluations, checkpoints and a time plan."""
    plain = ["--steps", "36", "--eval-every", "7", "--save-every", "12", "--decay-frac", "0.3", "--min-lr-frac", "0.1", "--print-every", "5"]
    timed = ["--steps", "40", "--decay-frac", "0.5", "--min-lr-frac", "0.1", "--time-limit-s", "100000", "--plan-reserve-s", "1", "--save-every", "100", "--eval-every", "10"]
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        for name, extra in (("plain", plain), ("timed", timed)):
            ref = None
            for tag, more in (("lag0", []), ("lag1", ["--async-steps", "1"]), ("lag2", ["--async-steps", "2"]), ("lag2io", ["--async-steps", "2", "--async-io"]), ("lag7", ["--async-steps", "7"])):
                r = run_tiny(tmp, f"{name}_{tag}", extra + more)
                assert r.returncode == 0, (name, tag, r.stdout[-1000:], r.stderr[-1000:])
                got = {"latest": sha(os.path.join(r.out, "ckpt_latest.bin")), "predecay": sha(os.path.join(r.out, "ckpt_predecay.bin")), "records": records(r.out), "lines": printed(r.stdout, ckpt_lines=False)}
                if name == "timed":
                    assert got["predecay"] is not None and "pre-decay checkpoint at step 20" in r.stdout
                if ref is None:
                    ref = got
                    assert len(ref["records"]) == (36 if name == "plain" else 40) and sum(1 for x in ref["records"] if "val_loss" in x) >= 4
                else:
                    for key in ref:
                        assert got[key] == ref[key], (name, tag, key, got[key] if key != "records" else "", ref[key] if key != "records" else "")
    print("--async-steps 0/1/2/7 (and --async-io): byte-identical checkpoints (last and pre-decay), the same log records and printed lines, with evaluations, checkpoints and a time plan: ok")


def check_pipeline_overlaps_host_work():
    """What the pipeline is for: host work after a step (here 80 ms of sleep per step, the stand-in for formatting, json and the dispatch of the next step) is hidden behind the device's work of the next step. 60 steps of a
    tiny model with 16 micro-steps each (about 70 ms): the foreground loop needs step + host work per step, the pipeline the larger of the two."""
    walls = {}
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        for lag in ("0", "2"):
            t0 = time.time()
            r = run_tiny(tmp, f"overlap{lag}", ["--steps", "60", "--tokens-per-step", "4096", "--eval-every", "1000", "--save-every", "1000", "--print-every", "1000", "--async-steps", lag], FORGE_TEST_HOST_WORK_MS="80")
            walls[lag] = time.time() - t0
            assert r.returncode == 0, r.stderr[-800:]
        assert walls["2"] < walls["0"] - 1.5, walls  # about 4 s expected: 60 x 70 ms
    print(f"pipeline: 60 steps with 80 ms of host work each take {walls['0']:.1f} s without and {walls['2']:.1f} s with --async-steps 2 (the work is hidden behind the device): ok")


def check_same_as_old_trainer():
    """The default behaviour against an older trainer (FORGE_TEST_OLD_TRAINER): the same files and lines with every option off, and with --async-steps 2 --async-io."""
    old = os.environ.get("FORGE_TEST_OLD_TRAINER")
    if not old:
        print("default behaviour against an older trainer: skipped (set FORGE_TEST_OLD_TRAINER)")
        return
    extra = ["--steps", "36", "--eval-every", "7", "--save-every", "12", "--decay-frac", "0.5", "--min-lr-frac", "0.1", "--print-every", "5", "--time-limit-s", "100000", "--plan-reserve-s", "1"]
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        runs = {}
        for tag, script, more in (("old", old, []), ("new", SCRIPT, []), ("new_async", SCRIPT, ["--async-steps", "2", "--async-io"])):
            r = run_tiny(tmp, tag, extra + more, script=script)
            assert r.returncode == 0, (tag, r.stdout[-800:], r.stderr[-800:])
            runs[tag] = {"latest": sha(os.path.join(r.out, "ckpt_latest.bin")), "predecay": sha(os.path.join(r.out, "ckpt_predecay.bin")), "records": records(r.out), "lines": printed(r.stdout, ckpt_lines=(tag != "new_async"))}
        for key in ("latest", "predecay", "records", "lines"):
            assert runs["new"][key] == runs["old"][key], key
        for key in ("latest", "predecay", "records"):
            assert runs["new_async"][key] == runs["old"][key], key
        assert runs["new_async"]["lines"] == [l for l in runs["old"]["lines"] if not l.startswith(("[train] checkpoint", "[train] pre-decay"))]
    print("default behaviour against the older trainer: byte-identical checkpoints, the same records and lines (also with --async-steps 2 --async-io): ok")


def check_guards_with_lag():
    """Every guard of the loop still fires with a pipeline: the NaN stop (exit 98, nothing written), poisoned parameters at a save, the fallback of a failing kernel (step and evaluation), the watchdog, the producer, a resume."""
    import shutil

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        for lag in ("1", "3"):
            r = run_tiny(tmp, f"good{lag}", ["--steps", "10", "--save-every", "5", "--async-steps", lag])
            assert r.returncode == 0, r.stderr[-1500:]
            good = sha(os.path.join(r.out, "ckpt_latest.bin"))
            open(os.path.join(r.out, "ckpt_latest.bin.tmp"), "wb").write(b"half a file of a killed session")
            out, args, env = tf._tiny_run_args(os.path.join(tmp, f"good{lag}"), ["--steps", "10", "--save-every", "5"])
            i = args.index("--steps")
            args2 = args[:i] + ["--steps", "40"] + args[i + 2 :] + ["--resume", "--optim", "adamw", "--async-steps", lag, "--lr", "1e9", "--clip", "0"]
            r2 = subprocess.run([sys.executable, SCRIPT] + args2, capture_output=True, text=True, env=env, timeout=600)
            assert r2.returncode == 98 and "are not finite" in r2.stdout and "checkpoint at step 20" not in r2.stdout, (r2.returncode, r2.stdout[-800:])
            assert sha(os.path.join(out, "ckpt_latest.bin")) == good, "the checkpoint was overwritten"
            assert not os.path.exists(os.path.join(out, "ckpt_latest.bin.tmp"))
            first_bad = int(re.search(r"at step (\d+) are not finite", r2.stdout).group(1))
            shown = [int(l.split()[1].split("/")[0]) for l in r2.stdout.splitlines() if l.startswith("step ")]
            assert shown and max(shown) < first_bad and first_bad <= 20, (shown, first_bad)  # the step lines stop before the bad step: nothing after it is booked
        for extra in (["--async-steps", "2"], ["--async-steps", "2", "--async-io"]):  # poisoned parameters (a finite loss first) at a save step
            r = run_tiny(tmp, "poison" + str(len(extra)), ["--steps", "30", "--save-every", "10"] + extra, FORGE_TEST_POISON_AT_STEP="10")
            assert r.returncode == 98 and "not finite at step 10" in r.stdout, (r.returncode, r.stdout[-800:])
            assert not os.path.exists(os.path.join(r.out, "ckpt_latest.bin"))
        for mode, needle in (("step", "failed in the full training step"), ("eval", "failed in the evaluation")):  # the first step is read at once whatever the lag; the evaluation empties the pipeline
            args = ["--steps", "12", "--eval-every", "6", "--eval-batches", "1", "--save-every", "100", "--attn-test-fail", mode, "--async-steps", "2"]
            r = run_tiny(tmp, "fail" + mode, args)
            assert r.returncode == 0 and needle in r.stdout and "step    12/12" in r.stdout and " val " in r.stdout, (mode, r.stdout[-1200:], r.stderr[-600:])
        t_hang = time.time()  # a hung step ends in the watchdog's exit code 97 (steps 1 to 6 are booked or in the pipeline)
        r = run_tiny(tmp, "hang", ["--steps", "30", "--watchdog-s", "20", "--stall-trace-s", "5", "--async-steps", "1"], FORGE_TEST_HANG_AT_STEP="8")
        assert r.returncode == 97 and "watchdog" in r.stdout, (r.returncode, r.stdout[-600:])
        assert r.stdout.count("[train] STALL: nothing completed for") == 1 and "most recent call first" in r.stdout and " in main" in r.stdout, r.stdout[-1500:]  # the stack of the hung loop, once
        r = run_tiny(tmp, "producer", ["--steps", "30", "--async-steps", "2"], FORGE_TEST_PRODUCER_FAIL="4")
        assert r.returncode != 0 and "the batch producer thread failed" in r.stderr, (r.returncode, r.stderr[-800:])
        # a resume with a lag: the step numbers and learning rates continue as in the uninterrupted run
        full = run_tiny(tmp, "full", ["--steps", "30", "--save-every", "10", "--decay-frac", "0.4", "--min-lr-frac", "0.1", "--async-steps", "2"])
        part = run_tiny(tmp, "part", ["--steps", "30", "--save-every", "10", "--decay-frac", "0.4", "--min-lr-frac", "0.1", "--async-steps", "2", "--watchdog-s", "40"], FORGE_TEST_HANG_AT_STEP="11")
        assert part.returncode == 97, (part.returncode, part.stdout[-400:])
        out, args, env = tf._tiny_run_args(os.path.join(tmp, "part"), ["--steps", "30", "--save-every", "10", "--decay-frac", "0.4", "--min-lr-frac", "0.1", "--async-steps", "2"])
        r = subprocess.run([sys.executable, SCRIPT] + args + ["--resume"], capture_output=True, text=True, env=env, timeout=600)
        assert r.returncode == 0 and "resumed from step 10" in r.stdout, r.stdout[-600:]
        a_, b_ = records(full.out), records(out)
        assert [(x["step"], x["lr"]) for x in a_] == [(x["step"], x["lr"]) for x in b_], "a resume with a lag does not continue the schedule"
    print(f"guards with a pipeline: NaN stop (exit 98, checkpoint kept, no booked step after the bad one), poison at a save, kernel fallbacks, watchdog (exit 97, with the stack of every thread once), producer failure, resume: ok")


def check_async_io():
    """--async-io: the files are the same as in the foreground (a periodic checkpoint is taken at the pre-decay step), a slow disk does not hold the loop (writes are skipped while one is running), a failing write is reported and
    the final one is exit code 96, the log is complete at the end and a slow log writer does not slow the loop."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        base = ["--steps", "40", "--decay-frac", "0.5", "--min-lr-frac", "0.1", "--time-limit-s", "100000", "--plan-reserve-s", "1", "--save-every", "100", "--eval-every", "10"]
        ref = run_tiny(tmp, "sync", base)
        io_ = run_tiny(tmp, "io", base + ["--async-io"])
        assert ref.returncode == 0 and io_.returncode == 0, (io_.stdout[-800:], io_.stderr[-800:])
        assert "pre-decay checkpoint at step 20 (written by a background thread)" in io_.stdout and "written by the background thread (ckpt_predecay.bin" in io_.stdout, io_.stdout[-1500:]
        for name in ("ckpt_predecay.bin", "ckpt_latest.bin"):
            assert sha(os.path.join(ref.out, name)) == sha(os.path.join(io_.out, name)) is not None, name
        assert records(ref.out) == records(io_.out)
        assert not [f for f in os.listdir(io_.out) if f.endswith(".tmp")]
        # periodic checkpoints in the background and a slow disk: writes are skipped while one runs, and the last one is complete (the final write waits for the thread)
        slow = run_tiny(tmp, "slow", ["--steps", "30", "--save-every", "6", "--async-io"], FORGE_TEST_SLOW_WRITE="2.5")
        assert slow.returncode == 0, (slow.stdout[-800:], slow.stderr[-800:])
        assert "skipped: the background thread is still writing the previous one" in slow.stdout and "written by the background thread" in slow.stdout, slow.stdout[-1800:]
        plain = run_tiny(tmp, "slowref", ["--steps", "30", "--save-every", "6"])
        assert sha(os.path.join(slow.out, "ckpt_latest.bin")) == sha(os.path.join(plain.out, "ckpt_latest.bin")) is not None
        assert not [f for f in os.listdir(slow.out) if f.endswith(".tmp")]
        # a failing write: the thread reports it, the run goes on, the final (foreground) write ends the run with exit code 96
        out_, args, env = tf._tiny_run_args(os.path.join(tmp, "fail"), ["--steps", "12", "--save-every", "5", "--async-io"])
        os.makedirs(os.path.join(out_, "ckpt_latest.bin.tmp", "x"))
        r = subprocess.run([sys.executable, SCRIPT] + args, capture_output=True, text=True, env=env, timeout=600)
        assert r.returncode == 96 and r.stdout.count("writing ckpt_latest.bin failed") == 3 and "step    12/12" in r.stdout and "the final checkpoint could not be written" in r.stdout, (r.returncode, r.stdout[-1500:])
        # a slow log writer: the loop is not held by it (its own clock at the last checkpoint is far below the time the writer needs), and every line is on the disk at the end
        t0 = time.time()
        r = run_tiny(tmp, "slowlog", ["--steps", "30", "--save-every", "30", "--async-io"], FORGE_TEST_SLOW_LOG="0.5")
        wall = time.time() - t0
        assert r.returncode == 0, r.stderr[-800:]
        loop_s = float(re.search(r"checkpoint at step 30 \((\d+) s elapsed", r.stdout).group(1))
        assert len(records(r.out)) == 30 and wall > 14.0 and loop_s < 12.0, (loop_s, wall, len(records(r.out)))
        print(f"async io: same pre-decay and final files as in the foreground, skipped writes on a slow disk, a failing write reported (exit 96 for the last one), slow log writer ({wall:.0f} s in all, the loop took {loop_s:.0f} s), complete log: ok")


def check_async_checkpoint_bytes():
    """The file of a BACKGROUND write (state copied to host memory while the next steps already run) equals the file that a foreground write of the same step makes: stop two runs at the same step, one writing the
    periodic checkpoint in the background and one in the foreground, and keep the files by renaming them from the pre-decay slot."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        # the pre-decay checkpoint (step 12 of 24) is written in the background with --async-io and is never overwritten; the foreground run writes the same file in the foreground
        base = ["--steps", "24", "--decay-frac", "0.5", "--time-limit-s", "100000", "--plan-reserve-s", "1", "--save-every", "100", "--eval-every", "100", "--async-steps", "2"]
        a_ = run_tiny(tmp, "bg", base + ["--async-io"])
        b_ = run_tiny(tmp, "fg", base)
        assert a_.returncode == 0 and b_.returncode == 0
        assert "pre-decay checkpoint at step 12 (written by a background thread)" in a_.stdout
        assert sha(os.path.join(a_.out, "ckpt_predecay.bin")) == sha(os.path.join(b_.out, "ckpt_predecay.bin")) is not None
        c = fj.cfg_from_checkpoint(os.path.join(a_.out, "ckpt_predecay.bin"))
        p, m, v, st = fj.load_checkpoint(os.path.join(a_.out, "ckpt_predecay.bin"), c)
        assert st == 12 and fj.tree_finite(p) and fj.tree_finite(m) and fj.tree_finite(v)
        # the writer thread on its own: a failing job leaves no .tmp, a good job leaves the file, the sync_every path writes the same bytes
        c2 = fj.Cfg(n_layer=2, n_embd=32, n_head=2, n_kv=1, hd=16, n_ff=64, vocab=300, seq=16, relu2=True, lambdas=True, ve_layers=(0,))
        p2 = fj.init_params(c2, 3)
        m2, v2 = ({n: x * k for n, x in p2.items()} for k in (0.5, 2.0))
        host = tuple({n: fj.host_copy(x) for n, x in t.items()} for t in (p2, m2, v2))
        fj.save_checkpoint(os.path.join(tmp, "a.bin"), c2, p2, m2, v2, 7, plan_steps=100)
        fj.save_checkpoint(os.path.join(tmp, "b.bin"), c2, *host, 7, plan_steps=100, sync_every=1000)  # fsync every 1000 bytes: many syncs
        assert sha(os.path.join(tmp, "a.bin")) == sha(os.path.join(tmp, "b.bin"))
        w = fj.CheckpointWriter()
        w.start(os.path.join(tmp, "w.bin"), c2, host, 7, 100, "w.bin")
        w.wait()
        assert w.collect()[2:3] == ("ok",) and sha(os.path.join(tmp, "w.bin")) == sha(os.path.join(tmp, "a.bin")) and not os.path.exists(os.path.join(tmp, "w.bin.tmp"))
        os.makedirs(os.path.join(tmp, "bad.bin.tmp", "x"))
        w.start(os.path.join(tmp, "bad.bin"), c2, host, 7, 100, "bad.bin")
        w.wait()
        r = w.collect()
        assert r[2] == "failed" and not os.path.exists(os.path.join(tmp, "bad.bin")), r
        # a snapshot is a real copy: changing the device array afterwards (donation) does not change it
        x = jnp.arange(8, dtype=jnp.float32)
        hx = fj.host_copy(x)
        assert hx.flags.owndata and hx.base is None and np.array_equal(hx, np.arange(8, dtype=np.float32))
    print("background checkpoint: the file of a background write equals the foreground one, a failing job leaves no half file, the snapshot owns its memory: ok")


def check_bench_worker_pick():
    """tpu/bench_step.py's worker with a pick (the variants after `base` of the autotune): the kernel is built without the self-test, the row says which pick it used; a model with sliding-window layers still gets the full
    selection (base has no window, so the pick says nothing about the second kernel); without a pick the selection runs as before."""
    sys.path.insert(0, HERE)
    import bench_step

    def worker_row(extra_args, pick, select):
        argv = ["--worker", "--name", "t", "--args=" + bench_step.ARGS_TINY + " " + extra_args, "--steps", "2", "--warmup", "1"] + (["--attn-pick", pick] if pick else [])
        saved = (jax.default_backend, fj.make_splash_attention, fj.select_attention, dict(fj.ATTN))
        jax.default_backend = lambda: "tpu"
        fj.make_splash_attention = lambda c_, mesh, blk, fused, **kw: (lambda c2, q, k, v, cdt, local=False: fj.attention_xla(c2, q, k, v, cdt, c2.window if local else None))
        fj.select_attention = select
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                rc = bench_step.main(argv)
        finally:
            jax.default_backend, fj.make_splash_attention, fj.select_attention = saved[:3]
            fj.ATTN.clear()
            fj.ATTN.update(saved[3])
        assert rc == 0
        return json.loads([l for l in buf.getvalue().splitlines() if l.startswith("{")][-1])

    def no_selection(*a_, **k_):
        raise AssertionError("the selection must not run")

    row = worker_row("", "512,1", no_selection)
    assert row["status"] == "ok" and row["attn_pick"] == "512,1" and row["attention"] == "splash block 512 fused backward", row
    chosen = []
    row = worker_row("--window 16 --window-layers auto", "512,1", lambda *a_, **k_: chosen.append(k_.get("candidates")) or None)
    assert row["status"] == "ok" and chosen == [((512, True),)] and row["attn_pick"] == "xla", (row, chosen)  # the windowed model selected (the restricted selection of the pick)
    row = worker_row("", "", lambda *a_, **k_: chosen.append("full") or None)
    assert row["status"] == "ok" and chosen[-1] == "full" and row["attn_pick"] == "xla"
    print("bench_step worker: a pick builds the kernel without the self-test, a windowed model still selects, no pick selects as before: ok")


def check_time_limit_with_pipeline():
    """A run that the time limit stops (no plan: --no-auto-steps) with a pipeline: it stops with an empty pipeline, so that the final checkpoint is the state after exactly the last step of log.jsonl, and the unfinished decay is
    reported as before."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        for tag, more in (("lag0", []), ("lag2", ["--async-steps", "2"]), ("lag2io", ["--async-steps", "2", "--async-io"])):
            r = run_tiny(tmp, tag, ["--steps", "1000000", "--no-auto-steps", "--time-limit-s", "14", "--save-every", "100000", "--eval-every", "100000", "--print-every", "100000"] + more)
            assert r.returncode == 0 and "time limit of 14 s reached: stop" in r.stdout and "the learning-rate decay is NOT finished" in r.stdout, (tag, r.returncode, r.stdout[-800:], r.stderr[-500:])
            recs = records(r.out)
            ck = os.path.join(r.out, "ckpt_latest.bin")
            c = fj.cfg_from_checkpoint(ck)
            step = fj.load_checkpoint(ck, c)[3]
            assert step == len(recs) == recs[-1]["step"] and step > 20, (tag, step, len(recs))
            assert not [f for f in os.listdir(r.out) if f.endswith(".tmp")]
    print("time limit with a pipeline: the run stops with an empty pipeline, the final checkpoint is the state after the last logged step, the unfinished decay is reported: ok")


def check_time_plan_with_pipeline():
    """The time planner with a pipeline and background writers (the same run as test_forge_jax.check_auto_steps): a schedule far longer than the time limit is planned so that the decay ends inside the limit, the plan is
    moved while the decay has not started, and the last learning rate is the floor."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        r = run_tiny(tmp, "plan", ["--steps", "1000000", "--decay-frac", "0.2", "--min-lr-frac", "0.1", "--time-limit-s", "45", "--plan-reserve-s", "6", "--eval-every", "50", "--eval-batches", "1", "--save-every", "50", "--lr", "3e-3",
                                   "--async-steps", "2", "--async-io"])
        assert r.returncode == 0, (r.stdout[-800:], r.stderr[-800:])
        assert "time plan:" in r.stdout and "time limit of" not in r.stdout, r.stdout[-1500:]
        last = [l for l in r.stdout.splitlines() if l.startswith("step ")][-1]
        total, done = int(last.split("/")[1].split()[0]), int(last.split()[1].split("/")[0])
        assert done == total and total < 1_000_000, last
        assert float(last.split(" lr ")[1].split()[0]) < 4e-4, last  # the schedule ended near its floor (3e-4)
        c = fj.cfg_from_checkpoint(os.path.join(r.out, "ckpt_latest.bin"))
        assert fj.load_checkpoint(os.path.join(r.out, "ckpt_latest.bin"), c)[3] == total and fj.checkpoint_plan(os.path.join(r.out, "ckpt_latest.bin")) == total
    print(f"time plan with a pipeline and background writers: planned {total} steps for a 45 s limit, finished the decay (last learning rate {float(last.split(' lr ')[1].split()[0]):.2e}): ok")


def check_eval_schedule():
    """--eval-early-until: every 100 steps until that step, then every --eval-every; the last step always."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        r = run_tiny(tmp, "early", ["--steps", "340", "--eval-every", "1000", "--eval-early-until", "250", "--eval-batches", "1", "--save-every", "1000", "--print-every", "1000", "--async-steps", "1"])
        assert r.returncode == 0, r.stdout[-600:] + r.stderr[-600:]
        at = [x["step"] for x in records(r.out) if "val_loss" in x]
        assert at == [100, 200, 340], at
        r = run_tiny(tmp, "late", ["--steps", "340", "--eval-every", "120", "--eval-early-until", "100", "--eval-batches", "1", "--save-every", "1000", "--print-every", "1000"])
        at = [x["step"] for x in records(r.out) if "val_loss" in x]
        assert at == [100, 120, 240, 340], at  # 100 from the early rule, then every 120, the last step
        r = run_tiny(tmp, "off", ["--steps", "250", "--eval-every", "120", "--eval-batches", "1", "--save-every", "1000", "--print-every", "1000"])
        assert [x["step"] for x in records(r.out) if "val_loss" in x] == [120, 240, 250]  # off by default: as before
    print("evaluation schedule: --eval-early-until adds the early evaluations, the default is unchanged: ok")


if __name__ == "__main__":
    check_scalars()
    check_parse_and_pick()
    check_pick_in_main()
    check_lag_changes_nothing()
    check_pipeline_overlaps_host_work()
    check_same_as_old_trainer()
    check_async_checkpoint_bytes()
    check_async_io()
    check_bench_worker_pick()
    check_eval_schedule()
    check_time_limit_with_pipeline()
    check_time_plan_with_pipeline()
    check_guards_with_lag()
    print("all async-loop checks passed")
