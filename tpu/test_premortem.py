"""Regression checks of the pre-mortem review of the TPU pre-training path (docs/TPU_PREMORTEM.md): each check reproduces one way in which a long session could be wasted or a model silently degraded.

    JAX_PLATFORMS=cpu python test_premortem.py          (from this folder; tiny models on 2 fake CPU devices, a few minutes)
"""
import hashlib
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=2")  # the trainers that the checks start (and the runner starts) see two devices
import jax.numpy as jnp

import forge_jax as fj
import test_forge_jax as tf

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "forge_jax.py")


def run(args, env, **extra_env):
    return subprocess.run([sys.executable, SCRIPT] + args, capture_output=True, text=True, env=dict(env, **extra_env), timeout=600)


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def check_nan_stops_without_overwriting_the_checkpoint():
    """A NaN loss used to be written into ckpt_latest.bin every save_every steps and the run ended with exit code 0 (found by a run with an absurd learning rate): now exit code 98 and the last good checkpoint stays."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        out, args, env = tf._tiny_run_args(tmp, ["--steps", "10", "--save-every", "5"])
        r = run(args, env)
        assert r.returncode == 0, r.stderr[-1500:]
        good = sha(os.path.join(out, "ckpt_latest.bin"))
        open(os.path.join(out, "ckpt_latest.bin.tmp"), "wb").write(b"half a file of a killed session")  # a stale temporary file of an earlier session is removed at the start
        i = args.index("--steps")
        args2 = args[:i] + ["--steps", "40"] + args[i + 2 :] + ["--resume", "--optim", "adamw", "--lr", "1e9", "--clip", "0"]
        j = args2.index("--lr")
        args2 = args2[:j] + args2[j + 2 :]  # the first --lr (3e-3) is replaced by the last one (argparse takes the last), keep only the absurd one
        r = run(args2, env)
        assert r.returncode == 98, (r.returncode, r.stdout[-800:])
        assert "are not finite" in r.stdout and "step    20/" not in r.stdout and "checkpoint at step 20" not in r.stdout, r.stdout[-800:]
        assert sha(os.path.join(out, "ckpt_latest.bin")) == good, "the checkpoint was overwritten"
        assert not os.path.exists(os.path.join(out, "ckpt_latest.bin.tmp"))
        c = fj.cfg_from_checkpoint(os.path.join(out, "ckpt_latest.bin"))
        p, m, v, st = fj.load_checkpoint(os.path.join(out, "ckpt_latest.bin"), c)
        assert st == 10 and fj.tree_finite(p) and fj.tree_finite(m) and fj.tree_finite(v)
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:  # a finite loss followed by poisoned parameters, at a save step: nothing is written
        out, args, env = tf._tiny_run_args(tmp, ["--steps", "30", "--save-every", "10"])
        r = run(args, env, FORGE_TEST_POISON_AT_STEP="10")
        assert r.returncode == 98 and "not finite at step 10" in r.stdout, (r.returncode, r.stdout[-800:])
        assert not os.path.exists(os.path.join(out, "ckpt_latest.bin"))
    print("NaN guard: exit code 98, the last good checkpoint is kept, a stale .tmp is removed: ok")


def check_save_streams_the_same_bytes():
    """save_checkpoint writes one array at a time now (host RAM 4 N instead of 12 N at the peak): the file is byte for byte what the concatenating version wrote."""
    c = fj.Cfg(n_layer=2, n_embd=32, n_head=2, n_kv=1, hd=16, n_ff=64, vocab=300, seq=16, relu2=True, lambdas=True, ve_layers=(0,))
    p = fj.init_params(c, 3)
    m = {n: x * 0.5 for n, x in p.items()}
    v = {n: x * x for n, x in p.items()}
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        path = os.path.join(tmp, "c.bin")
        fj.save_checkpoint(path, c, p, m, v, 7, plan_steps=100)
        n = sum(int(np.prod(s)) for _, s in fj.layout(c))
        old = (fj.header_of(c, n, 7, plan_steps=100) + "\n").encode() + b"".join(fj.flat(c, t).astype("<f4").tobytes() for t in (p, m, v))
        assert open(path, "rb").read() == old
        assert os.path.getsize(path) == len(old) and not os.path.exists(path + ".tmp")
        # a file that broke off (a copy, a download) is refused with a clear message instead of a reshape error or silently wrong moments
        cut = os.path.join(tmp, "cut.bin")
        open(cut, "wb").write(old[:-1000])
        for with_state in (True,):
            try:
                fj.load_checkpoint(cut, c, with_state)
                raise AssertionError("a truncated checkpoint was loaded")
            except ValueError as e:
                assert "incomplete" in str(e), e
        fj.load_checkpoint(cut, c, with_state=False)  # the parameters alone are all there: a fine-tune may start from it
    print("checkpoint writer is byte-identical, a truncated file is refused: ok")


def check_failed_checkpoint_write_keeps_the_run():
    """A write that fails (the disk is full) must not end the run in the middle: the half file goes, training continues, the final failure ends with exit code 96. (A directory in the place of the temporary file makes open() fail.)"""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        out, args, env = tf._tiny_run_args(tmp, ["--steps", "12", "--save-every", "5"])
        os.makedirs(os.path.join(out, "ckpt_latest.bin.tmp", "x"))
        r = run(args, env)
        assert r.returncode == 96, (r.returncode, r.stdout[-800:], r.stderr[-500:])
        assert r.stdout.count("writing ckpt_latest.bin failed") == 3 and "step    12/12" in r.stdout and "the final checkpoint could not be written" in r.stdout, r.stdout[-1200:]
    print("failed checkpoint writes: the run goes on, exit code 96 only for the final one: ok")


def check_disk_preflight():
    """12 bytes per parameter, three files at the peak: the 365M model needs 13.1 GB of the 20 GB of /kaggle/working, 600M would need 21.6 GB. The check fails at the start, not at the first save hours later."""
    assert fj.out_space_needed(365_000_000, True) == 13_140_000_000
    assert fj.out_space_needed(555_000_000, True) <= 20e9 < fj.out_space_needed(556_000_000, True)
    assert fj.out_space_needed(833_000_000, False) <= 20e9 < fj.out_space_needed(834_000_000, False)
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        out, args, env = tf._tiny_run_args(tmp, ["--steps", "5", "--time-limit-s", "1000"])
        r = run(args, env, FORGE_OUT_QUOTA_GB="0.001")
        assert r.returncode == 2 and "need up to" in r.stdout and "step     1/" not in r.stdout, (r.returncode, r.stdout[-800:])
        r = run(args + ["--no-preflight"], env, FORGE_OUT_QUOTA_GB="0.001")
        assert r.returncode == 0, r.stdout[-800:]
    print("disk preflight: refuses a run whose checkpoints cannot fit, at the start: ok")


def check_data_preflight():
    """A shard shorter than a window, token ids beyond the vocabulary and a cool-down shard too short for a window used to be found (or not found: JAX clamps the ids) inside the run."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        main = os.path.join(tmp, "main")
        os.makedirs(main)
        ok = (np.arange(50_000) % 97).astype("<u2")
        ok.tofile(os.path.join(main, "a_000.u16"))
        ok.tofile(os.path.join(main, "b_001.u16"))
        d = fj.Data(main, val_tokens=2000)
        seen, flagged = fj.validate_shards(d, 64, 512)
        assert seen > 0 and flagged == 0.0
        try:
            fj.validate_shards(d, 64, 90)  # the ids go up to 96
            raise AssertionError("ids beyond the vocabulary were not found")
        except ValueError as e:
            assert "--vocab is 90" in str(e), e
        flagged_shard = (np.arange(50_000) % 97 | np.where(np.arange(50_000) % 2 == 0, 0x8000, 0)).astype("<u2")  # chat data: bit 15 is the loss mask
        flagged_shard.tofile(os.path.join(main, "c_002.u16"))
        d2 = fj.Data(main, val_tokens=2000)
        seen, flagged = fj.validate_shards(d2, 64, 512)
        assert 0.1 < flagged < 0.4, flagged  # one shard of three is half masked
        # a cool-down pool with a shard of 30 tokens: dropped with a warning (a window of 65 tokens does not fit)
        cool = os.path.join(tmp, "cool")
        os.makedirs(cool)
        ok[:30].tofile(os.path.join(cool, "c0.u16"))  # a short shard that sorts first: positions inside it give short windows
        ok[:300].tofile(os.path.join(cool, "c1.u16"))
        pool = fj.open_cool_pool(cool, "*.u16", True, min_len=65)
        assert [f.name for f in pool.files] == ["c1.u16"]
        rng = np.random.default_rng(1)
        d.cool = pool
        for _ in range(20):
            d.train_batch(rng, 8, 64, 1.0)  # every window of the pool is whole
        # the same pool without the filter breaks np.stack inside train_batch (what the run met at the first cool-down step)
        raw = fj.CoolPool(cool, True, "*.u16")
        d.cool = raw
        try:
            for _ in range(50):
                d.train_batch(rng, 8, 64, 1.0)
            raise AssertionError("expected a ragged batch")
        except ValueError:
            pass
        # the main data: a last shard that is too short for a window
        tiny = os.path.join(tmp, "tiny")
        os.makedirs(tiny)
        ok.tofile(os.path.join(tiny, "a.u16"))
        ok[:40].tofile(os.path.join(tiny, "b.u16"))
        try:
            fj.validate_shards(fj.Data(tiny, val_tokens=100), 64, 512)
            raise AssertionError("a short shard was accepted")
        except ValueError as e:
            assert "b.u16" in str(e), e
        # the whole run: ids beyond --vocab stop it at the start
        out, args, env = tf._tiny_run_args(tmp, ["--steps", "3"])
        i = args.index("--vocab")
        args[i + 1] = "64"
        r = run(args, env)
        assert r.returncode == 2 and "--vocab is 64" in r.stdout and "step     1/" not in r.stdout, (r.returncode, r.stdout[-600:])
    print("data preflight (short shards, ids beyond the vocabulary, short cool-down shards, bit-15 share): ok")


def check_batch_geometry_warning():
    """tokens per step is rounded UP to a multiple of devices x micro x seq: another micro (the runner's out-of-memory retry) can silently change the batch. The trainer now says so."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        out, args, env = tf._tiny_run_args(tmp, ["--steps", "2", "--micro", "8"])  # 2 devices x 8 x 64 = 1024 tokens, 512 were asked for
        r = run(args, env)
        assert r.returncode == 0 and "WARNING: 512 tokens per step were asked for" in r.stdout and "trains with 1024 tokens per step" in r.stdout, r.stdout[:1500]
        out, args, env = tf._tiny_run_args(tmp + "/b", ["--steps", "2", "--micro", "4"])
        r = run(args, env)
        assert r.returncode == 0 and "WARNING: 512 tokens" not in r.stdout
    print("batch geometry: a rounded batch is reported: ok")


def check_time_limit_after_the_pin_preparation():
    """The child process of _pin_jax_for_tpu started with the whole time limit again after minutes of pip install: now the preparation time is subtracted."""
    assert fj._shrink_time_limit(["f.py", "train", "--time-limit-s", "30000", "--steps", "5"], 600.4) == ["f.py", "train", "--time-limit-s", "29400", "--steps", "5"]
    assert fj._shrink_time_limit(["--time-limit-s=100"], 40) == ["--time-limit-s=60"]
    assert fj._shrink_time_limit(["--time-limit-s", "10"], 40) == ["--time-limit-s", "1"]
    assert fj._shrink_time_limit(["train", "--steps", "5"], 40) == ["train", "--steps", "5"]
    assert fj._shrink_time_limit(["--time-limit-s", "x"], 40) == ["--time-limit-s", "x"]
    print("time limit minus the preparation: ok")


def check_resume_continuity():
    """A resume continues the learning-rate schedule and the cool-down share exactly (they are functions of the step), and the Adam step counter restarts at the checkpoint's step."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        out_a, args, env = tf._tiny_run_args(os.path.join(tmp, "a"), ["--steps", "30", "--save-every", "10", "--decay-frac", "0.4", "--min-lr-frac", "0.1"])
        assert run(args, env).returncode == 0
        full = [json.loads(l) for l in open(os.path.join(out_a, "log.jsonl"))]
        out_b, args_b, env = tf._tiny_run_args(os.path.join(tmp, "b"), ["--steps", "30", "--save-every", "10", "--decay-frac", "0.4", "--min-lr-frac", "0.1"])
        # a first session of the same 30-step plan that is killed after step 10 (the watchdog ends a step that hangs; the checkpoint of step 10 is on disk)
        r = run(args_b + ["--watchdog-s", "40"], env, FORGE_TEST_HANG_AT_STEP="11")
        assert r.returncode == 97, (r.returncode, r.stdout[-400:])
        r = run(args_b + ["--resume"], env)
        assert r.returncode == 0 and "resumed from step 10" in r.stdout, r.stdout[-600:]
        split = [json.loads(l) for l in open(os.path.join(out_b, "log.jsonl"))]
        assert [(x["step"], x["lr"]) for x in split] == [(x["step"], x["lr"]) for x in full]
        c = fj.cfg_from_checkpoint(os.path.join(out_b, "ckpt_latest.bin"))
        assert fj.load_checkpoint(os.path.join(out_b, "ckpt_latest.bin"), c)[3] == 30
    print("resume: the step numbers and learning rates of a two-session run equal the uninterrupted run: ok")


def _load_runner(tmp, tag):
    """cloud/kaggle_tpu_train.py as a module with its input and output folders in tmp (the paths are read at import) and a fake accelerator probe (the CPU of this machine counts as a GPU, so that no pip install is tried)."""
    import importlib.util
    import shutil

    root = os.path.join(tmp, tag)
    inp, outp = os.path.join(root, "in"), os.path.join(root, "out")
    os.makedirs(os.path.join(inp, "forge-jax"))
    os.makedirs(os.path.join(inp, "forge-tpujob"))
    os.makedirs(outp)
    shutil.copy(SCRIPT, os.path.join(inp, "forge-jax", "forge_jax.py"))
    old = {k: os.environ.get(k) for k in ("FORGE_KAGGLE_IN", "FORGE_KAGGLE_OUT")}
    os.environ["FORGE_KAGGLE_IN"], os.environ["FORGE_KAGGLE_OUT"] = inp, outp
    try:
        spec = importlib.util.spec_from_file_location("kaggle_tpu_train_" + tag, os.path.join(HERE, "..", "cloud", "kaggle_tpu_train.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    mod.backend = lambda env=None: "gpu 2 fake (jax 0)"
    return mod, inp, outp


def check_runner_guards():
    """The Kaggle runner: a checkpoint of the previous session that is broken or has no moments stops the session at once (it used to be copied and fail in the trainer, or a retry started from nothing); a trainer that stops at
    its time limit before the end of the decay (exit code 0) is marked in exit.txt."""
    import shlex
    import shutil

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        out, args, env = tf._tiny_run_args(os.path.join(tmp, "x"), ["--steps", "12", "--save-every", "6"])
        assert run(args, env).returncode == 0
        good = os.path.join(out, "ckpt_latest.bin")
        mod0, _, _ = _load_runner(tmp, "probe")
        assert mod0.ckpt_complete(__import__("pathlib").Path(good))
        raw = open(good, "rb").read()
        cut = os.path.join(tmp, "cut.bin")
        open(cut, "wb").write(raw[:-5000])
        params_only = os.path.join(tmp, "po.bin")
        line = raw.split(b"\n", 1)[0] + b"\n"
        n = json.loads(line)["n_params"]
        open(params_only, "wb").write(raw[: len(line) + 4 * n])
        P = __import__("pathlib").Path
        assert not mod0.ckpt_complete(P(cut)) and not mod0.ckpt_complete(P(params_only)) and not mod0.ckpt_complete(P(os.path.join(tmp, "missing.bin")))
        assert mod0.decay_unfinished(["x\n", "[train] WARNING: stopped at step 5 of 9 planned: the learning-rate decay is NOT finished (learning rate 1e-3)\n"]) and not mod0.decay_unfinished(["step 9/9\n"])
        rest, keep, k = args[5:], [], 0
        while k < len(rest):  # the runner adds --data, --out and --micro itself
            if rest[k] == "--micro":
                k += 2
                continue
            keep.append(rest[k])
            k += 1
        base_args = shlex.join(keep)
        data_dir = os.path.join(tmp, "x", "data")
        for tag, ckpt, steps_flags in (("broken", cut, ""), ("unfinished", None, "--no-auto-steps --steps 1000000 --time-limit-s 25")):
            mod, inp, outp = _load_runner(tmp, tag)
            shutil.copytree(data_dir, os.path.join(inp, "forge-tokens"))
            settings = {"args": base_args + " " + steps_flags, "data": "forge-tokens", "micro": [2], "jax_pin": "", "require_accelerator": False}
            json.dump(settings, open(os.path.join(inp, "forge-tpujob", "settings.json"), "w"))
            if ckpt:
                os.makedirs(os.path.join(inp, "forge-ckpt-tpu"))
                shutil.copy(ckpt, os.path.join(inp, "forge-ckpt-tpu", "ckpt_latest.bin"))
            mod.main()
            exit_txt = open(os.path.join(outp, "exit.txt")).read()
            if tag == "broken":
                assert exit_txt.strip() == "bad checkpoint" and not os.path.exists(os.path.join(outp, "train.log")), exit_txt
            else:
                assert exit_txt.splitlines() == ["0", "decay unfinished"], (exit_txt, open(os.path.join(outp, "train.log")).read()[-800:])
    print("runner: a broken or moment-less checkpoint stops the session, an unfinished decay is marked in exit.txt: ok")


if __name__ == "__main__":
    check_runner_guards()
    check_save_streams_the_same_bytes()
    check_time_limit_after_the_pin_preparation()
    check_disk_preflight()
    check_data_preflight()
    check_batch_geometry_warning()
    check_failed_checkpoint_write_keeps_the_run()
    check_nan_stops_without_overwriting_the_checkpoint()
    check_resume_continuity()
    print("all premortem checks passed")
