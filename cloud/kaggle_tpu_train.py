"""Kaggle kernel: train Forge-1 with the JAX trainer (tpu/forge_jax.py) on a TPU (or a GPU), started from the Kaggle web page with the accelerator set to TPU.

Inputs (datasets of the account YOUR_KAGGLE_USER): forge-jax (forge_jax.py), forge-tokens (the token shards), forge-tpujob (settings.json), optionally forge-ckpt-tpu (ckpt_latest.bin + log.jsonl of the previous session of a chain).
settings.json:  {"args": "--layers 30 --dim 576 ... --steps 45000 --time-limit-s 28000", "data": "forge-tokens", "micro": [4, 2, 1], "require_accelerator": true}
The trainer runs with the first micro-batch size (sequences per device) that fits in memory. Outputs in /kaggle/working: run/ (log.jsonl, ckpt_latest.bin), train.log, env.txt, exit.txt.
Optional settings: "session_s" (the session limit, default 32400 = 9 h), "end_margin_s" (time kept at the end of the session, default 1500), "silence_s" (kill the trainer after this many seconds
without output, default 1500), "print_every" (step lines shown in the notebook: every n-th, default 10; train.log keeps all of them), "jax_pin" (a jax version installed into /tmp and used when it
sees the TPU, default "0.11.2", "" = the preinstalled JAX).
"code_dataset" (default "forge-jax"): the input dataset that holds forge_jax.py (and, for the autotune, bench_step.py); "@ds:NAME" inside "args" is replaced by the folder of the input dataset NAME.
"autotune" (off when absent): {"candidates": [{"name": "muon-shard", "flags": "--muon-shard", "env": {}}, ...], "budget_s": 1200, "per_variant_s": 420, "steps": 6, "min_gain_pct": 2.0, "on_resume": false, "code_dataset": "forge-jax-code"}:
before the training, a short benchmark of the candidates against `base` (tpu/bench_step.py, which must sit next to a byte-identical copy of the trainer: the folder of forge_jax.py or the code dataset) on the session's own TPU; the
fastest candidate that computes the same function (same first-step loss) and is at least min_gain_pct faster adds its flags (and environment, e.g. LIBTPU_INIT_ARGS) to the trainer's; anything that fails leaves the settings' own
flags. It saves a separate benchmark session (a second wait in Kaggle's queue). Candidates must never change the model (window, head size, batch): that is the settings' decision. Outputs: autotune/ (bench.json, decision.json).
The splash kernel that the benchmark's base variant chose (`attn_pick` of its row in bench.json, e.g. "512,1") is handed to the trainer as `--attn-pick 512,1`, which skips the trainer's own repeat of that selection (about 2 minutes
of the session); not when the settings give --attn or --attn-pick themselves. When the jax version is settled in the runner (the pin installed and checked, or no pin needed or wanted) the trainer runs with FORGE_NO_PIN=1: its own
pin logic would only probe jax once more (17 s).
"""
import json
import os
import queue
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

IN = Path(os.environ.get("FORGE_KAGGLE_IN", "/kaggle/input"))
OUT = Path(os.environ.get("FORGE_KAGGLE_OUT", "/kaggle/working"))
T0 = time.time()


def find(slug: str) -> Path:
    for cand in (IN / slug, *IN.glob(f"*/{slug}"), *IN.glob(f"*/*/{slug}"), *IN.glob(f"datasets/*/{slug}")):
        if cand.is_dir():
            return cand
    raise SystemExit(f"dataset {slug} not found under {IN}: {[str(p) for p in IN.glob('*')]}")


def find_opt(slug: str):
    try:
        return find(slug)
    except SystemExit:
        return None


def sh(cmd: str, timeout: float = 600, env=None) -> str:
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout, env=env).stdout.strip()
    except subprocess.TimeoutExpired:
        return f"(timed out after {timeout:.0f} s: {cmd[:80]})"


def backend(env=None) -> str:
    """'tpu 8 TPU v5 lite (jax 0.11.2)' etc. The marker keeps log lines that the TPU runtime may print around the probe (stderr is merged) from being taken for the answer; the timeout keeps a stuck
    TPU initialisation from holding the session."""
    out = sh(f"{sys.executable} -c \"import jax; d = jax.devices(); print('JAXPROBE', jax.__version__, jax.default_backend(), len(d), d[0].device_kind)\" 2>&1", timeout=600, env=env)
    lines = [l.split("JAXPROBE", 1)[1].strip() for l in out.splitlines() if "JAXPROBE" in l]
    if lines:
        ver, rest = lines[-1].split(" ", 1)
        return rest + f" (jax {ver})"
    return "none: " + " | ".join(out.splitlines()[-3:])


def pinned_jax(version: str):
    """Installs jax[tpu]==version (with the libtpu it pins) into its own directory and returns an environment that uses it, or None when the install fails or does not see an accelerator.
    Why: Kaggle's TPU image force-installs libtpu 0.0.17 (built 2025-06-12) next to its JAX, and the Pallas of jax 0.7.2 refuses a libtpu built before 2025-08-01, so the splash kernel cannot run there;
    the trainer was tested with jax 0.11.2 (CPU tests, and a deviceless TPU v5e compile with libtpu 0.0.48). The system packages stay untouched."""
    target = Path("/tmp/forge_jax_" + version.replace(".", "_"))
    say(f"installing jax[tpu]=={version} into {target} ...")
    say(sh(f"{sys.executable} -m pip install -q --target {target} 'jax[tpu]=={version}' 2>&1 | tail -n 3", timeout=1200))
    env = dict(os.environ, PYTHONPATH=str(target) + (os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else ""))
    be = backend(env)
    say(f"backend with the pinned jax: {be}")
    return env if be.startswith(("tpu", "gpu", "cuda")) else None


def say(msg: str) -> None:
    print(msg, flush=True)
    with open(OUT / "env.txt", "a") as f:
        f.write(msg + "\n")


OOM = re.compile(r"RESOURCE_EXHAUSTED|out of memory|Ran out of memory", re.IGNORECASE)


def ckpt_complete(path: Path) -> bool:
    """True when the file has the size its own header promises for a checkpoint with both Adam moments (header line + 12 bytes per parameter): a copy that broke off, or a weights-only file, cannot be resumed."""
    try:
        with open(path, "rb") as f:
            line = f.readline()
        return path.stat().st_size == len(line) + 12 * int(json.loads(line)["n_params"])
    except (OSError, ValueError, KeyError, TypeError):
        return False


def decay_unfinished(tail) -> bool:
    """True when the trainer's last lines say that it stopped at the time limit before the end of its learning-rate decay (exit code 0, but the model is not annealed: the session must be continued with --resume)."""
    return any("the learning-rate decay is NOT finished" in l for l in tail)


def choose_config(rows, min_gain_pct=2.0, first_loss_tol=2e-3, final_loss_tol=0.02):
    """The autotune's decision from the table of bench_step.py (a list of dicts, `base` among them): the fastest candidate that ran, computes the same function as base and is at least `min_gain_pct` percent faster.
    The same function = the same tokens per step, the loss of the very first step (initial parameters, same data: a difference means a different forward pass) within `first_loss_tol` and the loss after the timed steps
    within `final_loss_tol` of base's. Returns {"name", "flags", "env", "gain_pct", "reason"}; without a valid winner (base not measured, nothing faster, nothing equal) name is "base" with no flags."""
    keep = {"name": "base", "flags": "", "env": {}, "gain_pct": 0.0}
    base = next((r for r in rows if r.get("variant") == "base" and r.get("status") == "ok" and r.get("tokens_per_s")), None)
    if base is None:
        return dict(keep, reason="the base variant was not measured")
    best, why = None, []
    for r in rows:
        name = r.get("variant")
        if r is base or name == "base":
            continue
        if r.get("status") != "ok" or not r.get("tokens_per_s"):
            why.append(f"{name}: {r.get('status')}")
            continue
        gain = (r["tokens_per_s"] / base["tokens_per_s"] - 1.0) * 100.0
        if r.get("tokens_per_step") != base.get("tokens_per_step"):
            why.append(f"{name}: a different batch")
        elif base.get("first_loss") is not None and (r.get("first_loss") is None or abs(r["first_loss"] - base["first_loss"]) > first_loss_tol):
            why.append(f"{name}: the first-step loss differs ({r.get('first_loss')} against {base['first_loss']})")
        elif r.get("final_loss") is None or base.get("final_loss") is None or abs(r["final_loss"] - base["final_loss"]) > final_loss_tol:
            why.append(f"{name}: the loss after the timed steps differs ({r.get('final_loss')} against {base.get('final_loss')})")
        elif gain < min_gain_pct:
            why.append(f"{name}: only {gain:+.1f} percent")
        elif best is None or gain > best[0] + 1e-9 or (abs(gain - best[0]) <= 1e-9 and len(r.get("extra_flags", "")) < len(best[1].get("extra_flags", ""))):
            best = (gain, r)
    if best is None:
        return dict(keep, reason="no candidate is both equal and faster: " + "; ".join(why))
    gain, r = best
    return {"name": r["variant"], "flags": r.get("extra_flags", ""), "env": dict(r.get("env") or {}), "gain_pct": round(gain, 1), "reason": "fastest valid candidate" + (("; rejected: " + "; ".join(why)) if why else "")}


def attn_pick_flag(rows, base_args):
    """'--attn-pick 512,1' from the `attn_pick` field of the base row of the benchmark table (the splash kernel that the benchmark's own selection chose on this TPU a few minutes ago: the trainer then skips its repeat of that
    selection, 128 s of the 365M session), or '' when there is none: the base row failed or fell back to XLA's attention (the value 'xla'), the field is not of the form BLOCK,FUSED, or the settings choose the attention
    themselves (--attn, --attn-pick)."""
    base = next((r for r in rows if r.get("variant") == "base" and r.get("status") == "ok"), None)
    spec = str((base or {}).get("attn_pick") or "")
    if not re.fullmatch(r"\d+,[01]", spec) or "--attn" in base_args or "--attn-pick" in base_args:
        return ""
    return f"--attn-pick {spec}"


def run_autotune(auto, code: Path, base_args, micro: int, env=None):
    """The in-session autotune: a short benchmark (tpu/bench_step.py, one worker process per candidate, so that an out-of-memory error or a flag libtpu refuses costs only that candidate) of the equivalence-preserving
    candidates of settings["autotune"] against `base` with the run's own model arguments, on the same TPU, then choose_config. It replaces a separate benchmark session (a second wait in Kaggle's queue). The benchmark code
    must be the trainer's own file: bench_step.py is taken from the folder of forge_jax.py or from the dataset "code_dataset" (default forge-jax-code) and only used when that folder's forge_jax.py is byte-identical to the
    trainer's. Candidates must not change the model (no window, no head size, no batch): that is the settings' choice, never an autotune's.
    auto: {"candidates": [{"name", "flags", "env"}], "budget_s": 1200, "per_variant_s": 420, "steps": 6, "min_gain_pct": 2.0, "tiny": false}.
    Returns (flags, env_extra, note); ("", {}, note) = keep the settings' own flags."""
    import hashlib

    def digest(p: Path) -> str:
        return hashlib.sha256(p.read_bytes()).hexdigest()

    pair = None
    for d in (code.parent, find_opt(str(auto.get("code_dataset", "forge-jax-code")))):
        if d is not None and (d / "bench_step.py").exists() and (d / "forge_jax.py").exists():
            if digest(d / "forge_jax.py") == digest(code):
                pair = d
                break
    if pair is None:
        return "", {}, "no bench_step.py next to a copy of the trainer that is byte-identical to it: autotune skipped"
    work = OUT / "autotune"
    (work / "code").mkdir(parents=True, exist_ok=True)
    for name in ("forge_jax.py", "bench_step.py"):
        shutil.copy(pair / name, work / "code" / name)
    cands = auto.get("candidates") or [{"name": "muon-shard", "flags": "--muon-shard"}]
    (work / "variants.json").write_text(json.dumps([{"name": "base", "flags": ""}] + [c for c in cands if c.get("name") != "base"]))
    budget = float(auto.get("budget_s", 1200))
    cmd = [sys.executable, "-u", str(work / "code" / "bench_step.py"), "--out", str(work), "--variants-json", str(work / "variants.json"), "--deadline-s", f"{budget:.0f}",
           "--per-variant-s", str(auto.get("per_variant_s", 420)), "--steps", str(int(auto.get("steps", 6))), "--warmup", "2"]
    if auto.get("tiny"):
        cmd += ["--tiny"]
    else:
        cmd += ["--args=" + shlex.join(list(base_args) + ["--micro", str(micro)])]
    say("autotune: " + " ".join(shlex.quote(c) for c in cmd))
    t0 = time.time()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=budget + 180, env=env)
        (work / "bench.log").write_text((r.stdout or "") + "\n" + (r.stderr or ""))
        say(f"autotune: benchmark exit code {r.returncode} after {(time.time() - t0) / 60:.1f} min")
    except subprocess.TimeoutExpired:
        say(f"autotune: the benchmark did not end within {budget + 180:.0f} s: stopped, the settings' own flags are used")
        return "", {}, "benchmark timeout"
    try:
        rows = json.loads((work / "bench.json").read_text())["table"]
    except (OSError, ValueError, KeyError):
        return "", {}, "no bench.json: the settings' own flags are used"
    say("autotune table: " + ", ".join(f"{x['variant']}={x['status']}" + (f" {x['tokens_per_s']:.0f} tok/s" if x.get("status") == "ok" else "") for x in rows))
    pick = choose_config(rows, float(auto.get("min_gain_pct", 2.0)))
    (work / "decision.json").write_text(json.dumps(pick, indent=1))
    say(f"autotune: {pick['name']} (gain {pick['gain_pct']:+.1f} percent): {pick['reason']}")
    flags = pick["flags"]
    attn = attn_pick_flag(rows, base_args)
    if attn:
        say(f"autotune: the benchmark's attention pick goes to the trainer ({attn}): its own self-test of the splash kernels (about 2 minutes) is skipped")
        flags = (flags + " " + attn).strip()
    return flags, pick["env"], pick["name"]


def run_trainer(cmd, log, silence_s: float, print_every: int, env=None):
    """Runs one trainer process; returns (exit code, saw an out-of-memory message, killed for silence, last lines). Every line goes to train.log; the notebook shows every `print_every`-th step line
    and all other lines. The whole output is scanned for out-of-memory messages (an XLA message can be longer than any fixed tail)."""
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)
    lines: "queue.Queue" = queue.Queue()

    def reader():
        for line in p.stdout:
            lines.put(line)
        lines.put(None)

    threading.Thread(target=reader, daemon=True).start()
    tail, oom, silent = [], False, False
    step_re = re.compile(r"^step\s+(\d+)/")
    while True:
        try:
            line = lines.get(timeout=silence_s)
        except queue.Empty:
            say(f"no output from the trainer for {silence_s:.0f} s: stopping it (a hang would otherwise hold the session until Kaggle ends it)")
            silent = True
            p.kill()
            break
        if line is None:
            break
        log.write(line)
        log.flush()
        m = step_re.match(line)
        if m is None or int(m.group(1)) % print_every == 0 or " val " in line:
            print(line, end="", flush=True)
        if OOM.search(line):
            oom = True
        tail.append(line)
        tail = tail[-60:]
    rc = p.wait()
    return rc, oom, silent, tail


def main() -> None:
    settings = json.loads((find("forge-tpujob") / "settings.json").read_text())
    code = find(str(settings.get("code_dataset", "forge-jax"))) / "forge_jax.py"  # "code_dataset": a small dataset with the trainer (forge-jax-code), so that a code change does not re-upload the 1.2 GB of cool-down shards of forge-jax
    data = find(settings.get("data", "forge-tokens"))
    say(f"python {sys.version.split()[0]}, {sh('nproc')} cpus, " + sh("free -g | sed -n 2p"))
    say("devices: " + sh("ls /dev/accel* /dev/vfio 2>&1 | head -n 5"))
    be = backend()
    say("backend: " + be)
    say(sh("pip list 2>/dev/null | grep -i -E '^(jax|jaxlib|libtpu|numpy|ml.dtypes) ' | tr '\\n' ';'"))
    if not be.startswith(("tpu", "gpu", "cuda")):
        say("no TPU or GPU backend: trying jax[tpu] ...")
        say(sh("pip install -q -U 'jax[tpu]' -f https://storage.googleapis.com/jax-releases/libtpu_releases.html 2>&1 | tail -n 3", timeout=1200))
        be = backend()
        say("backend now: " + be)
        if settings.get("require_accelerator", True) and not be.startswith(("tpu", "gpu", "cuda")):
            say("still no accelerator: stopping at once so that no time is wasted (set the accelerator of this kernel to TPU in the editor and run it again)")
            (OUT / "exit.txt").write_text("no accelerator\n")
            return
    env = None
    pin_failed = False
    pin = str(settings.get("jax_pin", "0.11.2"))  # "" = use the preinstalled JAX
    if pin and be.startswith("tpu") and f"(jax {pin})" not in be:
        env = pinned_jax(pin)
        if env is None:
            pin_failed = True
            say(f"the pinned jax {pin} is not usable here: the preinstalled JAX is used")
    run = OUT / "run"
    run.mkdir(exist_ok=True)
    resume = False
    ck = find_opt("forge-ckpt-tpu")
    if ck is not None and (ck / "ckpt_latest.bin").exists() and (ck / "ckpt_latest.bin").stat().st_size > 1000 and not ckpt_complete(ck / "ckpt_latest.bin"):
        # starting from scratch instead would burn the whole session on a run that is already hours old, and a broken copy would only fail in the trainer: stop at once
        say(f"the checkpoint of the previous session ({ck / 'ckpt_latest.bin'}) is incomplete or has no optimizer state: stopping at once (upload it again, with the moments)")
        (OUT / "exit.txt").write_text("bad checkpoint\n")
        return
    if ck is not None and (ck / "ckpt_latest.bin").exists() and (ck / "ckpt_latest.bin").stat().st_size > 1000:
        shutil.copy(ck / "ckpt_latest.bin", run / "ckpt_latest.bin")
        if (ck / "log.jsonl").exists():
            shutil.copy(ck / "log.jsonl", run / "log.jsonl")
        resume = True
        say("resuming from the checkpoint of the previous session")
    session_s = float(settings.get("session_s", 9 * 3600))
    end_margin_s = float(settings.get("end_margin_s", 1500))
    silence_s = float(settings.get("silence_s", 1500))
    print_every = max(1, int(settings.get("print_every", 10)))
    base_args = [str(find(a_[4:])) if a_.startswith("@ds:") else a_ for a_ in shlex.split(settings["args"])]  # "@ds:NAME" in the arguments = the folder of the input dataset NAME (e.g. --cool-data @ds:forge-jax when the trainer comes from another dataset)
    limit_given = None
    for i, t in enumerate(base_args):  # the last --time-limit-s wins (argparse)
        if t == "--time-limit-s" and i + 1 < len(base_args):
            limit_given = float(base_args[i + 1])
    attn_forced = "--attn" in base_args
    micros = list(settings.get("micro", [4, 2, 1]))
    auto = settings.get("autotune")
    if auto and (be.startswith("tpu") or auto.get("tiny")) and (not resume or auto.get("on_resume")):
        try:  # a failed autotune must never cost the run: the settings' own flags stay
            flags, env_extra, note = run_autotune(auto, code, base_args, micros[0], env)
            if flags:
                base_args += shlex.split(flags)
            if env_extra:
                env = dict(os.environ if env is None else env)
                for k, v in env_extra.items():
                    env[k] = (env[k] + " " + v) if k == "LIBTPU_INIT_ARGS" and env.get(k) else v
            say(f"autotune result: {note}" + (f"; trainer flags added: {flags}" if flags else "") + (f"; environment: {env_extra}" if env_extra else ""))
        except Exception as e:  # noqa: BLE001
            say(f"autotune failed ({type(e).__name__}: {e}): the settings' own flags are used")
    if not pin_failed:
        # the jax version is settled here (pinned and checked above, or already the right one, or deliberately the preinstalled one): the trainer's own pin logic would only probe jax once more (17 s of the 365M session).
        # After a pin that failed here it stays on: its second try (pip install again) is the safety net
        env = dict(os.environ if env is None else env, FORGE_NO_PIN="1")
    attempts = [(m, None) for m in micros]  # (micro, attention override)
    rc, xla_retry_used, tail = 1, False, []
    with open(OUT / "train.log", "a") as log:
        while attempts:
            micro, attn = attempts.pop(0)
            # the trainer counts its time limit from its own start: give it what is left of the session (minus the trainer's start-up, the final checkpoint and Kaggle's saving of the outputs)
            left = session_s - (time.time() - T0) - end_margin_s
            if left < 1200:
                say(f"only {left:.0f} s of the session left: no further attempt")
                break
            limit = min(limit_given, left) if limit_given else left
            cmd = [sys.executable, "-u", str(code), "train", "--data", str(data), "--out", str(run), "--micro", str(micro)] + base_args + ["--time-limit-s", f"{limit:.0f}"]
            cmd += (["--attn", attn] if attn else []) + (["--resume"] if resume or (run / "ckpt_latest.bin").exists() else [])
            if not (run / "ckpt_latest.bin").exists() and (run / "log.jsonl").exists():
                (run / "log.jsonl").unlink()  # a retry that starts from step 0 must not append to the records of the failed attempt
            say("running: " + " ".join(cmd))
            t_att = time.time()
            rc, oom, silent, tail = run_trainer(cmd, log, silence_s, print_every, env)
            say(f"trainer exit code {rc} after {(time.time() - t_att) / 60:.1f} min" + (" (out of memory)" if oom else "") + (" (killed: no output)" if silent else ""))
            if rc == 0:
                break
            if oom and not silent and time.time() - T0 < 3600:
                say(f"micro-batch {micro} does not fit in memory: trying a smaller one")
                continue
            if not attn_forced and not xla_retry_used and time.time() - T0 < 7200:
                # a failure that the trainer's own fallbacks did not catch (a crash in the Pallas/Mosaic compiler, a hang): once more with XLA's attention, same micro-batch
                # (within the first 2 hours only: a resume then lies before the learning-rate decay, whose position the trainer does not store in the checkpoint)
                xla_retry_used = True
                attempts.insert(0, (micro, "xla"))
                say("the trainer failed (not out of memory): once more with --attn xla")
                continue
            break
    say(f"exit code {rc} after {(time.time() - T0) / 60:.1f} min")
    for stale in run.glob("ckpt_*.tmp"):  # the half-written file of a trainer that was killed during a checkpoint write (4.4 GB for the 365M model) must not be uploaded as output
        try:
            stale.unlink()
        except OSError:
            pass
    unfinished = rc == 0 and decay_unfinished(tail)
    if unfinished:
        say("WARNING: the trainer stopped at its time limit BEFORE the end of the learning-rate decay: exit code 0 does not mean the run is finished; continue with --resume in the next session")
    (OUT / "exit.txt").write_text(f"{rc}\n" + ("decay unfinished\n" if unfinished else ""))


if __name__ == "__main__":
    main()
