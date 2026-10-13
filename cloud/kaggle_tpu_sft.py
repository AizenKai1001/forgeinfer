"""Kaggle kernel: chat fine-tunes of a finished Forge-1 base with the JAX trainer (tpu/forge_jax.py) on a TPU, several runs one after the other in ONE session (a learning-rate grid costs one session start, not one per run).

Inputs (datasets of the account YOUR_KAGGLE_USER): forge-jax (forge_jax.py), forge-tpujob-sft (settings.json), the shard dataset named by "data" (default forge-sft365: the .u16 chat shards, bit 15 = not a target, the LAST file by name holds the
validation tail), and the dataset of the base model named by the "init" of every run ("dataset/file", e.g. "forge-base365/ckpt_latest.bin": the weights are enough, see tools/ckpt_params_only.py).
settings.json:
  {"data": "forge-sft365", "micro": [4, 2, 1], "require_accelerator": true,
   "runs": [{"name": "lr15", "init": "forge-base365/ckpt_latest.bin", "max_s": 5400, "args": "--layers 24 --dim 1024 ... --steps 2300 --lr 1.5e-4 --muon-lr 1.5e-4 --no-auto-steps --save-every 100000"}, ...]}
Every run writes to /kaggle/working/<name>/ (log.jsonl, train.log lines go to /kaggle/working/train.log): after a successful run the pre-decay checkpoint is removed and ckpt_latest.bin is cut down to the weights (header + parameters, 4 bytes per
parameter instead of 12), so that three 365M runs are 4.4 GB instead of 13 GB. summary.json lists the runs (exit code, minutes, last validation loss, checkpoint size); exit.txt is 0 when every run finished.
A run that does not fit in memory is retried with the next smaller micro-batch (the sequences per device; 4 x 8 devices x 2,048 tokens = 65,536 tokens per step), a crash that is not an out-of-memory error once with XLA's attention. Runs that no longer fit
into the session ("session_s", default 9 h, minus "end_margin_s", default 25 minutes) are skipped. Optional settings as in kaggle_tpu_train.py: "silence_s", "print_every", "jax_pin".
"""
import json
import os
import queue
import re
import shlex
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


def cut_params_only(path: Path) -> int:
    """Rewrites a forgeckpt1 checkpoint as its header line plus the parameters (the two Adam moments are dropped: a fine-tuned model is evaluated and served, not resumed). Returns the new size in bytes."""
    tmp = path.with_suffix(".part")
    with open(path, "rb") as f, open(tmp, "wb") as o:
        line = f.readline()
        n = int(json.loads(line)["n_params"])
        o.write(line)
        left = n * 4
        while left:
            b = f.read(min(left, 64 << 20))
            if not b:
                raise RuntimeError(f"{path} is shorter than its header says")
            o.write(b)
            left -= len(b)
    os.replace(tmp, path)
    return path.stat().st_size


def last_val(run: Path):
    """(step, validation loss) of the last evaluation in the run's log.jsonl, or None."""
    best = None
    try:
        with open(run / "log.jsonl") as f:
            for line in f:
                r = json.loads(line)
                if "val_loss" in r:
                    best = (r["step"], round(r["val_loss"], 4))
    except (OSError, ValueError):
        pass
    return best


def main() -> None:
    settings = json.loads((find("forge-tpujob-sft") / "settings.json").read_text())
    code = find("forge-jax") / "forge_jax.py"
    data = find(settings.get("data", "forge-sft365"))
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
    pin = str(settings.get("jax_pin", "0.11.2"))  # "" = use the preinstalled JAX
    if pin and be.startswith("tpu") and f"(jax {pin})" not in be:
        env = pinned_jax(pin)
        if env is None:
            say(f"the pinned jax {pin} is not usable here: the preinstalled JAX is used")
    session_s = float(settings.get("session_s", 9 * 3600))
    end_margin_s = float(settings.get("end_margin_s", 1500))
    silence_s = float(settings.get("silence_s", 1500))
    print_every = max(1, int(settings.get("print_every", 10)))
    results = []
    with open(OUT / "train.log", "a") as log:
        for r in settings["runs"]:
            name = r["name"]
            left = session_s - (time.time() - T0) - end_margin_s
            if left < 900:
                say(f"run {name}: only {left:.0f} s of the session left: skipped")
                results.append({"name": name, "skipped": True})
                continue
            ds, _, fn = r["init"].partition("/")
            init = find(ds) / (fn or "ckpt_latest.bin")
            run = OUT / name
            run.mkdir(exist_ok=True)
            base_args = shlex.split(r["args"])
            attn_forced = "--attn" in base_args
            attempts = [(m, None) for m in list(r.get("micro", settings.get("micro", [4, 2, 1])))]  # (micro, attention override)
            xla_retry_used = False
            rc, tail = 1, []
            t_run = time.time()
            while attempts:
                micro, attn = attempts.pop(0)
                left = session_s - (time.time() - T0) - end_margin_s
                if left < 600:
                    say(f"run {name}: only {left:.0f} s of the session left: no further attempt")
                    break
                limit = min(float(r.get("max_s", left)), left)
                for stale in ("ckpt_latest.bin", "ckpt_predecay.bin", "log.jsonl"):  # a failed attempt must not leave a half run behind
                    if (run / stale).exists():
                        (run / stale).unlink()
                cmd = [sys.executable, "-u", str(code), "train", "--data", str(data), "--out", str(run), "--micro", str(micro), "--init-from", str(init)] + base_args + ["--time-limit-s", f"{limit:.0f}"]
                cmd += ["--attn", attn] if attn else []
                say(f"run {name}: " + " ".join(cmd))
                t_att = time.time()
                rc, oom, silent, tail = run_trainer(cmd, log, silence_s, print_every, env)
                say(f"run {name}: trainer exit code {rc} after {(time.time() - t_att) / 60:.1f} min" + (" (out of memory)" if oom else "") + (" (killed: no output)" if silent else ""))
                if rc == 0:
                    break
                if oom and not silent:
                    say(f"run {name}: micro-batch {micro} does not fit in memory: trying a smaller one")
                    continue
                if not attn_forced and not xla_retry_used:
                    xla_retry_used = True
                    attempts.insert(0, (micro, "xla"))
                    say(f"run {name}: the trainer failed (not out of memory): once more with --attn xla")
                    continue
                break
            size = None
            ck = run / "ckpt_latest.bin"
            if rc == 0 and ck.exists():
                pre = run / "ckpt_predecay.bin"
                if pre.exists():
                    pre.unlink()
                size = cut_params_only(ck)
            unfinished = rc == 0 and any("the learning-rate decay is NOT finished" in l for l in tail)  # stopped at max_s before the end of the schedule: exit code 0, but the fine-tune is not annealed
            if unfinished:
                say(f"run {name}: WARNING: stopped at its time limit (max_s) before the end of the learning-rate decay: not a finished fine-tune")
            results.append({"name": name, "rc": rc, "minutes": round((time.time() - t_run) / 60, 1), "last_val": last_val(run), "ckpt_bytes": size, "decay_unfinished": unfinished})
            say(f"run {name}: " + json.dumps(results[-1]))
            (OUT / "summary.json").write_text(json.dumps(results, indent=1))
    ok = bool(results) and all(x.get("rc") == 0 and not x.get("decay_unfinished") for x in results)
    say(f"all runs done after {(time.time() - T0) / 60:.1f} min: " + ("every run finished" if ok else "NOT every run finished (see summary.json)"))
    (OUT / "exit.txt").write_text("0\n" if ok else "1\n")


if __name__ == "__main__":
    main()
