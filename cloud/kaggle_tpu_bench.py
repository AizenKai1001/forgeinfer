"""Kaggle kernel: ONE short TPU session that benchmarks the JAX training step (tpu/bench_step.py) to find out which speed options pay on the free v5e-8. Started from the Kaggle web page with the accelerator set to TPU,
like the training kernel. No training data are needed.

Inputs (datasets of the account YOUR_KAGGLE_USER): forge-jax-code (small: forge_jax.py and bench_step.py in its root) or else forge-jax with both files, or the dataset named by "code_dataset" (so that the 1.2 GB of cool-down shards
in forge-jax need not be uploaded again for every code change); optionally forge-tpujob-bench with a settings.json (every key optional):
  {"code_dataset": "forge-jax", "args": "<model arguments, default: the 365M model of m365_ve8.json>", "only": "base,muon-shard", "total_s": 1500, "per_variant_s": 420, "silence_s": 600, "jax_pin": "0.11.2", "require_accelerator": true}
The whole run (the JAX installation included) is held under "total_s" (default 1500 = 25 minutes): bench_step.py gets what is left of it as its deadline and starts no variant it cannot finish; a worker that has not reported after
"per_variant_s" is killed. Outputs in /kaggle/working: bench.json (the table, written after every variant), bench.log (everything the benchmark printed), env.txt (the environment and the decisions), exit.txt (0 = the base variant
was measured, 1 = it was not, "no accelerator" / "too little time" = nothing ran). The trainer's own JAX pinning is repeated here (kaggle_tpu_train.py explains why): the workers run with the pinned JAX.
Test switches (CPU, no network): FORGE_KAGGLE_IN / FORGE_KAGGLE_OUT (directories), FORGE_BENCH_TINY=1 (a tiny model on 8 fake CPU devices), FORGE_KAGGLE_NO_PIP=1 (never call pip).
"""
import json
import os
import queue
import shlex
import subprocess
import sys
import threading
import time
from pathlib import Path

IN = Path(os.environ.get("FORGE_KAGGLE_IN", "/kaggle/input"))
OUT = Path(os.environ.get("FORGE_KAGGLE_OUT", "/kaggle/working"))
NO_PIP = bool(os.environ.get("FORGE_KAGGLE_NO_PIP"))
TINY = bool(os.environ.get("FORGE_BENCH_TINY"))
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


def say(msg: str) -> None:
    print(msg, flush=True)
    with open(OUT / "env.txt", "a") as f:
        f.write(msg + "\n")


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


def run_bench(cmd, log, silence_s: float, hard_s: float, env=None):
    """Runs the benchmark process; every line goes to bench.log and to the notebook. Returns (exit code, killed for silence, killed at the hard limit). `silence_s` without any output, or `hard_s` in all, kills it
    (a hung TPU runtime would otherwise hold the session until Kaggle ends it)."""
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)
    lines: "queue.Queue" = queue.Queue()

    def reader():
        for line in p.stdout:
            lines.put(line)
        lines.put(None)

    threading.Thread(target=reader, daemon=True).start()
    t0 = time.time()
    silent = late = False
    while True:
        try:
            line = lines.get(timeout=max(1.0, min(silence_s, hard_s - (time.time() - t0))))
        except queue.Empty:
            if time.time() - t0 >= hard_s:
                say(f"the benchmark is still running after {hard_s:.0f} s: stopping it")
                late = True
            else:
                say(f"no output from the benchmark for {silence_s:.0f} s: stopping it")
                silent = True
            p.kill()
            break
        if line is None:
            break
        log.write(line)
        log.flush()
        print(line, end="", flush=True)
    rc = p.wait()
    return rc, silent, late


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    settings = {}
    sd = find_opt("forge-tpujob-bench")
    if sd is not None and (sd / "settings.json").exists():
        settings = json.loads((sd / "settings.json").read_text())
    jax_dir = bench = None
    for slug in ([str(settings["code_dataset"])] if settings.get("code_dataset") else ["forge-jax-code", "forge-jax"]):  # the first dataset that has both files
        d = find_opt(slug)
        if d is not None and (d / "bench_step.py").exists() and (d / "forge_jax.py").exists():
            jax_dir, bench = d, d / "bench_step.py"
            break
    if jax_dir is None:
        say("no dataset (forge-jax-code, forge-jax or the one named by code_dataset) has both forge_jax.py and bench_step.py in its root: stopping")
        (OUT / "exit.txt").write_text("no benchmark code\n")
        return
    say(f"code from the dataset {jax_dir.name}")
    total_s = float(settings.get("total_s", 1500))
    say(f"python {sys.version.split()[0]}, {sh('nproc')} cpus, " + sh("free -g | sed -n 2p") + f"; total time budget {total_s:.0f} s")
    say("devices: " + sh("ls /dev/accel* /dev/vfio 2>&1 | head -n 5"))
    be = backend()
    say("backend: " + be)
    say(sh("pip list 2>/dev/null | grep -i -E '^(jax|jaxlib|libtpu|numpy|ml.dtypes) ' | tr '\\n' ';'"))
    accel = be.startswith(("tpu", "gpu", "cuda"))
    if not accel and not TINY:
        if not NO_PIP:
            say("no TPU or GPU backend: trying jax[tpu] ...")
            say(sh("pip install -q -U 'jax[tpu]' -f https://storage.googleapis.com/jax-releases/libtpu_releases.html 2>&1 | tail -n 3", timeout=1200))
            be = backend()
            say("backend now: " + be)
            accel = be.startswith(("tpu", "gpu", "cuda"))
        if not accel and settings.get("require_accelerator", True):
            say("still no accelerator: stopping at once so that no time is wasted (set the accelerator of this kernel to TPU in the editor and run it again)")
            (OUT / "exit.txt").write_text("no accelerator\n")
            return
    env = None
    pin = str(settings.get("jax_pin", "0.11.2"))  # "" = use the preinstalled JAX
    if pin and be.startswith("tpu") and f"(jax {pin})" not in be and not NO_PIP:
        env = pinned_jax(pin)
        if env is None:
            say(f"the pinned jax {pin} is not usable here: the preinstalled JAX is used")
    left = total_s - (time.time() - T0) - 60  # 60 s to write the outputs
    if left < 300:
        say(f"only {left:.0f} s of the {total_s:.0f} s budget left after the set-up: too little for a benchmark")
        (OUT / "exit.txt").write_text("too little time\n")
        return
    cmd = [sys.executable, "-u", str(bench), "--out", str(OUT), "--deadline-s", f"{left:.0f}", "--per-variant-s", str(settings.get("per_variant_s", 420))]
    if settings.get("args"):
        cmd += ["--args=" + settings["args"]]
    if settings.get("only"):
        cmd += ["--only", settings["only"]]
    if TINY:
        cmd += ["--tiny", "--steps", "3"]
    say("running: " + " ".join(shlex.quote(c) for c in cmd))
    with open(OUT / "bench.log", "a") as log:
        rc, silent, late = run_bench(cmd, log, float(settings.get("silence_s", 600)), left + 30, env)
    say(f"benchmark exit code {rc} after {(time.time() - T0) / 60:.1f} min" + (" (killed: no output)" if silent else "") + (" (killed: time limit)" if late else ""))
    ok = False
    try:
        rows = json.loads((OUT / "bench.json").read_text())["table"]
        ok = any(r["variant"] == "base" and r.get("status") == "ok" for r in rows)
        say("variants: " + ", ".join(f"{r['variant']}={r['status']}" + (f" {r['tokens_per_s']:.0f} tok/s" if r.get("status") == "ok" else "") for r in rows))
    except (OSError, ValueError, KeyError):
        say("no bench.json was written")
    say(f"done after {(time.time() - T0) / 60:.1f} min: " + ("the base variant was measured" if ok else "the base variant was NOT measured (see bench.log)"))
    (OUT / "exit.txt").write_text("0\n" if ok else "1\n")


if __name__ == "__main__":
    main()
