"""Kaggle script kernel: run `forge train` arms (the Forge-1 proxy A/B sweeps) on the free T4s, one arm per GPU at a time, no internet needed.

Inputs (Kaggle datasets of the account YOUR_KAGGLE_USER, staged by cloud/kaggle_stage.py): forge-bin (the Linux build of `forge`), forge-data-web (token shards), forge-job (settings.json). Outputs in /kaggle/working:
one <arm>.log per arm, summary.jsonl (one JSON line per finished arm), env.txt (GPUs, driver). settings.json:
  {"data": "forge-data-web", "arms": [{"name": "base", "args": "--layers 12 --dim 384 ... --steps 500 ..."}, ...], "max_hours": 8}
The arguments of an arm go to `forge train` after `--data DIR --out DIR`; the arm is skipped when the time budget (max_hours of the whole session) is used up.
"""
import json
import os
import re
import shlex
import shutil
import subprocess
import threading
import time
from pathlib import Path

IN = Path("/kaggle/input")
OUT = Path("/kaggle/working")
T0 = time.time()


def find(slug: str) -> Path:
    """The folder of an attached dataset (the layout of /kaggle/input differs between Kaggle image versions)."""
    for cand in (IN / slug, *IN.glob(f"*/{slug}"), *IN.glob(f"*/*/{slug}"), *IN.glob(f"datasets/*/{slug}")):
        if cand.is_dir():
            return cand
    raise SystemExit(f"dataset {slug} not found under {IN}: {[str(p) for p in IN.glob('*')]}")


def find_prefix(prefix: str) -> Path:
    """The attached dataset whose folder name starts with `prefix` (forge-job, forge-job2, ...: one job dataset per kernel)."""
    for d in sorted(list(IN.glob(f"{prefix}*")) + list(IN.glob(f"*/{prefix}*")) + list(IN.glob(f"*/*/{prefix}*"))):
        if d.is_dir():
            return d
    raise SystemExit(f"no dataset starting with {prefix} under {IN}")


def sh(cmd: str) -> str:
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()


def main() -> None:
    job = find_prefix("forge-job")
    settings = json.loads((job / "settings.json").read_text())
    binary = find(settings.get("bin", "forge-bin")) / "forge"  # "bin": the slug of the dataset with the build to test (an experimental build lives in forge-bin2 so that the chain's forge-bin stays untouched)
    shutil.copy(binary, "/tmp/forge")
    os.chmod("/tmp/forge", 0o755)
    data = find(settings.get("data", "forge-data-web"))
    n_gpu = len([l for l in sh("nvidia-smi -L").splitlines() if l.startswith("GPU")])
    (OUT / "env.txt").write_text(sh("nvidia-smi") + "\n" + sh("ldd /tmp/forge") + "\n" + sh("/tmp/forge info 2>&1 | head -n 20") + "\n")
    print(f"{n_gpu} GPU(s); data {data}; {len(settings['arms'])} arms; budget {settings.get('max_hours', 8)} h", flush=True)
    arms = list(settings["arms"])
    lock = threading.Lock()
    budget_s = settings.get("max_hours", 8) * 3600

    def worker(gpu: int) -> None:
        while True:
            with lock:
                if not arms or time.time() - T0 > budget_s:
                    return
                arm = arms.pop(0)
            name = arm["name"]
            out_dir = f"/tmp/run_{name}"
            cmd = f"/tmp/forge train --data {shlex.quote(str(data))} --out {out_dir} {arm['args']}"
            log = OUT / f"{name}.log"
            t = time.time()
            with open(log, "w") as f:
                rc = subprocess.run(cmd, shell=True, stdout=f, stderr=subprocess.STDOUT, env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}).returncode
            text = log.read_text()
            vals = re.findall(r"step\s+(\d+)/(\d+) .*?([\d.]+) tok/s.*?val ([\d.]+)", text)
            last = re.findall(r"step\s+(\d+)/(\d+) loss ([\d.]+).*?([\d.]+) tok/s", text)
            rec = {"name": name, "gpu": gpu, "returncode": rc, "seconds": round(time.time() - t), "args": arm["args"],
                   "val": [float(v[3]) for v in vals], "last_step": last[-1][0] if last else None, "tok_s": float(last[-1][3]) if last else None}
            with lock:
                with open(OUT / "summary.jsonl", "a") as sf:
                    sf.write(json.dumps(rec) + "\n")
            shutil.rmtree(out_dir, ignore_errors=True)
            print(f"[{name}] gpu {gpu}: rc {rc}, {rec['seconds']} s, val {rec['val'][-1:] }", flush=True)

    threads = [threading.Thread(target=worker, args=(g,)) for g in range(max(n_gpu, 1))]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    print(f"done in {(time.time() - T0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
