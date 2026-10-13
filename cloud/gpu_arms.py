"""Run the arms of a proxy sweep one after the other on the GPU of this machine (the runner of the Lightning jobs; the same arm format as the Kaggle sweeps: cloud/kaggle_kernel.py).

    python gpu_arms.py --forge /path/forge --data /path/shards --out /path/out --arms-json '[{"name": "x", "args": "--layers 12 ..."}]'

Writes <out>/<arm>.log, <out>/summary.jsonl (one JSON line per finished arm: name, seconds, val list, last step, tokens per second) and <out>/env.txt.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path


def sh(cmd: str) -> str:
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--forge", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arms-json", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    os.chmod(a.forge, 0o755)
    (out / "env.txt").write_text(sh("nvidia-smi") + "\n" + sh("ldd " + a.forge) + "\n")
    arms = json.loads(a.arms_json)
    t00 = time.time()
    for arm in arms:
        name = arm["name"]
        run_dir = f"/tmp/run_{name}"
        cmd = f"{a.forge} train --data {a.data} --out {run_dir} {arm['args']}"
        log = out / f"{name}.log"
        t = time.time()
        with open(log, "w") as f:
            rc = subprocess.run(cmd, shell=True, stdout=f, stderr=subprocess.STDOUT).returncode
        text = log.read_text()
        vals = re.findall(r"step\s+(\d+)/(\d+) .*?([\d.]+) tok/s.*?val ([\d.]+)", text)
        last = re.findall(r"step\s+(\d+)/(\d+) loss ([\d.]+).*?([\d.]+) tok/s", text)
        rec = {"name": name, "gpu": 0, "returncode": rc, "seconds": round(time.time() - t), "args": arm["args"], "val": [float(v[3]) for v in vals], "last_step": last[-1][0] if last else None, "tok_s": float(last[-1][3]) if last else None}
        with open(out / "summary.jsonl", "a") as sf:
            sf.write(json.dumps(rec) + "\n")
        shutil.rmtree(run_dir, ignore_errors=True)
        print(f"[{name}] rc {rc}, {rec['seconds']} s, val {rec['val'][-1:]}", flush=True)
    print(f"done in {(time.time() - t00) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
