"""Kaggle script kernel: a long Forge-1 training run on the free GPUs (all GPUs of the session, data parallel through host memory), resumable across sessions.

Inputs (Kaggle datasets of the account YOUR_KAGGLE_USER, staged by cloud/kaggle_stage.py): forge-bin (the Linux build of `forge`), forge-tokens (the token shards), forge-trainjob (settings.json) and, from the second session
on, forge-ckpt (ckpt_latest.bin and log.jsonl of the previous session). Output in /kaggle/working: run/ckpt_latest.bin, run/log.jsonl, train.log.
settings.json:  {"args": "--layers 30 --dim 576 ... --steps 45000 ...", "data": "forge-tokens", "hours": 11.2, "files": ["fwedu_000.u16", ...]}      ("files": optional subset of the shards in the dataset)
The session stops itself with a checkpoint at the first optimizer step past `hours` (Kaggle ends a session after 12 hours); `kaggle_chain.py` starts the next one from that checkpoint.
"""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

IN = Path("/kaggle/input")
OUT = Path("/kaggle/working")


def find(slug: str):
    for cand in (IN / slug, *IN.glob(f"*/{slug}"), *IN.glob(f"*/*/{slug}"), *IN.glob(f"datasets/*/{slug}")):
        if cand.is_dir():
            return cand
    return None


def main() -> None:
    job = find("forge-trainjob")
    settings = json.loads((job / "settings.json").read_text())
    shutil.copy(find("forge-bin") / "forge", "/tmp/forge")
    os.chmod("/tmp/forge", 0o755)
    data = find(settings.get("data", "forge-tokens"))
    if settings.get("files"):
        # a subset of the shards: a directory of links
        sub = Path("/tmp/train_data")
        sub.mkdir(exist_ok=True)
        for f in settings["files"]:
            (sub / f).symlink_to(data / f)
        data = sub
    run = OUT / "run"
    run.mkdir(parents=True, exist_ok=True)
    ckpt = find("forge-ckpt")
    resume = ""
    if ckpt is not None and (ckpt / "ckpt_latest.bin").exists():
        t = time.time()
        shutil.copy(ckpt / "ckpt_latest.bin", run / "ckpt_latest.bin")
        if (ckpt / "log.jsonl").exists():
            shutil.copy(ckpt / "log.jsonl", run / "log.jsonl")
        resume = "--resume"
        print(f"checkpoint copied in {time.time() - t:.0f} s", flush=True)
    gpus = len([l for l in subprocess.run("nvidia-smi -L", shell=True, capture_output=True, text=True).stdout.splitlines() if l.startswith("GPU")])
    hours = float(settings.get("hours", 11.2))
    cmd = f"/tmp/forge train --data {data} --out {run} --gpus {max(gpus, 1)} --time-limit-s {int(hours * 3600)} {resume} {settings['args']}"
    print(cmd, flush=True)
    (OUT / "env.txt").write_text(subprocess.run("nvidia-smi", shell=True, capture_output=True, text=True).stdout)
    with open(OUT / "train.log", "a") as log:
        # the whole log goes to train.log; every 20th step line, the evaluations and the [train] lines also go to the kernel's log page, which shows them while the session runs
        p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        n = 0
        for line in p.stdout:
            log.write(line)
            log.flush()
            n += 1
            if " val " in line or line.startswith("[") or n % 20 == 0:
                print(line, end="", flush=True)
        p.wait()
    print("forge train exited with", p.returncode, flush=True)
    print(subprocess.run(f"ls -la {run}; tail -n 3 {OUT / 'train.log'}", shell=True, capture_output=True, text=True).stdout, flush=True)
    sys.exit(0 if p.returncode == 0 else 1)


if __name__ == "__main__":
    main()
