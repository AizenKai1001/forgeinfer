"""Run a long Forge-1 training on Kaggle as a chain of sessions (each one at most 12 hours): push the kernel, wait until it ends, take the checkpoint from its output, make it the dataset forge-ckpt, push again,
until the log shows the target step. Runs on the PC; every Kaggle call goes through <dev>\\kaggle\\kg.sh (the API token stays there).

    python cloud/kaggle_chain.py --name forge-train --target-step 45000 [--max-hops 8] [--poll-s 300]

Prerequisites (see cloud/kaggle_stage.py): datasets forge-bin, forge-tokens, forge-trainjob exist, the kernel folder was staged with `trainkernel --with-ckpt`, and a dataset forge-ckpt exists (an empty placeholder is fine for
the first hop: the kernel starts from scratch when it finds no ckpt_latest.bin in it). Progress goes to cloud/kaggle_chain.log; the downloaded outputs of every hop to <dev>\\kaggle\\out\\<name>_<hop>.
"""
import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

_GIT_BASH = "C:/Program Files/Git/bin/bash.exe"  # a process started without a shell (the detached chain) has no `bash` on its PATH
KG = [_GIT_BASH if Path(_GIT_BASH).exists() else "bash", "<dev>/kaggle/kg.sh"]
STAGE = Path(r"<dev>\kaggle\stage")
OUTS = Path(r"<dev>\kaggle\out")
LOG = Path(__file__).resolve().parent / "kaggle_chain.log"


def say(msg: str) -> None:
    line = time.strftime("%Y-%m-%d %I:%M %p") + "  " + msg
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def kg(*args: str, timeout: int = 1800, cwd: str | None = None) -> str:
    """One call of the Kaggle CLI. Calls that upload files run inside the folder with `-p .`: a -p path with a drive letter breaks the CLI's upload cache (the upload is then silently skipped)."""
    r = subprocess.run([*KG, *args], capture_output=True, text=True, timeout=timeout, cwd=cwd)
    return (r.stdout + r.stderr).strip()


def kernel_status(ref: str) -> str:
    out = kg("kernels", "status", ref, timeout=120)
    for word in ("COMPLETE", "ERROR", "CANCEL", "RUNNING", "QUEUED"):
        if word in out.upper():
            return word
    return out[-80:]


def last_step(run_dir: Path) -> int:
    log = run_dir / "log.jsonl"
    if not log.exists():
        return 0
    step = 0
    for line in log.read_text().splitlines():
        try:
            step = json.loads(line)["step"]
        except Exception:  # noqa: BLE001
            pass
    return step


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="forge-train")
    ap.add_argument("--target-step", type=int, required=True)
    ap.add_argument("--max-hops", type=int, default=8)
    ap.add_argument("--poll-s", type=int, default=300)
    ap.add_argument("--first-hop", type=int, default=1)
    ap.add_argument("--attach", action="store_true", help="the first hop of this run is already running on Kaggle: do not push it, only wait for it")
    a = ap.parse_args()
    ref = f"YOUR_KAGGLE_USER/{a.name}"
    kernel_dir = STAGE / f"{a.name}-kernel"
    stop_file = STAGE / "STOP_CHAIN"  # create this file to end the chain after the hop that is running (no further hop is pushed)
    for hop in range(a.first_hop, a.max_hops + 1):
        attached = a.attach and hop == a.first_hop
        if stop_file.exists() and not attached:
            say(f"stop file {stop_file} found: hop {hop} is not started")
            break
        if attached:
            say(f"hop {hop}: attached to the running kernel {ref}")
        else:
            say(f"hop {hop}: push {ref}")
            say(kg("kernels", "push", "-p", ".", cwd=str(kernel_dir)))
            time.sleep(60)
        while True:
            st = kernel_status(ref)
            if st in ("COMPLETE", "ERROR", "CANCEL"):
                break
            time.sleep(a.poll_s)
        say(f"hop {hop}: kernel status {st}")
        out = OUTS / f"{a.name}_{hop}"
        if out.exists():
            shutil.rmtree(out)
        out.mkdir(parents=True)
        say(kg("kernels", "output", ref, "-p", str(out), timeout=7200)[-300:])
        run = out / "run"
        step = last_step(run)
        say(f"hop {hop}: log reaches step {step} of {a.target_step}; checkpoint present: {(run / 'ckpt_latest.bin').exists()}")
        if step >= a.target_step:
            say("target reached")
            break
        if not (run / "ckpt_latest.bin").exists():
            say("no checkpoint in the output: stopping the chain")
            sys.exit(1)
        stage = STAGE / "forge-ckpt"
        stage.mkdir(parents=True, exist_ok=True)
        shutil.copy2(run / "ckpt_latest.bin", stage / "ckpt_latest.bin")
        if (run / "log.jsonl").exists():
            shutil.copy2(run / "log.jsonl", stage / "log.jsonl")
        (stage / "dataset-metadata.json").write_text(json.dumps({"title": "Forge ckpt", "id": "YOUR_KAGGLE_USER/forge-ckpt", "licenses": [{"name": "CC0-1.0"}]}, indent=1), encoding="utf-8")
        say("dataset version: " + kg("datasets", "version", "-p", ".", "-m", f"hop {hop}, step {step}", timeout=7200, cwd=str(stage))[-200:])
        for _ in range(60):
            if "ready" in kg("datasets", "status", "YOUR_KAGGLE_USER/forge-ckpt", timeout=120).lower():
                break
            time.sleep(20)
    say("chain finished")


if __name__ == "__main__":
    main()
