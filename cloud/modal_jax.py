"""Modal app that runs the JAX trainer (tpu/forge_jax.py) on a cloud GPU: the fallback path for fine-tunes when the Rust engine cannot read or run a checkpoint written by the TPU run
(for example fp16 overflow of a bf16-trained model), and the way to continue a TPU model in its own numerics (bf16 activations, fp32 master weights, the same loss and optimizer code).

NOTHING HERE RUNS, AND NOTHING IS BILLED, UNTIL A COMMAND BELOW IS STARTED. From Git Bash: export MSYS_NO_PATHCONV=1 PYTHONUTF8=1 PYTHONIOENCODING=utf-8 first (see cloud/modal_forge.py).

  modal run cloud/modal_jax.py::smoke                      # about 0.1 USD: the tiny JAX checkpoint /runs/jxcompat/ckpt_latest.bin is fine-tuned for 12 steps on the SFT shards (needs the volumes of modal_forge.py)
  modal run cloud/modal_jax.py::train --name NAME --approved-usd 3.5 --max-hours 1.0 --data dir:tokens_sft_human_replay --args="--init-from /runs/tpu1/ckpt_latest.bin <structure flags> --seq 2048 --micro 4 ..."

The arguments after --args are those of `python tpu/forge_jax.py train` (same names as `forge train` for the common ones: --layers --dim --heads --kv --ffn --qk-norm --relu2 --lambdas --value-emb --lr-table-mult
--lr-emb-mult --lr-scalar-mult --seq --micro --tokens-per-step --steps --warmup --decay-frac --min-lr-frac --optim muon --lr --muon-lr --muon-wd --val-tokens --eval-every --eval-batches --save-every --seed --init-from).
The data directory of the volume forge-data is copied to the container disk first (SFT shards carry the loss mask in bit 15; the JAX trainer reads it as the Rust dev build does). Results: /runs/NAME/ in the volume forge-runs
(ckpt_latest.bin is the same forgeckpt1 format as the Rust trainer's). Money: every call has a hard timeout and prints the worst case before it starts; the estimated cost is appended to cloud/ledger.jsonl.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
APP_NAME = "forge1-jax"
GPU_USD_PER_HOUR = {"T4": 0.59, "L4": 0.80, "A10": 1.10, "L40S": 1.95, "A100-40GB": 2.10, "A100-80GB": 2.50, "H100": 3.95}
CPU, MEM_MIB = 4, 16384
CPU_USD_PER_CORE_S, MEM_USD_PER_GIB_S = 0.0000131, 0.00000222
LEDGER = HERE / "ledger.jsonl"
JAX_VERSION = "0.11.2"   # the version the trainer was tested with (tpu/forge_jax.py, _pin_jax_for_tpu)


def cost_usd(gpu: str, seconds: float) -> float:
    return (GPU_USD_PER_HOUR[gpu] / 3600.0 + CPU * CPU_USD_PER_CORE_S + (MEM_MIB / 1024.0) * MEM_USD_PER_GIB_S) * seconds


def ledger(entry: dict) -> None:
    entry = {"time": time.strftime("%Y-%m-%d %I:%M %p"), **entry}
    with open(LEDGER, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04", add_python="3.12")   # jax 0.11.x needs Python 3.12 or newer
    .pip_install(f"jax[cuda12]=={JAX_VERSION}", "numpy")
    .add_local_file(str(REPO / "tpu" / "forge_jax.py"), "/code/forge_jax.py", copy=True)
)
app = modal.App(APP_NAME)
data_vol = modal.Volume.from_name("forge-data", create_if_missing=True)
runs_vol = modal.Volume.from_name("forge-runs", create_if_missing=True)
VOLUMES = {"/data": data_vol, "/runs": runs_vol}


def _run(name: str, args: list[str], max_hours: float, data: str) -> dict:
    out = f"/runs/{name}"
    Path(out).mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    Path("/tmp/train_data").mkdir(exist_ok=True)
    if data.startswith("dir:"):
        rels = [f"{data[4:]}/{q.name}" for q in sorted(Path(f"/data/{data[4:]}").iterdir())]
    else:
        rels = data.split(",")
    for rel in rels:
        shutil.copyfile(f"/data/{rel}", f"/tmp/train_data/{Path(rel).name}")
    print(f"data copied in {time.time() - t0:.0f} s: {len(rels)} files", flush=True)
    cmd = ["python", "-u", "/code/forge_jax.py", "train", "--data", "/tmp/train_data", "--out", out, "--no-pin", *args]
    stop = threading.Event()

    def committer():
        while not stop.wait(300):
            try:
                runs_vol.commit()
            except Exception as e:  # noqa: BLE001
                print("commit failed:", e, flush=True)

    threading.Thread(target=committer, daemon=True).start()
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    last = [time.time()]

    def watchdog():
        while p.poll() is None:
            time.sleep(5)
            if time.time() - t0 > max_hours * 3600 + 30 or time.time() - last[0] > 900:
                print("watchdog: killing the trainer", flush=True)
                p.kill()
                return

    threading.Thread(target=watchdog, daemon=True).start()
    lines, steps = [], []
    for line in p.stdout:
        last[0] = time.time()
        lines.append(line.rstrip())
        m = re.search(r"step\s+(\d+)/(\d+).*?loss ([\d.]+)", line)
        if m:
            steps.append((int(m.group(1)), float(m.group(3))))
        if any(k in line for k in ("[train]", "[attn]", "val", "Error", "error", "Traceback", "NaN", "nan")) or (m and int(m.group(1)) % 50 == 0):
            print(line.rstrip(), flush=True)
    p.wait()
    stop.set()
    Path(out, "train_stdout.log").write_text("\n".join(lines) + "\n")
    runs_vol.commit()
    vals = [ln for ln in lines if " val " in ln][-5:]
    return {"name": name, "returncode": p.returncode, "seconds": time.time() - t0, "last_step": steps[-1][0] if steps else None, "last_loss": steps[-1][1] if steps else None, "val_tail": vals, "tail": lines[-6:]}


def _fn(gpu: str, timeout: int):
    return app.function(image=image, gpu=gpu, timeout=timeout, cpu=CPU, memory=MEM_MIB, retries=0, volumes=VOLUMES)


@_fn("A100-40GB", 5400)
def train_a100(name: str, args: list[str], max_hours: float, data: str) -> dict:
    return _run(name, args, max_hours, data)


@app.local_entrypoint()
def train(name: str, args: str = "", max_hours: float = 1.0, approved_usd: float = 0.0, data: str = "dir:tokens_sft_human_replay", gpu: str = "A100-40GB"):
    import shlex

    if gpu != "A100-40GB":
        raise SystemExit("only A100-40GB is wired in (add a function for another GPU)")
    worst = cost_usd(gpu, min(max_hours, 1.25) * 3600 + 120)
    print(f"jax train {name} on {gpu}: up to {max_hours} h = at most {worst:.2f} USD; approved {approved_usd:.2f} USD")
    if max_hours > 1.25:
        raise SystemExit("refused: the function timeout is 90 minutes; use --max-hours 1.25 or less")
    if worst > approved_usd:
        raise SystemExit(f"refused: the worst case {worst:.2f} USD is above the approved {approved_usd:.2f} USD (pass --approved-usd)")
    ledger({"call": "jax-train-start", "name": name, "gpu": gpu, "max_hours": max_hours, "worst_usd": round(worst, 2), "args": args})
    res = train_a100.remote(name, shlex.split(args), max_hours, data)
    print(json.dumps(res, indent=1))
    ledger({"call": "jax-train-end", "name": name, "seconds": round(res["seconds"]), "est_usd": round(cost_usd(gpu, res["seconds"]), 3), "last_step": res["last_step"]})


@app.local_entrypoint()
def smoke():
    """12 steps of the JAX trainer on the A100 from the tiny JAX checkpoint (uploaded by `modal volume put forge-runs scratch/jx_compat/ckpt_latest.bin /jxcompat/ckpt_latest.bin`)."""
    args = ("--init-from /runs/jxcompat/ckpt_latest.bin --layers 4 --dim 192 --heads 3 --kv 1 --ffn 512 --qk-norm --relu2 --lambdas --value-emb 0,3 --seq 512 --micro 8 --tokens-per-step 65536 --steps 12 "
            "--warmup 3 --decay-frac 0.85 --min-lr-frac 0.05 --optim muon --lr 3e-4 --muon-lr 3e-4 --muon-wd 0.01 --val-tokens 4000000 --eval-every 6 --eval-batches 4 --save-every 12 --seed 1")
    import shlex

    worst = cost_usd("A100-40GB", 600)
    print(f"jax smoke on A100-40GB: hard cap 10 minutes = at most {worst:.2f} USD")
    t0 = time.time()
    res = train_a100.remote("jax_smoke", shlex.split(args), 0.1, "dir:tokens_sft_human_replay")
    print(json.dumps(res, indent=1))
    est = cost_usd("A100-40GB", time.time() - t0)
    ledger({"call": "jax-smoke", "seconds": round(time.time() - t0), "est_usd": round(est, 3), "returncode": res["returncode"], "last_step": res["last_step"]})
    print(f"estimated cost {est:.2f} USD")
