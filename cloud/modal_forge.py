"""Modal app for the Forge-1 experiments and training (the from-scratch trainer `forge train` on cloud GPUs).

NOTHING HERE RUNS, AND NOTHING IS BILLED, UNTIL A COMMAND BELOW IS STARTED. Modal client: <dev>\\research\\neural-ai-from-scratch-20260919\\reasoner5b\\cloud\\.venv (modal 1.6.1), logged in as the user.
From Git Bash export MSYS_NO_PATHCONV=1 PYTHONUTF8=1 PYTHONIOENCODING=utf-8 first (see the memory note cloud-compute-modal).

  modal run cloud/modal_forge.py::smoke --gpu T4 [--micro 4] [--steps 30]     # one machine: image, kernels (cubin or PTX JIT), speed of the 31M proxy and of the 125M shape on random tokens (cents)
  modal run cloud/modal_forge.py::sweep --gpu A100-40GB --arms cloud/arms_round1.json   # the proxy A/B arms in parallel, one container each
  modal volume ls forge-data / modal volume get forge-runs <path> ...        # results

The image is built from the work tree (`crates/`, `tools/`, `Cargo.*`): CUDA 12.4 devel + Rust, `cargo build --release -p forge-cli` runs at image build time (std-only crates, nvcc compiles the sm_75 cubin and the compute_75 PTX;
newer GPUs JIT the PTX when the module loads). Money: every call has a hard timeout (the worst case is printed before the call) and the estimated cost of every finished call is appended to cloud/ledger.jsonl.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
APP_NAME = "forge1"
DATA_VOLUME = "forge-data"     # tokenizer + encoded token shards
RUNS_VOLUME = "forge-runs"     # logs, checkpoints, summaries of the runs
# Modal list prices, USD per hour (2026-10: L4 .80, A10 1.10, L40S 1.95, A100-40GB 2.10, A100-80GB 2.50, H100 3.95, T4 .59), CPU per core-second, memory per GiB-second
GPU_USD_PER_HOUR = {"T4": 0.59, "L4": 0.80, "A10": 1.10, "L40S": 1.95, "A100-40GB": 2.10, "A100-80GB": 2.50, "H100": 3.95, "H200": 4.54}
CPU_USD_PER_CORE_S = 0.0000131
MEM_USD_PER_GIB_S = 0.00000222
CPU, MEM_MIB = 4, 16384
LEDGER = HERE / "ledger.jsonl"


def cost_usd(gpu: str | None, seconds: float, cpu: float = CPU, mem_mib: int = MEM_MIB) -> float:
    rate = (GPU_USD_PER_HOUR[gpu] / 3600.0 if gpu else 0.0) + cpu * CPU_USD_PER_CORE_S + (mem_mib / 1024.0) * MEM_USD_PER_GIB_S
    return rate * seconds


def ledger(entry: dict) -> None:
    entry = {"time": time.strftime("%Y-%m-%d %I:%M %p"), **entry}
    with open(LEDGER, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


# tools/ holds dozens of scripts that change all day and an edit there would rebuild the image (5 minutes of CPU) at the next call: the image only needs the evaluation ladder and the chat checklist
IGNORE = ["**/target/**", "**/.git/**", "tests/golden/**", "docs/**", "runs/**", "data/**", "cloud/**", "**/__pycache__/**", "**/*.pyc", "scratch/**",
          "tools/**", "!tools/f1_eval.sh", "!tools/f1_eval_reference.json", "!tools/sft_data/sft_checklist.sh", "tpu/**", "bench/**", "spikes/**", "README.md", ".gitattributes", ".gitignore"]
PATH = "/root/.cargo/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("curl", "ca-certificates", "build-essential", "pkg-config")
    .run_commands("curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable")
    .env({"PATH": PATH})
    .pip_install("pyarrow==17.0.0", "numpy==1.26.4")
    .add_local_dir(str(REPO), "/src", copy=True, ignore=IGNORE)
    .run_commands("cd /src && cargo build --release -p forge-cli 2>&1 | grep -v '^warning\\|^ *|\\|^ *=\\|^ *-->\\|^$' | tail -n 15; ls -la /src/target/release/forge")
)

app = modal.App(APP_NAME)
data_vol = modal.Volume.from_name(DATA_VOLUME, create_if_missing=True)
runs_vol = modal.Volume.from_name(RUNS_VOLUME, create_if_missing=True)
VOLUMES = {"/data": data_vol, "/runs": runs_vol}
FORGE = "/src/target/release/forge"


def sh(cmd: str, timeout: int = 600) -> str:
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout).stdout.strip()


def run_train(args: list[str], log_path: str | None = None, timeout_s: int = 900, silence_s: int = 900) -> dict:
    """Run `forge train` with the arguments, return the timeline: seconds until the first step (module load + JIT) and the tok/s of every step.
    A watchdog thread kills the trainer after `timeout_s` seconds or after `silence_s` seconds without a line of output (a hung trainer would otherwise run up the function's own timeout, up to a day of GPU)."""
    t0 = time.time()
    p = subprocess.Popen([FORGE, "train", *args], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    last = [time.time()]

    def watchdog():
        while p.poll() is None:
            time.sleep(5)
            now = time.time()
            if now - t0 > timeout_s + 30 or now - last[0] > silence_s:
                print(f"watchdog: killing the trainer ({'time cap' if now - t0 > timeout_s + 30 else 'no output for %d s' % silence_s})", flush=True)
                p.kill()
                return

    threading.Thread(target=watchdog, daemon=True).start()
    first = None
    steps = []
    lines = []
    for line in p.stdout:
        last[0] = time.time()
        lines.append(line.rstrip())
        m = re.search(r"step\s+(\d+)/(\d+) loss ([\d.]+).*?([\d.]+) tok/s \(([\d.]+) s\)(?:\s+val ([\d.]+))?", line)
        if m:
            if first is None:
                first = time.time() - t0
            steps.append({"step": int(m.group(1)), "loss": float(m.group(3)), "tok_s": float(m.group(4)), "val": float(m.group(6)) if m.group(6) else None})
            print(line.rstrip(), flush=True)
        elif "[train]" in line or "error" in line.lower() or "panick" in line.lower() or "[kprof]" in line or re.match(r"\s*[\d.]+ %\s", line) or line.startswith("kernel time"):
            print(line.rstrip(), flush=True)
        if time.time() - t0 > timeout_s:
            p.kill()
            lines.append("TIMEOUT")
            break
    p.wait()
    if log_path:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        Path(log_path).write_text("\n".join(lines) + "\n")
    return {"returncode": p.returncode, "seconds": time.time() - t0, "first_step_s": first, "steps": steps, "tail": lines[-6:]}


def _smoke(micro: int, steps: int, prof: bool = False) -> dict:
    if prof:
        os.environ["FORGE_KPROF"] = "1"
    import numpy as np

    info = sh("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader")
    Path("/tmp/data").mkdir(exist_ok=True)
    rng = np.random.default_rng(1)
    rng.integers(0, 32768, size=16_000_000, dtype=np.uint16).tofile("/tmp/data/rand.u16")
    out = {"gpu": info, "forge": sh(f"ls -la {FORGE}")}
    shapes = {
        "proxy31M": ["--layers", "12", "--dim", "384", "--heads", "6", "--kv", "2", "--ffn", "1024"],
        "f125": ["--layers", "30", "--dim", "576", "--heads", "9", "--kv", "3", "--ffn", "1536"],
    }
    for name, shape in shapes.items():
        args = ["--data", "/tmp/data", "--out", f"/tmp/run_{name}", *shape, "--seq", "2048", "--micro", str(micro), "--tokens-per-step", "65536", "--steps", str(steps), "--warmup", "5",
                "--qk-norm", "--optim", "muon", "--lr", "2e-3", "--muon-lr", "2e-3", "--eval-every", "100000", "--save-every", "100000", "--val-tokens", "1000000"]
        r = run_train(args, timeout_s=420)
        if os.environ.get("FORGE_KPROF"):
            print(f"== kernel table of {name} ==")
        tps = [s["tok_s"] for s in r["steps"][5:]] or [s["tok_s"] for s in r["steps"]]
        out[name] = {"returncode": r["returncode"], "first_step_s": r["first_step_s"], "tok_s_mean": sum(tps) / max(len(tps), 1), "tok_s_max": max(tps) if tps else 0, "tail": r["tail"],
                     "mem": sh("nvidia-smi --query-gpu=memory.used --format=csv,noheader")}
    return out


def _wrap(gpu: str, timeout: int):
    def deco(fn):
        return app.function(image=image, gpu=gpu, timeout=timeout, cpu=CPU, memory=MEM_MIB, retries=0, volumes=VOLUMES)(fn)
    return deco


@_wrap("T4", 900)
def smoke_t4(micro: int, steps: int, prof: bool = False) -> dict:
    return _smoke(micro, steps, prof)


@_wrap("A100-40GB", 900)
def smoke_a100(micro: int, steps: int, prof: bool = False) -> dict:
    return _smoke(micro, steps, prof)


@_wrap("H100", 900)
def smoke_h100(micro: int, steps: int, prof: bool = False) -> dict:
    return _smoke(micro, steps, prof)


@_wrap("L4", 900)
def smoke_l4(micro: int, steps: int, prof: bool = False) -> dict:
    return _smoke(micro, steps, prof)


@app.function(image=image, timeout=300, cpu=1, memory=2048, volumes=VOLUMES)
def export_bin() -> str:
    """Copy the Linux build of `forge` (made at image build time, Ubuntu 22.04, glibc 2.35) into the runs volume: `modal volume get forge-runs bin/forge <local path>`."""
    Path("/runs/bin").mkdir(parents=True, exist_ok=True)
    sh(f"cp {FORGE} /runs/bin/forge && chmod +x /runs/bin/forge")
    runs_vol.commit()
    return sh("ls -la /runs/bin/forge; ldd /runs/bin/forge; /runs/bin/forge info 2>&1 | head -n 5")


@app.local_entrypoint()
def binary():
    print(export_bin.remote())



PROXY = ["--layers", "12", "--dim", "384", "--heads", "6", "--kv", "2", "--ffn", "1024", "--seq", "2048", "--micro", "8", "--tokens-per-step", "65536", "--steps", "500", "--warmup", "30",
         "--eval-every", "100", "--eval-batches", "8", "--save-every", "100000", "--lr", "2e-3", "--optim", "muon", "--muon-lr", "2e-3", "--muon-wd", "0.05", "--qk-norm"]


def merge_args(base: list[str], extra: list[str]) -> list[str]:
    """`forge train` takes the FIRST occurrence of a flag: remove from `base` every flag that `extra` sets again (same function as cloud/kaggle_stage.py)."""
    def pairs(tokens):
        out, i = [], 0
        while i < len(tokens):
            if tokens[i].startswith("--") and i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
                out.append((tokens[i], tokens[i + 1]))
                i += 2
            else:
                out.append((tokens[i], None))
                i += 1
        return out

    over = {k for k, _ in pairs(extra)}
    return [x for k, v in pairs(base) + [] if k not in over for x in ((k,) if v is None else (k, v))] + list(extra)


def _local_copy(src: str, dst: str) -> None:
    Path(dst).mkdir(parents=True, exist_ok=True)
    for f in sorted(Path(src).glob("*.u16")):
        if not (Path(dst) / f.name).exists():
            shutil.copyfile(f, Path(dst) / f.name)


def _sweep_arm(tag: str, name: str, extra: str, data_files: list[str]) -> dict:
    """One proxy arm (the 31M model of tools/f1_ab.sh) on the shard(s) `data_files` of the data volume; log and summary go to /runs/sweeps/<tag>/."""
    import shlex

    ddir = f"/tmp/proxy_data_{name}"
    Path(ddir).mkdir(parents=True, exist_ok=True)
    for f in data_files:
        shutil.copyfile(f"/data/tokens_f1/{f}", f"{ddir}/{f}")
    out = f"/tmp/run_{name}"
    r = run_train(["--data", ddir, "--out", out, *merge_args(PROXY, shlex.split(extra))], log_path=f"/runs/sweeps/{tag}/{name}.log", timeout_s=1500)
    vals = [s["val"] for s in r["steps"] if s["val"] is not None]
    tps = [s["tok_s"] for s in r["steps"][20:]] or [0.0]
    shutil.rmtree(out, ignore_errors=True)
    shutil.rmtree(ddir, ignore_errors=True)
    runs_vol.commit()
    return {"name": name, "extra": extra, "returncode": r["returncode"], "val": vals, "tok_s": sum(tps) / len(tps), "seconds": r["seconds"], "first_step_s": r["first_step_s"]}


def _train(name: str, args: list[str], max_hours: float, data: str = "") -> dict:
    """The long run: `forge train` on the whole mixture in /data/tokens_f1 (copied to the container's disk first), resumable: the checkpoint and the log live in /runs/<name> and the volume is committed every 5 minutes."""
    out = f"/runs/{name}"
    Path(out).mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if data:
        # a list of shard files of the data volume, comma separated (for example "tokens_sft/sft_000.u16,tokens_f1/cosmo_000.u16"): only these are copied;
        # "dir:NAME" copies every file of /data/NAME (for example dir:tokens_sft_human_replay: the replay shard sorts first, zzz_val.u16 last)
        Path("/tmp/train_data").mkdir(exist_ok=True)
        rels = [f"{data[4:]}/{q.name}" for q in sorted(Path(f"/data/{data[4:]}").iterdir())] if data.startswith("dir:") else data.split(",")
        for rel in rels:
            shutil.copyfile(f"/data/{rel}", f"/tmp/train_data/{Path(rel).name}")
        src = "/tmp/train_data"
    else:
        _local_copy("/data/tokens_f1", "/tmp/tokens_f1")
        src = "/tmp/tokens_f1"
    print(f"data copied to the container disk in {time.time() - t0:.0f} s", flush=True)
    full = ["--data", src, "--out", out, *args]
    if (Path(out) / "ckpt_latest.bin").exists():
        full.append("--resume")
        print("resuming from the checkpoint in the volume", flush=True)
    stop = threading.Event()

    def committer():
        while not stop.wait(300):
            try:
                runs_vol.commit()
            except Exception as e:  # noqa: BLE001
                print("commit failed:", e, flush=True)

    threading.Thread(target=committer, daemon=True).start()
    r = run_train(full, log_path=f"{out}/train_stdout.log", timeout_s=int(max_hours * 3600))
    stop.set()
    runs_vol.commit()
    vals = [(s["step"], s["val"]) for s in r["steps"] if s["val"] is not None]
    return {"name": name, "returncode": r["returncode"], "seconds": r["seconds"], "last_step": r["steps"][-1]["step"] if r["steps"] else None, "val_tail": vals[-5:], "tail": r["tail"]}


def _eval(name: str, ckpt: str, tasks: list[str]) -> dict:
    """Benchmarks of a checkpoint in the runs volume: HellaSwag and the multiple-choice sets of tools/mc_prepare.py (files under /data/eval)."""
    res = {}
    for t in tasks:
        path = f"/data/eval/{t}.jsonl"
        cmd = ([FORGE, "eval-hellaswag", ckpt, "--tokenizer", "/data/tokenizer_f1.json", "--data", path] if t == "hellaswag_val" else [FORGE, "eval-mc", ckpt, "--tokenizer", "/data/tokenizer_f1.json", "--data", path])
        res[t] = subprocess.run(cmd, capture_output=True, text=True).stdout.strip().splitlines()[-1:]
        print(t, res[t], flush=True)
    return res


def _gpu_fn(gpu: str, timeout: int):
    return app.function(image=image, gpu=gpu, timeout=timeout, cpu=CPU, memory=MEM_MIB, retries=0, volumes=VOLUMES)


@_gpu_fn("A100-40GB", 1800)
def sweep_a100(tag: str, name: str, extra: str, data_files: list[str]) -> dict:
    return _sweep_arm(tag, name, extra, data_files)


@_gpu_fn("H100", 1800)
def sweep_h100(tag: str, name: str, extra: str, data_files: list[str]) -> dict:
    return _sweep_arm(tag, name, extra, data_files)


@_gpu_fn("A100-40GB", 86000)
def train_a100(name: str, args: list[str], max_hours: float, data: str = "") -> dict:
    return _train(name, args, max_hours, data)


@_gpu_fn("H100", 86000)
def train_h100(name: str, args: list[str], max_hours: float, data: str = "") -> dict:
    return _train(name, args, max_hours, data)


@_gpu_fn("A100-40GB", 5400)
def train_a100_short(name: str, args: list[str], max_hours: float, data: str = "") -> dict:
    """The same with a function timeout of 90 minutes: the cost of a hang is capped at about 3.6 USD (tests, relay checks, fine-tuning runs)."""
    return _train(name, args, max_hours, data)


@_gpu_fn("T4", 3600)
def eval_t4(name: str, ckpt: str, tasks: list[str]) -> dict:
    return _eval(name, ckpt, tasks)


@_gpu_fn("T4", 3600)
def ladder_t4(name: str, ckpt: str, label: str, extra: str = "") -> dict:
    """The evaluation ladder of tools/f1_eval.sh (quick mode, all tasks) on a checkpoint of the runs volume; data pack in /data/eval_pack (modal volume put forge-data eval_pack /eval_pack), results in /runs/<name>/ladder_<label>/."""
    import shlex

    out = f"/runs/{name}/ladder_{label}"
    cmd = ["bash", "/src/tools/f1_eval.sh", ckpt, "--quick", "--tasks", "all", "--data", "/data/eval_pack", "--forge", FORGE, "--tokenizer", "/data/eval_pack/tokenizer_f1.json", "--out", out, "--name", label, *shlex.split(extra)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    print(p.stdout[-4000:], flush=True)
    print(p.stderr[-1500:], flush=True)
    runs_vol.commit()
    return {"returncode": p.returncode, "tail": p.stdout.strip().splitlines()[-14:]}


@_gpu_fn("T4", 1800)
def checklist_t4(name: str, ckpt: str, label: str) -> dict:
    """The 20-prompt qualitative checklist of the chat fine-tune (tools/sft_data/sft_checklist.sh) on a checkpoint of the runs volume; the text goes to /runs/<name>/checklist_<label>.txt."""
    out = f"/runs/{name}/checklist_{label}.txt"
    env = dict(os.environ, FORGE_BIN=FORGE, TOK="/data/eval_pack/tokenizer_f1.json")
    p = subprocess.run(["bash", "/src/tools/sft_data/sft_checklist.sh", ckpt, out], capture_output=True, text=True, env=env)
    print(p.stdout[-1500:], p.stderr[-1500:], flush=True)
    runs_vol.commit()
    text = Path(out).read_text() if Path(out).exists() else ""
    return {"returncode": p.returncode, "chars": len(text), "head": text[:1500], "text": text}


@_gpu_fn("T4", 1800)
def probe_t4(name: str, ckpt: str, label: str, prompts: list[dict]) -> dict:
    """A held-out prompt probe (list of {"kind", "prompt", optional "system"}): each prompt greedy through `forge chat` like the checklist; the text goes to /runs/<name>/probe_<label>.txt. Used for the safety behaviour: refuse the harmful requests, help with the harmless look-alikes."""
    lines = []
    for i, p in enumerate(prompts, 1):
        if p.get("turns"):   # a multi-turn chat: the user turns are fed to `forge chat --interactive`, one per line
            cmd = [FORGE, "chat", ckpt, "--tokenizer", "/data/eval_pack/tokenizer_f1.json", "--interactive", "--tokens", "60", "--temp", "0", "--seq", "768"]
            r = subprocess.run(cmd, input="\n".join(p["turns"]) + "\n", capture_output=True, text=True)
            shown = " || ".join(p["turns"])
        else:
            cmd = [FORGE, "chat", ckpt, "--tokenizer", "/data/eval_pack/tokenizer_f1.json", "--chat", p["prompt"], "--tokens", "160", "--temp", "0", "--seq", "768"]
            if p.get("system"):
                cmd += ["--system", p["system"]]
            r = subprocess.run(cmd, capture_output=True, text=True)
            shown = p["prompt"]
        lines += [f"=== {i} [{p.get('kind', '?')}] {shown}", "    -> " + (r.stdout + r.stderr).strip(), ""]
    text = "\n".join(lines)
    Path(f"/runs/{name}/probe_{label}.txt").write_text(text)
    runs_vol.commit()
    return {"returncode": 0, "prompts": len(prompts), "chars": len(text), "text": text}


SWEEP = {"A100-40GB": sweep_a100, "H100": sweep_h100}
TRAIN = {"A100-40GB": train_a100, "H100": train_h100}
TRAIN_SHORT = {"A100-40GB": train_a100_short}


@app.local_entrypoint()
def sweep(gpu: str = "A100-40GB", arms: str = "round2a", tag: str = "", data: str = "fwedu_000.u16", dry: bool = False):
    """The proxy arms of cloud/kaggle_stage.py (ARMS[arms]) in parallel, one container each. Hard ceiling per arm 30 min."""
    import sys

    sys.path.insert(0, str(HERE))
    import kaggle_stage as ks

    spec = ks.ARMS[arms]
    tag = tag or f"{arms}_{time.strftime('%m%d_%H%M')}"
    worst = len(spec) * cost_usd(gpu, 1800)
    est = len(spec) * cost_usd(gpu, 330)
    print(f"{len(spec)} arms of {arms} on {gpu}, tag {tag}: estimated {est:.2f} USD, hard ceiling {worst:.2f} USD")
    if dry:
        return
    t0 = time.time()
    results = list(SWEEP[gpu].starmap([(tag, n, extra, [data]) for n, extra in spec]))
    dt = time.time() - t0
    for r in results:
        print(f"{r['name']:14s} val {[round(v, 4) for v in r['val'][-2:]]} {r['tok_s'] / 1000:.0f}k tok/s  {r['seconds']:.0f} s  rc {r['returncode']}  {r['extra']}")
    spent = sum(cost_usd(gpu, r["seconds"] + 60) for r in results)
    ledger({"call": "sweep", "gpu": gpu, "arms": arms, "tag": tag, "wall_s": round(dt), "est_usd": round(spent, 2), "results": [{k: r[k] for k in ("name", "val", "tok_s")} for r in results]})
    print(f"wall {dt:.0f} s, estimated cost {spent:.2f} USD (results also in the runs volume under sweeps/{tag})")


@app.local_entrypoint()
def train(name: str, gpu: str = "A100-40GB", args: str = "", max_hours: float = 16.0, approved_usd: float = 0.0, data: str = ""):
    """The long run. Refuses to start unless --approved-usd covers the worst case (max-hours at the GPU's rate). Use `modal run --detach` so that closing this window does not stop it."""
    import shlex

    worst = cost_usd(gpu, max_hours * 3600)
    print(f"train {name} on {gpu}: up to {max_hours} h = at most {worst:.2f} USD; approved {approved_usd:.2f} USD")
    if worst > approved_usd:
        raise SystemExit(f"refused: the worst case {worst:.2f} USD is above the approved {approved_usd:.2f} USD (pass --approved-usd)")
    ledger({"call": "train-start", "name": name, "gpu": gpu, "max_hours": max_hours, "worst_usd": round(worst, 2), "args": args})
    fn = TRAIN_SHORT[gpu] if (max_hours <= 1.25 and gpu in TRAIN_SHORT) else TRAIN[gpu]
    print(f"function timeout {'90 minutes' if fn is not TRAIN[gpu] else '24 hours'} ({fn.tag if hasattr(fn, 'tag') else ''})")
    res = fn.remote(name, shlex.split(args), max_hours, data)
    print(json.dumps(res, indent=1))


@app.local_entrypoint()
def ladder(name: str, ckpt: str = "", label: str = "x", extra: str = "", approved_usd: float = 0.0):
    """The evaluation ladder on a T4 (about 10 minutes): `modal run cloud/modal_forge.py::ladder --name tpu1 --label tpu1 --approved-usd 1` scores /runs/<name>/ckpt_latest.bin (or --ckpt PATH)."""
    worst = cost_usd("T4", 3600)
    if worst > approved_usd:
        raise SystemExit(f"refused: the worst case {worst:.2f} USD (one hour of T4) is above the approved {approved_usd:.2f} USD (pass --approved-usd)")
    res = ladder_t4.remote(name, ckpt or f"/runs/{name}/ckpt_latest.bin", label, extra)
    print(json.dumps(res, indent=1))
    ledger({"call": "ladder", "name": name, "label": label, "returncode": res["returncode"]})


@app.local_entrypoint()
def checklist(name: str, ckpt: str = "", label: str = "x", approved_usd: float = 0.0):
    """The qualitative checklist on a T4 (a few minutes): `modal run cloud/modal_forge.py::checklist --name sft_tpu1_human --label human --approved-usd 0.6`."""
    worst = cost_usd("T4", 1800)
    if worst > approved_usd:
        raise SystemExit(f"refused: the worst case {worst:.2f} USD (30 minutes of T4) is above the approved {approved_usd:.2f} USD (pass --approved-usd)")
    res = checklist_t4.remote(name, ckpt or f"/runs/{name}/ckpt_latest.bin", label)
    text = res.pop("text", "")
    print(json.dumps(res, indent=1))
    ledger({"call": "checklist", "name": name, "label": label, "returncode": res["returncode"]})
    if text:   # the full answers go to scratch/ and are scored by tools/sft_data/score_checklist.py (rules of the "pass if" lines; a first pass, read the answers too)
        import sys

        dest = REPO / "scratch" / f"checklist_{name}_{label}.txt"
        dest.parent.mkdir(exist_ok=True)
        dest.write_text(text, encoding="utf-8")
        sys.path.insert(0, str(REPO / "tools" / "sft_data"))
        import score_checklist as sc

        _, summ = sc.score_text(text)
        print(f"answers in {dest}\nautomatic score {summ['total']}/{summ['max']} ({summ['label']}); story {summ['story']}; flags {summ['flags'] or 'none'}")
        print("   " + "; ".join(f"{c} {s}/{m}" for c, (s, m) in summ["categories"].items()))


@app.local_entrypoint()
def probe(name: str, prompts_file: str, ckpt: str = "", label: str = "x", approved_usd: float = 0.0):
    """A held-out prompt probe on a T4 (a few minutes, about 4 cents): `modal run cloud/modal_forge.py::probe --name sft_tpu1_humansafe_2250_2e-4 --prompts-file scratch/safety_probe.json --label safety --approved-usd 0.6`; the answers go to scratch/probe_<name>_<label>.txt (scored by tools/sft_data/score_probe.py)."""
    worst = cost_usd("T4", 1800)
    if worst > approved_usd:
        raise SystemExit(f"refused: the worst case {worst:.2f} USD (30 minutes of T4) is above the approved {approved_usd:.2f} USD (pass --approved-usd)")
    prompts = json.loads(Path(prompts_file).read_text(encoding="utf-8"))
    res = probe_t4.remote(name, ckpt or f"/runs/{name}/ckpt_latest.bin", label, prompts)
    text = res.pop("text", "")
    print(json.dumps(res, indent=1))
    ledger({"call": "probe", "name": name, "label": label, "prompts": len(prompts), "returncode": res["returncode"]})
    if text:
        dest = REPO / "scratch" / f"probe_{name}_{label}.txt"
        dest.parent.mkdir(exist_ok=True)
        dest.write_text(text, encoding="utf-8")
        print(f"answers in {dest}")


@app.local_entrypoint()
def evaluate(name: str, ckpt: str, tasks: str = "hellaswag_val,arc_easy,arc_challenge,piqa,winogrande,sciq,openbookqa"):
    res = eval_t4.remote(name, ckpt, tasks.split(","))
    print(json.dumps(res, indent=1))
    ledger({"call": "eval", "name": name, "ckpt": ckpt, "result": res})



# an image that also holds the golden files and the compiled test binaries: `modal run cloud/modal_forge.py::tests --gpu A100-40GB` runs the GEMM and trainer tests on that GPU
test_image = (
    image.add_local_dir(str(REPO / "tests" / "golden" / "train_tiny"), "/src/tests/golden/train_tiny", copy=True)
    .add_local_dir(str(REPO / "tests" / "golden" / "muon"), "/src/tests/golden/muon", copy=True)
    .run_commands("cd /src && cargo test --release -p forge-gpu --test gemm_s80 --no-run 2>&1 | tail -n 2; cargo test --release -p forge-train --test tiny --test muon --no-run 2>&1 | tail -n 3")
)


def _tests(which: str) -> str:
    out = []
    for cmd in ("cargo test --release -p forge-gpu --test gemm_s80 -- --nocapture --test-threads=1", "cargo test --release -p forge-train --test tiny --test muon -- --nocapture --test-threads=1"):
        if which and which not in cmd:
            continue
        r = subprocess.run(f"cd /src && {cmd} 2>&1", shell=True, capture_output=True, text=True, timeout=1200)
        text = r.stdout
        keep = [ln for ln in text.splitlines() if re.search(r"^test |test result|TFLOPs|Ampere|sm_|panicked|error|FAILED|max .diff.", ln)]
        out.append(f"$ {cmd}" + chr(10) + chr(10).join(keep)[-6000:])
    return chr(10).join(out)


@app.function(image=test_image, gpu="A100-40GB", timeout=1500, cpu=CPU, memory=MEM_MIB, retries=0)
def tests_a100(which: str = "") -> str:
    return _tests(which)


@app.function(image=test_image, gpu="H100", timeout=1500, cpu=CPU, memory=MEM_MIB, retries=0)
def tests_h100(which: str = "") -> str:
    return _tests(which)


@app.local_entrypoint()
def tests(gpu: str = "A100-40GB", which: str = ""):
    fn = {"A100-40GB": tests_a100, "H100": tests_h100}[gpu]
    print(f"tests on {gpu}: hard ceiling 25 min = at most {cost_usd(gpu, 1500):.2f} USD")
    t0 = time.time()
    res = fn.remote(which)
    print(res)
    est = cost_usd(gpu, time.time() - t0)
    ledger({"call": "tests", "gpu": gpu, "which": which, "est_usd": round(est, 3)})
    print(f"estimated cost {est:.2f} USD")


SMOKE = {"T4": smoke_t4, "A100-40GB": smoke_a100, "H100": smoke_h100, "L4": smoke_l4}


@app.local_entrypoint()
def smoke(gpu: str = "T4", micro: int = 4, steps: int = 30, prof: bool = False):
    fn = SMOKE[gpu]
    worst = cost_usd(gpu, 900)
    print(f"smoke on {gpu}: micro {micro}, {steps} steps; hard timeout 15 min = at most {worst:.2f} USD")
    t0 = time.time()
    res = fn.remote(micro, steps, prof)
    dt = time.time() - t0
    est = cost_usd(gpu, dt)
    print(json.dumps(res, indent=1))
    ledger({"call": "smoke", "gpu": gpu, "micro": micro, "seconds": round(dt), "est_usd": round(est, 3), "result": {k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk != "tail"}) for k, v in res.items()}})
    print(f"wall {dt:.0f} s (including the image start), estimated cost {est:.2f} USD")
