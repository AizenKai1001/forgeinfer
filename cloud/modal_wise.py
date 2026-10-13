"""Weight-space interpolation between a base checkpoint and its fine-tune (WiSE-FT, Wortsman et al. 2022) on a CPU container of Modal: the result keeps most of the chat behaviour and loses less of the base model's
general ability (the 'alignment tax'). Reads and writes the volume forge-runs, nothing is uploaded from the PC; the cost is a few cents of CPU.

  modal run cloud/modal_wise.py::interp --base tpu1 --ft sft_tpu1_human_2250_2e-4 --alphas 0.3,0.5,0.7
writes /runs/wise_<ft>_a030/ckpt_latest.bin and so on: params = (1 - alpha) * base + alpha * fine-tune (alpha 0 = base, 1 = fine-tune); the optimizer sections are copied from the fine-tune (they are not used for evaluation or chat).
Then evaluate as usual:  modal run cloud/modal_forge.py::ladder --name wise_sft_tpu1_human_2250_2e-4_a050 --label lad --approved-usd 1   and   ::checklist ...
The checkpoint format is the forgeckpt1 file of tpu/forge_jax.py: one JSON header line, then n_params float32 values of parameters, the same count for the first and for the second Adam moment.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
app = modal.App("forge1-wise")
runs_vol = modal.Volume.from_name("forge-runs", create_if_missing=True)
image = modal.Image.debian_slim(python_version="3.12").pip_install("numpy")


@app.function(image=image, cpu=2, memory=8192, timeout=1800, volumes={"/runs": runs_vol}, retries=0)
def interp_remote(base: str, ft: str, alphas: list[float]) -> list[dict]:
    import numpy as np

    runs_vol.reload()
    pb, pf = Path(f"/runs/{base}/ckpt_latest.bin"), Path(f"/runs/{ft}/ckpt_latest.bin")
    out = []
    with open(pb, "rb") as fb, open(pf, "rb") as ff:
        hb, hf = fb.readline(), ff.readline()
        jb, jf = json.loads(hb), json.loads(hf)
        n = jf["n_params"]
        keys = ["n_layer", "n_embd", "n_head", "n_kv", "head_dim", "n_ff", "vocab", "qk_norm", "relu2", "lambdas", "skips", "untied", "ve_layers"]
        assert jb["n_params"] == n and all(jb["cfg"].get(k) == jf["cfg"].get(k) for k in keys), "the checkpoints do not have the same structure"
        offb, offf = len(hb), len(hf)
    for a in alphas:
        name = f"wise_{ft}_a{int(round(a * 100)):03d}"
        d = Path(f"/runs/{name}")
        d.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        chunk = 8_000_000   # float32 values per chunk
        with open(pb, "rb") as fb, open(pf, "rb") as ff, open(d / "ckpt_latest.bin", "wb") as fo:
            fb.seek(offb)
            ff.seek(offf)
            fo.write(hf)
            done = 0
            while done < n:
                k = min(chunk, n - done)
                xb = np.frombuffer(fb.read(4 * k), dtype="<f4")
                xf = np.frombuffer(ff.read(4 * k), dtype="<f4")
                fo.write(((1.0 - a) * xb + a * xf).astype("<f4").tobytes())
                done += k
            # the two Adam sections: copied from the fine-tune
            fb.seek(offb + 4 * n)
            rest = 2 * 4 * n
            ff.seek(offf + 4 * n)
            while rest > 0:
                buf = ff.read(min(64 << 20, rest))
                fo.write(buf)
                rest -= len(buf)
        size = (d / "ckpt_latest.bin").stat().st_size
        out.append({"name": name, "alpha": a, "bytes": size, "seconds": round(time.time() - t0, 1)})
        runs_vol.commit()
    return out


@app.local_entrypoint()
def interp(base: str, ft: str, alphas: str = "0.3,0.5,0.7"):
    a = [float(x) for x in alphas.split(",")]
    res = interp_remote.remote(base, ft, a)
    print(json.dumps(res, indent=1))
    with open(HERE / "ledger.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"time": time.strftime("%Y-%m-%d %I:%M %p"), "call": "wise-interp", "base": base, "ft": ft, "alphas": a, "est_usd": 0.02}) + "\n")
