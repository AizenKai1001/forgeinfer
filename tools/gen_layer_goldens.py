"""Per-layer golden tensors for testing single decode stages (norm+QKV+RoPE, attention, O-projection, norm+gate/up+SwiGLU, down-projection) of the engine on REAL data.
    ~/forgeinfer/venv/bin/python tools/gen_layer_goldens.py
Runs the numpy specification (tools/ref_forward_numpy.py, float32) on a 13-token prompt of SmolLM2-135M and, for layers 0, 7, 15 and 29, writes every intermediate tensor of the LAST token plus the
K/V of all 13 positions (the cache the decode step reads) as little-endian float32 files into tests/golden/layer_smollm2/, with a manifest of shapes. The decode contract of layer l at position t:
    x_in[t] --norm--> h[t] --Wq,Wk,Wv,rope--> q[t], k[0..t], v[0..t] --attention--> o[t] --Wo,+x_in--> x_mid[t] --norm--> h2[t] --Wg,Wu,swiglu--> a[t] --Wd,+x_mid--> x_out[t]
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ref_forward_numpy as F

LAYERS = (0, 7, 15, 29)


def main():
    cfg, w = F.load("smollm2")
    gold = json.load(open(os.path.join(F.GOLD, "fwd_smollm2.json")))
    prompt = gold["prompts"][3]                                               # 13 tokens
    ids = prompt["ids"]
    trace = {}
    F.forward(cfg, w, ids, False, trace=trace)
    out_dir = os.path.join(F.GOLD, "layer_smollm2")
    os.makedirs(out_dir, exist_ok=True)
    T = len(ids)
    manifest = {"model": "SmolLM2-135M-Instruct", "prompt_ids": ids, "position": T - 1, "layers": {}, "note": "float32 little endian; per-token tensors are those of the last position; k and v hold all positions [T, kv_heads, head_dim] after RoPE"}
    for l in LAYERS:
        tr = trace[l]
        entries = {}
        for name, arr in tr.items():
            a = arr if name in ("k", "v") else arr[T - 1]
            a = np.ascontiguousarray(a, dtype="<f4")
            fn = f"L{l}_{name}.f32"
            a.tofile(os.path.join(out_dir, fn))
            entries[name] = {"file": fn, "shape": list(a.shape)}
        manifest["layers"][str(l)] = entries
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(manifest, f, indent=1)
    total = sum(os.path.getsize(os.path.join(out_dir, fn)) for fn in os.listdir(out_dir))
    print(f"wrote {len(os.listdir(out_dir))} files, {total / 1024:.0f} KiB, layers {LAYERS}, position {T - 1}")


if __name__ == "__main__":
    main()
