"""Golden dequantization vectors for our own quantization code, recorded from real GGUF files with the reference reader (the `gguf` Python package, an ORACLE only).

Run on the homelab:   ~/forgeinfer/venv/bin/python tools/gen_quant_goldens.py
For every quantization type found in the oracle GGUF files it takes the first tensor of that type and writes
    tests/golden/quant_<TYPE>.in    raw bytes of the first 64 blocks (exactly as stored in the file)
    tests/golden/quant_<TYPE>.f32   the dequantized values as little-endian float32 (blocks x values-per-block)
    tests/golden/quant_index.json   {type: {"block_bytes", "block_values", "blocks", "source_tensor"}}
Real blocks (not random bytes) keep the scale fields sane, so the comparison is meaningful.
"""
import json
import os
import sys

import numpy as np
from gguf import GGUFReader
from gguf.quants import dequantize
from gguf.constants import GGML_QUANT_SIZES

ROOT = os.path.expanduser("~/forgeinfer/oracle/models")
FILES = [
    "bartowski__SmolLM2-135M-Instruct-GGUF/SmolLM2-135M-Instruct-f16.gguf",
    "bartowski__SmolLM2-135M-Instruct-GGUF/SmolLM2-135M-Instruct-Q8_0.gguf",
    "bartowski__SmolLM2-135M-Instruct-GGUF/SmolLM2-135M-Instruct-Q6_K.gguf",
    "bartowski__SmolLM2-135M-Instruct-GGUF/SmolLM2-135M-Instruct-Q4_K_M.gguf",
    "bartowski__SmolLM2-135M-Instruct-GGUF/SmolLM2-135M-Instruct-Q4_0.gguf",
    "Qwen__Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf",
]
BLOCKS = 64


def main() -> int:
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests", "golden")
    os.makedirs(out_dir, exist_ok=True)
    index = {}
    for rel in FILES:
        r = GGUFReader(os.path.join(ROOT, rel))
        for t in r.tensors:
            name = t.tensor_type.name
            if name in index or name == "F32":
                continue
            block_values, block_bytes = GGML_QUANT_SIZES[t.tensor_type]
            raw = np.asarray(t.data).reshape(-1).view(np.uint8)
            n_blocks = min(BLOCKS, raw.size // block_bytes)
            chunk = raw[: n_blocks * block_bytes].reshape(n_blocks, block_bytes)
            vals = dequantize(chunk, t.tensor_type).astype("<f4")
            chunk.tofile(os.path.join(out_dir, f"quant_{name}.in"))
            vals.reshape(-1).tofile(os.path.join(out_dir, f"quant_{name}.f32"))
            index[name] = {"block_bytes": int(block_bytes), "block_values": int(block_values), "blocks": int(n_blocks), "source_tensor": t.name, "source_file": os.path.basename(rel),
                           "min": float(vals.min()), "max": float(vals.max())}
            print(f"{name:6s} {n_blocks} blocks x {block_bytes} bytes -> {block_values} values each   from {t.name} in {os.path.basename(rel)}   range [{vals.min():.4f}, {vals.max():.4f}]")
    with open(os.path.join(out_dir, "quant_index.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(index, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
