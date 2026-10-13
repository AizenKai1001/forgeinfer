"""Golden forward-pass outputs of a reference model (Hugging Face transformers + PyTorch CPU float32, an ORACLE only) for our own engine's tests.

Run on the homelab:   ~/forgeinfer/venv/bin/python tools/gen_forward_goldens.py [smollm2|qwen3]
Writes into tests/golden/ :
  fwd_<name>.json             prompts, token ids, the top-20 next-token logits per prompt, 48 greedy tokens, and the file names of the raw dumps
  fwd_<name>_p0_logits.f32    ALL-position logits of prompt 0   [T, vocab]  float32 little endian
  fwd_<name>_p0_hidden.f32    hidden states of prompt 0         [layers+1, T, hidden] float32 (index 0 = embedding output, index L = output of the last block BEFORE the final norm, then one more
                              slice with the final-norm output): shape [layers+2, T, hidden]
  fwd_<name>_pK_last.f32      last-position logits of prompt K  [vocab] float32
"""
import json
import os
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = os.path.expanduser("~/forgeinfer/oracle/models")
MODELS = {"smollm2": "HuggingFaceTB__SmolLM2-135M-Instruct", "qwen3": "Qwen__Qwen3-0.6B", "qwen35": "Qwen__Qwen3.5-0.8B"}
PROMPTS = {
    "smollm2": ["The capital of France is", "def fibonacci(n):", "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n", "Once upon a time, in a small village by the sea,",
                "1 2 3 4 5 6 7 8", "Hello"],
    "qwen3": ["The capital of France is", "def fibonacci(n):", "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n", "Once upon a time, in a small village by the sea,",
              "1 2 3 4 5 6 7 8", "Hello"],
}
PROMPTS["qwen35"] = PROMPTS["qwen3"] + ["The quick brown fox jumps over the lazy dog. " * 12]          # the long prompt (about 120 tokens) crosses the 64-token chunk size of the delta rule


def main() -> int:
    name = sys.argv[1] if len(sys.argv) > 1 else "smollm2"
    path = os.path.join(ROOT, MODELS[name])
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests", "golden")
    os.makedirs(out_dir, exist_ok=True)
    torch.set_num_threads(4)
    tok = AutoTokenizer.from_pretrained(path)
    try:
        model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32)
    model.eval()
    record = {"name": name, "model": MODELS[name], "config": json.load(open(os.path.join(path, "config.json"))), "prompts": []}
    for k, text in enumerate(PROMPTS[name]):
        ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids
        with torch.no_grad():
            out = model(ids, output_hidden_states=True)
            logits = out.logits[0].float().numpy()                                  # [T, V]
            if k == 0:
                hs = [h[0].float().numpy() for h in out.hidden_states]               # embedding, each block output ..., the LAST entry is already after the final norm in HF
                np.stack(hs).astype("<f4").tofile(os.path.join(out_dir, f"fwd_{name}_p0_hidden.f32"))
                logits.astype("<f4").tofile(os.path.join(out_dir, f"fwd_{name}_p0_logits.f32"))
            logits[-1].astype("<f4").tofile(os.path.join(out_dir, f"fwd_{name}_p{k}_last.f32"))
            gen = model.generate(ids, max_new_tokens=48, do_sample=False, pad_token_id=tok.eos_token_id or 0)[0][ids.shape[1]:].tolist()
        last = logits[-1]
        top = np.argsort(-last)[:20]
        record["prompts"].append({"text": text, "ids": ids[0].tolist(), "top20": [[int(i), float(last[i])] for i in top], "greedy48": gen, "last_logits_file": f"fwd_{name}_p{k}_last.f32"})
        print(name, k, repr(text[:40]), "ids", ids.shape[1], "top1", int(top[0]), repr(tok.decode([int(top[0])])), flush=True)
    record["hidden_note"] = "HF output_hidden_states: [embeddings, block1..blockL]; the last entry has the final norm applied. Shape [L+1, T, H]."
    with open(os.path.join(out_dir, f"fwd_{name}.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(record, f, ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
