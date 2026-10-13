#!/usr/bin/env python3
"""page_eval.py "Model title on the page" TAG [--default "Title of the model to switch back to"]

The whole page-side evaluation of one chat model, with the page's configuration (system prompt "You are a helpful assistant.", temperature 0.5), about 12 minutes on the homelab GPU:
  everyday   the 61 everyday requests (scratch/everyday60.json): how many are answered without a refusal marker
  harm A/B   the held-out harmful requests and harmless look-alikes (scratch/safety_probe.json, safety_probe_b.json), 2 samples per prompt
  arithmetic the 100-item arithmetic probe (scratch/arith_probe.json), 1 sample per prompt
  skills     the single-turn items of probe C (scratch/skills_probe.json)
  checklist  the 20-prompt checklist through the page (scratch/page_checklist.py, 38 points)
Prints one summary block and leaves the page on --default (default: the model that was current before). Needs the model in the page's list (tools/chat_add_model.py) and a restarted proxy.
Run on the PC (ssh key of the homelab); files are written as scratch/pe_<TAG>_*.txt.
"""
import argparse
import json
import subprocess
import sys

SYS = "You are a helpful assistant."
PY = sys.executable


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.stdout + r.stderr


def probe(model, prompts, out, n, temp="0.5", extra=()):
    return run([PY, "tools/page_probe.py", prompts, out, "--model", model, "--n", str(n), "--temp", temp, "--system", SYS, *extra])


def line(cmd_out, idx):
    ls = [x for x in cmd_out.splitlines() if x.strip()]
    return ls[idx].strip() if len(ls) > idx else "?"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("tag")
    ap.add_argument("--default", default="Forge-1 (TPU model, 45,000 steps, chat-tuned)")
    a = ap.parse_args()
    t = a.tag
    res = {}
    probe(a.model, "scratch/everyday60.json", f"scratch/pe_{t}_everyday.txt", 1)
    res["everyday"] = line(run([PY, "tools/sft_data/score_probe.py", f"scratch/pe_{t}_everyday.txt"]), 1)
    probe(a.model, "scratch/safety_probe.json", f"scratch/pe_{t}_A.txt", 2)
    res["harm A"] = line(run([PY, "tools/sft_data/score_probe.py", f"scratch/pe_{t}_A.txt"]), 1)
    probe(a.model, "scratch/safety_probe_b.json", f"scratch/pe_{t}_B.txt", 2)
    res["harm B"] = line(run([PY, "tools/sft_data/score_probe.py", f"scratch/pe_{t}_B.txt"]), 1)
    probe(a.model, "scratch/arith_probe.json", f"scratch/pe_{t}_arith.txt", 1, extra=("--rep", "1.0", "--max-tokens", "400"))   # what the page sends for messages with digits
    out = run([PY, "tools/sft_data/score_skills.py", "scratch/arith_probe.json", f"scratch/pe_{t}_arith.txt"])
    res["arithmetic"] = next((x.strip() for x in out.splitlines() if "total" in x), "?")
    items = json.load(open("scratch/skills_probe.json", encoding="utf-8"))
    single = [i for i in items if not i.get("turns")]
    json.dump(single, open(f"scratch/pe_{t}_skills_prompts.json", "w", encoding="utf-8"))
    probe(a.model, f"scratch/pe_{t}_skills_prompts.json", f"scratch/pe_{t}_skills.txt", 1)
    out = run([PY, "tools/sft_data/score_skills.py", f"scratch/pe_{t}_skills_prompts.json", f"scratch/pe_{t}_skills.txt"])
    res["skills (22 single-turn items)"] = next((x.strip() for x in out.splitlines() if "total" in x), "?")
    out = run([PY, "scratch/page_checklist.py", a.model, SYS, "0.5", f"pe_{t}_chk"])
    res["checklist (38)"] = out.strip().splitlines()[-1][:200] if out.strip() else "?"
    # back to the default model
    run([PY, "tools/page_probe.py", "scratch/everyday_prompts.json", "scratch/pe_tmp.txt", "--model", a.default, "--n", "1", "--temp", "0.5"])
    print(f"=== page evaluation of: {a.model}")
    for k, v in res.items():
        print(f"{k:32s} {v}")


if __name__ == "__main__":
    main()
