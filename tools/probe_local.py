#!/usr/bin/env python3
"""probe_local.py CKPT_ON_HOMELAB PROMPTS.json OUT.txt

The greedy probe of cloud/modal_forge.py::probe, run on the homelab GPU instead of a Modal T4 (free): every prompt of PROMPTS.json ({"kind", "prompt", optional "system"}, optional "turns" for a multi-turn chat)
goes through `forge chat` with --temp 0 and the answers are written in the probe format (`=== N [kind] prompt` / `    -> answer`) for score_probe.py, score_skills.py. Needs the GPU to be free enough for the 1.3 GB
inference run next to the chat engine (it is not while a trainer holds 6.3 GB: wait for the run to end). Run on the PC (ssh key of the homelab). About 3 seconds per prompt.
  python tools/probe_local.py /mnt/data/forgeinfer-data/models/tpu1_patch9_skills.bin scratch/arith_probe.json scratch/probe_local_patch9_arith.txt 400   (the last argument: answer limit in tokens, default 160)
"""
import os
import json
import subprocess
import sys

KEY = os.path.expanduser("~/.ssh/id_ed25519")
HOST = "user@homelab"
SSH = ["ssh", "-i", KEY, "-o", "BatchMode=yes", "-o", "IdentitiesOnly=yes", "-o", "ConnectTimeout=10", HOST]

REMOTE = r'''
import json, subprocess, sys
cfg = json.load(open("/tmp/probe_local_cfg.json"))
F = "/home/user/forgeinfer/target-dev3/release/forge"
TOK = "/home/user/forgeinfer/data/tokenizer_f1.json"
out = []
for p in cfg["prompts"]:
    if p.get("turns"):
        cmd = ["nice", "-n", "5", F, "chat", cfg["ckpt"], "--tokenizer", TOK, "--interactive", "--tokens", "60", "--temp", "0", "--seq", "768"]
        r = subprocess.run(cmd, input="\n".join(p["turns"]) + "\n", capture_output=True, text=True)
    else:
        cmd = ["nice", "-n", "5", F, "chat", cfg["ckpt"], "--tokenizer", TOK, "--chat", p["prompt"], "--tokens", str(cfg.get("tokens", 160)), "--temp", "0", "--seq", "768"]
        if p.get("system"):
            cmd += ["--system", p["system"]]
        r = subprocess.run(cmd, capture_output=True, text=True)
    out.append((r.stdout + r.stderr).strip())
print(json.dumps(out))
'''


def main():
    ckpt, prompts_file, out = sys.argv[1:4]
    tokens = int(sys.argv[4]) if len(sys.argv) > 4 else 160
    prompts = json.load(open(prompts_file, encoding="utf-8"))
    subprocess.run(SSH + ["cat > /tmp/probe_local_cfg.json"], input=json.dumps({"ckpt": ckpt, "prompts": prompts, "tokens": tokens}).encode(), check=True)
    r = subprocess.run(SSH + ["python3", "-"], input=REMOTE.encode(), capture_output=True, check=True)
    answers = json.loads(r.stdout.decode().strip().splitlines()[-1])
    lines = []
    for i, (p, a) in enumerate(zip(prompts, answers), 1):
        shown = " || ".join(p["turns"]) if p.get("turns") else p["prompt"]
        lines += [f"=== {i} [{p.get('kind', '?')}] {shown}", "    -> " + a, ""]
    open(out, "w", encoding="utf-8").write("\n".join(lines))
    print(f"{len(answers)} answers -> {out}")


if __name__ == "__main__":
    main()
