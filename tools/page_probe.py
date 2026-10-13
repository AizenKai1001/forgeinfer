#!/usr/bin/env python3
"""page_probe.py PROMPTS.json OUT.txt [--model "Title of the model"] [--n 1] [--restore]

Runs a prompt file (the format of scratch/safety_probe.json: [{"kind", "prompt"}]) through the CHAT PAGE of the homelab (the proxy on 127.0.0.1:8191 there, reached over ssh), with the page's own sampling settings (temperature 0.7,
top-k 40, top-p 0.95, repeat penalty 1.1), and writes the answers in the probe format (`=== N [kind] prompt` / `    -> answer`) so that tools/sft_data/score_probe.py can score them. Greedy probes (cloud/modal_forge.py::probe) show
the best answer of a model; this shows what a person sees on the page. --model switches the page to that model first (the exact title in the model list) and --restore switches back to the model that was current before.
With --n 3 every prompt is asked three times and the three answers are written as three sections. Run on the PC (needs the ssh key of the homelab).
"""
import os
import argparse
import json
import subprocess
import sys

KEY = os.path.expanduser("~/.ssh/id_ed25519")
HOST = "user@homelab"
SSH = ["ssh", "-i", KEY, "-o", "BatchMode=yes", "-o", "IdentitiesOnly=yes", "-o", "ConnectTimeout=10", HOST]

REMOTE = r'''
import json, urllib.request
cfg = json.load(open("/tmp/page_probe_cfg.json"))
base = "http://127.0.0.1:8191"
def get(p):
    return json.load(urllib.request.urlopen(base + p, timeout=30))
def post(p, d, t=120):
    req = urllib.request.Request(base + p, data=json.dumps(d).encode(), headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=t))
before = get("/api/models")["current"]
if cfg["model"] and cfg["model"] != before:
    post("/api/select", {"name": cfg["model"]}, 180)
now = get("/api/models")["current"]
out = []
for p in cfg["prompts"]:
    answers = []
    for _ in range(cfg["n"]):
        try:
            r = post("/v1/chat/completions", {"model": "forge-1", "messages": ([{"role": "system", "content": cfg["system"]}] if cfg["system"] else []) + [{"role": "user", "content": p["prompt"]}], "max_tokens": cfg.get("max_tokens", 160), **cfg["extra"]})
            answers.append(r["choices"][0]["message"]["content"])
        except Exception as e:
            answers.append("[error] " + str(e))
    out.append(answers)
if cfg["restore"] and now != before:
    post("/api/select", {"name": before}, 180)
print(json.dumps({"model": now, "before": before, "answers": out}))
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompts")
    ap.add_argument("out")
    ap.add_argument("--model", default="")
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--restore", action="store_true")
    ap.add_argument("--temp", type=float, default=None, help="sampling temperature (default: the page's own 0.7)")
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--max-tokens", type=int, default=160, help="answer length limit (long worked arithmetic needs 400)")
    ap.add_argument("--rep", type=float, default=None, help="repetition_penalty of the request (the page engine default is 1.1)")
    ap.add_argument("--system", default="", help="a system prompt for every question")
    a = ap.parse_args()
    prompts = json.load(open(a.prompts, encoding="utf-8"))
    extra = {}
    if a.temp is not None:
        extra["temperature"] = a.temp
    if a.top_p is not None:
        extra["top_p"] = a.top_p
    if a.rep is not None:
        extra["repetition_penalty"] = a.rep
    cfg = {"model": a.model, "n": a.n, "restore": a.restore, "prompts": prompts, "extra": extra, "system": a.system, "max_tokens": a.max_tokens}
    subprocess.run(SSH + ["cat > /tmp/page_probe_cfg.json"], input=json.dumps(cfg).encode(), check=True)
    r = subprocess.run(SSH + ["python3", "-"], input=REMOTE.encode(), capture_output=True, check=True)
    res = json.loads(r.stdout.decode().strip().splitlines()[-1])
    lines, k = [], 0
    for p, answers in zip(prompts, res["answers"]):
        for ans in answers:
            k += 1
            lines += [f"=== {k} [{p.get('kind', '?')}] {p['prompt']}", "    -> " + ans.strip(), ""]
    open(a.out, "w", encoding="utf-8").write("\n".join(lines))
    print(f"model on the page: {res['model']} (before: {res['before']}); {k} answers -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
