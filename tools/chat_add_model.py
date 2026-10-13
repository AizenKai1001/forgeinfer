#!/usr/bin/env python3
"""chat_add_model.py --ckpt LOCAL.bin --file NAME.bin --title TEXT --note TEXT [--default] [--restart]

Copies a checkpoint to the homelab and adds an entry to ~/forgeinfer/chat_models.json (the model list of the chat page: tools/chat_proxy.py re-reads it whenever it changes, so the new model is in the page's list within
seconds and NOTHING needs a restart). The file goes to the NVMe (~/forgeinfer/models_fast: a model loads in about 2 s) when it has 30 GB free, else to the hard drive (/mnt/data/forgeinfer-data/models: 127 MB/s, a model
switch takes about 12 s); --hdd forces the hard drive. Nothing is replaced: an existing entry with the same title is updated, the others stay; --default makes the new entry the default model of the page (the user's
decision, so it is off unless asked for; the default is what the engine loads when the machine or the page starts); --restart runs ~/forgeinfer/start_chat.sh (about ten seconds without the chat page) and is only needed
to make a new DEFAULT the running model. Run on the PC; needs the ssh key of the homelab (see the memory note homelab-control-channel).
"""
import argparse
import base64
import json
import subprocess
import sys

KEY = os.path.expanduser("~/.ssh/id_ed25519")
HOST = "user@homelab"
SSH = ["ssh", "-i", KEY, "-o", "BatchMode=yes", "-o", "IdentitiesOnly=yes", "-o", "ConnectTimeout=10", HOST]
MODELS_DIR = "/mnt/data/forgeinfer-data/models"  # the big hard drive (127 MB/s)
FAST_DIR = "/home/user/forgeinfer/models_fast"      # the NVMe (2.4 GB/s)
FAST_MIN_FREE_GB = 30

REMOTE = r'''
import base64, json, os, sys
entry = json.loads(base64.b64decode(sys.argv[1]))
path = os.path.expanduser("~/forgeinfer/chat_models.json")
cfg = json.load(open(path))
cfg["models"][entry["title"]] = {"ckpt": entry["ckpt"], "note": entry["note"]}
if entry["default"]:
    cfg["default"] = entry["title"]
tmp = path + ".tmp"
json.dump(cfg, open(tmp, "w"), indent=2)
os.replace(tmp, path)
print("models:", len(cfg["models"]), "default:", cfg["default"])
'''


RESTART = r"""
import json, os, subprocess
cfg = json.load(open(os.path.expanduser("~/forgeinfer/chat_models.json")))
ck = cfg["models"][cfg["default"]]["ckpt"]
print("restarting the chat engine with", ck)
print(subprocess.run(["bash", os.path.expanduser("~/forgeinfer/start_chat.sh"), ck], capture_output=True, text=True).stdout)
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="", help="a local checkpoint file (copied with scp)")
    ap.add_argument("--remote-ckpt", default="", help="a checkpoint file that already is on the homelab (copied there with cp, for example a training run)")
    ap.add_argument("--file", required=True, help="file name on the homelab, e.g. patch3_human.bin")
    ap.add_argument("--title", required=True)
    ap.add_argument("--note", required=True)
    ap.add_argument("--default", action="store_true")
    ap.add_argument("--restart", action="store_true")
    ap.add_argument("--hdd", action="store_true", help="put the file on the hard drive even when the NVMe has room")
    a = ap.parse_args()
    free_gb = 0
    if not a.hdd:
        r = subprocess.run(SSH + [f"mkdir -p {FAST_DIR}; df --output=avail -BG {FAST_DIR} | tail -n 1"], capture_output=True, text=True)
        digits = "".join(ch for ch in r.stdout if ch.isdigit())
        free_gb = int(digits) if digits else 0
    models_dir = FAST_DIR if free_gb >= FAST_MIN_FREE_GB else MODELS_DIR
    print(f"[chat_add_model] {a.file} goes to {models_dir} ({'NVMe, ' + str(free_gb) + ' GB free' if models_dir == FAST_DIR else 'hard drive'})", flush=True)
    remote_path = f"{models_dir}/{a.file}"
    if a.remote_ckpt:
        subprocess.run(SSH + [f"cp {a.remote_ckpt} {remote_path}.part && mv {remote_path}.part {remote_path} && ls -la {remote_path}"], check=True)
    else:
        subprocess.run(["scp", "-i", KEY, "-o", "BatchMode=yes", "-o", "IdentitiesOnly=yes", "-q", a.ckpt, f"{HOST}:{remote_path}.part"], check=True)
        subprocess.run(SSH + [f"mv {remote_path}.part {remote_path} && ls -la {remote_path}"], check=True)
    entry = {"title": a.title, "ckpt": remote_path, "note": a.note, "default": a.default}
    b64 = base64.b64encode(json.dumps(entry).encode()).decode()
    subprocess.run(SSH + ["python3", "-", b64], input=REMOTE.encode(), check=True)
    if a.restart:   # the proxy reads the model list once at its start: restart engine and proxy with the default model of the list
        subprocess.run(SSH + ["python3", "-"], input=RESTART.encode(), check=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
