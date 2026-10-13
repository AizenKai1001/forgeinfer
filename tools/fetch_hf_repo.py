#!/usr/bin/env python3
"""fetch_hf_repo.py REPO OUT_DIR [--only SUBSTR[,SUBSTR...]] [--skip SUBSTR[,...]] [--no-verify]

Downloads the files of a public Hugging Face model repository with curl (resumable: an interrupted run continues where it stopped),
one file at a time, into OUT_DIR, and checks the SHA-256 of every LFS file against the hash the Hub reports.
    python3 tools/fetch_hf_repo.py Qwen/Qwen3.5-27B /mnt/data/forgeinfer-hf/Qwen__Qwen3.5-27B
Nothing is deleted; a partial file is kept as NAME.part until it is complete.
"""
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.request

repo, out = sys.argv[1], sys.argv[2]
only = skip = []
verify = True
args = sys.argv[3:]
i = 0
while i < len(args):
    if args[i] == "--only":
        only = args[i + 1].split(",")
        i += 2
    elif args[i] == "--skip":
        skip = args[i + 1].split(",")
        i += 2
    elif args[i] == "--no-verify":
        verify = False
        i += 1
    else:
        raise SystemExit("unknown argument " + args[i])

os.makedirs(out, exist_ok=True)
with urllib.request.urlopen(f"https://huggingface.co/api/models/{repo}/tree/main", timeout=60) as r:
    tree = [f for f in json.load(r) if f.get("type") == "file"]
tree.sort(key=lambda f: f["path"])
todo = [f for f in tree if (not only or any(s in f["path"] for s in only)) and not any(s in f["path"] for s in skip)]
total = sum(f["size"] for f in todo)
print(f"{repo}: {len(todo)} files, {total / 1e9:.1f} GB -> {out}", flush=True)
done_bytes = 0
t0 = time.time()
for f in todo:
    name, size = f["path"], f["size"]
    dest = os.path.join(out, name)
    if os.path.exists(dest) and os.path.getsize(dest) == size:
        print(f"have {name}", flush=True)
    else:
        os.makedirs(os.path.dirname(dest) or out, exist_ok=True)
        part = dest + ".part"
        url = f"https://huggingface.co/{repo}/resolve/main/{name}"
        for attempt in range(8):
            rc = subprocess.call(["curl", "-L", "-sS", "--fail", "-C", "-", "--retry", "3", "--connect-timeout", "30", "-o", part, url])
            if rc == 0 and os.path.exists(part) and os.path.getsize(part) == size:
                os.rename(part, dest)
                break
            print(f"retry {attempt + 1} for {name} (curl exit {rc})", flush=True)
            time.sleep(5)
        else:
            raise SystemExit(f"could not download {name}")
    done_bytes += size
    el = time.time() - t0
    print(f"ok   {name}  {size / 1e9:.2f} GB   total {done_bytes / 1e9:.1f}/{total / 1e9:.1f} GB  {el / 60:.1f} min", flush=True)
    oid = (f.get("lfs") or {}).get("oid")
    if verify and oid:
        h = hashlib.sha256()
        with open(dest, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 22), b""):
                h.update(chunk)
        print(("sha256 OK   " if h.hexdigest() == oid else "sha256 BAD  ") + name, flush=True)
print("all done", flush=True)
