#!/usr/bin/env python3
"""hf_sizes.py REPO [PATH] [--type datasets]: list a public Hugging Face repository directory (recursively, size per entry and the total), no token needed.   hf_sizes.py HuggingFaceTB/smollm-corpus cosmopedia-v2"""
import json
import sys
import urllib.request

repo = sys.argv[1]
path = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else ""
kind = "datasets"
url = f"https://huggingface.co/api/{kind}/{repo}/tree/main/{path}?recursive=true&expand=false"
items = []
while url:
    req = urllib.request.Request(url, headers={"User-Agent": "forgeinfer-tools"})
    with urllib.request.urlopen(req, timeout=60) as r:
        items += json.load(r)
        link = r.headers.get("Link", "")
    url = None
    for part in link.split(","):
        if 'rel="next"' in part:
            url = part.split(";")[0].strip().strip("<>")
files = [i for i in items if i.get("type") == "file"]
total = sum(i.get("size", 0) for i in files)
print(f"{repo}/{path}: {len(files)} files, {total / 1e9:.1f} GB")
for i in files[:int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3].isdigit() else 6]:
    print("  ", i["path"], f'{i.get("size", 0) / 1e6:.0f} MB')
