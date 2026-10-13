#!/usr/bin/env python3
"""probe_cards.py REPO [REPO ...]: license, gating and file layout of Hugging Face datasets, straight from the hub API (no login).
Prints: id, license (cardData.license), gated flag, tags that start with 'license:', total size of the parquet/json files, and the file list (first 40).
Run on the homelab with ~/forgeinfer/venv/bin/python (stdlib only)."""
import json
import re
import sys
import urllib.request


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "forgeinfer-sft-data"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def readme(repo):
    req = urllib.request.Request(f"https://huggingface.co/datasets/{repo}/raw/main/README.md", headers={"User-Agent": "forgeinfer-sft-data"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read().decode("utf-8", "replace")


args = [a for a in sys.argv[1:] if not a.startswith("--")]
show_readme = "--readme" in sys.argv
for repo in args:
    if show_readme:
        try:
            txt = readme(repo)
            head = txt.split("---")[1] if txt.startswith("---") else ""
            print(f"== README {repo}: frontmatter license lines: {[l for l in head.splitlines() if 'licen' in l.lower()]}")
            for i, line in enumerate(txt.splitlines()):
                if re.search(r"licen[cs]e|terms of use|OpenAI|gpt-?[34]|llama|apache|cc-by|odc", line, re.I):
                    print("     ", i, line.strip()[:260])
        except Exception as e:  # noqa: BLE001
            print(f"== README {repo}: ERROR {e}")
        continue
    try:
        info = get(f"https://huggingface.co/api/datasets/{repo}?blobs=true")
    except Exception as e:  # noqa: BLE001
        print(f"== {repo}: ERROR {e}")
        continue
    card = info.get("cardData") or {}
    lic = card.get("license")
    tags = [t for t in info.get("tags", []) if t.startswith("license")]
    sib = info.get("siblings", [])
    data_files = [s for s in sib if s["rfilename"].endswith((".parquet", ".jsonl", ".json", ".json.gz", ".jsonl.gz", ".csv", ".arrow", ".zst"))]
    total = sum((s.get("size") or 0) for s in data_files)
    print(f"== {repo}: license={lic!r} tags={tags} gated={info.get('gated')} private={info.get('private')} downloads={info.get('downloads')} data_files={len(data_files)} total={total / 1e6:.0f} MB sha={info.get('sha', '')[:8]}")
    for s in data_files[:40]:
        print(f"     {s['rfilename']}  {(s.get('size') or 0) / 1e6:.1f} MB")
    if len(data_files) > 40:
        print(f"     ... {len(data_files) - 40} more")
