#!/usr/bin/env python3
"""fetch_cards.py REPO [REPO ...]: save the dataset card (README.md) and the hub metadata (license, gating, tags, file list) of public Hugging Face datasets
to /mnt/data/forgeinfer-data/sft_work/cards/<owner>__<name>.{md,json} and print the license facts (cardData.license, gated flag, license tags, size of the data files).
No login: a gated dataset (needs the user to accept terms) is reported as gated and skipped.  Run on the homelab with ~/forgeinfer/venv/bin/python (stdlib only)."""
import json
import os
import sys
import urllib.request

OUT = "/mnt/data/forgeinfer-data/sft_work/cards"
os.makedirs(OUT, exist_ok=True)
UA = {"User-Agent": "forgeinfer-sft-data"}


def get(url, as_json=False):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        data = r.read()
    return json.loads(data) if as_json else data.decode("utf-8", "replace")


for repo in sys.argv[1:]:
    base = repo.replace("/", "__")
    try:
        info = get(f"https://huggingface.co/api/datasets/{repo}?blobs=true", True)
    except Exception as e:  # noqa: BLE001
        print(f"== {repo}: API ERROR {e}")
        continue
    json.dump(info, open(os.path.join(OUT, base + ".json"), "w"), indent=1)
    card = info.get("cardData") or {}
    tags = [t for t in info.get("tags", []) if t.startswith("license")]
    sib = info.get("siblings", [])
    data_files = [s for s in sib if s["rfilename"].endswith((".parquet", ".jsonl", ".json", ".json.gz", ".jsonl.gz", ".csv", ".arrow", ".zst"))]
    total = sum((s.get("size") or 0) for s in data_files)
    print(f"== {repo}: license={card.get('license')!r} tags={tags} gated={info.get('gated')} private={info.get('private')} data_files={len(data_files)} total={total / 1e6:.0f} MB downloads={info.get('downloads')} sha={info.get('sha', '')[:8]} lastModified={info.get('lastModified')}")
    try:
        md = get(f"https://huggingface.co/datasets/{repo}/raw/main/README.md")
        open(os.path.join(OUT, base + ".md"), "w").write(md)
        print(f"   README saved: {len(md)} chars")
    except Exception as e:  # noqa: BLE001
        print(f"   README ERROR {e}")
