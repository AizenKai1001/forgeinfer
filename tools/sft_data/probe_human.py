#!/usr/bin/env python3
"""probe_human.py REPO [REPO ...] [--grep REGEX]: hub facts of Hugging Face datasets for the human-written-only chat mix (see ForgeInfer - SFT Data Mix (human-written only) in the research notes).

For every repo: the API record (license from cardData and the license: tags, gating, size of the data files, last modification), the card (README.md) saved to
/mnt/data/forgeinfer-data/sft_work_human/cards/<owner>__<name>.{md,json}, and the card lines that say who wrote the data (default regex: human, written, annotat, crowd, volunteer, gpt, openai, llm,
language model, synthetic, generated, translat, machine, licen).  No login: a gated dataset is reported as gated.  Run on the homelab with ~/forgeinfer/venv/bin/python (stdlib only).
"""
import json
import os
import re
import sys
import urllib.request

OUT = os.environ.get("HUMAN_CARDS", "/mnt/data/forgeinfer-data/sft_work_human/cards")
os.makedirs(OUT, exist_ok=True)
UA = {"User-Agent": "forgeinfer-sft-human"}
DEFAULT = r"human|written|annotat|crowd|volunteer|gpt|openai|\bllm|language model|synthetic|generated|translat|machine|licen|employee|professional|worker|turk"


def get(url, as_json=False):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        data = r.read()
    return json.loads(data) if as_json else data.decode("utf-8", "replace")


args = [a for a in sys.argv[1:]]
rx = re.compile(DEFAULT, re.I)
if "--grep" in args:
    k = args.index("--grep")
    rx = re.compile(args[k + 1], re.I)
    del args[k : k + 2]
show = 14
for repo in args:
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
    data_files = [s for s in sib if s["rfilename"].endswith((".parquet", ".jsonl", ".json", ".json.gz", ".jsonl.gz", ".csv", ".arrow", ".zst", ".tsv", ".txt", ".gz", ".7z", ".zip", ".tar"))]
    total = sum((s.get("size") or 0) for s in data_files)
    print(f"== {repo}: license={card.get('license')!r} tags={tags} gated={info.get('gated')} downloads={info.get('downloads')} data_files={len(data_files)} total={total / 1e6:.0f} MB lastModified={info.get('lastModified')}")
    for s in data_files[:show]:
        print(f"     {s['rfilename']}  {(s.get('size') or 0) / 1e6:.1f} MB")
    if len(data_files) > show:
        print(f"     ... {len(data_files) - show} more files")
    try:
        md = get(f"https://huggingface.co/datasets/{repo}/raw/main/README.md")
        open(os.path.join(OUT, base + ".md"), "w", encoding="utf-8").write(md)
    except Exception as e:  # noqa: BLE001
        print(f"   README ERROR {e}")
        continue
    n = 0
    for i, line in enumerate(md.splitlines()):
        if rx.search(line) and len(line.strip()) > 12:
            print(f"   [{i}] {line.strip()[:300]}")
            n += 1
            if n >= 24:
                print("   ...")
                break
