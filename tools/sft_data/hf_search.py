#!/usr/bin/env python3
"""hf_search.py QUERY [QUERY ...] [--limit N]: search the Hugging Face hub for datasets by name (public API, no token); prints id, downloads, likes, license tag and last modification, most downloaded first.
Run on the homelab with ~/forgeinfer/venv/bin/python (stdlib only)."""
import json
import sys
import urllib.parse
import urllib.request

args = sys.argv[1:]
limit = 15
if "--limit" in args:
    k = args.index("--limit")
    limit = int(args[k + 1])
    del args[k : k + 2]
for q in args:
    url = "https://huggingface.co/api/datasets?" + urllib.parse.urlencode({"search": q, "limit": limit, "sort": "downloads", "direction": -1})
    req = urllib.request.Request(url, headers={"User-Agent": "forgeinfer-sft-human"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            items = json.load(r)
    except Exception as e:  # noqa: BLE001
        print(f"== {q}: ERROR {e}")
        continue
    print(f"== {q}: {len(items)} results")
    for it in items:
        lic = [t for t in it.get("tags", []) if t.startswith("license:")]
        print(f"   {it['id']:70s} dl={it.get('downloads')} likes={it.get('likes')} {lic} gated={it.get('gated')} mod={str(it.get('lastModified', ''))[:10]}")
