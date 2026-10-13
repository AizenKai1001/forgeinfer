#!/usr/bin/env python3
"""fetch_human.py GROUP [GROUP ...] [--dry-run]: download the raw files of the human-written-only SFT sources (human_cfg.py) from the Hugging Face hub to
/mnt/data/forgeinfer-data/sft_work_human/raw/<owner>__<name>/ (public files only, no login).

Groups: se (Stack Exchange per-site parquet files of HuggingFaceH4/stack-exchange-preferences), shp (stanfordnlp/SHP subreddits), small (the other sources).
One file at a time; after every file the page cache of the file is dropped (posix_fadvise DONTNEED after fsync) and the loop sleeps so that the average write speed stays below --mb-s
(default 35 MB/s): the disk and the page cache are shared with the production training run.  Existing files of the right size are skipped (resumable).  Run on the homelab:
    nice -n 15 ionice -c3 ~/forgeinfer/venv/bin/python fetch_human.py se shp small
"""
import fnmatch
import json
import os
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import human_cfg as C  # noqa: E402

os.environ.setdefault("HF_HOME", os.path.join(C.WORK, "hf_home"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
from huggingface_hub import hf_hub_download  # noqa: E402

MBS = 35.0
DRY = "--dry-run" in sys.argv


def hub_files(repo):
    """{filename: size} of a dataset repo from the hub API (cached in the cards directory)."""
    cache = os.path.join(C.CARDS, repo.replace("/", "__") + ".json")
    if os.path.exists(cache):
        info = json.load(open(cache))
    else:
        req = urllib.request.Request(f"https://huggingface.co/api/datasets/{repo}?blobs=true", headers={"User-Agent": "forgeinfer-sft-human"})
        info = json.load(urllib.request.urlopen(req, timeout=60))
        json.dump(info, open(cache, "w"), indent=1)
    return {s["rfilename"]: s.get("size") or 0 for s in info.get("siblings", [])}


def drop_cache(path):
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        except OSError:
            pass
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        os.close(fd)
    except OSError:
        pass


def fetch(repo, name, size):
    dest_dir = os.path.join(C.RAW, repo.replace("/", "__"))
    dest = os.path.join(dest_dir, name)
    if os.path.exists(dest) and (size == 0 or os.path.getsize(dest) == size):
        return 0
    t0 = time.time()
    if DRY:
        print(f"   would fetch {repo}/{name} {size / 1e6:.1f} MB")
        return 0
    try:
        hf_hub_download(repo, name, repo_type="dataset", local_dir=dest_dir)
    except Exception as e:  # noqa: BLE001
        print(f"   FAILED {repo}/{name}: {type(e).__name__}: {e}", flush=True)
        return 0
    got = os.path.getsize(dest)
    drop_cache(dest)
    # the hub cache keeps no second copy with local_dir, but remove the .cache lock files of finished downloads
    wait = got / (MBS * 1e6) - (time.time() - t0)
    print(f"[{time.strftime('%H:%M:%S')}] {repo}/{name} {got / 1e6:.1f} MB in {time.time() - t0:.0f} s", flush=True)
    if wait > 0:
        time.sleep(wait)
    return got


def spread(files, k):
    if k is None or k >= len(files):
        return files
    return [files[int(i * len(files) / k)] for i in range(k)]


def group_se():
    allf = hub_files(C.SE_REPO)
    total = 0
    for site, (_, k) in C.SE_SITES.items():
        fs = sorted(f for f in allf if f.startswith(f"data/{site}/") and f.endswith(".parquet"))
        if not fs:
            print(f"   no files for site {site}")
            continue
        for f in spread(fs, k):
            total += fetch(C.SE_REPO, f, allf[f])
    return total


def group_shp():
    allf = hub_files(C.SHP_REPO)
    total = 0
    for sub in C.SHP_SUBS:
        for f in sorted(allf):
            if f.startswith(sub + "/") and f.endswith(".json"):
                total += fetch(C.SHP_REPO, f, allf[f])
    return total


def group_small():
    total = 0
    for repo, patterns in C.SMALL:
        allf = hub_files(repo)
        for pat in patterns:
            hit = sorted(f for f in allf if fnmatch.fnmatch(f, pat))
            if not hit:
                print(f"   NO MATCH {repo} {pat}")
            for f in hit:
                total += fetch(repo, f, allf[f])
    return total


if __name__ == "__main__":
    os.makedirs(C.RAW, exist_ok=True)
    os.makedirs(C.CARDS, exist_ok=True)
    groups = [a for a in sys.argv[1:] if not a.startswith("--")] or ["se", "shp", "small"]
    for g in groups:
        t0 = time.time()
        print(f"== group {g}", flush=True)
        n = {"se": group_se, "shp": group_shp, "small": group_small}[g]()
        print(f"== group {g} done: {n / 1e6:.0f} MB in {time.time() - t0:.0f} s", flush=True)
