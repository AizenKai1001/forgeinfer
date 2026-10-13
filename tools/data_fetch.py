"""Streaming data preparation for ForgeInfer pre-training (runs on the homelab, uses the Python oracle environment only for reading parquet; the tokenization itself is done by the Rust tool).

    ~/forgeinfer/venv/bin/python tools/data_fetch.py <dataset> <shard> [--out DIR] [--raw DIR] [--keep]
    datasets: fwedu   FineWeb-Edu sample-10BT (14 shards, 0.7 B tokens each)
              cosmo   Cosmopedia v2, synthetic textbooks and stories (104 shards, 0.27 B tokens each)
              fmath   FineMath-3plus, web pages with mathematics (128 shards, 0.27 B tokens each)
              owmath  OpenWebMath (114 shards, 0.13 B tokens each)
              wiki    English Wikipedia 2023-11-01 (41 shards, 0.16 B tokens each)
              gut     Project Gutenberg, English books (52 shards)
              pycode  codeparrot-clean, Python source files from GitHub (55 files of ~1 GB of code each)
    (Python-Edu and the Stack list only blob ids - the code itself lives on Software Heritage - and are not used.)
    e.g.  data_fetch.py fwedu 0        data_fetch.py cosmo 3        data_fetch.py pycode 1

Downloads ONE shard (resumable), extracts the text column in batches and writes the documents separated by a NUL byte (0x00 never occurs in text) to <out>/<name>_<shard>.txt, then deletes the shard
(unless --keep): only one shard is in flight at a time. The shard number is the position in the repository's sorted file list (cached in <raw>/lists). Prints documents, bytes and the speed.
"""
import argparse
import gzip
import json
import os
import sys
import time
import urllib.request

# name: (repository, path prefix, file suffix, text column)
SOURCES = {
    "fwedu": ("HuggingFaceFW/fineweb-edu", "sample/10BT/", ".parquet", "text"),
    "cosmo": ("HuggingFaceTB/smollm-corpus", "cosmopedia-v2/", ".parquet", "text"),
    "fmath": ("HuggingFaceTB/finemath", "finemath-3plus/", ".parquet", "text"),
    "owmath": ("open-web-math/open-web-math", "data/", ".parquet", "text"),
    "wiki": ("wikimedia/wikipedia", "20231101.en/", ".parquet", "text"),
    "gut": ("manu/project_gutenberg", "data/en-", ".parquet", "text"),
    "pycode": ("codeparrot/codeparrot-clean", "file-", ".json.gz", "content"),
}


def file_list(repo, prefix, suffix, cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    cache = os.path.join(cache_dir, repo.replace("/", "__") + "__" + prefix.replace("/", "_") + ".json")
    if os.path.exists(cache):
        return json.load(open(cache))
    url = f"https://huggingface.co/api/datasets/{repo}/tree/main/{prefix.rsplit('/', 1)[0] if '/' in prefix else ''}?recursive=true"
    names = []
    while url:
        req = urllib.request.Request(url, headers={"User-Agent": "forgeinfer-tools"})
        with urllib.request.urlopen(req, timeout=60) as r:
            names += [i["path"] for i in json.load(r) if i.get("type") == "file"]
            link = r.headers.get("Link", "")
        url = None
        for part in link.split(","):
            if 'rel="next"' in part:
                url = part.split(";")[0].strip().strip("<>")
    names = sorted(n for n in names if n.startswith(prefix) and n.endswith(suffix))
    json.dump(names, open(cache, "w"))
    return names


def download(url, dest):
    have = os.path.getsize(dest + ".part") if os.path.exists(dest + ".part") else 0
    req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=120) as r, open(dest + ".part", "ab" if have else "wb") as f:
        total = have
        while True:
            chunk = r.read(1 << 22)
            if not chunk:
                break
            f.write(chunk)
            total += len(chunk)
    os.rename(dest + ".part", dest)
    print(f"downloaded {total / 1e6:.0f} MB in {time.time() - t0:.0f} s", flush=True)


def documents(path, col):
    """The texts of a downloaded shard, one at a time."""
    if path.endswith(".json.gz"):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line).get(col) or ""
    else:
        import pyarrow.parquet as pq

        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=2000, columns=[col]):
            for text in batch.column(0).to_pylist():
                yield text or ""


def documents_raw(path, col):
    """The values of one column of a parquet file as Python objects (lists of messages for chat data), one at a time."""
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=2000, columns=[col]):
        for v in batch.column(0).to_pylist():
            yield v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=sorted(SOURCES))
    ap.add_argument("shard", type=int)
    ap.add_argument("--out", default="/mnt/data/forgeinfer-data/text")
    ap.add_argument("--raw", default="/mnt/data/forgeinfer-data/raw")
    ap.add_argument("--keep", action="store_true")
    a = ap.parse_args()
    repo, prefix, suffix, col = SOURCES[a.dataset]
    names = file_list(repo, prefix, suffix, os.path.join(a.raw, "lists"))
    if a.shard >= len(names):
        sys.exit(f"{a.dataset} has {len(names)} shards")
    name = names[a.shard]
    os.makedirs(a.out, exist_ok=True)
    raw = os.path.join(a.raw, f"raw_{a.dataset}_{a.shard}" + suffix)
    if not os.path.exists(raw):
        download(f"https://huggingface.co/datasets/{repo}/resolve/main/{name}", raw)
    out_path = os.path.join(a.out, f"{a.dataset}_{a.shard:03d}.txt")
    docs = nbytes = 0
    t0 = time.time()
    with open(out_path, "wb") as f:
        for text in documents(raw, col):
            if not text:
                continue
            b = text.replace("\x00", " ").encode("utf-8")
            f.write(b)
            f.write(b"\x00")
            docs += 1
            nbytes += len(b)
    print(f"{a.dataset} shard {a.shard} ({name}): {docs} documents, {nbytes / 1e9:.2f} GB of text -> {out_path} ({time.time() - t0:.0f} s)", flush=True)
    if not a.keep:
        os.remove(raw)


if __name__ == "__main__":
    sys.exit(main())
