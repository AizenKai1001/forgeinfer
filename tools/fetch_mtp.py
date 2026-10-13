#!/usr/bin/env python3
"""Fetch only the multi-token-prediction head (the `mtp.*` tensors) of a Hugging Face Qwen3.5 checkpoint and write it as one small safetensors file.

The GGUF files do not contain the head. The checkpoint is several GB, so the tensors are read with HTTP range requests: the safetensors header (or the shard index) says
where they are.

    fetch_mtp.py <repo> <out.safetensors>
    fetch_mtp.py Qwen/Qwen3.5-4B ~/forgeinfer/oracle/models/Qwen__Qwen3.5-4B/mtp.safetensors
"""
import json
import os
import struct
import sys
import time
import urllib.error
import urllib.request

CHUNK = 32 << 20


def url(repo, name):
    return f"https://huggingface.co/{repo}/resolve/main/{name}"


def fetch(u, start=None, end=None, tries=6):
    """GET u (optionally bytes start..end inclusive), following redirects, with retries."""
    for attempt in range(tries):
        req = urllib.request.Request(u, headers={"User-Agent": "forgeinfer-fetch/1"})
        if start is not None:
            req.add_header("Range", f"bytes={start}-{end}")
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                data = r.read()
            if start is not None and len(data) != end - start + 1:
                raise IOError(f"short read: {len(data)} of {end - start + 1}")
            return data
        except (urllib.error.URLError, IOError, TimeoutError) as e:
            if isinstance(e, urllib.error.HTTPError) and e.code == 404:
                raise
            print(f"  retry {attempt + 1}: {e}", file=sys.stderr)
            time.sleep(2 + 3 * attempt)
    raise RuntimeError(f"cannot fetch {u}")


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    repo, out = sys.argv[1], os.path.expanduser(sys.argv[2])
    try:
        index = json.loads(fetch(url(repo, "model.safetensors.index.json")))["weight_map"]
    except urllib.error.HTTPError as e:
        if e.code != 404:
            raise
        index = None
    if index is None:
        files = {"model.safetensors": None}   # a single file: every tensor is in it
    else:
        files = {}
        for name, f in index.items():
            if name.startswith("mtp."):
                files.setdefault(f, []).append(name)
        if not files:
            print("the checkpoint has no mtp.* tensors")
            return 1
    tensors = {}   # name -> (dtype, shape, bytes)
    for f in files:
        u = url(repo, f)
        n = struct.unpack("<Q", fetch(u, 0, 7))[0]
        header = json.loads(fetch(u, 8, 8 + n - 1))
        base = 8 + n
        names = [k for k in header if k.startswith("mtp.")]
        print(f"{f}: {len(names)} mtp tensors")
        for k in names:
            t = header[k]
            b0, b1 = t["data_offsets"]
            parts = []
            p = b0
            while p < b1:
                q = min(p + CHUNK, b1)
                parts.append(fetch(u, base + p, base + q - 1))
                p = q
            tensors[k] = (t["dtype"], t["shape"], b"".join(parts))
            print(f"  {k} {t['dtype']} {t['shape']} {b1 - b0} bytes")
    if not tensors:
        print("no mtp.* tensors found")
        return 1
    hdr, off = {}, 0
    for k in sorted(tensors):
        d, shape, data = tensors[k]
        hdr[k] = {"dtype": d, "shape": shape, "data_offsets": [off, off + len(data)]}
        off += len(data)
    hdr["__metadata__"] = {"format": "pt", "source": repo}
    h = json.dumps(hdr, separators=(",", ":")).encode()
    h += b" " * (-len(h) % 8)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    tmp = out + ".part"
    with open(tmp, "wb") as fo:
        fo.write(struct.pack("<Q", len(h)))
        fo.write(h)
        for k in sorted(tensors):
            fo.write(tensors[k][2])
    os.replace(tmp, out)
    print(f"wrote {out}: {len(tensors)} tensors, {off / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
