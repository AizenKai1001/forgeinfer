#!/usr/bin/env python3
"""Text for measuring quantization quality, several domains, each with held-out evaluation text and separate calibration text (other documents):
   en    FineWeb-Edu web text            (Hugging Face datasets server)
   code  Python standard library         (local)
   zh    Chinese Wikipedia               (datasets server)
   de    German Wikipedia                (datasets server)
   math  GSM8K question + worked answer  (datasets server)
   chat  UltraChat dialogues             (datasets server)
A domain whose download fails is skipped.

    make_eval_text.py [outdir]      default ~/forgeinfer/data/eval
writes eval_<d>.txt and calib_<d>.txt per domain
"""
import json
import os
import sys
import time
import urllib.parse
import urllib.request

OUT = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/forgeinfer/data/eval")
os.makedirs(OUT, exist_ok=True)


def rows(dataset, config, split, offset, length=100):
    q = urllib.parse.urlencode({"dataset": dataset, "config": config, "split": split, "offset": offset, "length": length})
    for attempt in range(5):
        try:
            with urllib.request.urlopen("https://datasets-server.huggingface.co/rows?" + q, timeout=60) as r:
                return [x["row"] for x in json.load(r)["rows"]]
        except Exception as e:  # noqa: BLE001
            print("retry", dataset, attempt, e, file=sys.stderr)
            time.sleep(3 * (attempt + 1))
    raise RuntimeError("datasets server not reachable for " + dataset)


def write(name, docs, max_chars):
    text = "\n\n".join(docs)[:max_chars]
    with open(os.path.join(OUT, name), "w", encoding="utf-8") as f:
        f.write(text)
    print(name, len(docs), "documents,", len(text), "characters")


def collect(dataset, config, split, first, requests, render, keep=lambda t: 200 < len(t) < 12000, step=997):
    docs, total = [], 0
    for i in range(requests):
        for r in rows(dataset, config, split, first + i * step):
            t = render(r)
            if t and keep(t):
                docs.append(t.strip())
                total += len(t)
    return docs


def web(name, dataset, config, split, first_eval, first_calib, requests, render, max_chars, **kw):
    try:
        write("eval_" + name + ".txt", collect(dataset, config, split, first_eval, requests, render, **kw), max_chars)
        write("calib_" + name + ".txt", collect(dataset, config, split, first_calib, requests, render, **kw), max_chars)
    except Exception as e:  # noqa: BLE001
        print("skipped", name, e, file=sys.stderr)


def code(name, parity, max_chars):
    root = "/usr/lib/python3.12"
    files = []
    for d, dirs, fs in os.walk(root):
        dirs[:] = sorted(x for x in dirs if x not in ("test", "tests", "idlelib", "site-packages", "__pycache__", "lib2to3", "turtledemo", "pydoc_data", "ensurepip"))
        for f in sorted(fs):
            if f.endswith(".py"):
                p = os.path.join(d, f)
                if 2000 < os.path.getsize(p) < 60000:
                    files.append(p)
    parts, total = [], 0
    for i, p in enumerate(files):
        if i % 2 != parity:
            continue
        try:
            t = open(p, encoding="utf-8").read()
        except UnicodeDecodeError:
            continue
        parts.append(f"# file: {os.path.relpath(p, root)}\n{t}")
        total += len(t)
        if total >= max_chars:
            break
    write(name, parts, max_chars)


if __name__ == "__main__":
    web("en", "HuggingFaceFW/fineweb-edu", "sample-10BT", "train", 300000, 450000, 12, lambda r: r["text"], 900_000, keep=lambda t: 800 < len(t) < 12000)
    code("eval_code.txt", 0, 500_000)
    code("calib_code.txt", 1, 500_000)
    web("zh", "wikimedia/wikipedia", "20231101.zh", "train", 3000, 90000, 10, lambda r: r["text"], 300_000, keep=lambda t: 600 < len(t) < 8000, step=211)
    web("de", "wikimedia/wikipedia", "20231101.de", "train", 3000, 90000, 10, lambda r: r["text"], 500_000, keep=lambda t: 800 < len(t) < 12000, step=211)
    web("math", "openai/gsm8k", "main", "train", 0, 3800, 8, lambda r: "Question: " + r["question"] + "\nAnswer: " + r["answer"], 300_000, keep=lambda t: len(t) > 100, step=100)
    web("chat", "HuggingFaceH4/ultrachat_200k", "default", "train_sft", 0, 60000, 12, lambda r: "\n".join(("User: " if m["role"] == "user" else "Assistant: ") + m["content"] for m in r["messages"]), 600_000, keep=lambda t: 300 < len(t) < 12000, step=300)
