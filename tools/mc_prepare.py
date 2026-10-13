"""Multiple-choice evaluation sets in one jsonl format, converted the way EleutherAI's lm-evaluation-harness prepares them (runs on the homelab with the oracle venv; needs pyarrow).

    ~/forgeinfer/venv/bin/python tools/mc_prepare.py [--out DIR] [task ...]       tasks: arc_easy arc_challenge piqa winogrande sciq openbookqa (default: all)

One line per example: {"choices": [{"ctx": "...", "cont": " ..."}, ...], "label": k}. Every choice is scored as the log-likelihood of `cont` given `ctx` (the continuation starts with a space, as in the
harness); `acc` takes the highest total, `acc_norm` the highest total divided by the number of characters of the continuation. Winogrande uses the harness's partial scoring (the same continuation after the
sentence prefix with each option filled in); the other tasks share one context. The data comes from the Hugging Face hub (parquet files of the official datasets), one evaluation split each.
"""
import argparse
import json
import os
import sys
import urllib.request

import pyarrow.parquet as pq

# task: (repository, path prefix to search under, split name that selects the evaluation file)
SOURCES = {
    "arc_easy": ("allenai/ai2_arc", "ARC-Easy/", "test"),
    "arc_challenge": ("allenai/ai2_arc", "ARC-Challenge/", "test"),
    "piqa": ("nthngdy/piqa", "data/", "validation"),
    "winogrande": ("allenai/winogrande", "winogrande_xl/", "validation"),
    "sciq": ("allenai/sciq", "data/", "test"),
    "openbookqa": ("allenai/openbookqa", "main/", "test"),
}


def hub_files(repo, ref="main"):
    url = f"https://huggingface.co/api/datasets/{repo}/tree/{ref}?recursive=true"
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
    return names


def fetch(repo, path, dest, ref="main"):
    url = f"https://huggingface.co/datasets/{repo}/resolve/{ref.replace('/', '%2F')}/{path}"
    req = urllib.request.Request(url, headers={"User-Agent": "forgeinfer-tools"})
    with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
        f.write(r.read())


def find_parquet(repo, prefix, split):
    for ref in ("main", "refs/convert/parquet"):
        try:
            names = hub_files(repo, ref)
        except Exception as e:  # noqa: BLE001
            print(f"  {repo}@{ref}: {e}", file=sys.stderr)
            continue
        cand = [n for n in names if n.endswith(".parquet") and n.startswith(prefix) and (os.path.basename(n).startswith(split) or f"/{split}/" in "/" + n)]
        if cand:
            return ref, sorted(cand)[0]
    raise SystemExit(f"no parquet file for {repo} {prefix} {split}")


def rows(path):
    t = pq.read_table(path).to_pylist()
    return t


def conv_arc(r):
    labels = list(r["choices"]["label"])
    return {"choices": [{"ctx": f"Question: {r['question']}\nAnswer:", "cont": " " + t} for t in r["choices"]["text"]], "label": labels.index(r["answerKey"])}


def conv_piqa(r):
    ctx = f"Question: {r['goal']}\nAnswer:"
    return {"choices": [{"ctx": ctx, "cont": " " + r["sol1"]}, {"ctx": ctx, "cont": " " + r["sol2"]}], "label": int(r["label"])}


def conv_winogrande(r):
    s = r["sentence"]
    i = s.index("_")
    cont = " " + s[i + 1:].strip()
    return {"choices": [{"ctx": s[:i] + r["option1"], "cont": cont}, {"ctx": s[:i] + r["option2"], "cont": cont}], "label": int(r["answer"]) - 1}


def conv_sciq(r):
    ctx = f"{r['support'].lstrip()}\nQuestion: {r['question']}\nAnswer:"
    opts = [r["distractor1"], r["distractor2"], r["distractor3"], r["correct_answer"]]
    return {"choices": [{"ctx": ctx, "cont": " " + o} for o in opts], "label": 3}


def conv_obqa(r):
    labels = list(r["choices"]["label"])
    return {"choices": [{"ctx": r["question_stem"], "cont": " " + t} for t in r["choices"]["text"]], "label": labels.index(r["answerKey"])}


CONVERT = {"arc_easy": conv_arc, "arc_challenge": conv_arc, "piqa": conv_piqa, "winogrande": conv_winogrande, "sciq": conv_sciq, "openbookqa": conv_obqa}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tasks", nargs="*")
    ap.add_argument("--out", default=os.path.expanduser("~/forgeinfer/data/eval/mc"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for task in a.tasks or list(SOURCES):
        repo, prefix, split = SOURCES[task]
        try:
            ref, name = find_parquet(repo, prefix, split)
            tmp = os.path.join(a.out, f"_{task}.parquet")
            fetch(repo, name, tmp, ref)
            data = [CONVERT[task](r) for r in rows(tmp)]
            os.remove(tmp)
        except SystemExit as e:
            print(f"{task}: {e}", file=sys.stderr)
            continue
        with open(os.path.join(a.out, f"{task}.jsonl"), "w", encoding="utf-8") as f:
            for d in data:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        print(f"{task}: {len(data)} examples from {repo}@{ref}:{name}")


if __name__ == "__main__":
    main()
