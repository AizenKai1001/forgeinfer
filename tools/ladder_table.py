#!/usr/bin/env python3
"""ladder_table.py: side-by-side table of several evaluation-ladder results (the results.json files written by tools/f1_eval.sh, also those downloaded from Modal).

  python tools/ladder_table.py "v1 s12500=scratch/modal_ladder_s12500/results.json" "TPU model=scratch/ladder_tpu1_tpu1.json" ... [--domains]

Columns are the checkpoints, rows the tasks (headline metric, one decimal), then LAMBADA word perplexity and the held-out loss averaged over the seven kinds of text (nats per token, lower is better); --domains adds the seven losses.
The standard error of a 500-item subsample is about 2 points: differences below that are noise; the perplexities are the reliable signal.
"""
import json
import sys

TASKS = ["hellaswag", "arc_easy", "arc_challenge", "piqa", "winogrande", "openbookqa", "sciq", "lambada_openai", "mmlu_cloze"]
NAMES = {"hellaswag": "HellaSwag", "arc_easy": "ARC-Easy", "arc_challenge": "ARC-Challenge", "piqa": "PIQA", "winogrande": "WinoGrande", "openbookqa": "OpenBookQA", "sciq": "SciQ",
         "lambada_openai": "LAMBADA acc", "mmlu_cloze": "MMLU cloze"}


def main(argv):
    domains = "--domains" in argv
    pairs = [a for a in argv if "=" in a and not a.startswith("--")]
    if not pairs:
        print(__doc__)
        return 1
    labels, results = [], []
    for p in pairs:
        label, path = p.split("=", 1)
        with open(path, encoding="utf-8") as fh:
            results.append(json.load(fh))
        labels.append(label)
    w = max(14, max(len(x) for x in labels) + 1)
    print(f"{'':16s}" + "".join(f"{x:>{w}s}" for x in labels))
    for t in TASKS:
        vals = []
        for r in results:
            v = (r.get("tasks") or {}).get(t)
            vals.append(f"{v['headline']:.1f}" if v and v.get("headline") is not None else "-")
        print(f"{NAMES[t]:16s}" + "".join(f"{x:>{w}s}" for x in vals))
    row = []
    for r in results:
        v = (r.get("tasks") or {}).get("lambada_openai")
        row.append(f"{v['perplexity']:.1f}" if v and v.get("perplexity") else "-")
    print(f"{'LAMBADA ppl':16s}" + "".join(f"{x:>{w}s}" for x in row))
    if domains:
        for d in ["web", "cosmo", "fmath", "owmath", "wiki", "books", "python"]:
            print(f"{'nats ' + d:16s}" + "".join(f"{((r.get('ppl') or {}).get('domains') or {}).get(d, {}).get('nats', float('nan')):{w}.4f}" for r in results))
    print(f"{'mean nats':16s}" + "".join(f"{((r.get('ppl') or {}).get('mean_nats') or float('nan')):{w}.4f}" for r in results))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
