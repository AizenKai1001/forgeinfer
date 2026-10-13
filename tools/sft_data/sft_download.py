#!/usr/bin/env python3
"""sft_download.py [GROUP ...]: download the raw SFT sources and the benchmark test sets for decontamination to /mnt/data/forgeinfer-data/sft_raw/<owner>__<name>/ (public files only, no login).
Groups: bench, small, smoltalk, tulu, ultrachat, (no argument = all).  Run on the homelab:  nice -n 15 ~/forgeinfer/venv/bin/python sft_download.py [group ...]
Uses huggingface_hub.snapshot_download with allow_patterns, 2 download workers; the cache stays under /mnt/data (HF_HOME is set below)."""
import os
import sys
import time

ROOT = "/mnt/data/forgeinfer-data/sft_raw"
os.environ.setdefault("HF_HOME", "/mnt/data/forgeinfer-data/sft_work/hf_home")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
from huggingface_hub import snapshot_download  # noqa: E402

GROUPS = {
    # benchmark test sets: only used to build the n-gram index of the decontamination (never trained on)
    "bench": [
        ("Rowan/hellaswag", ["data/validation-*", "data/test-*"]),
        ("allenai/ai2_arc", ["ARC-Challenge/test-*", "ARC-Challenge/validation-*", "ARC-Easy/test-*", "ARC-Easy/validation-*"]),
        ("baber/piqa", ["piqa_validation.parquet", "piqa_test.parquet"]),
        ("allenai/winogrande", ["winogrande_xl/validation-*", "winogrande_xl/test-*"]),
        ("cais/mmlu", ["all/test-*", "all/validation-*", "all/dev-*"]),
        ("openai/gsm8k", ["main/test-*"]),
        ("openai/openai_humaneval", ["openai_humaneval/test-*"]),
        ("google-research-datasets/mbpp", ["full/*", "sanitized/*"]),
        ("EleutherAI/lambada_openai", ["data/lambada_test.jsonl"]),
        ("allenai/sciq", ["data/test-*", "data/validation-*"]),
        ("allenai/openbookqa", ["main/test-*", "main/validation-*"]),
        ("google/IFEval", ["ifeval_input_data.jsonl"]),
        ("truthfulqa/truthful_qa", ["generation/validation-*"]),
        ("EleutherAI/hendrycks_math", ["*/test-*"]),
    ],
    # small sources
    "small": [
        ("databricks/databricks-dolly-15k", ["databricks-dolly-15k.jsonl"]),
        ("OpenAssistant/oasst2", ["2023-11-05_oasst2_ready.trees.jsonl.gz"]),
        ("meta-math/MetaMathQA", ["MetaMathQA-395K.json"]),
        ("microsoft/orca-math-word-problems-200k", ["*.parquet", "*.json", "*.jsonl"]),
        ("openai/gsm8k", ["main/train-*", "socratic/train-*"]),
        ("EleutherAI/hendrycks_math", ["*/train-*"]),
        ("ise-uiuc/Magicoder-OSS-Instruct-75K", ["data-oss_instruct-decontaminated.jsonl"]),
        ("LDJnr/Capybara", ["*.jsonl", "*.json", "*.parquet"]),
    ],
    "smoltalk": [
        ("HuggingFaceTB/smoltalk", [f"data/{c}/train-*" for c in ["smol-summarize", "smol-rewrite", "smol-constraints", "everyday-conversations", "systemchats-30k", "self-oss-instruct", "smol-magpie-ultra"]]),
    ],
    "tulu": [
        ("allenai/tulu-3-sft-mixture", ["data/*.parquet"]),
    ],
    "ultrachat": [
        ("HuggingFaceH4/ultrachat_200k", ["data/train_sft-*"]),
    ],
}


def main():
    groups = [a for a in sys.argv[1:]] or list(GROUPS)
    for g in groups:
        for repo, patterns in GROUPS[g]:
            dest = os.path.join(ROOT, repo.replace("/", "__"))
            t0 = time.time()
            print(f"[{time.strftime('%H:%M:%S')}] {repo} {patterns[:3]}... -> {dest}", flush=True)
            try:
                snapshot_download(repo_id=repo, repo_type="dataset", allow_patterns=patterns, local_dir=dest, max_workers=2)
            except Exception as e:  # noqa: BLE001
                print(f"   FAILED {repo}: {type(e).__name__}: {e}", flush=True)
                continue
            size = sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(dest) for f in fs if ".cache" not in d)
            print(f"   done in {time.time() - t0:.0f} s, {size / 1e6:.0f} MB on disk", flush=True)


if __name__ == "__main__":
    main()
