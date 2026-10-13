"""List the files and sizes of open dataset repositories on the Hugging Face hub (public API, no token), to plan the training-data budget.
    python3 tools/hf_dataset_sizes.py
"""
import json
import urllib.request

REPOS = [("HuggingFaceFW/fineweb-edu", "sample/10BT"), ("HuggingFaceFW/fineweb-edu", "sample/100BT"), ("HuggingFaceTB/smollm-corpus", ""), ("HuggingFaceTB/smollm-corpus", "fineweb-edu-dedup"),
         ("HuggingFaceTB/smollm-corpus", "cosmopedia-v2"), ("HuggingFaceTB/smollm-corpus", "python-edu"), ("roneneldan/TinyStories", ""), ("mlfoundations/dclm-baseline-1.0-parquet", "")]


def tree(repo, path=""):
    url = f"https://huggingface.co/api/datasets/{repo}/tree/main" + (f"/{path}" if path else "")
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.load(r)


def main():
    for repo, path in REPOS:
        try:
            items = tree(repo, path)
        except Exception as e:
            print(f"{repo}/{path}: ERROR {e}")
            continue
        files = [i for i in items if i["type"] == "file"]
        dirs = [i["path"] for i in items if i["type"] == "directory"]
        total = sum(i.get("size", 0) for i in files)
        first = [(f["path"].split("/")[-1], round(f.get("size", 0) / 1e6)) for f in files[:3]]
        print(f"{repo}/{path}: {len(files)} files, {total / 1e9:.1f} GB, subdirectories {dirs[:8]}, first files (name, MB) {first}")


if __name__ == "__main__":
    main()
