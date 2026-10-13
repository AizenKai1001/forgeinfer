"""Chat (instruction-tuning) text for Forge-1, rendered as ChatML, ready for `forge tok-encode` (runs on the homelab with the oracle venv; CPU only).

    ~/forgeinfer/venv/bin/python tools/sft_prepare.py OUT.txt [--sources smoltalk,ultrachat] [--max-convs 400000] [--max-chars 7000] [--raw DIR]

Every conversation becomes one document (documents are separated by a NUL byte, like the pre-training text):
    <|im_start|>system\\n...<|im_end|>\\n<|im_start|>user\\n...<|im_end|>\\n<|im_start|>assistant\\n...<|im_end|>\\n
Sources (public datasets of the Hugging Face hub; their licences allow this use):
    smoltalk    HuggingFaceTB/smoltalk, split all (about 1.1 M conversations: everyday chat, rewriting, summarising, maths, code, instruction following, tool calls)
    ultrachat   HuggingFaceH4/ultrachat_200k, split train_sft (multi-turn general chat)
Conversations with a role other than system/user/assistant, without an assistant turn, or longer than --max-chars are dropped. The special-token strings inside the text are neutralised.
The output is shuffled by a fixed seed within each parquet file group, so a prefix of the file is already a mixture.
"""
import argparse
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data_fetch  # noqa: E402  (file_list, download)

SOURCES = {
    "smoltalk": ("HuggingFaceTB/smoltalk", "data/all/train-", ".parquet", "messages"),
    "ultrachat": ("HuggingFaceH4/ultrachat_200k", "data/train_sft-", ".parquet", "messages"),
}
SPECIALS = ["<|im_start|>", "<|im_end|>", "<|endoftext|>", "<tool_call>", "</tool_call>", "<tool_response>", "</tool_response>", "<think>", "</think>", "<|pad|>"]


def clean(text):
    for sp in SPECIALS:
        text = text.replace(sp, sp.replace("<", "(").replace(">", ")"))
    return text.replace("\x00", " ").strip()


def render(msgs, max_chars):
    parts = []
    has_assistant = False
    for m in msgs:
        role = m.get("role")
        if role not in ("system", "user", "assistant"):
            return None
        content = clean(m.get("content") or "")
        if not content:
            return None
        has_assistant |= role == "assistant"
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
    text = "".join(parts)
    if not has_assistant or len(text) > max_chars:
        return None
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--sources", default="smoltalk,ultrachat")
    ap.add_argument("--max-convs", type=int, default=400000)
    ap.add_argument("--max-chars", type=int, default=7000)
    ap.add_argument("--raw", default="/mnt/data/forgeinfer-data/raw")
    a = ap.parse_args()
    names = a.sources.split(",")
    per_source = a.max_convs // len(names)
    rng = random.Random(7)
    total = 0
    with open(a.out, "wb") as f:
        for name in names:
            repo, prefix, suffix, col = SOURCES[name]
            files = data_fetch.file_list(repo, prefix, suffix, os.path.join(a.raw, "lists"))
            kept = 0
            for i, fname in enumerate(files):
                if kept >= per_source:
                    break
                raw = os.path.join(a.raw, f"raw_sft_{name}_{i}{suffix}")
                if not os.path.exists(raw):
                    data_fetch.download(f"https://huggingface.co/datasets/{repo}/resolve/main/{fname}", raw)
                docs = []
                for msgs in data_fetch.documents_raw(raw, col):
                    text = render(msgs, a.max_chars)
                    if text is not None:
                        docs.append(text)
                rng.shuffle(docs)
                for text in docs:
                    if kept >= per_source:
                        break
                    f.write(text.encode("utf-8"))
                    f.write(b"\x00")
                    kept += 1
                os.remove(raw)
                print(f"{name}: file {i + 1}/{len(files)}: {kept} conversations so far", flush=True)
            total += kept
            print(f"{name}: {kept} conversations", flush=True)
    print(f"{total} conversations -> {a.out}")


if __name__ == "__main__":
    main()
