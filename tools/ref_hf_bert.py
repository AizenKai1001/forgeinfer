"""The Hugging Face reference of a BERT-family embedding model (bge, MiniLM, e5, gte ...): token ids and vectors made by `transformers` from the model's own folder (config.json, model.safetensors, vocab and
tokenizer files, as `huggingface.co/BAAI/bge-small-en-v1.5` ships them). The Rust `forge-embed` and `forge-embed-gpu` are checked against it (crates/forge-embed/tests/golden/*.json are made by this file).

    python tools/ref_hf_bert.py MODEL_DIR --pooling cls|mean --golden OUT.json [--texts tools/embed_texts_23.json] [--prefix "query: "]
    python tools/ref_hf_bert.py MODEL_DIR --pooling cls TEXT [TEXT ...]          cosine similarities of the texts, token ids

The vector is what sentence-transformers makes of the model: the [CLS] vector (bge) or the mean of the token vectors (MiniLM, gte, e5), normalised to length one. Needs torch and transformers (the homelab's
~/forgeinfer/venv has them).
"""
import argparse
import json
import sys

import torch
from transformers import AutoModel, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--pooling", choices=["cls", "mean"], required=True)
    ap.add_argument("--golden", help="write the ids and vectors of the texts as the golden file of the Rust tests")
    ap.add_argument("--texts", default=None, help="a JSON list of texts (default: the texts on the command line)")
    ap.add_argument("--prefix", default="", help="put in front of every text (e5: 'query: ')")
    ap.add_argument("texts_cmd", nargs="*")
    a = ap.parse_args()
    texts = json.load(open(a.texts, encoding="utf-8")) if a.texts else a.texts_cmd
    tok = AutoTokenizer.from_pretrained(a.model_dir)
    model = AutoModel.from_pretrained(a.model_dir, torch_dtype=torch.float32).eval()
    max_len = int(getattr(model.config, "max_position_embeddings", 512))
    cases = []
    with torch.no_grad():
        for text in texts:
            enc = tok(a.prefix + text, truncation=True, max_length=max_len, return_tensors="pt")
            h = model(**enc).last_hidden_state[0]
            v = h[0] if a.pooling == "cls" else h.mean(dim=0)
            v = v / v.norm().clamp(min=1e-12)
            cases.append({"text": text, "ids": enc["input_ids"][0].tolist(), "vector": [float(x) for x in v]})
    if a.golden:
        json.dump({"model": a.model_dir.rstrip("/").split("/")[-1], "pooling": a.pooling, "prefix": a.prefix, "cases": cases}, open(a.golden, "w", encoding="utf-8"), ensure_ascii=False)
        print("wrote", a.golden, len(cases), "cases", file=sys.stderr)
        return
    for c in cases:
        print(c["ids"][:12], "..." if len(c["ids"]) > 12 else "", repr(c["text"][:50]))
    base = cases[0]["vector"]
    for c in cases[1:]:
        print(round(sum(x * y for x, y in zip(base, c["vector"])), 4), repr(c["text"][:60]))


if __name__ == "__main__":
    main()
