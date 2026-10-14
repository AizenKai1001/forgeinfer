"""An executable SPECIFICATION of the NomicBERT embedding model (nomic-embed-text-v1.5 as a GGUF file), written in numpy from the architecture of Hugging Face's `NomicBertModel` and llama.cpp's `nomic-bert`;
the Rust `forge-embed` is checked against it (crates/forge-embed/tests/golden/nomic_v15_q4km.json is made by this file).

    python tools/ref_nomic_bert.py MODEL.gguf TEXT [TEXT ...]                 cosine similarities of the texts, token ids
    python tools/ref_nomic_bert.py MODEL.gguf --golden OUT.json TEXT ...      the ids and vectors as the golden file of the Rust tests
    python tools/ref_nomic_bert.py MODEL.gguf --ids-only TEXT ...              only the token ids (needs `transformers` for the reference tokenizer; without it a small ASCII WordPiece is used)

Per token: x = LayerNorm(token_embd[id] + token_types[0]); per layer: q, k, v = rows of Wqkv x (q first, then k, then v); rotary embedding on q and k (NeoX layout: dimension i with i + d/2, base 1000, position =
index in the text); a = softmax(q k^T / sqrt(d)) v over ALL tokens; x = LayerNorm(x + Wo a); x = LayerNorm(x + Wdown (silu(Wgate x) * Wup x)); the vector is the mean of the token vectors, normalised."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ref_gguf  # noqa: E402


def load(path):
    kv, infos, buf, data0 = ref_gguf.read_gguf(path)
    assert kv["general.architecture"] == "nomic-bert", kv["general.architecture"]
    t = lambda name: ref_gguf.tensor(kv, infos, buf, data0, name).astype(np.float32)
    m = {
        "kv": kv,
        "tok_emb": t("token_embd.weight"),
        "type_emb": t("token_types.weight")[0],
        "emb_norm": (t("token_embd_norm.weight"), t("token_embd_norm.bias")),
        "layers": [],
    }
    for i in range(kv["nomic-bert.block_count"]):
        b = f"blk.{i}"
        m["layers"].append({
            "qkv": t(f"{b}.attn_qkv.weight"), "out": t(f"{b}.attn_output.weight"), "up": t(f"{b}.ffn_up.weight"), "gate": t(f"{b}.ffn_gate.weight"), "down": t(f"{b}.ffn_down.weight"),
            "n1": (t(f"{b}.attn_output_norm.weight"), t(f"{b}.attn_output_norm.bias")), "n2": (t(f"{b}.layer_output_norm.weight"), t(f"{b}.layer_output_norm.bias")),
        })
    return m


def unphantom(t):
    """llama.cpp's converter stores BERT's vocabulary with phantom spaces: a word-initial token gets U+2581 in front, a continuation piece loses its ##, [bracketed] tokens stay."""
    if t.startswith("▁"):
        return t[1:]
    if t.startswith("[") and t.endswith("]"):
        return t
    return "##" + t


def tokenizer(kv):
    tokens = list(kv["tokenizer.ggml.tokens"])
    if any(t.startswith("▁") for t in tokens):
        tokens = [unphantom(t) for t in tokens]
    try:
        import tempfile
        from transformers import BertTokenizer
        d = tempfile.mkdtemp()
        with open(os.path.join(d, "vocab.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(tokens) + "\n")
        tok = BertTokenizer(os.path.join(d, "vocab.txt"), do_lower_case=True)
        return lambda text: tok.encode(text, add_special_tokens=True, truncation=True, max_length=kv["nomic-bert.context_length"])
    except Exception as e:  # a small WordPiece for plain ASCII text
        print("(no transformers: the ASCII WordPiece of this file is used:", e, ")", file=sys.stderr)
    vocab = {t: i for i, t in enumerate(tokens)}
    punct = set(range(33, 48)) | set(range(58, 65)) | set(range(91, 97)) | set(range(123, 127))

    def encode(text):
        words, cur = [], ""
        for ch in text.lower():
            if ch.isspace():
                if cur:
                    words.append(cur)
                    cur = ""
            elif ord(ch) in punct:
                if cur:
                    words.append(cur)
                    cur = ""
                words.append(ch)
            else:
                cur += ch
        if cur:
            words.append(cur)
        ids = [vocab["[CLS]"]]
        for w in words:
            pieces, start, bad = [], 0, False
            while start < len(w):
                end, found = len(w), None
                while start < end:
                    s = w[start:end] if start == 0 else "##" + w[start:end]
                    if s in vocab:
                        found = vocab[s]
                        break
                    end -= 1
                if found is None:
                    bad = True
                    break
                pieces.append(found)
                start = end
            ids.extend([vocab["[UNK]"]] if bad else pieces)
        ids.append(vocab["[SEP]"])
        return ids

    return encode


def layer_norm(x, w_b, eps):
    w, b = w_b
    mean = x.mean(-1, keepdims=True)
    var = ((x - mean) ** 2).mean(-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * w + b


def embed_ids(m, ids):
    kv = m["kv"]
    n_head, e = kv["nomic-bert.attention.head_count"], kv["nomic-bert.embedding_length"]
    hd = e // n_head
    eps = np.float32(kv["nomic-bert.attention.layer_norm_epsilon"])
    base = float(kv["nomic-bert.rope.freq_base"])
    n = len(ids)
    x = layer_norm(m["tok_emb"][ids] + m["type_emb"], m["emb_norm"], eps)
    inv_freq = base ** (-np.arange(0, hd, 2, dtype=np.float64) / hd)
    ang = np.arange(n, dtype=np.float64)[:, None] * inv_freq[None, :]
    cos, sin = np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)

    def rope(z):  # z: [n, heads, hd], NeoX layout
        a, b = z[..., : hd // 2], z[..., hd // 2:]
        return np.concatenate([a * cos[:, None, :] - b * sin[:, None, :], b * cos[:, None, :] + a * sin[:, None, :]], -1)

    for l in m["layers"]:
        qkv = x @ l["qkv"].T
        q, k, v = (qkv[:, i * e:(i + 1) * e].reshape(n, n_head, hd) for i in range(3))
        q, k = rope(q), rope(k)
        s = np.einsum("thd,shd->hts", q, k) / np.float32(np.sqrt(hd))
        s = s - s.max(-1, keepdims=True)
        p = np.exp(s)
        p = p / p.sum(-1, keepdims=True)
        a = np.einsum("hts,shd->thd", p, v).reshape(n, e)
        x = layer_norm(x + a @ l["out"].T, l["n1"], eps)
        g = x @ l["gate"].T
        h = (g / (1 + np.exp(-g))) * (x @ l["up"].T)
        x = layer_norm(x + h @ l["down"].T, l["n2"], eps)
    v = x.mean(0) if kv.get("nomic-bert.pooling_type", 1) != 2 else x[0]
    return (v / np.linalg.norm(v)).astype(np.float32)


def main():
    args = sys.argv[1:]
    if not args:
        sys.exit(__doc__)
    path = args.pop(0)
    golden = None
    ids_only = False
    texts = []
    while args:
        a = args.pop(0)
        if a == "--golden":
            golden = args.pop(0)
        elif a == "--ids-only":
            ids_only = True
        elif a == "--texts-file":
            texts.extend(json.load(open(args.pop(0), encoding="utf-8")))
        else:
            texts.append(a)
    m = load(path)
    enc = tokenizer(m["kv"])
    cases = []
    for t in texts:
        ids = [int(i) for i in enc(t)]
        case = {"text": t, "ids": ids}
        if not ids_only:
            case["vector"] = [round(float(x), 7) for x in embed_ids(m, np.array(ids))]
        cases.append(case)
        print(repr(t), len(ids), "tokens", ids[:12], "..." if len(ids) > 12 else "")
    if golden:
        json.dump({"model": os.path.basename(path), "cases": cases}, open(golden, "w"))
        print("written", golden)
    elif not ids_only and len(cases) > 1:
        vs = [np.array(c["vector"]) for c in cases]
        for i in range(len(vs)):
            for j in range(i + 1, len(vs)):
                print(f"cosine({i}, {j}) = {float(vs[i] @ vs[j]):.5f}")


if __name__ == "__main__":
    main()
