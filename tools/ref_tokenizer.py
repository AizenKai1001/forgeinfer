"""An executable SPECIFICATION of the byte-level BPE tokenizers we must reproduce (SmolLM2: Digits + GPT-2 regex; Qwen3: NFC + Qwen2 regex), written as plain character scanners with
no regex engine, checked against the 475 recorded golden cases of each model. The Rust tokenizer is a port of this file.
    python3 tools/ref_tokenizer.py smollm2|qwen3        (run on the homelab; reads the tokenizer.json of the oracle models)
"""
import json
import os
import sys
import unicodedata

ROOT = os.path.expanduser("~/forgeinfer/oracle/models")
MODELS = {"smollm2": "HuggingFaceTB__SmolLM2-135M-Instruct", "qwen3": "Qwen__Qwen3-0.6B", "qwen35": "Qwen__Qwen3.5-0.8B"}
GOLD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests", "golden")

# ---- character classes (Unicode general categories; White_Space as a fixed list because str.isspace() also accepts U+001C..U+001F, which regex \s does not)
WS = set([0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x20, 0x85, 0xA0, 0x1680, 0x2028, 0x2029, 0x202F, 0x205F, 0x3000] + list(range(0x2000, 0x200B)))


def is_space(c): return ord(c) in WS
def is_letter(c): return unicodedata.category(c)[0] == "L"
def is_number(c): return unicodedata.category(c)[0] == "N"
def is_mark(c): return unicodedata.category(c)[0] == "M"
def is_newline(c): return c == "\r" or c == "\n"


def bytes_to_unicode():
    """GPT-2's byte -> printable character table: printable bytes map to themselves, the others to U+0100 + n."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(0xA1, 0xAD)) + list(range(0xAE, 0x100))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


B2U = bytes_to_unicode()

CONTRACTIONS = ["s", "t", "re", "ve", "m", "ll", "d"]


def contraction_len(t, i, ignore_case):
    """Length of an English contraction ('s 't 're 've 'm 'll 'd) starting at t[i], or 0."""
    if t[i] != "'":
        return 0
    for suf in CONTRACTIONS:
        seg = t[i + 1:i + 1 + len(suf)]
        if (seg.lower() if ignore_case else seg) == suf:
            return 1 + len(suf)
    return 0


def run(t, i, pred):
    j = i
    while j < len(t) and pred(t[j]):
        j += 1
    return j


def ws_run_ending_before_nonspace(t, i):
    r"""`\\s+(?!\\S)`: the longest whitespace run from i that is not followed by a non-space (the last space of a run before a word is left for the word). Returns the end or i if none."""
    j = run(t, i, is_space)
    if j == i:
        return i
    if j == len(t):
        return j
    return j - 1 if j - 1 > i else i          # followed by a non-space character: give the last whitespace back; a single space cannot be split


def split_gpt2(t):
    r"""' ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+' after the case-sensitive contractions."""
    out, i = [], 0
    while i < len(t):
        n = contraction_len(t, i, False)
        if n:
            out.append(t[i:i + n]); i += n; continue
        j = i + 1 if t[i] == " " else i
        if j < len(t) and is_letter(t[j]):
            k = run(t, j, is_letter); out.append(t[i:k]); i = k; continue
        if j < len(t) and is_number(t[j]):
            k = run(t, j, is_number); out.append(t[i:k]); i = k; continue
        if j < len(t) and not is_space(t[j]) and not is_letter(t[j]) and not is_number(t[j]):
            k = run(t, j, lambda c: not is_space(c) and not is_letter(c) and not is_number(c)); out.append(t[i:k]); i = k; continue
        k = ws_run_ending_before_nonspace(t, i)
        if k > i:
            out.append(t[i:k]); i = k; continue
        k = run(t, i, is_space)
        out.append(t[i:k]); i = k
    return out


def split_qwen2(t, marks=False):
    r"""Qwen2 (Qwen3):  (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
    Qwen3.5 (marks=True): the letter class also takes combining marks: [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+ and  ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*  (Thai, Devanagari ... vowel signs are marks)."""
    lettish = (lambda c: is_letter(c) or is_mark(c)) if marks else is_letter
    out, i = [], 0
    while i < len(t):
        n = contraction_len(t, i, True)
        if n:
            out.append(t[i:i + n]); i += n; continue
        c = t[i]
        j = i + 1 if (not is_newline(c) and not is_letter(c) and not is_number(c)) else i     # optional one char that is not CR, LF, letter, number
        if j < len(t) and lettish(t[j]):
            k = run(t, j, lettish); out.append(t[i:k]); i = k; continue
        if is_number(c):
            out.append(c); i += 1; continue
        j = i + 1 if c == " " else i
        if j < len(t) and not is_space(t[j]) and not lettish(t[j]) and not is_number(t[j]):
            k = run(t, j, lambda x: not is_space(x) and not lettish(x) and not is_number(x))
            k = run(t, k, is_newline)
            out.append(t[i:k]); i = k; continue
        w = run(t, i, is_space)
        last_nl = max((p for p in range(i, w) if is_newline(t[p])), default=-1)
        if last_nl >= 0:
            out.append(t[i:last_nl + 1]); i = last_nl + 1; continue
        k = ws_run_ending_before_nonspace(t, i)
        if k > i:
            out.append(t[i:k]); i = k; continue
        out.append(t[i:w]); i = w
    return out


def split_digits_individual(t):
    out, buf = [], ""
    for c in t:
        if is_number(c):
            if buf:
                out.append(buf); buf = ""
            out.append(c)
        else:
            buf += c
    if buf:
        out.append(buf)
    return out


class Tok:
    def __init__(self, path, kind):
        j = json.load(open(path, encoding="utf-8"))
        self.kind = kind
        self.vocab = j["model"]["vocab"]
        self.rank = {}
        for r, m in enumerate(j["model"]["merges"]):
            a, b = (m if isinstance(m, list) else m.split(" "))
            self.rank[(a, b)] = r
        self.added = sorted(((a["content"], a["id"]) for a in j["added_tokens"]), key=lambda x: -len(x[0]))
        self.nfc = (j.get("normalizer") or {}).get("type") == "NFC"

    def bpe(self, piece):
        syms = [B2U[b] for b in piece.encode("utf-8")]
        while len(syms) > 1:
            best, at = None, -1
            for i in range(len(syms) - 1):
                r = self.rank.get((syms[i], syms[i + 1]))
                if r is not None and (best is None or r < best):
                    best, at = r, i
            if best is None:
                break
            syms[at:at + 2] = [syms[at] + syms[at + 1]]
        return [self.vocab[s] for s in syms]

    def pretokenize(self, text):
        if self.kind in ("qwen2", "qwen35"):
            return split_qwen2(text, marks=(self.kind == "qwen35"))
        pieces = []
        for part in split_digits_individual(text):
            pieces.extend(split_gpt2(part))
        return pieces

    def encode(self, text):
        # 1. added tokens are cut out of the raw text first (leftmost, longest), the rest is normalized and pre-tokenized
        segs, i, buf = [], 0, ""
        while i < len(text):
            hit = next(((s, tid) for s, tid in self.added if text.startswith(s, i)), None)
            if hit:
                if buf:
                    segs.append(("text", buf)); buf = ""
                segs.append(("id", hit[1])); i += len(hit[0])
            else:
                buf += text[i]; i += 1
        if buf:
            segs.append(("text", buf))
        ids = []
        for kind, v in segs:
            if kind == "id":
                ids.append(v)
                continue
            if self.nfc:
                v = unicodedata.normalize("NFC", v)
            for piece in self.pretokenize(v):
                ids.extend(self.bpe(piece))
        return ids


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "smollm2"
    tok = Tok(os.path.join(ROOT, MODELS[name], "tokenizer.json"), {"qwen3": "qwen2", "qwen35": "qwen35"}.get(name, "gpt2"))
    gold = json.load(open(os.path.join(GOLD, f"tok_{name}.json"), encoding="utf-8"))
    bad = 0
    for case in gold["cases"]:
        got = tok.encode(case["text"])
        if got != case["ids"]:
            bad += 1
            if bad <= 8:
                print("MISMATCH", repr(case["text"][:60]), "\n   want", case["ids"][:20], case["tokens"][:12], "\n   got ", got[:20])
    print(f"{name}: {len(gold['cases']) - bad}/{len(gold['cases'])} golden cases exact")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
