"""Shared helpers of the Forge-1 SFT data pipeline (see ForgeInfer - SFT Data Mix in the research notes).

Everything here is deterministic and stdlib + numpy + tokenizers only. Conversations are lists of [role, content] pairs, role in {system, user, assistant}.
The rendering is EXACTLY the plain ChatML of crates/forge-chat (`render_chatml`): <|im_start|>role\\ncontent<|im_end|>\\n per message, no injected system turn; one document per conversation
(documents are separated by a NUL byte in the corpus files; `forge tok-encode` appends <|endoftext|> after each document).
"""
import hashlib
import json
import os

os.environ.setdefault("RAYON_NUM_THREADS", "2")  # the tokenizers library: at most two threads (the production run shares this machine)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import random
import re
import zlib

import numpy as np

TOKENIZER = os.environ.get("FORGE_TOKENIZER", "/home/user/forgeinfer/data/tokenizer_f1.json")
RAW = "/mnt/data/forgeinfer-data/sft_raw"
WORK = "/mnt/data/forgeinfer-data/sft_work"
OUT = "/mnt/data/forgeinfer-data/tokens_sft_mix2"
SEED = 20261010

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# text cleaning

SPECIAL_STRINGS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|pad|>", "<|reserved_", "<tool_call>", "</tool_call>", "<tool_response>", "</tool_response>", "<think>", "</think>"]
_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f​‌‍⁠﻿]")
_TRAIL = re.compile(r"[ \t]+\n")
_MANY_NL = re.compile(r"\n{4,}")


def clean(s):
    """Normalise one message: newlines, control characters (NUL is the document separator!), trailing blanks, runs of blank lines; strip the ends (a leading blank would change the role header's tokens)."""
    if not isinstance(s, str):
        return ""
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = _CTRL.sub("", s)
    s = _TRAIL.sub("\n", s)
    s = _MANY_NL.sub("\n\n\n", s)
    return s.strip()


def render(msgs):
    return "".join(f"<|im_start|>{r}\n{c}<|im_end|>\n" for r, c in msgs)


def has_special(text):
    return any(t in text for t in SPECIAL_STRINGS)


# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# quality filters: every function returns a reason string (drop) or None (keep)

_VENDOR = (
    r"(?:open\s?ai|chat\s?gpt|gpt-?\d[\w.\-]*|\bgpt\b|anthropic|claude|google|gemini|\bbard\b|\bpalm\b|deepmind|\bmeta\b|meta ai|llama[\w.\-]*|facebook|microsoft|copilot|\bbing\b|alibaba|qwen|tongyi|deepseek|mistral|mixtral|cohere|"
    r"hugging\s?face|smollm|allen institute|\bai2\b|\btulu\b|\btülu\b|stability ai|nvidia|amazon|alexa|\bsiri\b|baidu|ernie|zhipu|chatglm|vicuna|alpaca|lmsys|falcon|databricks|\bdolly\b|\bxai\b|grok|inflection|"
    r"perplexity|character\.ai|replika|\bkimi\b|moonshot|minimax|\bibm\b|watson|granite|\bphi-\d|wizardlm|openchat|open\s?assistant|\bpi\b ai|orca|zephyr|nous research|hermes|magpie|argilla|distilabel)"
)
_SELF = r"(?:\bi am\b|\bi'm\b|\bi’m\b|\bmy name is\b|\bcall me\b|\bi was\b|\bi have been\b|\bi've been\b|\bi’ve been\b|\bi\b[^.?!\n]{0,40}\b(?:trained|created|developed|built|made|designed|powered|fine-tuned)\b)"
IDENT_RES = [
    re.compile(_SELF + r"[^.?!\n]{0,90}" + _VENDOR, re.I),
    re.compile(r"\b(?:created|developed|built|trained|made|designed|released|fine-tuned|powered|owned)\b[^.?!\n]{0,25}\bby\b[^.?!\n]{0,40}" + _VENDOR + r"[^.?!\n]{0,60}\b(?:i|me|my)\b", re.I),
    re.compile(r"\b(?:i|me|my)\b[^.?!\n]{0,80}\b(?:created|developed|built|trained|made|designed|released|fine-tuned|powered)\b[^.?!\n]{0,25}\bby\b[^.?!\n]{0,40}" + _VENDOR, re.I),
    re.compile(r"\bchat\s?gpt\b", re.I),
    re.compile(r"\bas an? (?:ai|artificial intelligence)(?: language)? (?:model|assistant|chatbot)\b", re.I),
    re.compile(r"\bas an? (?:large )?language model\b", re.I),
    re.compile(r"\bi(?:'m| am) an? (?:large )?language model\b", re.I),
    re.compile(r"\b(?:my|the) (?:training|knowledge) (?:data )?(?:cutoff|cut-off|date)\b|\bknowledge cut-?off\b|\bas of my (?:last|latest|knowledge|training)\b|\bmy training data (?:only )?(?:goes|extends|includes|ends)\b", re.I),
    re.compile(r"\byou are (?:an? )?(?:chat\s?gpt|claude|gemini|bard|llama|gpt)\b", re.I),
]
_MARKUP_ARTIFACTS = re.compile(r"\[/?INST\]|<</?SYS>>|</?s>|###\s*(?:Instruction|Response|Input|Output)\s*:|<\|[a-z_]+\|>|\bBEGININPUT\b|\bENDINPUT\b|\bBEGININSTRUCTION\b|\bENDINSTRUCTION\b", re.I)
_HTML_JUNK = re.compile(r"</?(?:div|span|p|br|table|tr|td|ul|li|html|body|head|script|style)\b[^>]*>|&nbsp;|&amp;|&lt;|&gt;|&quot;", re.I)
_ROLE_LEAK = re.compile(r"^\s*(?:Human|User|Assistant|AI|Bot|System)\s*:", re.I | re.M)
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_EMAIL_OK = re.compile(r"@(?:example|test|domain|email|mail|yourdomain|yourcompany|company|sample|acme|localhost)[\w.-]*\.", re.I)
_PHONE = re.compile(r"(?<!\d)(?:\+?\d{1,2}[ .-])?\(?\d{3}\)?[ .-]\d{3}[ .-]\d{4}(?!\d)")
_URL = re.compile(r"https?://\S+")
_WORD = re.compile(r"[a-z']+")
_STOP = {"the", "and", "to", "of", "a", "in", "is", "that", "it", "for", "you", "with", "as", "are", "this", "on", "be", "or", "can", "your", "have", "not", "an", "by", "i", "was", "if", "at"}
_BAD_WORDS = re.compile(
    r"\b(?:fuck\w*|shit\w*|bitch\w*|cunt\w*|nigg\w*|faggot\w*|retard\w*|porn\w*|nsfw|erotic\w*|orgasm\w*|cock|pussy|masturbat\w*|rape\w*|rapist\w*|incest\w*|hentai|onlyfans|slut\w*|whore\w*|dildo\w*|blowjob\w*|horny|boobs?|titties|handjob\w*|bdsm|fetish\w*)\b", re.I)
_URL_TOKEN = re.compile(r"https?://[^\s)>\]\"'`]+|www\.[^\s)>\]\"'`]+")


def fake_url(msgs):
    """An assistant message that contains a URL that appears in no earlier message of the conversation: small models learn to invent links from such answers."""
    seen = ""
    for r, c in msgs:
        if r == "assistant":
            for m in _URL_TOKEN.finditer(c):
                u = m.group(0).rstrip(".,;:!?")
                if u not in seen:
                    return u[:60]
        seen += "\n" + c
    return None


def ascii_ratio(s):
    return len(s.encode("ascii", "ignore")) / max(1, len(s))


_FENCE = re.compile(r"```.*?```", re.S)


def english_ok(text):
    if ascii_ratio(text) < 0.90:
        return False
    prose = _FENCE.sub(" ", text)
    words = _WORD.findall(prose.lower())
    if len(words) >= 25:
        stop = sum(1 for w in words if w in _STOP)
        if stop / len(words) < 0.12:
            return False
    return True


_ANYWHERE = re.compile(r"\bchat\s?gpt\b|\bopen\s?ai\b|\bgpt-?[34]\b|\banthropic\b|\bclaude\b", re.I)


_IDENT_GROUPS = {"vendor": [0, 1, 2, 3, 8], "boiler": [4, 5, 6], "cutoff": [7]}


def identity_hit(text, groups=("vendor", "boiler", "cutoff")):
    """Other vendors' identity boilerplate in the assistant/system text (patterns IDENT_RES): 'vendor' = claims to be / to be made by another company or model, 'boiler' = 'as an AI language model',
    'cutoff' = statements about a training or knowledge cut-off date (a made-up date would be a false claim of Forge-1)."""
    for g in groups:
        for i in _IDENT_GROUPS[g]:
            m = IDENT_RES[i].search(text)
            if m:
                return m.group(0)[:80]
    return None


def fences_ok(text):
    return sum(1 for ln in text.split("\n") if ln.lstrip().startswith("```")) % 2 == 0


def degenerate(text):
    if len(text) < 400:
        return False
    z = len(zlib.compress(text.encode("utf-8", "ignore"), 6)) / max(1, len(text.encode("utf-8", "ignore")))
    if z < 0.10:
        return True
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    if len(lines) >= 10 and len(set(lines)) / len(lines) < 0.45 and not any(ln.startswith(("```", "|", "-", "*", "#")) for ln in lines[:3]):
        return True
    return False


def looks_truncated(text):
    """An answer that ends in the middle of a sentence: long, last character a letter or comma, last line long."""
    if len(text) < 700:
        return False
    last = text.rstrip()[-1:]
    tail = text.rstrip().split("\n")[-1]
    return (last.isalpha() or last in ",;(") and len(tail) > 100


def basic_checks(msgs, *, allow_html=False, check_pii=True, need_english=True, min_answer_chars=12, ban_words=False, check_urls=True, check_vendor=True, ident_groups=("vendor", "boiler", "cutoff")):
    """Checks every conversation passes. Returns a reason or None."""
    roles = [r for r, _ in msgs]
    if not msgs or roles[-1] != "assistant":
        return "last_not_assistant"
    i = 0
    if roles[0] == "system":
        i = 1
    exp = "user"
    for r in roles[i:]:
        if r != exp:
            return "bad_role_order"
        exp = "assistant" if exp == "user" else "user"
    full = "\n".join(c for _, c in msgs)
    for r, c in msgs:
        if not c:
            return "empty_message"
        if r == "assistant" and len(c) < min_answer_chars:
            return "short_answer"
    if has_special(full):
        return "special_token_text"
    if "�" in full or "Ã¢" in full or "â€" in full or "Ã©" in full:
        return "mojibake"
    if need_english and not english_ok(full):
        return "not_english"
    if check_vendor and _ANYWHERE.search(full):
        return "vendor_mention"
    if check_vendor and identity_hit("\n".join(c for r, c in msgs if r != "user"), ident_groups):
        return "identity_boilerplate"
    if check_urls and fake_url(msgs):
        return "fabricated_url"
    if _MARKUP_ARTIFACTS.search(full):
        return "markup_artifact"
    if not allow_html and _HTML_JUNK.search(full):
        return "html_junk"
    for r, c in msgs:
        if not fences_ok(c):
            return "unbalanced_fence"
        if r == "assistant":
            if degenerate(c):
                return "degenerate"
            if _ROLE_LEAK.search(c):
                return "role_leak"
    if looks_truncated(msgs[-1][1]):
        return "truncated"
    if check_pii:
        for m in _EMAIL.finditer(full):
            if not _EMAIL_OK.search(m.group(0)):
                return "email"
        if _PHONE.search(full):
            return "phone"
    if ban_words and _BAD_WORDS.search(full):
        return "explicit_words"
    for _, c in msgs:
        if c.count("\\n") >= 3 and "\n" not in c:
            return "escaped_newlines"
    return None


# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# n-grams for the decontamination and the near-duplicate sketches

_TOKRE = re.compile(r"[a-z0-9]+")
_P = np.uint64(0x9E3779B97F4A7C15)
_WCACHE = {}


def words_of(text):
    return _TOKRE.findall(text.lower())


def word_ids(text):
    out = []
    cache = _WCACHE
    if len(cache) > 3_000_000:
        cache.clear()
    for w in _TOKRE.findall(text.lower()):
        v = cache.get(w)
        if v is None:
            v = (zlib.crc32(w.encode()) * 0x9E3779B1 + len(w) * 0x85EBCA6B) & 0xFFFFFFFFFFFFFFFF
            v = (v * 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
            cache[w] = v
        out.append(v)
    return np.array(out, dtype=np.uint64)


def ngram_hashes(ids, n):
    """Hashes of all n-grams of an array of word ids (a polynomial hash with uint64 wrap-around)."""
    L = len(ids)
    if L < n:
        return np.zeros(0, dtype=np.uint64)
    h = np.zeros(L - n + 1, dtype=np.uint64)
    with np.errstate(over="ignore"):
        for k in range(n):
            h = h * _P + ids[k : L - n + 1 + k]
    return h


# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# the benchmark index (built by sft_bench_index.py)

class BenchIndex:
    def __init__(self, path=None, n=10):
        path = path or os.path.join(WORK, "bench_index.npz")
        z = np.load(path, allow_pickle=True)
        self.n = int(z["n"])
        self.hashes = z["hashes"]  # sorted unique uint64
        self.owner = z["owner"]  # benchmark id of every hash (index into names), same order as hashes
        self.names = [str(x) for x in z["names"]]

    def hits(self, text):
        """Names of the benchmarks that share at least one n-gram with `text` (empty list if none)."""
        ids = word_ids(text)
        h = ngram_hashes(ids, self.n)
        if len(h) == 0:
            return []
        pos = np.searchsorted(self.hashes, h)
        pos[pos >= len(self.hashes)] = 0
        ok = self.hashes[pos] == h
        if not ok.any():
            return []
        return sorted({self.names[o] for o in np.unique(self.owner[pos[ok]])})


# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# token counting with the HF tokenizers library (verified identical to `forge tokenize` on 37 test strings, see ForgeInfer - SFT Data Mix)

_TOK = None


def tokenizer():
    global _TOK
    if _TOK is None:
        from tokenizers import Tokenizer

        _TOK = Tokenizer.from_file(TOKENIZER)
    return _TOK


def count_tokens(texts):
    """Number of tokens of every text (specials in the text are recognised like in `forge tok-encode`)."""
    tok = tokenizer()
    out = []
    for i in range(0, len(texts), 1024):
        out.extend(len(e.ids) for e in tok.encode_batch(texts[i : i + 1024]))
    return out


def stable_seed(*parts):
    return (SEED + zlib.crc32("|".join(str(p) for p in parts).encode())) & 0x7FFFFFFF


def rng_for(*parts):
    return random.Random(stable_seed(*parts))


def write_jsonl(path, records):
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# token accounting of the rendered ChatML document:  sum over messages of (header + content + tail)  +  1 (<|endoftext|> appended by `forge tok-encode`)
# the pieces never merge across the special tokens, so the sum equals the token count of the whole document (checked against the encoder in sft_verify.py)

_HDR = {}
_TAIL = [None]


def header_tokens(role):
    """Tokens of '<|im_start|>role\n'."""
    if role not in _HDR:
        _HDR[role] = len(tokenizer().encode(f"<|im_start|>{role}\n").ids)
    return _HDR[role]


def tail_tokens():
    """Tokens of '<|im_end|>\n'."""
    if _TAIL[0] is None:
        _TAIL[0] = len(tokenizer().encode("<|im_end|>\n").ids)
    return _TAIL[0]


def content_token_counts(convs):
    """convs: list of message lists -> list of lists with the token count of every message's content (one batched call of the tokenizer)."""
    flat = [c for msgs in convs for _, c in msgs]
    cnt = []
    tok = tokenizer()
    for i in range(0, len(flat), 2048):
        cnt.extend(len(e.ids) for e in tok.encode_batch(flat[i : i + 2048]))
    out, k = [], 0
    for msgs in convs:
        out.append(cnt[k : k + len(msgs)])
        k += len(msgs)
    return out


def doc_tokens(msgs, counts):
    return sum(header_tokens(r) + n + tail_tokens() for (r, _), n in zip(msgs, counts)) + 1


def supervised_tokens(msgs, counts):
    """Tokens that carry a loss after `forge chat-mask`: the assistant's text and its <|im_end|>."""
    return sum(n + 1 for (r, _), n in zip(msgs, counts) if r == "assistant")


def fit_prefix(msgs, counts, max_tok, max_assistant=99):
    """Longest prefix of the conversation that ends with an assistant message, has at most `max_assistant` assistant turns and at most `max_tok` document tokens.
    Returns (msgs, counts) or None when not even the first exchange fits."""
    total, best, n_ass = 1, 0, 0
    for i, ((r, _), n) in enumerate(zip(msgs, counts)):
        total += header_tokens(r) + n + tail_tokens()
        if total > max_tok:
            break
        if r == "assistant":
            n_ass += 1
            if n_ass > max_assistant:
                break
            best = i + 1
    if best == 0:
        return None
    return msgs[:best], counts[:best]


# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# duplicates

_NORM = re.compile(r"[^a-z0-9]+")


def norm_text(s):
    return _NORM.sub(" ", s.lower()).strip()


def text_hash(s):
    """64-bit hash of the normalised text (letters and digits, lower case)."""
    return int.from_bytes(hashlib.blake2b(norm_text(s).encode("utf-8", "ignore"), digest_size=8).digest(), "little")


def conv_hash(msgs):
    return text_hash("\n".join(c for r, c in msgs if r != "system"))


def prompt_hash(msgs):
    """Hash of everything the user said (the same prompts answered twice are a duplicate even when the answers differ; a greeting alone does not make two chats the same)."""
    return text_hash("\n".join(c for r, c in msgs if r == "user"))


# MinHash of word shingles for near duplicates: K permutations x^64 mix, bands of R rows
MH_K = 64
_MH_A = np.array([(0x9E3779B97F4A7C15 * (2 * i + 1) + 0x1234567) & 0xFFFFFFFFFFFFFFFF | 1 for i in range(MH_K)], dtype=np.uint64)
_MH_B = np.array([(0xC2B2AE3D27D4EB4F * (i + 7) + 0x2545F4914F6CDD1D) & 0xFFFFFFFFFFFFFFFF for i in range(MH_K)], dtype=np.uint64)


def minhash(text, n=3):
    """MinHash signature (uint64[K]) of the word n-gram shingles of `text`; None when the text has fewer than n+2 words (short texts are only compared exactly)."""
    ids = word_ids(text)
    if len(ids) < n + 2:
        return None
    h = np.unique(ngram_hashes(ids, n))
    with np.errstate(over="ignore"):
        return (h[:, None] * _MH_A[None, :] + _MH_B[None, :]).min(axis=0)


def shingle_set(text, n=3):
    ids = word_ids(text)
    return set(ngram_hashes(ids, n).tolist()) if len(ids) >= n else set()


def jaccard(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)
