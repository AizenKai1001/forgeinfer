#!/usr/bin/env python3
"""sft_sources.py: adapters of the raw SFT sources of the Forge-1 chat mix (see ForgeInfer - SFT Data Mix in the research notes).

Every adapter is a generator  adapter(rng, frac) -> dicts {"id", "msgs": [[role, content], ...], "meta": {...}}  over the CANDIDATES of one source: the rows that pass the cheap metadata filters
of the source, thinned by a Bernoulli sample with probability `frac` (the pool builder asks for about twice the rows it needs). The content is RAW here: cleaning, filters, decontamination and
token counting happen in sft_pools.py. `meta["stratum"]` names the sub-population that the pool builder can give its own quota (category, dataset type ...).

CLI:  sft_sources.py extract_tulu    one pass over the six Tulu-3 parquet files, writes the sub-sources we use to sft_work/extract/tulu__<source>.jsonl (the parquet rows are sorted by source:
                                     scanning for one source at a time would read the whole 1.4 GB again and again)
      sft_sources.py peek NAME [N]   print N candidate conversations of a source
"""
import glob
import gzip
import json
import os
import random
import re
import sys

import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sft_lib import RAW, WORK  # noqa: E402

EXTRACT = os.path.join(WORK, "extract")


def files(pattern):
    return sorted(glob.glob(os.path.join(RAW, pattern)))


def pq_rows(pattern, columns=None, batch=4000):
    for p in files(pattern):
        pf = pq.ParquetFile(p)
        for b in pf.iter_batches(batch_size=batch, columns=columns):
            yield from b.to_pylist()


def jsonl_rows(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def pairs(messages):
    return [[m["role"], m["content"]] for m in messages]


# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# general chat

def magpie(rng, frac):
    """smol-magpie-ultra (Apache-2.0): only conversations rated good/excellent and very easy/easy/medium; the pool builder cuts every conversation to its first 1-3 exchanges."""
    i = 0
    for r in pq_rows("HuggingFaceTB__smoltalk/data/smol-magpie-ultra/train-*.parquet"):
        i += 1
        if r["quality"] not in ("good", "excellent") or r["difficulty"] not in ("very easy", "easy", "medium"):
            continue
        if rng.random() > frac:
            continue
        yield {"id": f"magpie:{i}", "msgs": pairs(r["messages"]), "meta": {"stratum": r["category"], "difficulty": r["difficulty"], "quality": r["quality"], "score": r["reward_model_score"]}}


def ultrachat(rng, frac):
    """ultrachat_200k train_sft (MIT)."""
    for r in pq_rows("HuggingFaceH4__ultrachat_200k/data/train_sft-*.parquet", columns=["prompt_id", "messages"]):
        if rng.random() > frac:
            continue
        yield {"id": f"ultrachat:{r['prompt_id']}", "msgs": pairs(r["messages"]), "meta": {}}


def _oasst_ok(m):
    if m.get("deleted") or m.get("lang") != "en" or m.get("review_result") is not True or m.get("synthetic"):
        return False
    tox = (m.get("detoxify") or {}).get("toxicity")
    return tox is None or tox < 0.3


def oasst2(rng, frac):
    """OpenAssistant oasst2 (Apache-2.0): per message tree the best-ranked path of English, reviewed, non-deleted, non-synthetic, non-toxic messages; ONE conversation per tree."""
    path = os.path.join(RAW, "OpenAssistant__oasst2", "2023-11-05_oasst2_ready.trees.jsonl.gz")
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            tree = json.loads(line)
            node, msgs = tree["prompt"], []
            while node is not None:
                if not _oasst_ok(node):
                    break
                msgs.append(["user" if node["role"] == "prompter" else "assistant", node["text"]])
                kids = [k for k in (node.get("replies") or []) if _oasst_ok(k)]
                if not kids:
                    break
                kids.sort(key=lambda k: (k.get("rank") is None, k.get("rank") if k.get("rank") is not None else 0))
                if node["role"] == "prompter" and len(kids) > 1 and kids[0].get("rank") not in (0, None):
                    break
                node = kids[0]
            if msgs and msgs[-1][0] == "user":
                msgs.pop()  # a conversation ends with the assistant
            if len(msgs) >= 2 and rng.random() <= frac:
                yield {"id": "oasst2:" + tree["message_tree_id"], "msgs": msgs, "meta": {}}


_CITE = re.compile(r"\[\d{1,3}\]")


def dolly(rng, frac):
    """databricks-dolly-15k (CC-BY-SA-3.0, human-written): instruction (+ context) -> response."""
    for i, r in enumerate(jsonl_rows(os.path.join(RAW, "databricks__databricks-dolly-15k", "databricks-dolly-15k.jsonl"))):
        if rng.random() > frac:
            continue
        ctx = _CITE.sub("", r.get("context") or "").strip()
        user = r["instruction"].strip() + ("\n\n" + ctx if ctx else "")
        yield {"id": f"dolly:{i}", "msgs": [["user", user], ["assistant", _CITE.sub("", r["response"]).strip()]], "meta": {"stratum": r.get("category", "")}}


def capybara(rng, frac):
    """LDJnr/Capybara CapybaraPure_Decontaminated (Apache-2.0): multi-turn, input/output pairs."""
    for i, r in enumerate(jsonl_rows(os.path.join(RAW, "LDJnr__Capybara", "CapybaraPure_Decontaminated.jsonl"))):
        if rng.random() > frac:
            continue
        msgs = []
        for t in r["conversation"]:
            msgs.append(["user", t["input"]])
            msgs.append(["assistant", t["output"]])
        yield {"id": f"capybara:{i}", "msgs": msgs, "meta": {"stratum": r.get("source", "")}}


def _smoltalk(name, tag):
    def gen(rng, frac):
        i = 0
        for r in pq_rows(f"HuggingFaceTB__smoltalk/data/{name}/train-*.parquet", columns=["messages"]):
            i += 1
            if rng.random() > frac:
                continue
            msgs = [m for m in pairs(r["messages"]) if not (m[0] == "system" and not (m[1] or "").strip())]
            yield {"id": f"{tag}:{i}", "msgs": msgs, "meta": {}}

    gen.__doc__ = f"smoltalk subset {name} (Apache-2.0 for the new subsets; see the license table of the research note)."
    return gen


everyday = _smoltalk("everyday-conversations", "everyday")
systemchats = _smoltalk("systemchats-30k", "systemchats")
summarize = _smoltalk("smol-summarize", "summarize")
rewrite = _smoltalk("smol-rewrite", "rewrite")
constraints = _smoltalk("smol-constraints", "constraints")
selfoss = _smoltalk("self-oss-instruct", "selfoss")


# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# math

_CALC = re.compile(r"<<[^<>]*>>")


def gsm8k(rng, frac):
    """GSM8K TRAIN (MIT, human-written): the calculator annotations are removed and the solution ends with the two closing lines '#### n' and 'The answer is: n' (the convention of MetaMathQA's GSM types)."""
    for i, r in enumerate(pq_rows("openai__gsm8k/main/train-*.parquet")):
        if rng.random() > frac:
            continue
        body, _, final = r["answer"].rpartition("####")
        body = _CALC.sub("", body).strip()
        final = final.strip().replace(",", "")
        if not body or not final:
            continue
        yield {"id": f"gsm8k:{i}", "msgs": [["user", r["question"].strip()], ["assistant", f"{body}\n#### {final}\nThe answer is: {final}"]], "meta": {}}


def _boxed(s):
    """Content of the last \\boxed{...} of a MATH solution (balanced braces), or None."""
    k = s.rfind("\\boxed")
    if k < 0:
        return None
    j = s.find("{", k)
    if j < 0:
        return None
    depth, out = 0, []
    for ch in s[j:]:
        if ch == "{":
            depth += 1
            if depth == 1:
                continue
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return "".join(out)
        out.append(ch)
    return None


def hendrycks_math(rng, frac):
    """MATH TRAIN (MIT, human-written), levels 1-3 only; the solution gets the closing line 'The answer is: ...' (the style of MetaMathQA)."""
    for p in files("EleutherAI__hendrycks_math/*/train-*.parquet"):
        for i, r in enumerate(pq.ParquetFile(p).read().to_pylist()):
            if r["level"] not in ("Level 1", "Level 2", "Level 3") or rng.random() > frac:
                continue
            ans = _boxed(r["solution"])
            if not ans:
                continue
            yield {"id": f"math:{os.path.basename(os.path.dirname(p))}:{i}", "msgs": [["user", r["problem"].strip()], ["assistant", r["solution"].strip() + f"\nThe answer is: {ans}"]], "meta": {"stratum": r["type"]}}


METAMATH_WEIGHTS = {"GSM_AnsAug": 1.0, "GSM_Rephrased": 0.8, "GSM_SV": 0.25, "GSM_FOBAR": 0.25, "MATH_AnsAug": 0.35, "MATH_Rephrased": 0.0, "MATH_SV": 0.0, "MATH_FOBAR": 0.0}


def metamath(rng, frac):
    """MetaMathQA (MIT; GPT-3.5 rewrites of GSM8K/MATH TRAIN questions): mostly the GSM types, a few MATH_AnsAug; the responses already end with 'The answer is: ...'."""
    with open(os.path.join(RAW, "meta-math__MetaMathQA", "MetaMathQA-395K.json"), encoding="utf-8") as f:
        data = json.load(f)
    for i, r in enumerate(data):
        w = METAMATH_WEIGHTS.get(r["type"], 0.0)
        if w == 0.0 or rng.random() > frac * w:
            continue
        yield {"id": f"metamath:{i}", "msgs": [["user", r["query"].strip()], ["assistant", r["response"].strip()]], "meta": {"stratum": r["type"]}}
    del data


def orcamath(rng, frac):
    """orca-math-word-problems-200k (MIT; GPT-4-Turbo answers)."""
    for i, r in enumerate(pq_rows("microsoft__orca-math-word-problems-200k/data/*.parquet")):
        if rng.random() > frac:
            continue
        yield {"id": f"orcamath:{i}", "msgs": [["user", r["question"].strip()], ["assistant", r["answer"].strip()]], "meta": {}}


# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# code

def magicoder(rng, frac):
    """Magicoder-OSS-Instruct-75K, decontaminated file (MIT; gpt-3.5-turbo-1106): all Python, a few shell/TypeScript."""
    other = {"shell": 0.35, "typescript": 0.15}
    for i, r in enumerate(jsonl_rows(os.path.join(RAW, "ise-uiuc__Magicoder-OSS-Instruct-75K", "data-oss_instruct-decontaminated.jsonl"))):
        w = 1.0 if r["lang"] == "python" else other.get(r["lang"], 0.0)
        if w == 0.0 or rng.random() > frac * w:
            continue
        yield {"id": f"magicoder:{i}", "msgs": [["user", r["problem"].strip()], ["assistant", r["solution"].strip()]], "meta": {"stratum": r["lang"]}}


# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# the Tulu-3 SFT mixture (ODC-BY-1.0 for the collection; per-subset licenses are in the card)

TULU_WANTED = {
    "coconot": "ai2-adapt-dev/coconot_converted",
    "wildguard": "ai2-adapt-dev/tulu_v3.9_synthetic_finalresp_wildguardmixtrain_decontaminated_50k",
    "tulu_gsm": "ai2-adapt-dev/tulu_v3.9_open_math_2_gsm8k_50k",
    "tulu_grade": "allenai/tulu-3-sft-personas-math-grade",
    "tulu_code": "ai2-adapt-dev/personahub_code_v2_34999",
    "evolcode": "ai2-adapt-dev/evol_codealpaca_heval_decontaminated",
}


def extract_tulu():
    os.makedirs(EXTRACT, exist_ok=True)
    want = {v: k for k, v in TULU_WANTED.items()}
    outs = {k: open(os.path.join(EXTRACT, f"tulu__{k}.jsonl"), "w", encoding="utf-8") for k in TULU_WANTED}
    n = {k: 0 for k in TULU_WANTED}
    for p in files("allenai__tulu-3-sft-mixture/data/*.parquet"):
        pf = pq.ParquetFile(p)
        for b in pf.iter_batches(batch_size=4000, columns=["id", "messages", "source"]):
            for r in b.to_pylist():
                k = want.get(r["source"])
                if k is None:
                    continue
                n[k] += 1
                outs[k].write(json.dumps({"id": r["id"], "msgs": pairs(r["messages"])}, ensure_ascii=False) + "\n")
        print("done", os.path.basename(p), n, flush=True)
    for f in outs.values():
        f.close()


_BOXED = re.compile(r"\\boxed\{([^{}]*)\}")
_NUMERIC = re.compile(r"-?\$?\d[\d,]*\.?\d*%?")
_HASH_ANS = re.compile(r"\n#{3,5}\s*(-?\$?\d[\d,]*\.?\d*)\s*$")


def _fix_boxed(text):
    """OpenMathInstruct style 'is \\boxed{95}.' -> 'is 95.' plus the closing lines '#### 95' and 'The answer is: 95' (the same ending in every grade-school math source); None when the answer is not a plain number."""
    m = list(_BOXED.finditer(text))
    if not m:
        return None
    ans = m[-1].group(1).strip().replace(",", "").replace("$", "")
    if not _NUMERIC.fullmatch(ans):
        return None
    return text[: m[-1].start()] + m[-1].group(1) + text[m[-1].end() :].rstrip() + f"\n#### {ans}\nThe answer is: {ans}"


def _fix_hash(text):
    """GSM8K style closing line '#### 132' is kept and 'The answer is: 132' is added after it; None when the text has no such line."""
    m = _HASH_ANS.search(text)
    if not m:
        return None
    n = m.group(1).replace(",", "").replace("$", "")
    return text[: m.start()].rstrip() + f"\n#### {n}\nThe answer is: {n}"


def _tulu(key, python_only=False, post=None):
    def gen(rng, frac):
        for r in jsonl_rows(os.path.join(EXTRACT, f"tulu__{key}.jsonl")):
            if rng.random() > frac:
                continue
            msgs = r["msgs"]
            if python_only:
                user = "\n".join(c for ro, c in msgs if ro == "user").lower()
                ass = "\n".join(c for ro, c in msgs if ro == "assistant")
                if "```python" not in ass and "python" not in user:
                    continue
            if post is not None:
                fixed = post(msgs[-1][1])
                if fixed is None:
                    continue
                msgs = msgs[:-1] + [[msgs[-1][0], fixed]]
            yield {"id": f"{key}:{r['id']}", "msgs": msgs, "meta": {}}

    gen.__doc__ = f"tulu-3-sft-mixture sub-source {TULU_WANTED[key]}"
    return gen


coconot = _tulu("coconot")
wildguard = _tulu("wildguard")
tulu_gsm = _tulu("tulu_gsm", post=_fix_boxed)
tulu_grade = _tulu("tulu_grade", post=_fix_hash)
tulu_code = _tulu("tulu_code")
evolcode = _tulu("evolcode", python_only=True)

ADAPTERS = {
    "magpie": magpie, "ultrachat": ultrachat, "oasst2": oasst2, "dolly": dolly, "capybara": capybara, "everyday": everyday, "systemchats": systemchats, "summarize": summarize,
    "rewrite": rewrite, "constraints": constraints, "selfoss": selfoss, "gsm8k": gsm8k, "hendrycks": hendrycks_math, "metamath": metamath, "orcamath": orcamath, "magicoder": magicoder,
    "coconot": coconot, "wildguard": wildguard, "tulu_gsm": tulu_gsm, "tulu_grade": tulu_grade, "tulu_code": tulu_code, "evolcode": evolcode,
}

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "extract_tulu":
        extract_tulu()
    elif cmd == "peek":
        name, n = sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 2
        rng = random.Random(1)
        for k, c in enumerate(ADAPTERS[name](rng, 0.05)):
            print(json.dumps(c, ensure_ascii=False)[:1500])
            print()
            if k + 1 >= n:
                break
    else:
        print(__doc__)
