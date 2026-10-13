#!/usr/bin/env python3
"""Build the data of the Forge-1 evaluation ladder (evaluation splits ONLY, never training splits) and a manifest with sizes and checksums.

    python3 tools/f1_eval_data.py [--out DIR] [--tokenizer FILE] [--val-domains DIR] [--tasks a,b,...] [--seed N] [--offline]

Default output: /mnt/data/forgeinfer-data/eval (the homelab). Needs pyarrow and tokenizers (the homelab venv ~/forgeinfer/venv has both); no GPU.

Layout of OUT (the driver tools/f1_eval.sh reads MANIFEST.json, which is the single source of truth for task -> file -> command):
    raw/        the downloaded source files (parquet / jsonl) of the pinned dataset revisions, as downloaded
    mc/         one jsonl per multiple-choice task for `forge eval-mc`:  {"choices": [{"ctx": ..., "cont": " ..."}, ...], "label": k}
    hellaswag/  hellaswag_val.jsonl for `forge eval-hellaswag` (the raw HellaSwag fields)
    lambada/    lambada_openai.jsonl for `forge eval-lambada`:  {"ctx": "... text before the last word", "cont": " last-word"}
    ppl/        copies of the per-domain validation token files (uint16) for `forge eval-ppl`
    MANIFEST.json, MANIFEST.md

Every task file is SHUFFLED once with a fixed seed, so `--limit N` (the first N lines) is a uniform random subsample of the evaluation split (the quick mode of the driver); a full run gives the
same numbers whatever the order. The prompts and the scoring follow EleutherAI's lm-evaluation-harness (zero-shot; see the research note "ForgeInfer - Evaluation Ladder").
"""
import argparse
import hashlib
import json
import math
import os
import platform
import random
import shutil
import sys
import time
import urllib.parse
import urllib.request

UA = {"User-Agent": "forgeinfer-tools"}
HUB = "https://huggingface.co"
SEED = 20261010
SEQ_STEPS = [64, 128, 256, 512, 1024]  # powers of two: the trainer's forward needs batch x seq <= 2048 or a multiple of 2048 (cross-entropy chunk), and the kernels like multiples of 64


# ---------------------------------------------------------------- hub access
def http_json(url, retries=4):
    last = None
    for a in range(retries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
                return json.load(r), r.headers
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(2 * (a + 1))
    raise RuntimeError(f"{url}: {last}")


def hub_info(repo, ref=None):
    """commit sha and license of a dataset repository (at `ref`: a branch name such as refs/convert/parquet, or None for main)"""
    j, _ = http_json(f"{HUB}/api/datasets/{repo}")
    card = j.get("cardData") or {}
    lic = card.get("license")
    if isinstance(lic, list):
        lic = ",".join(lic)
    sha = j.get("sha")
    if ref:
        k, _ = http_json(f"{HUB}/api/datasets/{repo}/revision/{urllib.parse.quote(ref, safe='')}")
        sha = k.get("sha")
    return {"repo": repo, "ref": ref or "main", "commit": sha, "license_metadata": lic}


def hub_get(repo, commit, path, dest, expect_size=None, retries=4):
    url = f"{HUB}/datasets/{repo}/resolve/{commit}/{urllib.parse.quote(path)}"
    last = None
    for a in range(retries):
        try:
            tmp = dest + ".part"
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=180) as r, open(tmp, "wb") as f:
                shutil.copyfileobj(r, f)
            if expect_size is not None and os.path.getsize(tmp) != expect_size:
                raise RuntimeError(f"size {os.path.getsize(tmp)} != {expect_size}")
            os.replace(tmp, dest)
            return url
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(3 * (a + 1))
    raise RuntimeError(f"{url}: {last}")


def hub_tree_sizes(repo, commit):
    """{path: size} of the files of the repository at a commit"""
    out = {}
    url = f"{HUB}/api/datasets/{repo}/tree/{commit}?recursive=true"
    while url:
        j, headers = http_json(url)
        for i in j:
            if i.get("type") == "file":
                out[i["path"]] = i.get("size")
        url = None
        for part in headers.get("Link", "").split(","):
            if 'rel="next"' in part:
                url = part.split(";")[0].strip().strip("<>")
    return out


def readme_license_lines(repo, commit):
    """what the dataset card says about the license: the first non-empty line under every 'Licensing' heading, and the 'license:' entries of the front matter"""
    try:
        req = urllib.request.Request(f"{HUB}/datasets/{repo}/resolve/{commit}/README.md", headers=UA)
        with urllib.request.urlopen(req, timeout=60) as r:
            txt = r.read().decode("utf-8", "replace")
    except Exception:  # noqa: BLE001
        return []
    lines = txt.splitlines()
    out = []
    for i, ln in enumerate(lines):
        if ln.startswith("#") and "licens" in ln.lower() and "table" not in ln.lower():
            nxt = [x.strip() for x in lines[i + 1:i + 6] if x.strip() and not x.startswith("#")]
            if nxt:
                out.append(f"{ln.strip('# ').strip()}: {nxt[0][:160]}")
        if ln.strip() == "license:" and i + 1 < len(lines):
            out.append("front matter license: " + lines[i + 1].strip("- ").strip())
    return out[:4]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


# ---------------------------------------------------------------- converters (the lm-evaluation-harness prompts, zero-shot)
def conv_arc(r):
    labels = list(r["choices"]["label"])
    return {"choices": [{"ctx": f"Question: {r['question']}\nAnswer:", "cont": " " + t} for t in r["choices"]["text"]], "label": labels.index(r["answerKey"])}


def conv_piqa(r):
    ctx = f"Question: {r['goal']}\nAnswer:"
    return {"choices": [{"ctx": ctx, "cont": " " + r["sol1"]}, {"ctx": ctx, "cont": " " + r["sol2"]}], "label": int(r["label"])}


def conv_winogrande(r):
    # the harness's partial scoring: the sentence prefix with each option filled in is the context, the rest of the sentence is the scored continuation
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
    return {"choices": [{"ctx": r["question_stem"], "cont": " " + t} for t in r["choices"]["text"]], "label": labels.index(r["answerKey"].strip())}


def mmlu_subject(r):
    return r["subject"].replace("_", " ")


def conv_mmlu_letter(r):
    # lm-evaluation-harness `mmlu` (zero-shot): description + question + lettered options, the four letters are the candidate continuations
    desc = f"The following are multiple choice questions (with answers) about {mmlu_subject(r)}.\n\n"
    c = list(r["choices"])
    ctx = desc + r["question"].strip() + "".join(f"\n{L}. {c[i]}" for i, L in enumerate("ABCD")) + "\nAnswer:"
    return {"choices": [{"ctx": ctx, "cont": f" {L}"} for L in "ABCD"], "label": int(r["answer"])}


def conv_mmlu_cloze(r):
    # cloze formulation (the answer TEXT is scored, as in the SmolLM2 / OLMES "MMLU (cloze)" rows): question, then each answer text as the continuation
    ctx = f"The following are questions about {mmlu_subject(r)}.\nQuestion: {r['question'].strip()}\nAnswer:"
    return {"choices": [{"ctx": ctx, "cont": " " + str(t).strip()} for t in r["choices"]], "label": int(r["answer"])}


def conv_hellaswag(r):
    return {k: r[k] for k in ("ind", "activity_label", "ctx_a", "ctx_b", "ctx", "endings", "source_id", "split", "split_type", "label")}


# ---------------------------------------------------------------- task table
# kind: mc (forge eval-mc), hellaswag (forge eval-hellaswag), lambada (forge eval-lambada)
# files: (repo, ref or None, path in the repo, reader); headline: the metric the driver puts first (see the research note for the published conventions)
TASKS = {
    "hellaswag": dict(title="HellaSwag", kind="hellaswag", repo="Rowan/hellaswag", paths=["data/validation-00000-of-00001.parquet"], split="validation", n=10042, chance=25.0, headline="acc_norm",
                      harness="hellaswag", quick=1000, out="hellaswag/hellaswag_val.jsonl", conv=conv_hellaswag),
    "arc_easy": dict(title="ARC-Easy", kind="mc", repo="allenai/ai2_arc", paths=["ARC-Easy/test-00000-of-00001.parquet"], split="test", n=2376, chance=25.0, headline="acc",
                     harness="arc_easy", quick=500, out="mc/arc_easy.jsonl", conv=conv_arc),
    "arc_challenge": dict(title="ARC-Challenge", kind="mc", repo="allenai/ai2_arc", paths=["ARC-Challenge/test-00000-of-00001.parquet"], split="test", n=1172, chance=25.0, headline="acc_norm",
                          harness="arc_challenge", quick=500, out="mc/arc_challenge.jsonl", conv=conv_arc),
    "piqa": dict(title="PIQA", kind="mc", repo="ybisk/piqa", ref="refs/convert/parquet", paths=["plain_text/validation/0000.parquet"], split="validation", n=1838, chance=50.0, headline="acc",
                 harness="piqa", quick=500, out="mc/piqa.jsonl", conv=conv_piqa),
    "winogrande": dict(title="WinoGrande", kind="mc", repo="allenai/winogrande", paths=["winogrande_xl/validation-00000-of-00001.parquet"], split="validation", n=1267, chance=50.0, headline="acc",
                       harness="winogrande", quick=500, out="mc/winogrande.jsonl", conv=conv_winogrande),
    "openbookqa": dict(title="OpenBookQA", kind="mc", repo="allenai/openbookqa", paths=["main/test-00000-of-00001.parquet"], split="test", n=500, chance=25.0, headline="acc_norm",
                       harness="openbookqa", quick=500, out="mc/openbookqa.jsonl", conv=conv_obqa),
    "sciq": dict(title="SciQ", kind="mc", repo="allenai/sciq", paths=["data/test-00000-of-00001.parquet"], split="test", n=1000, chance=25.0, headline="acc",
                 harness="sciq", quick=500, out="mc/sciq.jsonl", conv=conv_sciq),
    "lambada_openai": dict(title="LAMBADA (OpenAI)", kind="lambada", repo="EleutherAI/lambada_openai", paths=["data/lambada_test.jsonl"], split="test", n=5153, chance=0.0, headline="acc",
                           harness="lambada_openai", quick=1000, out="lambada/lambada_openai.jsonl", conv=None),
    "mmlu_cloze": dict(title="MMLU (cloze, sanity)", kind="mc", repo="cais/mmlu", paths=["all/test-00000-of-00001.parquet"], split="test", n=14042, chance=25.0, headline="acc_norm",
                       harness="(lighteval/OLMES-style cloze; no exact harness task)", quick=500, full_limit=2000, max_len=512, out="mc/mmlu_cloze.jsonl", conv=conv_mmlu_cloze),
    "mmlu_letter": dict(title="MMLU (letters, sanity)", kind="mc", repo="cais/mmlu", paths=["all/test-00000-of-00001.parquet"], split="test", n=14042, chance=25.0, headline="acc",
                        harness="mmlu (zero-shot, letter choices)", quick=500, full_limit=2000, max_len=512, out="mc/mmlu_letter.jsonl", conv=conv_mmlu_letter),
}
ORDER = list(TASKS)
# what the pinned dataset cards say about the license (read on 2026-10-10); "not stated" = the card has no license information
LICENSE_SUMMARY = {
    "Rowan/hellaswag": "MIT (card: https://github.com/rowanz/hellaswag/blob/master/LICENSE)",
    "allenai/ai2_arc": "CC BY-SA 4.0 (card metadata)",
    "ybisk/piqa": "unknown (card says Unknown)",
    "allenai/winogrande": "not stated on the dataset card",
    "allenai/openbookqa": "unknown (card metadata: unknown)",
    "allenai/sciq": "CC BY-NC 3.0 (non-commercial: evaluation only, do not redistribute the data in a product)",
    "EleutherAI/lambada_openai": "MIT per card metadata; the card text says Modified MIT (https://github.com/openai/gpt-2/blob/master/LICENSE)",
    "cais/mmlu": "MIT (card metadata and card text)",
}
PPL_DOMAINS = ["web", "cosmo", "fmath", "owmath", "wiki", "books", "python"]


def read_rows(path):
    if path.endswith(".jsonl"):
        with open(path, encoding="utf-8") as f:
            return [json.loads(ln) for ln in f if ln.strip()]
    import pyarrow.parquet as pq

    return pq.read_table(path).to_pylist()


def lambada_convert(rows):
    out = []
    for r in rows:
        text = r["text"]
        ctx, last = text.rsplit(" ", 1)
        out.append({"ctx": ctx, "cont": " " + last})
    return out


# ---------------------------------------------------------------- token statistics with the Forge-1 tokenizer (the same encoding rule as the Rust evaluators)
def hs_preprocess(text):
    t = text.strip().replace(" [title]", ". ")
    out, depth = [], 0
    for c in t:
        if c == "[":
            depth += 1
        elif c == "]" and depth > 0:
            depth -= 1
        elif depth == 0:
            out.append(c)
    return "".join(out).replace("  ", " ")


def pct(sorted_vals, q):
    if not sorted_vals:
        return 0
    return sorted_vals[min(len(sorted_vals) - 1, int(math.ceil(q * len(sorted_vals))) - 1)]


def example_pairs(task, e):
    """(context, continuation) of every scored sequence of one example, as the Rust evaluators build them"""
    if task["kind"] == "hellaswag":
        ctx = hs_preprocess(f"{e['activity_label']}: {e['ctx_a']} {e['ctx_b'][:1].upper() + e['ctx_b'][1:].lower()}")
        return [(ctx, " " + hs_preprocess(end)) for end in e["endings"]]
    if task["kind"] == "lambada":
        return [(e["ctx"], e["cont"])]
    return [(c["ctx"], c["cont"]) for c in e["choices"]]


def seq_lengths(tok, task, examples):
    """token length of every scored sequence (context + continuation), per example, and the length of the continuation part"""
    pairs = [pr for e in examples for pr in example_pairs(task, e)]
    ctx_ids = tok.encode_batch([p[0] for p in pairs], add_special_tokens=False)
    whole = tok.encode_batch([p[0] + p[1] for p in pairs], add_special_tokens=False)
    lens, cont_lens = [], []
    for i, (c, w) in enumerate(zip(ctx_ids, whole)):
        if len(w.ids) > len(c.ids) and w.ids[: len(c.ids)] == c.ids:
            lens.append(len(w.ids))
            cont_lens.append(len(w.ids) - len(c.ids))
        else:  # the tokenization of context + continuation does not extend the one of the context: the Rust code concatenates the two encodings
            n = len(tok.encode(pairs[i][1], add_special_tokens=False).ids)
            lens.append(len(c.ids) + n)
            cont_lens.append(n)
    per_ex, k = [], 0
    for e in examples:
        m = len(example_pairs(task, e))
        per_ex.append((lens[k:k + m], cont_lens[k:k + m]))
        k += m
    return per_ex


def seq_stats(per_ex):
    lens = [v for l, _ in per_ex for v in l]
    cont_lens = [v for _, c in per_ex for v in c]
    s = sorted(lens)
    need = lambda q: next((v for v in SEQ_STEPS if v >= pct(s, q) - 1), SEQ_STEPS[-1])  # a row holds seq + 1 tokens
    seq_max = need(1.0)
    return {"sequences": len(lens), "mean": round(sum(lens) / len(lens), 1), "p99": pct(s, 0.99), "p999": pct(s, 0.999), "max": s[-1], "cont_mean": round(sum(cont_lens) / len(cont_lens), 2),
            "cont_max": max(cont_lens), "seq_p999": need(0.999), "seq_max": seq_max, "truncated_at_seq_max": sum(1 for v in lens if v > seq_max + 1)}


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="/mnt/data/forgeinfer-data/eval")
    ap.add_argument("--tokenizer", default=os.path.expanduser("~/forgeinfer/data/tokenizer_f1.json"))
    ap.add_argument("--val-domains", default=os.path.expanduser("~/forgeinfer/data/val_domains"))
    ap.add_argument("--tasks", default="")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--offline", action="store_true", help="reuse OUT/raw and the existing MANIFEST.json (no network)")
    a = ap.parse_args()
    out = os.path.abspath(os.path.expanduser(a.out))
    for d in ("raw", "mc", "hellaswag", "lambada", "ppl"):
        os.makedirs(os.path.join(out, d), exist_ok=True)
    want = [t for t in (a.tasks.split(",") if a.tasks else ORDER) if t]
    for t in want:
        if t not in TASKS:
            sys.exit(f"unknown task {t}; choose from {', '.join(ORDER)}")
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(a.tokenizer)
    tok_copy = os.path.join(out, "tokenizer_f1.json")
    if not os.path.exists(tok_copy) or sha256_file(tok_copy) != sha256_file(a.tokenizer):
        shutil.copyfile(a.tokenizer, tok_copy)  # the eval directory is self-contained
    man_path = os.path.join(out, "MANIFEST.json")
    manifest = {}
    if os.path.exists(man_path):
        with open(man_path, encoding="utf-8") as f:
            manifest = json.load(f)
    manifest.setdefault("tasks", {})
    manifest.update({"schema": 1, "generator": "tools/f1_eval_data.py", "seed": a.seed, "python": platform.python_version(), "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
                     "tokenizer": {"file": os.path.basename(a.tokenizer), "sha256": sha256_file(a.tokenizer), "vocab": tok.get_vocab_size()}})
    info_cache = {}
    for name in want:
        t = TASKS[name]
        print(f"== {name}: {t['title']} ({t['repo']}, {t['split']})", flush=True)
        key = (t["repo"], t.get("ref"))
        if a.offline:
            src = manifest["tasks"][name]["source"]
        else:
            if key not in info_cache:
                info = hub_info(t["repo"], t.get("ref"))
                info["readme_license_lines"] = readme_license_lines(t["repo"], info["commit"])
                info["sizes"] = hub_tree_sizes(t["repo"], info["commit"])
                info_cache[key] = info
            info = info_cache[key]
            src = {"repo": t["repo"], "ref": info["ref"], "commit": info["commit"], "license_metadata": info["license_metadata"], "readme_license_lines": info["readme_license_lines"], "files": []}
            for p in t["paths"]:
                dest = os.path.join(out, "raw", f"{t['repo'].replace('/', '__')}__{p.replace('/', '__')}")
                if not (os.path.exists(dest) and os.path.getsize(dest) == info["sizes"].get(p)):
                    hub_get(t["repo"], info["commit"], p, dest, info["sizes"].get(p))
                    print(f"   downloaded {p} ({os.path.getsize(dest)} bytes)", flush=True)
                src["files"].append({"path": p, "raw": os.path.relpath(dest, out).replace(os.sep, "/"), "bytes": os.path.getsize(dest), "sha256": sha256_file(dest)})
        raw = os.path.join(out, src["files"][0]["raw"])
        rows = read_rows(raw)
        assert len(rows) == t["n"], f"{name}: {len(rows)} rows in the split, expected {t['n']}"
        dropped = 0
        if t.get("max_len"):
            # MMLU: a handful of very long questions would force a huge padded row on every example. They are dropped, judged on the longer (letter) format so that the cloze and the letter
            # variants keep exactly the same questions in the same order (the number is recorded in the manifest)
            lens = seq_lengths(tok, {"kind": "mc"}, [conv_mmlu_letter(r) for r in rows])
            keep = [i for i, (l, _) in enumerate(lens) if max(l) <= t["max_len"] + 1]
            dropped = len(rows) - len(keep)
            rows = [rows[i] for i in keep]
        if t["kind"] == "lambada":
            examples = lambada_convert(rows)
        else:
            examples = [t["conv"](r) for r in rows]
        random.Random(a.seed).shuffle(examples)
        per_ex = seq_lengths(tok, t, examples)
        dest = os.path.join(out, t["out"])
        with open(dest, "w", encoding="utf-8", newline="\n") as f:
            for e in examples:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        st = seq_stats(per_ex)
        chance = t["chance"]
        if t["kind"] == "mc":
            chance = round(100.0 * sum(1.0 / len(e["choices"]) for e in examples) / len(examples), 2)
        n_choices = (len(examples[0]["endings"]) if t["kind"] == "hellaswag" else (len(examples[0]["choices"]) if t["kind"] == "mc" else 1))
        seq = 256 if t["kind"] == "hellaswag" else st["seq_max"]
        entry = {"title": t["title"], "kind": t["kind"], "file": t["out"], "examples": len(examples), "examples_in_split": t["n"], "dropped_long": dropped, "choices": n_choices, "chance": chance,
                 "headline": t["headline"], "harness_task": t["harness"], "split": t["split"], "shots": 0, "quick": min(t["quick"], len(examples)), "full_limit": min(t.get("full_limit", len(examples)), len(examples)),
                 "seq": seq, "seq_note": "smallest power of two that holds every sequence (hellaswag: fixed 256 in the Rust code)", "tokens": st, "bytes": os.path.getsize(dest), "sha256": sha256_file(dest), "source": src,
                 "license": LICENSE_SUMMARY.get(t["repo"], "")}
        manifest["tasks"][name] = entry
        print(f"   {len(examples)} examples ({dropped} dropped as too long) -> {t['out']}  tokens/sequence mean {st['mean']} p99.9 {st['p999']} max {st['max']}  seq {seq}", flush=True)
    # per-domain validation token files: copies, so that the eval directory is self-contained (and can be packed for a cloud machine)
    ppl = {}
    for d in PPL_DOMAINS:
        srcp = os.path.join(os.path.expanduser(a.val_domains), f"{d}.u16")
        dst = os.path.join(out, "ppl", f"{d}.u16")
        if os.path.exists(srcp) and (not os.path.exists(dst) or os.path.getsize(dst) != os.path.getsize(srcp)):
            shutil.copyfile(srcp, dst)
        if os.path.exists(dst):
            ppl[d] = {"file": f"ppl/{d}.u16", "bytes": os.path.getsize(dst), "tokens": os.path.getsize(dst) // 2, "sha256": sha256_file(dst)}
    manifest["ppl"] = {"files": ppl, "note": "held-out tail of the training mixture of run f1_v1 (zzz_val), per source; token ids of tokenizer_f1.json; eval-ppl scores `windows` evenly spaced windows of seq + 1 tokens"}
    manifest["order"] = ORDER
    with open(man_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(manifest, f, indent=1, ensure_ascii=False)
        f.write("\n")
    write_markdown(out, manifest)
    print("manifest written:", man_path)


def write_markdown(out, m):
    L = ["# Forge-1 evaluation ladder data", "", f"Generated by tools/f1_eval_data.py on {m['updated']} (seed {m['seed']}, Python {m['python']}); tokenizer {m['tokenizer']['file']} (sha256 {m['tokenizer']['sha256'][:16]}..., vocabulary {m['tokenizer']['vocab']}).", "",
         "Only evaluation splits were downloaded (validation or test, never a training split). Each task file is shuffled once with the seed, so the first N lines are a uniform random subsample.", "",
         "| task | file | examples | choices | chance % | split | source (pinned commit) | license (as the card states it) | sha256 (file) |", "|---|---|---|---|---|---|---|---|---|"]
    for name in m["order"]:
        if name not in m["tasks"]:
            continue
        t = m["tasks"][name]
        s = t["source"]
        n_txt = f"{t['examples']}" + (f" of {t['examples_in_split']} ({t['dropped_long']} over-long questions dropped)" if t.get("dropped_long") else "")
        L.append(f"| {name} | {t['file']} | {n_txt} | {t['choices']} | {t['chance']} | {t['split']} | {s['repo']}@{s['commit'][:10]} ({s['ref']}) | {t.get('license', '')} | {t['sha256'][:16]} |")
    L += ["", "## License notes (lines of the dataset cards that mention a license)", ""]
    seen = set()
    for name in m["order"]:
        if name not in m["tasks"]:
            continue
        s = m["tasks"][name]["source"]
        if s["repo"] in seen:
            continue
        seen.add(s["repo"])
        L.append(f"- **{s['repo']}** (metadata: {s.get('license_metadata')}): " + ("; ".join(x.replace("|", "/") for x in s.get("readme_license_lines", [])) or "no license line in the card"))
    L += ["", "## Raw files", "", "| raw file | bytes | sha256 |", "|---|---|---|"]
    seen_raw = set()
    for name in m["order"]:
        if name in m["tasks"]:
            for fl in m["tasks"][name]["source"]["files"]:
                if fl["raw"] not in seen_raw:
                    seen_raw.add(fl["raw"])
                    L.append(f"| {fl['raw']} | {fl['bytes']} | {fl['sha256']} |")
    L += ["", "## Token statistics (Forge-1 tokenizer, context + continuation per scored sequence)", "", "| task | sequences | mean | p99 | p99.9 | max | continuation mean | seq used |", "|---|---|---|---|---|---|---|---|"]
    for name in m["order"]:
        if name in m["tasks"]:
            t = m["tasks"][name]
            k = t["tokens"]
            L.append(f"| {name} | {k['sequences']} | {k['mean']} | {k['p99']} | {k['p999']} | {k['max']} | {k['cont_mean']} | {t['seq']} |")
    L += ["", "## Per-domain validation token files", "", "| domain | file | tokens | sha256 |", "|---|---|---|---|"]
    for d, v in m.get("ppl", {}).get("files", {}).items():
        L.append(f"| {d} | {v['file']} | {v['tokens']} | {v['sha256'][:16]} |")
    with open(os.path.join(out, "MANIFEST.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
