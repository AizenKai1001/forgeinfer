#!/usr/bin/env python3
"""human_sources.py: adapters of the human-written-only SFT sources (see ForgeInfer - SFT Data Mix (human-written only) in the research notes).

An adapter is a generator  adapter(ctx) -> dicts {"id", "msgs": [[role, content], ...], "meta": {"stratum": ...}}  over the CANDIDATES of one source, in a seeded random order; `ctx.rng` is the
random generator and `ctx.full(stratum)` says that the token quota of a stratum (a Stack Exchange site, a subreddit ...) is reached, so that the adapter can stop reading it.  The content is RAW here
(cleaning, the quality filters, decontamination and token counting happen in human_pools.py), except for the MECHANICAL transformations that give the text its chat form:
HTML to markdown (Stack Exchange), detokenisation (WritingPrompts), templating of the user side (summaries, reading comprehension, code tasks), the closing line 'The answer is: X' of the math sources.
The assistant side of every conversation is text that a human wrote; no adapter calls a model.

CLI:  human_sources.py peek NAME [N]   print N candidate conversations of a source
"""
import csv
import glob
import json
import os
import re
import sys

import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import human_cfg as C  # noqa: E402
import human_lib as H  # noqa: E402
import sft_lib as L  # noqa: E402
import sft_sources as S  # noqa: E402  (the adapters of the first build that are reused unchanged: oasst2, dolly, gsm8k)


def raw(*parts):
    return os.path.join(C.RAW, *parts)


def parquet_rows(pattern, columns=None, batch=2000):
    for p in sorted(glob.glob(raw(pattern))):
        pf = pq.ParquetFile(p)
        for b in pf.iter_batches(batch_size=batch, columns=columns):
            yield from b.to_pylist()


def pick(rng, seq):
    return seq[rng.randrange(len(seq))]


def code_block(code, lang="python"):
    code = code.replace("\r\n", "\n").replace("\r", "\n").replace("\t", "    ").rstrip()
    return f"```{lang}\n{code}\n```"


# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# general chat (humans wrote prompt and answer)

def oasst2(ctx):
    """OpenAssistant oasst2 (Apache-2.0): best-ranked path of every message tree (English, reviewed, not deleted, not synthetic, toxicity < 0.3): the adapter of the first build."""
    yield from S.oasst2(ctx.rng, 1.0)


def dolly(ctx):
    """databricks-dolly-15k (CC-BY-SA-3.0): instruction (+ context) -> response written by Databricks employees."""
    yield from S.dolly(ctx.rng, 1.0)


def no_robots(ctx):
    """HuggingFaceH4/no_robots (CC-BY-NC-4.0): 10,000 instructions and demonstrations written by skilled human annotators; some have a system prompt, some are multi-turn."""
    rows = list(parquet_rows("HuggingFaceH4__no_robots/data/*.parquet"))
    ctx.rng.shuffle(rows)
    for r in rows:
        msgs = [[m["role"], m["content"]] for m in r["messages"] if m["role"] in ("system", "user", "assistant")]
        yield {"id": "no_robots:" + r["prompt_id"], "msgs": msgs, "meta": {"stratum": r.get("category", "")}}


def aya(ctx):
    """CohereLabs/aya_dataset (Apache-2.0), English rows of the ORIGINAL annotations only (prompts and completions written from scratch by annotators; the 're-annotations' are human edits of machine output and are not used)."""
    rows = []
    for r in parquet_rows("CohereLabs__aya_dataset/data/*.parquet", columns=["inputs", "targets", "language", "annotation_type", "user_id"]):
        if r["language"] != "English" or r["annotation_type"] != "original-annotations":
            continue
        rows.append(r)
    ctx.rng.shuffle(rows)
    for i, r in enumerate(rows):
        yield {"id": f"aya:{i}", "msgs": [["user", r["inputs"]], ["assistant", r["targets"]]], "meta": {}}


# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Stack Exchange (answers written before the release of ChatGPT)

def _se_pick_answer(answers):
    """The best answer of a question: highest pm_score (log2 of the upvotes + 1 when accepted), accepted first; at least 3 (or accepted with at least 2)."""
    best = None
    for a in answers or []:
        key = (a["pm_score"], bool(a["selected"]), len(a["text"] or ""))
        if best is None or key > best[0]:
            best = (key, a)
    if best is None:
        return None
    a = best[1]
    if a["pm_score"] >= 3 or (a["selected"] and a["pm_score"] >= 2):
        return a
    return None


def stackexchange(ctx):
    """HuggingFaceH4/stack-exchange-preferences (CC-BY-SA-4.0): one question and its best answer per thread, questions dated before SE_DATE_MAX, per site until its token quota is reached."""
    repo = C.SE_REPO.replace("/", "__")
    sites = list(C.SE_SITES)
    ctx.rng.shuffle(sites)
    for site in sites:
        files = sorted(glob.glob(raw(repo, "data", site, "*.parquet")))
        ctx.rng.shuffle(files)
        for f in files:
            if ctx.full(site):
                break
            pf = pq.ParquetFile(f)
            for rg in range(pf.num_row_groups):
                if ctx.full(site):
                    break
                rows = pf.read_row_group(rg, columns=["qid", "question", "answers", "date"]).to_pylist()
                cands = []
                for r in rows:
                    if not r["date"] or r["date"] >= C.SE_DATE_MAX:
                        continue
                    q = r["question"] or ""
                    if not (60 <= len(q) <= 3500):
                        continue
                    a = _se_pick_answer(r["answers"])
                    if a is None or not (80 <= len(a["text"] or "") <= 4500):
                        continue
                    cands.append((r, a))
                ctx.rng.shuffle(cands)
                for r, a in cands:
                    if ctx.full(site):
                        break
                    qm, bad, ql = H.html_to_md(r["question"])
                    if bad or not qm or ql >= 3:
                        continue
                    am, bad, al = H.html_to_md(a["text"])
                    if bad or not am or al >= 2 or (al == 1 and len(am.split()) < 40):  # a link whose URL is gone leaves a dangling 'see here'
                        continue
                    yield {"id": f"se:{site}:{r['qid']}:{a['answer_id']}", "msgs": [["user", qm], ["assistant", am]], "meta": {"stratum": site, "score": a["pm_score"]}}


# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Reddit answers (SHP: naturally occurring, human-written; comments written before 2022-09-01)

_HTML_ENT = re.compile(r"&(?:amp|lt|gt|quot|apos|#\d+|#x[0-9a-fA-F]+);")


def html_unescape(s):
    import html

    return html.unescape(s) if "&" in s else s


_PARA = re.compile(r"(?<=[.!?:\"')\]])[ \t]{2,}(?=\S)")


def _reddit_text(s):
    """SHP replaced the line breaks of the comments by two spaces: after a sentence end they become a paragraph break again (mechanical formatting)."""
    s = s or ""
    for _ in range(2):
        s = html_unescape(s)
    return _PARA.sub("\n\n", s.replace("​", "")).strip()


def shp(ctx):
    """stanfordnlp/SHP: per post the highest-scored top-level comment (score at least 6, upvote ratio of the post at least 0.75, written before SHP_TS_MAX) as the answer to the post (title + body); per subreddit until its token quota is reached."""
    subs = list(C.SHP_SUBS)
    ctx.rng.shuffle(subs)
    for sub in subs:
        posts = {}
        for split in ("train", "validation", "test"):
            p = raw(C.SHP_REPO.replace("/", "__"), sub, split + ".json")
            if not os.path.exists(p):
                continue
            with open(p, encoding="utf-8") as f:
                for line in f:
                    r = json.loads(line)
                    d = posts.setdefault(r["post_id"], {"history": r["history"], "ratio": r["upvote_ratio"], "c": {}})
                    d["c"][r["c_root_id_A"]] = (r["score_A"], r["created_at_utc_A"], r["human_ref_A"])
                    d["c"][r["c_root_id_B"]] = (r["score_B"], r["created_at_utc_B"], r["human_ref_B"])
        cands = []
        for pid, d in posts.items():
            if d["ratio"] < 0.75:
                continue
            cid, (score, ts, text) = max(d["c"].items(), key=lambda kv: kv[1][0])
            if score < 6 or ts >= C.SHP_TS_MAX:
                continue
            cands.append((pid, cid, d["history"], text, score))
        del posts
        cands.sort()
        ctx.rng.shuffle(cands)
        for pid, cid, hist, text, score in cands:
            if ctx.full(sub):
                break
            yield {"id": f"shp:{sub}:{pid}:{cid}", "msgs": [["user", _reddit_text(hist)], ["assistant", _reddit_text(text)]], "meta": {"stratum": sub, "score": score}}


# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# math with human solutions

def gsm8k(ctx):
    """GSM8K TRAIN (MIT): the adapter of the first build (calculator tags removed, closing lines '#### n' and 'The answer is: n')."""
    yield from S.gsm8k(ctx.rng, 1.0)


def hendrycks_math(ctx):
    """MATH TRAIN (MIT), all levels: human solutions; the closing line 'The answer is: ...' is added (the convention of the first mix); problems and solutions with [asy] figures are skipped."""
    for p in sorted(glob.glob(os.path.join(L.RAW, "EleutherAI__hendrycks_math", "*", "train-*.parquet"))):
        rows = pq.ParquetFile(p).read().to_pylist()
        ctx.rng.shuffle(rows)
        for i, r in enumerate(rows):
            if "[asy]" in r["problem"] or "[asy]" in r["solution"]:
                continue
            ans = S._boxed(r["solution"])
            if not ans:
                continue
            yield {"id": f"math:{os.path.basename(os.path.dirname(p))}:{i}", "msgs": [["user", r["problem"].strip()], ["assistant", r["solution"].strip() + f"\nThe answer is: {ans}"]], "meta": {"stratum": r["level"]}}


_AQ_TAIL = [
    re.compile(r"(?is)(?:\n|^)\s*(?:so\s*,?\s*)?(?:the\s+)?(?:correct\s+)?(?:answer|ans|option|choice)\s*(?:is|:|=|-)?\s*[\(\[]?\s*[A-E]\s*[\)\]]?\s*\.?\s*$"),
    re.compile(r"(?s)(?:\n|^)\s*[\(\[]?[A-E][\)\]]?\s*\.?\s*$"),
]


def aqua_rat(ctx):
    """AQuA-RAT train (Apache-2.0): algebra word problems with options and a natural-language rationale written by crowd workers; the closing answer line of the rationale is replaced by 'The answer is: X'."""
    rows = list(parquet_rows("deepmind__aqua_rat/raw/train-*.parquet"))
    ctx.rng.shuffle(rows)
    for i, r in enumerate(rows):
        rat = (r["rationale"] or "").strip()
        if not rat:
            continue
        for _ in range(2):
            for rx in _AQ_TAIL:
                rat = rx.sub("", rat).strip()
        opts = []
        for o in r["options"]:
            m = re.match(r"\s*([A-E])\s*[\)\.:]\s*(.*)", o)
            if not m:
                opts = None
                break
            opts.append(f"{m.group(1)}) {m.group(2).strip()}")
        if not opts or r["correct"] not in "ABCDE" or len(rat.split()) < 12:
            continue
        yield {"id": f"aqua:{i}", "msgs": [["user", r["question"].strip() + "\n\nOptions:\n" + "\n".join(opts)], ["assistant", rat + f"\nThe answer is: {r['correct']}"]], "meta": {}}


# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# code with human solutions

def _pick_solution(sols, rng):
    ok = []
    for s in sols:
        s = s.replace("\r\n", "\n").strip("\n")
        if not (80 <= len(s) <= 1800) or s.count("\n") < 3 or not s.isascii():
            continue
        if re.search(r"\bexec\(|\beval\(|__import__|\blambda\b.*\blambda\b", s):
            continue
        if max(len(ln) for ln in s.split("\n")) > 110:
            continue
        ok.append(s)
    return pick(rng, ok) if ok else None


def apps(ctx):
    """APPS train (MIT): programming problems from competitive-programming sites with human Python solutions (introductory and interview level); one valid solution per problem."""
    rows = []
    with open(raw("codeparrot__apps", "train.jsonl"), encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["difficulty"] not in ("introductory", "interview") or not r["solutions"]:
                continue
            rows.append(r)
    ctx.rng.shuffle(rows)
    for r in rows:
        try:
            sols = json.loads(r["solutions"])
        except Exception:  # noqa: BLE001
            continue
        sol = _pick_solution(sols, ctx.rng)
        if sol is None:
            continue
        q = r["question"].strip()
        start = (r.get("starter_code") or "").strip()
        if start:
            prompt = f"Solve the following programming problem in Python. Complete the starter code.\n\n{q}\n\nStarter code:\n```python\n{start}\n```"
        else:
            prompt = f"Solve the following programming problem in Python. Read the input from standard input and print the answer.\n\n{q}"
        yield {"id": f"apps:{r['id']}", "msgs": [["user", prompt], ["assistant", code_block(sol)]], "meta": {"stratum": r["difficulty"]}}


def mbpp_train(ctx):
    """MBPP train + validation problems (CC-BY-4.0; task_id 511-974: the test problems 11-510 and the few-shot prompts 1-10 are NOT used): crowd-written task, human solution, first test case in the prompt."""
    rows = list(parquet_rows("google-research-datasets__mbpp/full/train-*.parquet")) + list(parquet_rows("google-research-datasets__mbpp/full/validation-*.parquet"))
    ctx.rng.shuffle(rows)
    for r in rows:
        if r["task_id"] < 511:
            continue
        yield {"id": f"mbpp:{r['task_id']}", "msgs": [["user", f"{r['text'].strip()}\nYour code should pass this test:\n{r['test_list'][0].strip()}"], ["assistant", code_block(r["code"])]], "meta": {}}


def rosetta(ctx):
    """Rosetta Code (GFDL): task description + a human Python solution."""
    rows = {}
    for r in parquet_rows("christopher__rosetta-code/data/*.parquet", columns=["task_name", "task_description", "language_name", "code"]):
        if r["language_name"] != "Python":
            continue
        d = re.sub(r"\n{3,}", "\n\n", H.ascii_typography((r["task_description"] or "").strip()))
        if "displaystyle" in d or not d.isascii() or len(d) < 60 or len(d) > 1400:
            continue
        code = H.ascii_typography((r["code"] or "").strip("\n"))
        if not (80 <= len(code) <= 1800) or code.count("\n") < 3 or not code.isascii() or max(len(ln) for ln in code.split("\n")) > 110:
            continue
        rows.setdefault(r["task_name"], (d, []))[1].append(code)
    names = sorted(rows)
    ctx.rng.shuffle(names)
    for n in names:
        d, codes = rows[n]
        code = pick(ctx.rng, codes)
        yield {"id": "rosetta:" + n, "msgs": [["user", f"Write a Python program for the Rosetta Code task \"{n}\".\n\n{d}"], ["assistant", code_block(code)]], "meta": {}}


_SIG = re.compile(r"^def\s+(\w+)\s*\(([^)]*)\)\s*(?:->\s*[^:]+)?:", re.S)
_CSN_T = [
    "Write a Python function with this signature: `{sig}`\n\n{doc}",
    "Implement the following Python function.\n\n```python\n{sig}\n```\n\n{doc}",
    "{doc}\n\nWrite this as a Python function named `{name}`.",
    "Write a Python function called `{name}`. {doc}",
]


def code_search_net(ctx):
    """CodeSearchNet Python (functions of open-source GitHub projects with a docstring): docstring -> function.  User side templated, assistant side = the function as written by its author."""
    rows = []
    for p in sorted(glob.glob(raw("code-search-net__code_search_net", "python", "train-*.parquet"))):
        pf = pq.ParquetFile(p)
        for b in pf.iter_batches(batch_size=4000, columns=["func_name", "func_code_string", "func_documentation_string"]):
            for r in b.to_pylist():
                code = r["func_code_string"] or ""
                doc = (r["func_documentation_string"] or "").strip()
                if not (4 <= code.count("\n") + 1 <= 40) or len(code) > 1700 or not code.isascii() or not doc.isascii():
                    continue
                if not (8 <= len(doc.split()) <= 90) or doc.count("\n") > 6:
                    continue
                m = _SIG.match(code)
                if not m or m.group(2).split(",")[0].strip() in ("self", "cls") or m.group(1).startswith("__"):
                    continue
                if re.search(r"TODO|FIXME|XXX|HACK|:param|:type|:return|:rtype|@param|\.\. |>>>|\bhttp|py2|python2|Python 2|\bprint [\"']|Deprecated", code + doc):
                    continue
                rows.append((r["func_name"], code, doc, m.group(0).strip(), m.group(1)))
        if len(rows) > 400000:
            break
    ctx.rng.shuffle(rows)
    for i, (fn, code, doc, sig, name) in enumerate(rows):
        doc = re.sub(r"\s*\n\s*", " ", doc).strip()
        user = pick(ctx.rng, _CSN_T).format(sig=sig, doc=doc, name=name)
        yield {"id": f"csn:{i}:{fn}", "msgs": [["user", user], ["assistant", code_block(code)]], "meta": {}}


# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# reading comprehension and short-answer QA (answers written / marked by crowd workers)

_SQ_FIRST = [
    "Read the passage and answer the question.\n\nPassage: {c}\n\nQuestion: {q}",
    "{c}\n\n{q}",
    "Context: {c}\n\nQuestion: {q}\nAnswer briefly.",
    "Use the text below to answer my questions.\n\n{c}\n\nQuestion: {q}",
]


def squad(ctx):
    """SQuAD v1.1 train (CC-BY-SA-4.0): questions written by crowd workers on Wikipedia paragraphs, the answer is a span of the paragraph; up to 4 questions of a paragraph form one multi-turn conversation."""
    by_ctx = {}
    for r in parquet_rows("rajpurkar__squad/plain_text/train-*.parquet"):
        if not r["answers"]["text"]:
            continue
        by_ctx.setdefault(r["context"], []).append((r["question"], r["answers"]["text"][0]))
    keys = sorted(by_ctx)
    ctx.rng.shuffle(keys)
    for i, c in enumerate(keys):
        qa = by_ctx[c]
        ctx.rng.shuffle(qa)
        qa = qa[:4]
        tpl = pick(ctx.rng, _SQ_FIRST)
        plain = tpl.startswith("{c}")
        msgs = []
        for k, (q, a) in enumerate(qa):
            if k == 0:
                msgs.append(["user", tpl.format(c=c.strip(), q=q.strip())])
            else:
                msgs.append(["user", q.strip() if plain else f"Question: {q.strip()}"])
            msgs.append(["assistant", a.strip()])
        yield {"id": f"squad:{i}", "msgs": msgs, "meta": {}}


def coqa(ctx):
    """CoQA train (licenses per passage domain: CC-BY-SA-4.0 Wikipedia and literature, Apache CNN; the MCTest and RACE passages are NOT used): conversational questions about a passage, answers written by crowd workers."""
    rows = [r for r in parquet_rows("stanfordnlp__coqa/data/train-*.parquet") if r["source"] in ("wikipedia", "gutenberg", "cnn")]
    ctx.rng.shuffle(rows)
    for i, r in enumerate(rows):
        qs, ans = r["questions"], r["answers"]["input_text"]
        if not qs or len(qs) != len(ans):
            continue
        msgs = []
        for k, (q, a) in enumerate(zip(qs, ans)):
            if k == 0:
                msgs.append(["user", f"Read the story and answer my questions.\n\nStory: {r['story'].strip()}\n\nQuestion: {q.strip()}"])
            else:
                msgs.append(["user", f"Question: {q.strip()}"])
            msgs.append(["assistant", a.strip()])
        yield {"id": f"coqa:{i}", "msgs": msgs, "meta": {"stratum": r["source"]}}


_NQ_T = ["{q}", "{q}", "{q}", "Answer in a few words: {q}", "{q}\nGive a short answer.", "Q: {q}\nA:"]


def nq_open(ctx):
    """NQ-Open train (CC-BY-SA-3.0): real search-engine questions with the short answer that annotators marked in Wikipedia."""
    rows = list(parquet_rows("google-research-datasets__nq_open/nq_open/train-*.parquet"))
    ctx.rng.shuffle(rows)
    for i, r in enumerate(rows):
        q = r["question"].strip()
        if not r["answer"]:
            continue
        q = q[0].upper() + q[1:] + ("" if q.endswith("?") else "?")
        yield {"id": f"nq:{i}", "msgs": [["user", pick(ctx.rng, _NQ_T).format(q=q)], ["assistant", r["answer"][0].strip()]], "meta": {}}


def trivia_qa(ctx):
    """TriviaQA rc.nocontext train (license unknown on the card): questions written by trivia enthusiasts, answer = the trivia answer (a Wikipedia entity or value)."""
    rows = list(parquet_rows("mandarjoshi__trivia_qa/rc.nocontext/train-*.parquet", columns=["question", "question_id", "answer"]))
    ctx.rng.shuffle(rows)
    for r in rows:
        a = (r["answer"] or {}).get("value") or ""
        if not a:
            continue
        yield {"id": "trivia:" + r["question_id"], "msgs": [["user", pick(ctx.rng, _NQ_T).format(q=r["question"].strip())], ["assistant", a.strip()]], "meta": {}}


# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# writing: summaries and stories written by humans

_CNN_BAD = re.compile(r"E-mail to a friend|Click here|READ MORE|Read more|Related:|Photo:|Story Highlights|STORY HIGHLIGHTS|Watch|WATCH|CNN\.com|ONLY ON|Follow|iReport|Sign up|\bSee also\b|By \. |PUBLISHED:")
_CNN_T = [
    "Summarize the following news article in two or three sentences.\n\n{a}",
    "{a}\n\nSummarize this article.",
    "Write a short summary of this article.\n\n{a}",
    "Give me the key points of this news story.\n\n{a}",
    "TL;DR of the following article:\n\n{a}",
]


def cnn_dailymail(ctx):
    """CNN / DailyMail 3.0.0 train (Apache-2.0): news articles written by CNN journalists (only articles with the '(CNN)' dateline are used) and their highlights as written by the article author."""
    rows = [r for r in parquet_rows("abisee__cnn_dailymail/3.0.0/train-*.parquet", columns=["article", "highlights", "id"]) if "(CNN)" in r["article"][:140]]
    ctx.rng.shuffle(rows)
    for r in rows:
        art = r["article"].strip()
        if _CNN_BAD.search(art) or len(art.split()) > 460 or len(art.split()) < 120:
            continue
        hl = [re.sub(r"\s+([.,;:!?])", r"\1", h.strip()) for h in r["highlights"].split("\n") if h.strip()]
        if not hl:
            continue
        summary = "\n".join("- " + h for h in hl) if ctx.rng.random() < 0.5 else " ".join(hl)
        yield {"id": "cnn:" + r["id"], "msgs": [["user", pick(ctx.rng, _CNN_T).format(a=art)], ["assistant", summary]], "meta": {}}


_DS_T = ["Summarize this conversation.\n\n{d}", "{d}\n\nWhat is this conversation about? Summarize it briefly.", "Write a short summary of the following dialogue.\n\n{d}"]


def dialogsum(ctx):
    """DialogSum train (CC-BY-NC-SA-4.0): dialogues with summaries written by human annotators."""
    with open(raw("knkarthick__dialogsum", "train.csv"), encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    ctx.rng.shuffle(rows)
    for r in rows:
        yield {"id": "dialogsum:" + r["id"], "msgs": [["user", pick(ctx.rng, _DS_T).format(d=r["dialogue"].strip())], ["assistant", r["summary"].strip()]], "meta": {}}


_WP_TAG = re.compile(r"^\s*\[\s*[A-Za-z]{2}\s*\]\s*")
_WP_BAD = re.compile(r"(?i)\b(?:edit|thanks for reading|part (?:one|two|three|\d)|/r/|reddit|subreddit|upvote|my first (?:post|story|prompt|attempt)|feedback|critique|constructive|sorry for|not sure if this|wrote this on|on mobile|formatting|typos?)\b|\bTLDR\b|\[\s*[A-Z]{2}\s*\]")
_WP_T = ["Write a short story based on this prompt:\n\n{p}", "{p}", "Write a story: {p}", "Prompt: {p}\n\nWrite the story.", "Here is a writing prompt. Write a short story for it.\n\n{p}"]


def writing_prompts(ctx):
    """WritingPrompts train (MIT; Fan et al. 2018, stories written by Reddit users): prompt -> story.  Detokenised mechanically (the archive tokenised the text); stories with leftover tokenisation, forum chatter or edits are dropped."""
    rows = []
    for r in parquet_rows("euclaise__writingprompts/data/train-*.parquet", batch=4000):
        st = r["story"] or ""
        if not (700 <= len(st) <= 3600):
            continue
        rows.append(r)
        if len(rows) >= 60000:
            break
    ctx.rng.shuffle(rows)
    for i, r in enumerate(rows):
        p = H.detokenize(_WP_TAG.sub("", r["prompt"]))
        st = H.detokenize(r["story"])
        if H.tokenization_artifacts(st) > 1 or H.tokenization_artifacts(p) > 1 or _WP_BAD.search(st) or _WP_BAD.search(p) or len(p.split()) < 8 or "<" in st:
            continue
        yield {"id": f"wp:{i}", "msgs": [["user", pick(ctx.rng, _WP_T).format(p=p)], ["assistant", st]], "meta": {}}


# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

ADAPTERS = {
    "oasst2": oasst2, "dolly": dolly, "no_robots": no_robots, "aya": aya, "se": stackexchange, "shp": shp, "gsm8k": gsm8k, "hendrycks": hendrycks_math, "aqua": aqua_rat, "apps": apps, "mbpp": mbpp_train,
    "rosetta": rosetta, "csn": code_search_net, "squad": squad, "coqa": coqa, "nq": nq_open, "trivia": trivia_qa, "cnn": cnn_dailymail, "dialogsum": dialogsum, "wp": writing_prompts,
}


class _Ctx:
    def __init__(self, seed_name):
        self.rng = L.rng_for("human-pool", seed_name)

    def full(self, stratum):
        return False


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "peek":
        name, n = sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 2
        k = 0
        for c in ADAPTERS[name](_Ctx(name)):
            print(json.dumps(c, ensure_ascii=False)[:2400])
            print()
            k += 1
            if k >= n:
                break
    else:
        print(__doc__)
