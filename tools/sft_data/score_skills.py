#!/usr/bin/env python3
"""score_skills.py PROBE.json OUT1.txt [OUT2.txt ...] [--verbose]: scores the output of cloud/modal_forge.py::probe on scratch/skills_probe.json (probe C: held-out prompts of the families of tools/sft_data/sft_skills.py).

Each item of the JSON has a "check" and an "expect": last_number (the last number of the answer), contains, regex, equals (the whole answer, punctuation and case ignored), one_word, bullets_n, list_n, one_sentence (one sentence that
contains the word), code (a function with def and return that parses). One point per item, printed by family and in total for every output file; with several files a comparison table follows. Standard library only.
"""
import ast
import json
import re
import sys

SECTION = re.compile(r"^=== (\d+) \[(\w+)\] (.*)$", re.M)
NOISE = re.compile(r"^\s*\[[a-z_ -]+\]")


def answers(text):
    out = {}
    marks = list(SECTION.finditer(text))
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        body = text[m.end():end]
        body = body.split("->", 1)[1] if "->" in body else body
        lines = [ln for ln in body.splitlines() if not NOISE.match(ln)]
        out[int(m.group(1))] = "\n".join(lines).strip("\n")
    return out


def last_turn(raw):
    """The interactive output is all the answers (stdout) followed by the prompts '> > >' (stderr): drop the prompts, the last non-empty line is the answer to the last user turn."""
    t = re.sub(r"(?:>\s*)+$", "", raw.strip()).strip()
    lines = [ln for ln in t.splitlines() if ln.strip()]
    return lines[-1].strip() if lines else ""


def norm(s):
    return re.sub(r"[^a-z0-9 ]+", "", s.lower()).strip()


def code_ok(ans):
    m = re.search(r"```(?:python)?\n(.*?)```", ans, re.S)
    code = m.group(1) if m else ans
    if "def " not in code or "return" not in code:
        return False
    try:
        ast.parse(code)
    except SyntaxError:
        return False
    return True


def judge(item, ans):
    c, e = item["check"], item["expect"]
    if item.get("turns"):
        ans = last_turn(ans)
    if c == "last_number":
        nums = re.findall(r"-?\d[\d,]*\.?\d*", ans.replace("$", ""))
        return bool(nums) and nums[-1].replace(",", "").rstrip(".") == e
    if c == "recall":   # the last answer must name the fact and address the user ("Your name is Hana"), not parrot "My name is Hana"
        return e.lower() in ans.lower() and bool(re.search(r"\byour\b|\byou\b", ans, re.I))
    if c == "contains":
        return e.lower() in ans.lower()
    if c == "regex":
        return bool(re.search(e, ans, re.I))
    if c == "equals":
        return norm(ans) == norm(e)
    if c == "one_word":
        w = re.findall(r"[A-Za-z']+", ans)
        return len(w) == 1
    if c == "bullets_n":
        n = len([ln for ln in ans.splitlines() if re.match(r"\s*([-*•]|\d+[.)])\s+\S", ln)])
        return n == int(e)
    if c == "list_n":
        lines = [ln for ln in ans.splitlines() if re.match(r"\s*([-*•]|\d+[.)])\s+\S", ln)]
        n = len(lines) if lines else len([p for p in re.split(r",|\n", ans) if p.strip()])
        return n == int(e)
    if c == "one_sentence":
        s = [x for x in re.split(r"(?<=[.!?])\s+", ans.strip()) if x.strip()]
        return len(s) == 1 and e.lower() in ans.lower()
    if c == "code":
        return code_ok(ans)
    raise ValueError(c)


def score(items, text, verbose=False):
    ans = answers(text)
    fam, rows = {}, []
    for i, it in enumerate(items, 1):
        ok = judge(it, ans.get(i, ""))
        f = fam.setdefault(it["kind"], [0, 0])
        f[0] += int(ok)
        f[1] += 1
        rows.append(ok)
        if verbose or not ok:
            print(f"   {'ok  ' if ok else 'FAIL'} {i:2d} [{it['kind']}] {(it.get('prompt') or '')[:48]} -> {ans.get(i, '')[:100]!r}")
    return fam, rows


def main(argv):
    verbose = "--verbose" in argv
    argv = [a for a in argv if a != "--verbose"]
    items = json.load(open(argv[0], encoding="utf-8"))
    table = []
    for path in argv[1:]:
        print(path)
        fam, rows = score(items, open(path, encoding="utf-8").read(), verbose)
        tot = sum(rows)
        print(f"  total {tot}/{len(rows)}   " + "; ".join(f"{k} {v[0]}/{v[1]}" for k, v in fam.items()))
        table.append((path, rows))
    if len(table) > 1:
        print("item  " + "  ".join(f"{i + 1:>2}" for i in range(len(table))))
        for k, it in enumerate(items):
            print(f"{k + 1:4d}  " + "  ".join(f"{int(rows[k]):>2}" for _, rows in table) + f"   [{it['kind']}] {(it.get('prompt') or '')[:50]}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
