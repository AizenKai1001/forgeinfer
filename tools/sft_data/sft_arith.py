#!/usr/bin/env python3
"""sft_arith.py [--out DIR] [--n N] [--name NAME]: a LARGE arithmetic set with ONE worked format, for the Forge-1 chat patch (computed by this program, re-checked at the end, no model output).

Why: patch 1 and 2 taught the FORMAT of a word-problem answer ("Step 1: ... The answer is: N") but the numbers stayed wrong (5 + 7 = 13, 63 / 9 = 11, 345 + 278 = 503). Two things were wrong in the data: direct answers and
worked answers competed for the same prompts (the model took the short one), and 6,000 table facts seen ~18 times each are too few to be learned. Here every calculation that is not a basic fact is WORKED in place-value steps,
the basic facts (single digits, times tables to 12 x 12) are plain, and there are about 100,000 examples (each seen about once: learning the procedure, not memorizing the examples):
  facts (30 %)        a + b, a - b, a x b, a / b with the basic ranges, several phrasings
  worked (35 %)       multi-digit sums, differences and products written in place-value steps ("345 + 278: 300 + 200 = 500, 40 + 70 = 110, 5 + 8 = 13, 500 + 110 + 13 = 623")
  word problems (35 %) the generators of sft_skills.py with every non-basic calculation of the solution expanded in the same worked format
Output: DIR/NAME.jsonl (record format of sft_pools.py).
"""
import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_argv = sys.argv
sys.argv = [sys.argv[0]]
import sft_skills as S  # noqa: E402  (its argparse reads no arguments here; the module is only used for its problem generators)

sys.argv = _argv
import sft_lib as L  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=os.path.join(L.WORK, "pool"))
ap.add_argument("--n", type=int, default=100000)
ap.add_argument("--name", default="arith")
args = ap.parse_args()
rng = L.rng_for("arith")
pick, ri = S.pick, S.ri
S.rng = rng   # the problem generators of sft_skills use this module's generator: one seeded stream for everything


def parts(x):
    """345 -> [300, 40, 5] (the non-zero place values, largest first)."""
    s = str(x)
    return [int(d) * 10 ** (len(s) - 1 - i) for i, d in enumerate(s) if d != "0"]


def worked_add(a, b):
    """'45 + 27: 40 + 20 = 60, 5 + 7 = 12, 60 + 12 = 72' (the caller appends ' = c' when the expression is part of a step)."""
    places = max(len(str(a)), len(str(b)))
    lines, terms = [], []
    for i in range(places - 1, -1, -1):
        pa = (a // 10 ** i) % 10 * 10 ** i
        pb = (b // 10 ** i) % 10 * 10 ** i
        if pa and pb:
            lines.append(f"{pa} + {pb} = {pa + pb}")
            terms.append(pa + pb)
        elif pa or pb:
            terms.append(pa or pb)
    if len(terms) == 1:
        return f"{a} + {b} = {a + b}"
    if not lines:
        return f"{a} + {b} = {a + b}"
    return f"{a} + {b}: " + ", ".join(lines) + ", " + " + ".join(str(t) for t in terms) + f" = {a + b}"


def worked_sub(a, b):
    """Subtract the parts of b one after the other; a single-digit part that needs a borrow goes through the next lower ten first (25 - 7: 25 - 5 = 20, 20 - 2 = 18)."""
    cur, lines = a, []
    for y in parts(b):
        if y < 10 and cur % 10 < y:
            low = cur % 10
            if low:
                lines.append(f"{cur} - {low} = {cur - low}")
                cur -= low
                y -= low
            lines.append(f"{cur} - {y} = {cur - y}")
            cur -= y
        else:
            lines.append(f"{cur} - {y} = {cur - y}")
            cur -= y
    return f"{a} - {b}: " + ", ".join(lines)


def worked_mul(a, b):
    if len(parts(b)) >= 2:
        prods = [(a, p) for p in parts(b)]
    elif len(parts(a)) >= 2:
        prods = [(p, b) for p in parts(a)]
    else:
        return f"{a} x {b} = {a * b}"
    lines = [f"{x} x {y} = {x * y}" for x, y in prods]
    if len(prods) == 1:
        return f"{a} x {b} = {a * b}"
    return f"{a} x {b}: " + ", ".join(lines) + ", " + " + ".join(str(x * y) for x, y in prods) + f" = {a * b}"


RE_EXPR = re.compile(r"(?<![\w.$])(?<![-+x/] )(\d+) ([-+x/]) (\d+) = (\d+)(?=[.\s]|$)")   # the second lookbehind keeps the tail of a chain ('15 + 8 + 12 = 35') out of it


def expand(text):
    """Every 'a op b = c' of a solution that is not a basic fact becomes its worked version."""
    def rep(m):
        a, op, b, c = int(m.group(1)), m.group(2), int(m.group(3)), int(m.group(4))
        if op == "+" and (a >= 10 or b >= 10):
            return worked_add(a, b)
        if op == "-" and (b >= 10 or (a >= 20 and b > a % 10)):
            return worked_sub(a, b)
        if op == "x" and (a > 12 or b > 12):
            return worked_mul(a, b)
        return m.group(0)
    return RE_EXPR.sub(rep, text)


def make_facts_one():
    t = rng.randrange(4)
    if t == 0:
        a, b = ri(0, 20), ri(0, 20)
        op, c = "+", a + b
    elif t == 1:
        a, b = ri(0, 20), ri(0, 20)
        a, b = max(a, b), min(a, b)
        op, c = "-", a - b
    elif t == 2:
        a, b = ri(2, 12), ri(2, 12)
        op, c = "x", a * b
    else:
        b, c = ri(2, 12), ri(2, 12)
        a, op = b * c, "/"
    w = S.OPWORD[op]
    q = pick([f"What is {a} {op} {b}?", f"{a} {op} {b} = ?", f"Calculate {a} {op} {b}.", f"What is {a} {w} {b}?", f"What's {a} {w} {b}?", f"Compute {a} {op} {b}.", f"{a} {op} {b}"])
    return "arith_fact", [("user", q), ("assistant", pick([f"{a} {op} {b} = {c}.", f"{a} {op} {b} = {c}"]))]


def make_worked_one():
    t = rng.randrange(6)
    if t == 0:
        a, b, op = ri(11, 999), ri(11, 999), "+"
    elif t == 1:
        a, b, op = ri(11, 99), ri(2, 9), "+"
    elif t == 2:
        a, b = ri(30, 999), ri(11, 999)
        a, b, op = max(a, b), min(a, b), "-"
    elif t == 3:
        a, b, op = ri(13, 99), ri(2, 9), "x"
    elif t == 4:
        a, b, op = ri(13, 49), ri(11, 29), "x"
    else:
        a, b, op = ri(2, 12), ri(13, 99), "x"
    if op == "+":
        c, body = a + b, worked_add(a, b)
    elif op == "-":
        c, body = a - b, worked_sub(a, b)
    else:
        c, body = a * b, worked_mul(a, b)
    q = pick([f"What is {a} {op} {b}?", f"Calculate {a} {op} {b}.", f"What is {a} {S.OPWORD[op]} {b}?", f"{a} {op} {b} = ?", f"Compute {a} {op} {b}."])
    return "arith_worked", [("user", q), ("assistant", f"{body}.\nSo {a} {op} {b} = {c}.")]


def make_word_one():
    pool = [f for f, w in zip(S.PROBLEMS, S.WEIGHTS) for _ in range(w)]
    while True:
        q, a = pick(pool)()
        if any(b in q for b in S.BLACKLIST):
            continue
        return "math_word", [("user", q), ("assistant", pick(S.OPENERS) + expand(a))]


def verify(convs):
    """Re-check every 'a op b (op c ...) = d' of every answer."""
    pat = re.compile(r"(?<![\w.$])(\d+(?:\.\d+)?(?: [-+x/] \d+(?:\.\d+)?)+) = (\d+(?:\.\d+)?)")
    bad = 0
    for g, msgs in convs:
        for expr, c in pat.findall(msgs[-1][1]):
            try:
                r = eval(expr.replace(" x ", " * "), {"__builtins__": {}}, {})
            except ZeroDivisionError:
                bad += 1
                print("DIVISION BY ZERO", msgs[-1][1][:160])
                continue
            if abs(r - float(c)) > 1e-9:
                bad += 1
                print("BAD ARITHMETIC", expr, "=", c, "|", msgs[-1][1][:140])
    return bad


def write(name, convs):
    msgs_all = [[list(m) for m in msgs] for _, msgs in convs]
    counts = L.content_token_counts([[tuple(m) for m in ms] for ms in msgs_all])
    recs = []
    for i, ((group, _), msgs, cnt) in enumerate(zip(convs, msgs_all, counts)):
        t = [tuple(m) for m in msgs]
        recs.append({"id": f"{name}:{i}", "src": name, "msgs": msgs, "ntok": L.doc_tokens(t, cnt), "nsup": L.supervised_tokens(t, cnt), "stratum": group})
    os.makedirs(args.out, exist_ok=True)
    L.write_jsonl(os.path.join(args.out, f"{name}.jsonl"), recs)
    tok = sum(r["ntok"] for r in recs)
    by = {}
    for r in recs:
        by[r["stratum"]] = by.get(r["stratum"], 0) + 1
    print(f"{name}: {len(recs)} conversations, {tok} tokens, mean {tok / len(recs):.0f}")
    print("  " + ", ".join(f"{k} {v}" for k, v in sorted(by.items())))


def all_facts():
    """About 2,900 basic facts: sums and differences with the ones and tens digits (the steps of every worked calculation), tables, tens x digit, tens and hundreds sums."""
    f = []
    for a in range(0, 21):
        for b in range(0, 21):
            f.append(("+", a, b, a + b))
    for a in range(0, 100):
        for b in range(0, 10):
            f.append(("+", a, b, a + b))
            if a >= b:
                f.append(("-", a, b, a - b))
    for a in range(0, 21):
        for b in range(0, a + 1):
            f.append(("-", a, b, a - b))
    for a in range(2, 13):
        for b in range(2, 13):
            f.append(("x", a, b, a * b))
            f.append(("/", a * b, a, b))
    for a in range(1, 10):
        for b in range(2, 10):
            f.append(("x", a * 10, b, a * 10 * b))
    for a in range(1, 10):
        for b in range(1, 10):
            f.append(("+", a * 10, b * 10, (a + b) * 10))
            f.append(("+", a * 100, b * 100, (a + b) * 100))
            if a >= b:
                f.append(("-", a * 10, b * 10, (a - b) * 10))
    return sorted(set(f))


def make_big_facts(n):
    """n basic-fact examples drawn with repetition (no dedup): every fact is seen hundreds of times, which is what memorizing a table takes."""
    facts = all_facts()
    print(len(facts), "distinct facts")
    out = []
    for _ in range(n):
        op, a, b, c = pick(facts)
        w = S.OPWORD[op]
        q = pick([f"What is {a} {op} {b}?", f"{a} {op} {b} = ?", f"Calculate {a} {op} {b}.", f"What is {a} {w} {b}?", f"What's {a} {w} {b}?", f"Compute {a} {op} {b}.", f"{a} {op} {b}"])
        out.append(("arith_fact", [("user", q), ("assistant", pick([f"{a} {op} {b} = {c}.", f"{a} {op} {b} = {c}"]))]))
    return out


if __name__ == "__main__" and os.environ.get("ARITH_MODE") == "facts":
    convs = make_big_facts(args.n)
    bad = verify(convs)
    if bad:
        sys.exit(f"{bad} wrong results: fix the generators")
    write(args.name, convs)
    sys.exit(0)

if __name__ == "__main__":
    convs, seen = [], set()
    while len(convs) < args.n:
        r = rng.random()
        g, msgs = make_facts_one() if r < 0.30 else (make_worked_one() if r < 0.65 else make_word_one())
        key = tuple(msgs)
        if key in seen:
            continue
        seen.add(key)
        convs.append((g, msgs))
    bad = verify(convs)
    if bad:
        sys.exit(f"{bad} wrong results: fix the generators")
    write(args.name, convs)
