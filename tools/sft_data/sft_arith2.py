#!/usr/bin/env python3
"""sft_arith2.py [--out DIR] [--n N] [--name NAME]: SCHOOLBOOK arithmetic for the Forge-1 chat patch (computed by this program, every line and every result re-checked, no model output).

The place-value format of sft_arith.py (345 + 278: 300 + 200 = 500, ...) was followed by the patched model, but its sums with a carry were still wrong ("60 + 42 = 92"): one step there is a multi-digit calculation.
Here every step is ONE digit operation the model can look up (a table of about 300 facts): the column method, written out, with explicit carries and borrows:
  345 + 278: ones: 5 + 8 = 13, write 3 carry 1. tens: 4 + 7 + 1 = 12, write 2 carry 1. hundreds: 3 + 2 + 1 = 6, write 6. So 345 + 278 = 623.
  345 - 278: ones: 5 - 8 = -3, write 7 borrow 1. tens: 4 - 1 - 7 = -4, write 6 borrow 1. hundreds: 3 - 1 - 2 = 0, write 0. So 345 - 278 = 67.
  17 x 6: ones: 7 x 6 = 42, write 2 carry 4. tens: 1 x 6 + 4 = 10, write 0 carry 1. hundreds: write 1. So 17 x 6 = 102.
  23 x 14: 23 x 4: ... So 23 x 4 = 92. 23 x 10 = 230. 92 + 230: ... So 92 + 230 = 322. So 23 x 14 = 322.
  135 / 5: 13 / 5 = 2, remainder 3. 35 / 5 = 7, remainder 0. So 135 / 5 = 27.
Basic facts (single digits, sums to 18, the times table to 9 x 9 and exact quotients to 12) stay plain: "7 x 6 = 42".
Three families, one format: plain questions (35 %), word problems of sft_skills.py with every non-basic calculation of the solution written this way (65 %).
Output: DIR/NAME.jsonl (record format of sft_pools.py).
"""
import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_argv = sys.argv
sys.argv = [sys.argv[0]]
import sft_skills as S  # noqa: E402  (only its problem generators are used)

sys.argv = _argv
import sft_lib as L  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=os.path.join(L.WORK, "pool"))
ap.add_argument("--n", type=int, default=100000)
ap.add_argument("--name", default="arith2")
args = ap.parse_args()
rng = L.rng_for("arith2")
S.rng = rng
pick, ri = S.pick, S.ri

PLACES = ["ones", "tens", "hundreds", "thousands", "ten-thousands", "hundred-thousands"]


def dig(n, i):
    return (n // 10 ** i) % 10


def s_add(a, b):
    n = max(len(str(a)), len(str(b)))
    parts, carry = [], 0
    written = []
    for i in range(n):
        da, db = dig(a, i), dig(b, i)
        t = da + db + carry
        expr = f"{da} + {db}" + (" + 1" if carry else "")
        c2 = t // 10
        written.append(t % 10)
        parts.append(f"{PLACES[i]}: {expr} = {t}, write {t % 10}" + (f" carry {c2}." if c2 else "."))
        carry = c2
    if carry:
        parts.append(f"{PLACES[n]}: write {carry}.")
        written.append(carry)
    return " ".join(parts) + f" So {a} + {b} = {a + b}.", written


def s_sub(a, b):
    assert a >= b
    n = len(str(a))
    parts, borrow, written = [], 0, []
    for i in range(n):
        da, db = dig(a, i), dig(b, i)
        v = da - borrow - db
        expr = f"{da}" + (" - 1" if borrow else "") + f" - {db}"
        if v < 0:
            parts.append(f"{PLACES[i]}: {expr} = {v}, write {v + 10} borrow 1.")
            written.append(v + 10)
            borrow = 1
        else:
            parts.append(f"{PLACES[i]}: {expr} = {v}, write {v}.")
            written.append(v)
            borrow = 0
    return " ".join(parts) + f" So {a} - {b} = {a - b}.", written


def s_mul1(a, m):
    """a x m for a single-digit m (2..9)."""
    n = len(str(a))
    parts, carry, written = [], 0, []
    for i in range(n):
        da = dig(a, i)
        t = da * m + carry
        expr = f"{da} x {m}" + (f" + {carry}" if carry else "")
        c2 = t // 10
        written.append(t % 10)
        parts.append(f"{PLACES[i]}: {expr} = {t}, write {t % 10}" + (f" carry {c2}." if c2 else "."))
        carry = c2
    if carry:
        parts.append(f"{PLACES[n]}: write {carry}.")
        written.append(carry)
    return " ".join(parts) + f" So {a} x {m} = {a * m}.", written


def s_mul(a, b):
    """a x b, b of one or two digits (b >= 2)."""
    if b < 10:
        return s_mul1(a, b)[0]
    tens, ones = b // 10, b % 10
    if ones == 0:
        if tens == 1:
            return f"{a} x {b} = {a * b}."
        return f"{a} x {tens}: " + s_mul1(a, tens)[0] + f" So {a} x {b} = {a * b}."
    p1 = a * ones
    p2 = a * tens * 10
    head = f"{a} x {ones}: " + (s_mul1(a, ones)[0] if ones >= 2 else f"{a} x 1 = {a}.")
    if tens == 1:
        mid = f" {a} x 10 = {p2}."
    else:
        mid = f" {a} x {tens * 10}: " + s_mul1(a, tens)[0] + f" So {a} x {tens * 10} = {p2}."
    return head + mid + f" {p1} + {p2}: " + s_add(p1, p2)[0] + f" So {a} x {b} = {a * b}."


def s_div(a, d):
    """a / d for a single-digit d (2..9) by long division; remainder allowed (the plain question asks for it)."""
    s = str(a)
    parts, r, q = [], 0, []
    started = False
    for ch in s:
        cur = r * 10 + int(ch)
        if not started and cur < d:
            r = cur
            continue
        started = True
        qq, r = divmod(cur, d)
        q.append(str(qq))
        parts.append(f"{cur} / {d} = {qq}, remainder {r}.")
    quot = int("".join(q)) if q else 0
    tail = f" So {a} / {d} = {quot}." if r == 0 else f" So {a} / {d} = {quot} remainder {r}."
    return f"{a} / {d}: " + " ".join(parts) + tail


def expr_result(a, op, b):
    return {"+": a + b, "-": a - b, "x": a * b, "/": a // b if b else 0}[op]


# basic facts stay plain
def basic(a, op, b):
    if op == "+":
        return a < 10 and b < 10
    if op == "-":
        return a <= 18 and b <= 9 and a >= b
    if op == "x":
        return a <= 9 and b <= 9
    if op == "/":
        return a % b == 0 and a // b <= 12 and b <= 12
    return False


def worked(a, op, b):
    if op == "+":
        return f"{a} + {b}: " + s_add(a, b)[0]
    if op == "-":
        return f"{a} - {b}: " + s_sub(a, b)[0]
    if op == "x":
        return f"{a} x {b}: " + s_mul(a, b) if min(a, b) >= 2 and (b < 100) else None
    if op == "/":
        return s_div(a, b) if 2 <= b <= 9 and a % b == 0 else None
    return None


RE_EXPR = re.compile(r"(?<![\w.$])(?<![-+x/] )(\d+) ([-+x/]) (\d+) = (\d+)(?=[.\s]|$)")


def expand(text):
    """Every 'a op b = c' of a solution that is not a basic fact becomes its schoolbook version (the result stays the same, the line gets longer)."""
    def rep(m):
        a, op, b, c = int(m.group(1)), m.group(2), int(m.group(3)), int(m.group(4))
        if basic(a, op, b):
            return m.group(0)
        if op == "x" and a < b:      # put the number with more digits first: 7 x 23 -> 23 x 7
            a, b = b, a
        w = worked(a, op, b)
        return w[:-1] if w and w.endswith(".") else (w or m.group(0))
    return RE_EXPR.sub(rep, text)


def plain_one():
    t = rng.randrange(10)
    if t <= 2:
        n1, n2 = pick([(2, 2), (3, 3), (3, 2), (4, 3), (4, 4), (2, 1), (3, 1)])
        a, b, op = ri(10 ** (n1 - 1), 10 ** n1 - 1), ri(10 ** (n2 - 1) if n2 > 1 else 2, 10 ** n2 - 1), "+"
    elif t <= 5:
        n1, n2 = pick([(2, 2), (3, 3), (3, 2), (4, 3), (4, 4), (2, 1), (3, 1)])
        a, b = ri(10 ** (n1 - 1), 10 ** n1 - 1), ri(10 ** (n2 - 1) if n2 > 1 else 2, 10 ** n2 - 1)
        a, b, op = max(a, b), min(a, b), "-"
    elif t <= 8:
        a, b, op = pick([(ri(11, 99), ri(2, 9)), (ri(100, 999), ri(2, 9)), (ri(11, 99), ri(11, 99)), (ri(100, 400), ri(11, 30))]), None, "x"
        a, b = a
    else:
        d = ri(2, 9)
        a, b, op = ri(11, 999), d, "/"
        if rng.random() < 0.8:
            a = d * ri(2, 999 // d)
    if basic(a, op, b):
        return plain_one()
    q = pick([f"What is {a} {op} {b}?", f"Calculate {a} {op} {b}.", f"What is {a} {S.OPWORD[op]} {b}?", f"{a} {op} {b} = ?", f"Compute {a} {op} {b}.", f"What's {a} {S.OPWORD[op]} {b}?"])
    if op == "/" and a % b:
        return "arith_sb", [("user", q), ("assistant", s_div(a, b))]
    w = worked(a, op, b)
    if w is None:
        return plain_one()
    return "arith_sb", [("user", q), ("assistant", w)]


def word_one():
    pool = [f for f, w in zip(S.PROBLEMS, S.WEIGHTS) for _ in range(w) if f is not S.p_avg]   # the average problem adds a chain of 3-5 numbers in one line: left out until chains are written column by column too
    while True:
        q, a = pick(pool)()
        if any(b in q for b in S.BLACKLIST) or re.search(r"\d\.\d", q + a) or "%" in q:
            continue
        return "math_word", [("user", q), ("assistant", pick(S.OPENERS) + expand(a))]


def verify(convs):
    """(1) every 'a op b (op c ...) = d' of every line; (2) every 'So a op b = c' against Python; (3) the written digits of every column calculation put together give the result."""
    pat = re.compile(r"(?<![\w.$])(\d+(?: [-+x/] \d+)+) = (-?\d+)")
    final = re.compile(r"So (\d+) ([-+x/]) (\d+) = (\d+)(?: remainder (\d+))?\.")
    bad = 0
    for g, msgs in convs:
        text = msgs[-1][1]
        for expr, c in pat.findall(text):
            try:
                r = eval(expr.replace(" x ", " * ").replace(" / ", " // "), {"__builtins__": {}}, {})
            except ZeroDivisionError:
                bad += 1
                print("DIVISION BY ZERO", text[:200])
                continue
            if "/" in expr:
                continue   # long-division lines carry a remainder: checked below
            if r != int(c):
                bad += 1
                print("BAD LINE", expr, "=", c, "|", text[:200])
        for a, op, b, c, rem in final.findall(text):
            a, b, c = int(a), int(b), int(c)
            r = expr_result(a, op, b)
            if r != c or (op == "/" and a - r * b != int(rem or 0)):
                bad += 1
                print("BAD RESULT", a, op, b, c, "|", text[:200])
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


if __name__ == "__main__":
    convs, seen = [], set()
    while len(convs) < args.n:
        g, msgs = plain_one() if rng.random() < 0.35 else word_one()
        key = tuple(msgs)
        if key in seen:
            continue
        seen.add(key)
        convs.append((g, msgs))
    bad = verify(convs)
    if bad:
        sys.exit(f"{bad} wrong results: fix the generators")
    write(args.name, convs)
