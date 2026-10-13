#!/usr/bin/env python3
"""sft_drills.py [--out DIR] [--n N] [--name NAME]: DRILLS of the micro-facts of the column method, in the exact line formats of the worked answers of sft_arith2.py / sft_arith3.py (computed by this program, re-checked, no model output).

The arithmetic probe (100 fresh problems) showed that the remaining mistakes of the skills patch are single lines inside a correct procedure: "4 x 8 + 7 = 37" instead of 39, "9 x 8 + 4 = 85" instead of 76, "42 / 6 = 6, remainder 2" instead of "7, remainder 0"; long division and one-digit multipliers fail for that reason, the procedure itself is followed. The lines are table lookups
(digit x digit + carry: 576 facts; a two-digit number divided by a digit with its remainder: 700 facts; digit sums with a carry; differences with a borrow; the split of a two-digit product into the digit written and the carry), so they are drilled
as separate short examples, 20 to 150 repetitions of every fact:
  D1 'a x b + c'   (25 %)    D2 'n / d = q, remainder r'  (25 %)    D3 'digit + digit + carry' (10 %)    D4 'digit - borrow - digit' with negatives (10 %)
  D5 tens sums 'x0 + y0'   (5 %)    D6 'a x b = p, write w carry c' (25 %)
Output: DIR/NAME.jsonl (record format of sft_pools.py).
"""
import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sft_lib as L  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=os.path.join(L.WORK, "pool"))
ap.add_argument("--n", type=int, default=90000)
ap.add_argument("--name", default="drills")
args = ap.parse_args()
rng = L.rng_for("drills")


def pick(lst):
    return lst[rng.randrange(len(lst))]


def d1():
    a, b, c = rng.randint(2, 9), rng.randint(2, 9), rng.randint(0, 8)
    e = f"{a} x {b}" + (f" + {c}" if c else "")
    q = pick([f"What is {e}?", f"{e} = ?", f"Calculate {e}.", f"Compute {e}."])
    return "drill_mul", q, f"{e} = {a * b + c}."


def d2():
    d = rng.randint(2, 9)
    n = rng.randint(d, 89)
    qq, r = divmod(n, d)
    q = pick([f"What is {n} / {d}? Give the quotient and the remainder.", f"Divide {n} by {d}: quotient and remainder?", f"{n} / {d} = ? (with remainder)", f"What is {n} divided by {d}, with the remainder?"])
    return "drill_div", q, f"{n} / {d} = {qq}, remainder {r}."


def d3():
    a, b = rng.randint(0, 9), rng.randint(0, 9)
    c = rng.randint(0, 1)
    e = f"{a} + {b}" + (" + 1" if c else "")
    return "drill_add", pick([f"What is {e}?", f"{e} = ?", f"Calculate {e}."]), f"{e} = {a + b + c}."


def d4():
    a, b = rng.randint(0, 9), rng.randint(0, 9)
    c = rng.randint(0, 1)
    e = f"{a}" + (" - 1" if c else "") + f" - {b}"
    r = a - c - b
    return "drill_sub", pick([f"What is {e}?", f"{e} = ?", f"Calculate {e}."]), f"{e} = {r}" + (f", write {r + 10} borrow 1." if r < 0 else f", write {r}.")


def d5():
    x, y = rng.randint(1, 9), rng.randint(1, 9)
    return "drill_tens", pick([f"What is {x * 10} + {y * 10}?", f"{x * 10} + {y * 10} = ?"]), f"{x * 10} + {y * 10} = {(x + y) * 10}."


def d6():
    a, b = rng.randint(2, 9), rng.randint(2, 9)
    p = a * b
    q = pick([f"Multiply {a} x {b}: which digit do you write and what do you carry?", f"{a} x {b}: write which digit, carry what?", f"What do you write and what do you carry for {a} x {b}?"])
    return "drill_carry", q, f"{a} x {b} = {p}, write {p % 10}" + (f" carry {p // 10}." if p >= 10 else ".")


KINDS = [(d1, 25), (d2, 25), (d3, 10), (d4, 10), (d5, 5), (d6, 25)]
POOL = [f for f, w in KINDS for _ in range(w)]


def verify(convs):
    pat = re.compile(r"(\d+(?: [-+x/] \d+)+) = (-?\d+)")
    bad = 0
    for g, msgs in convs:
        a = msgs[-1][1]
        for expr, c in pat.findall(a):
            if "/" in expr:
                n, d = (int(x) for x in expr.split(" / "))
                m = re.search(r"= (\d+), remainder (\d+)", a)
                if not m or int(m.group(1)) != n // d or int(m.group(2)) != n % d:
                    bad += 1
                    print("BAD DIV", a)
                continue
            if eval(expr.replace(" x ", " * ")) != int(c):
                bad += 1
                print("BAD", a)
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
    print(f"{name}: {len(recs)} conversations, {tok} tokens, mean {tok / len(recs):.0f}  " + ", ".join(f"{k} {v}" for k, v in sorted(by.items())))


if __name__ == "__main__":
    convs = []
    for _ in range(args.n):
        g, q, a = pick(POOL)()
        convs.append((g, [("user", q), ("assistant", a)]))
    if verify(convs):
        sys.exit("wrong results: fix the generators")
    write(args.name, convs)
