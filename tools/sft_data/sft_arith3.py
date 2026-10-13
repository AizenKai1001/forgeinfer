#!/usr/bin/env python3
"""sft_arith3.py [--out DIR] [--n N] [--name NAME]: the schoolbook arithmetic set of sft_arith2.py with what the first patches still missed (written by the agent for the project, computed and re-checked, no model output):

  - DRILLS of the micro-facts that every column step needs, in the line format of the worked steps: "9 x 9 + 4 = 85.", "4 + 7 + 1 = 12.", "3 - 1 - 5 = -3.", and the basic facts "5 + 7 = 12." (the checklist's apples problem still came out as 5 + 7 = 13);
  - the times table to 12 x 12 as plain facts (the probe asks "7 x 12");
  - WORD PROBLEMS with more wording: pronouns ("he gives 4 apples to his sister"), money, ages, distances, groups, totals, two and three steps, so that the format is not tied to the first templates;
  - DECIMALS in rate problems (60 miles per hour for 2.5 hours): the column method on the number without the point, then one place back: "60 x 25 = 1500, 1500 / 10 = 150".
Mix per set: 25 percent drills and facts, 30 percent plain column calculations (sft_arith2.plain_one), 45 percent word problems (the new templates and the old ones). Output: DIR/NAME.jsonl.
"""
import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_argv = sys.argv
sys.argv = [sys.argv[0]]
import sft_arith2 as A2  # noqa: E402  (its helpers: column method, expand, verify)

sys.argv = _argv
import sft_lib as L  # noqa: E402

S = A2.S
ap = argparse.ArgumentParser()
ap.add_argument("--out", default=os.path.join(L.WORK, "pool"))
ap.add_argument("--n", type=int, default=100000)
ap.add_argument("--name", default="arith3")
ap.add_argument("--stream", default="", help="another random stream (fresh examples): any text")
args = ap.parse_args()
rng = L.rng_for("arith3", args.stream) if args.stream else L.rng_for("arith3")
A2.rng = rng
S.rng = rng
pick, ri = S.pick, S.ri
sol = S.sol

FEM = ["Mia", "Sara", "Ana", "Priya", "Lily", "Zoe", "Eva", "Ruth", "Nina", "Ella", "Tara", "Maya", "Anna", "Iris", "Nora"]
MAS = ["Ben", "Leo", "Omar", "Jack", "Noah", "Sam", "Kai", "Hugo", "Max", "Ravi", "Finn", "Luis", "Dev", "Theo", "Alex"]
REL = ["sister", "brother", "friend", "cousin", "neighbor", "teacher", "classmate"]


def person():
    if rng.random() < 0.5:
        return pick(FEM), "she", "her", "her"
    return pick(MAS), "he", "his", "him"


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# drills

def drill_one():
    t = rng.randrange(8)
    if t == 0:
        a, b = ri(2, 9), ri(2, 9)
        c = ri(0, 8)
        e = f"{a} x {b}" + (f" + {c}" if c else "")
        r = a * b + c
        q = pick([f"What is {e}?", f"{e} = ?", f"Calculate {e}."])
        return "arith_drill", [("user", q), ("assistant", f"{e} = {r}.")]
    if t == 1:
        a, b, c = ri(0, 9), ri(0, 9), ri(0, 1)
        e = f"{a} + {b}" + (f" + {c}" if c else "")
        q = pick([f"What is {e}?", f"{e} = ?", f"Calculate {e}."])
        return "arith_drill", [("user", q), ("assistant", f"{e} = {a + b + c}.")]
    if t == 2:
        a, b, c = ri(0, 9), ri(0, 9), ri(0, 1)
        e = f"{a}" + (f" - {c}" if c else "") + f" - {b}"
        r = a - c - b
        q = pick([f"What is {e}?", f"{e} = ?", f"Calculate {e}."])
        return "arith_drill", [("user", q), ("assistant", f"{e} = {r}.")]
    if t == 3:
        a, b = ri(0, 9), ri(0, 9)
        q = pick([f"What is {a} + {b}?", f"{a} + {b} = ?", f"What is {a} plus {b}?", f"What's {a} plus {b}?"])
        return "arith_drill", [("user", q), ("assistant", f"{a} + {b} = {a + b}.")]
    if t == 4:
        a = ri(2, 18)
        b = ri(0, min(9, a))
        q = pick([f"What is {a} - {b}?", f"{a} - {b} = ?", f"What is {a} minus {b}?"])
        return "arith_drill", [("user", q), ("assistant", f"{a} - {b} = {a - b}.")]
    if t == 5:
        a, b = ri(2, 12), ri(2, 12)
        q = pick([f"What is {a} x {b}?", f"{a} x {b} = ?", f"What is {a} times {b}?", f"What's {a} times {b}?"])
        return "arith_drill", [("user", q), ("assistant", f"{a} x {b} = {a * b}.")]
    if t == 6:
        b, c = ri(2, 12), ri(2, 12)
        q = pick([f"What is {b * c} / {b}?", f"{b * c} / {b} = ?", f"What is {b * c} divided by {b}?"])
        return "arith_drill", [("user", q), ("assistant", f"{b * c} / {b} = {c}.")]
    a, b = ri(10, 99), ri(2, 9)
    r = a % b
    q = f"What is {a} / {b}?"
    return "arith_drill", [("user", q), ("assistant", A2.s_div(a, b))]


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# word problems with more wording; each returns (question, solution text before expansion)

def w_pron_addsub():
    n, he, his, him = person()
    (s, p), a, b = pick(S.ITEMS), ri(3, 40), ri(2, 30)
    c = ri(1, a + b - 1)
    rel = pick(REL)
    q = pick([f"{n} has {a} {p} and buys {b} more. Then {he} gives {c} {p} to {his} {rel}. How many {p} does {n} have now?",
              f"{n} had {a} {p}. {he.capitalize()} found {b} more and then gave {c} to {his} {rel}. How many {p} does {n} have now?",
              f"{n} starts with {a} {p}, gets {b} more from a friend and then loses {c}. How many {p} are left?"])
    return q, sol([f"{a} + {b} = {a + b}.", f"{a + b} - {c} = {a + b - c}."], a + b - c)


def w_basket():
    n, he, his, him = person()
    m = pick(MAS + FEM)
    (s, p), a = pick(S.ITEMS), ri(10, 80)
    b, c = ri(2, a - 1), ri(2, 40)
    q = f"There are {a} {p} in a basket. {n} takes {b} and then {m} adds {c}. How many {p} are in the basket now?"
    return q, sol([f"{a} - {b} = {a - b}.", f"{a - b} + {c} = {a - b + c}."], a - b + c)


def w_daily():
    n, he, his, him = person()
    a, d = ri(5, 40), ri(2, 9)
    q = pick([f"{n} reads {a} pages every day for {d} days. How many pages does {he} read in total?", f"{n} saves {a} dollars every week for {d} weeks. How much money does {he} have then?",
              f"{n} walks {a} minutes every day for {d} days. How many minutes is that?"])
    unit = "pages" if "pages" in q else ("dollars" if "dollars" in q else "minutes")
    return q, sol([f"{a} x {d} = {a * d}."], a * d, f" {unit}")


def w_money():
    n, he, his, him = person()
    a = ri(10, 99)
    b = ri(2, a - 1)
    thing = pick(["book", "toy", "game", "T-shirt", "ticket", "lunch", "kite", "hat"])
    q = pick([f"{n} has ${a} and spends ${b} on a {thing}. How much money does {he} have left?", f"{n} has ${a}. {he.capitalize()} buys a {thing} for ${b}. How much money is left?"])
    return q, sol([f"{a} - {b} = {a - b}."], f"${a - b}")


def w_animals():
    a, b = ri(8, 90), ri(8, 90)
    who = pick(["farmer", "zoo keeper", "girl on a farm", "shepherd"])
    x, y = pick([("cows", "sheep"), ("ducks", "chickens"), ("horses", "goats"), ("cats", "dogs")])
    q = f"A {who} has {a} {x} and {b} {y}. How many animals are there in total?"
    return q, sol([f"{a} + {b} = {a + b}."], a + b)


def w_distance():
    n, he, his, him = person()
    a, b = ri(2, 40), ri(2, 40)
    q = f"{n} runs {a} miles on Monday and {b} miles on Tuesday. How many miles does {he} run in total?"
    return q, sol([f"{a} + {b} = {a + b}."], a + b, " miles")


def w_groups():
    k, m = ri(2, 12), ri(3, 25)
    (s, p) = pick(S.ITEMS)
    q = pick([f"A bag holds {m} {p}. How many {p} are in {k} bags?", f"Each box has {m} {p}. How many {p} are in {k} boxes?"])
    return q, sol([f"{m} x {k} = {m * k}."], m * k)


def w_class():
    a = ri(15, 40)
    b = ri(1, a - 1)
    q = pick([f"A class has {a} students. {b} of them are absent. How many students are present?", f"There are {a} children at a party and {b} go home. How many children are left?"])
    return q, sol([f"{a} - {b} = {a - b}."], a - b)


def w_age():
    n, he, his, him = person()
    m = pick(MAS + FEM)
    while m == n:
        m = pick(MAS + FEM)
    a, b = ri(5, 40), ri(1, 12)
    q = f"{n} is {a} years old. {m} is {b} years older than {n}. How old is {m}?"
    return q, sol([f"{a} + {b} = {a + b}."], a + b)


def w_total_cost():
    n, he, his, him = person()
    k, c, d = ri(2, 9), ri(2, 12), ri(3, 30)
    q = f"{n} buys {k} pens for ${c} each and a notebook for ${d}. How much does {he} pay in total?"
    return q, sol([f"{k} x {c} = {k * c}.", f"{k * c} + {d} = {k * c + d}."], f"${k * c + d}")


def w_rate_dec():
    v = pick(S.VEHICLES)
    sp = 2 * ri(5, 45)   # an even speed: speed x (a multiple of 5) is a multiple of 10, so that the result of the decimal problem is a whole number
    t10 = pick([15, 25, 35, 45])
    t = t10 / 10
    if sp == 60 and t10 == 25:
        return w_rate_dec()
    d = sp * t10 // 10
    q = pick([f"A {v} travels at {sp} miles per hour for {S.num(t)} hours. How far does it go?", f"A {v} moves at {sp} kilometers per hour for {S.num(t)} hours. How many kilometers does it cover?"])
    unit = " miles" if "miles" in q else " kilometers"
    steps = [f"{sp} x {S.num(t)}: first {A2.s_mul(sp, t10)} The number {S.num(t)} has one decimal place, so divide by 10: {sp * t10} / 10 = {d}."]
    return q, sol(steps, d, unit)


NEW_PROBLEMS = [w_pron_addsub, w_basket, w_daily, w_money, w_animals, w_distance, w_groups, w_class, w_age, w_total_cost, w_rate_dec]
OLD = [f for f, w in zip(S.PROBLEMS, S.WEIGHTS) for _ in range(w) if f not in (S.p_avg, S.p_percent, S.p_rate)]


def word_one():
    while True:
        f = pick(NEW_PROBLEMS + NEW_PROBLEMS + OLD)
        q, a = f()
        if any(b in q for b in S.BLACKLIST) or "%" in q:
            continue
        text = a if f is w_rate_dec else A2.expand(a)   # the decimal rate solution is written out completely: expanding it again would nest the column method twice
        return "math_word", [("user", q), ("assistant", pick(S.OPENERS) + text)]


def verify(convs):
    bad = A2.verify(convs)
    # the decimal rate lines: 'A x B: first ... So A x B10 = P. ... P / 10 = D.' are checked as ordinary lines by A2.verify except the division by 10: check it here
    pat = re.compile(r"(\d+) / 10 = (\d+)\.")
    for g, msgs in convs:
        for p, d in pat.findall(msgs[-1][1]):
            if int(p) // 10 != int(d) or int(p) % 10:
                bad += 1
                print("BAD DECIMAL", msgs[-1][1][:200])
    return bad


if __name__ == "__main__":
    convs, seen = [], set()
    while len(convs) < args.n:
        r = rng.random()
        if r < 0.15:
            g, msgs = drill_one()
        elif r < 0.25:
            g, msgs = drill_one()
        elif r < 0.55:
            g, msgs = A2.plain_one()
        else:
            g, msgs = word_one()
        key = tuple(msgs)
        if key in seen:
            continue
        seen.add(key)
        convs.append((g, msgs))
    bad = verify(convs)
    if bad:
        sys.exit(f"{bad} wrong results: fix the generators")
    A2.args.out = args.out
    A2.write(args.name, convs)
