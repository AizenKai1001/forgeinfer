#!/usr/bin/env python3
"""sft_skills.py [--out DIR] [--scale S]: PROGRAMMATIC skill conversations for the Forge-1 chat fine-tune. Every answer is computed by this program or taken from a small hand-written table, so it is correct by construction
(the arithmetic of every solution is re-checked at the end). No model output is copied. The families (what the human-written mix teaches badly, see the 20-prompt checklist):
  math_word     two-step and one-step word problems (add, subtract, multiply, divide, rate x time, price and change, area, percent, average, comparisons) with a step-by-step solution ending "The answer is: N"
  arithmetic    plain sums, differences, products and exact quotients
  memory        multi-turn chats in which the user tells a fact (name, colour, pet, city ...) and later asks for it, with filler turns; asking for a fact never told gets an honest "you haven't told me"
  one_word      "Answer in one word", "answer yes or no", "reply with only a number" from tables of facts
  transform     exact text operations: upper / lower case, reverse a word, count letters or words, sort numbers, repeat a word
  bullets       "write exactly N bullet points about T" from hand-written facts, "list N examples of C"
  rewrite       polite rewrites (hand-written pairs) and one-sentence summaries of templated news paragraphs
  code          short Python functions with a code block and a one-line explanation
Held out on purpose (they are the checklist and must stay unseen): Tom's apples, the 60 mph train, the sum of squares, the palindrome function, "Give me the report now", the committee / park summary, "benefits of exercise",
the capital of France, the colour of the sky. The probe scratch/skills_probe.json (C) uses other prompts of the same families.
Output: DIR/skills.jsonl in the record format of sft_pools.py (id, src, msgs, ntok, nsup, stratum).
"""
import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sft_lib as L  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=os.path.join(L.WORK, "pool"))
ap.add_argument("--scale", type=float, default=1.0, help="multiplies the number of conversations of every family")
ap.add_argument("--no-arith", action="store_true", help="leave out word problems and arithmetic (they come from sft_arith.py)")
ap.add_argument("--name", default="skills", help="output name (DIR/NAME.jsonl)")
args = ap.parse_args()
rng = L.rng_for("skills")


def pick(lst):
    return lst[rng.randrange(len(lst))]


def ri(a, b):
    return rng.randint(a, b)


def num(x):
    return str(int(x)) if float(x) == int(x) else str(x)


NAMES = ["Mia", "Ben", "Sara", "Leo", "Ana", "Omar", "Priya", "Jack", "Lily", "Noah", "Zoe", "Sam", "Eva", "Kai", "Ruth", "Hugo", "Nina", "Max", "Ella", "Ravi", "Tara", "Finn", "Maya", "Luis", "Anna", "Dev", "Iris", "Theo", "Nora", "Alex"]
ITEMS = [("apple", "apples"), ("pencil", "pencils"), ("cookie", "cookies"), ("sticker", "stickers"), ("marble", "marbles"), ("book", "books"), ("card", "cards"), ("balloon", "balloons"), ("orange", "oranges"),
         ("stamp", "stamps"), ("toy car", "toy cars"), ("crayon", "crayons"), ("cupcake", "cupcakes"), ("bottle", "bottles"), ("ticket", "tickets"), ("notebook", "notebooks"), ("lemon", "lemons"), ("coin", "coins"),
         ("cherry", "cherries"), ("sandwich", "sandwiches")]
VEHICLES = ["car", "bus", "train", "cyclist", "truck", "boat", "runner"]
GENERIC_SYSTEM = ["You are a helpful assistant.", "You are a helpful, honest and concise assistant.", "You are a friendly assistant. Keep your answers short.", "You are Forge-1, a small language model built from scratch. Answer clearly and be honest about what you don't know.",
                  "You are a helpful AI assistant.", "Answer as briefly as you can."]


def sol(steps, final, unit=""):
    body = steps[0] if len(steps) == 1 else "\n".join(f"Step {i + 1}: {s}" for i, s in enumerate(steps))
    return f"{body}\nThe answer is: {final}{unit}"


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# math word problems: each function returns (question, answer)

def p_add():
    n, (s, p), a, b = pick(NAMES), pick(ITEMS), ri(3, 70), ri(2, 45)
    c = a + b
    q = pick([f"{n} has {a} {p} and gets {b} more. How many {p} does {n} have now?", f"A box holds {a} {p}. {n} puts {b} more in. How many {p} are in the box now?",
              f"{n} collected {a} {p} on Monday and {b} {p} on Tuesday. How many {p} did {n} collect in all?"])
    return q, sol([f"{a} + {b} = {c}."], c)


def p_sub():
    n, (s, p), a = pick(NAMES), pick(ITEMS), ri(15, 90)
    b = ri(3, a - 2)
    c = a - b
    q = pick([f"{n} had {a} {p} and gave {b} of them to a friend. How many {p} does {n} have left?", f"There were {a} {p} on a table. {n} took {b}. How many {p} are left on the table?",
              f"{n} bought {a} {p} and used {b}. How many {p} remain?"])
    return q, sol([f"{a} - {b} = {c}."], c)


def p_two_step_addsub():
    n, m, (s, p) = pick(NAMES), pick(NAMES), pick(ITEMS)
    a, b = ri(8, 40), ri(3, 30)
    c = ri(2, a + b - 1)
    d = a + b - c
    q = f"{n} has {a} {p} and finds {b} more. Then {n} gives {c} {p} to {m}. How many {p} does {n} have now?"
    return q, sol([f"{a} + {b} = {a + b}.", f"{a + b} - {c} = {d}."], d)


def p_mult():
    n, (s, p), k, m = pick(NAMES), pick(ITEMS), ri(2, 12), ri(2, 12)
    c = k * m
    q = pick([f"{n} has {k} boxes with {m} {p} in each box. How many {p} does {n} have in total?", f"A shelf has {k} rows of {m} {p}. How many {p} are on the shelf?",
              f"{n} buys {k} packs of {p}. Each pack has {m} {p}. How many {p} is that?"])
    return q, sol([f"{k} x {m} = {c}."], c)


def p_div():
    n, (s, p), k, m = pick(NAMES), pick(ITEMS), ri(2, 9), ri(2, 12)
    a = k * m
    q = pick([f"{n} has {a} {p} and shares them equally among {k} friends. How many {p} does each friend get?", f"{a} {p} are packed equally into {k} bags. How many {p} are in each bag?",
              f"{n} splits {a} {p} into {k} equal groups. How many {p} are in each group?"])
    return q, sol([f"{a} / {k} = {m}."], m)


def p_mult_sub():
    n, (s, p), k, m = pick(NAMES), pick(ITEMS), ri(2, 9), ri(3, 12)
    b = ri(1, k * m - 1)
    c = k * m - b
    q = f"{n} has {k} bags with {m} {p} in each bag. {n} gives away {b} {p}. How many {p} does {n} have left?"
    return q, sol([f"{k} x {m} = {k * m}.", f"{k * m} - {b} = {c}."], c)


def p_mult_add():
    n, (s, p), k, m, b = pick(NAMES), pick(ITEMS), ri(2, 9), ri(2, 12), ri(2, 30)
    c = k * m + b
    q = f"{n} has {k} boxes with {m} {p} each and then buys {b} more {p}. How many {p} does {n} have now?"
    return q, sol([f"{k} x {m} = {k * m}.", f"{k * m} + {b} = {c}."], c)


def p_rate():
    v, sp, t = pick(VEHICLES), ri(4, 18) * 5, pick([2, 3, 4, 5, 6, 1.5, 3.5, 4.5])
    d = sp * t
    if sp == 60 and t == 2.5:
        sp = 45
        d = sp * t
    q = pick([f"A {v} travels at {sp} miles per hour for {num(t)} hours. How far does it go?", f"A {v} moves at {sp} km per hour. How many kilometers does it cover in {num(t)} hours?"])
    unit = " miles" if "miles" in q else " km"
    return q, sol([f"{sp} x {num(t)} = {num(d)}{unit}."], num(d), unit)


def p_price():
    n, (s, p), k, c = pick(NAMES), pick(ITEMS), ri(2, 12), ri(2, 15)
    q = pick([f"One {s} costs ${c}. How much do {k} {p} cost?", f"{n} buys {k} {p} at ${c} each. How much does {n} pay in total?"])
    return q, sol([f"{k} x {c} = {k * c}."], f"${k * c}")


def p_change():
    n, (s, p), k, c = pick(NAMES), pick(ITEMS), ri(2, 8), ri(2, 9)
    paid = (k * c // 10 + 1) * 10 if k * c % 10 else k * c + 10
    q = f"{n} buys {k} {p} that cost ${c} each and pays with ${paid}. How much change does {n} get?"
    return q, sol([f"{k} x {c} = {k * c}.", f"{paid} - {k * c} = {paid - k * c}."], f"${paid - k * c}")


def p_area():
    l, w = ri(3, 20), ri(2, 15)
    if rng.random() < 0.5:
        q = f"A rectangle is {l} cm long and {w} cm wide. What is its area?"
        return q, sol([f"area = length x width = {l} x {w} = {l * w}."], l * w, " square cm")
    q = f"A rectangle is {l} cm long and {w} cm wide. What is its perimeter?"
    return q, sol(["perimeter = 2 x (length + width).", f"{l} + {w} = {l + w}.", f"2 x {l + w} = {2 * (l + w)}."], 2 * (l + w), " cm")


def p_percent():
    price, x = ri(2, 20) * 10, pick([10, 20, 25, 50])
    disc = price * x // 100
    q = pick([f"A jacket costs ${price}. It is on sale for {x}% off. What is the sale price?", f"A {pick(['bag', 'lamp', 'game', 'watch'])} usually costs ${price}. This week it is {x}% cheaper. How much does it cost now?"])
    return q, sol([f"{x}% of {price} = {price * x // 100}.", f"{price} - {disc} = {price - disc}."], f"${price - disc}")


def p_avg():
    k = pick([3, 4, 5])
    avg = ri(5, 30)
    xs = [ri(max(1, avg - 6), avg + 6) for _ in range(k - 1)]
    xs.append(avg * k - sum(xs))
    if min(xs) < 0:
        return p_avg()
    n = pick(NAMES)
    q = f"{n} scored {', '.join(str(x) for x in xs)} in {k} games. What is the average score?"
    return q, sol([f"{' + '.join(str(x) for x in xs)} = {sum(xs)}.", f"{sum(xs)} / {k} = {avg}."], avg)


def p_compare():
    n, m, (s, p) = pick(NAMES), pick(NAMES), pick(ITEMS)
    while m == n:
        m = pick(NAMES)
    a, b = ri(10, 60), ri(10, 60)
    while a == b:
        b = ri(10, 60)
    hi, lo = (n, m) if a > b else (m, n)
    q = f"{n} has {a} {p} and {m} has {b} {p}. Who has more, and by how many?"
    return q, sol([f"Compare {a} and {b}: {max(a, b)} is bigger.", f"{max(a, b)} - {min(a, b)} = {abs(a - b)}."], f"{hi} has {abs(a - b)} more")


def p_twice():
    n, m, (s, p), a = pick(NAMES), pick(NAMES), pick(ITEMS), ri(3, 30)
    while m == n:
        m = pick(NAMES)
    q = f"{n} has {a} {p}. {m} has twice as many. How many {p} do they have together?"
    return q, sol([f"{m} has 2 x {a} = {2 * a}.", f"{a} + {2 * a} = {3 * a}."], 3 * a)


def p_fraction():
    n, (s, p), q4 = pick(NAMES), pick(ITEMS), ri(2, 20)
    a = 4 * q4
    word, part = pick([("half", a // 2), ("a quarter", a // 4)])
    q = f"{n} had {a} {p} and gave away {word} of them. How many {p} are left?"
    return q, sol([f"{word} of {a} is {part}.", f"{a} - {part} = {a - part}."], a - part)


PROBLEMS = [p_add, p_sub, p_two_step_addsub, p_mult, p_div, p_mult_sub, p_mult_add, p_rate, p_price, p_change, p_area, p_percent, p_avg, p_compare, p_twice, p_fraction]
WEIGHTS = [3, 3, 4, 3, 3, 3, 3, 2, 3, 2, 2, 2, 2, 2, 2, 2]
BLACKLIST = ["Tom has 5 apples", "60 miles per hour for 2.5 hours"]
OPENERS = ["", "", "", "Let's work through it.\n", "Let's think step by step.\n", "Here is the solution.\n"]


def make_math(n):
    out = []
    pool = [f for f, w in zip(PROBLEMS, WEIGHTS) for _ in range(w)]
    while len(out) < n:
        q, a = pick(pool)()
        if any(b in q for b in BLACKLIST):
            continue
        out.append(("math_word", [("user", q), ("assistant", pick(OPENERS) + a)]))
    return out


OPWORD = {"+": "plus", "-": "minus", "x": "times", "/": "divided by"}


def make_facts(n):
    """The single-digit and teen tables (about 900 facts: sums to 20, differences, times tables 2-12 and exact quotients), each in several phrasings: the basic facts of every longer calculation."""
    facts = []
    for a in range(0, 21):
        for b in range(0, 21):
            facts.append(("+", a, b, a + b))
            if a >= b:
                facts.append(("-", a, b, a - b))
    for a in range(2, 13):
        for b in range(2, 13):
            facts.append(("x", a, b, a * b))
            facts.append(("/", a * b, a, b))
    seen, out = set(), []
    tries = 0
    while len(out) < n and tries < n * 6:
        tries += 1
        op, a, b, c = pick(facts)
        q = pick([f"What is {a} {op} {b}?", f"{a} {op} {b} = ?", f"Calculate {a} {op} {b}.", f"What is {a} {OPWORD[op]} {b}?", f"What's {a} {OPWORD[op]} {b}?", f"Compute {a} {op} {b}.", f"{a} {op} {b}"])
        if q in seen:
            continue
        seen.add(q)
        out.append(("arith_fact", [("user", q), ("assistant", pick([f"{a} {op} {b} = {c}.", f"{a} {op} {b} = {c}"]))]))
    return out


def parts(x):
    """345 -> [300, 40, 5] (the non-zero place values)."""
    s = str(x)
    return [int(d) * 10 ** (len(s) - 1 - i) for i, d in enumerate(s) if d != "0"]


def make_cot(n):
    """Longer sums, differences and products worked out in place-value steps, so that carrying is written down instead of guessed."""
    out = []
    while len(out) < n:
        t = rng.randrange(5)
        if t == 0:
            a, b = ri(100, 999), ri(100, 999)
            pa, pb = parts(a), parts(b)
            if len(pa) != len(pb):
                continue
            steps = [f"{x} + {y} = {x + y}." for x, y in zip(pa, pb)]
            tot = " + ".join(str(x + y) for x, y in zip(pa, pb))
            body = "Split into hundreds, tens and ones.\n" + "\n".join(steps) + f"\n{tot} = {a + b}.\nSo {a} + {b} = {a + b}."
            q = pick([f"What is {a} + {b}?", f"Calculate {a} + {b}.", f"What is {a} plus {b}?"])
        elif t == 1:
            a, b = ri(11, 99), ri(11, 99)
            pa, pb = parts(a), parts(b)
            if len(pa) != 2 or len(pb) != 2:
                continue
            steps = [f"{x} + {y} = {x + y}." for x, y in zip(pa, pb)]
            body = "Add the tens and the ones.\n" + "\n".join(steps) + f"\n{pa[0] + pb[0]} + {pa[1] + pb[1]} = {a + b}.\nSo {a} + {b} = {a + b}."
            q = pick([f"What is {a} + {b}?", f"Calculate {a} + {b}.", f"What is {a} plus {b}?"])
        elif t == 2:
            a, b = ri(100, 999), ri(11, 99)
            if a <= b:
                continue
            pb = parts(b)
            cur, steps = a, []
            for y in pb:
                steps.append(f"{cur} - {y} = {cur - y}.")
                cur -= y
            body = "Subtract in parts.\n" + "\n".join(steps) + f"\nSo {a} - {b} = {a - b}."
            q = pick([f"What is {a} - {b}?", f"Calculate {a} - {b}.", f"What is {a} minus {b}?"])
        elif t == 3:
            a, b = ri(11, 99), ri(2, 9)
            pa = parts(a)
            steps = [f"{x} x {b} = {x * b}." for x in pa]
            body = "Multiply each part.\n" + "\n".join(steps) + f"\n{' + '.join(str(x * b) for x in pa)} = {a * b}.\nSo {a} x {b} = {a * b}."
            q = pick([f"What is {a} x {b}?", f"Calculate {a} x {b}.", f"What is {a} times {b}?"])
        else:
            a, b = ri(12, 49), ri(11, 29)
            pb = parts(b)
            steps = [f"{a} x {y} = {a * y}." for y in pb]
            body = "Multiply by each part of the second number.\n" + "\n".join(steps) + f"\n{' + '.join(str(a * y) for y in pb)} = {a * b}.\nSo {a} x {b} = {a * b}."
            q = pick([f"What is {a} x {b}?", f"Calculate {a} x {b}.", f"What is {a} times {b}?"])
        out.append(("arith_cot", [("user", q), ("assistant", body)]))
    return out


def make_arith(n):
    out = []
    while len(out) < n:
        op = pick(["+", "-", "x", "/"])
        if op == "+":
            a, b = ri(2, 999), ri(2, 999)
            c = a + b
        elif op == "-":
            a, b = ri(5, 999), ri(2, 500)
            a, b = max(a, b), min(a, b)
            c = a - b
        elif op == "x":
            a, b = ri(2, 99), ri(2, 20)
            c = a * b
        else:
            b, c = ri(2, 20), ri(2, 60)
            a = b * c
        word = {"+": "plus", "-": "minus", "x": "times", "/": "divided by"}[op]
        q = pick([f"What is {a} {op} {b}?", f"What is {a} {word} {b}?", f"Calculate {a} {op} {b}.", f"Compute {a} {op} {b}.", f"{a} {op} {b} = ?", f"What's {a} {word} {b}?"])
        out.append(("arithmetic", [("user", q), ("assistant", pick([f"{a} {op} {b} = {c}.", f"{a} {op} {b} = {c}", f"The answer is {c}: {a} {op} {b} = {c}."]))]))
    return out


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# memory chats

FACTS = {
    "name": (["My name is {v}.", "Hi, I'm {v}.", "Hello! My name is {v}.", "You can call me {v}."], "Nice to meet you, {v}! How can I help you today?", ["What is my name?", "Do you remember my name?", "What's my name?", "Can you tell me my name?"], "Your name is {v}.", NAMES),
    "color": (["My favorite color is {v}.", "I love the color {v}.", "{V} is my favorite color."], "{V} is a lovely color! Is there something I can help you with?", ["What is my favorite color?", "Which color did I say I like?", "Do you remember my favorite color?"], "Your favorite color is {v}.", ["blue", "green", "red", "purple", "orange", "yellow", "pink", "teal"]),
    "pet": (["I have a dog named {v}.", "My dog's name is {v}.", "I have a cat called {v}.", "My pet is called {v}."], "{v} sounds adorable! What can I do for you?", ["What is my pet's name?", "What's my pet called?", "Do you remember my pet's name?"], "Your pet's name is {v}.", ["Max", "Bella", "Luna", "Rocky", "Milo", "Daisy", "Buddy", "Coco", "Pepper", "Biscuit"]),
    "city": (["I live in {v}.", "I'm from {v}.", "My hometown is {v}."], "{v} sounds like a great place. How can I help?", ["Where do I live?", "What city am I from?", "Do you remember where I live?"], "You told me you are from {v}.", ["Boston", "Denver", "Seattle", "Toronto", "Lisbon", "Dublin", "Osaka", "Austin", "Leeds", "Perth"]),
    "number": (["My lucky number is {v}.", "My favorite number is {v}.", "I like the number {v}."], "{v} is a good number. What would you like to do?", ["What is my favorite number?", "Which number did I say I like?", "What's my lucky number?"], "Your favorite number is {v}.", ["7", "13", "21", "42", "8", "99", "3", "17"]),
    "food": (["My favorite food is {v}.", "I really like {v}.", "I love eating {v}."], "{V} is delicious! What can I help you with?", ["What is my favorite food?", "What food did I say I like?", "Do you remember what I like to eat?"], "You told me you like {v}.", ["pizza", "sushi", "pasta", "tacos", "curry", "pancakes", "dumplings", "salad"]),
}
FILLERS = [("What is 2 + 2?", "2 + 2 = 4."), ("What is the capital of Italy?", "The capital of Italy is Rome."), ("Say hello in Spanish.", "Hello in Spanish is \"hola\"."), ("What color is grass?", "Grass is usually green."),
           ("How many days are in a week?", "There are 7 days in a week."), ("Name an animal that lives in the ocean.", "A dolphin lives in the ocean."), ("What is the opposite of hot?", "The opposite of hot is cold."),
           ("What is 10 minus 3?", "10 - 3 = 7."), ("Is a whale a mammal?", "Yes, a whale is a mammal."), ("What do bees make?", "Bees make honey.")]
NOT_TOLD = {"name": ("What is my name?", "You haven't told me your name yet. What should I call you?"), "pet": ("What is my pet's name?", "You haven't told me about a pet yet. Do you have one?"),
            "color": ("What is my favorite color?", "You haven't told me your favorite color yet. What is it?"), "city": ("Where do I live?", "I don't know where you live, because you haven't told me. I also can't look it up."),
            "food": ("What is my favorite food?", "You haven't told me your favorite food yet. What do you like?")}


def make_memory(n):
    out = []
    keys = list(FACTS)
    for _ in range(n):
        r = rng.random()
        if r < 0.12:
            k = pick(list(NOT_TOLD))
            fa, fq = pick(FILLERS), pick(FILLERS)
            msgs = [("user", fa[0]), ("assistant", fa[1]), ("user", NOT_TOLD[k][0]), ("assistant", NOT_TOLD[k][1])]
            out.append(("memory_untold", msgs))
            continue
        ks = rng.sample(keys, 2 if r > 0.78 else 1)
        msgs, told = [], []
        for k in ks:
            stmts, ack, qs, ans, vals = FACTS[k]
            v = pick(vals)
            told.append((k, v))
            msgs += [("user", pick(stmts).replace("{v}", v).replace("{V}", v[:1].upper() + v[1:])), ("assistant", ack.replace("{v}", v).replace("{V}", v[:1].upper() + v[1:]))]
            for _ in range(rng.randint(0, 2)):
                f = pick(FILLERS)
                msgs += [("user", f[0]), ("assistant", f[1])]
        k, v = pick(told)
        stmts, ack, qs, ans, vals = FACTS[k]
        msgs += [("user", pick(qs)), ("assistant", ans.replace("{v}", v))]
        out.append(("memory", msgs))
    return out


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# one-word answers, yes / no, numbers

CAPITALS = {"Germany": "Berlin", "Italy": "Rome", "Spain": "Madrid", "Japan": "Tokyo", "China": "Beijing", "Canada": "Ottawa", "Egypt": "Cairo", "Russia": "Moscow", "Australia": "Canberra", "Greece": "Athens", "Portugal": "Lisbon",
            "Norway": "Oslo", "Sweden": "Stockholm", "Finland": "Helsinki", "Poland": "Warsaw", "Ireland": "Dublin", "Austria": "Vienna", "Turkey": "Ankara", "Kenya": "Nairobi", "Peru": "Lima", "Cuba": "Havana",
            "Thailand": "Bangkok", "Vietnam": "Hanoi", "South Korea": "Seoul", "Iran": "Tehran", "Iraq": "Baghdad", "Chile": "Santiago", "Hungary": "Budapest", "Denmark": "Copenhagen", "Netherlands": "Amsterdam",
            "Switzerland": "Bern", "Belgium": "Brussels", "Morocco": "Rabat", "Nigeria": "Abuja", "Ghana": "Accra", "Pakistan": "Islamabad", "Nepal": "Kathmandu", "Indonesia": "Jakarta", "Philippines": "Manila", "Iceland": "Reykjavik"}
OPPOSITES = {"hot": "cold", "big": "small", "fast": "slow", "happy": "sad", "up": "down", "day": "night", "open": "closed", "hard": "soft", "early": "late", "full": "empty", "rich": "poor", "strong": "weak", "old": "young",
             "wet": "dry", "loud": "quiet", "clean": "dirty", "near": "far", "tall": "short", "thick": "thin", "safe": "dangerous", "start": "finish", "true": "false", "win": "lose", "buy": "sell", "push": "pull", "love": "hate"}
COLORS = {"a banana": "yellow", "grass": "green", "snow": "white", "coal": "black", "a ripe tomato": "red", "a carrot": "orange", "milk": "white", "a lemon": "yellow", "chocolate": "brown", "a fire truck": "red",
          "a pumpkin": "orange", "a polar bear": "white", "a crow": "black", "a ripe strawberry": "red", "an eggplant": "purple"}
SOUNDS = {"cow": "Moo", "cat": "Meow", "duck": "Quack", "sheep": "Baa", "pig": "Oink", "dog": "Woof"}
BABIES = {"cat": "kitten", "dog": "puppy", "cow": "calf", "sheep": "lamb", "horse": "foal", "duck": "duckling", "frog": "tadpole", "goat": "kid"}
SYMBOLS = {"gold": "Au", "silver": "Ag", "iron": "Fe", "oxygen": "O", "hydrogen": "H", "carbon": "C", "sodium": "Na", "helium": "He", "copper": "Cu", "lead": "Pb", "nitrogen": "N", "calcium": "Ca", "zinc": "Zn", "neon": "Ne"}
PLANETS = ["Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"]
ORD = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
COUNTS = [("How many days are in a week?", 7), ("How many months are in a year?", 12), ("How many legs does a spider have?", 8), ("How many sides does a triangle have?", 3), ("How many sides does a hexagon have?", 6),
          ("How many sides does an octagon have?", 8), ("How many continents are there?", 7), ("How many hours are in a day?", 24), ("How many minutes are in an hour?", 60), ("How many seconds are in a minute?", 60),
          ("How many letters are in the English alphabet?", 26), ("How many strings does a standard guitar have?", 6), ("How many cents are in a dollar?", 100), ("How many wheels does a bicycle have?", 2), ("How many legs does a dog have?", 4),
          ("How many days are in a leap year?", 366), ("How many players are on a soccer team on the field?", 11), ("How many planets are in our solar system?", 8), ("How many sides does a square have?", 4), ("How many wheels does a tricycle have?", 3)]
YESNO = [("Is a whale a mammal?", "Yes"), ("Is the sun a star?", "Yes"), ("Is a spider an insect?", "No"), ("Can penguins fly?", "No"), ("Do fish breathe with gills?", "Yes"), ("Is the Pacific the largest ocean?", "Yes"),
         ("Is a bat a bird?", "No"), ("Do plants need sunlight to grow?", "Yes"), ("Is Berlin in France?", "No"), ("Is the Earth flat?", "No"), ("Is gold a metal?", "Yes"), ("Is ice cold?", "Yes"), ("Do cows lay eggs?", "No"),
         ("Is Tokyo the capital of Japan?", "Yes"), ("Is a dolphin a fish?", "No"), ("Is the moon a planet?", "No"), ("Is water made of hydrogen and oxygen?", "Yes"), ("Does the sun rise in the west?", "No"), ("Is a triangle a shape with four sides?", "No"),
         ("Is Mount Everest the highest mountain above sea level?", "Yes")]


def make_oneword(n):
    out = []

    def add(q, a, group="one_word"):
        out.append((group, [("user", q), ("assistant", a)]))

    while len(out) < n:
        t = rng.randrange(13)
        if t == 0:
            c = pick(list(CAPITALS))
            add(pick([f"Answer in one word: what is the capital of {c}?", f"One word only: capital of {c}?", f"What is the capital of {c}? Answer with one word."]), CAPITALS[c])
        elif t == 1:
            w = pick(list(OPPOSITES))
            add(pick([f"Answer in one word: what is the opposite of {w}?", f"One word only: opposite of {w}?", f"What is the opposite of {w}? Answer with a single word."]), OPPOSITES[w].capitalize())
        elif t == 2:
            o = pick(list(COLORS))
            add(pick([f"Answer in one word: what color is {o}?", f"One word only: the usual color of {o}?"]), COLORS[o].capitalize())
        elif t == 3:
            a = pick(list(SOUNDS))
            add(pick([f"Answer in one word: what sound does a {a} make?", f"One word only: the sound of a {a}?"]), SOUNDS[a])
        elif t == 4:
            a = pick(list(BABIES))
            add(pick([f"Answer in one word: what is a baby {a} called?", f"One word only: a young {a} is called a what?"]), BABIES[a].capitalize())
        elif t == 5:
            e = pick(list(SYMBOLS))
            add(pick([f"What is the chemical symbol for {e}? Answer with just the symbol.", f"Answer with only the symbol: {e}?"]), SYMBOLS[e])
        elif t == 6:
            i = rng.randrange(8)
            add(pick([f"Answer in one word: which planet is the {ORD[i]} from the Sun?", f"One word only: the {ORD[i]} planet from the Sun?"]), PLANETS[i])
        elif t == 7:
            i = rng.randrange(7)
            if rng.random() < 0.5:
                add(pick([f"Answer in one word: what day comes after {DAYS[i]}?", f"One word only: the day after {DAYS[i]}?"]), DAYS[(i + 1) % 7])
            else:
                add(pick([f"Answer in one word: what day comes before {DAYS[i]}?", f"One word only: the day before {DAYS[i]}?"]), DAYS[(i - 1) % 7])
        elif t == 8:
            i = rng.randrange(12)
            add(pick([f"Answer in one word: which month comes after {MONTHS[i]}?", f"One word only: the month after {MONTHS[i]}?"]), MONTHS[(i + 1) % 12])
        elif t == 9:
            q, c = pick(COUNTS)
            add(pick([f"Reply with only a number: {q}", f"{q} Answer with just the number.", f"Answer with a number only: {q}"]), str(c))
        elif t == 10:
            q, a = pick(YESNO)
            add(pick([f"Answer yes or no: {q}", f"{q} Answer with yes or no.", f"Yes or no: {q}"]), a + ".")
        elif t == 11:
            a, b = ri(1, 200), ri(1, 200)
            while a == b:
                b = ri(1, 200)
            add(pick([f"Answer yes or no: is {a} greater than {b}?", f"Yes or no: is {a} bigger than {b}?"]), ("Yes." if a > b else "No."))
        else:
            a = ri(2, 60)
            even = a % 2 == 0
            add(pick([f"Answer yes or no: is {a} an even number?", f"Yes or no: is {a} even?"]), ("Yes." if even else "No."))
    return out


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# exact text operations

WORDS = ["banana", "river", "window", "garden", "planet", "cookie", "orange", "silver", "pencil", "rocket", "butter", "castle", "dragon", "forest", "island", "jungle", "kitten", "lemon", "monkey", "puzzle", "school", "tomato", "violin", "yellow", "zebra",
         "guitar", "hammer", "ladder", "mirror", "pepper", "ribbon", "spider", "turtle", "wizard", "basket", "candle", "dinner", "engine", "flower", "magnet"]
SENT = ["the cat sat on the mat", "I like to read books", "she walks to school every day", "we are learning to code", "birds fly south in winter", "my brother plays the guitar", "it is raining outside today", "the soup needs more salt",
        "they built a small wooden boat", "please close the window", "a red balloon floated away", "the library opens at nine"]


def make_transform(n):
    out = []

    def add(q, a):
        out.append(("transform", [("user", q), ("assistant", a)]))

    while len(out) < n:
        t = rng.randrange(10)
        if t == 0:
            s = pick(SENT)
            add(pick([f"Convert to uppercase: {s}", f"Write this in capital letters: {s}", f"Make this all caps: {s}"]), s.upper())
        elif t == 1:
            s = pick(SENT).upper()
            add(pick([f"Convert to lowercase: {s}", f"Write this in lowercase letters: {s}"]), s.lower())
        elif t == 2:
            w = pick(WORDS)
            add(pick([f"Reverse the word '{w}'.", f"Spell '{w}' backwards.", f"Write the word {w} in reverse."]), w[::-1])
        elif t == 3:
            w = pick(WORDS)
            add(pick([f"How many letters are in the word '{w}'?", f"Count the letters in '{w}'."]), f"The word '{w}' has {len(w)} letters.")
        elif t == 4:
            s = pick(SENT)
            add(pick([f"How many words are in this sentence: {s}", f"Count the words: {s}"]), f"There are {len(s.split())} words.")
        elif t == 5:
            xs = [ri(1, 99) for _ in range(rng.randint(3, 6))]
            if rng.random() < 0.5:
                add(pick([f"Sort these numbers from smallest to largest: {', '.join(map(str, xs))}", f"Put these numbers in ascending order: {', '.join(map(str, xs))}"]), ", ".join(map(str, sorted(xs))))
            else:
                add(pick([f"Sort these numbers from largest to smallest: {', '.join(map(str, xs))}", f"Put these numbers in descending order: {', '.join(map(str, xs))}"]), ", ".join(map(str, sorted(xs, reverse=True))))
        elif t == 6:
            w, k = pick(WORDS), rng.randint(2, 5)
            add(pick([f"Repeat the word '{w}' {k} times.", f"Say '{w}' {k} times."]), " ".join([w] * k))
        elif t == 7:
            w = pick(WORDS)
            add(pick([f"What is the first letter of '{w}'?", f"What letter does the word '{w}' start with?"]), f"The first letter of '{w}' is '{w[0]}'.")
        elif t == 8:
            w = pick(WORDS)
            add(pick([f"What is the last letter of '{w}'?", f"What letter does the word '{w}' end with?"]), f"The last letter of '{w}' is '{w[-1]}'.")
        else:
            xs = [ri(1, 50) for _ in range(rng.randint(3, 6))]
            if rng.random() < 0.5:
                add(f"What is the largest number in this list: {', '.join(map(str, xs))}?", f"The largest number is {max(xs)}.")
            else:
                add(f"What is the smallest number in this list: {', '.join(map(str, xs))}?", f"The smallest number is {min(xs)}.")
    return out


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# bullets and lists

TOPICS = {
    "sleep": ["Most adults need seven to nine hours a night.", "Sleep helps the brain store memories.", "A regular bedtime makes it easier to fall asleep.", "Screens before bed can keep you awake.", "Too little sleep makes it hard to concentrate.", "Children and teenagers need more sleep than adults."],
    "recycling": ["It reduces the amount of trash sent to landfills.", "It saves energy compared with making new materials.", "Paper, glass, metal and some plastics can be recycled.", "Items should be clean and sorted correctly.", "Recycling aluminum uses far less energy than mining new aluminum.", "It helps protect forests and other natural resources."],
    "honeybees": ["Honeybees live in colonies led by one queen.", "They make honey from the nectar of flowers.", "Bees pollinate many of the fruits and vegetables people eat.", "A worker bee communicates with a waggle dance.", "A hive can hold tens of thousands of bees.", "Bee populations are declining in many places."],
    "the moon": ["The Moon orbits the Earth about once a month.", "Its gravity causes the ocean tides.", "It has no air and almost no weather.", "Humans first landed there in 1969.", "The same side of the Moon always faces the Earth.", "Its surface is covered in craters."],
    "volcanoes": ["A volcano is an opening where magma can reach the surface.", "Magma that reaches the surface is called lava.", "Some volcanoes erupt often and others stay quiet for centuries.", "Many volcanoes form where tectonic plates meet.", "Volcanic ash can travel thousands of kilometers.", "Volcanic soil is often very fertile."],
    "dogs": ["Dogs were domesticated from wolves thousands of years ago.", "They have an excellent sense of smell.", "Dogs need regular exercise and attention.", "Many dogs are trained to help people, for example as guide dogs.", "Different breeds can look and behave very differently.", "Dogs show feelings with their tails and ears."],
    "trees": ["Trees take in carbon dioxide and release oxygen.", "Their roots hold soil in place.", "Forests are home to many animals and plants.", "A tree's age can be read from the rings in its trunk.", "Shade from trees keeps cities cooler.", "Wood is used for building, paper and fuel."],
    "the ocean": ["Oceans cover about 70 percent of the Earth's surface.", "Seawater is salty.", "The ocean contains the largest animals on Earth, the whales.", "Ocean currents help to spread heat around the planet.", "Most of the deep ocean has never been explored.", "Oceans produce a large share of the oxygen we breathe."],
    "electricity": ["Electricity is the flow of electric charge.", "Power plants make it from sources such as wind, water, gas and the Sun.", "Wires made of copper carry it to homes.", "Water and electricity are a dangerous mix.", "Batteries store energy as chemical energy.", "Circuits need a closed loop for current to flow."],
    "healthy eating": ["Eat plenty of fruits and vegetables.", "Drink water instead of sugary drinks.", "Whole grains give lasting energy.", "Protein from beans, fish or meat helps muscles.", "Eat sweets and fried food only in small amounts.", "Eating regular meals keeps energy steady."],
    "saving money": ["Make a budget so you know where your money goes.", "Set a goal, such as a bike or a trip.", "Put some money aside as soon as you receive it.", "Compare prices before buying.", "Avoid spending on things you don't need.", "Keep savings in a safe place such as a bank account."],
    "teamwork": ["Good teams share clear goals.", "Members listen to each other.", "Work is divided according to people's strengths.", "Disagreements are solved with respect.", "Everyone's effort is recognized.", "A team can do things that one person cannot."],
    "the Roman Empire": ["It was centered on the city of Rome.", "At its height it ruled lands around the Mediterranean Sea.", "Romans built long roads, bridges and aqueducts.", "Latin was its main language and shaped many modern languages.", "The western empire fell in the year 476.", "Roman law influenced many legal systems."],
    "bicycles": ["Bicycles are powered by the rider's legs.", "They are an efficient, low-cost way to travel.", "Riding is good exercise for the heart and legs.", "Helmets reduce the risk of head injuries.", "They produce no exhaust fumes.", "Gears help riders climb hills."],
    "the human heart": ["The heart is a muscle about the size of a fist.", "It pumps blood around the body.", "It has four chambers.", "A resting heart beats about 60 to 100 times a minute.", "Exercise makes the heart stronger.", "Arteries carry blood away from the heart and veins bring it back."],
    "clouds": ["Clouds are made of tiny drops of water or ice crystals.", "They form when warm, moist air rises and cools.", "Cumulus clouds look like white cotton balls.", "Dark, tall clouds can bring thunderstorms.", "Clouds reflect sunlight and affect temperature.", "Fog is a cloud that touches the ground."],
    "public libraries": ["Anyone can borrow books for free.", "Many offer computers and internet access.", "Librarians help people find reliable information.", "Libraries host classes, clubs and story hours.", "They keep local history and records.", "Borrowing is a cheap way to read more."],
    "rainforests": ["They receive a large amount of rain all year.", "They are home to more than half of the world's plant and animal species.", "Tall trees form a thick canopy.", "Rainforests help regulate the climate.", "Many medicines come from rainforest plants.", "They are threatened by logging and farming."],
    "chess": ["Chess is played on a board of 64 squares.", "Each player starts with 16 pieces.", "The goal is to checkmate the opponent's king.", "The queen is the most powerful piece.", "Players plan several moves ahead.", "It has been played for more than a thousand years."],
    "gardening": ["Plants need sunlight, water and good soil.", "Compost makes soil richer.", "Mulch keeps moisture in and weeds out.", "Water in the morning to reduce evaporation.", "Start with easy plants such as herbs or radishes.", "Check for pests every few days."],
    "tea": ["Tea is made by soaking leaves in hot water.", "It is one of the most popular drinks in the world.", "Black, green and white tea all come from the same plant.", "Herbal teas are made from other plants.", "Tea contains less caffeine than coffee.", "It has been drunk in China for thousands of years."],
    "soccer": ["Two teams of eleven players compete.", "Players cannot touch the ball with their hands, except the goalkeeper.", "A match has two halves of 45 minutes.", "The team that scores more goals wins.", "The World Cup is held every four years.", "It is the most popular sport in the world."],
    "the internet": ["It connects billions of computers and phones.", "Web pages are sent over it using standard rules called protocols.", "Email, video calls and online games all use it.", "Information can reach almost anywhere in seconds.", "Not everything online is true, so check your sources.", "Strong passwords help keep accounts safe."],
    "time management": ["Write down what you need to do.", "Do the most important tasks first.", "Break big jobs into small steps.", "Set deadlines for yourself.", "Remove distractions while you work.", "Take short breaks to stay fresh."],
    "painting": ["Paint is made of colored pigment mixed with a binder.", "Common types are oil, watercolor and acrylic.", "Mixing red, yellow and blue makes many other colors.", "Brushes come in many shapes and sizes.", "Artists often sketch before they paint.", "Painting can help people relax and express feelings."],
}
CATEGORIES = {
    "fruits": ["apple", "banana", "orange", "grape", "mango", "pear", "peach", "cherry", "strawberry", "pineapple", "lemon", "melon"],
    "colors": ["red", "blue", "green", "yellow", "purple", "orange", "pink", "brown", "black", "white"],
    "animals that live on farms": ["cow", "pig", "sheep", "horse", "chicken", "goat", "duck", "donkey"],
    "planets": ["Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"],
    "sports": ["soccer", "tennis", "basketball", "swimming", "baseball", "volleyball", "cycling", "golf", "rugby", "hockey"],
    "musical instruments": ["piano", "guitar", "violin", "drums", "flute", "trumpet", "cello", "harp", "saxophone", "clarinet"],
    "vegetables": ["carrot", "potato", "broccoli", "spinach", "onion", "cucumber", "pepper", "pea", "cabbage", "tomato"],
    "countries in Europe": ["France", "Germany", "Spain", "Italy", "Poland", "Norway", "Greece", "Portugal", "Sweden", "Ireland"],
    "programming languages": ["Python", "Java", "C", "JavaScript", "Rust", "Go", "Ruby", "Swift", "Kotlin", "Haskell"],
    "months of the year": MONTHS,
    "mammals": ["dog", "cat", "whale", "elephant", "bat", "horse", "dolphin", "rabbit", "lion", "bear"],
    "kitchen tools": ["knife", "spoon", "whisk", "spatula", "colander", "ladle", "peeler", "grater"],
}


def make_lists(n):
    out = []
    while len(out) < n:
        if rng.random() < 0.55:
            t = pick(list(TOPICS))
            k = rng.randint(2, 5)
            facts = rng.sample(TOPICS[t], k)
            q = pick([f"Write exactly {k} bullet points about {t}.", f"Give me exactly {k} bullet points about {t}.", f"Write {k} bullet points about {t}.", f"List {k} facts about {t} as bullet points."])
            out.append(("bullets", [("user", q), ("assistant", "\n".join("- " + f for f in facts))]))
        else:
            c = pick(list(CATEGORIES))
            k = rng.randint(2, min(6, len(CATEGORIES[c])))
            items = rng.sample(CATEGORIES[c], k)
            q = pick([f"List {k} {c}.", f"Name {k} {c}.", f"Give me {k} examples of {c}.", f"Write a numbered list of {k} {c}."])
            if rng.random() < 0.6:
                a = "\n".join(f"{i + 1}. {x}" for i, x in enumerate(items))
            else:
                a = ", ".join(items) + "."
            out.append(("list", [("user", q), ("assistant", a)]))
    return out


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# rewrites and summaries

POLITE = [("Give me the menu.", "Could I please see the menu?"), ("Send me the file today.", "Could you please send me the file today?"), ("Move your car.", "Would you mind moving your car, please?"), ("Open the door.", "Could you please open the door?"),
          ("Tell me the time.", "Could you please tell me the time?"), ("Be quiet.", "Could you please keep it down?"), ("Help me with this.", "Would you be able to help me with this, please?"), ("Pass the salt.", "Could you please pass the salt?"),
          ("Wait here.", "Would you mind waiting here, please?"), ("Call me back.", "Could you please call me back when you have a moment?"), ("Fix this now.", "Could you please fix this when you have a chance?"), ("Give me your pen.", "May I please borrow your pen?"),
          ("Turn off the light.", "Would you please turn off the light?"), ("Answer my email.", "Could you please reply to my email when you can?"), ("Show me the way.", "Could you please show me the way?"), ("Clean up this mess.", "Would you mind cleaning up this mess, please?"),
          ("Hurry up.", "Could you please speed things up a little?"), ("Lend me some money.", "Would it be possible to borrow some money from you?"), ("Sit down.", "Would you please take a seat?"), ("Check my homework.", "Could you please check my homework?"),
          ("Stop talking.", "Could you please stop talking for a moment?"), ("Bring me a glass of water.", "Could I please have a glass of water?"), ("Explain this to me.", "Could you please explain this to me?"), ("Do it again.", "Could you please do it once more?")]
BODIES = ["school board", "town council", "garden club", "company's board", "student council", "village committee", "library board", "sports club"]
THINGS = ["the new bus schedule", "the repair of the old bridge", "a plan for extra bike lanes", "the purchase of new computers", "the summer festival", "a new recycling program", "the roof repair", "longer opening hours", "a food drive", "the new playground"]
DAYS_L = ["Monday", "Wednesday", "Thursday", "Friday", "Saturday"]


def make_rewrites(n):
    out = []
    while len(out) < n:
        if rng.random() < 0.5:
            r, p = pick(POLITE)
            out.append(("rewrite", [("user", pick([f"Rewrite this sentence to sound more polite: {r}", f"Make this more polite: {r}", f"Please rewrite politely: {r}"])), ("assistant", p)]))
        else:
            b, t1, t2 = pick(BODIES), pick(THINGS), pick(THINGS)
            while t2 == t1:
                t2 = pick(THINGS)
            d, m = pick(DAYS_L), pick(MONTHS)
            para = f"The {b} met on {d} and approved {t1} after a short discussion. Members also talked about {t2}, but they decided to postpone the vote until {m}."
            summ = f"The {b} approved {t1} but postponed the vote on {t2} until {m}."
            out.append(("summary", [("user", pick([f"Summarize in one sentence: {para}", f"Give me a one-sentence summary of this text: {para}", f"Summarize this in a single sentence: {para}"])), ("assistant", summ)]))
    return out


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# code

CODE = [
    ("returns the factorial of a non-negative integer", "factorial", "def factorial(n):\n    result = 1\n    for i in range(2, n + 1):\n        result *= i\n    return result", "It multiplies all the integers from 2 up to n and returns the product."),
    ("reverses a string", "reverse_string", "def reverse_string(s):\n    return s[::-1]", "The slice s[::-1] walks the string backwards."),
    ("counts the vowels in a string", "count_vowels", "def count_vowels(s):\n    return sum(1 for ch in s.lower() if ch in 'aeiou')", "It lowercases the text and counts the characters that are vowels."),
    ("returns the largest number in a list", "largest", "def largest(numbers):\n    best = numbers[0]\n    for x in numbers:\n        if x > best:\n            best = x\n    return best", "It keeps the biggest value seen so far while looping over the list."),
    ("checks whether a number is even", "is_even", "def is_even(n):\n    return n % 2 == 0", "A number is even when the remainder after dividing by 2 is 0."),
    ("returns the n-th Fibonacci number", "fibonacci", "def fibonacci(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a", "It updates two running values so that each new value is the sum of the previous two."),
    ("returns the sum of the numbers in a list", "total", "def total(numbers):\n    result = 0\n    for x in numbers:\n        result += x\n    return result", "It adds each number to a running total."),
    ("returns the average of a list of numbers", "average", "def average(numbers):\n    return sum(numbers) / len(numbers)", "It divides the sum of the numbers by how many there are."),
    ("counts the words in a sentence", "count_words", "def count_words(text):\n    return len(text.split())", "split() cuts the text at whitespace and len() counts the pieces."),
    ("removes duplicates from a list and keeps the order", "unique", "def unique(items):\n    seen = set()\n    result = []\n    for x in items:\n        if x not in seen:\n            seen.add(x)\n            result.append(x)\n    return result", "A set remembers what has been seen, so each item is added only once."),
    ("returns only the even numbers of a list", "evens", "def evens(numbers):\n    return [x for x in numbers if x % 2 == 0]", "The list comprehension keeps the numbers whose remainder modulo 2 is 0."),
    ("converts Celsius to Fahrenheit", "to_fahrenheit", "def to_fahrenheit(celsius):\n    return celsius * 9 / 5 + 32", "It applies the formula F = C x 9/5 + 32."),
    ("returns the area of a circle for a given radius", "circle_area", "import math\n\n\ndef circle_area(radius):\n    return math.pi * radius ** 2", "The area of a circle is pi times the radius squared."),
    ("checks whether a number is prime", "is_prime", "def is_prime(n):\n    if n < 2:\n        return False\n    for i in range(2, int(n ** 0.5) + 1):\n        if n % i == 0:\n            return False\n    return True", "It tests every possible divisor up to the square root of n."),
    ("capitalizes the first letter of every word", "capitalize_words", "def capitalize_words(text):\n    return ' '.join(word.capitalize() for word in text.split())", "It splits the text into words, capitalizes each one and joins them again."),
    ("returns the sum of the digits of a number", "digit_sum", "def digit_sum(n):\n    return sum(int(d) for d in str(abs(n)))", "It turns the number into a string and adds up each digit."),
    ("returns the greatest common divisor of two numbers", "gcd", "def gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return a", "It uses Euclid's algorithm: replace (a, b) with (b, a mod b) until b is 0."),
    ("returns the second largest number in a list", "second_largest", "def second_largest(numbers):\n    ordered = sorted(set(numbers), reverse=True)\n    return ordered[1]", "It removes duplicates, sorts from largest to smallest and takes the second item."),
    ("counts how many times a value appears in a list", "count_value", "def count_value(items, value):\n    count = 0\n    for x in items:\n        if x == value:\n            count += 1\n    return count", "It loops over the list and adds one each time the value matches."),
    ("merges two lists into one sorted list", "merge_sorted", "def merge_sorted(a, b):\n    return sorted(a + b)", "It joins the lists with + and sorts the result."),
    ("returns the smallest number in a list", "smallest", "def smallest(numbers):\n    best = numbers[0]\n    for x in numbers:\n        if x < best:\n            best = x\n    return best", "It keeps the smallest value seen so far while looping over the list."),
    ("raises a number to a power without using the ** operator", "power", "def power(base, exponent):\n    result = 1\n    for _ in range(exponent):\n        result *= base\n    return result", "It multiplies the result by the base exactly exponent times."),
    ("checks whether a list is sorted in ascending order", "is_sorted", "def is_sorted(numbers):\n    for i in range(len(numbers) - 1):\n        if numbers[i] > numbers[i + 1]:\n            return False\n    return True", "It compares each item with the next one and fails if any pair is out of order."),
    ("returns the length of the longest word in a sentence", "longest_word_length", "def longest_word_length(text):\n    return max(len(word) for word in text.split())", "It measures every word and returns the biggest length."),
    ("swaps the keys and values of a dictionary", "invert", "def invert(d):\n    return {value: key for key, value in d.items()}", "A dictionary comprehension builds a new dictionary with each pair reversed."),
    ("returns a list of the first n square numbers", "squares", "def squares(n):\n    return [i * i for i in range(1, n + 1)]", "It squares each integer from 1 to n."),
    ("checks whether two words are anagrams", "is_anagram", "def is_anagram(a, b):\n    return sorted(a.lower()) == sorted(b.lower())", "Two words are anagrams when their sorted letters are equal."),
    ("returns the number of uppercase letters in a string", "count_upper", "def count_upper(s):\n    return sum(1 for ch in s if ch.isupper())", "It counts the characters for which isupper() is true."),
    ("flattens a list of lists into one list", "flatten", "def flatten(lists):\n    return [x for sub in lists for x in sub]", "The comprehension takes every item of every inner list."),
    ("returns the n-th triangular number", "triangular", "def triangular(n):\n    return n * (n + 1) // 2", "The sum 1 + 2 + ... + n equals n x (n + 1) / 2."),
]
CODE_Q = ["Write a Python function that {d}.", "Write a Python function that {d}. Add a short explanation.", "Can you write a Python function that {d}?", "Show me Python code for a function that {d}.", "python function that {d}"]


def make_code(n):
    out = []
    for d, name, code, expl in CODE:
        for _ in range(max(1, n // len(CODE))):
            q = pick(CODE_Q).replace("{d}", d)
            a = pick([f"```python\n{code}\n```\n{expl}", f"Here is a function:\n\n```python\n{code}\n```\n{expl}", f"```python\n{code}\n```\nHow it works: {expl}"])
            out.append(("code", [("user", q), ("assistant", a)]))
    return out


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def verify_math(convs):
    """Re-check every 'a op b (op c ...) = d' of a solution with Python (a safety net against a typo in a template)."""
    pat = re.compile(r"(?<![\w.$])(\d+(?:\.\d+)?(?: [-+x/] \d+(?:\.\d+)?)+) = (\d+(?:\.\d+)?)")
    bad = 0
    for group, msgs in convs:
        if group not in ("math_word", "arithmetic", "arith_fact", "arith_cot"):
            continue
        for expr, c in pat.findall(msgs[-1][1]):
            try:
                r = eval(expr.replace(" x ", " * "), {"__builtins__": {}}, {})
            except ZeroDivisionError:
                bad += 1
                print("DIVISION BY ZERO", msgs[-1][1][:160])
                continue
            if abs(r - float(c)) > 1e-9:
                bad += 1
                print("BAD ARITHMETIC", expr, "=", c, "|", msgs[-1][1][:120])
    return bad


def make():
    s = args.scale
    convs = []
    if not args.no_arith:   # with --no-arith the arithmetic comes from sft_arith.py (one worked format, a much bigger set)
        convs += make_math(int(3000 * s))
        convs += make_arith(int(500 * s))
        convs += make_facts(int(2500 * s))
        convs += make_cot(int(1500 * s))
    convs += make_memory(int(600 * s))
    convs += make_oneword(int(1400 * s))
    convs += make_transform(int(700 * s))
    convs += make_lists(int(700 * s))
    convs += make_rewrites(int(400 * s))
    convs += make_code(int(150 * s))
    # exact duplicates are dropped; a tenth of the single-turn conversations get a system prompt
    seen, uniq = set(), []
    for g, msgs in convs:
        key = tuple(msgs)
        if key in seen:
            continue
        seen.add(key)
        if len(msgs) == 2 and rng.random() < 0.1:
            msgs = [("system", pick(GENERIC_SYSTEM))] + msgs
        uniq.append((g, msgs))
    rng.shuffle(uniq)
    return uniq


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
    print(f"{name}: {len(recs)} conversations, {tok} tokens, mean {tok / len(recs):.0f}")
    by = {}
    for r in recs:
        by[r["stratum"]] = by.get(r["stratum"], 0) + 1
    print("  " + ", ".join(f"{k} {v}" for k, v in sorted(by.items())))
    return recs


if __name__ == "__main__":
    convs = make()
    bad = verify_math(convs)
    if bad:
        sys.exit(f"{bad} wrong arithmetic results: fix the templates")
    write(args.name, convs)
