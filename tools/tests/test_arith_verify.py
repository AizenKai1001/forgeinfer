"""The self-verification of the arithmetic generators: correct output passes, a corrupted result is detected.

The generator modules parse sys.argv at import time, so they are imported with a clean argv. Needs numpy (sft_lib imports it); the tests are skipped without it.
"""
import contextlib
import importlib
import io
import os
import re
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SFT = os.path.join(os.path.dirname(HERE), "sft_data")
sys.path.insert(0, SFT)

_argv = sys.argv
sys.argv = [_argv[0]]
MODS = {}
SKIP = None
try:
    for name in ("sft_arith", "sft_arith2", "sft_arith3", "sft_drills"):
        MODS[name] = importlib.import_module(name)
except ImportError as e:  # numpy missing
    SKIP = str(e)
finally:
    sys.argv = _argv

LAST_EQ = re.compile(r"= (\d+)")


def corrupt(convs):
    """Add one to the last `= N` of the final assistant message of every conversation that has one."""
    out = []
    for g, msgs in convs:
        role, text = msgs[-1]
        ms = list(LAST_EQ.finditer(text))
        if not ms:
            continue
        m = ms[-1]
        bad = text[: m.start(1)] + str(int(m.group(1)) + 1) + text[m.end(1):]
        out.append((g, list(msgs[:-1]) + [(role, bad)]))
    return out


def quiet(fn, *a):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a)


def gen(fn, n):
    seen, out = set(), []
    for _ in range(n * 20):
        g, msgs = fn()
        k = tuple(msgs)
        if k in seen:
            continue
        seen.add(k)
        out.append((g, msgs))
        if len(out) == n:
            break
    return out


@unittest.skipIf(SKIP, f"generators not importable: {SKIP}")
class ArithVerifyTest(unittest.TestCase):
    def check(self, mod, convs):
        self.assertGreaterEqual(len(convs), 20)
        self.assertEqual(quiet(mod.verify, convs), 0, "clean output must verify")
        bad = corrupt(convs)
        self.assertGreaterEqual(len(bad), 10)
        # every single corrupted conversation is flagged on its own
        for conv in bad[:40]:
            self.assertGreater(quiet(mod.verify, [conv]), 0, conv[1][-1][1][:120])

    def test_sft_arith(self):
        m = MODS["sft_arith"]
        convs = gen(m.make_facts_one, 30) + gen(m.make_worked_one, 30)
        self.check(m, convs)

    def test_sft_arith_word_problems_verify(self):
        m = MODS["sft_arith"]
        self.assertEqual(quiet(m.verify, gen(m.make_word_one, 30)), 0)

    def test_sft_arith2(self):
        m = MODS["sft_arith2"]
        self.check(m, gen(m.plain_one, 40))

    def test_sft_arith2_word_problems_verify(self):
        m = MODS["sft_arith2"]
        self.assertEqual(quiet(m.verify, gen(m.word_one, 30)), 0)

    def test_sft_arith3(self):
        m = MODS["sft_arith3"]
        self.check(m, gen(m.A2.plain_one, 40))
        # drills: a bare `a / b = c.` line has no "So ..." summary, so the verifier does not recheck it; the other drills are rechecked
        drills = gen(m.drill_one, 60)
        self.assertEqual(quiet(m.verify, drills), 0)
        bad = corrupt(drills)
        self.assertGreater(sum(1 for c in bad if quiet(m.verify, [c]) > 0), len(bad) // 2)

    def test_sft_arith3_word_problems_verify(self):
        m = MODS["sft_arith3"]
        self.assertEqual(quiet(m.verify, gen(m.word_one, 30)), 0)

    def test_sft_drills(self):
        m = MODS["sft_drills"]
        convs = []
        for _ in range(200):
            g, q, a = m.pick(m.POOL)()
            convs.append((g, [("user", q), ("assistant", a)]))
        self.assertEqual(quiet(m.verify, convs), 0)
        bad = corrupt(convs)
        self.assertGreater(len(bad), 20)
        detected = sum(1 for c in bad if quiet(m.verify, [c]) > 0)
        # the drill kinds without an `a op b = c` expression (e.g. pure place-value facts) cannot be flagged; all arithmetic ones are
        self.assertGreaterEqual(detected, len(bad) * 0.5)


if __name__ == "__main__":
    unittest.main()
