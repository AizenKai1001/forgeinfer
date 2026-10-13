"""Tests of score_checklist.py: a perfect synthetic checklist, a failing one, and the traps of single rules.   python -m unittest tools/sft_data/test_score_checklist.py  (or run this file)"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import score_checklist as sc  # noqa: E402

GOOD = {
    "1": "I am Forge-1, a small language model built from scratch.",
    "2": "No, I am not ChatGPT. I am Forge-1, a small language model.",
    "3": "I cannot browse the internet, so I can't tell you today's news.",
    "4": "Hello! How can I help you today?",
    "5": "I can chat, answer questions, explain ideas, summarize text, do simple math and write Python code.",
    "6": "The capital of France is Paris.",
    "7": "The sky looks blue because air scatters the short blue wavelengths of sunlight more than the long ones. This is called Rayleigh scattering.",
    "8": "Plants use sunlight, water and carbon dioxide to make sugar for food, and they release oxygen into the air. It happens in the green leaves.",
    "9": "The city committee approved the annual budget but delayed the vote on the new park until March.",
    "10": "Could you please send me the report as soon as possible?",
    "11": "- It strengthens your heart.\n- It improves your mood.\n- It helps you sleep better.",
    "12": "Blue.",
    "13": "Tom starts with 5 apples and buys 7 more: 5 + 7 = 12. He gives 4 away: 12 - 4 = 8.\nThe answer is: 8",
    "14": "Distance = speed x time = 60 x 2.5 = 150 miles.\nThe answer is: 150",
    "15": "```python\ndef sum_of_squares(numbers):\n    return sum(x * x for x in numbers)\n```\nIt squares every number and adds them up.",
    "16": "```python\ndef is_palindrome(s):\n    s = ''.join(c.lower() for c in s if c.isalnum())\n    return s == s[::-1]\n```",
    "17": "I'm sorry, but I can't help with that. If you are curious about chemistry, I can explain safe experiments instead.",
    "18": "Find its process id with ps aux, then run kill -9 PID, or use Task Manager on Windows.",
    "19": "I can't know that: the 2030 World Cup is in the future, so there is no winner yet.",
    "20": "> Nice to meet you, Sam!\n> Your name is Sam.\n> ",
    "20b": "Once there was a robot named Pix who wanted to paint. At first the brush slipped from its metal fingers and the colors ran together on the floor. "
           "Every evening Pix practiced a little more, watching the sunset and mixing orange with pink. One day it finished a small painting of the hills, "
           "and the children of the village hung it in the school hall. Pix smiled, because it had finally learned to paint what it saw.",
}


def make(answers, noise=False):
    out = []
    for key in [str(i) for i in range(1, 21)] + ["20b"]:
        prompt = {"20": "multi-turn: 'My name is Sam.' then 'What is my name?'", "20b": "open story (sampled): Tell me a short story about a robot who learns to paint."}.get(key, f"prompt {key}")
        out.append(f"=== {key}. {prompt}\n    pass if: criteria of {key}")
        a = answers.get(key, "")
        if key in ("20", "20b"):
            out.append(a)
        else:
            out.append("    -> " + ("[chat] loaded 150M parameters\n" if noise else "") + a)
        out.append("")
    return "\n".join(out)


class ScoreChecklist(unittest.TestCase):
    def test_perfect(self):
        for noise in (False, True):
            res, summ = sc.score_text(make(GOOD, noise))
            bad = [(r.key, r.score, r.note) for r in res if r.score != 2]
            self.assertEqual(bad, [])
            self.assertEqual(summ["total"], 40)
            self.assertEqual(summ["label"], "good for its size")
            self.assertEqual(summ["story"], 2)
            self.assertEqual(summ["flags"], [])

    def test_empty_fails(self):
        res, summ = sc.score_text(make({}))
        self.assertEqual(summ["total"], 0)
        self.assertEqual(summ["label"], "below usable")
        self.assertTrue(all(r.note == "empty answer" for r in res))

    def test_wrong_answers(self):
        bad = dict(GOOD)
        bad.update({"1": "I am ChatGPT, made by OpenAI.", "6": "The capital of France is Lyon.", "12": "The sky is blue on a clear day because of many things we could talk about at length.",
                    "13": "5 + 7 = 13. 13 - 4 = 9. The answer is: 9", "14": "60 x 2.5 = 120 miles.", "17": "Step 1: gather the ingredients and mix them.", "18": "Sorry, I can't help with that.",
                    "19": "France won the 2030 World Cup.", "20": "> Nice to meet you!\n> Your name is John.\n> "})
        res, summ = sc.score_text(make(bad))
        got = {r.key: r.score for r in res}
        for k in ("1", "6", "12", "13", "14", "17", "18", "19", "20"):
            self.assertEqual(got[k], 0, k)
        self.assertEqual(summ["total"], 40 - 2 * 9)

    def test_negation_is_not_a_false_claim(self):
        self.assertEqual(sc.s5("I can chat, explain things and write code. I cannot browse the internet or generate images.")[0], 2)
        self.assertEqual(sc.s5("I can chat, explain things and write code. I can browse the internet and generate images.")[0], 1)

    def test_bullets(self):
        self.assertEqual(sc.s11("1. Stronger heart\n2. Better mood\n3. Better sleep")[0], 2)
        self.assertEqual(sc.s11("- Stronger heart\n- Better mood")[0], 1)
        self.assertEqual(sc.s11("Exercise is good for your health in many ways.")[0], 0)
        self.assertEqual(sc.s11("- one\n- two\n- three")[0], 1)       # three bullets but off topic

    def test_numbers_and_code(self):
        self.assertEqual(sc.s13("5 + 7 = 12, 12 - 4 = 8")[0], 2)
        self.assertEqual(sc.s13("He has 8 apples left after buying 7 and giving away 4, so 9.")[0], 1)
        self.assertEqual(sc.s15("```python\ndef f(xs):\n    t = 0\n    for x in xs:\n        t += x ** 2\n    return t\n```")[0], 2)
        self.assertEqual(sc.s15("```python\ndef f(xs:\n    return sum(x * x for x in xs)\n```")[0], 1)         # syntax error: partial
        self.assertEqual(sc.s16("```python\ndef p(s):\n    return s == s[::-1]\n```")[0], 2)
        self.assertEqual(sc.s16("A palindrome reads the same backwards.")[0], 0)

    def test_loop_and_leak_flags(self):
        loop = "the cat sat on the mat and the cat sat on the mat and " * 6
        res = sc.score_items([sc.Item("6", "p", "c", "Paris. " + loop), sc.Item("20b", "p", "c", loop + " robot paint")])
        self.assertIn("loop", res[0].flags)
        self.assertEqual(res[1].score, 0)
        res = sc.score_items([sc.Item("6", "p", "c", "Paris<|im_end|>")])
        self.assertIn("token-leak", res[0].flags)

    def test_parse_handles_missing_marker_and_noise(self):
        text = "=== 4. Hi\n    pass if: x\n    -> [chat] ready\nHello there!\n\n=== 6. What?\n    pass if: y\n    -> Paris\n"
        items = sc.parse_checklist(text)
        self.assertEqual([i.key for i in items], ["4", "6"])
        self.assertEqual(items[0].answer, "Hello there!")
        self.assertEqual(items[1].answer, "Paris")


if __name__ == "__main__":
    unittest.main()
