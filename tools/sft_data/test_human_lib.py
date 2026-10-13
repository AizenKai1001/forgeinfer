#!/usr/bin/env python3
"""test_human_lib.py: tests of the mechanical text transformations and the forum rules of human_lib.py (run on the homelab: ~/forgeinfer/venv/bin/python tools/sft_data/test_human_lib.py)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import human_lib as H  # noqa: E402

FAILS = []


def check(name, got, want):
    if got != want:
        FAILS.append(name)
        print(f"FAIL {name}\n  got:  {got!r}\n  want: {want!r}")
    else:
        print(f"ok   {name}")


md, bad, links = H.html_to_md("<p>Is it possible?</p>\n\n<pre><code>/**\n * @test\n */\n--group a|b\n</code></pre>\n\n<p>Use <code>x</code> and <em>only</em> this.</p>")
check("pre block keeps stars and gets a fence", md, "Is it possible?\n\n```\n/**\n * @test\n */\n--group a|b\n```\n\nUse `x` and *only* this.")
check("no bad tag", (bad, links), (None, 0))

md, bad, links = H.html_to_md('<pre class="lang-py prettyprint-override"><code>def f(x):\n    return x ** 2\n</code></pre>')
check("language hint and power operator", md, "```py\ndef f(x):\n    return x ** 2\n```")

md, bad, links = H.html_to_md("<ul>\n<li><code>--a</code> works</li>\n<li>second <strong>item</strong></li>\n</ul>\n<p>After.</p>")
check("list", md, "- `--a` works\n- second **item**\n\nAfter.")

md, bad, links = H.html_to_md("<ol><li>one</li><li>two</li></ol>")
check("ordered list", md, "1. one\n2. two")

md, bad, links = H.html_to_md("<blockquote>\n<p>quoted text</p>\n</blockquote>\n<p>answer <a href='http://x.org/y'>here</a></p>")
check("blockquote and link", (md, links), ("> quoted text\n\nanswer here", 1))

md, bad, links = H.html_to_md('<p>see <img src="a.png"> this</p>')
check("image marks the post bad", bad, "img")

md, bad, links = H.html_to_md("<p>Value is 10<sup>3</sup> and H<sub>2</sub>O &amp; more &lt;b&gt;</p>")
check("sup, sub, entities", md, "Value is 10^3 and H_2O & more <b>")

md, bad, links = H.html_to_md("<p><strong></strong>Empty <em> </em>spans.</p>")
check("empty emphasis removed", md, "Empty spans.")

check("detokenize", H.detokenize("I was n't walking `` Oh , no '' he said . It 's fine ."), 'I wasn\'t walking "Oh, no" he said. It\'s fine.')
check("detokenize paragraphs", H.detokenize("One . \n \n Two ."), "One.\n\nTwo.")
check("wp artifacts", H.tokenization_artifacts("clean text, with commas. And a quote \"here\"."), 0)

check("meta: comments", H.forum_answer_problem("As mentioned in the comments, use sed."), "meta_reference")
check("meta: username", H.forum_answer_problem("@john you can do this with awk."), "meta_reference")
check("meta: the OP", H.forum_answer_problem("The OP should try sed."), "meta_reference")
check("no meta: op-amp", H.forum_answer_problem("Use an op-amp as a buffer. The output follows the input and the circuit is stable up to a few megahertz."), None)
check("no meta: the answer is", H.forum_answer_problem("The answer is yes. You can use a pressure cooker and it saves about half of the time."), None)
check("life story", H.forum_answer_problem("My wife and I tried this last year and it worked for us. We used a small pot."), "life_story")
check("clean explanation", H.forum_answer_problem("Water boils at a lower temperature at altitude because the air pressure is lower. At 3000 m it boils near 90 C."), None)
check("question refers to title", H.forum_question_problem("As the title says, how do I fix this?"), "question_refers_elsewhere")
check("question ok", H.forum_question_problem("How can I make my cookies chewy instead of crisp? I use brown sugar and butter."), None)
check("english on prose only", H.english_ok_human("Let $x$ be a real number with $\\frac{a}{b} = \\sqrt{c}$. Then the sum of the series over all terms is the same as the integral of the function over the interval, and we are done."), True)

print("FAILED:", FAILS if FAILS else "none")
sys.exit(1 if FAILS else 0)
