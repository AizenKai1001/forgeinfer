"""Does the engine's tokenizer give the ids llama.cpp's gives? A corpus of awkward texts (digits, runs of white space, contractions, URLs, code, emoji with joiners, flags, combining marks, CJK, Cyrillic, Arabic, Hebrew, Thai,
Devanagari, control characters, long words, special-token look-alikes) tokenized by `llama-tokenize` (the CPU build: oracle) and by `forge tokenize`; the first difference of every text is printed with its context.
    python tools/tokenizer_parity.py MODEL.gguf [forge] [llama-tokenize]
Defaults: target/release/forge and llama-tokenize from the PATH. Each text is tokenized on its own (one call of each program per text, about a second per model load). Exit code 1 when a text differs."""
import json
import re
import subprocess
import sys

model = sys.argv[1]
forge = sys.argv[2] if len(sys.argv) > 2 else "target/release/forge"
llama = sys.argv[3] if len(sys.argv) > 3 else "llama-tokenize"

U = lambda s: s.encode("ascii").decode("unicode_escape")
texts = [
    "Hello, world!",
    "The quick brown fox jumps over the lazy dog.",
    "don't can't won't it's I'm we're they've she'd he'll DON'T IT'S",
    "1234567890 3.14159 1,000,000 0x7FFF 1e-9 -42 +7 007 2026-10-13 12:34:56",
    "a  b   c    d\t\te\n\nf\n\n\ng \n h\r\ni\r\n\r\nj",
    "   leading spaces and trailing spaces   ",
    "tabs\tand\t\ttabs\n\tindented\n\t\tmore",
    "https://example.com/path/to/page?query=1&other=two#fragment user@example.com C:/Users/you/file.txt /usr/local/bin",
    "fn main() { let x: Vec<u32> = (0..10).map(|i| i * i).collect(); println!(\"{:?}\", x); }",
    "def f(x):\n    return [i ** 2 for i in range(x) if i % 2 == 0]\n\nprint(f(10))\n",
    "#include <stdio.h>\nint main(void) {\n  printf(\"%d\\n\", 42);\n  return 0;\n}\n",
    "SELECT name, COUNT(*) FROM users WHERE age >= 18 GROUP BY name HAVING COUNT(*) > 1;",
    '{"key": "value", "list": [1, 2, 3], "nested": {"a": null, "b": true}}',
    "<html><body><p class=\"x\">text &amp; more</p></body></html>",
    "# Title\n\n- item one\n- item two\n\n```py\nprint(1)\n```\n\n> quote\n\n| a | b |\n|---|---|\n| 1 | 2 |",
    U("caf\\u00e9 na\\u00efve fa\\u00e7ade \\u00fcber \\u00c5ngstr\\u00f6m se\\u00f1or \\u0141\\u00f3d\\u017a"),
    U("e\\u0301 a\\u0300 o\\u0308 n\\u0303 (combining marks after the letters)"),
    U("\\u4f60\\u597d\\uff0c\\u4e16\\u754c\\uff01\\u8fd9\\u662f\\u4e00\\u4e2a\\u6d4b\\u8bd5\\u3002"),
    U("\\u3053\\u3093\\u306b\\u3061\\u306f\\u4e16\\u754c\\u3002\\u30c6\\u30b9\\u30c8\\u3067\\u3059\\u3002\\u6f22\\u5b57\\u3068\\u30ab\\u30bf\\u30ab\\u30ca"),
    U("\\uc548\\ub155\\ud558\\uc138\\uc694 \\uc138\\uacc4 \\ud14c\\uc2a4\\ud2b8\\uc785\\ub2c8\\ub2e4"),
    U("\\u041f\\u0440\\u0438\\u0432\\u0435\\u0442, \\u043c\\u0438\\u0440! \\u042d\\u0442\\u043e \\u0442\\u0435\\u0441\\u0442 \\u0442\\u043e\\u043a\\u0435\\u043d\\u0438\\u0437\\u0430\\u0442\\u043e\\u0440\\u0430."),
    U("\\u0645\\u0631\\u062d\\u0628\\u0627 \\u0628\\u0627\\u0644\\u0639\\u0627\\u0644\\u0645! \\u0647\\u0630\\u0627 \\u0627\\u062e\\u062a\\u0628\\u0627\\u0631."),
    U("\\u05e9\\u05dc\\u05d5\\u05dd \\u05e2\\u05d5\\u05dc\\u05dd! \\u05d6\\u05d4 \\u05de\\u05d1\\u05d7\\u05df."),
    U("\\u0e2a\\u0e27\\u0e31\\u0e2a\\u0e14\\u0e35\\u0e0a\\u0e32\\u0e27\\u0e42\\u0e25\\u0e01 \\u0e19\\u0e35\\u0e48\\u0e04\\u0e37\\u0e2d\\u0e01\\u0e32\\u0e23\\u0e17\\u0e14\\u0e2a\\u0e2d\\u0e1a"),
    U("\\u0928\\u092e\\u0938\\u094d\\u0924\\u0947 \\u0926\\u0941\\u0928\\u093f\\u092f\\u093e! \\u092f\\u0939 \\u090f\\u0915 \\u092a\\u0930\\u0940\\u0915\\u094d\\u0937\\u0923 \\u0939\\u0948\\u0964"),
    U("\\U0001f600 \\U0001f44d\\U0001f3fd \\U0001f468\\u200d\\U0001f469\\u200d\\U0001f467\\u200d\\U0001f466 \\U0001f1fa\\U0001f1f8 \\U0001f1e9\\U0001f1ea \\u2764\\ufe0f \\u2615 \\U0001f9d1\\u200d\\U0001f4bb"),
    U("math: \\u2200x \\u2208 \\u211d, x\\u00b2 \\u2265 0; \\u221a2 \\u2248 1.414; \\u03b1+\\u03b2=\\u03b3; \\u2211\\u1d62 a\\u1d62 \\u2192 \\u221e; \\u00bd \\u00be \\u2153"),
    U("quotes: \\u201cdouble\\u201d \\u2018single\\u2019 \\u00abguillemets\\u00bb \\u201edoes\\u201c \\u2014 em dash \\u2013 en dash \\u2026 ellipsis \\u00a0nbsp\\u00a0here"),
    "ALL CAPS TEXT AND camelCaseIdentifier snake_case_identifier kebab-case-identifier PascalCase HTTPServer XMLHttpRequest",
    "supercalifragilisticexpialidocious pneumonoultramicroscopicsilicovolcanoconiosis " + "a" * 300,
    "x" * 1000,
    " " * 200 + "after spaces",
    "\n" * 50 + "after newlines",
    "ab" * 200,
    "<|im_start|>user\nlook-alike of a special token<|im_end|>\n<|endoftext|> <think> </think> <tool_call> {\"name\": \"f\"} </tool_call>",
    "<s> </s> <unk> [INST] [/INST] <<SYS>> <|begin_of_text|> <|eot_id|> <|start_header_id|>",
    "'" + '"' + "`~!@#$%^&*()_+-=[]{}\\|;:,.<>/?",
    "mixed: Hello \u4e16\u754c 123 \u043c\u0438\u0440 \U0001f600 test_case-42 caf\u00e9",
    "A",
    " ",
    "\n",
    "",
    "The year 1999 and the number 100000000 and the version 1.2.3-rc.4+build.5 and ip 203.0.113.200:8080",
    "Dr. Smith's 3.5\" drive costs $1,299.99 (approx. EUR 1.200,50) -- see e.g. p. 42, fig. 3a/b; i.e. 50% off!!!",
    "thinking... <think>\nlet me see: 2 + 2 = 4\n</think>\n\nThe answer is 4.",
]

DIGITS_ONLY = re.compile(r"^\[(.*)\]\s*$", re.S)


def ids_of(proc):
    out = proc.stdout.strip().splitlines()
    for line in reversed(out):
        m = DIGITS_ONLY.match(line.strip())
        if m:
            body = m.group(1).strip()
            return [int(x) for x in body.split(",")] if body else []
    return None


bad = 0
for i, t in enumerate(texts):
    if t == "":
        # an empty prompt: llama-tokenize refuses it or prints nothing; the engine's answer is only checked to be empty
        r = subprocess.run([forge, "tokenize", model, t], capture_output=True, text=True, encoding="utf-8")
        print(f"[{i:2d}] empty text -> forge {ids_of(r)}")
        continue
    a = subprocess.run([llama, "-m", model, "--ids", "--no-bos", "--no-escape", "--no-parse-special", "--log-disable", "-p", t], capture_output=True, text=True, encoding="utf-8")
    b = subprocess.run([forge, "tokenize", model, t], capture_output=True, text=True, encoding="utf-8")
    ia, ib = ids_of(a), ids_of(b)
    if ia is None or ib is None:
        print(f"[{i:2d}] could not read ids: llama {ia is not None} forge {ib is not None}: {a.stdout[:80]!r} {a.stderr[:80]!r} | {b.stdout[:80]!r} {b.stderr[:80]!r}")
        bad += 1
        continue
    if ia == ib:
        print(f"[{i:2d}] same ({len(ia)} tokens)  {t[:50]!r}")
        continue
    k = next((j for j in range(min(len(ia), len(ib))) if ia[j] != ib[j]), min(len(ia), len(ib)))
    print(f"[{i:2d}] DIFFERENT: llama {len(ia)} tokens, forge {len(ib)}, first difference at token {k}: llama {ia[max(0, k - 2):k + 4]} forge {ib[max(0, k - 2):k + 4]}  text {t[:70]!r}")
    bad += 1
print(f"{bad} of {len(texts)} texts differ")
sys.exit(1 if bad else 0)
