"""Golden tokenizer outputs for our own tokenizer's tests.

The reference library (Hugging Face `tokenizers`) is an ORACLE only: it runs once, here, to record what the real tokenizers of the test models produce; our Rust tokenizer must reproduce
these ids exactly. Run on the homelab:   ~/forgeinfer/venv/bin/python tools/gen_tokenizer_goldens.py
Writes tests/golden/tok_<name>.json = {"name", "source", "cases": [{"text", "ids", "tokens"}]} (ids without special-token insertion; special tokens that appear inside the text are recognized, as in use).
"""
import json
import os
import random
import sys

from tokenizers import Tokenizer

ROOT = os.path.expanduser("~/forgeinfer/oracle/models")
MODELS = {
    "smollm2": os.path.join(ROOT, "HuggingFaceTB__SmolLM2-135M-Instruct", "tokenizer.json"),
    "qwen3": os.path.join(ROOT, "Qwen__Qwen3-0.6B", "tokenizer.json"),
    "qwen35": os.path.join(ROOT, "Qwen__Qwen3.5-0.8B", "tokenizer.json"),
}

HAND = [
    "", " ", "  ", "   ", "\n", "\n\n", "\n \n", "\t", "\r\n", " \n", "a", "A", "aa", "Hello", "Hello, world!", "hello world", " hello", "  hello", "hello  world", "hello\nworld",
    "The quick brown fox jumps over the lazy dog.", "   leading spaces", "trailing spaces   ", "tabs\tand\nnewlines\r\nmixed", "1", "12", "123", "1234567890", "3.14159", "1,000,000", "0.000123",
    "2026-10-09", "12:30 PM", "user@example.com", "https://example.com/path?x=1&y=2#frag", "C:/Users/you\\file.txt", "/usr/local/bin/python3",
    "def f(x):\n    return x * 2\n", "fn main() {\n    println!(\"hi {}\", 1 + 2);\n}\n", "if (a && b || !c) { d++; }", "SELECT * FROM users WHERE id = 7 AND name LIKE '%a%';",
    "x = [i**2 for i in range(10)]", "#include <stdio.h>\nint main(void){return 0;}", "<div class=\"a\">text</div>", "{\"key\": [1, 2, {\"n\": null}], \"ok\": true}",
    "naïve café façade déjà vu", "Zażółć gęślą jaźń", "日本語のテキストです。", "한국어 텍스트입니다", "Привет, мир! Как дела?", "مرحبا بالعالم", "שלום עולם", "สวัสดีชาวโลก", "नमस्ते दुनिया",
    "emoji 😀🎉👍🏽 and flags 🇺🇸🇯🇵", "ZWJ family 👨‍👩‍👧‍👦", "combining e\u0301 and a\u030a", "full-width ＡＢＣ１２３", "math ∑∫√∞ ≠ ≤ ≥", "ß ẞ ǅ İ ı",
    "don't can't won't I'm you're they've we'll he'd", "ALLCAPS and camelCase and snake_case and kebab-case", "Hello!!!   ???   ...   ---", "a" * 50, "ab" * 40, " " * 30 + "x", "\n" * 5 + "y",
    "<|im_start|>user\nHi there<|im_end|>\n<|im_start|>assistant\n", "<|endoftext|>", "text <|endoftext|> more", "<think>\nhmm\n</think>", "<tool_call>\n{\"name\": \"f\"}\n</tool_call>",
    "The year 1999 and 2026; 42nd; 3rd; 100%; $5.99; 5kg; 10x; v1.2.3-rc.1", "word1word2 word_3 word-4 word.5 word,6", "Tab\tseparated\tvalues\tin\ta\trow",
]

# fixed-seed pseudo-random strings from many scripts: coverage for the byte-level fallback and the pre-tokenizer
ALPHABETS = [
    "abcdefghijklmnopqrstuvwxyz", "ABCDEFGHIJKLMNOPQRSTUVWXYZ", "0123456789", " \t\n.,;:!?'\"()[]{}<>/\\|-_=+*&^%$#@~`", "αβγδεζηθικλμνξοπρστυφχψω", "абвгдеёжзийклмнопрстуфхцчшщъыьэюя",
    "あいうえおかきくけこさしすせそ", "日本語中文汉字漢字大小上下", "한국어가나다라마바사", "😀😃😄😁😆😅🤣😂🙂🙃😉😊😇🥰😍", "éèêëàâäôöùûüçñáíóú",
]


def random_cases(n: int, seed: int):
    rng = random.Random(seed)
    for _ in range(n):
        parts = []
        for _ in range(rng.randint(1, 6)):
            alpha = rng.choice(ALPHABETS)
            parts.append("".join(rng.choice(alpha) for _ in range(rng.randint(1, 14))))
        sep = rng.choice(["", " ", "  ", "\n", " "])
        yield sep.join(parts)


def main() -> int:
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests", "golden")
    os.makedirs(out_dir, exist_ok=True)
    texts = HAND + list(random_cases(400, 20261009))
    for name, path in MODELS.items():
        tok = Tokenizer.from_file(path)
        cases = []
        for t in texts:
            enc = tok.encode(t, add_special_tokens=False)
            cases.append({"text": t, "ids": enc.ids, "tokens": enc.tokens})
        dest = os.path.join(out_dir, f"tok_{name}.json")
        with open(dest, "w", encoding="utf-8", newline="\n") as f:
            json.dump({"name": name, "source": os.path.basename(os.path.dirname(path)), "vocab_size": tok.get_vocab_size(), "cases": cases}, f, ensure_ascii=False, separators=(",", ":"))
        print(f"{name}: {len(cases)} cases, vocab {tok.get_vocab_size()} -> {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
