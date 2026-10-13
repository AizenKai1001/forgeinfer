//! The tokenizer a GGUF file carries inside it must agree with the tokenizer.json of the same model on awkward text:
//! Unicode, digits, code, whitespace runs, special tokens written out in the text.
//! Needs model files (FORGE_MODELS, default ~/forgeinfer/oracle/models), so a missing file skips the case.

use forge_gguf::Gguf;
use forge_tok::Tokenizer;
use std::path::PathBuf;

fn models_dir() -> PathBuf {
    match std::env::var("FORGE_MODELS") {
        Ok(p) => PathBuf::from(p),
        Err(_) => PathBuf::from(std::env::var("HOME").unwrap_or_default()).join("forgeinfer/oracle/models"),
    }
}

fn corpus() -> Vec<String> {
    let mut v: Vec<String> = [
        "",
        "a",
        " ",
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog. It's a don't-care situation; we'll see; I'M SURE THEY'RE HERE.",
        "3.14159 1234567890 1,000,000 2026-10-13 0x7fffffff 1e-9 -42 +7 12345678901234567890",
        "fn main() {\n    let x: Vec<u32> = (0..10).map(|i| i * i).collect();\n    println!(\"{x:?}\");\n}\n",
        "def f(x):\n\treturn [i**2 for i in range(x) if i % 2 == 0]  # comment\n\n\nclass A:\n    pass",
        "{\"name\": \"forge\", \"values\": [1, 2.5, null, true], \"nested\": {\"a\": \"b\"}}",
        "# Title\n\n* item one\n* item two\n\n```python\nprint('hi')\n```\n\n> quote — with a dash\n",
        "https://example.com/path?query=1&b=2#frag user@example.com X:\\dir\\file.txt /usr/local/bin",
        "café naïve Zoë Straße ÅÄÖ ñandú œuvre Ærø İstanbul ǅ",
        "你好，世界！这是一个测试。日本語のテキスト、こんにちは。한국어 텍스트입니다.",
        "Привет, мир! Это тест. Ελληνικά κείμενο. مرحبا بالعالم עברית",
        "नमस्ते दुनिया ผมรักภาษาไทย தமிழ் மொழி",
        "emoji 😀 👍🏽 👨‍👩‍👧‍👦 🇩🇪 ❤️ ✨ 🧪 text after",
        "e\u{301} A\u{30a} a\u{308}\u{323} \u{1100}\u{1161}\u{11a8} \u{fb01} \u{2126} \u{212b}",
        "zero\u{200b}width\u{200d}joiner\u{feff}bom nbsp\u{a0}space em\u{2003}space",
        "word    four spaces\n\n\n  indented line\n\t\ttabbed\n   \ntrailing   \n",
        "line one\r\nline two\r\n\r\nline four\rline five",
        "\n\n\nleading newlines and a final newline\n",
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nHi!<|im_end|>\n<|im_start|>assistant\n",
        "<think>\nreasoning here\n</think>\n\nAnswer: 42<|endoftext|>",
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nhello<|eot_id|>",
        "<tool_call>{\"name\": \"f\", \"arguments\": {}}</tool_call> <|vision_start|><|image_pad|><|vision_end|>",
        "<|im_start|><|im_start|><|im_end|><|im_end|>no spaces<|endoftext|><|endoftext|>",
        "<|not_a_token|> <notatag> <| |> <|im_start| >",
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "!!!??? ... --- === ~~~ ((())) [[[]]] {{{}}} \\\\ //// ***** _____ ##### $$$$",
        "x\u{0}y \u{1}\u{2}\u{7f} tab\there",
        "I'll we've they'd she's It's DON'T 'tis 'twas rock'n'roll o'clock ''quoted'' '",
        "ſ ' ſ'S ’s ‘quoted’ “double” «guillemets» ¿qué? ¡hola!",
        "1 2 3 4 5 6 7 8 9 10 11 12 100 1000 10000 100000 0.5 .5 5. 1_000 1e10 3x4",
        "Ünïcödé ÀÉÎÕÜ àéîõü ŁÓDŹ ǆ ǈ ǌ ȸ ɐ ʰ ̀ ͂ ΐ",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    // a long mixed document: repeated paragraphs exercise the BPE merge cache and the chunking
    let para = "The model reads 1234 tokens per second on 8 GB of VRAM; naïve café 日本語 😀 — see https://example.com/a?b=c.\n";
    v.push(para.repeat(60));
    v
}

fn compare(label: &str, gguf: PathBuf, json: PathBuf) {
    if !gguf.exists() || !json.exists() {
        eprintln!("skipped {label}: {} or {} not found", gguf.display(), json.display());
        return;
    }
    let from_json = Tokenizer::from_file(&json).unwrap();
    let from_gguf = Tokenizer::from_gguf(&Gguf::open(&gguf).unwrap()).unwrap();
    // a GGUF vocabulary is padded to the embedding size with unused [PAD..] entries, so it may be longer, never shorter
    assert!(from_gguf.vocab_size() >= from_json.vocab_size(), "{label}: the GGUF vocabulary is shorter");
    let mut bad = 0;
    let cases = corpus();
    for t in &cases {
        let a = from_json.encode(t);
        let b = from_gguf.encode(t);
        if a != b {
            bad += 1;
            if bad <= 6 {
                let first = a.iter().zip(&b).position(|(x, y)| x != y).unwrap_or(a.len().min(b.len()));
                eprintln!(
                    "MISMATCH {label} {:?}\n  json  from {first}: {:?}\n  gguf  from {first}: {:?}",
                    t.chars().take(60).collect::<String>(),
                    &a[first..a.len().min(first + 8)],
                    &b[first..b.len().min(first + 8)]
                );
            }
            continue;
        }
        let back_json = from_json.decode(&a, false);
        let back_gguf = from_gguf.decode(&b, false);
        if back_json != back_gguf {
            bad += 1;
            if bad <= 6 {
                eprintln!("DECODE MISMATCH {label} {:?}", t.chars().take(60).collect::<String>());
            }
        }
    }
    // the special tokens the chat templates rely on resolve to the same ids
    for name in ["<|im_start|>", "<|im_end|>", "<|endoftext|>", "<think>", "</think>", "<|eot_id|>", "<|begin_of_text|>"] {
        let (a, b) = (from_json.token_id(name), from_gguf.token_id(name));
        if a != b {
            bad += 1;
            eprintln!("SPECIAL TOKEN {label} {name}: json {a:?} gguf {b:?}");
        }
    }
    assert_eq!(bad, 0, "{label}: {bad} differences");
    eprintln!("{label}: {} texts and the special tokens agree (vocabulary {})", cases.len(), from_gguf.vocab_size());
}

#[test]
fn smollm2() {
    let m = models_dir();
    compare(
        "smollm2",
        m.join("bartowski__SmolLM2-135M-Instruct-GGUF/SmolLM2-135M-Instruct-Q8_0.gguf"),
        m.join("HuggingFaceTB__SmolLM2-135M-Instruct/tokenizer.json"),
    );
}

#[test]
fn qwen3() {
    let m = models_dir();
    compare("qwen3", m.join("Qwen__Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf"), m.join("Qwen__Qwen3-0.6B/tokenizer.json"));
}

#[test]
fn qwen35_small() {
    let m = models_dir();
    compare(
        "qwen3.5-0.8b",
        m.join("unsloth__Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q8_0.gguf"),
        m.join("Qwen__Qwen3.5-0.8B/tokenizer.json"),
    );
}

#[test]
fn qwen35_4b_and_9b() {
    let m = models_dir();
    compare("qwen3.5-4b", m.join("unsloth__Qwen3.5-4B-GGUF/Qwen3.5-4B-Q4_K_M.gguf"), m.join("Qwen__Qwen3.5-4B/tokenizer.json"));
    compare("qwen3.5-9b", m.join("unsloth__Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf"), m.join("Qwen__Qwen3.5-9B/tokenizer.json"));
}
