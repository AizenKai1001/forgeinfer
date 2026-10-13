//! The tokenizer against the recorded Hugging Face goldens (475 cases per model) and Unicode's NFC conformance suite.

use forge_core::Json;
use forge_tok::normalize::nfc;
use forge_tok::Tokenizer;
use std::path::PathBuf;

fn golden_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/golden")
}

fn models_dir() -> PathBuf {
    match std::env::var("FORGE_MODELS") {
        Ok(p) => PathBuf::from(p),
        Err(_) => PathBuf::from(std::env::var("HOME").unwrap_or_default())
            .join("forgeinfer/oracle/models"),
    }
}

fn check(name: &str, model_dir: &str) {
    let path = models_dir().join(model_dir).join("tokenizer.json");
    if !path.exists() {
        eprintln!("skipped {name}: {} not found", path.display());
        return;
    }
    let tok = Tokenizer::from_file(&path).unwrap();
    check_with(name, &tok);
}

/// The same goldens against the tokenizer a GGUF file carries inside it.
fn check_gguf(name: &str, gguf: &str) {
    let path = models_dir().join(gguf);
    if !path.exists() {
        eprintln!("skipped {name} (GGUF): {} not found", path.display());
        return;
    }
    let tok = Tokenizer::from_gguf(&forge_gguf::Gguf::open(&path).unwrap()).unwrap();
    check_with(&format!("{name} (from the GGUF)"), &tok);
}

fn check_with(name: &str, tok: &Tokenizer) {
    let golden_name = name.split(' ').next().unwrap();
    let gold = Json::from_file(golden_dir().join(format!("tok_{golden_name}.json"))).unwrap();
    let cases = gold["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 475);
    let mut bad = 0;
    for case in cases {
        let text = case["text"].as_str().unwrap();
        let want: Vec<u32> = case["ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        let got = tok.encode(text);
        if got != want {
            bad += 1;
            if bad <= 5 {
                eprintln!(
                    "MISMATCH {name} {:?}\n  want {:?}\n  got  {:?}",
                    text.chars().take(60).collect::<String>(),
                    &want[..want.len().min(24)],
                    &got[..got.len().min(24)]
                );
            }
        }
        // decoding gives the text back (up to NFC where the tokenizer normalises)
        let back = tok.decode(&want, false);
        if back != text && back != nfc(text) {
            bad += 1;
            if bad <= 5 {
                eprintln!(
                    "DECODE MISMATCH {name} {:?} -> {:?}",
                    text.chars().take(40).collect::<String>(),
                    back.chars().take(40).collect::<String>()
                );
            }
        }
    }
    assert_eq!(bad, 0, "{name}: {bad} of 475 cases differ");
    eprintln!(
        "{name}: 475/475 cases exact (vocabulary {})",
        tok.vocab_size()
    );
}

#[test]
fn smollm2_gguf_tokenizer_matches_hugging_face() {
    check_gguf("smollm2", "bartowski__SmolLM2-135M-Instruct-GGUF/SmolLM2-135M-Instruct-Q8_0.gguf");
}

#[test]
fn qwen3_gguf_tokenizer_matches_hugging_face() {
    check_gguf("qwen3", "Qwen__Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf");
}

#[test]
fn qwen35_gguf_tokenizer_matches_hugging_face() {
    check_gguf("qwen35", "unsloth__Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q8_0.gguf");
}

#[test]
fn smollm2_tokenizer_matches_hugging_face() {
    check("smollm2", "HuggingFaceTB__SmolLM2-135M-Instruct");
}

#[test]
fn qwen3_tokenizer_matches_hugging_face() {
    check("qwen3", "Qwen__Qwen3-0.6B");
}

#[test]
fn qwen35_tokenizer_matches_hugging_face() {
    check("qwen35", "Qwen__Qwen3.5-0.8B");
}

fn parse_cps(field: &str) -> String {
    field
        .split_whitespace()
        .map(|h| char::from_u32(u32::from_str_radix(h, 16).unwrap()).unwrap())
        .collect()
}

#[test]
fn nfc_matches_the_unicode_conformance_suite() {
    let text = std::fs::read_to_string(golden_dir().join("NormalizationTest.txt")).unwrap();
    let mut lines = 0;
    for (n, line) in text.lines().enumerate() {
        let line = line.split('#').next().unwrap().trim();
        if line.is_empty() || line.starts_with('@') {
            continue;
        }
        let f: Vec<&str> = line.split(';').collect();
        if f.len() < 5 {
            continue;
        }
        let (c1, c2, c3, c4, c5) = (
            parse_cps(f[0]),
            parse_cps(f[1]),
            parse_cps(f[2]),
            parse_cps(f[3]),
            parse_cps(f[4]),
        );
        // NFC: c2 == toNFC(c1) == toNFC(c2) == toNFC(c3);  c4 == toNFC(c4) == toNFC(c5)
        assert_eq!(nfc(&c1), c2, "line {}: NFC(c1)", n + 1);
        assert_eq!(nfc(&c2), c2, "line {}: NFC(c2)", n + 1);
        assert_eq!(nfc(&c3), c2, "line {}: NFC(c3)", n + 1);
        assert_eq!(nfc(&c4), c4, "line {}: NFC(c4)", n + 1);
        assert_eq!(nfc(&c5), c4, "line {}: NFC(c5)", n + 1);
        lines += 1;
    }
    eprintln!("NFC conformance: {lines} test lines pass");
    assert!(lines > 18000);
}

#[test]
fn bpe_trainer_reproduces_the_specification_on_the_sample_corpus() {
    // the 3.9 MB sample used for the Python specification: 1,792 merges (vocab 2,048) gave 0.3821 tokens per byte on its last 200,000 characters (Hugging Face's trainer: 0.3821)
    let path =
        PathBuf::from(std::env::var("HOME").unwrap_or_default()).join("forgeinfer/bpe_sample.txt");
    if !path.exists() {
        eprintln!("skipped: {} not found", path.display());
        return;
    }
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|_| String::from_utf8_lossy(&std::fs::read(&path).unwrap()).into_owned());
    let cfg = forge_tok::train::TrainConfig {
        vocab_size: 2048,
        special_tokens: vec![],
        threads: 2,
        log_every: 0,
    };
    let counts = {
        let tmp = std::env::temp_dir().join("forge_bpe_sample.txt");
        std::fs::write(&tmp, text.as_bytes()).unwrap();
        forge_tok::train::count_words(&[tmp], forge_tok::PreTokenizer::Gpt2Digits, 2).unwrap()
    };
    let t0 = std::time::Instant::now();
    let trained = forge_tok::train::train_from_counts(counts, &cfg).unwrap();
    eprintln!(
        "trained {} merges in {:.2} s",
        trained.merges.len(),
        t0.elapsed().as_secs_f32()
    );
    assert_eq!(trained.symbol_bytes.len(), 2048);
    eprintln!(
        "{} merges produced {} distinct tokens",
        trained.merges.len(),
        trained.symbol_bytes.len() - 256
    );
    let tok = Tokenizer::from_json(&trained.to_tokenizer_json()).unwrap();
    let held: String = text
        .chars()
        .rev()
        .take(200_000)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    let ids = tok.encode(&held);
    assert_eq!(tok.decode(&ids, false), held, "round trip");
    let tpb = ids.len() as f64 / held.len() as f64;
    eprintln!("tokens per byte on the last 200,000 characters: {tpb:.4}");
    assert!((tpb - 0.3821).abs() < 0.001, "{tpb}");
}
