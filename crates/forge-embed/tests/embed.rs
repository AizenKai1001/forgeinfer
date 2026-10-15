//! Tests that need a real embedding model: set FORGE_EMBED_GGUF to an embedding GGUF file; without it they print "skipped".
//!   nomic-embed-text:  FORGE_EMBED_GGUF=nomic-embed-text-v1.5.Q4_K_M.gguf (the one LM Studio ships); the golden vectors in tests/golden/nomic_v15_q4km.json were made by tools/ref_nomic_bert.py (numpy,
//!                      independent of this code) from the same file.
//!   bge-small:         FORGE_EMBED_GGUF=bge-small-en-v1.5-f32.gguf FORGE_EMBED_GOLDEN=bge_small_en_v15_hf.json; the golden vectors were made by tools/ref_hf_bert.py (Hugging Face `transformers` on the
//!                      official checkpoint). For a quantized file of the model also set FORGE_EMBED_MIN_COSINE=0.999 (the f32 file reaches 0.99995 and more).
//! FORGE_EMBED_GOLDEN is the name of a file in tests/golden (default: the nomic one).

use forge_core::Json;
use forge_embed::{cosine, Arch, Embedder};
use std::path::PathBuf;

fn model() -> Option<Embedder> {
    let Ok(p) = std::env::var("FORGE_EMBED_GGUF") else {
        eprintln!("skipped: FORGE_EMBED_GGUF is not set");
        return None;
    };
    Some(Embedder::load(p).expect("the embedding model loads"))
}

fn golden() -> Json {
    let name = std::env::var("FORGE_EMBED_GOLDEN").unwrap_or_else(|_| "nomic_v15_q4km.json".to_string());
    Json::from_file(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/golden").join(name)).expect("the golden file")
}

fn min_cosine() -> f32 {
    std::env::var("FORGE_EMBED_MIN_COSINE").ok().and_then(|v| v.parse().ok()).unwrap_or(0.99995)
}

#[test]
fn related_texts_are_closer_than_unrelated_ones() {
    let Some(e) = model() else { return };
    // nomic-embed-text was trained with a prefix per task; the BERT family needs none
    let (query, document) = if e.config().arch == Arch::NomicBert { ("search_query: ", "search_document: ") } else { ("", "") };
    let q = e.embed(&format!("{query}What is the capital of France?"));
    let d1 = e.embed(&format!("{document}Paris is the capital and largest city of France."));
    let d2 = e.embed(&format!("{document}The mitochondria is the powerhouse of the cell."));
    assert_eq!(q.vector.len(), e.dim());
    for v in [&q.vector, &d1.vector, &d2.vector] {
        let norm: f32 = v.iter().map(|a| a * a).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-4, "length one, got {norm}");
    }
    let (s1, s2) = (cosine(&q.vector, &d1.vector), cosine(&q.vector, &d2.vector));
    eprintln!("query vs paris {s1:.4}, query vs mitochondria {s2:.4}");
    assert!(s1 > s2 + 0.1, "{s1} against {s2}");
    // the same text twice: the same vector, bit for bit
    assert_eq!(e.embed(&format!("{query}What is the capital of France?")).vector, q.vector);
}

#[test]
fn token_ids_and_vectors_match_the_reference() {
    let Some(e) = model() else { return };
    let g = golden();
    let mut worst = 1.0f32;
    for case in g["cases"].as_array().expect("cases") {
        let text = case["text"].as_str().unwrap();
        let ids: Vec<u32> = case["ids"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
        assert_eq!(e.tokenizer().encode_truncated(text, e.config().ctx), ids, "token ids of {text:?}");
        if let Some(vec) = case.get("vector") {
            let want: Vec<f32> = vec.as_array().unwrap().iter().map(|v| v.as_f64().unwrap() as f32).collect();
            let got = e.embed(text).vector;
            let c = cosine(&got, &want);
            eprintln!("{text:?}: cosine to the reference {c:.7}");
            worst = worst.min(c);
        }
    }
    assert!(worst > min_cosine(), "the vectors must match the reference, worst cosine {worst}");
}

#[test]
fn word_order_changes_a_bert_vector() {
    let Some(e) = model() else { return };
    // the positions are part of the model: the same words in another order are another vector (the mean of a bag of words would not tell them apart, learned or rotary positions do)
    let a = e.embed("the dog bites the man");
    let b = e.embed("the man bites the dog");
    let c = cosine(&a.vector, &b.vector);
    eprintln!("same words, other order: cosine {c:.5}");
    assert!(c < 0.99999, "the two orders gave the same vector ({c})");
    assert_eq!(a.tokens, b.tokens);
}

#[test]
fn a_text_longer_than_the_context_is_cut_and_an_empty_one_works() {
    let Some(e) = model() else { return };
    let long: String = (0..e.config().ctx * 2).map(|i| format!("w{i} ")).collect();
    let r = e.embed(&long);
    assert_eq!(r.tokens, e.config().ctx, "cut to the context");
    assert!(r.vector.iter().all(|v| v.is_finite()));
    let empty = e.embed("");
    assert_eq!(empty.tokens, 2, "[CLS] and [SEP]");
    assert!(empty.vector.iter().all(|v| v.is_finite()));
}
