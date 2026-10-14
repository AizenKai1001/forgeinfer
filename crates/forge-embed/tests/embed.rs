//! Tests that need a real embedding model: set FORGE_EMBED_GGUF to a nomic-embed-text GGUF file (nomic-embed-text-v1.5.Q4_K_M.gguf is the one LM Studio ships); without it they print "skipped".
//! The golden vectors in tests/golden/nomic_v15_q4km.json were made by tools/ref_nomic_bert.py (numpy, independent of this code) from the same file.

use forge_core::Json;
use forge_embed::{cosine, Embedder};
use std::path::PathBuf;

fn model() -> Option<Embedder> {
    let Ok(p) = std::env::var("FORGE_EMBED_GGUF") else {
        eprintln!("skipped: FORGE_EMBED_GGUF is not set");
        return None;
    };
    Some(Embedder::load(p).expect("the embedding model loads"))
}

fn golden() -> Json {
    Json::from_file(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/golden/nomic_v15_q4km.json")).expect("the golden file")
}

#[test]
fn related_texts_are_closer_than_unrelated_ones() {
    let Some(e) = model() else { return };
    let q = e.embed("search_query: What is the capital of France?");
    let d1 = e.embed("search_document: Paris is the capital and largest city of France.");
    let d2 = e.embed("search_document: The mitochondria is the powerhouse of the cell.");
    assert_eq!(q.vector.len(), 768);
    for v in [&q.vector, &d1.vector, &d2.vector] {
        let norm: f32 = v.iter().map(|a| a * a).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-4, "length one, got {norm}");
    }
    let (s1, s2) = (cosine(&q.vector, &d1.vector), cosine(&q.vector, &d2.vector));
    eprintln!("query vs paris {s1:.4}, query vs mitochondria {s2:.4}");
    assert!(s1 > s2 + 0.1, "{s1} against {s2}");
    // the same text twice: the same vector, bit for bit
    assert_eq!(e.embed("search_query: What is the capital of France?").vector, q.vector);
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
    assert!(worst > 0.99995, "the vectors must match the numpy reference, worst cosine {worst}");
}
