//! Tests that need a GPU and a real embedding model: set FORGE_EMBED_GGUF to a nomic-embed-text GGUF file (FORGE_EMBED_GPU: the card's number, default 0); without them they print "skipped".
//! The golden vectors are those of the CPU tests (crates/forge-embed/tests/golden), made by tools/ref_nomic_bert.py (numpy) from the same file.

use forge_core::Json;
use forge_embed::{cosine, Embedder, Encode};
use forge_embed_gpu::GpuEmbedder;
use std::path::PathBuf;

fn model() -> Option<GpuEmbedder> {
    let Ok(p) = std::env::var("FORGE_EMBED_GGUF") else {
        eprintln!("skipped: FORGE_EMBED_GGUF is not set");
        return None;
    };
    let ordinal = std::env::var("FORGE_EMBED_GPU").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    let parts = Embedder::load(p).expect("the embedding model loads").into_parts();
    match GpuEmbedder::new(parts, ordinal) {
        Ok(g) => Some(g),
        Err(e) => {
            eprintln!("skipped: no usable GPU ({e})");
            None
        }
    }
}

fn golden() -> Vec<(String, Vec<f32>)> {
    let g = Json::from_file(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../forge-embed/tests/golden/nomic_v15_q4km.json")).expect("the golden file");
    g["cases"]
        .as_array()
        .expect("cases")
        .iter()
        .filter_map(|c| {
            let v = c.get("vector")?;
            Some((c["text"].as_str().unwrap().to_string(), v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect()))
        })
        .collect()
}

#[test]
fn vectors_match_the_reference() {
    let Some(g) = model() else { return };
    let cases = golden();
    assert!(cases.len() >= 20);
    let mut worst = 1.0f32;
    for (text, want) in &cases {
        let got = g.embed_many(&[text.as_str()]).expect("embeds").remove(0).vector;
        let c = cosine(&got, want);
        eprintln!("{text:?}: cosine to the reference {c:.7}");
        worst = worst.min(c);
    }
    // fp16 weights and activations on the card: not the 1.0000000 of the f32 CPU path, but far above the noise of the quantized file itself
    assert!(worst > 0.999, "worst cosine {worst}");
}

#[test]
fn a_pack_gives_the_vectors_of_the_single_texts() {
    let Some(g) = model() else { return };
    let cases = golden();
    let texts: Vec<&str> = cases.iter().map(|c| c.0.as_str()).collect();
    let pack = g.embed_many(&texts).expect("embeds");
    assert_eq!(pack.len(), texts.len());
    for ((text, want), got) in cases.iter().zip(&pack) {
        let single = g.embed_many(&[text.as_str()]).expect("embeds").remove(0);
        assert_eq!(single.tokens, got.tokens, "token count of {text:?}");
        let c = cosine(&single.vector, &got.vector);
        assert!(c > 0.99999, "{text:?}: single against pack {c}");
        assert!(cosine(&got.vector, want) > 0.999, "{text:?} against the reference");
    }
}

#[test]
fn texts_that_need_several_passes_keep_their_order() {
    let Some(g) = model() else { return };
    // three texts of the full context (more rows than one pass takes) and short ones between them
    let long: String = (0..g.config().ctx).map(|i| format!("word{} ", i % 97)).collect();
    let long2: String = (0..g.config().ctx).map(|i| format!("thing{} ", i % 89)).collect();
    let texts = ["a short one", long.as_str(), "another short text", long2.as_str(), long.as_str(), "the end"];
    let pack = g.embed_many(&texts).expect("embeds");
    assert_eq!(pack.len(), texts.len());
    assert!(pack.iter().map(|e| e.tokens).sum::<usize>() > g.rows_max());
    for (t, got) in texts.iter().zip(&pack) {
        let single = g.embed_many(&[*t]).expect("embeds").remove(0);
        let c = cosine(&single.vector, &got.vector);
        assert!(c > 0.99999, "{:?}: single against pack {c}", &t[..t.len().min(20)]);
        assert_eq!(single.tokens, got.tokens);
    }
}

#[test]
fn an_empty_text_and_a_text_beyond_the_context_work() {
    let Some(g) = model() else { return };
    let beyond: String = (0..g.config().ctx * 2).map(|i| format!("w{i} ")).collect();
    let r = g.embed_many(&["", beyond.as_str()]).expect("embeds");
    assert_eq!(r[0].tokens, 2, "[CLS] and [SEP]");
    assert_eq!(r[1].tokens, g.config().ctx, "cut to the context");
    for e in &r {
        let norm: f32 = e.vector.iter().map(|a| a * a).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-3, "length one, got {norm}");
    }
}
