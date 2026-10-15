//! Tests that need a GPU and a real embedding model: set FORGE_EMBED_GGUF to an embedding GGUF file (FORGE_EMBED_GPU: the card's number, default 0); without them they print "skipped".
//! The golden vectors are those of the CPU tests (crates/forge-embed/tests/golden), made by tools/ref_nomic_bert.py (numpy) from the nomic-embed-text file, or by tools/ref_hf_bert.py (Hugging Face) for a
//! BERT-family model: FORGE_EMBED_GOLDEN is the name of the golden file (default: the nomic one; bge-small-en-v1.5 in f32: FORGE_EMBED_GOLDEN=bge_small_en_v15_hf.json). A quantized file of a BERT model
//! cannot meet the CPU-exact 0.99999 checks of the lazy encoder test: use the f32 file there.

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
    let name = std::env::var("FORGE_EMBED_GOLDEN").unwrap_or_else(|_| "nomic_v15_q4km.json".to_string());
    let g = Json::from_file(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../forge-embed/tests/golden").join(name)).expect("the golden file");
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
    assert!(cases.len() >= 8);
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
    // texts of the full context, enough of them for more rows than one pass takes (3 of 2048 tokens for nomic-embed-text, 10 of 512 for a BERT model), and short ones between them
    let ctx = g.config().ctx;
    let longs: Vec<String> = (0..g.rows_max() / ctx + 2).map(|k| (0..ctx).map(|i| format!("word{} ", (i + 13 * k) % (97 - k))).collect()).collect();
    let mut texts: Vec<&str> = vec!["a short one"];
    for (k, l) in longs.iter().enumerate() {
        texts.push(l);
        if k == 0 {
            texts.push("another short text");
        }
    }
    texts.push("the end");
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
fn the_lazy_encoder_loads_on_the_first_request_and_falls_back_to_the_cpu_without_room() {
    use forge_embed_gpu::{LazyEncoder, Place};
    let Ok(p) = std::env::var("FORGE_EMBED_GGUF") else {
        eprintln!("skipped: FORGE_EMBED_GGUF is not set");
        return;
    };
    // nothing is loaded at first: the configuration comes from the metadata
    let lazy = LazyEncoder::new(&p, Place::Auto, 0).expect("the metadata reads");
    let want = golden();
    assert_eq!(lazy.dim(), want[0].1.len());
    assert!(lazy.backend().contains("first use"), "{}", lazy.backend());
    let first = lazy.embed_many(&[want[0].0.as_str()]).expect("the first request loads the model and embeds");
    assert!(!lazy.backend().contains("first use"), "{}", lazy.backend());
    assert!(cosine(&first[0].vector, &want[0].1) > 0.999, "the lazily loaded model gives the reference vector");
    // a card without room (the threshold is raised above any card): the CPU copy, the same vectors
    std::env::set_var("FORGE_EMBED_MIN_FREE_MB", "10000000");
    let cpu = LazyEncoder::new(&p, Place::Auto, 0).unwrap();
    let r = cpu.embed_many(&[want[0].0.as_str()]).unwrap();
    std::env::remove_var("FORGE_EMBED_MIN_FREE_MB");
    assert!(cpu.backend().starts_with("CPU"), "{}", cpu.backend());
    // the CPU copy is f32 and so exact for an f32 file; a quantized file differs from an f32 reference by its own rounding (FORGE_EMBED_MIN_COSINE=0.999 for those)
    let min: f32 = std::env::var("FORGE_EMBED_MIN_COSINE").ok().and_then(|v| v.parse().ok()).unwrap_or(0.99999);
    assert!(cosine(&r[0].vector, &want[0].1) > min);
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
