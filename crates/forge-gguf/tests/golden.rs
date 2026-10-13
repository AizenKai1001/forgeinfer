//! Bit-exact dequantisation against the recorded golden blocks, and container checks against the real oracle models on the homelab (skipped when the models are absent).

use forge_core::Json;
use forge_gguf::{dequantize, DType, Gguf, SafeTensorsDir};
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

fn dtype_named(name: &str) -> DType {
    match name {
        "F16" => DType::F16,
        "Q8_0" => DType::Q8_0,
        "Q6_K" => DType::Q6_K,
        "Q5_0" => DType::Q5_0,
        "Q4_K" => DType::Q4_K,
        "Q4_1" => DType::Q4_1,
        "Q4_0" => DType::Q4_0,
        other => panic!("unknown golden type {other}"),
    }
}

#[test]
fn dequantisation_matches_the_golden_blocks_bit_for_bit() {
    let index = Json::from_file(golden_dir().join("quant_index.json")).unwrap();
    let mut checked = 0;
    for (name, meta) in index.as_object().unwrap() {
        let dt = dtype_named(name);
        assert_eq!(
            dt.block_bytes(),
            meta["block_bytes"].as_usize().unwrap(),
            "{name} block bytes"
        );
        assert_eq!(
            dt.block_values(),
            meta["block_values"].as_usize().unwrap(),
            "{name} block values"
        );
        let raw = std::fs::read(golden_dir().join(format!("quant_{name}.in"))).unwrap();
        let want: Vec<f32> = std::fs::read(golden_dir().join(format!("quant_{name}.f32")))
            .unwrap()
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let got = dequantize(dt, &raw, want.len()).unwrap();
        let mut bad = 0;
        for (i, (a, b)) in got.iter().zip(&want).enumerate() {
            if a.to_bits() != b.to_bits() {
                if bad < 5 {
                    eprintln!("{name}[{i}]: got {a:e} want {b:e}");
                }
                bad += 1;
            }
        }
        assert_eq!(bad, 0, "{name}: {bad} of {} values differ", want.len());
        checked += 1;
    }
    assert_eq!(checked, 7);
}

#[test]
fn gguf_container_of_the_oracle_models() {
    let dir = models_dir().join("bartowski__SmolLM2-135M-Instruct-GGUF");
    let f16 = dir.join("SmolLM2-135M-Instruct-f16.gguf");
    if !f16.exists() {
        eprintln!("skipped: {} not found", f16.display());
        return;
    }
    let g = Gguf::open(&f16).unwrap();
    assert_eq!(g.architecture(), Some("llama"));
    assert_eq!(g.get_u64("llama.block_count"), Some(30));
    assert_eq!(g.get_u64("llama.embedding_length"), Some(576));
    let toks = g.get_str_array("tokenizer.ggml.tokens").unwrap();
    assert_eq!(toks.len(), 49152);
    let emb = g.tensor_info("token_embd.weight").unwrap();
    assert_eq!(emb.shape(), vec![49152, 576]);
    assert_eq!(emb.dtype, DType::F16);

    // the Hugging Face checkpoint holds the same embedding (bf16 there, f16 here: equal up to the conversion rounding)
    let st =
        SafeTensorsDir::open(models_dir().join("HuggingFaceTB__SmolLM2-135M-Instruct")).unwrap();
    let a = g.tensor_f32("token_embd.weight").unwrap();
    let b = st.tensor_f32("model.embed_tokens.weight").unwrap();
    assert_eq!(a.len(), b.len());
    let worst = a
        .iter()
        .zip(&b)
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max);
    eprintln!("embedding: GGUF f16 versus safetensors, worst abs difference {worst:e}");
    assert!(worst < 2e-3, "{worst}");

    // the Q8_0 file: every tensor type is one we can read, and Q8_0 stays close to f16
    let q8 = Gguf::open(dir.join("SmolLM2-135M-Instruct-Q8_0.gguf")).unwrap();
    for t in &q8.tensors {
        assert!(
            !matches!(t.dtype, DType::Other(_)),
            "{} has type {}",
            t.name,
            t.dtype.name()
        );
    }
    let c = q8.tensor_f32("token_embd.weight").unwrap();
    let worst = a
        .iter()
        .zip(&c)
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max);
    eprintln!("embedding: Q8_0 versus f16, worst abs difference {worst:e}");
    assert!(worst < 2e-2);

    // the K-quant file mixes Q8_0, Q5_0, Q4_K and Q6_K: all of them must be readable and finite
    let k = Gguf::open(dir.join("SmolLM2-135M-Instruct-Q4_K_M.gguf")).unwrap();
    let mut seen = std::collections::BTreeSet::new();
    for t in &k.tensors {
        seen.insert(t.dtype.name());
        let v = k.tensor_f32(&t.name).unwrap();
        assert!(
            v.iter().all(|x| x.is_finite()),
            "{} has non-finite values",
            t.name
        );
    }
    eprintln!("Q4_K_M tensor types: {seen:?}");
}

#[test]
fn safetensors_directory_of_the_oracle_models() {
    let d = models_dir().join("Qwen__Qwen3-0.6B");
    if !d.exists() {
        eprintln!("skipped: {} not found", d.display());
        return;
    }
    let st = SafeTensorsDir::open(&d).unwrap();
    let (_, t) = st.tensor("model.embed_tokens.weight").unwrap();
    assert_eq!(t.shape, vec![151936, 1024]);
    let n = st.tensor_f32("model.norm.weight").unwrap();
    assert_eq!(n.len(), 1024);
    assert!(n.iter().all(|x| x.is_finite()));
}
