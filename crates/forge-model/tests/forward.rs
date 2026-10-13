//! The CPU reference forward pass against the recorded Hugging Face goldens (last-position logits of six prompts, all-position logits of prompt 0, 48 greedy tokens).
//! The models live on the homelab (FORGE_MODELS or ~/forgeinfer/oracle/models); the tests skip when they are absent.

use forge_core::Json;
use forge_model::{argmax, Model, Session};
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

fn read_f32(path: PathBuf) -> Vec<f32> {
    std::fs::read(path)
        .unwrap()
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

/// Returns the worst last-logit difference over all prompts.
fn check(
    golden: &str,
    model_path: PathBuf,
    tol: f32,
    greedy_prompts: &[usize],
    strict_top1: bool,
) -> f32 {
    if !model_path.exists() {
        eprintln!("skipped {golden}: {} not found", model_path.display());
        return 0.0;
    }
    let t0 = std::time::Instant::now();
    let model = Model::load(&model_path).unwrap();
    eprintln!(
        "loaded {} in {:.1} s ({} layers, d {}, vocab {})",
        model_path.display(),
        t0.elapsed().as_secs_f32(),
        model.cfg.n_layer,
        model.cfg.n_embd,
        model.cfg.vocab
    );
    let gold = Json::from_file(golden_dir().join(format!("fwd_{golden}.json"))).unwrap();
    let mut worst = 0f32;
    for (pi, p) in gold["prompts"].as_array().unwrap().iter().enumerate() {
        let ids: Vec<u32> = p["ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        let want = read_f32(golden_dir().join(p["last_logits_file"].as_str().unwrap()));
        let mut s = Session::new(&model, 256);
        let t1 = std::time::Instant::now();
        let logits = s.prefill(&ids);
        let d = max_diff(&logits, &want);
        worst = worst.max(d);
        let top1 = p["top20"][0][0].as_u64().unwrap() as u32;
        eprintln!(
            "{golden} prompt {pi} ({} tokens): max |logit diff| {d:.2e}, top-1 {} ({:.0} ms)",
            ids.len(),
            if argmax(&logits) == top1 {
                "same"
            } else {
                "DIFFERENT"
            },
            t1.elapsed().as_secs_f64() * 1e3
        );
        if strict_top1 {
            assert_eq!(argmax(&logits), top1, "top-1 differs on prompt {pi}");
        } else {
            // a lossy format may flip a near-tie: the reference's top token must stay among our best five
            let mut order: Vec<usize> = (0..logits.len()).collect();
            order.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());
            assert!(
                order[..5].contains(&(top1 as usize)),
                "reference top-1 {top1} not in our top five on prompt {pi}"
            );
        }
        assert!(d < tol, "prompt {pi}: logits differ by {d}");
        if pi == 0 {
            // all positions
            let all = read_f32(golden_dir().join(format!("fwd_{golden}_p0_logits.f32")));
            let mut s = Session::new(&model, 256);
            let mut got = Vec::new();
            for &t in &ids {
                got.extend(s.step(t, true).unwrap());
            }
            let d = max_diff(&got, &all);
            eprintln!("{golden} prompt 0: all-position logits max diff {d:.2e}");
            assert!(d < tol, "all-position logits differ by {d}");
        }
        if greedy_prompts.contains(&pi) {
            let want_ids: Vec<u32> = p["greedy48"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as u32)
                .collect();
            let mut s = Session::new(&model, 256);
            let mut logits = s.prefill(&ids);
            let mut got = Vec::new();
            for _ in 0..want_ids.len() {
                let t = argmax(&logits);
                got.push(t);
                logits = s.step(t, true).unwrap();
            }
            let same = got
                .iter()
                .zip(&want_ids)
                .take_while(|(a, b)| a == b)
                .count();
            eprintln!(
                "{golden} prompt {pi}: greedy {} tokens, first {same} identical",
                want_ids.len()
            );
            assert_eq!(
                got, want_ids,
                "greedy continuation differs after {same} tokens"
            );
        }
    }
    worst
}

#[test]
fn smollm2_from_the_hugging_face_checkpoint() {
    check(
        "smollm2",
        models_dir().join("HuggingFaceTB__SmolLM2-135M-Instruct"),
        1e-3,
        &[0, 3],
        true,
    );
}

#[test]
fn smollm2_from_the_f16_gguf() {
    // the f16 file holds the bf16 weights rounded to f16: the logits agree to about 1e-3; llama.cpp's permuted Q/K layout is undone at load
    check(
        "smollm2",
        models_dir().join("bartowski__SmolLM2-135M-Instruct-GGUF/SmolLM2-135M-Instruct-f16.gguf"),
        5e-2,
        &[0],
        true,
    );
}

#[test]
fn smollm2_from_the_q8_0_gguf_keeps_the_top_token() {
    let worst = check(
        "smollm2",
        models_dir().join("bartowski__SmolLM2-135M-Instruct-GGUF/SmolLM2-135M-Instruct-Q8_0.gguf"),
        1.0,
        &[],
        false,
    );
    eprintln!("Q8_0 worst last-logit difference {worst:.3}");
}

#[test]
fn qwen3_from_the_hugging_face_checkpoint() {
    check(
        "qwen3",
        models_dir().join("Qwen__Qwen3-0.6B"),
        2e-3,
        &[0],
        true,
    );
}

#[test]
fn qwen35_from_the_hugging_face_checkpoint() {
    // the hybrid model: 18 Gated DeltaNet layers and 6 gated full-attention layers; the 121-token prompt crosses the 64-token chunk of the reference
    check(
        "qwen35",
        models_dir().join("Qwen__Qwen3.5-0.8B"),
        5e-3,
        &[0],
        true,
    );
}

#[test]
fn qwen35_from_the_q8_0_gguf_keeps_the_top_token() {
    // GGUF conventions of the family: +1 already baked into the norms, ssm_a = -exp(A_log), squeezed conv weight, post_attention_norm as the pre-MLP norm
    let worst = check(
        "qwen35",
        models_dir().join("unsloth__Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q8_0.gguf"),
        1.5,
        &[],
        false,
    );
    eprintln!("Q8_0 worst last-logit difference {worst:.3}");
}

#[test]
fn qwen35_from_the_q4_k_m_gguf_keeps_the_top_token() {
    let worst = check(
        "qwen35",
        models_dir().join("unsloth__Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf"),
        6.0,
        &[],
        false,
    );
    eprintln!("Q4_K_M worst last-logit difference {worst:.3}");
}

/// The MTP head of Qwen3.5-0.8B as a drafter: teacher-forced on the golden prompts, its arg-max for position t + 2 (from the hidden state at t and the token at t + 1) should be the main model's own
/// arg-max for that position almost every time (numpy experiment tools/ref_qwen35_mtp.py: 95 % on the 121-token prompt).
#[test]
fn qwen35_mtp_head_predicts_the_main_models_next_token() {
    let path = models_dir().join("Qwen__Qwen3.5-0.8B");
    if !path.exists() {
        eprintln!("skipped: model missing");
        return;
    }
    let model = Model::load(&path).unwrap();
    assert!(model.mtp.is_some(), "the checkpoint has an MTP head");
    let gold = Json::from_file(golden_dir().join("fwd_qwen35.json")).unwrap();
    let ids: Vec<u32> = gold["prompts"][6]["ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect();
    let mut s = Session::new(&model, 256);
    let (mut main_arg, mut hidden) = (Vec::new(), Vec::new());
    for &t in &ids {
        let logits = s.step(t, true).unwrap();
        main_arg.push(argmax(&logits));
        hidden.push(s.last_hidden.clone());
    }
    let mut agree = 0;
    let n = ids.len() - 2;
    for t in 0..ids.len() - 1 {
        let logits = s.mtp_step(&hidden[t], ids[t + 1], t);
        if t < n && argmax(&logits) == main_arg[t + 1] {
            agree += 1;
        }
    }
    eprintln!("MTP head: {agree} of {n} drafts equal the main model's own next-next token");
    assert!(
        agree as f64 > 0.85 * n as f64,
        "only {agree} of {n} drafts agree"
    );
}
