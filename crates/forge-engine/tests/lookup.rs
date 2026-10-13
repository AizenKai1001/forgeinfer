//! Prompt lookup through the engine, on a model WITHOUT a prediction head: greedy output must equal plain decoding's (up to the numerical noise of near-tied tokens), the lookup must actually be used
//! on repeated text, and a sampled request with a seed must be reproducible. Skipped when the model file or a CUDA device is missing.
//!
//!   FORGE_LOOKUP_MODEL=/path/model.gguf cargo test --release -p forge-engine --test lookup -- --nocapture --test-threads=1
//! (default model: the homelab's Qwen3-4B Q4_K_M under $FORGE_MODELS or ~/forgeinfer/oracle/models).

use forge_cuda::Device;
use forge_engine::{
    Engine, EngineConfig, Event, FinishReason, Prompt, Request, SamplingParams, Usage, WeightMode,
};
use std::path::{Path, PathBuf};

fn model_path() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("FORGE_LOOKUP_MODEL") {
        return Some(PathBuf::from(p));
    }
    let dir = match std::env::var("FORGE_MODELS") {
        Ok(p) => PathBuf::from(p),
        Err(_) => PathBuf::from(std::env::var("HOME").unwrap_or_default())
            .join("forgeinfer/oracle/models"),
    };
    Some(dir.join("unsloth__Qwen3-4B-GGUF/Qwen3-4B-Q4_K_M.gguf"))
}

/// `speculate` false turns every kind of speculation off (no lookup); true leaves the lookup on (the model has no head here).
fn config(model: &Path, speculate: bool) -> EngineConfig {
    EngineConfig {
        model: model.to_path_buf(),
        tokenizer: None,
        weights: WeightMode::Native,
        n_ctx: 2048,
        device: 0,
        alias: "t".into(),
        chat: None,
        max_queue: 4,
        defaults: SamplingParams::default(),
        mtp: None,
        speculate,
    }
}

fn generate(e: &Engine, prompt: &str, sampling: SamplingParams, max_tokens: usize) -> (String, Usage, FinishReason) {
    let g = e
        .start(Request {
            prompt: Prompt::Raw(prompt.to_string()),
            sampling,
            max_tokens: Some(max_tokens),
            stop: vec![],
        })
        .unwrap();
    let mut text = String::new();
    let mut usage = Usage::default();
    let mut reason = FinishReason::Cancelled;
    for ev in g.rx.iter() {
        match ev {
            Event::Content(s) => text.push_str(&s),
            Event::Finished { usage: u, reason: r, .. } => {
                usage = u;
                reason = r;
                break;
            }
            Event::Error(err) => panic!("engine error: {err}"),
            _ => {}
        }
    }
    (text, usage, reason)
}

fn greedy() -> SamplingParams {
    SamplingParams {
        temperature: Some(0.0),
        ..Default::default()
    }
}

/// Prompts that end in the middle of a repetition: the continuation copies the text before it.
const PROMPTS: [&str; 4] = [
    "red green blue yellow orange purple red green blue yellow orange purple red green blue yellow orange purple red green blue yellow",
    "fn add(a: i32, b: i32) -> i32 {\n    a + b\n}\n\nfn sub(a: i32, b: i32) -> i32 {\n    a - b\n}\n\nfn mul(a: i32, b: i32) -> i32 {\n    a * b\n}\n\nfn div(a: i32, b: i32) -> i32 {\n",
    "The capital of France is Paris. The capital of Spain is Madrid. The capital of Italy is Rome. The capital of Germany is Berlin. The capital of France is",
    "Once upon a time there was a small village by the sea.",
];

#[test]
fn lookup_matches_plain_decoding_and_is_used() {
    let Some(model) = model_path().filter(|p| p.exists()) else {
        eprintln!("skipped: model file missing");
        return;
    };
    if Device::count().unwrap_or(0) == 0 {
        eprintln!("skipped: no CUDA device");
        return;
    }
    let n = 120;
    // plain decoding first (the engine is dropped before the next one is loaded)
    let plain: Vec<(String, Usage, FinishReason)> = {
        let e = Engine::load(config(&model, false)).unwrap();
        let r = PROMPTS.iter().map(|p| generate(&e, p, greedy(), n)).collect();
        drop(e);
        r
    };
    std::thread::sleep(std::time::Duration::from_secs(3));
    let e = Engine::load(config(&model, true)).unwrap();
    let (mut cycles, mut accepted, mut same, mut total, mut tokens) = (0usize, 0usize, 0usize, 0usize, 0usize);
    for (i, p) in PROMPTS.iter().enumerate() {
        let (text, usage, reason) = generate(&e, p, greedy(), n);
        let (ptext, pusage, preason) = &plain[i];
        let common = text.chars().zip(ptext.chars()).take_while(|(a, b)| a == b).count();
        eprintln!(
            "prompt {i}: {} tokens (plain {}), {} verification cycles, {} drafts accepted, {} of {} characters equal to plain decoding",
            usage.completion_tokens,
            pusage.completion_tokens,
            usage.draft_cycles,
            usage.draft_accepted,
            common,
            ptext.chars().count()
        );
        assert_eq!(pusage.draft_cycles, 0, "plain decoding must not speculate");
        cycles += usage.draft_cycles;
        accepted += usage.draft_accepted;
        same += common;
        total += ptext.chars().count();
        tokens += usage.completion_tokens;
        if text == *ptext {
            assert_eq!(reason, *preason);
            assert_eq!(usage.completion_tokens, pusage.completion_tokens);
        }
    }
    eprintln!("{tokens} tokens, {cycles} lookup cycles, {accepted} drafts accepted, {:.1} % of the characters equal to plain decoding", 100.0 * same as f64 / total as f64);
    assert!(cycles > 0, "the lookup was never used on repeated text");
    assert!(accepted as f64 > 0.5 * cycles as f64, "the drafts of the lookup are rarely right: {accepted} of {cycles} cycles");
    // near-tied logits may flip a token between the single-token and the multi-row kernels; the texts must agree on nearly everything
    assert!(same as f64 > 0.7 * total as f64, "lookup output differs from plain decoding: {same} of {total} characters equal");

    // a sampled request with a seed is reproducible, with the lookup in the loop
    let sampling = SamplingParams {
        temperature: Some(0.7),
        top_k: Some(40),
        top_p: Some(0.95),
        seed: Some(11),
        ..Default::default()
    };
    let (a, ua, ra) = generate(&e, PROMPTS[0], sampling.clone(), 80);
    let (b, ub, _) = generate(&e, PROMPTS[0], sampling, 80);
    assert_eq!(a, b, "the same seed must give the same text");
    assert_eq!(ua.draft_accepted, ub.draft_accepted);
    assert_eq!(ra, FinishReason::Length);
    assert_eq!(ua.completion_tokens, 80);
}
