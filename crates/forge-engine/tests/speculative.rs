//! Speculative decoding through the engine (Qwen3.5-0.8B Q4_K_M with the MTP head from the Hugging Face checkpoint): greedy output must equal plain decoding's (up to the numerical noise of
//! near-tied tokens), the head's drafts must be accepted most of the time, and sampling must stay sensible. Skipped when the files or a CUDA device are missing.

use forge_cuda::Device;
use forge_engine::{
    Engine, EngineConfig, Event, FinishReason, Prompt, Request, SamplingParams, Usage, WeightMode,
};
use std::path::{Path, PathBuf};

fn models_dir() -> PathBuf {
    match std::env::var("FORGE_MODELS") {
        Ok(p) => PathBuf::from(p),
        Err(_) => PathBuf::from(std::env::var("HOME").unwrap_or_default())
            .join("forgeinfer/oracle/models"),
    }
}

/// The head: a `mtp.safetensors` fetched into the Hugging Face directory (tools/fetch_mtp.py), else the directory itself when it holds the whole checkpoint.
fn head(hf: &Path) -> Option<PathBuf> {
    let f = hf.join("mtp.safetensors");
    if f.exists() {
        Some(f)
    } else if hf.join("model.safetensors").exists()
        || hf.join("model.safetensors.index.json").exists()
    {
        Some(hf.to_path_buf())
    } else {
        None
    }
}

fn config(gguf: &Path, hf: &Path, speculate: bool) -> EngineConfig {
    EngineConfig {
        model: gguf.to_path_buf(),
        tokenizer: Some(hf.join("tokenizer.json")),
        weights: WeightMode::Native,
        n_ctx: 2048,
        device: 0,
        alias: "t".into(),
        chat: None,
        max_queue: 4,
        defaults: SamplingParams::default(),
        mtp: if speculate { head(hf) } else { None },
        speculate,
    }
}

fn generate(
    e: &Engine,
    prompt: &str,
    sampling: SamplingParams,
    max_tokens: usize,
    stop: Vec<String>,
) -> (String, Usage, FinishReason) {
    let g = e
        .start(Request {
            prompt: Prompt::Raw(prompt.to_string()),
            sampling,
            max_tokens: Some(max_tokens),
            stop,
        })
        .unwrap();
    let mut text = String::new();
    let mut usage = Usage::default();
    let mut reason = FinishReason::Cancelled;
    for ev in g.rx.iter() {
        match ev {
            Event::Content(s) => text.push_str(&s),
            Event::Finished {
                usage: u,
                reason: r,
                ..
            } => {
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

const PROMPTS: [&str; 6] = [
    "The capital of France is",
    "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
    "In 1969, the Apollo 11 mission",
    "Q: What is the difference between a process and a thread?\nA:",
    "Once upon a time, in a small village by the sea, there lived",
    "The three laws of thermodynamics state that",
];

#[test]
fn speculative_greedy_matches_plain_decoding() {
    // FORGE_SPEC_SIZE=4B (or 9B) runs the same test on the bigger models, which need the whole GPU
    let size = std::env::var("FORGE_SPEC_SIZE").unwrap_or_else(|_| "0.8B".into());
    let gguf = models_dir().join(format!(
        "unsloth__Qwen3.5-{size}-GGUF/Qwen3.5-{size}-Q4_K_M.gguf"
    ));
    let hf = models_dir().join(format!("Qwen__Qwen3.5-{size}"));
    if !gguf.exists() || head(&hf).is_none() || Device::count().unwrap_or(0) == 0 {
        eprintln!("skipped: model, MTP head or CUDA device missing");
        return;
    }
    let n = 96;
    // plain decoding first (the engine is dropped before the next one is loaded: the GPU is shared with a trainer)
    let plain: Vec<(String, Usage, FinishReason)> = {
        let e = Engine::load(config(&gguf, &hf, false)).unwrap();
        assert!(!e.info.speculative);
        let r = PROMPTS
            .iter()
            .map(|p| generate(&e, p, greedy(), n, vec![]))
            .collect();
        drop(e);
        r
    };
    std::thread::sleep(std::time::Duration::from_secs(3));
    let e = Engine::load(config(&gguf, &hf, true)).unwrap();
    assert!(e.info.speculative, "the MTP head did not load");
    let (mut cycles, mut accepted, mut same_chars, mut total_chars, mut tokens) =
        (0usize, 0usize, 0usize, 0usize, 0usize);
    for (i, p) in PROMPTS.iter().enumerate() {
        let (text, usage, reason) = generate(&e, p, greedy(), n, vec![]);
        let (ptext, pusage, preason) = &plain[i];
        let common = text
            .chars()
            .zip(ptext.chars())
            .take_while(|(a, b)| a == b)
            .count();
        eprintln!(
            "prompt {i}: {} tokens (plain {}), {} drafts of which {} accepted, {} of {} characters equal to plain decoding, finish {:?}",
            usage.completion_tokens,
            pusage.completion_tokens,
            usage.draft_cycles,
            usage.draft_accepted,
            common,
            ptext.chars().count(),
            reason
        );
        if common < ptext.chars().count() {
            eprintln!(
                "  plain:       {:?}\n  speculative: {:?}",
                &ptext
                    .chars()
                    .skip(common.saturating_sub(20))
                    .take(80)
                    .collect::<String>(),
                &text
                    .chars()
                    .skip(common.saturating_sub(20))
                    .take(80)
                    .collect::<String>()
            );
        }
        cycles += usage.draft_cycles;
        accepted += usage.draft_accepted;
        same_chars += common;
        total_chars += ptext.chars().count();
        tokens += usage.completion_tokens;
        if text == *ptext {
            // the same text: the same ending (a model may stop on its own before the token limit)
            assert_eq!(reason, *preason);
            assert_eq!(usage.completion_tokens, pusage.completion_tokens);
        }
    }
    let rate = accepted as f64 / cycles.max(1) as f64;
    eprintln!("{tokens} tokens in {cycles} verification steps ({:.2} tokens per step), {:.1} % of the drafts accepted, {:.1} % of the characters equal to plain decoding", tokens as f64 / cycles.max(1) as f64, 100.0 * rate, 100.0 * same_chars as f64 / total_chars as f64);
    assert!(cycles > 0);
    assert!(rate > 0.5, "the drafts are rarely accepted: {rate}");
    // near-tied logits may flip a token between the two numerical paths (the verification kernels round differently from the single-token ones), and one flip early in an answer costs the rest of that answer:
    // 77 % of the characters were equal on the RTX 2060 SUPER with 4 of 6 answers identical and the others each diverging at a word that has two fluent continuations ("crew consisted of" / "mission was a
    // joint effort"). Wrong drafts accepted by a broken verification would give unrelated text, far below this bound.
    assert!(same_chars as f64 > 0.6 * total_chars as f64, "speculative decoding differs from plain decoding: {same_chars} of {total_chars} characters equal");

    // sampling: sensible output, fewer drafts accepted than with greedy decoding, the stop string and the token limit still work
    let sampling = SamplingParams {
        temperature: Some(0.7),
        top_k: Some(40),
        top_p: Some(0.95),
        seed: Some(7),
        ..Default::default()
    };
    let (text, usage, reason) = generate(&e, PROMPTS[3], sampling.clone(), 80, vec![]);
    eprintln!(
        "sampled: {} tokens, {} of {} drafts accepted, {:?}",
        usage.completion_tokens, usage.draft_accepted, usage.draft_cycles, text
    );
    assert_eq!(reason, FinishReason::Length);
    assert_eq!(usage.completion_tokens, 80);
    assert!(usage.draft_cycles > 0 && !text.trim().is_empty());
    let (text2, usage2, _) = generate(&e, PROMPTS[3], sampling, 80, vec![]);
    assert_eq!(text, text2, "the same seed must give the same text");
    assert_eq!(usage.draft_accepted, usage2.draft_accepted);
    let (text3, _, reason3) = generate(&e, PROMPTS[0], greedy(), 64, vec![".".into()]);
    eprintln!("stop string: {:?} {:?}", text3, reason3);
    assert_eq!(reason3, FinishReason::Stop);
    assert!(!text3.contains('.'));
}

/// Why does greedy speculative output sometimes differ from plain decoding after many tokens? Because two logits were within the numerical noise of the two kernel paths. This prints, for every prompt,
/// the margin between the best and the second best logit at each greedy step of PLAIN decoding and the smallest margins with the character offset where that token starts
/// (FORGE_SPEC_DIAG=1 FORGE_SPEC_SIZE=9B cargo test --release -p forge-engine --test speculative near_tie -- --nocapture).
#[test]
fn near_tie_diagnosis() {
    if std::env::var("FORGE_SPEC_DIAG").is_err() {
        return;
    }
    let size = std::env::var("FORGE_SPEC_SIZE").unwrap_or_else(|_| "0.8B".into());
    let gguf = models_dir().join(format!(
        "unsloth__Qwen3.5-{size}-GGUF/Qwen3.5-{size}-Q4_K_M.gguf"
    ));
    let hf = models_dir().join(format!("Qwen__Qwen3.5-{size}"));
    if !gguf.exists() || Device::count().unwrap_or(0) == 0 {
        return;
    }
    let tok = forge_tok::Tokenizer::from_file(hf.join("tokenizer.json")).unwrap();
    let model = forge_model::Model::load_native(&gguf).unwrap();
    let gpu = forge_gpu::GpuModel::upload(&model, forge_gpu::Fmt::Native, 0).unwrap();
    drop(model);
    for (pi, p) in PROMPTS.iter().enumerate() {
        let mut sess = forge_gpu::GpuSession::new(&gpu, 512).unwrap();
        let ids = tok.encode(p);
        sess.prefill_batched(&ids).unwrap();
        let mut gaps: Vec<(f32, usize, usize)> = Vec::new();
        let mut text = String::new();
        let mut out: Vec<u32> = Vec::new();
        for step in 0..96 {
            let l = sess.logits().unwrap();
            let (mut b1, mut b2, mut i1) = (f32::NEG_INFINITY, f32::NEG_INFINITY, 0usize);
            for (i, &v) in l.iter().enumerate() {
                if v > b1 {
                    b2 = b1;
                    b1 = v;
                    i1 = i;
                } else if v > b2 {
                    b2 = v;
                }
            }
            gaps.push((b1 - b2, step, text.chars().count()));
            out.push(i1 as u32);
            text = tok.decode(&out, false);
            sess.advance(i1 as u32).unwrap();
        }
        gaps.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        eprintln!(
            "prompt {pi}: smallest logit margins (margin, step, character offset): {:?}",
            &gaps[..4]
        );
    }
}
