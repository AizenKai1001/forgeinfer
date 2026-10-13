//! A training checkpoint with sliding-window attention (`window`, `window_layers` in its header) is served by the engine through the KV-cache `Decoder` (not the no-cache fallback), and the
//! greedy text it streams is the text the trainer's own forward pass (`Trainer::logits_at`, which applies the window mask of the training kernels) produces token by token. The context is longer
//! than the window, so the window matters; a second request that shares a prefix with the first one goes through the prefix reuse. Needs a CUDA device (skipped without one).

use forge_cuda::Device;
use forge_engine::{
    Engine, EngineConfig, Event, FinishReason, Prompt, Request, SamplingParams, Usage, WeightMode,
};
use forge_tok::tokenizer::bytes_to_unicode;
use forge_train::{config_from_checkpoint, TrainConfig, Trainer};

/// A tokenizer in which every one of the 512 ids has ASCII text: ids 0..25 are the letters a-z, 26 is the space, 27..509 are the three-letter words AAA, AAB, ...; 510 and 511 are the special
/// tokens that end a generation. No merges: a prompt of letters and spaces encodes character by character.
fn tokenizer_json() -> String {
    let b2u = bytes_to_unicode();
    let mut vocab: Vec<(String, usize)> = Vec::new();
    for c in b'a'..=b'z' {
        vocab.push((b2u[c as usize].to_string(), vocab.len()));
    }
    vocab.push((b2u[b' ' as usize].to_string(), 26));
    for i in 0..483usize {
        let w: String = [i / 676, (i / 26) % 26, i % 26]
            .iter()
            .map(|&d| (b'A' + d as u8) as char)
            .collect();
        vocab.push((w, 27 + i));
    }
    assert_eq!(vocab.len(), 510);
    let entries: Vec<String> = vocab
        .iter()
        .map(|(t, id)| format!("\"{t}\":{id}"))
        .collect();
    format!(
        "{{\"model\":{{\"type\":\"BPE\",\"vocab\":{{{}}},\"merges\":[]}},\"added_tokens\":[{{\"id\":510,\"content\":\"<|im_end|>\",\"special\":true}},{{\"id\":511,\"content\":\"<|endoftext|>\",\"special\":true}}],\"pre_tokenizer\":{{\"type\":\"ByteLevel\",\"add_prefix_space\":false}}}}",
        entries.join(",")
    )
}

fn greedy() -> SamplingParams {
    SamplingParams {
        temperature: Some(0.0),
        top_k: Some(1),
        repetition_penalty: Some(1.0),
        ..Default::default()
    }
}

fn generate(e: &Engine, prompt: &str, max_tokens: usize) -> (String, Usage, FinishReason, usize) {
    let g = e
        .start(Request {
            prompt: Prompt::Raw(prompt.to_string()),
            sampling: greedy(),
            max_tokens: Some(max_tokens),
            stop: vec![],
        })
        .unwrap();
    let (mut text, mut cached) = (String::new(), 0);
    for ev in g.rx.iter() {
        match ev {
            Event::Started { cached_tokens, .. } => cached = cached_tokens,
            Event::Content(s) => text.push_str(&s),
            Event::Finished { usage, reason, .. } => return (text, usage, reason, cached),
            Event::Error(err) => panic!("engine error: {err}"),
            _ => {}
        }
    }
    panic!("the engine stream ended without a Finished event");
}

/// The trainer's forward pass over the whole text for every new token, greedy; the text of the produced ids and how many there are.
fn reference(tr: &mut Trainer, e: &Engine, prompt: &str, max_tokens: usize) -> (String, usize) {
    let mut seq = e.tok.encode(prompt);
    let (mut text, mut n) = (Vec::new(), 0);
    while n < max_tokens && seq.len() < tr.cfg.seq_len {
        let z = tr.logits_at(&seq, seq.len()).unwrap();
        let mut best = 0;
        for (i, &x) in z.iter().enumerate() {
            if x > z[best] {
                best = i;
            }
        }
        let id = best as u32;
        if id >= 510 {
            break;
        }
        text.extend_from_slice(e.tok.token_bytes(id));
        seq.push(id);
        n += 1;
    }
    (String::from_utf8(text).unwrap(), n)
}

#[test]
fn the_engine_serves_a_window_checkpoint_with_the_kv_decoder() {
    if Device::count().unwrap_or(0) == 0 {
        eprintln!("skipped: no CUDA device");
        return;
    }
    let dir = std::env::temp_dir().join(format!("forge-window-engine-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    // a random 4-layer model, window 16 in layers 0, 1, 2 (`auto`); the engine serves 256 positions
    let window_layers = TrainConfig::auto_window_layers(4);
    let cfg = TrainConfig {
        n_layer: 4,
        n_embd: 128,
        n_head: 2,
        n_kv: 1,
        head_dim: 64,
        n_ff: 256,
        vocab: 512,
        seq_len: 256,
        batch: 1,
        qk_norm: true,
        ce_chunk: 64,
        relu2: true,
        lambdas: true,
        ve_layers: vec![0, 3],
        window: 16,
        window_layers,
        ..TrainConfig::default()
    };
    let mut tr = Trainer::new(cfg, 0).unwrap();
    tr.init_random(21).unwrap();
    // larger weights (not the norm gains and the lambdas): a sharper, more varied output than the all-alike text of the default initialisation
    let mut named = tr.params_named().unwrap();
    for (k, v) in named.iter_mut() {
        if !k.contains("norm") && !k.contains("lam") {
            v.iter_mut().for_each(|x| *x *= 4.0);
        }
    }
    tr.set_params(&named).unwrap();
    let ckpt = dir.join("window.ckpt");
    tr.save_checkpoint(&ckpt, 1024.0).unwrap();
    drop(tr);
    let tok_path = dir.join("tokenizer.json");
    std::fs::write(&tok_path, tokenizer_json()).unwrap();

    let engine = Engine::load(EngineConfig {
        model: ckpt.clone(),
        tokenizer: Some(tok_path),
        weights: WeightMode::F16,
        n_ctx: 256,
        device: 0,
        alias: "window".into(),
        chat: None,
        max_queue: 2,
        defaults: SamplingParams::default(),
        mtp: None,
        speculate: false,
    })
    .unwrap();
    eprintln!("engine: {}", engine.info.weights);
    assert!(
        engine.info.weights.contains("KV cache") && !engine.info.weights.contains("without"),
        "the window checkpoint must be served through the KV-cache decoder, got: {}",
        engine.info.weights
    );

    // the reference trainer: the same checkpoint, the same context length
    let mut rc = config_from_checkpoint(&ckpt).unwrap();
    assert_eq!((rc.window, rc.window_layers.clone()), (16, vec![0, 1, 2]));
    rc.seq_len = 256;
    rc.batch = 1;
    rc.ce_chunk = 32;
    let mut reference_tr = Trainer::new_inference(rc, 0).unwrap();
    reference_tr.load_checkpoint(&ckpt).unwrap();

    let first = "the quick brown fox jumps over the lazy dog and then the dog runs after the fox";
    // shares its first 60 characters with the first prompt (prefix reuse), then continues differently
    let second = format!("{} cat sat on a mat", &first[..60]);
    for (i, prompt) in [first.to_string(), second].iter().enumerate() {
        let (text, usage, reason, cached) = generate(&engine, prompt, 48);
        let (want, n) = reference(&mut reference_tr, &engine, prompt, 48);
        eprintln!(
            "request {i}: {} prompt tokens, {} cached, {} generated ({reason:?}); engine text {text:?}",
            usage.prompt_tokens, cached, usage.completion_tokens
        );
        assert!(
            usage.prompt_tokens > 16,
            "the prompt must be longer than the window"
        );
        if i == 1 {
            assert!(cached >= 40, "prefix reuse kept only {cached} positions");
        }
        assert_eq!(
            text, want,
            "request {i}: the engine's greedy text differs from the trainer's forward pass"
        );
        assert_eq!(usage.completion_tokens, n);
        assert!(n >= 10, "the reference generated only {n} tokens");
    }
    let _ = std::fs::remove_dir_all(&dir);
}
