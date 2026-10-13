//! `forge serve CKPT --q8` on a training checkpoint: the engine loads the checkpoint through the KV-cache `Decoder` with Q8 weights (the matrices quantized on the device, the trainer's fp32 / fp16
//! weights freed), and the greedy text it streams is the text of a Q8 decoder driven by hand on the same checkpoint (same kernels, same bits, so the same arg-max at every step); a second request that
//! shares a prefix with the first one goes through the prefix reuse. The weights it reports are smaller than the fp16 ones. Needs a CUDA device (skipped without one).

use forge_cuda::Device;
use forge_engine::{
    Engine, EngineConfig, Event, FinishReason, Prompt, Request, SamplingParams, Usage, WeightMode,
};
use forge_tok::tokenizer::bytes_to_unicode;
use forge_train::{config_from_checkpoint, Decoder, TrainConfig, Trainer};

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

fn argmax(z: &[f32]) -> u32 {
    let mut best = 0;
    for (i, &x) in z.iter().enumerate() {
        if x > z[best] {
            best = i;
        }
    }
    best as u32
}

/// A Q8 decoder driven by hand (greedy, the prompt at once, then one token per step): the text of the produced ids and how many there are.
fn reference(
    tr: &Trainer,
    dec: &mut Decoder,
    e: &Engine,
    prompt: &str,
    max_tokens: usize,
) -> (String, usize) {
    let ids = e.tok.encode(prompt);
    dec.reset();
    let mut logits = dec.feed(tr, &ids).unwrap();
    let (mut text, mut n) = (Vec::new(), 0);
    while n < max_tokens && dec.len() + 1 < tr.cfg.seq_len {
        let id = argmax(&logits);
        if id >= 510 {
            break;
        }
        text.extend_from_slice(e.tok.token_bytes(id));
        n += 1;
        logits = dec.feed(tr, &[id]).unwrap();
    }
    (String::from_utf8(text).unwrap(), n)
}

#[test]
fn the_engine_serves_a_checkpoint_with_q8_weights() {
    if Device::count().unwrap_or(0) == 0 {
        eprintln!("skipped: no CUDA device");
        return;
    }
    let dir = std::env::temp_dir().join(format!("forge-q8-engine-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
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
    let ckpt = dir.join("q8.ckpt");
    tr.save_checkpoint(&ckpt, 1024.0).unwrap();
    let n_params = tr.n_params;
    drop(tr);
    let tok_path = dir.join("tokenizer.json");
    std::fs::write(&tok_path, tokenizer_json()).unwrap();

    let engine = Engine::load(EngineConfig {
        model: ckpt.clone(),
        tokenizer: Some(tok_path),
        weights: WeightMode::Q8,
        n_ctx: 256,
        device: 0,
        alias: "q8".into(),
        chat: None,
        max_queue: 2,
        defaults: SamplingParams::default(),
        mtp: None,
        speculate: false,
    })
    .unwrap();
    eprintln!(
        "engine: {} ({} bytes of weights, fp16 would be {})",
        engine.info.weights,
        engine.info.weight_bytes,
        n_params * 2
    );
    assert!(
        engine.info.weights.contains("Q8") && engine.info.weights.contains("KV cache"),
        "the checkpoint must be served through the Q8 KV-cache decoder, got: {}",
        engine.info.weights
    );
    assert!(
        engine.info.weight_bytes < n_params * 2,
        "Q8 weights ({}) must be smaller than fp16 ones ({})",
        engine.info.weight_bytes,
        n_params * 2
    );

    // the reference: the same checkpoint, a Q8 decoder driven by hand
    let mut rc = config_from_checkpoint(&ckpt).unwrap();
    rc.seq_len = 256;
    rc.batch = 1;
    rc.ce_chunk = 32;
    let mut reference_tr = Trainer::new_inference(rc, 0).unwrap();
    reference_tr.load_checkpoint(&ckpt).unwrap();
    let mut reference_dec = Decoder::new(&reference_tr, 512.min(256)).unwrap();
    reference_dec.enable_q8(&mut reference_tr, false).unwrap();

    let first = "the quick brown fox jumps over the lazy dog and then the dog runs after the fox";
    // shares its first 60 characters with the first prompt (prefix reuse), then continues differently
    let second = format!("{} cat sat on a mat", &first[..60]);
    for (i, prompt) in [first.to_string(), second].iter().enumerate() {
        let (text, usage, reason, cached) = generate(&engine, prompt, 48);
        let (want, n) = reference(&reference_tr, &mut reference_dec, &engine, prompt, 48);
        eprintln!(
            "request {i}: {} prompt tokens, {} cached, {} generated ({reason:?}); engine text {text:?}",
            usage.prompt_tokens, cached, usage.completion_tokens
        );
        if i == 1 {
            assert!(cached >= 40, "prefix reuse kept only {cached} positions");
        }
        assert_eq!(
            text, want,
            "request {i}: the engine's greedy text differs from the Q8 decoder driven by hand"
        );
        // the engine counts the end token that stops a generation as produced
        assert!(usage.completion_tokens == n || usage.completion_tokens == n + 1);
        assert!(n >= 10, "the reference generated only {n} tokens");
    }
    let _ = std::fs::remove_dir_all(&dir);
}
