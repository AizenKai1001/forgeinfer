//! `forge serve <model>`: the OpenAI-compatible server (forge-serve) over the GPU engine (forge-engine).
//!
//!   forge serve <model> [--native | --q8 | --f16] [--tokenizer FILE] [--host H] [--port P] [--ctx N] [--alias NAME]
//!                       [--chat chatml|smollm2|qwen3|qwen35|qwen35-think] [--api-key KEY] [--queue N]
//!                       [--temp T] [--top-k K] [--top-p P] [--min-p P] [--repeat-penalty R] [--device D]
//!                       [--mtp DIR|FILE] [--no-speculate] [--no-lookup] [--lookup-draft N]
//!                       [--kv fp16|int8|int4|k8v4] [--depth N] [--chunk N] [--threads N] [--host-ffn LAYERS|all] [--vram-reserve MIB] [--prompt-cache-mb MB]
//!   --kv           quantized KV cache (int8 halves its memory with a negligible loss); --depth: drafted tokens per step of the MTP head (2 or 3 is best); --chunk: prompt tokens per prefill pass;
//!   --prompt-cache-mb  host memory for the conversations the GPU has been used away from (default 2048, 0 = off): going back to a long document in an earlier chat copies its KV entries back instead of reading it again;
//!   --no-lookup    no prompt-lookup drafting (on by default, any model: tokens that repeat earlier text are drafted from the text itself and verified in one sweep; --lookup-draft: at most N of them, default 11; a run starts at 7 and doubles after a cycle whose drafts were all right);
//!   --no-speculate no speculative decoding at all (neither the MTP head nor the lookup);
//!   --threads      CPU threads for the feed-forward layers kept in host memory; --host-ffn: force those layers (default: chosen automatically when the model does not fit the GPU);
//!   --vram-reserve MiB of GPU memory left free for other programs.

use forge_chat::ChatFormat;
use forge_engine::{Engine, EngineConfig, SamplingParams, WeightMode};
use forge_serve::ServerConfig;
use std::path::PathBuf;
use std::sync::Arc;

fn flag<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str())
}

fn parse_chat(name: &str) -> Option<ChatFormat> {
    Some(match name {
        "chatml" => ChatFormat::ChatMl,
        "smollm2" => ChatFormat::SmolLm2,
        "llama3" => ChatFormat::Llama3,
        "qwen25" | "qwen2" => ChatFormat::Qwen25,
        "qwen3" => ChatFormat::Qwen3,
        "qwen35" => ChatFormat::Qwen35 {
            thinking_default: false,
        },
        "qwen35-think" => ChatFormat::Qwen35 {
            thinking_default: true,
        },
        _ => return None,
    })
}

pub fn serve(args: &[String]) -> i32 {
    let Some(model) = args.get(1).filter(|a| !a.starts_with("--")) else {
        eprintln!("usage: forge serve <model> [--native | --q8 | --f16] [--tokenizer FILE] [--host H] [--port P] [--ctx N] [--alias NAME] [--chat FORMAT] [--api-key KEY] [--queue N]");
        return 2;
    };
    let model = PathBuf::from(model);
    // a downloaded GGUF is already quantized: its blocks are read as they are unless asked otherwise; our own full-precision checkpoints default to F16
    let is_gguf = model
        .extension()
        .map_or(false, |e| e.eq_ignore_ascii_case("gguf"));
    let weights = if args.iter().any(|a| a == "--native") {
        WeightMode::Native
    } else if args.iter().any(|a| a == "--q8") {
        WeightMode::Q8
    } else if args.iter().any(|a| a == "--f16") || !is_gguf {
        WeightMode::F16
    } else {
        WeightMode::Native
    };
    let chat = match flag(args, "--chat") {
        Some(name) => match parse_chat(name) {
            Some(c) => Some(c),
            None => {
                eprintln!("unknown --chat {name:?} (chatml, smollm2, llama3, qwen25, qwen3, qwen35, qwen35-think)");
                return 2;
            }
        },
        None => None,
    };
    let num = |name: &str| flag(args, name).and_then(|v| v.parse::<f64>().ok());
    let alias = flag(args, "--alias")
        .map(|s| s.to_string())
        .unwrap_or_else(|| {
            model
                .file_stem()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_else(|| "forgeinfer".to_string())
        });
    // Qwen3.5's multi-token-prediction head (not in the GGUF files): `--mtp` names a Hugging Face directory or a safetensors file with the `mtp.*` tensors; a `mtp.safetensors` beside the model is found by itself
    let mtp = flag(args, "--mtp").map(PathBuf::from).or_else(|| {
        let beside = model.with_file_name("mtp.safetensors");
        beside.exists().then_some(beside)
    });
    let cfg = EngineConfig {
        model,
        mtp,
        speculate: !args.iter().any(|a| a == "--no-speculate"),
        tokenizer: flag(args, "--tokenizer").map(PathBuf::from),
        weights,
        n_ctx: flag(args, "--ctx")
            .and_then(|v| v.parse().ok())
            .unwrap_or(4096),
        device: flag(args, "--device")
            .and_then(|v| v.parse().ok())
            .unwrap_or(0),
        alias,
        chat,
        max_queue: flag(args, "--queue")
            .and_then(|v| v.parse().ok())
            .unwrap_or(8),
        defaults: SamplingParams {
            temperature: num("--temp").map(|v| v as f32),
            top_k: num("--top-k").map(|v| v as usize),
            top_p: num("--top-p").map(|v| v as f32),
            min_p: num("--min-p").map(|v| v as f32),
            repetition_penalty: num("--repeat-penalty").map(|v| v as f32),
            seed: None,
            json_object: false,
            json_schema: None,
        },
    };
    // switches the runtime reads from the environment, as flags: an app or a script should not have to know the variable names
    for (name, var) in [
        ("--kv", "FORGE_KV"),
        ("--depth", "FORGE_SPEC_DEPTH"),
        ("--chunk", "FORGE_PREFILL_CHUNK"),
        ("--threads", "FORGE_CPU_THREADS"),
        ("--host-ffn", "FORGE_HOST_FFN"),
        ("--vram-reserve", "FORGE_VRAM_RESERVE_MIB"),
        ("--lookup-draft", "FORGE_PLD_DRAFT"),
        ("--prompt-cache-mb", "FORGE_PROMPT_CACHE_MB"),
    ] {
        if let Some(v) = flag(args, name) {
            std::env::set_var(var, v);
        }
    }
    if args.iter().any(|a| a == "--no-lookup") {
        std::env::set_var("FORGE_PLD", "0");
    }
    let t0 = std::time::Instant::now();
    let engine = match Engine::load(cfg) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("cannot load the model: {e}");
            return 1;
        }
    };
    eprintln!(
        "[serve] {} ready in {:.1} s: {} ({:.0} M parameters, {:.0} MB of weights on the GPU), {:?} chat format, {} tokens of context{}",
        engine.info.alias,
        t0.elapsed().as_secs_f64(),
        engine.info.weights,
        engine.info.n_params as f64 / 1e6,
        engine.info.weight_bytes as f64 / 1e6,
        engine.info.chat,
        engine.info.n_ctx,
        if engine.info.speculative { ", speculative decoding with the MTP head" } else { "" }
    );
    let sc = ServerConfig {
        host: flag(args, "--host").unwrap_or("127.0.0.1").to_string(),
        port: flag(args, "--port")
            .and_then(|v| v.parse().ok())
            .unwrap_or(8080),
        api_key: flag(args, "--api-key").map(|s| s.to_string()),
    };
    match forge_serve::serve(Arc::new(engine), sc) {
        Ok(()) => 0,
        Err(e) => {
            eprintln!("server error: {e}");
            1
        }
    }
}
