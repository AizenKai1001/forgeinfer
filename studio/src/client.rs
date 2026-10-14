//! The OpenAI-compatible client: the server's model list and streamed chat completions (server-sent events).
//! Works against the ForgeInfer engine, llama.cpp's server, Ollama, LM Studio, vLLM: whatever speaks `/v1/chat/completions`.

use std::io::{BufRead, BufReader};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Sender;
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde_json::{json, Value};

use crate::chats::{Role, Stats};
use crate::settings::{GenParams, ThinkingMode};

#[derive(Clone, Debug, PartialEq)]
pub struct Endpoint {
    /// `http://host:port`, no `/v1`.
    pub base: String,
    pub key: String,
}

impl Endpoint {
    pub fn local(port: u16, key: &str) -> Endpoint {
        Endpoint {
            base: format!("http://127.0.0.1:{port}"),
            key: key.to_string(),
        }
    }

    pub fn url(&self, path: &str) -> String {
        format!("{}/v1/{}", self.base.trim_end_matches('/'), path)
    }
}

fn agent() -> ureq::Agent {
    let config = ureq::Agent::config_builder()
        .timeout_connect(Some(Duration::from_secs(5)))
        .http_status_as_error(false)
        .build();
    ureq::Agent::new_with_config(config)
}

/// The context length in the model list of a ForgeInfer server (`meta.n_ctx` of the first model).
pub fn context_from_models(v: &Value) -> Option<u32> {
    v["data"][0]["meta"]["n_ctx"].as_u64().filter(|n| *n > 0).map(|n| n as u32)
}

/// The context length in the properties of a server (`/props`: ForgeInfer has `n_ctx`, llama.cpp `default_generation_settings.n_ctx`).
pub fn context_from_props(v: &Value) -> Option<u32> {
    v["n_ctx"]
        .as_u64()
        .or_else(|| v["default_generation_settings"]["n_ctx"].as_u64())
        .filter(|n| *n > 0)
        .map(|n| n as u32)
}

/// The context length the server says it was started with: the model list of a ForgeInfer server has it, else `/props`.
pub fn loaded_context(ep: &Endpoint) -> Option<u32> {
    let get = |url: String| -> Option<Value> {
        let mut req = agent().get(url).config().timeout_global(Some(Duration::from_secs(8))).build();
        if !ep.key.is_empty() {
            req = req.header("Authorization", format!("Bearer {}", ep.key));
        }
        let mut resp = req.call().ok()?;
        if !(200..300).contains(&resp.status().as_u16()) {
            return None;
        }
        serde_json::from_str(&resp.body_mut().read_to_string().ok()?).ok()
    };
    get(ep.url("models")).and_then(|v| context_from_models(&v)).or_else(|| get(format!("{}/props", ep.base.trim_end_matches('/'))).and_then(|v| context_from_props(&v)))
}

/// The ids the server offers (`GET /v1/models`).
pub fn list_models(ep: &Endpoint) -> Result<Vec<String>, String> {
    let mut req = agent()
        .get(ep.url("models"))
        .config()
        .timeout_global(Some(Duration::from_secs(8)))
        .build();
    if !ep.key.is_empty() {
        req = req.header("Authorization", format!("Bearer {}", ep.key));
    }
    let mut resp = req
        .call()
        .map_err(|e| format!("cannot reach {}: {e}", ep.base))?;
    let status = resp.status().as_u16();
    let body = resp
        .body_mut()
        .read_to_string()
        .map_err(|e| e.to_string())?;
    if !(200..300).contains(&status) {
        return Err(error_text(status, &body));
    }
    let v: Value = serde_json::from_str(&body).map_err(|e| format!("not a model list: {e}"))?;
    let mut ids: Vec<String> = v["data"]
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|m| m["id"].as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default();
    if ids.is_empty() {
        // Ollama's native shape
        if let Some(a) = v["models"].as_array() {
            ids = a
                .iter()
                .filter_map(|m| {
                    m["name"]
                        .as_str()
                        .or(m["model"].as_str())
                        .map(str::to_string)
                })
                .collect();
        }
    }
    Ok(ids)
}

fn error_text(status: u16, body: &str) -> String {
    if let Ok(v) = serde_json::from_str::<Value>(body) {
        if let Some(m) = v["error"]["message"]
            .as_str()
            .or(v["error"].as_str())
            .or(v["message"].as_str())
        {
            return format!("HTTP {status}: {m}");
        }
    }
    let short: String = body.chars().take(300).collect();
    format!("HTTP {status}: {short}")
}

#[derive(Clone, Debug)]
pub struct ChatParams {
    pub model: String,
    pub messages: Vec<(Role, String)>,
    pub gen: GenParams,
}

/// How far the server has read the prompt (`prompt_progress` of the stream, llama.cpp's `return_progress`): `processed` of `total` tokens are in its cache, the first `cached` of them from an earlier request.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PromptProgress {
    pub processed: u32,
    pub total: u32,
    pub cached: u32,
    pub time_ms: f32,
}

impl PromptProgress {
    /// The share of the prompt that is in the cache.
    pub fn fraction(&self) -> f32 {
        (self.processed as f32 / self.total.max(1) as f32).clamp(0.0, 1.0)
    }

    /// Seconds the rest of the prompt will take at the speed so far (None before there is a speed to go by).
    pub fn seconds_left(&self) -> Option<f32> {
        let done = self.processed.saturating_sub(self.cached) as f32;
        if done < 1.0 || self.time_ms < 50.0 {
            return None;
        }
        Some((self.total.saturating_sub(self.processed)) as f32 / (done / (self.time_ms / 1e3)))
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum ChatEvent {
    Content(String),
    Reasoning(String),
    /// The server is still reading the prompt.
    Progress(PromptProgress),
    Finished(Stats),
    Failed(String),
}

/// The request body: the standard fields plus the samplers local servers understand.
pub fn chat_body(p: &ChatParams, extended: bool) -> Value {
    let messages: Vec<Value> = p
        .messages
        .iter()
        .map(|(r, c)| json!({"role": r.api(), "content": c}))
        .collect();
    let mut body = json!({
        "model": p.model,
        "messages": messages,
        "stream": true,
        "stream_options": {"include_usage": true},
        "temperature": p.gen.temperature,
        "top_p": p.gen.top_p,
    });
    if p.gen.max_tokens > 0 {
        body["max_tokens"] = json!(p.gen.max_tokens);
    }
    if extended {
        // a long prompt takes seconds to read: servers that know the field report how far they are (llama.cpp, ForgeInfer)
        body["return_progress"] = json!(true);
        if p.gen.top_k > 0 {
            body["top_k"] = json!(p.gen.top_k);
        }
        if p.gen.min_p > 0.0 {
            body["min_p"] = json!(p.gen.min_p);
        }
        body["repetition_penalty"] = json!(p.gen.repeat_penalty);
        body["repeat_penalty"] = json!(p.gen.repeat_penalty);
        match p.gen.thinking {
            ThinkingMode::Default => {}
            ThinkingMode::On => body["chat_template_kwargs"] = json!({"enable_thinking": true}),
            ThinkingMode::Off => body["chat_template_kwargs"] = json!({"enable_thinking": false}),
        }
    }
    body
}

/// What one `data:` line of the stream carries.
#[derive(Debug, Default, PartialEq)]
pub struct Chunk {
    pub content: String,
    pub reasoning: String,
    pub finish: Option<String>,
    pub completion_tokens: Option<u32>,
    pub prompt_tokens: Option<u32>,
    /// Prompt tokens the server did not have to read again (OpenAI's `prompt_tokens_details.cached_tokens`, llama.cpp's `timings.cache_n`).
    pub cached_tokens: Option<u32>,
    /// The server's own decode speed, when it reports one.
    pub tok_per_s: Option<f32>,
    /// A report of how far the server has read the prompt.
    pub progress: Option<PromptProgress>,
}

pub fn parse_chunk(data: &str) -> Option<Chunk> {
    let v: Value = serde_json::from_str(data).ok()?;
    let mut c = Chunk::default();
    let choice = &v["choices"][0];
    let delta = &choice["delta"];
    if let Some(s) = delta["content"].as_str() {
        c.content = s.to_string();
    }
    if let Some(s) = delta["reasoning_content"]
        .as_str()
        .or(delta["reasoning"].as_str())
    {
        c.reasoning = s.to_string();
    }
    c.finish = choice["finish_reason"].as_str().map(str::to_string);
    if let Some(u) = v.get("usage").filter(|u| u.is_object()) {
        c.completion_tokens = u["completion_tokens"].as_u64().map(|n| n as u32);
        c.prompt_tokens = u["prompt_tokens"].as_u64().map(|n| n as u32);
        c.cached_tokens = u["prompt_tokens_details"]["cached_tokens"].as_u64().map(|n| n as u32);
    }
    if let Some(p) = v.get("prompt_progress").filter(|p| p.is_object()) {
        if let (Some(total), Some(processed)) = (p["total"].as_u64(), p["processed"].as_u64()) {
            c.progress = Some(PromptProgress {
                processed: processed as u32,
                total: total as u32,
                cached: p["cache"].as_u64().unwrap_or(0) as u32,
                time_ms: p["time_ms"].as_f64().unwrap_or(0.0) as f32,
            });
        }
    }
    if let Some(t) = v.get("timings").filter(|t| t.is_object()) {
        c.tok_per_s = t["predicted_per_second"].as_f64().map(|x| x as f32);
        if c.completion_tokens.is_none() {
            c.completion_tokens = t["predicted_n"].as_u64().map(|n| n as u32);
        }
        if c.prompt_tokens.is_none() {
            c.prompt_tokens = t["prompt_n"].as_u64().map(|n| n as u32);
        }
        if c.cached_tokens.is_none() {
            c.cached_tokens = t["cache_n"].as_u64().map(|n| n as u32);
        }
    }
    Some(c)
}

/// Run one chat completion on the calling thread, reporting through `emit`. Returns when the stream ends, fails or `cancel` is set.
pub fn stream_chat(
    ep: &Endpoint,
    p: &ChatParams,
    cancel: &AtomicBool,
    emit: &mut dyn FnMut(ChatEvent),
) {
    let t0 = Instant::now();
    let mut extended = true;
    let resp = loop {
        let body = chat_body(p, extended);
        let mut req = agent()
            .post(ep.url("chat/completions"))
            .header("Content-Type", "application/json");
        if !ep.key.is_empty() {
            req = req.header("Authorization", format!("Bearer {}", ep.key));
        }
        let mut resp = match req.send_json(&body) {
            Ok(r) => r,
            Err(e) => {
                emit(ChatEvent::Failed(format!(
                    "cannot reach the server at {}: {e}",
                    ep.base
                )));
                return;
            }
        };
        let status = resp.status().as_u16();
        if (200..300).contains(&status) {
            break resp;
        }
        let text = resp.body_mut().read_to_string().unwrap_or_default();
        // a strict server that rejects the extra samplers: try again with the standard fields only
        if extended && (status == 400 || status == 422) && {
            let low = text.to_lowercase();
            low.contains("unrecognized")
                || low.contains("unknown")
                || low.contains("extra")
                || low.contains("not permitted")
                || low.contains("unsupported")
        } {
            extended = false;
            continue;
        }
        emit(ChatEvent::Failed(error_text(status, &text)));
        return;
    };
    let mut reader = BufReader::new(resp.into_body().into_reader());
    let mut line = String::new();
    let mut first: Option<f32> = None;
    let mut pieces = 0u32;
    let mut finish = String::new();
    let mut usage_completion: Option<u32> = None;
    let mut usage_prompt: Option<u32> = None;
    let mut usage_cached: Option<u32> = None;
    let mut server_speed: Option<f32> = None;
    loop {
        if cancel.load(Ordering::Relaxed) {
            return;
        }
        line.clear();
        match reader.read_line(&mut line) {
            Ok(0) => break,
            Ok(_) => {}
            Err(e) => {
                if cancel.load(Ordering::Relaxed) {
                    return;
                }
                emit(ChatEvent::Failed(format!("the connection broke: {e}")));
                return;
            }
        }
        let l = line.trim();
        let Some(data) = l.strip_prefix("data:") else {
            continue;
        };
        let data = data.trim();
        if data == "[DONE]" {
            break;
        }
        let Some(chunk) = parse_chunk(data) else {
            continue;
        };
        if let Some(p) = chunk.progress {
            emit(ChatEvent::Progress(p));
        }
        if !chunk.reasoning.is_empty() || !chunk.content.is_empty() {
            if first.is_none() {
                first = Some(t0.elapsed().as_secs_f32());
            }
            pieces += 1;
        }
        if !chunk.reasoning.is_empty() {
            emit(ChatEvent::Reasoning(chunk.reasoning));
        }
        if !chunk.content.is_empty() {
            emit(ChatEvent::Content(chunk.content));
        }
        if let Some(f) = chunk.finish {
            finish = f;
        }
        usage_completion = chunk.completion_tokens.or(usage_completion);
        usage_prompt = chunk.prompt_tokens.or(usage_prompt);
        usage_cached = chunk.cached_tokens.or(usage_cached);
        server_speed = chunk.tok_per_s.or(server_speed);
    }
    if cancel.load(Ordering::Relaxed) {
        return;
    }
    let total = t0.elapsed().as_secs_f32();
    let tokens = usage_completion.unwrap_or(pieces);
    let first_token_s = first.unwrap_or(total);
    let decode_s = (total - first_token_s).max(0.001);
    let tok_per_s = server_speed.unwrap_or_else(|| {
        if tokens > 1 {
            (tokens - 1) as f32 / decode_s
        } else {
            0.0
        }
    });
    emit(ChatEvent::Finished(Stats {
        completion_tokens: tokens,
        prompt_tokens: usage_prompt.unwrap_or(0),
        cached_tokens: usage_cached.unwrap_or(0),
        tok_per_s,
        first_token_s,
        total_s: total,
        finish,
    }));
}

/// Start a completion on its own thread. Events arrive tagged with `request`; setting `cancel` stops reading.
pub fn spawn_chat<T: Send + 'static>(
    ep: Endpoint,
    p: ChatParams,
    cancel: Arc<AtomicBool>,
    tx: Sender<T>,
    wrap: impl Fn(ChatEvent) -> T + Send + 'static,
    wake: impl Fn() + Send + 'static,
) {
    std::thread::Builder::new()
        .name("chat-stream".into())
        .spawn(move || {
            let mut emit = |ev: ChatEvent| {
                let _ = tx.send(wrap(ev));
                wake();
            };
            stream_chat(&ep, &p, &cancel, &mut emit);
        })
        .ok();
}

/// A reply that carries its reasoning inline as `<think>...</think>` (servers that do not split it out): the three parts.
pub fn split_think(text: &str) -> (String, String, bool) {
    let t = text.trim_start();
    let Some(rest) = t.strip_prefix("<think>") else {
        return (String::new(), text.to_string(), false);
    };
    match rest.find("</think>") {
        Some(end) => (
            rest[..end].trim().to_string(),
            rest[end + "</think>".len()..].trim_start().to_string(),
            false,
        ),
        None => (rest.trim_start().to_string(), String::new(), true),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn content_reasoning_and_finish_chunks() {
        let c = parse_chunk(
            r#"{"choices":[{"index":0,"delta":{"content":"Hel"},"finish_reason":null}]}"#,
        )
        .unwrap();
        assert_eq!(c.content, "Hel");
        assert_eq!(c.finish, None);
        let r = parse_chunk(r#"{"choices":[{"delta":{"reasoning_content":"hmm"}}]}"#).unwrap();
        assert_eq!(r.reasoning, "hmm");
        let f = parse_chunk(r#"{"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":20,"completion_tokens":8},"timings":{"predicted_per_second":363.4}}"#).unwrap();
        assert_eq!(f.finish.as_deref(), Some("stop"));
        assert_eq!((f.prompt_tokens, f.completion_tokens), (Some(20), Some(8)));
        assert!((f.tok_per_s.unwrap() - 363.4).abs() < 0.01);
        assert_eq!(f.cached_tokens, None);
        // how much of the prompt the engine already had: OpenAI's field, or llama.cpp's
        let c = parse_chunk(r#"{"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":6399,"completion_tokens":24,"prompt_tokens_details":{"cached_tokens":6354}}}"#).unwrap();
        assert_eq!((c.prompt_tokens, c.cached_tokens), (Some(6399), Some(6354)));
        let l = parse_chunk(r#"{"choices":[{"delta":{},"finish_reason":"stop"}],"timings":{"cache_n":120,"prompt_n":30,"predicted_per_second":50.0}}"#).unwrap();
        assert_eq!((l.prompt_tokens, l.cached_tokens), (Some(30), Some(120)));
        assert!(parse_chunk("not json").is_none());
    }

    #[test]
    fn the_context_a_server_started_with() {
        let models: Value = serde_json::from_str(r#"{"object":"list","data":[{"id":"m","meta":{"n_ctx":262144,"n_vocab":248320}}]}"#).unwrap();
        assert_eq!(context_from_models(&models), Some(262144));
        let none: Value = serde_json::from_str(r#"{"data":[{"id":"m"}]}"#).unwrap();
        assert_eq!(context_from_models(&none), None);
        let props: Value = serde_json::from_str(r#"{"model":"m","n_ctx":16384}"#).unwrap();
        assert_eq!(context_from_props(&props), Some(16384));
        let llama: Value = serde_json::from_str(r#"{"default_generation_settings":{"n_ctx":8192},"total_slots":1}"#).unwrap();
        assert_eq!(context_from_props(&llama), Some(8192));
        assert_eq!(context_from_props(&serde_json::from_str::<Value>(r#"{"n_ctx":0}"#).unwrap()), None);
    }

    #[test]
    fn prompt_progress_chunks() {
        let c = parse_chunk(r#"{"choices":[{"index":0,"delta":{},"finish_reason":null}],"prompt_progress":{"total":27000,"cache":1000,"processed":13288,"time_ms":3000.0}}"#).unwrap();
        let p = c.progress.unwrap();
        assert_eq!((p.processed, p.total, p.cached), (13288, 27000, 1000));
        assert!(c.content.is_empty() && c.reasoning.is_empty() && c.finish.is_none());
        assert!((p.fraction() - 0.4922).abs() < 0.001);
        // 12,288 tokens in 3 s: the remaining 13,712 take 3.35 s
        assert!((p.seconds_left().unwrap() - 3.348).abs() < 0.01);
        // no speed yet: no estimate
        let early = PromptProgress { processed: 1024, total: 9000, cached: 1024, time_ms: 10.0 };
        assert!(early.seconds_left().is_none());
        assert!(parse_chunk(r#"{"choices":[{"delta":{"content":"x"}}]}"#).unwrap().progress.is_none());
    }

    #[test]
    fn the_request_body_carries_the_samplers() {
        let p = ChatParams {
            model: "m".into(),
            messages: vec![(Role::User, "hi".into())],
            gen: GenParams {
                max_tokens: 100,
                ..Default::default()
            },
        };
        let b = chat_body(&p, true);
        assert_eq!(b["stream"], true);
        assert_eq!(b["max_tokens"], 100);
        assert_eq!(b["top_k"], 40);
        assert!(
            b.get("chat_template_kwargs").is_none(),
            "the model's own default is left alone"
        );
        let mut q = p.clone();
        q.gen.thinking = ThinkingMode::Off;
        assert_eq!(
            chat_body(&q, true)["chat_template_kwargs"]["enable_thinking"],
            false
        );
        assert_eq!(b["return_progress"], true);
        let plain = chat_body(&p, false);
        assert!(plain.get("return_progress").is_none());
        assert!(plain.get("top_k").is_none());
        assert!(plain.get("chat_template_kwargs").is_none());
    }

    #[test]
    fn inline_think_tags_are_split() {
        assert_eq!(
            split_think("<think>step one</think>\n\nAnswer"),
            ("step one".into(), "Answer".into(), false)
        );
        assert_eq!(
            split_think("<think>still going"),
            ("still going".into(), String::new(), true)
        );
        assert_eq!(split_think("plain"), (String::new(), "plain".into(), false));
    }
}
