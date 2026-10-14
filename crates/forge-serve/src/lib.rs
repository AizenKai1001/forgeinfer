//! An OpenAI-compatible HTTP server over `forge-engine`, std only. One thread per connection, `Connection: close` always.
//!
//!   GET  /health, /v1/health        {"status": "ok"}
//!   GET  /v1/models, /props         the served model
//!   POST /v1/chat/completions       messages, tools, streaming (SSE) with reasoning_content and tool_calls deltas, usage
//!   POST /v1/completions            raw text completion
//!   POST /apply-template            the rendered prompt of a chat request (llama.cpp's route: the oracle for template tests)
//!   POST /tokenize, /detokenize     llama.cpp shapes
//!
//! The wire shapes follow what Ember and the OpenAI SDKs read (research note "ForgeInfer Research - Ember Integration and Serving Spec"): first chunk with the assistant role, `reasoning_content` for
//! thinking text, tool calls as one chunk with id / name and empty arguments followed by argument fragments, a final chunk with the finish reason, optional usage, `[DONE]`; errors as JSON with the
//! HTTP status chosen so that Ember's text-based mapping works; an SSE comment every 15 s while a request waits; closing the socket cancels the generation. A streamed request with
//! `return_progress: true` (llama.cpp's field) also gets a chunk with `prompt_progress` {total, cache, processed, time_ms} after every 1,024 tokens of a long prompt, before the first token.

use forge_core::Json;
use forge_engine::{Engine, EngineError, Event, FinishReason, Prompt, Request, SamplingParams};
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::mpsc::RecvTimeoutError;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub api_key: Option<String>,
}

fn s(v: &str) -> Json {
    Json::Str(v.to_string())
}

fn obj(fields: Vec<(&str, Json)>) -> Json {
    Json::Obj(
        fields
            .into_iter()
            .map(|(k, v)| (k.to_string(), v))
            .collect(),
    )
}

fn to_text(j: &Json) -> String {
    let mut out = String::new();
    j.write(&mut out);
    out
}

fn now() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

fn hex_id() -> String {
    let t = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
    let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    format!(
        "{:012x}{:012x}",
        t & 0xFFFF_FFFF_FFFF,
        n.wrapping_mul(0x9E37_79B9_7F4A_7C15) & 0xFFFF_FFFF_FFFF
    )
}

struct HttpRequest {
    method: String,
    path: String,
    headers: Vec<(String, String)>,
    body: Vec<u8>,
}

impl HttpRequest {
    fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(k, _)| k == name)
            .map(|(_, v)| v.as_str())
    }
}

enum ReadError {
    Closed,
    TooLarge,
    Bad(&'static str),
}

const MAX_BODY: usize = 8 << 20;
const MAX_CONNECTIONS: usize = 256;

fn read_request(stream: &mut TcpStream) -> Result<HttpRequest, ReadError> {
    let mut buf: Vec<u8> = Vec::with_capacity(4096);
    let mut chunk = [0u8; 4096];
    let head_end = loop {
        if let Some(p) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
            break p;
        }
        if buf.len() > 64 * 1024 {
            return Err(ReadError::Bad("headers too large"));
        }
        let n = stream.read(&mut chunk).map_err(|_| ReadError::Closed)?;
        if n == 0 {
            return Err(ReadError::Closed);
        }
        buf.extend_from_slice(&chunk[..n]);
    };
    let head = String::from_utf8_lossy(&buf[..head_end]).to_string();
    let mut lines = head.split("\r\n");
    let first = lines.next().unwrap_or("");
    let mut parts = first.split(' ');
    let method = parts.next().unwrap_or("").to_string();
    let target = parts.next().unwrap_or("/").to_string();
    let path = target.split('?').next().unwrap_or("/").to_string();
    let headers: Vec<(String, String)> = lines
        .filter_map(|l| {
            l.split_once(':')
                .map(|(k, v)| (k.trim().to_lowercase(), v.trim().to_string()))
        })
        .collect();
    let mut req = HttpRequest {
        method,
        path,
        headers,
        body: Vec::new(),
    };
    let mut body = buf[head_end + 4..].to_vec();
    if req
        .header("expect")
        .map(|v| v.eq_ignore_ascii_case("100-continue"))
        .unwrap_or(false)
    {
        let _ = stream.write_all(b"HTTP/1.1 100 Continue\r\n\r\n");
    }
    if req
        .header("transfer-encoding")
        .map(|v| v.to_lowercase().contains("chunked"))
        .unwrap_or(false)
    {
        // chunked request body
        let mut out = Vec::new();
        loop {
            // read a chunk-size line
            let line_end = loop {
                if let Some(p) = body.windows(2).position(|w| w == b"\r\n") {
                    break p;
                }
                let n = stream.read(&mut chunk).map_err(|_| ReadError::Closed)?;
                if n == 0 {
                    return Err(ReadError::Closed);
                }
                body.extend_from_slice(&chunk[..n]);
            };
            let size = usize::from_str_radix(
                String::from_utf8_lossy(&body[..line_end])
                    .split(';')
                    .next()
                    .unwrap_or("0")
                    .trim(),
                16,
            )
            .map_err(|_| ReadError::Bad("bad chunk size"))?;
            body.drain(..line_end + 2);
            if size == 0 {
                break;
            }
            if out.len() + size > MAX_BODY {
                return Err(ReadError::TooLarge);
            }
            while body.len() < size + 2 {
                let n = stream.read(&mut chunk).map_err(|_| ReadError::Closed)?;
                if n == 0 {
                    return Err(ReadError::Closed);
                }
                body.extend_from_slice(&chunk[..n]);
            }
            out.extend_from_slice(&body[..size]);
            body.drain(..size + 2);
        }
        req.body = out;
    } else {
        let len: usize = req
            .header("content-length")
            .and_then(|v| v.parse().ok())
            .unwrap_or(0);
        if len > MAX_BODY {
            return Err(ReadError::TooLarge);
        }
        while body.len() < len {
            let n = stream.read(&mut chunk).map_err(|_| ReadError::Closed)?;
            if n == 0 {
                return Err(ReadError::Closed);
            }
            body.extend_from_slice(&chunk[..n]);
        }
        body.truncate(len);
        req.body = body;
    }
    Ok(req)
}

fn status_text(code: u16) -> &'static str {
    match code {
        200 => "OK",
        400 => "Bad Request",
        401 => "Unauthorized",
        404 => "Not Found",
        405 => "Method Not Allowed",
        413 => "Payload Too Large",
        429 => "Too Many Requests",
        500 => "Internal Server Error",
        503 => "Service Unavailable",
        _ => "OK",
    }
}

fn respond_json(stream: &mut TcpStream, code: u16, body: &Json, extra: &str) {
    let text = to_text(body);
    let head = format!("HTTP/1.1 {code} {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n{extra}\r\n", status_text(code), text.len());
    let _ = stream.write_all(head.as_bytes());
    let _ = stream.write_all(text.as_bytes());
    let _ = stream.flush();
}

fn error_body(message: &str, kind: &str, code: u16) -> Json {
    obj(vec![(
        "error",
        obj(vec![
            ("message", s(message)),
            ("type", s(kind)),
            ("param", Json::Null),
            ("code", Json::Int(code as i64)),
        ]),
    )])
}

fn respond_error(stream: &mut TcpStream, code: u16, kind: &str, message: &str) {
    let extra = if code == 429 {
        "Retry-After: 5\r\n"
    } else {
        ""
    };
    respond_json(stream, code, &error_body(message, kind, code), extra);
}

fn engine_error(stream: &mut TcpStream, e: &EngineError) {
    match e {
        EngineError::InvalidRequest(m) => respond_error(stream, 400, "invalid_request_error", m),
        EngineError::Template(m) => respond_error(stream, 400, "invalid_request_error", m),
        EngineError::ContextOverflow {
            prompt_tokens,
            n_ctx,
        } => {
            let mut b = error_body(&e.to_string(), "exceed_context_size_error", 400);
            if let Json::Obj(o) = &mut b {
                if let Json::Obj(inner) = &mut o[0].1 {
                    inner.push(("n_prompt_tokens".into(), Json::Int(*prompt_tokens as i64)));
                    inner.push(("n_ctx".into(), Json::Int(*n_ctx as i64)));
                }
            }
            respond_json(stream, 400, &b, "");
        }
        EngineError::Busy => respond_error(stream, 429, "rate_limit_error", &e.to_string()),
        EngineError::Device(m) => respond_error(stream, 500, "server_error", m),
    }
}

// ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// request parsing

/// message contents as plain strings: a list of text parts is joined, null stays null
fn normalize_messages(messages: &[Json]) -> Result<Vec<Json>, String> {
    let mut out = Vec::new();
    for m in messages {
        let fields = m.as_object().ok_or("every message must be an object")?;
        let role = m
            .get("role")
            .and_then(|r| r.as_str())
            .ok_or("a message without a role")?;
        let role = match role {
            "developer" => "system",
            "system" | "user" | "assistant" | "tool" => role,
            other => return Err(format!("unknown message role {other:?}")),
        };
        let mut nf: Vec<(String, Json)> = Vec::new();
        for (k, v) in fields {
            match k.as_str() {
                "role" => nf.push((k.clone(), s(role))),
                "content" => match v {
                    Json::Arr(parts) => {
                        let mut t = String::new();
                        for p in parts {
                            if let Some(x) = p.get("text").and_then(|x| x.as_str()) {
                                t.push_str(x);
                            }
                        }
                        nf.push((k.clone(), Json::Str(t)));
                    }
                    other => nf.push((k.clone(), other.clone())),
                },
                _ => nf.push((k.clone(), v.clone())),
            }
        }
        out.push(Json::Obj(nf));
    }
    Ok(out)
}

fn sampling_from(j: &Json) -> SamplingParams {
    SamplingParams {
        temperature: j
            .get("temperature")
            .and_then(|v| v.as_f64())
            .map(|v| v as f32),
        top_p: j.get("top_p").and_then(|v| v.as_f64()).map(|v| v as f32),
        top_k: j.get("top_k").and_then(|v| v.as_usize()),
        min_p: j.get("min_p").and_then(|v| v.as_f64()).map(|v| v as f32),
        repetition_penalty: j
            .get("repetition_penalty")
            .and_then(|v| v.as_f64())
            .map(|v| v as f32),
        seed: j.get("seed").and_then(|v| v.as_u64()),
        json_object: false,
        json_schema: None,
    }
}

/// What `response_format` asks for.
enum Format {
    Text,
    /// `{"type": "json_object"}`: any JSON object
    Object,
    /// `{"type": "json_schema", "json_schema": {"name": ..., "schema": {...}}}`: a document that follows the schema
    Schema(std::sync::Arc<forge_engine::constrain::SchemaDoc>),
}

/// `response_format`: `text` (or nothing) is the default; `json_object` and `json_schema` constrain the answer (the schema is compiled here, so that an unsupported one is a 400 and not a
/// failure in the middle of a generation).
fn response_format(body: &Json) -> Result<Format, String> {
    let Some(rf) = body.get("response_format") else { return Ok(Format::Text) };
    match rf.get("type").and_then(|t| t.as_str()) {
        None | Some("text") => Ok(Format::Text),
        Some("json_object") => Ok(Format::Object),
        Some("json_schema") => {
            let schema = rf
                .get("json_schema")
                .and_then(|j| j.get("schema"))
                .ok_or("response_format json_schema needs json_schema.schema")?;
            let doc = forge_engine::constrain::SchemaDoc::from_json(schema).map_err(|e| format!("response_format json_schema.schema: {e}"))?;
            Ok(Format::Schema(std::sync::Arc::new(doc)))
        }
        Some(other) => Err(format!("response_format type {other:?} is not supported: use \"json_schema\", \"json_object\" or \"text\"")),
    }
}

fn stops_from(j: &Json) -> Vec<String> {
    match j.get("stop") {
        Some(Json::Str(x)) => vec![x.clone()],
        Some(Json::Arr(a)) => a
            .iter()
            .filter_map(|x| x.as_str().map(|s| s.to_string()))
            .take(4)
            .collect(),
        _ => Vec::new(),
    }
}

fn chat_request(body: &Json) -> Result<Request, String> {
    let messages = body
        .get("messages")
        .and_then(|m| m.as_array())
        .ok_or("messages must be an array")?;
    if messages.is_empty() || messages.len() > 4000 {
        return Err("messages must hold between 1 and 4000 entries".into());
    }
    if body
        .get("n")
        .and_then(|n| n.as_i64())
        .map(|n| n != 1)
        .unwrap_or(false)
    {
        return Err("n must be 1".into());
    }
    let messages = normalize_messages(messages)?;
    let tool_choice_none = body.get("tool_choice").and_then(|t| t.as_str()) == Some("none");
    let tools: Vec<Json> = if tool_choice_none {
        Vec::new()
    } else {
        body.get("tools")
            .and_then(|t| t.as_array())
            .map(|a| a.to_vec())
            .unwrap_or_default()
    };
    if tools.len() > 64 {
        return Err("at most 64 tools".into());
    }
    let format = response_format(body)?;
    let json = !matches!(format, Format::Text);
    let mut enable_thinking = body
        .get("chat_template_kwargs")
        .and_then(|k| k.get("enable_thinking"))
        .and_then(|v| v.as_bool());
    if enable_thinking.is_none()
        && body.get("reasoning_effort").and_then(|v| v.as_str()) == Some("none")
    {
        enable_thinking = Some(false);
    }
    if json {
        // the constraint governs the whole answer: there is no room for a reasoning block in front of it
        enable_thinking = Some(false);
    }
    let max_tokens = match (
        body.get("max_tokens").and_then(|v| v.as_usize()),
        body.get("max_completion_tokens").and_then(|v| v.as_usize()),
    ) {
        (Some(_), Some(_)) => {
            return Err("give max_tokens or max_completion_tokens, not both".into())
        }
        (a, b) => a.or(b),
    };
    Ok(Request {
        prompt: Prompt::Chat {
            messages,
            tools,
            enable_thinking,
        },
        sampling: SamplingParams {
            json_object: matches!(format, Format::Object),
            json_schema: match &format {
                Format::Schema(d) => Some(d.clone()),
                _ => None,
            },
            ..sampling_from(body)
        },
        max_tokens,
        stop: stops_from(body),
    })
}

// ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

fn finish_str(r: FinishReason) -> &'static str {
    match r {
        FinishReason::Stop => "stop",
        FinishReason::Length => "length",
        FinishReason::ToolCalls => "tool_calls",
        FinishReason::Cancelled => "stop",
    }
}

fn chunk(id: &str, created: i64, model: &str, delta: Json, finish: Option<&str>) -> String {
    let c = obj(vec![
        ("id", s(id)),
        ("object", s("chat.completion.chunk")),
        ("created", Json::Int(created)),
        ("model", s(model)),
        (
            "choices",
            Json::Arr(vec![obj(vec![
                ("index", Json::Int(0)),
                ("delta", delta),
                ("finish_reason", finish.map(s).unwrap_or(Json::Null)),
            ])]),
        ),
    ]);
    format!("data: {}\n\n", to_text(&c))
}

fn write_chunked(stream: &mut TcpStream, data: &str) -> std::io::Result<()> {
    stream.write_all(format!("{:x}\r\n", data.len()).as_bytes())?;
    stream.write_all(data.as_bytes())?;
    stream.write_all(b"\r\n")?;
    stream.flush()
}

fn usage_json(u: &forge_engine::Usage, timings: &forge_engine::Timings) -> (Json, Json) {
    let usage = obj(vec![
        ("prompt_tokens", Json::Int(u.prompt_tokens as i64)),
        ("completion_tokens", Json::Int(u.completion_tokens as i64)),
        (
            "total_tokens",
            Json::Int((u.prompt_tokens + u.completion_tokens) as i64),
        ),
        (
            "prompt_tokens_details",
            obj(vec![("cached_tokens", Json::Int(u.cached_tokens as i64))]),
        ),
    ]);
    let new_prompt = u.prompt_tokens.saturating_sub(u.cached_tokens);
    let t = obj(vec![
        ("cache_n", Json::Int(u.cached_tokens as i64)),
        ("prompt_n", Json::Int(new_prompt as i64)),
        ("prompt_ms", Json::Float(timings.prefill_ms)),
        (
            "prompt_per_second",
            Json::Float(if timings.prefill_ms > 0.0 {
                new_prompt as f64 / (timings.prefill_ms / 1e3)
            } else {
                0.0
            }),
        ),
        ("predicted_n", Json::Int(u.completion_tokens as i64)),
        ("predicted_ms", Json::Float(timings.decode_ms)),
        (
            "predicted_per_second",
            Json::Float(if timings.decode_ms > 0.0 {
                u.completion_tokens as f64 / (timings.decode_ms / 1e3)
            } else {
                0.0
            }),
        ),
        ("draft_n", Json::Int(u.draft_cycles as i64)),
        ("draft_n_accepted", Json::Int(u.draft_accepted as i64)),
    ]);
    (usage, t)
}

/// Close the generation when the client hangs up: a thread blocks on the socket; end of stream (or an error) means the peer is gone.
fn watch_disconnect(stream: &TcpStream, cancel: Arc<std::sync::atomic::AtomicBool>) {
    if let Ok(mut s2) = stream.try_clone() {
        std::thread::spawn(move || {
            let mut b = [0u8; 256];
            loop {
                match s2.read(&mut b) {
                    Ok(0) | Err(_) => {
                        cancel.store(true, std::sync::atomic::Ordering::SeqCst);
                        return;
                    }
                    Ok(_) => {}
                }
            }
        });
    }
}

/// One server-sent chunk that reports the prompt's progress (llama.cpp's `return_progress`): an empty delta and `prompt_progress` {total, cache, processed, time_ms}, where processed / total is the share of the
/// prompt in the cache and (processed - cache) / (total - cache) the share of the work this request has to do.
fn progress_chunk(id: &str, created: i64, model: &str, chat: bool, processed: usize, total: usize, cached: usize, elapsed_ms: f64) -> String {
    let p = obj(vec![
        ("total", Json::Int(total as i64)),
        ("cache", Json::Int(cached as i64)),
        ("processed", Json::Int(processed as i64)),
        ("time_ms", Json::Float(elapsed_ms)),
    ]);
    let choice = if chat {
        obj(vec![("index", Json::Int(0)), ("delta", obj(vec![])), ("finish_reason", Json::Null)])
    } else {
        obj(vec![("index", Json::Int(0)), ("text", s("")), ("finish_reason", Json::Null)])
    };
    let c = obj(vec![
        ("id", s(id)),
        ("object", s(if chat { "chat.completion.chunk" } else { "text_completion" })),
        ("created", Json::Int(created)),
        ("model", s(model)),
        ("choices", Json::Arr(vec![choice])),
        ("prompt_progress", p),
    ]);
    format!("data: {}\n\n", to_text(&c))
}

#[allow(clippy::too_many_arguments)]
fn handle_generation(
    stream: &mut TcpStream,
    engine: &Engine,
    req: Request,
    chat: bool,
    stream_mode: bool,
    include_usage: bool,
    return_progress: bool,
) {
    let model = engine.info.alias.clone();
    let generation = match engine.start(req) {
        Ok(g) => g,
        Err(e) => {
            engine_error(stream, &e);
            return;
        }
    };
    watch_disconnect(stream, generation.cancel_handle());
    let id = if chat {
        format!("chatcmpl-{}", hex_id())
    } else {
        format!("cmpl-{}", hex_id())
    };
    let created = now();
    if stream_mode {
        let head = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\nX-Accel-Buffering: no\r\nTransfer-Encoding: chunked\r\n\r\n";
        if stream.write_all(head.as_bytes()).is_err() {
            generation.cancel();
            return;
        }
        if write_chunked(
            stream,
            &chunk(
                &id,
                created,
                &model,
                obj(vec![("role", s("assistant")), ("content", s(""))]),
                None,
            ),
        )
        .is_err()
        {
            generation.cancel();
            return;
        }
        loop {
            let ev = match generation.rx.recv_timeout(Duration::from_secs(15)) {
                Ok(ev) => ev,
                Err(RecvTimeoutError::Timeout) => {
                    if write_chunked(stream, ": keep-alive\n\n").is_err() {
                        generation.cancel();
                        return;
                    }
                    continue;
                }
                Err(RecvTimeoutError::Disconnected) => break,
            };
            let data = match ev {
                Event::Started { .. } => continue,
                Event::Progress {
                    processed,
                    total,
                    cached,
                    elapsed_ms,
                } => {
                    if !return_progress {
                        continue;
                    }
                    progress_chunk(&id, created, &model, chat, processed, total, cached, elapsed_ms)
                }
                Event::Reasoning(t) => chunk(
                    &id,
                    created,
                    &model,
                    obj(vec![("reasoning_content", Json::Str(t))]),
                    None,
                ),
                Event::Content(t) => {
                    if chat {
                        chunk(
                            &id,
                            created,
                            &model,
                            obj(vec![("content", Json::Str(t))]),
                            None,
                        )
                    } else {
                        format!(
                            "data: {}\n\n",
                            to_text(&obj(vec![
                                ("id", s(&id)),
                                ("object", s("text_completion")),
                                ("created", Json::Int(created)),
                                ("model", s(&model)),
                                (
                                    "choices",
                                    Json::Arr(vec![obj(vec![
                                        ("index", Json::Int(0)),
                                        ("text", Json::Str(t)),
                                        ("finish_reason", Json::Null)
                                    ])])
                                )
                            ]))
                        )
                    }
                }
                Event::ToolCallStart {
                    index,
                    id: cid,
                    name,
                } => chunk(
                    &id,
                    created,
                    &model,
                    obj(vec![(
                        "tool_calls",
                        Json::Arr(vec![obj(vec![
                            ("index", Json::Int(index as i64)),
                            ("id", Json::Str(cid)),
                            ("type", s("function")),
                            (
                                "function",
                                obj(vec![("name", Json::Str(name)), ("arguments", s(""))]),
                            ),
                        ])]),
                    )]),
                    None,
                ),
                Event::ToolCallArgs { index, fragment } => chunk(
                    &id,
                    created,
                    &model,
                    obj(vec![(
                        "tool_calls",
                        Json::Arr(vec![obj(vec![
                            ("index", Json::Int(index as i64)),
                            ("function", obj(vec![("arguments", Json::Str(fragment))])),
                        ])]),
                    )]),
                    None,
                ),
                Event::ToolCallEnd { .. } => continue,
                Event::Finished {
                    reason,
                    usage,
                    timings,
                } => {
                    let mut out = if chat {
                        chunk(&id, created, &model, obj(vec![]), Some(finish_str(reason)))
                    } else {
                        format!(
                            "data: {}\n\n",
                            to_text(&obj(vec![
                                ("id", s(&id)),
                                ("object", s("text_completion")),
                                ("created", Json::Int(created)),
                                ("model", s(&model)),
                                (
                                    "choices",
                                    Json::Arr(vec![obj(vec![
                                        ("index", Json::Int(0)),
                                        ("text", s("")),
                                        ("finish_reason", s(finish_str(reason)))
                                    ])])
                                )
                            ]))
                        )
                    };
                    let (u, t) = usage_json(&usage, &timings);
                    if include_usage {
                        out.push_str(&format!(
                            "data: {}\n\n",
                            to_text(&obj(vec![
                                ("id", s(&id)),
                                ("object", s("chat.completion.chunk")),
                                ("created", Json::Int(created)),
                                ("model", s(&model)),
                                ("choices", Json::Arr(vec![])),
                                ("usage", u),
                                ("timings", t)
                            ]))
                        ));
                    }
                    out.push_str("data: [DONE]\n\n");
                    if write_chunked(stream, &out).is_err() {
                        generation.cancel();
                        return;
                    }
                    let _ = stream.write_all(b"0\r\n\r\n");
                    return;
                }
                Event::Error(e) => {
                    let out = format!(
                        "data: {}\n\ndata: [DONE]\n\n",
                        to_text(&error_body(&e.to_string(), "server_error", 500))
                    );
                    let _ = write_chunked(stream, &out);
                    let _ = stream.write_all(b"0\r\n\r\n");
                    return;
                }
            };
            if write_chunked(stream, &data).is_err() {
                generation.cancel();
                return;
            }
        }
    } else {
        let (mut content, mut reasoning) = (String::new(), String::new());
        let mut calls: Vec<(String, String, String)> = Vec::new(); // id, name, arguments
        let (finish, usage, timings) = loop {
            match generation.rx.recv_timeout(Duration::from_secs(3600)) {
                Ok(Event::Content(t)) => content.push_str(&t),
                Ok(Event::Reasoning(t)) => reasoning.push_str(&t),
                Ok(Event::ToolCallStart { id: cid, name, .. }) => {
                    calls.push((cid, name, String::new()))
                }
                Ok(Event::ToolCallArgs { index, fragment }) => {
                    if let Some(c) = calls.get_mut(index) {
                        c.2.push_str(&fragment);
                    }
                }
                Ok(Event::ToolCallEnd { .. }) | Ok(Event::Started { .. }) | Ok(Event::Progress { .. }) => {}
                Ok(Event::Finished {
                    reason,
                    usage,
                    timings,
                }) => break (reason, usage, timings),
                Ok(Event::Error(e)) => {
                    engine_error(stream, &e);
                    return;
                }
                Err(_) => {
                    respond_error(stream, 500, "server_error", "the engine stopped answering");
                    return;
                }
            }
        };
        let (u, t) = usage_json(&usage, &timings);
        let body = if chat {
            let mut msg: Vec<(&str, Json)> = vec![
                ("role", s("assistant")),
                (
                    "content",
                    if content.is_empty() && !calls.is_empty() {
                        Json::Null
                    } else {
                        Json::Str(content)
                    },
                ),
            ];
            if !reasoning.is_empty() {
                msg.push(("reasoning_content", Json::Str(reasoning)));
            }
            if !calls.is_empty() {
                msg.push((
                    "tool_calls",
                    Json::Arr(
                        calls
                            .iter()
                            .map(|(cid, name, args)| {
                                obj(vec![
                                    ("id", s(cid)),
                                    ("type", s("function")),
                                    (
                                        "function",
                                        obj(vec![("name", s(name)), ("arguments", s(args))]),
                                    ),
                                ])
                            })
                            .collect(),
                    ),
                ));
            }
            obj(vec![
                ("id", s(&id)),
                ("object", s("chat.completion")),
                ("created", Json::Int(created)),
                ("model", s(&model)),
                (
                    "choices",
                    Json::Arr(vec![obj(vec![
                        ("index", Json::Int(0)),
                        ("message", obj(msg)),
                        ("finish_reason", s(finish_str(finish))),
                    ])]),
                ),
                ("usage", u),
                ("timings", t),
            ])
        } else {
            obj(vec![
                ("id", s(&id)),
                ("object", s("text_completion")),
                ("created", Json::Int(created)),
                ("model", s(&model)),
                (
                    "choices",
                    Json::Arr(vec![obj(vec![
                        ("index", Json::Int(0)),
                        ("text", Json::Str(content)),
                        ("finish_reason", s(finish_str(finish))),
                    ])]),
                ),
                ("usage", u),
                ("timings", t),
            ])
        };
        respond_json(stream, 200, &body, "");
    }
}

fn handle(mut stream: TcpStream, engine: Arc<Engine>, cfg: Arc<ServerConfig>) {
    let _ = stream.set_nodelay(true);
    // a client that connects and says nothing must not hold a thread forever; once the request is in, the socket is only watched for the peer hanging up
    let _ = stream.set_read_timeout(Some(Duration::from_secs(60)));
    let req = match read_request(&mut stream) {
        Ok(r) => {
            let _ = stream.set_read_timeout(None);
            r
        }
        Err(ReadError::Closed) => return,
        Err(ReadError::TooLarge) => {
            return respond_error(
                &mut stream,
                413,
                "invalid_request_error",
                "request body too large",
            )
        }
        Err(ReadError::Bad(m)) => {
            return respond_error(&mut stream, 400, "invalid_request_error", m)
        }
    };
    if let Some(key) = &cfg.api_key {
        let ok = req
            .header("authorization")
            .map(|h| h.strip_prefix("Bearer ").unwrap_or("") == key)
            .unwrap_or(false);
        if !ok && req.path != "/health" && req.path != "/v1/health" {
            return respond_error(&mut stream, 401, "invalid_request_error", "invalid API key");
        }
    }
    let t0 = Instant::now();
    let (method, path) = (req.method.as_str(), req.path.as_str());
    let body_json = || -> Result<Json, String> {
        Json::parse(&String::from_utf8_lossy(&req.body))
            .map_err(|e| format!("invalid JSON body: {e}"))
    };
    match (method, path) {
        ("GET", "/health") | ("GET", "/v1/health") => {
            respond_json(&mut stream, 200, &obj(vec![("status", s("ok"))]), "")
        }
        ("GET", "/v1/models") => {
            let i = &engine.info;
            let m = obj(vec![
                ("id", s(&i.alias)),
                ("object", s("model")),
                ("created", Json::Int(now())),
                ("owned_by", s("forgeinfer")),
                (
                    "meta",
                    obj(vec![
                        ("n_ctx", Json::Int(i.n_ctx as i64)),
                        ("n_vocab", Json::Int(i.vocab as i64)),
                        ("n_params", Json::Int(i.n_params as i64)),
                    ]),
                ),
            ]);
            respond_json(
                &mut stream,
                200,
                &obj(vec![("object", s("list")), ("data", Json::Arr(vec![m]))]),
                "",
            );
        }
        ("GET", "/props") => {
            let i = &engine.info;
            let st = engine.stats();
            respond_json(
                &mut stream,
                200,
                &obj(vec![
                    ("model", s(&i.alias)),
                    ("n_ctx", Json::Int(i.n_ctx as i64)),
                    ("chat_format", s(&format!("{:?}", i.chat))),
                    ("weights", s(&i.weights)),
                    ("weight_bytes", Json::Int(i.weight_bytes as i64)),
                    ("supports_tools", Json::Bool(i.supports_tools)),
                    ("speculative_decoding", Json::Bool(i.speculative)),
                    ("requests", Json::Int(st.requests as i64)),
                    ("cached_tokens", Json::Int(st.cached_tokens as i64)),
                    ("prompt_tokens", Json::Int(st.prompt_tokens as i64)),
                    ("completion_tokens", Json::Int(st.completion_tokens as i64)),
                    ("draft_cycles", Json::Int(st.draft_cycles as i64)),
                    ("draft_accepted", Json::Int(st.draft_accepted as i64)),
                    ("build", s("forgeinfer")),
                ]),
                "",
            );
        }
        ("POST", "/v1/chat/completions") => {
            let body = match body_json() {
                Ok(b) => b,
                Err(m) => return respond_error(&mut stream, 400, "invalid_request_error", &m),
            };
            let request = match chat_request(&body) {
                Ok(r) => r,
                Err(m) => return respond_error(&mut stream, 400, "invalid_request_error", &m),
            };
            let streaming = body
                .get("stream")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            let usage = body
                .get("stream_options")
                .and_then(|o| o.get("include_usage"))
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            let progress = body
                .get("return_progress")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            handle_generation(&mut stream, &engine, request, true, streaming, usage, progress);
            eprintln!(
                "[serve] POST {path} stream={streaming} {:.0} ms",
                t0.elapsed().as_secs_f64() * 1e3
            );
        }
        ("POST", "/v1/completions") => {
            let body = match body_json() {
                Ok(b) => b,
                Err(m) => return respond_error(&mut stream, 400, "invalid_request_error", &m),
            };
            let prompt = match body.get("prompt").and_then(|p| p.as_str()) {
                Some(p) => p.to_string(),
                None => {
                    return respond_error(
                        &mut stream,
                        400,
                        "invalid_request_error",
                        "prompt must be a string",
                    )
                }
            };
            let max_tokens = body.get("max_tokens").and_then(|v| v.as_usize());
            let request = Request {
                prompt: Prompt::Raw(prompt),
                sampling: sampling_from(&body),
                max_tokens,
                stop: stops_from(&body),
            };
            let streaming = body
                .get("stream")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            let progress = body
                .get("return_progress")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            handle_generation(&mut stream, &engine, request, false, streaming, false, progress);
            eprintln!(
                "[serve] POST {path} stream={streaming} {:.0} ms",
                t0.elapsed().as_secs_f64() * 1e3
            );
        }
        ("POST", "/apply-template") => {
            let body = match body_json() {
                Ok(b) => b,
                Err(m) => return respond_error(&mut stream, 400, "invalid_request_error", &m),
            };
            match chat_request(&body)
                .and_then(|r| engine.render(&r.prompt).map_err(|e| e.to_string()))
            {
                Ok((text, ids, _)) => respond_json(
                    &mut stream,
                    200,
                    &obj(vec![
                        ("prompt", Json::Str(text)),
                        ("n_tokens", Json::Int(ids.len() as i64)),
                    ]),
                    "",
                ),
                Err(m) => respond_error(&mut stream, 400, "invalid_request_error", &m),
            }
        }
        ("POST", "/tokenize") => {
            let body = match body_json() {
                Ok(b) => b,
                Err(m) => return respond_error(&mut stream, 400, "invalid_request_error", &m),
            };
            let text = body.get("content").and_then(|c| c.as_str()).unwrap_or("");
            let ids = engine.tok.encode(text);
            respond_json(
                &mut stream,
                200,
                &obj(vec![(
                    "tokens",
                    Json::Arr(ids.into_iter().map(|i| Json::Int(i as i64)).collect()),
                )]),
                "",
            );
        }
        ("POST", "/detokenize") => {
            let body = match body_json() {
                Ok(b) => b,
                Err(m) => return respond_error(&mut stream, 400, "invalid_request_error", &m),
            };
            let ids: Vec<u32> = body
                .get("tokens")
                .and_then(|t| t.as_array())
                .map(|a| {
                    a.iter()
                        .filter_map(|v| v.as_u64().map(|x| x as u32))
                        .collect()
                })
                .unwrap_or_default();
            respond_json(
                &mut stream,
                200,
                &obj(vec![("content", Json::Str(engine.tok.decode(&ids, false)))]),
                "",
            );
        }
        (_, "/health")
        | (_, "/v1/models")
        | (_, "/props")
        | (_, "/v1/chat/completions")
        | (_, "/v1/completions") => {
            let head = "HTTP/1.1 405 Method Not Allowed\r\nAllow: GET, POST\r\nContent-Type: application/json\r\nConnection: close\r\nContent-Length: ";
            let b = to_text(&error_body(
                "method not allowed",
                "invalid_request_error",
                405,
            ));
            let _ = stream.write_all(format!("{head}{}\r\n\r\n{b}", b.len()).as_bytes());
        }
        _ => respond_error(
            &mut stream,
            404,
            "invalid_request_error",
            &format!("unknown route {method} {path}"),
        ),
    }
    let _ = stream.flush();
    let _ = stream.shutdown(std::net::Shutdown::Both);
}

/// Serve until the process ends (blocking).
pub fn serve(engine: Arc<Engine>, cfg: ServerConfig) -> std::io::Result<()> {
    let listener = TcpListener::bind((cfg.host.as_str(), cfg.port))?;
    eprintln!(
        "[serve] listening on http://{}:{} (model {}, {} tokens of context)",
        cfg.host, cfg.port, engine.info.alias, engine.info.n_ctx
    );
    let cfg = Arc::new(cfg);
    let live = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    for conn in listener.incoming() {
        match conn {
            Ok(mut stream) => {
                if live.load(std::sync::atomic::Ordering::SeqCst) >= MAX_CONNECTIONS {
                    respond_error(
                        &mut stream,
                        503,
                        "server_error",
                        "too many open connections",
                    );
                    continue;
                }
                live.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                let (e, c, l) = (engine.clone(), cfg.clone(), live.clone());
                std::thread::spawn(move || {
                    handle(stream, e, c);
                    l.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
                });
            }
            Err(e) => eprintln!("[serve] accept failed: {e}"),
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use forge_engine::Prompt;

    fn request(body: &str) -> Result<Request, String> {
        chat_request(&Json::parse(body).unwrap())
    }

    #[test]
    fn response_format_json_object_constrains_the_answer_and_switches_thinking_off() {
        let plain = request(r#"{"messages": [{"role": "user", "content": "hi"}]}"#).unwrap();
        assert!(!plain.sampling.json_object);
        let text = request(r#"{"messages": [{"role": "user", "content": "hi"}], "response_format": {"type": "text"}}"#).unwrap();
        assert!(!text.sampling.json_object);
        let json = request(r#"{"messages": [{"role": "user", "content": "hi"}], "temperature": 0.3, "chat_template_kwargs": {"enable_thinking": true}, "response_format": {"type": "json_object"}}"#).unwrap();
        assert!(json.sampling.json_object);
        assert_eq!(json.sampling.temperature, Some(0.3), "the other sampling fields are kept");
        match json.prompt {
            Prompt::Chat { enable_thinking, .. } => assert_eq!(enable_thinking, Some(false), "a reasoning block cannot come before a constrained answer"),
            Prompt::Raw(_) => panic!("a chat request"),
        }
        let schema = request(r#"{"messages": [{"role": "user", "content": "hi"}], "response_format": {"type": "json_schema", "json_schema": {"name": "x", "strict": true, "schema": {"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]}}}}"#).unwrap();
        assert!(!schema.sampling.json_object && schema.sampling.json_schema.is_some());
        match schema.prompt {
            Prompt::Chat { enable_thinking, .. } => assert_eq!(enable_thinking, Some(false)),
            Prompt::Raw(_) => panic!("a chat request"),
        }
        let err = |rf: &str| request(&format!(r#"{{"messages": [{{"role": "user", "content": "hi"}}], "response_format": {rf}}}"#)).err().unwrap();
        assert!(err(r#"{"type": "json_schema"}"#).contains("json_schema.schema"));
        assert!(err(r#"{"type": "json_schema", "json_schema": {"schema": {"type": "foo"}}}"#).contains("unknown type"));
        assert!(err(r#"{"type": "xml"}"#).contains("not supported"));
    }
}

