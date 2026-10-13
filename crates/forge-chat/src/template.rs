//! Chat templates as plain code (no Jinja interpreter): SmolLM2 (ChatML with an injected default system turn), Qwen3 (JSON tool calls) and Qwen3.5 (XML tool calls). Verified byte for byte against the
//! output of each model's own Jinja template (tests/golden/chat_*.json, also with the messages in OpenAI wire form). `tools/ref_chat_templates.py` is the executable specification.
//!
//! Messages and tools are `forge_core::Json` values in the layout of the OpenAI chat API after `wire_to_template` (tool-call arguments as objects when they parse): `{"role", "content", "reasoning_content",
//! "tool_calls": [{"function": {"name", "arguments"}}]}`.

use crate::pyfmt::{py_json, py_str, py_strip};
use forge_core::{Error, Json, Result};

fn err(msg: &str) -> Error {
    Error::msg(msg.to_string())
}

fn role(m: &Json) -> &str {
    m.get("role").and_then(|r| r.as_str()).unwrap_or("")
}

/// A content that is a string; anything else counts as "" (Qwen3's `m["content"] if isinstance(.., str) else ""`).
fn content_str(m: &Json) -> &str {
    m.get("content").and_then(|c| c.as_str()).unwrap_or("")
}

/// Qwen3.5 `render_content`, text only: a string, null, or a list of `{"text": ..}` parts.
fn text_of(c: Option<&Json>) -> Result<String> {
    match c {
        None | Some(Json::Null) => Ok(String::new()),
        Some(Json::Str(s)) => Ok(s.clone()),
        Some(Json::Arr(parts)) => {
            let mut out = String::new();
            for p in parts {
                match p.get("text").and_then(|t| t.as_str()) {
                    Some(t) => out.push_str(t),
                    None => return Err(err("Unexpected content type.")),
                }
            }
            Ok(out)
        }
        _ => Err(err("Unexpected content type.")),
    }
}

/// A user message that is not just tool results wrapped in a user turn.
fn is_query(text: &str) -> bool {
    !(text.starts_with("<tool_response>") && text.ends_with("</tool_response>"))
}

/// (reasoning, answer) of an assistant message: the `reasoning_content` field if it is a string, else the `<think>` block split out of the text.
pub fn split_think(content: &str, given: Option<&str>) -> (String, String) {
    if let Some(g) = given {
        return (g.to_string(), content.to_string());
    }
    if !content.contains("</think>") {
        return (String::new(), content.to_string());
    }
    let first = content.split("</think>").next().unwrap();
    let reasoning = first
        .trim_end_matches('\n')
        .rsplit("<think>")
        .next()
        .unwrap()
        .trim_start_matches('\n');
    let answer = content
        .rsplit("</think>")
        .next()
        .unwrap()
        .trim_start_matches('\n');
    (reasoning.to_string(), answer.to_string())
}

fn function_of(tc: &Json) -> &Json {
    match tc.get("function") {
        Some(f) if f.as_object().map(|o| !o.is_empty()).unwrap_or(false) => f,
        _ => tc,
    }
}

fn tool_calls_of(m: &Json) -> &[Json] {
    m.get("tool_calls")
        .and_then(|t| t.as_array())
        .unwrap_or(&[])
}

fn set_key(obj: &mut Json, key: &str, value: Json) {
    if let Json::Obj(o) = obj {
        for (k, v) in o.iter_mut() {
            if k == key {
                *v = value;
                return;
            }
        }
        o.push((key.to_string(), value));
    }
}

/// OpenAI wire format -> template input: tool-call `arguments` arrive as a JSON string and become an object when they parse (Qwen3.5 needs a mapping); text that is not JSON stays text.
pub fn wire_to_template(messages: &[Json]) -> Vec<Json> {
    let mut out: Vec<Json> = messages.to_vec();
    for m in out.iter_mut() {
        if let Json::Obj(fields) = m {
            for (k, v) in fields.iter_mut() {
                if k != "tool_calls" {
                    continue;
                }
                if let Json::Arr(calls) = v {
                    for c in calls.iter_mut() {
                        let has_fn = c
                            .get("function")
                            .map(|f| f.as_object().map(|o| !o.is_empty()).unwrap_or(false))
                            .unwrap_or(false);
                        let target: &mut Json = if has_fn {
                            if let Json::Obj(o) = c {
                                &mut o.iter_mut().find(|(k, _)| k == "function").unwrap().1
                            } else {
                                unreachable!()
                            }
                        } else {
                            c
                        };
                        if let Some(a) = target
                            .get("arguments")
                            .and_then(|a| a.as_str())
                            .map(|s| s.to_string())
                        {
                            let text = if a.is_empty() { "{}".to_string() } else { a };
                            if let Ok(parsed) = Json::parse(&text) {
                                set_key(target, "arguments", parsed);
                            }
                        }
                    }
                }
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// ChatML

/// Plain ChatML: every message is a turn, no default system prompt, no tools.
pub fn render_chatml(messages: &[Json], add_generation_prompt: bool) -> Result<String> {
    let mut out = String::new();
    for m in messages {
        out.push_str("<|im_start|>");
        out.push_str(role(m));
        out.push('\n');
        out.push_str(&text_of(m.get("content")).unwrap_or_default());
        out.push_str("<|im_end|>\n");
    }
    if add_generation_prompt {
        out.push_str("<|im_start|>assistant\n");
    }
    Ok(out)
}

// ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// Llama 3

/// Today's date as the Llama 3.1 / 3.2 templates print it ("26 Jul 2024"), from the system clock.
pub fn today_dd_mon_yyyy() -> String {
    let secs = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0) as i64;
    let z = secs.div_euclid(86_400) + 719_468; // days since 0000-03-01
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    const MONTHS: [&str; 12] = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
    format!("{d:02} {} {y}", MONTHS[(m - 1) as usize])
}

/// Llama 3.x: `<|begin_of_text|>`, a system turn with the knowledge cutoff and the date (as the 3.1 / 3.2 templates add it) followed by the system prompt, then every message as
/// `<|start_header_id|>role<|end_header_id|>\n\n content <|eot_id|>`; no tools.
pub fn render_llama3(messages: &[Json], add_generation_prompt: bool, date: &str) -> Result<String> {
    let mut out = String::from("<|begin_of_text|>");
    let (system, rest) = match messages.first() {
        Some(m) if role(m) == "system" => (content_str(m).trim(), &messages[1..]),
        _ => ("", messages),
    };
    out.push_str("<|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: ");
    out.push_str(date);
    out.push_str("\n\n");
    out.push_str(system);
    out.push_str("<|eot_id|>");
    for m in rest {
        out.push_str("<|start_header_id|>");
        out.push_str(role(m));
        out.push_str("<|end_header_id|>\n\n");
        out.push_str(content_str(m).trim());
        out.push_str("<|eot_id|>");
    }
    if add_generation_prompt {
        out.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
    }
    Ok(out)
}

// ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// SmolLM2

/// ChatML with an injected default system turn; `tools`, `tool_calls` and `reasoning_content` are ignored by this template.
pub fn render_smollm2(messages: &[Json], add_generation_prompt: bool) -> Result<String> {
    let mut out = String::new();
    for (i, m) in messages.iter().enumerate() {
        if i == 0 && role(m) != "system" {
            out.push_str("<|im_start|>system\nYou are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>\n");
        }
        out.push_str("<|im_start|>");
        out.push_str(role(m));
        out.push('\n');
        out.push_str(content_str(m));
        out.push_str("<|im_end|>\n");
    }
    if add_generation_prompt {
        out.push_str("<|im_start|>assistant\n");
    }
    Ok(out)
}

// ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// Qwen3

const Q3_HEAD: &str = "# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>";
const Q3_TAIL: &str = "\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>";

pub fn render_qwen3(
    messages: &[Json],
    tools: &[Json],
    add_generation_prompt: bool,
    enable_thinking: Option<bool>,
) -> Result<String> {
    if messages.is_empty() {
        return Err(err("no messages"));
    }
    let n = messages.len();
    let mut out = String::new();
    if !tools.is_empty() {
        out.push_str("<|im_start|>system\n");
        if role(&messages[0]) == "system" {
            out.push_str(content_str(&messages[0]));
            out.push_str("\n\n");
        }
        out.push_str(Q3_HEAD);
        for t in tools {
            out.push('\n');
            out.push_str(&py_json(t));
        }
        out.push_str(Q3_TAIL);
        out.push_str("<|im_end|>\n");
    } else if role(&messages[0]) == "system" {
        out.push_str("<|im_start|>system\n");
        out.push_str(content_str(&messages[0]));
        out.push_str("<|im_end|>\n");
    }
    // index of the last real user query (none found: the last index, so no turn counts as "after" it)
    let lq = (0..n)
        .rev()
        .find(|&i| {
            role(&messages[i]) == "user"
                && messages[i]
                    .get("content")
                    .and_then(|c| c.as_str())
                    .map(is_query)
                    .unwrap_or(false)
        })
        .unwrap_or(n - 1);
    for (i, m) in messages.iter().enumerate() {
        let r = role(m);
        let content = content_str(m);
        if r == "user" || (r == "system" && i > 0) {
            out.push_str("<|im_start|>");
            out.push_str(r);
            out.push('\n');
            out.push_str(content);
            out.push_str("<|im_end|>\n");
        } else if r == "assistant" {
            let (reasoning, content) =
                split_think(content, m.get("reasoning_content").and_then(|c| c.as_str()));
            if i > lq && (i == n - 1 || !reasoning.is_empty()) {
                out.push_str("<|im_start|>assistant\n<think>\n");
                out.push_str(reasoning.trim_matches('\n'));
                out.push_str("\n</think>\n\n");
                out.push_str(content.trim_start_matches('\n'));
            } else {
                out.push_str("<|im_start|>assistant\n");
                out.push_str(&content);
            }
            for (j, tc) in tool_calls_of(m).iter().enumerate() {
                let f = function_of(tc);
                if (j == 0 && !content.is_empty()) || j > 0 {
                    out.push('\n');
                }
                out.push_str("<tool_call>\n{\"name\": \"");
                out.push_str(f.get("name").and_then(|x| x.as_str()).unwrap_or(""));
                out.push_str("\", \"arguments\": ");
                match f.get("arguments") {
                    Some(Json::Str(s)) => out.push_str(s),
                    Some(a) => out.push_str(&py_json(a)),
                    None => out.push_str("null"),
                }
                out.push_str("}\n</tool_call>");
            }
            out.push_str("<|im_end|>\n");
        } else if r == "tool" {
            // consecutive tool results share one user turn
            if i == 0 || role(&messages[i - 1]) != "tool" {
                out.push_str("<|im_start|>user");
            }
            out.push_str("\n<tool_response>\n");
            out.push_str(content);
            out.push_str("\n</tool_response>");
            if i == n - 1 || role(&messages[i + 1]) != "tool" {
                out.push_str("<|im_end|>\n");
            }
        }
    }
    if add_generation_prompt {
        out.push_str("<|im_start|>assistant\n");
        if enable_thinking == Some(false) {
            out.push_str("<think>\n\n</think>\n\n");
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// Qwen3.5

const Q35_HEAD: &str = "# Tools\n\nYou have access to the following functions:\n\n<tools>";
const Q35_TAIL: &str = "\n</tools>\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>";

/// `thinking_default` is what an absent `enable_thinking` means: false in the 0.8B and 2B templates, true in the 4B and 9B ones (their only difference).
pub fn render_qwen35(
    messages: &[Json],
    tools: &[Json],
    add_generation_prompt: bool,
    enable_thinking: Option<bool>,
    thinking_default: bool,
) -> Result<String> {
    if messages.is_empty() {
        return Err(err("No messages provided."));
    }
    let n = messages.len();
    let trimmed = |m: &Json| -> Result<String> {
        Ok(crate::pyfmt::py_strip(&text_of(m.get("content"))?).to_string())
    };
    let mut out = String::new();
    let sys0 = if role(&messages[0]) == "system" {
        trimmed(&messages[0])?
    } else {
        String::new()
    };
    if !tools.is_empty() {
        out.push_str("<|im_start|>system\n");
        out.push_str(Q35_HEAD);
        for t in tools {
            out.push('\n');
            out.push_str(&py_json(t));
        }
        out.push_str(Q35_TAIL);
        if !sys0.is_empty() {
            out.push_str("\n\n");
            out.push_str(&sys0);
        }
        out.push_str("<|im_end|>\n");
    } else if role(&messages[0]) == "system" {
        out.push_str("<|im_start|>system\n");
        out.push_str(&sys0);
        out.push_str("<|im_end|>\n");
    }
    let mut lq: Option<usize> = None;
    for i in (0..n).rev() {
        if role(&messages[i]) == "user" && is_query(&trimmed(&messages[i])?) {
            lq = Some(i);
            break;
        }
    }
    let lq = lq.ok_or_else(|| err("No user query found in messages."))?;
    for (i, m) in messages.iter().enumerate() {
        let r = role(m);
        let content = trimmed(m)?;
        match r {
            "system" => {
                if i > 0 {
                    return Err(err("System message must be at the beginning."));
                }
            }
            "user" => {
                out.push_str("<|im_start|>user\n");
                out.push_str(&content);
                out.push_str("<|im_end|>\n");
            }
            "assistant" => {
                let (reasoning, content) = split_think(
                    &content,
                    m.get("reasoning_content").and_then(|c| c.as_str()),
                );
                out.push_str("<|im_start|>assistant\n");
                if i > lq {
                    out.push_str("<think>\n");
                    out.push_str(py_strip(&reasoning));
                    out.push_str("\n</think>\n\n");
                }
                out.push_str(&content);
                for (j, tc) in tool_calls_of(m).iter().enumerate() {
                    let f = function_of(tc);
                    if j == 0 && !py_strip(&content).is_empty() {
                        out.push_str("\n\n");
                    } else if j > 0 {
                        out.push('\n');
                    }
                    out.push_str("<tool_call>\n<function=");
                    out.push_str(f.get("name").and_then(|x| x.as_str()).unwrap_or(""));
                    out.push_str(">\n");
                    match f.get("arguments") {
                        None | Some(Json::Null) => {}
                        Some(Json::Obj(args)) => {
                            for (k, v) in args {
                                out.push_str("<parameter=");
                                out.push_str(k);
                                out.push_str(">\n");
                                match v {
                                    Json::Obj(_) | Json::Arr(_) => out.push_str(&py_json(v)),
                                    other => out.push_str(&py_str(other)),
                                }
                                out.push_str("\n</parameter>\n");
                            }
                        }
                        Some(Json::Str(a)) if a.is_empty() => {}
                        Some(_) => return Err(err("tool_call arguments must be an object")),
                    }
                    out.push_str("</function>\n</tool_call>");
                }
                out.push_str("<|im_end|>\n");
            }
            "tool" => {
                if i > 0 && role(&messages[i - 1]) != "tool" {
                    out.push_str("<|im_start|>user");
                }
                out.push_str("\n<tool_response>\n");
                out.push_str(&content);
                out.push_str("\n</tool_response>");
                if i == n - 1 || role(&messages[i + 1]) != "tool" {
                    out.push_str("<|im_end|>\n");
                }
            }
            _ => return Err(err("Unexpected message role.")),
        }
    }
    if add_generation_prompt {
        // "<think>\n" leaves the block OPEN: the model's first tokens are its reasoning
        out.push_str("<|im_start|>assistant\n");
        out.push_str(if enable_thinking.unwrap_or(thinking_default) {
            "<think>\n"
        } else {
            "<think>\n\n</think>\n\n"
        });
    }
    Ok(out)
}

#[cfg(test)]
mod llama3_tests {
    use super::*;

    #[test]
    fn llama3_prompt() {
        let msgs = Json::parse(r#"[{"role":"system","content":" Be brief. "},{"role":"user","content":"Hi"},{"role":"assistant","content":"Hello!"},{"role":"user","content":"Bye"}]"#).unwrap();
        let m = msgs.as_array().unwrap();
        let p = render_llama3(m, true, "26 Jul 2024").unwrap();
        assert_eq!(
            p,
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\nBe brief.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nHi<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nHello!<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nBye<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        );
    }

    #[test]
    fn the_date_is_well_formed() {
        let d = today_dd_mon_yyyy();
        let parts: Vec<&str> = d.split(' ').collect();
        assert_eq!(parts.len(), 3, "{d}");
        assert!(parts[0].len() == 2 && parts[0].parse::<u32>().unwrap() >= 1);
        assert!(parts[2].parse::<u32>().unwrap() >= 2025);
    }
}
