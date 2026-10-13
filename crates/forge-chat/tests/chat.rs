//! Chat templates against the golden renderings recorded from each model's own Jinja template (28 SmolLM2 + 56 Qwen3 + 56 Qwen3.5 cases, each also from the OpenAI wire form of the conversation),
//! and the output parser against the table of `tools/ref_chat_templates.py` for every chunking.

use forge_chat::*;
use forge_core::Json;
use std::path::PathBuf;

fn golden(name: &str) -> Vec<Json> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/golden")
        .join(format!("chat_{name}.json"));
    let j = Json::from_file(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    j["cases"].as_array().unwrap().to_vec()
}

/// The conversation as a client sends it: tool-call arguments as JSON strings, null content next to tool calls.
fn to_wire(messages: &[Json]) -> Vec<Json> {
    messages
        .iter()
        .map(|m| {
            let mut fields: Vec<(String, Json)> = m.as_object().unwrap().to_vec();
            let has_calls = m
                .get("tool_calls")
                .and_then(|t| t.as_array())
                .map(|a| !a.is_empty())
                .unwrap_or(false);
            for (k, v) in fields.iter_mut() {
                if k == "tool_calls" {
                    if let Json::Arr(calls) = v {
                        for c in calls.iter_mut() {
                            if let Json::Obj(cf) = c {
                                for (ck, cv) in cf.iter_mut() {
                                    if ck == "function" {
                                        if let Json::Obj(ff) = cv {
                                            for (fk, fv) in ff.iter_mut() {
                                                if fk == "arguments" {
                                                    let mut s = String::new();
                                                    fv.write(&mut s);
                                                    *fv = Json::Str(s);
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                if k == "content" && has_calls && v.as_str().map(|s| s.is_empty()).unwrap_or(true) {
                    *v = Json::Null;
                }
            }
            Json::Obj(fields)
        })
        .collect()
}

fn check(name: &str, fmt: ChatFormat) {
    let cases = golden(name);
    assert!(!cases.is_empty());
    let mut bad = 0;
    for c in &cases {
        let messages = c["messages"].as_array().unwrap().to_vec();
        let tools: Vec<Json> = c["tools"]
            .as_array()
            .map(|a| a.to_vec())
            .unwrap_or_default();
        let gen = c["add_generation_prompt"].as_bool().unwrap();
        let think = c["kwargs"].get("enable_thinking").and_then(|t| t.as_bool());
        let want = c["text"].as_str().unwrap();
        for (form, msgs) in [
            ("stored", messages.clone()),
            ("wire", wire_to_template(&to_wire(&messages))),
        ] {
            let got = fmt.render(&msgs, &tools, gen, think).unwrap();
            if got != want {
                bad += 1;
                eprintln!("GOLDEN FAIL {name} {} ({form}) gen={gen} think={think:?}\n  got  {got:?}\n  want {want:?}", c["name"].as_str().unwrap());
            }
        }
    }
    assert_eq!(bad, 0, "{name}: {bad} renderings differ");
    eprintln!(
        "{name}: {} golden renderings exact (stored and wire form)",
        cases.len()
    );
}

#[test]
fn smollm2_renders_like_its_jinja_template() {
    check("smollm2", ChatFormat::SmolLm2);
}

#[test]
fn qwen3_renders_like_its_jinja_template() {
    check("qwen3", ChatFormat::Qwen3);
}

#[test]
fn qwen35_renders_like_its_jinja_template() {
    check(
        "qwen35",
        ChatFormat::Qwen35 {
            thinking_default: false,
        },
    );
}

#[test]
fn qwen35_thinking_default_depends_on_the_checkpoint() {
    let msgs = vec![Json::parse(r#"{"role": "user", "content": "hi"}"#).unwrap()];
    let off = ChatFormat::Qwen35 {
        thinking_default: false,
    }
    .render(&msgs, &[], true, None)
    .unwrap();
    let on = ChatFormat::Qwen35 {
        thinking_default: true,
    }
    .render(&msgs, &[], true, None)
    .unwrap();
    assert!(
        off.ends_with("<think>\n\n</think>\n\n")
            && on.ends_with("<|im_start|>assistant\n<think>\n")
    );
    assert!(
        ChatFormat::Qwen35 {
            thinking_default: true
        }
        .thinking_open(None)
            && !ChatFormat::Qwen35 {
                thinking_default: true
            }
            .thinking_open(Some(false))
    );
}

// ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// parser

fn tools() -> Vec<Json> {
    vec![Json::parse(r#"{"type": "function", "function": {"name": "run_python", "parameters": {"type": "object", "properties": {"code": {"type": "string"}, "n": {"type": "integer"}, "ok": {"type": "boolean"}, "xs": {"type": "array"}}}}}"#).unwrap()]
}

struct Case {
    fmt: CallFormat,
    thinking_open: bool,
    with_tools: bool,
    text: &'static str,
    reasoning: &'static str,
    answer: &'static str,
    calls: Vec<(&'static str, Option<&'static str>)>, // (name, arguments JSON or None when the text is not valid JSON)
}

fn cases() -> Vec<Case> {
    const JS: CallFormat = CallFormat::Json;
    const XM: CallFormat = CallFormat::Xml;
    let c = |fmt, thinking_open, with_tools, text, reasoning, answer, calls| Case {
        fmt,
        thinking_open,
        with_tools,
        text,
        reasoning,
        answer,
        calls,
    };
    vec![
        c(JS, false, false, "I will do both.\n<tool_call>\n{\"name\": \"a\", \"arguments\": {\"x\": \"}\\\"{\"}}\n</tool_call>\n<tool_call>\n{\"name\": \"b\", \"arguments\": {}}\n</tool_call>", "", "I will do both.", vec![("a", Some("{\"x\": \"}\\\"{\"}")), ("b", Some("{}"))]),
        c(JS, false, false, "<think>\nplan\n</think>\n\nOK.\n\n<tool_call>\n{\"arguments\": {\"a\": 1}, \"name\": \"k\"}\n</tool_call>", "plan", "OK.", vec![("k", Some("{\"a\": 1}"))]),
        c(JS, false, false, "<tool_call>\n{\"name\": \"k\", \"arguments\": {\"a\": 1}}", "", "", vec![("k", Some("{\"a\": 1}"))]),
        c(JS, false, false, "<tool_call>\n{\"name\": \"k\", \"arguments\": {\"a\": [1, 2", "", "", vec![("k", None)]),
        c(JS, false, false, "I would write <tool_call> to call a tool. a < b and 1<2 <b", "", "I would write <tool_call> to call a tool. a < b and 1<2 <b", vec![]),
        c(XM, true, false, "Think.\n</think>\n\nHi!", "Think.", "Hi!", vec![]),
        c(XM, true, false, "Hmm.\n<tool_call>\n<function=f>\n</function>\n</tool_call>", "Hmm.", "", vec![("f", Some("{}"))]),
        c(XM, false, true, "Sure.\n\n<tool_call>\n<function=run_python>\n<parameter=code>\nprint(\"hi\")\nx = 1 < 2\n</parameter>\n</function>\n</tool_call>", "", "Sure.", vec![("run_python", Some("{\"code\": \"print(\\\"hi\\\")\\nx = 1 < 2\"}"))]),
        c(XM, false, true, "<tool_call>\n<function=run_python>\n<parameter=n>\n5\n</parameter>\n<parameter=ok>\nTrue\n</parameter>\n<parameter=xs>\n[1, \"a\"]\n</parameter>\n</function>\n</tool_call>", "", "", vec![("run_python", Some("{\"n\": 5, \"ok\": true, \"xs\": [1, \"a\"]}"))]),
        c(XM, false, true, "<tool_call>\n<function=run_python>\n<parameter=code>\nx\n<parameter=n>\n7", "", "", vec![("run_python", Some("{\"code\": \"x\", \"n\": 7}"))]),
        c(XM, false, false, "<tool_call>\n<function=a>\n<parameter=p>\nv\n</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=b>\n</function>\n</tool_call>", "", "", vec![("a", Some("{\"p\": \"v\"}")), ("b", Some("{}"))]),
        c(XM, false, false, "<tool_call>\ngarbage\n</tool_call>tail", "", "<tool_call>\ngarbage\n</tool_call>tail", vec![]),
    ]
}

#[test]
fn parser_gives_the_same_result_for_every_chunking() {
    let tl = tools();
    let mut runs = 0;
    for (ci, c) in cases().iter().enumerate() {
        let t: &[Json] = if c.with_tools { &tl } else { &[] };
        let whole = parse(c.fmt, c.text, &[], c.thinking_open, t);
        let calls: Vec<(String, Option<Json>)> = whole
            .calls
            .iter()
            .map(|(n, a)| (n.clone(), Json::parse(a).ok()))
            .collect();
        let want: Vec<(String, Option<Json>)> = c
            .calls
            .iter()
            .map(|(n, a)| (n.to_string(), a.map(|a| Json::parse(a).unwrap())))
            .collect();
        assert_eq!(
            (whole.reasoning.as_str(), whole.content.as_str()),
            (c.reasoning, c.answer),
            "case {ci}: {:?}",
            c.text
        );
        assert_eq!(calls, want, "case {ci}: {:?}", c.text);
        let n = c.text.len();
        for step in [1usize, 2, 3, 5, 8] {
            let cuts: Vec<usize> = (step..n).step_by(step).collect();
            assert_eq!(
                parse(c.fmt, c.text, &cuts, c.thinking_open, t),
                whole,
                "case {ci}: cuts every {step}"
            );
            runs += 1;
        }
        for i in 0..=n {
            assert_eq!(
                parse(c.fmt, c.text, &[i], c.thinking_open, t),
                whole,
                "case {ci}: single cut at {i}"
            );
            runs += 1;
        }
    }
    eprintln!(
        "{} parser cases, {runs} chunkings all equal the one-shot parse",
        cases().len()
    );
}

#[test]
fn a_call_cut_off_by_max_tokens_is_reported_incomplete() {
    let mut p = OutputParser::new(CallFormat::Xml, false, &[]);
    let mut ev = p.feed("<tool_call>\n<function=f>\n<parameter=a>\nx", false);
    ev.extend(p.finish());
    let ends: Vec<&Event> = ev
        .iter()
        .filter(|e| matches!(e, Event::CallEnd { .. }))
        .collect();
    assert_eq!(
        ends,
        vec![&Event::CallEnd {
            index: 0,
            complete: false
        }]
    );
}
