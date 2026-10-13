//! Incremental parsing of model output into reasoning, answer text and tool calls (Qwen3 JSON calls `<tool_call>{"name": .., "arguments": ..}</tool_call>`, Qwen3.5 XML calls
//! `<tool_call><function=NAME><parameter=KEY>VALUE</parameter></function></tool_call>`). `feed` accepts any split of the text and gives the same events whatever the chunking (tested for every
//! single cut of every case). `tools/ref_chat_templates.py` is the executable specification.

use crate::pyfmt::{py_json, py_rstrip, py_strip};
use forge_core::Json;
use std::collections::HashMap;

const THINK_O: &str = "<think>";
const THINK_C: &str = "</think>";
const CALL_O: &str = "<tool_call>";
const CALL_C: &str = "</tool_call>";

#[derive(Clone, Debug, PartialEq)]
pub enum Event {
    Reason(String),
    Content(String),
    CallStart {
        index: usize,
        name: String,
    },
    CallArgs {
        index: usize,
        fragment: String,
    },
    /// `complete` is false when the call was cut off (max tokens) or its arguments are not finished: the engine must drop it.
    CallEnd {
        index: usize,
        complete: bool,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CallFormat {
    /// Qwen3: `<tool_call>{"name": .., "arguments": {..}}</tool_call>`
    Json,
    /// Qwen3.5: `<tool_call><function=NAME><parameter=K>V</parameter></function></tool_call>`
    Xml,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Mode {
    Content,
    Reason,
    Call,
}

/// Length in bytes of the longest suffix of `buf` that is a proper prefix of one of the tags (that text must wait for more input). Tags are ASCII, so a matching suffix starts on a char boundary.
fn hold(buf: &str, tags: &[&str]) -> usize {
    let b = buf.as_bytes();
    let maxk = tags.iter().map(|t| t.len()).max().unwrap_or(1) - 1;
    for k in (1..=maxk.min(b.len())).rev() {
        let tail = &b[b.len() - k..];
        if tags
            .iter()
            .any(|t| t.len() > k && t.as_bytes().starts_with(tail))
        {
            return k;
        }
    }
    0
}

/// Index just after the JSON value that starts at `s[k]`, or None while the text ends inside it.
fn value_end(s: &str, k: usize) -> Option<usize> {
    let b = s.as_bytes();
    if k >= b.len() {
        return None;
    }
    if !matches!(b[k], b'"' | b'{' | b'[') {
        let e = [b',', b'}']
            .iter()
            .filter_map(|c| b[k..].iter().position(|x| x == c).map(|p| p + k))
            .min();
        return e;
    }
    let (mut depth, mut instr, mut esc) = (0i32, false, false);
    for i in k..b.len() {
        let ch = b[i];
        if instr {
            let (new_esc, new_instr) = (!esc && ch == b'\\', esc || ch != b'"');
            esc = new_esc;
            instr = new_instr;
        } else if ch == b'"' {
            instr = true;
        } else if ch == b'{' || ch == b'[' {
            depth += 1;
        } else if ch == b'}' || ch == b']' {
            depth -= 1;
        }
        if !instr && depth == 0 && matches!(ch, b'"' | b'}' | b']') {
            return Some(i + 1);
        }
    }
    None
}

/// Qwen3 body `{"name": N, "arguments": A}` -> (N or None, raw text of A so far, A complete?). Any key order; "parameters" counts as "arguments".
fn scan_json(s: &str) -> (Option<String>, String, bool) {
    let b = s.as_bytes();
    let (mut name, mut args, mut done) = (None, String::new(), false);
    let mut i = s.find('{').map(|p| p + 1).unwrap_or(0);
    while i > 0 {
        let j = s[i..].find('"').map(|p| p + i);
        let e = j.and_then(|j| value_end(s, j));
        let c = e.and_then(|e| s[e..].find(':').map(|p| p + e));
        let mut k = c.map(|c| c + 1).unwrap_or(0);
        while k > 0 && k < b.len() && matches!(b[k], b' ' | b'\t' | b'\r' | b'\n') {
            k += 1;
        }
        let c = match c {
            Some(c) => c,
            None => break,
        };
        let _ = c;
        if k >= b.len() {
            break;
        }
        let key = Json::parse(&s[j.unwrap()..e.unwrap()])
            .ok()
            .and_then(|v| v.as_str().map(|x| x.to_string()))
            .unwrap_or_default();
        let e = value_end(s, k);
        if key == "arguments" || key == "parameters" {
            args = s[k..e.unwrap_or(b.len())].to_string();
            done = e.is_some();
        } else if key == "name" {
            if let Some(e) = e {
                name = Json::parse(&s[k..e])
                    .ok()
                    .and_then(|v| v.as_str().map(|x| x.to_string()));
            }
        }
        match e {
            None => break,
            Some(e) => i = e,
        }
    }
    (name, args, done)
}

/// Typed value of an XML parameter (vLLM's rules): "null" is null for every type; text that does not fit its type stays a string.
fn coerce(raw: &str, typ: &str) -> Json {
    let t = typ.to_lowercase();
    let r = py_strip(raw);
    if r.to_lowercase() == "null" {
        return Json::Null;
    }
    match t.as_str() {
        "integer" | "int" => {
            if let Ok(v) = r.parse::<i64>() {
                return Json::Int(v);
            }
        }
        "number" | "float" => {
            if let Ok(f) = r.parse::<f64>() {
                let lower = r.to_lowercase();
                if f.is_finite()
                    && f.fract() == 0.0
                    && !lower.contains('.')
                    && !lower.contains('e')
                    && f.abs() < 9e18
                {
                    return Json::Int(f as i64);
                }
                return Json::Float(f);
            }
        }
        "boolean" | "bool" => return Json::Bool(r.to_lowercase() == "true"),
        "object" | "array" => {
            if let Ok(v) = Json::parse(r) {
                return v;
            }
        }
        _ => {}
    }
    Json::Str(raw.to_string())
}

/// Qwen3.5 body `<function=N>\n<parameter=K>\nV\n</parameter>..</function>` -> (N or None, JSON text of the arguments so far, finished?). String values stream (an unfinished JSON string has no closing quote);
/// other types appear once their end is seen.
fn scan_xml(
    s: &str,
    closed: bool,
    schemas: &HashMap<String, HashMap<String, String>>,
) -> (Option<String>, String, bool) {
    let i = match s.find("<function=") {
        Some(i) => i,
        None => return (None, String::new(), false),
    };
    let j = match s[i..].find('>') {
        Some(p) => p + i,
        None => return (None, String::new(), false),
    };
    let name = py_strip(&s[i + 10..j]).to_string();
    let props = schemas.get(&name);
    let ends_at = ["</parameter>", "<parameter=", "</function>"];
    let mut items: Vec<String> = Vec::new();
    let mut pos = j + 1;
    let mut finished = false;
    loop {
        let p = s[pos..].find("<parameter=").map(|x| x + pos);
        let f = s[pos..].find("</function>").map(|x| x + pos);
        if let Some(f) = f {
            if p.map(|p| f < p).unwrap_or(true) {
                finished = true;
                break;
            }
        }
        let p = match p {
            Some(p) => p,
            None => break,
        };
        let q = match s[p..].find('>') {
            Some(x) => x + p,
            None => break,
        };
        let key = py_strip(&s[p + 11..q]).to_string();
        let vs = q + 1 + if s[q + 1..].starts_with('\n') { 1 } else { 0 };
        let typ = props
            .and_then(|pr| pr.get(&key))
            .map(|t| t.as_str())
            .unwrap_or("string");
        let found = ends_at
            .iter()
            .filter_map(|t| s[vs..].find(t).map(|x| x + vs))
            .min();
        let raw = &s[vs..found.unwrap_or(s.len())];
        if found.is_some() || closed {
            // the value is complete: its end tag came, or the call was cut and we make the best of it
            let raw = raw.strip_suffix('\n').unwrap_or(raw);
            items.push(format!(
                "{}: {}",
                py_json(&Json::Str(key.clone())),
                py_json(&coerce(raw, typ))
            ));
            match found {
                None => break,
                Some(fd) => {
                    pos = fd
                        + if s[fd..].starts_with("</parameter>") {
                            12
                        } else {
                            0
                        };
                    continue;
                }
            }
        }
        if typ == "string" {
            // still streaming: hold back a possible end tag and one newline
            let mut k = raw.len() - hold(raw, &ends_at);
            if raw[..k].ends_with('\n') {
                k -= 1;
            }
            let mut piece = py_json(&Json::Str(raw[..k].to_string()));
            piece.pop();
            items.push(format!("{}: {}", py_json(&Json::Str(key.clone())), piece));
        }
        break;
    }
    let body = format!(
        "{{{}{}",
        items.join(", "),
        if finished || closed { "}" } else { "" }
    );
    (Some(name), body, finished)
}

/// Incremental parser. `thinking_open`: the prompt ended inside `<think>` (the first tokens are reasoning).
pub struct OutputParser {
    fmt: CallFormat,
    schemas: HashMap<String, HashMap<String, String>>,
    mode: Mode,
    buf: String,
    ws: String,
    sent: String,
    started: bool,
    n: usize,
    cut: bool,
    pre: String,
    back: Mode,
}

impl OutputParser {
    pub fn new(fmt: CallFormat, thinking_open: bool, tools: &[Json]) -> OutputParser {
        let mut schemas = HashMap::new();
        for t in tools {
            let f = t.get("function").unwrap_or(t);
            if let Some(name) = f.get("name").and_then(|n| n.as_str()) {
                let mut props = HashMap::new();
                if let Some(Json::Obj(p)) = f.get("parameters").and_then(|p| p.get("properties")) {
                    for (k, v) in p {
                        props.insert(
                            k.clone(),
                            v.get("type")
                                .and_then(|t| t.as_str())
                                .unwrap_or("string")
                                .to_string(),
                        );
                    }
                }
                schemas.insert(name.to_string(), props);
            }
        }
        let mode = if thinking_open {
            Mode::Reason
        } else {
            Mode::Content
        };
        OutputParser {
            fmt,
            schemas,
            mode,
            buf: String::new(),
            ws: String::new(),
            sent: String::new(),
            started: false,
            n: 0,
            cut: true,
            pre: String::new(),
            back: mode,
        }
    }

    pub fn feed(&mut self, text: &str, fin: bool) -> Vec<Event> {
        self.buf.push_str(text);
        let mut ev = Vec::new();
        loop {
            let again = if self.mode == Mode::Call {
                self.step_call(&mut ev, fin)
            } else {
                self.step_text(&mut ev, fin)
            };
            if !again {
                break;
            }
        }
        ev
    }

    pub fn finish(&mut self) -> Vec<Event> {
        self.feed("", true)
    }

    fn emit(mode: Mode, ev: &mut Vec<Event>, s: String) {
        ev.push(if mode == Mode::Reason {
            Event::Reason(s)
        } else {
            Event::Content(s)
        });
    }

    /// trailing whitespace waits: a tag, a tool call or the end may follow and swallow it
    fn say(&mut self, ev: &mut Vec<Event>, s: &str, boundary: bool) {
        let s = format!("{}{}", self.ws, s);
        let keep = py_rstrip(&s).len();
        if boundary {
            self.ws.clear();
            self.pre = s[keep..].to_string();
        } else {
            self.ws = s[keep..].to_string();
            self.pre.clear();
        }
        if keep > 0 {
            Self::emit(self.mode, ev, s[..keep].to_string());
        }
    }

    fn step_text(&mut self, ev: &mut Vec<Event>, fin: bool) -> bool {
        let tags: [&str; 2] = if self.mode == Mode::Content {
            [THINK_O, CALL_O]
        } else {
            [THINK_C, CALL_O]
        };
        let b: String = if self.cut {
            self.buf.trim_start_matches('\n').to_string()
        } else {
            self.buf.clone()
        };
        self.cut = self.cut && b.is_empty();
        let hit = tags.iter().filter_map(|t| b.find(t).map(|i| (i, *t))).min();
        if let Some((i, t)) = hit {
            self.say(ev, &b[..i], true);
            self.buf = b[i + t.len()..].to_string();
            if t == CALL_O {
                // also ends a reasoning block that was never closed
                self.back = self.mode;
                self.mode = Mode::Call;
                self.started = false;
                self.sent.clear();
            } else {
                self.mode = if t == THINK_O {
                    Mode::Reason
                } else {
                    Mode::Content
                };
                self.cut = true;
            }
            return true;
        }
        let k = if fin {
            b.len()
        } else {
            b.len() - hold(&b, &tags)
        };
        self.say(ev, &b[..k], fin);
        self.buf = b[k..].to_string();
        false
    }

    fn step_call(&mut self, ev: &mut Vec<Event>, fin: bool) -> bool {
        let b = self.buf.clone();
        let end = b.find(CALL_C);
        let closed = end.is_some() || fin;
        let body = &b[..end.unwrap_or(b.len())];
        let (name, args, done) = match self.fmt {
            CallFormat::Json => scan_json(body),
            CallFormat::Xml => scan_xml(body, closed, &self.schemas),
        };
        if name.is_none() && !closed {
            return false; // not even the name yet
        }
        let name = match name {
            Some(n) => n,
            None => {
                // not a call after all: keep the text, as Ember does
                let text = format!(
                    "{}{}{}",
                    self.pre,
                    CALL_O,
                    match end {
                        Some(e) => &b[..e + CALL_C.len()],
                        None => &b[..],
                    }
                );
                Self::emit(self.back, ev, text);
                self.buf = match end {
                    Some(e) => b[e + CALL_C.len()..].to_string(),
                    None => String::new(),
                };
                self.mode = self.back;
                return end.is_some();
            }
        };
        if !self.started {
            ev.push(Event::CallStart {
                index: self.n,
                name,
            });
            self.started = true;
            self.n += 1;
        }
        if args.len() > self.sent.len() {
            ev.push(Event::CallArgs {
                index: self.n - 1,
                fragment: args[self.sent.len()..].to_string(),
            });
            self.sent = args;
        }
        if closed {
            ev.push(Event::CallEnd {
                index: self.n - 1,
                complete: done,
            });
            self.buf = match end {
                Some(e) => b[e + CALL_C.len()..].to_string(),
                None => String::new(),
            };
            self.mode = Mode::Content;
            self.cut = true;
        }
        closed && end.is_some()
    }
}

/// The parsed pieces of a whole output.
#[derive(Debug, Default, PartialEq)]
pub struct Parsed {
    pub reasoning: String,
    pub content: String,
    /// (name, arguments text)
    pub calls: Vec<(String, String)>,
}

/// One-shot or chunked parse (text cut at the given byte offsets, which must be char boundaries).
pub fn parse(
    fmt: CallFormat,
    text: &str,
    cuts: &[usize],
    thinking_open: bool,
    tools: &[Json],
) -> Parsed {
    let mut p = OutputParser::new(fmt, thinking_open, tools);
    let mut events = Vec::new();
    let mut pts = vec![0usize];
    pts.extend_from_slice(cuts);
    pts.push(text.len());
    for w in pts.windows(2) {
        events.extend(p.feed(&text[w[0]..w[1]], false));
    }
    events.extend(p.finish());
    let mut out = Parsed::default();
    for e in events {
        match e {
            Event::Reason(s) => out.reasoning.push_str(&s),
            Event::Content(s) => out.content.push_str(&s),
            Event::CallStart { name, .. } => out.calls.push((name, String::new())),
            Event::CallArgs { index, fragment } => out.calls[index].1.push_str(&fragment),
            Event::CallEnd { .. } => {}
        }
    }
    out
}
