//! Constrained decoding: a byte-level automaton for JSON that the sampler consults. `response_format: {"type": "json_object"}` of the OpenAI API (and the same in Ollama and LM Studio) promises that the
//! answer parses; here the promise is kept by construction: before a token is drawn, the candidates whose bytes would break the grammar are removed, and the generation ends when the top-level object
//! is closed. `response_format: json_schema` goes further: the automaton follows a JSON Schema (see the second half of this file).
//!
//! The automaton is the grammar of RFC 8259 with an object at the top (what `json_object` means). It reads bytes, so a token that spells several structural characters (`"},`) is checked as a whole,
//! and it is small enough to clone (a stack of open containers and a few flags): `accepts` runs the candidate on a copy, `advance` on the real state. Runs of white space are capped (a model that is
//! stuck can emit new lines forever); strings take any byte from 0x20 up (the continuation bytes of a character split over two tokens included) with the JSON escapes checked.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Frame {
    Obj,
    Arr,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum St {
    /// a value may start here (white space allowed)
    Value,
    /// right after `[`: a value or `]`
    ArrFirst,
    /// right after `{`: a key or `}`
    KeyOrEnd,
    /// after `,` in an object: a key
    Key,
    /// after a key: `:`
    Colon,
    /// after a value inside a container: `,` or the closer
    After,
    /// inside a string (`key`: an object key, so the colon follows)
    Str { key: bool, esc: Esc },
    Num(Num),
    /// a literal (`true`, `false`, `null`): the text and how many bytes were read
    Lit { word: &'static [u8], at: u8 },
    /// the top-level object is closed
    Done,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Esc {
    None,
    /// after a backslash
    Slash,
    /// inside \uXXXX: hex digits still to come
    Hex(u8),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Num {
    Minus,
    Zero,
    Int,
    Dot,
    Frac,
    E,
    ESign,
    Exp,
}

impl Num {
    /// can the number end here
    fn complete(self) -> bool {
        matches!(self, Num::Zero | Num::Int | Num::Frac | Num::Exp)
    }
}

/// The longest run of white space between two tokens of the JSON text.
const MAX_WS_RUN: u8 = 24;

fn is_ws(b: u8) -> bool {
    matches!(b, b' ' | b'\t' | b'\n' | b'\r')
}

#[derive(Clone, Debug)]
pub struct Plain {
    stack: Vec<Frame>,
    st: St,
    ws: u8,
}

impl Plain {
    /// Any JSON object.
    pub fn json_object() -> Plain {
        Plain {
            stack: Vec::new(),
            st: St::Value,
            ws: 0,
        }
    }

    /// The top-level object is closed: the answer is complete.
    pub fn is_complete(&self) -> bool {
        self.st == St::Done
    }

    /// Would `bytes` be a valid continuation (nothing is changed).
    pub fn accepts(&self, bytes: &[u8]) -> bool {
        if bytes.is_empty() || self.is_complete() {
            return false;
        }
        let mut c = self.clone();
        bytes.iter().all(|&b| c.step(b))
    }

    /// Consume `bytes` (false: the grammar is broken and the state is unspecified).
    pub fn advance(&mut self, bytes: &[u8]) -> bool {
        bytes.iter().all(|&b| self.step(b))
    }

    fn after_value(&mut self) {
        self.st = if self.stack.is_empty() { St::Done } else { St::After };
    }

    /// The first byte of a value. At the top level only `{` is a value (json_object).
    fn start_value(&mut self, b: u8) -> bool {
        let top = self.stack.is_empty();
        match b {
            b'{' => {
                self.stack.push(Frame::Obj);
                self.st = St::KeyOrEnd;
                true
            }
            _ if top => false,
            b'[' => {
                self.stack.push(Frame::Arr);
                self.st = St::ArrFirst;
                true
            }
            b'"' => {
                self.st = St::Str { key: false, esc: Esc::None };
                true
            }
            b'-' => {
                self.st = St::Num(Num::Minus);
                true
            }
            b'0' => {
                self.st = St::Num(Num::Zero);
                true
            }
            b'1'..=b'9' => {
                self.st = St::Num(Num::Int);
                true
            }
            b't' => {
                self.st = St::Lit { word: b"true", at: 1 };
                true
            }
            b'f' => {
                self.st = St::Lit { word: b"false", at: 1 };
                true
            }
            b'n' => {
                self.st = St::Lit { word: b"null", at: 1 };
                true
            }
            _ => false,
        }
    }

    fn step(&mut self, b: u8) -> bool {
        // white space: allowed between tokens, in a capped run
        let ws_ok = matches!(self.st, St::Value | St::ArrFirst | St::KeyOrEnd | St::Key | St::Colon | St::After | St::Done);
        if ws_ok && is_ws(b) {
            self.ws += 1;
            return self.ws <= MAX_WS_RUN;
        }
        if !matches!(self.st, St::Str { .. }) {
            self.ws = 0;
        }
        match self.st {
            St::Done => false,
            St::Value => self.start_value(b),
            St::ArrFirst => {
                if b == b']' {
                    self.stack.pop();
                    self.after_value();
                    true
                } else {
                    self.start_value(b)
                }
            }
            St::KeyOrEnd => match b {
                b'"' => {
                    self.st = St::Str { key: true, esc: Esc::None };
                    true
                }
                b'}' => {
                    self.stack.pop();
                    self.after_value();
                    true
                }
                _ => false,
            },
            St::Key => {
                if b == b'"' {
                    self.st = St::Str { key: true, esc: Esc::None };
                    true
                } else {
                    false
                }
            }
            St::Colon => {
                if b == b':' {
                    self.st = St::Value;
                    true
                } else {
                    false
                }
            }
            St::After => match (self.stack.last().copied(), b) {
                (Some(Frame::Obj), b',') => {
                    self.st = St::Key;
                    true
                }
                (Some(Frame::Arr), b',') => {
                    self.st = St::Value;
                    true
                }
                (Some(Frame::Obj), b'}') | (Some(Frame::Arr), b']') => {
                    self.stack.pop();
                    self.after_value();
                    true
                }
                _ => false,
            },
            St::Str { key, esc } => match esc {
                Esc::None => match b {
                    b'"' => {
                        if key {
                            self.st = St::Colon;
                        } else {
                            self.after_value();
                        }
                        true
                    }
                    b'\\' => {
                        self.st = St::Str { key, esc: Esc::Slash };
                        true
                    }
                    0..=0x1f => false,
                    _ => true,
                },
                Esc::Slash => match b {
                    b'"' | b'\\' | b'/' | b'b' | b'f' | b'n' | b'r' | b't' => {
                        self.st = St::Str { key, esc: Esc::None };
                        true
                    }
                    b'u' => {
                        self.st = St::Str { key, esc: Esc::Hex(4) };
                        true
                    }
                    _ => false,
                },
                Esc::Hex(n) => {
                    if b.is_ascii_hexdigit() {
                        self.st = St::Str { key, esc: if n == 1 { Esc::None } else { Esc::Hex(n - 1) } };
                        true
                    } else {
                        false
                    }
                }
            },
            St::Lit { word, at } => {
                if word[at as usize] == b {
                    if at as usize + 1 == word.len() {
                        self.after_value();
                    } else {
                        self.st = St::Lit { word, at: at + 1 };
                    }
                    true
                } else {
                    false
                }
            }
            St::Num(n) => {
                let next = match (n, b) {
                    (Num::Minus, b'0') => Some(Num::Zero),
                    (Num::Minus, b'1'..=b'9') => Some(Num::Int),
                    (Num::Zero, b'.') | (Num::Int, b'.') => Some(Num::Dot),
                    (Num::Int, b'0'..=b'9') => Some(Num::Int),
                    (Num::Dot, b'0'..=b'9') | (Num::Frac, b'0'..=b'9') => Some(Num::Frac),
                    (Num::Zero, b'e' | b'E') | (Num::Int, b'e' | b'E') | (Num::Frac, b'e' | b'E') => Some(Num::E),
                    (Num::E, b'+' | b'-') => Some(Num::ESign),
                    (Num::E, b'0'..=b'9') | (Num::ESign, b'0'..=b'9') | (Num::Exp, b'0'..=b'9') => Some(Num::Exp),
                    _ => None,
                };
                match next {
                    Some(m) => {
                        self.st = St::Num(m);
                        true
                    }
                    // the number ends before this byte, which belongs to what follows
                    None if n.complete() => {
                        self.after_value();
                        self.step(b)
                    }
                    None => false,
                }
            }
        }
    }
}

// =================================================================================================================================================================================================
// JSON Schema: the same byte-level reading, guided by a schema (`response_format: {"type": "json_schema", ...}`).
//
// Supported: type (one of object, array, string, number, integer, boolean, null, or a list of them), properties / required / additionalProperties, items / minItems / maxItems, enum and const,
// anyOf / oneOf (the first alternative whose first byte fits wins: alternatives that start alike are not told apart), `$ref` to `#`, `#/$defs/x` and `#/definitions/x` (recursive schemas work),
// true as "anything". Properties are written in the order the schema lists them (optional ones may be left out; a required one may not be skipped), keys that are not declared are refused unless
// additionalProperties allows them (absent means no, as in OpenAI's strict mode). Not checked: pattern, format, minLength / maxLength, minimum / maximum, uniqueItems, allOf.
// =================================================================================================================================================================================================

use forge_core::Json;
use std::sync::Arc;

#[derive(Clone, Debug)]
struct Prop {
    /// the key as it is written between the quotes (JSON escapes included)
    name: Vec<u8>,
    node: usize,
    required: bool,
}

#[derive(Clone, Debug)]
enum Node {
    Null,
    Bool,
    Int,
    Num,
    /// a string; with `options` only these contents (as written between the quotes)
    Str { options: Option<Vec<Vec<u8>>> },
    /// enum / const values that are not all strings: their JSON text
    Lits(Vec<Vec<u8>>),
    Obj { props: Vec<Prop>, extra: Option<usize> },
    Arr { items: usize, min: usize, max: usize },
    Union(Vec<usize>),
}

/// A compiled schema: nodes in an arena (so that `$ref` cycles are just indices).
#[derive(Clone, Debug)]
pub struct SchemaDoc {
    nodes: Vec<Node>,
    root: usize,
}

struct Builder<'a> {
    root_json: &'a Json,
    nodes: Vec<Node>,
    refs: Vec<(String, usize)>,
    any: Option<usize>,
}

const MAX_PROPS: usize = 64;
const MAX_DEPTH: usize = 40;

fn escaped(s: &str) -> Vec<u8> {
    let q = Json::Str(s.to_string()).to_string();
    q.as_bytes()[1..q.len() - 1].to_vec()
}

impl<'a> Builder<'a> {
    fn push(&mut self, n: Node) -> usize {
        self.nodes.push(n);
        self.nodes.len() - 1
    }

    /// "anything": a union of every JSON value, recursive through arrays and objects.
    fn any(&mut self) -> usize {
        if let Some(a) = self.any {
            return a;
        }
        let u = self.push(Node::Union(Vec::new()));
        self.any = Some(u);
        let kinds = [
            self.push(Node::Null),
            self.push(Node::Bool),
            self.push(Node::Num),
            self.push(Node::Str { options: None }),
            self.push(Node::Arr { items: u, min: 0, max: usize::MAX }),
            self.push(Node::Obj { props: Vec::new(), extra: Some(u) }),
        ];
        self.nodes[u] = Node::Union(kinds.to_vec());
        u
    }

    fn build(&mut self, s: &Json, depth: usize) -> Result<usize, String> {
        if depth > MAX_DEPTH {
            return Err("the schema nests too deeply".into());
        }
        match s {
            Json::Bool(true) => return Ok(self.any()),
            Json::Bool(false) => return Err("a schema that is false allows nothing".into()),
            Json::Obj(_) => {}
            _ => return Err("a schema must be an object (or true)".into()),
        }
        if let Some(r) = s.get("$ref").and_then(|r| r.as_str()) {
            if let Some((_, i)) = self.refs.iter().find(|(k, _)| k == r) {
                return Ok(*i);
            }
            let target = if r == "#" {
                self.root_json
            } else if let Some(path) = r.strip_prefix("#/") {
                let mut cur = self.root_json;
                for part in path.split('/') {
                    let key = part.replace("~1", "/").replace("~0", "~");
                    cur = cur.get(&key).ok_or_else(|| format!("$ref {r} does not resolve"))?;
                }
                cur
            } else {
                return Err(format!("$ref {r}: only references inside the schema are supported"));
            };
            let slot = self.push(Node::Null);
            self.refs.push((r.to_string(), slot));
            let built = self.build(target, depth + 1)?;
            self.nodes[slot] = Node::Union(vec![built]);
            return Ok(slot);
        }
        if let Some(c) = s.get("const") {
            return Ok(self.literals(std::slice::from_ref(c)));
        }
        if let Some(e) = s.get("enum").and_then(|e| e.as_array()) {
            if e.is_empty() {
                return Err("an enum needs at least one value".into());
            }
            return Ok(self.literals(e));
        }
        for key in ["anyOf", "oneOf"] {
            if let Some(alts) = s.get(key).and_then(|a| a.as_array()) {
                let mut ids = Vec::new();
                for a in alts {
                    ids.push(self.build(a, depth + 1)?);
                }
                return Ok(self.push(Node::Union(ids)));
            }
        }
        if s.get("allOf").is_some() {
            return Err("allOf is not supported".into());
        }
        let types: Vec<String> = match s.get("type") {
            Some(Json::Str(t)) => vec![t.clone()],
            Some(Json::Arr(a)) => a.iter().filter_map(|t| t.as_str().map(|x| x.to_string())).collect(),
            Some(_) => return Err("type must be a string or a list of strings".into()),
            None => {
                // no type: the keywords say what it is
                if s.get("properties").is_some() {
                    vec!["object".into()]
                } else if s.get("items").is_some() {
                    vec!["array".into()]
                } else {
                    return Ok(self.any());
                }
            }
        };
        let mut ids = Vec::new();
        for t in &types {
            ids.push(self.typed(t, s, depth)?);
        }
        if ids.len() == 1 {
            Ok(ids[0])
        } else {
            Ok(self.push(Node::Union(ids)))
        }
    }

    fn literals(&mut self, values: &[Json]) -> usize {
        if values.iter().all(|v| matches!(v, Json::Str(_))) {
            let options = values.iter().map(|v| escaped(v.as_str().unwrap())).collect();
            self.push(Node::Str { options: Some(options) })
        } else {
            let options = values.iter().map(|v| v.to_string().into_bytes()).collect();
            self.push(Node::Lits(options))
        }
    }

    fn typed(&mut self, t: &str, s: &Json, depth: usize) -> Result<usize, String> {
        Ok(match t {
            "null" => self.push(Node::Null),
            "boolean" => self.push(Node::Bool),
            "integer" => self.push(Node::Int),
            "number" => self.push(Node::Num),
            "string" => self.push(Node::Str { options: None }),
            "array" => {
                let items = match s.get("items") {
                    Some(i) => self.build(i, depth + 1)?,
                    None => self.any(),
                };
                let min = s.get("minItems").and_then(|v| v.as_usize()).unwrap_or(0);
                let max = s.get("maxItems").and_then(|v| v.as_usize()).unwrap_or(usize::MAX);
                if min > max {
                    return Err("minItems is larger than maxItems".into());
                }
                self.push(Node::Arr { items, min, max })
            }
            "object" => {
                let required: Vec<&str> = s
                    .get("required")
                    .and_then(|r| r.as_array())
                    .map(|a| a.iter().filter_map(|x| x.as_str()).collect())
                    .unwrap_or_default();
                let mut props = Vec::new();
                let slot = self.push(Node::Null);
                if let Some(Json::Obj(members)) = s.get("properties") {
                    if members.len() > MAX_PROPS {
                        return Err(format!("an object may declare at most {MAX_PROPS} properties"));
                    }
                    for (k, v) in members {
                        let node = self.build(v, depth + 1)?;
                        props.push(Prop { name: escaped(k), node, required: required.contains(&k.as_str()) });
                    }
                }
                for r in &required {
                    if !props.iter().any(|p| p.name == escaped(r)) {
                        return Err(format!("required property {r:?} is not declared"));
                    }
                }
                let extra = match s.get("additionalProperties") {
                    Some(Json::Bool(true)) => Some(self.any()),
                    Some(Json::Obj(_)) => Some(self.build(s.get("additionalProperties").unwrap(), depth + 1)?),
                    Some(Json::Bool(false)) => None,
                    // absent: an object without declared properties is free-form, one with them takes only those
                    _ => (props.is_empty()).then(|| self.any()),
                };
                self.nodes[slot] = Node::Obj { props, extra };
                slot
            }
            other => return Err(format!("unknown type {other:?}")),
        })
    }
}

impl SchemaDoc {
    /// Compile a JSON Schema; an error names what is not supported.
    pub fn from_json(schema: &Json) -> Result<SchemaDoc, String> {
        let mut b = Builder { root_json: schema, nodes: Vec::new(), refs: Vec::new(), any: None };
        let root = b.build(schema, 0)?;
        Ok(SchemaDoc { nodes: b.nodes, root })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SFrame {
    Obj { node: u32, next: u16, seen: u64 },
    Arr { node: u32, count: u32 },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SSt {
    Value,
    ArrFirst,
    KeyOrEnd,
    Key,
    Colon,
    After,
    /// a string value; `enum_`: its content must be one of the node's options (matched byte by byte in `buf`)
    Str { esc: Esc, enum_: bool },
    /// an object key, free-form (additionalProperties allows other keys than the declared ones)
    KeyFree { esc: Esc },
    /// an object key that must be one of the declared names still allowed (matched byte by byte in `buf`)
    KeyDecl,
    Num { n: Num, int_only: bool },
    Lit { word: &'static [u8], at: u8 },
    /// one of the literal options of an enum / const (`buf` holds what has been read)
    Lits,
    Done,
}

/// The schema-guided automaton.
#[derive(Clone, Debug)]
pub struct Sch {
    doc: Arc<SchemaDoc>,
    stack: Vec<SFrame>,
    st: SSt,
    /// the node of the value that is being read or about to be read
    cur: usize,
    ws: u8,
    buf: Vec<u8>,
}

fn num_next(n: Num, b: u8, int_only: bool) -> Option<Num> {
    match (n, b) {
        (Num::Minus, b'0') => Some(Num::Zero),
        (Num::Minus, b'1'..=b'9') => Some(Num::Int),
        (Num::Zero, b'.') | (Num::Int, b'.') if !int_only => Some(Num::Dot),
        (Num::Int, b'0'..=b'9') => Some(Num::Int),
        (Num::Dot, b'0'..=b'9') | (Num::Frac, b'0'..=b'9') => Some(Num::Frac),
        (Num::Zero, b'e' | b'E') | (Num::Int, b'e' | b'E') | (Num::Frac, b'e' | b'E') if !int_only => Some(Num::E),
        (Num::E, b'+' | b'-') => Some(Num::ESign),
        (Num::E, b'0'..=b'9') | (Num::ESign, b'0'..=b'9') | (Num::Exp, b'0'..=b'9') => Some(Num::Exp),
        _ => None,
    }
}

impl Sch {
    fn new(doc: Arc<SchemaDoc>) -> Sch {
        let cur = doc.root;
        Sch { doc, stack: Vec::new(), st: SSt::Value, cur, ws: 0, buf: Vec::new() }
    }

    fn after_value(&mut self) {
        match self.stack.last_mut() {
            None => self.st = SSt::Done,
            Some(SFrame::Arr { count, .. }) => {
                *count += 1;
                self.st = SSt::After;
            }
            Some(SFrame::Obj { .. }) => self.st = SSt::After,
        }
    }

    /// Which node reads a value that starts with byte `b`: a union is resolved by the first alternative that fits.
    fn resolve(&self, node: usize, b: u8, depth: usize) -> Option<usize> {
        if depth > MAX_DEPTH {
            return None;
        }
        match &self.doc.nodes[node] {
            Node::Union(alts) => alts.iter().find_map(|&a| self.resolve(a, b, depth + 1)),
            Node::Null => (b == b'n').then_some(node),
            Node::Bool => (b == b't' || b == b'f').then_some(node),
            Node::Int | Node::Num => (b == b'-' || b.is_ascii_digit()).then_some(node),
            Node::Str { .. } => (b == b'"').then_some(node),
            Node::Lits(opts) => opts.iter().any(|o| o.first() == Some(&b)).then_some(node),
            Node::Obj { .. } => (b == b'{').then_some(node),
            Node::Arr { .. } => (b == b'[').then_some(node),
        }
    }

    fn start_value(&mut self, b: u8) -> bool {
        let Some(node) = self.resolve(self.cur, b, 0) else { return false };
        self.cur = node;
        match &self.doc.nodes[node] {
            Node::Null => self.st = SSt::Lit { word: b"null", at: 1 },
            Node::Bool => {
                self.st = if b == b't' { SSt::Lit { word: b"true", at: 1 } } else { SSt::Lit { word: b"false", at: 1 } };
            }
            Node::Int | Node::Num => {
                let int_only = matches!(self.doc.nodes[node], Node::Int);
                let n = match b {
                    b'-' => Num::Minus,
                    b'0' => Num::Zero,
                    _ => Num::Int,
                };
                self.st = SSt::Num { n, int_only };
            }
            Node::Str { options } => {
                self.buf.clear();
                self.st = SSt::Str { esc: Esc::None, enum_: options.is_some() };
            }
            Node::Lits(opts) => {
                self.buf.clear();
                self.buf.push(b);
                // a literal that nothing extends is complete at once
                let done = opts.iter().any(|o| *o == self.buf) && !opts.iter().any(|o| o.len() > self.buf.len() && o.starts_with(&self.buf));
                if done {
                    self.after_value();
                } else {
                    self.st = SSt::Lits;
                }
            }
            Node::Obj { .. } => {
                self.stack.push(SFrame::Obj { node: node as u32, next: 0, seen: 0 });
                self.st = SSt::KeyOrEnd;
            }
            Node::Arr { .. } => {
                self.stack.push(SFrame::Arr { node: node as u32, count: 0 });
                self.st = SSt::ArrFirst;
            }
            Node::Union(_) => return false,
        }
        true
    }

    /// The declared properties a key may name now: not seen yet, after the last one written, and not past a required one that is still missing.
    fn allowed_props(&self, node: u32, next: u16, seen: u64) -> Vec<usize> {
        let Node::Obj { props, .. } = &self.doc.nodes[node as usize] else { return Vec::new() };
        let mut out = Vec::new();
        for (i, p) in props.iter().enumerate() {
            if i < next as usize || seen & (1u64 << i) != 0 {
                continue;
            }
            out.push(i);
            if p.required {
                break;
            }
        }
        out
    }

    fn missing_required(&self, node: u32, seen: u64) -> bool {
        let Node::Obj { props, .. } = &self.doc.nodes[node as usize] else { return false };
        props.iter().enumerate().any(|(i, p)| p.required && seen & (1u64 << i) == 0)
    }

    fn obj_top(&self) -> Option<(u32, u16, u64)> {
        match self.stack.last() {
            Some(SFrame::Obj { node, next, seen }) => Some((*node, *next, *seen)),
            _ => None,
        }
    }

    fn can_take_more_keys(&self, node: u32, next: u16, seen: u64) -> bool {
        let Node::Obj { extra, .. } = &self.doc.nodes[node as usize] else { return false };
        extra.is_some() || !self.allowed_props(node, next, seen).is_empty()
    }

    fn step(&mut self, b: u8) -> bool {
        let ws_ok = matches!(self.st, SSt::Value | SSt::ArrFirst | SSt::KeyOrEnd | SSt::Key | SSt::Colon | SSt::After | SSt::Done);
        if ws_ok && is_ws(b) {
            self.ws += 1;
            return self.ws <= MAX_WS_RUN;
        }
        if !matches!(self.st, SSt::Str { .. } | SSt::KeyFree { .. } | SSt::KeyDecl) {
            self.ws = 0;
        }
        match self.st {
            SSt::Done => false,
            SSt::Value => self.start_value(b),
            SSt::ArrFirst => {
                let Some(SFrame::Arr { node, .. }) = self.stack.last().copied() else { return false };
                let Node::Arr { items, min, max } = self.doc.nodes[node as usize] else { return false };
                if b == b']' {
                    if min == 0 {
                        self.stack.pop();
                        self.after_value();
                        return true;
                    }
                    return false;
                }
                if max == 0 {
                    return false;
                }
                self.cur = items;
                self.start_value(b)
            }
            SSt::KeyOrEnd | SSt::Key => {
                let Some((node, next, seen)) = self.obj_top() else { return false };
                if b == b'}' && self.st == SSt::KeyOrEnd {
                    if self.missing_required(node, seen) {
                        return false;
                    }
                    self.stack.pop();
                    self.after_value();
                    return true;
                }
                if b != b'"' || !self.can_take_more_keys(node, next, seen) {
                    return false;
                }
                self.buf.clear();
                let free = matches!(&self.doc.nodes[node as usize], Node::Obj { extra: Some(_), .. });
                self.st = if free { SSt::KeyFree { esc: Esc::None } } else { SSt::KeyDecl };
                true
            }
            SSt::KeyDecl => {
                let Some((node, next, seen)) = self.obj_top() else { return false };
                let Node::Obj { props, .. } = &self.doc.nodes[node as usize] else { return false };
                let allowed = self.allowed_props(node, next, seen);
                if b == b'"' {
                    let Some(&i) = allowed.iter().find(|&&i| props[i].name == self.buf) else { return false };
                    let value_node = props[i].node;
                    if let Some(SFrame::Obj { next, seen, .. }) = self.stack.last_mut() {
                        *next = i as u16 + 1;
                        *seen |= 1u64 << i;
                    }
                    self.cur = value_node;
                    self.st = SSt::Colon;
                    return true;
                }
                let want = self.buf.len();
                if allowed.iter().any(|&i| props[i].name.len() > want && props[i].name.starts_with(&self.buf) && props[i].name[want] == b) {
                    self.buf.push(b);
                    true
                } else {
                    false
                }
            }
            SSt::KeyFree { esc } => {
                let ok = match esc {
                    Esc::None => match b {
                        b'"' => {
                            let Some((node, _, seen)) = self.obj_top() else { return false };
                            let Node::Obj { props, extra, .. } = &self.doc.nodes[node as usize] else { return false };
                            let declared = props.iter().position(|p| p.name == self.buf);
                            let value_node = match (declared, extra) {
                                (Some(i), _) => {
                                    if seen & (1u64 << i) != 0 {
                                        return false; // a key twice
                                    }
                                    if let Some(SFrame::Obj { seen, .. }) = self.stack.last_mut() {
                                        *seen |= 1u64 << i;
                                    }
                                    props[i].node
                                }
                                (None, Some(e)) => *e,
                                (None, None) => return false,
                            };
                            self.cur = value_node;
                            self.st = SSt::Colon;
                            return true;
                        }
                        b'\\' => {
                            self.st = SSt::KeyFree { esc: Esc::Slash };
                            true
                        }
                        0..=0x1f => false,
                        _ => true,
                    },
                    Esc::Slash => match b {
                        b'"' | b'\\' | b'/' | b'b' | b'f' | b'n' | b'r' | b't' => {
                            self.st = SSt::KeyFree { esc: Esc::None };
                            true
                        }
                        b'u' => {
                            self.st = SSt::KeyFree { esc: Esc::Hex(4) };
                            true
                        }
                        _ => false,
                    },
                    Esc::Hex(n) => {
                        if b.is_ascii_hexdigit() {
                            self.st = SSt::KeyFree { esc: if n == 1 { Esc::None } else { Esc::Hex(n - 1) } };
                            true
                        } else {
                            false
                        }
                    }
                };
                if ok && !matches!(self.st, SSt::Colon) {
                    self.buf.push(b);
                }
                ok
            }
            SSt::Colon => {
                if b == b':' {
                    self.st = SSt::Value;
                    true
                } else {
                    false
                }
            }
            SSt::After => match self.stack.last().copied() {
                Some(SFrame::Obj { node, next, seen }) => match b {
                    b',' => {
                        if self.can_take_more_keys(node, next, seen) {
                            self.st = SSt::Key;
                            true
                        } else {
                            false
                        }
                    }
                    b'}' => {
                        if self.missing_required(node, seen) {
                            return false;
                        }
                        self.stack.pop();
                        self.after_value();
                        true
                    }
                    _ => false,
                },
                Some(SFrame::Arr { node, count }) => {
                    let Node::Arr { items, min, max } = self.doc.nodes[node as usize] else { return false };
                    match b {
                        b',' => {
                            if (count as usize) < max {
                                self.cur = items;
                                self.st = SSt::Value;
                                true
                            } else {
                                false
                            }
                        }
                        b']' => {
                            if (count as usize) < min {
                                return false;
                            }
                            self.stack.pop();
                            self.after_value();
                            true
                        }
                        _ => false,
                    }
                }
                None => false,
            },
            SSt::Str { esc, enum_ } => {
                if enum_ {
                    let Node::Str { options: Some(opts) } = &self.doc.nodes[self.cur] else { return false };
                    if b == b'"' {
                        if opts.iter().any(|o| *o == self.buf) {
                            self.after_value();
                            return true;
                        }
                        return false;
                    }
                    let want = self.buf.len();
                    if opts.iter().any(|o| o.len() > want && o.starts_with(&self.buf) && o[want] == b) {
                        self.buf.push(b);
                        return true;
                    }
                    return false;
                }
                match esc {
                    Esc::None => match b {
                        b'"' => {
                            self.after_value();
                            true
                        }
                        b'\\' => {
                            self.st = SSt::Str { esc: Esc::Slash, enum_: false };
                            true
                        }
                        0..=0x1f => false,
                        _ => true,
                    },
                    Esc::Slash => match b {
                        b'"' | b'\\' | b'/' | b'b' | b'f' | b'n' | b'r' | b't' => {
                            self.st = SSt::Str { esc: Esc::None, enum_: false };
                            true
                        }
                        b'u' => {
                            self.st = SSt::Str { esc: Esc::Hex(4), enum_: false };
                            true
                        }
                        _ => false,
                    },
                    Esc::Hex(n) => {
                        if b.is_ascii_hexdigit() {
                            self.st = SSt::Str { esc: if n == 1 { Esc::None } else { Esc::Hex(n - 1) }, enum_: false };
                            true
                        } else {
                            false
                        }
                    }
                }
            }
            SSt::Lit { word, at } => {
                if word[at as usize] == b {
                    if at as usize + 1 == word.len() {
                        self.after_value();
                    } else {
                        self.st = SSt::Lit { word, at: at + 1 };
                    }
                    true
                } else {
                    false
                }
            }
            SSt::Lits => {
                let Node::Lits(opts) = &self.doc.nodes[self.cur] else { return false };
                let mut next = self.buf.clone();
                next.push(b);
                if opts.iter().any(|o| o.starts_with(&next)) {
                    self.buf = next;
                    let exact = opts.iter().any(|o| *o == self.buf);
                    let longer = opts.iter().any(|o| o.len() > self.buf.len() && o.starts_with(&self.buf));
                    if exact && !longer {
                        self.after_value();
                    }
                    true
                } else if opts.iter().any(|o| *o == self.buf) {
                    // the option read so far is complete and this byte belongs to what follows
                    self.after_value();
                    self.step(b)
                } else {
                    false
                }
            }
            SSt::Num { n, int_only } => match num_next(n, b, int_only) {
                Some(m) => {
                    self.st = SSt::Num { n: m, int_only };
                    true
                }
                None if n.complete() => {
                    self.after_value();
                    self.step(b)
                }
                None => false,
            },
        }
    }
}

/// What the sampler consults: any JSON object, or a document that follows a schema.
#[derive(Clone, Debug)]
pub enum Constraint {
    Plain(Plain),
    Schema(Sch),
}

impl Constraint {
    /// Any JSON object.
    pub fn json_object() -> Constraint {
        Constraint::Plain(Plain::json_object())
    }

    /// A document that follows `doc`.
    pub fn from_schema(doc: Arc<SchemaDoc>) -> Constraint {
        Constraint::Schema(Sch::new(doc))
    }

    /// The answer is complete (the top-level value is closed).
    pub fn is_complete(&self) -> bool {
        match self {
            Constraint::Plain(p) => p.is_complete(),
            Constraint::Schema(s) => s.st == SSt::Done,
        }
    }

    /// Would `bytes` be a valid continuation (nothing is changed).
    pub fn accepts(&self, bytes: &[u8]) -> bool {
        if bytes.is_empty() || self.is_complete() {
            return false;
        }
        match self {
            Constraint::Plain(p) => p.accepts(bytes),
            Constraint::Schema(s) => {
                let mut c = s.clone();
                bytes.iter().all(|&b| c.step(b))
            }
        }
    }

    /// Consume `bytes` (false: the grammar is broken and the state is unspecified).
    pub fn advance(&mut self, bytes: &[u8]) -> bool {
        match self {
            Constraint::Plain(p) => p.advance(bytes),
            Constraint::Schema(s) => bytes.iter().all(|&b| s.step(b)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use forge_core::Json;
    use std::sync::Arc;

    fn feed(s: &str) -> Option<Constraint> {
        let mut c = Constraint::json_object();
        if c.advance(s.as_bytes()) {
            Some(c)
        } else {
            None
        }
    }

    #[test]
    fn valid_objects_are_accepted_and_complete_exactly_at_the_closing_brace() {
        for text in [
            r#"{}"#,
            r#"{"a": 1}"#,
            r#"{ "a" : [1, 2.5, -3e+2, 0, 0.5E-3, true, false, null, "x\"y\\z\né", {"b": []}] , "c" : {} }"#,
            "{\n  \"name\": \"Ada\",\n  \"tags\": [\"a\", \"b\"],\n  \"n\": 12\n}",
            r#"{"é":"日本語 and emoji 😀"}"#,
        ] {
            let c = feed(text).unwrap_or_else(|| panic!("rejected: {text}"));
            assert!(c.is_complete(), "not complete: {text}");
            // one byte short: valid so far but not complete
            let short = &text[..text.len() - 1];
            let c2 = feed(short).unwrap_or_else(|| panic!("a prefix was rejected: {short}"));
            assert!(!c2.is_complete(), "complete too early: {short}");
        }
    }

    #[test]
    fn broken_json_is_rejected_at_the_first_wrong_byte() {
        for text in [
            "[1]",                 // the top level must be an object
            "\"a\"",
            "hello",
            r#"{a: 1}"#,           // unquoted key
            r#"{"a" 1}"#,          // no colon
            r#"{"a": 1,}"#,        // trailing comma
            r#"{"a": [1,]}"#,
            r#"{"a": 01}"#,        // leading zero
            r#"{"a": 1.}"#,
            r#"{"a": .5}"#,
            r#"{"a": -}"#,
            r#"{"a": tru}"#,
            r#"{"a": "x\q"}"#,     // bad escape
            r#"{"a": "\u12G4"}"#,
            "{\"a\": \"line\nbreak\"}", // a raw control character in a string
            r#"{"a": 1}}"#,        // text after the end
            r#"{"a": }"#,
            r#"{"a": [}"#,
            r#"{"a": {]}"#,
        ] {
            assert!(feed(text).is_none(), "accepted: {text}");
        }
    }

    #[test]
    fn accepts_does_not_change_the_state_and_judges_a_token_as_a_whole() {
        let mut c = Constraint::json_object();
        assert!(c.advance(br#"{"a": [1"#));
        assert!(c.accepts(b"2, 3]"));
        assert!(c.accepts(b"]"));
        assert!(c.accepts(br#"], "b": "#));
        assert!(!c.accepts(b"}"), "the array is still open");
        assert!(!c.accepts(b"x"));
        assert!(!c.accepts(b""));
        assert!(c.advance(b"]}"));
        assert!(c.is_complete());
        assert!(!c.accepts(b" "), "nothing follows a complete answer");
    }

    #[test]
    fn a_number_ends_with_the_byte_that_follows_it() {
        let mut c = Constraint::json_object();
        assert!(c.advance(br#"{"a": 12"#));
        assert!(c.accepts(b","));
        assert!(c.accepts(b"}"));
        assert!(c.accepts(b" }"));
        assert!(!c.accepts(b"x"));
        let mut d = Constraint::json_object();
        assert!(d.advance(br#"{"a": -"#));
        assert!(!d.accepts(b"}"), "a minus alone is not a number");
        assert!(!d.accepts(b","));
        assert!(d.accepts(b"1"));
    }

    #[test]
    fn runs_of_white_space_are_capped() {
        let mut c = Constraint::json_object();
        assert!(c.advance(b"{"));
        assert!(c.advance(&[b' '; 24]));
        assert!(!c.accepts(b" "), "25 blanks in a row");
        assert!(c.accepts(b"\"a\""));
        // a string keeps its blanks
        let mut d = Constraint::json_object();
        assert!(d.advance(br#"{"a": ""#));
        assert!(d.advance(&[b' '; 200]));
        assert!(d.accepts(b"\""));
    }

    #[test]
    fn deep_nesting_is_tracked() {
        let mut c = Constraint::json_object();
        let open = r#"{"a":[{"b":[{"c":"#.repeat(1);
        assert!(c.advance(open.as_bytes()));
        assert!(c.advance(b"1}]}]}"));
        assert!(c.is_complete());
    }

    fn sch(schema: &str) -> Constraint {
        let doc = SchemaDoc::from_json(&Json::parse(schema).unwrap()).unwrap();
        Constraint::from_schema(Arc::new(doc))
    }

    /// the text is a complete document of the schema
    fn full(schema: &str, text: &str) -> bool {
        let mut c = sch(schema);
        c.advance(text.as_bytes()) && c.is_complete()
    }

    /// the text is rejected somewhere (or does not complete)
    fn bad(schema: &str, text: &str) -> bool {
        !full(schema, text)
    }

    const PERSON: &str = r#"{"type":"object","properties":{"name":{"type":"string"},"age":{"type":"integer"},"tags":{"type":"array","items":{"type":"string"}}},"required":["name","age"]}"#;

    #[test]
    fn an_object_schema_fixes_the_keys_their_order_and_their_types() {
        for t in [
            r#"{"name":"Ada","age":36}"#,
            r#"{"name":"Ada","age":36,"tags":["a","b"]}"#,
            r#"{ "name" : "Ada" , "age" : 36 , "tags" : [ ] }"#,
            "{\n  \"name\": \"Grace \\\"Amazing\\\" Hopper\",\n  \"age\": -1\n}",
        ] {
            assert!(full(PERSON, t), "rejected: {t}");
        }
        for t in [
            r#"{"age":36,"name":"Ada"}"#,           // the order of the schema
            r#"{"name":"Ada"}"#,                    // a required property is missing
            r#"{"name":"Ada","age":3.5}"#,          // an integer
            r#"{"name":"Ada","age":36,"extra":1}"#, // a key that is not declared
            r#"{"name":1,"age":2}"#,
            r#"{"name":"Ada","age":36,"tags":["a",1]}"#,
            r#"{"name":"Ada","age":36,"name":"B"}"#,
            r#"["a"]"#,
        ] {
            assert!(bad(PERSON, t), "accepted: {t}");
        }
        // keys are matched byte by byte: after `{"na` only `me` can follow
        let mut c = sch(PERSON);
        assert!(c.advance(br#"{"na"#));
        assert!(c.accepts(b"me\""));
        assert!(!c.accepts(b"x"));
        assert!(!c.accepts(b"g"));
    }

    #[test]
    fn enums_constants_and_unions() {
        let s = r#"{"type":"object","properties":{"color":{"enum":["red","green","blue"]},"n":{"enum":[1,2,10]},"flag":{"const":true},"v":{"type":["string","null"]}},"required":["color","n","flag","v"]}"#;
        assert!(full(s, r#"{"color":"green","n":10,"flag":true,"v":null}"#));
        assert!(full(s, r#"{"color":"red","n":1,"flag":true,"v":"x"}"#));
        assert!(full(s, r#"{"color":"blue","n":2,"flag":true,"v":"x"}"#));
        for t in [
            r#"{"color":"pink","n":1,"flag":true,"v":null}"#,
            r#"{"color":"red","n":3,"flag":true,"v":null}"#,
            r#"{"color":"red","n":1,"flag":false,"v":null}"#,
            r#"{"color":"red","n":1,"flag":true,"v":1}"#,
        ] {
            assert!(bad(s, t), "accepted: {t}");
        }
        let mut c = sch(s);
        assert!(c.advance(br#"{"color":"gr"#));
        assert!(c.accepts(b"een"));
        assert!(!c.accepts(b"x"));
        assert!(!c.accepts(b"\""), "`gr` is not one of the options");
    }

    #[test]
    fn a_token_that_closes_an_enum_string_and_goes_on_is_judged_as_a_whole() {
        let s = r#"{"type":"object","properties":{"sentiment":{"enum":["positive","negative","neutral"]},"confidence":{"type":"number"}},"required":["sentiment","confidence"]}"#;
        let mut c = sch(s);
        assert!(c.advance(b"{
  \"sentiment\": \"neg"));
        assert!(c.accepts(b"ative\","));
        assert!(c.accepts(b"ative\",
"));
        assert!(c.accepts(b"ative\""));
        assert!(!c.accepts(b"ative!"));
        assert!(!c.accepts(b"\""), "`neg` is not an option");
        assert!(c.advance(b"ative\""));
        assert!(c.accepts(b","));
        assert!(c.accepts(b",
  \"conf"));
        assert!(!c.accepts(b"}"), "confidence is required");
    }

    #[test]
    fn array_limits_and_open_objects() {
        let s = r#"{"type":"object","properties":{"xs":{"type":"array","items":{"type":"integer"},"minItems":1,"maxItems":2}},"required":["xs"]}"#;
        assert!(full(s, r#"{"xs":[1]}"#) && full(s, r#"{"xs":[1,2]}"#));
        assert!(bad(s, r#"{"xs":[]}"#) && bad(s, r#"{"xs":[1,2,3]}"#));
        assert!(full(r#"{"type":"object","additionalProperties":true}"#, r#"{"anything":[1,{"b":null}],"x":"y"}"#));
        assert!(full(r#"{"type":"object"}"#, r#"{"free":"form"}"#));
        let typed = r#"{"type":"object","additionalProperties":{"type":"integer"}}"#;
        assert!(full(typed, r#"{"a":1,"b":2}"#) && full(typed, "{}"));
        assert!(bad(typed, r#"{"a":"x"}"#));
        assert!(full(r#"{"type":"array","items":{"type":"number"}}"#, "[1, 2.5, -3e2]"));
        assert!(full(r#"{"type":"object","properties":{"a":{"type":"array","maxItems":0}},"required":["a"]}"#, r#"{"a":[]}"#));
    }

    #[test]
    fn references_and_alternatives() {
        let tree = r##"{"$defs":{"node":{"type":"object","properties":{"v":{"type":"integer"},"kids":{"type":"array","items":{"$ref":"#/$defs/node"}}},"required":["v"]}},"$ref":"#/$defs/node"}"##;
        assert!(full(tree, r#"{"v":1,"kids":[{"v":2},{"v":3,"kids":[{"v":4}]}]}"#));
        assert!(bad(tree, r#"{"v":1,"kids":[{"kids":[]}]}"#));
        let either = r#"{"anyOf":[{"type":"object","properties":{"a":{"type":"integer"}},"required":["a"]},{"type":"string"}]}"#;
        assert!(full(either, r#""hi""#) && full(either, r#"{"a":1}"#));
        assert!(bad(either, "7"));
    }

    #[test]
    fn a_schema_that_cannot_be_followed_is_refused_with_a_reason() {
        let err = |s: &str| SchemaDoc::from_json(&Json::parse(s).unwrap()).err().unwrap();
        assert!(err(r#"{"type":"foo"}"#).contains("unknown type"));
        assert!(err(r#"{"allOf":[{}]}"#).contains("allOf"));
        assert!(err("false").contains("false"));
        assert!(err(r##"{"$ref":"#/$defs/missing"}"##).contains("does not resolve"));
        assert!(err(r#"{"type":"object","properties":{"a":{}},"required":["b"]}"#).contains("not declared"));
        assert!(err(r#"{"type":"array","minItems":3,"maxItems":1}"#).contains("minItems"));
        assert!(err(r#"{"enum":[]}"#).contains("enum"));
    }
}
