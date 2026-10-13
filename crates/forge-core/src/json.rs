//! A small, strict JSON parser and writer (RFC 8259) with the two extensions that Python's `json` module emits and model files contain: `NaN`, `Infinity` and `-Infinity` literals.
//! Objects keep their key order (a `Vec` of pairs): tokenizer.json files rely on the order of `vocab` and `added_tokens`, and objects are small except `vocab`, which callers iterate.

use crate::error::{Error, Result};
use std::fmt;
use std::ops::Index;

#[derive(Debug, Clone, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    /// A number written without fraction or exponent that fits in i64.
    Int(i64),
    Float(f64),
    Str(String),
    Arr(Vec<Json>),
    Obj(Vec<(String, Json)>),
}

static NULL: Json = Json::Null;

const MAX_DEPTH: usize = 512;

impl Json {
    pub fn parse(text: &str) -> Result<Json> {
        Json::parse_bytes(text.as_bytes())
    }

    pub fn parse_bytes(b: &[u8]) -> Result<Json> {
        // tolerate a UTF-8 byte order mark
        let b = if b.starts_with(&[0xEF, 0xBB, 0xBF]) {
            &b[3..]
        } else {
            b
        };
        let mut p = Parser { b, i: 0 };
        p.ws();
        let v = p.value(0)?;
        p.ws();
        if p.i != b.len() {
            return Err(p.fail("trailing characters after the JSON value"));
        }
        Ok(v)
    }

    pub fn from_file(path: impl AsRef<std::path::Path>) -> Result<Json> {
        let bytes = std::fs::read(path.as_ref())
            .map_err(|e| Error::msg(format!("{}: {e}", path.as_ref().display())))?;
        Json::parse_bytes(&bytes)
    }

    /// Object member by key (the first one if a key repeats).
    pub fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Obj(m) => m.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    pub fn at(&self, i: usize) -> Option<&Json> {
        match self {
            Json::Arr(a) => a.get(i),
            _ => None,
        }
    }

    /// Follow a chain of object keys: `j.path(&["model", "vocab"])`.
    pub fn path(&self, keys: &[&str]) -> Option<&Json> {
        let mut cur = self;
        for k in keys {
            cur = cur.get(k)?;
        }
        Some(cur)
    }

    pub fn is_null(&self) -> bool {
        matches!(self, Json::Null)
    }
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Json::Str(s) => Some(s),
            _ => None,
        }
    }
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Json::Bool(b) => Some(*b),
            _ => None,
        }
    }
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Json::Int(i) => Some(*i),
            Json::Float(f) if f.fract() == 0.0 && f.abs() < 9.0e18 => Some(*f as i64),
            _ => None,
        }
    }
    pub fn as_u64(&self) -> Option<u64> {
        self.as_i64().and_then(|i| u64::try_from(i).ok())
    }
    pub fn as_usize(&self) -> Option<usize> {
        self.as_u64().map(|u| u as usize)
    }
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Json::Int(i) => Some(*i as f64),
            Json::Float(f) => Some(*f),
            _ => None,
        }
    }
    pub fn as_array(&self) -> Option<&[Json]> {
        match self {
            Json::Arr(a) => Some(a),
            _ => None,
        }
    }
    pub fn as_object(&self) -> Option<&[(String, Json)]> {
        match self {
            Json::Obj(m) => Some(m),
            _ => None,
        }
    }

    /// Compact serialisation. Non-finite floats become `null` (JSON has no representation).
    pub fn write(&self, out: &mut String) {
        match self {
            Json::Null => out.push_str("null"),
            Json::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
            Json::Int(i) => out.push_str(&i.to_string()),
            Json::Float(f) => {
                if f.is_finite() {
                    let s = f.to_string();
                    out.push_str(&s);
                    if !s.contains('.') && !s.contains('e') && !s.contains('E') {
                        out.push_str(".0");
                    }
                } else {
                    out.push_str("null");
                }
            }
            Json::Str(s) => write_str(s, out),
            Json::Arr(a) => {
                out.push('[');
                for (i, v) in a.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    v.write(out);
                }
                out.push(']');
            }
            Json::Obj(m) => {
                out.push('{');
                for (i, (k, v)) in m.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    write_str(k, out);
                    out.push(':');
                    v.write(out);
                }
                out.push('}');
            }
        }
    }
}

pub fn write_str(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0C}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
}

impl fmt::Display for Json {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut s = String::new();
        self.write(&mut s);
        f.write_str(&s)
    }
}

/// `j["a"]["b"]` never panics: a missing key or a wrong type gives `Json::Null`.
impl Index<&str> for Json {
    type Output = Json;
    fn index(&self, key: &str) -> &Json {
        self.get(key).unwrap_or(&NULL)
    }
}

impl Index<usize> for Json {
    type Output = Json;
    fn index(&self, i: usize) -> &Json {
        self.at(i).unwrap_or(&NULL)
    }
}

struct Parser<'a> {
    b: &'a [u8],
    i: usize,
}

impl<'a> Parser<'a> {
    fn fail(&self, what: &str) -> Error {
        // line and column for the error message
        let upto = &self.b[..self.i.min(self.b.len())];
        let line = upto.iter().filter(|&&c| c == b'\n').count() + 1;
        let col = upto.iter().rev().take_while(|&&c| c != b'\n').count() + 1;
        Error::Parse(format!("JSON: {what} at line {line}, column {col}"))
    }

    fn ws(&mut self) {
        while self.i < self.b.len() && matches!(self.b[self.i], b' ' | b'\t' | b'\n' | b'\r') {
            self.i += 1;
        }
    }

    fn eat(&mut self, lit: &[u8]) -> bool {
        if self.b[self.i..].starts_with(lit) {
            self.i += lit.len();
            true
        } else {
            false
        }
    }

    fn value(&mut self, depth: usize) -> Result<Json> {
        if depth > MAX_DEPTH {
            return Err(self.fail("nesting too deep"));
        }
        match self.b.get(self.i) {
            None => Err(self.fail("unexpected end of input")),
            Some(b'{') => self.object(depth),
            Some(b'[') => self.array(depth),
            Some(b'"') => Ok(Json::Str(self.string()?)),
            Some(b't') if self.eat(b"true") => Ok(Json::Bool(true)),
            Some(b'f') if self.eat(b"false") => Ok(Json::Bool(false)),
            Some(b'n') if self.eat(b"null") => Ok(Json::Null),
            Some(b'N') if self.eat(b"NaN") => Ok(Json::Float(f64::NAN)),
            Some(b'I') if self.eat(b"Infinity") => Ok(Json::Float(f64::INFINITY)),
            Some(b'-') if self.b[self.i..].starts_with(b"-Infinity") => {
                self.i += 9;
                Ok(Json::Float(f64::NEG_INFINITY))
            }
            Some(b'-') | Some(b'0'..=b'9') => self.number(),
            Some(_) => Err(self.fail("unexpected character")),
        }
    }

    fn object(&mut self, depth: usize) -> Result<Json> {
        self.i += 1; // {
        let mut members = Vec::new();
        self.ws();
        if self.b.get(self.i) == Some(&b'}') {
            self.i += 1;
            return Ok(Json::Obj(members));
        }
        loop {
            self.ws();
            if self.b.get(self.i) != Some(&b'"') {
                return Err(self.fail("expected a string key"));
            }
            let k = self.string()?;
            self.ws();
            if self.b.get(self.i) != Some(&b':') {
                return Err(self.fail("expected ':'"));
            }
            self.i += 1;
            self.ws();
            let v = self.value(depth + 1)?;
            members.push((k, v));
            self.ws();
            match self.b.get(self.i) {
                Some(b',') => self.i += 1,
                Some(b'}') => {
                    self.i += 1;
                    return Ok(Json::Obj(members));
                }
                _ => return Err(self.fail("expected ',' or '}'")),
            }
        }
    }

    fn array(&mut self, depth: usize) -> Result<Json> {
        self.i += 1; // [
        let mut items = Vec::new();
        self.ws();
        if self.b.get(self.i) == Some(&b']') {
            self.i += 1;
            return Ok(Json::Arr(items));
        }
        loop {
            self.ws();
            items.push(self.value(depth + 1)?);
            self.ws();
            match self.b.get(self.i) {
                Some(b',') => self.i += 1,
                Some(b']') => {
                    self.i += 1;
                    return Ok(Json::Arr(items));
                }
                _ => return Err(self.fail("expected ',' or ']'")),
            }
        }
    }

    fn hex4(&mut self) -> Result<u32> {
        if self.i + 4 > self.b.len() {
            return Err(self.fail("truncated \\u escape"));
        }
        let mut v = 0u32;
        for k in 0..4 {
            let d = (self.b[self.i + k] as char)
                .to_digit(16)
                .ok_or_else(|| self.fail("bad hex digit in \\u escape"))?;
            v = v * 16 + d;
        }
        self.i += 4;
        Ok(v)
    }

    fn string(&mut self) -> Result<String> {
        self.i += 1; // opening quote
        let mut out = String::new();
        loop {
            // copy the run of ordinary bytes in one go
            let start = self.i;
            while self.i < self.b.len()
                && self.b[self.i] != b'"'
                && self.b[self.i] != b'\\'
                && self.b[self.i] >= 0x20
            {
                self.i += 1;
            }
            if self.i > start {
                match std::str::from_utf8(&self.b[start..self.i]) {
                    Ok(s) => out.push_str(s),
                    Err(_) => return Err(self.fail("invalid UTF-8 in string")),
                }
            }
            match self.b.get(self.i) {
                None => return Err(self.fail("unterminated string")),
                Some(b'"') => {
                    self.i += 1;
                    return Ok(out);
                }
                Some(b'\\') => {
                    self.i += 1;
                    let c = *self
                        .b
                        .get(self.i)
                        .ok_or_else(|| self.fail("truncated escape"))?;
                    self.i += 1;
                    match c {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\u{08}'),
                        b'f' => out.push('\u{0C}'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => {
                            let hi = self.hex4()?;
                            if (0xD800..0xDC00).contains(&hi) {
                                // a high surrogate must be followed by \uDC00..\uDFFF
                                if self.b[self.i..].starts_with(b"\\u") {
                                    self.i += 2;
                                    let lo = self.hex4()?;
                                    if (0xDC00..0xE000).contains(&lo) {
                                        let cp = 0x10000 + ((hi - 0xD800) << 10) + (lo - 0xDC00);
                                        out.push(char::from_u32(cp).unwrap_or('\u{FFFD}'));
                                    } else {
                                        out.push('\u{FFFD}');
                                        out.push(char::from_u32(lo).unwrap_or('\u{FFFD}'));
                                    }
                                } else {
                                    out.push('\u{FFFD}');
                                }
                            } else {
                                out.push(char::from_u32(hi).unwrap_or('\u{FFFD}'));
                                // a lone low surrogate becomes U+FFFD
                            }
                        }
                        _ => return Err(self.fail("invalid escape")),
                    }
                }
                Some(_) => return Err(self.fail("control character in string")),
            }
        }
    }

    fn number(&mut self) -> Result<Json> {
        let start = self.i;
        if self.b[self.i] == b'-' {
            self.i += 1;
        }
        let digits_from = self.i;
        while self.i < self.b.len() && self.b[self.i].is_ascii_digit() {
            self.i += 1;
        }
        if self.i == digits_from {
            return Err(self.fail("expected digits"));
        }
        let mut is_float = false;
        if self.b.get(self.i) == Some(&b'.') {
            is_float = true;
            self.i += 1;
            let f0 = self.i;
            while self.i < self.b.len() && self.b[self.i].is_ascii_digit() {
                self.i += 1;
            }
            if self.i == f0 {
                return Err(self.fail("expected digits after '.'"));
            }
        }
        if matches!(self.b.get(self.i), Some(b'e') | Some(b'E')) {
            is_float = true;
            self.i += 1;
            if matches!(self.b.get(self.i), Some(b'+') | Some(b'-')) {
                self.i += 1;
            }
            let e0 = self.i;
            while self.i < self.b.len() && self.b[self.i].is_ascii_digit() {
                self.i += 1;
            }
            if self.i == e0 {
                return Err(self.fail("expected exponent digits"));
            }
        }
        let text =
            std::str::from_utf8(&self.b[start..self.i]).map_err(|_| self.fail("bad number"))?;
        if !is_float {
            if let Ok(v) = text.parse::<i64>() {
                return Ok(Json::Int(v));
            }
        }
        text.parse::<f64>()
            .map(Json::Float)
            .map_err(|_| self.fail("bad number"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalars_and_containers() {
        let j = Json::parse(r#" {"a": [1, -2, 3.5, 1e3, true, null], "b": {"c": "x\ny\u00e9\ud83d\ude00"}, "n": NaN} "#).unwrap();
        assert_eq!(j["a"][0].as_i64(), Some(1));
        assert_eq!(j["a"][1].as_i64(), Some(-2));
        assert_eq!(j["a"][2].as_f64(), Some(3.5));
        assert_eq!(j["a"][3].as_f64(), Some(1000.0));
        assert_eq!(j["a"][4].as_bool(), Some(true));
        assert!(j["a"][5].is_null());
        assert_eq!(j["b"]["c"].as_str(), Some("x\nyé😀"));
        assert!(j["n"].as_f64().unwrap().is_nan());
        assert!(j["missing"]["deeper"].is_null());
        assert_eq!(
            j.path(&["b", "c"]).and_then(|v| v.as_str()),
            Some("x\nyé😀")
        );
    }

    #[test]
    fn round_trip() {
        let src = r#"{"k":[1,2.5,"a\"b\\c\u0001",{"z":null}],"e":[],"o":{}}"#;
        let j = Json::parse(src).unwrap();
        let out = j.to_string();
        assert_eq!(Json::parse(&out).unwrap(), j);
        // the source is already in the canonical escape form
        assert_eq!(out, src);
    }

    #[test]
    fn errors() {
        for bad in [
            "",
            "{",
            "[1,]",
            "{\"a\" 1}",
            "tru",
            "\"abc",
            "01x",
            "[1 2]",
            "{\"a\":1}x",
            "-",
            "1.",
            "\"\\q\"",
        ] {
            assert!(Json::parse(bad).is_err(), "should reject {bad:?}");
        }
    }

    #[test]
    fn big_integers_fall_back_to_float() {
        let j = Json::parse("[9223372036854775807, 9223372036854775808]").unwrap();
        assert_eq!(j[0].as_i64(), Some(i64::MAX));
        assert!(matches!(j[1], Json::Float(_)));
    }
}
