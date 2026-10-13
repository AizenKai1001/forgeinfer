//! Constrained decoding: a byte-level automaton for JSON that the sampler consults. `response_format: {"type": "json_object"}` of the OpenAI API (and the same in Ollama and LM Studio) promises that the
//! answer parses; here the promise is kept by construction: before a token is drawn, the candidates whose bytes would break the grammar are removed, and the generation ends when the top-level object
//! is closed.
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
pub struct Constraint {
    stack: Vec<Frame>,
    st: St,
    ws: u8,
}

impl Constraint {
    /// Any JSON object.
    pub fn json_object() -> Constraint {
        Constraint {
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

#[cfg(test)]
mod tests {
    use super::*;

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
}
