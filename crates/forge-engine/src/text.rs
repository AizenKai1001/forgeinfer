//! Streaming text helpers: bytes of generated tokens to UTF-8 text without ever splitting a character, and stop-string matching that holds back text which could still become a stop string.

/// Collects token bytes and releases the longest valid UTF-8 prefix; an invalid byte becomes U+FFFD, an incomplete tail waits for the next token.
#[derive(Default)]
pub struct Utf8Stream {
    pending: Vec<u8>,
}

impl Utf8Stream {
    pub fn push(&mut self, bytes: &[u8]) -> String {
        self.pending.extend_from_slice(bytes);
        let mut out = String::new();
        loop {
            match std::str::from_utf8(&self.pending) {
                Ok(s) => {
                    out.push_str(s);
                    self.pending.clear();
                    break;
                }
                Err(e) => {
                    let valid = e.valid_up_to();
                    out.push_str(std::str::from_utf8(&self.pending[..valid]).unwrap());
                    match e.error_len() {
                        Some(n) => {
                            out.push('\u{fffd}');
                            self.pending.drain(..valid + n);
                        }
                        None => {
                            self.pending.drain(..valid);
                            break; // an incomplete character: wait for more bytes
                        }
                    }
                }
            }
        }
        out
    }

    /// The end of the stream: an incomplete tail becomes U+FFFD.
    pub fn finish(&mut self) -> String {
        if self.pending.is_empty() {
            String::new()
        } else {
            self.pending.clear();
            "\u{fffd}".to_string()
        }
    }
}

/// Matches stop strings in a stream. `feed` returns (text safe to emit, stop found): after a stop string is found the text before it is returned and nothing after.
pub struct StopMatcher {
    stops: Vec<String>,
    held: String,
    pub stopped: bool,
}

impl StopMatcher {
    pub fn new(stops: Vec<String>) -> StopMatcher {
        StopMatcher {
            stops: stops.into_iter().filter(|s| !s.is_empty()).collect(),
            held: String::new(),
            stopped: false,
        }
    }

    pub fn feed(&mut self, text: &str) -> String {
        if self.stopped {
            return String::new();
        }
        if self.stops.is_empty() {
            return text.to_string();
        }
        self.held.push_str(text);
        // the earliest stop string wins
        let hit = self
            .stops
            .iter()
            .filter_map(|s| self.held.find(s.as_str()))
            .min();
        if let Some(i) = hit {
            self.stopped = true;
            let out = self.held[..i].to_string();
            self.held.clear();
            return out;
        }
        // hold back the longest suffix that is a proper prefix of a stop string
        let mut keep = 0;
        for s in &self.stops {
            let max = s.len().saturating_sub(1).min(self.held.len());
            for k in (1..=max).rev() {
                if self.held.is_char_boundary(self.held.len() - k)
                    && s.is_char_boundary(k)
                    && s.starts_with(&self.held[self.held.len() - k..])
                {
                    keep = keep.max(k);
                    break;
                }
            }
        }
        let cut = self.held.len() - keep;
        let out = self.held[..cut].to_string();
        self.held = self.held[cut..].to_string();
        out
    }

    /// The end of the stream without a stop: whatever was held back is text.
    pub fn flush(&mut self) -> String {
        std::mem::take(&mut self.held)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn utf8_split_across_tokens() {
        let mut u = Utf8Stream::default();
        let e = "é€😀".as_bytes();
        let mut out = String::new();
        for b in e {
            out.push_str(&u.push(&[*b]));
        }
        assert_eq!(out, "é€😀");
        assert_eq!(u.finish(), "");
        assert_eq!(u.push(&[0xE2, 0x82]), "");
        assert_eq!(u.finish(), "\u{fffd}");
        assert_eq!(
            Utf8Stream::default().push(&[b'a', 0xFF, b'b']),
            "a\u{fffd}b"
        );
    }

    #[test]
    fn stop_strings_across_chunks() {
        let mut m = StopMatcher::new(vec!["</end>".into(), "STOP".into()]);
        let mut out = String::new();
        for chunk in ["Hello ", "wor", "ld <", "/en", "d> tail"] {
            out.push_str(&m.feed(chunk));
        }
        assert!(m.stopped);
        assert_eq!(out, "Hello world ");
        let mut m = StopMatcher::new(vec!["STOP".into()]);
        let mut out = String::new();
        for chunk in ["ab", "ST", "OX", "end"] {
            out.push_str(&m.feed(chunk));
        }
        out.push_str(&m.flush());
        assert!(!m.stopped);
        assert_eq!(out, "abSTOXend");
    }
}
