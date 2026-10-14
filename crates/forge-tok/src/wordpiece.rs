//! BERT's WordPiece tokenizer, the one of the embedding models (nomic-embed-text, BERT, MiniLM): the basic tokenizer (drop control characters, white space to blanks, blanks around CJK characters, lower case,
//! strip accents, split at punctuation) followed by greedy longest-match-first word pieces with the `##` prefix for the pieces after the first. The result is what Hugging Face's `BertTokenizer` gives
//! (`do_lower_case` on, `strip_accents` unset, which means on with lower case), checked against it in tests/wordpiece.rs.

use crate::normalize::nfd;
use crate::unicode::{is_nonspacing_mark, is_other, is_punct};
use forge_core::{Error, Result};
use forge_gguf::Gguf;
use std::collections::HashMap;

/// Words longer than this many characters are one unknown token.
const MAX_WORD_CHARS: usize = 100;

pub struct WordPiece {
    tokens: Vec<String>,
    vocab: HashMap<String, u32>,
    pub unk: u32,
    pub cls: u32,
    pub sep: u32,
    pub pad: u32,
    /// lower case and strip accents (an uncased vocabulary)
    pub lowercase: bool,
}

/// BERT's token string for a token of llama.cpp's "phantom space" vocabulary (see `WordPiece::from_gguf`).
fn unphantom(t: &str) -> String {
    if let Some(rest) = t.strip_prefix('\u{2581}') {
        rest.to_string()
    } else if t.starts_with('[') && t.ends_with(']') {
        t.to_string()
    } else {
        format!("##{t}")
    }
}

fn is_cjk(c: char) -> bool {
    matches!(c as u32, 0x4E00..=0x9FFF | 0x3400..=0x4DBF | 0x20000..=0x2A6DF | 0x2A700..=0x2B73F | 0x2B740..=0x2B81F | 0x2B820..=0x2CEAF | 0xF900..=0xFAFF | 0x2F800..=0x2FA1F)
}

/// BERT counts every ASCII symbol as punctuation, not only the Unicode P* ones.
fn is_bert_punct(c: char) -> bool {
    matches!(c as u32, 33..=47 | 58..=64 | 91..=96 | 123..=126) || is_punct(c)
}

fn is_blank(c: char) -> bool {
    matches!(c, ' ' | '\t' | '\n' | '\r') || matches!(c as u32, 0xA0 | 0x1680 | 0x2000..=0x200A | 0x202F | 0x205F | 0x3000)
}

fn is_control(c: char) -> bool {
    !matches!(c, '\t' | '\n' | '\r') && is_other(c)
}

impl WordPiece {
    pub fn from_tokens(tokens: Vec<String>, unk: u32, cls: u32, sep: u32, pad: u32) -> WordPiece {
        let vocab: HashMap<String, u32> = tokens.iter().enumerate().map(|(i, t)| (t.clone(), i as u32)).collect();
        // an uncased vocabulary has no capitalised common words
        let lowercase = !(vocab.contains_key("The") || vocab.contains_key("A") || vocab.contains_key("He"));
        WordPiece { tokens, vocab, unk, cls, sep, pad, lowercase }
    }

    /// The tokenizer of a GGUF file whose `tokenizer.ggml.model` is `bert`.
    pub fn from_gguf(g: &Gguf) -> Result<WordPiece> {
        if g.get_str("tokenizer.ggml.model") != Some("bert") {
            return Err(Error::msg("the GGUF file has no BERT (WordPiece) tokenizer"));
        }
        let mut tokens: Vec<String> = g
            .get_str_array("tokenizer.ggml.tokens")
            .ok_or_else(|| Error::msg("tokenizer.ggml.tokens is missing"))?
            .into_iter()
            .map(|s| s.to_string())
            .collect();
        // llama.cpp's converter writes the vocabulary with "phantom spaces": a word-initial token gets the SentencePiece marker U+2581 in front, a continuation piece loses its `##`, the bracketed
        // special tokens stay; turn it back into BERT's own form
        if tokens.iter().any(|t| t.starts_with('\u{2581}')) {
            tokens = tokens.into_iter().map(|t| unphantom(&t)).collect();
        }
        let id = |key: &str, default: &str| -> u32 {
            g.get_u64(key).map(|v| v as u32).or_else(|| tokens.iter().position(|t| t == default).map(|p| p as u32)).unwrap_or(0)
        };
        let unk = id("tokenizer.ggml.unknown_token_id", "[UNK]");
        let cls = id("tokenizer.ggml.bos_token_id", "[CLS]");
        let sep = g.get_u64("tokenizer.ggml.seperator_token_id").or_else(|| g.get_u64("tokenizer.ggml.eos_token_id")).map(|v| v as u32).unwrap_or_else(|| id("", "[SEP]"));
        let pad = id("tokenizer.ggml.padding_token_id", "[PAD]");
        Ok(WordPiece::from_tokens(tokens, unk, cls, sep, pad))
    }

    pub fn vocab_size(&self) -> usize {
        self.tokens.len()
    }

    pub fn token(&self, id: u32) -> Option<&str> {
        self.tokens.get(id as usize).map(|s| s.as_str())
    }

    /// The words of the basic tokenizer: control characters dropped, blanks around CJK characters, lower case and stripped accents (uncased vocabularies), every punctuation character a word of its own.
    pub fn basic_words(&self, text: &str) -> Vec<String> {
        let mut spaced = String::with_capacity(text.len() + 8);
        for c in text.chars() {
            if c as u32 == 0 || c as u32 == 0xFFFD || is_control(c) {
                continue;
            }
            if is_blank(c) {
                spaced.push(' ');
            } else if is_cjk(c) {
                spaced.push(' ');
                spaced.push(c);
                spaced.push(' ');
            } else {
                spaced.push(c);
            }
        }
        let mut words = Vec::new();
        for raw in spaced.split(' ').filter(|w| !w.is_empty()) {
            let word: String = if self.lowercase {
                // Python's str.lower() and then NFD without the nonspacing marks
                let lower: String = raw.chars().flat_map(|c| c.to_lowercase()).collect();
                nfd(&lower).chars().filter(|&c| !is_nonspacing_mark(c)).collect()
            } else {
                raw.to_string()
            };
            let mut cur = String::new();
            for c in word.chars() {
                if is_bert_punct(c) {
                    if !cur.is_empty() {
                        words.push(std::mem::take(&mut cur));
                    }
                    words.push(c.to_string());
                } else {
                    cur.push(c);
                }
            }
            if !cur.is_empty() {
                words.push(cur);
            }
        }
        words
    }

    /// Greedy longest-match-first pieces of one word (`##` before every piece after the first); an unknown word is one `[UNK]`.
    fn pieces(&self, word: &str, out: &mut Vec<u32>) {
        let chars: Vec<char> = word.chars().collect();
        if chars.len() > MAX_WORD_CHARS {
            out.push(self.unk);
            return;
        }
        let mark = out.len();
        let mut start = 0;
        while start < chars.len() {
            let mut end = chars.len();
            let mut found = None;
            while start < end {
                let piece: String = chars[start..end].iter().collect();
                let key = if start > 0 { format!("##{piece}") } else { piece };
                if let Some(&id) = self.vocab.get(&key) {
                    found = Some(id);
                    break;
                }
                end -= 1;
            }
            match found {
                Some(id) => {
                    out.push(id);
                    start = end;
                }
                None => {
                    out.truncate(mark);
                    out.push(self.unk);
                    return;
                }
            }
        }
    }

    /// The token ids of a text, with `[CLS]` in front and `[SEP]` behind when `add_special`.
    pub fn encode(&self, text: &str, add_special: bool) -> Vec<u32> {
        let mut ids = Vec::new();
        if add_special {
            ids.push(self.cls);
        }
        for w in self.basic_words(text) {
            self.pieces(&w, &mut ids);
        }
        if add_special {
            ids.push(self.sep);
        }
        ids
    }

    /// Like `encode` with special tokens, cut to at most `max_len` tokens (the text is shortened, `[SEP]` stays last).
    pub fn encode_truncated(&self, text: &str, max_len: usize) -> Vec<u32> {
        let mut ids = self.encode(text, true);
        if ids.len() > max_len && max_len >= 2 {
            ids.truncate(max_len - 1);
            ids.push(self.sep);
        }
        ids
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny() -> WordPiece {
        let words = [
            "[PAD]", "[UNK]", "[CLS]", "[SEP]", "hello", "world", "!", ",", "un", "##aff", "##able", "cafe", "a", "##b", ".", "the", "$", "-", "\u{4e2d}", "\u{6587}", "it", "'", "s", "##s", "x",
        ];
        WordPiece::from_tokens(words.iter().map(|s| s.to_string()).collect(), 1, 2, 3, 0)
    }

    fn texts(wp: &WordPiece, ids: &[u32]) -> Vec<String> {
        ids.iter().map(|&i| wp.token(i).unwrap().to_string()).collect()
    }

    #[test]
    fn words_and_punctuation() {
        let wp = tiny();
        assert!(wp.lowercase);
        assert_eq!(texts(&wp, &wp.encode("Hello, World!", true)), ["[CLS]", "hello", ",", "world", "!", "[SEP]"]);
        assert_eq!(texts(&wp, &wp.encode("  hello \t\n world  ", false)), ["hello", "world"]);
        // every punctuation character is a word of its own, ASCII symbols too
        assert_eq!(wp.basic_words("a.b-c$d"), ["a", ".", "b", "-", "c", "$", "d"]);
        assert_eq!(texts(&wp, &wp.encode("it's", false)), ["it", "'", "s"]);
    }

    #[test]
    fn word_pieces() {
        let wp = tiny();
        assert_eq!(texts(&wp, &wp.encode("unaffable", false)), ["un", "##aff", "##able"]);
        // one unknown piece makes the whole word [UNK]
        assert_eq!(texts(&wp, &wp.encode("unaffablez", false)), ["[UNK]"]);
        assert_eq!(texts(&wp, &wp.encode("ab", false)), ["a", "##b"]);
        assert_eq!(texts(&wp, &wp.encode(&"x".repeat(101), false)), ["[UNK]"]);
        assert_eq!(wp.encode(&"x".repeat(100), false).len(), 1); // "xxxx..." is not in the vocabulary: [UNK] too, but through the pieces
    }

    #[test]
    fn accents_cjk_and_controls() {
        let wp = tiny();
        assert_eq!(texts(&wp, &wp.encode("Caf\u{e9}", false)), ["cafe"]);
        assert_eq!(texts(&wp, &wp.encode("Cafe\u{301}", false)), ["cafe"]);
        assert_eq!(texts(&wp, &wp.encode("\u{4e2d}\u{6587}hello", false)), ["\u{4e2d}", "\u{6587}", "hello"]);
        assert_eq!(texts(&wp, &wp.encode("hel\u{200b}lo\u{0}\u{7}", false)), ["hello"]);
        assert_eq!(texts(&wp, &wp.encode("hello\u{a0}world", false)), ["hello", "world"]);
    }

    #[test]
    fn the_phantom_space_vocabulary_of_a_gguf_file_is_turned_back() {
        assert_eq!(unphantom("\u{2581}the"), "the");
        assert_eq!(unphantom("ing"), "##ing");
        assert_eq!(unphantom("[CLS]"), "[CLS]");
        assert_eq!(unphantom("[unused12]"), "[unused12]");
        assert_eq!(unphantom("\u{2581}!"), "!");
        assert_eq!(unphantom("_"), "##_");
    }

    #[test]
    fn truncation_keeps_the_separator() {
        let wp = tiny();
        let ids = wp.encode_truncated("hello world hello world hello", 5);
        assert_eq!(ids.len(), 5);
        assert_eq!((ids[0], ids[4]), (2, 3));
        assert_eq!(wp.encode_truncated("hello", 5), [2, 4, 3]);
    }
}
