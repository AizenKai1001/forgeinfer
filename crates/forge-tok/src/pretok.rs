//! Pre-tokenizers as plain character scanners (no regex engine). Each function returns the pieces as half-open ranges of char indices.
//! tools/ref_tokenizer.py is the executable specification; the regexes they implement:
//!   GPT-2 (SmolLM2, after a Digits splitter):  's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
//!   Qwen2 (Qwen3):  (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
//!   Qwen3.5: as Qwen2 with the letter class widened to \p{L}\p{M} (combining marks stay inside words).

use crate::unicode::{is_letter, is_mark, is_newline, is_number, is_space};

pub type Span = (usize, usize);

const CONTRACTIONS: [&[char]; 7] = [
    &['s'],
    &['t'],
    &['r', 'e'],
    &['v', 'e'],
    &['m'],
    &['l', 'l'],
    &['d'],
];

/// Length (in chars) of an English contraction starting at t[i], or 0.
fn contraction_len(t: &[char], i: usize, ignore_case: bool) -> usize {
    if t[i] != '\'' {
        return 0;
    }
    for suf in CONTRACTIONS {
        if i + 1 + suf.len() > t.len() {
            continue;
        }
        let ok = suf.iter().enumerate().all(|(k, &want)| {
            let c = t[i + 1 + k];
            if ignore_case {
                // case-insensitive as in (?i:...): ASCII letters, plus U+017F (long s) which folds to 's'
                c.to_ascii_lowercase() == want || (want == 's' && c == '\u{17F}')
            } else {
                c == want
            }
        });
        if ok {
            return 1 + suf.len();
        }
    }
    0
}

fn run(t: &[char], mut i: usize, pred: impl Fn(char) -> bool) -> usize {
    while i < t.len() && pred(t[i]) {
        i += 1;
    }
    i
}

/// `\s+(?!\S)`: the longest whitespace run from i that is not followed by a non-space (the last space of a run before a word is left for the word). Returns the end, or i if none.
fn ws_run_ending_before_nonspace(t: &[char], i: usize) -> usize {
    let j = run(t, i, is_space);
    if j == i {
        return i;
    }
    if j == t.len() {
        return j;
    }
    if j - 1 > i {
        j - 1
    } else {
        i
    }
}

pub fn split_gpt2(t: &[char]) -> Vec<Span> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < t.len() {
        let n = contraction_len(t, i, false);
        if n > 0 {
            out.push((i, i + n));
            i += n;
            continue;
        }
        let j = if t[i] == ' ' { i + 1 } else { i };
        if j < t.len() && is_letter(t[j]) {
            let k = run(t, j, is_letter);
            out.push((i, k));
            i = k;
            continue;
        }
        if j < t.len() && is_number(t[j]) {
            let k = run(t, j, is_number);
            out.push((i, k));
            i = k;
            continue;
        }
        if j < t.len() && !is_space(t[j]) && !is_letter(t[j]) && !is_number(t[j]) {
            let k = run(t, j, |c| !is_space(c) && !is_letter(c) && !is_number(c));
            out.push((i, k));
            i = k;
            continue;
        }
        let k = ws_run_ending_before_nonspace(t, i);
        if k > i {
            out.push((i, k));
            i = k;
            continue;
        }
        let k = run(t, i, is_space).max(i + 1);
        out.push((i, k));
        i = k;
    }
    out
}

pub fn split_qwen2(t: &[char], marks: bool) -> Vec<Span> {
    split_qwen2_digits(t, marks, 1)
}

/// The Qwen2 pattern with `\p{N}{1,digits}` for the numbers (`digits` 1: Qwen; 3: Llama 3 and its relatives).
pub fn split_qwen2_digits(t: &[char], marks: bool, digits: usize) -> Vec<Span> {
    let lettish = |c: char| is_letter(c) || (marks && is_mark(c));
    let mut out = Vec::new();
    let mut i = 0;
    while i < t.len() {
        let n = contraction_len(t, i, true);
        if n > 0 {
            out.push((i, i + n));
            i += n;
            continue;
        }
        let c = t[i];
        // [^\r\n\p{L}\p{N}]? followed by letters
        let j = if !is_newline(c) && !is_letter(c) && !is_number(c) {
            i + 1
        } else {
            i
        };
        if j < t.len() && lettish(t[j]) {
            let k = run(t, j, lettish);
            out.push((i, k));
            i = k;
            continue;
        }
        // the optional prefix may stay empty (regex backtracking): in the Qwen3.5 class a combining mark is a "letter" itself, so a mark with nothing letter-like after it still starts a piece
        if lettish(c) {
            let k = run(t, i, lettish);
            out.push((i, k));
            i = k;
            continue;
        }
        if is_number(c) {
            let mut k = i + 1;
            while k < t.len() && k - i < digits && is_number(t[k]) {
                k += 1;
            }
            out.push((i, k));
            i = k;
            continue;
        }
        // ` ?[^\s\p{L}\p{N}]+[\r\n]*`
        let j = if c == ' ' { i + 1 } else { i };
        if j < t.len() && !is_space(t[j]) && !lettish(t[j]) && !is_number(t[j]) {
            let k = run(t, j, |x| !is_space(x) && !lettish(x) && !is_number(x));
            let k = run(t, k, is_newline);
            out.push((i, k));
            i = k;
            continue;
        }
        // `\s*[\r\n]+`: whitespace up to and including the last newline of the run
        let w = run(t, i, is_space);
        if let Some(p) = (i..w).rev().find(|&p| is_newline(t[p])) {
            out.push((i, p + 1));
            i = p + 1;
            continue;
        }
        let k = ws_run_ending_before_nonspace(t, i);
        if k > i {
            out.push((i, k));
            i = k;
            continue;
        }
        // every scanner step must consume at least one character, whatever the input
        let w = w.max(i + 1);
        out.push((i, w));
        i = w;
    }
    out
}

/// The Digits pre-tokenizer with individual_digits: every number character is a piece of its own, the text between them stays together.
pub fn split_digits_individual(t: &[char]) -> Vec<Span> {
    let mut out = Vec::new();
    let mut start = 0;
    for (i, &c) in t.iter().enumerate() {
        if is_number(c) {
            if start < i {
                out.push((start, i));
            }
            out.push((i, i + 1));
            start = i + 1;
        }
    }
    if start < t.len() {
        out.push((start, t.len()));
    }
    out
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PreTokenizer {
    /// The GPT-2 regex alone (GPT-2, Llama-2-style byte-level BPE without a digit splitter).
    Gpt2,
    /// Digits (individual) followed by the GPT-2 regex: SmolLM2, GPT-2-style models.
    Gpt2Digits,
    /// The Qwen2 / Qwen3 pattern.
    Qwen2,
    /// The Qwen3.5 pattern (letters and combining marks).
    Qwen35,
    /// The Llama 3 pattern: the Qwen2 one with numbers in groups of up to three digits.
    Llama3,
}

impl PreTokenizer {
    pub fn split(self, t: &[char]) -> Vec<Span> {
        match self {
            PreTokenizer::Gpt2 => split_gpt2(t),
            PreTokenizer::Qwen2 => split_qwen2(t, false),
            PreTokenizer::Qwen35 => split_qwen2(t, true),
            PreTokenizer::Llama3 => split_qwen2_digits(t, false, 3),
            PreTokenizer::Gpt2Digits => {
                let mut out = Vec::new();
                for (a, b) in split_digits_individual(t) {
                    for (x, y) in split_gpt2(&t[a..b]) {
                        out.push((a + x, a + y));
                    }
                }
                out
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pieces(f: impl Fn(&[char]) -> Vec<Span>, s: &str) -> Vec<String> {
        let t: Vec<char> = s.chars().collect();
        f(&t)
            .into_iter()
            .map(|(a, b)| t[a..b].iter().collect())
            .collect()
    }

    #[test]
    fn gpt2_basics() {
        assert_eq!(
            pieces(split_gpt2, "Hello world! It's 2024."),
            vec!["Hello", " world", "!", " It", "'s", " 2024", "."]
        );
        assert_eq!(pieces(split_gpt2, "a  b"), vec!["a", " ", " b"]);
        assert_eq!(
            pieces(split_gpt2, "end   \n\nx"),
            vec!["end", "   \n", "\n", "x"]
        );
    }

    #[test]
    fn qwen2_basics() {
        assert_eq!(
            pieces(|t| split_qwen2(t, false), "Don't stop 123!\n\nok"),
            vec!["Don", "'t", " stop", " ", "1", "2", "3", "!\n\n", "ok"]
        );
        assert_eq!(
            pieces(|t| split_qwen2(t, false), "x  \ny"),
            vec!["x", "  \n", "y"]
        );
    }

    #[test]
    fn a_lone_combining_mark_makes_a_piece_of_its_own() {
        let q35 = |t: &[char]| split_qwen2(t, true);
        assert_eq!(pieces(q35, "1\u{fe0f}"), vec!["1", "\u{fe0f}"]);
        assert_eq!(pieces(q35, "\u{301}"), vec!["\u{301}"]);
        assert_eq!(
            pieces(q35, "a.\u{301}.\n\u{301}"),
            vec!["a", ".\u{301}", ".\n", "\u{301}"]
        );
        assert_eq!(
            pieces(|t| split_qwen2(t, false), "1\u{301}"),
            vec!["1", "\u{301}"]
        );
    }

    #[test]
    fn llama3_groups_digits_by_three() {
        assert_eq!(
            pieces(|t| split_qwen2_digits(t, false, 3), "x 1234567 y"),
            vec!["x", " ", "123", "456", "7", " y"]
        );
    }

    #[test]
    fn every_variant_tiles_any_text_without_empty_pieces() {
        let atoms: Vec<&str> = vec![
            "a", "Z", "é", "Ω", "你", "1", "٣", "²", " ", "  ", "\t", "\n", "\r\n", ".", "!", "'",
            "'s", "'LL", "\u{301}", "\u{fe0f}", "\u{200d}", "\u{20e3}", "😀", "_", "-", "\u{a0}",
            "\u{2028}", "ß", "ǅ", "\u{0}",
        ];
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        for _ in 0..20000 {
            let n = (next() % 14) as usize;
            let mut s = String::new();
            for _ in 0..n {
                s.push_str(atoms[(next() % atoms.len() as u64) as usize]);
            }
            let t: Vec<char> = s.chars().collect();
            for pre in [
                PreTokenizer::Gpt2,
                PreTokenizer::Gpt2Digits,
                PreTokenizer::Qwen2,
                PreTokenizer::Qwen35,
                PreTokenizer::Llama3,
            ] {
                let mut at = 0;
                for (a, b) in pre.split(&t) {
                    assert_eq!(a, at, "{pre:?} {s:?}");
                    assert!(b > a, "{pre:?} {s:?}: empty piece");
                    at = b;
                }
                assert_eq!(at, t.len(), "{pre:?} {s:?}");
            }
        }
    }

    #[test]
    fn digits() {
        assert_eq!(
            pieces(split_digits_individual, "ab12c"),
            vec!["ab", "1", "2", "c"]
        );
    }
}
