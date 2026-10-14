//! Character classes used by the pre-tokenizers, from the generated Unicode 15.1 range tables (tools/gen_unicode_tables.py).
//! `\p{L}` = letters (Lu Ll Lt Lm Lo), `\p{N}` = numbers (Nd Nl No), `\p{M}` = marks (Mn Mc Me), `\s` = the White_Space property.

use crate::unicode_punct::{MN, OTHER, PUNCT};
use crate::unicode_tables::{LETTER, MARK, NUMBER};

fn in_ranges(table: &[(u32, u32)], c: u32) -> bool {
    // first range whose end >= c
    let i = table.partition_point(|&(_, hi)| hi < c);
    i < table.len() && table[i].0 <= c
}

pub fn is_letter(c: char) -> bool {
    let u = c as u32;
    if u < 0x80 {
        return c.is_ascii_alphabetic();
    }
    in_ranges(&LETTER, u)
}

pub fn is_number(c: char) -> bool {
    let u = c as u32;
    if u < 0x80 {
        return c.is_ascii_digit();
    }
    in_ranges(&NUMBER, u)
}

pub fn is_mark(c: char) -> bool {
    let u = c as u32;
    u >= 0x300 && in_ranges(&MARK, u)
}

/// Unicode punctuation (P*): what BERT's basic tokenizer splits words at (it also counts the ASCII symbols as punctuation; see `wordpiece`).
pub fn is_punct(c: char) -> bool {
    let u = c as u32;
    if u < 0x80 {
        return matches!(u, 0x21..=0x23 | 0x25..=0x2A | 0x2C..=0x2F | 0x3A | 0x3B | 0x3F | 0x40 | 0x5B..=0x5D | 0x5F | 0x7B | 0x7D);
    }
    in_ranges(&PUNCT, u)
}

/// Nonspacing marks (Mn): the accents that BERT strips after NFD.
pub fn is_nonspacing_mark(c: char) -> bool {
    let u = c as u32;
    u >= 0x300 && in_ranges(&MN, u)
}

/// Controls, format characters, surrogates, private use and unassigned code points (C*).
pub fn is_other(c: char) -> bool {
    in_ranges(&OTHER, c as u32)
}

/// `\s` of the regex engines the tokenizers were built with (White_Space; NOT the C0 separators U+001C..U+001F that Python's `str.isspace` accepts).
pub fn is_space(c: char) -> bool {
    matches!(c as u32, 0x09..=0x0D | 0x20 | 0x85 | 0xA0 | 0x1680 | 0x2000..=0x200A | 0x2028 | 0x2029 | 0x202F | 0x205F | 0x3000)
}

pub fn is_newline(c: char) -> bool {
    c == '\r' || c == '\n'
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classes() {
        assert!(is_letter('a') && is_letter('é') && is_letter('中') && is_letter('ก'));
        assert!(!is_letter('1') && !is_letter(' ') && !is_letter('\u{0E31}')); // Thai vowel sign is a mark, not a letter
        assert!(is_mark('\u{0E31}') && is_mark('\u{0301}'));
        assert!(is_number('7') && is_number('٣') && is_number('½'));
        assert!(!is_number('a'));
        assert!(is_punct('!') && is_punct('-') && is_punct('\u{2014}') && is_punct('\u{00BF}') && is_punct('_') && !is_punct('$') && !is_punct('+') && !is_punct('a'));
        assert!(is_nonspacing_mark('\u{0301}') && !is_nonspacing_mark('\u{0903}') && !is_nonspacing_mark('e'));
        assert!(is_other('\u{0000}') && is_other('\u{200B}') && is_other('\u{007F}') && is_other('\u{0378}') && !is_other('a') && !is_other(' '));
        assert!(
            is_space(' ') && is_space('\u{3000}') && is_space('\u{00A0}') && !is_space('\u{001F}')
        );
    }
}
