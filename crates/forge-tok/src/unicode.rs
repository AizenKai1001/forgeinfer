//! Character classes used by the pre-tokenizers, from the generated Unicode 15.1 range tables (tools/gen_unicode_tables.py).
//! `\p{L}` = letters (Lu Ll Lt Lm Lo), `\p{N}` = numbers (Nd Nl No), `\p{M}` = marks (Mn Mc Me), `\s` = the White_Space property.

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
        assert!(
            is_space(' ') && is_space('\u{3000}') && is_space('\u{00A0}') && !is_space('\u{001F}')
        );
    }
}
