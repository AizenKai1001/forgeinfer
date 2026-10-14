//! Unicode NFC (canonical decomposition, canonical ordering, canonical composition) from the generated tables (tools/gen_unicode_norm.py).
//! Qwen's tokenizer.json declares an NFC normalizer. Conformance is tested against Unicode's NormalizationTest.txt (tests/golden).

use crate::unicode_norm_tables::{CCC, COMPOSE, DECOMP_DATA, DECOMP_INDEX};

const S_BASE: u32 = 0xAC00;
const L_BASE: u32 = 0x1100;
const V_BASE: u32 = 0x1161;
const T_BASE: u32 = 0x11A7;
const L_COUNT: u32 = 19;
const V_COUNT: u32 = 21;
const T_COUNT: u32 = 28;
const N_COUNT: u32 = V_COUNT * T_COUNT;
const S_COUNT: u32 = L_COUNT * N_COUNT;

/// Canonical combining class of a code point.
pub fn ccc(c: u32) -> u8 {
    if c < 0x300 {
        return 0;
    }
    let i = CCC.partition_point(|&(_, hi, _)| hi < c);
    if i < CCC.len() && CCC[i].0 <= c {
        CCC[i].2
    } else {
        0
    }
}

fn decompose_into(c: u32, out: &mut Vec<u32>) {
    if c < 0xC0 {
        out.push(c);
        return;
    }
    if (S_BASE..S_BASE + S_COUNT).contains(&c) {
        let s = c - S_BASE;
        out.push(L_BASE + s / N_COUNT);
        out.push(V_BASE + (s % N_COUNT) / T_COUNT);
        let t = s % T_COUNT;
        if t != 0 {
            out.push(T_BASE + t);
        }
        return;
    }
    match DECOMP_INDEX.binary_search_by_key(&c, |&(cp, _, _)| cp) {
        Ok(i) => {
            let (_, off, len) = DECOMP_INDEX[i];
            out.extend_from_slice(&DECOMP_DATA[off as usize..off as usize + len as usize]);
        }
        Err(_) => out.push(c),
    }
}

fn compose_pair(a: u32, b: u32) -> Option<u32> {
    // Hangul: L + V -> LV, LV + T -> LVT
    if (L_BASE..L_BASE + L_COUNT).contains(&a) && (V_BASE..V_BASE + V_COUNT).contains(&b) {
        return Some(S_BASE + ((a - L_BASE) * V_COUNT + (b - V_BASE)) * T_COUNT);
    }
    if (S_BASE..S_BASE + S_COUNT).contains(&a)
        && (a - S_BASE).is_multiple_of(T_COUNT)
        && (T_BASE + 1..T_BASE + T_COUNT).contains(&b)
    {
        return Some(a + (b - T_BASE));
    }
    COMPOSE
        .binary_search_by(|&(x, y, _)| (x, y).cmp(&(a, b)))
        .ok()
        .map(|i| COMPOSE[i].2)
}

/// True when the text is certainly already in NFC (ASCII only), so callers can skip the work.
fn is_ascii(s: &str) -> bool {
    s.is_ascii()
}

/// Canonical decomposition with canonical ordering (NFD), for BERT's accent stripping.
pub fn nfd(s: &str) -> String {
    if s.is_ascii() {
        return s.to_string();
    }
    let mut d: Vec<u32> = Vec::with_capacity(s.len());
    for ch in s.chars() {
        decompose_into(ch as u32, &mut d);
    }
    let mut i = 0;
    while i < d.len() {
        if ccc(d[i]) == 0 {
            i += 1;
            continue;
        }
        let mut j = i;
        while j < d.len() && ccc(d[j]) != 0 {
            j += 1;
        }
        d[i..j].sort_by_key(|&c| ccc(c));
        i = j;
    }
    d.into_iter().map(|c| char::from_u32(c).unwrap_or('\u{FFFD}')).collect()
}

pub fn nfc(s: &str) -> String {
    if is_ascii(s) {
        return s.to_string();
    }
    // 1. canonical decomposition
    let mut d: Vec<u32> = Vec::with_capacity(s.len());
    for ch in s.chars() {
        decompose_into(ch as u32, &mut d);
    }
    // 2. canonical ordering: stable sort of every run of non-starters by combining class
    let mut i = 0;
    while i < d.len() {
        if ccc(d[i]) == 0 {
            i += 1;
            continue;
        }
        let mut j = i;
        while j < d.len() && ccc(d[j]) != 0 {
            j += 1;
        }
        d[i..j].sort_by_key(|&c| ccc(c)); // sort_by_key is stable
        i = j;
    }
    // 3. canonical composition
    let mut out: Vec<u32> = Vec::with_capacity(d.len());
    let mut starter: Option<usize> = None;
    let mut last_cc: i32 = 0;
    for &c in &d {
        let cc = ccc(c) as i32;
        if let Some(si) = starter {
            let adjacent = out.len() - 1 == si;
            if adjacent || last_cc < cc {
                if let Some(p) = compose_pair(out[si], c) {
                    out[si] = p;
                    continue;
                }
            }
        }
        if cc == 0 {
            starter = Some(out.len());
            last_cc = 0;
        } else {
            last_cc = cc;
        }
        out.push(c);
    }
    out.into_iter()
        .map(|c| char::from_u32(c).unwrap_or('\u{FFFD}'))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_cases() {
        assert_eq!(nfc("e\u{0301}"), "\u{00E9}");
        assert_eq!(nfc("\u{00E9}"), "\u{00E9}");
        assert_eq!(nfc("a\u{0323}\u{0307}"), "\u{1EA1}\u{0307}"); // dot below composes first, dot above stays (class 230 after 220)
        assert_eq!(nfc("a\u{0307}\u{0323}"), "\u{1EA1}\u{0307}"); // canonical reordering
        assert_eq!(nfc("\u{1100}\u{1161}\u{11A8}"), "\u{AC01}"); // Hangul L V T
        assert_eq!(nfc("\u{AC01}"), "\u{AC01}");
        assert_eq!(nfc("\u{212B}"), "\u{00C5}"); // ANGSTROM SIGN -> A with ring
        assert_eq!(nfc("hello"), "hello");
        assert_eq!(nfd("\u{00E9}"), "e\u{0301}");
        assert_eq!(nfd("\u{1EA1}\u{0307}"), "a\u{0323}\u{0307}");
        assert_eq!(nfd("\u{AC01}"), "\u{1100}\u{1161}\u{11A8}");
        assert_eq!(nfd("hello"), "hello");
    }
}
