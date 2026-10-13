//! The fast encoder: a byte-level scanner for the GPT-2 pre-tokenizer (with or without the digit splitter), a cache from pre-tokenized words to their token ids, and merge buffers that live from word to word.
//!
//! What it takes from the study of gigatoken (Marcel Roed, MIT; its README, notes and source were read for ideas, no code was taken): no regex engine, a table of character classes indexed by the first byte,
//! letters scanned eight bytes at a time, one pass without building intermediate vectors, and above all a cache of words (text is a long tail of a few thousand frequent words, so most merge runs are repeats):
//! words of up to 15 bytes are packed into two 64-bit key words, the token ids sit in the same 32-byte slot (about 90 % of the words are one token, 98 % at most two), and two slots share one 64-byte line,
//! so a lookup is one memory access. The output is identical to `Tokenizer::encode_reference` (differential tests below, on random text with every kind of character the pre-tokenizer treats specially,
//! and on the Hugging Face goldens in tests/golden.rs). Tokenizers with another pre-tokenizer or with NFC take the reference path.

use crate::tokenizer::{BpeScratch, Tokenizer};
use crate::unicode::{is_letter, is_number, is_space};
use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

// ---- hashing ------------------------------------------------------------------------------------------------------------------------------------------------------------------

/// A multiplicative hasher (the idea of rustc's FxHash) with a final mix, so that the low bits can index a table. The std hasher (SipHash) costs more than the work it guards here.
#[derive(Default, Clone, Copy)]
pub struct FxHasher {
    h: u64,
}

const FX_K: u64 = 0x517c_c1b7_2722_0a95;

impl FxHasher {
    #[inline(always)]
    fn add(&mut self, w: u64) {
        self.h = (self.h.rotate_left(5) ^ w).wrapping_mul(FX_K);
    }
}

impl Hasher for FxHasher {
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        let mut b = bytes;
        while b.len() >= 8 {
            self.add(u64::from_le_bytes([
                b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
            ]));
            b = &b[8..];
        }
        if b.len() >= 4 {
            self.add(u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as u64);
            b = &b[4..];
        }
        for &x in b {
            self.add(x as u64);
        }
    }

    #[inline]
    fn write_u32(&mut self, i: u32) {
        self.add(i as u64);
    }

    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.add(i);
    }

    #[inline]
    fn finish(&self) -> u64 {
        let mut x = self.h;
        x ^= x >> 32;
        x = x.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        x ^ (x >> 29)
    }
}

pub type FxBuild = BuildHasherDefault<FxHasher>;

/// Position of the first `needle` at or after `from`, eight bytes per step (the zero-byte test of the bit tricks collection: exact for the lowest match).
pub fn find_byte(hay: &[u8], needle: u8, from: usize) -> Option<usize> {
    const LO: u64 = 0x0101_0101_0101_0101;
    const HI: u64 = 0x8080_8080_8080_8080;
    let pat = LO.wrapping_mul(needle as u64);
    let n = hay.len();
    let mut i = from;
    while i + 8 <= n {
        let x = u64::from_le_bytes(hay[i..i + 8].try_into().unwrap()) ^ pat;
        let z = x.wrapping_sub(LO) & !x & HI;
        if z != 0 {
            return Some(i + (z.trailing_zeros() >> 3) as usize);
        }
        i += 8;
    }
    while i < n {
        if hay[i] == needle {
            return Some(i);
        }
        i += 1;
    }
    None
}

// ---- the word cache -----------------------------------------------------------------------------------------------------------------------------------------------------------

/// Words of up to this many bytes live in the main table (two 64-bit key words, the ids inside the slot); longer ones, up to `MAX_LONG`, in a map; longer still are not cached.
const SHORT: usize = 15;
const MAX_LONG: usize = 48;
/// The tables start over when they hold this many words (the main one needs 64 MB at that size).
const MAX_SHORT_ENTRIES: usize = 1 << 20;
const MAX_LONG_ENTRIES: usize = 1 << 17;
/// Token ids that fit in a slot; a word with more is stored in the spill arena.
const INLINE: usize = 3;

/// 32 bytes: the word packed into a 128-bit key (the bytes of the word, zero padded, and its length in the top byte; `hi` = 0 marks an empty slot), three token ids, and `meta` = number of ids
/// (bits 4-7; 15 = the ids are in the spill arena: ids[0] = offset, ids[1] = count).
#[derive(Clone, Copy)]
#[repr(C)]
struct Slot {
    lo: u64,
    hi: u64,
    ids: [u32; INLINE],
    meta: u32,
}

const EMPTY_SLOT: Slot = Slot {
    lo: 0,
    hi: 0,
    ids: [0; INLINE],
    meta: 0,
};

/// Two slots, one 64-byte cache line: the unit of probing.
#[derive(Clone, Copy)]
#[repr(C, align(64))]
struct Pair([Slot; 2]);

const EMPTY_PAIR: Pair = Pair([EMPTY_SLOT; 2]);

#[inline(always)]
fn load8(b: &[u8], i: usize) -> u64 {
    u64::from_le_bytes(b[i..i + 8].try_into().unwrap())
}

/// The 128-bit key of the word `b[s..e]` (1 to 15 bytes) as two words: its bytes, zero padded, with its length in the top byte (one unaligned 128-bit load and a mask when 16 bytes are available).
#[inline(always)]
fn key_of(b: &[u8], s: usize, e: usize) -> (u64, u64) {
    let len = e - s;
    let x = if s + 16 <= b.len() {
        u128::from_le_bytes(b[s..s + 16].try_into().unwrap()) & ((1u128 << (8 * len)) - 1)
    } else {
        let mut tmp = [0u8; 16];
        tmp[..len].copy_from_slice(&b[s..e]);
        u128::from_le_bytes(tmp)
    };
    let x = x | ((len as u128) << 120);
    (x as u64, (x >> 64) as u64)
}

/// Fibonacci hashing: the top bits of the product index the table.
#[inline(always)]
fn hash_key(lo: u64, hi: u64) -> u64 {
    (lo ^ hi.wrapping_mul(0xC2B2_AE3D_27D4_EB4F)).wrapping_mul(0x9E37_79B9_7F4A_7C15)
}

/// Open addressing over pairs of slots, load factor up to 3/4; a pair that contains an empty slot ends every lookup that reaches it (words are written into the first empty slot of their probe walk).
struct WordCache {
    pairs: Vec<Pair>,
    pair_mask: usize,
    /// 64 - log2(number of pairs)
    shift: u32,
    len: usize,
    spill: Vec<u32>,
}

impl WordCache {
    fn new() -> WordCache {
        let n = 1 << 6;
        WordCache {
            pairs: vec![EMPTY_PAIR; n],
            pair_mask: n - 1,
            shift: 64 - n.trailing_zeros(),
            len: 0,
            spill: Vec::new(),
        }
    }

    #[inline(always)]
    fn find(&self, lo: u64, hi: u64, h: u64) -> Result<&Slot, usize> {
        let mut p = (h >> self.shift) as usize;
        loop {
            let pair = &self.pairs[p];
            let a = &pair.0[0];
            if a.lo == lo && a.hi == hi {
                return Ok(a);
            }
            if a.hi == 0 {
                return Err(p * 2);
            }
            let b = &pair.0[1];
            if b.lo == lo && b.hi == hi {
                return Ok(b);
            }
            if b.hi == 0 {
                return Err(p * 2 + 1);
            }
            p = (p + 1) & self.pair_mask;
        }
    }

    #[inline(always)]
    fn get(&self, lo: u64, hi: u64, h: u64) -> Option<&Slot> {
        self.find(lo, hi, h).ok()
    }

    #[inline(always)]
    fn ids<'a>(&'a self, s: &'a Slot) -> &'a [u32] {
        let n = ((s.meta >> 4) & 15) as usize;
        if n <= INLINE {
            &s.ids[..n]
        } else {
            &self.spill[s.ids[0] as usize..s.ids[0] as usize + s.ids[1] as usize]
        }
    }

    /// Adds a word that is not in the table (the caller has looked it up).
    fn insert(&mut self, lo: u64, hi: u64, h: u64, ids: &[u32]) {
        if self.len >= MAX_SHORT_ENTRIES {
            self.clear();
        } else if (self.len + 1) * 4 > self.pairs.len() * 2 * 3 {
            self.grow();
        }
        let pos = match self.find(lo, hi, h) {
            Ok(_) => return,
            Err(pos) => pos,
        };
        let mut slot = Slot {
            lo,
            hi,
            ids: [0; INLINE],
            meta: 0,
        };
        if ids.len() <= INLINE {
            slot.ids[..ids.len()].copy_from_slice(ids);
            slot.meta |= (ids.len() as u32) << 4;
        } else {
            slot.ids[0] = self.spill.len() as u32;
            slot.ids[1] = ids.len() as u32;
            slot.meta |= 15 << 4;
            self.spill.extend_from_slice(ids);
        }
        self.pairs[pos >> 1].0[pos & 1] = slot;
        self.len += 1;
    }

    fn clear(&mut self) {
        self.pairs.iter_mut().for_each(|p| *p = EMPTY_PAIR);
        self.spill.clear();
        self.len = 0;
    }

    fn grow(&mut self) {
        let n = self.pairs.len() * 2;
        let old = std::mem::replace(&mut self.pairs, vec![EMPTY_PAIR; n]);
        self.pair_mask = n - 1;
        self.shift -= 1;
        for s in old.iter().flat_map(|p| p.0.iter()).filter(|s| s.hi != 0) {
            let h = hash_key(s.lo, s.hi);
            let mut p = (h >> self.shift) as usize;
            'walk: loop {
                for k in 0..2 {
                    if self.pairs[p].0[k].hi == 0 {
                        self.pairs[p].0[k] = *s;
                        break 'walk;
                    }
                }
                p = (p + 1) & self.pair_mask;
            }
        }
    }
}

/// Appends the first `n` (at most `INLINE`) of the three ids of a slot. Always writes all three (a fixed-size copy instead of a call with a variable length) and keeps only `n` of them.
#[inline(always)]
fn push_inline(out: &mut Vec<u32>, ids: &[u32; INLINE], n: usize) {
    debug_assert!(n <= INLINE);
    out.reserve(INLINE);
    let len = out.len();
    // SAFETY: room for INLINE more elements was reserved above; the copy fills spare capacity only, and the length grows by n <= INLINE of those elements.
    unsafe {
        std::ptr::copy_nonoverlapping(ids.as_ptr(), out.as_mut_ptr().add(len), INLINE);
        out.set_len(len + n);
    }
}

// ---- the scanner --------------------------------------------------------------------------------------------------------------------------------------------------------------

const OTHER: u8 = 0;
const LETTER: u8 = 1;
const NUMBER: u8 = 2;
const SPACE: u8 = 3;
const END: u8 = 255;

/// The class of every ASCII character (`\p{L}`, `\p{N}`, `\s`, anything else), built from the same definitions as the character scanners.
const ASCII_CLASS: [u8; 128] = {
    let mut t = [OTHER; 128];
    let mut i = 0;
    while i < 128 {
        let c = i as u8;
        t[i] = if c.is_ascii_alphabetic() {
            LETTER
        } else if c.is_ascii_digit() {
            NUMBER
        } else if matches!(c, 0x09..=0x0D | 0x20) {
            SPACE
        } else {
            OTHER
        };
        i += 1;
    }
    t
};

/// Class and width in bytes of the character that starts at byte `i` (the text is valid UTF-8 and `i` is a character boundary).
#[inline(always)]
fn class_at(b: &[u8], i: usize) -> (u8, usize) {
    let x = b[i];
    if x < 0x80 {
        (ASCII_CLASS[x as usize], 1)
    } else {
        class_wide(b, i)
    }
}

#[inline(never)]
fn class_wide(b: &[u8], i: usize) -> (u8, usize) {
    let x = b[i] as u32;
    let (c, w) = if x >= 0xF0 {
        (
            ((x & 0x07) << 18)
                | ((b[i + 1] as u32 & 0x3F) << 12)
                | ((b[i + 2] as u32 & 0x3F) << 6)
                | (b[i + 3] as u32 & 0x3F),
            4,
        )
    } else if x >= 0xE0 {
        (
            ((x & 0x0F) << 12) | ((b[i + 1] as u32 & 0x3F) << 6) | (b[i + 2] as u32 & 0x3F),
            3,
        )
    } else {
        (((x & 0x1F) << 6) | (b[i + 1] as u32 & 0x3F), 2)
    };
    let c = char::from_u32(c).unwrap_or('\u{FFFD}');
    let k = if is_letter(c) {
        LETTER
    } else if is_number(c) {
        NUMBER
    } else if is_space(c) {
        SPACE
    } else {
        OTHER
    };
    (k, w)
}

/// End of the run of characters of class `class` that continues from byte `e`.
#[inline(always)]
fn run_of(b: &[u8], mut e: usize, class: u8) -> usize {
    while e < b.len() {
        let (k, w) = class_at(b, e);
        if k != class {
            break;
        }
        e += w;
    }
    e
}

const HI_BITS: u64 = 0x8080_8080_8080_8080;

/// 0x80 in every byte of `x` that is NOT an ASCII letter (eight bytes at once: lower-case, then the range test a..z by two additions that cannot carry into the next byte).
#[inline(always)]
fn not_letter_mask(x: u64) -> u64 {
    let y = (x | 0x2020_2020_2020_2020) & 0x7F7F_7F7F_7F7F_7F7F;
    let ge_a = y.wrapping_add(0x1F1F_1F1F_1F1F_1F1F) & HI_BITS; // y >= 0x61
    let gt_z = y.wrapping_add(0x0505_0505_0505_0505) & HI_BITS; // y >= 0x7B
    !(ge_a & !gt_z & !x) & HI_BITS
}

/// End of the run of letters that continues from byte `e`: eight ASCII letters per step while they last, the character classes (which know the non-ASCII letters) at the edges.
#[inline(always)]
fn letter_run_end(b: &[u8], mut e: usize) -> usize {
    let n = b.len();
    loop {
        while e + 8 <= n {
            let m = not_letter_mask(load8(b, e));
            if m == 0 {
                e += 8;
                continue;
            }
            e += (m.trailing_zeros() >> 3) as usize;
            break;
        }
        if e >= n {
            return e;
        }
        let (k, w) = class_at(b, e);
        if k != LETTER {
            return e;
        }
        e += w;
    }
}

/// Length of the English contraction (`'s 't 're 've 'm 'll 'd`, lower case) that starts at the apostrophe at `b[i]`, or 0.
#[inline]
fn contraction(b: &[u8], i: usize) -> usize {
    let r = &b[i + 1..];
    match r.first() {
        Some(b's') | Some(b't') | Some(b'm') | Some(b'd') => 2,
        Some(b'r') | Some(b'v') if r.get(1) == Some(&b'e') => 3,
        Some(b'l') if r.get(1) == Some(&b'l') => 3,
        _ => 0,
    }
}

/// The pieces of the GPT-2 pre-tokenizer as byte ranges, in order; `digits` = the Digits(individual) splitter in front of it (every number character is a piece of its own and ends the piece before it).
/// The same decisions as `pretok::split_gpt2` / `PreTokenizer::Gpt2Digits`:  's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
pub fn scan_gpt2(b: &[u8], digits: bool, f: &mut impl FnMut(usize, usize)) {
    let n = b.len();
    let mut i = 0;
    while i < n {
        let (k0, w0) = class_at(b, i);
        if digits && k0 == NUMBER {
            f(i, i + w0);
            i += w0;
            continue;
        }
        if b[i] == b'\'' {
            let m = contraction(b, i);
            if m > 0 {
                f(i, i + m);
                i += m;
                continue;
            }
        }
        // an optional single space, then a run of letters, of numbers or of anything else but whitespace
        let (j, kj, wj) = if b[i] == b' ' {
            if i + 1 < n {
                let (k, w) = class_at(b, i + 1);
                (i + 1, k, w)
            } else {
                (i + 1, END, 0)
            }
        } else {
            (i, k0, w0)
        };
        if kj == LETTER || kj == OTHER || (kj == NUMBER && !digits) {
            let e = if kj == LETTER {
                letter_run_end(b, j + wj)
            } else {
                run_of(b, j + wj, kj)
            };
            f(i, e);
            i = e;
            continue;
        }
        // whitespace: `\s+(?!\S)` (all of the run, or all but its last character when a word follows, which then takes that space), else the run
        let mut e = i;
        let mut last = i;
        while e < n {
            let (k, w) = class_at(b, e);
            if k != SPACE {
                break;
            }
            last = e;
            e += w;
        }
        let chunk_end = e >= n || (digits && class_at(b, e).0 == NUMBER);
        let cut = if chunk_end || last == i { e } else { last };
        f(i, cut);
        i = cut;
    }
}

// ---- the encoder --------------------------------------------------------------------------------------------------------------------------------------------------------------

/// An encoder for one thread: the tokenizer plus a cache of words and merge buffers. Create one per thread and keep it across documents; `Tokenizer::encode` makes a fresh one per call.
pub struct Encoder<'a> {
    tok: &'a Tokenizer,
    cache: WordCache,
    /// words of 16 to `MAX_LONG` bytes
    long: HashMap<Box<[u8]>, Box<[u32]>, FxBuild>,
    scratch: BpeScratch,
    tmp: Vec<u32>,
}

impl<'a> Encoder<'a> {
    pub fn new(tok: &'a Tokenizer) -> Encoder<'a> {
        Encoder {
            tok,
            cache: WordCache::new(),
            long: HashMap::with_hasher(FxBuild::default()),
            scratch: BpeScratch::default(),
            tmp: Vec::new(),
        }
    }

    /// Number of distinct words held by the cache.
    pub fn cached_words(&self) -> usize {
        self.cache.len + self.long.len()
    }

    pub fn encode(&mut self, text: &str) -> Vec<u32> {
        let mut ids = Vec::with_capacity(text.len() / 3 + 1);
        self.encode_into(text, &mut ids);
        ids
    }

    /// Appends the ids of `text` to `out` (no BOS/EOS); added tokens found in the text become their ids.
    pub fn encode_into(&mut self, text: &str, out: &mut Vec<u32>) {
        let tok = self.tok;
        let bytes = text.as_bytes();
        let mut seg_start = 0usize;
        if !tok.added.is_empty() {
            let mut i = 0;
            while i < bytes.len() {
                // jump to the next byte that can start an added token (eight bytes per step when they all start with the same byte, as `<` in the usual chat and tool markers)
                match tok.added_single {
                    Some(b) => match find_byte(bytes, b, i) {
                        Some(j) => i = j,
                        None => break,
                    },
                    None => {
                        while i < bytes.len() && !tok.added_first[bytes[i] as usize] {
                            i += 1;
                        }
                        if i >= bytes.len() {
                            break;
                        }
                    }
                }
                if let Some(a) = tok
                    .added
                    .iter()
                    .find(|a| bytes[i..].starts_with(a.content.as_bytes()))
                {
                    if seg_start < i {
                        self.segment(&text[seg_start..i], out);
                    }
                    out.push(a.id);
                    i += a.content.len();
                    seg_start = i;
                    continue;
                }
                i += 1;
            }
        }
        if seg_start < bytes.len() {
            self.segment(&text[seg_start..], out);
        }
    }

    fn segment(&mut self, seg: &str, out: &mut Vec<u32>) {
        let tok = self.tok;
        match tok.fast {
            Some(digits) => {
                let b = seg.as_bytes();
                let Encoder {
                    cache,
                    long,
                    scratch,
                    tmp,
                    ..
                } = self;
                scan_gpt2(b, digits, &mut |s, e| {
                    let len = e - s;
                    if len <= SHORT {
                        let (lo, hi) = key_of(b, s, e);
                        let h = hash_key(lo, hi);
                        if let Some(slot) = cache.get(lo, hi, h) {
                            let n = ((slot.meta >> 4) & 15) as usize;
                            if n <= INLINE {
                                push_inline(out, &slot.ids, n);
                            } else {
                                out.extend_from_slice(cache.ids(slot));
                            }
                        } else {
                            tmp.clear();
                            tok.bpe_scratch(&b[s..e], tmp, scratch);
                            out.extend_from_slice(tmp);
                            cache.insert(lo, hi, h, tmp);
                        }
                    } else if len <= MAX_LONG {
                        if let Some(ids) = long.get(&b[s..e]) {
                            out.extend_from_slice(ids);
                        } else {
                            tmp.clear();
                            tok.bpe_scratch(&b[s..e], tmp, scratch);
                            out.extend_from_slice(tmp);
                            if long.len() >= MAX_LONG_ENTRIES {
                                long.clear();
                            }
                            long.insert(b[s..e].into(), tmp.as_slice().into());
                        }
                    } else {
                        tok.bpe_scratch(&b[s..e], out, scratch);
                    }
                });
            }
            None => tok.encode_segment(seg, out),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pretok::PreTokenizer;
    use crate::train::{train_from_counts, TrainConfig};

    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            self.0 >> 33
        }
        fn below(&mut self, n: usize) -> usize {
            self.next() as usize % n
        }
    }

    /// Characters and short sequences the pre-tokenizer treats specially.
    const ATOMS: [&str; 46] = [
        "a",
        "b",
        "Z",
        "hello",
        "world",
        "é",
        "中",
        "ก",
        "\u{0e31}",
        "\u{0301}",
        "1",
        "7",
        "٣",
        "½",
        " ",
        "  ",
        "   ",
        "\t",
        "\n",
        "\r\n",
        "\u{a0}",
        "\u{3000}",
        "\u{2003}",
        "\u{85}",
        "'",
        "'s",
        "'re",
        "'ll",
        "'d",
        "'m",
        "'t",
        "'ve",
        "'S",
        "!",
        ".",
        ",",
        "-",
        "_",
        "\u{1F600}",
        "\u{FFFD}",
        "\0",
        "—",
        "«",
        "$",
        "internationalization",
        "abcdefghijklmnopqrstuvwxyz",
    ];

    fn random_text(rng: &mut Lcg, max_atoms: usize) -> String {
        let n = rng.below(max_atoms + 1);
        (0..n).map(|_| ATOMS[rng.below(ATOMS.len())]).collect()
    }

    fn reference_spans(pre: PreTokenizer, s: &str) -> Vec<(usize, usize)> {
        let chars: Vec<char> = s.chars().collect();
        let mut offs = Vec::with_capacity(chars.len() + 1);
        let mut o = 0;
        for c in &chars {
            offs.push(o);
            o += c.len_utf8();
        }
        offs.push(o);
        pre.split(&chars)
            .into_iter()
            .map(|(a, b)| (offs[a], offs[b]))
            .collect()
    }

    #[test]
    fn the_byte_scanner_cuts_where_the_character_scanners_cut() {
        let mut rng = Lcg(12345);
        for (pre, digits) in [
            (PreTokenizer::Gpt2, false),
            (PreTokenizer::Gpt2Digits, true),
        ] {
            for _ in 0..60_000 {
                let s = random_text(&mut rng, 30);
                let mut got = Vec::new();
                scan_gpt2(s.as_bytes(), digits, &mut |a, b| got.push((a, b)));
                assert_eq!(got, reference_spans(pre, &s), "{pre:?} on {s:?}");
            }
        }
    }

    #[test]
    fn find_byte_agrees_with_the_plain_search() {
        let mut rng = Lcg(77);
        for _ in 0..50_000 {
            let n = rng.below(60);
            let hay: Vec<u8> = (0..n)
                .map(|_| {
                    if rng.below(6) == 0 {
                        b'<'
                    } else {
                        rng.next() as u8
                    }
                })
                .collect();
            let from = if n == 0 { 0 } else { rng.below(n + 1) };
            let want = hay[from.min(n)..]
                .iter()
                .position(|&b| b == b'<')
                .map(|p| p + from);
            assert_eq!(find_byte(&hay, b'<', from), want, "{hay:?} from {from}");
        }
    }

    #[test]
    fn eight_letters_at_a_time_agree_with_the_byte_test() {
        let mut rng = Lcg(5);
        // every byte value in every position, then random words
        for pos in 0..8 {
            for v in 0..=255u8 {
                let mut w = [b'a'; 8];
                w[pos] = v;
                let m = not_letter_mask(u64::from_le_bytes(w));
                for (k, &c) in w.iter().enumerate() {
                    assert_eq!(
                        m >> (8 * k + 7) & 1 == 1,
                        !c.is_ascii_alphabetic(),
                        "byte {c:#x} at {k}"
                    );
                }
                assert_eq!(m & !HI_BITS, 0);
            }
        }
        for _ in 0..100_000 {
            let w: [u8; 8] = std::array::from_fn(|_| {
                if rng.below(4) == 0 {
                    rng.next() as u8
                } else {
                    b"azAZ@[`{09 _"[rng.below(12)]
                }
            });
            let m = not_letter_mask(u64::from_le_bytes(w));
            for (k, &c) in w.iter().enumerate() {
                assert_eq!(m >> (8 * k + 7) & 1 == 1, !c.is_ascii_alphabetic());
            }
        }
    }

    #[test]
    fn the_word_cache_returns_what_it_was_given() {
        let mut c = WordCache::new();
        let mut rng = Lcg(7);
        let mut want: HashMap<Vec<u8>, Vec<u32>> = HashMap::new();
        let mut buf = vec![0u8; 40];
        for _ in 0..200_000 {
            let len = 1 + rng.below(SHORT);
            let w: Vec<u8> = (0..len).map(|_| b'a' + rng.below(5) as u8).collect();
            buf[..len].copy_from_slice(&w);
            let (lo, hi) = key_of(&buf, 0, len);
            let h = hash_key(lo, hi);
            if want.contains_key(&w) {
                let slot = c.get(lo, hi, h).expect("a stored word is found");
                assert_eq!(c.ids(slot), &want[&w][..]);
            } else {
                assert!(c.get(lo, hi, h).is_none(), "{w:?}");
                let ids: Vec<u32> = (0..1 + rng.below(6)).map(|_| rng.next() as u32).collect(); // 1..=6 ids: inline and spilled
                c.insert(lo, hi, h, &ids);
                want.insert(w, ids);
            }
        }
        assert_eq!(c.len, want.len());
    }

    /// A small tokenizer trained on generated text (so that the test needs no model files).
    fn small_tokenizer() -> Tokenizer {
        let mut rng = Lcg(99);
        let syll = [
            "ka", "to", "ri", "me", "sun", "la", "ven", "ox", "ing", "er", "the", "and", "é", "中",
        ];
        let mut counts: HashMap<Vec<u8>, u64> = HashMap::new();
        for _ in 0..40_000 {
            let w: String = (0..1 + rng.below(3))
                .map(|_| syll[rng.below(syll.len())])
                .collect();
            let w = if rng.below(3) > 0 { format!(" {w}") } else { w };
            *counts.entry(w.into_bytes()).or_insert(0) += 1 + rng.below(5) as u64;
        }
        let cfg = TrainConfig {
            vocab_size: 700,
            special_tokens: vec![
                "<|endoftext|>".into(),
                "<|im_end|>".into(),
                "<think>".into(),
            ],
            threads: 1,
            log_every: 0,
        };
        let trained = train_from_counts(counts, &cfg).unwrap();
        Tokenizer::from_json(&trained.to_tokenizer_json()).unwrap()
    }

    #[test]
    fn the_fast_encoder_gives_the_reference_ids() {
        let tok = small_tokenizer();
        assert_eq!(tok.fast, Some(true));
        let mut enc = tok.encoder();
        let mut rng = Lcg(2024);
        let words = [
            "kato",
            " rimesun",
            "venox",
            " thing",
            "andthe",
            "Ünïcode",
            "x",
            " kakakakakakakakakakakakaka",
            "tototototototototototototototototototototototototo",
        ];
        for round in 0..30_000 {
            let mut s = String::new();
            for _ in 0..rng.below(14) {
                match rng.below(8) {
                    0 => s.push_str("<|im_end|>"),
                    1 => s.push_str("<think>"),
                    2 | 3 => s.push_str(words[rng.below(words.len())]),
                    _ => s.push_str(ATOMS[rng.below(ATOMS.len())]),
                }
            }
            let want = tok.encode_reference(&s);
            // the same encoder many times (cache hits from earlier texts), a fresh one, and the convenience call
            assert_eq!(enc.encode(&s), want, "round {round}: {s:?}");
            if round % 50 == 0 {
                assert_eq!(tok.encode(&s), want);
                assert_eq!(tok.decode(&want, false), tok.decode(&tok.encode(&s), false));
            }
        }
        assert!(enc.cached_words() > 10);
    }

    /// Speed of the pieces of the encoder on a text file: `FORGE_BENCH_TEXT=/path cargo test --release -p forge-tok -- --ignored --nocapture bench`
    #[test]
    #[ignore]
    fn bench_pieces() {
        let Ok(path) = std::env::var("FORGE_BENCH_TEXT") else {
            return;
        };
        let tok_path = std::env::var("FORGE_BENCH_TOKENIZER").unwrap_or_else(|_| {
            format!(
                "{}/forgeinfer/data/tokenizer_f1.json",
                std::env::var("HOME").unwrap()
            )
        });
        let tok = Tokenizer::from_file(tok_path).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        let docs: Vec<&[u8]> = bytes.split(|&b| b == 0).filter(|d| !d.is_empty()).collect();
        let texts: Vec<String> = docs
            .iter()
            .map(|d| String::from_utf8_lossy(d).into_owned())
            .collect();
        let total: usize = texts.iter().map(|t| t.len()).sum();
        let mb = total as f64 / 1e6;
        let t0 = std::time::Instant::now();
        let mut pieces = 0usize;
        let mut checksum = 0usize;
        for t in &texts {
            scan_gpt2(t.as_bytes(), true, &mut |a, b| {
                pieces += 1;
                checksum = checksum.wrapping_add(a ^ b);
            });
        }
        let dt = t0.elapsed().as_secs_f64();
        eprintln!(
            "scan only: {:.1} MB/s ({} pieces, {:.1} bytes per piece, checksum {checksum})",
            mb / dt,
            pieces,
            total as f64 / pieces as f64
        );
        // scan + the key and the hash of every short word, no table access
        let t0 = std::time::Instant::now();
        let mut acc = 0u64;
        for t in &texts {
            let b = t.as_bytes();
            scan_gpt2(b, true, &mut |x, y| {
                let len = y - x;
                if len <= SHORT {
                    let (lo, hi) = key_of(b, x, y);
                    acc = acc.wrapping_add(hash_key(lo, hi));
                }
            });
        }
        eprintln!(
            "scan + key + hash, no table: {:.1} MB/s (acc {acc})",
            mb / t0.elapsed().as_secs_f64()
        );
        // scan + hash lookups of the words (a table of the distinct words of the whole text, so every lookup hits)
        let mut c = WordCache::new();
        let mut miss = 0usize;
        for pass in 0..2 {
            let t0 = std::time::Instant::now();
            let mut hits = 0usize;
            for t in &texts {
                let b = t.as_bytes();
                scan_gpt2(b, true, &mut |x, y| {
                    let len = y - x;
                    if len <= SHORT {
                        let (lo, hi) = key_of(b, x, y);
                        let h = hash_key(lo, hi);
                        if c.get(lo, hi, h).is_some() {
                            hits += 1;
                        } else if pass == 0 {
                            c.insert(lo, hi, h, &[1, 2]);
                            miss += 1;
                        }
                    }
                });
            }
            let dt = t0.elapsed().as_secs_f64();
            eprintln!("scan + cache lookups, pass {pass}: {:.1} MB/s ({hits} hits, {miss} words inserted)", mb / dt);
        }
        // the whole encoder, with a cold and a warm cache
        for pass in 0..2 {
            let mut enc = tok.encoder();
            let t0 = std::time::Instant::now();
            let mut out = Vec::new();
            for t in &texts {
                out.clear();
                enc.encode_into(t, &mut out);
            }
            let dt = t0.elapsed().as_secs_f64();
            eprintln!(
                "encoder, fresh cache, run {pass}: {:.1} MB/s ({} cached words)",
                mb / dt,
                enc.cached_words()
            );
        }
        let mut enc = tok.encoder();
        let mut out = Vec::new();
        for t in &texts {
            out.clear();
            enc.encode_into(t, &mut out);
        }
        let t0 = std::time::Instant::now();
        for t in &texts {
            out.clear();
            enc.encode_into(t, &mut out);
        }
        eprintln!(
            "encoder, warm cache: {:.1} MB/s",
            mb / t0.elapsed().as_secs_f64()
        );
    }
}
