//! The prompt cache: the conversations the GPU's KV cache has been used away from, kept in page-locked host memory so that going back to one costs a copy over PCIe instead of reading its prompt again.
//!
//! The engine holds ONE conversation on the GPU (`slot`: its tokens, the session: their KV entries and, for a hybrid model, the recurrent state). A request that shares little with it used to throw it away: a
//! long document read in one chat had to be read again when the user came back from another chat (16,000 tokens: 4 seconds on the 4B of an RTX 4060, 11 on a 2060 SUPER, 75 on the 27B). Now, when a request is
//! about to discard most of a conversation of at least `MIN_STORE` tokens, its KV entries (and the state of a hybrid model at its last snapshot) are copied to host memory (a budget: FORGE_PROMPT_CACHE_MB,
//! default 2,048, 0 = off; the least recently stored go first), and a request that continues a stored conversation brings it back (12 GB/s: 16,000 tokens of the 4B hybrid are 0.5 GB, 40 ms).
//!
//! A dense model can resume from the longest prefix a stored conversation shares with the prompt (its KV entries are position-wise); a hybrid model only from a conversation whose stored tokens are ALL a prefix of
//! the prompt (the recurrent state summarises exactly those tokens: the message boundary of its last snapshot).

use super::Snap;
use forge_core::Result;
use forge_cuda::PinnedBuffer;
use forge_gpu::{GpuSession, KvBlob, StateSnapshot};
use std::time::Instant;

/// A conversation with fewer tokens than this is cheaper to read again than to keep.
pub const MIN_STORE: usize = 1024;
/// A stored conversation is brought back only when it saves at least this many tokens of prefill.
pub const MIN_GAIN: usize = 512;

pub struct Entry<P> {
    pub tokens: Vec<u32>,
    pub payload: P,
    pub bytes: usize,
    used: u64,
}

/// The entries and their budget (generic over what an entry carries, so that the policy is tested without a GPU).
pub struct Store<P> {
    entries: Vec<Entry<P>>,
    budget: usize,
    bytes: usize,
    clock: u64,
}

impl<P> Store<P> {
    pub fn new(budget: usize) -> Store<P> {
        Store { entries: Vec::new(), budget, bytes: 0, clock: 0 }
    }

    pub fn budget(&self) -> usize {
        self.budget
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn bytes(&self) -> usize {
        self.bytes
    }

    /// The entry a prompt can resume from and how many positions of it it can use: for a model that cannot roll back (`exact_prefix`) an entry whose tokens are all a prefix of the prompt (and leave a token to
    /// evaluate), otherwise the entry that shares the longest prefix with the prompt (at most all but its last token). The longest wins; None when nothing is shared.
    pub fn best(&self, ids: &[u32], exact_prefix: bool) -> Option<(usize, usize)> {
        let mut best: Option<(usize, usize)> = None;
        for (i, e) in self.entries.iter().enumerate() {
            let n = if exact_prefix {
                if e.tokens.len() < ids.len() && ids.starts_with(&e.tokens) {
                    e.tokens.len()
                } else {
                    0
                }
            } else {
                e.tokens.iter().zip(ids).take_while(|(a, b)| a == b).count().min(ids.len().saturating_sub(1))
            };
            if n > 0 && best.map_or(true, |(_, bn)| n > bn) {
                best = Some((i, n));
            }
        }
        best
    }

    pub fn take(&mut self, i: usize) -> Entry<P> {
        let e = self.entries.remove(i);
        self.bytes -= e.bytes;
        e
    }

    /// Count `bytes` of idle memory (a spare buffer) against the budget; false when they do not fit next to the entries.
    pub fn reserve(&mut self, bytes: usize) -> bool {
        if self.bytes + bytes > self.budget {
            return false;
        }
        self.bytes += bytes;
        true
    }

    /// The reverse of `reserve`.
    pub fn unreserve(&mut self, bytes: usize) {
        self.bytes = self.bytes.saturating_sub(bytes);
    }

    /// Make room for `bytes` more by dropping the entries stored longest ago; false when the budget could not hold them even when empty.
    pub fn make_room(&mut self, bytes: usize) -> bool {
        if bytes > self.budget {
            return false;
        }
        while self.bytes + bytes > self.budget {
            let Some(oldest) = self.entries.iter().enumerate().min_by_key(|(_, e)| e.used).map(|(i, _)| i) else {
                break;
            };
            self.take(oldest);
        }
        true
    }

    /// Keep a conversation (its first tokens and what the payload carries). The entries whose tokens are a prefix of the new one are the same conversation, shorter: they go. False when it does not fit.
    pub fn put(&mut self, tokens: Vec<u32>, payload: P, bytes: usize) -> bool {
        let mut i = 0;
        while i < self.entries.len() {
            if tokens.starts_with(&self.entries[i].tokens) {
                self.take(i);
            } else {
                i += 1;
            }
        }
        if !self.make_room(bytes) {
            return false;
        }
        self.clock += 1;
        self.bytes += bytes;
        self.entries.push(Entry { tokens, payload, bytes, used: self.clock });
        true
    }
}

/// What a stored conversation carries besides its tokens: the KV entries of its positions and, for a hybrid model, the recurrent state after them.
pub struct Conv {
    kv: KvBlob,
    state: Option<StateSnapshot>,
}

/// The stored conversations and one spare page-locked buffer: locking 1 GB of host memory takes 0.3 s and unlocking it as long, so the buffer a conversation came back from is kept for the next one that is
/// stored (a swap is a pair: one comes back, the one it replaces goes out). The spare counts against the budget.
pub struct Cache {
    store: Store<Conv>,
    spare: Option<PinnedBuffer>,
}

impl Cache {
    pub fn new(budget: usize) -> Cache {
        Cache { store: Store::new(budget), spare: None }
    }
}

/// Before a request is served: bring back a stored conversation when the prompt continues it and that saves a lot of prefill, and store the conversation the GPU holds when this request is about to discard most
/// of it. Anything that goes wrong leaves the engine as it was (a failed copy to the host) or empty (a failed copy back: the session is reset, the prompt is read in full).
pub fn swap(sess: &mut GpuSession, slot: &mut Vec<u32>, snaps: &mut Vec<Snap>, cache: &mut Cache, ids: &[u32]) {
    if cache.store.budget() == 0 || ids.len() < 2 {
        return;
    }
    if let Err(e) = swap_inner(sess, slot, snaps, cache, ids) {
        eprintln!("[cache] failed ({e}): the conversation is read again");
        slot.clear();
        snaps.clear();
        let _ = sess.reset();
    }
}

fn swap_inner(sess: &mut GpuSession, slot: &mut Vec<u32>, snaps: &mut Vec<Snap>, cache: &mut Cache, ids: &[u32]) -> Result<()> {
    let dense = sess.can_truncate();
    let have = slot.len().min(sess.position());
    // how much of what the GPU holds this prompt can use as it is
    let usable_now = if dense {
        slot.iter().zip(ids).take_while(|(a, b)| a == b).count().min(ids.len() - 1).min(have)
    } else if have == slot.len() && ids.len() > have && ids.starts_with(slot) {
        have
    } else {
        snaps.iter().filter(|s| s.tokens.len() < ids.len() && slot.starts_with(&s.tokens) && ids.starts_with(&s.tokens)).map(|s| s.tokens.len()).max().unwrap_or(0)
    };
    let found = cache.store.best(ids, !dense).filter(|&(_, n)| n >= usable_now + MIN_GAIN);
    // (taken first: storing the conversation of the GPU must not make room by dropping it)
    let entry = found.map(|(i, n)| (cache.store.take(i), n));
    if have >= MIN_STORE && usable_now * 2 < have {
        if let Err(e) = keep(sess, slot, snaps, cache, have) {
            eprintln!("[cache] the conversation of {have} tokens could not be kept ({e})");
        }
    }
    if let Some((e, n)) = entry {
        let t0 = Instant::now();
        let Entry { tokens, payload: Conv { kv, state }, bytes, .. } = e;
        sess.adopt(&kv, n, state.as_ref())?;
        slot.clear();
        slot.extend_from_slice(&tokens[..n]);
        snaps.clear();
        if let Some(st) = state {
            snaps.push(Snap { tokens: tokens[..n].to_vec(), state: st });
        }
        // the buffer the conversation came back from waits for the next one to be stored
        let buf = kv.into_buffer();
        if cache.spare.is_none() && cache.store.reserve(buf.len()) {
            cache.spare = Some(buf);
        }
        eprintln!("[cache] a conversation of {n} tokens is back on the GPU ({:.0} MiB, {:.0} ms); {} others stored ({:.0} MiB)", bytes as f64 / 1048576.0, t0.elapsed().as_secs_f64() * 1e3, cache.store.len(), cache.store.bytes() as f64 / 1048576.0);
    }
    Ok(())
}

/// Copy the conversation of the GPU to the cache: a dense model's first `have` positions; a hybrid model's up to its last snapshot (where the next turn resumes), or, with no snapshot, all of them.
fn keep(sess: &mut GpuSession, slot: &[u32], snaps: &mut Vec<Snap>, cache: &mut Cache, have: usize) -> Result<()> {
    let t0 = Instant::now();
    let (tokens, state) = if sess.can_truncate() {
        (slot[..have].to_vec(), None)
    } else if let Some(sn) = snaps.pop() {
        (sn.tokens, Some(sn.state))
    } else if have == slot.len() {
        (slot.to_vec(), Some(sess.snapshot_state()?))
    } else {
        return Ok(());
    };
    let n = tokens.len();
    if n < MIN_STORE {
        return Ok(());
    }
    let need = sess.kv_blob_bytes(n);
    let extra = state.as_ref().map_or(0, |s| s.bytes());
    // the spare buffer is used when it fits; either way its bytes stop counting as idle
    let spare = cache.spare.take();
    if let Some(b) = &spare {
        cache.store.unreserve(b.len());
    }
    let reuse = spare.filter(|b| spare_fits(b.len(), need));
    let size = reuse.as_ref().map_or_else(|| buffer_size(need, extra, cache.store.budget()), |b| b.len());
    // (room first: the pinned memory of the entries that go is freed before the new buffer is locked)
    if !cache.store.make_room(size + extra) {
        return Ok(());
    }
    let buf = match reuse {
        Some(b) => b,
        None => PinnedBuffer::alloc(size)?,
    };
    let kv = sess.export_kv_with(n, Some(buf))?;
    let bytes = kv.capacity() + extra;
    if cache.store.put(tokens, Conv { kv, state }, bytes) {
        eprintln!("[cache] a conversation of {n} tokens was stored ({:.0} MiB, {:.0} ms); {} stored in all ({:.0} MiB of {:.0})", bytes as f64 / 1048576.0, t0.elapsed().as_secs_f64() * 1e3, cache.store.len(), cache.store.bytes() as f64 / 1048576.0, cache.store.budget() as f64 / 1048576.0);
    }
    Ok(())
}

/// The host buffer for a conversation of `need` bytes of KV entries (and `extra` bytes of recurrent state beside it, which the budget must hold too): an eighth bigger than the entries, because a conversation grows
/// by a few turns before it is stored again and the buffer it came from should hold the next version as well (locking a new one costs 0.3 s per GB); never more than the budget allows, never less than needed.
fn buffer_size(need: usize, extra: usize, budget: usize) -> usize {
    (need + need / 8).min(budget.saturating_sub(extra)).max(need)
}

/// Is a spare buffer of `len` bytes right for a conversation that needs `need`: big enough, and not so big that the conversation would hold on to memory it does not use.
fn spare_fits(len: usize, need: usize) -> bool {
    len >= need && len <= need * 2
}

#[cfg(test)]
mod tests {
    use super::*;

    fn toks(n: usize, tag: u32) -> Vec<u32> {
        (0..n as u32).map(|i| tag * 100_000 + i).collect()
    }

    #[test]
    fn a_dense_model_resumes_from_the_longest_shared_prefix() {
        let mut s: Store<&str> = Store::new(1000);
        let mut doc = toks(2000, 1);
        s.put(doc.clone(), "doc", 100);
        s.put(toks(1500, 2), "other", 100);
        // a prompt that starts with 1,200 tokens of the document and then goes its own way
        let mut ask = doc[..1200].to_vec();
        ask.extend(toks(50, 9));
        assert_eq!(s.best(&ask, false), Some((0, 1200)));
        // all but the last token at most
        doc.push(7);
        assert_eq!(s.best(&doc[..2000], false), Some((0, 1999)));
        // nothing shared: nothing
        assert_eq!(s.best(&toks(100, 5), false), None);
        // the other entry
        let mut ask2 = toks(1500, 2);
        ask2.push(4);
        assert_eq!(s.best(&ask2, false), Some((1, 1500)));
    }

    #[test]
    fn a_hybrid_model_needs_the_stored_tokens_to_be_a_prefix_of_the_prompt() {
        let mut s: Store<()> = Store::new(1000);
        s.put(toks(1500, 1), (), 100);
        s.put(toks(1200, 2), (), 100);
        let mut p = toks(1500, 1);
        p.extend(toks(30, 8));
        assert_eq!(s.best(&p, true), Some((0, 1500)));
        // the same tokens but the prompt ends inside them: the state is of a position beyond the prompt
        assert_eq!(s.best(&toks(1499, 1), true), None);
        // equal to the stored tokens: nothing left to evaluate
        assert_eq!(s.best(&toks(1500, 1), true), None);
        // a prefix that is shared only in part: no state for it
        let mut q = toks(1000, 2);
        q.extend(toks(500, 7));
        assert_eq!(s.best(&q, true), None);
        // the longer of two matches wins
        let mut r = toks(1200, 2);
        r.extend(toks(10, 3));
        s.put(r.clone()[..1200 + 5].to_vec(), (), 100);
        let mut long = r;
        long.extend(toks(10, 4));
        assert_eq!(s.best(&long, true).map(|(_, n)| n), Some(1205));
    }

    #[test]
    fn the_budget_drops_the_entries_stored_longest_ago_and_a_longer_version_replaces_a_shorter_one() {
        let mut s: Store<u8> = Store::new(300);
        assert!(s.put(toks(1100, 1), 1, 100));
        assert!(s.put(toks(1100, 2), 2, 100));
        assert!(s.put(toks(1100, 3), 3, 100));
        assert_eq!((s.len(), s.bytes()), (3, 300));
        // a fourth: the first one goes
        assert!(s.put(toks(1100, 4), 4, 100));
        assert_eq!((s.len(), s.bytes()), (3, 300));
        assert_eq!(s.best(&{ let mut p = toks(1100, 1); p.push(0); p }, false), None);
        assert!(s.best(&{ let mut p = toks(1100, 2); p.push(0); p }, false).is_some());
        // the same conversation, longer: it replaces the shorter version instead of adding to it
        let mut longer = toks(1100, 2);
        longer.extend(toks(400, 6));
        assert!(s.put(longer, 22, 150));
        // (the shorter version went, 150 more bytes did not fit next to the other two, so the one stored longest ago went too)
        assert_eq!((s.len(), s.bytes()), (2, 250));
        // one that cannot fit at all is refused and nothing is dropped for it
        assert!(!s.put(toks(1100, 5), 5, 301));
        assert_eq!(s.len(), 2);
        // taking an entry frees its bytes
        let (i, _) = s.best(&{ let mut p = toks(1100, 4); p.push(0); p }, false).unwrap();
        assert_eq!(s.take(i).payload, 4);
        assert_eq!(s.bytes(), 150);
        // a budget of zero keeps nothing
        let mut off: Store<u8> = Store::new(0);
        assert!(!off.put(toks(1100, 1), 1, 1));
    }

    #[test]
    fn a_buffer_has_room_for_the_next_turns_and_a_spare_is_used_only_when_it_fits() {
        // an eighth to spare
        assert_eq!(buffer_size(800, 0, 10_000), 900);
        // the budget caps it (and the recurrent state beside it), but never below what the entries need
        assert_eq!(buffer_size(800, 0, 850), 850);
        assert_eq!(buffer_size(800, 100, 850), 800);
        assert_eq!(buffer_size(800, 100, 1000), 900);
        // the next version of a conversation (a few turns longer) fits the buffer of the last one
        let first = buffer_size(897 << 20, 0, 2048 << 20);
        assert!(spare_fits(first, 903 << 20));
        assert!(spare_fits(first, 1000 << 20));
        // a conversation that outgrew it, or one that would waste most of a big buffer: not
        assert!(!spare_fits(first, 1100 << 20));
        assert!(!spare_fits(first, 400 << 20));
        assert!(spare_fits(first, first / 2 + 1));
    }

    #[test]
    fn an_idle_buffer_counts_against_the_budget_until_it_is_used_or_freed() {
        let mut s: Store<u8> = Store::new(300);
        assert!(s.put(toks(1100, 1), 1, 100));
        // a spare buffer of 100 fits next to it, a second one of 150 does not
        assert!(s.reserve(100));
        assert!(!s.reserve(150));
        assert_eq!(s.bytes(), 200);
        // while it is reserved a new entry makes room for itself by dropping the oldest, not by touching the reservation
        assert!(s.put(toks(1100, 2), 2, 100));
        assert!(s.put(toks(1100, 3), 3, 100));
        assert_eq!((s.len(), s.bytes()), (2, 300));
        // using the spare gives its bytes back to the budget
        s.unreserve(100);
        assert_eq!(s.bytes(), 200);
        assert!(s.put(toks(1100, 4), 4, 100));
        assert_eq!((s.len(), s.bytes()), (3, 300));
        // (an unreserve never goes below zero)
        let mut t: Store<u8> = Store::new(300);
        t.unreserve(50);
        assert_eq!(t.bytes(), 0);
    }
}
