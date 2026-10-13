//! Prompt-lookup drafting: the tokens that followed an earlier occurrence of the text's last few tokens are proposed as drafts and verified exactly like the MTP head's (the main model's own
//! sample at every position decides, so the output has the distribution of plain decoding). Copying, quoting, code edits and structured output repeat themselves, and then a whole run of
//! tokens is confirmed in ONE sweep of the weights; in free prose the lookup finds nothing or is wrong, and an adaptive guard keeps it out of the way. Works for every model (no head needed).
//!
//! The guard compares what a lookup cycle gives with what the alternative gives, from the numbers of the request itself: the drafts accepted per cycle are averaged per length of the matched n-gram
//! (a longer match is a better predictor), and a lookup is used only when its average beats the break-even point against the head's own chain (about `0.85 (1 + a) - 1` accepted drafts, with `a` the
//! head's average: its cycle costs 6 ms of drafting more) or, without a head, against one plain step (0.3 drafts).
//!
//! On by default (`--no-lookup`, `FORGE_PLD=0` or `--no-speculate` turn it off); `FORGE_PLD_DRAFT` (most drafted tokens, default 11; a run starts at 7 and doubles after a cycle whose drafts were all right), `FORGE_PLD_MIN` / `FORGE_PLD_MAX` (shortest and longest matched n-gram, default 3 and 6) tune it.

/// The tokens after the most recent earlier occurrence of the last `n` tokens of `hist` (n from `max_n` down to `min_n`: the longest match wins), at most `k` of them, with the n that matched;
/// None when nothing matches.
pub fn find(hist: &[u32], min_n: usize, max_n: usize, k: usize) -> Option<(usize, Vec<u32>)> {
    let len = hist.len();
    if k == 0 || min_n == 0 {
        return None;
    }
    for n in (min_n..=max_n.max(min_n)).rev() {
        if len <= n {
            continue;
        }
        let key = &hist[len - n..];
        // an earlier start i < len - n, scanning backwards so that the most recent occurrence wins
        let mut i = len - n;
        while i > 0 {
            i -= 1;
            if hist[i] == key[0] && hist[i..i + n] == *key {
                let from = i + n;
                let end = (from + k).min(len);
                return Some((n, hist[from..end].to_vec()));
            }
        }
    }
    None
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// Longest n-gram length that has its own average.
const BUCKETS: usize = 12;
/// What a bucket is believed to give before it has been tried, and the pull towards it (a bucket that did badly is tried again after a while).
const PRIOR: f32 = 2.0;
const PULL: f32 = 0.01;

pub struct Pld {
    pub on: bool,
    min_n: usize,
    max_n: usize,
    pub max_draft: usize,
    min_draft: usize,
    /// the run proposed now: it doubles after a cycle whose drafts were all right (a long copy); shrinking it after a miss measured worse (extra rows are cheap)
    cur: usize,
    has_head: bool,
    /// drafts accepted per lookup cycle, averaged for each matched length
    by_len: [f32; BUCKETS],
    /// drafts accepted per cycle of the head's own chain (only with a head)
    head_avg: f32,
}

impl Pld {
    /// `has_head`: an MTP head drafts the other cycles; `max_rows`: the most rows one verification sweep takes.
    pub fn from_env(has_head: bool, max_rows: usize) -> Pld {
        let on = std::env::var("FORGE_PLD")
            .map(|v| v != "0" && !v.eq_ignore_ascii_case("off") && !v.is_empty())
            .unwrap_or(true)
            && max_rows >= 3;
        let min_n = env_usize("FORGE_PLD_MIN", 3).clamp(1, BUCKETS - 1);
        Pld {
            on,
            min_n,
            max_n: env_usize("FORGE_PLD_MAX", 6).clamp(min_n, BUCKETS - 1),
            max_draft: env_usize("FORGE_PLD_DRAFT", 11).clamp(1, max_rows.saturating_sub(1).max(1)),
            min_draft: 2,
            cur: 7,
            has_head,
            by_len: [PRIOR; BUCKETS],
            head_avg: 1.5,
        }
    }

    /// The accepted drafts per lookup cycle a match must promise to be worth a cycle.
    fn break_even(&self) -> f32 {
        if self.has_head {
            (0.85 * (1.0 + self.head_avg) - 1.0).max(0.3)
        } else {
            0.3
        }
    }

    /// Once per cycle: every average drifts back towards its prior, so that a length that did badly gets another chance after a while.
    pub fn tick(&mut self) {
        for v in self.by_len.iter_mut() {
            *v += (PRIOR - *v) * PULL;
        }
    }

    /// The drafts for the text `hist` (it ends with the pending token) and the matched length; `room`: how many rows after the pending token still fit in the context.
    pub fn propose(&self, hist: &[u32], room: usize) -> Option<(usize, Vec<u32>)> {
        if !self.on {
            return None;
        }
        let (n, d) = find(hist, self.min_n, self.max_n, self.cur.min(self.max_draft).min(room))?;
        (d.len() >= self.min_draft && self.by_len[n] >= self.break_even()).then_some((n, d))
    }

    /// The outcome of a lookup cycle that matched `n` tokens: `accepted` of its `proposed` drafts were right.
    pub fn record(&mut self, n: usize, accepted: usize, proposed: usize) {
        let v = &mut self.by_len[n];
        *v = 0.7 * *v + 0.3 * accepted as f32;
        if accepted >= proposed {
            self.cur = (self.cur * 2).min(self.max_draft.max(1));
        }
    }

    /// The outcome of a cycle of the head's chain: `accepted` of its drafts were right.
    pub fn record_head(&mut self, accepted: usize) {
        self.head_avg = 0.9 * self.head_avg + 0.1 * accepted as f32;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pld(has_head: bool) -> Pld {
        Pld {
            on: true,
            min_n: 2,
            max_n: 4,
            max_draft: 7,
            min_draft: 2,
            cur: 7,
            has_head,
            by_len: [PRIOR; BUCKETS],
            head_avg: 1.5,
        }
    }

    #[test]
    fn the_most_recent_longest_match_continues() {
        // ... 5 6 7 8 9 ... 1 2 3 4 | 5 6 7 8 9 (end: 5 6 7 8 matches the earlier run)
        let hist = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 5, 6, 7, 8];
        assert_eq!(find(&hist, 3, 5, 4), Some((4, vec![9, 10, 11, 5])));
        assert_eq!(find(&hist, 3, 5, 2), Some((4, vec![9, 10])));
        // the match is the text's own end: not a match
        assert!(find(&[1, 2, 3, 4], 2, 4, 3).is_none());
        // too short a history
        assert!(find(&[7], 1, 3, 3).is_none());
        assert!(find(&[], 1, 3, 3).is_none());
    }

    #[test]
    fn a_longer_context_beats_a_more_recent_shorter_one() {
        // "9 1 2 3" occurred earlier followed by 40; the more recent "8 1 2 3" shares only 3 tokens, then continues with 50
        let hist = [9, 1, 2, 3, 40, 41, 8, 1, 2, 3, 50, 51, 9, 1, 2, 3];
        // n = 4 ("9 1 2 3") is found first: the first occurrence
        assert_eq!(find(&hist, 3, 4, 2), Some((4, vec![40, 41])));
        // limited to 3-grams the most recent one wins: "1 2 3" at index 7
        assert_eq!(find(&hist, 3, 3, 2), Some((3, vec![50, 51])));
    }

    #[test]
    fn a_continuation_may_run_to_the_pending_token() {
        // "a b a b a b": the earlier "a b" is followed by the rest of the text itself
        let hist = [1, 2, 1, 2, 1, 2];
        assert_eq!(find(&hist, 2, 2, 5), Some((2, vec![1, 2])));
    }

    #[test]
    fn a_length_that_does_badly_is_dropped_and_comes_back() {
        let mut p = pld(false);
        let hist = [5, 1, 2, 9, 3, 1, 2];
        assert!(p.propose(&hist, 10).is_some());
        for _ in 0..8 {
            p.record(2, 0, 4);
        }
        assert!(p.propose(&hist, 10).is_none(), "eight misses in a row put the 2-gram below the break-even point");
        for _ in 0..200 {
            p.tick();
        }
        assert!(p.propose(&hist, 10).is_some(), "and it is tried again after a while");
        p.record(2, 6, 6);
        assert!(p.propose(&hist, 10).is_some());
    }

    #[test]
    fn with_a_head_the_bar_follows_the_head() {
        let mut p = pld(true);
        let hist = [5, 1, 2, 9, 3, 1, 2];
        // the head averages 1.5 accepted drafts per cycle: the lookup must promise 0.85 * 2.5 - 1 = 1.1; the prior of 2 does
        assert!(p.propose(&hist, 10).is_some());
        // the head gets better (code that it predicts well): 3 accepted per cycle moves the bar to 2.4, above the prior
        for _ in 0..40 {
            p.record_head(3);
        }
        assert!(p.propose(&hist, 10).is_none());
        // a lookup that has proven itself (5 accepted) beats it
        p.record(2, 7, 7);
        p.record(2, 7, 7);
        p.record(2, 7, 7);
        assert!(p.propose(&hist, 10).is_some());
    }

    #[test]
    fn the_run_doubles_after_a_full_hit_and_stays_after_a_miss() {
        let mut p = pld(false);
        p.max_draft = 15;
        p.cur = 7;
        let hist: Vec<u32> = (0..60).chain(0..20).collect(); // an earlier run of 40+ tokens follows the matched n-gram
        assert_eq!(p.propose(&hist, 100).unwrap().1.len(), 7);
        p.record(4, 7, 7);
        assert_eq!(p.propose(&hist, 100).unwrap().1.len(), 14);
        p.record(4, 14, 14);
        assert_eq!(p.propose(&hist, 100).unwrap().1.len(), 15, "capped by max_draft");
        p.record(4, 5, 15);
        assert_eq!(p.propose(&hist, 100).unwrap().1.len(), 15, "a miss does not shorten it");
        assert_eq!(p.propose(&hist, 6).unwrap().1.len(), 6, "the room caps it");
    }

    #[test]
    fn proposals_need_two_tokens() {
        let p = pld(false);
        assert_eq!(p.propose(&[5, 1, 2, 9, 3, 1, 2], 10), Some((2, vec![9, 3, 1, 2])));
        assert_eq!(p.propose(&[5, 1, 2, 9, 3, 1, 2], 3), Some((2, vec![9, 3, 1])), "the room caps it");
        // "5 5" is followed by one known token only: too little to be worth a sweep
        assert!(p.propose(&[5, 5, 5], 10).is_none());
        assert!(p.propose(&[9, 9, 1, 2], 10).is_none());
    }
}
