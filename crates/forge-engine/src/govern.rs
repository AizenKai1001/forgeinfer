//! The speed governor of the speculative decoding loop.
//!
//! Drafting pays only when verifying the drafts costs little more than a plain step, and that depends on the context and on the text. A verification sweep of four rows reads the whole KV cache of
//! every attention layer once per row, so at 64,000 positions a cycle of the head's chain costs about three plain steps and gives 2.3 tokens: on the RTX 2060 SUPER (Qwen3.5-4B) the cycles decode
//! 1.27 times as fast as plain steps at 4,000 positions, as fast at 16,000 and 24 percent slower at 64,000.
//!
//! The governor times every iteration of the loop and keeps, for each way of taking the next token or tokens (a plain step, a cycle of the head's chain, a cycle of drafts looked up in the text), the
//! tokens it gave per millisecond (moving averages); the loop takes the best one and tries the others now and then (the more often, the closer an arm is to the best), because the numbers move with the
//! context and the text. A plain step keeps the head's cache in step (one KV-only head step), so drafting can start again at any time.
//!
//! When the chain is clearly slower than plain steps (below `DROP_BELOW` of their rate after `DROP_AFTER` measurements) the governor gives it up for the rest of the request: the head's cache is no longer kept
//! in step (that costs a KV-only head step per plain token) and the chain is not tried again; the next request of the chat starts with a head that is not ready and decodes plainly too, until a new chat.
//!
//! It is only consulted from `FORGE_GOVERN_FROM` positions on (default 6,000: below that drafting won everywhere it was measured, and the loop behaves exactly as without a governor);
//! `FORGE_GOVERN=0` turns it off.

use std::time::Duration;

/// The ways of taking the next token or tokens.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Arm {
    /// one step of the main model
    Plain = 0,
    /// a cycle of the MTP head's chain of drafts, verified in one sweep
    Head = 1,
    /// a cycle of drafts looked up in the text itself, verified in one sweep
    Lookup = 2,
}

/// The averages are running means over at most this many iterations; the weight of what an arm has measured halves every `FORGET` iterations it is not used (its old numbers may be out of date), so a
/// new measurement of an arm that has been left alone for long counts a lot.
const WINDOW: u32 = 20;
const FORGET: f64 = 96.0;
/// Iterations an arm is measured before the governor trusts it (the tokens a cycle of the chain gives vary from 1 to 4, so it needs more).
fn min_tries(a: usize) -> u32 {
    if a == Arm::Head as usize {
        DROP_AFTER
    } else {
        3
    }
}
/// A chain that has been measured this often and gives less than `DROP_BELOW` of the plain steps' rate is given up.
const DROP_AFTER: u32 = 6;
const DROP_BELOW: f64 = 0.78;
/// The arm in use is left only for one that is this much faster.
const HYSTERESIS: f64 = 1.08;
/// Positions from which the governor decides (default).
pub const DEFAULT_FROM: usize = 6000;

/// Iterations between two turns of an arm that is not the best, by how close its rate is to the best arm's: a turn of a clearly slower arm costs more than one of an equal arm.
fn explore_gap(ratio: f64) -> u64 {
    if ratio >= 0.9 {
        24
    } else if ratio >= 0.8 {
        48
    } else if ratio >= 0.65 {
        96
    } else {
        192
    }
}

pub struct Governor {
    enabled: bool,
    from: usize,
    /// the last `pick` was at or beyond `from` with the governor on
    active: bool,
    /// the chain is given up for the rest of the request
    dropped: bool,
    /// moving averages of the milliseconds and the tokens of an iteration, per arm
    ms: [f64; 3],
    tok: [f64; 3],
    /// iterations measured, and the index of the last one, per arm
    n: [u32; 3],
    last: [u64; 3],
    /// iterations recorded so far
    it: u64,
    cur: usize,
    /// iterations taken per arm (for the statistics)
    pub used: [u64; 3],
}

impl Governor {
    pub fn new(enabled: bool, from: usize) -> Governor {
        Governor {
            enabled,
            from,
            active: false,
            dropped: false,
            ms: [0.0; 3],
            tok: [0.0; 3],
            n: [0; 3],
            last: [0; 3],
            it: 0,
            cur: Arm::Head as usize,
            used: [0; 3],
        }
    }

    /// `FORGE_GOVERN=0` turns the governor off, `FORGE_GOVERN_FROM=N` sets the position from which it decides.
    pub fn from_env() -> Governor {
        let enabled = std::env::var("FORGE_GOVERN").map(|v| v != "0" && !v.eq_ignore_ascii_case("off")).unwrap_or(true);
        let from = std::env::var("FORGE_GOVERN_FROM").ok().and_then(|v| v.parse().ok()).unwrap_or(DEFAULT_FROM);
        Governor::new(enabled, from)
    }

    /// Tokens per millisecond an arm has given, None while it has not been measured.
    fn rate(&self, a: usize) -> Option<f64> {
        (self.n[a] > 0 && self.ms[a] > 0.0).then(|| self.tok[a] / self.ms[a])
    }

    /// The way to take the next token(s) at position `pos`; `avail`: which arms are possible now (indexed by `Arm`; a plain step always is).
    /// Without the governor, or before its position, the choice is what the loop always did: the lookup when it offers drafts, else the head, else a plain step.
    pub fn pick(&mut self, pos: usize, avail: [bool; 3]) -> Arm {
        self.active = self.enabled && pos >= self.from;
        let avail = [avail[0], avail[1] && !self.dropped, avail[2]];
        let arm = |a: usize| match a {
            0 => Arm::Plain,
            1 => Arm::Head,
            _ => Arm::Lookup,
        };
        let default = if avail[2] {
            2
        } else if avail[1] {
            1
        } else {
            0
        };
        if !self.enabled || pos < self.from || (!avail[1] && !avail[2]) {
            return arm(default);
        }
        // an arm that has not been measured enough yet (the lookup only when it is offered)
        for a in [2usize, 1, 0] {
            if avail[a] && self.n[a] < min_tries(a) {
                return arm(a);
            }
        }
        // the best arm by tokens per millisecond, the one in use kept unless another is clearly faster
        let mut best = default;
        let mut best_rate = 0.0;
        for a in 0..3 {
            if avail[a] {
                if let Some(r) = self.rate(a) {
                    if r > best_rate {
                        best = a;
                        best_rate = r;
                    }
                }
            }
        }
        if best != self.cur && avail[self.cur] {
            if let Some(rc) = self.rate(self.cur) {
                if best_rate < rc * HYSTERESIS {
                    best = self.cur;
                    best_rate = rc;
                }
            }
        }
        self.cur = best;
        // now and then another arm gets a turn: the one that has waited longest beyond its own gap
        let mut cand = None;
        let mut longest = 0u64;
        for a in 0..3 {
            if a == best || !avail[a] {
                continue;
            }
            let ratio = self.rate(a).map(|r| r / best_rate).unwrap_or(1.0);
            let waited = self.it - self.last[a];
            if waited >= explore_gap(ratio) && waited > longest {
                longest = waited;
                cand = Some(a);
            }
        }
        match cand {
            Some(a) => arm(a),
            None => arm(best),
        }
    }

    /// A new request on the same engine: what the arms measured before is kept as a starting point (a short answer must not pay for measuring again) but counts for no more than the minimum number
    /// of measurements, so that the new text decides quickly; a chain that was given up is tried again when the head is usable (it is, after a new chat has been read).
    pub fn new_request(&mut self, head_usable: bool) {
        for a in 0..3 {
            self.n[a] = self.n[a].min(min_tries(a));
        }
        if head_usable {
            self.dropped = false;
        }
    }

    /// The chain has been given up: the head's cache need not be kept in step any more.
    pub fn head_dropped(&self) -> bool {
        self.dropped
    }

    /// The outcome of an iteration: `tokens` tokens emitted in `took`.
    pub fn record(&mut self, arm: Arm, tokens: usize, took: Duration) {
        let a = arm as usize;
        let ms = took.as_secs_f64() * 1e3;
        if self.n[a] == 0 {
            self.ms[a] = ms;
            self.tok[a] = tokens as f64;
        } else {
            let waited = (self.it - self.last[a]) as f64;
            let n_eff = self.n[a].min(WINDOW) as f64 * 0.5f64.powf(waited / FORGET);
            let alpha = 1.0 / (n_eff + 1.0);
            // (one stalled iteration must not move the average much)
            let ms = ms.min(self.ms[a] * 4.0);
            self.ms[a] += alpha * (ms - self.ms[a]);
            self.tok[a] += alpha * (tokens as f64 - self.tok[a]);
        }
        self.n[a] += 1;
        self.last[a] = self.it;
        self.it += 1;
        self.used[a] += 1;
        if self.active && !self.dropped && self.n[1] >= DROP_AFTER && self.n[0] >= 3 {
            if let (Some(rh), Some(rp)) = (self.rate(1), self.rate(0)) {
                if rh < DROP_BELOW * rp {
                    self.dropped = true;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Run `iters` iterations in which every arm gives a fixed (tokens, milliseconds); returns how often each arm was taken.
    fn run(g: &mut Governor, pos: usize, avail: [bool; 3], cost: [(usize, f64); 3], iters: usize) -> [usize; 3] {
        let mut taken = [0usize; 3];
        for _ in 0..iters {
            let a = g.pick(pos, avail);
            taken[a as usize] += 1;
            let (t, ms) = cost[a as usize];
            g.record(a, t, Duration::from_secs_f64(ms / 1e3));
        }
        taken
    }

    #[test]
    fn without_the_governor_the_loop_keeps_its_order() {
        for g in [Governor::new(false, 0), Governor::new(true, 100_000)] {
            let mut g = g;
            assert_eq!(g.pick(50_000, [true, true, true]), Arm::Lookup);
            assert_eq!(g.pick(50_000, [true, true, false]), Arm::Head);
            assert_eq!(g.pick(50_000, [true, false, false]), Arm::Plain);
        }
    }

    #[test]
    fn drafting_that_loses_at_a_long_context_gives_way_to_plain_steps() {
        // 48,000 positions: a plain step 14 ms; a cycle of the chain 32 ms for 2 tokens (0.875 of the plain rate)
        let mut g = Governor::new(true, 0);
        let taken = run(&mut g, 48_000, [true, true, false], [(1, 14.0), (2, 32.0), (0, 1.0)], 400);
        assert!(taken[0] > 340, "plain steps should dominate: {taken:?}");
        assert!(taken[1] >= 5 && taken[1] <= 25, "the chain is still tried now and then, not often: {taken:?}");
        assert!(!g.head_dropped(), "a chain at 87 percent of the plain rate is not given up");
    }

    #[test]
    fn a_chain_far_behind_is_given_up_and_stays_given_up() {
        // 200,000 positions: a plain step 30 ms; a cycle of the chain 110 ms for 2 tokens (0.6 of the plain rate)
        let mut g = Governor::new(true, 0);
        let taken = run(&mut g, 200_000, [true, true, false], [(1, 30.0), (2, 110.0), (0, 1.0)], 300);
        assert!(g.head_dropped(), "{taken:?}");
        assert_eq!(taken[1], 6, "only the first measurements: {taken:?}");
        // the lookup arm is still there
        assert_eq!(g.pick(200_000, [true, true, false]), Arm::Plain);
    }

    #[test]
    fn nothing_is_given_up_below_the_governors_position() {
        let mut g = Governor::new(true, 100_000);
        run(&mut g, 5_000, [true, true, false], [(1, 30.0), (2, 110.0), (0, 1.0)], 100);
        assert!(!g.head_dropped());
    }

    #[test]
    fn drafting_that_wins_keeps_the_chain() {
        // 4,000 positions: a plain step 10 ms; a cycle 17 ms for 2.2 tokens
        let mut g = Governor::new(true, 0);
        let taken = run(&mut g, 4_000, [true, true, false], [(1, 10.0), (2, 17.0), (0, 1.0)], 400);
        assert!(taken[1] > 330, "the chain should dominate: {taken:?}");
        assert!(taken[0] >= 8, "a plain step is still timed now and then: {taken:?}");
    }

    #[test]
    fn it_follows_the_text() {
        // the chain first loses a little (0.8 of the plain rate), then (a repetitive text) wins
        let mut g = Governor::new(true, 0);
        let a = run(&mut g, 30_000, [true, true, false], [(1, 12.0), (2, 30.0), (0, 1.0)], 300);
        assert!(a[0] > 5 * a[1], "{a:?}");
        assert!(!g.head_dropped());
        let b = run(&mut g, 30_000, [true, true, false], [(1, 12.0), (4, 30.0), (0, 1.0)], 700);
        assert!(b[1] > 300, "after the text became predictable the chain should be taken again: {b:?}");
    }

    #[test]
    fn an_arm_that_is_not_possible_is_never_taken() {
        let mut g = Governor::new(true, 0);
        let t = run(&mut g, 20_000, [true, false, true], [(1, 12.0), (3, 20.0), (6, 40.0)], 200);
        assert_eq!(t[1], 0, "{t:?}");
        // the lookup that gives 6 tokens in 40 ms beats a plain step (12 ms for 1)
        assert!(t[2] > t[0], "{t:?}");
    }

    #[test]
    fn a_second_request_starts_from_what_the_first_measured() {
        let mut g = Governor::new(true, 0);
        let first = run(&mut g, 20_000, [true, true, false], [(1, 12.0), (3, 20.0), (0, 1.0)], 60);
        assert!(first[0] >= 3 && first[1] >= 6, "the first request measures every arm: {first:?}");
        g.new_request(true);
        // no forced measuring any more: the next iterations all go to the best arm until the gap of another arm is over
        let second = run(&mut g, 20_000, [true, true, false], [(1, 12.0), (3, 20.0), (0, 1.0)], 10);
        assert_eq!(second[1], 10, "{second:?}");
    }

    #[test]
    fn a_chain_that_was_given_up_comes_back_with_a_usable_head() {
        let mut g = Governor::new(true, 0);
        run(&mut g, 200_000, [true, true, false], [(1, 30.0), (2, 110.0), (0, 1.0)], 40);
        assert!(g.head_dropped());
        g.new_request(false);
        assert!(g.head_dropped(), "the head is still not usable: the chain stays given up");
        g.new_request(true);
        assert!(!g.head_dropped());
    }

    #[test]
    fn a_stalled_iteration_does_not_flip_the_choice() {
        let mut g = Governor::new(true, 0);
        run(&mut g, 20_000, [true, true, false], [(1, 12.0), (3, 20.0), (0, 1.0)], 100);
        // one iteration of the chain that took ten times as long (the GPU was busy with something else)
        g.record(Arm::Head, 3, Duration::from_millis(200));
        let t = run(&mut g, 20_000, [true, true, false], [(1, 12.0), (3, 20.0), (0, 1.0)], 10);
        assert!(t[1] >= 8, "the chain stays the choice: {t:?}");
    }
}
