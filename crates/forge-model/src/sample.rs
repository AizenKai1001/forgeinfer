//! Token sampling on the CPU (tools/ref_sampling_numpy.py is the executable specification of the chain).
//! Chain order, as in llama.cpp's default: repetition penalty -> temperature scaling -> top-k -> top-p -> min-p -> draw. Temperature 0 (or top-k 1) is greedy.
//!
//! Speed matters here: a 152k-token vocabulary is scanned once per generated token. The chain is evaluated on the *support* only, the tokens whose probability is at least 1e-13 of the best one
//! (a few hundred to a few thousand): top-k selects within it without sorting, min-p is a threshold on the logit, and only the survivors are sorted for the top-p cumulative sum. The probabilities of
//! the dropped tail add up to far less than the rounding error of the f32 sum, so the result is the specified distribution.

pub struct SamplerConfig {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub min_p: f32,
    pub repetition_penalty: f32,
    pub seed: u64,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        SamplerConfig {
            temperature: 0.0,
            top_k: 40,
            top_p: 0.95,
            min_p: 0.05,
            repetition_penalty: 1.0,
            seed: 1,
        }
    }
}

/// xorshift64* - small, fast and reproducible (the engine uses a counter-based generator on the GPU).
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng(seed.max(1).wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    /// uniform in [0, 1)
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
}

pub struct Sampler {
    pub cfg: SamplerConfig,
    rng: Rng,
}

/// largest value (8 independent lanes so the loop vectorises)
fn max_of(l: &[f32]) -> f32 {
    let mut m = [f32::NEG_INFINITY; 8];
    let chunks = l.chunks_exact(8);
    let rem = chunks.remainder();
    for c in chunks {
        for i in 0..8 {
            m[i] = if c[i] > m[i] { c[i] } else { m[i] };
        }
    }
    let mut best = f32::NEG_INFINITY;
    for v in m.iter().chain(rem.iter()) {
        if *v > best {
            best = *v;
        }
    }
    best
}

/// descending by logit, ties broken by the lower token id (the stable order of the specification)
fn by_logit_desc(a: &(f32, u32), b: &(f32, u32)) -> std::cmp::Ordering {
    b.0.partial_cmp(&a.0)
        .unwrap_or(std::cmp::Ordering::Equal)
        .then(a.1.cmp(&b.1))
}

impl Sampler {
    pub fn new(cfg: SamplerConfig) -> Sampler {
        let rng = Rng::new(cfg.seed);
        Sampler { cfg, rng }
    }

    /// The distribution the next token is drawn from: (token, probability) of the kept candidates, most probable first, probabilities summing to 1.
    /// `l` must already carry the repetition penalty. Temperature <= 0 gives the one-hot greedy choice.
    pub fn distribution(&self, l: &[f32]) -> Vec<(u32, f32)> {
        let c = &self.cfg;
        let m = max_of(l);
        if self.is_greedy() || !m.is_finite() {
            let best = l.iter().position(|&x| x == m).unwrap_or(0);
            return vec![(best as u32, 1.0)];
        }
        // the support: logits within 30 temperatures of the best (probability ratio e^-30 = 9e-14)
        let cutoff = m - 30.0 * c.temperature;
        let mut cand: Vec<(f32, u32)> = Vec::with_capacity(2048);
        for (i, &x) in l.iter().enumerate() {
            if x >= cutoff {
                cand.push((x, i as u32));
            }
        }
        self.chain(cand, m, || {
            l.iter().position(|&x| x == m).unwrap_or(0) as u32
        })
    }

    /// Temperature <= 0 or top-k 1: the arg-max.
    pub fn is_greedy(&self) -> bool {
        self.cfg.temperature <= 0.0 || self.cfg.top_k == 1
    }

    /// How far below the best logit a token can be and still matter: 30 temperatures (probability ratio 9e-14); 0 for greedy decoding. A device that extracts the candidates
    /// {logit >= max - cutoff} hands the sampler exactly the support `distribution` works on.
    pub fn cutoff(&self) -> f32 {
        if self.is_greedy() {
            0.0
        } else {
            30.0 * self.cfg.temperature
        }
    }

    /// Draw from candidates (logit, token) that already carry the repetition penalty and hold the whole support (see `cutoff`), in any order.
    pub fn sample_candidates(&mut self, cand: Vec<(f32, u32)>) -> u32 {
        let m = cand
            .iter()
            .fold(f32::NEG_INFINITY, |a, c| if c.0 > a { c.0 } else { a });
        let best = || {
            cand.iter()
                .filter(|c| c.0 == m)
                .map(|c| c.1)
                .min()
                .unwrap_or(0)
        };
        if self.is_greedy() || !m.is_finite() || cand.is_empty() {
            return best();
        }
        let b = best();
        let dist = self.chain(cand, m, || b);
        if dist.len() == 1 {
            return dist[0].0;
        }
        let mut r = self.rng.next_f32();
        for &(i, p) in &dist {
            r -= p;
            if r <= 0.0 {
                return i;
            }
        }
        dist[dist.len() - 1].0
    }

    /// The sampler chain on the support: top-k, min-p, top-p, normalisation.
    fn chain(&self, mut cand: Vec<(f32, u32)>, m: f32, best: impl Fn() -> u32) -> Vec<(u32, f32)> {
        let c = &self.cfg;
        // top-k: the k best of the support (selection, no full sort)
        if c.top_k > 0 && cand.len() > c.top_k {
            cand.select_nth_unstable_by(c.top_k - 1, by_logit_desc);
            cand.truncate(c.top_k);
        }
        // normalisation over the top-k set (top-p works on these probabilities, as in llama.cpp)
        let inv_t = 1.0 / c.temperature;
        let z: f32 = cand.iter().map(|&(x, _)| ((x - m) * inv_t).exp()).sum();
        // min-p: p_i >= min_p * p_best  <=>  logit_i >= m + T ln(min_p); a prefix of the sorted order, so it can be applied before sorting
        if c.min_p > 0.0 {
            let floor = m + c.temperature * c.min_p.ln();
            cand.retain(|&(x, _)| x >= floor);
            if cand.is_empty() {
                // min_p > 1: only the best token passes
                return vec![(best(), 1.0)];
            }
        }
        cand.sort_unstable_by(by_logit_desc);
        let mut p: Vec<f32> = cand
            .iter()
            .map(|&(x, _)| ((x - m) * inv_t).exp() / z)
            .collect();
        // top-p: the smallest prefix whose mass reaches top_p (never empty)
        if c.top_p < 1.0 {
            let mut cum = 0.0;
            for (i, v) in p.iter().enumerate() {
                cum += v;
                if cum >= c.top_p {
                    p.truncate(i + 1);
                    break;
                }
            }
        }
        let kept = p.len();
        let zk: f32 = p.iter().sum();
        cand.iter()
            .take(kept)
            .zip(p.iter())
            .map(|(&(_, i), &v)| (i, v / zk))
            .collect()
    }

    /// `history`: recent token ids for the repetition penalty (divide positive logits, multiply negative ones, once per distinct token, as in CTRL / llama.cpp).
    pub fn sample(&mut self, logits: &[f32], history: &[u32]) -> u32 {
        let penalised;
        let l: &[f32] = if self.cfg.repetition_penalty != 1.0 && !history.is_empty() {
            let mut v = logits.to_vec();
            let mut seen: Vec<u32> = history.to_vec();
            seen.sort_unstable();
            seen.dedup();
            let pen = self.cfg.repetition_penalty;
            for t in seen {
                if let Some(x) = v.get_mut(t as usize) {
                    *x = if *x > 0.0 { *x / pen } else { *x * pen };
                }
            }
            penalised = v;
            &penalised
        } else {
            logits
        };
        let dist = self.distribution(l);
        if dist.len() == 1 {
            return dist[0].0;
        }
        let mut r = self.rng.next_f32();
        for &(i, p) in &dist {
            r -= p;
            if r <= 0.0 {
                return i;
            }
        }
        dist[dist.len() - 1].0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// the straightforward specification: full softmax-free chain on the sorted vocabulary
    fn reference(l: &[f32], c: &SamplerConfig) -> Vec<(u32, f32)> {
        if c.temperature <= 0.0 || c.top_k == 1 {
            let mut best = 0;
            for (i, &x) in l.iter().enumerate() {
                if x > l[best] {
                    best = i;
                }
            }
            return vec![(best as u32, 1.0)];
        }
        let mut idx: Vec<u32> = (0..l.len() as u32).collect();
        idx.sort_by(|&a, &b| {
            l[b as usize]
                .partial_cmp(&l[a as usize])
                .unwrap()
                .then(a.cmp(&b))
        });
        let k = if c.top_k == 0 {
            l.len()
        } else {
            c.top_k.min(l.len())
        };
        idx.truncate(k);
        let mx = l[idx[0] as usize];
        let mut p: Vec<f64> = idx
            .iter()
            .map(|&i| (((l[i as usize] - mx) / c.temperature) as f64).exp())
            .collect();
        let z: f64 = p.iter().sum();
        p.iter_mut().for_each(|v| *v /= z);
        let mut keep = p.len();
        if c.top_p < 1.0 {
            let mut cum = 0.0;
            for (i, v) in p.iter().enumerate() {
                cum += v;
                if cum >= c.top_p as f64 {
                    keep = i + 1;
                    break;
                }
            }
        }
        let floor = c.min_p as f64 * p[0];
        while keep > 1 && p[keep - 1] < floor {
            keep -= 1;
        }
        let zk: f64 = p[..keep].iter().sum();
        (0..keep).map(|i| (idx[i], (p[i] / zk) as f32)).collect()
    }

    fn fake_logits(n: usize, seed: u64) -> Vec<f32> {
        let mut r = Rng::new(seed);
        // a language-model-like spread: most tokens far below, a handful near the top
        (0..n)
            .map(|_| {
                (r.next_f32() - 0.5) * 8.0
                    + if r.next_f32() < 0.001 {
                        14.0 * r.next_f32()
                    } else {
                        0.0
                    }
            })
            .collect()
    }

    #[test]
    fn distribution_matches_the_specification() {
        for (si, &(t, k, p, mp)) in [
            (0.8, 40, 0.95, 0.0),
            (0.7, 0, 0.95, 0.05),
            (1.0, 0, 1.0, 0.0),
            (1.0, 0, 1.0, 0.02),
            (0.6, 20, 0.9, 0.1),
            (1.5, 100, 0.8, 0.0),
            (0.3, 5, 1.0, 0.5),
        ]
        .iter()
        .enumerate()
        {
            let l = fake_logits(5000, 100 + si as u64);
            let cfg = SamplerConfig {
                temperature: t,
                top_k: k,
                top_p: p,
                min_p: mp,
                repetition_penalty: 1.0,
                seed: 1,
            };
            let want = reference(&l, &cfg);
            let got = Sampler::new(SamplerConfig { ..cfg }).distribution(&l);
            assert_eq!(
                got.len(),
                want.len(),
                "case {si} ({t} {k} {p} {mp}): kept {} tokens, specification keeps {}",
                got.len(),
                want.len()
            );
            for (g, w) in got.iter().zip(want.iter()) {
                assert_eq!(g.0, w.0, "case {si}: token order differs");
                assert!(
                    (g.1 - w.1).abs() < 2e-5,
                    "case {si}: probability {} vs {}",
                    g.1,
                    w.1
                );
            }
        }
    }

    #[test]
    fn candidates_give_the_same_draws_as_the_full_logits() {
        let l = fake_logits(5000, 11);
        let m = l.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        for (t, k, p, mp) in [
            (0.8f32, 40usize, 0.95f32, 0.0f32),
            (0.7, 0, 0.95, 0.05),
            (1.0, 0, 1.0, 0.0),
            (0.6, 20, 0.9, 0.1),
            (0.0, 40, 0.95, 0.0),
            (0.9, 1, 1.0, 0.0),
        ] {
            let mk = || {
                Sampler::new(SamplerConfig {
                    temperature: t,
                    top_k: k,
                    top_p: p,
                    min_p: mp,
                    repetition_penalty: 1.0,
                    seed: 3,
                })
            };
            let (mut a, mut b) = (mk(), mk());
            for round in 0..40 {
                let full = a.sample(&l, &[]);
                let cutoff = b.cutoff();
                let mut cand: Vec<(f32, u32)> = l
                    .iter()
                    .enumerate()
                    .filter(|(_, &x)| x >= m - cutoff)
                    .map(|(i, &x)| (x, i as u32))
                    .collect();
                if round % 2 == 1 {
                    cand.reverse(); // the device appends in no particular order
                }
                assert_eq!(
                    full,
                    b.sample_candidates(cand),
                    "t {t} k {k} p {p} min_p {mp} round {round}"
                );
            }
        }
    }

    #[test]
    fn repetition_penalty_applies_once_per_distinct_token() {
        // two equal best tokens; token 0 is in the history three times: with the penalty applied once (2.0) it falls below token 1, applied three times it would too, so use a mild penalty
        let logits = vec![2.0f32, 1.7, 0.0, -1.0];
        let mut s = Sampler::new(SamplerConfig {
            temperature: 0.0,
            repetition_penalty: 1.1,
            ..Default::default()
        });
        assert_eq!(
            s.sample(&logits, &[0, 0, 0]),
            0,
            "2.0 / 1.1 = 1.82 still beats 1.7; applied three times it would not"
        );
        let mut s = Sampler::new(SamplerConfig {
            temperature: 0.0,
            repetition_penalty: 1.3,
            ..Default::default()
        });
        assert_eq!(s.sample(&logits, &[0]), 1);
        // negative logits are multiplied (pushed down further)
        let mut s = Sampler::new(SamplerConfig {
            temperature: 0.0,
            repetition_penalty: 2.0,
            ..Default::default()
        });
        assert_eq!(s.sample(&[-1.0, -1.5], &[0]), 1);
    }

    #[test]
    fn greedy_and_seeded_draws_are_reproducible() {
        let l = fake_logits(2000, 7);
        let mut a = Sampler::new(SamplerConfig {
            temperature: 0.0,
            ..Default::default()
        });
        let g = a.sample(&l, &[]);
        assert_eq!(
            g,
            l.iter()
                .enumerate()
                .fold(0, |b, (i, &x)| if x > l[b] { i } else { b }) as u32
        );
        let draw = |seed| {
            let mut s = Sampler::new(SamplerConfig {
                temperature: 0.9,
                top_k: 50,
                top_p: 0.95,
                min_p: 0.02,
                repetition_penalty: 1.0,
                seed,
            });
            (0..20).map(|_| s.sample(&l, &[])).collect::<Vec<_>>()
        };
        assert_eq!(draw(5), draw(5));
        assert_ne!(draw(5), draw(6));
    }
}
