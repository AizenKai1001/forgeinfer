//! A CPU stand-in for forge_train::Trainer with the API surface that eval_lambada.rs uses. The "model" is an oracle that looks at the true next token (the full `ids` are passed to logits_at / token_nll):
//! logit +GAP for the true next token, 0 elsewhere (FAKE_MODE=wrong: +GAP for a different token). Both entry points use the same distribution, so the self-check of eval_lambada must be 0.
use std::path::Path;

pub const VOCAB: usize = 32768;
pub const GAP: f32 = 10.0;

#[derive(Clone, Debug)]
pub struct TrainConfig {
    pub seq_len: usize,
    pub batch: usize,
    pub ce_chunk: usize,
    pub vocab: usize,
}

pub fn config_from_checkpoint(_ckpt: &Path) -> Result<TrainConfig, String> {
    Ok(TrainConfig { seq_len: 2048, batch: 1, ce_chunk: 2048, vocab: VOCAB })
}

pub struct Trainer {
    cfg: TrainConfig,
    pub calls_logits: usize,
    pub calls_nll: usize,
}

fn wrong() -> bool {
    std::env::var("FAKE_MODE").map(|v| v == "wrong").unwrap_or(false)
}

impl Trainer {
    pub fn new_inference(cfg: TrainConfig, _dev: usize) -> Result<Trainer, String> {
        assert!(cfg.batch == 1, "the stub supports batch 1 only");
        Ok(Trainer { cfg, calls_logits: 0, calls_nll: 0 })
    }
    pub fn load_checkpoint(&mut self, _p: &Path) -> Result<f32, String> {
        Ok(0.0)
    }
    /// logits that follow ids[..len]; the true next token is ids[len] when it exists
    pub fn logits_at(&mut self, ids: &[u32], len: usize) -> Result<Vec<f32>, String> {
        assert!(len >= 1 && len <= self.cfg.seq_len && ids.len() >= len, "logits_at contract violated: len {len}, ids {}, seq {}", ids.len(), self.cfg.seq_len);
        self.calls_logits += 1;
        let mut z = vec![0f32; self.cfg.vocab];
        if len < ids.len() {
            let t = ids[len] as usize;
            let favoured = if wrong() { (t + 1) % self.cfg.vocab } else { t };
            z[favoured] = GAP;
        }
        Ok(z)
    }
    /// per-position negative log-likelihood of the next token for a row of seq_len + 1 tokens (batch 1): position j is -log p(tokens[j + 1] | tokens[..= j])
    pub fn token_nll(&mut self, tokens: &[u32]) -> Result<Vec<f32>, String> {
        assert_eq!(tokens.len(), self.cfg.batch * (self.cfg.seq_len + 1), "token_nll contract violated");
        self.calls_nll += 1;
        let mut out = Vec::with_capacity(self.cfg.seq_len);
        for j in 0..self.cfg.seq_len {
            let z = self.logits_at(tokens, j + 1)?;
            let mx = z.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let sum: f64 = z.iter().map(|&l| ((l - mx) as f64).exp()).sum();
            out.push((-((z[tokens[j + 1] as usize] - mx) as f64 - sum.ln())) as f32);
        }
        Ok(out)
    }
}
