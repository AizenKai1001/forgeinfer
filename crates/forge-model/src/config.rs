//! Model hyper-parameters, read from GGUF metadata or a Hugging Face config.json.

use forge_core::{Error, Json, Result};
use forge_gguf::Gguf;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Arch {
    /// Llama family (SmolLM2, Llama-3 ...): RMSNorm, GQA attention with RoPE, SwiGLU.
    Llama,
    /// Qwen2 / Qwen2.5: as Llama (rotate-half RoPE, no QK-norm) plus a bias on the query, key and value projections.
    Qwen2,
    /// Qwen3: as Llama plus RMSNorm on every query and key head before RoPE.
    Qwen3,
    /// Qwen3.5 (dense text decoder): three Gated DeltaNet layers (a fixed-size recurrent state) for every gated full-attention layer, partial RoPE, head width 256.
    Qwen35,
}

impl Arch {
    /// RMSNorm on every query and key head before RoPE (Qwen3 and Qwen3.5)
    pub fn has_qk_norm(self) -> bool {
        matches!(self, Arch::Qwen3 | Arch::Qwen35)
    }
}

/// The Gated DeltaNet (linear attention) layers of a hybrid model.
#[derive(Clone, Debug, PartialEq)]
pub struct DeltaCfg {
    /// key (and query) heads
    pub n_k: usize,
    /// value heads (a multiple of `n_k`)
    pub n_v: usize,
    pub head_k: usize,
    pub head_v: usize,
    /// causal depthwise convolution width
    pub conv: usize,
    /// the order of the value heads: Hugging Face groups them (value head h pairs with key head h / (n_v / n_k)); llama.cpp's converter stores models with n_v != n_k in a tiled order (key head h mod n_k)
    pub tiled_v: bool,
}

impl DeltaCfg {
    pub fn key_dim(&self) -> usize {
        self.n_k * self.head_k
    }
    pub fn value_dim(&self) -> usize {
        self.n_v * self.head_v
    }
    /// channels of the convolution: q, k and v
    pub fn conv_dim(&self) -> usize {
        2 * self.key_dim() + self.value_dim()
    }
    /// the key head that value head `h` pairs with
    pub fn key_head(&self, h: usize) -> usize {
        if self.tiled_v {
            h % self.n_k
        } else {
            h / (self.n_v / self.n_k)
        }
    }
}

#[derive(Clone, Debug)]
pub struct Config {
    pub arch: Arch,
    pub n_layer: usize,
    pub n_embd: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub head_dim: usize,
    pub n_ff: usize,
    pub vocab: usize,
    pub rms_eps: f32,
    pub rope_theta: f32,
    pub max_ctx: usize,
    pub qk_norm: bool,
    /// the query, key and value projections carry a bias (Qwen2)
    pub qkv_bias: bool,
    pub tie_embeddings: bool,
    /// head dimensions that rotate (the first `rotary_dim`, in pairs (i, i + rotary_dim/2)); equals `head_dim` unless the model uses partial RoPE
    pub rotary_dim: usize,
    /// the q projection has a gate half per head (output multiplied by sigmoid(gate) before the output projection)
    pub attn_gate: bool,
    /// 0: every layer is attention; n > 0: layer i is attention iff (i + 1) % n == 0, all others are Gated DeltaNet
    pub full_attn_interval: usize,
    pub delta: Option<DeltaCfg>,
}

impl Config {
    pub fn q_dim(&self) -> usize {
        self.n_head * self.head_dim
    }
    pub fn kv_dim(&self) -> usize {
        self.n_head_kv * self.head_dim
    }
    /// rows of the q projection (twice q_dim when each head carries a gate)
    pub fn q_rows(&self) -> usize {
        self.q_dim() * if self.attn_gate { 2 } else { 1 }
    }
    pub fn is_delta(&self, layer: usize) -> bool {
        self.full_attn_interval > 0 && !(layer + 1).is_multiple_of(self.full_attn_interval)
    }
    pub fn n_delta_layers(&self) -> usize {
        (0..self.n_layer).filter(|&l| self.is_delta(l)).count()
    }

    /// Parameters per layer that are matrices, plus the embedding table (a quick size figure).
    pub fn approx_params(&self) -> usize {
        let mlp = 3 * self.n_embd * self.n_ff;
        let attn = self.n_embd * (self.q_rows() + 2 * self.kv_dim()) + self.q_dim() * self.n_embd;
        let delta = self
            .delta
            .as_ref()
            .map(|d| {
                self.n_embd * (d.conv_dim() + d.value_dim() + 2 * d.n_v)
                    + d.value_dim() * self.n_embd
            })
            .unwrap_or(0);
        let mut total = self.vocab * self.n_embd * if self.tie_embeddings { 1 } else { 2 };
        for l in 0..self.n_layer {
            total += mlp + if self.is_delta(l) { delta } else { attn };
        }
        total
    }

    pub fn from_gguf(g: &Gguf) -> Result<Config> {
        let a = g
            .architecture()
            .ok_or_else(|| Error::format("GGUF without general.architecture"))?;
        let arch = match a {
            "llama" => Arch::Llama,
            "qwen2" => Arch::Qwen2,
            "qwen3" => Arch::Qwen3,
            "qwen35" => Arch::Qwen35,
            other => return Err(Error::unsupported(format!("model architecture {other:?}"))),
        };
        let u = |k: &str| {
            g.get_u64(&format!("{a}.{k}"))
                .ok_or_else(|| Error::format(format!("GGUF key {a}.{k} missing")))
        };
        if let Some(t) = g.get_str(&format!("{a}.rope.scaling.type")).filter(|t| *t != "none") {
            eprintln!("[model] note: the file asks for {t} rope scaling, which is not implemented: positions use plain RoPE");
        }
        let n_embd = u("embedding_length")? as usize;
        let n_head = u("attention.head_count")? as usize;
        if n_head == 0 {
            return Err(Error::format("GGUF: attention.head_count is 0"));
        }
        let n_head_kv = g
            .get_u64(&format!("{a}.attention.head_count_kv"))
            .unwrap_or(n_head as u64) as usize;
        let head_dim = g
            .get_u64(&format!("{a}.attention.key_length"))
            .map(|v| v as usize)
            .unwrap_or(n_embd / n_head);
        let emb = g
            .tensor_info("token_embd.weight")
            .ok_or_else(|| Error::format("GGUF without token_embd.weight"))?;
        let vocab = emb.dims.get(1).copied().unwrap_or(0) as usize;
        let mut n_layer = u("block_count")? as usize;
        if n_layer == 0 {
            return Err(Error::format("GGUF: block_count is 0"));
        }
        let (mut rotary_dim, mut attn_gate, mut interval, mut delta) =
            (head_dim, false, 0usize, None);
        if arch == Arch::Qwen35 {
            // multi-token-prediction layers are appended after the trunk: not used
            let mtp = g.get_u64(&format!("{a}.nextn_predict_layers")).unwrap_or(0) as usize;
            n_layer = n_layer
                .checked_sub(mtp)
                .filter(|&n| n > 0)
                .ok_or_else(|| Error::format("GGUF: nextn_predict_layers exceeds block_count"))?;
            rotary_dim = u("rope.dimension_count")? as usize;
            attn_gate = true;
            interval = g
                .get_u64(&format!("{a}.full_attention_interval"))
                .unwrap_or(4) as usize;
            let n_k = u("ssm.group_count")? as usize;
            let n_v = u("ssm.time_step_rank")? as usize;
            let head_k = u("ssm.state_size")? as usize;
            let inner = u("ssm.inner_size")? as usize;
            if n_v == 0 {
                return Err(Error::format("GGUF: ssm.time_step_rank is 0"));
            }
            delta = Some(DeltaCfg {
                n_k,
                n_v,
                head_k,
                head_v: inner / n_v,
                conv: u("ssm.conv_kernel")? as usize,
                tiled_v: n_v != n_k,
            });
        }
        Ok(Config {
            arch,
            n_layer,
            n_embd,
            n_head,
            n_head_kv,
            head_dim,
            n_ff: u("feed_forward_length")? as usize,
            vocab,
            rms_eps: g
                .get_f64(&format!("{a}.attention.layer_norm_rms_epsilon"))
                .unwrap_or(1e-5) as f32,
            rope_theta: g.get_f64(&format!("{a}.rope.freq_base")).unwrap_or(10000.0) as f32,
            max_ctx: g.get_u64(&format!("{a}.context_length")).unwrap_or(4096) as usize,
            qk_norm: arch.has_qk_norm(),
            qkv_bias: arch == Arch::Qwen2,
            tie_embeddings: g.tensor_info("output.weight").is_none(),
            rotary_dim,
            attn_gate,
            full_attn_interval: interval,
            delta,
        })
    }

    pub fn from_hf_json(c: &Json) -> Result<Config> {
        let mt = c["model_type"].as_str().unwrap_or("");
        let arch = match mt {
            "llama" => Arch::Llama,
            "qwen2" => Arch::Qwen2,
            "qwen3" => Arch::Qwen3,
            "qwen3_5" | "qwen3_5_text" => Arch::Qwen35,
            other => {
                return Err(Error::unsupported(format!(
                    "Hugging Face model_type {other:?}"
                )))
            }
        };
        if arch == Arch::Qwen2 && c["use_sliding_window"].as_bool().unwrap_or(false) {
            return Err(Error::unsupported("Qwen2 with sliding-window attention (use_sliding_window)"));
        }
        if !c["rope_scaling"].is_null() {
            eprintln!("[model] note: config.json has rope_scaling, which is not implemented: positions use plain RoPE");
        }
        // the multimodal Qwen3.5 repositories nest the text model
        let t = if arch == Arch::Qwen35 && !c["text_config"].is_null() {
            &c["text_config"]
        } else {
            c
        };
        let g = |k: &str| {
            t[k].as_usize()
                .ok_or_else(|| Error::format(format!("config.json: {k} missing")))
        };
        let n_embd = g("hidden_size")?;
        let n_head = g("num_attention_heads")?;
        if n_head == 0 {
            return Err(Error::format("config.json: num_attention_heads is 0"));
        }
        let head_dim = t["head_dim"].as_usize().unwrap_or(n_embd / n_head);
        let (mut rope_theta, mut rotary_dim, mut attn_gate, mut interval, mut delta) = (
            t["rope_theta"].as_f64().unwrap_or(10000.0),
            head_dim,
            false,
            0usize,
            None,
        );
        if arch == Arch::Qwen35 {
            let rp = &t["rope_parameters"];
            rope_theta = rp["rope_theta"].as_f64().unwrap_or(1e7);
            rotary_dim = (head_dim as f64 * rp["partial_rotary_factor"].as_f64().unwrap_or(0.25))
                .round() as usize;
            attn_gate = t["attn_output_gate"].as_bool().unwrap_or(true);
            interval = t["full_attention_interval"].as_usize().unwrap_or(4);
            let (n_k, n_v) = (g("linear_num_key_heads")?, g("linear_num_value_heads")?);
            delta = Some(DeltaCfg {
                n_k,
                n_v,
                head_k: g("linear_key_head_dim")?,
                head_v: g("linear_value_head_dim")?,
                conv: g("linear_conv_kernel_dim")?,
                tiled_v: false,
            });
        }
        Ok(Config {
            arch,
            n_layer: g("num_hidden_layers")?,
            n_embd,
            n_head,
            n_head_kv: t["num_key_value_heads"].as_usize().unwrap_or(n_head),
            head_dim,
            n_ff: g("intermediate_size")?,
            vocab: g("vocab_size")?,
            rms_eps: t["rms_norm_eps"].as_f64().unwrap_or(1e-5) as f32,
            rope_theta: rope_theta as f32,
            max_ctx: t["max_position_embeddings"].as_usize().unwrap_or(4096),
            qk_norm: arch.has_qk_norm(),
            qkv_bias: arch == Arch::Qwen2,
            tie_embeddings: t["tie_word_embeddings"]
                .as_bool()
                .or_else(|| c["tie_word_embeddings"].as_bool())
                .unwrap_or(false),
            rotary_dim,
            attn_gate,
            full_attn_interval: interval,
            delta,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GOOD: &str = r#"{"model_type":"llama","hidden_size":64,"num_attention_heads":4,"num_hidden_layers":2,"intermediate_size":128,"vocab_size":100}"#;

    #[test]
    fn a_valid_hf_config_parses() {
        let c = Config::from_hf_json(&Json::parse(GOOD).unwrap()).unwrap();
        assert_eq!((c.n_head, c.head_dim), (4, 16));
    }

    #[test]
    fn zero_attention_heads_is_an_error_not_a_division_by_zero() {
        let bad = GOOD.replace("\"num_attention_heads\":4", "\"num_attention_heads\":0");
        assert!(Config::from_hf_json(&Json::parse(&bad).unwrap()).is_err());
    }
}
