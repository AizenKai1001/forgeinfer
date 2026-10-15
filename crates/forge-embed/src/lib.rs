//! Embedding models on the CPU: the NomicBERT encoder behind nomic-embed-text-v1.5 (and its relatives) and the plain BERT encoder behind bge, MiniLM, e5 and gte, read from a GGUF file. The tokenizer is
//! BERT's WordPiece (`forge_tok::WordPiece`), the forward pass is f32 over the dequantized weights, the vector is the mean of the last layer's token vectors (or the `[CLS]` vector, as the file says),
//! normalised to length one.
//!
//! The architecture of `nomic-bert` (llama.cpp's name, Hugging Face's `NomicBertModel`), per token:
//!   x = LayerNorm(token_embd[id] + token_types[0])                                        (no position embedding: the positions enter through the rotary embedding)
//!   per layer:  q, k, v = split(Wqkv x)  (the rows of the fused matrix are q, then k, then v) ; rotary embedding (NeoX layout: dimension i is rotated with i + d/2, base 1000) on q and k;
//!               a = softmax(q k^T / sqrt(d)) v over ALL tokens of the text (no causal mask) ; x = LayerNorm1(x + Wo a)
//!               x = LayerNorm2(x + Wdown (silu(Wgate x) * Wup x))
//! The architecture of `bert` (Hugging Face's `BertModel`): the same shape with learned positions, biases on every matrix and a GELU feed-forward layer:
//!   x = LayerNorm(token_embd[id] + position_embd[position] + token_types[0])
//!   per layer:  q, k, v = Wq x + bq, Wk x + bk, Wv x + bv (three matrices, kept as one fused q|k|v matrix here) ; the same attention ; x = LayerNorm1(x + Wo a + bo)
//!               x = LayerNorm2(x + Wdown gelu(Wup x + bup) + bdown)                    (gelu(x) = x * Phi(x), the exact one with erf)
//! The numbers are written to match tools/ref_nomic_bert.py (numpy), tools/ref_hf_bert.py (Hugging Face `transformers`) and llama.cpp's embeddings of the same file (cosine above 0.999 for the quantized weights).

mod simd;

use forge_core::{Error, Result};
use forge_gguf::Gguf;
use forge_tok::WordPiece;
use std::path::Path;

#[derive(Clone, Debug)]
pub struct Config {
    pub arch: Arch,
    pub n_layer: usize,
    pub n_embd: usize,
    pub n_head: usize,
    pub n_ff: usize,
    /// the base of the rotary embedding (`Arch::NomicBert` only)
    pub rope_base: f32,
    pub eps: f32,
    /// the longest text in tokens
    pub ctx: usize,
    /// the vector is the mean of the token vectors (else the first token's)
    pub mean_pool: bool,
    pub vocab: usize,
}

/// The two encoder families this crate runs (the `general.architecture` of the GGUF file): the weights and the forward pass differ.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Arch {
    /// `nomic-bert` (nomic-embed-text): rotary positions, one fused q|k|v matrix, a SwiGLU feed-forward layer, no biases.
    NomicBert,
    /// `bert` (bge, MiniLM, e5, gte, BERT itself): learned positions, q, k and v with biases, a bias on every other matrix, a GELU feed-forward layer.
    Bert,
}

impl Arch {
    pub fn gguf_name(self) -> &'static str {
        match self {
            Arch::NomicBert => "nomic-bert",
            Arch::Bert => "bert",
        }
    }

    pub fn from_gguf_name(name: &str) -> Option<Arch> {
        match name {
            "nomic-bert" => Some(Arch::NomicBert),
            "bert" => Some(Arch::Bert),
            _ => None,
        }
    }
}

/// A LayerNorm's scale and shift.
pub struct Norm {
    pub w: Vec<f32>,
    pub b: Vec<f32>,
}

/// The weights of one encoder layer, every matrix row-major [out][in] (a GPU copy reads them from here). The biases are empty for `Arch::NomicBert`, `gate` for `Arch::Bert`.
pub struct Layer {
    /// fused q, k and v projections: 3 x n_embd rows (a `bert` file keeps them as three matrices: they are joined at load time)
    pub qkv: Vec<f32>,
    /// 3 x n_embd values
    pub qkv_bias: Vec<f32>,
    pub out: Vec<f32>,
    pub out_bias: Vec<f32>,
    pub up: Vec<f32>,
    pub up_bias: Vec<f32>,
    pub gate: Vec<f32>,
    pub down: Vec<f32>,
    pub down_bias: Vec<f32>,
    /// after the attention
    pub norm1: Norm,
    /// after the feed-forward layer
    pub norm2: Norm,
}

/// What an embedding backend (the CPU `Embedder`, a GPU copy of it) offers the server.
pub trait Encode: Send + Sync {
    /// The embedding of each text, in the order of the texts. An error is a failure of the device, never of the text (a text longer than the context is cut).
    fn embed_many(&self, texts: &[&str]) -> std::result::Result<Vec<Embedding>, String>;
    fn config(&self) -> &Config;
    fn name(&self) -> &str;
    /// where it runs, for the log
    fn backend(&self) -> String;
    fn dim(&self) -> usize {
        self.config().n_embd
    }
}

/// The rotary tables of a text of n tokens: cos and sin of position * base^(-2 i / head_dim), [n][head_dim / 2] each.
pub fn rope_tables(n: usize, head_dim: usize, base: f32) -> (Vec<f32>, Vec<f32>) {
    let half = head_dim / 2;
    let mut cos = vec![0f32; n * half];
    let mut sin = vec![0f32; n * half];
    for i in 0..half {
        let inv_freq = (base as f64).powf(-2.0 * i as f64 / head_dim as f64);
        for t in 0..n {
            let a = t as f64 * inv_freq;
            cos[t * half + i] = a.cos() as f32;
            sin[t * half + i] = a.sin() as f32;
        }
    }
    (cos, sin)
}

/// A loaded model taken apart: a GPU backend uploads the weights and keeps the tokenizer.
pub struct Parts {
    pub cfg: Config,
    pub name: String,
    pub tok: WordPiece,
    pub tok_emb: Vec<f32>,
    /// the row of token type 0
    pub type_emb: Vec<f32>,
    /// the learned position embeddings, [ctx][n_embd] (`Arch::Bert`; empty for `Arch::NomicBert`, whose positions are rotary)
    pub pos_emb: Vec<f32>,
    pub emb_norm: Norm,
    pub layers: Vec<Layer>,
}

pub struct Embedder {
    cfg: Config,
    name: String,
    tok: WordPiece,
    tok_emb: Vec<f32>,
    type_emb: Vec<f32>,
    pos_emb: Vec<f32>,
    emb_norm: Norm,
    layers: Vec<Layer>,
    threads: usize,
    avx2: bool,
}

/// One embedding and what it cost.
#[derive(Clone, Debug)]
pub struct Embedding {
    /// length one
    pub vector: Vec<f32>,
    /// tokens of the text, with `[CLS]` and `[SEP]`
    pub tokens: usize,
}

/// Is this GGUF file an embedding model this crate can run (a `nomic-bert` or a `bert` encoder with a WordPiece vocabulary)? A BERT-shaped model with another tokenizer (bge-m3, multilingual-e5: SentencePiece) is not.
pub fn is_embedding_gguf(path: impl AsRef<Path>) -> bool {
    Gguf::open(path).ok().map(|g| g.architecture().and_then(Arch::from_gguf_name).is_some() && g.get_str("tokenizer.ggml.model") == Some("bert")).unwrap_or(false)
}

fn layer_norm_rows(x: &mut [f32], n: usize, e: usize, norm: &Norm, eps: f32) {
    for t in 0..n {
        let row = &mut x[t * e..(t + 1) * e];
        let mean = row.iter().sum::<f32>() / e as f32;
        let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / e as f32;
        let inv = 1.0 / (var + eps).sqrt();
        for i in 0..e {
            row[i] = (row[i] - mean) * inv * norm.w[i] + norm.b[i];
        }
    }
}

/// x[t][i] += bias[i] for every row t of `cols` values.
fn add_bias(x: &mut [f32], cols: usize, bias: &[f32]) {
    debug_assert_eq!(bias.len(), cols);
    for row in x.chunks_mut(cols) {
        for (v, b) in row.iter_mut().zip(bias) {
            *v += b;
        }
    }
}

/// The Taylor coefficients 1 / (n! (2n + 1)) of erf(x) = 2 / sqrt(pi) * x * sum_n c_n (-x^2)^n.
fn erf_coefficients() -> &'static [f64; 41] {
    static C: std::sync::OnceLock<[f64; 41]> = std::sync::OnceLock::new();
    C.get_or_init(|| {
        let mut c = [0f64; 41];
        let mut factorial = 1f64;
        for (n, v) in c.iter_mut().enumerate() {
            if n > 0 {
                factorial *= n as f64;
            }
            *v = 1.0 / (factorial * (2 * n + 1) as f64);
        }
        c
    })
}

/// erfc(x) for x >= 0, to full f64 precision (relative error below 1e-13): the Taylor series of erf below 2 (41 terms, the remainder is 1e-25), above 2 Laplace's continued fraction
/// erfc(x) = exp(-x^2) / sqrt(pi) / (x + (1/2) / (x + (2/2) / (x + (3/2) / (x + ...)))).
fn erfc_positive(x: f64) -> f64 {
    if x < 2.0 {
        let y = -x * x;
        let sum = erf_coefficients().iter().rev().fold(0.0, |acc, &c| acc * y + c);
        1.0 - 2.0 / std::f64::consts::PI.sqrt() * x * sum
    } else if x > 27.0 {
        // exp(-729) is below the smallest f64
        0.0
    } else {
        let mut f = x;
        for k in (1..=60).rev() {
            f = x + k as f64 * 0.5 / f;
        }
        (-x * x).exp() / (std::f64::consts::PI.sqrt() * f)
    }
}

/// The error function.
pub fn erf(x: f32) -> f32 {
    let e = 1.0 - erfc_positive((x as f64).abs());
    (if x < 0.0 { -e } else { e }) as f32
}

/// BERT's activation, x * Phi(x) = x / 2 * erfc(-x / sqrt(2)): the exact one of Hugging Face's `gelu` (not the tanh approximation of llama.cpp's `ggml_gelu`), computed through erfc so that the
/// small values on the left of zero keep their digits.
pub fn gelu(x: f32) -> f32 {
    let z = -(x as f64) * std::f64::consts::FRAC_1_SQRT_2;
    let c = if z >= 0.0 { erfc_positive(z) } else { 2.0 - erfc_positive(-z) };
    (0.5 * x as f64 * c) as f32
}

/// What a `nomic-bert` or `bert` GGUF file says about the model, without its weights.
#[derive(Clone, Debug)]
pub struct Info {
    pub cfg: Config,
    pub name: String,
}

/// Read the metadata of an embedding GGUF file (a few milliseconds: no tensor is read): for a server that loads the weights only when the first text arrives.
pub fn read_info(path: impl AsRef<Path>) -> Result<Info> {
    let g = Gguf::open(path.as_ref())?;
    info_of(&g).map(|(info, _)| info)
}

fn info_of(g: &Gguf) -> Result<(Info, WordPiece)> {
    let name = g.architecture().ok_or_else(|| Error::msg("the GGUF file has no architecture"))?.to_string();
    let arch = Arch::from_gguf_name(&name).ok_or_else(|| Error::msg(format!("the embedding architecture {name:?} is not supported (nomic-bert and bert are)")))?;
    let key = |k: &str| format!("{name}.{k}");
    let uint = |k: &str| -> Result<usize> { g.get_u64(&key(k)).map(|v| v as usize).ok_or_else(|| Error::msg(format!("{} is missing", key(k)))) };
    let tok = WordPiece::from_gguf(g)?;
    let mut cfg = Config {
        arch,
        n_layer: uint("block_count")?,
        n_embd: uint("embedding_length")?,
        n_head: uint("attention.head_count")?,
        n_ff: uint("feed_forward_length")?,
        rope_base: g.get_f64(&key("rope.freq_base")).unwrap_or(1000.0) as f32,
        eps: g.get_f64(&key("attention.layer_norm_epsilon")).unwrap_or(1e-12) as f32,
        ctx: uint("context_length").unwrap_or(if arch == Arch::Bert { 512 } else { 2048 }),
        mean_pool: g.get_u64(&key("pooling_type")).unwrap_or(1) != 2,
        vocab: tok.vocab_size(),
    };
    if cfg.n_head == 0 || cfg.n_embd % cfg.n_head != 0 || (arch == Arch::NomicBert && (cfg.n_embd / cfg.n_head) % 2 != 0) {
        return Err(Error::msg("the head width must divide the embedding width (and be even for rotary positions)"));
    }
    if arch == Arch::Bert {
        // a text cannot be longer than the table of learned positions (a file whose metadata promises more than its table has is cut to the table)
        let positions = g.tensor_info("position_embd.weight").map(|t| t.shape()).filter(|s| s.len() == 2 && s[1] == cfg.n_embd).map(|s| s[0]).ok_or_else(|| Error::msg("position_embd.weight is missing or has the wrong shape"))?;
        cfg.ctx = cfg.ctx.min(positions);
    }
    let name = g.get_str("general.name").unwrap_or("embedding model").to_string();
    Ok((Info { cfg, name }, tok))
}

/// The three matrices (or biases) of q, k and v of a `bert` layer as one fused matrix: the rows of q, then of k, then of v.
fn fuse(g: &Gguf, prefix: &str, what: &str) -> Result<Vec<f32>> {
    let mut all = Vec::new();
    for part in ["q", "k", "v"] {
        all.extend(g.tensor_f32(&format!("{prefix}.attn_{part}.{what}"))?);
    }
    Ok(all)
}

impl Embedder {
    /// Load a `nomic-bert` or `bert` GGUF file (every tensor is dequantized to f32: about 550 MB for nomic-embed-text-v1.5, 130 MB for bge-small-en-v1.5).
    pub fn load(path: impl AsRef<Path>) -> Result<Embedder> {
        let g = Gguf::open(path.as_ref())?;
        let (info, tok) = info_of(&g)?;
        let cfg = info.cfg;
        let t = |name: &str| -> Result<Vec<f32>> { g.tensor_f32(name) };
        let tok_emb = t("token_embd.weight")?;
        if tok_emb.len() != cfg.vocab * cfg.n_embd {
            return Err(Error::msg(format!("token_embd.weight has {} values, expected {} x {}", tok_emb.len(), cfg.vocab, cfg.n_embd)));
        }
        let types = t("token_types.weight")?;
        let norm = |prefix: &str| -> Result<Norm> { Ok(Norm { w: t(&format!("{prefix}.weight"))?, b: t(&format!("{prefix}.bias"))? }) };
        let bert = cfg.arch == Arch::Bert;
        let pos_emb = if bert { t("position_embd.weight")? } else { Vec::new() };
        let mut layers = Vec::with_capacity(cfg.n_layer);
        for i in 0..cfg.n_layer {
            let b = format!("blk.{i}");
            let layer = if bert {
                Layer {
                    qkv: fuse(&g, &b, "weight")?,
                    qkv_bias: fuse(&g, &b, "bias")?,
                    out: t(&format!("{b}.attn_output.weight"))?,
                    out_bias: t(&format!("{b}.attn_output.bias"))?,
                    up: t(&format!("{b}.ffn_up.weight"))?,
                    up_bias: t(&format!("{b}.ffn_up.bias"))?,
                    gate: Vec::new(),
                    down: t(&format!("{b}.ffn_down.weight"))?,
                    down_bias: t(&format!("{b}.ffn_down.bias"))?,
                    norm1: norm(&format!("{b}.attn_output_norm"))?,
                    norm2: norm(&format!("{b}.layer_output_norm"))?,
                }
            } else {
                Layer {
                    qkv: t(&format!("{b}.attn_qkv.weight"))?,
                    qkv_bias: Vec::new(),
                    out: t(&format!("{b}.attn_output.weight"))?,
                    out_bias: Vec::new(),
                    up: t(&format!("{b}.ffn_up.weight"))?,
                    up_bias: Vec::new(),
                    gate: t(&format!("{b}.ffn_gate.weight"))?,
                    down: t(&format!("{b}.ffn_down.weight"))?,
                    down_bias: Vec::new(),
                    norm1: norm(&format!("{b}.attn_output_norm"))?,
                    norm2: norm(&format!("{b}.layer_output_norm"))?,
                }
            };
            let (e, ff) = (cfg.n_embd, cfg.n_ff);
            let sizes = [(layer.qkv.len(), 3 * e * e), (layer.out.len(), e * e), (layer.up.len(), ff * e), (layer.down.len(), e * ff)];
            if sizes.iter().any(|(got, want)| got != want) || (bert && (layer.qkv_bias.len() != 3 * e || layer.out_bias.len() != e || layer.up_bias.len() != ff || layer.down_bias.len() != e)) {
                return Err(Error::msg(format!("the tensors of layer {i} do not have the sizes the configuration asks for (embedding {e}, feed-forward {ff})")));
            }
            layers.push(layer);
        }
        if bert && pos_emb.len() < cfg.ctx * cfg.n_embd {
            return Err(Error::msg(format!("position_embd.weight has {} values, fewer than the {} x {} the context needs", pos_emb.len(), cfg.ctx, cfg.n_embd)));
        }
        let threads = std::env::var("FORGE_EMBED_THREADS").ok().and_then(|v| v.parse().ok()).unwrap_or_else(|| std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).min(16)).max(1);
        Ok(Embedder {
            name: g.get_str("general.name").unwrap_or("embedding model").to_string(),
            emb_norm: norm("token_embd_norm")?,
            type_emb: types[..cfg.n_embd].to_vec(),
            pos_emb,
            tok_emb,
            tok,
            layers,
            cfg,
            threads,
            avx2: simd::has_avx2(),
        })
    }

    pub fn config(&self) -> &Config {
        &self.cfg
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    /// The dimension of the vectors.
    pub fn dim(&self) -> usize {
        self.cfg.n_embd
    }

    /// Take the model apart (for a backend that keeps its own copy of the weights).
    pub fn into_parts(self) -> Parts {
        Parts { cfg: self.cfg, name: self.name, tok: self.tok, tok_emb: self.tok_emb, type_emb: self.type_emb, pos_emb: self.pos_emb, emb_norm: self.emb_norm, layers: self.layers }
    }

    pub fn tokenizer(&self) -> &WordPiece {
        &self.tok
    }

    /// out[t * rows + o] = x[t] . w[o] for x of n rows of k values and w of `rows` rows of k values, split over the threads by output rows.
    fn matmul(&self, x: &[f32], n: usize, k: usize, w: &[f32], rows: usize, out: &mut [f32]) {
        debug_assert_eq!(x.len(), n * k);
        debug_assert_eq!(w.len(), rows * k);
        debug_assert_eq!(out.len(), n * rows);
        let avx2 = self.avx2;
        // tiles of eight output rows, taken by the threads as they get free (the cores of a hybrid CPU differ in speed, so a fixed split would wait for the slowest)
        const TILE: usize = 8;
        let tiles = rows.div_ceil(TILE);
        let nt = self.threads.min(tiles).max(1);
        let next = std::sync::atomic::AtomicUsize::new(0);
        let parts: Vec<(usize, Vec<f32>)> = std::thread::scope(|s| {
            let handles: Vec<_> = (0..nt)
                .map(|_| {
                    let next = &next;
                    s.spawn(move || {
                        let mut mine: Vec<(usize, Vec<f32>)> = Vec::new();
                        loop {
                            let tile = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            if tile >= tiles {
                                break;
                            }
                            let (r0, r1) = (tile * TILE, ((tile + 1) * TILE).min(rows));
                            let mut part = vec![0f32; (r1 - r0) * n];
                            simd::matmul_rows(x, n, k, w, r0, r1, avx2, &mut part);
                            mine.push((r0, part));
                        }
                        mine
                    })
                })
                .collect();
            handles.into_iter().flat_map(|h| h.join().expect("a matmul thread panicked")).collect()
        });
        for (r0, part) in parts {
            let cnt = part.len() / n;
            for oi in 0..cnt {
                for t in 0..n {
                    out[t * rows + r0 + oi] = part[oi * n + t];
                }
            }
        }
    }

    /// x = gelu(x + bias) for rows of `cols` values, over the threads when there is a lot of it (a text of a few hundred tokens has a million values per layer).
    fn bias_gelu(&self, x: &mut [f32], cols: usize, bias: &[f32]) {
        let apply = |part: &mut [f32]| {
            for row in part.chunks_mut(cols) {
                for (v, b) in row.iter_mut().zip(bias) {
                    *v = gelu(*v + b);
                }
            }
        };
        let rows = x.len() / cols;
        let nt = if x.len() < 1 << 16 { 1 } else { self.threads.min(rows).max(1) };
        if nt == 1 {
            apply(x);
            return;
        }
        let per = rows.div_ceil(nt) * cols;
        std::thread::scope(|s| {
            for part in x.chunks_mut(per) {
                s.spawn(move || apply(part));
            }
        });
    }

    /// Rotary embedding (NeoX layout) of the `heads` heads of one projection (q or k) held in `qkv` at column offset `col0` of rows of `stride` values.
    fn rotary(&self, qkv: &mut [f32], n: usize, stride: usize, col0: usize, cos: &[f32], sin: &[f32]) {
        let (heads, hd) = (self.cfg.n_head, self.cfg.n_embd / self.cfg.n_head);
        let half = hd / 2;
        for t in 0..n {
            for h in 0..heads {
                let base = t * stride + col0 + h * hd;
                for i in 0..half {
                    let (c, s) = (cos[t * half + i], sin[t * half + i]);
                    let (a, b) = (qkv[base + i], qkv[base + i + half]);
                    qkv[base + i] = a * c - b * s;
                    qkv[base + i + half] = b * c + a * s;
                }
            }
        }
    }

    /// Attention over all tokens (no mask): ctx[t] = softmax(q[t] k^T / sqrt(d)) v for every head.
    fn attention(&self, qkv: &[f32], n: usize, ctx: &mut [f32]) {
        let (heads, e) = (self.cfg.n_head, self.cfg.n_embd);
        let hd = e / heads;
        let scale = 1.0 / (hd as f32).sqrt();
        let avx2 = self.avx2;
        let nt = self.threads.min(n).max(1);
        let rows_per = n.div_ceil(nt);
        std::thread::scope(|s| {
            for (ci, out_rows) in ctx.chunks_mut(rows_per * e).enumerate() {
                s.spawn(move || {
                    let mut scores = vec![0f32; n];
                    for (ri, out_row) in out_rows.chunks_mut(e).enumerate() {
                        let t = ci * rows_per + ri;
                        for h in 0..heads {
                            let q = &qkv[t * 3 * e + h * hd..t * 3 * e + (h + 1) * hd];
                            let mut mx = f32::NEG_INFINITY;
                            for (sidx, sc) in scores.iter_mut().enumerate() {
                                let k = &qkv[sidx * 3 * e + e + h * hd..sidx * 3 * e + e + (h + 1) * hd];
                                *sc = simd::dot(q, k, avx2) * scale;
                                mx = mx.max(*sc);
                            }
                            let mut sum = 0f32;
                            for sc in scores.iter_mut() {
                                *sc = (*sc - mx).exp();
                                sum += *sc;
                            }
                            let inv = 1.0 / sum;
                            let o = &mut out_row[h * hd..(h + 1) * hd];
                            o.iter_mut().for_each(|v| *v = 0.0);
                            for (sidx, sc) in scores.iter().enumerate() {
                                let p = *sc * inv;
                                let v = &qkv[sidx * 3 * e + 2 * e + h * hd..sidx * 3 * e + 2 * e + (h + 1) * hd];
                                for d in 0..hd {
                                    o[d] += p * v[d];
                                }
                            }
                        }
                    }
                });
            }
        });
    }

    /// The vector of a token sequence (it must start with `[CLS]` and end with `[SEP]`, and fit the context).
    pub fn embed_ids(&self, ids: &[u32]) -> Vec<f32> {
        let (n, e, ff) = (ids.len(), self.cfg.n_embd, self.cfg.n_ff);
        let bert = self.cfg.arch == Arch::Bert;
        assert!(n > 0, "an empty token sequence");
        assert!(!bert || n <= self.cfg.ctx, "{n} tokens do not fit the {} learned positions", self.cfg.ctx);
        let mut x = vec![0f32; n * e];
        for (t, &id) in ids.iter().enumerate() {
            let id = (id as usize).min(self.cfg.vocab - 1);
            let row = &self.tok_emb[id * e..(id + 1) * e];
            for i in 0..e {
                x[t * e + i] = row[i] + self.type_emb[i];
            }
            if bert {
                for (v, p) in x[t * e..(t + 1) * e].iter_mut().zip(&self.pos_emb[t * e..(t + 1) * e]) {
                    *v += p;
                }
            }
        }
        layer_norm_rows(&mut x, n, e, &self.emb_norm, self.cfg.eps);
        let (cos, sin) = if bert { (Vec::new(), Vec::new()) } else { rope_tables(n, e / self.cfg.n_head, self.cfg.rope_base) };
        let mut qkv = vec![0f32; n * 3 * e];
        let mut ctx = vec![0f32; n * e];
        let mut tmp = vec![0f32; n * e];
        let mut up = vec![0f32; n * ff];
        let mut gate = if bert { Vec::new() } else { vec![0f32; n * ff] };
        for l in &self.layers {
            self.matmul(&x, n, e, &l.qkv, 3 * e, &mut qkv);
            if bert {
                add_bias(&mut qkv, 3 * e, &l.qkv_bias);
            } else {
                self.rotary(&mut qkv, n, 3 * e, 0, &cos, &sin);
                self.rotary(&mut qkv, n, 3 * e, e, &cos, &sin);
            }
            self.attention(&qkv, n, &mut ctx);
            self.matmul(&ctx, n, e, &l.out, e, &mut tmp);
            if bert {
                add_bias(&mut tmp, e, &l.out_bias);
            }
            for i in 0..n * e {
                x[i] += tmp[i];
            }
            layer_norm_rows(&mut x, n, e, &l.norm1, self.cfg.eps);
            self.matmul(&x, n, e, &l.up, ff, &mut up);
            if bert {
                self.bias_gelu(&mut up, ff, &l.up_bias);
            } else {
                self.matmul(&x, n, e, &l.gate, ff, &mut gate);
                for i in 0..n * ff {
                    let g = gate[i];
                    up[i] *= g / (1.0 + (-g).exp());
                }
            }
            self.matmul(&up, n, ff, &l.down, e, &mut tmp);
            if bert {
                add_bias(&mut tmp, e, &l.down_bias);
            }
            for i in 0..n * e {
                x[i] += tmp[i];
            }
            layer_norm_rows(&mut x, n, e, &l.norm2, self.cfg.eps);
        }
        let mut v = vec![0f32; e];
        if self.cfg.mean_pool {
            for t in 0..n {
                for i in 0..e {
                    v[i] += x[t * e + i];
                }
            }
            for a in v.iter_mut() {
                *a /= n as f32;
            }
        } else {
            v.copy_from_slice(&x[..e]);
        }
        let norm = v.iter().map(|a| a * a).sum::<f32>().sqrt().max(1e-12);
        v.iter_mut().for_each(|a| *a /= norm);
        v
    }

    /// The embedding of a text (cut to the model's context).
    pub fn embed(&self, text: &str) -> Embedding {
        let ids = self.tok.encode_truncated(text, self.cfg.ctx);
        let tokens = ids.len();
        Embedding { vector: self.embed_ids(&ids), tokens }
    }
}

impl Encode for Embedder {
    fn embed_many(&self, texts: &[&str]) -> std::result::Result<Vec<Embedding>, String> {
        Ok(texts.iter().map(|t| Embedder::embed(self, t)).collect())
    }
    fn config(&self) -> &Config {
        &self.cfg
    }
    fn name(&self) -> &str {
        &self.name
    }
    fn backend(&self) -> String {
        format!("CPU, {} threads{}", self.threads, if self.avx2 { ", AVX2" } else { "" })
    }
}

/// The first `dims` values of a vector, normalised to length one again (Matryoshka-style truncation, OpenAI's `dimensions`).
pub fn truncate_normalized(v: &[f32], dims: usize) -> Vec<f32> {
    let d = dims.clamp(1, v.len());
    let mut out = v[..d].to_vec();
    let norm = out.iter().map(|a| a * a).sum::<f32>().sqrt().max(1e-12);
    out.iter_mut().for_each(|a| *a /= norm);
    out
}

/// The cosine similarity of two vectors.
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb).max(1e-30)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truncation_renormalizes() {
        let v = vec![0.6f32, 0.0, 0.8, 0.0];
        let t = truncate_normalized(&v, 3);
        assert!((t.iter().map(|a| a * a).sum::<f32>() - 1.0).abs() < 1e-6);
        assert!((t[0] - 0.6).abs() < 1e-6 && (t[2] - 0.8).abs() < 1e-6);
        assert_eq!(truncate_normalized(&v, 0).len(), 1);
        assert_eq!(truncate_normalized(&v, 99).len(), 4);
    }

    #[test]
    fn cosine_of_parallel_and_orthogonal_vectors() {
        assert!((cosine(&[1.0, 2.0], &[2.0, 4.0]) - 1.0).abs() < 1e-6);
        assert!(cosine(&[1.0, 0.0], &[0.0, 3.0]).abs() < 1e-6);
    }

    #[test]
    fn erf_matches_the_values_of_a_math_library() {
        // (x, erf(x), erfc(x)) from Python's math module
        let table = [
            (0.1, 0.1124629160182849, 0.8875370839817152),
            (0.5, 0.5204998778130465, 0.4795001221869534),
            (1.0, 0.8427007929497149, 0.1572992070502851),
            (1.5, 0.9661051464753108, 0.033894853524689274),
            (1.99, 0.995111413199617, 0.004888586800383002),
            (2.0, 0.9953222650189527, 0.004677734981047265),
            (2.5, 0.999593047982555, 0.000406952017444959),
            (3.0, 0.9999779095030014, 2.209049699858544e-05),
            (4.0, 0.9999999845827421, 1.5417257900280017e-08),
            (5.0, 0.9999999999984626, 1.5374597944280351e-12),
            (6.0, 1.0, 2.1519736712498913e-17),
        ];
        for (x, erf_want, erfc_want) in table {
            // erfc is checked in relative terms all the way into the tail, erf as the f32 it is returned as
            let c = erfc_positive(x);
            assert!(((c - erfc_want) / erfc_want).abs() < 1e-12, "erfc({x}) = {c}, wanted {erfc_want}");
            assert!((erf(x as f32) as f64 - erf_want).abs() < 1.5e-7 * erf_want.max(1e-3), "erf({x}) = {}, wanted {erf_want}", erf(x as f32));
            assert_eq!(erf(-x as f32), -erf(x as f32));
        }
        assert_eq!(erf(0.0), 0.0);
        assert_eq!(erf(1000.0), 1.0);
        assert_eq!(erfc_positive(1000.0), 0.0);
    }

    #[test]
    fn gelu_is_x_times_the_normal_distribution_function() {
        // (x, gelu(x)) from Python: x * 0.5 * math.erfc(-x / math.sqrt(2))
        let table = [
            (-6.0, -5.91952587022621e-09),
            (-4.0, -0.00012668496733247986),
            (-3.0, -0.004049694094890288),
            (-2.0, -0.04550026389635843),
            (-1.0, -0.15865525393145707),
            (-0.5, -0.15426876936299347),
            (-0.1, -0.0460172162722971),
            (0.1, 0.053982783727702904),
            (0.5, 0.3457312306370065),
            (1.0, 0.8413447460685428),
            (2.0, 1.9544997361036418),
            (3.0, 2.99595030590511),
            (6.0, 5.999999994080474),
        ];
        for (x, want) in table {
            let got = gelu(x as f32) as f64;
            // a single rounding to f32 (and the f32 input): relative error of a few millionths of a percent at most
            assert!(((got - want) / want).abs() < 2e-7, "gelu({x}) = {got}, wanted {want}");
        }
        assert_eq!(gelu(0.0), 0.0);
        // large inputs: the identity on the right, nothing on the left
        assert_eq!(gelu(30.0), 30.0);
        assert_eq!(gelu(-30.0), 0.0);
        assert!(gelu(f32::MIN_POSITIVE).is_finite());
    }

    #[test]
    fn bias_helpers_work_on_every_row() {
        let mut x = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        add_bias(&mut x, 3, &[10.0, 20.0, 30.0]);
        assert_eq!(x, vec![11.0, 22.0, 33.0, 14.0, 25.0, 36.0]);
    }

    #[test]
    fn arch_names_round_trip() {
        for a in [Arch::NomicBert, Arch::Bert] {
            assert_eq!(Arch::from_gguf_name(a.gguf_name()), Some(a));
        }
        assert_eq!(Arch::from_gguf_name("llama"), None);
    }
}
