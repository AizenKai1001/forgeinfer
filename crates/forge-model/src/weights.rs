//! Weights of a decoder-only model as f32 matrices (the CPU reference path dequantises everything at load), from a GGUF file or a Hugging Face checkpoint directory.
//! All weights are normalised to ONE convention: RoPE pairs (i, i + rotary_dim/2) ("rotate half"). llama.cpp stores Llama-architecture Q and K rows permuted for adjacent-pair RoPE;
//! that permutation is undone at load. RMSNorm weights are plain multipliers: Hugging Face's zero-centred norms of Qwen3.5 (y = x * rsqrt(..) * (1 + w)) get the 1 added at load, as
//! llama.cpp's converter does for GGUF files.

use crate::config::{Arch, Config};
use forge_core::{Error, Json, Result};
use forge_gguf::{DType, Gguf, SafeTensors, SafeTensorsDir};
use std::path::Path;

/// The bytes of a quantized tensor: owned, or a range of a memory-mapped model file (no copy: a 27B model's 16 GB of blocks stay in the page cache instead of in the process).
pub enum RawBytes {
    Owned(Vec<u8>),
    Mapped(std::sync::Arc<forge_core::Mmap>, usize, usize),
}

impl std::ops::Deref for RawBytes {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        match self {
            RawBytes::Owned(v) => v,
            RawBytes::Mapped(m, off, len) => &m[*off..*off + *len],
        }
    }
}

impl From<Vec<u8>> for RawBytes {
    fn from(v: Vec<u8>) -> RawBytes {
        RawBytes::Owned(v)
    }
}

/// Quantized tensor data in the native GGUF block layout (row-major: every row is `cols / block_values` blocks).
pub struct Raw {
    pub dtype: DType,
    pub bytes: RawBytes,
}

/// Row-major [rows][cols] matrix: a linear layer weight is [out][in]. `w` holds the f32 values (empty when the model was loaded native-only); `raw` the quantized blocks of a GGUF tensor of a
/// format the GPU kernels read directly (Q8_0, Q4_0, Q5_0, Q4_K, Q5_K, Q6_K), when requested.
pub struct Mat {
    pub rows: usize,
    pub cols: usize,
    pub w: Vec<f32>,
    pub raw: Option<Raw>,
}

impl Mat {
    pub fn empty() -> Mat {
        Mat {
            rows: 0,
            cols: 0,
            w: Vec::new(),
            raw: None,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct LoadOpts {
    /// dequantize quantized tensors to f32 (needed by the CPU reference path)
    pub dequantize: bool,
    /// keep the quantized blocks of the formats the GPU reads natively
    pub keep_raw: bool,
}

/// The weights of a Gated DeltaNet layer (Qwen3.5), in the order of the checkpoint's value heads.
pub struct DeltaLayer {
    /// [conv_dim][n_embd]: the q, k and v channels
    pub w_qkv: Mat,
    /// [value_dim][n_embd]: the output gate z
    pub w_z: Mat,
    /// [n_v][n_embd]: input of the decay gate
    pub w_a: Mat,
    /// [n_v][n_embd]: input of the write strength beta
    pub w_b: Mat,
    /// [conv_dim][conv]: depthwise causal convolution, w[c][conv-1] multiplies the current token
    pub conv: Vec<f32>,
    /// [n_v]: -exp(A_log), the (negative) decay rate
    pub a_neg: Vec<f32>,
    /// [n_v]
    pub dt_bias: Vec<f32>,
    /// [head_v]: weight of the gated RMSNorm (plain)
    pub norm: Vec<f32>,
    /// [n_embd][value_dim]
    pub w_out: Mat,
}

pub struct Layer {
    pub attn_norm: Vec<f32>,
    /// attention layers (empty matrices in a Gated DeltaNet layer); `wq` has `cfg.q_rows()` rows: per head the query, then the gate when the model has one
    pub wq: Mat,
    pub wk: Mat,
    pub wv: Mat,
    pub wo: Mat,
    pub q_norm: Option<Vec<f32>>,
    pub k_norm: Option<Vec<f32>>,
    /// biases of the q, k and v projections (Qwen2): [q_rows], [kv_dim], [kv_dim]
    pub bq: Option<Vec<f32>>,
    pub bk: Option<Vec<f32>>,
    pub bv: Option<Vec<f32>>,
    pub ffn_norm: Vec<f32>,
    pub w_gate: Mat,
    pub w_up: Mat,
    pub w_down: Mat,
    pub delta: Option<DeltaLayer>,
}

/// Qwen3.5's multi-token-prediction head: from the main model's final-normed hidden state at position t and the embedding of the token at t + 1 it predicts the token at t + 2
/// (a draft for speculative decoding). z = fc(concat(rms(embedding, norm_e), rms(hidden, norm_h))); one gated-attention decoder layer with its own KV cache; logits = rms(z, out_norm) . embedding^T.
pub struct MtpWeights {
    /// [n_embd][2 n_embd]
    pub fc: Mat,
    pub norm_e: Vec<f32>,
    pub norm_h: Vec<f32>,
    pub out_norm: Vec<f32>,
    pub layer: Layer,
}

pub struct Model {
    pub cfg: Config,
    /// [vocab][n_embd]
    pub tok_emb: Mat,
    pub layers: Vec<Layer>,
    pub out_norm: Vec<f32>,
    /// None = tied to the embedding table
    pub lm_head: Option<Mat>,
    /// the MTP head of a Qwen3.5 checkpoint, when it was found (the Hugging Face directory, or `mtp.safetensors` next to a GGUF file)
    pub mtp: Option<MtpWeights>,
}

enum Container {
    Gguf(Gguf),
    Hf(SafeTensorsDir),
}

/// A rule of a recipe (FORGE_RECIPE=recipe.json): these tensors of these layers come from another GGUF file of the same model (a different quantization), as they are: model surgery at the container
/// level, no re-quantization. {"overrides": [{"from": "x.gguf", "layers": "0-7,60-63", "tensors": ["w_gate", "w_up", "w_down"], "if_dtype": ["Q3_K"]}]}; "layers" defaults to all, "if_dtype" (optional)
/// takes the tensor only when the other file stores it in one of these types. Tensor names: w_gate w_up w_down wq wk wv wo wqkv wz wa wb w_out tok_emb lm_head.
struct Override {
    layers: Option<Vec<bool>>,
    tensors: Vec<String>,
    if_dtype: Vec<String>,
    g: Gguf,
}

struct Source {
    c: Container,
    arch: Arch,
    /// Hugging Face: prefix of the model tensors ("model." or "model.language_model.")
    hf_prefix: String,
    overrides: Vec<Override>,
}

/// "0-7,60-63,10" -> a flag per layer
fn parse_layer_list(text: &str, n_layer: usize) -> Vec<bool> {
    let mut sel = vec![false; n_layer];
    for part in text.split(',').map(|p| p.trim()).filter(|p| !p.is_empty()) {
        let (a, b) = match part.split_once('-') {
            Some((a, b)) => (a.trim().parse::<usize>(), b.trim().parse::<usize>()),
            None => (part.parse::<usize>(), part.parse::<usize>()),
        };
        if let (Ok(a), Ok(b)) = (a, b) {
            for l in a..=b.min(n_layer.saturating_sub(1)) {
                sel[l] = true;
            }
        }
    }
    sel
}

/// The overrides of the recipe named by FORGE_RECIPE (none when it is unset).
fn load_recipe(n_layer: usize) -> Result<Vec<Override>> {
    match std::env::var("FORGE_RECIPE") {
        Ok(path) => load_recipe_file(&path, n_layer),
        Err(_) => Ok(Vec::new()),
    }
}

fn load_recipe_file(path: &str, n_layer: usize) -> Result<Vec<Override>> {
    let j = Json::from_file(path)?;
    let mut out = Vec::new();
    for r in j.get("overrides").and_then(|v| v.as_array()).unwrap_or(&[]) {
        let from = r
            .get("from")
            .and_then(|v| v.as_str())
            .ok_or_else(|| Error::format("recipe: an override without \"from\""))?;
        let layers = r.get("layers").and_then(|v| v.as_str()).map(|t| parse_layer_list(t, n_layer));
        let strings = |k: &str| -> Vec<String> {
            r.get(k)
                .and_then(|v| v.as_array())
                .map(|a| a.iter().filter_map(|x| x.as_str().map(|s| s.to_string())).collect())
                .unwrap_or_default()
        };
        out.push(Override {
            layers,
            tensors: strings("tensors"),
            if_dtype: strings("if_dtype"),
            g: Gguf::open(from)?,
        });
    }
    eprintln!("[model] recipe {path}: {} overrides", out.len());
    Ok(out)
}

impl Source {
    /// canonical name -> name inside the container
    fn names(&self, canon: &str) -> String {
        let qwen35 = self.arch == Arch::Qwen35;
        match &self.c {
            Container::Gguf(_) => match canon {
                "tok_emb" => "token_embd.weight".into(),
                "out_norm" => "output_norm.weight".into(),
                "lm_head" => "output.weight".into(),
                c => {
                    let (l, rest) = c.split_once('.').unwrap();
                    let n = &l[1..];
                    let name = match rest {
                        "attn_norm" => "attn_norm.weight",
                        "wq" => "attn_q.weight",
                        "wk" => "attn_k.weight",
                        "wv" => "attn_v.weight",
                        "wo" => "attn_output.weight",
                        "q_norm" => "attn_q_norm.weight",
                        "k_norm" => "attn_k_norm.weight",
                        "bq" => "attn_q.bias",
                        "bk" => "attn_k.bias",
                        "bv" => "attn_v.bias",
                        // in the Qwen3.5 family the pre-MLP norm is called post_attention_norm
                        "ffn_norm" => {
                            if qwen35 {
                                "post_attention_norm.weight"
                            } else {
                                "ffn_norm.weight"
                            }
                        }
                        "w_gate" => "ffn_gate.weight",
                        "w_up" => "ffn_up.weight",
                        "w_down" => "ffn_down.weight",
                        "wqkv" => "attn_qkv.weight",
                        "wz" => "attn_gate.weight",
                        "wa" => "ssm_alpha.weight",
                        "wb" => "ssm_beta.weight",
                        "conv" => "ssm_conv1d.weight",
                        "a" => "ssm_a",
                        "dt_bias" => "ssm_dt.bias",
                        "ssm_norm" => "ssm_norm.weight",
                        "w_out" => "ssm_out.weight",
                        other => panic!("unknown weight {other}"),
                    };
                    format!("blk.{n}.{name}")
                }
            },
            Container::Hf(_) => {
                let p = &self.hf_prefix;
                match canon {
                    "tok_emb" => format!("{p}embed_tokens.weight"),
                    "out_norm" => format!("{p}norm.weight"),
                    "lm_head" => "lm_head.weight".into(),
                    c => {
                        let (l, rest) = c.split_once('.').unwrap();
                        let n = &l[1..];
                        let name = match rest {
                            "attn_norm" => "input_layernorm.weight",
                            "wq" => "self_attn.q_proj.weight",
                            "wk" => "self_attn.k_proj.weight",
                            "wv" => "self_attn.v_proj.weight",
                            "wo" => "self_attn.o_proj.weight",
                            "q_norm" => "self_attn.q_norm.weight",
                            "k_norm" => "self_attn.k_norm.weight",
                            "bq" => "self_attn.q_proj.bias",
                            "bk" => "self_attn.k_proj.bias",
                            "bv" => "self_attn.v_proj.bias",
                            "ffn_norm" => "post_attention_layernorm.weight",
                            "w_gate" => "mlp.gate_proj.weight",
                            "w_up" => "mlp.up_proj.weight",
                            "w_down" => "mlp.down_proj.weight",
                            "wqkv" => "linear_attn.in_proj_qkv.weight",
                            "wz" => "linear_attn.in_proj_z.weight",
                            "wa" => "linear_attn.in_proj_a.weight",
                            "wb" => "linear_attn.in_proj_b.weight",
                            "conv" => "linear_attn.conv1d.weight",
                            "a" => "linear_attn.A_log",
                            "dt_bias" => "linear_attn.dt_bias",
                            "ssm_norm" => "linear_attn.norm.weight",
                            "w_out" => "linear_attn.out_proj.weight",
                            other => panic!("unknown weight {other}"),
                        };
                        format!("{p}layers.{n}.{name}")
                    }
                }
            }
        }
    }

    fn vec(&self, canon: &str) -> Result<Vec<f32>> {
        let n = self.names(canon);
        match &self.c {
            Container::Gguf(g) => {
                // llama.cpp names the decay tensor `ssm_a` (some files add `.weight`)
                if g.tensor_info(&n).is_none() && canon.ends_with(".a") {
                    return g.tensor_f32(&format!("{n}.weight"));
                }
                g.tensor_f32(&n)
            }
            Container::Hf(d) => d.tensor_f32(&n),
        }
    }

    /// An RMSNorm weight as a plain multiplier.
    fn norm(&self, canon: &str) -> Result<Vec<f32>> {
        let mut v = self.vec(canon)?;
        if self.arch == Arch::Qwen35 && matches!(self.c, Container::Hf(_)) {
            v.iter_mut().for_each(|x| *x += 1.0);
        }
        Ok(v)
    }

    fn has(&self, canon: &str) -> bool {
        let n = self.names(canon);
        match &self.c {
            Container::Gguf(g) => g.tensor_info(&n).is_some(),
            Container::Hf(d) => d.tensor(&n).is_some(),
        }
    }

    /// The file a tensor comes from when a recipe overrides it.
    fn override_gguf(&self, canon: &str, name: &str) -> Option<&Gguf> {
        let (layer, tensor) = match canon.split_once('.') {
            Some((l, t)) => (l[1..].parse::<usize>().ok(), t),
            None => (None, canon),
        };
        self.overrides.iter().find_map(|o| {
            if !o.tensors.iter().any(|t| t == tensor) {
                return None;
            }
            if let (Some(sel), Some(l)) = (&o.layers, layer) {
                if !sel.get(l).copied().unwrap_or(false) {
                    return None;
                }
            }
            let t = o.g.tensor_info(name)?;
            if !o.if_dtype.is_empty() && !o.if_dtype.iter().any(|d| *d == t.dtype.name()) {
                return None;
            }
            Some(&o.g)
        })
    }

    fn mat(&self, canon: &str, rows: usize, cols: usize, o: &LoadOpts) -> Result<Mat> {
        // a model quantized by `forge quantize`: the matrix is a U8 tensor of raw GGUF blocks, described in the file's metadata
        if let Container::Hf(d) = &self.c {
            let name = self.names(canon);
            if let Some((st, t)) = d.tensor(&name) {
                if let Some(q) = quant_entry(st, &name) {
                    let (dtype, qr, qc) = q?;
                    if (qr, qc) != (rows, cols)
                        || !cols.is_multiple_of(dtype.block_values())
                        || t.end - t.start != dtype.bytes_for(rows * cols)
                    {
                        return Err(Error::format(format!(
                            "{name}: quantized as {} x {} {:?}, expected {rows} x {cols}",
                            qr, qc, dtype
                        )));
                    }
                    let bytes = st.bytes(t).to_vec();
                    let w = if o.dequantize {
                        forge_gguf::dequantize(dtype, &bytes, rows * cols)?
                    } else {
                        Vec::new()
                    };
                    return Ok(Mat {
                        rows,
                        cols,
                        w,
                        raw: if o.keep_raw || !o.dequantize {
                            Some(Raw { dtype, bytes: bytes.into() })
                        } else {
                            None
                        },
                    });
                }
            }
        }
        if let Container::Gguf(base) = &self.c {
            let name = self.names(canon);
            let g = self.override_gguf(canon, &name).unwrap_or(base);
            let t = g
                .tensor_info(&name)
                .ok_or_else(|| Error::msg(format!("tensor {name} not found in the GGUF file")))?;
            let native = matches!(
                t.dtype,
                DType::Q8_0 | DType::Q4_0 | DType::Q5_0 | DType::Q4_K | DType::Q5_K | DType::Q6_K | DType::Q3_K | DType::Q2_K
            );
            if o.keep_raw && native {
                if t.n_elements() != rows * cols || !cols.is_multiple_of(t.dtype.block_values()) {
                    return Err(Error::format(format!(
                        "{name}: {} values in {:?}, expected {rows}x{cols}",
                        t.n_elements(),
                        t.dtype
                    )));
                }
                let (map, off, len) = g.tensor_range(t);
                let raw = Raw {
                    dtype: t.dtype,
                    bytes: RawBytes::Mapped(map, off, len),
                };
                let w = if o.dequantize {
                    g.tensor_f32(&name)?
                } else {
                    Vec::new()
                };
                return Ok(Mat {
                    rows,
                    cols,
                    w,
                    raw: Some(raw),
                });
            }
        }
        let w = self.vec(canon)?;
        if w.len() != rows * cols {
            return Err(Error::format(format!(
                "{}: {} values, expected {rows}x{cols}",
                self.names(canon),
                w.len()
            )));
        }
        Ok(Mat {
            rows,
            cols,
            w,
            raw: None,
        })
    }
}

/// The quantization record of tensor `name` in a file written by `forge quantize` (a safetensors file whose `__metadata__.quant` is a JSON object name -> {"dtype", "rows", "cols"}).
pub fn quant_entry(
    st: &forge_gguf::SafeTensors,
    name: &str,
) -> Option<Result<(DType, usize, usize)>> {
    let text = st.metadata.get("quant")?.as_str()?;
    let j = Json::parse(text).ok()?;
    let e = j.get(name)?;
    let dt = match e.get("dtype").and_then(|v| v.as_str())? {
        "Q8_0" => DType::Q8_0,
        "Q4_0" => DType::Q4_0,
        "Q5_0" => DType::Q5_0,
        "Q4_K" => DType::Q4_K,
        "Q5_K" => DType::Q5_K,
        "Q6_K" => DType::Q6_K,
        other => {
            return Some(Err(Error::format(format!(
                "{name}: unknown quantization {other}"
            ))))
        }
    };
    Some(Ok((
        dt,
        e.get("rows")?.as_usize()?,
        e.get("cols")?.as_usize()?,
    )))
}

/// Undo llama.cpp's Q/K row permutation: it stores the two rows of every rotation pair (2a, 2a+1) of a head adjacent; rotate-half wants them at (a, a + head_dim/2).
fn unpermute_rows(m: &mut Mat, heads: usize) {
    let hd = m.rows / heads;
    let mut out = vec![0f32; m.w.len()];
    let mut raw_out = m.raw.as_ref().map(|r| vec![0u8; r.bytes.len()]);
    let row_bytes = m
        .raw
        .as_ref()
        .map(|r| m.cols / r.dtype.block_values() * r.dtype.block_bytes())
        .unwrap_or(0);
    for h in 0..heads {
        for a in 0..hd / 2 {
            for b in 0..2 {
                let src = h * hd + a * 2 + b;
                let dst = h * hd + b * (hd / 2) + a;
                if !m.w.is_empty() {
                    out[dst * m.cols..(dst + 1) * m.cols]
                        .copy_from_slice(&m.w[src * m.cols..(src + 1) * m.cols]);
                }
                if let (Some(ro), Some(r)) = (raw_out.as_mut(), m.raw.as_ref()) {
                    ro[dst * row_bytes..(dst + 1) * row_bytes]
                        .copy_from_slice(&r.bytes[src * row_bytes..(src + 1) * row_bytes]);
                }
            }
        }
    }
    if !m.w.is_empty() {
        m.w = out;
    }
    if let (Some(ro), Some(r)) = (raw_out, m.raw.as_mut()) {
        r.bytes = ro.into();
    }
}

impl Model {
    /// Load a `.gguf` file or a Hugging Face directory (config.json + safetensors), everything as f32 (the CPU reference path).
    pub fn load(path: impl AsRef<Path>) -> Result<Model> {
        Model::load_with(
            path,
            LoadOpts {
                dequantize: true,
                keep_raw: false,
            },
        )
    }

    /// Load a GGUF for the GPU kernels that read its quantized blocks directly: no f32 copy of the quantized tensors (a 9B Q4_K_M stays 5 GB in host memory instead of 36 GB).
    pub fn load_native(path: impl AsRef<Path>) -> Result<Model> {
        Model::load_with(
            path,
            LoadOpts {
                dequantize: false,
                keep_raw: true,
            },
        )
    }

    pub fn load_with(path: impl AsRef<Path>, opts: LoadOpts) -> Result<Model> {
        let path = path.as_ref();
        if path.is_dir() {
            let cfg = Config::from_hf_json(&Json::from_file(path.join("config.json"))?)?;
            let d = SafeTensorsDir::open(path)?;
            let hf_prefix = if d
                .tensor("model.language_model.embed_tokens.weight")
                .is_some()
            {
                "model.language_model."
            } else {
                "model."
            };
            let src = Source {
                c: Container::Hf(d),
                arch: cfg.arch,
                hf_prefix: hf_prefix.to_string(),
                overrides: Vec::new(),
            };
            let mut m = Model::build(cfg, src, false, &opts)?;
            if m.cfg.arch == Arch::Qwen35 {
                m.load_mtp(path, &opts)?;
            }
            Ok(m)
        } else {
            let g = Gguf::open(path)?;
            let cfg = Config::from_gguf(&g)?;
            let permuted = cfg.arch == Arch::Llama; // llama-architecture GGUFs carry the permuted Q/K layout
            let overrides = load_recipe(cfg.n_layer)?;
            let src = Source {
                c: Container::Gguf(g),
                arch: cfg.arch,
                hf_prefix: String::new(),
                overrides,
            };
            Model::build(cfg, src, permuted, &opts)
        }
    }

    fn build(cfg: Config, src: Source, permuted_qk: bool, o: &LoadOpts) -> Result<Model> {
        let (e, q, kv, f) = (cfg.n_embd, cfg.q_dim(), cfg.kv_dim(), cfg.n_ff);
        let tok_emb = src.mat("tok_emb", cfg.vocab, e, o)?;
        let out_norm = src.norm("out_norm")?;
        let lm_head = if !cfg.tie_embeddings && src.has("lm_head") {
            Some(src.mat("lm_head", cfg.vocab, e, o)?)
        } else {
            None
        };
        let mut layers = Vec::with_capacity(cfg.n_layer);
        for l in 0..cfg.n_layer {
            let n = |s: &str| format!("L{l}.{s}");
            // the bias of a projection (models with `qkv_bias`), checked against the rows of its matrix
            let bias = |what: &str, rows: usize| -> Result<Option<Vec<f32>>> {
                if !cfg.qkv_bias {
                    return Ok(None);
                }
                let v = src.vec(&n(what))?;
                if v.len() != rows {
                    return Err(Error::format(format!("{}: {} values, expected {rows}", src.names(&n(what)), v.len())));
                }
                Ok(Some(v))
            };
            let common = (src.norm(&n("attn_norm"))?, src.norm(&n("ffn_norm"))?);
            let (w_gate, w_up, w_down) = (
                src.mat(&n("w_gate"), f, e, o)?,
                src.mat(&n("w_up"), f, e, o)?,
                src.mat(&n("w_down"), e, f, o)?,
            );
            if cfg.is_delta(l) {
                let d = cfg.delta.as_ref().ok_or_else(|| {
                    Error::format("a Gated DeltaNet layer without DeltaNet hyper-parameters")
                })?;
                let a_raw = src.vec(&n("a"))?;
                // Hugging Face stores A_log; llama.cpp's converter stores -exp(A_log)
                let a_neg = if matches!(src.c, Container::Hf(_)) {
                    a_raw.iter().map(|v| -v.exp()).collect()
                } else {
                    a_raw
                };
                layers.push(Layer {
                    attn_norm: common.0,
                    wq: Mat::empty(),
                    wk: Mat::empty(),
                    wv: Mat::empty(),
                    wo: Mat::empty(),
                    q_norm: None,
                    k_norm: None,
                    bq: None,
                    bk: None,
                    bv: None,
                    ffn_norm: common.1,
                    w_gate,
                    w_up,
                    w_down,
                    delta: Some(DeltaLayer {
                        w_qkv: src.mat(&n("wqkv"), d.conv_dim(), e, o)?,
                        w_z: src.mat(&n("wz"), d.value_dim(), e, o)?,
                        w_a: src.mat(&n("wa"), d.n_v, e, o)?,
                        w_b: src.mat(&n("wb"), d.n_v, e, o)?,
                        conv: src.vec(&n("conv"))?,
                        a_neg,
                        dt_bias: src.vec(&n("dt_bias"))?,
                        norm: src.vec(&n("ssm_norm"))?,
                        w_out: src.mat(&n("w_out"), e, d.value_dim(), o)?,
                    }),
                });
                continue;
            }
            let mut wq = src.mat(&n("wq"), cfg.q_rows(), e, o)?;
            let mut wk = src.mat(&n("wk"), kv, e, o)?;
            if permuted_qk {
                unpermute_rows(&mut wq, cfg.n_head);
                unpermute_rows(&mut wk, cfg.n_head_kv);
            }
            layers.push(Layer {
                attn_norm: common.0,
                wq,
                wk,
                wv: src.mat(&n("wv"), kv, e, o)?,
                wo: src.mat(&n("wo"), e, q, o)?,
                q_norm: if cfg.qk_norm {
                    Some(src.norm(&n("q_norm"))?)
                } else {
                    None
                },
                k_norm: if cfg.qk_norm {
                    Some(src.norm(&n("k_norm"))?)
                } else {
                    None
                },
                bq: bias("bq", cfg.q_rows())?,
                bk: bias("bk", kv)?,
                bv: bias("bv", kv)?,
                ffn_norm: common.1,
                w_gate,
                w_up,
                w_down,
                delta: None,
            });
        }
        Ok(Model {
            cfg,
            tok_emb,
            layers,
            out_norm,
            lm_head,
            mtp: None,
        })
    }

    /// Load the MTP head from a Hugging Face directory (tensors `mtp.*`) or a single safetensors file with the same tensor names; returns None when the tensors are absent.
    pub fn load_mtp(&mut self, path: impl AsRef<Path>, o: &LoadOpts) -> Result<()> {
        let path = path.as_ref();
        enum St {
            Dir(SafeTensorsDir),
            File(SafeTensors),
        }
        let st = if path.is_dir() {
            St::Dir(SafeTensorsDir::open(path)?)
        } else {
            St::File(SafeTensors::open(path)?)
        };
        let has = |n: &str| match &st {
            St::Dir(d) => d.tensor(n).is_some(),
            St::File(f) => f.get(n).is_some(),
        };
        if !has("mtp.fc.weight") {
            return Ok(());
        }
        let get = |n: &str| -> Result<Vec<f32>> {
            match &st {
                St::Dir(d) => d.tensor_f32(n),
                St::File(f) => {
                    let t = f
                        .get(n)
                        .ok_or_else(|| Error::format(format!("tensor {n} missing")))?;
                    f.to_f32(t)
                }
            }
        };
        let cfg = &self.cfg;
        let (e, f) = (cfg.n_embd, cfg.n_ff);
        let mat = |n: &str, rows: usize, cols: usize| -> Result<Mat> {
            let w = get(n)?;
            if w.len() != rows * cols {
                return Err(Error::format(format!(
                    "{n}: {} values, expected {rows}x{cols}",
                    w.len()
                )));
            }
            Ok(Mat {
                rows,
                cols,
                w,
                raw: None,
            })
        };
        // Hugging Face norms are zero-centred: add the 1 here, as for the main model
        let norm =
            |n: &str| -> Result<Vec<f32>> { Ok(get(n)?.into_iter().map(|v| v + 1.0).collect()) };
        let _ = o;
        let l = "mtp.layers.0.";
        let layer = Layer {
            attn_norm: norm(&format!("{l}input_layernorm.weight"))?,
            wq: mat(&format!("{l}self_attn.q_proj.weight"), cfg.q_rows(), e)?,
            wk: mat(&format!("{l}self_attn.k_proj.weight"), cfg.kv_dim(), e)?,
            wv: mat(&format!("{l}self_attn.v_proj.weight"), cfg.kv_dim(), e)?,
            wo: mat(&format!("{l}self_attn.o_proj.weight"), e, cfg.q_dim())?,
            q_norm: Some(norm(&format!("{l}self_attn.q_norm.weight"))?),
            k_norm: Some(norm(&format!("{l}self_attn.k_norm.weight"))?),
            bq: None,
            bk: None,
            bv: None,
            ffn_norm: norm(&format!("{l}post_attention_layernorm.weight"))?,
            w_gate: mat(&format!("{l}mlp.gate_proj.weight"), f, e)?,
            w_up: mat(&format!("{l}mlp.up_proj.weight"), f, e)?,
            w_down: mat(&format!("{l}mlp.down_proj.weight"), e, f)?,
            delta: None,
        };
        let w = MtpWeights {
            fc: mat("mtp.fc.weight", e, 2 * e)?,
            norm_e: norm("mtp.pre_fc_norm_embedding.weight")?,
            norm_h: norm("mtp.pre_fc_norm_hidden.weight")?,
            out_norm: norm("mtp.norm.weight")?,
            layer,
        };
        self.mtp = Some(w);
        Ok(())
    }
}


/// The canonical name (`L12.w_gate`, `tok_emb`) of a GGUF tensor name, for the recipe rules; None for tensors a recipe does not address (norms, small tensors).
fn canon_of_gguf(name: &str) -> Option<String> {
    match name {
        "token_embd.weight" => return Some("tok_emb".into()),
        "output.weight" => return Some("lm_head".into()),
        _ => {}
    }
    let rest = name.strip_prefix("blk.")?;
    let (n, tail) = rest.split_once('.')?;
    n.parse::<usize>().ok()?;
    let canon = match tail {
        "attn_q.weight" => "wq",
        "attn_k.weight" => "wk",
        "attn_v.weight" => "wv",
        "attn_output.weight" => "wo",
        "ffn_gate.weight" => "w_gate",
        "ffn_up.weight" => "w_up",
        "ffn_down.weight" => "w_down",
        "attn_qkv.weight" => "wqkv",
        "attn_gate.weight" => "wz",
        "ssm_alpha.weight" => "wa",
        "ssm_beta.weight" => "wb",
        "ssm_out.weight" => "w_out",
        _ => return None,
    };
    Some(format!("L{n}.{canon}"))
}

/// `forge gguf-zero BASE.gguf OUT.gguf NAME[,NAME..]`: a copy of BASE in which the tensors with these canonical names (`w_down`, `wo`, `w_out`, `w_gate` ...; every layer) are filled with zero bytes. All ggml formats
/// dequantize an all-zero block to 0.0, so a recipe that takes a layer's tensor from this file removes that layer's contribution to the residual stream: the donor of the layer-ablation studies.
pub fn zero_copy(base: &Path, out: &Path, names: &[String]) -> Result<(usize, u64)> {
    use std::io::{Seek, SeekFrom, Write};
    std::fs::copy(base, out).map_err(|e| Error::msg(format!("{}: {e}", out.display())))?;
    let g = Gguf::open(base)?;
    let mut f = std::fs::OpenOptions::new().write(true).open(out).map_err(|e| Error::msg(format!("{}: {e}", out.display())))?;
    let zeros = vec![0u8; 1 << 20];
    let (mut n, mut bytes) = (0usize, 0u64);
    for t in &g.tensors {
        let Some(canon) = canon_of_gguf(&t.name) else { continue };
        let tn = canon.split_once('.').map(|(_, x)| x).unwrap_or(canon.as_str());
        if !names.iter().any(|x| x == tn) {
            continue;
        }
        let len = g.tensor_bytes(t).len();
        f.seek(SeekFrom::Start((g.data_offset + t.offset as usize) as u64)).map_err(|e| Error::msg(e.to_string()))?;
        let mut left = len;
        while left > 0 {
            let k = left.min(zeros.len());
            f.write_all(&zeros[..k]).map_err(|e| Error::msg(e.to_string()))?;
            left -= k;
        }
        n += 1;
        bytes += len as u64;
    }
    f.flush().map_err(|e| Error::msg(e.to_string()))?;
    Ok((n, bytes))
}

/// `forge gguf-stitch BASE.gguf RECIPE.json OUT.gguf`: writes a new GGUF whose tensors are those of BASE except where the recipe (FORGE_RECIPE format) takes a tensor from another file of the same
/// model. The metadata section is carried over byte for byte; the result loads in llama.cpp, LM Studio and this engine. Returns (tensors taken from other files, bytes written).
pub fn stitch(base: &Path, recipe: &Path, out: &Path) -> Result<(usize, u64)> {
    use std::io::Write;
    let g = Gguf::open(base)?;
    let n_layer = g.get_u64(&format!("{}.block_count", g.architecture().unwrap_or("qwen35"))).unwrap_or(0) as usize;
    let overrides = load_recipe_file(&recipe.to_string_lossy(), n_layer)?;
    let mut chosen: Vec<(&Gguf, usize)> = Vec::new(); // (file, tensor index in it)
    let mut taken = 0usize;
    for (i, t) in g.tensors.iter().enumerate() {
        let src = canon_of_gguf(&t.name).and_then(|canon| {
            // the rule lookup of `Source::override_gguf`, on the loaded overrides
            let (layer, tensor) = match canon.split_once('.') {
                Some((l, tn)) => (l[1..].parse::<usize>().ok(), tn),
                None => (None, canon.as_str()),
            };
            overrides.iter().find_map(|o| {
                if !o.tensors.iter().any(|x| x == tensor) {
                    return None;
                }
                if let (Some(sel), Some(l)) = (&o.layers, layer) {
                    if !sel.get(l).copied().unwrap_or(false) {
                        return None;
                    }
                }
                let ot = o.g.tensor_info(&t.name)?;
                if !o.if_dtype.is_empty() && !o.if_dtype.iter().any(|d| *d == ot.dtype.name()) {
                    return None;
                }
                Some(&o.g)
            })
        });
        match src {
            Some(og) => {
                let idx = og.tensors.iter().position(|x| x.name == t.name).unwrap();
                if og.tensors[idx].dims != t.dims {
                    return Err(Error::format(format!("{}: shape differs between the files", t.name)));
                }
                taken += 1;
                chosen.push((og, idx));
            }
            None => chosen.push((&g, i)),
        }
    }
    let align = g.alignment;
    let pad = |n: usize| (n + align - 1) / align * align;
    // tensor table
    let mut table: Vec<u8> = Vec::new();
    let mut off = 0usize;
    for (j, (src, idx)) in chosen.iter().enumerate() {
        let t = &src.tensors[*idx];
        let name = &g.tensors[j].name;
        table.extend_from_slice(&(name.len() as u64).to_le_bytes());
        table.extend_from_slice(name.as_bytes());
        table.extend_from_slice(&(t.dims.len() as u32).to_le_bytes());
        for d in &t.dims {
            table.extend_from_slice(&d.to_le_bytes());
        }
        table.extend_from_slice(&t.dtype.ggml_id().to_le_bytes());
        table.extend_from_slice(&(off as u64).to_le_bytes());
        off += pad(src.tensor_bytes(t).len());
    }
    let mut f = std::io::BufWriter::new(std::fs::File::create(out).map_err(|e| Error::msg(format!("{}: {e}", out.display())))?);
    let mut header = Vec::new();
    header.extend_from_slice(b"GGUF");
    header.extend_from_slice(&3u32.to_le_bytes());
    header.extend_from_slice(&(g.tensors.len() as u64).to_le_bytes());
    header.extend_from_slice(&(g.metadata.len() as u64).to_le_bytes());
    header.extend_from_slice(g.kv_bytes());
    header.extend_from_slice(&table);
    let data_start = pad(header.len());
    header.resize(data_start, 0);
    f.write_all(&header).map_err(|e| Error::msg(e.to_string()))?;
    let mut total = data_start as u64;
    for (src, idx) in &chosen {
        let b = src.tensor_bytes(&src.tensors[*idx]);
        f.write_all(b).map_err(|e| Error::msg(e.to_string()))?;
        let padn = pad(b.len()) - b.len();
        f.write_all(&vec![0u8; padn]).map_err(|e| Error::msg(e.to_string()))?;
        total += (b.len() + padn) as u64;
    }
    f.flush().map_err(|e| Error::msg(e.to_string()))?;
    Ok((taken, total))
}
