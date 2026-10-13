//! Training of the decoder transformer (Llama block, optional Qwen3-style QK-norm, tied embedding / output head) with hand-written forward and backward passes on our own CUDA kernels.
//!
//! Precision: fp32 master weights, gradients and optimizer state; fp16 working copy of the weights; fp32 residual stream; fp16 GEMM operands and saved activations; fp32 accumulation everywhere;
//! dynamic loss scaling (the scale multiplies the gradient of the loss, the optimizer unscales). Everything lives in flat arrays: one for the parameters, one for the gradients, two for Adam.
//!
//! One `forward_backward` call processes a micro-batch of `batch` sequences of `seq_len` tokens and ACCUMULATES gradients; `step` clips, checks for non-finite values and applies AdamW.

use forge_core::half::f32_to_f16;
use forge_core::{Error, Result};
use forge_cuda::{Args, Device, DeviceBuffer, Event, Function, Module, Stream};
use std::collections::HashMap;

pub mod decode;
pub use decode::{AttnKind, Decoder};

/// Bit 31 of a token id in a batch: the token is an input like any other, but it is not a target. The position that predicts it has no loss and no gradient (chat fine-tuning marks the prompt tokens
/// this way; the data files carry the mark in bit 15 of the u16 token and the loader moves it up here).
pub const IGNORE_TARGET: u32 = 0x8000_0000;

#[derive(Clone, Debug)]
pub struct TrainConfig {
    pub n_layer: usize,
    pub n_embd: usize,
    pub n_head: usize,
    pub n_kv: usize,
    pub head_dim: usize,
    pub n_ff: usize,
    pub vocab: usize,
    pub seq_len: usize,
    /// sequences per micro-batch
    pub batch: usize,
    pub qk_norm: bool,
    pub rope_theta: f32,
    pub eps: f32,
    /// rows of logits materialised at a time (cross entropy is computed chunk by chunk)
    pub ce_chunk: usize,
    // ---- Forge-1 structure options (all off = the plain Llama-style decoder, with the same parameter layout and numerics as before) ----
    /// separate output head (zero-initialised) instead of the tied input embedding
    pub untied: bool,
    /// ReLU squared MLP (up and down matrices) instead of SwiGLU (gate, up, down)
    pub relu2: bool,
    /// logit soft cap: logits = cap * tanh(z / cap); 0 = off
    pub softcap: f32,
    /// per-layer residual scalars: the input of layer l is a_l * (output of layer l - 1) + b_l * (input embedding)
    pub lambdas: bool,
    /// U-net skip connections: the second half of the layers also receives c_l * (output of the mirrored layer of the first half)
    pub skips: bool,
    /// value embeddings: in these layers v = lv * v + lve * ve_l[token] (a [vocab][kv_dim] table per layer)
    pub ve_layers: Vec<usize>,
    /// hashed n-gram memory: the orders (2..=4, at most three) of the token n-grams that index a table of 2^ngram_bits rows x ngram_dim; the summed rows go through a [n_embd][ngram_dim] matrix into the embedding output
    pub ngram: Vec<usize>,
    pub ngram_bits: usize,
    pub ngram_dim: usize,
    /// gated attention: the output of every head is multiplied by sigmoid(W_h . h1[..K]) before the output projection (h1 = the normalised attention input, its first K channels; one weight row per
    /// head, a [n_layer][n_head][K] tensor "agate", zero-initialised = gates of 0.5). 0 = off; K = n_embd is the head-wise gate of Qiu et al. 2025, a small K the sparse gate of modded-nanogpt.
    pub attn_gate: usize,
    /// Peri-LN (Kim et al. 2025): an RMSNorm with its own gain (initialised to 1) on the output of every attention and MLP sublayer before it is added to the residual stream: x + Norm(F(Norm(x)))
    pub peri_norm: bool,
    /// Peri-LN's optional initial embedding normalisation: the embedding output (with the n-gram memory, if any) goes through an RMSNorm with a gain (initialised to 1)
    pub emb_norm: bool,
    /// sliding-window attention: in the layers of `window_layers`, query i sees only the keys i - window < j <= i (0 = off: every layer is dense causal). Head dimension 64 only.
    pub window: usize,
    pub window_layers: Vec<usize>,
}

impl Default for TrainConfig {
    fn default() -> Self {
        TrainConfig {
            n_layer: 4,
            n_embd: 256,
            n_head: 4,
            n_kv: 2,
            head_dim: 64,
            n_ff: 768,
            vocab: 32768,
            seq_len: 2048,
            batch: 1,
            qk_norm: false,
            rope_theta: 10000.0,
            eps: 1e-5,
            ce_chunk: 2048,
            untied: false,
            relu2: false,
            softcap: 0.0,
            lambdas: false,
            skips: false,
            ve_layers: Vec::new(),
            ngram: Vec::new(),
            ngram_bits: 0,
            ngram_dim: 0,
            attn_gate: 0,
            peri_norm: false,
            emb_norm: false,
            window: 0,
            window_layers: Vec::new(),
        }
    }
}

impl TrainConfig {
    /// the input of every layer goes through the mixing step (a * x + b * x0 + c * skip)
    pub fn mixing(&self) -> bool {
        self.lambdas || self.skips
    }
    /// number of layers in the first (encoder) half: their outputs are skip sources
    pub fn n_enc(&self) -> usize {
        self.n_layer / 2
    }
    /// the attention window of layer `l` (None: dense causal)
    pub fn window_of(&self, l: usize) -> Option<usize> {
        if self.window > 0 && self.window_layers.contains(&l) {
            Some(self.window)
        } else {
            None
        }
    }
    /// `--window-layers auto`: three of every four layers windowed; every fourth layer (3, 7, 11, ...) and the last layer keep dense attention
    pub fn auto_window_layers(n_layer: usize) -> Vec<usize> {
        (0..n_layer)
            .filter(|&l| l % 4 != 3 && l + 1 != n_layer)
            .collect()
    }
    /// the layer whose output feeds layer `l` through a skip connection
    pub fn skip_source(&self, l: usize) -> Option<usize> {
        if self.skips && l >= self.n_layer - self.n_enc() {
            Some(self.n_layer - 1 - l)
        } else {
            None
        }
    }
    /// 1 for ReLU squared (up), 2 for SwiGLU (gate | up)
    pub fn gate_width(&self) -> usize {
        if self.relu2 {
            1
        } else {
            2
        }
    }
    /// a one-line description of the structure options for logs
    pub fn structure(&self) -> String {
        let mut s = Vec::new();
        if self.untied {
            s.push("untied head".to_string());
        }
        if self.relu2 {
            s.push("relu2".into());
        }
        if self.softcap > 0.0 {
            s.push(format!("softcap {}", self.softcap));
        }
        if self.lambdas {
            s.push("lambdas".into());
        }
        if self.skips {
            s.push("skips".into());
        }
        if !self.ve_layers.is_empty() {
            s.push(format!("value-emb {:?}", self.ve_layers));
        }
        if !self.ngram.is_empty() {
            s.push(format!(
                "n-gram memory {:?} (2^{} rows x {})",
                self.ngram, self.ngram_bits, self.ngram_dim
            ));
        }
        if self.attn_gate > 0 {
            s.push(format!(
                "attention gate (first {} channels)",
                self.attn_gate
            ));
        }
        if self.peri_norm {
            s.push("Peri-LN output norms".into());
        }
        if self.emb_norm {
            s.push("embedding norm".into());
        }
        if self.window > 0 && !self.window_layers.is_empty() {
            s.push(format!(
                "sliding window {} in layers {:?}",
                self.window, self.window_layers
            ));
        }
        if s.is_empty() {
            "plain".into()
        } else {
            s.join(", ")
        }
    }
}

impl TrainConfig {
    pub fn q_dim(&self) -> usize {
        self.n_head * self.head_dim
    }
    pub fn kv_dim(&self) -> usize {
        self.n_kv * self.head_dim
    }
    pub fn tokens(&self) -> usize {
        self.batch * self.seq_len
    }
}

#[derive(Clone, Debug)]
pub struct ParamInfo {
    pub name: String,
    pub shape: Vec<usize>,
    pub offset: usize,
    pub len: usize,
}

#[derive(Clone, Default)]
struct LayerOff {
    attn_norm: usize,
    wq: usize,
    wk: usize,
    wv: usize,
    wo: usize,
    q_norm: usize,
    k_norm: usize,
    ffn_norm: usize,
    w_gate: usize,
    w_up: usize,
    w_down: usize,
    /// value-embedding table of this layer (usize::MAX = none) and its (lv, lve) scalars
    ve: usize,
    ve_sc: usize,
}

struct Kernels {
    gemm_nt_f16: Function,
    gemm_nt_f32: Function,
    gemm_nt_add: Function,
    gemm_nn_f16: Function,
    gemm_nn_f32: Function,
    gemm_nn_add: Function,
    gemm_tn_add: Function,
    gemm_tn_f16: Function,
    gemm_nt_axpy: Function,
    gemm_nt_res: Function,
    gemm_nn_axpy: Function,
    /// Turing ReLU^2 epilogues (None with the Ampere GEMMs: the separate kernels are used there); the forward one is `fw_nt_relu2`
    gemm_nn_relu2_bwd: Option<Function>,
    /// the forward pass's NT GEMMs: gemm_nt_f16, gemm_nt_res and the ReLU^2 forward epilogue (gemm_nt_relu2), or with FORGE_F16ACC=P their fp16-accumulate versions (`*_hP`, Turing only)
    fw_nt_f16: Function,
    fw_nt_res: Function,
    fw_nt_relu2: Option<Function>,
    /// 128 x 144 tile version (FORGE_GEMM_W144, Turing only) of gemm_nn_f32 for widths that are a multiple of 144 (the data gradients into the residual width)
    w144: Option<Function>,
    ce_reg: Function,
    relu2: Function,
    relu2_bwd: Function,
    ce_cap: Function,
    mix_fwd: Function,
    mix_bwd: Function,
    rmsnorm_mix_bwd: Function,
    mix_rmsnorm_fwd: Function,
    add_to_f32_f16: Function,
    add_f32: Function,
    ve_fwd: Function,
    ve_bwd: Function,
    ngram_idx: Function,
    ngram_gather: Function,
    ngram_scatter: Function,
    muon_prep: Function,
    muon_normalize: Function,
    muon_apply: Function,
    // Muon options (train_opt.cu)
    muon_normalize_m: Function,
    normuon_lanes: Function,
    normuon_ratio: Function,
    muon_apply_opt: Function,
    // gated attention (train_opt.cu)
    attn_gate_fwd: Function,
    attn_gate_mul: Function,
    attn_gate_bwd: Function,
    attn_gate_dw: Function,
    attn_gate_dx: Function,
    // Peri-LN (train_opt.cu)
    rmsnorm_res_fwd_h: Function,
    rmsnorm_res_fwd_f: Function,
    rmsnorm_bwd_h: Function,
    adamw_cwd: Function,
    // document masking (flash_doc.cu)
    flash_doc_64: Function,
    flash_doc_128: Function,
    flash_doc_dkdv_64: Function,
    flash_doc_dq_64: Function,
    embed_batch: Function,
    rmsnorm_fwd: Function,
    qkv_rope_train: Function,
    flash_train_64: Function,
    flash_train_128: Function,
    swiglu2: Function,
    swiglu2_bwd: Function,
    ce: Function,
    mean: Function,
    rmsnorm_bwd: Function,
    reduce_partials: Function,
    f32_to_f16: Function,
    fill_f32: Function,
    embed_bwd: Function,
    attn_bwd_prep: Function,
    attn_dkdv: Function,
    attn_dq: Function,
    flash_dkdv_64: Function,
    flash_dq_64: Function,
    /// second-generation training attention (flash_train.cu): prefetching forward, longest-first dQ, sliding-window variants
    flash_fwd_t64: Function,
    flash_fwd_t64_win: Function,
    flash_dkdv_64_win: Function,
    flash_dq_t64: Function,
    flash_dq_64_win: Function,
    flash_dkdv_ds_64: Function,
    flash_dq_ds_64: Function,
    qkv_bwd: Function,
    grad_sumsq: Function,
    grad_finalize: Function,
    adamw: Function,
}

struct LayerActs {
    x: DeviceBuffer,      // fp32 [N][E] layer input
    out: DeviceBuffer, // fp32 [N][E] layer output (only with mixing: the next layer's input is a mix of it; otherwise 1 element)
    xmid: DeviceBuffer, // fp32 [N][E]
    rstd1: DeviceBuffer, // [N]
    rstd2: DeviceBuffer, // [N]
    rstd_q: DeviceBuffer, // [N * n_head]
    rstd_k: DeviceBuffer, // [N * n_kv]
    qkv: DeviceBuffer, // fp16 raw projections [N][qd + 2 kvd]
    q: DeviceBuffer,   // fp16 rotated [N][qd]
    k: DeviceBuffer,   // fp16 rotated [N][kvd]
    v: DeviceBuffer,   // fp16 [N][kvd]
    o: DeviceBuffer,   // fp16 [N][qd]
    lse: DeviceBuffer, // fp32 [N * n_head]
    gu: DeviceBuffer,  // fp16 [N][2F]: gate | up
    h1: DeviceBuffer, // fp16 [N][E] normalised input of the attention block (saved for the weight gradient)
    h2: DeviceBuffer, // fp16 [N][E] normalised input of the MLP
    act: DeviceBuffer, // fp16 [N][F] silu(gate) * up
    gate: DeviceBuffer, // fp32 [N][n_head] attention gates (1 element when the gate is off)
    y1: DeviceBuffer, // Peri-LN: fp16 [N][E] attention sublayer output before its norm (1 element when off)
    y2: DeviceBuffer, // Peri-LN: fp16 [N][E] MLP sublayer output before its norm
    rstd_p1: DeviceBuffer,
    rstd_p2: DeviceBuffer,
}

pub struct StepStats {
    pub grad_norm: f32,
    pub found_inf: bool,
}

/// Which lanes NorMuon normalises (the second moment is kept per lane).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NorMuonAxis {
    /// lanes along the longer dimension: the rows of a square or tall matrix, the columns of a wide one (modded-nanogpt's rule; for an exactly orthogonal update only these norms differ)
    Long,
    /// always the rows, i.e. the output neurons (the NorMuon paper's reference code)
    Rows,
}

/// Options of the Muon update. `MuonOpts::default()` is the update as it always was (Jordan's quintic, five iterations, no NorMuon, plain decay): the trainer then launches exactly the kernels of before.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MuonOpts {
    /// Polar Express (Amsel et al. 2025, arXiv 2505.16932): per-iteration minimax coefficients (`polar_express_coeffs`) and the input normalised by 1.02 |X|_F + 1e-6, instead of Jordan's fixed
    /// quintic (3.4445, -4.7750, 2.0315) after |X|_F + 1e-7
    pub polar: bool,
    /// iterations of the orthogonalisation (default 5; Polar Express has tables for 3 to 7)
    pub iters: usize,
    /// NorMuon (Li et al. 2025, arXiv 2510.05491): after the orthogonalisation every lane is divided by the root of a running average (decay `normuon_beta2`) of its mean square, then the matrix is
    /// rescaled to its norm before the division (so the step size of the update is unchanged). The running averages live in the second-moment array of the matrix (unused by Muon), so they are
    /// part of every checkpoint.
    pub normuon: Option<NorMuonAxis>,
    pub normuon_beta2: f32,
    /// cautious weight decay (Chen et al. 2025, arXiv 2510.12402) on the Muon matrices: an entry decays only where the update has its sign (moves it towards zero)
    pub cautious_wd: bool,
}

impl Default for MuonOpts {
    fn default() -> Self {
        MuonOpts {
            polar: false,
            iters: 5,
            normuon: None,
            normuon_beta2: 0.95,
            cautious_wd: false,
        }
    }
}

impl MuonOpts {
    /// one line for logs
    pub fn describe(&self) -> String {
        let mut s = vec![format!(
            "{} x{}",
            if self.polar {
                "Polar Express"
            } else {
                "Newton-Schulz (Jordan)"
            },
            self.iters
        )];
        if let Some(a) = self.normuon {
            s.push(format!(
                "NorMuon ({}, beta2 {})",
                if a == NorMuonAxis::Long {
                    "lanes along the longer dimension"
                } else {
                    "rows"
                },
                self.normuon_beta2
            ));
        }
        if self.cautious_wd {
            s.push("cautious weight decay".into());
        }
        s.join(", ")
    }
}

/// Jordan's quintic Newton-Schulz coefficients (the same for every iteration)
pub const NS_JORDAN: (f32, f32, f32) = (3.4445, -4.7750, 2.0315);

/// Polar Express coefficients for `iters` iterations (3 to 7): optimal_composition(l = 1e-3, iters, safety 2e-2, cushion 0.02) of the authors' reference code (tools/ref_muon.py regenerates them; the
/// five-iteration set is the one of modded-nanogpt record 38). The first iterations are the same for every length; only the last one, which has no safety factor, differs.
pub fn polar_express_coeffs(iters: usize) -> Option<Vec<(f32, f32, f32)>> {
    // the greedy sequence of polynomials with the safety factor (every iteration but the last uses these, in order)
    const SAFE: [(f32, f32, f32); 6] = [
        (8.156554524902461, -22.48329292557794, 15.878769915207442),
        (4.042929935166736, -2.8089174659087117, 0.500017845105131),
        (3.891667802292663, -2.772484153217688, 0.5060648178503397),
        (3.285753657755654, -2.3681294933425376, 0.464490242330031),
        (2.300530711627095, -1.6111665557258386, 0.38333744275452697),
        (1.8631210546382566, -1.2042160621002678, 0.342187956052336),
    ];
    // the last polynomial (no safety factor) of a composition of `iters` iterations
    let last: (f32, f32, f32) = match iters {
        3 => (3.9695011583385162, -2.942182363267836, 0.558736450563465),
        4 => (3.351468730910767, -2.5130779633710483, 0.5128347598303034),
        5 => (2.3465413258596373, -1.709782838268706, 0.42323551169305224),
        6 => (1.9003834757310218, -1.2779237188293011, 0.3778031533636295),
        7 => (1.87502235952885, -1.2500248436955763, 0.3750024843773982),
        _ => return None,
    };
    let mut v: Vec<(f32, f32, f32)> = SAFE[..iters - 1].to_vec();
    v.push(last);
    Some(v)
}

/// Scratch of one Muon update in flight: the fp32 update, two fp16 Newton-Schulz iterates, the Gram and polynomial matrices, the norm partials, and NorMuon's lane statistics
/// ([lane_pow | gain] of `lanes` floats each, then the norm-restoring factor).
struct MuonScratch {
    u32: DeviceBuffer,
    x: [DeviceBuffer; 2],
    a: DeviceBuffer,
    b: DeviceBuffer,
    partial: DeviceBuffer,
    nm: DeviceBuffer,
    lanes: usize,
}

impl MuonScratch {
    fn new(n: usize, m: usize, lanes: usize) -> Result<MuonScratch> {
        Ok(MuonScratch {
            u32: alloc_f32(n)?,
            x: [alloc_f16(n)?, alloc_f16(n)?],
            a: alloc_f16(m * m)?,
            b: alloc_f16(m * m)?,
            partial: alloc_f32(64)?,
            nm: alloc_f32(2 * lanes + 32)?,
            lanes,
        })
    }
}

/// Kernel-level options of the training step. Defaults: everything on. Environment switches (read once when the trainer is built):
///   FORGE_SPLITK=N   split the reduction of every weight-gradient GEMM into N slices on Turing (1 = never split; default: chosen per shape from the block slots of the device)
///   FORGE_NO_FUSE=1  ReLU^2 and its backward as separate kernels instead of GEMM epilogues
///   FORGE_CE_OLD=1   the three-pass cross-entropy kernel instead of the register-resident one
///   FORGE_ATTN_OLD=1 the first-generation attention kernels (flash.cu / flash_bwd.cu) for the dense layers
///   FORGE_ATTN_DS=0/1 dense attention backward with the dS spill (dK dV stores its fp16 dS tiles, dQ is one matmul per tile: bit-identical, ~38 MB of extra memory per sequence); default: on for 34-SM devices (2060 SUPER), off elsewhere until measured
///   FORGE_NO_FUSE_MIX=1 the layer-input mixing and the attention norm as separate kernels (forward and backward)
///   FORGE_GEMM_E2=0  the Turing GEMMs with fp16 outputs or the fp32 residual epilogue with their original epilogue instead of the staged one (store_acc_e2: 128-bit stores of whole sectors through
///                    shared memory; bit-identical results)
///   FORGE_GEMM_W144=0 no 128 x 144 tiles for the fp32 data-gradient GEMMs whose width is a multiple of 144 (Forge-1's 576: 4 column tiles instead of 4.5; bit-identical results); default: used
///                    when they need fewer waves of the block slots (2060 SUPER yes, T4 no); =force: always
///   FORGE_F16ACC=P   the forward NT GEMMs (qkv, attention output, MLP up and down) accumulate in fp16 and add into fp32 every P k8 MMA steps, P = 2, 4, 8 or 16 (Turing only: GeForce cards run fp16
///                    accumulation at twice the rate of fp32); 0 or unset: the fp32-accumulate kernels. Not bit-identical (error ~2x the fp16 output rounding at P = 8: tests/f16acc.rs)
#[derive(Clone, Copy, Debug)]
struct KOpts {
    split_k: Option<usize>,
    fuse_relu2: bool,
    ce_reg: bool,
    new_attn: bool,
    fuse_mix: bool,
    f16acc: usize,
    gemm_e2: bool,
    /// dS spill attention backward: None = by device (on for 34 SMs), Some = forced
    attn_ds: Option<bool>,
    /// 0 off, 1 when it saves a wave (default), 2 always (tests)
    w144: u8,
}

impl KOpts {
    fn from_env() -> KOpts {
        let on = |name: &str| {
            std::env::var(name)
                .map(|v| v != "0" && !v.is_empty())
                .unwrap_or(false)
        };
        KOpts {
            split_k: std::env::var("FORGE_SPLITK")
                .ok()
                .and_then(|v| v.parse().ok())
                .map(|v: usize| v.max(1)),
            fuse_relu2: !on("FORGE_NO_FUSE"),
            ce_reg: !on("FORGE_CE_OLD"),
            new_attn: !on("FORGE_ATTN_OLD"),
            fuse_mix: !on("FORGE_NO_FUSE_MIX"),
            f16acc: std::env::var("FORGE_F16ACC")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0),
            attn_ds: std::env::var("FORGE_ATTN_DS").ok().map(|v| v != "0" && !v.is_empty()),
            gemm_e2: std::env::var("FORGE_GEMM_E2")
                .map(|v| v != "0")
                .unwrap_or(true),
            w144: match std::env::var("FORGE_GEMM_W144").as_deref() {
                Ok("0") => 0,
                Ok("force") => 2,
                _ => 1,
            },
        }
    }
}

/// Slices of the reduction of a weight-gradient GEMM on Turing (two 128 x 128 blocks per SM): the count that minimises the time of the busiest SM, at most 8, every slice a multiple of 16 deep and at least
/// 256 deep. Model: a block is (k tiles of its slice + 24) units of work (the 24: prologue, epilogue and atomics); an SM runs its blocks in pairs at full rate, a block alone at 0.55 of it. Calibrated on a T4
/// (kbench, job kernel-opt-job1b: it picks the measured best factor for 7 of the 8 Forge-1 shapes at 4096 and 8192 tokens, the 8th within 6 %). Few output tiles and a long reduction (576 x 576 over 8192
/// tokens: 25 tiles) split; the vocabulary head (1280 tiles) does not.
pub fn split_k_turing(tiles: usize, k: usize, sm_count: usize) -> usize {
    // 34 SMs (RTX 2060 SUPER): refitted to kbench on that card at 2048 and 4096 tokens (the T4 constants split too little there: 576 x 576 at 4096 tokens 0.168 -> 0.145 ms, 960 x 576 at 2048
    // tokens 0.157 -> 0.127 ms); every other SM count keeps the T4 calibration
    let (overhead_ktiles, alone) = if sm_count == 34 { (6.0, 0.7) } else { (24.0, 0.55) };
    let sms = sm_count.max(1);
    let mut best = (f64::MAX, 1usize);
    for gz in 1..=8usize {
        if k % (16 * gz) != 0 || (gz > 1 && k / gz < 256) {
            continue;
        }
        let w = (k / gz / 16) as f64 + overhead_ktiles;
        let per_sm = (tiles * gz + sms - 1) / sms;
        let cost = (per_sm / 2 * 2) as f64 * w + if per_sm % 2 == 1 { w / alone } else { 0.0 };
        if cost < best.0 - 1e-9 {
            best = (cost, gz);
        }
    }
    best.1
}

pub struct Trainer {
    pub cfg: TrainConfig,
    dev: Device,
    _module: Module,
    k: Kernels,
    stream: Stream,
    pub params: Vec<ParamInfo>,
    index: HashMap<String, usize>,
    emb_off: usize,
    out_norm_off: usize,
    head_off: usize,
    ng_off: Vec<usize>,
    ngw_off: usize,
    lam_off: usize,
    velam_off: usize,
    gate_off: usize,
    pnorm_off: usize,
    enorm_off: usize,
    layer_off: Vec<LayerOff>,
    pub n_params: usize,
    decay_end: usize,
    master: DeviceBuffer,
    grads: DeviceBuffer,
    adam_m: DeviceBuffer,
    adam_v: DeviceBuffer,
    work16: DeviceBuffer,
    acts: Vec<LayerActs>,
    x_final: DeviceBuffer,
    // scratch
    hf16: DeviceBuffer,
    rstd_f: DeviceBuffer,
    z: DeviceBuffer,
    dz: DeviceBuffer,
    loss_rows: DeviceBuffer,
    loss: DeviceBuffer,
    tokens: DeviceBuffer,
    targets: DeviceBuffer,
    cos: DeviceBuffer,
    sin: DeviceBuffer,
    dh32: DeviceBuffer,
    dx: DeviceBuffer,
    dxn: DeviceBuffer,
    dy16: DeviceBuffer,
    da16: DeviceBuffer,
    dgu16: DeviceBuffer,
    do16: DeviceBuffer,
    dq32: DeviceBuffer,
    /// dS spill of the dense attention backward (FORGE_ATTN_DS): fp16 64 x 64 tiles, `batch * n_head * nt (nt + 1) / 2` of them, nt = seq_len / 64
    ds_buf: Option<DeviceBuffer>,
    dk32: DeviceBuffer,
    dv32: DeviceBuffer,
    dqkv16: DeviceBuffer,
    dvec: DeviceBuffer,
    partial: DeviceBuffer,
    sumsq_partial: DeviceBuffer,
    nonfinite_partial: DeviceBuffer,
    ts: DeviceBuffer,
    // Forge-1 structure options: the embedding output kept for the per-layer mixing, its gradient, the gradients coming back through the skip connections and the partial sums of the scalar gradients
    x0: DeviceBuffer,
    dx0: DeviceBuffer,
    dskip: Vec<DeviceBuffer>,
    sc_partial: DeviceBuffer,
    // n-gram memory: the table row of every position per order, the summed rows (fp16) and their gradient
    ng_idx: Vec<DeviceBuffer>,
    ng_m: DeviceBuffer,
    ng_dm: DeviceBuffer,
    lr_mults: (f32, f32, f32, f32),
    /// AdamW learning-rate multiplier of the attention gate weights
    gate_lr_mult: f32,
    /// cautious weight decay in AdamW (the decayed AdamW parameters: the embedding, and with plain AdamW the matrices too)
    adam_cwd: bool,
    /// document masking: the end-of-text token id (a query sees no key before the start of its document); the document start of every row of the micro-batch (int32 [N])
    doc_eos: Option<u32>,
    dstart: DeviceBuffer,
    // attention gate scratch: the gated head outputs (fp16 [N][q_dim]), the gradients of the gate logits ([N][n_head]) and the per-block partial sums of the weight gradient
    og16: DeviceBuffer,
    gdpre: DeviceBuffer,
    gpart: DeviceBuffer,
    // Peri-LN: the embedding output before its norm (fp32 [N][E]), its rstd, and the gradient scratch of the extra norms (fp32 [N][E])
    xe: DeviceBuffer,
    rstd_e: DeviceBuffer,
    dpost: DeviceBuffer,
    /// set by `logits_at`: stop after the final norm and compute the logits of this one row
    logits_row: Option<usize>,
    /// set by `logits_at`: run the forward pass over this many positions per sequence instead of the full `seq_len` (generation cost follows the context length)
    t_override: Option<usize>,
    /// set by `set_window`: training windows of this many positions; a micro-batch then holds batch * seq_len / window rows (the same number of tokens)
    win: Option<usize>,
    /// the Ampere GEMM kernels are in use (dynamic shared memory, K a multiple of 32)
    s80: bool,
    /// built by `new_inference`: no gradients, no Adam state
    inference: bool,
    /// the fp32 master weights and the fp16 working copy were freed after `Decoder::enable_q8` made its Q8 copy: only that decoder can still run the model
    weights_released: bool,
    /// streaming multiprocessors of the device (the weight-gradient GEMMs split their long reduction when they would not fill them)
    sm_count: usize,
    /// kernel-level options of the step (all on by default; environment switches for A/B runs and as a fallback)
    kopt: KOpts,
    // Muon: the matrices it updates (offset, rows, cols) and one scratch per stream: the matrices are updated on several streams at once (their small GEMMs fill few SMs each: 4 to 90 blocks);
    // stream 0 is the trainer's stream, the others wait for the gradient norm and the main stream waits for them before the step ends (FORGE_MUON_STREAMS=1: all on the main stream)
    muon_mats: Vec<(usize, usize, usize)>,
    mu: Vec<MuonScratch>,
    mu_streams: Vec<Stream>,
    mu_done: Vec<Event>,
    mu_start: Event,
    /// options of the Muon update (default: as it always was)
    muon_opts: MuonOpts,
    pub step_count: usize,
}

/// The flat parameter layout (matrices that get weight decay first, then the norm weights). Pure: no GPU needed (used by the trainer and by the exporter).
pub struct Layout {
    pub params: Vec<ParamInfo>,
    pub emb_off: usize,
    layer_off: Vec<LayerOff>,
    pub decay_end: usize,
    pub out_norm_off: usize,
    /// the separate output head (= emb_off when the head is tied)
    pub head_off: usize,
    /// the n-gram memory tables (one per order) and the matrix that projects their sum into the embedding output
    pub ng_off: Vec<usize>,
    pub ngw_off: usize,
    /// per-layer (a, b, c) mixing scalars [n_layer][3] (0 when the mixing is off)
    pub lam_off: usize,
    /// (lv, lve) of the value-embedding layers, in the order of `ve_layers`
    pub velam_off: usize,
    /// the attention gate weights [n_layer][n_head][attn_gate] (0 when the gate is off)
    pub gate_off: usize,
    /// Peri-LN gains: [n_layer][2][n_embd] (attention, MLP) and the embedding norm's gain [n_embd] (0 when off)
    pub pnorm_off: usize,
    pub enorm_off: usize,
    pub n_params: usize,
    pub index: HashMap<String, usize>,
}

/// Is this parameter updated by Muon (a 2-D hidden matrix)? The embedding-like tables and the scalar groups are not.
fn is_muon_matrix(name: &str, shape: &[usize]) -> bool {
    shape.len() == 2
        && name != "emb"
        && name != "head"
        && !name.starts_with("ve")
        && !name.starts_with("ng")
        && name != "lam"
        && name != "velam"
}

pub fn param_layout(c: &TrainConfig) -> Layout {
    let (e, qd, kvd, ff, v) = (c.n_embd, c.q_dim(), c.kv_dim(), c.n_ff, c.vocab);
    let mut params: Vec<ParamInfo> = Vec::new();
    let mut off = 0usize;
    let push =
        |params: &mut Vec<ParamInfo>, off: &mut usize, name: String, shape: Vec<usize>| -> usize {
            let len = shape.iter().product();
            let o = *off;
            params.push(ParamInfo {
                name,
                shape,
                offset: o,
                len,
            });
            *off += len;
            o
        };
    let emb_off = push(&mut params, &mut off, "emb".into(), vec![v, e]);
    let mut layer_off = vec![LayerOff::default(); c.n_layer];
    for l in 0..c.n_layer {
        let lo = &mut layer_off[l];
        lo.wq = push(&mut params, &mut off, format!("l{l}.wq"), vec![qd, e]);
        lo.wk = push(&mut params, &mut off, format!("l{l}.wk"), vec![kvd, e]);
        lo.wv = push(&mut params, &mut off, format!("l{l}.wv"), vec![kvd, e]);
        lo.wo = push(&mut params, &mut off, format!("l{l}.wo"), vec![e, qd]);
        if !c.relu2 {
            lo.w_gate = push(&mut params, &mut off, format!("l{l}.w_gate"), vec![ff, e]);
        }
        lo.w_up = push(&mut params, &mut off, format!("l{l}.w_up"), vec![ff, e]);
        lo.w_down = push(&mut params, &mut off, format!("l{l}.w_down"), vec![e, ff]);
        lo.ve = usize::MAX;
    }
    let decay_end = off;
    for l in 0..c.n_layer {
        let lo = &mut layer_off[l];
        lo.attn_norm = push(&mut params, &mut off, format!("l{l}.attn_norm"), vec![e]);
        lo.ffn_norm = push(&mut params, &mut off, format!("l{l}.ffn_norm"), vec![e]);
        if c.qk_norm {
            lo.q_norm = push(
                &mut params,
                &mut off,
                format!("l{l}.q_norm"),
                vec![c.head_dim],
            );
            lo.k_norm = push(
                &mut params,
                &mut off,
                format!("l{l}.k_norm"),
                vec![c.head_dim],
            );
        }
    }
    let out_norm_off = push(&mut params, &mut off, "out_norm".into(), vec![e]);
    // Forge-1 additions, after everything of the plain layout (the plain layout is therefore unchanged): head, value-embedding tables, scalar groups
    let head_off = if c.untied {
        push(&mut params, &mut off, "head".into(), vec![v, e])
    } else {
        emb_off
    };
    for &l in &c.ve_layers {
        layer_off[l].ve = push(&mut params, &mut off, format!("ve{l}"), vec![v, kvd]);
    }
    let mut ng_off = Vec::new();
    for &o in &c.ngram {
        ng_off.push(push(
            &mut params,
            &mut off,
            format!("ng{o}"),
            vec![1usize << c.ngram_bits, c.ngram_dim],
        ));
    }
    let ngw_off = if c.ngram.is_empty() {
        0
    } else {
        push(&mut params, &mut off, "ngw".into(), vec![e, c.ngram_dim])
    };
    let lam_off = if c.mixing() {
        push(&mut params, &mut off, "lam".into(), vec![c.n_layer, 3])
    } else {
        0
    };
    let velam_off = if c.ve_layers.is_empty() {
        0
    } else {
        push(
            &mut params,
            &mut off,
            "velam".into(),
            vec![c.ve_layers.len(), 2],
        )
    };
    let gate_off = if c.attn_gate == 0 {
        0
    } else {
        push(
            &mut params,
            &mut off,
            "agate".into(),
            vec![c.n_layer, c.n_head, c.attn_gate],
        )
    };
    // the norm kernels read gains as float4: the Peri-LN gains start at a multiple of four floats (the padding before them belongs to no parameter and stays 0)
    if c.peri_norm || c.emb_norm {
        off = (off + 3) & !3;
    }
    let pnorm_off = if c.peri_norm {
        push(&mut params, &mut off, "pnorm".into(), vec![c.n_layer, 2, e])
    } else {
        0
    };
    let enorm_off = if c.emb_norm {
        push(&mut params, &mut off, "enorm".into(), vec![e])
    } else {
        0
    };
    for (k, &l) in c.ve_layers.iter().enumerate() {
        layer_off[l].ve_sc = velam_off + 2 * k;
    }
    let n_params = off;
    let index = params
        .iter()
        .enumerate()
        .map(|(i, p)| (p.name.clone(), i))
        .collect();
    Layout {
        params,
        emb_off,
        layer_off,
        decay_end,
        out_norm_off,
        head_off,
        ng_off,
        ngw_off,
        lam_off,
        velam_off,
        gate_off,
        pnorm_off,
        enorm_off,
        n_params,
        index,
    }
}

/// Where a trained parameter lives in a Hugging Face style checkpoint.
fn hf_name(name: &str) -> String {
    match name {
        "emb" => "model.embed_tokens.weight".into(),
        "out_norm" => "model.norm.weight".into(),
        n => {
            let (l, rest) = n.split_once('.').unwrap();
            let sub = match rest {
                "wq" => "self_attn.q_proj",
                "wk" => "self_attn.k_proj",
                "wv" => "self_attn.v_proj",
                "wo" => "self_attn.o_proj",
                "q_norm" => "self_attn.q_norm",
                "k_norm" => "self_attn.k_norm",
                "attn_norm" => "input_layernorm",
                "ffn_norm" => "post_attention_layernorm",
                "w_gate" => "mlp.gate_proj",
                "w_up" => "mlp.up_proj",
                "w_down" => "mlp.down_proj",
                other => other,
            };
            format!("model.layers.{}.{}.weight", &l[1..], sub)
        }
    }
}

/// The model configuration of a checkpoint header (batch 1, sequence length as trained); the structure options of older checkpoints default to off.
fn config_from_header(h: &forge_core::Json, ce_chunk: usize) -> Result<TrainConfig> {
    let g = |k: &str| {
        h["cfg"][k].as_usize().ok_or_else(|| {
            Error::msg(format!(
                "checkpoint header lacks cfg.{k} (written by an older version)"
            ))
        })
    };
    let ve_layers: Vec<usize> = h["cfg"]["ve_layers"]
        .as_array()
        .map(|a| a.iter().filter_map(|v| v.as_usize()).collect())
        .unwrap_or_default();
    Ok(TrainConfig {
        n_layer: g("n_layer")?,
        n_embd: g("n_embd")?,
        n_head: g("n_head")?,
        n_kv: g("n_kv")?,
        head_dim: g("head_dim")?,
        n_ff: g("n_ff")?,
        vocab: g("vocab")?,
        seq_len: g("seq_len")?,
        batch: 1,
        qk_norm: h["cfg"]["qk_norm"].as_bool().unwrap_or(false),
        rope_theta: h["cfg"]["rope_theta"].as_f64().unwrap_or(10000.0) as f32,
        eps: h["cfg"]["eps"].as_f64().unwrap_or(1e-5) as f32,
        ce_chunk,
        untied: h["cfg"]["untied"].as_bool().unwrap_or(false),
        relu2: h["cfg"]["relu2"].as_bool().unwrap_or(false),
        softcap: h["cfg"]["softcap"].as_f64().unwrap_or(0.0) as f32,
        lambdas: h["cfg"]["lambdas"].as_bool().unwrap_or(false),
        skips: h["cfg"]["skips"].as_bool().unwrap_or(false),
        ve_layers,
        ngram: h["cfg"]["ngram"]
            .as_array()
            .map(|a| a.iter().filter_map(|v| v.as_usize()).collect())
            .unwrap_or_default(),
        ngram_bits: h["cfg"]["ngram_bits"].as_usize().unwrap_or(0),
        ngram_dim: h["cfg"]["ngram_dim"].as_usize().unwrap_or(0),
        attn_gate: h["cfg"]["attn_gate"].as_usize().unwrap_or(0),
        peri_norm: h["cfg"]["peri_norm"].as_bool().unwrap_or(false),
        emb_norm: h["cfg"]["emb_norm"].as_bool().unwrap_or(false),
        window: h["cfg"]["window"].as_usize().unwrap_or(0),
        window_layers: h["cfg"]["window_layers"]
            .as_array()
            .map(|a| a.iter().filter_map(|v| v.as_usize()).collect())
            .unwrap_or_default(),
    })
}

/// The model configuration stored in a checkpoint header (batch 1, sequence length as trained).
pub fn config_from_checkpoint(ckpt: &std::path::Path) -> Result<TrainConfig> {
    use std::io::BufRead;
    let mut f = std::io::BufReader::new(std::fs::File::open(ckpt)?);
    let mut line = String::new();
    f.read_line(&mut line)?;
    config_from_header(&forge_core::Json::parse(line.trim())?, 2048)
}

/// The parameter section of a checkpoint as one flat host array (no GPU needed).
pub fn read_checkpoint_params(ckpt: &std::path::Path) -> Result<Vec<f32>> {
    use std::io::{BufRead, Read};
    let mut f = std::io::BufReader::new(std::fs::File::open(ckpt)?);
    let mut line = String::new();
    f.read_line(&mut line)?;
    let h = forge_core::Json::parse(line.trim())?;
    let n = h["n_params"]
        .as_usize()
        .filter(|_| h["magic"].as_str() == Some("forgeckpt1"))
        .ok_or_else(|| Error::msg("not a forge checkpoint"))?;
    let mut raw = vec![0u8; n * 4];
    f.read_exact(&mut raw)?;
    Ok(raw
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

/// Convert a training checkpoint into a Hugging Face style directory (model.safetensors in fp16 or fp32 + config.json) that `forge_model::Model::load` and the GPU engine read.
pub fn export_hf(
    ckpt: &std::path::Path,
    out_dir: &std::path::Path,
    fp16: bool,
) -> Result<TrainConfig> {
    use std::io::{BufRead, Read, Write};
    let mut f = std::io::BufReader::new(std::fs::File::open(ckpt)?);
    let mut line = String::new();
    f.read_line(&mut line)?;
    let h = forge_core::Json::parse(line.trim())?;
    let cfg = config_from_header(&h, 1)?;
    if cfg.structure() != "plain" {
        return Err(Error::msg(format!("export_hf writes the plain Llama-style layout; this checkpoint has Forge-1 structure options ({})", cfg.structure())));
    }
    let lay = param_layout(&cfg);
    if h["n_params"].as_usize() != Some(lay.n_params) {
        return Err(Error::msg(
            "checkpoint size does not match its configuration",
        ));
    }
    let mut raw = vec![0u8; lay.n_params * 4];
    f.read_exact(&mut raw)?;
    let vals: Vec<f32> = raw
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    std::fs::create_dir_all(out_dir)?;
    let esize = if fp16 { 2 } else { 4 };
    let mut header = Vec::new();
    let mut data_off = 0usize;
    for p in &lay.params {
        let shape: Vec<String> = p.shape.iter().map(|d| d.to_string()).collect();
        header.push(format!(
            "\"{}\":{{\"dtype\":\"{}\",\"shape\":[{}],\"data_offsets\":[{},{}]}}",
            hf_name(&p.name),
            if fp16 { "F16" } else { "F32" },
            shape.join(","),
            data_off,
            data_off + p.len * esize
        ));
        data_off += p.len * esize;
    }
    let header_json = format!("{{{}}}", header.join(","));
    let mut out =
        std::io::BufWriter::new(std::fs::File::create(out_dir.join("model.safetensors"))?);
    out.write_all(&(header_json.len() as u64).to_le_bytes())?;
    out.write_all(header_json.as_bytes())?;
    for p in &lay.params {
        let slice = &vals[p.offset..p.offset + p.len];
        if fp16 {
            let bytes: Vec<u8> = slice
                .iter()
                .flat_map(|v| f32_to_f16(*v).to_le_bytes())
                .collect();
            out.write_all(&bytes)?;
        } else {
            let bytes: Vec<u8> = slice.iter().flat_map(|v| v.to_le_bytes()).collect();
            out.write_all(&bytes)?;
        }
    }
    out.flush()?;
    let model_type = if cfg.qk_norm { "qwen3" } else { "llama" };
    let config = format!(
        "{{\"model_type\":\"{}\",\"hidden_size\":{},\"num_hidden_layers\":{},\"num_attention_heads\":{},\"num_key_value_heads\":{},\"head_dim\":{},\"intermediate_size\":{},\"vocab_size\":{},\"rms_norm_eps\":{},\"rope_theta\":{},\"max_position_embeddings\":{},\"tie_word_embeddings\":true}}\n",
        model_type, cfg.n_embd, cfg.n_layer, cfg.n_head, cfg.n_kv, cfg.head_dim, cfg.n_ff, cfg.vocab, cfg.eps, cfg.rope_theta, cfg.seq_len
    );
    std::fs::write(out_dir.join("config.json"), config)?;
    Ok(cfg)
}

/// dynamic shared memory of the Ampere GEMM kernels (gemm_s80.cu: three stages of two 128 x 136 half tiles)
const GEMM80_SMEM: u32 = 3 * 2 * 32 * 136 * 2;

const RB_BLOCKS: usize = 128; // blocks of the RMSNorm backward (partial dw buffers)
const QB_BLOCKS: usize = 128; // blocks of the q/k backward (partial q-norm / k-norm weight gradients)
const SUMSQ_BLOCKS: usize = 256;
const GATE_BLOCKS: usize = 128; // blocks of the attention gate's weight gradient (partial sums)

fn alloc_f32(n: usize) -> Result<DeviceBuffer> {
    DeviceBuffer::alloc(n * 4)
}
fn alloc_f16(n: usize) -> Result<DeviceBuffer> {
    DeviceBuffer::alloc(n * 2)
}

impl Trainer {
    pub fn new(cfg: TrainConfig, device_ordinal: usize) -> Result<Trainer> {
        Trainer::build(cfg, device_ordinal, false)
    }

    /// A trainer for generation and evaluation only: the gradients and the Adam state are not allocated (a 125M model needs about 1 GB instead of 2.5 GB for its parameters), a checkpoint loads only its
    /// parameters, and `forward_backward` / `step` are not available.
    pub fn new_inference(cfg: TrainConfig, device_ordinal: usize) -> Result<Trainer> {
        Trainer::build(cfg, device_ordinal, true)
    }

    fn build(cfg: TrainConfig, device_ordinal: usize, inference: bool) -> Result<Trainer> {
        let dev = Device::open(device_ordinal)?;
        let (image, _) = forge_kernels::image_for(dev.info.cc_major, dev.info.cc_minor)
            .ok_or_else(|| Error::msg("no GPU kernels for this device"))?;
        let module = Module::load(image)?;
        let f = |n: &str| module.function(n);
        // GEMM variants: the Ampere kernels (cp.async pipeline, m16n8k16: `*_s80`) on sm_80 and newer, the Turing ones elsewhere; FORGE_NO_S80=1 forces the Turing ones, FORGE_S80=1 the Ampere source with its
        // emulation of the new instructions on any device (a correctness test on Turing). They need K to be a multiple of 32 in every GEMM: checked below, the Turing kernels are used otherwise.
        let mut s80 = (forge_kernels::has_s80(dev.info.cc_major)
            && std::env::var("FORGE_NO_S80").is_err())
            || std::env::var("FORGE_S80").is_ok();
        {
            let c = &cfg;
            let dims = [
                c.n_embd,
                c.q_dim(),
                c.kv_dim(),
                c.n_ff,
                c.n_ff * c.gate_width(),
                c.q_dim() + 2 * c.kv_dim(),
                c.vocab,
                c.tokens(),
                c.ngram_dim,
            ];
            if s80 && dims.iter().any(|d| d % 32 != 0 && *d != 0) {
                eprintln!(
                    "[train] a GEMM dimension is not a multiple of 32: the Ampere GEMMs are off"
                );
                s80 = false;
            }
        }
        let g = |n: &str| -> Result<Function> {
            if s80 {
                let func = module.function(&format!("{n}_s80"))?;
                func.set_max_dynamic_smem(GEMM80_SMEM as i32)?;
                Ok(func)
            } else {
                module.function(n)
            }
        };
        // FORGE_F16ACC=P: fp16-accumulate forward GEMMs (`gemm_nt_*_hP`, Turing kernels only)
        let hacc = if s80 { 0 } else { KOpts::from_env().f16acc };
        if ![0, 2, 4, 8, 16].contains(&hacc) {
            return Err(Error::unsupported("FORGE_F16ACC must be 0, 2, 4, 8 or 16 (k8 MMA steps between the flushes into fp32)"));
        }
        // FORGE_GEMM_E2 (default on): the gemm3 kernels with the staged epilogue (`*_e2`, Turing kernels only; bit-identical) for the fp16 outputs and the fp32 residual epilogue. Not for the plain fp32
        // outputs (head logits, the data gradients into dh32): measured slower there (kbench --only e2 on the 2060 SUPER: head 0.40x, 4096 x 576 fp32 0.97-0.98x)
        let e2 = !s80 && KOpts::from_env().gemm_e2;
        let fe = |n: &str| -> Result<Function> {
            if e2 {
                f(&format!("{n}_e2"))
            } else {
                f(n)
            }
        };
        let ge = |n: &str| -> Result<Function> {
            if e2 {
                f(&format!("{n}_e2"))
            } else {
                g(n)
            }
        };
        let fw = |n: &str| -> Result<Function> {
            if hacc > 0 {
                f(&format!("{n}_h{hacc}"))
            } else {
                ge(n)
            }
        };
        let k = Kernels {
            gemm_nt_f16: ge("gemm_nt_f16")?,
            gemm_nt_f32: g("gemm_nt_f32")?,
            gemm_nt_add: g("gemm_nt_add")?,
            gemm_nn_f16: ge("gemm_nn_f16")?,
            gemm_nn_f32: g("gemm_nn_f32")?,
            gemm_nn_add: g("gemm_nn_add")?,
            gemm_tn_add: g("gemm_tn_add")?,
            gemm_tn_f16: g("gemm_tn_f16")?,
            gemm_nt_axpy: g("gemm_nt_axpy")?,
            gemm_nt_res: ge("gemm_nt_res")?,
            gemm_nn_axpy: g("gemm_nn_axpy")?,
            gemm_nn_relu2_bwd: if s80 {
                None
            } else {
                Some(fe("gemm_nn_relu2_bwd")?)
            },
            fw_nt_f16: fw("gemm_nt_f16")?,
            fw_nt_res: fw("gemm_nt_res")?,
            fw_nt_relu2: if s80 {
                None
            } else if hacc > 0 {
                Some(f(&format!("gemm_nt_relu2_h{hacc}"))?)
            } else {
                Some(fe("gemm_nt_relu2")?)
            },
            // only the fp32 data gradients: 4096 x 576 over K = 2304 / 960: 1.05-1.06x on the 2060 SUPER (kbench --only e2); the residual forward (attention output 0.85x, MLP down 1.02x) and the fp16
            // data gradient (1.03x, the staged epilogue gives 1.05x) stay on the 128 x 128 kernels
            w144: if s80 || KOpts::from_env().w144 == 0 {
                None
            } else {
                let w = f("gemm_nn_f32_w144")?;
                // 17.5 KB of shared memory per block: two blocks per SM need the 64 KB configuration
                w.set_smem_carveout(100)?;
                Some(w)
            },
            ce_reg: f("ce_reg_kernel")?,
            relu2: f("relu2_kernel")?,
            relu2_bwd: f("relu2_bwd_kernel")?,
            ce_cap: f("ce_cap_kernel")?,
            mix_fwd: f("mix_fwd_kernel")?,
            mix_bwd: f("mix_bwd_kernel")?,
            rmsnorm_mix_bwd: f("rmsnorm_mix_bwd_kernel")?,
            mix_rmsnorm_fwd: f("mix_rmsnorm_fwd_kernel")?,
            add_to_f32_f16: f("add_to_f32_f16_kernel")?,
            add_f32: f("add_f32_kernel")?,
            ve_fwd: f("ve_fwd_kernel")?,
            ve_bwd: f("ve_bwd_kernel")?,
            ngram_idx: f("ngram_idx_kernel")?,
            ngram_gather: f("ngram_gather_kernel")?,
            ngram_scatter: f("ngram_scatter_kernel")?,
            muon_prep: f("muon_prep_kernel")?,
            muon_normalize: f("muon_normalize_kernel")?,
            muon_apply: f("muon_apply_kernel")?,
            muon_normalize_m: f("muon_normalize_m_kernel")?,
            normuon_lanes: f("normuon_lanes_kernel")?,
            normuon_ratio: f("normuon_ratio_kernel")?,
            muon_apply_opt: f("muon_apply_opt_kernel")?,
            attn_gate_fwd: f("attn_gate_fwd_kernel")?,
            attn_gate_mul: f("attn_gate_mul_kernel")?,
            attn_gate_bwd: f("attn_gate_bwd_kernel")?,
            attn_gate_dw: f("attn_gate_dw_kernel")?,
            attn_gate_dx: f("attn_gate_dx_kernel")?,
            rmsnorm_res_fwd_h: f("rmsnorm_res_fwd_h_kernel")?,
            rmsnorm_res_fwd_f: f("rmsnorm_res_fwd_f_kernel")?,
            rmsnorm_bwd_h: f("rmsnorm_bwd_h_kernel")?,
            adamw_cwd: f("adamw_cwd_kernel")?,
            flash_doc_64: f("flash_fwd_train_doc_64")?,
            flash_doc_128: f("flash_fwd_train_doc_128")?,
            flash_doc_dkdv_64: f("flash_bwd_dkdv_doc_64")?,
            flash_doc_dq_64: f("flash_bwd_dq_doc_64")?,
            embed_batch: f("embed_batch_kernel")?,
            rmsnorm_fwd: f("rmsnorm_fwd_kernel")?,
            qkv_rope_train: f("qkv_rope_train_kernel")?,
            flash_train_64: f("flash_fwd_train_64")?,
            flash_train_128: f("flash_fwd_train_128")?,
            swiglu2: f("swiglu2_kernel")?,
            swiglu2_bwd: f("swiglu2_bwd_kernel")?,
            ce: f("ce_kernel")?,
            mean: f("mean_kernel")?,
            rmsnorm_bwd: f("rmsnorm_bwd_kernel")?,
            reduce_partials: f("reduce_partials_kernel")?,
            f32_to_f16: f("f32_to_f16_kernel")?,
            fill_f32: f("fill_f32_kernel")?,
            embed_bwd: f("embed_bwd_kernel")?,
            attn_bwd_prep: f("attn_bwd_prep_kernel")?,
            attn_dkdv: f("attn_bwd_dkdv_naive_kernel")?,
            attn_dq: f("attn_bwd_dq_naive_kernel")?,
            flash_dkdv_64: f("flash_bwd_dkdv_64")?,
            flash_dq_64: f("flash_bwd_dq_64")?,
            flash_fwd_t64: f("flash_fwd_t64")?,
            flash_fwd_t64_win: f("flash_fwd_t64_win")?,
            flash_dkdv_64_win: f("flash_bwd_dkdv_64_win")?,
            flash_dq_t64: f("flash_bwd_dq_t64")?,
            flash_dq_64_win: f("flash_bwd_dq_64_win")?,
            flash_dkdv_ds_64: f("flash_bwd_dkdv_ds_64")?,
            flash_dq_ds_64: f("flash_bwd_dq_ds_64")?,
            qkv_bwd: f("qkv_bwd_kernel")?,
            grad_sumsq: f("grad_sumsq_kernel")?,
            grad_finalize: f("grad_finalize_kernel")?,
            adamw: f("adamw_kernel")?,
        };
        let c = &cfg;
        if c.head_dim != 64 && c.head_dim != 128 {
            return Err(Error::unsupported("head_dim must be 64 or 128"));
        }
        if c.n_embd > 1024 {
            return Err(Error::unsupported(
                "the RMSNorm kernels support widths up to 1024",
            ));
        }
        if c.seq_len % 32 != 0
            || c.n_embd % 32 != 0
            || c.n_ff % 32 != 0
            || c.vocab % 32 != 0
            || c.q_dim() % 8 != 0
        {
            return Err(Error::unsupported(
                "sequence length, widths and vocabulary must be multiples of 32",
            ));
        }
        let (e, qd, kvd, ff, v) = (c.n_embd, c.q_dim(), c.kv_dim(), c.n_ff, c.vocab);
        if c.skips && c.n_layer < 2 {
            return Err(Error::unsupported(
                "skip connections need at least two layers",
            ));
        }
        if c.ve_layers.iter().any(|&l| l >= c.n_layer)
            || c.ve_layers.windows(2).any(|w| w[0] >= w[1])
        {
            return Err(Error::unsupported(
                "value-embedding layers must be distinct, increasing layer indices",
            ));
        }
        if c.softcap < 0.0 {
            return Err(Error::unsupported(
                "the logit soft cap must not be negative",
            ));
        }
        if c.attn_gate > c.n_embd || (c.attn_gate > 0 && c.n_head > 32) {
            return Err(Error::unsupported(
                "the attention gate reads at most n_embd channels and supports up to 32 heads",
            ));
        }
        if c.window > 0 && !c.window_layers.is_empty() {
            if c.head_dim != 64 || c.seq_len % 64 != 0 {
                return Err(Error::unsupported("sliding-window attention needs head dimension 64 and a sequence length that is a multiple of 64"));
            }
            if c.window_layers.iter().any(|&l| l >= c.n_layer) {
                return Err(Error::unsupported(
                    "window layers must be layer indices of the model",
                ));
            }
        }
        if !c.ngram.is_empty() {
            if c.ngram.len() > 3
                || c.ngram.iter().any(|&o| !(2..=4).contains(&o))
                || c.ngram.windows(2).any(|w| w[0] >= w[1])
            {
                return Err(Error::unsupported("the n-gram memory takes up to three distinct, increasing orders between 2 and 4"));
            }
            if c.ngram_bits < 4 || c.ngram_bits > 24 || c.ngram_dim == 0 || c.ngram_dim % 32 != 0 {
                return Err(Error::unsupported("the n-gram memory needs 4 to 24 index bits and a row width that is a multiple of 32"));
            }
        }
        let Layout {
            params,
            emb_off,
            layer_off,
            decay_end,
            out_norm_off,
            head_off,
            ng_off,
            ngw_off,
            lam_off,
            velam_off,
            gate_off,
            pnorm_off,
            enorm_off,
            n_params,
            index,
        } = param_layout(c);
        let n = c.tokens();
        let mix = c.mixing();
        let gw = c.gate_width();
        let ce = c.ce_chunk.min(n);
        if n % ce != 0 {
            return Err(Error::unsupported(
                "tokens per micro-batch must be a multiple of the cross-entropy chunk",
            ));
        }
        let half = c.head_dim / 2;
        let max_pos = c.seq_len;
        let mut cos = vec![0f32; max_pos * half];
        let mut sin = vec![0f32; max_pos * half];
        for p in 0..max_pos {
            for i in 0..half {
                let a =
                    p as f64 * (c.rope_theta as f64).powf(-((2 * i) as f64) / c.head_dim as f64);
                cos[p * half + i] = a.cos() as f32;
                sin[p * half + i] = a.sin() as f32;
            }
        }
        let mut acts = Vec::new();
        let n_acts = if inference && !c.skips {
            c.n_layer.min(2)
        } else {
            c.n_layer
        };
        for _ in 0..n_acts {
            acts.push(LayerActs {
                x: alloc_f32(n * e)?,
                out: alloc_f32(if mix { n * e } else { 1 })?,
                xmid: alloc_f32(n * e)?,
                rstd1: alloc_f32(n)?,
                rstd2: alloc_f32(n)?,
                rstd_q: alloc_f32(if c.qk_norm { n * c.n_head } else { 1 })?,
                rstd_k: alloc_f32(if c.qk_norm { n * c.n_kv } else { 1 })?,
                qkv: alloc_f16(n * (qd + 2 * kvd))?,
                q: alloc_f16(n * qd)?,
                k: alloc_f16(n * kvd)?,
                v: alloc_f16(n * kvd)?,
                o: alloc_f16(n * qd)?,
                lse: alloc_f32(n * c.n_head)?,
                gu: alloc_f16(n * gw * ff)?,
                h1: alloc_f16(n * e)?,
                h2: alloc_f16(n * e)?,
                act: alloc_f16(n * ff)?,
                gate: alloc_f32(if c.attn_gate > 0 { n * c.n_head } else { 1 })?,
                y1: alloc_f16(if c.peri_norm { n * e } else { 1 })?,
                y2: alloc_f16(if c.peri_norm { n * e } else { 1 })?,
                rstd_p1: alloc_f32(if c.peri_norm { n } else { 1 })?,
                rstd_p2: alloc_f32(if c.peri_norm { n } else { 1 })?,
            });
        }
        let muon_mats: Vec<(usize, usize, usize)> = params
            .iter()
            .filter(|p| is_muon_matrix(&p.name, &p.shape))
            .map(|p| (p.offset, p.shape[0], p.shape[1]))
            .collect();
        let mu_n = muon_mats.iter().map(|m| m.1 * m.2).max().unwrap_or(1);
        let mu_m = muon_mats.iter().map(|m| m.1.min(m.2)).max().unwrap_or(1);
        let nm_lanes = muon_mats.iter().map(|m| m.1.max(m.2)).max().unwrap_or(1);
        let sm_count_dev = dev.info.sm_count.max(1) as usize;
        let n_mu = if inference || muon_mats.is_empty() {
            1
        } else {
            std::env::var("FORGE_MUON_STREAMS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(4usize)
                .clamp(1, 8)
        };
        let t = Trainer {
            dev,
            _module: module,
            k,
            stream: Stream::new()?,
            params,
            index,
            emb_off,
            out_norm_off,
            head_off,
            ng_off,
            ngw_off,
            lam_off,
            velam_off,
            gate_off,
            pnorm_off,
            enorm_off,
            layer_off,
            n_params,
            decay_end,
            master: alloc_f32(n_params)?,
            grads: alloc_f32(if inference { 1 } else { n_params })?,
            adam_m: alloc_f32(if inference { 1 } else { n_params })?,
            adam_v: alloc_f32(if inference { 1 } else { n_params })?,
            work16: alloc_f16(n_params)?,
            acts,
            x_final: alloc_f32(n * e)?,
            hf16: alloc_f16(n * e)?,
            rstd_f: alloc_f32(n)?,
            z: alloc_f32(ce * v)?,
            dz: alloc_f16(ce * v)?,
            loss_rows: alloc_f32(n)?,
            loss: alloc_f32(1)?,
            tokens: DeviceBuffer::alloc(n * 4)?,
            targets: DeviceBuffer::alloc(n * 4)?,
            cos: {
                let b = alloc_f32(cos.len())?;
                b.upload_f32(&cos)?;
                b
            },
            sin: {
                let b = alloc_f32(sin.len())?;
                b.upload_f32(&sin)?;
                b
            },
            dh32: alloc_f32(n * e)?,
            dx: alloc_f32(n * e)?,
            dxn: alloc_f32(n * e)?,
            dy16: alloc_f16(n * e)?,
            da16: alloc_f16(n * ff)?,
            dgu16: alloc_f16(n * gw * ff)?,
            do16: alloc_f16(n * qd)?,
            dq32: alloc_f32(n * qd)?,
            ds_buf: if !inference
                && !s80
                && c.head_dim == 64
                && c.seq_len % 64 == 0
                && KOpts::from_env().new_attn
                && KOpts::from_env().attn_ds.unwrap_or(sm_count_dev == 34)
            {
                let nt = c.seq_len / 64;
                Some(alloc_f16(c.batch * c.n_head * (nt * (nt + 1) / 2) * 4096)?)
            } else {
                None
            },
            dk32: alloc_f32(n * kvd)?,
            dv32: alloc_f32(n * kvd)?,
            dqkv16: alloc_f16(n * (qd + 2 * kvd))?,
            dvec: alloc_f32(n * c.n_head)?,
            partial: alloc_f32((RB_BLOCKS * e).max(2 * QB_BLOCKS * c.head_dim))?,
            sumsq_partial: alloc_f32(SUMSQ_BLOCKS)?,
            nonfinite_partial: alloc_f32(SUMSQ_BLOCKS)?,
            ts: alloc_f32(16)?,
            x0: alloc_f32(if mix { n * e } else { 1 })?,
            dx0: alloc_f32(if mix { n * e } else { 1 })?,
            dskip: {
                let mut d = Vec::new();
                if c.skips {
                    for _ in 0..c.n_enc() {
                        d.push(alloc_f32(n * e)?);
                    }
                }
                d
            },
            sc_partial: alloc_f32(3 * 256)?,
            ng_idx: {
                let mut d = Vec::new();
                for _ in 0..c.ngram.len() {
                    d.push(DeviceBuffer::alloc(n * 4)?);
                }
                d
            },
            ng_m: alloc_f16(if c.ngram.is_empty() {
                1
            } else {
                n * c.ngram_dim
            })?,
            ng_dm: alloc_f16(if c.ngram.is_empty() {
                1
            } else {
                n * c.ngram_dim
            })?,
            lr_mults: (1.0, 1.0, 1.0, 1.0),
            gate_lr_mult: 1.0,
            adam_cwd: false,
            doc_eos: None,
            dstart: DeviceBuffer::alloc(n * 4)?,
            og16: alloc_f16(if c.attn_gate > 0 { n * qd } else { 1 })?,
            gdpre: alloc_f32(if c.attn_gate > 0 { n * c.n_head } else { 1 })?,
            gpart: alloc_f32(if c.attn_gate > 0 {
                GATE_BLOCKS * c.n_head * c.attn_gate
            } else {
                1
            })?,
            xe: alloc_f32(if c.emb_norm { n * e } else { 1 })?,
            rstd_e: alloc_f32(if c.emb_norm { n } else { 1 })?,
            dpost: alloc_f32(if c.peri_norm || c.emb_norm { n * e } else { 1 })?,
            logits_row: None,
            t_override: None,
            win: None,
            s80,
            inference,
            weights_released: false,
            sm_count: sm_count_dev,
            kopt: KOpts::from_env(),
            muon_mats,
            mu: (0..n_mu)
                .map(|_| MuonScratch::new(mu_n, mu_m, nm_lanes))
                .collect::<Result<Vec<_>>>()?,
            mu_streams: (1..n_mu)
                .map(|_| Stream::new())
                .collect::<Result<Vec<_>>>()?,
            mu_done: (1..n_mu)
                .map(|_| Event::new())
                .collect::<Result<Vec<_>>>()?,
            mu_start: Event::new()?,
            muon_opts: MuonOpts::default(),
            step_count: 0,
            cfg,
        };
        t.adam_m.zero(&t.stream)?;
        t.adam_v.zero(&t.stream)?;
        t.grads.zero(&t.stream)?;
        t.stream.synchronize()?;
        Ok(t)
    }

    /// Total bytes of device memory this trainer allocated (estimate for planning).
    pub fn device_info(&self) -> String {
        format!(
            "{} ({} parameters{})",
            self.dev.info.name,
            self.n_params,
            if self.s80 { ", Ampere GEMMs" } else { "" }
        )
    }

    pub fn param(&self, name: &str) -> Option<&ParamInfo> {
        self.index.get(name).map(|&i| &self.params[i])
    }

    /// Initialise the parameters on the host from named tensors (missing ones keep 0) and refresh the fp16 working copy.
    pub fn set_params(&mut self, named: &HashMap<String, Vec<f32>>) -> Result<()> {
        let mut flat = vec![0f32; self.n_params];
        for p in &self.params {
            if let Some(v) = named.get(&p.name) {
                if v.len() != p.len {
                    return Err(Error::msg(format!(
                        "parameter {} has {} values, expected {}",
                        p.name,
                        v.len(),
                        p.len
                    )));
                }
                flat[p.offset..p.offset + p.len].copy_from_slice(v);
            }
        }
        self.master.upload_f32(&flat)?;
        self.refresh_work()
    }

    fn refresh_work(&self) -> Result<()> {
        let n = self.n_params;
        self.k.f32_to_f16.launch(
            (((n + 255) / 256).min(65535) as u32, 1, 1),
            (256, 1, 1),
            0,
            &self.stream,
            &Args::new()
                .buf(&self.master)
                .buf(&self.work16)
                .u64(n as u64),
        )?;
        self.stream.synchronize()
    }

    pub fn params_named(&self) -> Result<HashMap<String, Vec<f32>>> {
        let flat = self.master.download_f32(self.n_params)?;
        Ok(self
            .params
            .iter()
            .map(|p| (p.name.clone(), flat[p.offset..p.offset + p.len].to_vec()))
            .collect())
    }

    /// The accumulated gradients multiplied by `unscale` (pass 1 / loss_scale).
    pub fn grads_named(&self, unscale: f32) -> Result<HashMap<String, Vec<f32>>> {
        let flat = self.grads.download_f32(self.n_params)?;
        Ok(self
            .params
            .iter()
            .map(|p| {
                (
                    p.name.clone(),
                    flat[p.offset..p.offset + p.len]
                        .iter()
                        .map(|v| v * unscale)
                        .collect(),
                )
            })
            .collect())
    }

    /// The accumulated (loss-scaled) gradients of all parameters as one flat host array (data-parallel training sums them over the devices).
    pub fn download_grads(&self) -> Result<Vec<f32>> {
        self.grads.download_f32(self.n_params)
    }

    /// Replace the gradients by a flat host array (the sum over the devices).
    pub fn upload_grads(&self, flat: &[f32]) -> Result<()> {
        if flat.len() != self.n_params {
            return Err(Error::msg("gradient array has the wrong length"));
        }
        self.grads.upload_f32(flat)
    }

    pub fn zero_grads(&self) -> Result<()> {
        self.grads.zero(&self.stream)
    }

    /// Random initialisation (GPT-2 style): matrices N(0, 0.02), output projections scaled by 1/sqrt(2 L), norm weights 1.
    pub fn init_random(&mut self, seed: u64) -> Result<()> {
        let mut rng = Lcg(seed.max(1));
        let std = 0.02f32;
        let resid = std / ((2 * self.cfg.n_layer) as f32).sqrt();
        let mut named: HashMap<String, Vec<f32>> = HashMap::new();
        for p in &self.params {
            let v: Vec<f32> = if p.name.ends_with("_norm") {
                vec![1.0; p.len]
            } else if p.name == "head" {
                vec![0.0; p.len] // the untied head starts at zero (all logits equal)
            } else if p.name == "lam" {
                (0..p.len)
                    .map(|i| if i % 3 == 0 { 1.0 } else { 0.0 })
                    .collect() // (a, b, c) = (1, 0, 0): the plain residual stream at the start
            } else if p.name == "velam" {
                vec![1.0; p.len] // (lv, lve) = (1, 1)
            } else if p.name == "pnorm" || p.name == "enorm" {
                vec![1.0; p.len] // Peri-LN gains start at 1 (as in the paper)
            } else if p.name == "agate" {
                vec![0.0; p.len] // gates of 0.5 at the start (as modded-nanogpt's attention gate)
            } else {
                let s = if p.name.ends_with(".wo") || p.name.ends_with(".w_down") {
                    resid
                } else {
                    std
                };
                (0..p.len).map(|_| rng.normal() * s).collect()
            };
            named.insert(p.name.clone(), v);
        }
        self.set_params(&named)
    }

    /// The activation buffers of layer `l` (an inference trainer shares two sets between all layers).
    fn act(&self, l: usize) -> &LayerActs {
        &self.acts[l % self.acts.len()]
    }

    fn w16(&self, off: usize) -> u64 {
        self.work16.ptr() + (off * 2) as u64
    }
    fn g32(&self, off: usize) -> u64 {
        self.grads.ptr() + (off * 4) as u64
    }
    fn m32(&self, off: usize) -> u64 {
        self.master.ptr() + (off * 4) as u64
    }

    /// The 128 x 144 tile version of a GEMM whose width is a multiple of 144 (FORGE_GEMM_W144, bit-identical) when it needs fewer waves of the device's block slots (two per SM), and the tile width
    /// of the function to launch. 4096 x 576: 160 blocks of 128 x 128 are 2.35 waves on a 2060 SUPER (68 slots: the 128 blocks of 128 x 144 are 1.9 waves, measured 1.05-1.06x) but exactly 2 waves on
    /// a T4 (80 slots: 128 x 144 measured 0.80-0.87x there).
    fn w144_for(&self, f: Function, m: usize, n: usize) -> (Function, usize) {
        match self.k.w144 {
            Some(w) if n % 144 == 0 && f.id() == self.k.gemm_nn_f32.id() => {
                let slots = 2 * self.sm_count.max(1);
                let rows = (m + 127) / 128;
                let waves = |blocks: usize| ((blocks + slots - 1) / slots) as f64;
                if self.kopt.w144 == 2
                    || waves(rows * (n / 144)) * 1.125 < waves(rows * ((n + 127) / 128))
                {
                    (w, 144)
                } else {
                    (f, 128)
                }
            }
            _ => (f, 128),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn gemm(
        &self,
        f: Function,
        a: u64,
        b: u64,
        c: u64,
        m: usize,
        n: usize,
        kk: usize,
        lda: usize,
        ldb: usize,
        ldc: usize,
    ) -> Result<()> {
        self.gemm_on(&self.stream, f, a, b, c, m, n, kk, lda, ldb, ldc)
    }

    #[allow(clippy::too_many_arguments)]
    fn gemm_on(
        &self,
        st: &Stream,
        f: Function,
        a: u64,
        b: u64,
        c: u64,
        m: usize,
        n: usize,
        kk: usize,
        lda: usize,
        ldb: usize,
        ldc: usize,
    ) -> Result<()> {
        let (f, bn) = self.w144_for(f, m, n);
        let (gx, gy) = ((n + bn - 1) / bn, (m + 127) / 128);
        // split-K: a weight-gradient GEMM (C += A^T B over the tokens) with few output tiles and a long reduction is cut into slices that add into C with atomics
        let mut gz = 1usize;
        if self.s80 && f.id() == self.k.gemm_tn_add.id() {
            while gz < 8
                && gx * gy * gz * 10 < self.sm_count * 2 * 9
                && kk % (32 * gz * 2) == 0
                && kk / (gz * 2) >= 1024
            {
                gz *= 2;
            }
        } else if !self.s80 && f.id() == self.k.gemm_tn_add.id() {
            gz = match self.kopt.split_k {
                Some(s) if kk % (16 * s) == 0 => s,
                Some(_) => 1,
                None => split_k_turing(gx * gy, kk, self.sm_count),
            };
        }
        let grid = (gx as u32, gy as u32, gz as u32);
        f.launch(
            grid,
            (256, 1, 1),
            if self.s80 { GEMM80_SMEM } else { 0 },
            st,
            &Args::new()
                .ptr(a)
                .ptr(b)
                .ptr(c)
                .i32(m as i32)
                .i32(n as i32)
                .i32(kk as i32)
                .i32(lda as i32)
                .i32(ldb as i32)
                .i32(ldc as i32)
                .f32(1.0),
        )
    }

    /// GEMM with the axpy epilogue: C = alpha * (A B) + beta * R (fp16 everywhere), R with row stride `ldr`.
    #[allow(clippy::too_many_arguments)]
    fn gemm_axpy(
        &self,
        f: Function,
        a: u64,
        b: u64,
        c: u64,
        m: usize,
        n: usize,
        kk: usize,
        lda: usize,
        ldb: usize,
        ldc: usize,
        alpha: f32,
        r: u64,
        ldr: usize,
        beta: f32,
    ) -> Result<()> {
        self.gemm_axpy_on(
            &self.stream,
            f,
            a,
            b,
            c,
            m,
            n,
            kk,
            lda,
            ldb,
            ldc,
            alpha,
            r,
            ldr,
            beta,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn gemm_axpy_on(
        &self,
        st: &Stream,
        f: Function,
        a: u64,
        b: u64,
        c: u64,
        m: usize,
        n: usize,
        kk: usize,
        lda: usize,
        ldb: usize,
        ldc: usize,
        alpha: f32,
        r: u64,
        ldr: usize,
        beta: f32,
    ) -> Result<()> {
        let (f, bn) = self.w144_for(f, m, n);
        let grid = (((n + bn - 1) / bn) as u32, ((m + 127) / 128) as u32, 1);
        f.launch(
            grid,
            (256, 1, 1),
            if self.s80 { GEMM80_SMEM } else { 0 },
            st,
            &Args::new()
                .ptr(a)
                .ptr(b)
                .ptr(c)
                .i32(m as i32)
                .i32(n as i32)
                .i32(kk as i32)
                .i32(lda as i32)
                .i32(ldb as i32)
                .i32(ldc as i32)
                .f32(alpha)
                .ptr(r)
                .i32(ldr as i32)
                .f32(beta),
        )
    }

    fn rmsnorm_fwd(
        &self,
        x: &DeviceBuffer,
        w_off: usize,
        h: &DeviceBuffer,
        rstd: &DeviceBuffer,
    ) -> Result<()> {
        let c = &self.cfg;
        let n = self
            .t_override
            .map(|t| c.batch * t)
            .unwrap_or_else(|| c.tokens());
        self.k.rmsnorm_fwd.launch(
            (((n + 7) / 8) as u32, 1, 1),
            (256, 1, 1),
            0,
            &self.stream,
            &Args::new()
                .buf(x)
                .ptr(self.m32(w_off))
                .buf(h)
                .buf(rstd)
                .i32(n as i32)
                .i32(c.n_embd as i32)
                .f32(c.eps),
        )
    }

    /// dx_out = dres + rmsnorm_backward(dy, x) (and its fp16 copy in `dx16`); the weight gradient is accumulated into `grads[w_off..]`
    #[allow(clippy::too_many_arguments)]
    fn rmsnorm_bwd(
        &self,
        dy: &DeviceBuffer,
        x: &DeviceBuffer,
        w_off: usize,
        rstd: &DeviceBuffer,
        dres: Option<&DeviceBuffer>,
        dx_out: &DeviceBuffer,
        dx16: Option<&DeviceBuffer>,
    ) -> Result<()> {
        let c = &self.cfg;
        let n = c.tokens();
        let blocks = RB_BLOCKS.min((n + 7) / 8);
        self.k.rmsnorm_bwd.launch(
            (blocks as u32, 1, 1),
            (256, 1, 1),
            (8 * c.n_embd * 4) as u32,
            &self.stream,
            &Args::new()
                .buf(dy)
                .buf(x)
                .ptr(self.m32(w_off))
                .buf(rstd)
                .ptr(dres.map(|b| b.ptr()).unwrap_or(0))
                .buf(dx_out)
                .ptr(dx16.map(|b| b.ptr()).unwrap_or(0))
                .buf(&self.partial)
                .i32(n as i32)
                .i32(c.n_embd as i32),
        )?;
        self.k.reduce_partials.launch(
            (((c.n_embd + 255) / 256) as u32, 1, 1),
            (256, 1, 1),
            0,
            &self.stream,
            &Args::new()
                .buf(&self.partial)
                .i32(blocks as i32)
                .i32(c.n_embd as i32)
                .ptr(self.g32(w_off)),
        )
    }

    /// rmsnorm_bwd with an fp16 norm input (the saved Peri-LN sublayer output); no residual term
    fn rmsnorm_bwd_h(
        &self,
        dy: &DeviceBuffer,
        x16: &DeviceBuffer,
        w_off: usize,
        rstd: &DeviceBuffer,
        dx_out: &DeviceBuffer,
        dx16: &DeviceBuffer,
    ) -> Result<()> {
        let c = &self.cfg;
        let n = c.tokens();
        let blocks = RB_BLOCKS.min((n + 7) / 8);
        self.k.rmsnorm_bwd_h.launch(
            (blocks as u32, 1, 1),
            (256, 1, 1),
            (8 * c.n_embd * 4) as u32,
            &self.stream,
            &Args::new()
                .buf(dy)
                .buf(x16)
                .ptr(self.m32(w_off))
                .buf(rstd)
                .ptr(0)
                .buf(dx_out)
                .buf(dx16)
                .buf(&self.partial)
                .i32(n as i32)
                .i32(c.n_embd as i32),
        )?;
        self.k.reduce_partials.launch(
            (((c.n_embd + 255) / 256) as u32, 1, 1),
            (256, 1, 1),
            0,
            &self.stream,
            &Args::new()
                .buf(&self.partial)
                .i32(blocks as i32)
                .i32(c.n_embd as i32)
                .ptr(self.g32(w_off)),
        )
    }

    fn f32_to_f16(&self, src: &DeviceBuffer, dst: &DeviceBuffer, n: usize) -> Result<()> {
        self.k.f32_to_f16.launch(
            (((n + 255) / 256).min(65535) as u32, 1, 1),
            (256, 1, 1),
            0,
            &self.stream,
            &Args::new().buf(src).buf(dst).u64(n as u64),
        )
    }

    /// Forward and backward of one micro-batch. `tokens` holds `batch` rows of `seq_len + 1` ids (inputs are the first seq_len of every row, targets the last seq_len).
    /// Gradients are accumulated (scaled by `loss_scale / tokens_per_step`). Returns the mean cross-entropy of this micro-batch.
    pub fn forward_backward(
        &mut self,
        tokens: &[u32],
        loss_scale: f32,
        tokens_per_step: usize,
    ) -> Result<f32> {
        self.run(tokens, loss_scale, tokens_per_step, true)
    }

    /// Forward only: per-position negative log-likelihood of the next token (B * T values; position j of a row is -log p(tokens[j + 1] | tokens[..= j])).
    pub fn token_nll(&mut self, tokens: &[u32]) -> Result<Vec<f32>> {
        self.run(tokens, 1.0, self.cfg.tokens(), false)?;
        self.loss_rows.download_f32(self.cfg.tokens())
    }

    /// Forward only, for generation: the logits (after the soft cap, if the model has one) that follow the first `len` tokens of `ids` (len <= seq_len; the rest of the window is padding and cannot influence
    /// earlier positions). Row 0 of the micro-batch is used; with batch > 1 the other rows are padding too.
    pub fn logits_at(&mut self, ids: &[u32], len: usize) -> Result<Vec<f32>> {
        let (t_max, b) = (self.cfg.seq_len, self.cfg.batch);
        assert!(len >= 1 && len <= t_max && ids.len() >= len);
        // the window grows with the context in steps of 64 positions: the cost of a token follows the length of the text, not the allocated window
        let t = ((len + 63) / 64 * 64).min(t_max);
        let mut window = vec![0u32; b * (t + 1)];
        window[..len].copy_from_slice(&ids[..len]);
        self.logits_row = Some(len - 1);
        self.t_override = Some(t);
        let r = self.run(&window, 1.0, b * t, false);
        self.logits_row = None;
        self.t_override = None;
        r?;
        let mut z = self.z.download_f32(self.cfg.vocab)?;
        if self.cfg.softcap > 0.0 {
            let cap = self.cfg.softcap;
            for x in z.iter_mut() {
                *x = cap * (*x / cap).tanh();
            }
        }
        Ok(z)
    }

    /// Forward only: the mean cross-entropy of a micro-batch (no gradients touched).
    pub fn eval_loss(&mut self, tokens: &[u32]) -> Result<f32> {
        self.run(tokens, 1.0, self.cfg.tokens(), false)
    }

    /// Sequence-length schedule: from now on a micro-batch is `batch * seq_len / w` rows of `w` positions (`tokens` passed to `forward_backward` / `eval_loss` / `token_nll` then hold that many rows
    /// of w + 1 ids); None goes back to `batch` rows of `seq_len`. RoPE has no learned positions, so the same parameters serve every window length. w: a multiple of 64 that divides batch * seq_len.
    pub fn set_window(&mut self, w: Option<usize>) -> Result<()> {
        if let Some(w) = w {
            let n = self.cfg.tokens();
            if w == 0 || w % 64 != 0 || w > self.cfg.seq_len || n % w != 0 {
                return Err(Error::unsupported("a training window must be a multiple of 64, at most seq_len, and divide the tokens of a micro-batch"));
            }
        }
        self.win = w.filter(|&w| w != self.cfg.seq_len);
        Ok(())
    }

    /// the training window set by `set_window` (None = seq_len)
    pub fn window(&self) -> Option<usize> {
        self.win
    }

    fn run(
        &mut self,
        tokens: &[u32],
        loss_scale: f32,
        tokens_per_step: usize,
        with_grad: bool,
    ) -> Result<f32> {
        if with_grad && self.inference {
            return Err(Error::msg(
                "this trainer was built for inference (Trainer::new_inference): no gradients",
            ));
        }
        if self.weights_released {
            return Err(Error::msg(
                "the fp32 and fp16 weights of this trainer were released for the Q8 decoder (Decoder::enable_q8): only that decoder can run the model",
            ));
        }
        let c = self.cfg.clone();
        // rows and positions of this micro-batch: generation (t_override) pads row 0 of the usual batch; a training window keeps the number of tokens and changes the shape
        let (b, t) = match (self.t_override, self.win) {
            (Some(t), _) => (c.batch, t),
            (None, Some(w)) => (c.tokens() / w, w),
            (None, None) => (c.batch, c.seq_len),
        };
        let (e, qd, kvd, ff, v) = (c.n_embd, c.q_dim(), c.kv_dim(), c.n_ff, c.vocab);
        let n = b * t;
        assert_eq!(tokens.len(), b * (t + 1));
        let mut inp = Vec::with_capacity(n * 4);
        let mut tgt = Vec::with_capacity(n * 4);
        let mut n_valid = 0usize;
        for r in 0..b {
            for i in 0..t {
                inp.extend_from_slice(&(tokens[r * (t + 1) + i] & !IGNORE_TARGET).to_le_bytes());
                // a token with the ignore bit is an input like any other, but the position that predicts it has no loss and no gradient (the kernels skip a target outside the vocabulary)
                let y = tokens[r * (t + 1) + i + 1];
                let y = if y & IGNORE_TARGET != 0 {
                    u32::MAX
                } else {
                    n_valid += 1;
                    y
                };
                tgt.extend_from_slice(&y.to_le_bytes());
            }
        }
        self.tokens.upload(&inp)?;
        self.targets.upload(&tgt)?;
        if let Some(eos) = self.doc_eos {
            // the document start of every position: an end-of-text input token opens a new document
            let mut ds = Vec::with_capacity(n * 4);
            for r in 0..b {
                let mut start = 0usize;
                for i in 0..t {
                    if tokens[r * (t + 1) + i] & !IGNORE_TARGET == eos {
                        start = i;
                    }
                    ds.extend_from_slice(&(start as i32).to_le_bytes());
                }
            }
            self.dstart.upload(&ds)?;
        }
        let st = &self.stream;
        let k = &self.k;
        let stride = qd + 2 * kvd;
        let (mix, n_enc) = (c.mixing(), c.n_enc());
        let ne = n * e;
        let grid_e = (((ne + 255) / 256).min(65535) as u32, 1, 1);
        // ---- forward ----
        // where the embedding output goes (the mixing reads it from x0); with the embedding norm the raw embedding (and the n-gram memory) goes to xe first
        let emb_out = if mix { &self.x0 } else { &self.act(0).x };
        let emb_dst = if c.emb_norm { &self.xe } else { emb_out };
        k.embed_batch.launch(
            (n as u32, 1, 1),
            (256, 1, 1),
            0,
            st,
            &Args::new()
                .ptr(self.w16(self.emb_off))
                .buf(&self.tokens)
                .buf(emb_dst)
                .i32(n as i32)
                .i32(e as i32),
        )?;
        let n_ng = c.ngram.len();
        let ng_tab = |i: usize| {
            if i < n_ng {
                self.w16(self.ng_off[i])
            } else {
                0
            }
        };
        let ng_idx = |i: usize| if i < n_ng { self.ng_idx[i].ptr() } else { 0 };
        if n_ng > 0 {
            // hashed n-gram memory: the embedding output gets  sum over the orders of table_n[hash of the last n tokens]  projected by ngw
            let dm = c.ngram_dim;
            for (i, &o) in c.ngram.iter().enumerate() {
                k.ngram_idx.launch(
                    (((n + 255) / 256).min(65535) as u32, 1, 1),
                    (256, 1, 1),
                    0,
                    st,
                    &Args::new()
                        .buf(&self.tokens)
                        .buf(&self.ng_idx[i])
                        .i32(n as i32)
                        .i32(t as i32)
                        .i32(o as i32)
                        .i32(c.ngram_bits as i32),
                )?;
            }
            k.ngram_gather.launch(
                (n as u32, 1, 1),
                (128, 1, 1),
                0,
                st,
                &Args::new()
                    .ptr(ng_tab(0))
                    .ptr(ng_idx(0))
                    .ptr(ng_tab(1))
                    .ptr(ng_idx(1))
                    .ptr(ng_tab(2))
                    .ptr(ng_idx(2))
                    .buf(&self.ng_m)
                    .i32(dm as i32),
            )?;
            self.gemm_axpy(
                k.gemm_nt_res,
                self.ng_m.ptr(),
                self.w16(self.ngw_off),
                emb_dst.ptr(),
                n,
                e,
                dm,
                dm,
                dm,
                e,
                1.0,
                emb_dst.ptr(),
                e,
                0.0,
            )?;
        }
        if c.emb_norm {
            k.rmsnorm_res_fwd_f.launch(
                (((n + 7) / 8) as u32, 1, 1),
                (256, 1, 1),
                0,
                st,
                &Args::new()
                    .buf(&self.xe)
                    .ptr(self.m32(self.enorm_off))
                    .ptr(0)
                    .buf(emb_out)
                    .buf(&self.rstd_e)
                    .i32(n as i32)
                    .i32(e as i32)
                    .f32(c.eps),
            )?;
        }
        for l in 0..c.n_layer {
            let lo = self.layer_off[l].clone();
            let a = self.act(l);
            if mix && c.lambdas && c.skip_source(l).is_none() && self.kopt.fuse_mix {
                // layer input = a * (output of the previous layer) + b * (embedding) and its RMSNorm in one pass (mix_rmsnorm_fwd_kernel; bit-identical to the two kernels below)
                let xprev: &DeviceBuffer = if l == 0 {
                    &self.x0
                } else {
                    &self.act(l - 1).out
                };
                k.mix_rmsnorm_fwd.launch(
                    (((n + 7) / 8) as u32, 1, 1),
                    (256, 1, 1),
                    0,
                    st,
                    &Args::new()
                        .buf(xprev)
                        .buf(&self.x0)
                        .ptr(self.m32(self.lam_off + 3 * l))
                        .buf(&a.x)
                        .ptr(self.m32(lo.attn_norm))
                        .buf(&a.h1)
                        .buf(&a.rstd1)
                        .i32(n as i32)
                        .i32(e as i32)
                        .f32(c.eps),
                )?;
            } else if mix {
                // layer input = a * (output of the previous layer) + b * (embedding) + c * (output of the mirrored layer of the first half)
                let xprev: &DeviceBuffer = if l == 0 {
                    &self.x0
                } else {
                    &self.act(l - 1).out
                };
                let src = c.skip_source(l);
                let xskip: &DeviceBuffer = match src {
                    Some(j) => &self.act(j).out,
                    None => xprev,
                };
                k.mix_fwd.launch(
                    grid_e,
                    (256, 1, 1),
                    0,
                    st,
                    &Args::new()
                        .buf(xprev)
                        .buf(&self.x0)
                        .buf(xskip)
                        .buf(&a.x)
                        .u64(ne as u64)
                        .ptr(self.m32(self.lam_off + 3 * l))
                        .i32(c.lambdas as i32)
                        .i32(src.is_some() as i32),
                )?;
                self.rmsnorm_fwd(&a.x, lo.attn_norm, &a.h1, &a.rstd1)?;
            } else {
                self.rmsnorm_fwd(&a.x, lo.attn_norm, &a.h1, &a.rstd1)?;
            }
            // wq, wk, wv are adjacent in the flat parameter array: one GEMM for the whole projection
            self.gemm(
                k.fw_nt_f16,
                a.h1.ptr(),
                self.w16(lo.wq),
                a.qkv.ptr(),
                n,
                stride,
                e,
                e,
                e,
                stride,
            )?;
            let (qn, kn) = if c.qk_norm {
                (self.m32(lo.q_norm), self.m32(lo.k_norm))
            } else {
                (0, 0)
            };
            k.qkv_rope_train.launch(
                ((((n * (c.n_head + 2 * c.n_kv)) + 7) / 8) as u32, 1, 1),
                (256, 1, 1),
                0,
                st,
                &Args::new()
                    .buf(&a.qkv)
                    .buf(&a.q)
                    .buf(&a.k)
                    .buf(&a.v)
                    .buf(&self.cos)
                    .buf(&self.sin)
                    .ptr(qn)
                    .ptr(kn)
                    .buf(&a.rstd_q)
                    .buf(&a.rstd_k)
                    .i32(n as i32)
                    .i32(t as i32)
                    .i32(c.n_head as i32)
                    .i32(c.n_kv as i32)
                    .i32(c.head_dim as i32)
                    .f32(c.eps),
            )?;
            if lo.ve != usize::MAX {
                // value embedding: v = lv * v + lve * ve[token]
                k.ve_fwd.launch(
                    (n as u32, 1, 1),
                    (128, 1, 1),
                    0,
                    st,
                    &Args::new()
                        .buf(&a.v)
                        .ptr(self.w16(lo.ve))
                        .buf(&self.tokens)
                        .ptr(self.m32(lo.ve_sc))
                        .i32(kvd as i32),
                )?;
            }
            // a window at least as long as the sequence is the dense causal case
            let win = c.window_of(l).filter(|&w| w < t);
            if self.doc_eos.is_some() {
                // document masking: the masked copies of the first-generation kernels (flash_doc.cu); set_doc_mask refuses sliding windows
                let flash = if c.head_dim == 64 {
                    k.flash_doc_64
                } else {
                    k.flash_doc_128
                };
                flash.launch(
                    (((t + 63) / 64) as u32, c.n_head as u32, b as u32),
                    (128, 1, 1),
                    0,
                    st,
                    &Args::new()
                        .buf(&a.q)
                        .buf(&a.k)
                        .buf(&a.v)
                        .buf(&a.o)
                        .buf(&a.lse)
                        .buf(&self.dstart)
                        .i32(t as i32)
                        .i32(c.n_head as i32)
                        .i32(c.n_kv as i32),
                )?;
            } else {
                let flash = match (c.head_dim, win) {
                    (64, Some(_)) => k.flash_fwd_t64_win,
                    (64, None) if self.kopt.new_attn => k.flash_fwd_t64,
                    (64, None) => k.flash_train_64,
                    _ => k.flash_train_128,
                };
                flash.launch(
                    (((t + 63) / 64) as u32, c.n_head as u32, b as u32),
                    (128, 1, 1),
                    0,
                    st,
                    &Args::new()
                        .buf(&a.q)
                        .buf(&a.k)
                        .buf(&a.v)
                        .buf(&a.o)
                        .buf(&a.lse)
                        .i32(t as i32)
                        .i32(c.n_head as i32)
                        .i32(c.n_kv as i32)
                        // the dense kernel ignores the window; non-zero asks it for the longest-first launch order over the whole grid (devices with the dS spill)
                        .i32(win.unwrap_or(if self.ds_buf.is_some() { 1 } else { 0 }) as i32),
                )?;
            }
            let o_in = if c.attn_gate > 0 {
                // gated attention: og = o * sigmoid(W h1[..K]) per head (the gates are kept for the backward)
                k.attn_gate_fwd.launch(
                    (((n + 7) / 8) as u32, 1, 1),
                    (256, 1, 1),
                    0,
                    st,
                    &Args::new()
                        .buf(&a.h1)
                        .ptr(self.m32(self.gate_off + l * c.n_head * c.attn_gate))
                        .buf(&a.o)
                        .buf(&a.gate)
                        .buf(&self.og16)
                        .i32(n as i32)
                        .i32(e as i32)
                        .i32(c.attn_gate as i32)
                        .i32(c.n_head as i32)
                        .i32(c.head_dim as i32),
                )?;
                self.og16.ptr()
            } else {
                a.o.ptr()
            };
            if c.peri_norm {
                // Peri-LN: xmid = x + Norm(o Wo^T)
                self.gemm(
                    k.fw_nt_f16,
                    o_in,
                    self.w16(lo.wo),
                    a.y1.ptr(),
                    n,
                    e,
                    qd,
                    qd,
                    qd,
                    e,
                )?;
                k.rmsnorm_res_fwd_h.launch(
                    (((n + 7) / 8) as u32, 1, 1),
                    (256, 1, 1),
                    0,
                    st,
                    &Args::new()
                        .buf(&a.y1)
                        .ptr(self.m32(self.pnorm_off + 2 * l * e))
                        .buf(&a.x)
                        .buf(&a.xmid)
                        .buf(&a.rstd_p1)
                        .i32(n as i32)
                        .i32(e as i32)
                        .f32(c.eps),
                )?;
            } else {
                // xmid = x + o Wo^T (the GEMM epilogue adds the fp32 residual: no copy)
                self.gemm_axpy(
                    k.fw_nt_res,
                    o_in,
                    self.w16(lo.wo),
                    a.xmid.ptr(),
                    n,
                    e,
                    qd,
                    qd,
                    qd,
                    e,
                    1.0,
                    a.x.ptr(),
                    e,
                    0.0,
                )?;
            }
            self.rmsnorm_fwd(&a.xmid, lo.ffn_norm, &a.h2, &a.rstd2)?;
            if c.relu2 {
                match k.fw_nt_relu2 {
                    // u and relu(u)^2 straight from the GEMM epilogue (bit-identical to the two-kernel path below)
                    Some(fused) if self.kopt.fuse_relu2 => self.gemm_axpy(
                        fused,
                        a.h2.ptr(),
                        self.w16(lo.w_up),
                        a.gu.ptr(),
                        n,
                        ff,
                        e,
                        e,
                        e,
                        ff,
                        1.0,
                        a.act.ptr(),
                        ff,
                        0.0,
                    )?,
                    _ => {
                        self.gemm(
                            k.fw_nt_f16,
                            a.h2.ptr(),
                            self.w16(lo.w_up),
                            a.gu.ptr(),
                            n,
                            ff,
                            e,
                            e,
                            e,
                            ff,
                        )?;
                        k.relu2.launch(
                            (((n * ff + 255) / 256).min(65535) as u32, 1, 1),
                            (256, 1, 1),
                            0,
                            st,
                            &Args::new().buf(&a.gu).buf(&a.act).u64((n * ff) as u64),
                        )?;
                    }
                }
            } else {
                self.gemm(
                    k.fw_nt_f16,
                    a.h2.ptr(),
                    self.w16(lo.w_gate),
                    a.gu.ptr(),
                    n,
                    2 * ff,
                    e,
                    e,
                    e,
                    2 * ff,
                )?;
                k.swiglu2.launch(
                    (((n * ff / 8 + 255) / 256).min(65535) as u32, 1, 1),
                    (256, 1, 1),
                    0,
                    st,
                    &Args::new()
                        .buf(&a.gu)
                        .buf(&a.act)
                        .i32(n as i32)
                        .i32(ff as i32),
                )?;
            }
            let next: &DeviceBuffer = if l + 1 < c.n_layer {
                if mix {
                    &a.out
                } else {
                    &self.act(l + 1).x
                }
            } else {
                &self.x_final
            };
            if c.peri_norm {
                // Peri-LN: out = xmid + Norm(act Wdown^T)
                self.gemm(
                    k.fw_nt_f16,
                    a.act.ptr(),
                    self.w16(lo.w_down),
                    a.y2.ptr(),
                    n,
                    e,
                    ff,
                    ff,
                    ff,
                    e,
                )?;
                k.rmsnorm_res_fwd_h.launch(
                    (((n + 7) / 8) as u32, 1, 1),
                    (256, 1, 1),
                    0,
                    st,
                    &Args::new()
                        .buf(&a.y2)
                        .ptr(self.m32(self.pnorm_off + (2 * l + 1) * e))
                        .buf(&a.xmid)
                        .buf(next)
                        .buf(&a.rstd_p2)
                        .i32(n as i32)
                        .i32(e as i32)
                        .f32(c.eps),
                )?;
            } else {
                self.gemm_axpy(
                    k.fw_nt_res,
                    a.act.ptr(),
                    self.w16(lo.w_down),
                    next.ptr(),
                    n,
                    e,
                    ff,
                    ff,
                    ff,
                    e,
                    1.0,
                    a.xmid.ptr(),
                    e,
                    0.0,
                )?;
            }
        }
        // final norm and the cross-entropy head (logits chunk by chunk; the head's gradients are produced right here)
        self.rmsnorm_fwd(&self.x_final, self.out_norm_off, &self.hf16, &self.rstd_f)?;
        if let Some(pos) = self.logits_row {
            // generation: the logits of one position only (the first row of `z`)
            self.gemm(
                k.gemm_nt_f32,
                self.hf16.ptr() + (pos * e * 2) as u64,
                self.w16(self.head_off),
                self.z.ptr(),
                1,
                v,
                e,
                e,
                e,
                v,
            )?;
            st.synchronize()?;
            return Ok(0.0);
        }
        let ce = c.ce_chunk.min(n);
        let dz_scale = loss_scale / tokens_per_step as f32;
        for ch in 0..n / ce {
            let hrow = self.hf16.ptr() + (ch * ce * e * 2) as u64;
            self.gemm(
                k.gemm_nt_f32,
                hrow,
                self.w16(self.head_off),
                self.z.ptr(),
                ce,
                v,
                e,
                e,
                e,
                v,
            )?;
            let ce_args = Args::new()
                .buf(&self.z)
                .ptr(self.targets.ptr() + (ch * ce * 4) as u64)
                .ptr(self.loss_rows.ptr() + (ch * ce * 4) as u64)
                .buf(&self.dz)
                .i32(v as i32)
                .f32(dz_scale);
            if c.softcap > 0.0 {
                k.ce_cap.launch(
                    (ce as u32, 1, 1),
                    (256, 1, 1),
                    0,
                    st,
                    &ce_args.f32(c.softcap),
                )?;
            } else if self.kopt.ce_reg && v % 4 == 0 && v <= 32768 {
                // one read of the logits (the row lives in the registers of 1024 threads)
                k.ce_reg
                    .launch((ce as u32, 1, 1), (1024, 1, 1), 0, st, &ce_args)?;
            } else {
                k.ce.launch((ce as u32, 1, 1), (256, 1, 1), 0, st, &ce_args)?;
            }
            if !with_grad {
                continue;
            }
            // dh = dZ . Head  (NN: A = dZ [ce][V], B = Head stored [k = V][n = E]; the tied head is the embedding)
            self.gemm(
                k.gemm_nn_f32,
                self.dz.ptr(),
                self.w16(self.head_off),
                self.dh32.ptr() + (ch * ce * e * 4) as u64,
                ce,
                e,
                v,
                v,
                e,
                e,
            )?;
            // dHead += dZ^T . h  (TN: A = dZ stored [k = ce][m = V], B = h chunk [k = ce][n = E])
            self.gemm(
                k.gemm_tn_add,
                self.dz.ptr(),
                hrow,
                self.g32(self.head_off),
                v,
                e,
                ce,
                v,
                e,
                e,
            )?;
        }
        // the mean over the positions that have a target (all of them, unless the data marks tokens as ignored)
        k.mean.launch(
            (1, 1, 1),
            (256, 1, 1),
            0,
            st,
            &Args::new()
                .buf(&self.loss_rows)
                .i32(n as i32)
                .i32(n_valid.max(1) as i32)
                .buf(&self.loss),
        )?;
        if !with_grad {
            st.synchronize()?;
            return Ok(self.loss.download_f32(1)?[0]);
        }
        // ---- backward ----
        self.rmsnorm_bwd(
            &self.dh32,
            &self.x_final,
            self.out_norm_off,
            &self.rstd_f,
            None,
            &self.dx,
            Some(&self.dy16),
        )?;
        if mix {
            self.dx0.zero(st)?;
        }
        for l in (0..c.n_layer).rev() {
            let lo = self.layer_off[l].clone();
            let a = self.act(l);
            if c.skips && l < n_enc {
                // this layer's output also fed a skip connection: its gradient is the stream's plus the one that came back through the skip
                k.add_to_f32_f16.launch(
                    grid_e,
                    (256, 1, 1),
                    0,
                    st,
                    &Args::new()
                        .buf(&self.dx)
                        .buf(&self.dskip[l])
                        .buf(&self.dy16)
                        .u64(ne as u64),
                )?;
            }
            // MLP branch: dx is the gradient of this layer's output, dy16 its fp16 copy (written by the previous RMSNorm backward)
            if c.peri_norm {
                // through the output norm: dy16 becomes the gradient of the MLP sublayer output (the residual path keeps dx)
                self.rmsnorm_bwd_h(
                    &self.dx,
                    &a.y2,
                    self.pnorm_off + (2 * l + 1) * e,
                    &a.rstd_p2,
                    &self.dpost,
                    &self.dy16,
                )?;
            }
            let fused_bwd = match k.gemm_nn_relu2_bwd {
                Some(f) if c.relu2 && self.kopt.fuse_relu2 => Some(f),
                _ => None,
            };
            match fused_bwd {
                // the gradient of the up projection's output straight from the GEMM epilogue: du = fp16(dy W_down) * 2 relu(u) (bit-identical to the two-kernel path)
                Some(f) => self.gemm_axpy(
                    f,
                    self.dy16.ptr(),
                    self.w16(lo.w_down),
                    self.dgu16.ptr(),
                    n,
                    ff,
                    e,
                    e,
                    ff,
                    ff,
                    1.0,
                    a.gu.ptr(),
                    ff,
                    0.0,
                )?,
                None => self.gemm(
                    k.gemm_nn_f16,
                    self.dy16.ptr(),
                    self.w16(lo.w_down),
                    self.da16.ptr(),
                    n,
                    ff,
                    e,
                    e,
                    ff,
                    ff,
                )?,
            }
            self.gemm(
                k.gemm_tn_add,
                self.dy16.ptr(),
                a.act.ptr(),
                self.g32(lo.w_down),
                e,
                ff,
                n,
                e,
                ff,
                ff,
            )?;
            if c.relu2 {
                if fused_bwd.is_none() {
                    k.relu2_bwd.launch(
                        (((n * ff + 255) / 256).min(65535) as u32, 1, 1),
                        (256, 1, 1),
                        0,
                        st,
                        &Args::new()
                            .buf(&self.da16)
                            .buf(&a.gu)
                            .buf(&self.dgu16)
                            .u64((n * ff) as u64),
                    )?;
                }
                self.gemm(
                    k.gemm_nn_f32,
                    self.dgu16.ptr(),
                    self.w16(lo.w_up),
                    self.dh32.ptr(),
                    n,
                    e,
                    ff,
                    ff,
                    e,
                    e,
                )?;
                self.gemm(
                    k.gemm_tn_add,
                    self.dgu16.ptr(),
                    a.h2.ptr(),
                    self.g32(lo.w_up),
                    ff,
                    e,
                    n,
                    ff,
                    e,
                    e,
                )?;
            } else {
                k.swiglu2_bwd.launch(
                    (((n * ff / 8 + 255) / 256).min(65535) as u32, 1, 1),
                    (256, 1, 1),
                    0,
                    st,
                    &Args::new()
                        .buf(&self.da16)
                        .buf(&a.gu)
                        .buf(&self.dgu16)
                        .i32(n as i32)
                        .i32(ff as i32),
                )?;
                // w_gate and w_up are adjacent: one data-gradient GEMM (K = 2F) and one weight-gradient GEMM (M = 2F)
                self.gemm(
                    k.gemm_nn_f32,
                    self.dgu16.ptr(),
                    self.w16(lo.w_gate),
                    self.dh32.ptr(),
                    n,
                    e,
                    2 * ff,
                    2 * ff,
                    e,
                    e,
                )?;
                self.gemm(
                    k.gemm_tn_add,
                    self.dgu16.ptr(),
                    a.h2.ptr(),
                    self.g32(lo.w_gate),
                    2 * ff,
                    e,
                    n,
                    2 * ff,
                    e,
                    e,
                )?;
            }
            self.rmsnorm_bwd(
                &self.dh32,
                &a.xmid,
                lo.ffn_norm,
                &a.rstd2,
                Some(&self.dx),
                &self.dxn,
                Some(&self.dy16),
            )?;
            // attention branch: dxn is the gradient of xmid (dy16 its fp16 copy)
            if c.peri_norm {
                self.rmsnorm_bwd_h(
                    &self.dxn,
                    &a.y1,
                    self.pnorm_off + 2 * l * e,
                    &a.rstd_p1,
                    &self.dpost,
                    &self.dy16,
                )?;
            }
            self.gemm(
                k.gemm_nn_f16,
                self.dy16.ptr(),
                self.w16(lo.wo),
                self.do16.ptr(),
                n,
                qd,
                e,
                e,
                qd,
                qd,
            )?;
            let gate_w = self.gate_off + l * c.n_head * c.attn_gate;
            if c.attn_gate > 0 {
                // the output projection saw o * g: recompute it for the weight gradient, then turn do16 into the gradient of o and collect the gradients of the gate logits
                k.attn_gate_mul.launch(
                    (((n * qd / 2 + 255) / 256).min(65535) as u32, 1, 1),
                    (256, 1, 1),
                    0,
                    st,
                    &Args::new()
                        .buf(&a.o)
                        .buf(&a.gate)
                        .buf(&self.og16)
                        .i32(n as i32)
                        .i32(c.n_head as i32)
                        .i32(c.head_dim as i32),
                )?;
                self.gemm(
                    k.gemm_tn_add,
                    self.dy16.ptr(),
                    self.og16.ptr(),
                    self.g32(lo.wo),
                    e,
                    qd,
                    n,
                    e,
                    qd,
                    qd,
                )?;
                k.attn_gate_bwd.launch(
                    (((n * c.n_head + 7) / 8) as u32, 1, 1),
                    (256, 1, 1),
                    0,
                    st,
                    &Args::new()
                        .buf(&self.do16)
                        .buf(&a.o)
                        .buf(&a.gate)
                        .buf(&self.gdpre)
                        .i32(n as i32)
                        .i32(c.n_head as i32)
                        .i32(c.head_dim as i32),
                )?;
                let gb = GATE_BLOCKS.min(n);
                k.attn_gate_dw.launch(
                    (gb as u32, 1, 1),
                    (256, 1, 1),
                    0,
                    st,
                    &Args::new()
                        .buf(&self.gdpre)
                        .buf(&a.h1)
                        .buf(&self.gpart)
                        .i32(n as i32)
                        .i32(e as i32)
                        .i32(c.attn_gate as i32)
                        .i32(c.n_head as i32),
                )?;
                let hk = c.n_head * c.attn_gate;
                k.reduce_partials.launch(
                    (((hk + 255) / 256) as u32, 1, 1),
                    (256, 1, 1),
                    0,
                    st,
                    &Args::new()
                        .buf(&self.gpart)
                        .i32(gb as i32)
                        .i32(hk as i32)
                        .ptr(self.g32(gate_w)),
                )?;
            } else {
                self.gemm(
                    k.gemm_tn_add,
                    self.dy16.ptr(),
                    a.o.ptr(),
                    self.g32(lo.wo),
                    e,
                    qd,
                    n,
                    e,
                    qd,
                    qd,
                )?;
            }
            k.attn_bwd_prep.launch(
                (((n * c.n_head + 7) / 8) as u32, 1, 1),
                (256, 1, 1),
                0,
                st,
                &Args::new()
                    .buf(&a.o)
                    .buf(&self.do16)
                    .buf(&self.dvec)
                    .i32(n as i32)
                    .i32(c.n_head as i32)
                    .i32(c.head_dim as i32),
            )?;
            let win = c.window_of(l).filter(|&w| w < t);
            let flash_bwd =
                c.head_dim == 64 && t % 64 == 0 && std::env::var("FORGE_NAIVE_ATTN").is_err();
            if win.is_some() && !flash_bwd {
                return Err(Error::unsupported(
                    "sliding-window attention has no reference backward (FORGE_NAIVE_ATTN)",
                ));
            }
            if self.doc_eos.is_some() {
                if t % 64 != 0 {
                    return Err(Error::unsupported(
                        "document masking needs a window that is a multiple of 64",
                    ));
                }
                let args = |dst_a: &DeviceBuffer, dst_b: Option<&DeviceBuffer>| {
                    let mut a = Args::new()
                        .buf(&a.q)
                        .buf(&a.k)
                        .buf(&a.v)
                        .buf(&self.do16)
                        .buf(&a.lse)
                        .buf(&self.dvec)
                        .buf(&self.dstart)
                        .buf(dst_a);
                    if let Some(bb) = dst_b {
                        a = a.buf(bb);
                    }
                    a.i32(t as i32).i32(c.n_head as i32).i32(c.n_kv as i32)
                };
                k.flash_doc_dkdv_64.launch(
                    ((t / 64) as u32, c.n_kv as u32, b as u32),
                    (128, 1, 1),
                    0,
                    st,
                    &args(&self.dk32, Some(&self.dv32)),
                )?;
                k.flash_doc_dq_64.launch(
                    ((t / 64) as u32, c.n_head as u32, b as u32),
                    (128, 1, 1),
                    0,
                    st,
                    &args(&self.dq32, None),
                )?;
            } else if flash_bwd {
                let args = |dst_a: &DeviceBuffer, dst_b: Option<&DeviceBuffer>| {
                    let mut a = Args::new()
                        .buf(&a.q)
                        .buf(&a.k)
                        .buf(&a.v)
                        .buf(&self.do16)
                        .buf(&a.lse)
                        .buf(&self.dvec)
                        .buf(dst_a);
                    if let Some(bb) = dst_b {
                        a = a.buf(bb);
                    }
                    a.i32(t as i32)
                        .i32(c.n_head as i32)
                        .i32(c.n_kv as i32)
                        .i32(win.unwrap_or(0) as i32)
                };
                // dense layers with the dS spill: dK dV stores the dS tiles, dQ is one matmul per tile
                let ds_fits = self
                    .ds_buf
                    .as_ref()
                    .map(|bf| bf.len() >= b * c.n_head * (t / 64) * (t / 64 + 1) / 2 * 4096 * 2)
                    .unwrap_or(false);
                if win.is_none() && ds_fits {
                    let dsb = self.ds_buf.as_ref().unwrap();
                    k.flash_dkdv_ds_64.launch(
                        ((t / 64) as u32, c.n_kv as u32, b as u32),
                        (128, 1, 1),
                        0,
                        st,
                        &Args::new()
                            .buf(&a.q)
                            .buf(&a.k)
                            .buf(&a.v)
                            .buf(&self.do16)
                            .buf(&a.lse)
                            .buf(&self.dvec)
                            .buf(&self.dk32)
                            .buf(&self.dv32)
                            .buf(dsb)
                            .i32(t as i32)
                            .i32(c.n_head as i32)
                            .i32(c.n_kv as i32),
                    )?;
                    k.flash_dq_ds_64.launch(
                        ((t / 64) as u32, c.n_head as u32, b as u32),
                        (128, 1, 1),
                        0,
                        st,
                        &Args::new()
                            .buf(&a.k)
                            .buf(dsb)
                            .buf(&self.dq32)
                            .i32(t as i32)
                            .i32(c.n_head as i32)
                            .i32(c.n_kv as i32),
                    )?;
                } else {
                let (fdkdv, fdq) = match win {
                    Some(_) => (k.flash_dkdv_64_win, k.flash_dq_64_win),
                    None if self.kopt.new_attn => (k.flash_dkdv_64, k.flash_dq_t64),
                    None => (k.flash_dkdv_64, k.flash_dq_64),
                };
                fdkdv.launch(
                    ((t / 64) as u32, c.n_kv as u32, b as u32),
                    (128, 1, 1),
                    0,
                    st,
                    &args(&self.dk32, Some(&self.dv32)),
                )?;
                fdq.launch(
                    ((t / 64) as u32, c.n_head as u32, b as u32),
                    (128, 1, 1),
                    0,
                    st,
                    &args(&self.dq32, None),
                )?;
                }
            } else {
                k.attn_dkdv.launch(
                    (((b * t * c.n_kv + 127) / 128) as u32, 1, 1),
                    (128, 1, 1),
                    0,
                    st,
                    &Args::new()
                        .buf(&a.q)
                        .buf(&a.k)
                        .buf(&a.v)
                        .buf(&self.do16)
                        .buf(&a.lse)
                        .buf(&self.dvec)
                        .buf(&self.dk32)
                        .buf(&self.dv32)
                        .i32(b as i32)
                        .i32(t as i32)
                        .i32(c.n_head as i32)
                        .i32(c.n_kv as i32)
                        .i32(c.head_dim as i32),
                )?;
                k.attn_dq.launch(
                    (((b * t * c.n_head + 127) / 128) as u32, 1, 1),
                    (128, 1, 1),
                    0,
                    st,
                    &Args::new()
                        .buf(&a.q)
                        .buf(&a.k)
                        .buf(&a.v)
                        .buf(&self.do16)
                        .buf(&a.lse)
                        .buf(&self.dvec)
                        .buf(&self.dq32)
                        .i32(b as i32)
                        .i32(t as i32)
                        .i32(c.n_head as i32)
                        .i32(c.n_kv as i32)
                        .i32(c.head_dim as i32),
                )?;
            }
            if lo.ve != usize::MAX {
                // value embedding: dve[token] += lve * dv, gradients of (lv, lve), and dv *= lv (the gradient of the raw projection output)
                let vb = 128usize.min(n);
                k.ve_bwd.launch(
                    (vb as u32, 1, 1),
                    (128, 1, 1),
                    0,
                    st,
                    &Args::new()
                        .buf(&self.dv32)
                        .buf(&a.qkv)
                        .i32(stride as i32)
                        .i32((qd + kvd) as i32)
                        .ptr(self.w16(lo.ve))
                        .buf(&self.tokens)
                        .ptr(self.m32(lo.ve_sc))
                        .ptr(self.g32(lo.ve))
                        .buf(&self.sc_partial)
                        .i32(n as i32)
                        .i32(kvd as i32),
                )?;
                k.reduce_partials.launch(
                    (1, 1, 1),
                    (32, 1, 1),
                    0,
                    st,
                    &Args::new()
                        .buf(&self.sc_partial)
                        .i32(vb as i32)
                        .i32(2)
                        .ptr(self.g32(lo.ve_sc)),
                )?;
            }
            let (qn, kn) = if c.qk_norm {
                (self.m32(lo.q_norm), self.m32(lo.k_norm))
            } else {
                (0, 0)
            };
            let qb = QB_BLOCKS.min(n);
            k.qkv_bwd.launch(
                (qb as u32, 1, 1),
                (256, 1, 1),
                0,
                st,
                &Args::new()
                    .buf(&self.dq32)
                    .buf(&self.dk32)
                    .buf(&self.dv32)
                    .buf(&a.qkv)
                    .buf(&a.rstd_q)
                    .buf(&a.rstd_k)
                    .ptr(qn)
                    .ptr(kn)
                    .buf(&self.cos)
                    .buf(&self.sin)
                    .buf(&self.dqkv16)
                    .buf(&self.partial)
                    .i32(n as i32)
                    .i32(t as i32)
                    .i32(c.n_head as i32)
                    .i32(c.n_kv as i32)
                    .i32(c.head_dim as i32)
                    .i32(((n + qb - 1) / qb) as i32),
            )?;
            if c.qk_norm {
                // q-norm / k-norm weight gradients: partial[kind][block][hd] reduced in fixed order
                for (kind, g_off) in [(0usize, lo.q_norm), (1usize, lo.k_norm)] {
                    k.reduce_partials.launch(
                        (1, 1, 1),
                        (128, 1, 1),
                        0,
                        st,
                        &Args::new()
                            .ptr(self.partial.ptr() + (kind * qb * c.head_dim * 4) as u64)
                            .i32(qb as i32)
                            .i32(c.head_dim as i32)
                            .ptr(self.g32(g_off)),
                    )?;
                }
            }
            let dqkv = self.dqkv16.ptr();
            self.gemm(
                k.gemm_nn_f32,
                dqkv,
                self.w16(lo.wq),
                self.dh32.ptr(),
                n,
                e,
                stride,
                stride,
                e,
                e,
            )?;
            if c.attn_gate > 0 {
                // the gate read the first channels of the normalised input too
                k.attn_gate_dx.launch(
                    (((n * c.attn_gate + 255) / 256).min(65535) as u32, 1, 1),
                    (256, 1, 1),
                    0,
                    st,
                    &Args::new()
                        .buf(&self.gdpre)
                        .ptr(self.m32(gate_w))
                        .buf(&self.dh32)
                        .i32(n as i32)
                        .i32(e as i32)
                        .i32(c.attn_gate as i32)
                        .i32(c.n_head as i32),
                )?;
            }
            self.gemm(
                k.gemm_tn_add,
                dqkv,
                a.h1.ptr(),
                self.g32(lo.wq),
                stride,
                e,
                n,
                stride,
                e,
                e,
            )?;
            if mix && c.lambdas && c.skip_source(l).is_none() && self.kopt.fuse_mix {
                // attention-norm backward and mixing backward in one pass (rmsnorm_mix_bwd_kernel): dx = a d, dy16 its fp16 copy, dx0 += b d, partial sums of the gradients of (a, b)
                let xprev: &DeviceBuffer = if l == 0 {
                    &self.x0
                } else {
                    &self.act(l - 1).out
                };
                let blocks = RB_BLOCKS.min((n + 7) / 8);
                k.rmsnorm_mix_bwd.launch(
                    (blocks as u32, 1, 1),
                    (256, 1, 1),
                    (8 * e * 4) as u32,
                    st,
                    &Args::new()
                        .buf(&self.dh32)
                        .buf(&a.x)
                        .ptr(self.m32(lo.attn_norm))
                        .buf(&a.rstd1)
                        .buf(&self.dxn)
                        .buf(&self.dx)
                        .buf(&self.dy16)
                        .buf(&self.partial)
                        .i32(n as i32)
                        .i32(e as i32)
                        .buf(xprev)
                        .buf(&self.x0)
                        .buf(&self.dx0)
                        .ptr(self.m32(self.lam_off + 3 * l))
                        .buf(&self.sc_partial),
                )?;
                k.reduce_partials.launch(
                    (((e + 255) / 256) as u32, 1, 1),
                    (256, 1, 1),
                    0,
                    st,
                    &Args::new()
                        .buf(&self.partial)
                        .i32(blocks as i32)
                        .i32(e as i32)
                        .ptr(self.g32(lo.attn_norm)),
                )?;
                k.reduce_partials.launch(
                    (1, 1, 1),
                    (32, 1, 1),
                    0,
                    st,
                    &Args::new()
                        .buf(&self.sc_partial)
                        .i32(blocks as i32)
                        .i32(3)
                        .ptr(self.g32(self.lam_off + 3 * l)),
                )?;
                if l == 0 {
                    k.add_f32.launch(
                        grid_e,
                        (256, 1, 1),
                        0,
                        st,
                        &Args::new().buf(&self.dx0).buf(&self.dx).u64(ne as u64),
                    )?;
                }
                continue;
            }
            self.rmsnorm_bwd(
                &self.dh32,
                &a.x,
                lo.attn_norm,
                &a.rstd1,
                Some(&self.dxn),
                &self.dx,
                Some(&self.dy16),
            )?;
            if mix {
                // dx is now the gradient of the mixed layer input: split it between the previous layer's output (a *), the embedding (b *) and the skip source (c *), and collect the scalars' gradients
                let xprev: &DeviceBuffer = if l == 0 {
                    &self.x0
                } else {
                    &self.act(l - 1).out
                };
                let src = c.skip_source(l);
                let xskip: &DeviceBuffer = match src {
                    Some(j) => &self.act(j).out,
                    None => xprev,
                };
                let dskip: &DeviceBuffer = match src {
                    Some(j) => &self.dskip[j],
                    None => &self.dx0,
                };
                let mb = ((ne + 255) / 256).min(256);
                k.mix_bwd.launch(
                    (mb as u32, 1, 1),
                    (256, 1, 1),
                    0,
                    st,
                    &Args::new()
                        .buf(&self.dx)
                        .buf(xprev)
                        .buf(&self.x0)
                        .buf(xskip)
                        .buf(&self.dx0)
                        .buf(dskip)
                        .buf(&self.dy16)
                        .buf(&self.sc_partial)
                        .u64(ne as u64)
                        .ptr(self.m32(self.lam_off + 3 * l))
                        .i32(c.lambdas as i32)
                        .i32(src.is_some() as i32),
                )?;
                k.reduce_partials.launch(
                    (1, 1, 1),
                    (32, 1, 1),
                    0,
                    st,
                    &Args::new()
                        .buf(&self.sc_partial)
                        .i32(mb as i32)
                        .i32(3)
                        .ptr(self.g32(self.lam_off + 3 * l)),
                )?;
                if l == 0 {
                    // the first layer's "previous output" is the embedding itself
                    k.add_f32.launch(
                        grid_e,
                        (256, 1, 1),
                        0,
                        st,
                        &Args::new().buf(&self.dx0).buf(&self.dx).u64(ne as u64),
                    )?;
                }
            }
        }
        let mut demb: &DeviceBuffer = if mix { &self.dx0 } else { &self.dx };
        if c.emb_norm {
            // back through the embedding norm to the raw embedding (and n-gram) output
            self.rmsnorm_bwd(
                demb,
                &self.xe,
                self.enorm_off,
                &self.rstd_e,
                None,
                &self.dpost,
                None,
            )?;
            demb = &self.dpost;
        }
        if n_ng > 0 {
            // gradient of the embedding output -> the projection, and the rows of the tables that were read
            let dm = c.ngram_dim;
            self.f32_to_f16(demb, &self.dy16, ne)?;
            self.gemm(
                k.gemm_nn_f16,
                self.dy16.ptr(),
                self.w16(self.ngw_off),
                self.ng_dm.ptr(),
                n,
                dm,
                e,
                e,
                dm,
                dm,
            )?;
            self.gemm(
                k.gemm_tn_add,
                self.dy16.ptr(),
                self.ng_m.ptr(),
                self.g32(self.ngw_off),
                e,
                dm,
                n,
                e,
                dm,
                dm,
            )?;
            let ng_grad = |i: usize| {
                if i < n_ng {
                    self.g32(self.ng_off[i])
                } else {
                    0
                }
            };
            k.ngram_scatter.launch(
                (n as u32, 1, 1),
                (128, 1, 1),
                0,
                st,
                &Args::new()
                    .buf(&self.ng_dm)
                    .ptr(ng_idx(0))
                    .ptr(ng_grad(0))
                    .ptr(ng_idx(1))
                    .ptr(ng_grad(1))
                    .ptr(ng_idx(2))
                    .ptr(ng_grad(2))
                    .i32(dm as i32),
            )?;
        }
        k.embed_bwd.launch(
            (n as u32, 1, 1),
            (256, 1, 1),
            0,
            st,
            &Args::new()
                .buf(demb)
                .buf(&self.tokens)
                .ptr(self.g32(self.emb_off))
                .i32(e as i32),
        )?;
        st.synchronize()?;
        let l = self.loss.download_f32(1)?;
        Ok(l[0])
    }

    /// Clip, check for non-finite gradients and apply AdamW. `loss_scale` is the scale the gradients were produced with. Returns the gradient norm and whether the step was skipped.
    #[allow(clippy::too_many_arguments)]
    pub fn step(
        &mut self,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
        max_norm: f32,
        loss_scale: f32,
    ) -> Result<StepStats> {
        let st = &self.stream;
        let k = &self.k;
        let n = self.n_params;
        self.ts.upload_f32(&[
            loss_scale,
            0.0,
            0.0,
            1.0 / loss_scale,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ])?;
        k.grad_sumsq.launch(
            (SUMSQ_BLOCKS as u32, 1, 1),
            (256, 1, 1),
            0,
            st,
            &Args::new()
                .buf(&self.grads)
                .u64(n as u64)
                .buf(&self.sumsq_partial)
                .buf(&self.nonfinite_partial),
        )?;
        k.grad_finalize.launch(
            (1, 1, 1),
            (32, 1, 1),
            0,
            st,
            &Args::new()
                .buf(&self.sumsq_partial)
                .buf(&self.nonfinite_partial)
                .i32(SUMSQ_BLOCKS as i32)
                .f32(max_norm)
                .buf(&self.ts),
        )?;
        self.step_count += 1;
        let t = self.step_count as f32;
        let bc1 = 1.0 - beta1.powf(t);
        let bc2 = 1.0 - beta2.powf(t);
        for (lo, hi, mult) in self.adam_groups(false) {
            self.adamw_range(lo, hi, lr * mult, beta1, beta2, eps, weight_decay, bc1, bc2)?;
        }
        st.synchronize()?;
        let ts = self.ts.download_f32(4)?;
        if ts[1] != 0.0 {
            self.step_count -= 1; // the update was skipped: do not advance the bias-correction counter
        }
        Ok(StepStats {
            grad_norm: ts[2],
            found_inf: ts[1] != 0.0,
        })
    }

    /// Learning-rate multipliers of AdamW's parameter groups: the (tied or input) embedding, the separate output head, the lookup tables (value embeddings, n-gram memory and its projection), the scalars
    /// (per-layer mixing, value-embedding mixing). Default 1.
    pub fn set_lr_mults(&mut self, emb: f32, head: f32, table: f32, scalar: f32) {
        self.lr_mults = (emb, head, table, scalar);
    }

    /// AdamW learning-rate multiplier of the attention gate weights (default 1)
    pub fn set_gate_lr_mult(&mut self, m: f32) {
        self.gate_lr_mult = m;
    }

    /// Cautious weight decay in AdamW (Chen et al. 2025; modded-nanogpt record 50): a decayed element decays only where its update moves it towards zero. Default off.
    pub fn set_adam_cautious_wd(&mut self, on: bool) {
        self.adam_cwd = on;
    }

    /// Document masking for every forward and backward pass from now on (training and evaluation): `eos` (the end-of-text id) opens a new document, and a position attends only to the positions of
    /// its own document (RoPE is relative, so no position reset is needed). None = plain causal attention. Needs head_dim 64 (the masked backward kernels exist for it). The masked kernels are copies
    /// of the first-generation attention kernels (flash_doc.cu): with the mask on, the dense layers do not use kernel-opt's newer attention kernels; sliding windows are refused.
    pub fn set_doc_mask(&mut self, eos: Option<u32>) -> Result<()> {
        if eos.is_some() && self.cfg.head_dim != 64 {
            return Err(Error::unsupported("document masking needs head_dim 64"));
        }
        if eos.is_some() && self.cfg.window > 0 && !self.cfg.window_layers.is_empty() {
            return Err(Error::unsupported("document masking and sliding-window attention cannot be combined (the masked kernels have no window)"));
        }
        self.doc_eos = eos;
        Ok(())
    }

    /// the end-of-text id of the document masking (None = plain causal attention)
    pub fn doc_mask(&self) -> Option<u32> {
        self.doc_eos
    }

    /// The parameter ranges AdamW updates, with their learning-rate multipliers, neighbours with equal multipliers merged. `muon`: the hidden matrices belong to Muon and are left out.
    fn adam_groups(&self, muon: bool) -> Vec<(usize, usize, f32)> {
        let emb_end = self.cfg.vocab * self.cfg.n_embd;
        let norm_end = self.out_norm_off + self.cfg.n_embd;
        // the recipe-opt tensors at the end of the layout (attention gates, Peri-LN gains) follow the scalars
        let c = &self.cfg;
        let tail_end = [
            (c.attn_gate > 0).then_some(self.gate_off),
            c.peri_norm.then_some(self.pnorm_off),
            c.emb_norm.then_some(self.enorm_off),
        ]
        .into_iter()
        .flatten()
        .min()
        .unwrap_or(self.n_params);
        let scalar_start = if self.cfg.mixing() {
            self.lam_off
        } else if !self.cfg.ve_layers.is_empty() {
            self.velam_off
        } else {
            tail_end
        };
        let (m_emb, m_head, m_tab, m_sc) = self.lr_mults;
        let head_end = if self.cfg.untied {
            self.head_off + emb_end
        } else {
            norm_end
        };
        let mut raw = vec![(0, emb_end, m_emb)];
        if !muon {
            raw.push((emb_end, self.decay_end, 1.0));
        }
        raw.push((self.decay_end, norm_end, 1.0));
        raw.push((norm_end, head_end, m_head));
        raw.push((head_end, scalar_start, m_tab));
        raw.push((scalar_start, tail_end, m_sc));
        if c.attn_gate > 0 {
            raw.push((
                self.gate_off,
                self.gate_off + c.n_layer * c.n_head * c.attn_gate,
                self.gate_lr_mult,
            ));
        }
        if c.peri_norm {
            raw.push((
                self.pnorm_off,
                self.pnorm_off + 2 * c.n_layer * c.n_embd,
                1.0,
            ));
        }
        if c.emb_norm {
            raw.push((self.enorm_off, self.enorm_off + c.n_embd, 1.0));
        }
        let mut groups: Vec<(usize, usize, f32)> = Vec::new();
        for (lo, hi, m) in raw {
            if hi <= lo {
                continue;
            }
            match groups.last_mut() {
                Some(last) if last.1 == lo && last.2 == m => last.1 = hi,
                _ => groups.push((lo, hi, m)),
            }
        }
        groups
    }

    /// AdamW on the parameters [lo, hi) with the learning rate `lr` (weight decay below `decay_end` as everywhere).
    #[allow(clippy::too_many_arguments)]
    fn adamw_range(
        &self,
        lo: usize,
        hi: usize,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
        bc1: f32,
        bc2: f32,
    ) -> Result<()> {
        let n = hi - lo;
        let decay_rel = self.decay_end.saturating_sub(lo).min(n);
        let kernel = if self.adam_cwd && decay_rel > 0 {
            self.k.adamw_cwd
        } else {
            self.k.adamw
        };
        kernel.launch(
            (((n + 255) / 256).min(4096) as u32, 1, 1),
            (256, 1, 1),
            0,
            &self.stream,
            &Args::new()
                .ptr(self.m32(lo))
                .ptr(self.g32(lo))
                .ptr(self.adam_m.ptr() + (lo * 4) as u64)
                .ptr(self.adam_v.ptr() + (lo * 4) as u64)
                .ptr(self.w16(lo))
                .u64(n as u64)
                .u64(decay_rel as u64)
                .f32(lr)
                .f32(beta1)
                .f32(beta2)
                .f32(eps)
                .f32(weight_decay)
                .f32(bc1)
                .f32(bc2)
                .buf(&self.ts)
                .u64(0)
                .u64(0),
        )
    }

    /// Five (or `iters`) Newton-Schulz iterations on the normalised matrix in `scr.x[0]` (rows x cols, row-major) on the stream `st`; returns which of the two iterate buffers holds the result.
    /// Works on the wide orientation: for rows <= cols the Gram matrix is X X^T and X' = a X + B X; for a tall matrix the transpose is iterated implicitly (X^T X and X' = a X + X B).
    fn muon_newton_schulz(
        &self,
        st: &Stream,
        scr: &MuonScratch,
        rows: usize,
        cols: usize,
        iters: usize,
    ) -> Result<usize> {
        self.muon_newton_schulz_coefs(st, scr, rows, cols, &vec![NS_JORDAN; iters])
    }

    /// The iterations X' = a X + (b A + c A A) X  (A = X X^T) on the stream `st` with the coefficients (a, b, c) of every iteration in turn (Jordan's: the same three numbers each time; Polar Express: one
    /// triple per iteration).
    fn muon_newton_schulz_coefs(
        &self,
        st: &Stream,
        scr: &MuonScratch,
        rows: usize,
        cols: usize,
        coefs: &[(f32, f32, f32)],
    ) -> Result<usize> {
        let k = &self.k;
        let tall = rows > cols;
        let (m, n) = if tall { (cols, rows) } else { (rows, cols) };
        if m % 16 != 0 || n % 16 != 0 {
            return Err(Error::unsupported(
                "Muon needs matrix dimensions that are multiples of 16",
            ));
        }
        let (a, b) = (scr.a.ptr(), scr.b.ptr());
        let mut cur = 0usize;
        for &(ca, cb, cc) in coefs {
            let x = scr.x[cur].ptr();
            let xn = scr.x[1 - cur].ptr();
            if !tall {
                self.gemm_on(st, k.gemm_nt_f16, x, x, a, m, m, n, n, n, m)?;
            } else {
                self.gemm_on(st, k.gemm_tn_f16, x, x, a, m, m, n, m, m, m)?;
            }
            // B = b A + c A A  (A is symmetric, so the NT product is A A)
            self.gemm_axpy_on(st, k.gemm_nt_axpy, a, a, b, m, m, m, m, m, m, cc, a, m, cb)?;
            if !tall {
                self.gemm_axpy_on(
                    st,
                    k.gemm_nn_axpy,
                    b,
                    x,
                    xn,
                    m,
                    n,
                    m,
                    m,
                    n,
                    n,
                    1.0,
                    x,
                    n,
                    ca,
                )?;
            } else {
                self.gemm_axpy_on(
                    st,
                    k.gemm_nn_axpy,
                    x,
                    b,
                    xn,
                    n,
                    m,
                    m,
                    m,
                    m,
                    m,
                    1.0,
                    x,
                    m,
                    ca,
                )?;
            }
            cur = 1 - cur;
        }
        Ok(cur)
    }

    /// Set the options of the Muon update (see `MuonOpts`). Polar Express has coefficient tables for 3 to 7 iterations.
    pub fn set_muon_opts(&mut self, o: MuonOpts) -> Result<()> {
        if o.iters == 0 || (o.polar && polar_express_coeffs(o.iters).is_none()) {
            return Err(Error::unsupported(
                "Muon iterations: 1 or more (Polar Express: 3 to 7)",
            ));
        }
        if !(0.0..1.0).contains(&o.normuon_beta2) {
            return Err(Error::unsupported("the NorMuon decay must be in [0, 1)"));
        }
        self.muon_opts = o;
        Ok(())
    }

    pub fn muon_opts(&self) -> &MuonOpts {
        &self.muon_opts
    }

    /// One Muon update of the matrix at parameter offset `off`: momentum, Nesterov, normalisation, Newton-Schulz, decoupled weight decay and the step (scale 0.2 sqrt(max(rows, cols)) so that the
    /// update has the RMS of an AdamW step and shares its learning-rate schedule).
    #[allow(clippy::too_many_arguments)]
    fn muon_matrix(
        &self,
        st: &Stream,
        scr: &MuonScratch,
        off: usize,
        rows: usize,
        cols: usize,
        lr: f32,
        wd: f32,
        mu: f32,
    ) -> Result<()> {
        let n = rows * cols;
        let k = &self.k;
        let o = &self.muon_opts;
        let blocks = ((n + 255) / 256).min(1024) as u32;
        k.muon_prep.launch(
            (64, 1, 1),
            (256, 1, 1),
            0,
            st,
            &Args::new()
                .ptr(self.g32(off))
                .ptr(self.adam_m.ptr() + (off * 4) as u64)
                .buf(&scr.u32)
                .u64(n as u64)
                .f32(mu)
                .buf(&self.ts)
                .buf(&scr.partial),
        )?;
        let cur = if o.polar {
            // Polar Express: |X|_F * 1.02 + 1e-6 puts every singular value safely below 1, then one minimax polynomial per iteration
            k.muon_normalize_m.launch(
                (blocks, 1, 1),
                (256, 1, 1),
                0,
                st,
                &Args::new()
                    .buf(&scr.u32)
                    .buf(&scr.partial)
                    .i32(64)
                    .buf(&scr.x[0])
                    .u64(n as u64)
                    .buf(&self.ts)
                    .f32(1.02)
                    .f32(1e-6),
            )?;
            self.muon_newton_schulz_coefs(
                st,
                scr,
                rows,
                cols,
                &polar_express_coeffs(o.iters).expect("checked by set_muon_opts"),
            )?
        } else {
            k.muon_normalize.launch(
                (blocks, 1, 1),
                (256, 1, 1),
                0,
                st,
                &Args::new()
                    .buf(&scr.u32)
                    .buf(&scr.partial)
                    .buf(&scr.x[0])
                    .u64(n as u64)
                    .buf(&self.ts),
            )?;
            self.muon_newton_schulz(st, scr, rows, cols, o.iters)?
        };
        let scale = 0.2 * (rows.max(cols) as f32).sqrt();
        if o.normuon.is_none() && !o.cautious_wd {
            k.muon_apply.launch(
                (blocks, 1, 1),
                (256, 1, 1),
                0,
                st,
                &Args::new()
                    .ptr(self.m32(off))
                    .ptr(self.work16.ptr() + (off * 2) as u64)
                    .buf(&scr.x[cur])
                    .u64(n as u64)
                    .f32(lr)
                    .f32(wd)
                    .f32(scale)
                    .buf(&self.ts),
            )?;
            return Ok(());
        }
        // NorMuon scratch of this stream: [lane_pow | gain] of the largest matrix dimension, then the norm-restoring factor
        let nl = scr.lanes;
        let (lane_pow, gain, ratio) = (
            scr.nm.ptr(),
            scr.nm.ptr() + (nl * 4) as u64,
            scr.nm.ptr() + (2 * nl * 4) as u64,
        );
        let lane_mode = match o.normuon {
            None => 0,
            Some(axis) => {
                let by_row = axis == NorMuonAxis::Rows || rows >= cols;
                let (lanes, len) = if by_row { (rows, cols) } else { (cols, rows) };
                // the running mean squares of the lanes live in the matrix's (otherwise unused) second-moment array: checkpoints carry them
                let v = self.adam_v.ptr() + (off * 4) as u64;
                let grid = if by_row {
                    ((rows + 7) / 8) as u32
                } else {
                    ((cols + 255) / 256) as u32
                };
                k.normuon_lanes.launch(
                    (grid, 1, 1),
                    (256, 1, 1),
                    0,
                    st,
                    &Args::new()
                        .buf(&scr.x[cur])
                        .ptr(v)
                        .ptr(lane_pow)
                        .ptr(gain)
                        .i32(rows as i32)
                        .i32(cols as i32)
                        .i32(by_row as i32)
                        .f32(o.normuon_beta2)
                        .buf(&self.ts),
                )?;
                k.normuon_ratio.launch(
                    (1, 1, 1),
                    (256, 1, 1),
                    0,
                    st,
                    &Args::new()
                        .ptr(lane_pow)
                        .ptr(gain)
                        .i32(lanes as i32)
                        .i32(len as i32)
                        .ptr(ratio)
                        .buf(&self.ts),
                )?;
                if by_row {
                    1
                } else {
                    2
                }
            }
        };
        k.muon_apply_opt.launch(
            (blocks, 1, 1),
            (256, 1, 1),
            0,
            st,
            &Args::new()
                .ptr(self.m32(off))
                .ptr(self.work16.ptr() + (off * 2) as u64)
                .buf(&scr.x[cur])
                .u64(n as u64)
                .i32(cols as i32)
                .f32(lr)
                .f32(wd)
                .f32(scale)
                .buf(&self.ts)
                .ptr(gain)
                .ptr(ratio)
                .i32(lane_mode)
                .i32(o.cautious_wd as i32),
        )?;
        Ok(())
    }

    /// Like `step`, but the 2-D hidden matrices (everything except the tied embedding and the norm gains) are updated by Muon (learning rate `lr_muon`, decoupled decay `muon_wd`, momentum `mu`)
    /// and only the embedding and the norm gains by AdamW. The Adam first-moment array doubles as Muon's momentum buffer; the second-moment array stays unused for those matrices.
    #[allow(clippy::too_many_arguments)]
    pub fn step_muon(
        &mut self,
        lr: f32,
        lr_muon: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
        muon_wd: f32,
        mu: f32,
        max_norm: f32,
        loss_scale: f32,
    ) -> Result<StepStats> {
        let st = &self.stream;
        let k = &self.k;
        let n = self.n_params;
        self.ts.upload_f32(&[
            loss_scale,
            0.0,
            0.0,
            1.0 / loss_scale,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ])?;
        k.grad_sumsq.launch(
            (SUMSQ_BLOCKS as u32, 1, 1),
            (256, 1, 1),
            0,
            st,
            &Args::new()
                .buf(&self.grads)
                .u64(n as u64)
                .buf(&self.sumsq_partial)
                .buf(&self.nonfinite_partial),
        )?;
        k.grad_finalize.launch(
            (1, 1, 1),
            (32, 1, 1),
            0,
            st,
            &Args::new()
                .buf(&self.sumsq_partial)
                .buf(&self.nonfinite_partial)
                .i32(SUMSQ_BLOCKS as i32)
                .f32(max_norm)
                .buf(&self.ts),
        )?;
        self.step_count += 1;
        let t = self.step_count as f32;
        let bc1 = 1.0 - beta1.powf(t);
        let bc2 = 1.0 - beta2.powf(t);
        for (lo, hi, mult) in self.adam_groups(true) {
            self.adamw_range(lo, hi, lr * mult, beta1, beta2, eps, weight_decay, bc1, bc2)?;
        }
        // the matrices round robin over the Muon streams (the extra streams start after the gradient norm and the clip factor are known; the main stream waits for all of them)
        let ns = self.mu.len();
        if ns > 1 {
            self.mu_start.record(st)?;
            for s in &self.mu_streams {
                s.wait_event(&self.mu_start)?;
            }
        }
        for i in 0..self.muon_mats.len() {
            let (off, rows, cols) = self.muon_mats[i];
            let j = i % ns;
            let stream = if j == 0 { st } else { &self.mu_streams[j - 1] };
            self.muon_matrix(stream, &self.mu[j], off, rows, cols, lr_muon, muon_wd, mu)?;
        }
        for (s, e) in self.mu_streams.iter().zip(&self.mu_done) {
            e.record(s)?;
            st.wait_event(e)?;
        }
        st.synchronize()?;
        let ts = self.ts.download_f32(4)?;
        if ts[1] != 0.0 {
            self.step_count -= 1;
        }
        Ok(StepStats {
            grad_norm: ts[2],
            found_inf: ts[1] != 0.0,
        })
    }

    /// Test hook: the Muon direction for a host matrix (normalise, `iters` Newton-Schulz iterations, no momentum). The matrix must fit the scratch sized from this model's own matrices.
    pub fn muon_direction(
        &mut self,
        g: &[f32],
        rows: usize,
        cols: usize,
        iters: usize,
    ) -> Result<Vec<f32>> {
        let n = rows * cols;
        if g.len() != n || n * 4 > self.mu[0].u32.len() {
            return Err(Error::msg("matrix does not fit the Muon scratch"));
        }
        self.ts.upload_f32(&[
            1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ])?;
        let gd =
            DeviceBuffer::from_bytes(&g.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>())?;
        let mom = alloc_f32(n)?;
        mom.zero(&self.stream)?;
        let (st, k) = (&self.stream, &self.k);
        let blocks = ((n + 255) / 256).min(1024) as u32;
        let scr = &self.mu[0];
        k.muon_prep.launch(
            (64, 1, 1),
            (256, 1, 1),
            0,
            st,
            &Args::new()
                .buf(&gd)
                .buf(&mom)
                .buf(&scr.u32)
                .u64(n as u64)
                .f32(0.0)
                .buf(&self.ts)
                .buf(&scr.partial),
        )?;
        k.muon_normalize.launch(
            (blocks, 1, 1),
            (256, 1, 1),
            0,
            st,
            &Args::new()
                .buf(&scr.u32)
                .buf(&scr.partial)
                .buf(&scr.x[0])
                .u64(n as u64)
                .buf(&self.ts),
        )?;
        let cur = self.muon_newton_schulz(st, scr, rows, cols, iters)?;
        st.synchronize()?;
        let mut raw = vec![0u8; n * 2];
        scr.x[cur].download(&mut raw)?;
        Ok(raw
            .chunks_exact(2)
            .map(|c| forge_core::half::f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect())
    }

    pub fn stream(&self) -> &Stream {
        &self.stream
    }

    /// All parameters as one flat host array (the fp32 masters).
    pub fn download_params(&self) -> Result<Vec<f32>> {
        self.master.download_f32(self.n_params)
    }

    /// Replace all parameters by a flat host array (and refresh the fp16 working copy); the optimizer state is not touched.
    pub fn upload_params(&mut self, flat: &[f32]) -> Result<()> {
        if flat.len() != self.n_params {
            return Err(Error::msg("parameter array has the wrong length"));
        }
        self.master.upload_f32(flat)?;
        self.refresh_work()
    }

    /// Write parameters, Adam moments and the step counter to `path` (atomically: temp file + rename). Format: one JSON line (config summary, step, loss scale) then three raw f32 arrays.
    pub fn save_checkpoint(&self, path: &std::path::Path, loss_scale: f32) -> Result<()> {
        self.save_checkpoint_with(path, loss_scale, None)
    }

    /// Like `save_checkpoint`, but the parameter section holds `params` (a flat host array, e.g. an average of the parameters) instead of the trainer's own; the moments and the step are the trainer's.
    /// Every tool that reads checkpoints reads such a file like any other.
    pub fn save_checkpoint_params(
        &self,
        path: &std::path::Path,
        loss_scale: f32,
        params: &[f32],
    ) -> Result<()> {
        if params.len() != self.n_params {
            return Err(Error::msg("parameter array has the wrong length"));
        }
        self.save_checkpoint_with(path, loss_scale, Some(params))
    }

    fn save_checkpoint_with(
        &self,
        path: &std::path::Path,
        loss_scale: f32,
        params: Option<&[f32]>,
    ) -> Result<()> {
        use std::io::Write;
        let tmp = path.with_extension("tmp");
        let mut f = std::io::BufWriter::new(std::fs::File::create(&tmp)?);
        let header = format!(
            "{{\"magic\":\"forgeckpt1\",\"n_params\":{},\"step\":{},\"loss_scale\":{},\"cfg\":{{\"n_layer\":{},\"n_embd\":{},\"n_head\":{},\"n_kv\":{},\"head_dim\":{},\"n_ff\":{},\"vocab\":{},\"seq_len\":{},\"qk_norm\":{},\"rope_theta\":{},\"eps\":{},\"untied\":{},\"relu2\":{},\"softcap\":{},\"lambdas\":{},\"skips\":{},\"ve_layers\":[{}],\"ngram\":[{}],\"ngram_bits\":{},\"ngram_dim\":{},\"attn_gate\":{},\"peri_norm\":{},\"emb_norm\":{},\"window\":{},\"window_layers\":[{}]}}}}\n",
            self.n_params,
            self.step_count,
            loss_scale,
            self.cfg.n_layer,
            self.cfg.n_embd,
            self.cfg.n_head,
            self.cfg.n_kv,
            self.cfg.head_dim,
            self.cfg.n_ff,
            self.cfg.vocab,
            self.cfg.seq_len,
            self.cfg.qk_norm,
            self.cfg.rope_theta,
            self.cfg.eps,
            self.cfg.untied,
            self.cfg.relu2,
            self.cfg.softcap,
            self.cfg.lambdas,
            self.cfg.skips,
            self.cfg.ve_layers.iter().map(|l| l.to_string()).collect::<Vec<_>>().join(","),
            self.cfg.ngram.iter().map(|l| l.to_string()).collect::<Vec<_>>().join(","),
            self.cfg.ngram_bits,
            self.cfg.ngram_dim,
            self.cfg.attn_gate,
            self.cfg.peri_norm,
            self.cfg.emb_norm,
            self.cfg.window,
            self.cfg.window_layers.iter().map(|l| l.to_string()).collect::<Vec<_>>().join(",")
        );
        f.write_all(header.as_bytes())?;
        if let Some(p) = params {
            for chunk in p.chunks(16 << 20) {
                let bytes: Vec<u8> = chunk.iter().flat_map(|v| v.to_le_bytes()).collect();
                f.write_all(&bytes)?;
            }
        }
        for buf in [&self.master, &self.adam_m, &self.adam_v]
            .into_iter()
            .skip(params.is_some() as usize)
        {
            // stream in 64 MB pieces
            let total = self.n_params * 4;
            let mut done = 0;
            let mut chunk = vec![0u8; 64 << 20];
            while done < total {
                let n = chunk.len().min(total - done);
                buf.download_at(done, &mut chunk[..n])?;
                f.write_all(&chunk[..n])?;
                done += n;
            }
        }
        f.flush()?;
        drop(f);
        std::fs::rename(&tmp, path)?;
        Ok(())
    }

    /// Load only the parameters of a checkpoint (a fresh optimizer state and step counter 0): the start of fine-tuning from a pre-trained model. The model structure must be the same.
    pub fn load_params(&mut self, path: &std::path::Path) -> Result<()> {
        use std::io::{BufRead, Read};
        let mut f = std::io::BufReader::new(std::fs::File::open(path)?);
        let mut line = String::new();
        f.read_line(&mut line)?;
        let h = forge_core::Json::parse(line.trim())?;
        if h["magic"].as_str() != Some("forgeckpt1")
            || h["n_params"].as_usize() != Some(self.n_params)
        {
            return Err(Error::msg("checkpoint does not match this model"));
        }
        let total = self.n_params * 4;
        let mut done = 0;
        let mut chunk = vec![0u8; 64 << 20];
        while done < total {
            let n = chunk.len().min(total - done);
            f.read_exact(&mut chunk[..n])?;
            self.master.upload_at(done, &chunk[..n])?;
            done += n;
        }
        self.refresh_work()
    }

    /// Load a checkpoint written by `save_checkpoint`; returns the loss scale stored in it.
    pub fn load_checkpoint(&mut self, path: &std::path::Path) -> Result<f32> {
        use std::io::{BufRead, Read, Seek};
        let mut f = std::io::BufReader::new(std::fs::File::open(path)?);
        let mut line = String::new();
        f.read_line(&mut line)?;
        let h = forge_core::Json::parse(line.trim())?;
        if h["magic"].as_str() != Some("forgeckpt1")
            || h["n_params"].as_usize() != Some(self.n_params)
        {
            return Err(Error::msg("checkpoint does not match this model"));
        }
        self.step_count = h["step"].as_usize().unwrap_or(0);
        for (which, buf) in [&self.master, &self.adam_m, &self.adam_v]
            .into_iter()
            .enumerate()
        {
            let total = self.n_params * 4;
            if self.inference && which > 0 {
                f.seek_relative(total as i64)?; // the Adam state is not needed to generate
                continue;
            }
            let mut done = 0;
            let mut chunk = vec![0u8; 64 << 20];
            while done < total {
                let n = chunk.len().min(total - done);
                f.read_exact(&mut chunk[..n])?;
                buf.upload_at(done, &chunk[..n])?;
                done += n;
            }
        }
        self.refresh_work()?;
        Ok(h["loss_scale"].as_f64().unwrap_or(1024.0) as f32)
    }
}

fn st_copy(st: &Stream, dst: u64, src: u64, bytes: usize) -> Result<()> {
    st.copy_d2d(dst, src, bytes)
}

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 33
    }
    fn uniform(&mut self) -> f32 {
        (self.next() as f32 + 0.5) / (1u64 << 31) as f32
    }
    fn normal(&mut self) -> f32 {
        let (u1, u2) = (self.uniform().max(1e-7), self.uniform());
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
    }
}

#[allow(dead_code)]
fn _unused() -> u16 {
    f32_to_f16(0.0)
}
