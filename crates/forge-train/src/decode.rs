//! KV-cache generation from a training checkpoint (any Forge-1 structure: ReLU^2 or SwiGLU MLP, per-layer lambdas, skips, value embeddings, qk-norm, GQA, RoPE, tied or separate head, soft cap,
//! n-gram memory).
//!
//! `Trainer::logits_at` recomputes the whole text for every new token. A `Decoder` keeps the keys (after qk-norm and RoPE) and values (after the value-embedding mixing) of every position in fp16,
//! exactly as the trainer's forward pass produces them, and computes only the new positions:
//!  - `feed` with several tokens (a prompt, or the new part of a chat prompt after the shared prefix) runs the trainer's own kernels over those rows: the same tensor-core GEMMs, RMSNorm, mixing, RoPE and
//!    value-embedding kernels, and flash.cu's attention over the cache (`flash_fwd_64/128` with a position offset; for a query row it walks the same 64-key tiles from position 0 as the training kernel);
//!  - `feed` with ONE token (the decoding loop) runs the fused single-token kernels of decode_f1.cu, captured once as a CUDA graph (one graph per attention variant): per layer one kernel for
//!    mixing + norm + q|k|v projection, the attention for one query with the qk-norm + RoPE + value embedding + cache append of its heads in front (from 256 cached positions on: a separate
//!    post-processing kernel and the split attention, scores of all key tiles in parallel then the softmax chain), output projection + residual, norm + up projection + ReLU^2, down projection +
//!    residual; then the final norm and the head. Their matrix-vector products repeat the GEMM's tensor-core chain along K for one row, so the result is the same bits.
//! Sliding-window models (`window`, `window_layers`): in a window layer both paths mask the keys the way the training kernels do (query i sees i - window < j <= i; tiles below the window are skipped);
//! the cache stays full length. A window of 0 (or a layer outside `window_layers`) runs the unchanged dense kernels.
//! Both paths produce bitwise the logits `logits_at` would (tests/decode.rs and `forge gen-check` check it), so caching changes the speed and nothing else.
//! On a T4 with the 150M production checkpoint: about 510 tokens/s at a short context and 320 at 1,968 cached tokens, against 138 and 22 for `logits_at` (research note "ForgeInfer - Forge-1 KV-Cache
//! Decoding (2026-10)").

use crate::{TrainConfig, Trainer};
use forge_core::{Error, Result};
use forge_cuda::{Args, DeviceBuffer, Function, GraphExec};

mod q8;
pub use q8::Q8Info;

/// warps per block of the matrix-vector kernels (8 output columns per warp); FORGE_KV_GV_WARPS overrides it (1 to 8)
const GV_WARPS: usize = 4;

struct FastKernels {
    embed: Function,
    qkv_post: Function,
    attn_fused: Function,
    attn_gqa: Function,
    attn_flash: Function,
    /// flash.cu's kernel with a sliding window, one block per query head (head dimension 64 with window layers only)
    attn_flash_win: Option<Function>,
    attn_scores: Function,
    attn_pv: Function,
    mixnorm_f16: Function,
    norm_f16: Function,
    vec_res: Function,
    norm_relu2: Function,
    vec_f32: Function,
}

impl FastKernels {
    fn load(tr: &Trainer) -> Result<FastKernels> {
        let m = &tr._module;
        // the matrix-vector products follow the GEMM the trainer uses: k16 steps with the Ampere kernels, k8 steps otherwise
        let sfx = if tr.s80 { "_k16" } else { "" };
        // the matrix-vector kernels: one weight tile through shared memory at a time (default: fastest on the T4), FORGE_KV_GEMV=3 batches of eight tiles per warp, =0 warp-specialised
        let var = match gemv_variant() {
            1 => "_t1",
            3 => "",
            _ => "_ws",
        };
        let g = |n: &str| m.function(&format!("{n}{sfx}{var}"));
        Ok(FastKernels {
            embed: m.function("f1d_embed")?,
            qkv_post: m.function("f1d_qkv_post")?,
            attn_fused: m.function(&format!("f1d_attn_fused_{}", tr.cfg.head_dim))?,
            attn_gqa: m.function(&format!("f1d_attn_gqa_{}", tr.cfg.head_dim))?,
            attn_flash: m.function(&format!("f1d_attn_{}", tr.cfg.head_dim))?,
            attn_flash_win: if tr.cfg.window > 0 && !tr.cfg.window_layers.is_empty() {
                Some(m.function("f1d_attn_win_64")?)
            } else {
                None
            },
            attn_scores: m.function(&format!("f1d_attn_scores_{}", tr.cfg.head_dim))?,
            attn_pv: m.function(&format!("f1d_attn_pv_{}", tr.cfg.head_dim))?,
            mixnorm_f16: g("f1d_gemv_mixnorm_f16")?,
            norm_f16: g("f1d_gemv_norm_f16")?,
            vec_res: g("f1d_gemv_vec_res")?,
            norm_relu2: g("f1d_gemv_norm_relu2")?,
            vec_f32: g("f1d_gemv_vec_f32")?,
        })
    }
}

/// The single-token attention (FORGE_KV_ATTN): `auto` (default) the fused kernel below `split_from` cached positions and the split pair from there on; `fused` one block per KV head that also does
/// the q/k/v post-processing (qk-norm, RoPE, value embedding, cache append) for its heads, the work of a key tile split over four warps; `gqa` the same attention after a separate post-processing
/// kernel; `flash` flash.cu's kernel itself, one block per query head; `split` the scores of all key tiles in parallel, then one warp per (KV head, two output tiles) for the softmax chain and P V.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum AttnKind {
    Auto,
    Fused,
    Gqa,
    Flash,
    Split,
}

/// FORGE_KV_GEMV: 1 = one 32-k weight tile through shared memory at a time per warp (default), 3 = batches of eight tiles per warp, 0 = warp-specialised (one 8-column tile per block of four
/// warps that share the loading). All three are bitwise equal; on the T4 (job kv4, alternating runs) 1 gave 510-515 tokens/s, 0 487-490, 3 467-469.
fn gemv_variant() -> u32 {
    std::env::var("FORGE_KV_GEMV")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1)
}

fn attn_kind_from_env() -> AttnKind {
    match std::env::var("FORGE_KV_ATTN").as_deref() {
        Ok("flash") => AttnKind::Flash,
        Ok("gqa") => AttnKind::Gqa,
        Ok("fused") => AttnKind::Fused,
        Ok("split") => AttnKind::Split,
        _ => AttnKind::Auto,
    }
}

/// Arguments of one matrix-vector kernel launch (pointers 0 = unused).
#[derive(Default, Clone, Copy)]
struct Gv {
    w: u64,
    n: usize,
    k: usize,
    vec: bool,
    a16: u64,
    xin: u64,
    x0: u64,
    xskip: u64,
    sc: u64,
    use_ab: bool,
    use_c: bool,
    xout: u64,
    nw: u64,
    out: u64,
    r: u64,
}

/// the matrix-vector kernels of the single-token step (the fp16 ones in `FastKernels`, the Q8 ones in `q8::Q8Kernels`)
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind {
    /// layer-input mixing + attention norm -> q | k | v (fp16 out)
    MixNorm,
    /// norm -> fp16 out (q | k | v without mixing, SwiGLU gate | up)
    Norm,
    /// fp16 vector -> fp32 out + residual (attention output projection, MLP down projection)
    VecRes,
    /// norm -> up projection + ReLU^2 (fp16 out)
    NormRelu2,
    /// fp16 vector -> fp32 out (the head's logits)
    VecF32,
}

pub struct Decoder {
    max_ctx: usize,
    chunk: usize,
    /// the tokens whose keys and values are in the cache (positions 0 .. ids.len())
    ids: Vec<u32>,
    kc: DeviceBuffer,
    vc: DeviceBuffer,
    /// u32 [max_ctx]: the same tokens on the device (value embeddings, n-gram hashes)
    hist: DeviceBuffer,
    /// the single-token step's (token, position)
    step: DeviceBuffer,
    // activations of up to `chunk` rows (the single-token step uses row 0)
    x0: DeviceBuffer,
    xin: DeviceBuffer,
    ys: [DeviceBuffer; 2],
    xmid: DeviceBuffer,
    outs: Vec<DeviceBuffer>,
    x_final: DeviceBuffer,
    h: DeviceBuffer,
    qkv: DeviceBuffer,
    q: DeviceBuffer,
    o: DeviceBuffer,
    gu: DeviceBuffer,
    act: DeviceBuffer,
    rstd: DeviceBuffer,
    rstd_q: DeviceBuffer,
    rstd_k: DeviceBuffer,
    hf: DeviceBuffer,
    z: DeviceBuffer,
    ng_idx: Vec<DeviceBuffer>,
    ng_m: DeviceBuffer,
    flash: Function,
    /// flash.cu's kernel with a sliding window (a model with window layers only)
    flash_win: Option<Function>,
    fast: Option<FastKernels>,
    /// captured steps: [0] with the single attention kernel, [1] with the split attention
    graph: [Option<GraphExec>; 2],
    /// scores of the split attention: [n_kv][16][max_ctx] fp32
    sbuf: DeviceBuffer,
    /// `Auto` switches to the split attention from this many cached positions on (FORGE_KV_SPLIT_FROM)
    pub split_from: usize,
    /// single tokens go through the fused kernels (off: every feed runs the row path; FORGE_KV_GENERIC=1 sets it off)
    pub use_fast: bool,
    /// the fused step is captured as a CUDA graph after its first direct run (FORGE_KV_NOGRAPH=1 sets it off)
    pub use_graph: bool,
    fast_steps: [usize; 2],
    gv_warps: usize,
    attn: AttnKind,
    /// Q8 weights (`enable_q8`): the single-token step and the prompt pieces read these instead of the trainer's fp16 matrices
    q8: Option<q8::Q8Weights>,
}

fn f32s(n: usize) -> Result<DeviceBuffer> {
    DeviceBuffer::alloc(n.max(1) * 4)
}
fn f16s(n: usize) -> Result<DeviceBuffer> {
    DeviceBuffer::alloc(n.max(1) * 2)
}

impl Decoder {
    /// A cache for `tr.cfg.seq_len` positions (the trainer's RoPE tables cover that many) and scratch for prompt pieces of up to `chunk` rows.
    pub fn new(tr: &Trainer, chunk: usize) -> Result<Decoder> {
        let c = &tr.cfg;
        if c.batch != 1 {
            return Err(Error::unsupported(
                "the KV-cache decoder needs a trainer built with batch 1",
            ));
        }
        // the structure options reproduced here; an option added to the trainer later shows up in `structure()` and is refused (the caller falls back to the no-cache forward)
        let known = TrainConfig {
            untied: c.untied,
            relu2: c.relu2,
            softcap: c.softcap,
            lambdas: c.lambdas,
            skips: c.skips,
            ve_layers: c.ve_layers.clone(),
            ngram: c.ngram.clone(),
            ngram_bits: c.ngram_bits,
            ngram_dim: c.ngram_dim,
            // sliding-window attention is reproduced (the window of every layer goes to the attention kernels)
            window: c.window,
            window_layers: c.window_layers.clone(),
            ..TrainConfig::default()
        };
        if known.structure() != c.structure() {
            return Err(Error::unsupported(format!("the KV-cache decoder does not implement every structure option of this model ({}); only {}", c.structure(), known.structure())));
        }
        let windowed = c.window > 0 && !c.window_layers.is_empty();
        if windowed && c.head_dim != 64 {
            return Err(Error::unsupported(
                "sliding-window attention needs head dimension 64",
            ));
        }
        let max_ctx = c.seq_len;
        let chunk = chunk.clamp(1, max_ctx);
        let (e, qd, kvd, ff) = (c.n_embd, c.q_dim(), c.kv_dim(), c.n_ff);
        let stride = qd + 2 * kvd;
        let mix = c.mixing();
        let n_outs = if !mix {
            0
        } else if c.skips {
            c.n_layer
        } else {
            2
        };
        let mut outs = Vec::new();
        for _ in 0..n_outs {
            outs.push(f32s(chunk * e)?);
        }
        let mut ng_idx = Vec::new();
        for _ in 0..c.ngram.len() {
            ng_idx.push(DeviceBuffer::alloc(max_ctx * 4)?);
        }
        let cache_elems = c.n_layer * max_ctx * kvd;
        let kc = f16s(cache_elems)?;
        let vc = f16s(cache_elems)?;
        let flash = tr._module.function(if c.head_dim == 64 {
            "flash_fwd_64"
        } else {
            "flash_fwd_128"
        })?;
        let flash_win = if windowed {
            Some(tr._module.function("flash_fwd_64_win")?)
        } else {
            None
        };
        // the fused single-token kernels: everything except the n-gram memory (that model goes through the row path), up to 16 query heads per KV head
        let fast_ok = c.ngram.is_empty()
            && c.n_head / c.n_kv <= 16
            && e % 32 == 0
            && qd % 32 == 0
            && ff % 32 == 0;
        let fast = if fast_ok {
            Some(FastKernels::load(tr)?)
        } else {
            None
        };
        let d = Decoder {
            max_ctx,
            chunk,
            ids: Vec::new(),
            kc,
            vc,
            hist: DeviceBuffer::alloc(max_ctx * 4)?,
            step: DeviceBuffer::alloc(16)?,
            x0: f32s(chunk * e)?,
            xin: f32s(chunk * e)?,
            ys: [f32s(chunk * e)?, f32s(chunk * e)?],
            xmid: f32s(chunk * e)?,
            outs,
            x_final: f32s(chunk * e)?,
            h: f16s(chunk * e)?,
            qkv: f16s(chunk * stride)?,
            q: f16s(chunk * qd)?,
            o: f16s(chunk * qd)?,
            gu: f16s(chunk * c.gate_width() * ff)?,
            act: f16s(chunk * ff)?,
            rstd: f32s(chunk)?,
            rstd_q: f32s(chunk * c.n_head)?,
            rstd_k: f32s(chunk * c.n_kv)?,
            hf: f16s(e)?,
            z: f32s(c.vocab)?,
            ng_idx,
            ng_m: f16s(if c.ngram.is_empty() {
                1
            } else {
                chunk * c.ngram_dim
            })?,
            flash,
            flash_win,
            fast,
            graph: [None, None],
            sbuf: f32s(c.n_kv * 16 * ((max_ctx + 63) / 64 * 64))?,
            split_from: std::env::var("FORGE_KV_SPLIT_FROM")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(256),
            use_fast: std::env::var("FORGE_KV_GENERIC").is_err(),
            use_graph: std::env::var("FORGE_KV_NOGRAPH").is_err(),
            fast_steps: [0, 0],
            gv_warps: std::env::var("FORGE_KV_GV_WARPS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(GV_WARPS)
                .clamp(1, 8),
            attn: attn_kind_from_env(),
            q8: None,
        };
        d.hist.zero(&tr.stream)?;
        tr.stream.synchronize()?;
        Ok(d)
    }

    /// positions in the cache
    pub fn len(&self) -> usize {
        self.ids.len()
    }
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }
    /// the tokens in the cache
    pub fn ids(&self) -> &[u32] {
        &self.ids
    }
    pub fn max_ctx(&self) -> usize {
        self.max_ctx
    }
    /// bytes of the key and value cache
    pub fn cache_bytes(&self) -> usize {
        self.kc.len() + self.vc.len()
    }
    /// do single tokens take the fused kernels?
    pub fn fast_path(&self) -> bool {
        self.use_fast && self.fast.is_some()
    }

    /// Choose the single-token attention kernel (tests compare the variants; the captured graph is rebuilt).
    pub fn set_attention(&mut self, kind: AttnKind) {
        self.attn = kind;
        self.graph = [None, None];
        self.fast_steps = [0, 0];
    }

    /// Keep the first `n` positions (prefix reuse: the entries after them are overwritten by whatever is fed next).
    pub fn truncate(&mut self, n: usize) {
        self.ids.truncate(n);
    }

    pub fn reset(&mut self) {
        self.ids.clear();
    }

    /// Append `toks` at the end of the cache and return the logits that follow the last of them (after the soft cap, as `Trainer::logits_at` returns them).
    pub fn feed(&mut self, tr: &Trainer, toks: &[u32]) -> Result<Vec<f32>> {
        if toks.is_empty() {
            return Err(Error::msg("nothing to feed"));
        }
        let p0 = self.ids.len();
        if p0 + toks.len() > self.max_ctx {
            return Err(Error::msg(format!(
                "the context holds {} tokens; {} + {} do not fit",
                self.max_ctx,
                p0,
                toks.len()
            )));
        }
        if let Some(&bad) = toks.iter().find(|&&t| t as usize >= tr.cfg.vocab) {
            return Err(Error::msg(format!(
                "token id {bad} is outside the vocabulary"
            )));
        }
        self.ids.extend_from_slice(toks);
        let r = if toks.len() == 1 && self.fast_path() {
            self.step_fast(tr, p0, toks[0])
        } else {
            let end = p0 + toks.len();
            let mut p = p0;
            let mut r = Ok(());
            while p < end {
                let n = (end - p).min(self.chunk);
                r = self.forward_rows(tr, p, n, p + n == end);
                if r.is_err() {
                    break;
                }
                p += n;
            }
            r
        };
        if let Err(e) = r {
            self.ids.truncate(p0);
            return Err(e);
        }
        tr.stream.synchronize()?;
        let mut z = self.z.download_f32(tr.cfg.vocab)?;
        if tr.cfg.softcap > 0.0 {
            let cap = tr.cfg.softcap;
            for x in z.iter_mut() {
                *x = cap * (*x / cap).tanh();
            }
        }
        Ok(z)
    }

    /// The cached keys and values of positions 0..len (all layers, fp16 bits): for tests that compare two ways of filling the cache.
    pub fn cache_snapshot(&self, tr: &Trainer) -> Result<Vec<u16>> {
        tr.stream.synchronize()?;
        let kvd = tr.cfg.kv_dim();
        let mut out = Vec::new();
        for buf in [&self.kc, &self.vc] {
            for l in 0..tr.cfg.n_layer {
                let mut raw = vec![0u8; self.ids.len() * kvd * 2];
                buf.download_at(l * self.max_ctx * kvd * 2, &mut raw)?;
                out.extend(
                    raw.chunks_exact(2)
                        .map(|c| u16::from_le_bytes([c[0], c[1]])),
                );
            }
        }
        Ok(out)
    }

    fn layer_k(&self, tr: &Trainer, l: usize) -> u64 {
        self.kc.ptr() + (l * self.max_ctx * tr.cfg.kv_dim() * 2) as u64
    }
    fn layer_v(&self, tr: &Trainer, l: usize) -> u64 {
        self.vc.ptr() + (l * self.max_ctx * tr.cfg.kv_dim() * 2) as u64
    }
    /// the output buffer of layer `l` when the layer inputs are mixed
    fn out_ptr(&self, tr: &Trainer, l: usize) -> u64 {
        if tr.cfg.skips {
            self.outs[l].ptr()
        } else {
            self.outs[l % 2].ptr()
        }
    }

    /// The trainer's forward pass over the rows p0 .. p0 + n (their tokens are in `ids`), attention over the cache; writes their keys and values into the cache and, with `logits`, the logits of
    /// the last row into `z`. Mirrors `Trainer::run` kernel for kernel.
    fn forward_rows(&mut self, tr: &Trainer, p0: usize, n: usize, logits: bool) -> Result<()> {
        let c = &tr.cfg;
        let (e, qd, kvd, ff, v) = (c.n_embd, c.q_dim(), c.kv_dim(), c.n_ff, c.vocab);
        let stride = qd + 2 * kvd;
        let st = &tr.stream;
        let k = &tr.k;
        let mix = c.mixing();
        let ne = n * e;
        let grid_e = (((ne + 255) / 256).min(65535) as u32, 1, 1);
        let bytes: Vec<u8> = self.ids[p0..p0 + n]
            .iter()
            .flat_map(|t| t.to_le_bytes())
            .collect();
        self.hist.upload_at(p0 * 4, &bytes)?;
        let tok = self.hist.ptr() + (p0 * 4) as u64;
        let emb_dst = if mix { self.x0.ptr() } else { self.ys[0].ptr() };
        if let Some(q) = &self.q8 {
            q.k.embed_rows.launch(
                (n as u32, 1, 1),
                (256, 1, 1),
                0,
                st,
                &Args::new()
                    .ptr(q.buf.ptr() + q.emb.off as u64)
                    .i32(e as i32)
                    .ptr(tok)
                    .ptr(emb_dst),
            )?;
        } else {
            k.embed_batch.launch(
                (n as u32, 1, 1),
                (256, 1, 1),
                0,
                st,
                &Args::new()
                    .ptr(tr.w16(tr.emb_off))
                    .ptr(tok)
                    .ptr(emb_dst)
                    .i32(n as i32)
                    .i32(e as i32),
            )?;
        }
        let n_ng = c.ngram.len();
        if n_ng > 0 {
            // the n-gram rows of every position from 0 (the hash looks back up to three tokens; the window of `logits_at` starts at position 0 as well)
            let dm = c.ngram_dim;
            let rows = p0 + n;
            for (i, &ord) in c.ngram.iter().enumerate() {
                k.ngram_idx.launch(
                    (((rows + 255) / 256).min(65535) as u32, 1, 1),
                    (256, 1, 1),
                    0,
                    st,
                    &Args::new()
                        .buf(&self.hist)
                        .buf(&self.ng_idx[i])
                        .i32(rows as i32)
                        .i32(self.max_ctx as i32)
                        .i32(ord as i32)
                        .i32(c.ngram_bits as i32),
                )?;
            }
            let tab = |i: usize| if i < n_ng { tr.w16(tr.ng_off[i]) } else { 0 };
            let idx = |i: usize| {
                if i < n_ng {
                    self.ng_idx[i].ptr() + (p0 * 4) as u64
                } else {
                    0
                }
            };
            k.ngram_gather.launch(
                (n as u32, 1, 1),
                (128, 1, 1),
                0,
                st,
                &Args::new()
                    .ptr(tab(0))
                    .ptr(idx(0))
                    .ptr(tab(1))
                    .ptr(idx(1))
                    .ptr(tab(2))
                    .ptr(idx(2))
                    .buf(&self.ng_m)
                    .i32(dm as i32),
            )?;
            tr.gemm_axpy(
                k.gemm_nt_res,
                self.ng_m.ptr(),
                tr.w16(tr.ngw_off),
                emb_dst,
                n,
                e,
                dm,
                dm,
                dm,
                e,
                1.0,
                emb_dst,
                e,
                0.0,
            )?;
        }
        let half = c.head_dim / 2;
        for l in 0..c.n_layer {
            let lo = tr.layer_off[l].clone();
            let ql = self.q8.as_ref().map(|q| q.layers[l]);
            let (kc, vc) = (self.layer_k(tr, l), self.layer_v(tr, l));
            let x_in = if mix {
                let xprev = if l == 0 {
                    self.x0.ptr()
                } else {
                    self.out_ptr(tr, l - 1)
                };
                let src = c.skip_source(l);
                let xskip = src.map(|j| self.out_ptr(tr, j)).unwrap_or(xprev);
                k.mix_fwd.launch(
                    grid_e,
                    (256, 1, 1),
                    0,
                    st,
                    &Args::new()
                        .ptr(xprev)
                        .buf(&self.x0)
                        .ptr(xskip)
                        .buf(&self.xin)
                        .u64(ne as u64)
                        .ptr(self.m32(tr, tr.lam_off + 3 * l))
                        .i32(c.lambdas as i32)
                        .i32(src.is_some() as i32),
                )?;
                self.xin.ptr()
            } else {
                self.ys[l % 2].ptr()
            };
            k.rmsnorm_fwd.launch(
                (((n + 7) / 8) as u32, 1, 1),
                (256, 1, 1),
                0,
                st,
                &Args::new()
                    .ptr(x_in)
                    .ptr(self.m32(tr, lo.attn_norm))
                    .buf(&self.h)
                    .buf(&self.rstd)
                    .i32(n as i32)
                    .i32(e as i32)
                    .f32(c.eps),
            )?;
            let w_qkv = self.weight16(tr, lo.wq, ql.map(|x| x.qkv))?;
            tr.gemm(
                k.gemm_nt_f16,
                self.h.ptr(),
                w_qkv,
                self.qkv.ptr(),
                n,
                stride,
                e,
                e,
                e,
                stride,
            )?;
            let (qn, kn) = if c.qk_norm {
                (self.m32(tr, lo.q_norm), self.m32(tr, lo.k_norm))
            } else {
                (0, 0)
            };
            // rotated q, and k / v straight into the cache at p0 (the RoPE tables are offset so that row r is position p0 + r)
            k.qkv_rope_train.launch(
                ((((n * (c.n_head + 2 * c.n_kv)) + 7) / 8) as u32, 1, 1),
                (256, 1, 1),
                0,
                st,
                &Args::new()
                    .buf(&self.qkv)
                    .buf(&self.q)
                    .ptr(kc + (p0 * kvd * 2) as u64)
                    .ptr(vc + (p0 * kvd * 2) as u64)
                    .ptr(tr.cos.ptr() + (p0 * half * 4) as u64)
                    .ptr(tr.sin.ptr() + (p0 * half * 4) as u64)
                    .ptr(qn)
                    .ptr(kn)
                    .buf(&self.rstd_q)
                    .buf(&self.rstd_k)
                    .i32(n as i32)
                    .i32(n as i32)
                    .i32(c.n_head as i32)
                    .i32(c.n_kv as i32)
                    .i32(c.head_dim as i32)
                    .f32(c.eps),
            )?;
            if lo.ve != usize::MAX {
                k.ve_fwd.launch(
                    (n as u32, 1, 1),
                    (128, 1, 1),
                    0,
                    st,
                    &Args::new()
                        .ptr(vc + (p0 * kvd * 2) as u64)
                        .ptr(self.ve16(tr, lo.ve))
                        .ptr(tok)
                        .ptr(self.m32(tr, lo.ve_sc))
                        .i32(kvd as i32),
                )?;
            }
            let attn_args = Args::new()
                .buf(&self.q)
                .ptr(kc)
                .ptr(vc)
                .buf(&self.o)
                .i32(n as i32)
                .i32(p0 as i32)
                .i32(c.n_head as i32)
                .i32(c.n_kv as i32);
            match (c.window_of(l), &self.flash_win) {
                // a local layer: query i sees the keys i - window < j <= i, the training kernel's rule
                (Some(w), Some(fw)) => fw.launch(
                    (((n + 63) / 64) as u32, c.n_head as u32, 1),
                    (128, 1, 1),
                    0,
                    st,
                    &attn_args.i32(w as i32),
                )?,
                _ => self.flash.launch(
                    (((n + 63) / 64) as u32, c.n_head as u32, 1),
                    (128, 1, 1),
                    0,
                    st,
                    &attn_args,
                )?,
            }
            let w_o = self.weight16(tr, lo.wo, ql.map(|x| x.wo))?;
            tr.gemm_axpy(
                k.gemm_nt_res,
                self.o.ptr(),
                w_o,
                self.xmid.ptr(),
                n,
                e,
                qd,
                qd,
                qd,
                e,
                1.0,
                x_in,
                e,
                0.0,
            )?;
            k.rmsnorm_fwd.launch(
                (((n + 7) / 8) as u32, 1, 1),
                (256, 1, 1),
                0,
                st,
                &Args::new()
                    .buf(&self.xmid)
                    .ptr(self.m32(tr, lo.ffn_norm))
                    .buf(&self.h)
                    .buf(&self.rstd)
                    .i32(n as i32)
                    .i32(e as i32)
                    .f32(c.eps),
            )?;
            let w_up = self.weight16(
                tr,
                if c.relu2 { lo.w_up } else { lo.w_gate },
                ql.map(|x| x.up),
            )?;
            if c.relu2 {
                tr.gemm(
                    k.gemm_nt_f16,
                    self.h.ptr(),
                    w_up,
                    self.gu.ptr(),
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
                    &Args::new()
                        .buf(&self.gu)
                        .buf(&self.act)
                        .u64((n * ff) as u64),
                )?;
            } else {
                tr.gemm(
                    k.gemm_nt_f16,
                    self.h.ptr(),
                    w_up,
                    self.gu.ptr(),
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
                        .buf(&self.gu)
                        .buf(&self.act)
                        .i32(n as i32)
                        .i32(ff as i32),
                )?;
            }
            let next = if l + 1 < c.n_layer {
                if mix {
                    self.out_ptr(tr, l)
                } else {
                    self.ys[(l + 1) % 2].ptr()
                }
            } else {
                self.x_final.ptr()
            };
            let w_down = self.weight16(tr, lo.w_down, ql.map(|x| x.down))?;
            tr.gemm_axpy(
                k.gemm_nt_res,
                self.act.ptr(),
                w_down,
                next,
                n,
                e,
                ff,
                ff,
                ff,
                e,
                1.0,
                self.xmid.ptr(),
                e,
                0.0,
            )?;
        }
        if logits {
            let last = self.x_final.ptr() + ((n - 1) * e * 4) as u64;
            k.rmsnorm_fwd.launch(
                (1, 1, 1),
                (256, 1, 1),
                0,
                st,
                &Args::new()
                    .ptr(last)
                    .ptr(self.m32(tr, tr.out_norm_off))
                    .buf(&self.hf)
                    .buf(&self.rstd)
                    .i32(1)
                    .i32(e as i32)
                    .f32(c.eps),
            )?;
            if let Some(q) = &self.q8 {
                self.gemv_q8(
                    tr,
                    Kind::VecF32,
                    q.head,
                    Gv {
                        a16: self.hf.ptr(),
                        out: self.z.ptr(),
                        ..Default::default()
                    },
                )?;
            } else {
                tr.gemm(
                    k.gemm_nt_f32,
                    self.hf.ptr(),
                    tr.w16(tr.head_off),
                    self.z.ptr(),
                    1,
                    v,
                    e,
                    e,
                    e,
                    v,
                )?;
            }
        }
        Ok(())
    }

    fn gemv(&self, tr: &Trainer, f: Function, a: Gv) -> Result<()> {
        let variant = gemv_variant();
        let w = if variant == 0 { 4 } else { self.gv_warps };
        let tiles = a.n / 8;
        let blocks = if variant == 0 {
            tiles
        } else {
            (tiles + w - 1) / w
        };
        // activation (fp16), the fp32 input of a norm prologue, and the weight staging: two stages of 16 x 512 bytes per block (warp-specialised), 8 tiles of 512 bytes per warp, or one
        let stage = match variant {
            0 => 2 * 16 * 512,
            1 => w * 512,
            _ => w * 4096,
        };
        let smem = a.k * 2 + if a.vec { 0 } else { a.k * 4 } + stage;
        f.launch(
            (blocks as u32, 1, 1),
            ((32 * w) as u32, 1, 1),
            smem as u32,
            &tr.stream,
            &Args::new()
                .ptr(a.w)
                .i32(a.n as i32)
                .i32(a.k as i32)
                .ptr(a.a16)
                .ptr(a.xin)
                .ptr(a.x0)
                .ptr(a.xskip)
                .ptr(a.sc)
                .i32(a.use_ab as i32)
                .i32(a.use_c as i32)
                .ptr(a.xout)
                .ptr(a.nw)
                .f32(tr.cfg.eps)
                .ptr(a.out)
                .ptr(a.r),
        )
    }

    /// One matrix-vector product of the single-token step: the fp16 kernel of `kind` on the trainer's matrix (`a.w`), or, with Q8 weights, the Q8 kernel on `m`.
    fn mv(&self, tr: &Trainer, kind: Kind, m: Option<q8::Q8Mat>, a: Gv) -> Result<()> {
        match (&self.q8, m) {
            (Some(_), Some(m)) => self.gemv_q8(tr, kind, m, a),
            _ => {
                let f = self.fast.as_ref().expect("fused kernels");
                let func = match kind {
                    Kind::MixNorm => f.mixnorm_f16,
                    Kind::Norm => f.norm_f16,
                    Kind::VecRes => f.vec_res,
                    Kind::NormRelu2 => f.norm_relu2,
                    Kind::VecF32 => f.vec_f32,
                };
                self.gemv(tr, func, a)
            }
        }
    }

    /// One token through the fused kernels; position and token come from `self.step` (device), so the same launches serve every position.
    fn enqueue_step(&self, tr: &Trainer, split: bool) -> Result<()> {
        let f = self.fast.as_ref().expect("fused kernels");
        let c = &tr.cfg;
        let st = &tr.stream;
        let k = &tr.k;
        let (e, qd, kvd, ff, v) = (c.n_embd, c.q_dim(), c.kv_dim(), c.n_ff, c.vocab);
        let stride = qd + 2 * kvd;
        let mix = c.mixing();
        let emb_dst = if mix { self.x0.ptr() } else { self.ys[0].ptr() };
        if let Some(q) = &self.q8 {
            q.k.embed.launch(
                (1, 1, 1),
                (256, 1, 1),
                0,
                st,
                &Args::new()
                    .ptr(q.buf.ptr() + q.emb.off as u64)
                    .i32(e as i32)
                    .buf(&self.step)
                    .ptr(emb_dst)
                    .buf(&self.hist),
            )?;
        } else {
            f.embed.launch(
                (1, 1, 1),
                (256, 1, 1),
                0,
                st,
                &Args::new()
                    .ptr(tr.w16(tr.emb_off))
                    .buf(&self.step)
                    .ptr(emb_dst)
                    .buf(&self.hist)
                    .i32(e as i32),
            )?;
        }
        let heads = c.n_head + 2 * c.n_kv;
        for l in 0..c.n_layer {
            let lo = tr.layer_off[l].clone();
            let ql = self.q8.as_ref().map(|q| q.layers[l]);
            let (kc, vc) = (self.layer_k(tr, l), self.layer_v(tr, l));
            let x_in;
            if mix {
                let xprev = if l == 0 {
                    self.x0.ptr()
                } else {
                    self.out_ptr(tr, l - 1)
                };
                let src = c.skip_source(l);
                let xskip = src.map(|j| self.out_ptr(tr, j)).unwrap_or(xprev);
                x_in = self.xin.ptr();
                self.mv(
                    tr,
                    Kind::MixNorm,
                    ql.map(|x| x.qkv),
                    Gv {
                        w: tr.w16(lo.wq),
                        n: stride,
                        k: e,
                        xin: xprev,
                        x0: self.x0.ptr(),
                        xskip,
                        sc: self.m32(tr, tr.lam_off + 3 * l),
                        use_ab: c.lambdas,
                        use_c: src.is_some(),
                        xout: x_in,
                        nw: self.m32(tr, lo.attn_norm),
                        out: self.qkv.ptr(),
                        ..Default::default()
                    },
                )?;
            } else {
                x_in = self.ys[l % 2].ptr();
                self.mv(
                    tr,
                    Kind::Norm,
                    ql.map(|x| x.qkv),
                    Gv {
                        w: tr.w16(lo.wq),
                        n: stride,
                        k: e,
                        xin: x_in,
                        nw: self.m32(tr, lo.attn_norm),
                        out: self.qkv.ptr(),
                        ..Default::default()
                    },
                )?;
            }
            let (qn, kn) = if c.qk_norm {
                (self.m32(tr, lo.q_norm), self.m32(tr, lo.k_norm))
            } else {
                (0, 0)
            };
            let (ve, ve_sc) = if lo.ve != usize::MAX {
                (self.ve16(tr, lo.ve), self.m32(tr, lo.ve_sc))
            } else {
                (0, 0)
            };
            let kind = if split {
                AttnKind::Split
            } else if self.attn == AttnKind::Auto {
                AttnKind::Fused
            } else {
                self.attn
            };
            // the sliding window of this layer for the attention kernels (0: dense causal)
            let win = c.window_of(l).unwrap_or(0) as i32;
            if kind == AttnKind::Fused {
                f.attn_fused.launch(
                    (c.n_kv as u32, 1, 1),
                    (128, 1, 1),
                    0,
                    st,
                    &Args::new()
                        .buf(&self.qkv)
                        .ptr(kc)
                        .ptr(vc)
                        .buf(&self.o)
                        .buf(&self.step)
                        .i32(c.n_head as i32)
                        .i32(c.n_kv as i32)
                        .buf(&tr.cos)
                        .buf(&tr.sin)
                        .ptr(qn)
                        .ptr(kn)
                        .ptr(ve)
                        .ptr(ve_sc)
                        .f32(c.eps)
                        .i32(win),
                )?;
            } else {
                f.qkv_post.launch(
                    (((heads + 7) / 8) as u32, 1, 1),
                    (256, 1, 1),
                    0,
                    st,
                    &Args::new()
                        .buf(&self.qkv)
                        .buf(&self.q)
                        .ptr(kc)
                        .ptr(vc)
                        .buf(&tr.cos)
                        .buf(&tr.sin)
                        .ptr(qn)
                        .ptr(kn)
                        .ptr(ve)
                        .ptr(ve_sc)
                        .buf(&self.step)
                        .i32(c.n_head as i32)
                        .i32(c.n_kv as i32)
                        .i32(c.head_dim as i32)
                        .f32(c.eps),
                )?;
                if kind == AttnKind::Split {
                    // score rows padded to whole 64-key tiles
                    let (kt, s_stride) = (
                        if c.head_dim == 64 { 64 } else { 32 },
                        (self.max_ctx + 63) / 64 * 64,
                    );
                    let s_args = |a: Args| {
                        a.buf(&self.step)
                            .i32(c.n_head as i32)
                            .i32(c.n_kv as i32)
                            .i32(s_stride as i32)
                            .i32(win)
                    };
                    f.attn_scores.launch(
                        ((s_stride / kt) as u32, c.n_kv as u32, 1),
                        (128, 1, 1),
                        0,
                        st,
                        &s_args(Args::new().buf(&self.q).ptr(kc).buf(&self.sbuf)),
                    )?;
                    f.attn_pv.launch(
                        (c.n_kv as u32, (c.head_dim / 16) as u32, 1),
                        (32, 1, 1),
                        0,
                        st,
                        &s_args(Args::new().ptr(vc).buf(&self.sbuf).buf(&self.o)),
                    )?;
                } else {
                    let args = Args::new()
                        .buf(&self.q)
                        .ptr(kc)
                        .ptr(vc)
                        .buf(&self.o)
                        .buf(&self.step)
                        .i32(c.n_head as i32)
                        .i32(c.n_kv as i32);
                    if kind == AttnKind::Gqa {
                        f.attn_gqa.launch(
                            (c.n_kv as u32, 1, 1),
                            (128, 1, 1),
                            0,
                            st,
                            &args.i32(win),
                        )?;
                    } else {
                        // flash.cu's kernel itself: the windowed copy for a local layer
                        let flash_grid = (1, c.n_head as u32, 1);
                        match (win > 0, &f.attn_flash_win) {
                            (true, Some(fw)) => {
                                fw.launch(flash_grid, (128, 1, 1), 0, st, &args.i32(win))?
                            }
                            _ => f.attn_flash.launch(flash_grid, (128, 1, 1), 0, st, &args)?,
                        }
                    }
                }
            }
            self.mv(
                    tr,
                    Kind::VecRes,
                    ql.map(|x| x.wo),
                    Gv {
                        w: tr.w16(lo.wo),
                    n: e,
                    k: qd,
                    vec: true,
                    a16: self.o.ptr(),
                    out: self.xmid.ptr(),
                    r: x_in,
                    ..Default::default()
                },
            )?;
            if c.relu2 {
                self.mv(
                    tr,
                    Kind::NormRelu2,
                    ql.map(|x| x.up),
                    Gv {
                        w: tr.w16(lo.w_up),
                        n: ff,
                        k: e,
                        xin: self.xmid.ptr(),
                        nw: self.m32(tr, lo.ffn_norm),
                        out: self.act.ptr(),
                        ..Default::default()
                    },
                )?;
            } else {
                self.mv(
                    tr,
                    Kind::Norm,
                    ql.map(|x| x.up),
                    Gv {
                        w: tr.w16(lo.w_gate),
                        n: 2 * ff,
                        k: e,
                        xin: self.xmid.ptr(),
                        nw: self.m32(tr, lo.ffn_norm),
                        out: self.gu.ptr(),
                        ..Default::default()
                    },
                )?;
                k.swiglu2.launch(
                    (((ff / 8 + 255) / 256) as u32, 1, 1),
                    (256, 1, 1),
                    0,
                    st,
                    &Args::new()
                        .buf(&self.gu)
                        .buf(&self.act)
                        .i32(1)
                        .i32(ff as i32),
                )?;
            }
            let next = if l + 1 < c.n_layer {
                if mix {
                    self.out_ptr(tr, l)
                } else {
                    self.ys[(l + 1) % 2].ptr()
                }
            } else {
                self.x_final.ptr()
            };
            self.mv(
                    tr,
                    Kind::VecRes,
                    ql.map(|x| x.down),
                    Gv {
                        w: tr.w16(lo.w_down),
                    n: e,
                    k: ff,
                    vec: true,
                    a16: self.act.ptr(),
                    out: next,
                    r: self.xmid.ptr(),
                    ..Default::default()
                },
            )?;
        }
        k.rmsnorm_fwd.launch(
            (1, 1, 1),
            (256, 1, 1),
            0,
            st,
            &Args::new()
                .buf(&self.x_final)
                .ptr(self.m32(tr, tr.out_norm_off))
                .buf(&self.hf)
                .buf(&self.rstd)
                .i32(1)
                .i32(e as i32)
                .f32(c.eps),
        )?;
        self.mv(
                    tr,
                    Kind::VecF32,
                    self.q8.as_ref().map(|q| q.head),
                    Gv {
                        w: tr.w16(tr.head_off),
                n: v,
                k: e,
                vec: true,
                a16: self.hf.ptr(),
                out: self.z.ptr(),
                ..Default::default()
            },
        )?;
        Ok(())
    }

    fn step_fast(&mut self, tr: &Trainer, pos: usize, token: u32) -> Result<()> {
        let mut b = [0u8; 8];
        b[..4].copy_from_slice(&token.to_le_bytes());
        b[4..].copy_from_slice(&(pos as u32).to_le_bytes());
        tr.stream.synchronize()?;
        self.step.upload(&b)?;
        let split = match self.attn {
            AttnKind::Split => true,
            AttnKind::Auto => pos >= self.split_from,
            _ => false,
        };
        let gi = split as usize;
        if self.use_graph && self.graph[gi].is_none() && self.fast_steps[gi] >= 1 {
            // the first step of each variant ran the kernels directly (lazy module loading happens outside the capture); from then on its captured graph is replayed
            match tr
                .stream
                .capture(|| self.enqueue_step(tr, split))
                .and_then(|g| g.instantiate())
            {
                Ok(g) => self.graph[gi] = Some(g),
                Err(e) => {
                    // e.g. another thread used the legacy stream during the capture: keep going without graphs
                    eprintln!("[decode] CUDA graph capture failed ({e}); the step kernels are launched directly from now on");
                    self.use_graph = false;
                }
            }
        }
        match (&self.graph[gi], self.use_graph) {
            (Some(g), true) => g.launch(&tr.stream)?,
            _ => self.enqueue_step(tr, split)?,
        }
        self.fast_steps[gi] += 1;
        Ok(())
    }
}
