//! Q8 weights for the KV-cache `Decoder` (`Decoder::enable_q8`, `forge serve CKPT --q8`).
//!
//! One decoded token streams every weight once, so its speed is the memory bandwidth divided by the bytes of the weights. Q8_0-style weights (int8 and one fp16 scale per 32 weights: 1.0625 bytes per
//! weight against 2) cut those bytes by 47 percent. The big matrices of the model (q|k|v, output projection, up (gate|up), down, the tied embedding / head) are quantized on the device from the fp32
//! master weights into the tiled layout of `kernels/decode_q8f.cu`; the norms, the layer scalars and the value-embedding tables (gathers: little traffic) stay in fp32 / fp16. The activations stay fp16
//! and the accumulation is fp32, so only the weights are rounded (the usual Q8_0 error: well under 0.01 nats of perplexity).
//!
//! The single-token step runs the fused Q8 matrix-vector kernels (same fusions as decode_f1.cu: norm prologues, ReLU^2 epilogue, residual add, the head's fp32 logits). A prompt piece of several tokens
//! dequantizes each matrix into one reusable fp16 scratch buffer and runs the trainer's tensor-core GEMM on it (the dequantization reads 1.06 and writes 2 bytes per weight once per piece: 2.2 ms
//! for the 365M model per piece of up to 512 rows, 6 ms of a 52 ms prompt of 1,219 tokens and 1.8 ms of a 7.7 ms prompt of 15 tokens).
//!
//! With `release`, the trainer's fp32 master weights (4 bytes per parameter) and its fp16 working copy (2 bytes) are freed once the Q8 copy and the small compact copies exist: the 365M model then holds
//! about 0.45 GB of weights on the device instead of 2.2 GB. The trainer itself cannot run the model afterwards (`Trainer::weights_released`).

use super::{Decoder, Kind};
use crate::Trainer;
use forge_core::{Error, Result};
use forge_cuda::{Args, DeviceBuffer, Function};

/// bytes of one super-chunk (64 k of one 16-row tile) in the tiled layout
pub(super) const SC_BYTES: usize = 1088;

pub(super) struct Q8Kernels {
    mixnorm_f16: Function,
    norm_f16: Function,
    vec_res: Function,
    norm_relu2: Function,
    vec_f32: Function,
    quantize: Function,
    pub dequant: Function,
    pub embed: Function,
    pub embed_rows: Function,
}

impl Q8Kernels {
    fn load(tr: &Trainer) -> Result<Q8Kernels> {
        let m = &tr._module;
        Ok(Q8Kernels {
            mixnorm_f16: m.function("q8f_gemv_mixnorm_f16")?,
            norm_f16: m.function("q8f_gemv_norm_f16")?,
            vec_res: m.function("q8f_gemv_vec_res")?,
            norm_relu2: m.function("q8f_gemv_norm_relu2")?,
            vec_f32: m.function("q8f_gemv_vec_f32")?,
            quantize: m.function("q8f_quantize")?,
            dequant: m.function("q8f_dequant")?,
            embed: m.function("q8f_embed")?,
            embed_rows: m.function("q8f_embed_rows")?,
        })
    }
    pub fn gemv(&self, kind: Kind) -> Function {
        match kind {
            Kind::MixNorm => self.mixnorm_f16,
            Kind::Norm => self.norm_f16,
            Kind::VecRes => self.vec_res,
            Kind::NormRelu2 => self.norm_relu2,
            Kind::VecF32 => self.vec_f32,
        }
    }
}

/// One quantized matrix: byte offset in the Q8 buffer, rows (outputs, a multiple of 16) and columns (inputs, a multiple of 64).
#[derive(Clone, Copy, Debug)]
pub(super) struct Q8Mat {
    pub off: usize,
    pub n: usize,
    pub k: usize,
}

impl Q8Mat {
    pub fn bytes(&self) -> usize {
        (self.n / 16) * (self.k / 64) * SC_BYTES
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) struct Q8Layer {
    pub qkv: Q8Mat,
    pub wo: Q8Mat,
    /// `up` of a ReLU^2 MLP, `gate | up` (2 ff rows) of a SwiGLU one
    pub up: Q8Mat,
    pub down: Q8Mat,
}

/// The parameters the decoder still reads in fp32 / fp16 once the trainer's own copies are freed: ranges of the flat parameter array (offset in elements, length), packed one after the other.
pub(super) struct Compact {
    seg32: Vec<(usize, usize, usize)>,
    buf32: DeviceBuffer,
    seg16: Vec<(usize, usize, usize)>,
    buf16: DeviceBuffer,
}

impl Compact {
    fn lookup(segs: &[(usize, usize, usize)], off: usize) -> usize {
        for &(orig, len, new) in segs {
            if off >= orig && off < orig + len {
                return new + (off - orig);
            }
        }
        panic!("parameter offset {off} is not part of the compact Q8 copy");
    }
    pub fn m32(&self, off: usize) -> u64 {
        self.buf32.ptr() + (Self::lookup(&self.seg32, off) * 4) as u64
    }
    pub fn w16(&self, off: usize) -> u64 {
        self.buf16.ptr() + (Self::lookup(&self.seg16, off) * 2) as u64
    }
}

pub(super) struct Q8Weights {
    pub k: Q8Kernels,
    pub buf: DeviceBuffer,
    pub layers: Vec<Q8Layer>,
    pub head: Q8Mat,
    /// the embedding table (the head's matrix when the head is tied)
    pub emb: Q8Mat,
    /// fp16 [rows][K] of the largest layer matrix: the dequantized weights of the matrix a prompt-piece GEMM is working on
    pub scratch: DeviceBuffer,
    pub compact: Option<Compact>,
    /// matrix-vector launch shape: warps per CTA (FORGE_Q8_WARPS), fixed k-slices per tile (FORGE_Q8_SPLIT, 0 = by the number of tiles)
    pub warps: usize,
    pub split: usize,
    pub matrix_bytes: usize,
}

impl Q8Weights {
    /// (warps per CTA, warps sharing a tile) of a matrix-vector product with `tiles` 16-row tiles and `k` columns: enough warps in flight (about 128 tiles x slices), at most 4 slices.
    pub fn shape(&self, tiles: usize, k: usize) -> (usize, usize) {
        let w = self.warps;
        let nsc = k / 64;
        let mut s = if self.split > 0 {
            self.split
        } else {
            let mut s = 1;
            while s < 4 && tiles * s < 128 {
                s *= 2;
            }
            s
        };
        s = s.min(w).min(nsc.max(1));
        while w % s != 0 {
            s -= 1;
        }
        (w, s.max(1))
    }
}

/// What `Decoder::enable_q8` did.
#[derive(Clone, Debug)]
pub struct Q8Info {
    /// bytes of the Q8 matrices on the device
    pub matrix_bytes: usize,
    /// bytes of the fp32 / fp16 parameters that stay (norms, scalars, value-embedding tables), 0 when the trainer's own copies are still there
    pub small_bytes: usize,
    /// bytes of fp32 master weights and fp16 working copy that were freed
    pub freed_bytes: usize,
    /// milliseconds for the quantization on the device
    pub quantize_ms: f64,
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

impl Decoder {
    /// Is the single-token step (and the prompt pieces' weights) using Q8 weights?
    pub fn is_q8(&self) -> bool {
        self.q8.is_some()
    }

    /// Bytes of device memory the weights of this decoder take in Q8 form (matrices plus the compact fp32 / fp16 copies), 0 without Q8.
    pub fn q8_weight_bytes(&self) -> usize {
        self.q8.as_ref().map_or(0, |q| {
            q.matrix_bytes
                + q.compact
                    .as_ref()
                    .map_or(0, |c| c.buf32.len() + c.buf16.len())
        })
    }

    /// Quantize the weight matrices of the model to Q8 (from the fp32 master weights of `tr`) and switch this decoder to them. With `release`, the trainer's fp32 master weights and fp16 working copy
    /// are freed afterwards (everything the decoder still reads in fp32 / fp16 is copied to small buffers first): the trainer then refuses to run the model on its own. Needs the fused single-token
    /// path (no n-gram memory) and widths that are multiples of 64.
    pub fn enable_q8(&mut self, tr: &mut Trainer, release: bool) -> Result<Q8Info> {
        if self.q8.is_some() {
            return Err(Error::msg("the decoder already uses Q8 weights"));
        }
        if tr.weights_released {
            return Err(Error::msg("the trainer's weights were already released"));
        }
        let c = tr.cfg.clone();
        if !self.fast_path() {
            return Err(Error::unsupported("Q8 weights need the fused single-token path (no n-gram memory, FORGE_KV_GENERIC unset)"));
        }
        let (e, qd, kvd, ff, v) = (c.n_embd, c.q_dim(), c.kv_dim(), c.n_ff, c.vocab);
        if e % 64 != 0 || qd % 64 != 0 || ff % 64 != 0 || (qd + 2 * kvd) % 16 != 0 || v % 16 != 0 {
            return Err(Error::unsupported(
                "Q8 weights need n_embd, q_dim and n_ff to be multiples of 64",
            ));
        }
        let k = Q8Kernels::load(tr)?;
        let gw = c.gate_width();
        // the matrices, in the order of the buffer
        let mut mats: Vec<(usize, usize, usize)> = Vec::new(); // (source offset in the flat parameters, rows, columns)
        for l in 0..c.n_layer {
            let lo = &tr.layer_off[l];
            mats.push((lo.wq, qd + 2 * kvd, e));
            mats.push((lo.wo, e, qd));
            mats.push((if c.relu2 { lo.w_up } else { lo.w_gate }, gw * ff, e));
            mats.push((lo.w_down, e, ff));
        }
        mats.push((tr.head_off, v, e));
        let separate_emb = tr.emb_off != tr.head_off;
        if separate_emb {
            mats.push((tr.emb_off, v, e));
        }
        let total: usize = mats
            .iter()
            .map(|&(_, n, kk)| (n / 16) * (kk / 64) * SC_BYTES)
            .sum();
        let buf = DeviceBuffer::alloc(total)?;
        let t0 = std::time::Instant::now();
        let mut off = 0usize;
        let mut placed: Vec<Q8Mat> = Vec::new();
        for &(src, n, kk) in &mats {
            let m = Q8Mat { off, n, k: kk };
            let warps = n * (kk / 32);
            k.quantize.launch(
                (((warps + 7) / 8) as u32, 1, 1),
                (256, 1, 1),
                0,
                &tr.stream,
                &Args::new()
                    .ptr(tr.m32(src))
                    .i32(n as i32)
                    .i32(kk as i32)
                    .ptr(buf.ptr() + off as u64),
            )?;
            off += m.bytes();
            placed.push(m);
        }
        tr.stream.synchronize()?;
        let quantize_ms = t0.elapsed().as_secs_f64() * 1e3;
        let mut layers = Vec::new();
        for l in 0..c.n_layer {
            layers.push(Q8Layer {
                qkv: placed[4 * l],
                wo: placed[4 * l + 1],
                up: placed[4 * l + 2],
                down: placed[4 * l + 3],
            });
        }
        let head = placed[4 * c.n_layer];
        let emb = if separate_emb {
            placed[4 * c.n_layer + 1]
        } else {
            head
        };
        // scratch for the dequantized matrix of a prompt-piece GEMM
        let scratch_elems = layers
            .iter()
            .flat_map(|l| [l.qkv, l.wo, l.up, l.down])
            .map(|m| m.n * m.k)
            .max()
            .unwrap_or(1);
        let scratch = DeviceBuffer::alloc(scratch_elems * 2)?;
        let mut q = Q8Weights {
            k,
            buf,
            layers,
            head,
            emb,
            scratch,
            compact: None,
            warps: env_usize("FORGE_Q8_WARPS", 4).clamp(1, 8),
            split: env_usize("FORGE_Q8_SPLIT", 0),
            matrix_bytes: total,
        };
        let mut info = Q8Info {
            matrix_bytes: total,
            small_bytes: 0,
            freed_bytes: 0,
            quantize_ms,
        };
        if release {
            // what the decoder still reads in fp32: the norms (attention, MLP, q, k per layer, the final one) from the end of the matrices on, the layer scalars and the value-embedding scalars;
            // in fp16: the value-embedding tables
            let mut s32: Vec<(usize, usize)> =
                vec![(tr.decay_end, tr.out_norm_off + e - tr.decay_end)];
            if c.mixing() {
                s32.push((tr.lam_off, 3 * c.n_layer));
            }
            if !c.ve_layers.is_empty() {
                s32.push((tr.velam_off, 2 * c.ve_layers.len()));
            }
            let mut s16: Vec<(usize, usize)> = Vec::new();
            if !c.ve_layers.is_empty() {
                let first = tr.layer_off[c.ve_layers[0]].ve;
                let last = tr.layer_off[*c.ve_layers.last().unwrap()].ve + v * kvd;
                s16.push((first, last - first));
            }
            let pack = |segs: &[(usize, usize)]| -> (Vec<(usize, usize, usize)>, usize) {
                let mut out = Vec::new();
                let mut at = 0usize;
                for &(o, len) in segs {
                    out.push((o, len, at));
                    at += len;
                }
                (out, at)
            };
            let (seg32, n32) = pack(&s32);
            let (seg16, n16) = pack(&s16);
            let buf32 = DeviceBuffer::alloc(n32.max(1) * 4)?;
            let buf16 = DeviceBuffer::alloc(n16.max(1) * 2)?;
            for &(o, len, at) in &seg32 {
                tr.stream
                    .copy_d2d(buf32.ptr() + (at * 4) as u64, tr.m32(o), len * 4)?;
            }
            for &(o, len, at) in &seg16 {
                // the fp16 value-embedding tables exactly as the trainer's working copy holds them
                tr.stream
                    .copy_d2d(buf16.ptr() + (at * 2) as u64, tr.w16(o), len * 2)?;
            }
            tr.stream.synchronize()?;
            info.small_bytes = n32 * 4 + n16 * 2;
            info.freed_bytes = tr.master.len() + tr.work16.len();
            q.compact = Some(Compact {
                seg32,
                buf32,
                seg16,
                buf16,
            });
            tr.master = DeviceBuffer::alloc(4)?;
            tr.work16 = DeviceBuffer::alloc(2)?;
            tr.weights_released = true;
        }
        self.q8 = Some(q);
        // the captured graphs hold the fp16 kernels
        self.graph = [None, None];
        self.fast_steps = [0, 0];
        Ok(info)
    }

    /// fp32 parameter at `off` of the flat array: the trainer's master copy, or the compact copy once the trainer's own was released
    pub(super) fn m32(&self, tr: &Trainer, off: usize) -> u64 {
        match self.q8.as_ref().and_then(|q| q.compact.as_ref()) {
            Some(c) => c.m32(off),
            None => tr.m32(off),
        }
    }

    /// fp16 value-embedding table at `off` (the trainer's working copy, or the compact copy)
    pub(super) fn ve16(&self, tr: &Trainer, off: usize) -> u64 {
        match self.q8.as_ref().and_then(|q| q.compact.as_ref()) {
            Some(c) => c.w16(off),
            None => tr.w16(off),
        }
    }

    /// Launch one Q8 matrix-vector kernel on `m`; `a` carries the activation and output pointers as for the fp16 kernels (`a.w` is not used).
    pub(super) fn gemv_q8(&self, tr: &Trainer, kind: Kind, m: Q8Mat, a: super::Gv) -> Result<()> {
        let q = self.q8.as_ref().expect("Q8 weights");
        let tiles = m.n / 16;
        let (w, s) = q.shape(tiles, m.k);
        let tb = w / s;
        let blocks = (tiles + tb - 1) / tb;
        let smem = 16 + m.k * 2 + w * 64 + 128;
        q.k.gemv(kind).launch(
            (blocks as u32, 1, 1),
            ((32 * w) as u32, 1, 1),
            smem as u32,
            &tr.stream,
            &Args::new()
                .ptr(q.buf.ptr() + m.off as u64)
                .i32(m.n as i32)
                .i32(m.k as i32)
                .i32(s as i32)
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

    /// The fp16 weights a prompt-piece GEMM reads: the trainer's working copy at `off`, or (Q8) the matrix `m` dequantized into the scratch buffer (valid until the next call).
    pub(super) fn weight16(&self, tr: &Trainer, off: usize, m: Option<Q8Mat>) -> Result<u64> {
        match (&self.q8, m) {
            (Some(q), Some(m)) => {
                // one warp per 512-byte block (16 rows x 32 columns)
                let warps = (m.n / 16) * (m.k / 32);
                q.k.dequant.launch(
                    (((warps + 7) / 8) as u32, 1, 1),
                    (256, 1, 1),
                    0,
                    &tr.stream,
                    &Args::new()
                        .ptr(q.buf.ptr() + m.off as u64)
                        .i32(m.n as i32)
                        .i32(m.k as i32)
                        .ptr(q.scratch.ptr()),
                )?;
                Ok(q.scratch.ptr())
            }
            _ => Ok(tr.w16(off)),
        }
    }
}
