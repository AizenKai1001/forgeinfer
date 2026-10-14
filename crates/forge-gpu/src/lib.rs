//! GPU backend: weights resident on the device (fp16 or Q8_0), the fused decode step (embedding, five stages per layer, final norm, head, argmax) captured once as CUDA graphs and replayed.
//! Per-token variables (token id, position) live in a device-resident step state, so consecutive greedy tokens need no host round trip.

use forge_core::half::f32_to_f16;
use forge_core::{Error, Result};
use forge_cuda::{Args, Device, DeviceBuffer, Event, Function, GraphExec, Module, PinnedBuffer, Stream};
use std::ffi::c_void;
use std::sync::{Arc, Mutex};
use forge_model::{Config, Mat, Model};

pub mod cpuk;
mod mrow;
mod qrepack;
pub use mrow::VERIFY_ROWS_MAX;
pub use qrepack::{repack, QFmt};

/// Storage format of the matrices on the device.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fmt {
    F16,
    /// int8 + one fp16 scale per 32 weights (llama.cpp's Q8_0 values, planar layout)
    Q8_0,
    /// every tensor in the format it has in the GGUF file (Q8_0, Q4_0, Q5_0, Q4_K, Q5_K, Q6_K), repacked to the planar layouts of the quantized kernels
    Native,
}

/// Kernel variants hoisting NI 16-byte chunks per lane (the row length decides which one fits).
struct NiKernels {
    list: Vec<(usize, Function)>,
    /// elements per 16-byte chunk (8 halves or 16 int8)
    chunk: usize,
}

impl NiKernels {
    fn load(module: &Module, base: &str, variants: &[usize], chunk: usize) -> Result<NiKernels> {
        let mut list = Vec::new();
        for &ni in variants {
            list.push((ni, module.function(&format!("{base}_{ni}"))?));
        }
        Ok(NiKernels { list, chunk })
    }
    /// The variant of a one-block kernel with 512 threads for vectors of `n` elements (NI elements per thread).
    fn for_nq(&self, n: usize) -> Result<Function> {
        let need = (n + 511) / 512;
        self.list
            .iter()
            .find(|(ni, _)| *ni >= need)
            .map(|(_, f)| *f)
            .ok_or_else(|| {
                Error::unsupported(format!("vector length {n} is too long for norm_quant"))
            })
    }
    /// The variant for rows of `k` elements.
    fn for_k(&self, k: usize) -> Result<Function> {
        let need = ((k / self.chunk) + 31) / 32;
        self.list
            .iter()
            .find(|(ni, _)| *ni >= need)
            .map(|(_, f)| *f)
            .ok_or_else(|| {
                Error::unsupported(format!(
                    "row length {k} needs more than {} chunks per lane",
                    self.list.last().unwrap().0
                ))
            })
    }
}

/// The kernels of one quantized stage for every weight format (compile-time specialised: one code path each) plus, for stages whose tensors can differ in format, a runtime-switch variant.
struct FmtKernels {
    by_fmt: Vec<NiKernels>,
    generic: Option<NiKernels>,
    /// streaming kernels for rows too long for the register-resident ones (mvq_res only): one per format, any row length
    big: Vec<Function>,
}

impl FmtKernels {
    fn load(module: &Module, stage: &str, with_generic: bool) -> Result<FmtKernels> {
        let mut by_fmt = Vec::new();
        for suf in ["q8", "q40", "q50", "q4k", "q5k", "q6k"] {
            by_fmt.push(NiKernels::load(
                module,
                &format!("{stage}_{suf}_kernel"),
                &NI_Q8P,
                16,
            )?);
        }
        let generic = if with_generic {
            Some(NiKernels::load(
                module,
                &format!("{stage}_x_kernel"),
                &NI_Q8P,
                16,
            )?)
        } else {
            None
        };
        let mut big = Vec::new();
        if stage == "mvq_res" {
            for suf in ["q8", "q40", "q50", "q4k", "q5k", "q6k"] {
                big.push(module.function(&format!("mvqb_res_{suf}_kernel"))?);
            }
        }
        Ok(FmtKernels {
            by_fmt,
            generic,
            big,
        })
    }

    /// Tasks per lane of a row of `k` elements, and the registers a pipelined (double-buffered) row plus the activation would need: rows beyond the register budget use the streaming kernel.
    fn use_big(f: QFmt, k: usize) -> bool {
        let tasks = match f {
            QFmt::Q8_0 => k / 16,
            QFmt::Q6_K => k / 64,
            _ => k / 32,
        };
        let n = (tasks + 31) / 32;
        let (row, act) = match f {
            QFmt::Q8_0 => (5 * n, 5 * n),
            QFmt::Q4_0 | QFmt::Q5_0 => (6 * n, 10 * n),
            QFmt::Q4_K | QFmt::Q5_K => (9 * n, 12 * n),
            QFmt::Q6_K => (14 * n, 20 * n),
            // host-resident formats: the device decode kernels never see them
            QFmt::Q3_K | QFmt::Q2_K => (14 * n, 20 * n),
        };
        let limit: usize = std::env::var("FORGE_BIG_REGS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(170);
        2 * row + act + 25 > limit
    }

    /// The variant for rows of `k` elements when every tensor of the launch has format `f`.
    fn uniform(&self, f: QFmt, k: usize) -> Result<Function> {
        self.by_fmt[f as usize].for_k(k)
    }

    /// The kernel of the residual matvec for rows of `k` elements and its dynamic shared memory: register-resident rows, or the streaming kernel for long ones.
    fn mv_fn(&self, f: QFmt, k: usize) -> Result<(Function, u32)> {
        if Self::use_big(f, k) {
            // the long-row kernels keep the activation vector in shared memory with its 16-byte chunks swizzled inside groups of 8
            if k % 128 != 0 {
                return Err(Error::unsupported(format!("a row of {k} elements is too long for the register-resident kernels and not a multiple of 128 for the streaming ones")));
            }
            Ok((
                self.big[f as usize],
                ((k + k / 32 * 8 + 15) / 16 * 16) as u32,
            ))
        } else {
            Ok((self.uniform(f, k)?, 0))
        }
    }

    /// The variant for tensors that may differ in format: the specialised one if they agree, else the runtime-switch kernel.
    fn mixed(&self, fs: &[QFmt], k: usize) -> Result<Function> {
        if fs.iter().all(|f| *f == fs[0]) {
            self.uniform(fs[0], k)
        } else {
            self.generic
                .as_ref()
                .ok_or_else(|| Error::unsupported("this stage needs tensors of one format"))?
                .for_k(k)
        }
    }
}

struct Kernels {
    embed: Function,
    embed_q8: Function,
    qk_norm_rope: Function,
    attention: Function,
    final_norm: Function,
    head: Function,
    argmax_partial: Function,
    argmax_final: Function,
    norm_qkv_rope: NiKernels,
    matvec_residual: NiKernels,
    norm_gateup: NiKernels,
    norm_qkv_rope_q8: NiKernels,
    matvec_residual_q8: NiKernels,
    norm_gateup_q8: NiKernels,
    head_q8: NiKernels,
    // quantized decode with pre-quantized activations
    norm_quant: NiKernels,
    quant_vec: NiKernels,
    attention_q: Function,
    qkvq: FmtKernels,
    mvq_res: FmtKernels,
    gateupq: FmtKernels,
    headq: FmtKernels,
    embed_q: Function,
    dequant_test: Function,
    dequant_f16: Function,
    dequant_rows_f16: Function,
    embed_q_batch: Function,
    // hybrid models (Qwen3.5)
    gdn_step: Function,
    qkv35_prep: Function,
    attention_g: Function,
    attn_split_g: Function,
    /// split-KV decode attention of dense models (decode.cu): partial softmax per block of AS_CH positions, exact merge to fp32 / to the quantized activation
    attn_split: Function,
    /// the same straight from a compressed KV cache (no dequantization pass): dense models, hybrid models
    attn_split_q: Function,
    attn_split_g_q: Function,
    /// the same for the rows of a multi-row step, four rows per block (they share the loads of keys and values; bit-identical partial results)
    attn_split_rows: Function,
    /// GQA-shared split attention (decode.cu `attn_gqa_impl`): one block per key-value head for the query heads that share it, bit-identical partial results. (head width, query heads per block padded to a power
    /// of two, compressed cache, kernel) for every instantiation of ATTN_GQA_KERNELS
    gqa: Vec<(usize, usize, bool, Function)>,
    attn_merge: Function,
    attn_merge_q: Function,
    attn_merge_g: Function,
    gdn_prep: Function,
    gdn_prep_tile: Function,
    /// the scan with 32 / 64 / 128 state columns per block (one column per 8 lanes), and the blocks of 128 threads with 16 / 32 / 64 columns of 1 / 2 / 4 columns per thread
    gdn_scan: [Function; 3],
    gdn_scan_nc: [Function; 3],
    gdn_post: Function,
    qkv35_rope_batch: Function,
    attn_prefill_256g: Function,
    flash_256g: Function,
    /// the Ampere attention kernels of flash_a80.cu (m16n8k16 MMAs, cp.async K / V tiles): Qwen3.5's gated head width 256 (warp pairs, 64 KB of dynamic shared memory), and the dense head widths [64, 128]
    /// (empty kernels below sm_80: the session picks them only from compute capability 8)
    flash_256g_a80: Function,
    flash_a80: [Function; 2],
    gemm_f32_out: Function,
    // MTP head
    mtp_prep: Function,
    mtp_prep_batch: Function,
    // two-token decode (speculative verification)
    embed_q_multi: Function,
    gdn_step2: Function,
    swiglu_quant_t: Function,
    argmax_rows: Function,
    argmax_rows_p: Function,
    argmax_rows_f: Function,
    mvt2_out: Vec<Function>,
    mvt2_res: Vec<Function>,
    // multi-row matvecs ([variant: 4, 8, 12, 16 rows][format]) and the glue of the multi-row prompt path (decode_m.cu)
    mm_out: Vec<Vec<Function>>,
    mm_res: Vec<Vec<Function>>,
    mm_reduce_out: Function,
    mm_reduce_res: Function,
    qkv_prep_multi: Function,
    gdn_stepn: Function,
    gdn_stepn_side: Function,
    // sampling on the device
    penalty: Function,
    logit_max: Function,
    collect: Function,
    // prefill
    gemm_f16_out: Function,
    gemm_f32_add: Function,
    /// fused dequantizing GEMMs (K-quant weights read in place): [Q4_K, Q5_K, Q6_K] x [fp16 out, fp32 out, fp32 add]
    gemmq: Vec<Function>,
    /// the same with fp16 accumulators (one k16 step per partial sum, sm_80 and newer: twice the tensor-core rate on GeForce cards), used when FORGE_F16ACC=1
    gemmq_h: Vec<Function>,
    /// the integer tensor-core GEMMs of gemm_mmq.cu (int8 activations, Q4_K / Q5_K / Q6_K weights read in place; sm_80 and newer, `MMQ_SMEM` / `MMQ6_SMEM` bytes of shared memory): [Q4_K, Q5_K, Q6_K] x [fp16 out, fp32 out, fp32 add]
    mmq: Vec<Function>,
    /// the device accepted the shared-memory size of `mmq`
    mmq_ok: bool,
    /// fp16 [M][K] -> int8 plus a scale and a sum per 32 values (the activation arrays of the decode kernels and the input of `mmq`)
    quant_rows_h: Function,
    swiglu_quant_rows: Function,
    fakeq_rows: Function,
    /// the plain tensor-core GEMMs of gemm_s80.cu (cp.async pipeline, m16n8k16; sm_80 and newer): C (+)= A B^T for contiguous A [M][K] and B [N][K], [fp16 out, fp32 out, fp32 add]; `gemm_s80h` accumulates in fp16
    /// (flushed into fp32 after every k16 step): 41 against 31 TFLOPs on the RTX 4060. They multiply a dequantized weight matrix (the long chunks of a prompt) and need `S80_SMEM` bytes of shared memory.
    gemm_s80: Vec<Function>,
    gemm_s80h: Vec<Function>,
    /// stand-alone fp16 dequantization of K-quant matrices: [Q4_K, Q5_K, Q6_K]
    dequantk: Vec<Function>,
    embed_batch: Function,
    rmsnorm_rows: Function,
    colsq: Function,
    gemm_tn_add: Function,
    qkv_rope: Function,
    attn_prefill_64: Function,
    attn_prefill_128: Function,
    flash_64: Function,
    flash_128: Function,
    swiglu: Function,
    copy_row: Function,
    set_pos: Function,
    add_f32: Function,
    kv_dequant: Function,
    kv_quant: Function,
}

/// A weight matrix on the device: fp16 values, or int8 values plus fp16 scales.
struct GpuMat {
    w: DeviceBuffer,
    s: Option<DeviceBuffer>,
    /// planes 3 and 4 of the K-quant / Q5 layouts (see qrepack.rs)
    x: Vec<DeviceBuffer>,
    /// the format of a quantized tensor (None: fp16)
    qf: Option<QFmt>,
}

impl GpuMat {
    fn bytes(&self) -> usize {
        self.w.len()
            + self.s.as_ref().map(|s| s.len()).unwrap_or(0)
            + self.x.iter().map(|b| b.len()).sum::<usize>()
    }

    /// Device pointers of the planes p0..p3 as the quantized kernels take them (0 for the planes a format does not have).
    fn ptrs(&self) -> [u64; 4] {
        let mut p = [0u64; 4];
        p[0] = self.w.ptr();
        let mut i = 1;
        if let Some(s) = &self.s {
            p[1] = s.ptr();
            i = 2;
        }
        for b in &self.x {
            p[i] = b.ptr();
            i += 1;
        }
        p
    }

    /// Bytes of the planes p0..p3 in the order of `ptrs` (0 for the planes a format does not have).
    fn plane_lens(&self) -> [usize; 4] {
        let mut p = [0usize; 4];
        p[0] = self.w.len();
        let mut i = 1;
        if let Some(s) = &self.s {
            p[1] = s.len();
            i = 2;
        }
        for b in &self.x {
            p[i] = b.len();
            i += 1;
        }
        p
    }

    /// A placeholder for the attention matrices of a Gated DeltaNet layer (and vice versa).
    fn dummy() -> Result<GpuMat> {
        Ok(GpuMat {
            w: DeviceBuffer::alloc(16)?,
            s: None,
            x: Vec::new(),
            qf: None,
        })
    }

    /// From the planes of qrepack::repack; `host`: the planes go to page-locked host memory that the kernels read over PCIe (and the CPU kernels read directly) instead of to the device.
    fn from_planes(qf: QFmt, planes: &[Vec<u8>], host: bool) -> Result<GpuMat> {
        let up = |p: &Vec<u8>| {
            if host {
                DeviceBuffer::from_bytes_host(p)
            } else {
                DeviceBuffer::from_bytes(p)
            }
        };
        let mut it = planes.iter();
        let w = up(it.next().unwrap())?;
        let s = it.next().map(|p| up(p)).transpose()?;
        let x = it.map(|p| up(p)).collect::<Result<Vec<_>>>()?;
        Ok(GpuMat {
            w,
            s,
            x,
            qf: Some(qf),
        })
    }

    /// The matrix lives in host memory (all its planes do).
    fn is_host(&self) -> bool {
        self.w.is_host()
    }

    /// Bytes on the device and bytes in host memory.
    fn split_bytes(&self) -> (usize, usize) {
        if self.is_host() {
            (0, self.bytes())
        } else {
            (self.bytes(), 0)
        }
    }
}

/// Where the weights go. The defaults put everything on the device; `from_env` reads FORGE_HOST_FFN and FORGE_HOST_EMBED.
#[derive(Clone, Debug, Default)]
pub struct UploadOpts {
    /// per layer: the FFN matrices (gate, up, down) live in page-locked host memory, which the GPU reads over PCIe (prefill, verification) and the CPU kernels read directly (decode); empty = none
    pub host_ffn: Vec<bool>,
    /// the token-embedding table lives in host memory (one row is read per token)
    pub host_embed: bool,
}

impl UploadOpts {
    /// FORGE_HOST_FFN = N (N layers, evenly spread), `all`, `list:3,5,9`, `none`; FORGE_HOST_EMBED = 1.
    pub fn from_env(n_layer: usize) -> UploadOpts {
        let mut o = UploadOpts::default();
        if let Ok(v) = std::env::var("FORGE_HOST_FFN") {
            o.host_ffn = Self::parse_layers(&v, n_layer);
        }
        o.host_embed = std::env::var("FORGE_HOST_EMBED").map(|v| v == "1").unwrap_or(false);
        o
    }

    pub fn parse_layers(v: &str, n_layer: usize) -> Vec<bool> {
        let mut sel = vec![false; n_layer];
        let v = v.trim();
        if v.is_empty() || v == "none" || v == "0" {
            return Vec::new();
        }
        if v == "all" {
            sel.iter_mut().for_each(|b| *b = true);
        } else if let Some(list) = v.strip_prefix("list:") {
            for t in list.split(',') {
                if let Ok(i) = t.trim().parse::<usize>() {
                    if i < n_layer {
                        sel[i] = true;
                    }
                }
            }
        } else if let Ok(n) = v.parse::<usize>() {
            let n = n.min(n_layer);
            for i in 0..n_layer {
                if (i + 1) * n / n_layer > i * n / n_layer {
                    sel[i] = true;
                }
            }
        }
        sel
    }

    fn host_ffn_of(&self, li: usize) -> bool {
        self.host_ffn.get(li).copied().unwrap_or(false)
    }
}

struct GpuLayer {
    attn_norm: DeviceBuffer,
    wq: GpuMat,
    wk: GpuMat,
    wv: GpuMat,
    wo: GpuMat,
    q_norm: Option<DeviceBuffer>,
    k_norm: Option<DeviceBuffer>,
    /// biases of the q, k and v projections (Qwen2), fp32
    bq: Option<DeviceBuffer>,
    bk: Option<DeviceBuffer>,
    bv: Option<DeviceBuffer>,
    ffn_norm: DeviceBuffer,
    w_gate: GpuMat,
    w_up: GpuMat,
    w_down: GpuMat,
    delta: Option<GpuDelta>,
    /// the FFN lives in host memory and the CPU kernels can multiply it (decode steps run it there)
    cpu: Option<cpuk::CpuFfn>,
}

impl GpuLayer {
    /// Device addresses of the q, k and v biases (0 = the model has none), as the attention kernels take them.
    fn bias_ptrs(&self) -> (u64, u64, u64) {
        let p = |b: &Option<DeviceBuffer>| b.as_ref().map(|b| b.ptr()).unwrap_or(0);
        (p(&self.bq), p(&self.bk), p(&self.bv))
    }
}

/// The weights of a Gated DeltaNet layer on the device (matrices quantized, the small per-head parameters fp32).
struct GpuDelta {
    w_qkv: GpuMat,
    w_z: GpuMat,
    w_a: GpuMat,
    w_b: GpuMat,
    conv: DeviceBuffer,
    a_neg: DeviceBuffer,
    dt_bias: DeviceBuffer,
    norm: DeviceBuffer,
    w_out: GpuMat,
}

/// The MTP head on the device (matrices quantized to the main model's format, or Q8_0 when the head comes from a Hugging Face checkpoint).
struct GpuMtp {
    fc: GpuMat,
    norm_e: DeviceBuffer,
    norm_h: DeviceBuffer,
    out_norm: DeviceBuffer,
    layer: GpuLayer,
}

pub struct GpuModel {
    pub dev: Device,
    pub cfg: Config,
    pub fmt: Fmt,
    /// all matrices are quantized (Q8_0 or native formats): the pre-quantized-activation decode path
    pub quantized: bool,
    _module: Module,
    k: Kernels,
    tok_emb: GpuMat,
    layers: Vec<GpuLayer>,
    out_norm: DeviceBuffer,
    lm_head: Option<GpuMat>,
    mtp: Option<GpuMtp>,
    pub weight_bytes: usize,
    /// of `weight_bytes`: the part in page-locked host memory (the GPU reads it over PCIe, the CPU kernels directly)
    pub host_weight_bytes: usize,
    /// of `host_weight_bytes`: the feed-forward matrices. A host-resident embedding table is read one row per token and changes nothing about how a prompt is processed; host-resident FFN weights do (a batched call
    /// reads them over PCIe once), so the prompt paths key on this and not on `host_weight_bytes`.
    pub host_ffn_bytes: usize,
    /// fp16 weights of a hybrid model, for evaluation only: the prefill path runs, the decode graphs (quantized kernels) are not built
    eval_only: bool,
}

fn f32_buffer(v: &[f32]) -> Result<DeviceBuffer> {
    let b = DeviceBuffer::alloc(v.len() * 4)?;
    b.upload_f32(v)?;
    Ok(b)
}

/// llama.cpp's Q8_0 quantiser: per 32 values d = amax / 127, q = round(x / d) with the unrounded d, the scale stored as fp16.
pub fn quantize_q8_0(w: &[f32]) -> (Vec<i8>, Vec<u16>) {
    assert!(w.len() % 32 == 0);
    let mut q = vec![0i8; w.len()];
    let mut s = Vec::with_capacity(w.len() / 32);
    for (bi, blk) in w.chunks_exact(32).enumerate() {
        let amax = blk.iter().fold(0f32, |m, v| m.max(v.abs()));
        let d = amax / 127.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        s.push(f32_to_f16(d));
        for (j, v) in blk.iter().enumerate() {
            q[bi * 32 + j] = (v * id).round() as i8;
        }
    }
    (q, s)
}

/// A matrix in its GGUF format (from the raw blocks), or Q8_0 quantized from the f32 values when the file stores it as a float type.
/// Bytes a matrix takes on the device in format `fmt`.
fn mat_device_bytes(m: &Mat, fmt: Fmt) -> usize {
    if m.rows == 0 {
        return 0;
    }
    match (fmt, &m.raw) {
        (Fmt::Native, Some(raw)) => raw.bytes.len(),
        (Fmt::F16, _) => m.rows * m.cols * 2,
        _ => m.rows * m.cols / 32 * 34,
    }
}

/// Working memory of a session next to the weights (bytes): KV cache, recurrent states, the prefill buffers, the dequantization scratch, the CUDA context and the rest.
fn session_reserve(c: &Config, ctx: usize, mtp: bool) -> usize {
    let (e, f, qd, kvd) = (c.n_embd, c.n_ff, c.q_dim(), c.kv_dim());
    let n_dn = c.n_delta_layers();
    let n_attn = c.n_layer - n_dn;
    let kv = match KvCfg::from_env(c) {
        Some(k) => k.total_bytes(c, n_attn, ctx),
        None => n_attn * ctx * kvd * 2 * 2,
    } + if mtp { ctx * kvd * 2 * 2 } else { 0 };
    let (states, hybrid_pf, hybrid_w) = match &c.delta {
        Some(d) => (
            n_dn * (d.n_v * 128 * 128 * 4 + d.n_v * (d.conv - 1) * 384 * 4),
            4 * d.conv_dim() + 16 * d.value_dim(),
            d.conv_dim().max(d.value_dim()) * e,
        ),
        None => (0, 0, 0),
    };
    let pf = prefill_chunk_len() * (6 * f + 6 * e + 2 * (c.q_rows() + 2 * kvd) + 4 * qd + hybrid_pf)
        + 2 * (f * e).max(c.q_rows() * e).max(hybrid_w);
    // a chain of drafts (FORGE_SPEC_DEPTH) keeps the recurrent state after every verified row but the last, and the multi-row buffers
    let depth = std::env::var("FORGE_SPEC_DEPTH").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(1);
    let deep = if depth >= 2 && mtp { (depth - 1) * states + (100 << 20) } else { 0 };
    kv + states + pf + (450 << 20) + deep
}

/// Where the weights go when the user did not say: everything on the device if it fits next to a session's working memory, else the embedding table, then the FFN of as many layers as needed
/// (spread evenly: the CPU kernels read them while the GPU does the rest). FORGE_CTX_HINT (default 4096) sizes the KV cache of the reserve, FORGE_VRAM_RESERVE_MIB adds to it.
fn plan_offload(model: &Model, fmt: Fmt, device_ordinal: usize) -> Result<UploadOpts> {
    let c = &model.cfg;
    let ctx = std::env::var("FORGE_CTX_HINT").ok().and_then(|v| v.parse().ok()).unwrap_or(4096usize).min(c.max_ctx);
    let extra = std::env::var("FORGE_VRAM_RESERVE_MIB").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(0) << 20;
    let reserve = session_reserve(c, ctx, model.mtp.is_some()) + extra + (128 << 20);
    let ffn_bytes: Vec<usize> = model
        .layers
        .iter()
        .map(|l| mat_device_bytes(&l.w_gate, fmt) + mat_device_bytes(&l.w_up, fmt) + mat_device_bytes(&l.w_down, fmt))
        .collect();
    let mut total = mat_device_bytes(&model.tok_emb, fmt) + model.lm_head.as_ref().map(|h| mat_device_bytes(h, fmt)).unwrap_or(0);
    for (l, fb) in model.layers.iter().zip(&ffn_bytes) {
        total += fb;
        total += mat_device_bytes(&l.wq, fmt) + mat_device_bytes(&l.wk, fmt) + mat_device_bytes(&l.wv, fmt) + mat_device_bytes(&l.wo, fmt);
        if let Some(d) = &l.delta {
            total += mat_device_bytes(&d.w_qkv, fmt) + mat_device_bytes(&d.w_z, fmt) + mat_device_bytes(&d.w_a, fmt) + mat_device_bytes(&d.w_b, fmt) + mat_device_bytes(&d.w_out, fmt);
        }
    }
    if let Some(w) = &model.mtp {
        let l = &w.layer;
        total += mat_device_bytes(&w.fc, fmt)
            + mat_device_bytes(&l.wq, fmt)
            + mat_device_bytes(&l.wk, fmt)
            + mat_device_bytes(&l.wv, fmt)
            + mat_device_bytes(&l.wo, fmt)
            + mat_device_bytes(&l.w_gate, fmt)
            + mat_device_bytes(&l.w_up, fmt)
            + mat_device_bytes(&l.w_down, fmt);
    }
    total += 16 << 20; // norms and small tensors
    let mut o = UploadOpts::default();
    // The embedding table of a model with a separate output matrix is read one row (2 to 4 KB) per token: it lives in page-locked host memory, which gives the card 0.3 to 0.6 GiB back for the KV cache and for the
    // rows of a speculative pass, at no measurable cost (Qwen3.5 9B decode 46.6 against 46.5 tokens/s, the rows it frees speed up speculation by up to 13 percent). A tied head reads the whole table every token, so that table
    // stays on the device. FORGE_HOST_EMBED=0 keeps the table on the device too.
    let embed_bytes = mat_device_bytes(&model.tok_emb, fmt);
    if model.lm_head.is_some() && (embed_bytes >= (64 << 20) || host_only(&model.tok_emb)) {
        o.host_embed = true;
        total = total.saturating_sub(embed_bytes);
    }
    // tensors in formats only the CPU reads (Q3_K, Q2_K) are in host memory whatever happens: they do not count against the device
    let mut forced = vec![false; c.n_layer];
    let all_bytes = total;
    for (i, l) in model.layers.iter().enumerate() {
        if host_only(&l.w_gate) || host_only(&l.w_up) || host_only(&l.w_down) {
            forced[i] = true;
            total = total.saturating_sub(ffn_bytes[i]);
        }
    }
    // Whether a thread that waits for the GPU spins or sleeps is fixed when the device's context is created, so it is decided here, before the device is opened: with FFN layers on the CPU (forced by their
    // format, or because the weights and the session do not fit the card) every core counts and the waiter sleeps.
    let cpu_ffn = forced.iter().any(|b| *b)
        || forge_cuda::device_total_mem(device_ordinal).map(|t| all_bytes + reserve > t.saturating_sub(192 << 20)).unwrap_or(false);
    if cpu_ffn {
        forge_cuda::set_blocking_sync_default(true);
    }
    let dev = Device::open(device_ordinal)?;
    let (free, _total) = dev.mem_info()?;
    let avail = free.saturating_sub(reserve);
    if total <= avail {
        o.host_ffn = forced;
        return Ok(o);
    }
    let mut need = total - avail;
    let need0 = need;
    let mut saved = 0usize;
    let mut shortfall = 0usize;
    if model.lm_head.is_some() && !o.host_embed {
        o.host_embed = true;
        saved += mat_device_bytes(&model.tok_emb, fmt);
    }
    if saved < need {
        need -= saved;
        // the layers that may be offloaded: FFN formats with a CPU kernel (the others would be read over PCIe, six to eight times slower)
        let ok = |i: usize| {
            let l = &model.layers[i];
            [&l.w_gate, &l.w_up, &l.w_down].iter().all(|m| match (&m.raw, fmt) {
                (Some(r), _) => QFmt::from_dtype(r.dtype).map(|q| cpuk::HostMat::usable(q, m.cols)).unwrap_or(false),
                (None, Fmt::Q8_0) => m.cols % 32 == 0,
                _ => false,
            })
        };
        let cand: Vec<usize> = (0..model.layers.len()).filter(|&i| ok(i) && !forced[i]).collect();
        // the candidates in a spread-out order (van der Corput: 0, 1/2, 1/4, 3/4, ...): any prefix of it is evenly spread over the layers
        let vdc = |mut k: usize| {
            let (mut f, mut r) = (0.5f64, 0.0f64);
            while k > 0 {
                if k & 1 == 1 {
                    r += f;
                }
                f /= 2.0;
                k >>= 1;
            }
            r
        };
        let mut order: Vec<usize> = (0..cand.len()).collect();
        order.sort_by(|&a, &b| vdc(a).partial_cmp(&vdc(b)).unwrap());
        let mut sel = forced.clone();
        let mut got = 0usize;
        for &k in &order {
            if got >= need {
                break;
            }
            sel[cand[k]] = true;
            got += ffn_bytes[cand[k]];
        }
        o.host_ffn = sel;
        shortfall = need0.saturating_sub(saved + got);
    }
    eprintln!(
        "[gpu] weights {:.2} GiB, free device memory {:.2} GiB, session reserve {:.2} GiB: {} FFN layers{} go to host memory (set FORGE_HOST_FFN=none to force the device)",
        total as f64 / (1u64 << 30) as f64,
        free as f64 / (1u64 << 30) as f64,
        reserve as f64 / (1u64 << 30) as f64,
        o.host_ffn.iter().filter(|b| **b).count(),
        if o.host_embed { " and the embedding table" } else { "" }
    );
    if shortfall > 0 {
        eprintln!(
            "[gpu] WARNING: the weights that must stay on the device exceed the card by {:.2} GiB even with every FFN layer in host memory: a long context, a chain of drafts or a large prefill chunk may run out of device memory (FORGE_KV=int8, FORGE_PREFILL_CHUNK=512, a smaller --ctx, or a recipe with fewer device-resident bytes help)",
            shortfall as f64 / (1u64 << 30) as f64
        );
    }
    Ok(o)
}

/// Formats that only the CPU kernels and the dequantizing prefill kernel read: a tensor in one of them must live in host memory (the device decode kernels do not know them).
fn host_only(m: &Mat) -> bool {
    matches!(m.raw.as_ref().map(|r| r.dtype), Some(forge_gguf::DType::Q3_K) | Some(forge_gguf::DType::Q2_K))
}

/// Load-time accounting (FORGE_LOAD_TRACE=1 prints it): nanoseconds spent repacking blocks into planes and copying the planes to the device.
static REPACK_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static COPY_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

fn upload_native(m: &Mat, host: bool) -> Result<GpuMat> {
    if let Some(raw) = &m.raw {
        let qf = QFmt::from_dtype(raw.dtype)
            .ok_or_else(|| Error::unsupported(format!("tensor format {:?}", raw.dtype)))?;
        if matches!(qf, QFmt::Q3_K | QFmt::Q2_K) && !host {
            return Err(Error::unsupported(format!(
                "{} tensors run only as FFN matrices in host memory (the device decode kernels do not read them); this tensor is not an FFN of a host-resident layer",
                qf.name()
            )));
        }
        let t = std::time::Instant::now();
        let planes = repack(qf, &raw.bytes)?;
        REPACK_NS.fetch_add(t.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
        let t = std::time::Instant::now();
        let g = GpuMat::from_planes(qf, &planes, host);
        COPY_NS.fetch_add(t.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
        return g;
    }
    upload_mat(m, Fmt::Q8_0, host)
}

fn upload_mat(m: &Mat, fmt: Fmt, host: bool) -> Result<GpuMat> {
    let up = |b: &[u8]| {
        if host {
            DeviceBuffer::from_bytes_host(b)
        } else {
            DeviceBuffer::from_bytes(b)
        }
    };
    match fmt {
        Fmt::Native => upload_native(m, host),
        Fmt::F16 => {
            let mut bytes = Vec::with_capacity(m.w.len() * 2);
            for &v in &m.w {
                bytes.extend_from_slice(&f32_to_f16(v).to_le_bytes());
            }
            Ok(GpuMat {
                w: up(&bytes)?,
                s: None,
                x: Vec::new(),
                qf: None,
            })
        }
        Fmt::Q8_0 => {
            if m.cols % 32 != 0 {
                return Err(Error::unsupported(format!(
                    "Q8_0 needs rows that are a multiple of 32 ({} columns)",
                    m.cols
                )));
            }
            let (q, s) = quantize_q8_0(&m.w);
            let qb = unsafe { std::slice::from_raw_parts(q.as_ptr() as *const u8, q.len()) };
            let mut sb = Vec::with_capacity(s.len() * 2);
            for v in s {
                sb.extend_from_slice(&v.to_le_bytes());
            }
            Ok(GpuMat {
                w: up(qb)?,
                s: Some(up(&sb)?),
                x: Vec::new(),
                qf: Some(QFmt::Q8_0),
            })
        }
    }
}

const NI_F16: [usize; 6] = [2, 3, 4, 6, 8, 12];
const NI_Q8: [usize; 5] = [2, 3, 4, 6, 8];
const NI_Q8P: [usize; 9] = [1, 2, 3, 4, 6, 8, 12, 16, 24];

/// Dynamic shared memory of the Q8 kernels for activation vectors of `k` elements: fp32 copy + int8 copy + scales + reduction scratch.
fn q8_smem(k: usize) -> u32 {
    ((4 * k + k + k / 8 + 128 + 15) / 16 * 16) as u32
}

impl GpuModel {
    /// The MTP head is loaded and every matrix is quantized, so speculative decoding (draft step + two-token verification) is available.
    /// The positions the model was trained for: the longest context a session should be opened with.
    pub fn max_positions(&self) -> usize {
        self.cfg.max_ctx
    }

    /// Bytes of device memory a session of `ctx` positions takes next to the weights (the KV cache or its compressed store and the fp16 scratch, the prefill buffers, the recurrent states, the CUDA graphs), as the
    /// planner of `GpuModel::upload` counts them (FORGE_KV, FORGE_PREFILL_CHUNK and FORGE_SPEC_DEPTH change it).
    pub fn session_bytes(&self, ctx: usize) -> usize {
        session_reserve(&self.cfg, ctx, self.mtp.is_some())
    }

    /// Free device memory now.
    pub fn free_device_bytes(&self) -> Result<usize> {
        Ok(self.dev.mem_info()?.0)
    }

    pub fn has_mtp(&self) -> bool {
        self.mtp.is_some() && self.quantized
    }

    /// Copy a CPU-loaded model to the device (matrices as `fmt`, norm weights as fp32).
    pub fn upload(model: &Model, fmt: Fmt, device_ordinal: usize) -> Result<GpuModel> {
        // FORGE_HOST_FFN / FORGE_HOST_EMBED decide by hand; without them the weights go to the device when they fit next to the working memory of a session, else the FFN of as many layers as needed
        // (and the embedding table) go to host memory
        let manual = std::env::var("FORGE_HOST_FFN").is_ok() || std::env::var("FORGE_HOST_EMBED").is_ok();
        let opts = if manual || fmt == Fmt::F16 {
            UploadOpts::from_env(model.cfg.n_layer)
        } else {
            plan_offload(model, fmt, device_ordinal)?
        };
        Self::upload_inner(model, fmt, device_ordinal, false, opts)
    }

    /// As `upload`, with an explicit placement of the weights (see `UploadOpts`).
    pub fn upload_with(
        model: &Model,
        fmt: Fmt,
        device_ordinal: usize,
        opts: UploadOpts,
    ) -> Result<GpuModel> {
        Self::upload_inner(model, fmt, device_ordinal, false, opts)
    }

    /// fp16 weights of any model, for evaluation (`prefill_logits`) only: a hybrid model gets no decode graphs. The unquantized reference the quantizers are measured against.
    pub fn upload_eval(model: &Model, device_ordinal: usize) -> Result<GpuModel> {
        Self::upload_inner(model, Fmt::F16, device_ordinal, true, UploadOpts::default())
    }

    fn upload_inner(
        model: &Model,
        fmt: Fmt,
        device_ordinal: usize,
        eval_only: bool,
        opts: UploadOpts,
    ) -> Result<GpuModel> {
        let dev = Device::open(device_ordinal)?;
        let cfg = model.cfg.clone();
        if cfg.delta.is_some() && fmt == Fmt::F16 && !eval_only {
            return Err(Error::unsupported(
                "hybrid (Gated DeltaNet) models run with quantized weights: use --q8 or --native"
                    .to_string(),
            ));
        }
        if cfg.qkv_bias && fmt != Fmt::Native && !eval_only {
            return Err(Error::unsupported(
                "models with biases on the q, k and v projections (Qwen2) run with the quantized weights of a GGUF file as they are (--native)".to_string(),
            ));
        }
        if let Some(d) = &cfg.delta {
            if d.head_k != 128
                || d.head_v != 128
                || d.conv < 2
                || d.conv > 4
                || d.n_v % d.n_k != 0
                || cfg.head_dim % 32 != 0
            {
                return Err(Error::unsupported(format!("Gated DeltaNet layout {d:?} (the kernels take head widths of 128 and a convolution of 2 to 4 columns)")));
            }
        }
        for (what, v) in [
            ("n_embd", cfg.n_embd),
            ("q_dim", cfg.q_dim()),
            ("kv_dim", cfg.kv_dim()),
            ("n_ff", cfg.n_ff),
        ] {
            if v % 32 != 0 {
                return Err(Error::unsupported(format!(
                    "{what} = {v} is not a multiple of 32"
                )));
            }
        }
        if cfg.head_dim > 256
            || cfg.head_dim % 8 != 0
            || (cfg.head_dim > 128 && cfg.delta.is_none())
        {
            return Err(Error::unsupported(format!("head_dim {}", cfg.head_dim)));
        }
        let (image, is_ptx) = forge_kernels::image_for(dev.info.cc_major, dev.info.cc_minor)
            .ok_or_else(|| {
                Error::msg(format!(
                    "no GPU kernels for compute capability {}.{}",
                    dev.info.cc_major, dev.info.cc_minor
                ))
            })?;
        let module = Module::load(image)?;
        eprintln!(
            "[gpu] {} (sm_{}{}, {} SMs, {:.1} GiB): kernels from {}, weights {:?}",
            dev.info.name,
            dev.info.cc_major,
            dev.info.cc_minor,
            dev.info.sm_count,
            dev.info.total_mem as f64 / (1u64 << 30) as f64,
            if is_ptx { "PTX (JIT)" } else { "cubin" },
            fmt
        );
        let mm_fns = |kind: &str| -> Result<Vec<Vec<Function>>> {
            mrow::MM_T
                .iter()
                .map(|t| {
                    ["q8", "q40", "q50", "q4k", "q5k", "q6k"]
                        .iter()
                        .map(|f| module.function(&format!("mm_{kind}_{f}_t{t}_kernel")))
                        .collect::<Result<Vec<_>>>()
                })
                .collect()
        };
        // the integer prompt GEMMs (gemm_mmq.cu) need sm_80 and 98,304 bytes of dynamic shared memory (the opt-in limit of a block is 99 KB on Ampere GeForce and Ada)
        let mmq = ["q4k", "q5k", "q6k"]
            .iter()
            .flat_map(|f| ["f16", "f32", "add"].iter().map(move |e| format!("mmq_{f}_{e}")))
            .map(|n| module.function(&n))
            .collect::<Result<Vec<_>>>()?;
        let mmq_ok = dev.info.cc_major >= 8
            && mmq
                .iter()
                .enumerate()
                .all(|(i, f)| f.set_max_dynamic_smem(if i / 3 == 2 { MMQ6_SMEM } else { MMQ_SMEM } as i32).is_ok());
        let k = Kernels {
            embed: module.function("embed_kernel")?,
            embed_q8: module.function("embed_q8_kernel")?,
            qk_norm_rope: module.function("qk_norm_rope_kernel")?,
            attention: module.function("attention_kernel")?,
            final_norm: module.function("final_norm_kernel")?,
            head: module.function("head_kernel")?,
            argmax_partial: module.function("argmax_partial_kernel")?,
            argmax_final: module.function("argmax_final_kernel")?,
            norm_qkv_rope: NiKernels::load(&module, "norm_qkv_rope_kernel", &NI_F16, 8)?,
            matvec_residual: NiKernels::load(&module, "matvec_residual_kernel", &NI_F16, 8)?,
            norm_gateup: NiKernels::load(&module, "norm_gateup_kernel", &NI_F16, 8)?,
            norm_qkv_rope_q8: NiKernels::load(&module, "norm_qkv_rope_q8_kernel", &NI_Q8, 16)?,
            matvec_residual_q8: NiKernels::load(&module, "matvec_residual_q8_kernel", &NI_Q8, 16)?,
            norm_gateup_q8: NiKernels::load(&module, "norm_gateup_q8_kernel", &NI_Q8, 16)?,
            head_q8: NiKernels::load(&module, "head_q8_kernel", &NI_Q8, 16)?,
            norm_quant: NiKernels::load(&module, "norm_quant_kernel", &[1, 2, 4, 8, 16, 32], 512)?,
            attention_q: module.function("attention_q_kernel")?,
            quant_vec: NiKernels::load(&module, "quant_kernel", &[1, 2, 4, 8, 16, 32], 512)?,
            qkvq: FmtKernels::load(&module, "qkvq", true)?,
            mvq_res: FmtKernels::load(&module, "mvq_res", false)?,
            gateupq: FmtKernels::load(&module, "gateupq", false)?,
            headq: FmtKernels::load(&module, "headq", false)?,
            embed_q: module.function("embed_q_kernel")?,
            dequant_test: module.function("dequant_test_kernel")?,
            dequant_f16: module.function("dequant_f16_kernel")?,
            dequant_rows_f16: module.function("dequant_rows_f16_kernel")?,
            embed_q_batch: module.function("embed_q_batch_kernel")?,
            gdn_step: module.function("gdn_step_kernel")?,
            qkv35_prep: module.function("qkv35_prep_kernel")?,
            attention_g: module.function("attention_g_kernel")?,
            attn_split_g: module.function("attn_split_g_kernel")?,
            attn_split: module.function("attn_split_kernel")?,
            attn_split_q: module.function("attn_split_q_kernel")?,
            attn_split_g_q: module.function("attn_split_g_q_kernel")?,
            attn_split_rows: module.function("attn_split_rows4_kernel")?,
            gqa: {
                let mut v = Vec::new();
                for hd in [64usize, 128, 256] {
                    for gp in [2usize, 4, 8] {
                        for (quant, tag) in [(false, "f16"), (true, "q")] {
                            if let Ok(f) = module.function(&format!("attn_gqa_{tag}_{hd}_{gp}_kernel")) {
                                v.push((hd, gp, quant, f));
                            }
                        }
                    }
                }
                v
            },
            attn_merge: module.function("attn_merge_kernel")?,
            attn_merge_q: module.function("attn_merge_q_kernel")?,
            attn_merge_g: module.function("attn_merge_g_kernel")?,
            gdn_prep: module.function("gdn_prep_kernel")?,
            gdn_prep_tile: module.function("gdn_prep_tile_kernel")?,
            gdn_scan: [
                module.function("gdn_scan_kernel_32")?,
                module.function("gdn_scan_kernel_64")?,
                module.function("gdn_scan_kernel_128")?,
            ],
            gdn_scan_nc: [
                module.function("gdn_scan_nc_16")?,
                module.function("gdn_scan_nc_32")?,
                module.function("gdn_scan_nc_64")?,
            ],
            gdn_post: module.function("gdn_post_kernel")?,
            qkv35_rope_batch: module.function("qkv35_rope_batch_kernel")?,
            attn_prefill_256g: module.function("attn_prefill_256g")?,
            flash_256g: module.function("flash_fwd_256g")?,
            flash_256g_a80: module.function("flash_fwd_256g_a80")?,
            flash_a80: [module.function("flash_fwd_64_a80")?, module.function("flash_fwd_128_a80")?],
            gemm_f32_out: module.function("gemm_nt_f32")?,
            mtp_prep: module.function("mtp_prep_kernel")?,
            mtp_prep_batch: module.function("mtp_prep_batch_kernel")?,
            embed_q_multi: module.function("embed_q_multi_kernel")?,
            gdn_step2: module.function("gdn_step2_kernel")?,
            swiglu_quant_t: module.function("swiglu_quant_t_kernel")?,
            argmax_rows: module.function("argmax_rows_kernel")?,
            argmax_rows_p: module.function("argmax_rows_partial_kernel")?,
            argmax_rows_f: module.function("argmax_rows_final_kernel")?,
            mvt2_out: ["q8", "q40", "q50", "q4k", "q5k", "q6k"]
                .iter()
                .map(|f| module.function(&format!("mvt2_out_{f}_kernel")))
                .collect::<Result<Vec<_>>>()?,
            mvt2_res: ["q8", "q40", "q50", "q4k", "q5k", "q6k"]
                .iter()
                .map(|f| module.function(&format!("mvt2_res_{f}_kernel")))
                .collect::<Result<Vec<_>>>()?,
            mm_out: mm_fns("out")?,
            mm_res: mm_fns("res")?,
            mm_reduce_out: module.function("mm_reduce_out_kernel")?,
            mm_reduce_res: module.function("mm_reduce_res_kernel")?,
            qkv_prep_multi: module.function("qkv_prep_multi_kernel")?,
            gdn_stepn: module.function("gdn_stepn_kernel")?,
            gdn_stepn_side: module.function("gdn_stepn_side_kernel")?,
            penalty: module.function("penalty_kernel")?,
            logit_max: module.function("logit_max_kernel")?,
            collect: module.function("collect_kernel")?,
            gemm_f16_out: module.function("gemm_nt_f16")?,
            gemm_f32_add: module.function("gemm_nt_add")?,
            gemm_s80: ["f16", "f32", "add"]
                .iter()
                .map(|e| module.function(&format!("gemm_nt_{e}_s80")))
                .collect::<Result<Vec<_>>>()?,
            gemm_s80h: ["f16", "f32", "add"]
                .iter()
                .map(|e| module.function(&format!("gemm_nt_{e}_s80h")))
                .collect::<Result<Vec<_>>>()?,
            dequantk: ["q4k", "q5k", "q6k"]
                .iter()
                .map(|f| module.function(&format!("dequantk_{f}_kernel")))
                .collect::<Result<Vec<_>>>()?,
            gemmq: ["q4k", "q5k", "q6k"]
                .iter()
                .flat_map(|f| {
                    ["f16", "f32", "add"]
                        .iter()
                        .map(move |e| format!("gemmq_{f}_{e}"))
                })
                .map(|n| module.function(&n))
                .collect::<Result<Vec<_>>>()?,
            gemmq_h: ["q4k", "q5k", "q6k"]
                .iter()
                .flat_map(|f| {
                    ["f16", "f32", "add"]
                        .iter()
                        .map(move |e| format!("gemmq_{f}_{e}_h"))
                })
                .map(|n| module.function(&n))
                .collect::<Result<Vec<_>>>()?,
            mmq,
            mmq_ok,
            quant_rows_h: module.function("quant_rows_h_kernel")?,
            swiglu_quant_rows: module.function("swiglu_quant_rows_kernel")?,
            fakeq_rows: module.function("fakeq_rows_kernel")?,
            embed_batch: module.function("embed_batch_kernel")?,
            rmsnorm_rows: module.function("rmsnorm_rows_kernel")?,
            colsq: module.function("colsq_accum_kernel")?,
            gemm_tn_add: module.function("gemm_tn_add")?,
            qkv_rope: module.function("qkv_rope_kernel")?,
            attn_prefill_64: module.function("attn_prefill_64")?,
            attn_prefill_128: module.function("attn_prefill_128")?,
            flash_64: module.function("flash_fwd_64")?,
            flash_128: module.function("flash_fwd_128")?,
            swiglu: module.function("swiglu_kernel")?,
            copy_row: module.function("copy_row_kernel")?,
            set_pos: module.function("set_pos_kernel")?,
            add_f32: module.function("resid_add_f32_kernel")?,
            kv_dequant: module.function("kv_dequant_kernel")?,
            kv_quant: module.function("kv_quant_kernel")?,
        };
        // the multi-row kernels of blocks of 12 warps use up to 62 KB of shared memory: allow more than the 48 KB of the default launch limit
        for f in k.gemm_s80.iter().chain(k.gemm_s80h.iter()) {
            f.set_max_dynamic_smem(S80_SMEM as i32)?;
        }
        if dev.info.cc_major >= 8 {
            k.flash_256g_a80.set_max_dynamic_smem(FLASH_PAIR_SMEM as i32)?;
        }
        for f in k.mm_out.iter().chain(k.mm_res.iter()).flatten() {
            f.set_max_dynamic_smem(64000)?;
        }
        let mut bytes = 0usize;
        let mut host_bytes = 0usize;
        let mut host_ffn_bytes = 0usize;
        // the embedding table may live in host memory only when the LM head is a separate matrix (a tied head reads the whole table every token); when page-locked memory cannot be had, it stays on the device
        let want_host_embed = opts.host_embed && model.lm_head.is_some();
        let tok_emb = match upload_mat(&model.tok_emb, fmt, want_host_embed) {
            Ok(m) => m,
            Err(e) if want_host_embed => {
                eprintln!("[gpu] the embedding table could not go to page-locked host memory ({e}); it stays on the device");
                upload_mat(&model.tok_emb, fmt, false)?
            }
            Err(e) => return Err(e),
        };
        bytes += tok_emb.bytes();
        host_bytes += tok_emb.split_bytes().1;
        let lm_head = match &model.lm_head {
            Some(h) => {
                let b = upload_mat(h, fmt, false)?;
                bytes += b.bytes();
                Some(b)
            }
            None => None,
        };
        let out_norm = f32_buffer(&model.out_norm)?;
        let mut layers = Vec::with_capacity(model.layers.len());
        for (li, l) in model.layers.iter().enumerate() {
            let host = opts.host_ffn_of(li) || host_only(&l.w_gate) || host_only(&l.w_up) || host_only(&l.w_down);
            let mut g = if let Some(dl) = &l.delta {
                let gd = GpuDelta {
                    w_qkv: upload_mat(&dl.w_qkv, fmt, false)?,
                    w_z: upload_mat(&dl.w_z, fmt, false)?,
                    w_a: upload_mat(&dl.w_a, fmt, false)?,
                    w_b: upload_mat(&dl.w_b, fmt, false)?,
                    conv: f32_buffer(&dl.conv)?,
                    a_neg: f32_buffer(&dl.a_neg)?,
                    dt_bias: f32_buffer(&dl.dt_bias)?,
                    norm: f32_buffer(&dl.norm)?,
                    w_out: upload_mat(&dl.w_out, fmt, false)?,
                };
                bytes += gd.w_qkv.bytes()
                    + gd.w_z.bytes()
                    + gd.w_a.bytes()
                    + gd.w_b.bytes()
                    + gd.w_out.bytes();
                GpuLayer {
                    attn_norm: f32_buffer(&l.attn_norm)?,
                    wq: GpuMat::dummy()?,
                    wk: GpuMat::dummy()?,
                    wv: GpuMat::dummy()?,
                    wo: GpuMat::dummy()?,
                    q_norm: None,
                    k_norm: None,
                    bq: None,
                    bk: None,
                    bv: None,
                    ffn_norm: f32_buffer(&l.ffn_norm)?,
                    w_gate: upload_mat(&l.w_gate, fmt, host)?,
                    w_up: upload_mat(&l.w_up, fmt, host)?,
                    w_down: upload_mat(&l.w_down, fmt, host)?,
                    delta: Some(gd),
                    cpu: None,
                }
            } else {
                GpuLayer {
                    attn_norm: f32_buffer(&l.attn_norm)?,
                    wq: upload_mat(&l.wq, fmt, false)?,
                    wk: upload_mat(&l.wk, fmt, false)?,
                    wv: upload_mat(&l.wv, fmt, false)?,
                    wo: upload_mat(&l.wo, fmt, false)?,
                    q_norm: l.q_norm.as_ref().map(|v| f32_buffer(v)).transpose()?,
                    k_norm: l.k_norm.as_ref().map(|v| f32_buffer(v)).transpose()?,
                    bq: l.bq.as_ref().map(|v| f32_buffer(v)).transpose()?,
                    bk: l.bk.as_ref().map(|v| f32_buffer(v)).transpose()?,
                    bv: l.bv.as_ref().map(|v| f32_buffer(v)).transpose()?,
                    ffn_norm: f32_buffer(&l.ffn_norm)?,
                    w_gate: upload_mat(&l.w_gate, fmt, host)?,
                    w_up: upload_mat(&l.w_up, fmt, host)?,
                    w_down: upload_mat(&l.w_down, fmt, host)?,
                    delta: None,
                    cpu: None,
                }
            };
            if l.delta.is_none() {
                bytes += g.wq.bytes() + g.wk.bytes() + g.wv.bytes() + g.wo.bytes();
            }
            if host && std::env::var("FORGE_CPU_FFN").map(|v| v != "0").unwrap_or(true) {
                g.cpu = Self::cpu_ffn_of(&g, &cfg);
            }
            bytes += g.w_gate.bytes() + g.w_up.bytes() + g.w_down.bytes();
            let ffn_host = g.w_gate.split_bytes().1 + g.w_up.split_bytes().1 + g.w_down.split_bytes().1;
            host_bytes += ffn_host;
            host_ffn_bytes += ffn_host;
            layers.push(g);
        }
        let mtp = match &model.mtp {
            Some(w) => {
                let l = &w.layer;
                let layer = GpuLayer {
                    attn_norm: f32_buffer(&l.attn_norm)?,
                    wq: upload_mat(&l.wq, fmt, false)?,
                    wk: upload_mat(&l.wk, fmt, false)?,
                    wv: upload_mat(&l.wv, fmt, false)?,
                    wo: upload_mat(&l.wo, fmt, false)?,
                    q_norm: l.q_norm.as_ref().map(|v| f32_buffer(v)).transpose()?,
                    k_norm: l.k_norm.as_ref().map(|v| f32_buffer(v)).transpose()?,
                    bq: l.bq.as_ref().map(|v| f32_buffer(v)).transpose()?,
                    bk: l.bk.as_ref().map(|v| f32_buffer(v)).transpose()?,
                    bv: l.bv.as_ref().map(|v| f32_buffer(v)).transpose()?,
                    ffn_norm: f32_buffer(&l.ffn_norm)?,
                    w_gate: upload_mat(&l.w_gate, fmt, false)?,
                    w_up: upload_mat(&l.w_up, fmt, false)?,
                    w_down: upload_mat(&l.w_down, fmt, false)?,
                    delta: None,
                    cpu: None,
                };
                let fc = upload_mat(&w.fc, fmt, false)?;
                bytes += fc.bytes()
                    + layer.wq.bytes()
                    + layer.wk.bytes()
                    + layer.wv.bytes()
                    + layer.wo.bytes()
                    + layer.w_gate.bytes()
                    + layer.w_up.bytes()
                    + layer.w_down.bytes();
                Some(GpuMtp {
                    fc,
                    norm_e: f32_buffer(&w.norm_e)?,
                    norm_h: f32_buffer(&w.norm_h)?,
                    out_norm: f32_buffer(&w.out_norm)?,
                    layer,
                })
            }
            None => None,
        };
        if std::env::var_os("FORGE_LOAD_TRACE").is_some() {
            eprintln!(
                "[load] repacking {:.2} s, copying to the device {:.2} s",
                REPACK_NS.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1e9,
                COPY_NS.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1e9
            );
        }
        if host_bytes > 0 {
            let n_ffn = opts.host_ffn.iter().filter(|b| **b).count();
            let embed_on_host = tok_emb.is_host();
            let what = match (host_ffn_bytes > 0, embed_on_host) {
                (true, true) => format!("{n_ffn} layers' FFN and the embedding table"),
                (true, false) => format!("{n_ffn} layers' FFN"),
                (false, _) => "the embedding table".to_string(),
            };
            eprintln!(
                "[gpu] weights: {:.2} GiB on the device, {:.2} GiB in page-locked host memory ({what})",
                (bytes - host_bytes) as f64 / (1u64 << 30) as f64,
                host_bytes as f64 / (1u64 << 30) as f64,
            );
        }
        Ok(GpuModel {
            dev,
            cfg,
            fmt,
            quantized: fmt != Fmt::F16,
            _module: module,
            k,
            tok_emb,
            layers,
            out_norm,
            lm_head,
            mtp,
            weight_bytes: bytes,
            host_weight_bytes: host_bytes,
            host_ffn_bytes,
            eval_only,
        })
    }
}

/// FORGE_KPROF=1: the per-kernel table of the launches so far (GPU time per kernel; launches inside captured graphs are not in it). None when the profiler is off.
pub fn kernel_profile_report() -> Option<String> {
    if std::env::var("FORGE_KPROF").is_ok() {
        Some(forge_cuda::kprof_report())
    } else {
        None
    }
}

/// The CPU kernels' view of a host-resident matrix (None: not in host memory, or no kernel for its format / row length).
fn host_mat_of(m: &GpuMat, rows: usize, cols: usize) -> Option<cpuk::HostMat> {
    let qf = m.qf?;
    if !m.is_host() || !cpuk::HostMat::usable(qf, cols) {
        return None;
    }
    let mut p = [std::ptr::null::<u8>(); 4];
    p[0] = m.w.host_bytes().as_ptr();
    let mut i = 1;
    if let Some(sb) = &m.s {
        p[1] = sb.host_bytes().as_ptr();
        i = 2;
    }
    for b in &m.x {
        p[i] = b.host_bytes().as_ptr();
        i += 1;
    }
    Some(cpuk::HostMat { fmt: qf, rows, cols, p })
}

impl GpuModel {
    fn cpu_ffn_of(g: &GpuLayer, c: &Config) -> Option<cpuk::CpuFfn> {
        Some(cpuk::CpuFfn {
            gate: host_mat_of(&g.w_gate, c.n_ff, c.n_embd)?,
            up: host_mat_of(&g.w_up, c.n_ff, c.n_embd)?,
            down: host_mat_of(&g.w_down, c.n_embd, c.n_ff)?,
            e: c.n_embd,
            f: c.n_ff,
            s256: cpuk::s256_default(),
        })
    }
}

/// What a host callback of the decode graph needs to run one layer's FFN on the CPU: the matrices, the pool, the pinned input / output buffers.
struct CpuCtx {
    ffn: cpuk::CpuFfn,
    cols: usize,
    pool: Arc<cpuk::Pool>,
    in_q: *const i8,
    in_d: *const f32,
    in_s: *const i32,
    out: *mut f32,
    scratch: Arc<Mutex<cpuk::FfnScratch>>,
    /// after this layer's FFN the pool's workers pull the first rows of the next host layer's weights into the cache while the GPU does its part
    post: cpuk::PostJob,
}

unsafe impl Send for CpuCtx {}
unsafe impl Sync for CpuCtx {}

impl CpuCtx {
    fn run(&self) {
        let e = self.ffn.e;
        let acts: Vec<cpuk::Act> = (0..self.cols)
            .map(|c| unsafe {
                cpuk::Act {
                    xq: std::slice::from_raw_parts(self.in_q.add(c * e), e),
                    xd: std::slice::from_raw_parts(self.in_d.add(c * e / 32), e / 32),
                    xs: std::slice::from_raw_parts(self.in_s.add(c * e / 32), e / 32),
                }
            })
            .collect();
        let out = unsafe { std::slice::from_raw_parts_mut(self.out, self.cols * e) };
        let mut sc = self.scratch.lock().unwrap();
        let t0 = if cpu_prof() { Some(std::time::Instant::now()) } else { None };
        self.ffn.run_post(&self.pool, &acts, &mut sc, out, Some(&self.post));
        if let Some(t0) = t0 {
            // FORGE_CPU_PROF=1: wall time of the FFN jobs (every 256 calls: mean per call and the weight bandwidth)
            static NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            static B: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            use std::sync::atomic::Ordering::Relaxed;
            let bytes: usize = [&self.ffn.gate, &self.ffn.up, &self.ffn.down]
                .iter()
                .map(|m| m.rows * m.cols / m.fmt.block().0 * m.fmt.block().1)
                .sum();
            NS.fetch_add(t0.elapsed().as_nanos() as u64, Relaxed);
            B.fetch_add(bytes as u64, Relaxed);
            let n = N.fetch_add(1, Relaxed) + 1;
            if n % 256 == 0 {
                let (ns, b) = (NS.swap(0, Relaxed), B.swap(0, Relaxed));
                eprintln!("[cpu] {:.2} ms per FFN call, {:.1} GB/s of weights ({} columns)", ns as f64 / 256.0 / 1e6, b as f64 / ns as f64, self.cols);
            }
        }
    }
}

fn cpu_prof() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("FORGE_CPU_PROF").map(|v| v == "1").unwrap_or(false))
}

unsafe extern "C" fn cpu_ffn_callback(p: *mut c_void) {
    let ctx = &*(p as *const CpuCtx);
    ctx.run();
}

/// The CPU side of a session: pinned copies of the FFN input (the quantized, normed residual stream) and output, one context per host-resident layer.
struct CpuState {
    hin_q: PinnedBuffer,
    hin_d: PinnedBuffer,
    hin_s: PinnedBuffer,
    hout: PinnedBuffer,
    d_out: DeviceBuffer,
    /// per layer, for 1..=4 columns (index cols - 1)
    ctx: Vec<Option<Vec<Box<CpuCtx>>>>,
}

impl CpuState {
    fn new(m: &GpuModel) -> Result<Option<CpuState>> {
        if !m.layers.iter().any(|l| l.cpu.is_some()) {
            return Ok(None);
        }
        let e = m.cfg.n_embd;
        let threads = std::env::var("FORGE_CPU_THREADS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            // one core stays free for the rest of the machine (the thread that launches the graph, the driver, Docker): on the 8-core i7-9700K 7 threads with dynamic chunks decode a 27B model 9 percent
            // faster than 6 (and 8 gain nothing more); with fixed slices per thread 8 were far slower than 6, one preempted thread held every call back
            .unwrap_or_else(|| std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).saturating_sub(1).clamp(1, 7))
            .max(1);
        let pool = Arc::new(cpuk::Pool::new(threads));
        eprintln!("[cpu] FFN layers on the CPU: {threads} threads, dynamic chunks {}", if std::env::var("FORGE_CPU_DYN").map(|v| v == "0").unwrap_or(false) { "off" } else { "on" });
        let hin_q = PinnedBuffer::alloc(4 * e)?;
        let hin_d = PinnedBuffer::alloc(4 * e / 32 * 4)?;
        let hin_s = PinnedBuffer::alloc(4 * e / 32 * 4)?;
        let hout = PinnedBuffer::alloc(4 * e * 4)?;
        let scratch = Arc::new(Mutex::new(cpuk::FfnScratch::new(m.cfg.n_ff, 4)));
        let (pq, pd, ps, po) = (
            hin_q.as_ptr() as *const i8,
            hin_d.as_ptr() as *const f32,
            hin_s.as_ptr() as *const i32,
            hout.as_ptr() as *mut f32,
        );
        let mut ctx: Vec<Option<Vec<Box<CpuCtx>>>> = m
            .layers
            .iter()
            .map(|l| {
                l.cpu.as_ref().map(|f| {
                    (1..=4)
                        .map(|cols| {
                            Box::new(CpuCtx {
                                ffn: cpuk::CpuFfn {
                                    gate: f.gate,
                                    up: f.up,
                                    down: f.down,
                                    e: f.e,
                                    f: f.f,
                                    s256: f.s256,
                                },
                                cols,
                                pool: pool.clone(),
                                in_q: pq,
                                in_d: pd,
                                in_s: ps,
                                out: po,
                                scratch: scratch.clone(),
                                post: cpuk::PostJob::none(),
                            })
                        })
                        .collect()
                })
            })
            .collect();
        // every host layer's FFN prefetches the first rows of the next host layer (the last one: of the first, for the next step)
        let host: Vec<usize> = (0..ctx.len()).filter(|&i| ctx[i].is_some()).collect();
        if host.len() > 1 {
            for (k, &li) in host.iter().enumerate() {
                let nj = host[(k + 1) % host.len()];
                for c in 0..4 {
                    let next: *const cpuk::CpuFfn = &ctx[nj].as_ref().unwrap()[c].ffn;
                    ctx[li].as_mut().unwrap()[c].post = cpuk::PostJob::prefetch_of(next);
                }
            }
        }
        eprintln!(
            "[gpu] {} layers' FFN run on the CPU ({} threads) during decode",
            m.layers.iter().filter(|l| l.cpu.is_some()).count(),
            threads
        );
        Ok(Some(CpuState {
            hin_q,
            hin_d,
            hin_s,
            hout,
            d_out: DeviceBuffer::alloc(4 * e * 4)?,
            ctx,
        }))
    }
}

const ARGMAX_BLOCKS: u32 = 128;
const SAMPLE_BLOCKS: u32 = 128;
/// positions per split of the split-KV attention kernels (decode.cu / decode_h.cu: AS_CH)
const AS_CH: usize = 1024;
/// the split kernels are used from a session context of one split on: below it the one-block kernel runs the same single block per head (and the merge launch is saved), above it a block per split of the
/// context reads keys and values at several times the speed (see `GpuSession::attn_split`)
const ATTN_SPLIT_FROM: usize = AS_CH;
/// the longest context the one-block-per-head decode attention of a dense model can take: scores[max_ctx] + q[hd] + part[1024] + red[32] floats in the 48 KB of dynamic shared memory a launch gets without an opt-in
/// (11,168 positions for a head width of 64, 11,104 for 128)
const DENSE_ONE_BLOCK_MAX: usize = 11_000;
/// the same limit of the hybrid kernels (256 threads, head width 256: scores[max_ctx] + q[256] + part[2048] + red[32] floats)
const HYBRID_ONE_BLOCK_MAX: usize = 9_900;
/// capacity of the candidate list the device builds for the sampler
const CAND_CAP: usize = 2048;

/// Which logits to look at: those of the last single-token step, or one of the two rows of the last two-token step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LogitsSrc {
    Main,
    Verify(usize),
    /// row i of the last `verify_rows`
    Rows(usize),
}

/// The tokens whose logit is within a cutoff of the best one, extracted on the device.
pub struct Candidates {
    pub max: f32,
    /// (logit, token), in no particular order
    pub items: Vec<(f32, u32)>,
    /// more candidates than fit: the list is incomplete and the caller must use the full logits
    pub overflow: bool,
}
const RING: usize = 4096;

/// One generation: KV cache, scratch buffers, the step state and the captured graphs.
/// Buffers of the two-token (verification) step: everything of a decode step, twice, plus the state after the first token for a rollback.
struct VerifyBufs {
    tokens: DeviceBuffer, // u32 [2]
    x: DeviceBuffer,      // fp32 [2][E]: the residual streams (before the final norm)
    xq_e: DeviceBuffer,
    xd_e: DeviceBuffer,
    xs_e: DeviceBuffer,
    xq_a: DeviceBuffer,
    xd_a: DeviceBuffer,
    xs_a: DeviceBuffer,
    xq_f: DeviceBuffer,
    xd_f: DeviceBuffer,
    xs_f: DeviceBuffer,
    mixed: DeviceBuffer,
    z: DeviceBuffer,
    ab: DeviceBuffer,
    qg: DeviceBuffer,
    k_raw: DeviceBuffer,
    v_raw: DeviceBuffer,
    q: DeviceBuffer,
    g: DeviceBuffer,
    u: DeviceBuffer,
    logits: DeviceBuffer, // fp32 [2][V]
    argmax: DeviceBuffer, // u32 [2]
    as_m: DeviceBuffer,
    as_l: DeviceBuffer,
    as_o: DeviceBuffer,
    s_side: DeviceBuffer,
    conv_side: DeviceBuffer,
    graph: Option<GraphExec>,
}

/// The recurrent state of a hybrid model at one position (the Gated DeltaNet states and convolution windows): enough, together with the attention layers' KV cache entries below that
/// position (which stay valid while the cached tokens do), to continue from there. Tens of MB, copied device to device.
pub struct StateSnapshot {
    pos: usize,
    s: SnapBuf,
    conv: SnapBuf,
    /// the residual stream of the last processed position (the MTP head's input for the pending position)
    hidden: SnapBuf,
    mtp_filled: usize,
}

impl StateSnapshot {
    /// The position the snapshot was taken at.
    pub fn position(&self) -> usize {
        self.pos
    }

    /// Memory the snapshot occupies (device or page-locked host memory).
    pub fn bytes(&self) -> usize {
        self.s.len() + self.conv.len() + self.hidden.len()
    }
}

/// The KV entries of the first `n` positions of a conversation, in page-locked host memory (`GpuSession::export_kv`).
pub struct KvBlob {
    n: usize,
    /// per region (the order of `GpuSession::kv_regions`): byte offset in `buf` of its first position, and the bytes per position
    layout: Vec<(usize, usize)>,
    buf: PinnedBuffer,
}

impl KvBlob {
    /// The positions the blob holds.
    pub fn positions(&self) -> usize {
        self.n
    }

    /// Bytes of the blob's entries.
    pub fn bytes(&self) -> usize {
        self.layout.iter().map(|(_, row)| row * self.n).sum()
    }

    /// Host memory the blob occupies (the buffer may be bigger than the entries when it was reused).
    pub fn capacity(&self) -> usize {
        self.buf.len()
    }

    /// The page-locked buffer, to export another conversation into (`GpuSession::export_kv_with`).
    pub fn into_buffer(self) -> PinnedBuffer {
        self.buf
    }
}

/// One buffer of a snapshot: on the device, or in page-locked host memory when weights live there too (a snapshot of the 27B's Gated DeltaNet states is 144 MiB: the 8 GB card cannot spare three of them).
enum SnapBuf {
    Dev(DeviceBuffer),
    Host(PinnedBuffer),
}

impl SnapBuf {
    fn alloc(bytes: usize, host: bool) -> Result<SnapBuf> {
        Ok(if host {
            SnapBuf::Host(PinnedBuffer::alloc(bytes)?)
        } else {
            SnapBuf::Dev(DeviceBuffer::alloc(bytes)?)
        })
    }

    fn len(&self) -> usize {
        match self {
            SnapBuf::Dev(b) => b.len(),
            SnapBuf::Host(p) => p.as_slice().len(),
        }
    }

    /// copy `src` (a device buffer) into the snapshot
    fn save(&self, st: &Stream, src: &DeviceBuffer) -> Result<()> {
        match self {
            SnapBuf::Dev(b) => st.copy_d2d(b.ptr(), src.ptr(), src.len()),
            SnapBuf::Host(p) => st.download_async(p, src, 0, src.len()),
        }
    }

    /// copy the snapshot back into `dst` (a device buffer)
    fn load(&self, st: &Stream, dst: &DeviceBuffer) -> Result<()> {
        match self {
            SnapBuf::Dev(b) => st.copy_d2d(dst.ptr(), b.ptr(), dst.len()),
            SnapBuf::Host(p) => st.upload_async(dst, 0, p, dst.len()),
        }
    }
}

/// A compressed KV cache (FORGE_KV): the bits of the K and V values (8 or 4) and the values per fp16 scale.
#[derive(Clone, Copy, Debug)]
struct KvCfg {
    bits_k: u32,
    bits_v: u32,
    group: usize,
}

impl KvCfg {
    /// FORGE_KV = int8 | int4 | k8v8 | k8v4 | k4v8 | k4v4 (unset, fp16, off: the plain fp16 cache); FORGE_KV_GROUP = 8 | 16 | 32 | 64 | 128 | 256 (default 32; at least 8 values per lane's share of a row, see kvq.cu). Quantized models (the hooks sit in the K-quant decode, prefill and multi-row paths).
    fn from_env(c: &Config) -> Option<KvCfg> {
        let v = std::env::var("FORGE_KV").ok()?;
        let (bits_k, bits_v) = match v.trim().to_ascii_lowercase().as_str() {
            "" | "fp16" | "off" | "none" | "0" => return None,
            "int8" | "k8v8" | "8" => (8, 8),
            "int4" | "k4v4" | "4" => (4, 4),
            "k8v4" => (8, 4),
            "k4v8" => (4, 8),
            other => {
                eprintln!("[forge] FORGE_KV={other}: not understood (int8, int4, k8v4, k4v8); the cache stays fp16");
                return None;
            }
        };
        if ![64, 128, 256].contains(&c.head_dim) {
            eprintln!("[forge] FORGE_KV: the compressed cache needs a head size of 64, 128 or 256; the cache stays fp16");
            return None;
        }
        let group = std::env::var("FORGE_KV_GROUP")
            .ok()
            .and_then(|g| g.parse::<usize>().ok())
            .filter(|g| g.is_power_of_two() && *g >= 8)
            .unwrap_or(32)
            .min(c.head_dim);
        Some(KvCfg { bits_k, bits_v, group })
    }

    /// bytes of the compressed K (or V) cache of ONE attention layer: (values, scales)
    fn layer_bytes(&self, c: &Config, ctx: usize, bits: u32) -> (usize, usize) {
        let rows = ctx * c.n_head_kv;
        (rows * c.head_dim * bits as usize / 8, rows * (c.head_dim / self.group) * 2)
    }

    /// bytes of the whole cache for `n_attn` layers, plus the fp16 scratch of one layer
    fn total_bytes(&self, c: &Config, n_attn: usize, ctx: usize) -> usize {
        let (kv, ks) = self.layer_bytes(c, ctx, self.bits_k);
        let (vv, vs) = self.layer_bytes(c, ctx, self.bits_v);
        n_attn * (kv + ks + vv + vs) + 2 * ctx * c.kv_dim() * 2
    }
}

struct KvQ {
    cfg: KvCfg,
    kq: DeviceBuffer,
    ks: DeviceBuffer,
    vq: DeviceBuffer,
    vs: DeviceBuffer,
}

impl KvQ {
    fn alloc(cfg: KvCfg, c: &Config, n_attn: usize, ctx: usize) -> Result<KvQ> {
        let (kv, ks) = cfg.layer_bytes(c, ctx, cfg.bits_k);
        let (vv, vs) = cfg.layer_bytes(c, ctx, cfg.bits_v);
        Ok(KvQ {
            cfg,
            kq: DeviceBuffer::alloc(n_attn * kv)?,
            ks: DeviceBuffer::alloc(n_attn * ks)?,
            vq: DeviceBuffer::alloc(n_attn * vv)?,
            vs: DeviceBuffer::alloc(n_attn * vs)?,
        })
    }

    /// device pointers of attention layer `slot`: [K values, K scales, V values, V scales]
    fn ptrs(&self, slot: usize, c: &Config, ctx: usize) -> [u64; 4] {
        let (kv, ks) = self.cfg.layer_bytes(c, ctx, self.cfg.bits_k);
        let (vv, vs) = self.cfg.layer_bytes(c, ctx, self.cfg.bits_v);
        [
            self.kq.ptr() + (slot * kv) as u64,
            self.ks.ptr() + (slot * ks) as u64,
            self.vq.ptr() + (slot * vv) as u64,
            self.vs.ptr() + (slot * vs) as u64,
        ]
    }
}

/// Blocks per SM of the dequantizer in the pipelined prefill (FORGE_PIPE_BLOCKS, default 2).
fn pipe_blocks_per_sm() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| std::env::var("FORGE_PIPE_BLOCKS").ok().and_then(|v| v.parse::<usize>().ok()).filter(|&n| n >= 1).unwrap_or(2))
}

/// What the pipelined prefill GEMM of host-resident matrices needs: a second stream for the dequantizer (it reads the weights over PCIe, the slow part) and the events that order it with the GEMMs.
struct PipeSync {
    dq: Stream,
    ready: [Event; 2],
    free: [Event; 2],
}

impl PipeSync {
    fn new() -> Result<PipeSync> {
        Ok(PipeSync {
            dq: Stream::new()?,
            ready: [Event::new()?, Event::new()?],
            free: [Event::new()?, Event::new()?],
        })
    }
}

pub struct GpuSession<'m> {
    m: &'m GpuModel,
    stream: Stream,
    /// FORGE_PREFILL_PIPE=1: FFN matrices in host memory are dequantized block by block on a second stream while the tensor cores multiply the block before. Bit-identical to the plain path but SLOWER on the
    /// 27B (130-139 tok/s against 154: the GEMM kernels and the PCIe-bound dequantizer fight for the SMs, and a dequantizer with few blocks cannot keep the link busy), so it is off by default
    pipe: Option<PipeSync>,
    max_ctx: usize,
    x: DeviceBuffer,
    q: DeviceBuffer,
    k_raw: DeviceBuffer,
    attn_out: DeviceBuffer,
    act: DeviceBuffer,
    h_final: DeviceBuffer,
    logits: DeviceBuffer,
    pv: DeviceBuffer,
    pi: DeviceBuffer,
    // pre-quantized activations (int8 values, fp32 scale and integer sum per 32 values): the normed residual stream, the attention output, the SwiGLU output
    xq_e: DeviceBuffer,
    xd_e: DeviceBuffer,
    xs_e: DeviceBuffer,
    xq_a: DeviceBuffer,
    xd_a: DeviceBuffer,
    xs_a: DeviceBuffer,
    xq_f: DeviceBuffer,
    xd_f: DeviceBuffer,
    xs_f: DeviceBuffer,
    old_q8: bool,
    state: DeviceBuffer,
    ring: DeviceBuffer,
    cos: DeviceBuffer,
    sin: DeviceBuffer,
    k_cache: DeviceBuffer,
    v_cache: DeviceBuffer,
    /// layer -> index among the attention layers (the KV cache holds attention layers only)
    kv_slot: Vec<usize>,
    /// layer -> index among the Gated DeltaNet layers
    dn_slot: Vec<usize>,
    // hybrid models: projections of a DeltaNet layer, gated attention scratch, the recurrent states
    h_mixed: DeviceBuffer,
    h_z: DeviceBuffer,
    h_ab: DeviceBuffer,
    h_qg: DeviceBuffer,
    h_v: DeviceBuffer,
    samp_partial: DeviceBuffer,
    samp_out: DeviceBuffer,
    samp_ids: DeviceBuffer,
    // split-KV attention (long contexts): per (head, split) softmax statistics and unnormalised outputs
    as_m: DeviceBuffer,
    as_l: DeviceBuffer,
    as_o: DeviceBuffer,
    dn_s: DeviceBuffer,
    dn_conv: DeviceBuffer,
    vb: Option<VerifyBufs>,
    // the MTP head: its attention layer's KV cache, scratch, and how many positions (from 0) have a valid KV entry
    mtp_k: DeviceBuffer,
    mtp_v: DeviceBuffer,
    mtp_x: DeviceBuffer,
    mtp_xq: DeviceBuffer,
    mtp_xd: DeviceBuffer,
    mtp_xs: DeviceBuffer,
    mtp_tok: DeviceBuffer,
    mtp_out: DeviceBuffer,
    /// the draft head scores only the first `draft_rows` tokens of the vocabulary (the BPE ids are in merge order, so these are the frequent ones): a fraction of the weight traffic of the full head
    draft_rows: usize,
    pub mtp_filled: usize,
    layers_graph: GraphExec,
    head_graph: GraphExec,
    pf: Option<PrefillBufs>,
    eval: Option<EvalBufs>,
    stats: Option<ActStats>,
    /// K-quant matrices are multiplied by the fused dequantizing GEMMs when the chunk has at most this many tokens: the decode inside the GEMM makes it about 18 % slower than the plain GEMM, the separate
    /// dequantization pass (fast kernels) costs a fixed ~70 ms on the 9B whatever the chunk size, so the fused kernel wins for the short chunks of a chat's follow-up turns (+50 % at 64 tokens, +15 % at 256)
    /// and loses from about 400 tokens. FORGE_FUSED_MAX=N sets the limit, FORGE_NO_FUSED=1 means 0.
    fused_max: [usize; 3],
    /// the fused GEMMs accumulate in fp16 and flush into fp32 after every k16 step (see `Kernels::gemmq_h`). On by default on GeForce-class Ampere and Ada (compute capability 8.6 and 8.9, where the
    /// fp32-accumulate MMA runs at half the rate of the fp16-accumulate one: +5 to 11 percent of prompt speed on an RTX 4060, outputs within 3e-5 nats of the fp32 form); FORGE_F16ACC=1 / 0 forces it
    /// on (compute capability 8 and newer) or off
    f16acc: bool,
    /// the dequantize-then-multiply path of the prompt GEMMs uses the Ampere kernels of gemm_s80.cu (sm_80 and newer; FORGE_PLAIN_S80=0 keeps the Turing-style ones of gemm.cu)
    plain_s80: bool,
    /// the Q4_K, Q5_K and Q6_K matrices of a prompt chunk of at least `mmq_min_m` tokens are multiplied by the integer tensor-core GEMMs of gemm_mmq.cu: the activations are quantized to int8 with one scale per 32 values (the
    /// numerics of the decode kernels and of llama.cpp's mul_mat_q kernels) and the weights stay 4-bit. On by default from compute capability 8.0 (FORGE_MMQ=0 turns it off, FORGE_MMQ_MIN_M=N sets the shortest chunk, 16)
    mmq: bool,
    mmq_min_m: usize,
    /// roles (bit i of `role`) whose matrices skip the integer GEMMs (FORGE_MMQ_SKIP, for quality studies)
    mmq_skip: u32,
    /// ... only for chunks that start at this position or later (FORGE_MMQ_SKIP_FROM, default 0: always): the integer GEMMs lose almost nothing in the first 2,000 tokens, the recurrent state of a hybrid model
    /// only accumulates their noise after that
    mmq_skip_from: usize,
    /// quality study (FORGE_FAKEQ): per role, the (group, mode) of an in-place quantize / dequantize of the activations before the fp16 GEMM
    fakeq: [Option<(i32, i32)>; 9],
    /// the start position of the prompt chunk being processed
    chunk_pos: std::cell::Cell<usize>,
    /// the role of the matrix `gemm_w` is about to multiply (set by the layer code)
    role: std::cell::Cell<u8>,
    /// the SwiGLU and the int8 quantization of the down projection's input in one kernel (FORGE_SWIGLU_Q=0 keeps the two kernels); bit-identical results
    swiglu_q: bool,
    /// the int8 activation buffers could not be allocated (a model that fills the card): the integer GEMMs stay off for the rest of the session
    mmq_no_mem: std::cell::Cell<bool>,
    /// the prompt attention runs on the Ampere kernels of flash_a80.cu (from compute capability 8; FORGE_FLASH_A80=0 keeps the sm_75 kernels of flash.cu / flash256.cu); fp32 accumulators throughout, bit-identical
    /// to the old kernels at head width 128 and 256
    flash_a80: bool,
    /// the Gated DeltaNet prompt scan on blocks of 128 threads with several state columns per thread (prefill_h.cu `gdn_scan_nc_*`, bit-identical results; FORGE_SCAN_NC=0 keeps one column per thread)
    scan_nc: bool,
    /// the decode attention runs in splits of AS_CH positions (decode.cu `attn_split_kernel`, decode_h.cu `attn_split_g_kernel`; an exact merge follows) instead of one block per head with all scores in shared
    /// memory. Beyond about 11,000 positions (the one-block kernel needs 4 bytes of shared memory per position of the session's context and fails to launch beyond 48 KB) it is the only way; below that it is the
    /// faster one for contexts of more than a thousand positions or so: one block per head leaves a card of 30 SMs with 32 blocks of 4 warps, too little to hide the latency of the key and value loads (Qwen3-4B on
    /// a 2060 SUPER at 6,400 positions: 29 ms per token with the one-block kernel, 14 ms with the split one; 42 new prompt tokens after it 910 ms against 225 ms; at 200 positions both take 10 ms; the hybrid
    /// Qwen3.5-4B, whose attention layers are a quarter of its layers: 12.7 ms against 8.8 ms per token at 6,400 positions, 10.2 against 8.5 at 2,900 and 9.3 against 8.4 at 1,700).
    /// FORGE_ATTN_SPLIT=1 / 0 forces it on / off (0 is ignored where the one-block kernel cannot launch)
    attn_split: bool,
    /// the decode attention reads the compressed KV cache directly (`attn_split_q_kernel`, `attn_split_g_q_kernel`) instead of dequantizing the whole context into the scratch first
    kv_fused: bool,
    /// the split attention runs one block per key-value head for the query heads that share it (decode.cu `attn_gqa_impl`): a key or value row is loaded and decoded once for all of them, and the cross-lane sums of
    /// the heads of a position are made together (6 shuffles for 20 with four heads). The partial results are bit-identical to the one-head kernels'. It needs fewer blocks (a quarter with four heads), so it is used
    /// for sessions with at least FORGE_ATTN_GQA_FROM splits of AS_CH positions (default: 32 blocks' worth, 32 / key-value heads rounded up: 8 for 4 key-value heads, 4 for 8, 16 for 2: below about
    /// 32 blocks the kernel loses, the Qwen3.5-0.8B with 2 key-value heads at 8,192 positions: -7 percent with 16 blocks, +1 to 2 percent at 16,384 with 34); FORGE_ATTN_GQA=0 turns it off. The value is the number of heads per block, the group size padded to a power of two (0: not used).
    attn_gqa: usize,
    /// calibration passes that start or stop inside the network (`eval_chunk_resid`): the residual stream to start from / the one the pass ended with
    resid_in: Option<Vec<f32>>,
    resid_out: Option<Vec<f32>>,
    /// chunks of at most this many tokens go through the multi-row kernels (mrow.rs) instead of the GEMM path (FORGE_MROW_MAX=N; 0: never)
    mrow_max: usize,
    /// tokens per multi-row group (4, 8, 12 or 16; FORGE_MROW_GROUP)
    mrow_group: usize,
    /// `eval_chunk` runs through the multi-row kernels too (FORGE_MROW_EVAL=1: the numerics of the path measured by `forge ppl`)
    mrow_eval: bool,
    /// `eval_chunk` runs every row through the single-token decode graphs (FORGE_EVAL_DECODE=1: the third implementation, with the same int8 activations as the multi-row path, to compare both with the GEMM path)
    eval_decode: bool,
    mb: Option<mrow::MrowBufs>,
    /// the launch configurations (warps per block, K slices) found by the autotuner of the multi-row path
    mm_tune: std::cell::RefCell<std::collections::HashMap<mrow::MmKey, (u32, usize)>>,
    cpu: Option<CpuState>,
    /// FORGE_KV: the KV cache of the attention layers is stored in 8 or 4 bits (k_cache / v_cache are then the fp16 scratch of ONE layer)
    kvq: Option<KvQ>,
    pub pos: usize,
    generated: usize,
}

/// Scratch buffers of the batched prefill path (one chunk of PREFILL_CHUNK tokens).
struct PrefillBufs {
    tokens: DeviceBuffer,
    x: DeviceBuffer,   // fp32 [T][E]
    h: DeviceBuffer,   // fp16 [T][E]
    qkv: DeviceBuffer, // fp16 [T][q_dim + 2 kv_dim]
    q: DeviceBuffer,   // fp16 [T][q_dim]
    o: DeviceBuffer,   // fp16 [T][q_dim]
    /// fp16 [T][F] twice: the gate output (first half) and the up output (second half). One allocation, so that a stretch of 2 x T x F halves is free while neither is live (the dequantization slabs of
    /// `gemm_w` live there during the attention, recurrent and down projections)
    gu: DeviceBuffer,
    /// bytes of one half of `gu`
    gu_half: u64,
    a: DeviceBuffer,   // fp16 [T][F]
    /// quantized models: a small matrix no fused kernel takes (the 32-row Q8_0 projections of a Gated DeltaNet layer), dequantized to fp16
    wscratch_small: DeviceBuffer,
    /// quantized models: a large matrix being multiplied, dequantized to fp16. Taken when the first matrix needs it: where the fused kernels read the quantized blocks directly (always on Ada) it is never
    /// touched and its 50 to 100 MiB stay free
    wscratch: std::sync::OnceLock<DeviceBuffer>,
    /// the size `wscratch` gets when it is taken
    wscratch_bytes: usize,
    /// the two blocks of rows of a host-resident matrix in flight (FFN size / 8 elements each; tiny when the pipelined path is off)
    wblk: [DeviceBuffer; 2],
    /// hybrid models only
    hy: Option<HybridBufs>,
    /// the int8 copy of an activation matrix for the integer GEMMs (see `GpuSession::mmq`), taken when the first of them runs
    qa: std::sync::OnceLock<QaBufs>,
}

/// int8 activations [rows][kmax] with the scale (fp32) and the fp16 minimum operand per 32 values ([rows][kmax / 32] each, in the tile layout quant_rows_h_kernel writes): the input arrays of gemm_mmq.cu.
/// `rows` is a multiple of the 128-row tile.
struct QaBufs {
    q: DeviceBuffer,
    d: DeviceBuffer,
    s: DeviceBuffer,
    rows: usize,
    kmax: usize,
}

impl PrefillBufs {
    fn g_ptr(&self) -> u64 {
        self.gu.ptr()
    }
    fn u_ptr(&self) -> u64 {
        self.gu.ptr() + self.gu_half
    }
}

/// Buffers of the evaluation path (`prefill_logits`).
struct EvalBufs {
    /// fp32 [EVAL_ROWS][vocab]
    logits: DeviceBuffer,
    /// the LM head's rows of one slice, dequantized to fp16: [HEAD_SLICE][E]
    wslice: DeviceBuffer,
}

/// Sums of squares of the activations that multiply the columns of the weight matrices (calibration of the quantizer), per layer and for the LM head.
struct ActStats {
    /// per layer: [input of the attention / DeltaNet projections (E), input of the output projection (q_dim or value_dim), input of gate/up (E), input of down (F)]
    layers: Vec<[DeviceBuffer; 4]>,
    /// input of the LM head (the final-normed residual stream)
    head: DeviceBuffer,
    tokens: usize,
    /// optional second moments sum x x^T of the same inputs (fp32 [D][D]): per layer like `layers` (None outside `hess_layers`), and for the head (when `hess_head`)
    hess: Option<(Vec<Option<[DeviceBuffer; 4]>>, Option<DeviceBuffer>)>,
    hess_layers: std::ops::Range<usize>,
    /// layers from this one on are not run (second moments of the early layers of a big model: the rest of the forward pass is not needed); usize::MAX: run them all
    stop_layer: usize,
    /// layers before this one are not run either: the pass starts from a residual stream handed in with `eval_chunk_resid` instead of the embedding (0: from the embedding)
    start_layer: usize,
    /// the pass ends inside the network and nothing but the layers' statistics is wanted: no head, no logits (`eval_chunk_resid` then returns the residual stream where the pass stopped)
    body_only: bool,
}

/// Rows per call of `prefill_logits`, and rows of the LM head per GEMM there.
pub const EVAL_ROWS: usize = 128;
const HEAD_SLICE: usize = 16384;

/// Extra scratch of a chunk through a Gated DeltaNet layer.
struct HybridBufs {
    mixed: DeviceBuffer, // fp32 [T][conv_dim]
    z: DeviceBuffer,     // fp32 [T][value_dim]
    ab: DeviceBuffer,    // fp32 [T][2 n_v]: alpha | beta
    pq: DeviceBuffer,    // fp32 [T][n_v][128]
    pk: DeviceBuffer,
    pv: DeviceBuffer,
    pg: DeviceBuffer,       // fp32 [T][n_v][3]: beta, decay, k.q
    so: DeviceBuffer,       // fp32 [T][value_dim]
    y: DeviceBuffer,        // fp16 [T][value_dim]
    mtp_cat: DeviceBuffer,  // fp16 [T][2 E]: input of the MTP head's fc
    mtp_z: DeviceBuffer,    // fp32 [T][E]
    next_tok: DeviceBuffer, // u32 [T]
}

/// Capacity of the prefill buffers (tokens per chunk at most); `prefill_batched` uses `prefill_chunk_len()` of them.
pub const PREFILL_CHUNK: usize = 1024;

/// Dynamic shared memory of the plain Ampere GEMMs (gemm_s80.cu: three stages of two 128 x 32 half tiles are 49,152 bytes; the kernel file reserves 52,224 for its largest layout)
const S80_SMEM: u32 = 52224;

/// The longest chunk the multi-row path takes when the integer GEMMs are on (`mrow::MROW_MAX_DEFAULT` otherwise)
const MROW_MAX_MMQ: usize = 32;

/// Dynamic shared memory of the integer GEMMs (gemm_mmq.cu `Smem`: two stages of the activation tile and the packed weights, the decoded scales and minimum operands)
const MMQ_SMEM: u32 = 81920;
/// ... and of the Q6_K one (`Smem6`: the weights take 112 bytes per row and stage, the scales 16 + 8 floats per column)
const MMQ6_SMEM: u32 = 98304;
/// Dynamic shared memory of the Ampere width-256 prompt attention (flash_a80.cu: two buffers of a 32-key K tile and V tile, 256 halves per row)
const FLASH_PAIR_SMEM: u32 = 65_536;

/// Bytes of the always-present dequantization scratch for small matrices (see `PrefillBufs::wscratch_small`): matrices of up to 2 M weights.
const WSCRATCH_SMALL: usize = 4 << 20;

/// The chunk length up to which the fused dequantizing GEMMs are used (see `GpuSession::fused_max`). 448 was measured on Turing (RTX 2060 SUPER, T4). On Ada (compute capability 8.9, the RTX 40 series) the fused kernel
/// won at every length measured on an RTX 4060 (Qwen3.5 4B and 9B, Q4_K_M: +5 to +7 percent at 554 tokens, +17 to +21 at 1,086, +8 to +10 at 2,169), so there it is always used. FORGE_FUSED_MAX overrides.
/// Per K-quant format ([Q4_K, Q5_K, Q6_K]). With the Ampere GEMM at fp16 accumulation (`plain_s80`, `f16acc`) the dequantize-then-multiply path (1.5x the speed of the fused kernels on the Q6_K and Q5_K
/// shapes of the 9B at 1,024 tokens, 0.9x on the Q4_K ones: the fused Q4_K kernel is the best of the three) wins above 256 tokens for Q6_K and 384 for Q5_K, and never for Q4_K.
fn default_fused_max(cc_major: i32, cc_minor: i32, two_pass_s80: bool) -> [usize; 3] {
    if (cc_major, cc_minor) == (8, 9) {
        if two_pass_s80 {
            [1 << 20, 384, 256]
        } else {
            [1 << 20; 3]
        }
    } else {
        [448; 3]
    }
}

/// Tokens per prefill chunk: 1024 (longer chunks were measured: 2048 is +2 % on the 4B and -4 % on the 9B, so the buffers stay at 1024). FORGE_PREFILL_CHUNK can lower it.
/// The roles of FORGE_MMQ_SKIP (names or groups, comma separated) as a bit mask: 0 dn_qkv, 1 dn_z, 2 dn_out, 3 attn_q, 4 attn_kv, 5 attn_o, 6 ffn_gate_up, 7 ffn_down; groups dn, attn, ffn, all.
/// FORGE_FAKEQ="roles:group:mode;roles:group:mode" (roles as in FORGE_MMQ_SKIP; group default 32, mode default 0) per role index.
fn fakeq_spec() -> [Option<(i32, i32)>; 9] {
    let mut out = [None; 9];
    for item in std::env::var("FORGE_FAKEQ").unwrap_or_default().split(';') {
        let mut parts = item.split(':');
        let names = parts.next().unwrap_or("");
        let group: i32 = parts.next().and_then(|v| v.parse().ok()).unwrap_or(32);
        let mode: i32 = parts.next().and_then(|v| v.parse().ok()).unwrap_or(0);
        let mut mask = 0u32;
        for name in names.split(',') {
            mask |= match name.trim() {
                "dn_qkv" => 1,
                "dn_z" => 2,
                "dn_out" => 4,
                "attn_q" => 8,
                "attn_kv" => 16,
                "attn_o" => 32,
                "ffn_gate_up" => 64,
                "ffn_down" => 128,
                "dn" => 1 | 2 | 4,
                "attn" => 8 | 16 | 32,
                "ffn" => 64 | 128,
                "all" => 255,
                _ => 0,
            };
        }
        for (r, slot) in out.iter_mut().enumerate().take(8) {
            if mask >> r & 1 != 0 {
                *slot = Some((group, mode));
            }
        }
    }
    out
}

fn mmq_skip_mask() -> u32 {
    let mut m = 0u32;
    for name in std::env::var("FORGE_MMQ_SKIP").unwrap_or_default().split(',') {
        m |= match name.trim() {
            "dn_qkv" => 1,
            "dn_z" => 2,
            "dn_out" => 4,
            "attn_q" => 8,
            "attn_kv" => 16,
            "attn_o" => 32,
            "ffn_gate_up" => 64,
            "ffn_down" => 128,
            "dn" => 1 | 2 | 4,
            "attn" => 8 | 16 | 32,
            "ffn" => 64 | 128,
            "all" => 255,
            _ => 0,
        };
    }
    m
}

pub fn prefill_chunk_len() -> usize {
    std::env::var("FORGE_PREFILL_CHUNK")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&n| n >= 16)
        .map(|n| n.min(PREFILL_CHUNK) / 16 * 16)
        .unwrap_or(1024)
}

impl<'m> GpuSession<'m> {
    pub fn new(m: &'m GpuModel, max_ctx: usize) -> Result<GpuSession<'m>> {
        let c = &m.cfg;
        let max_ctx = max_ctx.min(c.max_ctx).max(1);
        let half = c.rotary_dim / 2;
        let mut cos = vec![0f32; max_ctx * half];
        let mut sin = vec![0f32; max_ctx * half];
        for p in 0..max_ctx {
            for i in 0..half {
                let inv = (c.rope_theta as f64).powf(-((2 * i) as f64) / c.rotary_dim as f64) / c.rope_factors.as_ref().map_or(1.0, |f| f[i] as f64);
                let a = p as f64 * inv;
                cos[p * half + i] = a.cos() as f32;
                sin[p * half + i] = a.sin() as f32;
            }
        }
        let mut kv_slot = Vec::with_capacity(c.n_layer);
        let mut dn_slot = Vec::with_capacity(c.n_layer);
        let (mut n_attn, mut n_dn) = (0usize, 0usize);
        for l in 0..c.n_layer {
            kv_slot.push(n_attn);
            dn_slot.push(n_dn);
            if c.is_delta(l) {
                n_dn += 1;
            } else {
                n_attn += 1;
            }
        }
        let kvq_cfg = KvCfg::from_env(c).filter(|_| m.quantized && std::env::var("FORGE_OLD_Q8").is_err());
        let attn_split = match std::env::var("FORGE_ATTN_SPLIT").ok().as_deref() {
            Some("1") => true,
            Some("0") => max_ctx > if m.cfg.delta.is_some() { HYBRID_ONE_BLOCK_MAX } else { DENSE_ONE_BLOCK_MAX },
            _ => max_ctx > ATTN_SPLIT_FROM,
        };
        // the decode attention reads a compressed cache directly (no dequantization pass of the whole context per layer and token); FORGE_KV_FUSED=0 keeps the pass (for comparisons)
        let kv_fused = kvq_cfg.is_some() && attn_split && std::env::var("FORGE_KV_FUSED").map(|v| v != "0").unwrap_or(true);
        let attn_gqa = if attn_split
            && std::env::var("FORGE_ATTN_GQA").map(|v| v != "0").unwrap_or(true)
            && c.n_head_kv > 0
            && c.n_head % c.n_head_kv == 0
            && c.n_head / c.n_head_kv >= 2
            && (max_ctx + AS_CH - 1) / AS_CH >= std::env::var("FORGE_ATTN_GQA_FROM").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or((32 + c.n_head_kv - 1) / c.n_head_kv)
        {
            let gp = (c.n_head / c.n_head_kv).next_power_of_two();
            // the hybrid kernels are for head width 256, the dense ones for 64 and 128; an instantiation must exist for the plain and for the compressed cache
            let have = |quant: bool| m.k.gqa.iter().any(|e| e.0 == c.head_dim && e.1 == gp && e.2 == quant);
            if (c.delta.is_some() == (c.head_dim == 256)) && have(false) && have(true) {
                gp
            } else {
                0
            }
        } else {
            0
        };
        let kv_bytes = if kvq_cfg.is_some() { 1 } else { n_attn.max(1) } * max_ctx * c.kv_dim() * 2;
        let (dn_s_bytes, dn_conv_bytes) = c
            .delta
            .as_ref()
            .map(|d| (d.n_v * 128 * 128 * 4, d.n_v * (d.conv - 1) * 384 * 4))
            .unwrap_or((0, 0));
        let (conv_dim, value_dim, n_v) = c
            .delta
            .as_ref()
            .map(|d| (d.conv_dim(), d.value_dim(), d.n_v))
            .unwrap_or((0, 0, 0));
        let act_a = c.q_dim().max(value_dim);
        let stream = Stream::new()?;
        let f16acc = match std::env::var("FORGE_F16ACC").ok().as_deref() {
            Some("1") => m.dev.info.cc_major >= 8,
            Some(_) => false,
            None => matches!((m.dev.info.cc_major, m.dev.info.cc_minor), (8, 6) | (8, 9)),
        };
        let plain_s80 = m.dev.info.cc_major >= 8 && std::env::var("FORGE_PLAIN_S80").map(|v| v != "0").unwrap_or(true);
        let mmq_on = m.k.mmq_ok && std::env::var("FORGE_MMQ").map(|v| v != "0").unwrap_or(true);
        let flash_a80 = m.dev.info.cc_major >= 8 && std::env::var("FORGE_FLASH_A80").map(|v| v != "0").unwrap_or(true);
        let mut s = GpuSession {
            m,
            max_ctx,
            x: DeviceBuffer::alloc(c.n_embd * 4)?,
            q: DeviceBuffer::alloc(c.q_dim() * 4)?,
            k_raw: DeviceBuffer::alloc(c.kv_dim() * 4)?,
            attn_out: DeviceBuffer::alloc(c.q_dim() * 4)?,
            act: DeviceBuffer::alloc(c.n_ff * 4)?,
            h_final: DeviceBuffer::alloc(c.n_embd * 4)?,
            logits: DeviceBuffer::alloc(c.vocab * 4)?,
            pv: DeviceBuffer::alloc(2 * ARGMAX_BLOCKS as usize * 4)?,
            pi: DeviceBuffer::alloc(2 * ARGMAX_BLOCKS as usize * 4)?,
            xq_e: DeviceBuffer::alloc(c.n_embd)?,
            xd_e: DeviceBuffer::alloc(c.n_embd / 32 * 4)?,
            xs_e: DeviceBuffer::alloc(c.n_embd / 32 * 4)?,
            xq_a: DeviceBuffer::alloc(act_a)?,
            xd_a: DeviceBuffer::alloc(act_a / 32 * 4)?,
            xs_a: DeviceBuffer::alloc(act_a / 32 * 4)?,
            xq_f: DeviceBuffer::alloc(c.n_ff)?,
            xd_f: DeviceBuffer::alloc(c.n_ff / 32 * 4)?,
            xs_f: DeviceBuffer::alloc(c.n_ff / 32 * 4)?,
            old_q8: std::env::var("FORGE_OLD_Q8").is_ok(),
            state: DeviceBuffer::alloc(16)?,
            ring: DeviceBuffer::alloc(RING * 4)?,
            cos: f32_buffer(&cos)?,
            sin: f32_buffer(&sin)?,
            k_cache: DeviceBuffer::alloc(kv_bytes)?,
            v_cache: DeviceBuffer::alloc(kv_bytes)?,
            kv_slot,
            dn_slot,
            h_mixed: DeviceBuffer::alloc(conv_dim.max(1) * 4)?,
            h_z: DeviceBuffer::alloc(value_dim.max(1) * 4)?,
            h_ab: DeviceBuffer::alloc(n_v.max(1) * 8)?,
            h_qg: DeviceBuffer::alloc(c.q_rows() * 4)?,
            h_v: DeviceBuffer::alloc(c.kv_dim() * 4)?,
            samp_partial: DeviceBuffer::alloc(SAMPLE_BLOCKS as usize * 4)?,
            samp_out: DeviceBuffer::alloc((2 + 2 * CAND_CAP) * 4)?,
            samp_ids: DeviceBuffer::alloc(64 * 4)?,
            as_m: DeviceBuffer::alloc(c.n_head * ((max_ctx + AS_CH - 1) / AS_CH) * 4)?,
            as_l: DeviceBuffer::alloc(c.n_head * ((max_ctx + AS_CH - 1) / AS_CH) * 4)?,
            as_o: DeviceBuffer::alloc(c.n_head * ((max_ctx + AS_CH - 1) / AS_CH) * 256 * 4)?,
            dn_s: DeviceBuffer::alloc((dn_s_bytes * n_dn).max(16))?,
            dn_conv: DeviceBuffer::alloc((dn_conv_bytes * n_dn).max(16))?,
            vb: None,
            mtp_k: DeviceBuffer::alloc(if m.mtp.is_some() {
                max_ctx * c.kv_dim() * 2
            } else {
                16
            })?,
            mtp_v: DeviceBuffer::alloc(if m.mtp.is_some() {
                max_ctx * c.kv_dim() * 2
            } else {
                16
            })?,
            mtp_x: DeviceBuffer::alloc(c.n_embd * 4)?,
            mtp_xq: DeviceBuffer::alloc(2 * c.n_embd)?,
            mtp_xd: DeviceBuffer::alloc(2 * c.n_embd / 32 * 4)?,
            mtp_xs: DeviceBuffer::alloc(2 * c.n_embd / 32 * 4)?,
            mtp_tok: DeviceBuffer::alloc(8)?,
            mtp_out: DeviceBuffer::alloc(8)?,
            draft_rows: std::env::var("FORGE_DRAFT_ROWS")
                .ok()
                .and_then(|v| v.parse().ok())
                .filter(|&n: &usize| n > 0)
                .unwrap_or(65536)
                .min(c.vocab),
            mtp_filled: 0,
            layers_graph: placeholder_graph(&stream)?,
            head_graph: placeholder_graph(&stream)?,
            pf: None,
            eval: None,
            stats: None,
            f16acc,
            plain_s80,
            mmq: mmq_on,
            mmq_min_m: std::env::var("FORGE_MMQ_MIN_M")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(16),
            mmq_no_mem: std::cell::Cell::new(false),
            mmq_skip: mmq_skip_mask(),
            fakeq: fakeq_spec(),
            mmq_skip_from: std::env::var("FORGE_MMQ_SKIP_FROM").ok().and_then(|v| v.parse().ok()).unwrap_or(0),
            chunk_pos: std::cell::Cell::new(0),
            role: std::cell::Cell::new(0),
            swiglu_q: std::env::var("FORGE_SWIGLU_Q").map(|v| v != "0").unwrap_or(true),
            flash_a80,
            scan_nc: std::env::var("FORGE_SCAN_NC").map(|v| v != "0").unwrap_or(true),
            attn_split,
            kv_fused,
            attn_gqa,
            fused_max: if std::env::var("FORGE_NO_FUSED").is_ok() {
                [0; 3]
            } else {
                match std::env::var("FORGE_FUSED_MAX").ok().and_then(|v| v.parse().ok()) {
                    Some(v) => [v; 3],
                    None => default_fused_max(m.dev.info.cc_major, m.dev.info.cc_minor, plain_s80 && f16acc),
                }
            },
            resid_in: None,
            resid_out: None,
            // (with weights in host memory every group of rows would read them over PCIe again: the dequantizing path reads them once per chunk)
            // with weights in host memory the groups of rows of the multi-row path (up to 4 with the FFN on the CPU: one pass over the host weights at 0.3 s on the 27B) serve the few tokens at the end of a prompt;
            // longer chunks go through the dequantizing GEMM path, which reads the weights over PCIe once per chunk (0.8 s)
            mrow_max: std::env::var("FORGE_MROW_MAX")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(if m.host_ffn_bytes > 0 {
                    if m.layers.iter().any(|l| l.cpu.is_some()) {
                        4
                    } else {
                        0
                    }
                } else if kvq_cfg.is_some() {
                    0
                } else if mmq_on {
                    // with the integer GEMMs the GEMM path beats groups of 16 rows from the third group on (48 tokens of the 9B: 891 against 616 tokens/s; at 32 they are level)
                    MROW_MAX_MMQ
                } else {
                    mrow::MROW_MAX_DEFAULT
                }),
            mrow_group: std::env::var("FORGE_MROW_GROUP")
                .ok()
                .and_then(|v| v.parse().ok())
                .filter(|v| mrow::MM_T.contains(v))
                .unwrap_or(mrow::MROW_GROUP_DEFAULT),
            mrow_eval: std::env::var("FORGE_MROW_EVAL")
                .map(|v| v == "1")
                .unwrap_or(false),
            eval_decode: std::env::var("FORGE_EVAL_DECODE")
                .map(|v| v == "1")
                .unwrap_or(false),
            mb: None,
            mm_tune: Default::default(),
            stream,
            pipe: if std::env::var("FORGE_PREFILL_PIPE").map(|v| v == "1").unwrap_or(false) && m.layers.iter().any(|l| l.w_gate.is_host() && l.w_gate.qf.is_some()) {
                Some(PipeSync::new()?)
            } else {
                None
            },
            cpu: CpuState::new(m)?,
            kvq: match kvq_cfg {
                Some(cfg) => Some(KvQ::alloc(cfg, c, n_attn.max(1), max_ctx)?),
                None => None,
            },
            pos: 0,
            generated: 0,
        };
        s.state.upload(&[0u8; 16])?;
        if m.eval_only {
            s.reset()?;
        } else {
            s.build_graphs()?;
        }
        Ok(s)
    }

    fn cpu_ffn_layer(&self, li: usize) -> bool {
        self.cpu.as_ref().map(|c| c.ctx[li].is_some()).unwrap_or(false)
    }

    /// The FFN of layer `li` on the CPU for `cols` token columns (1..=4): the normed, quantized residual streams (`xq`, `xd`, `xs`, as the quantization kernel left them) go to pinned memory, a host node
    /// runs the FFN, the result comes back and is added to the residual streams `x` ([cols][E] fp32).
    fn enqueue_cpu_ffn(&self, li: usize, cols: usize, xq: &DeviceBuffer, xd: &DeviceBuffer, xs: &DeviceBuffer, x: &DeviceBuffer) -> Result<()> {
        self.enqueue_cpu_ffn_at(li, cols, xq, xd, xs, x.ptr())
    }

    /// As `enqueue_cpu_ffn`, the residual streams (fp32 [cols][E]) at device address `x`.
    fn enqueue_cpu_ffn_at(&self, li: usize, cols: usize, xq: &DeviceBuffer, xd: &DeviceBuffer, xs: &DeviceBuffer, x: u64) -> Result<()> {
        let m = self.m;
        let e = m.cfg.n_embd;
        let st = &self.stream;
        let cp = self.cpu.as_ref().unwrap();
        st.download_async(&cp.hin_q, xq, 0, cols * e)?;
        st.download_async(&cp.hin_d, xd, 0, cols * e / 32 * 4)?;
        st.download_async(&cp.hin_s, xs, 0, cols * e / 32 * 4)?;
        let ctx: &CpuCtx = &cp.ctx[li].as_ref().unwrap()[cols - 1];
        st.launch_host_func(cpu_ffn_callback, ctx as *const CpuCtx as *mut c_void)?;
        st.upload_async(&cp.d_out, 0, &cp.hout, cols * e * 4)?;
        m.k.add_f32.launch(
            (((cols * e) as u32 + 255) / 256, 1, 1),
            (256, 1, 1),
            0,
            st,
            &Args::new().ptr(x).buf(&cp.d_out).i32((cols * e) as i32),
        )?;
        Ok(())
    }

    /// byte offset of an attention layer's K (or V) cache (a compressed cache has one fp16 scratch layer: offset 0 for all)
    fn kv_off(&self, li: usize) -> u64 {
        if self.kvq.is_some() {
            return 0;
        }
        (self.kv_slot[li] * self.max_ctx * self.m.cfg.kv_dim() * 2) as u64
    }

    /// Does the decode attention read the compressed KV cache directly (`attn_split_q_kernel`, `attn_split_g_q_kernel`) instead of dequantizing the whole context into the scratch first? It does when the cache is compressed
    /// and the split attention kernels are in use; FORGE_KV_FUSED=0 keeps the dequantization pass (for comparisons). The dequantization wrote and read the whole context as fp16 for every layer of every token.
    fn kv_fused(&self) -> bool {
        self.kv_fused
    }

    /// The split attention launch of the fused kind: the arguments of `attn_split_kernel` / `attn_split_g_kernel` (`q`, the scratch rows of the step `kc` / `vc`, the partial results, the step state) and the compressed
    /// store of attention layer `li`; `rows` tokens of the step (grid z).
    #[allow(clippy::too_many_arguments)]
    fn launch_split_fused(&self, hybrid: bool, li: usize, rows: usize, q: u64, kc: u64, vc: u64, pm: u64, pl: u64, po: u64, state: u64, ns_max: usize) -> Result<()> {
        let kq = self.kvq.as_ref().expect("a compressed cache");
        let c = &self.m.cfg;
        let p = kq.ptrs(self.kv_slot[li], c, self.max_ctx);
        let args = Args::new()
            .ptr(q)
            .ptr(p[0])
            .ptr(p[1])
            .ptr(p[2])
            .ptr(p[3])
            .ptr(kc)
            .ptr(vc)
            .ptr(pm)
            .ptr(pl)
            .ptr(po)
            .ptr(state)
            .i32(c.n_head as i32)
            .i32(c.n_head_kv as i32)
            .i32(c.head_dim as i32)
            .i32(ns_max as i32)
            .i32(kq.cfg.group as i32)
            .i32(kq.cfg.bits_k as i32)
            .i32(kq.cfg.bits_v as i32);
        if let Some(f) = self.gqa_kernel(true) {
            // one block per key-value head (the head width is an argument for the scale only, the kernel is built for it)
            let gargs = Args::new()
                .ptr(q)
                .ptr(p[0])
                .ptr(p[1])
                .ptr(p[2])
                .ptr(p[3])
                .ptr(kc)
                .ptr(vc)
                .ptr(pm)
                .ptr(pl)
                .ptr(po)
                .ptr(state)
                .i32(c.n_head as i32)
                .i32(c.n_head_kv as i32)
                .i32(c.head_dim as i32)
                .i32(ns_max as i32)
                .i32(kq.cfg.group as i32)
                .i32(kq.cfg.bits_k as i32)
                .i32(kq.cfg.bits_v as i32);
            let grid = (c.n_head_kv as u32, ns_max as u32, rows as u32);
            return f.launch(grid, (if hybrid { 256 } else { 128 }, 1, 1), 0, &self.stream, &gargs);
        }
        if hybrid {
            self.m.k.attn_split_g_q.launch((c.n_head as u32, ns_max as u32, rows as u32), (256, 1, 1), 0, &self.stream, &args)
        } else {
            self.m.k.attn_split_q.launch((c.n_head as u32, ns_max as u32, rows as u32), (128, 1, 1), 0, &self.stream, &args)
        }
    }

    /// The GQA-shared split attention kernel of this session for the plain (`compressed` false) or the compressed cache, if the session uses one.
    fn gqa_kernel(&self, compressed: bool) -> Option<&Function> {
        if self.attn_gqa == 0 {
            return None;
        }
        let hd = self.m.cfg.head_dim;
        self.m.k.gqa.iter().find(|e| e.0 == hd && e.1 == self.attn_gqa && e.2 == compressed).map(|e| &e.3)
    }

    /// The split attention launch for the plain fp16 cache: `attn_split_kernel` / `attn_split_g_kernel` (one block per query head), or the GQA-shared kernel (one block per key-value head, the same partial
    /// results bit for bit) in a long session. `rows` tokens of the step (grid z).
    #[allow(clippy::too_many_arguments)]
    fn launch_split_plain(&self, hybrid: bool, rows: usize, q: u64, kc: u64, vc: u64, pm: u64, pl: u64, po: u64, state: u64, ns_max: usize) -> Result<()> {
        let c = &self.m.cfg;
        let k = &self.m.k;
        if let Some(f) = self.gqa_kernel(false) {
            let gargs = Args::new().ptr(q).ptr(kc).ptr(vc).ptr(pm).ptr(pl).ptr(po).ptr(state).i32(c.n_head as i32).i32(c.n_head_kv as i32).i32(c.head_dim as i32).i32(ns_max as i32);
            let grid = (c.n_head_kv as u32, ns_max as u32, rows as u32);
            return f.launch(grid, (if hybrid { 256 } else { 128 }, 1, 1), 0, &self.stream, &gargs);
        }
        let args = Args::new().ptr(q).ptr(kc).ptr(vc).ptr(pm).ptr(pl).ptr(po).ptr(state).i32(c.n_head as i32).i32(c.n_head_kv as i32).i32(c.head_dim as i32).i32(ns_max as i32);
        let grid = (c.n_head as u32, ns_max as u32, rows as u32);
        if hybrid {
            k.attn_split_g.launch(grid, (256, 1, 1), 0, &self.stream, &args)
        } else {
            k.attn_split.launch(grid, (128, 1, 1), 0, &self.stream, &args)
        }
    }

    /// compressed KV cache: decompress the `t` positions before the current ones of attention layer `li` into the scratch (`None`: as many as the step state's position says, for the captured graphs)
    fn kv_pre(&self, li: usize, t: Option<usize>, state: u64) -> Result<()> {
        let Some(q) = &self.kvq else {
            return Ok(());
        };
        let c = &self.m.cfg;
        let n = t.unwrap_or(self.max_ctx);
        if n == 0 {
            return Ok(());
        }
        let p = q.ptrs(self.kv_slot[li], c, self.max_ctx);
        let blocks = ((n * c.n_head_kv + 7) / 8).min(2048) as u32;
        self.m.k.kv_dequant.launch(
            (blocks, 2, 1),
            (256, 1, 1),
            0,
            &self.stream,
            &Args::new()
                .ptr(p[0])
                .ptr(p[1])
                .buf(&self.k_cache)
                .ptr(p[2])
                .ptr(p[3])
                .buf(&self.v_cache)
                .ptr(state)
                .i32(t.map(|v| v as i32).unwrap_or(-1))
                .i32(c.n_head_kv as i32)
                .i32(c.head_dim as i32)
                .i32(q.cfg.group as i32)
                .i32(q.cfg.bits_k as i32)
                .i32(q.cfg.bits_v as i32),
        )
    }

    /// compressed KV cache: compress the `n_new` rows the layer just wrote into the scratch (from position `pos0`, `None`: the step state's) into the store
    fn kv_post(&self, li: usize, pos0: Option<usize>, n_new: usize, state: u64) -> Result<()> {
        let Some(q) = &self.kvq else {
            return Ok(());
        };
        let c = &self.m.cfg;
        let p = q.ptrs(self.kv_slot[li], c, self.max_ctx);
        let blocks = ((n_new * c.n_head_kv + 7) / 8).clamp(1, 2048) as u32;
        self.m.k.kv_quant.launch(
            (blocks, 2, 1),
            (256, 1, 1),
            0,
            &self.stream,
            &Args::new()
                .buf(&self.k_cache)
                .ptr(p[0])
                .ptr(p[1])
                .buf(&self.v_cache)
                .ptr(p[2])
                .ptr(p[3])
                .ptr(state)
                .i32(pos0.map(|v| v as i32).unwrap_or(-1))
                .i32(n_new as i32)
                .i32(c.n_head_kv as i32)
                .i32(c.head_dim as i32)
                .i32(q.cfg.group as i32)
                .i32(q.cfg.bits_k as i32)
                .i32(q.cfg.bits_v as i32),
        )
    }

    fn attention_smem(&self) -> u32 {
        // scores[max_ctx] + q[hd] + part[NGRP * hd] + red[32], NGRP = 128 / (hd / 8)
        let hd = self.m.cfg.head_dim;
        let ngrp = 128 / (hd / 8);
        ((self.max_ctx + hd + ngrp * hd + 32) * 4) as u32
    }

    fn enqueue_layers(&self) -> Result<()> {
        let m = self.m;
        let c = &m.cfg;
        let k = &m.k;
        let st = &self.stream;
        let q8 = m.fmt == Fmt::Q8_0;
        if c.delta.is_some() {
            return self.enqueue_layers_h();
        }
        if m.quantized && !self.old_q8 {
            return self.enqueue_layers_q();
        }
        let (e, hd) = (c.n_embd as i32, c.head_dim as i32);
        let emb_args = if q8 {
            Args::new()
                .buf(&m.tok_emb.w)
                .buf(m.tok_emb.s.as_ref().unwrap())
                .buf(&self.state)
                .buf(&self.x)
                .i32(e)
        } else {
            Args::new()
                .buf(&m.tok_emb.w)
                .buf(&self.state)
                .buf(&self.x)
                .i32(e)
        };
        (if q8 { k.embed_q8 } else { k.embed }).launch(
            ((c.n_embd as u32 + 255) / 256, 1, 1),
            (256, 1, 1),
            0,
            st,
            &emb_args,
        )?;
        // threads per block of the stage kernels (warps own rows; rows per block = warps): an experiment knob, FORGE_TPB = 128 or 256
        let tpb: u32 = std::env::var("FORGE_TPB")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(128);
        let wpb = (tpb / 32) as usize;
        let units = (c.n_head + 2 * c.n_head_kv) * c.head_dim / 2;
        let qkv_blocks = ((units + wpb - 1) / wpb) as u32;
        let kv_dim = c.kv_dim();
        let rotate = if c.qk_norm { 0 } else { 1 };
        for (li, l) in m.layers.iter().enumerate() {
            let kc = self.k_cache.ptr() + self.kv_off(li);
            let vc = self.v_cache.ptr() + self.kv_off(li);
            // S1
            if q8 {
                k.norm_qkv_rope_q8.for_k(c.n_embd)?.launch(
                    (qkv_blocks, 1, 1),
                    (tpb, 1, 1),
                    q8_smem(c.n_embd),
                    st,
                    &Args::new()
                        .buf(&self.x)
                        .buf(&l.attn_norm)
                        .buf(&l.wq.w)
                        .buf(l.wq.s.as_ref().unwrap())
                        .buf(&l.wk.w)
                        .buf(l.wk.s.as_ref().unwrap())
                        .buf(&l.wv.w)
                        .buf(l.wv.s.as_ref().unwrap())
                        .buf(&self.q)
                        .buf(&self.k_raw)
                        .ptr(kc)
                        .ptr(vc)
                        .buf(&self.cos)
                        .buf(&self.sin)
                        .buf(&self.state)
                        .i32(e)
                        .i32(c.n_head as i32)
                        .i32(c.n_head_kv as i32)
                        .i32(hd)
                        .f32(c.rms_eps)
                        .i32(rotate),
                )?;
            } else {
                k.norm_qkv_rope.for_k(c.n_embd)?.launch(
                    (qkv_blocks, 1, 1),
                    (tpb, 1, 1),
                    ((c.n_embd + 32) * 4) as u32,
                    st,
                    &Args::new()
                        .buf(&self.x)
                        .buf(&l.attn_norm)
                        .buf(&l.wq.w)
                        .buf(&l.wk.w)
                        .buf(&l.wv.w)
                        .buf(&self.q)
                        .buf(&self.k_raw)
                        .ptr(kc)
                        .ptr(vc)
                        .buf(&self.cos)
                        .buf(&self.sin)
                        .buf(&self.state)
                        .i32(e)
                        .i32(c.n_head as i32)
                        .i32(c.n_head_kv as i32)
                        .i32(hd)
                        .f32(c.rms_eps)
                        .i32(rotate),
                )?;
            }
            if c.qk_norm {
                k.qk_norm_rope.launch(
                    ((c.n_head + c.n_head_kv) as u32, 1, 1),
                    (128, 1, 1),
                    0,
                    st,
                    &Args::new()
                        .buf(&self.q)
                        .buf(&self.k_raw)
                        .ptr(kc)
                        .buf(l.q_norm.as_ref().unwrap())
                        .buf(l.k_norm.as_ref().unwrap())
                        .buf(&self.cos)
                        .buf(&self.sin)
                        .buf(&self.state)
                        .i32(c.n_head as i32)
                        .i32(c.n_head_kv as i32)
                        .i32(hd)
                        .f32(c.rms_eps),
                )?;
            }
            // S2
            if self.attn_split {
                // long contexts: split-KV attention (partial softmax per block of AS_CH positions), then the exact merge
                let ns_max = (self.max_ctx + AS_CH - 1) / AS_CH;
                k.attn_split.launch(
                    (c.n_head as u32, ns_max as u32, 1),
                    (128, 1, 1),
                    0,
                    st,
                    &Args::new()
                        .buf(&self.q)
                        .ptr(kc)
                        .ptr(vc)
                        .buf(&self.as_m)
                        .buf(&self.as_l)
                        .buf(&self.as_o)
                        .buf(&self.state)
                        .i32(c.n_head as i32)
                        .i32(c.n_head_kv as i32)
                        .i32(hd)
                        .i32(ns_max as i32),
                )?;
                k.attn_merge.launch(
                    (c.n_head as u32, 1, 1),
                    (128, 1, 1),
                    0,
                    st,
                    &Args::new()
                        .buf(&self.as_m)
                        .buf(&self.as_l)
                        .buf(&self.as_o)
                        .buf(&self.attn_out)
                        .buf(&self.state)
                        .i32(ns_max as i32)
                        .i32(hd),
                )?;
            } else {
                k.attention.launch(
                    (c.n_head as u32, 1, 1),
                    (128, 1, 1),
                    self.attention_smem(),
                    st,
                    &Args::new()
                        .buf(&self.q)
                        .ptr(kc)
                        .ptr(vc)
                        .buf(&self.attn_out)
                        .buf(&self.state)
                        .i32(c.n_head as i32)
                        .i32(c.n_head_kv as i32)
                        .i32(hd)
                        .i32(self.max_ctx as i32),
                )?;
            }
            // S3 and S5 share the residual matvec kernel
            let blocks_e = (((c.n_embd + wpb - 1) / wpb) as u32, 1, 1);
            if q8 {
                k.matvec_residual_q8.for_k(c.q_dim())?.launch(
                    blocks_e,
                    (tpb, 1, 1),
                    q8_smem(c.q_dim()),
                    st,
                    &Args::new()
                        .buf(&l.wo.w)
                        .buf(l.wo.s.as_ref().unwrap())
                        .buf(&self.attn_out)
                        .buf(&self.x)
                        .i32(e)
                        .i32(c.q_dim() as i32),
                )?;
            } else {
                k.matvec_residual.for_k(c.q_dim())?.launch(
                    blocks_e,
                    (tpb, 1, 1),
                    (c.q_dim() * 4) as u32,
                    st,
                    &Args::new()
                        .buf(&l.wo.w)
                        .buf(&self.attn_out)
                        .buf(&self.x)
                        .i32(e)
                        .i32(c.q_dim() as i32),
                )?;
            }
            // S4
            let blocks_f = (((c.n_ff + 2 * wpb - 1) / (2 * wpb)) as u32, 1, 1);
            if q8 {
                k.norm_gateup_q8.for_k(c.n_embd)?.launch(
                    blocks_f,
                    (tpb, 1, 1),
                    q8_smem(c.n_embd),
                    st,
                    &Args::new()
                        .buf(&self.x)
                        .buf(&l.ffn_norm)
                        .buf(&l.w_gate.w)
                        .buf(l.w_gate.s.as_ref().unwrap())
                        .buf(&l.w_up.w)
                        .buf(l.w_up.s.as_ref().unwrap())
                        .buf(&self.act)
                        .i32(e)
                        .i32(c.n_ff as i32)
                        .f32(c.rms_eps),
                )?;
            } else {
                k.norm_gateup.for_k(c.n_embd)?.launch(
                    blocks_f,
                    (tpb, 1, 1),
                    ((c.n_embd + 32) * 4) as u32,
                    st,
                    &Args::new()
                        .buf(&self.x)
                        .buf(&l.ffn_norm)
                        .buf(&l.w_gate.w)
                        .buf(&l.w_up.w)
                        .buf(&self.act)
                        .i32(e)
                        .i32(c.n_ff as i32)
                        .f32(c.rms_eps),
                )?;
            }
            // S5
            if q8 {
                k.matvec_residual_q8.for_k(c.n_ff)?.launch(
                    blocks_e,
                    (tpb, 1, 1),
                    q8_smem(c.n_ff),
                    st,
                    &Args::new()
                        .buf(&l.w_down.w)
                        .buf(l.w_down.s.as_ref().unwrap())
                        .buf(&self.act)
                        .buf(&self.x)
                        .i32(e)
                        .i32(c.n_ff as i32),
                )?;
            } else {
                k.matvec_residual.for_k(c.n_ff)?.launch(
                    blocks_e,
                    (tpb, 1, 1),
                    (c.n_ff * 4) as u32,
                    st,
                    &Args::new()
                        .buf(&l.w_down.w)
                        .buf(&self.act)
                        .buf(&self.x)
                        .i32(e)
                        .i32(c.n_ff as i32),
                )?;
            }
        }
        Ok(())
    }

    /// Persistent grid for `f` with `threads` threads per block: as many blocks as fit on the device at once (Turing: 64K registers, 1024 threads, 16 blocks per SM), at most `max_blocks`.
    fn persistent_grid(&self, f: Function, threads: u32, max_blocks: usize) -> u32 {
        let regs = f.registers().unwrap_or(64).max(16) as u32;
        let per_warp = (regs * 32 + 255) / 256 * 256;
        let per_block = per_warp * (threads / 32);
        let by_regs = 65536 / per_block.max(1);
        let by_threads = 1024 / threads;
        let per_sm = by_regs.min(by_threads).min(16).max(1);
        ((self.m.dev.info.sm_count as u32) * per_sm)
            .min(max_blocks as u32)
            .max(1)
    }

    /// Threads per block and number of blocks for a persistent streaming kernel (one row per warp) with `smem` bytes of dynamic shared memory per block: the block size that keeps the most warps
    /// resident given the registers, the thread limit and the shared memory (64 KB per SM), and one wave of blocks (never more than fit at once, never more than the rows need).
    fn best_launch(&self, f: Function, smem: u32, rows: usize) -> (u32, u32) {
        let regs = f.registers().unwrap_or(64).max(16) as u32;
        let per_warp = (regs * 32 + 255) / 256 * 256;
        let by_smem = 65536 / (smem + 1024).max(1);
        let mut best = (256u32, 0u32, 1u32);
        for threads in [128u32, 192, 256, 320, 384, 512, 640, 768, 1024] {
            let warps = threads / 32;
            let bps = (65536 / (per_warp * warps).max(1))
                .min(1024 / threads)
                .min(by_smem)
                .min(16);
            if bps > 0 && bps * warps > best.1 {
                best = (threads, bps * warps, bps);
            }
        }
        let (threads, _, bps) = best;
        let nw = threads / 32;
        let grid = (self.m.dev.info.sm_count as u32 * bps)
            .min(((rows as u32) + nw - 1) / nw)
            .max(1);
        (threads, grid)
    }

    fn qp(a: Args, m: &GpuMat) -> Args {
        let p = m.ptrs();
        a.ptr(p[0]).ptr(p[1]).ptr(p[2]).ptr(p[3])
    }

    fn same(a: QFmt, b: QFmt) -> Result<QFmt> {
        if a == b {
            Ok(a)
        } else {
            Err(Error::unsupported(
                "ffn_gate and ffn_up must have the same format",
            ))
        }
    }

    fn fid(m: &GpuMat) -> i32 {
        m.qf.map(|f| f as i32).unwrap_or(0)
    }

    /// Quantized decode with pre-quantized activations (decode_q.cu): norm_quant, QKV + RoPE, (QK-norm), attention with a quantized output, Wo, norm_quant, gate/up + SwiGLU + quantize, down.
    /// Every tensor carries its own format (a Q4_K_M file mixes Q4_K and Q6_K); the formats are runtime arguments of the kernels.
    fn enqueue_layers_q(&self) -> Result<()> {
        let m = self.m;
        let c = &m.cfg;
        let k = &m.k;
        let st = &self.stream;
        let (e, hd) = (c.n_embd as i32, c.head_dim as i32);
        k.embed_q.launch(
            ((c.n_embd as u32 + 255) / 256, 1, 1),
            (256, 1, 1),
            0,
            st,
            &Self::qp(Args::new(), &m.tok_emb)
                .i32(Self::fid(&m.tok_emb))
                .buf(&self.state)
                .buf(&self.x)
                .i32(e),
        )?;
        let units = (c.n_head + 2 * c.n_head_kv) * c.head_dim / 2;
        let _ = units;
        let kv_dim = c.kv_dim();
        let rotate = if c.qk_norm { 0 } else { 1 };
        // one block normalises and quantizes the whole vector: element i = thread + k * 512, NI elements per thread
        let nq = k.norm_quant.for_nq(c.n_embd)?;
        let rows_blocks = (((c.n_embd + 7) / 8) as u32, 1, 1);
        for (li, l) in m.layers.iter().enumerate() {
            let kc = self.k_cache.ptr() + self.kv_off(li);
            let vc = self.v_cache.ptr() + self.kv_off(li);
            nq.launch(
                (1, 1, 1),
                (512, 1, 1),
                0,
                st,
                &Args::new()
                    .buf(&self.x)
                    .buf(&l.attn_norm)
                    .buf(&self.xq_e)
                    .buf(&self.xd_e)
                    .buf(&self.xs_e)
                    .i32(e)
                    .f32(c.rms_eps),
            )?;
            // Q/K/V units: n_head * hd/2 of q, n_kv * hd/2 of k, n_kv * hd/2 of v. Tensors of one format are one launch; a mixed layer (Q4_K_M: v is Q6_K in half of the layers) is split by format so that every
            // launch is a compile-time-specialised pipelined kernel.
            let half_d = c.head_dim / 2;
            let (nq_u, nk_u) = (c.n_head * half_d, c.n_head_kv * half_d);
            let total_u = nq_u + 2 * nk_u;
            let (fq, fk, fv) = (l.wq.qf.unwrap(), l.wk.qf.unwrap(), l.wv.qf.unwrap());
            let ranges: Vec<(QFmt, usize, usize)> = if fq == fk && fk == fv {
                vec![(fq, 0, total_u)]
            } else if fq == fk {
                vec![(fq, 0, nq_u + nk_u), (fv, nq_u + nk_u, total_u)]
            } else {
                vec![
                    (fq, 0, nq_u),
                    (fk, nq_u, nq_u + nk_u),
                    (fv, nq_u + nk_u, total_u),
                ]
            };
            if !self.kv_fused() {
                self.kv_pre(li, None, self.state.ptr())?;
            }
            let (bq, bk, bv) = l.bias_ptrs();
            for (fmt_r, lo, hi) in ranges {
                let qkv_fn = k.qkvq.uniform(fmt_r, c.n_embd)?;
                let mut a = Self::qp(Args::new(), &l.wq);
                a = Self::qp(a, &l.wk);
                a = Self::qp(a, &l.wv);
                qkv_fn.launch(
                    (self.persistent_grid(qkv_fn, 256, (hi - lo + 7) / 8), 1, 1),
                    (256, 1, 1),
                    0,
                    st,
                    &a.i32(fmt_r as i32)
                        .i32(fmt_r as i32)
                        .i32(fmt_r as i32)
                        .buf(&self.xq_e)
                        .buf(&self.xd_e)
                        .buf(&self.xs_e)
                        .buf(&self.q)
                        .buf(&self.k_raw)
                        .ptr(kc)
                        .ptr(vc)
                        .buf(&self.cos)
                        .buf(&self.sin)
                        .buf(&self.state)
                        .i32(e)
                        .i32(c.n_head as i32)
                        .i32(c.n_head_kv as i32)
                        .i32(hd)
                        .i32(rotate)
                        .i32(lo as i32)
                        .i32(hi as i32)
                        .ptr(bq)
                        .ptr(bk)
                        .ptr(bv),
                )?;
            }
            if c.qk_norm {
                k.qk_norm_rope.launch(
                    ((c.n_head + c.n_head_kv) as u32, 1, 1),
                    (128, 1, 1),
                    0,
                    st,
                    &Args::new()
                        .buf(&self.q)
                        .buf(&self.k_raw)
                        .ptr(kc)
                        .buf(l.q_norm.as_ref().unwrap())
                        .buf(l.k_norm.as_ref().unwrap())
                        .buf(&self.cos)
                        .buf(&self.sin)
                        .buf(&self.state)
                        .i32(c.n_head as i32)
                        .i32(c.n_head_kv as i32)
                        .i32(hd)
                        .f32(c.rms_eps),
                )?;
            }
            if self.attn_split {
                let ns_max = (self.max_ctx + AS_CH - 1) / AS_CH;
                if self.kv_fused() {
                    self.launch_split_fused(false, li, 1, self.q.ptr(), kc, vc, self.as_m.ptr(), self.as_l.ptr(), self.as_o.ptr(), self.state.ptr(), ns_max)?;
                } else {
                    self.launch_split_plain(false, 1, self.q.ptr(), kc, vc, self.as_m.ptr(), self.as_l.ptr(), self.as_o.ptr(), self.state.ptr(), ns_max)?;
                }
                k.attn_merge_q.launch(
                    (c.n_head as u32, 1, 1),
                    (128, 1, 1),
                    0,
                    st,
                    &Args::new()
                        .buf(&self.as_m)
                        .buf(&self.as_l)
                        .buf(&self.as_o)
                        .buf(&self.xq_a)
                        .buf(&self.xd_a)
                        .buf(&self.xs_a)
                        .buf(&self.state)
                        .i32(ns_max as i32)
                        .i32(hd),
                )?;
            } else {
                k.attention_q.launch(
                    (c.n_head as u32, 1, 1),
                    (128, 1, 1),
                    self.attention_smem(),
                    st,
                    &Args::new()
                        .buf(&self.q)
                        .ptr(kc)
                        .ptr(vc)
                        .buf(&self.xq_a)
                        .buf(&self.xd_a)
                        .buf(&self.xs_a)
                        .buf(&self.state)
                        .i32(c.n_head as i32)
                        .i32(c.n_head_kv as i32)
                        .i32(hd)
                        .i32(self.max_ctx as i32),
                )?;
            }
            self.kv_post(li, None, 1, self.state.ptr())?;
            let (wo_fn, wo_smem) = k.mvq_res.mv_fn(l.wo.qf.unwrap(), c.q_dim())?;
            wo_fn.launch(
                (
                    self.persistent_grid(wo_fn, 256, rows_blocks.0 as usize),
                    1,
                    1,
                ),
                (256, 1, 1),
                wo_smem,
                st,
                &Self::qp(Args::new(), &l.wo)
                    .i32(Self::fid(&l.wo))
                    .buf(&self.xq_a)
                    .buf(&self.xd_a)
                    .buf(&self.xs_a)
                    .buf(&self.x)
                    .i32(e)
                    .i32(c.q_dim() as i32),
            )?;
            nq.launch(
                (1, 1, 1),
                (512, 1, 1),
                0,
                st,
                &Args::new()
                    .buf(&self.x)
                    .buf(&l.ffn_norm)
                    .buf(&self.xq_e)
                    .buf(&self.xd_e)
                    .buf(&self.xs_e)
                    .i32(e)
                    .f32(c.rms_eps),
            )?;
            if self.cpu_ffn_layer(li) {
                self.enqueue_cpu_ffn(li, 1, &self.xq_e, &self.xd_e, &self.xs_e, &self.x)?;
                continue;
            }
            let g = Self::qp(Self::qp(Args::new(), &l.w_gate), &l.w_up);
            let gu_fn = k.gateupq.uniform(
                Self::same(l.w_gate.qf.unwrap(), l.w_up.qf.unwrap())?,
                c.n_embd,
            )?;
            gu_fn.launch(
                (self.persistent_grid(gu_fn, 512, c.n_ff / 32), 1, 1),
                (512, 1, 1),
                0,
                st,
                &g.i32(Self::fid(&l.w_gate))
                    .buf(&self.xq_e)
                    .buf(&self.xd_e)
                    .buf(&self.xs_e)
                    .buf(&self.xq_f)
                    .buf(&self.xd_f)
                    .buf(&self.xs_f)
                    .i32(e)
                    .i32((c.n_ff / 32) as i32),
            )?;
            let (down_fn, down_smem) = k.mvq_res.mv_fn(l.w_down.qf.unwrap(), c.n_ff)?;
            down_fn.launch(
                (
                    self.persistent_grid(down_fn, 256, rows_blocks.0 as usize),
                    1,
                    1,
                ),
                (256, 1, 1),
                down_smem,
                st,
                &Self::qp(Args::new(), &l.w_down)
                    .i32(Self::fid(&l.w_down))
                    .buf(&self.xq_f)
                    .buf(&self.xd_f)
                    .buf(&self.xs_f)
                    .buf(&self.x)
                    .i32(e)
                    .i32(c.n_ff as i32),
            )?;
        }
        Ok(())
    }

    /// A quantized matrix times a pre-quantized activation vector into an fp32 buffer (`rows` outputs of rows of `k` values); the generic stage behind the projections of the hybrid layers.
    #[allow(clippy::too_many_arguments)]
    fn mv_out(
        &self,
        w: &GpuMat,
        xq: &DeviceBuffer,
        xd: &DeviceBuffer,
        xs: &DeviceBuffer,
        out: u64,
        rows: usize,
        k: usize,
    ) -> Result<()> {
        let f = w.qf.ok_or_else(|| {
            Error::unsupported("hybrid models need quantized weights (--q8 or --native)")
        })?;
        let func = self.m.k.headq.uniform(f, k)?;
        func.launch(
            (self.persistent_grid(func, 256, (rows + 7) / 8), 1, 1),
            (256, 1, 1),
            0,
            &self.stream,
            &Self::qp(Args::new(), w)
                .i32(f as i32)
                .buf(xq)
                .buf(xd)
                .buf(xs)
                .ptr(out)
                .i32(rows as i32)
                .i32(k as i32),
        )
    }

    fn attention_smem_h(&self) -> u32 {
        // 256 threads: scores[max_ctx] + q[hd] + part[NGRP * hd] + red[32], NGRP = 256 / (hd / 8)
        let hd = self.m.cfg.head_dim;
        let ngrp = 256 / (hd / 8);
        ((self.max_ctx + hd + ngrp * hd + 32) * 4) as u32
    }

    /// Hybrid decode (Qwen3.5): per layer a Gated DeltaNet block (norm_quant, four projections, the state step, output projection) or a gated attention block (norm_quant, q / k / v projections,
    /// norm + partial RoPE + cache append, attention with the output gate, output projection), then the usual norm_quant, gate/up + SwiGLU, down.
    fn enqueue_layers_h(&self) -> Result<()> {
        let m = self.m;
        let c = &m.cfg;
        let k = &m.k;
        let st = &self.stream;
        let d = c.delta.as_ref().unwrap();
        let (e, hd) = (c.n_embd as i32, c.head_dim as i32);
        k.embed_q.launch(
            ((c.n_embd as u32 + 255) / 256, 1, 1),
            (256, 1, 1),
            0,
            st,
            &Self::qp(Args::new(), &m.tok_emb)
                .i32(Self::fid(&m.tok_emb))
                .buf(&self.state)
                .buf(&self.x)
                .i32(e),
        )?;
        let nq = k.norm_quant.for_nq(c.n_embd)?;
        let rows_blocks = (c.n_embd + 7) / 8;
        let s_bytes = d.n_v * 128 * 128 * 4;
        let conv_bytes = d.n_v * (d.conv - 1) * 384 * 4;
        for (li, l) in m.layers.iter().enumerate() {
            nq.launch(
                (1, 1, 1),
                (512, 1, 1),
                0,
                st,
                &Args::new()
                    .buf(&self.x)
                    .buf(&l.attn_norm)
                    .buf(&self.xq_e)
                    .buf(&self.xd_e)
                    .buf(&self.xs_e)
                    .i32(e)
                    .f32(c.rms_eps),
            )?;
            let (mix_k, mix_wo) = if let Some(dl) = &l.delta {
                let slot = self.dn_slot[li];
                self.mv_out(
                    &dl.w_qkv,
                    &self.xq_e,
                    &self.xd_e,
                    &self.xs_e,
                    self.h_mixed.ptr(),
                    d.conv_dim(),
                    c.n_embd,
                )?;
                self.mv_out(
                    &dl.w_z,
                    &self.xq_e,
                    &self.xd_e,
                    &self.xs_e,
                    self.h_z.ptr(),
                    d.value_dim(),
                    c.n_embd,
                )?;
                self.mv_out(
                    &dl.w_a,
                    &self.xq_e,
                    &self.xd_e,
                    &self.xs_e,
                    self.h_ab.ptr(),
                    d.n_v,
                    c.n_embd,
                )?;
                self.mv_out(
                    &dl.w_b,
                    &self.xq_e,
                    &self.xd_e,
                    &self.xs_e,
                    self.h_ab.ptr() + (d.n_v * 4) as u64,
                    d.n_v,
                    c.n_embd,
                )?;
                k.gdn_step.launch(
                    (d.n_v as u32, 1, 1),
                    (1024, 1, 1),
                    0,
                    st,
                    &Args::new()
                        .buf(&self.h_mixed)
                        .buf(&self.h_z)
                        .buf(&self.h_ab)
                        .buf(&dl.conv)
                        .buf(&dl.a_neg)
                        .buf(&dl.dt_bias)
                        .buf(&dl.norm)
                        .ptr(self.dn_conv.ptr() + (slot * conv_bytes) as u64)
                        .ptr(self.dn_s.ptr() + (slot * s_bytes) as u64)
                        .buf(&self.xq_a)
                        .buf(&self.xd_a)
                        .buf(&self.xs_a)
                        .i32(d.n_k as i32)
                        .i32(d.n_v as i32)
                        .i32(d.tiled_v as i32)
                        .i32(d.conv as i32)
                        .f32(c.rms_eps),
                )?;
                (d.value_dim(), &dl.w_out)
            } else {
                let kc = self.k_cache.ptr() + self.kv_off(li);
                let vc = self.v_cache.ptr() + self.kv_off(li);
                self.mv_out(
                    &l.wq,
                    &self.xq_e,
                    &self.xd_e,
                    &self.xs_e,
                    self.h_qg.ptr(),
                    c.q_rows(),
                    c.n_embd,
                )?;
                self.mv_out(
                    &l.wk,
                    &self.xq_e,
                    &self.xd_e,
                    &self.xs_e,
                    self.k_raw.ptr(),
                    c.kv_dim(),
                    c.n_embd,
                )?;
                self.mv_out(
                    &l.wv,
                    &self.xq_e,
                    &self.xd_e,
                    &self.xs_e,
                    self.h_v.ptr(),
                    c.kv_dim(),
                    c.n_embd,
                )?;
                if !self.kv_fused() {
                    self.kv_pre(li, None, self.state.ptr())?;
                }
                k.qkv35_prep.launch(
                    ((c.n_head + 2 * c.n_head_kv) as u32, 1, 1),
                    (256, 1, 1),
                    0,
                    st,
                    &Args::new()
                        .buf(&self.h_qg)
                        .buf(&self.k_raw)
                        .buf(&self.h_v)
                        .buf(l.q_norm.as_ref().unwrap())
                        .buf(l.k_norm.as_ref().unwrap())
                        .buf(&self.cos)
                        .buf(&self.sin)
                        .buf(&self.state)
                        .buf(&self.q)
                        .ptr(kc)
                        .ptr(vc)
                        .i32(c.n_head as i32)
                        .i32(c.n_head_kv as i32)
                        .i32(hd)
                        .i32(c.rotary_dim as i32)
                        .f32(c.rms_eps),
                )?;
                if self.attn_split {
                    // long contexts: split-KV attention (partial softmax per block of AS_CH positions), then an exact merge with the output gate and the quantization
                    let ns_max = (self.max_ctx + AS_CH - 1) / AS_CH;
                    if self.kv_fused() {
                        self.launch_split_fused(true, li, 1, self.q.ptr(), kc, vc, self.as_m.ptr(), self.as_l.ptr(), self.as_o.ptr(), self.state.ptr(), ns_max)?;
                    } else {
                        self.launch_split_plain(true, 1, self.q.ptr(), kc, vc, self.as_m.ptr(), self.as_l.ptr(), self.as_o.ptr(), self.state.ptr(), ns_max)?;
                    }
                    k.attn_merge_g.launch(
                        (c.n_head as u32, 1, 1),
                        (256, 1, 1),
                        0,
                        st,
                        &Args::new()
                            .buf(&self.as_m)
                            .buf(&self.as_l)
                            .buf(&self.as_o)
                            .ptr(self.h_qg.ptr() + (c.head_dim * 4) as u64)
                            .buf(&self.xq_a)
                            .buf(&self.xd_a)
                            .buf(&self.xs_a)
                            .buf(&self.state)
                            .i32(ns_max as i32)
                            .i32(2 * hd),
                    )?;
                } else {
                    k.attention_g.launch(
                        (c.n_head as u32, 1, 1),
                        (256, 1, 1),
                        self.attention_smem_h(),
                        st,
                        &Args::new()
                            .buf(&self.q)
                            .ptr(kc)
                            .ptr(vc)
                            .ptr(self.h_qg.ptr() + (c.head_dim * 4) as u64)
                            .buf(&self.xq_a)
                            .buf(&self.xd_a)
                            .buf(&self.xs_a)
                            .buf(&self.state)
                            .i32(c.n_head as i32)
                            .i32(c.n_head_kv as i32)
                            .i32(hd)
                            .i32(self.max_ctx as i32)
                            .i32(2 * hd),
                    )?;
                }
                self.kv_post(li, None, 1, self.state.ptr())?;
                (c.q_dim(), &l.wo)
            };
            let (wo_fn, wo_smem) = k.mvq_res.mv_fn(mix_wo.qf.unwrap(), mix_k)?;
            wo_fn.launch(
                (self.persistent_grid(wo_fn, 256, rows_blocks), 1, 1),
                (256, 1, 1),
                wo_smem,
                st,
                &Self::qp(Args::new(), mix_wo)
                    .i32(Self::fid(mix_wo))
                    .buf(&self.xq_a)
                    .buf(&self.xd_a)
                    .buf(&self.xs_a)
                    .buf(&self.x)
                    .i32(e)
                    .i32(mix_k as i32),
            )?;
            nq.launch(
                (1, 1, 1),
                (512, 1, 1),
                0,
                st,
                &Args::new()
                    .buf(&self.x)
                    .buf(&l.ffn_norm)
                    .buf(&self.xq_e)
                    .buf(&self.xd_e)
                    .buf(&self.xs_e)
                    .i32(e)
                    .f32(c.rms_eps),
            )?;
            if self.cpu_ffn_layer(li) {
                self.enqueue_cpu_ffn(li, 1, &self.xq_e, &self.xd_e, &self.xs_e, &self.x)?;
                continue;
            }
            let g = Self::qp(Self::qp(Args::new(), &l.w_gate), &l.w_up);
            let gu_fn = k.gateupq.uniform(
                Self::same(l.w_gate.qf.unwrap(), l.w_up.qf.unwrap())?,
                c.n_embd,
            )?;
            gu_fn.launch(
                (self.persistent_grid(gu_fn, 512, c.n_ff / 32), 1, 1),
                (512, 1, 1),
                0,
                st,
                &g.i32(Self::fid(&l.w_gate))
                    .buf(&self.xq_e)
                    .buf(&self.xd_e)
                    .buf(&self.xs_e)
                    .buf(&self.xq_f)
                    .buf(&self.xd_f)
                    .buf(&self.xs_f)
                    .i32(e)
                    .i32((c.n_ff / 32) as i32),
            )?;
            let (down_fn, down_smem) = k.mvq_res.mv_fn(l.w_down.qf.unwrap(), c.n_ff)?;
            down_fn.launch(
                (self.persistent_grid(down_fn, 256, rows_blocks), 1, 1),
                (256, 1, 1),
                down_smem,
                st,
                &Self::qp(Args::new(), &l.w_down)
                    .i32(Self::fid(&l.w_down))
                    .buf(&self.xq_f)
                    .buf(&self.xd_f)
                    .buf(&self.xs_f)
                    .buf(&self.x)
                    .i32(e)
                    .i32(c.n_ff as i32),
            )?;
        }
        Ok(())
    }

    fn enqueue_head_q(&self) -> Result<()> {
        let m = self.m;
        let c = &m.cfg;
        let k = &m.k;
        let st = &self.stream;
        let e = c.n_embd as i32;
        k.norm_quant.for_nq(c.n_embd)?.launch(
            (1, 1, 1),
            (512, 1, 1),
            0,
            st,
            &Args::new()
                .buf(&self.x)
                .buf(&m.out_norm)
                .buf(&self.xq_e)
                .buf(&self.xd_e)
                .buf(&self.xs_e)
                .i32(e)
                .f32(c.rms_eps),
        )?;
        let head = m.lm_head.as_ref().unwrap_or(&m.tok_emb);
        let head_fn = k.headq.uniform(head.qf.unwrap(), c.n_embd)?;
        head_fn.launch(
            (self.persistent_grid(head_fn, 256, (c.vocab + 7) / 8), 1, 1),
            (256, 1, 1),
            0,
            st,
            &Self::qp(Args::new(), head)
                .i32(Self::fid(head))
                .buf(&self.xq_e)
                .buf(&self.xd_e)
                .buf(&self.xs_e)
                .buf(&self.logits)
                .i32(c.vocab as i32)
                .i32(e),
        )?;
        k.argmax_partial.launch(
            (ARGMAX_BLOCKS, 1, 1),
            (256, 1, 1),
            0,
            st,
            &Args::new()
                .buf(&self.logits)
                .i32(c.vocab as i32)
                .buf(&self.pv)
                .buf(&self.pi),
        )?;
        k.argmax_final.launch(
            (1, 1, 1),
            (32, 1, 1),
            0,
            st,
            &Args::new()
                .buf(&self.pv)
                .buf(&self.pi)
                .i32(ARGMAX_BLOCKS as i32)
                .buf(&self.state)
                .buf(&self.ring)
                .i32(RING as i32),
        )?;
        Ok(())
    }

    fn enqueue_head(&self) -> Result<()> {
        let m = self.m;
        let c = &m.cfg;
        let k = &m.k;
        let st = &self.stream;
        let e = c.n_embd as i32;
        if m.quantized && !self.old_q8 {
            return self.enqueue_head_q();
        }
        k.final_norm.launch(
            (1, 1, 1),
            (256, 1, 1),
            0,
            st,
            &Args::new()
                .buf(&self.x)
                .buf(&m.out_norm)
                .buf(&self.h_final)
                .i32(e)
                .f32(c.rms_eps),
        )?;
        let head = m.lm_head.as_ref().unwrap_or(&m.tok_emb);
        let blocks = (m.dev.info.sm_count as u32) * 4;
        if m.fmt == Fmt::Q8_0 {
            k.head_q8.for_k(c.n_embd)?.launch(
                (blocks, 1, 1),
                (128, 1, 1),
                q8_smem(c.n_embd),
                st,
                &Args::new()
                    .buf(&head.w)
                    .buf(head.s.as_ref().unwrap())
                    .buf(&self.h_final)
                    .buf(&self.logits)
                    .i32(c.vocab as i32)
                    .i32(e),
            )?;
        } else {
            k.head.launch(
                (blocks, 1, 1),
                (128, 1, 1),
                (c.n_embd * 4) as u32,
                st,
                &Args::new()
                    .buf(&head.w)
                    .buf(&self.h_final)
                    .buf(&self.logits)
                    .i32(c.vocab as i32)
                    .i32(e),
            )?;
        }
        k.argmax_partial.launch(
            (ARGMAX_BLOCKS, 1, 1),
            (256, 1, 1),
            0,
            st,
            &Args::new()
                .buf(&self.logits)
                .i32(c.vocab as i32)
                .buf(&self.pv)
                .buf(&self.pi),
        )?;
        k.argmax_final.launch(
            (1, 1, 1),
            (32, 1, 1),
            0,
            st,
            &Args::new()
                .buf(&self.pv)
                .buf(&self.pi)
                .i32(ARGMAX_BLOCKS as i32)
                .buf(&self.state)
                .buf(&self.ring)
                .i32(RING as i32),
        )?;
        Ok(())
    }

    fn build_graphs(&mut self) -> Result<()> {
        // direct launches once (warms the lazily loaded kernels), then capture
        self.state.upload(&[0u8; 16])?;
        self.enqueue_layers()?;
        self.enqueue_head()?;
        self.stream.synchronize()?;
        self.layers_graph = self
            .stream
            .capture(|| self.enqueue_layers())?
            .instantiate()?;
        self.head_graph = self.stream.capture(|| self.enqueue_head())?.instantiate()?;
        self.reset()
    }

    pub fn reset(&mut self) -> Result<()> {
        self.state.upload(&[0u8; 16])?;
        if self.m.cfg.delta.is_some() {
            self.dn_s.zero(&self.stream)?;
            self.dn_conv.zero(&self.stream)?;
            self.stream.synchronize()?;
        }
        self.pos = 0;
        self.mtp_filled = 0;
        self.generated = 0;
        Ok(())
    }

    /// Multiply K-quant matrices with the fused dequantizing GEMMs in the prefill whatever the chunk size, or never (the default switches by chunk size, see `fused_max`; for A/B tests).
    pub fn set_fused(&mut self, on: bool) {
        self.fused_max = if on { [usize::MAX; 3] } else { [0; 3] };
    }

    /// The integer GEMMs of gemm_mmq.cu on or off (for A/B tests; stays off where the device cannot run them).
    pub fn set_mmq(&mut self, on: bool) {
        self.mmq = on && self.m.k.mmq_ok;
    }

    /// The fused SwiGLU + int8 quantization of the FFN's down-projection input on or off (for A/B tests).
    pub fn set_swiglu_q(&mut self, on: bool) {
        self.swiglu_q = on;
    }

    /// The fused GEMMs accumulate in fp16 (for A/B tests; ignored below compute capability 8).
    pub fn set_f16acc(&mut self, on: bool) {
        self.f16acc = on && self.m.dev.info.cc_major >= 8;
    }

    /// The Gated DeltaNet prompt scan with several state columns per thread (for A/B tests).
    pub fn set_scan_nc(&mut self, on: bool) {
        self.scan_nc = on;
    }

    /// The prompt attention on the Ampere kernels of flash_a80.cu (for A/B tests; stays off below compute capability 8).
    pub fn set_flash_a80(&mut self, on: bool) {
        self.flash_a80 = on && self.m.dev.info.cc_major >= 8;
    }

    pub fn max_ctx(&self) -> usize {
        self.max_ctx
    }

    /// Forget everything after position `pos` (the KV cache entries stay, later tokens overwrite them): prefix reuse for a new prompt that shares its first `pos` tokens with the cached one.
    pub fn truncate(&mut self, pos: usize) {
        assert!(
            pos <= self.pos,
            "cannot truncate beyond the cached positions"
        );
        assert!(
            pos == self.pos || self.can_truncate(),
            "a hybrid model's recurrent state cannot be rolled back"
        );
        self.pos = pos;
    }

    /// False for hybrid (Gated DeltaNet) models: their recurrent state summarises the whole sequence, so only an exact prefix of the cached tokens can be reused (see `reset`).
    pub fn can_truncate(&self) -> bool {
        self.m.cfg.delta.is_none()
    }

    fn set_token_pos(&self, token: u32, pos: u32) -> Result<()> {
        let mut b = [0u8; 8];
        b[..4].copy_from_slice(&token.to_le_bytes());
        b[4..].copy_from_slice(&pos.to_le_bytes());
        self.state.upload_at(0, &b)
    }

    /// Feed a prompt. The last token also runs the head, so `logits()` are valid afterwards and the device holds the first generated token.
    pub fn prefill(&mut self, tokens: &[u32]) -> Result<()> {
        assert!(
            !tokens.is_empty() && self.pos + tokens.len() <= self.max_ctx,
            "prompt does not fit the context"
        );
        let mtp = self.m.mtp.is_some();
        self.mtp_fill_pending(tokens[0])?;
        for (i, &t) in tokens.iter().enumerate() {
            self.set_token_pos(t, self.pos as u32)?;
            self.layers_graph.launch(&self.stream)?;
            if i + 1 == tokens.len() {
                self.head_graph.launch(&self.stream)?;
            }
            self.pos += 1;
            if mtp && i + 1 < tokens.len() {
                // the MTP head's entry for the position just processed: its hidden state is the residual stream the layers left in `x`, the next token is known
                self.mtp_step(tokens[i + 1], self.x.ptr(), self.pos - 1, false)?;
            }
        }
        self.stream.synchronize()?;
        self.generated = 1; // the head already produced the first new token
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn gemm(
        &self,
        f: Function,
        a: u64,
        b: &DeviceBuffer,
        c: u64,
        m: usize,
        n: usize,
        k: usize,
        ldc: usize,
    ) -> Result<()> {
        self.gemm_p(f, a, b.ptr(), c, m, n, k, ldc)
    }

    /// C (+)= A W^T for the weight matrix `g` ([n][k]): K-quant matrices go through the fused kernels that decode the packed weights on their way into shared memory; any other format is
    /// dequantized to fp16 first. `epi`: 0 fp16 out, 1 fp32 out, 2 add into fp32.
    #[allow(clippy::too_many_arguments)]
    fn gemm_w(
        &self,
        epi: usize,
        a: u64,
        g: &GpuMat,
        c: u64,
        m: usize,
        n: usize,
        k: usize,
        ldc: usize,
    ) -> Result<()> {
        if let Some((group, mode)) = self.fakeq[self.role.get() as usize] {
            self.m.k.fakeq_rows.launch((m as u32, 1, 1), (128, 1, 1), 0, &self.stream, &Args::new().ptr(a).i32(m as i32).i32(k as i32).i32(group).i32(mode))?;
        }
        let fmt_index = match g.qf {
            Some(QFmt::Q4_K) => Some(0),
            Some(QFmt::Q5_K) => Some(1),
            Some(QFmt::Q6_K) => Some(2),
            _ => None,
        };
        if g.is_host() && self.pipe.is_some() && g.qf.is_some() && self.pf.is_some() {
            return self.gemm_host(epi, a, g, c, m, n, k, ldc);
        }
        // Q4_K weights and a chunk long enough for the 128-row tiles: the integer GEMM on int8 activations
        if let Some(mf) = self.mmq_fmt_of(g, m, n, k, ldc) {
            if let Some((aq, ad, asum)) = self.quant_acts(a, m, k)? {
                return self.launch_mmq(mf, epi, g, (aq, ad, asum), c, m, n, k, ldc);
            }
        }
        // (a matrix in host memory goes through the dequantizing pass instead: the fused kernels read every weight once per tile of the output, which over PCIe is a multiple of the matrix)
        if let (Some(fi), true, 0) = (fmt_index, fmt_index.map_or(false, |f| m <= self.fused_max[f]) && !g.is_host(), k % 256) {
            let p = g.ptrs();
            let grid = (((n + 127) / 128) as u32, ((m + 127) / 128) as u32, 1);
            let table = if self.f16acc { &self.m.k.gemmq_h } else { &self.m.k.gemmq };
            return table[fi * 3 + epi].launch(
                grid,
                (256, 1, 1),
                0,
                &self.stream,
                &Args::new()
                    .ptr(a)
                    .ptr(p[0])
                    .ptr(p[1])
                    .ptr(p[2])
                    .ptr(p[3])
                    .ptr(c)
                    .i32(m as i32)
                    .i32(n as i32)
                    .i32(k as i32)
                    .i32(k as i32)
                    .i32(ldc as i32)
                    .f32(1.0),
            );
        }
        let kn = &self.m.k;
        // a large quantized matrix and the Ampere GEMM: the matrix is dequantized in slabs of rows into buffers of the prefill that are dead at this point, so that no scratch of its own is needed
        if self.plain_s80 && k % 32 == 0 && g.qf.is_some() && !g.is_host() && n * k * 2 > WSCRATCH_SMALL {
            if let Some((buf, cap)) = self.slab_scratch(c) {
                let rows_cap = cap / (2 * k) / 128 * 128;
                if rows_cap >= 128 {
                    let slabs = (n + rows_cap - 1) / rows_cap;
                    let rows_per = ((n + slabs - 1) / slabs + 127) / 128 * 128;
                    let esz = if epi == 0 { 2 } else { 4 };
                    let mut r0 = 0;
                    while r0 < n {
                        let rows = rows_per.min(n - r0);
                        self.dequant_rows(g, n, r0, rows, k, buf)?;
                        self.gemm_s80(epi, a, buf, c + (r0 * esz) as u64, m, rows, k, ldc)?;
                        r0 += rows;
                    }
                    return Ok(());
                }
            }
        }
        let w16 = self.wf16(g, n, k)?;
        if self.plain_s80 && k % 32 == 0 {
            return self.gemm_s80(epi, a, w16.ptr(), c, m, n, k, ldc);
        }
        let f = match epi {
            0 => kn.gemm_f16_out,
            1 => kn.gemm_f32_out,
            _ => kn.gemm_f32_add,
        };
        self.gemm(f, a, w16, c, m, n, k, ldc)
    }

    /// The format index (0 Q4_K, 1 Q5_K, 2 Q6_K) when `g` (n rows of k weights, output row stride ldc) goes through the integer GEMMs for a chunk of m tokens, else None.
    fn mmq_fmt_of(&self, g: &GpuMat, m: usize, n: usize, k: usize, ldc: usize) -> Option<usize> {
        let mf = match g.qf {
            Some(QFmt::Q4_K) => 0,
            Some(QFmt::Q5_K) => 1,
            Some(QFmt::Q6_K) => 2,
            _ => return None,
        };
        if self.mmq_skip >> self.role.get() & 1 != 0 && self.chunk_pos.get() >= self.mmq_skip_from {
            return None;
        }
        (self.mmq && !g.is_host() && m >= self.mmq_min_m && k % 256 == 0 && n % 8 == 0 && ldc % 2 == 0).then_some(mf)
    }

    /// C (+)= A W^T with the integer GEMM of gemm_mmq.cu on int8 activations that are already quantized (arrays q, d, s of `quant_acts`).
    #[allow(clippy::too_many_arguments)]
    fn launch_mmq(&self, mf: usize, epi: usize, g: &GpuMat, qa: (u64, u64, u64), c: u64, m: usize, n: usize, k: usize, ldc: usize) -> Result<()> {
        let p = g.ptrs();
        self.m.k.mmq[mf * 3 + epi.min(2)].launch(
            (((n + 127) / 128) as u32, ((m + 127) / 128) as u32, 1),
            (256, 1, 1),
            if mf == 2 { MMQ6_SMEM } else { MMQ_SMEM },
            &self.stream,
            &Args::new()
                .ptr(qa.0)
                .ptr(qa.1)
                .ptr(qa.2)
                .ptr(p[0])
                .ptr(p[1])
                .ptr(p[2])
                .ptr(p[3])
                .ptr(c)
                .i32(m as i32)
                .i32(n as i32)
                .i32(k as i32)
                .i32(ldc as i32),
        )
    }

    /// The down projection of a SwiGLU FFN for a prompt chunk: x += (silu(g) * u) W_down^T with g, u the gate and up outputs [m][f] in the prefill buffers. When the down matrix goes through the integer GEMMs
    /// the SwiGLU and the int8 quantization of its result happen in one pass (swiglu_quant_rows_kernel: the fp16 activations are never stored); otherwise the SwiGLU kernel writes them and `gemm_w` takes over.
    #[allow(clippy::too_many_arguments)]
    fn ffn_down(&self, pf: &PrefillBufs, wd: &GpuMat, li: usize, m: usize, f: usize, e: usize) -> Result<()> {
        let st = &self.stream;
        self.role.set(7);
        if self.swiglu_q && self.stats.is_none() {
            if let Some(mf) = self.mmq_fmt_of(wd, m, e, f, e) {
                if let Some(qa) = self.qa_buffers(m, f) {
                    self.m.k.swiglu_quant_rows.launch(
                        (m as u32, 1, 1),
                        (256, 1, 1),
                        0,
                        st,
                        &Args::new().ptr(pf.g_ptr()).ptr(pf.u_ptr()).ptr(qa.0).ptr(qa.1).ptr(qa.2).i32(m as i32).i32(f as i32),
                    )?;
                    return self.launch_mmq(mf, 2, wd, qa, pf.x.ptr(), m, e, f, e);
                }
            }
        }
        let n = (m * f) as u64;
        self.m.k.swiglu.launch(
            (((n + 255) / 256).min(4096) as u32, 1, 1),
            (256, 1, 1),
            0,
            st,
            &Args::new().ptr(pf.g_ptr()).ptr(pf.u_ptr()).buf(&pf.a).u64(n),
        )?;
        self.stat(li, 3, &pf.a, m, f)?;
        self.gemm_w(2, pf.a.ptr(), wd, pf.x.ptr(), m, e, f, e)
    }

    /// The int8 activation arrays of the integer GEMMs (q, d, s) for m tokens of k values, allocated on first use; None when they do not fit (the GEMMs then stay off) or the chunk is longer than they were made for.
    fn qa_buffers(&self, m: usize, k: usize) -> Option<(u64, u64, u64)> {
        let pf = self.pf.as_ref()?;
        if self.mmq_no_mem.get() {
            return None;
        }
        let qa = match pf.qa.get() {
            Some(b) => b,
            None => {
                let c = &self.m.cfg;
                let kmax = c.n_embd.max(c.q_dim()).max(c.n_ff).max(c.delta.as_ref().map_or(0, |d| d.value_dim()));
                let rows = (prefill_chunk_len().max(EVAL_ROWS) + 127) / 128 * 128;
                let made = (|| -> Result<QaBufs> {
                    Ok(QaBufs {
                        q: DeviceBuffer::alloc(rows * kmax)?,
                        d: DeviceBuffer::alloc(rows * kmax / 32 * 4)?,
                        s: DeviceBuffer::alloc(rows * kmax / 32 * 2)?,
                        rows,
                        kmax,
                    })
                })();
                match made {
                    Ok(b) => {
                        let _ = pf.qa.set(b);
                        pf.qa.get().expect("just set")
                    }
                    Err(e) => {
                        eprintln!("[gpu] the int8 activation buffers of the integer GEMMs ({:.0} MiB) do not fit ({e}): those GEMMs stay off", (rows * kmax + rows * kmax / 32 * 6) as f64 / (1u64 << 20) as f64);
                        self.mmq_no_mem.set(true);
                        return None;
                    }
                }
            }
        };
        if m > qa.rows || k > qa.kmax {
            return None;
        }
        Some((qa.q.ptr(), qa.d.ptr(), qa.s.ptr()))
    }

    /// Quantize the fp16 activations `a` [m][k] to int8 with a scale and a minimum operand per 32 values (the arrays gemm_mmq.cu reads) and return the device addresses of (q, d, s); None when the buffers cannot hold m x k.
    fn quant_acts(&self, a: u64, m: usize, k: usize) -> Result<Option<(u64, u64, u64)>> {
        let Some((q, d, s)) = self.qa_buffers(m, k) else {
            return Ok(None);
        };
        self.m.k.quant_rows_h.launch(
            (m as u32, 1, 1),
            (256, 1, 1),
            0,
            &self.stream,
            &Args::new().ptr(a).ptr(q).ptr(d).ptr(s).i32(m as i32).i32(k as i32),
        )?;
        Ok(Some((q, d, s)))
    }

    /// C (+)= A B^T with the Ampere kernels of gemm_s80.cu (A [m][k] and B [n][k] contiguous fp16): fp16 accumulation (flushed into fp32 after every k16 step) when `f16acc`, else fp32. `epi`: 0 fp16 out, 1 fp32 out,
    /// 2 add into fp32.
    #[allow(clippy::too_many_arguments)]
    fn gemm_s80(&self, epi: usize, a: u64, b: u64, c: u64, m: usize, n: usize, k: usize, ldc: usize) -> Result<()> {
        let kn = &self.m.k;
        let table = if self.f16acc { &kn.gemm_s80h } else { &kn.gemm_s80 };
        table[epi.min(2)].launch(
            (((n + 127) / 128) as u32, ((m + 127) / 128) as u32, 1),
            (256, 1, 1),
            S80_SMEM,
            &self.stream,
            &Args::new().ptr(a).ptr(b).ptr(c).i32(m as i32).i32(n as i32).i32(k as i32).i32(k as i32).i32(k as i32).i32(ldc as i32).f32(1.0),
        )
    }

    /// Where `gemm_w` may put dequantized rows of a weight matrix while it multiplies it: (device address, bytes) of a stretch of prefill buffers that is dead at this point. The gate and up outputs are
    /// produced from the normed residual stream and consumed by the SwiGLU right away, the SwiGLU output only lives until the down projection has read it: so the gate and up GEMMs may use the SwiGLU
    /// buffer, every other GEMM (attention and recurrent projections, the output projection, the down projection) both halves of `gu`.
    fn slab_scratch(&self, c: u64) -> Option<(u64, usize)> {
        let pf = self.pf.as_ref()?;
        if c == pf.g_ptr() || c == pf.u_ptr() {
            Some((pf.a.ptr(), pf.a.len()))
        } else {
            Some((pf.gu.ptr(), pf.gu.len()))
        }
    }

    /// Dequantize rows r0 .. r0 + rows of the quantized matrix `g` ([n_total][cols]) to fp16 at the device address `out`.
    fn dequant_rows(&self, g: &GpuMat, n_total: usize, r0: usize, rows: usize, cols: usize, out: u64) -> Result<()> {
        let qf = g.qf.unwrap();
        let (p, lens) = (g.ptrs(), g.plane_lens());
        let pp: [u64; 4] = std::array::from_fn(|i| if lens[i] == 0 { 0 } else { p[i] + (r0 * (lens[i] / n_total)) as u64 });
        let n8 = rows * cols / 8;
        let blocks = (((n8 + 255) / 256).min(16384)) as u32;
        let fast = match qf {
            QFmt::Q4_K => Some(0),
            QFmt::Q5_K => Some(1),
            QFmt::Q6_K => Some(2),
            _ => None,
        };
        if let (Some(fi), 0) = (fast, cols % 256) {
            self.m.k.dequantk[fi].launch(
                (blocks, 1, 1),
                (256, 1, 1),
                0,
                &self.stream,
                &Args::new().ptr(pp[0]).ptr(pp[1]).ptr(pp[2]).ptr(pp[3]).ptr(out).i32(rows as i32).i32(cols as i32),
            )
        } else {
            self.m.k.dequant_f16.launch(
                (blocks, 1, 1),
                (256, 1, 1),
                0,
                &self.stream,
                &Args::new().ptr(pp[0]).ptr(pp[1]).ptr(pp[2]).ptr(pp[3]).i32(qf as i32).ptr(out).i32(rows as i32).i32(cols as i32),
            )
        }
    }

    /// C (+)= A W^T for a matrix in host memory (see `gemm_w`): the rows of W are dequantized to fp16 in blocks on a second stream (the dequantizer reads the packed weights over PCIe at 8.5 GB/s, which takes
    /// longer than the tensor-core GEMM of the block), two blocks in flight; the GEMM of block b runs while block b + 1 is read. Same arithmetic as the unpipelined path: every output column is computed by the same
    /// kernel over the same K, so the result is bit-identical.
    #[allow(clippy::too_many_arguments)]
    fn gemm_host(&self, epi: usize, a: u64, g: &GpuMat, c: u64, m: usize, n: usize, k: usize, ldc: usize) -> Result<()> {
        let pf = self.pf.as_ref().unwrap();
        let pipe = self.pipe.as_ref().unwrap();
        let qf = g.qf.unwrap();
        let (p, lens) = (g.ptrs(), g.plane_lens());
        let kn = &self.m.k;
        let f = match epi {
            0 => kn.gemm_f16_out,
            1 => kn.gemm_f32_out,
            _ => kn.gemm_f32_add,
        };
        let fast = match qf {
            QFmt::Q4_K => Some(0),
            QFmt::Q5_K => Some(1),
            QFmt::Q6_K => Some(2),
            _ => None,
        };
        let esz = if epi == 0 { 2 } else { 4 };
        let blk_elems = pf.wblk[0].len() / 2;
        let rows_blk = ((blk_elems / k) / 128 * 128).max(128).min(n);
        let (mut r0, mut bi) = (0usize, 0usize);
        while r0 < n {
            let rows = rows_blk.min(n - r0);
            let slot = bi % 2;
            // the GEMM that last read this buffer (two blocks back, or in the previous matrix) must be done
            pipe.dq.wait_event(&pipe.free[slot])?;
            let pp: [u64; 4] = std::array::from_fn(|i| if lens[i] == 0 { 0 } else { p[i] + (r0 * (lens[i] / n)) as u64 });
            let n8 = rows * k / 8;
            // few blocks (grid-stride loop): the dequantizer is limited by PCIe, so a handful of warps per SM keep the link busy and leave the rest of every SM to the GEMM running next to it
            let blocks = ((n8 + 255) / 256).min(16384).min(self.m.dev.info.sm_count as usize * pipe_blocks_per_sm()) as u32;
            if let (Some(fi), 0) = (fast, k % 256) {
                self.m.k.dequantk[fi].launch(
                    (blocks, 1, 1),
                    (256, 1, 1),
                    0,
                    &pipe.dq,
                    &Args::new().ptr(pp[0]).ptr(pp[1]).ptr(pp[2]).ptr(pp[3]).buf(&pf.wblk[slot]).i32(rows as i32).i32(k as i32),
                )?;
            } else {
                self.m.k.dequant_f16.launch(
                    (blocks, 1, 1),
                    (256, 1, 1),
                    0,
                    &pipe.dq,
                    &Args::new().ptr(pp[0]).ptr(pp[1]).ptr(pp[2]).ptr(pp[3]).i32(qf as i32).buf(&pf.wblk[slot]).i32(rows as i32).i32(k as i32),
                )?;
            }
            pipe.ready[slot].record(&pipe.dq)?;
            self.stream.wait_event(&pipe.ready[slot])?;
            self.gemm_p(f, a, pf.wblk[slot].ptr(), c + (r0 * esz) as u64, m, rows, k, ldc)?;
            pipe.free[slot].record(&self.stream)?;
            r0 += rows;
            bi += 1;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn gemm_p(
        &self,
        f: Function,
        a: u64,
        b: u64,
        c: u64,
        m: usize,
        n: usize,
        k: usize,
        ldc: usize,
    ) -> Result<()> {
        let grid = (((n + 127) / 128) as u32, ((m + 127) / 128) as u32, 1);
        f.launch(
            grid,
            (256, 1, 1),
            0,
            &self.stream,
            &Args::new()
                .ptr(a)
                .ptr(b)
                .ptr(c)
                .i32(m as i32)
                .i32(n as i32)
                .i32(k as i32)
                .i32(k as i32)
                .i32(k as i32)
                .i32(ldc as i32)
                .f32(1.0),
        )
    }

    /// The fp16 weights of a matrix for the tensor-core GEMM: the matrix itself, or (quantized models) its dequantization in the scratch buffer.
    fn wf16<'a>(&'a self, g: &'a GpuMat, rows: usize, cols: usize) -> Result<&'a DeviceBuffer> {
        let Some(qf) = g.qf else { return Ok(&g.w) };
        let pf = self.pf.as_ref().unwrap();
        let scratch = if rows * cols * 2 <= WSCRATCH_SMALL {
            &pf.wscratch_small
        } else {
            match pf.wscratch.get() {
                Some(b) => b,
                None => {
                    let _ = pf.wscratch.set(DeviceBuffer::alloc(pf.wscratch_bytes)?);
                    pf.wscratch.get().expect("just set")
                }
            }
        };
        let p = g.ptrs();
        let n8 = rows * cols / 8;
        let blocks = (((n8 + 255) / 256).min(16384)) as u32;
        let fast = match qf {
            QFmt::Q4_K => Some(0),
            QFmt::Q5_K => Some(1),
            QFmt::Q6_K => Some(2),
            _ => None,
        };
        if let (Some(fi), 0) = (fast, cols % 256) {
            self.m.k.dequantk[fi].launch(
                (blocks, 1, 1),
                (256, 1, 1),
                0,
                &self.stream,
                &Args::new()
                    .ptr(p[0])
                    .ptr(p[1])
                    .ptr(p[2])
                    .ptr(p[3])
                    .buf(scratch)
                    .i32(rows as i32)
                    .i32(cols as i32),
            )?;
            return Ok(scratch);
        }
        self.m.k.dequant_f16.launch(
            (((n8 + 255) / 256).min(16384) as u32, 1, 1),
            (256, 1, 1),
            0,
            &self.stream,
            &Args::new()
                .ptr(p[0])
                .ptr(p[1])
                .ptr(p[2])
                .ptr(p[3])
                .i32(qf as i32)
                .buf(scratch)
                .i32(rows as i32)
                .i32(cols as i32),
        )?;
        Ok(scratch)
    }

    /// The dense models' prefill buffers (allocated on first use).
    fn ensure_pf_d(&mut self) -> Result<()> {
        let pfc = prefill_chunk_len(); // (the buffers hold one chunk: FORGE_PREFILL_CHUNK lowers their size too)
        let m = self.m;
        let c = &m.cfg;
        let blk = if self.pipe.is_some() { c.n_ff * c.n_embd / 8 * 2 } else { 16 };
        if self.pf.is_none() {
            let (e, qd, kvd, f) = (c.n_embd, c.q_dim(), c.kv_dim(), c.n_ff);
            self.pf = Some(PrefillBufs {
                tokens: DeviceBuffer::alloc(pfc * 4)?,
                x: DeviceBuffer::alloc(pfc * e * 4)?,
                h: DeviceBuffer::alloc(pfc * e * 2)?,
                qkv: DeviceBuffer::alloc(pfc * (qd + 2 * kvd) * 2)?,
                q: DeviceBuffer::alloc(pfc * qd * 2)?,
                o: DeviceBuffer::alloc(pfc * qd * 2)?,
                gu: DeviceBuffer::alloc(2 * pfc * f * 2)?,
                gu_half: (pfc * f * 2) as u64,
                a: DeviceBuffer::alloc(pfc * f * 2)?,
                wscratch_small: DeviceBuffer::alloc(WSCRATCH_SMALL)?,
                wscratch: std::sync::OnceLock::new(),
                wscratch_bytes: if m.quantized {
                    (f * e).max(qd * e).max(kvd * e) * 2
                } else {
                    16
                },
                wblk: [DeviceBuffer::alloc(blk)?, DeviceBuffer::alloc(blk)?],
                hy: None,
                qa: std::sync::OnceLock::new(),
            });
        }
        Ok(())
    }

    /// The prefill buffers of any model (allocated on first use).
    fn ensure_pf(&mut self) -> Result<()> {
        if self.m.cfg.delta.is_some() {
            self.ensure_pf_h()
        } else {
            self.ensure_pf_d()
        }
    }

    /// The hybrid models' prefill buffers (allocated on first use).
    fn ensure_pf_h(&mut self) -> Result<()> {
        let pfc = prefill_chunk_len();
        let m = self.m;
        let c = &m.cfg;
        let d = c.delta.as_ref().unwrap();
        let (e, qd, kvd, f) = (c.n_embd, c.q_dim(), c.kv_dim(), c.n_ff);
        let (q_rows, conv_dim, value_dim, n_v) = (c.q_rows(), d.conv_dim(), d.value_dim(), d.n_v);
        let stride = q_rows + 2 * kvd;
        let blk = if self.pipe.is_some() { f * e / 8 * 2 } else { 16 };
        if self.pf.is_none() {
            let biggest = (f * e)
                .max(q_rows * e)
                .max(kvd * e)
                .max(conv_dim * e)
                .max(value_dim * e);
            self.pf = Some(PrefillBufs {
                tokens: DeviceBuffer::alloc(pfc * 4)?,
                x: DeviceBuffer::alloc(pfc * e * 4)?,
                h: DeviceBuffer::alloc(pfc * e * 2)?,
                qkv: DeviceBuffer::alloc(pfc * stride * 2)?,
                q: DeviceBuffer::alloc(pfc * qd * 2)?,
                o: DeviceBuffer::alloc(pfc * qd * 2)?,
                gu: DeviceBuffer::alloc(2 * pfc * f * 2)?,
                gu_half: (pfc * f * 2) as u64,
                a: DeviceBuffer::alloc(pfc * f * 2)?,
                wscratch_small: DeviceBuffer::alloc(WSCRATCH_SMALL)?,
                wscratch: std::sync::OnceLock::new(),
                wscratch_bytes: biggest * 2,
                wblk: [DeviceBuffer::alloc(blk)?, DeviceBuffer::alloc(blk)?],
                hy: Some(HybridBufs {
                    mixed: DeviceBuffer::alloc(pfc * conv_dim * 4)?,
                    z: DeviceBuffer::alloc(pfc * value_dim * 4)?,
                    ab: DeviceBuffer::alloc(pfc * 2 * n_v * 4)?,
                    pq: DeviceBuffer::alloc(pfc * value_dim * 4)?,
                    pk: DeviceBuffer::alloc(pfc * value_dim * 4)?,
                    pv: DeviceBuffer::alloc(pfc * value_dim * 4)?,
                    pg: DeviceBuffer::alloc(pfc * n_v * 3 * 4)?,
                    so: DeviceBuffer::alloc(pfc * value_dim * 4)?,
                    y: DeviceBuffer::alloc(pfc * value_dim * 2)?,
                    mtp_cat: DeviceBuffer::alloc(if m.mtp.is_some() {
                        pfc * 2 * e * 2
                    } else {
                        16
                    })?,
                    mtp_z: DeviceBuffer::alloc(if m.mtp.is_some() {
                        pfc * e * 4
                    } else {
                        16
                    })?,
                    next_tok: DeviceBuffer::alloc(pfc * 4)?,
                }),
                qa: std::sync::OnceLock::new(),
            });
        }
        Ok(())
    }

    /// One chunk of a hybrid model (Qwen3.5) through the tensor-core path: GEMMs for the projections, the token-serial register scan for the DeltaNet recurrence, the CUDA-core attention for head width 256.
    fn prefill_chunk_h(
        &mut self,
        tokens: &[u32],
        pos0: usize,
        next_after: Option<u32>,
    ) -> Result<()> {
        if self.mrow_ok(tokens.len(), pos0) {
            // short chunk: the weights are streamed once for groups of up to 16 rows (mrow.rs); the residual streams of the rows end up in pf.x as after the GEMM path
            self.layers_m(tokens, pos0, false)?;
            if let Some(n) = self.mtp_fill_chunk(tokens, pos0, next_after)? {
                self.mtp_filled = pos0 + n;
            }
            return Ok(());
        }
        self.ensure_pf()?;
        let m = self.m;
        let c = &m.cfg;
        let k = &m.k;
        let st = &self.stream;
        let d = c.delta.as_ref().unwrap();
        let t = tokens.len();
        let (e, qd, kvd, f) = (c.n_embd, c.q_dim(), c.kv_dim(), c.n_ff);
        let (hd, nh, nkv) = (c.head_dim, c.n_head, c.n_head_kv);
        let (q_rows, conv_dim, value_dim, n_v) = (c.q_rows(), d.conv_dim(), d.value_dim(), d.n_v);
        let stride = q_rows + 2 * kvd;
        let pf = self.pf.as_ref().unwrap();
        let hy = pf.hy.as_ref().unwrap();
        let tok_bytes: Vec<u8> = tokens.iter().flat_map(|v| v.to_le_bytes()).collect();
        pf.tokens.upload(&tok_bytes)?;
        let start = self.stats.as_ref().map(|s| s.start_layer).unwrap_or(0);
        if start > 0 {
            // a calibration pass that starts inside the network: the residual stream comes from the caller
            let x = self
                .resid_in
                .take()
                .expect("a pass that starts inside the network needs its residual stream");
            let bytes: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();
            pf.x.upload(&bytes)?;
        } else if m.quantized {
            let p = m.tok_emb.ptrs();
            k.embed_q_batch.launch(
                (t as u32, 1, 1),
                (256, 1, 1),
                0,
                st,
                &Args::new()
                    .ptr(p[0])
                    .ptr(p[1])
                    .ptr(p[2])
                    .ptr(p[3])
                    .i32(m.tok_emb.qf.unwrap() as i32)
                    .buf(&pf.tokens)
                    .buf(&pf.x)
                    .i32(t as i32)
                    .i32(e as i32),
            )?;
        } else {
            k.embed_batch.launch(
                (t as u32, 1, 1),
                (256, 1, 1),
                0,
                st,
                &Args::new()
                    .buf(&m.tok_emb.w)
                    .buf(&pf.tokens)
                    .buf(&pf.x)
                    .i32(t as i32)
                    .i32(e as i32),
            )?;
        }
        let s_bytes = n_v * 128 * 128 * 4;
        let conv_bytes = n_v * (d.conv - 1) * 384 * 4;
        let stop = self
            .stats
            .as_ref()
            .map(|s| s.stop_layer)
            .unwrap_or(usize::MAX);
        for (li, l) in m.layers.iter().enumerate() {
            if li < start {
                continue;
            }
            if li >= stop {
                break;
            }
            k.rmsnorm_rows.launch(
                (t as u32, 1, 1),
                (256, 1, 1),
                0,
                st,
                &Args::new()
                    .buf(&pf.x)
                    .buf(&l.attn_norm)
                    .buf(&pf.h)
                    .i32(e as i32)
                    .f32(c.rms_eps),
            )?;
            self.stat(li, 0, &pf.h, t, e)?;
            if let Some(dl) = &l.delta {
                let slot = self.dn_slot[li];
                let conv_state = self.dn_conv.ptr() + (slot * conv_bytes) as u64;
                self.role.set(0);
                self.gemm_w(
                    1,
                    pf.h.ptr(),
                    &dl.w_qkv,
                    hy.mixed.ptr(),
                    t,
                    conv_dim,
                    e,
                    conv_dim,
                )?;
                self.role.set(1);
                self.gemm_w(
                    1,
                    pf.h.ptr(),
                    &dl.w_z,
                    hy.z.ptr(),
                    t,
                    value_dim,
                    e,
                    value_dim,
                )?;
                self.gemm_w(1, pf.h.ptr(), &dl.w_a, hy.ab.ptr(), t, n_v, e, 2 * n_v)?;
                self.gemm_w(
                    1,
                    pf.h.ptr(),
                    &dl.w_b,
                    hy.ab.ptr() + (n_v * 4) as u64,
                    t,
                    n_v,
                    e,
                    2 * n_v,
                )?;
                // 8 tokens per block (bit-identical to the one-token kernel; FORGE_SCAN_NC=0 keeps that one)
                let (prep_fn, prep_grid) = if self.scan_nc { (k.gdn_prep_tile, ((t + 7) / 8) as u32) } else { (k.gdn_prep, t as u32) };
                let mut prep_args = Args::new()
                    .buf(&hy.mixed)
                    .buf(&hy.ab)
                    .buf(&dl.conv)
                    .ptr(conv_state)
                    .buf(&dl.a_neg)
                    .buf(&dl.dt_bias)
                    .buf(&hy.pq)
                    .buf(&hy.pk)
                    .buf(&hy.pv)
                    .buf(&hy.pg);
                if self.scan_nc {
                    prep_args = prep_args.i32(t as i32);
                }
                prep_fn.launch(
                    (prep_grid, n_v as u32, 1),
                    (384, 1, 1),
                    0,
                    st,
                    &prep_args
                        .i32(d.n_k as i32)
                        .i32(n_v as i32)
                        .i32(d.tiled_v as i32)
                        .i32(d.conv as i32)
                        .i32(conv_dim as i32),
                )?;
                // spread each head over enough blocks to occupy the SMs: with 32 SMs or more 128 columns for 32 heads or more, 64 for 16, else 32; with fewer SMs always 32 columns (the RTX 4060 has 24 SMs: a block of
                // 1,024 threads fills one, so 32 heads ran in two waves; scan 1.77 -> 1.25 ms on the 9B, 1.74 -> 1.20 on the 4B, 0.84 -> 0.57 ms on the 0.8B)
                let sms = self.m.dev.info.sm_count as usize;
                let cpb: usize = std::env::var("FORGE_SCAN_CPB")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(if sms < 32 {
                        32
                    } else if n_v >= 32 {
                        128
                    } else if n_v >= 16 {
                        64
                    } else {
                        32
                    });
                // blocks of 128 threads with CPB / 16 columns per thread (gdn_scan_nc_*): the fewest blocks that still give every SM 2.5 of them (the RTX 4060: 64 blocks of 64 columns for 32 value heads,
                // 64 of 32 columns for 16 heads); FORGE_SCAN_NC=0 keeps the kernels of one column per thread above
                let (scan_fn, cpb) = if self.scan_nc && std::env::var("FORGE_SCAN_CPB").is_err() {
                    let want = (sms * 5 + 1) / 2;
                    let cpb = [64usize, 32, 16].into_iter().find(|&c| n_v * (128 / c) >= want).unwrap_or(16);
                    (k.gdn_scan_nc[match cpb { 16 => 0, 32 => 1, _ => 2 }], cpb)
                } else {
                    (
                        k.gdn_scan[match cpb {
                            32 => 0,
                            64 => 1,
                            _ => 2,
                        }],
                        cpb,
                    )
                };
                let scan_threads = if self.scan_nc && std::env::var("FORGE_SCAN_CPB").is_err() { 128 } else { cpb * 8 };
                scan_fn.launch(
                    (n_v as u32, (128 / cpb) as u32, 1),
                    (scan_threads as u32, 1, 1),
                    0,
                    st,
                    &Args::new()
                        .buf(&hy.pq)
                        .buf(&hy.pk)
                        .buf(&hy.pv)
                        .buf(&hy.pg)
                        .buf(&hy.mixed)
                        .ptr(conv_state)
                        .ptr(self.dn_s.ptr() + (slot * s_bytes) as u64)
                        .buf(&hy.so)
                        .i32(t as i32)
                        .i32(d.n_k as i32)
                        .i32(n_v as i32)
                        .i32(d.tiled_v as i32)
                        .i32(d.conv as i32)
                        .i32(conv_dim as i32),
                )?;
                k.gdn_post.launch(
                    (t as u32, n_v as u32, 1),
                    (128, 1, 1),
                    0,
                    st,
                    &Args::new()
                        .buf(&hy.so)
                        .buf(&hy.z)
                        .buf(&dl.norm)
                        .buf(&hy.y)
                        .i32(n_v as i32)
                        .f32(c.rms_eps),
                )?;
                self.stat(li, 1, &hy.y, t, value_dim)?;
                self.role.set(2);
                self.gemm_w(2, hy.y.ptr(), &dl.w_out, pf.x.ptr(), t, e, value_dim, e)?;
            } else {
                let kc = self.k_cache.ptr() + self.kv_off(li);
                let vc = self.v_cache.ptr() + self.kv_off(li);
                self.role.set(3);
                self.gemm_w(0, pf.h.ptr(), &l.wq, pf.qkv.ptr(), t, q_rows, e, stride)?;
                self.role.set(4);
                self.gemm_w(
                    0,
                    pf.h.ptr(),
                    &l.wk,
                    pf.qkv.ptr() + (q_rows * 2) as u64,
                    t,
                    kvd,
                    e,
                    stride,
                )?;
                self.gemm_w(
                    0,
                    pf.h.ptr(),
                    &l.wv,
                    pf.qkv.ptr() + ((q_rows + kvd) * 2) as u64,
                    t,
                    kvd,
                    e,
                    stride,
                )?;
                self.kv_pre(li, Some(pos0), self.state.ptr())?;
                k.qkv35_rope_batch.launch(
                    (t as u32, (nh + 2 * nkv) as u32, 1),
                    (256, 1, 1),
                    0,
                    st,
                    &Args::new()
                        .buf(&pf.qkv)
                        .buf(&pf.q)
                        .ptr(kc)
                        .ptr(vc)
                        .buf(&self.cos)
                        .buf(&self.sin)
                        .buf(l.q_norm.as_ref().unwrap())
                        .buf(l.k_norm.as_ref().unwrap())
                        .i32(pos0 as i32)
                        .i32(nh as i32)
                        .i32(nkv as i32)
                        .i32(hd as i32)
                        .i32(c.rotary_dim as i32)
                        .f32(c.rms_eps)
                        .i32(0),
                )?;
                let gate_ptr = pf.qkv.ptr() + (hd * 2) as u64;
                if std::env::var("FORGE_NO_FLASH").is_ok() {
                    k.attn_prefill_256g.launch(
                        (((t + 31) / 32) as u32, nh as u32, 1),
                        (128, 1, 1),
                        0,
                        st,
                        &Args::new()
                            .buf(&pf.q)
                            .ptr(kc)
                            .ptr(vc)
                            .buf(&pf.o)
                            .i32(t as i32)
                            .i32(pos0 as i32)
                            .i32(nh as i32)
                            .i32(nkv as i32)
                            .ptr(gate_ptr)
                            .i32(stride as i32)
                            .i32((2 * hd) as i32),
                    )?;
                } else {
                    // the Ampere kernel: warp pairs share S (grid z 1, 256 threads, 64 KB of dynamic shared memory); the old one gives each half of the 256 output dimensions its own block
                    let (fa, grid, block, smem) = if self.flash_a80 {
                        (k.flash_256g_a80, (((t + 63) / 64) as u32, nh as u32, 1), (256, 1, 1), FLASH_PAIR_SMEM)
                    } else {
                        (k.flash_256g, (((t + 63) / 64) as u32, nh as u32, 2), (128, 1, 1), 0)
                    };
                    fa.launch(
                        grid,
                        block,
                        smem,
                        st,
                        &Args::new()
                            .buf(&pf.q)
                            .ptr(kc)
                            .ptr(vc)
                            .buf(&pf.o)
                            .i32(t as i32)
                            .i32(pos0 as i32)
                            .i32(nh as i32)
                            .i32(nkv as i32)
                            .ptr(gate_ptr)
                            .i32(stride as i32)
                            .i32((2 * hd) as i32),
                    )?;
                }
                self.kv_post(li, Some(pos0), t, self.state.ptr())?;
                self.stat(li, 1, &pf.o, t, qd)?;
                self.role.set(5);
                self.gemm_w(2, pf.o.ptr(), &l.wo, pf.x.ptr(), t, e, qd, e)?;
            }
            k.rmsnorm_rows.launch(
                (t as u32, 1, 1),
                (256, 1, 1),
                0,
                st,
                &Args::new()
                    .buf(&pf.x)
                    .buf(&l.ffn_norm)
                    .buf(&pf.h)
                    .i32(e as i32)
                    .f32(c.rms_eps),
            )?;
            self.stat(li, 2, &pf.h, t, e)?;
            self.role.set(6);
            self.gemm_w(0, pf.h.ptr(), &l.w_gate, pf.g_ptr(), t, f, e, f)?;
            self.gemm_w(0, pf.h.ptr(), &l.w_up, pf.u_ptr(), t, f, e, f)?;
            self.ffn_down(pf, &l.w_down, li, t, f, e)?;
            self.role.set(8);
        }
        if self.resid_out.is_some() && self.stats.as_ref().map(|s| s.body_only).unwrap_or(false) {
            // (`eval_chunk_resid` asked for the residual stream where this pass stopped)
            self.stream.synchronize()?;
            self.resid_out = Some(pf.x.download_f32(t * e)?);
        }
        if let Some(s) = self.stats.as_mut() {
            s.tokens += t;
        }
        if let Some(n) = self.mtp_fill_chunk(tokens, pos0, next_after)? {
            self.mtp_filled = pos0 + n;
        }
        Ok(())
    }

    /// The MTP head's KV entries for the positions of a chunk whose next token is known (the chunk's last position when `next_after` is given): input (embedding of the next token, final-normed residual
    /// stream of the position, taken from pf.x), written through the head's fc / wk / wv. Returns the number of entries; the caller sets `mtp_filled`.
    fn mtp_fill_chunk(&self, tokens: &[u32], pos0: usize, next_after: Option<u32>) -> Result<Option<usize>> {
        let m = self.m;
        let c = &m.cfg;
        let k = &m.k;
        let st = &self.stream;
        let t = tokens.len();
        let (e, kvd) = (c.n_embd, c.kv_dim());
        let (hd, nh, nkv) = (c.head_dim, c.n_head, c.n_head_kv);
        let q_rows = c.q_rows();
        let stride = q_rows + 2 * kvd;
        let pf = self.pf.as_ref().unwrap();
        let hy = pf.hy.as_ref().unwrap();
        // (fp16 evaluation models have no quantized head matrices: no MTP entries then)
        if let Some(w) = m.mtp.as_ref().filter(|_| m.quantized) {
            // the MTP head's KV entries for the positions of this chunk whose next token is known: input (embedding of the next token, final-normed residual of the position)
            let n_fill = if next_after.is_some() { t } else { t - 1 };
            if n_fill > 0 {
                let mut nt: Vec<u32> = tokens[1..].to_vec();
                if let Some(a) = next_after {
                    nt.push(a);
                }
                nt.truncate(n_fill);
                let bytes: Vec<u8> = nt.iter().flat_map(|v| v.to_le_bytes()).collect();
                hy.next_tok.upload(&bytes)?;
                let p = m.tok_emb.ptrs();
                k.mtp_prep_batch.launch(
                    (n_fill as u32, 1, 1),
                    (256, 1, 1),
                    0,
                    st,
                    &Args::new()
                        .ptr(p[0])
                        .ptr(p[1])
                        .ptr(p[2])
                        .ptr(p[3])
                        .i32(m.tok_emb.qf.unwrap() as i32)
                        .buf(&hy.next_tok)
                        .buf(&pf.x)
                        .buf(&m.out_norm)
                        .buf(&w.norm_e)
                        .buf(&w.norm_h)
                        .buf(&hy.mtp_cat)
                        .i32(e as i32)
                        .f32(c.rms_eps),
                )?;
                self.gemm(
                    k.gemm_f32_out,
                    hy.mtp_cat.ptr(),
                    self.wf16(&w.fc, e, 2 * e)?,
                    hy.mtp_z.ptr(),
                    n_fill,
                    e,
                    2 * e,
                    e,
                )?;
                k.rmsnorm_rows.launch(
                    (n_fill as u32, 1, 1),
                    (256, 1, 1),
                    0,
                    st,
                    &Args::new()
                        .buf(&hy.mtp_z)
                        .buf(&w.layer.attn_norm)
                        .buf(&pf.h)
                        .i32(e as i32)
                        .f32(c.rms_eps),
                )?;
                self.gemm(
                    k.gemm_f16_out,
                    pf.h.ptr(),
                    self.wf16(&w.layer.wk, kvd, e)?,
                    pf.qkv.ptr() + (q_rows * 2) as u64,
                    n_fill,
                    kvd,
                    e,
                    stride,
                )?;
                self.gemm(
                    k.gemm_f16_out,
                    pf.h.ptr(),
                    self.wf16(&w.layer.wv, kvd, e)?,
                    pf.qkv.ptr() + ((q_rows + kvd) * 2) as u64,
                    n_fill,
                    kvd,
                    e,
                    stride,
                )?;
                k.qkv35_rope_batch.launch(
                    (n_fill as u32, (2 * nkv) as u32, 1),
                    (256, 1, 1),
                    0,
                    st,
                    &Args::new()
                        .buf(&pf.qkv)
                        .buf(&pf.q)
                        .buf(&self.mtp_k)
                        .buf(&self.mtp_v)
                        .buf(&self.cos)
                        .buf(&self.sin)
                        .buf(w.layer.q_norm.as_ref().unwrap())
                        .buf(w.layer.k_norm.as_ref().unwrap())
                        .i32(pos0 as i32)
                        .i32(nh as i32)
                        .i32(nkv as i32)
                        .i32(hd as i32)
                        .i32(c.rotary_dim as i32)
                        .f32(c.rms_eps)
                        .i32(nh as i32),
                )?;
                return Ok(Some(n_fill));
            }
        }
        Ok(None)
    }

    /// Process one chunk of `t` tokens (t <= PREFILL_CHUNK) starting at position `pos0` with the tensor-core GEMM path; fp16 weights only.
    fn prefill_chunk(
        &mut self,
        tokens: &[u32],
        pos0: usize,
        next_after: Option<u32>,
    ) -> Result<()> {
        self.chunk_pos.set(pos0);
        if self.m.cfg.delta.is_some() {
            return self.prefill_chunk_h(tokens, pos0, next_after);
        }
        if self.mrow_ok(tokens.len(), pos0) {
            // short chunk: the weights are streamed once for groups of up to 16 rows (mrow.rs); the residual streams of the rows end up in pf.x as after the GEMM path
            return self.layers_m(tokens, pos0, false);
        }
        self.ensure_pf()?;
        let m = self.m;
        let c = &m.cfg;
        let k = &m.k;
        let st = &self.stream;
        let t = tokens.len();
        let pf = self.pf.as_ref().unwrap();
        let tok_bytes: Vec<u8> = tokens.iter().flat_map(|v| v.to_le_bytes()).collect();
        pf.tokens.upload(&tok_bytes)?;
        let (e, qd, kvd, f) = (c.n_embd, c.q_dim(), c.kv_dim(), c.n_ff);
        let (hd, nh, nkv) = (c.head_dim, c.n_head, c.n_head_kv);
        let stride = qd + 2 * kvd;
        let start = self.stats.as_ref().map(|s| s.start_layer).unwrap_or(0);
        if start > 0 {
            // a calibration pass that starts inside the network: the residual stream comes from the caller
            let x = self
                .resid_in
                .take()
                .expect("a pass that starts inside the network needs its residual stream");
            let bytes: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();
            pf.x.upload(&bytes)?;
        } else if m.quantized {
            let p = m.tok_emb.ptrs();
            k.embed_q_batch.launch(
                (t as u32, 1, 1),
                (256, 1, 1),
                0,
                st,
                &Args::new()
                    .ptr(p[0])
                    .ptr(p[1])
                    .ptr(p[2])
                    .ptr(p[3])
                    .i32(m.tok_emb.qf.unwrap() as i32)
                    .buf(&pf.tokens)
                    .buf(&pf.x)
                    .i32(t as i32)
                    .i32(e as i32),
            )?;
        } else {
            k.embed_batch.launch(
                (t as u32, 1, 1),
                (256, 1, 1),
                0,
                st,
                &Args::new()
                    .buf(&m.tok_emb.w)
                    .buf(&pf.tokens)
                    .buf(&pf.x)
                    .i32(t as i32)
                    .i32(e as i32),
            )?;
        }
        let stop = self
            .stats
            .as_ref()
            .map(|s| s.stop_layer)
            .unwrap_or(usize::MAX);
        for (li, l) in m.layers.iter().enumerate() {
            if li < start {
                continue;
            }
            if li >= stop {
                break;
            }
            let kc = self.k_cache.ptr() + self.kv_off(li);
            let vc = self.v_cache.ptr() + self.kv_off(li);
            k.rmsnorm_rows.launch(
                (t as u32, 1, 1),
                (256, 1, 1),
                0,
                st,
                &Args::new()
                    .buf(&pf.x)
                    .buf(&l.attn_norm)
                    .buf(&pf.h)
                    .i32(e as i32)
                    .f32(c.rms_eps),
            )?;
            self.stat(li, 0, &pf.h, t, e)?;
            self.role.set(3);
            self.gemm_w(0, pf.h.ptr(), &l.wq, pf.qkv.ptr(), t, qd, e, stride)?;
            self.role.set(4);
            self.gemm_w(
                0,
                pf.h.ptr(),
                &l.wk,
                pf.qkv.ptr() + (qd * 2) as u64,
                t,
                kvd,
                e,
                stride,
            )?;
            self.gemm_w(
                0,
                pf.h.ptr(),
                &l.wv,
                pf.qkv.ptr() + ((qd + kvd) * 2) as u64,
                t,
                kvd,
                e,
                stride,
            )?;
            let (qn, kn) = match (&l.q_norm, &l.k_norm) {
                (Some(a), Some(b)) => (a.ptr(), b.ptr()),
                _ => (0, 0),
            };
            let (bq, bk, bv) = l.bias_ptrs();
            self.kv_pre(li, Some(pos0), self.state.ptr())?;
            k.qkv_rope.launch(
                (t as u32, (nh + 2 * nkv) as u32, 1),
                (64, 1, 1),
                0,
                st,
                &Args::new()
                    .buf(&pf.qkv)
                    .buf(&pf.q)
                    .ptr(kc)
                    .ptr(vc)
                    .buf(&self.cos)
                    .buf(&self.sin)
                    .ptr(qn)
                    .ptr(kn)
                    .i32(pos0 as i32)
                    .i32(nh as i32)
                    .i32(nkv as i32)
                    .i32(hd as i32)
                    .f32(c.rms_eps)
                    .ptr(bq)
                    .ptr(bk)
                    .ptr(bv),
            )?;
            let use_flash = std::env::var("FORGE_NO_FLASH").is_err();
            let (attn, qb) = if use_flash {
                if self.flash_a80 {
                    (k.flash_a80[(hd != 64) as usize], 64)
                } else {
                    (if hd == 64 { k.flash_64 } else { k.flash_128 }, 64)
                }
            } else if hd == 64 {
                (k.attn_prefill_64, 128)
            } else {
                (k.attn_prefill_128, 64)
            };
            attn.launch(
                (((t + qb - 1) / qb) as u32, nh as u32, 1),
                (128, 1, 1),
                0,
                st,
                &Args::new()
                    .buf(&pf.q)
                    .ptr(kc)
                    .ptr(vc)
                    .buf(&pf.o)
                    .i32(t as i32)
                    .i32(pos0 as i32)
                    .i32(nh as i32)
                    .i32(nkv as i32),
            )?;
            self.kv_post(li, Some(pos0), t, self.state.ptr())?;
            self.stat(li, 1, &pf.o, t, qd)?;
            self.role.set(5);
            self.gemm_w(2, pf.o.ptr(), &l.wo, pf.x.ptr(), t, e, qd, e)?;
            k.rmsnorm_rows.launch(
                (t as u32, 1, 1),
                (256, 1, 1),
                0,
                st,
                &Args::new()
                    .buf(&pf.x)
                    .buf(&l.ffn_norm)
                    .buf(&pf.h)
                    .i32(e as i32)
                    .f32(c.rms_eps),
            )?;
            self.stat(li, 2, &pf.h, t, e)?;
            self.role.set(6);
            self.gemm_w(0, pf.h.ptr(), &l.w_gate, pf.g_ptr(), t, f, e, f)?;
            self.gemm_w(0, pf.h.ptr(), &l.w_up, pf.u_ptr(), t, f, e, f)?;
            self.ffn_down(pf, &l.w_down, li, t, f, e)?;
            self.role.set(8);
        }
        if self.resid_out.is_some() && self.stats.as_ref().map(|s| s.body_only).unwrap_or(false) {
            // (`eval_chunk_resid` asked for the residual stream where this pass stopped)
            self.stream.synchronize()?;
            self.resid_out = Some(pf.x.download_f32(t * e)?);
        }
        if let Some(s) = self.stats.as_mut() {
            s.tokens += t;
        }
        Ok(())
    }

    /// Calibration statistics (see `ActStats`): add the sums of squares of the columns of `x` ([t][d], fp16) to the accumulator `which` of layer `li`.
    fn stat(&self, li: usize, which: usize, x: &DeviceBuffer, t: usize, d: usize) -> Result<()> {
        if let Some(s) = &self.stats {
            self.m.k.colsq.launch(
                (((d + 255) / 256) as u32, 1, 1),
                (256, 1, 1),
                0,
                &self.stream,
                &Args::new()
                    .buf(x)
                    .i32(t as i32)
                    .i32(d as i32)
                    .buf(&s.layers[li][which]),
            )?;
            if let Some((hl, _)) = &s.hess {
                if let Some(h) = &hl[li] {
                    self.gemm_tn_acc(x.ptr(), h[which].ptr(), t, d)?;
                }
            }
        }
        Ok(())
    }

    /// H += X^T X for X = [t][d] fp16 (the weight-gradient GEMM layout: both operands k-major), H fp32 [d][d].
    fn gemm_tn_acc(&self, x: u64, h: u64, t: usize, d: usize) -> Result<()> {
        assert!(
            t % 16 == 0 && d % 8 == 0,
            "second moments need whole 16-token chunks and widths that are multiples of 8"
        );
        let g = ((d + 127) / 128) as u32;
        self.m.k.gemm_tn_add.launch(
            (g, g, 1),
            (256, 1, 1),
            0,
            &self.stream,
            &Args::new()
                .ptr(x)
                .ptr(x)
                .ptr(h)
                .i32(d as i32)
                .i32(d as i32)
                .i32(t as i32)
                .i32(d as i32)
                .i32(d as i32)
                .i32(d as i32)
                .f32(1.0),
        )
    }

    /// Start collecting the calibration statistics of the importance of every weight column (the prefill paths accumulate; `take_stats` reads them out).
    pub fn enable_stats(&mut self, second_moments: bool) -> Result<()> {
        let n = self.m.layers.len();
        self.enable_stats_for(second_moments, 0..n, true)
    }

    /// Like `enable_stats`, with the second moments (the memory-hungry part: D x D floats per input) kept only for the layers in `layers` and for the head if `head`.
    pub fn enable_stats_for(
        &mut self,
        second_moments: bool,
        layers: std::ops::Range<usize>,
        head: bool,
    ) -> Result<()> {
        // the previous group's buffers (D x D floats per input) must be gone before the next group's are allocated, or two groups share the card
        self.stats = None;
        let c = &self.m.cfg;
        let (e, f) = (c.n_embd, c.n_ff);
        let zeros = |n: usize| -> Result<DeviceBuffer> {
            let b = DeviceBuffer::alloc(n * 4)?;
            b.upload(&vec![0u8; n * 4])?;
            Ok(b)
        };
        let mut stat_layers = Vec::new();
        for l in &self.m.layers {
            let out_dim = if l.delta.is_some() {
                c.delta.as_ref().unwrap().value_dim()
            } else {
                c.q_dim()
            };
            stat_layers.push([zeros(e)?, zeros(out_dim)?, zeros(e)?, zeros(f)?]);
        }
        let hess = if second_moments {
            let mut hl = Vec::new();
            for (i, l) in self.m.layers.iter().enumerate() {
                if !layers.contains(&i) {
                    hl.push(None);
                    continue;
                }
                let out_dim = if l.delta.is_some() {
                    c.delta.as_ref().unwrap().value_dim()
                } else {
                    c.q_dim()
                };
                hl.push(Some([
                    zeros(e * e)?,
                    zeros(out_dim * out_dim)?,
                    zeros(e * e)?,
                    zeros(f * f)?,
                ]));
            }
            Some((hl, if head { Some(zeros(e * e)?) } else { None }))
        } else {
            None
        };
        let body_only = second_moments && !head;
        let stop_layer = if body_only { layers.end } else { usize::MAX };
        self.stats = Some(ActStats {
            layers: stat_layers,
            head: zeros(e)?,
            tokens: 0,
            hess,
            hess_layers: layers,
            stop_layer,
            start_layer: 0,
            body_only,
        });
        Ok(())
    }

    /// Start the calibration passes at layer `l` from a residual stream given to `eval_chunk_resid` (a pass over layers l.. of a big model without running the layers before again).
    /// Call after `enable_stats_for`; `l` = the number of layers for a pass that only does the head.
    pub fn set_start_layer(&mut self, l: usize) -> Result<()> {
        let s = self
            .stats
            .as_mut()
            .ok_or_else(|| Error::unsupported("statistics were not enabled"))?;
        s.start_layer = l;
        Ok(())
    }

    /// One calibration chunk of a pass set up with `enable_stats_for` (+ `set_start_layer`): `input` is the residual stream at the start layer ([tokens][E], required when it is not 0), the result is
    /// the residual stream where a body-only pass stopped (to be the input of the next group of layers). Chunks of a body-only pass may have up to PREFILL_CHUNK tokens.
    pub fn eval_chunk_resid(
        &mut self,
        tokens: &[u32],
        input: Option<&[f32]>,
    ) -> Result<Option<Vec<f32>>> {
        let e = self.m.cfg.n_embd;
        let start = self.stats.as_ref().map(|s| s.start_layer).unwrap_or(0);
        if start > 0 {
            let x = input.ok_or_else(|| {
                Error::unsupported(
                    "a pass that starts inside the network needs its residual stream",
                )
            })?;
            assert_eq!(
                x.len(),
                tokens.len() * e,
                "residual stream of the wrong size"
            );
            self.resid_in = Some(x.to_vec());
        }
        self.resid_out = if self.stats.as_ref().map(|s| s.body_only).unwrap_or(false) {
            Some(Vec::new())
        } else {
            None
        };
        self.eval_chunk(tokens)?;
        Ok(self.resid_out.take())
    }

    /// The second moments E[x x^T] of the inputs (mean over the tokens seen) as (key, width, row-major matrix): "blk.N.attn_in" (input of the attention / DeltaNet projections), "blk.N.out_in"
    /// (input of the output projection), "blk.N.ffn_in" (gate / up), "blk.N.down_in" (down), and "head". Requires `enable_stats(true)`; call before `take_stats`.
    pub fn take_hessians(&mut self) -> Result<Vec<(String, usize, Vec<f32>)>> {
        self.stream.synchronize()?;
        let s = self
            .stats
            .as_ref()
            .ok_or_else(|| Error::unsupported("statistics were not enabled"))?;
        let (hl, hh) = s
            .hess
            .as_ref()
            .ok_or_else(|| Error::unsupported("second moments were not enabled"))?;
        let n = s.tokens.max(1) as f32;
        let c = &self.m.cfg;
        let (e, f) = (c.n_embd, c.n_ff);
        let read = |b: &DeviceBuffer| -> Result<Vec<f32>> {
            Ok(b.download_f32(b.len() / 4)?
                .into_iter()
                .map(|v| v / n)
                .collect())
        };
        let mut out = Vec::new();
        for (i, l) in self.m.layers.iter().enumerate() {
            let Some(h) = &hl[i] else { continue };
            let od = if l.delta.is_some() {
                c.delta.as_ref().unwrap().value_dim()
            } else {
                c.q_dim()
            };
            for (which, key, dim) in [
                (0, "attn_in", e),
                (1, "out_in", od),
                (2, "ffn_in", e),
                (3, "down_in", f),
            ] {
                out.push((format!("blk.{i}.{key}"), dim, read(&h[which])?));
            }
        }
        if let Some(hh) = hh {
            out.push(("head".to_string(), e, read(hh)?));
        }
        Ok(out)
    }

    /// The collected statistics as (GGUF tensor name, mean square of the activation per input column), and stop collecting.
    pub fn take_stats(&mut self) -> Result<Vec<(String, Vec<f32>)>> {
        self.stream.synchronize()?;
        let s = self
            .stats
            .take()
            .ok_or_else(|| Error::unsupported("statistics were not enabled"))?;
        let n = s.tokens.max(1) as f32;
        let read = |b: &DeviceBuffer| -> Result<Vec<f32>> {
            Ok(b.download_f32(b.len() / 4)?
                .into_iter()
                .map(|v| v / n)
                .collect())
        };
        let mut out = Vec::new();
        for (i, l) in self.m.layers.iter().enumerate() {
            let [a_in, o_in, f_in, d_in] = &s.layers[i];
            let (a_in, o_in, f_in, d_in) = (read(a_in)?, read(o_in)?, read(f_in)?, read(d_in)?);
            let b = format!("blk.{i}");
            if l.delta.is_some() {
                for nm in ["attn_qkv", "attn_gate", "ssm_alpha", "ssm_beta"] {
                    out.push((format!("{b}.{nm}.weight"), a_in.clone()));
                }
                out.push((format!("{b}.ssm_out.weight"), o_in));
            } else {
                for nm in ["attn_q", "attn_k", "attn_v"] {
                    out.push((format!("{b}.{nm}.weight"), a_in.clone()));
                }
                out.push((format!("{b}.attn_output.weight"), o_in));
            }
            out.push((format!("{b}.ffn_gate.weight"), f_in.clone()));
            out.push((format!("{b}.ffn_up.weight"), f_in));
            out.push((format!("{b}.ffn_down.weight"), d_in));
        }
        let head = read(&s.head)?;
        out.push(("output.weight".into(), head.clone()));
        out.push(("token_embd.weight".into(), head));
        Ok(out)
    }

    /// Prompt processing with the batched tensor-core path (chunks of PREFILL_CHUNK tokens). fp16 models only; afterwards the state is the same as after `prefill`.
    pub fn prefill_batched(&mut self, tokens: &[u32]) -> Result<()> {
        assert!(
            !tokens.is_empty() && self.pos + tokens.len() <= self.max_ctx,
            "prompt does not fit the context"
        );
        self.mtp_fill_pending(tokens[0])?;
        // a few tokens with the FFN on the CPU: groups of rows through the multi-row path (a pass over the host weights costs 0.3 s there, a GEMM chunk 1.1 s whatever its length)
        let rows_mode = self.m.host_ffn_bytes > 0 && self.mrow_max >= 2 && self.mrow_usable() && tokens.len() <= 3 * self.mrow_max;
        let mut done = 0;
        while done < tokens.len() {
            let n = if rows_mode { (tokens.len() - done).min(self.mrow_max) } else { (tokens.len() - done).min(prefill_chunk_len()) };
            let pos0 = self.pos;
            self.prefill_chunk(&tokens[done..done + n], pos0, tokens.get(done + n).copied())?;
            self.pos += n;
            done += n;
            if done == tokens.len() {
                // the residual stream of the last token goes through the decode head: final norm, logits, argmax
                let e = self.m.cfg.n_embd;
                let pf = self.pf.as_ref().unwrap();
                let src = pf.x.ptr() + ((n - 1) * e * 4) as u64;
                self.m.k.copy_row.launch(
                    (((e + 255) / 256) as u32, 1, 1),
                    (256, 1, 1),
                    0,
                    &self.stream,
                    &Args::new().ptr(src).buf(&self.x).i32(e as i32),
                )?;
                self.m.k.set_pos.launch(
                    (1, 1, 1),
                    (32, 1, 1),
                    0,
                    &self.stream,
                    &Args::new().buf(&self.state).u32((self.pos - 1) as u32),
                )?;
                self.head_graph.launch(&self.stream)?;
            }
        }
        self.stream.synchronize()?;
        self.generated = 1;
        Ok(())
    }

    /// Evaluation: run `tokens` (at most EVAL_ROWS) through the batched prefill path from the current position and return the logits after every one of them, [tokens.len()][vocab], computed
    /// with the same fp16-operand GEMMs as the prefill (the LM head is dequantized slice by slice). Used to measure the perplexity and the divergence of quantized weights.
    pub fn prefill_logits(&mut self, tokens: &[u32]) -> Result<Vec<f32>> {
        self.eval_chunk(tokens)?;
        let v = self.m.cfg.vocab;
        self.eval
            .as_ref()
            .unwrap()
            .logits
            .download_f32(tokens.len() * v)
    }

    /// `prefill_logits` without the copy of the logits to the host (they stay in the device buffer); calibration runs only need the side effects.
    pub fn eval_chunk(&mut self, tokens: &[u32]) -> Result<()> {
        let t = tokens.len();
        let body_only = self.stats.as_ref().map(|s| s.body_only).unwrap_or(false);
        assert!(
            t >= 1
                && t <= if body_only { PREFILL_CHUNK } else { EVAL_ROWS }
                && self.pos + t <= self.max_ctx,
            "bad evaluation chunk"
        );
        let pos0 = self.pos;
        if self.eval_decode && !body_only && !self.m.eval_only {
            return self.eval_chunk_decode(tokens);
        }
        if self.mrow_eval && !body_only && self.mrow_usable() {
            return self.eval_chunk_m(tokens);
        }
        self.prefill_chunk(tokens, pos0, None)?;
        self.pos += t;
        if body_only {
            // a calibration pass over the layers only (it may stop before the last one): the residual stream is not the model's output, no logits
            self.stream.synchronize()?;
            return Ok(());
        }
        let m = self.m;
        let c = &m.cfg;
        let (e, v) = (c.n_embd, c.vocab);
        if self.eval.is_none() {
            self.eval = Some(EvalBufs {
                logits: DeviceBuffer::alloc(EVAL_ROWS * v * 4)?,
                wslice: DeviceBuffer::alloc(if m.quantized { HEAD_SLICE * e * 2 } else { 16 })?,
            });
        }
        let ev = self.eval.as_ref().unwrap();
        let pf = self.pf.as_ref().unwrap();
        let st = &self.stream;
        m.k.rmsnorm_rows.launch(
            (t as u32, 1, 1),
            (256, 1, 1),
            0,
            st,
            &Args::new()
                .buf(&pf.x)
                .buf(&m.out_norm)
                .buf(&pf.h)
                .i32(e as i32)
                .f32(c.rms_eps),
        )?;
        if let Some(sv) = &self.stats {
            m.k.colsq.launch(
                (((e + 255) / 256) as u32, 1, 1),
                (256, 1, 1),
                0,
                st,
                &Args::new()
                    .buf(&pf.h)
                    .i32(t as i32)
                    .i32(e as i32)
                    .buf(&sv.head),
            )?;
            if let Some((_, Some(hh))) = &sv.hess {
                self.gemm_tn_acc(pf.h.ptr(), hh.ptr(), t, e)?;
            }
        }
        let head = m.lm_head.as_ref().unwrap_or(&m.tok_emb);
        let mut r0 = 0;
        while r0 < v {
            let rows = (v - r0).min(HEAD_SLICE);
            let out = ev.logits.ptr() + (r0 * 4) as u64;
            match head.qf {
                Some(qf) => {
                    let p = head.ptrs();
                    let n8 = rows * e / 8;
                    m.k.dequant_rows_f16.launch(
                        (((n8 + 255) / 256).min(16384) as u32, 1, 1),
                        (256, 1, 1),
                        0,
                        st,
                        &Args::new()
                            .ptr(p[0])
                            .ptr(p[1])
                            .ptr(p[2])
                            .ptr(p[3])
                            .i32(qf as i32)
                            .buf(&ev.wslice)
                            .i32(rows as i32)
                            .i32(e as i32)
                            .i32(r0 as i32),
                    )?;
                    self.gemm_p(
                        m.k.gemm_f32_out,
                        pf.h.ptr(),
                        ev.wslice.ptr(),
                        out,
                        t,
                        rows,
                        e,
                        v,
                    )?;
                }
                None => self.gemm_p(
                    m.k.gemm_f32_out,
                    pf.h.ptr(),
                    head.w.ptr() + (r0 * e * 2) as u64,
                    out,
                    t,
                    rows,
                    e,
                    v,
                )?,
            }
            r0 += rows;
        }
        st.synchronize()?;
        Ok(())
    }

    /// Feed one token and return the logits for the next one (used with CPU-side sampling; costs a synchronisation and a copy of the logits per token).
    pub fn step(&mut self, token: u32) -> Result<Vec<f32>> {
        assert!(
            self.pos < self.max_ctx,
            "context of {} tokens exhausted",
            self.max_ctx
        );
        self.set_token_pos(token, self.pos as u32)?;
        self.layers_graph.launch(&self.stream)?;
        self.head_graph.launch(&self.stream)?;
        self.stream.synchronize()?;
        self.pos += 1;
        self.logits()
    }

    pub fn logits(&self) -> Result<Vec<f32>> {
        self.logits.download_f32(self.m.cfg.vocab)
    }

    /// Feed one token and run the network up to the logits, without copying anything back: the next token is then read with `argmax_token` or chosen from `candidates`.
    pub fn advance(&mut self, token: u32) -> Result<()> {
        assert!(
            self.pos < self.max_ctx,
            "context of {} tokens exhausted",
            self.max_ctx
        );
        self.set_token_pos(token, self.pos as u32)?;
        self.layers_graph.launch(&self.stream)?;
        self.head_graph.launch(&self.stream)?;
        self.stream.synchronize()?;
        self.pos += 1;
        Ok(())
    }

    /// The arg-max token of the current logits (computed on the device by the head graph): 4 bytes instead of the whole vocabulary.
    pub fn argmax_token(&self) -> Result<u32> {
        let mut b = [0u8; 4];
        self.state.download_at(0, &mut b)?;
        Ok(u32::from_le_bytes(b))
    }

    fn src_ptr(&self, src: LogitsSrc) -> u64 {
        match src {
            LogitsSrc::Main => self.logits.ptr(),
            LogitsSrc::Verify(r) => {
                self.vb
                    .as_ref()
                    .expect("no verification step has run")
                    .logits
                    .ptr()
                    + (r * self.m.cfg.vocab * 4) as u64
            }
            LogitsSrc::Rows(r) => {
                self.mb
                    .as_ref()
                    .expect("no multi-row verification has run")
                    .vlogits_ptr()
                    + (r * self.m.cfg.vocab * 4) as u64
            }
        }
    }

    /// The full logits of a source (a vocabulary-sized copy: the fallback of the candidate path).
    pub fn logits_of(&self, src: LogitsSrc) -> Result<Vec<f32>> {
        match src {
            LogitsSrc::Main => self.logits(),
            LogitsSrc::Verify(r) => self.verify_logits(r),
            LogitsSrc::Rows(r) => self.rows_logits(r),
        }
    }

    /// The arg-max token of a source, computed on the device (4 bytes instead of the vocabulary).
    pub fn argmax_of(&self, src: LogitsSrc) -> Result<u32> {
        match src {
            LogitsSrc::Main => self.argmax_token(),
            LogitsSrc::Verify(r) => Ok(self.verify_argmax()?[r]),
            LogitsSrc::Rows(r) => Ok(self.rows_argmax()?[r]),
        }
    }

    /// First half of candidate extraction: the repetition penalty on the (distinct) recent tokens `ids` (at most 64), applied in place to the logits, and the maxima of the logits. Then
    /// `collect_candidates` can be called with different cutoffs. The penalty overwrites the penalised entries: use this or `logits_of` for a position, not both.
    pub fn prepare_candidates(&self, src: LogitsSrc, penalty: Option<(&[u32], f32)>) -> Result<()> {
        let k = &self.m.k;
        let st = &self.stream;
        let v = self.m.cfg.vocab as i32;
        let lp = self.src_ptr(src);
        if let Some((ids, pen)) = penalty {
            let ids = &ids[..ids.len().min(64)];
            if !ids.is_empty() && pen != 1.0 {
                let bytes: Vec<u8> = ids.iter().flat_map(|i| i.to_le_bytes()).collect();
                self.samp_ids.upload_at(0, &bytes)?;
                k.penalty.launch(
                    (1, 1, 1),
                    (64, 1, 1),
                    0,
                    st,
                    &Args::new()
                        .ptr(lp)
                        .buf(&self.samp_ids)
                        .i32(ids.len() as i32)
                        .f32(pen),
                )?;
            }
        }
        k.logit_max.launch(
            (SAMPLE_BLOCKS, 1, 1),
            (256, 1, 1),
            0,
            st,
            &Args::new()
                .ptr(lp)
                .i32(v)
                .buf(&self.samp_partial)
                .buf(&self.samp_out),
        )?;
        Ok(())
    }

    /// The tokens whose logit is within `cutoff` of the best one (after `prepare_candidates`).
    pub fn collect_candidates(&self, src: LogitsSrc, cutoff: f32) -> Result<Candidates> {
        let k = &self.m.k;
        let st = &self.stream;
        let v = self.m.cfg.vocab as i32;
        self.samp_out.upload_at(0, &[0u8; 4])?; // the candidate count (the previous call's kernels have finished)
        k.collect.launch(
            (SAMPLE_BLOCKS, 1, 1),
            (256, 1, 1),
            0,
            st,
            &Args::new()
                .ptr(self.src_ptr(src))
                .i32(v)
                .buf(&self.samp_partial)
                .i32(SAMPLE_BLOCKS as i32)
                .f32(cutoff)
                .i32(CAND_CAP as i32)
                .buf(&self.samp_out),
        )?;
        st.synchronize()?;
        let mut raw = vec![0u8; (2 + 2 * CAND_CAP) * 4];
        self.samp_out.download(&mut raw)?;
        let word = |i: usize| {
            u32::from_le_bytes([raw[4 * i], raw[4 * i + 1], raw[4 * i + 2], raw[4 * i + 3]])
        };
        let count = word(0) as usize;
        let n = count.min(CAND_CAP);
        let items = (0..n)
            .map(|j| (f32::from_bits(word(2 + CAND_CAP + j)), word(2 + j)))
            .collect();
        Ok(Candidates {
            max: f32::from_bits(word(1)),
            items,
            overflow: count > CAND_CAP,
        })
    }

    /// The candidates {logit >= max - cutoff} of the current logits, after a repetition penalty on the recent tokens.
    pub fn candidates(&self, cutoff: f32, penalty: Option<(&[u32], f32)>) -> Result<Candidates> {
        self.prepare_candidates(LogitsSrc::Main, penalty)?;
        self.collect_candidates(LogitsSrc::Main, cutoff)
    }

    /// Capacity of the candidate list.
    pub const fn candidate_capacity() -> usize {
        CAND_CAP
    }

    /// Greedy decoding entirely on the device: `n` tokens including the one the prefill produced. Returns the new token ids.
    pub fn decode_greedy(&mut self, n: usize) -> Result<Vec<u32>> {
        assert!(n >= 1 && n <= RING && self.pos + n <= self.max_ctx + 1);
        for _ in 1..n {
            self.layers_graph.launch(&self.stream)?;
            self.head_graph.launch(&self.stream)?;
            self.pos += 1;
        }
        self.stream.synchronize()?;
        let mut raw = vec![0u8; n * 4];
        self.ring.download_at(0, &mut raw)?;
        self.generated = n;
        Ok(raw
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    }

    /// Copy the recurrent state (hybrid models) so that a later request can continue from this position. Only meaningful for models with Gated DeltaNet layers.
    pub fn snapshot_state(&self) -> Result<StateSnapshot> {
        // page-locked host memory (a 9B snapshot is 50 MB, copied in 4 ms at the boundary of a message): device memory is the scarce resource of an 8 GB card, FORGE_SNAP_DEVICE=1 keeps them there
        let host = self.m.host_weight_bytes > 0 || std::env::var_os("FORGE_SNAP_DEVICE").is_none();
        let snap = StateSnapshot {
            pos: self.pos,
            s: SnapBuf::alloc(self.dn_s.len(), host)?,
            conv: SnapBuf::alloc(self.dn_conv.len(), host)?,
            hidden: SnapBuf::alloc(self.x.len(), host)?,
            mtp_filled: self.mtp_filled,
        };
        snap.hidden.save(&self.stream, &self.x)?;
        snap.s.save(&self.stream, &self.dn_s)?;
        snap.conv.save(&self.stream, &self.dn_conv)?;
        self.stream.synchronize()?;
        Ok(snap)
    }

    /// Return to the position of a snapshot. The KV cache of the attention layers must still hold the tokens of positions below it.
    pub fn restore_state(&mut self, snap: &StateSnapshot) -> Result<()> {
        snap.s.load(&self.stream, &self.dn_s)?;
        snap.conv.load(&self.stream, &self.dn_conv)?;
        snap.hidden.load(&self.stream, &self.x)?;
        self.stream.synchronize()?;
        self.pos = snap.pos;
        self.mtp_filled = snap.mtp_filled;
        self.generated = 0;
        Ok(())
    }

    /// The positions the session holds (the next token goes to this position).
    pub fn position(&self) -> usize {
        self.pos
    }

    /// Every region of the device that holds the KV entries of positions, in a fixed order: (address of position 0, bytes per position). Per attention layer K and V of the plain fp16 cache, or the four arrays of a
    /// compressed one (K values, K scales, V values, V scales), then the MTP head's K and V. Every region is position-major, so the entries of positions [0, n) are its first n rows.
    fn kv_regions(&self) -> Vec<(u64, usize)> {
        let c = &self.m.cfg;
        let n_attn = (0..c.n_layer).filter(|&l| !c.is_delta(l)).count();
        let mut v = Vec::new();
        for s in 0..n_attn {
            match &self.kvq {
                Some(q) => {
                    let p = q.ptrs(s, c, self.max_ctx);
                    let scales = c.n_head_kv * (c.head_dim / q.cfg.group) * 2;
                    v.push((p[0], c.n_head_kv * c.head_dim * q.cfg.bits_k as usize / 8));
                    v.push((p[1], scales));
                    v.push((p[2], c.n_head_kv * c.head_dim * q.cfg.bits_v as usize / 8));
                    v.push((p[3], scales));
                }
                None => {
                    let off = (s * self.max_ctx * c.kv_dim() * 2) as u64;
                    v.push((self.k_cache.ptr() + off, c.kv_dim() * 2));
                    v.push((self.v_cache.ptr() + off, c.kv_dim() * 2));
                }
            }
        }
        if self.m.mtp.is_some() {
            v.push((self.mtp_k.ptr(), c.kv_dim() * 2));
            v.push((self.mtp_v.ptr(), c.kv_dim() * 2));
        }
        v
    }

    /// Bytes of host memory `export_kv(n)` needs.
    pub fn kv_blob_bytes(&self, n: usize) -> usize {
        self.kv_regions().iter().map(|(_, row)| row * n).sum()
    }

    /// Copy the KV entries of positions [0, n) to page-locked host memory (n at most the positions the session holds). With the tokens and, for a hybrid model, a `StateSnapshot` of position n, this is
    /// everything a conversation needs to continue after the cache has been used for another one (see `adopt`). 12 GB/s over PCIe: 2.4 GB for 16,000 tokens of a dense 4B model take 0.2 s.
    pub fn export_kv(&self, n: usize) -> Result<KvBlob> {
        self.export_kv_with(n, None)
    }

    /// `export_kv` into a buffer of an earlier blob (`KvBlob::into_buffer`) when it is big enough: page-locking 1 GB of host memory takes 0.3 s and unlocking it as long, which the swap of two
    /// conversations of the prompt cache does not need to pay every time.
    pub fn export_kv_with(&self, n: usize, reuse: Option<PinnedBuffer>) -> Result<KvBlob> {
        if n > self.pos {
            return Err(Error::msg("export of positions the session does not hold"));
        }
        let regions = self.kv_regions();
        let total: usize = regions.iter().map(|(_, row)| row * n).sum();
        let buf = match reuse {
            Some(b) if b.len() >= total => b,
            _ => PinnedBuffer::alloc(total)?,
        };
        let mut layout = Vec::with_capacity(regions.len());
        let mut off = 0;
        for (addr, row) in &regions {
            self.stream.download_to_async(&buf, off, *addr, row * n)?;
            layout.push((off, *row));
            off += row * n;
        }
        self.stream.synchronize()?;
        Ok(KvBlob { n, layout, buf })
    }

    /// Write the first `m` positions of a blob into the cache (m at most the blob's); the position of the session does not change (see `adopt`).
    pub fn import_kv(&mut self, blob: &KvBlob, m: usize) -> Result<()> {
        if m > blob.n || m > self.max_ctx {
            return Err(Error::msg("import beyond the blob or the context"));
        }
        let regions = self.kv_regions();
        if regions.len() != blob.layout.len() || regions.iter().zip(&blob.layout).any(|((_, row), (_, brow))| row != brow) {
            return Err(Error::msg("the blob is of another model or cache format"));
        }
        for ((addr, row), (off, _)) in regions.iter().zip(&blob.layout) {
            self.stream.upload_from_async(*addr, &blob.buf, *off, row * m)?;
        }
        self.stream.synchronize()
    }

    /// Continue a conversation that was taken out of the cache: its first `n` positions (n at most the blob's: a dense model may take fewer than it was stored with) become the session's, and a hybrid model's
    /// recurrent state returns with them (`state`: a snapshot taken at position n; required for hybrid models, which cannot roll back, and must then be of exactly n positions).
    pub fn adopt(&mut self, blob: &KvBlob, n: usize, state: Option<&StateSnapshot>) -> Result<()> {
        self.import_kv(blob, n)?;
        // (the step state of the device: the output ring starts again, as after `reset`)
        self.state.upload(&[0u8; 16])?;
        match state {
            Some(s) => {
                if s.pos != n {
                    return Err(Error::msg("the snapshot is of another position"));
                }
                self.restore_state(s)
            }
            None => {
                if !self.can_truncate() {
                    return Err(Error::msg("a hybrid model's recurrent state must come back with the cache"));
                }
                self.pos = n;
                self.mtp_filled = 0;
                self.generated = 0;
                Ok(())
            }
        }
    }

    /// A single pending MTP slot (the last position of an earlier prefill, or of a restored snapshot) is filled once the token after it is known; its hidden state is still in `x`.
    fn mtp_fill_pending(&mut self, next_token: u32) -> Result<()> {
        if self.m.mtp.is_some() && self.pos > 0 && self.mtp_filled + 1 == self.pos {
            self.mtp_step(next_token, self.x.ptr(), self.pos - 1, false)?;
        }
        Ok(())
    }

    /// Is the MTP head usable for speculation from here: all positions below the last processed one have an entry.
    /// Limit the draft head to the first `n` tokens of the vocabulary (0 or more than the vocabulary: all of it).
    pub fn set_draft_rows(&mut self, n: usize) {
        self.draft_rows = if n == 0 {
            self.m.cfg.vocab
        } else {
            n.min(self.m.cfg.vocab)
        };
    }

    pub fn mtp_ready(&self) -> bool {
        self.m.mtp.is_some() && self.pos > 0 && self.mtp_filled + 1 == self.pos
    }

    /// One step of the MTP head at position `slot`: from the residual stream at `hidden` (device pointer, fp32 [E], before the final norm, of the main model's position `slot`) and the token that
    /// follows it, it writes its KV entry for `slot` and, when `want_draft`, returns its guess for the token after that (the arg-max). It reads the position from the step state, so it must run
    /// while no main step is in flight (every step here ends with a synchronisation).
    pub fn mtp_step(&mut self, token: u32, hidden: u64, slot: usize, want_draft: bool) -> Result<Option<u32>> {
        let norm = self.m.out_norm.ptr();
        self.mtp_step_ex(token, hidden, slot, want_draft, norm)
    }

    /// The next link of a chain of drafts: the head applied to ITS OWN output (the residual stream it left in `mtp_x`, normed by the head's final norm instead of the main model's) and the token it
    /// has just drafted (`token`), at head slot `slot` (the position of the main model whose token follows). The head's cache entry it writes is provisional: the entry the main model's hidden
    /// state would give replaces it once the verification has run (the caller restores `mtp_filled`). Returns the draft after `token`.
    pub fn mtp_chain_step(&mut self, token: u32, slot: usize) -> Result<u32> {
        let norm = self
            .m
            .mtp
            .as_ref()
            .ok_or_else(|| Error::unsupported("this model has no MTP head"))?
            .out_norm
            .ptr();
        let hidden = self.mtp_x.ptr();
        Ok(self.mtp_step_ex(token, hidden, slot, true, norm)?.expect("a draft was requested"))
    }

    fn mtp_step_ex(&mut self, token: u32, hidden: u64, slot: usize, want_draft: bool, hidden_norm: u64) -> Result<Option<u32>> {
        let m = self.m;
        let w = m
            .mtp
            .as_ref()
            .ok_or_else(|| Error::unsupported("this model has no MTP head"))?;
        let c = &m.cfg;
        let k = &m.k;
        let st = &self.stream;
        let (e, hd) = (c.n_embd as i32, c.head_dim as i32);
        self.set_token_pos(token, slot as u32)?;
        self.mtp_tok.upload_at(0, &token.to_le_bytes())?;
        let p = m.tok_emb.ptrs();
        k.mtp_prep.launch(
            (1, 1, 1),
            (512, 1, 1),
            0,
            st,
            &Args::new()
                .ptr(p[0])
                .ptr(p[1])
                .ptr(p[2])
                .ptr(p[3])
                .i32(Self::fid(&m.tok_emb))
                .buf(&self.mtp_tok)
                .ptr(hidden)
                .ptr(hidden_norm)
                .buf(&w.norm_e)
                .buf(&w.norm_h)
                .buf(&self.mtp_xq)
                .buf(&self.mtp_xd)
                .buf(&self.mtp_xs)
                .i32(e)
                .f32(c.rms_eps),
        )?;
        self.mv_out(
            &w.fc,
            &self.mtp_xq,
            &self.mtp_xd,
            &self.mtp_xs,
            self.mtp_x.ptr(),
            c.n_embd,
            2 * c.n_embd,
        )?;
        let l = &w.layer;
        let nq = k.norm_quant.for_nq(c.n_embd)?;
        nq.launch(
            (1, 1, 1),
            (512, 1, 1),
            0,
            st,
            &Args::new()
                .buf(&self.mtp_x)
                .buf(&l.attn_norm)
                .buf(&self.xq_e)
                .buf(&self.xd_e)
                .buf(&self.xs_e)
                .i32(e)
                .f32(c.rms_eps),
        )?;
        if want_draft {
            self.mv_out(
                &l.wq,
                &self.xq_e,
                &self.xd_e,
                &self.xs_e,
                self.h_qg.ptr(),
                c.q_rows(),
                c.n_embd,
            )?;
        }
        self.mv_out(
            &l.wk,
            &self.xq_e,
            &self.xd_e,
            &self.xs_e,
            self.k_raw.ptr(),
            c.kv_dim(),
            c.n_embd,
        )?;
        self.mv_out(
            &l.wv,
            &self.xq_e,
            &self.xd_e,
            &self.xs_e,
            self.h_v.ptr(),
            c.kv_dim(),
            c.n_embd,
        )?;
        k.qkv35_prep.launch(
            ((c.n_head + 2 * c.n_head_kv) as u32, 1, 1),
            (256, 1, 1),
            0,
            st,
            &Args::new()
                .buf(&self.h_qg)
                .buf(&self.k_raw)
                .buf(&self.h_v)
                .buf(l.q_norm.as_ref().unwrap())
                .buf(l.k_norm.as_ref().unwrap())
                .buf(&self.cos)
                .buf(&self.sin)
                .buf(&self.state)
                .buf(&self.q)
                .buf(&self.mtp_k)
                .buf(&self.mtp_v)
                .i32(c.n_head as i32)
                .i32(c.n_head_kv as i32)
                .i32(hd)
                .i32(c.rotary_dim as i32)
                .f32(c.rms_eps),
        )?;
        self.mtp_filled = self.mtp_filled.max(slot + 1);
        if !want_draft {
            return Ok(None);
        }
        let (kc, vc) = (self.mtp_k.ptr(), self.mtp_v.ptr());
        if self.attn_split {
            let ns_max = (self.max_ctx + AS_CH - 1) / AS_CH;
            k.attn_split_g.launch(
                (c.n_head as u32, ns_max as u32, 1),
                (256, 1, 1),
                0,
                st,
                &Args::new()
                    .buf(&self.q)
                    .ptr(kc)
                    .ptr(vc)
                    .buf(&self.as_m)
                    .buf(&self.as_l)
                    .buf(&self.as_o)
                    .buf(&self.state)
                    .i32(c.n_head as i32)
                    .i32(c.n_head_kv as i32)
                    .i32(hd)
                    .i32(ns_max as i32),
            )?;
            k.attn_merge_g.launch(
                (c.n_head as u32, 1, 1),
                (256, 1, 1),
                0,
                st,
                &Args::new()
                    .buf(&self.as_m)
                    .buf(&self.as_l)
                    .buf(&self.as_o)
                    .ptr(self.h_qg.ptr() + (c.head_dim * 4) as u64)
                    .buf(&self.xq_a)
                    .buf(&self.xd_a)
                    .buf(&self.xs_a)
                    .buf(&self.state)
                    .i32(ns_max as i32)
                    .i32(2 * hd),
            )?;
        } else {
            k.attention_g.launch(
                (c.n_head as u32, 1, 1),
                (256, 1, 1),
                self.attention_smem_h(),
                st,
                &Args::new()
                    .buf(&self.q)
                    .ptr(kc)
                    .ptr(vc)
                    .ptr(self.h_qg.ptr() + (c.head_dim * 4) as u64)
                    .buf(&self.xq_a)
                    .buf(&self.xd_a)
                    .buf(&self.xs_a)
                    .buf(&self.state)
                    .i32(c.n_head as i32)
                    .i32(c.n_head_kv as i32)
                    .i32(hd)
                    .i32(self.max_ctx as i32)
                    .i32(2 * hd),
            )?;
        }
        let rows_blocks = (c.n_embd + 7) / 8;
        let (wo_fn, wo_smem) = k.mvq_res.mv_fn(l.wo.qf.unwrap(), c.q_dim())?;
        wo_fn.launch(
            (self.persistent_grid(wo_fn, 256, rows_blocks), 1, 1),
            (256, 1, 1),
            wo_smem,
            st,
            &Self::qp(Args::new(), &l.wo)
                .i32(Self::fid(&l.wo))
                .buf(&self.xq_a)
                .buf(&self.xd_a)
                .buf(&self.xs_a)
                .buf(&self.mtp_x)
                .i32(e)
                .i32(c.q_dim() as i32),
        )?;
        nq.launch(
            (1, 1, 1),
            (512, 1, 1),
            0,
            st,
            &Args::new()
                .buf(&self.mtp_x)
                .buf(&l.ffn_norm)
                .buf(&self.xq_e)
                .buf(&self.xd_e)
                .buf(&self.xs_e)
                .i32(e)
                .f32(c.rms_eps),
        )?;
        let g = Self::qp(Self::qp(Args::new(), &l.w_gate), &l.w_up);
        let gu_fn = k.gateupq.uniform(
            Self::same(l.w_gate.qf.unwrap(), l.w_up.qf.unwrap())?,
            c.n_embd,
        )?;
        gu_fn.launch(
            (self.persistent_grid(gu_fn, 512, c.n_ff / 32), 1, 1),
            (512, 1, 1),
            0,
            st,
            &g.i32(Self::fid(&l.w_gate))
                .buf(&self.xq_e)
                .buf(&self.xd_e)
                .buf(&self.xs_e)
                .buf(&self.xq_f)
                .buf(&self.xd_f)
                .buf(&self.xs_f)
                .i32(e)
                .i32((c.n_ff / 32) as i32),
        )?;
        let (down_fn, down_smem) = k.mvq_res.mv_fn(l.w_down.qf.unwrap(), c.n_ff)?;
        down_fn.launch(
            (self.persistent_grid(down_fn, 256, rows_blocks), 1, 1),
            (256, 1, 1),
            down_smem,
            st,
            &Self::qp(Args::new(), &l.w_down)
                .i32(Self::fid(&l.w_down))
                .buf(&self.xq_f)
                .buf(&self.xd_f)
                .buf(&self.xs_f)
                .buf(&self.mtp_x)
                .i32(e)
                .i32(c.n_ff as i32),
        )?;
        nq.launch(
            (1, 1, 1),
            (512, 1, 1),
            0,
            st,
            &Args::new()
                .buf(&self.mtp_x)
                .buf(&w.out_norm)
                .buf(&self.xq_e)
                .buf(&self.xd_e)
                .buf(&self.xs_e)
                .i32(e)
                .f32(c.rms_eps),
        )?;
        let head = m.lm_head.as_ref().unwrap_or(&m.tok_emb);
        self.mv_out(
            head,
            &self.xq_e,
            &self.xd_e,
            &self.xs_e,
            self.logits.ptr(),
            self.draft_rows,
            c.n_embd,
        )?;
        k.argmax_rows_p.launch(
            (ARGMAX_BLOCKS, 1, 1),
            (256, 1, 1),
            0,
            st,
            &Args::new()
                .buf(&self.logits)
                .i32(self.draft_rows as i32)
                .buf(&self.pv)
                .buf(&self.pi),
        )?;
        k.argmax_rows_f.launch(
            (1, 1, 1),
            (32, 1, 1),
            0,
            st,
            &Args::new()
                .buf(&self.pv)
                .buf(&self.pi)
                .i32(ARGMAX_BLOCKS as i32)
                .buf(&self.mtp_out),
        )?;
        st.synchronize()?;
        let mut b = [0u8; 4];
        self.mtp_out.download_at(0, &mut b)?;
        Ok(Some(u32::from_le_bytes(b)))
    }

    /// Device pointer of the residual stream of row `row` (0 or 1) of the last two-token step (the input of the MTP head for that position).
    pub fn verify_hidden(&self, row: usize) -> u64 {
        self.vb
            .as_ref()
            .expect("no verification step has run")
            .x
            .ptr()
            + (row * self.m.cfg.n_embd * 4) as u64
    }

    /// Device pointer of the residual stream left by the last single-token step.
    pub fn hidden(&self) -> u64 {
        self.x.ptr()
    }

    /// A quantized matrix times the two activation vectors of a verification step (streamed once): out[t * ldo + r] = dot (or += when `res`).
    #[allow(clippy::too_many_arguments)]
    fn mv2(
        &self,
        w: &GpuMat,
        res: bool,
        xq: &DeviceBuffer,
        xd: &DeviceBuffer,
        xs: &DeviceBuffer,
        out: u64,
        ldo: usize,
        rows: usize,
        k: usize,
    ) -> Result<()> {
        let f =
            w.qf.ok_or_else(|| Error::unsupported("two-token decode needs quantized weights"))?;
        let func = if res {
            self.m.k.mvt2_res[f as usize]
        } else {
            self.m.k.mvt2_out[f as usize]
        };
        if k % 128 != 0 {
            return Err(Error::unsupported(format!(
                "two-token decode needs rows that are a multiple of 128 ({k})"
            )));
        }
        let smem = (2 * (k + k / 32 * 8)) as u32;
        let (threads, grid) = self.best_launch(func, smem, rows);
        func.launch(
            (grid, 1, 1),
            (threads, 1, 1),
            smem,
            &self.stream,
            &Self::qp(Args::new(), w)
                .buf(xq)
                .buf(xd)
                .buf(xs)
                .ptr(out)
                .i32(ldo as i32)
                .i32(rows as i32)
                .i32(k as i32),
        )
    }

    fn alloc_verify(&self) -> Result<VerifyBufs> {
        let c = &self.m.cfg;
        let d = c.delta.as_ref().ok_or_else(|| {
            Error::unsupported("two-token decode is implemented for hybrid models")
        })?;
        let (e, f, v) = (c.n_embd, c.n_ff, c.vocab);
        let act_a = c.q_dim().max(d.value_dim());
        let ns_max = (self.max_ctx + AS_CH - 1) / AS_CH;
        Ok(VerifyBufs {
            tokens: DeviceBuffer::alloc(8)?,
            x: DeviceBuffer::alloc(2 * e * 4)?,
            xq_e: DeviceBuffer::alloc(2 * e)?,
            xd_e: DeviceBuffer::alloc(2 * e / 32 * 4)?,
            xs_e: DeviceBuffer::alloc(2 * e / 32 * 4)?,
            xq_a: DeviceBuffer::alloc(2 * act_a)?,
            xd_a: DeviceBuffer::alloc(2 * act_a / 32 * 4)?,
            xs_a: DeviceBuffer::alloc(2 * act_a / 32 * 4)?,
            xq_f: DeviceBuffer::alloc(2 * f)?,
            xd_f: DeviceBuffer::alloc(2 * f / 32 * 4)?,
            xs_f: DeviceBuffer::alloc(2 * f / 32 * 4)?,
            mixed: DeviceBuffer::alloc(2 * d.conv_dim() * 4)?,
            z: DeviceBuffer::alloc(2 * d.value_dim() * 4)?,
            ab: DeviceBuffer::alloc(2 * 2 * d.n_v * 4)?,
            qg: DeviceBuffer::alloc(2 * c.q_rows() * 4)?,
            k_raw: DeviceBuffer::alloc(2 * c.kv_dim() * 4)?,
            v_raw: DeviceBuffer::alloc(2 * c.kv_dim() * 4)?,
            q: DeviceBuffer::alloc(2 * c.q_dim() * 4)?,
            g: DeviceBuffer::alloc(2 * f * 4)?,
            u: DeviceBuffer::alloc(2 * f * 4)?,
            logits: DeviceBuffer::alloc(2 * v * 4)?,
            argmax: DeviceBuffer::alloc(8)?,
            as_m: DeviceBuffer::alloc(2 * c.n_head * ns_max * 4)?,
            as_l: DeviceBuffer::alloc(2 * c.n_head * ns_max * 4)?,
            as_o: DeviceBuffer::alloc(2 * c.n_head * ns_max * 256 * 4)?,
            s_side: DeviceBuffer::alloc(self.dn_s.len())?,
            conv_side: DeviceBuffer::alloc(self.dn_conv.len())?,
            graph: None,
        })
    }

    /// The two-token step: tokens `vb.tokens[0]` and `vb.tokens[1]` at the positions `state.pos` and `state.pos + 1`, all layers, the final norm and the head for both (logits in `vb.logits`).
    fn enqueue_verify2(&self) -> Result<()> {
        let m = self.m;
        let c = &m.cfg;
        let k = &m.k;
        let st = &self.stream;
        let vb = self.vb.as_ref().unwrap();
        let d = c.delta.as_ref().unwrap();
        let (e, hd) = (c.n_embd as i32, c.head_dim as i32);
        k.embed_q_multi.launch(
            (((c.n_embd + 255) / 256) as u32, 2, 1),
            (256, 1, 1),
            0,
            st,
            &Self::qp(Args::new(), &m.tok_emb)
                .i32(Self::fid(&m.tok_emb))
                .buf(&vb.tokens)
                .buf(&vb.x)
                .i32(e),
        )?;
        let nq = k.norm_quant.for_nq(c.n_embd)?;
        let s_bytes = d.n_v * 128 * 128 * 4;
        let conv_bytes = d.n_v * (d.conv - 1) * 384 * 4;
        let (cd, vd, qr, kd, f) = (d.conv_dim(), d.value_dim(), c.q_rows(), c.kv_dim(), c.n_ff);
        let ns_max = (self.max_ctx + AS_CH - 1) / AS_CH;
        for (li, l) in m.layers.iter().enumerate() {
            nq.launch(
                (2, 1, 1),
                (512, 1, 1),
                0,
                st,
                &Args::new()
                    .buf(&vb.x)
                    .buf(&l.attn_norm)
                    .buf(&vb.xq_e)
                    .buf(&vb.xd_e)
                    .buf(&vb.xs_e)
                    .i32(e)
                    .f32(c.rms_eps),
            )?;
            let (mix_k, mix_wo) = if let Some(dl) = &l.delta {
                let slot = self.dn_slot[li];
                self.mv2(
                    &dl.w_qkv,
                    false,
                    &vb.xq_e,
                    &vb.xd_e,
                    &vb.xs_e,
                    vb.mixed.ptr(),
                    cd,
                    cd,
                    c.n_embd,
                )?;
                self.mv2(
                    &dl.w_z,
                    false,
                    &vb.xq_e,
                    &vb.xd_e,
                    &vb.xs_e,
                    vb.z.ptr(),
                    vd,
                    vd,
                    c.n_embd,
                )?;
                self.mv2(
                    &dl.w_a,
                    false,
                    &vb.xq_e,
                    &vb.xd_e,
                    &vb.xs_e,
                    vb.ab.ptr(),
                    2 * d.n_v,
                    d.n_v,
                    c.n_embd,
                )?;
                self.mv2(
                    &dl.w_b,
                    false,
                    &vb.xq_e,
                    &vb.xd_e,
                    &vb.xs_e,
                    vb.ab.ptr() + (d.n_v * 4) as u64,
                    2 * d.n_v,
                    d.n_v,
                    c.n_embd,
                )?;
                k.gdn_step2.launch(
                    (d.n_v as u32, 1, 1),
                    (1024, 1, 1),
                    0,
                    st,
                    &Args::new()
                        .buf(&vb.mixed)
                        .buf(&vb.z)
                        .buf(&vb.ab)
                        .buf(&dl.conv)
                        .buf(&dl.a_neg)
                        .buf(&dl.dt_bias)
                        .buf(&dl.norm)
                        .ptr(self.dn_conv.ptr() + (slot * conv_bytes) as u64)
                        .ptr(self.dn_s.ptr() + (slot * s_bytes) as u64)
                        .ptr(vb.conv_side.ptr() + (slot * conv_bytes) as u64)
                        .ptr(vb.s_side.ptr() + (slot * s_bytes) as u64)
                        .buf(&vb.xq_a)
                        .buf(&vb.xd_a)
                        .buf(&vb.xs_a)
                        .i32(d.n_k as i32)
                        .i32(d.n_v as i32)
                        .i32(d.tiled_v as i32)
                        .i32(d.conv as i32)
                        .f32(c.rms_eps),
                )?;
                (vd, &dl.w_out)
            } else {
                let kc = self.k_cache.ptr() + self.kv_off(li);
                let vc = self.v_cache.ptr() + self.kv_off(li);
                self.mv2(
                    &l.wq,
                    false,
                    &vb.xq_e,
                    &vb.xd_e,
                    &vb.xs_e,
                    vb.qg.ptr(),
                    qr,
                    qr,
                    c.n_embd,
                )?;
                self.mv2(
                    &l.wk,
                    false,
                    &vb.xq_e,
                    &vb.xd_e,
                    &vb.xs_e,
                    vb.k_raw.ptr(),
                    kd,
                    kd,
                    c.n_embd,
                )?;
                self.mv2(
                    &l.wv,
                    false,
                    &vb.xq_e,
                    &vb.xd_e,
                    &vb.xs_e,
                    vb.v_raw.ptr(),
                    kd,
                    kd,
                    c.n_embd,
                )?;
                if !self.kv_fused() {
                    self.kv_pre(li, None, self.state.ptr())?;
                }
                k.qkv35_prep.launch(
                    ((c.n_head + 2 * c.n_head_kv) as u32, 2, 1),
                    (256, 1, 1),
                    0,
                    st,
                    &Args::new()
                        .buf(&vb.qg)
                        .buf(&vb.k_raw)
                        .buf(&vb.v_raw)
                        .buf(l.q_norm.as_ref().unwrap())
                        .buf(l.k_norm.as_ref().unwrap())
                        .buf(&self.cos)
                        .buf(&self.sin)
                        .buf(&self.state)
                        .buf(&vb.q)
                        .ptr(kc)
                        .ptr(vc)
                        .i32(c.n_head as i32)
                        .i32(c.n_head_kv as i32)
                        .i32(hd)
                        .i32(c.rotary_dim as i32)
                        .f32(c.rms_eps),
                )?;
                let gate = vb.qg.ptr() + (c.head_dim * 4) as u64;
                if self.attn_split {
                    if self.kv_fused() {
                        self.launch_split_fused(true, li, 2, vb.q.ptr(), kc, vc, vb.as_m.ptr(), vb.as_l.ptr(), vb.as_o.ptr(), self.state.ptr(), ns_max)?;
                    } else {
                        self.launch_split_plain(true, 2, vb.q.ptr(), kc, vc, vb.as_m.ptr(), vb.as_l.ptr(), vb.as_o.ptr(), self.state.ptr(), ns_max)?;
                    }
                    k.attn_merge_g.launch(
                        (c.n_head as u32, 2, 1),
                        (256, 1, 1),
                        0,
                        st,
                        &Args::new()
                            .buf(&vb.as_m)
                            .buf(&vb.as_l)
                            .buf(&vb.as_o)
                            .ptr(gate)
                            .buf(&vb.xq_a)
                            .buf(&vb.xd_a)
                            .buf(&vb.xs_a)
                            .buf(&self.state)
                            .i32(ns_max as i32)
                            .i32(2 * hd),
                    )?;
                } else {
                    k.attention_g.launch(
                        (c.n_head as u32, 2, 1),
                        (256, 1, 1),
                        self.attention_smem_h(),
                        st,
                        &Args::new()
                            .buf(&vb.q)
                            .ptr(kc)
                            .ptr(vc)
                            .ptr(gate)
                            .buf(&vb.xq_a)
                            .buf(&vb.xd_a)
                            .buf(&vb.xs_a)
                            .buf(&self.state)
                            .i32(c.n_head as i32)
                            .i32(c.n_head_kv as i32)
                            .i32(hd)
                            .i32(self.max_ctx as i32)
                            .i32(2 * hd),
                    )?;
                }
                self.kv_post(li, None, 2, self.state.ptr())?;
                (c.q_dim(), &l.wo)
            };
            self.mv2(
                mix_wo,
                true,
                &vb.xq_a,
                &vb.xd_a,
                &vb.xs_a,
                vb.x.ptr(),
                c.n_embd,
                c.n_embd,
                mix_k,
            )?;
            nq.launch(
                (2, 1, 1),
                (512, 1, 1),
                0,
                st,
                &Args::new()
                    .buf(&vb.x)
                    .buf(&l.ffn_norm)
                    .buf(&vb.xq_e)
                    .buf(&vb.xd_e)
                    .buf(&vb.xs_e)
                    .i32(e)
                    .f32(c.rms_eps),
            )?;
            if self.cpu_ffn_layer(li) {
                self.enqueue_cpu_ffn(li, 2, &vb.xq_e, &vb.xd_e, &vb.xs_e, &vb.x)?;
                continue;
            }
            self.mv2(
                &l.w_gate,
                false,
                &vb.xq_e,
                &vb.xd_e,
                &vb.xs_e,
                vb.g.ptr(),
                f,
                f,
                c.n_embd,
            )?;
            self.mv2(
                &l.w_up,
                false,
                &vb.xq_e,
                &vb.xd_e,
                &vb.xs_e,
                vb.u.ptr(),
                f,
                f,
                c.n_embd,
            )?;
            k.swiglu_quant_t.launch(
                (((f / 32 + 7) / 8) as u32, 2, 1),
                (256, 1, 1),
                0,
                st,
                &Args::new()
                    .buf(&vb.g)
                    .buf(&vb.u)
                    .buf(&vb.xq_f)
                    .buf(&vb.xd_f)
                    .buf(&vb.xs_f)
                    .i32(f as i32),
            )?;
            self.mv2(
                &l.w_down,
                true,
                &vb.xq_f,
                &vb.xd_f,
                &vb.xs_f,
                vb.x.ptr(),
                c.n_embd,
                c.n_embd,
                f,
            )?;
        }
        // final norm + head for both tokens
        let nq_e = k.norm_quant.for_nq(c.n_embd)?;
        nq_e.launch(
            (2, 1, 1),
            (512, 1, 1),
            0,
            st,
            &Args::new()
                .buf(&vb.x)
                .buf(&m.out_norm)
                .buf(&vb.xq_e)
                .buf(&vb.xd_e)
                .buf(&vb.xs_e)
                .i32(e)
                .f32(c.rms_eps),
        )?;
        let head = m.lm_head.as_ref().unwrap_or(&m.tok_emb);
        self.mv2(
            head,
            false,
            &vb.xq_e,
            &vb.xd_e,
            &vb.xs_e,
            vb.logits.ptr(),
            c.vocab,
            c.vocab,
            c.n_embd,
        )?;
        k.argmax_rows_p.launch(
            (ARGMAX_BLOCKS, 2, 1),
            (256, 1, 1),
            0,
            st,
            &Args::new()
                .buf(&vb.logits)
                .i32(c.vocab as i32)
                .buf(&self.pv)
                .buf(&self.pi),
        )?;
        k.argmax_rows_f.launch(
            (2, 1, 1),
            (32, 1, 1),
            0,
            st,
            &Args::new()
                .buf(&self.pv)
                .buf(&self.pi)
                .i32(ARGMAX_BLOCKS as i32)
                .buf(&vb.argmax),
        )?;
        Ok(())
    }

    /// Run two tokens of the sequence at once (hybrid models): `u0` at the current position and `u1` after it. Afterwards `verify_logits(0 / 1)` are the logits after each token,
    /// the position has advanced by two and the recurrent state is the one after `u1`; if the second token turns out to be wrong, `rollback_verify` returns to the position after `u0`.
    pub fn verify2(&mut self, u0: u32, u1: u32) -> Result<()> {
        assert!(
            self.pos + 2 <= self.max_ctx,
            "context of {} tokens exhausted",
            self.max_ctx
        );
        if self.vb.is_none() {
            let vb = self.alloc_verify()?;
            self.vb = Some(vb);
        }
        let mut bytes = [0u8; 8];
        bytes[..4].copy_from_slice(&u0.to_le_bytes());
        bytes[4..].copy_from_slice(&u1.to_le_bytes());
        self.vb.as_ref().unwrap().tokens.upload(&bytes)?;
        self.set_token_pos(u0, self.pos as u32)?;
        if self.vb.as_ref().unwrap().graph.is_none() {
            // a direct run first (loads the lazily loaded kernels), then capture; the direct run advances the recurrent state, so it is saved and put back
            let saved = self.snapshot_state()?;
            self.enqueue_verify2()?;
            self.stream.synchronize()?;
            self.restore_state(&saved)?;
            let g = self
                .stream
                .capture(|| self.enqueue_verify2())?
                .instantiate()?;
            self.vb.as_mut().unwrap().graph = Some(g);
        }
        self.vb
            .as_ref()
            .unwrap()
            .graph
            .as_ref()
            .unwrap()
            .launch(&self.stream)?;
        self.stream.synchronize()?;
        self.pos += 2;
        Ok(())
    }

    /// Return to the position after the first token of the last `verify2` (the second token was rejected). The residual stream of the last processed position (`hidden`) is the first row's.
    pub fn rollback_verify(&mut self) -> Result<()> {
        let vb = self.vb.as_ref().expect("no verification step has run");
        self.stream
            .copy_d2d(self.dn_s.ptr(), vb.s_side.ptr(), self.dn_s.len())?;
        self.stream
            .copy_d2d(self.dn_conv.ptr(), vb.conv_side.ptr(), self.dn_conv.len())?;
        self.stream
            .copy_d2d(self.x.ptr(), vb.x.ptr(), self.x.len())?;
        self.stream.synchronize()?;
        self.pos -= 1;
        Ok(())
    }

    /// Keep both tokens of the last `verify2` (the second one was right): the residual stream of the last processed position (`hidden`) becomes the second row's.
    pub fn commit_verify(&mut self) -> Result<()> {
        let vb = self.vb.as_ref().expect("no verification step has run");
        self.stream
            .copy_d2d(self.x.ptr(), vb.x.ptr() + self.x.len() as u64, self.x.len())?;
        self.stream.synchronize()?;
        Ok(())
    }

    /// The logits after token `which` (0 or 1) of the last `verify2`, and the arg-max tokens of both.
    pub fn verify_logits(&self, which: usize) -> Result<Vec<f32>> {
        let vb = self.vb.as_ref().expect("no verification step has run");
        let v = self.m.cfg.vocab;
        let mut raw = vec![0u8; v * 4];
        vb.logits.download_at(which * v * 4, &mut raw)?;
        Ok(raw
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect())
    }

    pub fn verify_argmax(&self) -> Result<[u32; 2]> {
        let vb = self.vb.as_ref().expect("no verification step has run");
        let mut raw = [0u8; 8];
        vb.argmax.download(&mut raw)?;
        Ok([
            u32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]),
            u32::from_le_bytes([raw[4], raw[5], raw[6], raw[7]]),
        ])
    }

    pub fn stream(&self) -> &Stream {
        &self.stream
    }
}

fn placeholder_graph(stream: &Stream) -> Result<GraphExec> {
    // an empty captured graph, replaced once the real ones are built
    stream.capture(|| Ok(()))?.instantiate()
}

#[cfg(test)]
mod kvq_tests {
    use super::*;
    use forge_core::half::{f16_to_f32, f32_to_f16};

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn f(&mut self) -> f32 {
            (self.next() >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
        }
    }

    /// The arithmetic of kv_quant_kernel on the CPU: (store bytes, fp16 scales) of fp16 rows `vals` ([rows][hd], bits).
    fn cpu_quant(vals: &[u16], hd: usize, group: usize, bits: u32) -> (Vec<u8>, Vec<u16>) {
        let rows = vals.len() / hd;
        let ng = hd / group;
        let qmax = if bits == 8 { 127.0f32 } else { 7.0 };
        let rb = hd * bits as usize / 8;
        let mut store = vec![0u8; rows * rb];
        let mut scales = vec![0u16; rows * ng];
        for r in 0..rows {
            for g in 0..ng {
                let sl: Vec<f32> = vals[r * hd + g * group..r * hd + (g + 1) * group].iter().map(|&h| f16_to_f32(h)).collect();
                let amax = sl.iter().fold(0f32, |a, v| a.max(v.abs()));
                let ds = f16_to_f32(f32_to_f16(amax / qmax));
                let id = if ds != 0.0 { 1.0 / ds } else { 0.0 };
                scales[r * ng + g] = f32_to_f16(ds);
                for (e, v) in sl.iter().enumerate() {
                    let q = (v * id).round_ties_even().clamp(-qmax, qmax) as i32;
                    let idx = g * group + e;
                    if bits == 8 {
                        store[r * rb + idx] = q as i8 as u8;
                    } else {
                        let nib = ((q + 8) & 15) as u8;
                        store[r * rb + idx / 2] |= if idx % 2 == 0 { nib } else { nib << 4 };
                    }
                }
            }
        }
        (store, scales)
    }

    /// The values kv_dequant_kernel must produce from a store: fp16 bits.
    fn cpu_dequant(store: &[u8], scales: &[u16], hd: usize, group: usize, bits: u32) -> Vec<u16> {
        let ng = hd / group;
        let rb = hd * bits as usize / 8;
        let rows = scales.len() / ng;
        let mut out = vec![0u16; rows * hd];
        for r in 0..rows {
            for idx in 0..hd {
                let s = f16_to_f32(scales[r * ng + idx / group]);
                let q = if bits == 8 {
                    store[r * rb + idx] as i8 as i32
                } else {
                    let b = store[r * rb + idx / 2];
                    (if idx % 2 == 0 { b & 15 } else { b >> 4 }) as i32 - 8
                };
                out[r * hd + idx] = f32_to_f16(q as f32 * s);
            }
        }
        out
    }

    fn bytes_of(v: &[u16]) -> Vec<u8> {
        v.iter().flat_map(|h| h.to_le_bytes()).collect()
    }

    fn halves_of(b: &[u8]) -> Vec<u16> {
        b.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect()
    }

    /// rows with very different magnitudes, one outlier per row, one all-zero row
    fn make_rows(rng: &mut Rng, rows: usize, hd: usize, zero_row: usize) -> Vec<u16> {
        let mut v = vec![0u16; rows * hd];
        for r in 0..rows {
            let mag = (2.0f32).powi((rng.next() % 9) as i32 - 4);
            for i in 0..hd {
                let x = if r == zero_row { 0.0 } else { rng.f() * mag };
                v[r * hd + i] = f32_to_f16(x);
            }
            if r != zero_row {
                v[r * hd + (rng.next() as usize % hd)] = f32_to_f16(rng.f() * mag * 8.0);
            }
        }
        v
    }

    #[test]
    #[ignore = "needs a GPU"]
    fn kv_kernels_match_cpu_model() {
        let dev = Device::open(0).unwrap();
        let (image, _) = forge_kernels::image_for(dev.info.cc_major, dev.info.cc_minor).unwrap();
        let module = Module::load(image).unwrap();
        let quant = module.function("kv_quant_kernel").unwrap();
        let dequant = module.function("kv_dequant_kernel").unwrap();
        let stream = Stream::new().unwrap();
        let state = DeviceBuffer::alloc(16).unwrap();
        let mut rng = Rng(0x9E3779B97F4A7C15);
        let (n_kv, t) = (4usize, 37usize);
        let mut checked = 0;
        for hd in [64usize, 128, 256] {
            for group in [32usize, 64, 128, 256] {
                if group > hd {
                    continue;
                }
                for (bk, bv) in [(8u32, 8u32), (8, 4), (4, 8), (4, 4)] {
                    let rows = t * n_kv;
                    let kin = make_rows(&mut rng, rows, hd, 3);
                    let vin = make_rows(&mut rng, rows, hd, 5);
                    let (kq_ref, ks_ref) = cpu_quant(&kin, hd, group, bk);
                    let (vq_ref, vs_ref) = cpu_quant(&vin, hd, group, bv);
                    let ng = hd / group;
                    let (kq_len, vq_len) = (rows * hd * bk as usize / 8, rows * hd * bv as usize / 8);
                    let (ik, iv) = (DeviceBuffer::alloc(rows * hd * 2).unwrap(), DeviceBuffer::alloc(rows * hd * 2).unwrap());
                    ik.upload(&bytes_of(&kin)).unwrap();
                    iv.upload(&bytes_of(&vin)).unwrap();
                    let (qk, qv) = (DeviceBuffer::alloc(kq_len).unwrap(), DeviceBuffer::alloc(vq_len).unwrap());
                    let (sk, sv) = (DeviceBuffer::alloc(rows * ng * 2).unwrap(), DeviceBuffer::alloc(rows * ng * 2).unwrap());
                    qk.upload(&vec![0xAAu8; kq_len]).unwrap();
                    qv.upload(&vec![0xAAu8; vq_len]).unwrap();
                    // positions [0, 20) with a fixed start, then [20, 37) with the start taken from the step state (pos = 20)
                    let mut st = [0u8; 16];
                    st[4..8].copy_from_slice(&20u32.to_le_bytes());
                    state.upload(&st).unwrap();
                    let launch_quant = |pos0: i32, n_new: usize| {
                        quant
                            .launch(
                                (((n_new * n_kv + 7) / 8).max(1) as u32, 2, 1),
                                (256, 1, 1),
                                0,
                                &stream,
                                &Args::new()
                                    .buf(&ik)
                                    .buf(&qk)
                                    .buf(&sk)
                                    .buf(&iv)
                                    .buf(&qv)
                                    .buf(&sv)
                                    .buf(&state)
                                    .i32(pos0)
                                    .i32(n_new as i32)
                                    .i32(n_kv as i32)
                                    .i32(hd as i32)
                                    .i32(group as i32)
                                    .i32(bk as i32)
                                    .i32(bv as i32),
                            )
                            .unwrap();
                    };
                    launch_quant(0, 20);
                    launch_quant(-1, t - 20);
                    stream.synchronize().unwrap();
                    let mut got_k = vec![0u8; kq_len];
                    qk.download(&mut got_k).unwrap();
                    let mut got_ks = vec![0u8; rows * ng * 2];
                    sk.download(&mut got_ks).unwrap();
                    let mut got_v = vec![0u8; vq_len];
                    qv.download(&mut got_v).unwrap();
                    let mut got_vs = vec![0u8; rows * ng * 2];
                    sv.download(&mut got_vs).unwrap();
                    // the quantizer divides with the fast-math reciprocal: a value on a rounding boundary may land one step off, a scale one fp16 ulp off, both rarely
                    let (gs_k, gs_v) = (halves_of(&got_ks), halves_of(&got_vs));
                    let bad_scale = gs_k.iter().zip(&ks_ref).filter(|(a, b)| a != b).count() + gs_v.iter().zip(&vs_ref).filter(|(a, b)| a != b).count();
                    assert!(bad_scale * 500 <= 2 * rows * ng, "hd {hd} group {group} k{bk}v{bv}: {bad_scale} scales differ");
                    let bad_k = got_k.iter().zip(&kq_ref).filter(|(a, b)| a != b).count();
                    let bad_v = got_v.iter().zip(&vq_ref).filter(|(a, b)| a != b).count();
                    assert!(bad_k * 200 <= got_k.len() && bad_v * 200 <= got_v.len(), "hd {hd} group {group} k{bk}v{bv}: {bad_k}/{} K bytes, {bad_v}/{} V bytes differ", got_k.len(), got_v.len());
                    // dequantization of the reference store: bit exact (33 positions, then the count from the step state)
                    let (ok, ov) = (DeviceBuffer::alloc(rows * hd * 2).unwrap(), DeviceBuffer::alloc(rows * hd * 2).unwrap());
                    ok.upload(&vec![0xFFu8; rows * hd * 2]).unwrap();
                    ov.upload(&vec![0xFFu8; rows * hd * 2]).unwrap();
                    qk.upload(&kq_ref).unwrap();
                    qv.upload(&vq_ref).unwrap();
                    sk.upload(&bytes_of(&ks_ref)).unwrap();
                    sv.upload(&bytes_of(&vs_ref)).unwrap();
                    let n_deq = 33usize;
                    dequant
                        .launch(
                            (((n_deq * n_kv + 7) / 8).max(1) as u32, 2, 1),
                            (256, 1, 1),
                            0,
                            &stream,
                            &Args::new()
                                .buf(&qk)
                                .buf(&sk)
                                .buf(&ok)
                                .buf(&qv)
                                .buf(&sv)
                                .buf(&ov)
                                .buf(&state)
                                .i32(n_deq as i32)
                                .i32(n_kv as i32)
                                .i32(hd as i32)
                                .i32(group as i32)
                                .i32(bk as i32)
                                .i32(bv as i32),
                        )
                        .unwrap();
                    stream.synchronize().unwrap();
                    let want_k = cpu_dequant(&kq_ref, &ks_ref, hd, group, bk);
                    let want_v = cpu_dequant(&vq_ref, &vs_ref, hd, group, bv);
                    let mut raw = vec![0u8; rows * hd * 2];
                    ok.download(&mut raw).unwrap();
                    let gotk = halves_of(&raw);
                    ov.download(&mut raw).unwrap();
                    let gotv = halves_of(&raw);
                    let n = n_deq * n_kv * hd;
                    assert!(gotk[..n] == want_k[..n], "hd {hd} group {group} k{bk}: dequantized K differs");
                    assert!(gotv[..n] == want_v[..n], "hd {hd} group {group} v{bv}: dequantized V differs");
                    assert!(gotk[n..].iter().all(|&h| h == 0xFFFF) && gotv[n..].iter().all(|&h| h == 0xFFFF), "hd {hd} group {group}: dequantization wrote past its positions");
                    // the position count from the step state (20)
                    ok.upload(&vec![0xFFu8; rows * hd * 2]).unwrap();
                    dequant
                        .launch(
                            (((t * n_kv + 7) / 8).max(1) as u32, 2, 1),
                            (256, 1, 1),
                            0,
                            &stream,
                            &Args::new()
                                .buf(&qk)
                                .buf(&sk)
                                .buf(&ok)
                                .buf(&qv)
                                .buf(&sv)
                                .buf(&ov)
                                .buf(&state)
                                .i32(-1)
                                .i32(n_kv as i32)
                                .i32(hd as i32)
                                .i32(group as i32)
                                .i32(bk as i32)
                                .i32(bv as i32),
                        )
                        .unwrap();
                    stream.synchronize().unwrap();
                    ok.download(&mut raw).unwrap();
                    let gotk = halves_of(&raw);
                    let n = 20 * n_kv * hd;
                    assert!(gotk[..n] == want_k[..n] && gotk[n..].iter().all(|&h| h == 0xFFFF), "hd {hd} group {group} k{bk}: the step-state position count is not honoured");
                    checked += 1;
                }
            }
        }
        eprintln!("kv kernels: {checked} configurations match the CPU model");
    }
}
