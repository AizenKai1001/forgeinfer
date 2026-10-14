//! The GPU copy of an embedding model (`forge_embed`): the weights as fp16 on the card, the linear layers the tensor-core GEMMs of gemm.cu (`gemm_nt_f16` writes fp16, `gemm_nt_res` adds into the fp32
//! residual stream), the rest the small kernels of encoder.cu (embedding gather, LayerNorm, rotary embedding, attention over the tokens of a text, SwiGLU, pooling). The numbers follow the CPU path of
//! `forge_embed` (tests/gpu.rs compares them with the golden vectors); the weights cost 2 bytes per parameter (about 275 MB for nomic-embed-text-v1.5) and the working buffers about 120 MB.
//!
//! Texts go through the encoder in packs: the tokens of many texts are laid one after the other in the rows of one matrix (no padding), so that the GEMMs see thousands of rows even when every text is
//! 20 tokens long; attention, the rotary position and the pooling are told where each text starts (`starts`, `seq_of` of encoder.cu).

use forge_core::half::f32_to_f16;
use forge_core::{Error, Result};
use forge_cuda::{Args, Device, DeviceBuffer, Function, Module, Stream};
use forge_embed::{rope_tables, Config, Embedding, Encode, Parts};
use forge_tok::WordPiece;
use std::sync::Mutex;

struct GpuLayer {
    qkv: DeviceBuffer,
    out: DeviceBuffer,
    up: DeviceBuffer,
    gate: DeviceBuffer,
    down: DeviceBuffer,
    n1w: DeviceBuffer,
    n1b: DeviceBuffer,
    n2w: DeviceBuffer,
    n2b: DeviceBuffer,
}

struct Kernels {
    embed: Function,
    ln: Function,
    rope: Function,
    attn: Function,
    swiglu: Function,
    pool: Function,
    gemm_f16: Function,
    gemm_res: Function,
}

/// The working memory of one pack of texts, sized for `rows_max` tokens.
struct Work {
    stream: Stream,
    ids: DeviceBuffer,
    seq_of: DeviceBuffer,
    starts: DeviceBuffer,
    x32: DeviceBuffer,
    x16: DeviceBuffer,
    qkv16: DeviceBuffer,
    ctx16: DeviceBuffer,
    up16: DeviceBuffer,
    gate16: DeviceBuffer,
    h16: DeviceBuffer,
    pooled: DeviceBuffer,
}

pub struct GpuEmbedder {
    dev: Device,
    // the module must outlive its functions
    _module: Module,
    cfg: Config,
    name: String,
    tok: WordPiece,
    tok_emb: DeviceBuffer,
    type_row: DeviceBuffer,
    emb_w: DeviceBuffer,
    emb_b: DeviceBuffer,
    cos: DeviceBuffer,
    sin: DeviceBuffer,
    layers: Vec<GpuLayer>,
    k: Kernels,
    rows_max: usize,
    work: Mutex<Work>,
    label: String,
}

fn f16_buffer(v: &[f32]) -> Result<DeviceBuffer> {
    let mut bytes = Vec::with_capacity(v.len() * 2);
    for &x in v {
        bytes.extend_from_slice(&f32_to_f16(x).to_le_bytes());
    }
    DeviceBuffer::from_bytes(&bytes)
}

fn f32_buffer(v: &[f32]) -> Result<DeviceBuffer> {
    let mut bytes = Vec::with_capacity(v.len() * 4);
    for &x in v {
        bytes.extend_from_slice(&x.to_le_bytes());
    }
    DeviceBuffer::from_bytes(&bytes)
}

fn i32_bytes(v: impl Iterator<Item = i32>) -> Vec<u8> {
    v.flat_map(|x| x.to_le_bytes()).collect()
}

impl GpuEmbedder {
    /// Put a loaded model on GPU number `ordinal`.
    pub fn new(parts: Parts, ordinal: usize) -> Result<GpuEmbedder> {
        let dev = Device::open(ordinal)?;
        let (image, _ptx) = forge_kernels::image_for(dev.info.cc_major, dev.info.cc_minor)
            .ok_or_else(|| Error::msg(format!("no GPU kernels for compute capability {}.{}", dev.info.cc_major, dev.info.cc_minor)))?;
        let module = Module::load(image)?;
        let f = |name: &str| module.function(name);
        let k = Kernels {
            embed: f("enc_embed")?,
            ln: f("enc_ln")?,
            rope: f("enc_rope")?,
            attn: f("enc_attn")?,
            swiglu: f("enc_swiglu")?,
            pool: f("enc_pool")?,
            gemm_f16: f("gemm_nt_f16")?,
            gemm_res: f("gemm_nt_res")?,
        };
        let cfg = parts.cfg;
        let (e, ff, heads) = (cfg.n_embd, cfg.n_ff, cfg.n_head);
        let hd = e / heads;
        if e % 32 != 0 || ff % 32 != 0 || hd % 8 != 0 || hd > 256 {
            return Err(Error::msg(format!("the GPU encoder needs embedding and feed-forward widths that are multiples of 32 and a head width that is a multiple of 8 up to 256 (this model: {e}, {ff}, {hd})")));
        }
        let rows_max = cfg.ctx.max(4096);
        let mut layers = Vec::with_capacity(parts.layers.len());
        for l in &parts.layers {
            layers.push(GpuLayer {
                qkv: f16_buffer(&l.qkv)?,
                out: f16_buffer(&l.out)?,
                up: f16_buffer(&l.up)?,
                gate: f16_buffer(&l.gate)?,
                down: f16_buffer(&l.down)?,
                n1w: f32_buffer(&l.norm1.w)?,
                n1b: f32_buffer(&l.norm1.b)?,
                n2w: f32_buffer(&l.norm2.w)?,
                n2b: f32_buffer(&l.norm2.b)?,
            });
        }
        let (cos, sin) = rope_tables(cfg.ctx, hd, cfg.rope_base);
        let work = Work {
            stream: Stream::new()?,
            ids: DeviceBuffer::alloc(rows_max * 4)?,
            seq_of: DeviceBuffer::alloc(rows_max * 4)?,
            starts: DeviceBuffer::alloc((rows_max + 1) * 4)?,
            x32: DeviceBuffer::alloc(rows_max * e * 4)?,
            x16: DeviceBuffer::alloc(rows_max * e * 2)?,
            qkv16: DeviceBuffer::alloc(rows_max * 3 * e * 2)?,
            ctx16: DeviceBuffer::alloc(rows_max * e * 2)?,
            up16: DeviceBuffer::alloc(rows_max * ff * 2)?,
            gate16: DeviceBuffer::alloc(rows_max * ff * 2)?,
            h16: DeviceBuffer::alloc(rows_max * ff * 2)?,
            pooled: DeviceBuffer::alloc(rows_max * e * 4)?,
        };
        let label = format!("GPU {} (sm_{}{}), fp16 weights", dev.info.name, dev.info.cc_major, dev.info.cc_minor);
        Ok(GpuEmbedder {
            tok_emb: f16_buffer(&parts.tok_emb)?,
            type_row: f16_buffer(&parts.type_emb)?,
            emb_w: f32_buffer(&parts.emb_norm.w)?,
            emb_b: f32_buffer(&parts.emb_norm.b)?,
            cos: f32_buffer(&cos)?,
            sin: f32_buffer(&sin)?,
            layers,
            k,
            rows_max,
            work: Mutex::new(work),
            label,
            name: parts.name,
            tok: parts.tok,
            cfg,
            _module: module,
            dev,
        })
    }

    pub fn tokenizer(&self) -> &WordPiece {
        &self.tok
    }

    /// The most tokens one pass takes (more texts go in several passes).
    pub fn rows_max(&self) -> usize {
        self.rows_max
    }

    #[allow(clippy::too_many_arguments)]
    fn gemm_f16(&self, st: &Stream, a: u64, w: &DeviceBuffer, c: u64, m: usize, n: usize, k: usize) -> Result<()> {
        self.k.gemm_f16.launch(
            (n.div_ceil(128) as u32, m.div_ceil(128) as u32, 1),
            (256, 1, 1),
            0,
            st,
            &Args::new().ptr(a).ptr(w.ptr()).ptr(c).i32(m as i32).i32(n as i32).i32(k as i32).i32(k as i32).i32(k as i32).i32(n as i32).f32(1.0),
        )
    }

    /// x += a W^T on the fp32 residual stream (the output and the residual are the same matrix).
    #[allow(clippy::too_many_arguments)]
    fn gemm_res(&self, st: &Stream, a: u64, w: &DeviceBuffer, x: u64, m: usize, n: usize, k: usize) -> Result<()> {
        self.k.gemm_res.launch(
            (n.div_ceil(128) as u32, m.div_ceil(128) as u32, 1),
            (256, 1, 1),
            0,
            st,
            &Args::new().ptr(a).ptr(w.ptr()).ptr(x).i32(m as i32).i32(n as i32).i32(k as i32).i32(k as i32).i32(k as i32).i32(n as i32).f32(1.0).ptr(x).i32(n as i32).f32(1.0),
        )
    }

    /// The vectors of token sequences (each starts with `[CLS]`, ends with `[SEP]` and fits the context), in order.
    pub fn embed_batch(&self, seqs: &[Vec<u32>]) -> Result<Vec<Vec<f32>>> {
        for s in seqs {
            if s.is_empty() || s.len() > self.cfg.ctx {
                return Err(Error::msg(format!("a text of {} tokens does not fit the context of {}", s.len(), self.cfg.ctx)));
            }
        }
        self.dev.make_current()?;
        let w = self.work.lock().unwrap_or_else(|p| p.into_inner());
        let mut out = Vec::with_capacity(seqs.len());
        let mut i = 0;
        while i < seqs.len() {
            // as many texts as fit one pass (a text is at most `ctx` tokens, so one always fits)
            let (mut rows, mut j) = (0usize, i);
            while j < seqs.len() && (j == i || rows + seqs[j].len() <= self.rows_max) {
                rows += seqs[j].len();
                j += 1;
            }
            out.extend(self.run_pack(&w, &seqs[i..j], rows)?);
            i = j;
        }
        Ok(out)
    }

    /// One pass over `seqs` laid out in `rows` rows.
    fn run_pack(&self, w: &Work, seqs: &[Vec<u32>], rows: usize) -> Result<Vec<Vec<f32>>> {
        let (e, ff, heads) = (self.cfg.n_embd, self.cfg.n_ff, self.cfg.n_head);
        let hd = e / heads;
        let st = &w.stream;
        let max_n = seqs.iter().map(|s| s.len()).max().unwrap_or(1);
        let vocab = self.cfg.vocab as u32;
        let ids: Vec<u8> = seqs.iter().flat_map(|s| s.iter().flat_map(|&t| t.min(vocab - 1).to_le_bytes())).collect();
        let seq_of = i32_bytes(seqs.iter().enumerate().flat_map(|(si, s)| std::iter::repeat(si as i32).take(s.len())));
        let mut at = 0i32;
        let mut starts = Vec::with_capacity(seqs.len() + 1);
        for s in seqs {
            starts.push(at);
            at += s.len() as i32;
        }
        starts.push(at);
        w.ids.upload_at(0, &ids)?;
        w.seq_of.upload_at(0, &seq_of)?;
        w.starts.upload_at(0, &i32_bytes(starts.into_iter()))?;
        let k = &self.k;
        let rows_u = rows as u32;
        k.embed.launch((rows_u, 1, 1), (256, 1, 1), 0, st, &Args::new().ptr(w.ids.ptr()).ptr(self.tok_emb.ptr()).ptr(self.type_row.ptr()).ptr(w.x32.ptr()).i32(e as i32))?;
        let ln = |x: &DeviceBuffer, gw: &DeviceBuffer, gb: &DeviceBuffer| {
            k.ln.launch((rows_u, 1, 1), (256, 1, 1), 0, st, &Args::new().ptr(x.ptr()).ptr(gw.ptr()).ptr(gb.ptr()).ptr(w.x16.ptr()).i32(e as i32).f32(self.cfg.eps))
        };
        ln(&w.x32, &self.emb_w, &self.emb_b)?;
        let rope_threads = (rows * 2 * heads * (hd / 2)) as u32;
        let scale = 1.0 / (hd as f32).sqrt();
        for l in &self.layers {
            self.gemm_f16(st, w.x16.ptr(), &l.qkv, w.qkv16.ptr(), rows, 3 * e, e)?;
            k.rope.launch(
                (rope_threads.div_ceil(256), 1, 1),
                (256, 1, 1),
                0,
                st,
                &Args::new().ptr(w.qkv16.ptr()).ptr(self.cos.ptr()).ptr(self.sin.ptr()).ptr(w.starts.ptr()).ptr(w.seq_of.ptr()).i32(rows as i32).i32(heads as i32).i32(hd as i32).i32(e as i32),
            )?;
            k.attn.launch(
                (rows_u, heads as u32, 1),
                (128, 1, 1),
                ((hd + max_n) * 4) as u32,
                st,
                &Args::new().ptr(w.qkv16.ptr()).ptr(w.ctx16.ptr()).ptr(w.starts.ptr()).ptr(w.seq_of.ptr()).i32(heads as i32).i32(hd as i32).i32(e as i32).f32(scale),
            )?;
            self.gemm_res(st, w.ctx16.ptr(), &l.out, w.x32.ptr(), rows, e, e)?;
            ln(&w.x32, &l.n1w, &l.n1b)?;
            self.gemm_f16(st, w.x16.ptr(), &l.up, w.up16.ptr(), rows, ff, e)?;
            self.gemm_f16(st, w.x16.ptr(), &l.gate, w.gate16.ptr(), rows, ff, e)?;
            let count = (rows * ff) as u64;
            k.swiglu.launch((count.div_ceil(256) as u32, 1, 1), (256, 1, 1), 0, st, &Args::new().ptr(w.gate16.ptr()).ptr(w.up16.ptr()).ptr(w.h16.ptr()).u64(count))?;
            self.gemm_res(st, w.h16.ptr(), &l.down, w.x32.ptr(), rows, e, ff)?;
            ln(&w.x32, &l.n2w, &l.n2b)?;
        }
        k.pool.launch((seqs.len() as u32, 1, 1), (256, 1, 1), 0, st, &Args::new().ptr(w.x32.ptr()).ptr(w.pooled.ptr()).ptr(w.starts.ptr()).i32(e as i32).i32(self.cfg.mean_pool as i32))?;
        st.synchronize()?;
        let flat = w.pooled.download_f32(seqs.len() * e)?;
        Ok(flat.chunks(e).map(|c| c.to_vec()).collect())
    }
}

impl Encode for GpuEmbedder {
    fn embed_many(&self, texts: &[&str]) -> std::result::Result<Vec<Embedding>, String> {
        let seqs: Vec<Vec<u32>> = texts.iter().map(|t| self.tok.encode_truncated(t, self.cfg.ctx)).collect();
        let vectors = self.embed_batch(&seqs).map_err(|e| e.to_string())?;
        Ok(seqs.iter().zip(vectors).map(|(s, vector)| Embedding { vector, tokens: s.len() }).collect())
    }
    fn config(&self) -> &Config {
        &self.cfg
    }
    fn name(&self) -> &str {
        &self.name
    }
    fn backend(&self) -> String {
        self.label.clone()
    }
}
