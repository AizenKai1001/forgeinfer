//! The ForgeInfer engine: one model on the GPU behind a request queue.
//!
//! A dedicated thread owns the CUDA context, the weights and the KV cache of one session; requests arrive over a bounded queue, are answered with a stream of events. For every request the prompt is
//! rendered (chat template of the model family) and tokenized, the longest prefix shared with the tokens already in the KV cache is kept (prefix reuse), the rest is prefilled (tensor-core GEMM path
//! for long prompts), then tokens are sampled one by one; the text goes through a UTF-8 safe detokenizer, a stop-string matcher and the reasoning / tool-call parser, and comes out as `Event`s.

pub mod constrain;
pub mod pld;
pub mod text;

use forge_chat::{CallFormat, ChatFormat, Event as ParseEvent, OutputParser};
use forge_core::{Error, Json, Result};
use forge_gpu::{Fmt, GpuModel, GpuSession, LogitsSrc, StateSnapshot};
use forge_model::{Arch, LoadOpts, Model, Sampler, SamplerConfig};
use forge_tok::Tokenizer;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{channel, sync_channel, Receiver, Sender, SyncSender, TrySendError};
use std::sync::{Arc, Mutex};
use std::time::Instant;
use text::{StopMatcher, Utf8Stream};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WeightMode {
    /// fp16 matrices (any model)
    F16,
    /// Q8_0 quantized from the loaded values
    Q8,
    /// every tensor in the format of the GGUF file (Q8_0, Q4_0, Q5_0, Q4_K, Q5_K, Q6_K)
    Native,
}

#[derive(Clone, Debug, Default)]
pub struct SamplingParams {
    pub temperature: Option<f32>,
    pub top_k: Option<usize>,
    pub top_p: Option<f32>,
    pub min_p: Option<f32>,
    pub repetition_penalty: Option<f32>,
    pub seed: Option<u64>,
    /// `response_format: json_object`: the answer is a JSON object (candidates that would break the grammar are removed before every draw)
    pub json_object: bool,
    /// `response_format: json_schema`: the answer follows this schema (compiled by `constrain::SchemaDoc::from_json`)
    pub json_schema: Option<std::sync::Arc<constrain::SchemaDoc>>,
}

impl SamplingParams {
    fn over(&self, base: &SamplingParams) -> SamplingParams {
        SamplingParams {
            temperature: self.temperature.or(base.temperature),
            top_k: self.top_k.or(base.top_k),
            top_p: self.top_p.or(base.top_p),
            min_p: self.min_p.or(base.min_p),
            repetition_penalty: self.repetition_penalty.or(base.repetition_penalty),
            seed: self.seed.or(base.seed),
            json_object: self.json_object || base.json_object,
            json_schema: self.json_schema.clone().or_else(|| base.json_schema.clone()),
        }
    }
}

pub struct EngineConfig {
    pub model: PathBuf,
    pub tokenizer: Option<PathBuf>,
    pub weights: WeightMode,
    pub n_ctx: usize,
    pub device: usize,
    pub alias: String,
    /// None: chosen from the architecture
    pub chat: Option<ChatFormat>,
    /// requests waiting behind the running one
    pub max_queue: usize,
    pub defaults: SamplingParams,
    /// Qwen3.5: where the MTP head's tensors are (a Hugging Face directory or a safetensors file with the `mtp.*` tensors); with it, generation drafts one token ahead and verifies two at once
    pub mtp: Option<PathBuf>,
    /// use the MTP head for speculative decoding when it is loaded
    pub speculate: bool,
}

#[derive(Clone, Debug)]
pub struct ModelInfo {
    pub alias: String,
    pub n_ctx: usize,
    pub vocab: usize,
    pub n_params: usize,
    pub chat: ChatFormat,
    pub weights: String,
    pub weight_bytes: usize,
    pub supports_tools: bool,
    /// the MTP head is loaded and speculative decoding is on
    pub speculative: bool,
}

pub enum Prompt {
    /// OpenAI-style messages (tool-call arguments as JSON strings or objects) and tool definitions
    Chat {
        messages: Vec<Json>,
        tools: Vec<Json>,
        enable_thinking: Option<bool>,
    },
    /// text completion
    Raw(String),
}

pub struct Request {
    pub prompt: Prompt,
    pub sampling: SamplingParams,
    pub max_tokens: Option<usize>,
    pub stop: Vec<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FinishReason {
    Stop,
    Length,
    ToolCalls,
    Cancelled,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub cached_tokens: usize,
    pub completion_tokens: usize,
    /// speculative decoding: verification steps run, and how many of the drafted tokens were right
    pub draft_cycles: usize,
    pub draft_accepted: usize,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Timings {
    pub queue_ms: f64,
    pub prefill_ms: f64,
    pub decode_ms: f64,
}

#[derive(Clone, Debug)]
pub enum EngineError {
    InvalidRequest(String),
    Template(String),
    ContextOverflow { prompt_tokens: usize, n_ctx: usize },
    Busy,
    Device(String),
}

impl std::fmt::Display for EngineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EngineError::InvalidRequest(m) | EngineError::Template(m) | EngineError::Device(m) => write!(f, "{m}"),
            EngineError::ContextOverflow { prompt_tokens, n_ctx } => write!(f, "the prompt has {prompt_tokens} tokens but the context holds {n_ctx} (n_ctx exceeded: too many tokens)"),
            EngineError::Busy => write!(f, "the server is busy: the request queue is full"),
        }
    }
}

#[derive(Clone, Debug)]
pub enum Event {
    Started {
        prompt_tokens: usize,
        cached_tokens: usize,
    },
    Reasoning(String),
    Content(String),
    ToolCallStart {
        index: usize,
        id: String,
        name: String,
    },
    ToolCallArgs {
        index: usize,
        fragment: String,
    },
    ToolCallEnd {
        index: usize,
        complete: bool,
    },
    Finished {
        reason: FinishReason,
        usage: Usage,
        timings: Timings,
    },
    Error(EngineError),
}

/// A running generation: events arrive on `rx`; dropping it cancels the request.
pub struct Generation {
    pub rx: Receiver<Event>,
    cancel: Arc<AtomicBool>,
}

impl Generation {
    pub fn cancel(&self) {
        self.cancel.store(true, Ordering::SeqCst);
    }
    pub fn cancel_handle(&self) -> Arc<AtomicBool> {
        self.cancel.clone()
    }
}

impl Drop for Generation {
    fn drop(&mut self) {
        self.cancel.store(true, Ordering::SeqCst);
    }
}

struct Job {
    ids: Vec<u32>,
    /// number of prompt tokens before the final assistant turn begins (where the next request of the same conversation diverges); hybrid models snapshot their state there
    boundary: Option<usize>,
    sampling: SamplingParams,
    max_tokens: usize,
    stop: Vec<String>,
    tools: Vec<Json>,
    thinking_open: bool,
    call_format: Option<CallFormat>,
    tx: Sender<Event>,
    cancel: Arc<AtomicBool>,
    created: Instant,
}

pub struct Engine {
    tx: SyncSender<Job>,
    pub info: ModelInfo,
    pub tok: Arc<Tokenizer>,
    chat: ChatFormat,
    n_ctx: usize,
    stats: Arc<Mutex<Stats>>,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Stats {
    pub requests: u64,
    pub cached_tokens: u64,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub draft_cycles: u64,
    pub draft_accepted: u64,
}

static CALL_COUNTER: AtomicU64 = AtomicU64::new(1);

fn call_id() -> String {
    let n = CALL_COUNTER.fetch_add(1, Ordering::Relaxed);
    let t = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    format!(
        "call_{:012x}{:012x}",
        t & 0xFFFF_FFFF_FFFF,
        n.wrapping_mul(0x9E37_79B9_7F4A_7C15) & 0xFFFF_FFFF_FFFF
    )
}

/// The chat template a GGUF file carries (`tokenizer.chat_template`), if the model is a GGUF file.
fn gguf_chat_template(model_path: &std::path::Path) -> Option<String> {
    if model_path.is_dir() || !model_path.extension().map(|e| e.eq_ignore_ascii_case("gguf")).unwrap_or(false) {
        return None;
    }
    let g = forge_gguf::Gguf::open(model_path).ok()?;
    g.get_str("tokenizer.chat_template").map(|s| s.to_string())
}

fn default_chat(model_path: &std::path::Path, arch: Arch, n_embd: usize) -> ChatFormat {
    let name = model_path.to_string_lossy().to_lowercase();
    match arch {
        Arch::Qwen3 => ChatFormat::Qwen3,
        // the 0.8B and 2B templates default to answering without a thinking phase, the 4B and 9B think
        Arch::Qwen35 => ChatFormat::Qwen35 {
            thinking_default: n_embd > 2048,
        },
        Arch::Llama => {
            let template = gguf_chat_template(model_path).unwrap_or_default();
            if template.contains("<|start_header_id|>") {
                ChatFormat::Llama3
            } else if name.contains("smollm") {
                ChatFormat::SmolLm2
            } else {
                ChatFormat::ChatMl
            }
        }
    }
}

impl Engine {
    /// Load the model on the engine thread and return once it is ready (or failed).
    pub fn load(cfg: EngineConfig) -> Result<Engine> {
        let tok_path = match &cfg.tokenizer {
            Some(p) => p.clone(),
            None if cfg.model.is_dir() => cfg.model.join("tokenizer.json"),
            None => cfg
                .model
                .parent()
                .map(|p| p.join("tokenizer.json"))
                .unwrap_or_else(|| PathBuf::from("tokenizer.json")),
        };
        // A GGUF file carries its own tokenizer: it is used when no tokenizer.json is named or sits beside the model.
        let is_gguf = !cfg.model.is_dir() && cfg.model.extension().map(|e| e.eq_ignore_ascii_case("gguf")).unwrap_or(false);
        let tok = if is_gguf && cfg.tokenizer.is_none() && !tok_path.exists() {
            let g = forge_gguf::Gguf::open(&cfg.model).map_err(|e| Error::msg(format!("model {}: {e}", cfg.model.display())))?;
            Arc::new(Tokenizer::from_gguf(&g).map_err(|e| Error::msg(format!("tokenizer of {}: {e}", cfg.model.display())))?)
        } else {
            Arc::new(Tokenizer::from_file(&tok_path).map_err(|e| Error::msg(format!("tokenizer {}: {e}", tok_path.display())))?)
        };
        // capacity = requests waiting behind the running one (the engine thread takes a job out of the channel before it starts it)
        let (tx, rx) = sync_channel::<Job>(cfg.max_queue);
        let (ready_tx, ready_rx) =
            channel::<std::result::Result<(ModelInfo, ChatFormat), String>>();
        let tok2 = tok.clone();
        let stats = Arc::new(Mutex::new(Stats::default()));
        let stats2 = stats.clone();
        let from_trainer = is_training_checkpoint(&cfg.model);
        let mut cfg = cfg;
        if from_trainer {
            // the window of a training checkpoint's forward pass is a multiple of 64 positions and at most 2048
            cfg.n_ctx = (cfg.n_ctx.min(2048) / 64 * 64).max(64);
        }
        let n_ctx = cfg.n_ctx;
        std::thread::Builder::new()
            .name("forge-engine".into())
            .spawn(move || {
                if from_trainer {
                    engine_main_trainer(cfg, tok2, rx, ready_tx, stats2)
                } else {
                    engine_main(cfg, tok2, rx, ready_tx, stats2)
                }
            })
            .map_err(|e| Error::msg(format!("cannot start the engine thread: {e}")))?;
        let (info, chat) = ready_rx
            .recv()
            .map_err(|_| Error::msg("the engine thread died while loading"))?
            .map_err(Error::msg)?;
        Ok(Engine {
            tx,
            info,
            tok,
            chat,
            n_ctx,
            stats,
        })
    }

    pub fn stats(&self) -> Stats {
        *self.stats.lock().unwrap()
    }

    /// The prompt text and token ids a request would be answered from.
    pub fn render(
        &self,
        prompt: &Prompt,
    ) -> std::result::Result<(String, Vec<u32>, bool), EngineError> {
        match prompt {
            Prompt::Raw(t) => Ok((t.clone(), self.tok.encode(t), false)),
            Prompt::Chat {
                messages,
                tools,
                enable_thinking,
            } => {
                let msgs = forge_chat::wire_to_template(messages);
                let text = self
                    .chat
                    .render(&msgs, tools, true, *enable_thinking)
                    .map_err(|e| EngineError::Template(e.to_string()))?;
                let ids = self.tok.encode(&text);
                Ok((text, ids, self.chat.thinking_open(*enable_thinking)))
            }
        }
    }

    /// Queue a request. Errors that are known up front (template, context size, full queue) are returned directly.
    pub fn start(&self, req: Request) -> std::result::Result<Generation, EngineError> {
        let (text, ids, thinking_open) = self.render(&req.prompt)?;
        // chat prompts: where the last assistant turn starts (the part before it is what the next turn of the conversation will share).
        // The next turn re-renders this answer as a history message: the header "<|im_start|>assistant" plus a newline, then the content, without the opening of a thinking block that this prompt may end with.
        // So the longest part the two share is the prompt up to and including that header; the snapshot is taken there, which leaves the few tokens after it (4 for an empty thinking block) for a
        // single group of rows instead of a pass over the weights. (Where that prefix does not tokenize the same way inside the whole prompt, the older boundary before the header is used.)
        let boundary = match &req.prompt {
            Prompt::Chat { .. } => {
                let hdr = "<|im_start|>assistant\n";
                let consistent = |end: usize| -> Option<usize> {
                    let b = self.tok.encode(&text[..end]);
                    (!b.is_empty() && ids.starts_with(&b)).then_some(b.len())
                };
                text.rfind(hdr)
                    .and_then(|i| consistent(i + hdr.len()).or_else(|| consistent(i)))
            }
            Prompt::Raw(_) => None,
        };
        if ids.is_empty() {
            return Err(EngineError::InvalidRequest("the prompt is empty".into()));
        }
        if ids.len() >= self.n_ctx {
            return Err(EngineError::ContextOverflow {
                prompt_tokens: ids.len(),
                n_ctx: self.n_ctx,
            });
        }
        let remaining = self.n_ctx - ids.len();
        let max_tokens = req.max_tokens.unwrap_or(remaining).min(remaining).max(1);
        let (tools, call_format) = match &req.prompt {
            Prompt::Chat { tools, .. } => (tools.clone(), self.chat.call_format()),
            Prompt::Raw(_) => (Vec::new(), None),
        };
        let (etx, erx) = channel();
        let cancel = Arc::new(AtomicBool::new(false));
        let job = Job {
            ids,
            boundary,
            sampling: req.sampling,
            max_tokens,
            stop: req.stop,
            tools,
            thinking_open,
            call_format,
            tx: etx,
            cancel: cancel.clone(),
            created: Instant::now(),
        };
        match self.tx.try_send(job) {
            Ok(()) => Ok(Generation { rx: erx, cancel }),
            Err(TrySendError::Full(_)) => Err(EngineError::Busy),
            Err(TrySendError::Disconnected(_)) => {
                Err(EngineError::Device("the engine thread has stopped".into()))
            }
        }
    }
}

fn engine_main(
    cfg: EngineConfig,
    tok: Arc<Tokenizer>,
    rx: Receiver<Job>,
    ready: Sender<std::result::Result<(ModelInfo, ChatFormat), String>>,
    stats: Arc<Mutex<Stats>>,
) {
    let trace = std::env::var_os("FORGE_LOAD_TRACE").is_some();
    let t_load = Instant::now();
    let step = |what: &str, t: Instant| {
        if trace {
            eprintln!("[load] {what}: {:.2} s (total {:.2} s)", t.elapsed().as_secs_f64(), t_load.elapsed().as_secs_f64());
        }
    };
    let loaded = (|| -> std::result::Result<(Model,), String> {
        let m = if cfg.weights == WeightMode::Native {
            Model::load_native(&cfg.model)
        } else {
            Model::load(&cfg.model)
        };
        let mut m = m.map_err(|e| format!("model {}: {e}", cfg.model.display()))?;
        step("model file read and unpacked", t_load);
        if let Some(p) = &cfg.mtp {
            if m.mtp.is_none() {
                m.load_mtp(
                    p,
                    &LoadOpts {
                        dequantize: false,
                        keep_raw: true,
                    },
                )
                .map_err(|e| format!("MTP head {}: {e}", p.display()))?;
            }
        }
        Ok((m,))
    })();
    let (model,) = match loaded {
        Ok(v) => v,
        Err(e) => {
            let _ = ready.send(Err(e));
            return;
        }
    };
    let chat = cfg
        .chat
        .unwrap_or_else(|| default_chat(&cfg.model, model.cfg.arch, model.cfg.n_embd));
    let fmt = match cfg.weights {
        WeightMode::F16 => Fmt::F16,
        WeightMode::Q8 => Fmt::Q8_0,
        WeightMode::Native => Fmt::Native,
    };
    let n_params = model.cfg.approx_params();
    let vocab = model.cfg.vocab;
    let t_up = Instant::now();
    let gpu = match GpuModel::upload(&model, fmt, cfg.device) {
        Ok(g) => g,
        Err(e) => {
            let _ = ready.send(Err(format!("GPU upload: {e}")));
            return;
        }
    };
    drop(model);
    step("weights on the GPU", t_up);
    let t_sess = Instant::now();
    let mut sess = match GpuSession::new(&gpu, cfg.n_ctx) {
        Ok(s) => s,
        Err(e) => {
            let _ = ready.send(Err(format!("GPU session: {e}")));
            return;
        }
    };
    // The working memory of the first request is taken now: the batched prefill's buffers and, for a hybrid model, the state snapshots (50 MB each on the 9B) are made on first use, and on a nearly full
    // GPU (the 9B on an 8 GB card next to another process) the first answer failed in the middle. Now the engine does not start at all and says why.
    step("session", t_sess);
    let t_warm = Instant::now();
    if let Err(e) = warm_up(&mut sess) {
        let _ = ready.send(Err(format!("GPU warm-up: {e}")));
        return;
    }
    step("warm-up", t_warm);
    let n_ctx = sess.max_ctx();
    let info = ModelInfo {
        alias: cfg.alias.clone(),
        n_ctx,
        vocab,
        n_params,
        chat,
        weights: format!("{:?}", fmt),
        weight_bytes: gpu.weight_bytes,
        supports_tools: chat.call_format().is_some(),
        speculative: cfg.speculate && gpu.has_mtp(),
    };
    let _ = ready.send(Ok((info, chat)));
    let speculate = cfg.speculate && gpu.has_mtp();
    let lookup = cfg.speculate;
    let stop_ids: Vec<u32> = ["<|im_end|>", "<|endoftext|>", "<|end_of_text|>", "<|eot_id|>", "</s>"]
        .iter()
        .filter_map(|s| tok.token_id(s))
        .collect();
    let mut slot: Vec<u32> = Vec::new();
    let mut snaps: Vec<Snap> = Vec::new();
    while let Ok(job) = rx.recv() {
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            run_job(
                &mut sess,
                &mut slot,
                &mut snaps,
                &tok,
                &stop_ids,
                &cfg.defaults,
                speculate,
                lookup,
                &job,
                &stats,
            )
        }));
        match r {
            Ok(Ok(())) => {}
            Ok(Err(e)) => {
                eprintln!("[engine] the request failed: {e}");
                slot.clear();
                snaps.clear();
                let _ = sess.reset(); // the position of the session must follow the empty slot: a hybrid model cannot roll back, so without this every later request panicked on a stale position
                let _ = job
                    .tx
                    .send(Event::Error(EngineError::Device(e.to_string())));
            }
            Err(_) => {
                slot.clear();
                snaps.clear();
                let _ = sess.reset();
                let _ = job.tx.send(Event::Error(EngineError::Device(
                    "the engine panicked while serving the request".into(),
                )));
            }
        }
    }
}

/// One prompt of a full prefill chunk through the batched path and, for a hybrid model, as many state snapshots as a conversation keeps; afterwards the session is as new.
fn warm_up(sess: &mut GpuSession) -> Result<()> {
    let n = forge_gpu::prefill_chunk_len().min(sess.max_ctx());
    if n >= 16 {
        sess.prefill_batched(&vec![0u32; n])?;
    }
    // the multi-row kernels of short chunks load lazily too: one chunk of every group size
    if sess.prefill_batched_min() < 16 {
        for k in [3usize, 7, 12, 16, 20] {
            if k <= sess.max_ctx() {
                sess.reset()?;
                sess.prefill_batched(&vec![0u32; k])?;
            }
        }
    }
    if !sess.can_truncate() {
        let snaps = (0..MAX_SNAPS)
            .map(|_| sess.snapshot_state())
            .collect::<Result<Vec<_>>>()?;
        drop(snaps);
    }
    sess.reset()
}

/// Is `path` a checkpoint written by `forge train` (first line of the file: {"magic":"forgeckpt1", ...})?
fn is_training_checkpoint(path: &std::path::Path) -> bool {
    use std::io::Read;
    let mut head = [0u8; 24];
    path.is_file()
        && std::fs::File::open(path)
            .and_then(|mut f| f.read_exact(&mut head))
            .is_ok()
        && head.starts_with(b"{\"magic\":\"forgeckpt1\"")
}

/// The engine thread of a training checkpoint (any Forge-1 structure): the model runs through the trainer's own kernels with a KV cache (`forge_train::Decoder`: the prompt rows through the
/// trainer's GEMMs, then one token per step through fused kernels; bitwise the logits of the trainer's forward pass), and the cached positions a new prompt shares are kept (prefix reuse).
/// FORGE_NO_KV=1 selects the old path without a cache (every new token recomputes the text so far).
fn engine_main_trainer(
    cfg: EngineConfig,
    tok: Arc<Tokenizer>,
    rx: Receiver<Job>,
    ready: Sender<std::result::Result<(ModelInfo, ChatFormat), String>>,
    stats: Arc<Mutex<Stats>>,
) {
    use forge_train::{config_from_checkpoint, Decoder, Trainer};
    let loaded = (|| -> std::result::Result<(Trainer, Option<Decoder>), String> {
        let mut c = config_from_checkpoint(&cfg.model)
            .map_err(|e| format!("checkpoint {}: {e}", cfg.model.display()))?;
        c.seq_len = cfg.n_ctx;
        c.batch = 1;
        // only one row of logits is ever computed: 32 rows of cross-entropy scratch instead of n_ctx (n_ctx x vocab floats)
        c.ce_chunk = 32;
        let mut tr = Trainer::new_inference(c, cfg.device).map_err(|e| format!("GPU: {e}"))?;
        tr.load_checkpoint(&cfg.model)
            .map_err(|e| format!("checkpoint {}: {e}", cfg.model.display()))?;
        let mut dec = if std::env::var("FORGE_NO_KV").is_ok() {
            None
        } else {
            match Decoder::new(&tr, PREFILL_PIECE.min(512)) {
                Ok(d) => Some(d),
                Err(e) => {
                    // a structure option the decoder does not reproduce: the no-cache forward still serves the model
                    eprintln!("[engine] no KV cache for this checkpoint: {e}");
                    None
                }
            }
        };
        if cfg.weights == WeightMode::Q8 {
            // Q8 weights live in the decoder: the matrices are quantized on the device and the trainer's fp32 / fp16 weights are freed
            let d = dec.as_mut().ok_or_else(|| {
                "--q8 needs the KV-cache decoder (it is off or does not reproduce this checkpoint's structure)".to_string()
            })?;
            let t0 = Instant::now();
            let info = d
                .enable_q8(&mut tr, true)
                .map_err(|e| format!("Q8 weights: {e}"))?;
            eprintln!(
                "[engine] Q8 weights: {:.0} MB of matrices quantized in {:.0} ms (+ {:.0} MB of norms, scalars and value-embedding tables), {:.0} MB of fp32 / fp16 weights freed ({:.1} s)",
                info.matrix_bytes as f64 / 1e6,
                info.quantize_ms,
                info.small_bytes as f64 / 1e6,
                info.freed_bytes as f64 / 1e6,
                t0.elapsed().as_secs_f64()
            );
        }
        Ok((tr, dec))
    })();
    let (mut tr, mut dec) = match loaded {
        Ok(t) => t,
        Err(e) => {
            let _ = ready.send(Err(e));
            return;
        }
    };
    let chat = cfg.chat.unwrap_or(ChatFormat::ChatMl);
    let weights = match &dec {
        Some(d) if d.is_q8() => format!("training checkpoint, Q8 weights (int8 and an fp16 scale per 32), KV cache of {} positions ({:.0} MB)", d.max_ctx(), d.cache_bytes() as f64 / 1e6),
        Some(d) => format!("training checkpoint, fp16, KV cache of {} positions ({:.0} MB; logits bitwise equal to the trainer's forward)", d.max_ctx(), d.cache_bytes() as f64 / 1e6),
        None => "training checkpoint, fp16 forward without a KV cache".to_string(),
    };
    let weight_bytes = match &dec {
        Some(d) if d.is_q8() => d.q8_weight_bytes(),
        _ => tr.n_params * 2,
    };
    let info = ModelInfo {
        alias: cfg.alias.clone(),
        n_ctx: cfg.n_ctx,
        vocab: tr.cfg.vocab,
        n_params: tr.n_params,
        chat,
        weights,
        weight_bytes,
        supports_tools: false,
        speculative: false,
    };
    let _ = ready.send(Ok((info, chat)));
    let stop_ids: Vec<u32> = ["<|im_end|>", "<|endoftext|>"]
        .iter()
        .filter_map(|s| tok.token_id(s))
        .collect();
    while let Ok(job) = rx.recv() {
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            run_job_trainer(
                &mut tr,
                dec.as_mut(),
                cfg.n_ctx,
                &tok,
                &stop_ids,
                &cfg.defaults,
                &job,
                &stats,
            )
        }));
        let failed = match r {
            Ok(Ok(())) => false,
            Ok(Err(e)) => {
                let _ = job
                    .tx
                    .send(Event::Error(EngineError::Device(e.to_string())));
                true
            }
            Err(_) => {
                let _ = job.tx.send(Event::Error(EngineError::Device(
                    "the engine panicked while serving the request".into(),
                )));
                true
            }
        };
        if failed {
            if let Some(d) = dec.as_mut() {
                d.reset();
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn run_job_trainer(
    tr: &mut forge_train::Trainer,
    mut dec: Option<&mut forge_train::Decoder>,
    n_ctx: usize,
    tok: &Tokenizer,
    stop_ids: &[u32],
    defaults: &SamplingParams,
    job: &Job,
    stats: &Mutex<Stats>,
) -> Result<()> {
    let queue_ms = job.created.elapsed().as_secs_f64() * 1e3;
    if job.cancel.load(Ordering::SeqCst) {
        let _ = job.tx.send(Event::Finished {
            reason: FinishReason::Cancelled,
            usage: Usage::default(),
            timings: Timings {
                queue_ms,
                ..Default::default()
            },
        });
        return Ok(());
    }
    let ids = &job.ids;
    // prefix reuse: keep the cached positions the new prompt shares, but always evaluate at least its last token (its logits start the answer)
    let lcp = match dec.as_deref_mut() {
        Some(d) => {
            let lcp = d
                .ids()
                .iter()
                .zip(ids.iter())
                .take_while(|(a, b)| a == b)
                .count()
                .min(ids.len() - 1);
            d.truncate(lcp);
            lcp
        }
        None => 0,
    };
    let _ = job.tx.send(Event::Started {
        prompt_tokens: ids.len(),
        cached_tokens: lcp,
    });
    let p = job.sampling.over(defaults);
    let mut sampler = Sampler::new(SamplerConfig {
        temperature: p.temperature.unwrap_or(0.7),
        top_k: p.top_k.unwrap_or(40),
        top_p: p.top_p.unwrap_or(0.95),
        min_p: p.min_p.unwrap_or(0.0),
        repetition_penalty: p.repetition_penalty.unwrap_or(1.1),
        seed: p.seed.unwrap_or_else(|| {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(1)
        }),
    });
    let mut out = Output {
        job,
        tok,
        stop_ids,
        history: ids.iter().rev().take(64).rev().cloned().collect(),
        utf8: Utf8Stream::default(),
        stops: StopMatcher::new(job.stop.clone()),
        parser: job
            .call_format
            .map(|cf| OutputParser::new(cf, job.thinking_open, &job.tools)),
        calls_done: 0,
        produced: 0,
    };
    // the prompt: the part after the shared prefix goes into the cache in pieces (cancellation is checked between them); without a cache the first logits come from the whole prompt
    let t_prefill = Instant::now();
    let mut seq: Vec<u32> = ids.clone();
    let mut logits = match dec.as_deref_mut() {
        Some(d) => {
            let mut last = None;
            for piece in ids[lcp..].chunks(PREFILL_PIECE) {
                if job.cancel.load(Ordering::SeqCst) {
                    let _ = job.tx.send(Event::Finished {
                        reason: FinishReason::Cancelled,
                        usage: Usage {
                            prompt_tokens: ids.len(),
                            cached_tokens: lcp,
                            completion_tokens: 0,
                            ..Default::default()
                        },
                        timings: Timings {
                            queue_ms,
                            prefill_ms: t_prefill.elapsed().as_secs_f64() * 1e3,
                            decode_ms: 0.0,
                        },
                    });
                    return Ok(());
                }
                last = Some(d.feed(tr, piece)?);
            }
            last.expect("at least one prompt token is evaluated")
        }
        None => tr.logits_at(&seq, seq.len())?,
    };
    let prefill_ms = t_prefill.elapsed().as_secs_f64() * 1e3;
    let t_decode = Instant::now();
    let reason = loop {
        if job.cancel.load(Ordering::SeqCst) {
            break FinishReason::Cancelled;
        }
        let token = sampler.sample(&logits, &out.history);
        if let Some(r) = out.take(token) {
            break r;
        }
        seq.push(token);
        if seq.len() >= n_ctx {
            break FinishReason::Length;
        }
        logits = match dec.as_deref_mut() {
            Some(d) => d.feed(tr, &[token])?,
            None => tr.logits_at(&seq, seq.len())?,
        };
    };
    if !out.stops.stopped && reason != FinishReason::Cancelled {
        let tail = format!("{}{}", out.stops.flush(), out.utf8.finish());
        out.emit(&tail, false);
    }
    out.emit("", true);
    let reason = if reason == FinishReason::Stop && out.calls_done > 0 {
        FinishReason::ToolCalls
    } else {
        reason
    };
    let usage = Usage {
        prompt_tokens: ids.len(),
        cached_tokens: lcp,
        completion_tokens: out.produced,
        ..Default::default()
    };
    {
        let mut st = stats.lock().unwrap();
        st.requests += 1;
        st.cached_tokens += lcp as u64;
        st.prompt_tokens += ids.len() as u64;
        st.completion_tokens += out.produced as u64;
    }
    let _ = job.tx.send(Event::Finished {
        reason,
        usage,
        timings: Timings {
            queue_ms,
            prefill_ms,
            decode_ms: t_decode.elapsed().as_secs_f64() * 1e3,
        },
    });
    Ok(())
}

const PREFILL_PIECE: usize = 1024;
/// state snapshots kept for hybrid models
const MAX_SNAPS: usize = 3;

/// A hybrid model's recurrent state after `tokens` (a message boundary of an earlier prompt).
struct Snap {
    tokens: Vec<u32>,
    state: StateSnapshot,
}

/// The next token from the logits a source holds. Greedy decoding reads the arg-max the device already computed (4 bytes). Sampling copies back only a candidate list instead of a
/// vocabulary-sized vector, and applies the repetition penalty on the device: with a top-k limit the list only has to contain the k best tokens, so its cutoff (in temperatures below
/// the best logit, `delta`) is adapted from token to token until the list holds at least k and at most CAND_CAP entries; without a top-k limit the full support is extracted and a list
/// that does not fit falls back to the full logits.
fn next_token(
    sess: &GpuSession,
    src: LogitsSrc,
    sampler: &mut Sampler,
    history: &[u32],
    delta: &mut f32,
    tok: &Tokenizer,
    constraint: Option<&mut constrain::Constraint>,
) -> Result<u32> {
    if let Some(c) = constraint {
        return next_constrained(sess, src, sampler, history, tok, c);
    }
    let pen = sampler.cfg.repetition_penalty;
    if sampler.is_greedy() && pen == 1.0 {
        return sess.argmax_of(src);
    }
    let mut seen = history.to_vec();
    seen.sort_unstable();
    seen.dedup();
    let penalty = if pen != 1.0 && !seen.is_empty() {
        Some((seen.as_slice(), pen))
    } else {
        None
    };
    sess.prepare_candidates(src, penalty)?;
    let k = sampler.cfg.top_k;
    let stats = std::env::var("FORGE_SAMPLE_STATS").is_ok();
    let cap = GpuSession::candidate_capacity();
    if !sampler.is_greedy() && k > 0 && k <= cap / 2 {
        let t = sampler.cfg.temperature;
        let (mut lo, mut hi) = (0.0f32, 30.0f32); // the cutoff is somewhere in (lo, hi) temperatures
        for _ in 0..8 {
            let cand = sess.collect_candidates(src, *delta * t)?;
            if stats {
                eprintln!(
                    "[sample] cutoff {:.2} T: {} candidates{}",
                    *delta,
                    cand.items.len(),
                    if cand.overflow { " (overflow)" } else { "" }
                );
            }
            if cand.overflow {
                hi = *delta;
                *delta = if lo > 0.0 {
                    (lo * hi).sqrt()
                } else {
                    *delta * 0.6
                };
            } else if cand.items.len() < k && *delta < 30.0 {
                lo = *delta;
                *delta = if hi < 30.0 {
                    (lo * hi).sqrt()
                } else {
                    (*delta * 2.0).min(30.0)
                };
            } else {
                return Ok(sampler.sample_candidates(cand.items));
            }
        }
    } else {
        let cand = sess.collect_candidates(src, sampler.cutoff())?;
        if !cand.overflow {
            return Ok(sampler.sample_candidates(cand.items));
        }
    }
    // the candidate list never fitted: use the full logits (they already carry the penalty)
    let logits = sess.logits_of(src)?;
    Ok(sampler.sample(&logits, &[]))
}

/// The next token of a constrained answer (`response_format: json_object`): the candidates the sampler would look at, minus those whose bytes break the grammar; when none of them fits, every token of
/// the vocabulary is tried (the model wanted prose and the best token that continues the JSON is far down). The chosen token advances the constraint: whoever draws a token has chosen it for good
/// (a draft that differs is dropped, the drawn token is the answer).
fn next_constrained(
    sess: &GpuSession,
    src: LogitsSrc,
    sampler: &mut Sampler,
    history: &[u32],
    tok: &Tokenizer,
    c: &mut constrain::Constraint,
) -> Result<u32> {
    let pen = sampler.cfg.repetition_penalty;
    let mut seen = history.to_vec();
    seen.sort_unstable();
    seen.dedup();
    let penalty = if pen != 1.0 && !seen.is_empty() { Some((seen.as_slice(), pen)) } else { None };
    sess.prepare_candidates(src, penalty)?;
    let cand = sess.collect_candidates(src, sampler.cutoff())?;
    let mut valid: Vec<(f32, u32)> = if cand.overflow {
        Vec::new()
    } else {
        cand.items.into_iter().filter(|&(_, t)| c.accepts(tok.token_bytes(t))).collect()
    };
    if valid.is_empty() {
        let logits = sess.logits_of(src)?;
        // which first bytes the state allows: most tokens fall out here without a copy of the automaton
        let first: Vec<bool> = (0..=255u8).map(|b| c.accepts(&[b])).collect();
        valid = logits
            .iter()
            .enumerate()
            .filter(|(i, l)| {
                l.is_finite() && {
                    let b = tok.token_bytes(*i as u32);
                    !b.is_empty() && first[b[0] as usize] && c.accepts(b)
                }
            })
            .map(|(i, &l)| (l, i as u32))
            .collect();
        valid.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal).then(a.1.cmp(&b.1)));
        valid.truncate(256);
    }
    if valid.is_empty() {
        return Err(Error::msg("no token of the vocabulary continues the JSON"));
    }
    let t = sampler.sample_candidates(valid);
    c.advance(tok.token_bytes(t));
    Ok(t)
}

#[allow(clippy::too_many_arguments)]
fn run_job(
    sess: &mut GpuSession,
    slot: &mut Vec<u32>,
    snaps: &mut Vec<Snap>,
    tok: &Tokenizer,
    stop_ids: &[u32],
    defaults: &SamplingParams,
    speculate: bool,
    lookup: bool,
    job: &Job,
    stats: &Mutex<Stats>,
) -> Result<()> {
    let started = Instant::now();
    let queue_ms = job.created.elapsed().as_secs_f64() * 1e3;
    if job.cancel.load(Ordering::SeqCst) {
        let _ = job.tx.send(Event::Finished {
            reason: FinishReason::Cancelled,
            usage: Usage::default(),
            timings: Timings {
                queue_ms,
                ..Default::default()
            },
        });
        return Ok(());
    }
    let ids = &job.ids;
    // prefix reuse: keep the cached positions that the new prompt shares, but always evaluate at least the last token (its logits start the answer)
    let mut lcp = slot
        .iter()
        .zip(ids.iter())
        .take_while(|(a, b)| a == b)
        .count()
        .min(ids.len() - 1);
    if !sess.can_truncate() && lcp < slot.len() {
        // a hybrid model's recurrent state summarises everything it has seen and cannot be rolled back to an earlier position: continue from the latest snapshot whose tokens both the cached
        // sequence (its KV entries are still there) and the new prompt start with, else start over
        let best = snaps
            .iter()
            .enumerate()
            .filter(|(_, s)| {
                s.tokens.len() <= lcp && slot.starts_with(&s.tokens) && ids.starts_with(&s.tokens)
            })
            .max_by_key(|(_, s)| s.tokens.len())
            .map(|(i, _)| i);
        match best {
            Some(i) => {
                sess.restore_state(&snaps[i].state)?;
                lcp = snaps[i].tokens.len();
                snaps.retain(|s| s.tokens.len() <= lcp);
            }
            None => {
                sess.reset()?;
                slot.clear();
                snaps.clear();
                lcp = 0;
            }
        }
    }
    sess.truncate(lcp);
    slot.truncate(lcp);
    let _ = job.tx.send(Event::Started {
        prompt_tokens: ids.len(),
        cached_tokens: lcp,
    });
    let t_prefill = Instant::now();
    // hybrid models: take a snapshot at the message boundary on the way (the next turn of the conversation continues from there)
    let snap_at = job
        .boundary
        .filter(|&b| !sess.can_truncate() && b > lcp && b <= ids.len());
    let mut segments: Vec<(usize, usize)> = Vec::new();
    match snap_at {
        Some(b) if b < ids.len() => segments.extend([(lcp, b), (b, ids.len())]),
        _ => segments.push((lcp, ids.len())),
    }
    for (seg_start, seg_end) in segments {
        for piece in ids[seg_start..seg_end].chunks(PREFILL_PIECE) {
            if job.cancel.load(Ordering::SeqCst) {
                let _ = job.tx.send(Event::Finished {
                    reason: FinishReason::Cancelled,
                    usage: Usage {
                        prompt_tokens: ids.len(),
                        cached_tokens: lcp,
                        completion_tokens: 0,
                        ..Default::default()
                    },
                    timings: Timings {
                        queue_ms,
                        prefill_ms: t_prefill.elapsed().as_secs_f64() * 1e3,
                        decode_ms: 0.0,
                    },
                });
                return Ok(());
            }
            let t_piece = Instant::now();
            if piece.len() >= sess.prefill_batched_min() {
                sess.prefill_batched(piece)?;
            } else {
                sess.prefill(piece)?;
            }
            if std::env::var("FORGE_SPEC_STATS").is_ok() {
                eprintln!("[prefill] {} tokens in {:.0} ms ({})", piece.len(), t_piece.elapsed().as_secs_f64() * 1e3, if piece.len() >= sess.prefill_batched_min() { "batched" } else { "token by token" });
            }
            slot.extend_from_slice(piece);
        }
        if Some(seg_end) == snap_at {
            snaps.push(Snap {
                tokens: ids[..seg_end].to_vec(),
                state: sess.snapshot_state()?,
            });
            if snaps.len() > MAX_SNAPS {
                snaps.remove(0);
            }
        }
    }
    let prefill_ms = t_prefill.elapsed().as_secs_f64() * 1e3;
    // sampling
    let p = job.sampling.over(defaults);
    let mut sampler = Sampler::new(SamplerConfig {
        temperature: p.temperature.unwrap_or(0.7),
        top_k: p.top_k.unwrap_or(40),
        top_p: p.top_p.unwrap_or(0.95),
        min_p: p.min_p.unwrap_or(0.0),
        repetition_penalty: p.repetition_penalty.unwrap_or(1.0),
        seed: p.seed.unwrap_or_else(|| {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(1)
        }),
    });
    let mut out = Output {
        job,
        tok,
        stop_ids,
        history: ids.iter().rev().take(64).rev().cloned().collect(),
        utf8: Utf8Stream::default(),
        stops: StopMatcher::new(job.stop.clone()),
        parser: job
            .call_format
            .map(|cf| OutputParser::new(cf, job.thinking_open, &job.tools)),
        calls_done: 0,
        produced: 0,
    };
    let t_decode = Instant::now();
    let mut spec = SpecStats::default();
    let mut constraint = match (&p.json_schema, p.json_object) {
        (Some(doc), _) => Some(constrain::Constraint::from_schema(doc.clone())),
        (None, true) => Some(constrain::Constraint::json_object()),
        (None, false) => None,
    };
    let reason = generate(sess, slot, &mut sampler, &mut out, speculate, lookup, &mut constraint, &mut spec)?;
    if std::env::var("FORGE_SPEC_STATS").is_ok() && spec.cycles > 0 {
        eprintln!(
            "[spec] {} cycles ({} by lookup, {} of their drafts right), {} drafts accepted ({:.2} per cycle), by position in the chain {:?}; ms per cycle: draft {:.1}, verify sweep {:.1}, sampling {:.1}, cache update {:.1}",
            spec.cycles,
            spec.pld_cycles,
            spec.pld_accepted,
            spec.accepted,
            spec.accepted as f64 / spec.cycles as f64,
            &spec.by_depth[..7],
            spec.t_phase[0] / spec.cycles as f64 * 1e3,
            spec.t_phase[1] / spec.cycles as f64 * 1e3,
            spec.t_phase[2] / spec.cycles as f64 * 1e3,
            spec.t_phase[3] / spec.cycles as f64 * 1e3
        );
        if let Some(r) = forge_gpu::kernel_profile_report() {
            eprintln!("[kprof]
{r}");
        }
    }
    // the end of the stream: held-back text, an incomplete UTF-8 tail, the parser's last events
    if !out.stops.stopped && reason != FinishReason::Cancelled {
        let tail = format!("{}{}", out.stops.flush(), out.utf8.finish());
        out.emit(&tail, false);
    }
    out.emit("", true);
    let reason = if reason == FinishReason::Stop && out.calls_done > 0 {
        FinishReason::ToolCalls
    } else {
        reason
    };
    let usage = Usage {
        prompt_tokens: ids.len(),
        cached_tokens: lcp,
        completion_tokens: out.produced,
        draft_cycles: spec.cycles,
        draft_accepted: spec.accepted,
    };
    {
        let mut st = stats.lock().unwrap();
        st.requests += 1;
        st.cached_tokens += lcp as u64;
        st.prompt_tokens += ids.len() as u64;
        st.completion_tokens += out.produced as u64;
        st.draft_cycles += spec.cycles as u64;
        st.draft_accepted += spec.accepted as u64;
    }
    let _ = started;
    let _ = job.tx.send(Event::Finished {
        reason,
        usage,
        timings: Timings {
            queue_ms,
            prefill_ms,
            decode_ms: t_decode.elapsed().as_secs_f64() * 1e3,
        },
    });
    Ok(())
}

/// Speculative-decoding counters of one request.
#[derive(Default)]
struct SpecStats {
    /// verification steps (two tokens, or a chain of drafts: FORGE_SPEC_DEPTH)
    cycles: usize,
    /// drafted tokens the main model chose too
    accepted: usize,
    /// accepted drafts by position in the chain (index 0: the first draft)
    by_depth: [usize; 8],
    /// wall time of the phases of the cycles, seconds (FORGE_SPEC_STATS=1 prints the mean per cycle): drafting, the verification sweep, sampling the tokens, the head's cache updates
    t_phase: [f64; 4],
    /// cycles whose drafts came from the text itself (prompt lookup), and the drafts of them that were right
    pld_cycles: usize,
    pld_accepted: usize,
}

/// What happens to every token the generation loop accepts: the stop checks, the repetition-penalty history, detokenization, stop strings, the reasoning / tool-call parser and the events.
struct Output<'a> {
    job: &'a Job,
    tok: &'a Tokenizer,
    stop_ids: &'a [u32],
    history: Vec<u32>,
    utf8: Utf8Stream,
    stops: StopMatcher,
    parser: Option<OutputParser>,
    calls_done: usize,
    produced: usize,
}

impl Output<'_> {
    fn emit(&mut self, text: &str, fin: bool) {
        match &mut self.parser {
            Some(p) => {
                for ev in p.feed(text, fin) {
                    let out = match ev {
                        ParseEvent::Reason(s) => Event::Reasoning(s),
                        ParseEvent::Content(s) => Event::Content(s),
                        ParseEvent::CallStart { index, name } => Event::ToolCallStart {
                            index,
                            id: call_id(),
                            name,
                        },
                        ParseEvent::CallArgs { index, fragment } => {
                            Event::ToolCallArgs { index, fragment }
                        }
                        ParseEvent::CallEnd { index, complete } => {
                            if complete {
                                self.calls_done += 1;
                            }
                            Event::ToolCallEnd { index, complete }
                        }
                    };
                    let _ = self.job.tx.send(out);
                }
            }
            None => {
                if !text.is_empty() {
                    let _ = self.job.tx.send(Event::Content(text.to_string()));
                }
            }
        }
    }

    /// Accept the next generated token. `Some(reason)`: the generation ends here (the token is not fed to the model).
    fn take(&mut self, token: u32) -> Option<FinishReason> {
        self.produced += 1;
        if self.stop_ids.contains(&token) {
            return Some(FinishReason::Stop);
        }
        self.history.push(token);
        if self.history.len() > 64 {
            self.history.remove(0);
        }
        let piece = self.utf8.push(self.tok.token_bytes(token));
        let safe = self.stops.feed(&piece);
        if !safe.is_empty() {
            self.emit(&safe, false);
        }
        if self.stops.stopped {
            return Some(FinishReason::Stop);
        }
        if self.produced >= self.job.max_tokens {
            return Some(FinishReason::Length);
        }
        None
    }
}

/// The decoding loop. Every round takes the token the logits at the current position gave, feeds it, and chooses the next one.
///
/// With the model's MTP head (`speculate`), a round is a cycle: the head drafts the token after the pending one `y` (from the hidden state of the last position and `y`), the main model
/// runs `y` and the draft `d` through the network at once, and the token the main model itself chooses after `y` (`z`, a sample from its own distribution) is compared with `d`: equal,
/// and the second token's logits are already there (two tokens for the price of about one step); different, and the state returns to the position after `y` and `z` is the pending
/// token of the next round. Every emitted token is a sample of the main model's distribution given the text before it, so the output has exactly the distribution of plain decoding.
fn generate(
    sess: &mut GpuSession,
    slot: &mut Vec<u32>,
    sampler: &mut Sampler,
    out: &mut Output,
    speculate: bool,
    lookup: bool,
    constraint: &mut Option<constrain::Constraint>,
    spec: &mut SpecStats,
) -> Result<FinishReason> {
    // a chain of drafts needs the multi-row verification: as many rows as the model's weights allow (4 with weights in host memory), one of them the pending token
    let depth: usize = std::env::var("FORGE_SPEC_DEPTH").ok().and_then(|v| v.parse().ok()).unwrap_or(1).min(7).min(sess.max_verify_rows().saturating_sub(1));
    let mut pld = pld::Pld::from_env(speculate, sess.max_verify_rows());
    pld.on &= lookup;
    if (speculate && depth >= 2) || pld.on {
        return generate_deep(sess, slot, sampler, out, depth, spec, speculate, pld, constraint);
    }
    let job = out.job;
    let mut delta = 10.0f32;
    let mut token = next_token(sess, LogitsSrc::Main, sampler, &out.history, &mut delta, out.tok, constraint.as_mut())?;
    loop {
        if job.cancel.load(Ordering::SeqCst) {
            return Ok(FinishReason::Cancelled);
        }
        if let Some(r) = out.take(token) {
            return Ok(r);
        }
        if constraint.as_ref().is_some_and(|c| c.is_complete()) {
            return Ok(FinishReason::Stop);
        }
        if sess.pos + 1 >= sess.max_ctx() {
            return Ok(FinishReason::Length);
        }
        if speculate && sess.mtp_ready() {
            let pos = sess.pos;
            let hidden = sess.hidden();
            let d = sess
                .mtp_step(token, hidden, pos - 1, true)?
                .expect("the draft was requested");
            sess.verify2(token, d)?;
            slot.push(token);
            spec.cycles += 1;
            let z = next_token(
                sess,
                LogitsSrc::Verify(0),
                sampler,
                &out.history,
                &mut delta,
                out.tok,
                constraint.as_mut(),
            )?;
            if z == d {
                spec.accepted += 1;
                // the head's cache entry for the position of `token` needed the token after it
                let h0 = sess.verify_hidden(0);
                sess.mtp_step(d, h0, pos, false)?;
                sess.commit_verify()?;
                slot.push(d);
                if let Some(r) = out.take(z) {
                    return Ok(r);
                }
                if constraint.as_ref().is_some_and(|c| c.is_complete()) {
                    return Ok(FinishReason::Stop);
                }
                token = next_token(
                    sess,
                    LogitsSrc::Verify(1),
                    sampler,
                    &out.history,
                    &mut delta,
                    out.tok,
                    constraint.as_mut(),
                )?;
            } else {
                sess.rollback_verify()?;
                token = z;
            }
        } else {
            sess.advance(token)?;
            slot.push(token);
            token = next_token(sess, LogitsSrc::Main, sampler, &out.history, &mut delta, out.tok, constraint.as_mut())?;
        }
    }
}

/// The lookup's verification rows need their buffers (a hybrid model keeps a recurrent state per row, 50 MB each on the 4B): take them now, with fewer rows when the memory is not there (see
/// `reserve_verify_rows`).
fn reserve_lookup_rows(sess: &mut GpuSession, pld: &mut pld::Pld, head_rows: usize) {
    let headroom = 0usize;
    let mut rows = 1 + pld.max_draft;
    // (rows the head's own chain verifies anyway need no new decision)
    while rows >= 3 && rows > head_rows {
        match sess.reserve_verify_rows(rows, headroom) {
            Ok(()) => break,
            Err(e) => {
                if std::env::var("FORGE_SPEC_STATS").is_ok() {
                    eprintln!("[pld] {rows} rows: {e}");
                }
                // (the multi-row kernels come in tiles of 4, 8, 12 and 16 rows: fewer rows than the tile below cost the same as the whole tile)
                rows = match rows {
                    r if r > 12 => 12,
                    r if r > 8 => 8,
                    r if r > 4 => 4,
                    _ => 0,
                };
            }
        }
    }
    rows = rows.max(head_rows.min(1 + pld.max_draft));
    if rows >= 3 {
        pld.max_draft = rows - 1;
    } else {
        pld.on = false;
    }
    if std::env::var("FORGE_SPEC_STATS").is_ok() {
        eprintln!(
            "[pld] {}",
            if pld.on {
                format!("lookup drafts up to {} tokens per cycle", pld.max_draft)
            } else {
                "off (no memory for the verification rows)".to_string()
            }
        );
    }
}

/// The decoding loop with a CHAIN of drafts (FORGE_SPEC_DEPTH = 2 or more) and/or drafts looked up in the text (FORGE_PLD=1, see `pld`): the head drafts the token after the pending one `y`, then (applied to
/// its own output) the one after that, and so on up to `depth` tokens; or a run of tokens that followed the same words earlier in the text is proposed. The main model runs `y` and all the drafts through the
/// network in ONE sweep of the weights (`verify_rows`), and the tokens it chooses itself are compared with the drafts one after the other: the first row whose own token differs ends the cycle (its token is
/// the next pending one), the drafts before it were right and cost nothing. As in `generate`, every emitted token is a sample of the main model's distribution given the text before it, whoever drafted.
/// `speculate`: the model has an MTP head (its cache is kept in step through every cycle, also the lookup ones).
#[allow(clippy::too_many_arguments)]
fn generate_deep(
    sess: &mut GpuSession,
    slot: &mut Vec<u32>,
    sampler: &mut Sampler,
    out: &mut Output,
    depth: usize,
    spec: &mut SpecStats,
    speculate: bool,
    mut pld: pld::Pld,
    constraint: &mut Option<constrain::Constraint>,
) -> Result<FinishReason> {
    let job = out.job;
    let mut delta = 10.0f32;
    let mut token = next_token(sess, LogitsSrc::Main, sampler, &out.history, &mut delta, out.tok, constraint.as_mut())?;
    if pld.on {
        // (after the first token: the free memory is what is left of the request's own buffers)
        reserve_lookup_rows(sess, &mut pld, if speculate { depth + 1 } else { 0 });
    }
    loop {
        if job.cancel.load(Ordering::SeqCst) {
            return Ok(FinishReason::Cancelled);
        }
        if let Some(r) = out.take(token) {
            return Ok(r);
        }
        if constraint.as_ref().is_some_and(|c| c.is_complete()) {
            return Ok(FinishReason::Stop);
        }
        if sess.pos + 1 >= sess.max_ctx() {
            return Ok(FinishReason::Length);
        }
        pld.tick();
        let room = sess.max_ctx() - sess.pos - 1; // rows after the pending token that still fit
        let head_ok = speculate && sess.mtp_ready();
        // drafts from the text itself first: a run that repeats earlier words costs no head steps at all
        let mut drafts: Vec<u32> = Vec::new();
        let mut lookup_n = 0usize;
        if pld.on && room >= 1 {
            slot.push(token);
            if let Some((n, d)) = pld.propose(slot, room) {
                drafts = d;
                lookup_n = n;
            }
            slot.pop();
        }
        let from_lookup = !drafts.is_empty();
        if !from_lookup && !(head_ok && room >= 1) {
            sess.advance(token)?;
            slot.push(token);
            token = next_token(sess, LogitsSrc::Main, sampler, &out.history, &mut delta, out.tok, constraint.as_mut())?;
            continue;
        }
        let pos = sess.pos;
        let t_a = std::time::Instant::now();
        let mut rows: Vec<u32> = Vec::with_capacity(drafts.len().max(depth) + 1);
        rows.push(token);
        if from_lookup {
            rows.extend_from_slice(&drafts);
            if head_ok {
                // the head's entry for the position before the pending token (a head cycle writes it while drafting)
                let hidden = sess.hidden();
                sess.mtp_step(token, hidden, pos - 1, false)?;
            }
        } else {
            let n_draft = depth.min(room);
            let hidden = sess.hidden();
            rows.push(sess.mtp_step(token, hidden, pos - 1, true)?.expect("the draft was requested"));
            let filled = sess.mtp_filled;
            for j in 1..n_draft {
                let d = sess.mtp_chain_step(rows[j], pos - 1 + j)?;
                rows.push(d);
            }
            // (the chain wrote provisional entries for positions the main model has not seen: only what the verification confirms is real)
            sess.mtp_filled = filled;
        }
        let t_b = std::time::Instant::now();
        sess.verify_rows(&rows)?;
        let t_c = std::time::Instant::now();
        slot.push(token);
        spec.cycles += 1;
        let mut keep = 1usize;
        let mut z = next_token(sess, LogitsSrc::Rows(0), sampler, &out.history, &mut delta, out.tok, constraint.as_mut())?;
        let mut fin = None;
        while keep < rows.len() && z == rows[keep] {
            spec.accepted += 1;
            spec.by_depth[(keep - 1).min(7)] += 1;
            slot.push(rows[keep]);
            let hit = out.take(z).or_else(|| constraint.as_ref().is_some_and(|c| c.is_complete()).then_some(FinishReason::Stop));
            keep += 1;
            if hit.is_some() {
                fin = hit;
                break;
            }
            z = next_token(sess, LogitsSrc::Rows(keep - 1), sampler, &out.history, &mut delta, out.tok, constraint.as_mut())?;
        }
        if from_lookup {
            spec.pld_cycles += 1;
            spec.pld_accepted += keep - 1;
            pld.record(lookup_n, keep - 1, rows.len() - 1);
        } else if pld.on && head_ok {
            pld.record_head(keep - 1);
        }
        let t_d = std::time::Instant::now();
        sess.accept_rows(keep)?;
        if head_ok {
            // the head's cache entries for the confirmed positions need the token after each of them
            for j in 0..keep - 1 {
                let h = sess.rows_hidden(j);
                sess.mtp_step(rows[j + 1], h, pos + j, false)?;
            }
        }
        let t_e = std::time::Instant::now();
        spec.t_phase[0] += (t_b - t_a).as_secs_f64();
        spec.t_phase[1] += (t_c - t_b).as_secs_f64();
        spec.t_phase[2] += (t_d - t_c).as_secs_f64();
        spec.t_phase[3] += (t_e - t_d).as_secs_f64();
        if let Some(r) = fin {
            return Ok(r);
        }
        token = z;
    }
}
