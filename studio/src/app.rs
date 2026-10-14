//! The application state and what happens to it: loading models, chatting, downloading, scanning. The screens in `ui` only draw this state and queue `Action`s.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use eframe::egui;

use crate::chats::{self, Chat, ChatMeta, Citation, Message, Role, Store};
use crate::client::{self, ChatEvent, ChatParams, Endpoint, PromptProgress};
use crate::engine::{EngineManager, EngineState, Launch};
use crate::hardware::{self, HardwareInfo};
use crate::hf::{self, Progress, Repo, RepoFile};
use crate::library::{self, Inspect, ModelEntry};
use crate::rag::{self, DocIndex, Strategy};
use crate::settings::{self, data_dir, EngineKind, GenParams, OverflowPolicy, Settings, ThinkingMode};
use crate::util::{free_port, new_id};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Screen {
    Chat,
    Library,
    Discover,
    Server,
    Hardware,
    Logs,
    Settings,
}

impl Screen {
    pub const ALL: [Screen; 7] = [
        Screen::Chat,
        Screen::Library,
        Screen::Discover,
        Screen::Server,
        Screen::Hardware,
        Screen::Logs,
        Screen::Settings,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Screen::Chat => "Chat",
            Screen::Library => "My models",
            Screen::Discover => "Discover",
            Screen::Server => "Local server",
            Screen::Hardware => "Hardware",
            Screen::Logs => "Logs",
            Screen::Settings => "Settings",
        }
    }

    pub fn from_name(s: &str) -> Option<Screen> {
        Screen::ALL.iter().copied().find(|x| {
            x.label().to_lowercase().replace(' ', "")
                == s.to_lowercase().replace(['-', '_', ' '], "")
                || format!("{x:?}").to_lowercase() == s.to_lowercase()
        })
    }
}

/// Everything the windows ask the app to do.
#[derive(Debug)]
pub enum Action {
    NewChat,
    OpenChat(String),
    DeleteChat(String),
    Send,
    Stop,
    Regenerate,
    DeleteMessage(usize),
    RenameChat(String, String),
    RemoveAttachment(usize),
    /// Remove a model file from the disk (asked twice in the library).
    DeleteModel(PathBuf),
    /// Edit user message `i` to this text, drop the conversation after it and ask again.
    EditResend(usize, String),
    LoadModel(PathBuf),
    Eject,
    Rescan,
    AddDir(String),
    RemoveDir(usize),
    Inspect(PathBuf),
    /// Fetch the prediction head of a Qwen3.5 model (the model file's path).
    GetHead(PathBuf),
    RefreshExternal,
    HfSearch,
    HfOpen(String),
    Download {
        repo: String,
        file: RepoFile,
    },
    CancelDownload(u64),
    /// Embed the lines of the playground with the loaded embedding model.
    Embed,
    Toast(String),
}

pub enum Msg {
    Scan(Vec<ModelEntry>),
    Hardware(HardwareInfo),
    Live {
        gpus: Vec<hardware::Gpu>,
        ram_free: u64,
    },
    Chat {
        req: u64,
        ev: ChatEvent,
    },
    HfSearch {
        seq: u64,
        result: Result<Vec<Repo>, String>,
    },
    HfFiles {
        repo: String,
        result: Result<Vec<RepoFile>, String>,
    },
    DownloadDone {
        id: u64,
        result: Result<PathBuf, String>,
    },
    Models(Result<Vec<String>, String>),
    Inspected(Result<Inspect, String>),
    Embedded(Result<EmbedOut, String>),
    /// A PDF, Office, OpenDocument, EPUB or HTML file has been read for the next message (or could not be).
    Attached(Result<chats::Attachment, String>),
    /// A search of the files of a chat is over: the index that now holds them and the passages found for the question.
    Retrieved {
        chat_id: String,
        msg_id: String,
        result: Result<(DocIndex, Vec<Citation>), String>,
    },
}

/// The search of the attached files that is running (chat with documents).
pub struct Retrieval {
    pub chat_id: String,
    pub msg_id: String,
    pub started: Instant,
    pub progress: rag::Progress,
}

/// What the embedding playground got back: the texts it sent, their vectors and what it cost.
#[derive(Clone, Debug, PartialEq)]
pub struct EmbedOut {
    pub texts: Vec<String>,
    pub vectors: Vec<Vec<f32>>,
    /// tokens the server counted
    pub tokens: u64,
    pub millis: f32,
}

/// The lines the embedding playground starts with.
pub const EMBED_SAMPLE: &str = "search_query: What is the capital of France?\nsearch_document: Paris is the capital and largest city of France.\nsearch_document: The Eiffel Tower was finished in 1889 and stands in Paris.\nsearch_document: The mitochondria is the powerhouse of the cell.\nsearch_document: Berlin has been the capital of Germany since 1990.";

/// The answer being generated right now.
pub struct Generation {
    pub req: u64,
    pub cancel: Arc<AtomicBool>,
    pub started: Instant,
    pub reasoning_started: Option<Instant>,
    /// How far the server has read the prompt (the last report; the bar shows it until the first token).
    pub progress: Option<PromptProgress>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum DlState {
    Running,
    Done(PathBuf),
    Failed(String),
}

pub struct Download {
    pub id: u64,
    pub repo: String,
    pub file: RepoFile,
    pub progress: Arc<Progress>,
    pub state: DlState,
    pub started: Instant,
    pub speed: f64,
    last: (Instant, u64),
}

pub struct Toast {
    pub text: String,
    pub at: Instant,
}

/// Command line options for testing and screenshots (`forge-studio --help`).
#[derive(Clone, Debug, Default)]
pub struct StartupOpts {
    pub screen: Option<Screen>,
    pub shot: Option<PathBuf>,
    /// Seconds to wait before taking the screenshot.
    pub shot_delay: f32,
    pub demo: bool,
    pub load: Option<PathBuf>,
    /// Messages to send, one after the other as soon as a model is ready (each when the answer to the one before is complete).
    pub send: Vec<String>,
    /// Take the screenshot (and close) once the reply to `send` is complete.
    pub shot_after_reply: bool,
    pub size: Option<(f32, f32)>,
    pub select_model: Option<String>,
    pub hf_query: Option<String>,
    pub inspect: Option<PathBuf>,
    /// Press Stop this many seconds after a generation starts (for tests).
    pub stop_after: Option<f32>,
    /// Open the newest saved chat instead of an empty one.
    pub open_last: bool,
    /// Start with the editor open on this message (screenshots of the editor).
    pub edit_msg: Option<usize>,
    /// Start with the newest chat of the list being renamed (screenshots).
    pub rename_first: bool,
    /// Text files attached to the first message (tests and screenshots; the same as dropping them on the window).
    pub attach: Vec<PathBuf>,
    /// Start with the parameters panel open (screenshots).
    pub show_params: bool,
    /// Embed the sample lines as soon as an embedding model is ready (tests and screenshots).
    pub embed: bool,
    /// Show the passages the model read open, under the questions about searched files (screenshots).
    pub open_passages: bool,
}

pub struct ShotState {
    pub path: PathBuf,
    pub at: Instant,
    pub requested: bool,
    pub wait_reply: bool,
}

pub struct StudioApp {
    pub settings: Settings,
    saved: Settings,
    dirty_since: Option<Instant>,
    pub screen: Screen,
    tx: Sender<Msg>,
    rx: Receiver<Msg>,
    pub ctx: egui::Context,
    pub actions: Vec<Action>,

    pub hw: HardwareInfo,
    pub library: Vec<ModelEntry>,
    pub scanning: bool,
    pub lib_filter: String,
    pub new_dir: String,
    pub inspect: Option<(PathBuf, Result<Inspect, String>)>,
    pub inspect_loading: bool,

    pub engine: EngineManager,
    pub external_models: Vec<String>,
    pub external_status: String,
    pub external_pick: String,

    pub store: Store,
    pub chat_list: Vec<ChatMeta>,
    pub chat: Chat,
    pub input: String,
    pub focus_input: bool,
    pub generation: Option<Generation>,
    next_req: u64,
    pub scroll_to_bottom: bool,
    pub show_params: bool,
    pub last_speed: Option<f32>,
    /// A failed save was already reported.
    save_warned: bool,
    /// The name typed in the parameters panel for a new preset.
    pub preset_name: String,
    /// The model card whose Delete button was pressed once (the second press deletes).
    pub confirm_delete: Option<PathBuf>,
    /// Text files dropped on the window, for the next message.
    pub attachments: Vec<crate::chats::Attachment>,
    /// The search box above the chat list.
    pub chat_filter: String,
    /// A chat being renamed in the list: (id, text, first frame).
    pub renaming: Option<(String, String, bool)>,
    /// A user message being edited: (index, text, first frame).
    pub editing: Option<(usize, String, bool)>,

    /// The embedding playground: the lines typed, whether a request is out, its answer or error.
    pub embed_text: String,
    pub embed_busy: bool,
    pub embed_out: Option<EmbedOut>,
    pub embed_error: String,
    pending_embed: bool,

    /// Chat with documents: the passages of the files of each chat with their vectors, and the search that is running.
    pub doc_index: HashMap<String, DocIndex>,
    pub retrieval: Option<Retrieval>,
    /// What the parts of the files were summarized to for a request to summarize them (shared with the thread that does it).
    pub summaries: Arc<Mutex<rag::SummaryCache>>,

    pub hf_query: String,
    pub hf_results: Vec<Repo>,
    pub hf_busy: bool,
    pub hf_error: String,
    hf_seq: u64,
    pub hf_repo: Option<String>,
    pub hf_files: Vec<RepoFile>,
    pub hf_files_busy: bool,
    pub downloads: Vec<Download>,

    pub toasts: Vec<Toast>,
    pub shot: Option<ShotState>,
    pub opts: StartupOpts,
    pending_send: std::collections::VecDeque<String>,
    pending_load: Option<PathBuf>,
    no_head: std::collections::HashSet<PathBuf>,
    live_at: Instant,
    live_busy: bool,
    last_activity: Instant,
    seen_log: usize,
    fps: Option<(Instant, u32)>,
}

impl StudioApp {
    pub fn new(cc: &eframe::CreationContext<'_>, opts: StartupOpts) -> StudioApp {
        crate::theme::apply(&cc.egui_ctx);
        crate::theme::install_fonts(&cc.egui_ctx);
        let settings = Settings::load();
        cc.egui_ctx.set_zoom_factor(settings.zoom.clamp(0.7, 2.0));
        let (tx, rx) = channel();
        let store = Store::new(chats::chats_dir(&data_dir()));
        let chat_list = store.list();
        let chat = if opts.open_last {
            chat_list
                .first()
                .and_then(|m| store.load(&m.id))
                .unwrap_or_default()
        } else {
            Chat::new()
        };
        let mut app = StudioApp {
            saved: settings.clone(),
            settings,
            dirty_since: None,
            screen: opts.screen.unwrap_or(Screen::Chat),
            tx,
            rx,
            ctx: cc.egui_ctx.clone(),
            actions: Vec::new(),
            hw: HardwareInfo::default(),
            library: Vec::new(),
            scanning: false,
            lib_filter: String::new(),
            new_dir: String::new(),
            inspect: None,
            inspect_loading: false,
            engine: EngineManager::default(),
            external_models: Vec::new(),
            external_status: String::new(),
            external_pick: String::new(),
            store,
            chat_list,
            chat,
            input: String::new(),
            focus_input: true,
            generation: None,
            next_req: 1,
            scroll_to_bottom: false,
            show_params: false,
            last_speed: None,
            save_warned: false,
            preset_name: String::new(),
            confirm_delete: None,
            attachments: Vec::new(),
            chat_filter: String::new(),
            renaming: None,
            editing: None,
            embed_text: EMBED_SAMPLE.to_string(),
            embed_busy: false,
            embed_out: None,
            embed_error: String::new(),
            pending_embed: opts.embed,
            doc_index: HashMap::new(),
            retrieval: None,
            summaries: Arc::new(Mutex::new(HashMap::new())),
            hf_query: opts.hf_query.clone().unwrap_or_default(),
            hf_results: Vec::new(),
            hf_busy: false,
            hf_error: String::new(),
            hf_seq: 0,
            hf_repo: None,
            hf_files: Vec::new(),
            hf_files_busy: false,
            downloads: Vec::new(),
            toasts: Vec::new(),
            shot: opts.shot.clone().map(|path| ShotState {
                path,
                at: Instant::now() + Duration::from_secs_f32(opts.shot_delay.max(0.5)),
                requested: false,
                wait_reply: opts.shot_after_reply,
            }),
            pending_send: opts.send.iter().cloned().collect(),
            pending_load: opts.load.clone(),
            no_head: Default::default(),
            live_at: Instant::now(),
            live_busy: false,
            last_activity: Instant::now(),
            seen_log: 0,
            fps: std::env::var_os("FORGE_STUDIO_FPS").map(|_| (Instant::now(), 0)),
            opts: opts.clone(),
        };
        if opts.demo {
            app.chat = demo_chat();
        }
        if let Some(i) = opts.edit_msg {
            if let Some(m) = app.chat.messages.get(i) {
                app.editing = Some((i, m.content.clone(), true));
            }
        }
        for p in &opts.attach {
            match crate::office::read_document(p, 24 << 20) {
                Ok(text) => app.attachments.push(crate::chats::Attachment {
                    name: p
                        .file_name()
                        .map(|s| s.to_string_lossy().to_string())
                        .unwrap_or_else(|| "file".into()),
                    text,
                }),
                Err(e) => eprintln!("--attach: {e}"),
            }
        }
        app.show_params = opts.show_params;
        if opts.rename_first {
            if let Some(m) = app.chat_list.first() {
                app.renaming = Some((m.id.clone(), m.title.clone(), true));
            }
        }
        app.detect_hardware();
        app.rescan();
        if app.settings.engine == EngineKind::External {
            app.refresh_external();
        }
        if let Some(p) = opts.inspect.clone() {
            app.inspect_model(p);
        }
        if let Some(q) = opts.hf_query.clone() {
            if !q.is_empty() {
                app.hf_search();
            }
        }
        app
    }

    /// What worker threads call when they have news. Repaints are spaced by 50 ms (about 20 per second): a window that redraws on every streamed
    /// token competes with the engine for the GPU (on Windows the graphics and compute work take turns) and measurably slows generation down.
    pub fn wake(&self) -> Arc<dyn Fn() + Send + Sync> {
        let ctx = self.ctx.clone();
        Arc::new(move || ctx.request_repaint_after(Duration::from_millis(50)))
    }

    pub fn toast(&mut self, text: impl Into<String>) {
        self.toasts.push(Toast {
            text: text.into(),
            at: Instant::now(),
        });
    }

    fn spawn<F: FnOnce(Sender<Msg>, Arc<dyn Fn() + Send + Sync>) + Send + 'static>(&self, f: F) {
        let tx = self.tx.clone();
        let wake = self.wake();
        std::thread::spawn(move || f(tx, wake));
    }

    pub fn detect_hardware(&mut self) {
        self.spawn(|tx, wake| {
            let _ = tx.send(Msg::Hardware(hardware::detect()));
            wake();
        });
    }

    pub fn rescan(&mut self) {
        if self.scanning {
            return;
        }
        self.scanning = true;
        let dirs = self.settings.dirs();
        self.spawn(move |tx, wake| {
            let _ = tx.send(Msg::Scan(library::scan(&dirs, true)));
            wake();
        });
    }

    pub fn inspect_model(&mut self, path: PathBuf) {
        self.inspect_loading = true;
        let entry = self.library.iter().find(|e| e.path == path).cloned();
        let shown = path.clone();
        self.spawn(move |tx, wake| {
            let (name, source) = entry
                .map(|e| (Some(e.name), e.source))
                .unwrap_or((None, "file".to_string()));
            let r = library::inspect(&path, name, &source);
            let _ = tx.send(Msg::Inspected(r));
            wake();
        });
        self.inspect = Some((shown, Err("reading the file ...".to_string())));
    }

    pub fn refresh_external(&mut self) {
        let ep = Endpoint {
            base: self.settings.external_base(),
            key: self.settings.external_key.clone(),
        };
        self.external_status = "checking ...".to_string();
        self.spawn(move |tx, wake| {
            let _ = tx.send(Msg::Models(client::list_models(&ep)));
            wake();
        });
    }

    pub fn hf_search(&mut self) {
        let q = self.hf_query.trim().to_string();
        if q.is_empty() {
            return;
        }
        self.hf_busy = true;
        self.hf_error.clear();
        self.hf_seq += 1;
        let seq = self.hf_seq;
        self.spawn(move |tx, wake| {
            let _ = tx.send(Msg::HfSearch {
                seq,
                result: hf::search(&q),
            });
            wake();
        });
    }

    pub fn hf_open(&mut self, repo: String) {
        self.hf_repo = Some(repo.clone());
        self.hf_files.clear();
        self.hf_files_busy = true;
        self.spawn(move |tx, wake| {
            let r = hf::files(&repo);
            let _ = tx.send(Msg::HfFiles { repo, result: r });
            wake();
        });
    }

    pub fn start_download(&mut self, repo: String, file: RepoFile) {
        let dest = hf::destination(
            std::path::Path::new(&self.settings.download_dir),
            &repo,
            &file,
        );
        if self
            .downloads
            .iter()
            .any(|d| d.repo == repo && d.file.path == file.path && d.state == DlState::Running)
        {
            return;
        }
        let id = self.next_req;
        self.next_req += 1;
        let progress = Arc::new(Progress::default());
        progress.total.store(file.size, Ordering::Relaxed);
        self.downloads.push(Download {
            id,
            repo: repo.clone(),
            file: file.clone(),
            progress: progress.clone(),
            state: DlState::Running,
            started: Instant::now(),
            speed: 0.0,
            last: (Instant::now(), 0),
        });
        self.spawn(move |tx, wake| {
            let w = wake.clone();
            let r = hf::download(&repo, &file, &dest, &progress, &move || w());
            let _ = tx.send(Msg::DownloadDone { id, result: r });
            wake();
        });
    }

    /// Whether the chosen engine can be started at all: the program must exist (a running server needs nothing).
    pub fn engine_program_problem(&self) -> Option<String> {
        match self.settings.engine {
            EngineKind::Forge if settings::find_forge(&self.settings).is_none() => Some("The ForgeInfer engine program (forge) was not found. Build it (docs/GETTING_STARTED.md) and put it next to this app, or set its path in Settings.".to_string()),
            EngineKind::LlamaCpp if settings::find_llama_server(&self.settings).is_none() => Some("llama.cpp's llama-server was not found. Install it (github.com/ggml-org/llama.cpp/releases) and set its path in Settings, or connect to a running server.".to_string()),
            _ => None,
        }
    }

    /// The server the chat talks to and the model name to send, when there is one.
    pub fn active_endpoint(&self) -> Option<(Endpoint, String)> {
        if self.settings.engine == EngineKind::External {
            let name = if !self.external_pick.is_empty() {
                self.external_pick.clone()
            } else {
                self.external_models.first().cloned()?
            };
            return Some((
                Endpoint {
                    base: self.settings.external_base(),
                    key: self.settings.external_key.clone(),
                },
                name,
            ));
        }
        match self.engine.state() {
            EngineState::Ready { model, port, .. } => {
                Some((Endpoint::local(port, &self.settings.api_key), model))
            }
            _ => None,
        }
    }

    /// The name of what answers now, for the title bar.
    pub fn active_name(&self) -> Option<String> {
        self.active_endpoint().map(|(_, n)| n)
    }

    /// The loaded (or loading) model is an embedding model: the chat screen shows the embedding playground for it.
    pub fn active_is_embedding(&self) -> bool {
        self.settings.engine == EngineKind::Forge
            && matches!(self.engine.state(), EngineState::Ready { .. } | EngineState::Loading { .. })
            && self.engine.current().is_some_and(|l| self.library.iter().any(|e| e.path == l.model && e.is_embedding()))
    }

    /// Send the lines of the playground to the loaded embedding model (one text per line, at most 64).
    pub fn embed_now(&mut self) {
        let Some((ep, model)) = self.active_endpoint() else {
            self.embed_error = "No embedding model is loaded yet.".to_string();
            return;
        };
        let texts: Vec<String> = self.embed_text.lines().map(str::trim).filter(|l| !l.is_empty()).take(64).map(str::to_string).collect();
        if texts.is_empty() {
            self.embed_error = "Type some text first, one per line.".to_string();
            return;
        }
        self.embed_busy = true;
        self.embed_error.clear();
        self.spawn(move |tx, wake| {
            let t0 = Instant::now();
            let r = client::embed(&ep, &model, &texts).map(|(vectors, tokens)| EmbedOut { texts, vectors, tokens, millis: t0.elapsed().as_secs_f32() * 1e3 });
            let _ = tx.send(Msg::Embedded(r));
            wake();
        });
    }

    pub fn load_model(&mut self, path: PathBuf) {
        if self.settings.engine == EngineKind::External {
            self.toast("The engine is set to a running server: pick a model there, or change the engine in Settings.");
            return;
        }
        let binary = match self.settings.engine {
            EngineKind::LlamaCpp => settings::find_llama_server(&self.settings),
            _ => settings::find_forge(&self.settings),
        };
        let Some(binary) = binary else {
            let what = if self.settings.engine == EngineKind::LlamaCpp {
                "llama-server"
            } else {
                "forge"
            };
            self.toast(format!(
                "Cannot find the {what} program. Set its path in Settings."
            ));
            self.screen = Screen::Settings;
            return;
        };
        let name = self
            .library
            .iter()
            .find(|e| e.path == path)
            .map(|e| e.name.clone())
            .unwrap_or_else(|| {
                path.file_stem()
                    .map(|s| s.to_string_lossy().to_string())
                    .unwrap_or_else(|| "model".into())
            });
        self.generation_cancel();
        self.engine.stop();
        let port = free_port(self.settings.port.max(1024));
        let mtp = self.head_for(&path);
        let launch = Launch {
            kind: self.settings.engine,
            binary,
            model: path.clone(),
            name: name.clone(),
            port,
            host: if self.settings.serve_on_lan {
                "0.0.0.0".to_string()
            } else {
                "127.0.0.1".to_string()
            },
            ctx: self.settings.context_len,
            kv: self.settings.kv_cache,
            depth: if mtp.is_some() && self.settings.spec_depth == 0 {
                3
            } else {
                self.settings.spec_depth
            },
            mtp,
            lookup: self.settings.lookup,
            prompt_cache_mb: self.settings.prompt_cache_mb,
            // chats search long attached files with an embedding model that runs inside the chat engine (an embedding model itself is served alone)
            embedding: if self.settings.engine == EngineKind::Forge && !self.library.iter().any(|e| e.path == path && e.is_embedding()) { self.retrieval_model_path() } else { None },
            api_key: self.settings.api_key.clone(),
            extra: self
                .settings
                .extra_args
                .split_whitespace()
                .map(str::to_string)
                .collect(),
        };
        match self.engine.start(launch, self.wake()) {
            Ok(()) => {
                self.last_activity = Instant::now();
                self.settings.last_model = path.display().to_string();
            }
            Err(e) => self.toast(e),
        }
    }

    /// The context length the loaded model has: what the engine says it opened (the automatic setting is only known then), else the setting.
    pub fn effective_ctx(&self) -> u32 {
        self.engine.loaded_ctx().unwrap_or(if self.settings.context_len == 0 { 8192 } else { self.settings.context_len })
    }

    /// The prediction head to use for a model: speculative decoding is on, the engine is ForgeInfer, a head file is found, and it did not fail to load before.
    pub fn head_for(&self, model: &std::path::Path) -> Option<PathBuf> {
        if !self.settings.speculative
            || self.settings.engine != EngineKind::Forge
            || self.no_head.contains(model)
        {
            return None;
        }
        let arch = match self.library.iter().find(|e| e.path == model) {
            Some(e) => e.arch.clone(),
            None => library::read_entry(model, None, "file").arch,
        };
        if arch != "qwen35" {
            return None;
        }
        let mut dirs = self.settings.dirs();
        dirs.push(PathBuf::from(&self.settings.download_dir));
        crate::mtp::find_head(model, &dirs)
    }

    /// Download the prediction head of a Qwen3.5 model from its Hugging Face repository into `<download folder>/mtp/`.
    pub fn get_head(&mut self, model: PathBuf) {
        let stem = model
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_default();
        let Some(repo) = crate::mtp::repo_of(&stem) else {
            self.toast("No known original checkpoint for this model: put a <name>-mtp.safetensors file next to it.");
            return;
        };
        let base = crate::mtp::base_name(&stem);
        let dest = std::path::Path::new(&self.settings.download_dir)
            .join("mtp")
            .join(format!("{base}-mtp.safetensors"));
        if self.downloads.iter().any(|d| {
            d.repo == repo
                && d.state == DlState::Running
                && d.file.path.ends_with("-mtp.safetensors")
        }) {
            return;
        }
        let id = self.next_req;
        self.next_req += 1;
        let progress = Arc::new(Progress::default());
        let file = RepoFile {
            path: format!("{base}-mtp.safetensors (prediction head)"),
            size: 0,
        };
        self.downloads.push(Download {
            id,
            repo: repo.clone(),
            file,
            progress: progress.clone(),
            state: DlState::Running,
            started: Instant::now(),
            speed: 0.0,
            last: (Instant::now(), 0),
        });
        self.toast(format!("Fetching the prediction head of {repo} (about 0.2 to 0.9 GB): see Discover > Downloads."));
        self.spawn(move |tx, wake| {
            let r = crate::mtp::fetch_head(&repo, &dest, &progress).map(|_| dest.clone());
            let _ = tx.send(Msg::DownloadDone { id, result: r });
            wake();
        });
    }

    pub fn eject(&mut self) {
        self.generation_cancel();
        self.engine.stop();
    }

    fn generation_cancel(&mut self) {
        self.cancel_retrieval(false);
        if let Some(g) = self.generation.take() {
            g.cancel.store(true, Ordering::Relaxed);
            if let Some(m) = self.chat.messages.last_mut() {
                m.streaming = false;
            }
        }
    }

    pub fn save_chat(&mut self) {
        if self.chat.messages.is_empty() || self.opts.demo {
            return;
        }
        self.chat.updated = crate::util::unix_secs();
        if let Err(e) = self.store.save(&self.chat) {
            self.warn_once_saving(&e.to_string());
        }
        self.chat_list = self.store.list();
    }

    /// A chat or the settings could not be written: say so once (the folder may be read-only), not at every answer.
    pub fn warn_once_saving(&mut self, why: &str) {
        if !self.save_warned {
            self.save_warned = true;
            self.toast(format!("Could not save to the data folder ({why}). Chats and settings of this session will be lost: see Settings > About for the folder."));
        }
    }

    /// An answer is streaming, or the files are being searched for the question that comes before it.
    pub fn is_generating(&self) -> bool {
        self.generation.is_some() || self.retrieval.is_some()
    }

    /// The embedding model chats search their files with: the one in Settings, else an embedding model of the library, else the one LM Studio ships.
    pub fn retrieval_model_path(&self) -> Option<PathBuf> {
        let given = self.settings.retrieval_model.trim();
        if !given.is_empty() {
            let p = PathBuf::from(given);
            return p.is_file().then_some(p);
        }
        self.library.iter().find(|e| e.is_embedding()).map(|e| e.path.clone()).or_else(library::lmstudio_embedding_model)
    }

    /// Will a message with these files and this text have the files of the chat searched, instead of sent whole? Only where an embedding model is running; then LM Studio's rule: the files that do not fit
    /// 70 percent of what is left of the context are searched, and a chat that searches its files searches for every question.
    pub fn would_search(&self, attachments: &[crate::chats::Attachment], text: &str) -> bool {
        if self.embedding_endpoint().is_none() {
            return false;
        }
        if self.chat.messages.iter().any(|m| m.by_retrieval) {
            return true;
        }
        if attachments.is_empty() {
            return false;
        }
        let tokens = |chars: usize| crate::chats::estimate_tokens(chars);
        let file_tokens: usize = attachments.iter().map(|a| tokens(a.text.chars().count())).sum();
        let occupied = tokens(self.chat.api_messages(&self.settings.system_prompt).iter().map(|(_, t)| t.chars().count()).sum());
        rag::choose(file_tokens, tokens(text.chars().count()), self.effective_ctx() as usize, occupied) == Strategy::Retrieve
    }

    /// Where a chat can have its files searched: the chat engine, when it was started with an embedding model (ForgeInfer only).
    pub fn embedding_endpoint(&self) -> Option<(Endpoint, String)> {
        if self.settings.engine != EngineKind::Forge {
            return None;
        }
        match self.engine.state() {
            EngineState::Ready { port, .. } if self.engine.current().is_some_and(|l| l.embedding.is_some()) => Some((Endpoint::local(port, &self.settings.api_key), "embedding".to_string())),
            _ => None,
        }
    }

    pub fn send(&mut self) {
        let text = self.input.trim().to_string();
        if text.is_empty() || self.is_generating() {
            return;
        }
        let Some((ep, model)) = self.active_endpoint() else {
            self.toast("Load a model first (pick one in the bar above, or in My models).");
            return;
        };
        let mut msg = Message::new(Role::User, text);
        msg.attachments = self.attachments.clone();
        // the whole conversation has to fit into the context the model was loaded with
        let mut probe = self.chat.clone();
        probe.messages.push(msg.clone());
        let chars: usize = probe
            .api_messages(&self.settings.system_prompt)
            .iter()
            .map(|(_, t)| t.chars().count())
            .sum();
        let tokens = crate::chats::estimate_tokens(chars);
        let ctx = self.effective_ctx() as usize;
        // Chat with documents: files that do not fit the context are searched instead of sent whole, and once a chat searches its files every question does (as in LM Studio)
        let embed = self.embedding_endpoint();
        if self.would_search(&msg.attachments, &msg.content) {
            msg.by_retrieval = true;
            msg.retrieving = true;
            let id = msg.id.clone();
            self.attachments.clear();
            self.chat.messages.push(msg);
            self.input.clear();
            self.chat.retitle();
            self.begin_retrieval(id);
            return;
        }
        if !self.attachments.is_empty() && tokens + 300 > ctx {
            let hint = if self.settings.engine == EngineKind::Forge && embed.is_none() { ", or let the chat search the files: Settings > Documents (an embedding model such as nomic-embed-text)" } else { "" };
            self.toast(format!("The conversation with the attached files is about {tokens} tokens and the context is {ctx}: raise the context length in Settings (it applies when a model loads) or attach less{hint}."));
            return;
        }
        self.attachments.clear();
        self.chat.messages.push(msg);
        self.input.clear();
        self.chat.retitle();
        self.begin_answer(ep, model);
    }

    /// Look for the passages of the files of this chat that answer the question of message `msg_id` (embedding the files first when the chat's index does not hold them), then answer.
    fn begin_retrieval(&mut self, msg_id: String) {
        let Some((ep, name)) = self.embedding_endpoint() else {
            if let Some(m) = self.chat.messages.iter_mut().find(|m| m.id == msg_id) {
                m.retrieving = false;
                m.error = Some("No embedding model is loaded to search the files with.".to_string());
            }
            return;
        };
        let Some(question) = self.chat.messages.iter().find(|m| m.id == msg_id).map(|m| m.content.clone()) else {
            return;
        };
        // every file of this chat that is searched: the index keeps what it already holds
        let docs: Vec<(String, String)> = self.chat.messages.iter().filter(|m| m.by_retrieval).flat_map(|m| m.attachments.iter().map(|a| (a.name.clone(), a.text.clone()))).collect();
        let chat_id = self.chat.id.clone();
        let index = self.doc_index.remove(&chat_id).unwrap_or_default();
        let (limit, threshold) = (self.settings.docs_limit.max(1) as usize, self.settings.docs_threshold);
        let file = self.engine.current().and_then(|l| l.embedding.clone()).map(|p| p.file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_default()).unwrap_or_default();
        let (doc_prefix, query_prefix) = rag::prefixes_for(&file);
        let progress = rag::Progress::default();
        // a request to summarize files that are too long to read at once: the chat model reads them part by part and summarizes each (the passages that best match such a request are a poor summary)
        let summary_ctx = if self.settings.docs_summarize && rag::wants_summary(&question) { Some(self.effective_ctx() as usize) } else { None };
        let chat = if summary_ctx.is_some() { self.active_endpoint() } else { None };
        let gen = self.settings.gen.clone();
        let cache = self.summaries.clone();
        self.retrieval = Some(Retrieval { chat_id: chat_id.clone(), msg_id: msg_id.clone(), started: Instant::now(), progress: progress.clone() });
        self.spawn(move |tx, wake| {
            let result = (|| {
                // (the files are cut into parts here, on this thread: a long one takes a moment)
                let targets = rag::files_named(&question, &docs);
                let plan = summary_ctx.and_then(|ctx| rag::summary_plan(&targets, ctx));
                if let (Some(plan), Some((chat_ep, chat_model))) = (plan, chat) {
                    progress.summarizing.store(true, Ordering::Relaxed);
                    let params = |prompt: &str| ChatParams {
                        model: chat_model.clone(),
                        messages: vec![(Role::User, prompt.to_string())],
                        gen: GenParams { temperature: 0.2, max_tokens: (plan.words * 2 + 40) as u32, thinking: ThinkingMode::Off, ..gen.clone() },
                    };
                    let mut ask = |prompt: &str| client::complete(&chat_ep, &params(prompt), &progress.cancel);
                    match rag::summarize(&cache, &targets, &plan, &progress, &mut ask) {
                        Ok(citations) => return Ok((index, citations)),
                        Err(e) if progress.cancel.load(Ordering::Relaxed) => return Err(e),
                        // the model could not do it (the server failed, say): the passages that match answer instead
                        Err(_) => {
                            progress.summarizing.store(false, Ordering::Relaxed);
                            progress.done.store(0, Ordering::Relaxed);
                            progress.total.store(0, Ordering::Relaxed);
                        }
                    }
                }
                let mut embed = |texts: &[String]| client::embed(&ep, &name, texts).map(|r| r.0);
                rag::search(index, &docs, &question, limit, threshold, (doc_prefix, query_prefix), &progress, &mut embed)
            })();
            let _ = tx.send(Msg::Retrieved { chat_id, msg_id, result });
            wake();
        });
    }

    /// Stop looking for passages. The message that asked for the search is taken back out of the chat (its text and files return to the input box when `restore` is set).
    fn cancel_retrieval(&mut self, restore: bool) {
        let Some(r) = self.retrieval.take() else {
            return;
        };
        r.progress.cancel.store(true, Ordering::Relaxed);
        if let Some(i) = self.chat.messages.iter().position(|m| m.id == r.msg_id) {
            let m = self.chat.messages.remove(i);
            if restore {
                self.input = m.content;
                self.attachments = m.attachments;
            }
        }
    }

    pub fn regenerate(&mut self) {
        if self.generation.is_some() {
            return;
        }
        // drop the last answer (and everything after the last user message) and ask again
        while matches!(self.chat.messages.last(), Some(m) if m.role == Role::Assistant) {
            self.chat.messages.pop();
        }
        if self.chat.messages.is_empty() {
            return;
        }
        let Some((ep, model)) = self.active_endpoint() else {
            self.toast("Load a model first.");
            return;
        };
        self.begin_answer(ep, model);
    }

    fn begin_answer(&mut self, ep: Endpoint, model: String) {
        self.editing = None;
        // the conversation has to fit into the context the model was loaded with: the oldest turns are left out when it does not (the chat keeps showing them)
        let ctx = self.effective_ctx() as usize;
        if self.settings.context_overflow == OverflowPolicy::Drop && ctx > 0 {
            let reserve = if self.settings.gen.max_tokens > 0 { self.settings.gen.max_tokens as usize } else { 1024 }.min(ctx / 4);
            let budget = (ctx * 95 / 100).saturating_sub(reserve);
            let dropped = self.chat.fit(&self.settings.system_prompt, budget);
            if dropped > 0 {
                self.toast(format!("The conversation no longer fits the context ({ctx} tokens): the oldest {dropped} messages are left out of what the model reads (they stay on screen)."));
            }
        }
        let mut reply = Message::new(Role::Assistant, "");
        reply.streaming = true;
        let messages = self.chat.api_messages(&self.settings.system_prompt);
        self.chat.messages.push(reply);
        self.chat.model = model.clone();
        let req = self.next_req;
        self.next_req += 1;
        let cancel = Arc::new(AtomicBool::new(false));
        self.generation = Some(Generation {
            req,
            cancel: cancel.clone(),
            started: Instant::now(),
            reasoning_started: None,
            progress: None,
        });
        self.scroll_to_bottom = true;
        let params = ChatParams {
            model,
            messages,
            gen: self.settings.gen.clone(),
        };
        let wake = self.wake();
        client::spawn_chat(
            ep,
            params,
            cancel,
            self.tx.clone(),
            move |ev| Msg::Chat { req, ev },
            move || wake(),
        );
        self.save_chat();
    }

    pub fn stop_generation(&mut self) {
        // Stop during the search of the files: the question goes back to the input box
        self.cancel_retrieval(true);
        self.generation_cancel();
        self.save_chat();
    }

    pub fn new_chat(&mut self) {
        self.generation_cancel();
        self.editing = None;
        self.attachments.clear();
        self.save_chat();
        self.chat = Chat::new();
        self.input.clear();
        self.focus_input = true;
    }

    pub fn open_chat(&mut self, id: &str) {
        if self.chat.id == id {
            return;
        }
        self.generation_cancel();
        self.editing = None;
        self.save_chat();
        if let Some(c) = self.store.load(id) {
            self.chat = c;
            self.scroll_to_bottom = true;
        }
    }

    pub fn delete_chat(&mut self, id: &str) {
        self.store.delete(id);
        if self.chat.id == id {
            self.generation_cancel();
            self.editing = None;
            self.chat = Chat::new();
        }
        self.chat_list = self.store.list();
    }

    /// Remove a `.gguf` file from the disk (never a folder, never a model that is loaded).
    pub fn delete_model(&mut self, path: &std::path::Path) {
        if !crate::util::is_gguf_file(path) {
            return;
        }
        if self.engine.current().is_some_and(|l| l.model == path) {
            self.toast("Eject the model first: it is loaded.");
            return;
        }
        let name = path
            .file_name()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_default();
        match std::fs::remove_file(path) {
            Ok(()) => {
                self.toast(format!("Deleted {name}."));
                self.rescan();
            }
            Err(e) => self.toast(format!("Could not delete {name}: {e}")),
        }
    }

    pub fn rename_chat(&mut self, id: &str, title: &str) {
        if self.opts.demo {
            return;
        }
        if self.chat.id == id {
            if self.chat.set_title(title) && !self.chat.messages.is_empty() {
                let _ = self.store.save(&self.chat);
            }
        } else if let Some(mut c) = self.store.load(id) {
            if c.set_title(title) {
                let _ = self.store.save(&c);
            }
        }
        self.chat_list = self.store.list();
    }

    pub fn edit_and_resend(&mut self, i: usize, text: &str) {
        if self.is_generating() {
            return;
        }
        let Some((ep, model)) = self.active_endpoint() else {
            self.toast("Load a model first (pick one in the bar above, or in My models).");
            return;
        };
        if self.chat.replace_from(i, text) {
            // a question about searched files finds its passages again for the new wording
            if self.chat.messages.last().is_some_and(|m| m.by_retrieval) && self.embedding_endpoint().is_some() {
                if let Some(m) = self.chat.messages.last_mut() {
                    m.retrieving = true;
                    let id = m.id.clone();
                    self.begin_retrieval(id);
                    return;
                }
            }
            self.begin_answer(ep, model);
        }
    }

    fn handle(&mut self, m: Msg) {
        match m {
            Msg::Scan(entries) => {
                self.scanning = false;
                self.library = entries;
                if let Some(p) = self.pending_load.take() {
                    self.load_model(p);
                } else if self.settings.auto_load_last
                    && !self.settings.last_model.is_empty()
                    && !self.engine.state().is_busy()
                {
                    let p = PathBuf::from(&self.settings.last_model);
                    if p.is_file() {
                        self.load_model(p);
                    }
                }
                if let Some(name) = self.opts.select_model.take() {
                    if let Some(e) = self
                        .library
                        .iter()
                        .find(|e| e.name.to_lowercase().contains(&name.to_lowercase()))
                        .cloned()
                    {
                        self.load_model(e.path);
                    }
                }
            }
            Msg::Hardware(h) => self.hw = h,
            Msg::Live { gpus, ram_free } => {
                self.live_busy = false;
                if !gpus.is_empty() {
                    self.hw.gpus = gpus;
                }
                if ram_free > 0 {
                    self.hw.ram_free = ram_free;
                }
            }
            Msg::Chat { req, ev } => self.on_chat_event(req, ev),
            Msg::HfSearch { seq, result } => {
                if seq == self.hf_seq {
                    self.hf_busy = false;
                    match result {
                        Ok(r) => {
                            self.hf_results = r;
                            if self.hf_results.is_empty() {
                                self.hf_error = "No GGUF repositories matched.".to_string();
                            }
                        }
                        Err(e) => self.hf_error = e,
                    }
                }
            }
            Msg::HfFiles { repo, result } => {
                if self.hf_repo.as_deref() == Some(repo.as_str()) {
                    self.hf_files_busy = false;
                    match result {
                        Ok(f) => self.hf_files = f,
                        Err(e) => self.hf_error = e,
                    }
                }
            }
            Msg::DownloadDone { id, result } => {
                let mut finished = None;
                if let Some(d) = self.downloads.iter_mut().find(|d| d.id == id) {
                    d.state = match &result {
                        Ok(p) => DlState::Done(p.clone()),
                        Err(e) => DlState::Failed(e.clone()),
                    };
                    finished = Some(d.file.file_name().to_string());
                }
                if let (Some(name), Ok(_)) = (finished, &result) {
                    self.toast(format!("Downloaded {name}"));
                    self.rescan();
                }
            }
            Msg::Models(r) => match r {
                Ok(list) => {
                    self.external_status = format!("connected, {} model(s)", list.len());
                    if !list.contains(&self.external_pick) {
                        self.external_pick = list.first().cloned().unwrap_or_default();
                    }
                    self.external_models = list;
                }
                Err(e) => {
                    self.external_status = e;
                    self.external_models.clear();
                }
            },
            Msg::Attached(r) => match r {
                Ok(a) => {
                    if self.attachments.iter().all(|x| x.name != a.name) {
                        self.attachments.push(a);
                    }
                    self.screen = Screen::Chat;
                    self.focus_input = true;
                }
                Err(e) => self.toast(e),
            },
            Msg::Retrieved { chat_id, msg_id, result } => {
                // a search that was stopped or left behind (another chat, a reloaded model) finds its message gone: the index it built is still kept
                let current = self.retrieval.as_ref().is_some_and(|r| r.msg_id == msg_id);
                let waiting = self.chat.id == chat_id && current && self.chat.messages.iter().any(|m| m.id == msg_id && m.retrieving);
                match result {
                    Ok((index, citations)) => {
                        self.doc_index.insert(chat_id, index);
                        if !waiting {
                            return;
                        }
                        self.retrieval = None;
                        if let Some(m) = self.chat.messages.iter_mut().find(|m| m.id == msg_id) {
                            m.citations = citations;
                            m.retrieving = false;
                        }
                        match self.active_endpoint() {
                            Some((ep, model)) => self.begin_answer(ep, model),
                            None => self.toast("The model was ejected while the files were searched."),
                        }
                    }
                    Err(e) => {
                        if !waiting {
                            return;
                        }
                        self.retrieval = None;
                        if let Some(m) = self.chat.messages.iter_mut().find(|m| m.id == msg_id) {
                            m.retrieving = false;
                            m.error = Some(format!("Could not search the files: {e}"));
                        }
                        self.save_chat();
                    }
                }
            }
            Msg::Embedded(r) => {
                self.embed_busy = false;
                match r {
                    Ok(out) => {
                        self.embed_error.clear();
                        self.embed_out = Some(out);
                    }
                    Err(e) => self.embed_error = e,
                }
            }
            Msg::Inspected(r) => {
                self.inspect_loading = false;
                let path = self
                    .inspect
                    .as_ref()
                    .map(|(p, _)| p.clone())
                    .unwrap_or_default();
                self.inspect = Some((path, r));
            }
        }
    }

    fn on_chat_event(&mut self, req: u64, ev: ChatEvent) {
        let Some(g) = self.generation.as_mut() else {
            return;
        };
        if g.req != req {
            return;
        }
        let Some(m) = self.chat.messages.last_mut() else {
            return;
        };
        match ev {
            ChatEvent::Reasoning(s) => {
                if g.reasoning_started.is_none() {
                    g.reasoning_started = Some(Instant::now());
                }
                m.reasoning.push_str(&s);
                self.scroll_to_bottom = true;
            }
            ChatEvent::Progress(p) => g.progress = Some(p),
            ChatEvent::Content(s) => {
                if let Some(t) = g.reasoning_started {
                    if m.thought_secs == 0.0 {
                        m.thought_secs = t.elapsed().as_secs_f32();
                    }
                }
                m.content.push_str(&s);
                self.scroll_to_bottom = true;
            }
            ChatEvent::Finished(stats) => {
                if let Some(t) = g.reasoning_started {
                    if m.thought_secs == 0.0 {
                        m.thought_secs = t.elapsed().as_secs_f32();
                    }
                }
                self.last_speed = Some(stats.tok_per_s);
                m.stats = Some(stats);
                m.streaming = false;
                self.generation = None;
                self.save_chat();
            }
            ChatEvent::Failed(e) => {
                m.error = Some(e);
                m.streaming = false;
                self.generation = None;
                self.save_chat();
            }
        }
    }

    /// Called once per frame before drawing.
    pub fn pump(&mut self) {
        while let Ok(m) = self.rx.try_recv() {
            self.handle(m);
        }
        self.engine.poll();
        if let (
            EngineState::Failed {
                message,
                during_load,
                ..
            },
            Some(l),
        ) = (self.engine.state(), self.engine.current().cloned())
        {
            // only a load that failed with a message that names the head is one to retry without it (a computer without a CUDA device fails the same way with and without it,
            // and an engine that dies while it answers is not a loading problem)
            if during_load && l.mtp.is_some() && message.contains("MTP head") {
                // the prediction head did not load (a head of another model, a broken file): run without it
                self.no_head.insert(l.model.clone());
                self.toast("The prediction head could not be used: starting the model without speculative decoding.");
                self.load_model(l.model.clone());
            }
        }
        self.toasts
            .retain(|t| t.at.elapsed() < Duration::from_secs(6));
        for d in &mut self.downloads {
            if d.state == DlState::Running {
                let done = d.progress.done.load(Ordering::Relaxed);
                let dt = d.last.0.elapsed().as_secs_f64();
                if dt >= 0.5 {
                    d.speed = (done.saturating_sub(d.last.1)) as f64 / dt;
                    d.last = (Instant::now(), done);
                }
            }
        }
        // a .gguf file or a folder dropped on the window: its folder joins the model folders
        let dropped: Vec<PathBuf> = self.ctx.input(|i| {
            i.raw
                .dropped_files
                .iter()
                .map(|f| f.path().to_path_buf())
                .collect()
        });
        for p in dropped {
            // a text file (not a model) is attached to the next message
            let is_model = p.is_dir()
                || p.extension()
                    .and_then(|e| e.to_str())
                    .is_some_and(|e| e.eq_ignore_ascii_case("gguf"));
            if !is_model {
                // a chat that can search its files takes much longer ones than one that has to send them whole
                let limit = if self.embedding_endpoint().is_some() { 24 << 20 } else { 2 << 20 };
                // a PDF, Office, OpenDocument, EPUB or HTML file is read on a thread of its own (a long PDF takes a moment)
                if crate::office::is_office(&p) {
                    let name = p.file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_else(|| "file".into());
                    self.toast(format!("Reading {name} ..."));
                    self.spawn(move |tx, wake| {
                        let r = crate::office::read_document(&p, limit).map(|text| crate::chats::Attachment { name, text });
                        let _ = tx.send(Msg::Attached(r));
                        wake();
                    });
                    continue;
                }
                match crate::util::read_text_file(&p, limit) {
                    Ok(text) => {
                        let name = p
                            .file_name()
                            .map(|s| s.to_string_lossy().to_string())
                            .unwrap_or_else(|| "file".into());
                        if self.attachments.iter().all(|a| a.name != name) {
                            self.attachments.push(crate::chats::Attachment {
                                name: name.clone(),
                                text,
                            });
                        }
                        self.screen = Screen::Chat;
                        self.focus_input = true;
                    }
                    Err(e) => self.toast(e),
                }
                continue;
            }
            let dir = if p.is_dir() {
                p.clone()
            } else {
                p.parent()
                    .map(|d| d.to_path_buf())
                    .unwrap_or_else(|| p.clone())
            };
            let shown = dir.display().to_string();
            if self.settings.model_dirs.contains(&shown) {
                self.toast("That folder is already in My models.");
            } else {
                self.settings.model_dirs.push(shown.clone());
                self.rescan();
                self.toast(format!("Added {shown} to My models."));
            }
            self.screen = Screen::Library;
        }
        // GPU memory in use and free RAM change while models load: look again every few seconds
        let every = if self.screen == Screen::Hardware || self.engine.state().is_busy() {
            Duration::from_secs(3)
        } else {
            Duration::from_secs(12)
        };
        if !self.live_busy && self.live_at.elapsed() > every && self.hw.threads > 0 {
            self.live_busy = true;
            self.live_at = Instant::now();
            self.spawn(|tx, wake| {
                let (gpus, ram_free) = hardware::detect_live();
                let _ = tx.send(Msg::Live { gpus, ram_free });
                wake();
            });
        }
        // idle unload: any request the engine logs (from this window or from another program) counts as activity
        let log_len = self.engine.log_len();
        if log_len != self.seen_log {
            if self
                .engine
                .log()
                .iter()
                .skip(self.seen_log.min(log_len))
                .any(|l| l.contains("POST /v1/") || l.contains("POST /completion"))
            {
                self.last_activity = Instant::now();
            }
            self.seen_log = log_len;
        }
        if self.generation.is_some() {
            self.last_activity = Instant::now();
        }
        let idle = Duration::from_secs(60 * self.settings.idle_eject_minutes as u64);
        if self.settings.idle_eject_minutes > 0
            && matches!(self.engine.state(), EngineState::Ready { .. })
            && self.last_activity.elapsed() > idle
        {
            let minutes = self.settings.idle_eject_minutes;
            self.eject();
            self.toast(format!(
                "The model was unloaded after {minutes} idle minutes (Settings)."
            ));
        }
        // settings: save a moment after the last change
        if self.settings != self.saved {
            let since = *self.dirty_since.get_or_insert_with(Instant::now);
            if since.elapsed() > Duration::from_millis(700) {
                if let Err(e) = self.settings.save() {
                    self.warn_once_saving(&e.to_string());
                }
                self.saved = self.settings.clone();
                self.dirty_since = None;
            }
            self.ctx.request_repaint_after(Duration::from_millis(300));
        }
        if let (Some(after), Some(g)) = (self.opts.stop_after, self.generation.as_ref()) {
            if g.started.elapsed().as_secs_f32() >= after {
                self.stop_generation();
            }
        }
        // test hooks: send a message once the model is ready
        if !self.pending_send.is_empty() && self.active_endpoint().is_some() && !self.is_generating() {
            self.input = self.pending_send.pop_front().unwrap_or_default();
            self.send();
        }
        // test hook: embed the sample lines once an embedding model is ready
        if self.pending_embed && self.active_is_embedding() && self.active_endpoint().is_some() && !self.embed_busy {
            self.pending_embed = false;
            self.embed_now();
        }
        if self.engine.state().is_busy()
            || self.generation.is_some()
            || self.embed_busy
            || self.retrieval.is_some()
            || self.downloads.iter().any(|d| d.state == DlState::Running)
        {
            self.ctx.request_repaint_after(Duration::from_millis(250));
        }
    }

    pub fn apply_actions(&mut self) {
        let actions = std::mem::take(&mut self.actions);
        for a in actions {
            match a {
                Action::NewChat => self.new_chat(),
                Action::OpenChat(id) => self.open_chat(&id),
                Action::DeleteChat(id) => self.delete_chat(&id),
                Action::Send => self.send(),
                Action::Stop => self.stop_generation(),
                Action::Regenerate => self.regenerate(),
                Action::RenameChat(id, title) => self.rename_chat(&id, &title),
                Action::DeleteModel(p) => self.delete_model(&p),
                Action::RemoveAttachment(i) => {
                    if i < self.attachments.len() {
                        self.attachments.remove(i);
                    }
                }
                Action::EditResend(i, text) => self.edit_and_resend(i, &text),
                Action::DeleteMessage(i) => {
                    if i < self.chat.messages.len() && self.generation.is_none() {
                        self.chat.messages.remove(i);
                        self.save_chat();
                    }
                }
                Action::LoadModel(p) => self.load_model(p),
                Action::Eject => self.eject(),
                Action::Rescan => self.rescan(),
                Action::AddDir(d) => {
                    let d = d.trim().to_string();
                    if !d.is_empty() && !self.settings.model_dirs.contains(&d) {
                        self.settings.model_dirs.push(d);
                        self.rescan();
                    }
                }
                Action::RemoveDir(i) => {
                    if i < self.settings.model_dirs.len() {
                        self.settings.model_dirs.remove(i);
                        self.rescan();
                    }
                }
                Action::Inspect(p) => self.inspect_model(p),
                Action::GetHead(p) => self.get_head(p),
                Action::RefreshExternal => self.refresh_external(),
                Action::HfSearch => self.hf_search(),
                Action::HfOpen(r) => self.hf_open(r),
                Action::Download { repo, file } => self.start_download(repo, file),
                Action::CancelDownload(id) => {
                    if let Some(d) = self.downloads.iter().find(|d| d.id == id) {
                        d.progress.cancel.store(true, Ordering::Relaxed);
                    }
                }
                Action::Embed => self.embed_now(),
                Action::Toast(t) => self.toast(t),
            }
        }
    }

    /// Screenshot plumbing for `--shot`.
    pub fn handle_shot(&mut self, ctx: &egui::Context) {
        let ready_for_reply = {
            let waiting = self.shot.as_ref().is_some_and(|s| s.wait_reply);
            !waiting
                || (self.opts.embed && !self.pending_embed && !self.embed_busy && (self.embed_out.is_some() || !self.embed_error.is_empty()))
                || (!self.opts.send.is_empty()
                    && self.pending_send.is_empty()
                    && !self.is_generating()
                    && self.chat.messages.last().is_some_and(|m| m.role == Role::Assistant && m.stats.is_some() || m.error.is_some()))
        };
        let Some(shot) = self.shot.as_mut() else {
            return;
        };
        if !shot.requested && Instant::now() >= shot.at && ready_for_reply && !self.scanning {
            shot.requested = true;
            ctx.send_viewport_cmd(egui::ViewportCommand::Screenshot(egui::UserData::default()));
        }
        let image = ctx.input(|i| {
            i.events.iter().find_map(|e| match e {
                egui::Event::Screenshot { image, .. } => Some(image.clone()),
                _ => None,
            })
        });
        if let Some(img) = image {
            let [w, h] = img.size;
            let path = shot.path.clone();
            if let Some(dir) = path.parent() {
                let _ = std::fs::create_dir_all(dir);
            }
            let result = (|| -> Result<(), String> {
                let file = std::fs::File::create(&path).map_err(|e| e.to_string())?;
                let mut enc = png::Encoder::new(std::io::BufWriter::new(file), w as u32, h as u32);
                enc.set_color(png::ColorType::Rgba);
                enc.set_depth(png::BitDepth::Eight);
                let mut writer = enc.write_header().map_err(|e| e.to_string())?;
                writer
                    .write_image_data(img.as_raw())
                    .map_err(|e| e.to_string())?;
                Ok(())
            })();
            match result {
                Ok(()) => println!("screenshot saved: {} ({w}x{h})", path.display()),
                Err(e) => eprintln!("screenshot failed: {e}"),
            }
            self.shot = None;
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
        }
        ctx.request_repaint_after(Duration::from_millis(200));
    }

    pub fn flush_settings(&mut self) {
        let _ = self.settings.save();
        self.save_chat();
    }
}

fn demo_chat() -> Chat {
    let mut c = Chat::new();
    c.title = "Explain KV caches".to_string();
    c.messages.push(Message::new(Role::User, "Explain the KV cache in two paragraphs, then show a tiny Rust example and a comparison table."));
    let mut a = Message::new(
        Role::Assistant,
        "## The KV cache\n\nWhen a transformer generates text it computes **keys** and **values** for every token it has seen. Instead of recomputing them for each new token, the engine *keeps* them in memory: that is the *KV cache*. It makes each new token cost one step instead of re-reading the whole conversation, at the price of memory that grows with the context length.\n\nOn a small GPU the cache is often the limit, so engines store it in fewer bits. See [the ForgeInfer notes](https://github.com/) for numbers.\n\n```rust\nstruct KvCache {\n    k: Vec<f16>, // [layers][tokens][kv_heads * head_dim]\n    v: Vec<f16>,\n    len: usize,\n}\n```\n\n| format | memory per 1k tokens | quality loss |\n|---|---|---|\n| fp16 | 100% | none |\n| int8 | 53% | negligible |\n| 8-bit keys, 4-bit values | 41% | small |\n| int4 | 28% | visible on some models |\n\n- Longer context needs more memory\n- Quantizing the cache halves it\n  - int8 is almost free\n  - int4 starts to show\n\n> A bigger cache is the price of a longer memory.\n",
    );
    a.reasoning = "The user wants a short explanation, a code sample and a table. Keep it concise and accurate.".to_string();
    a.thought_secs = 2.4;
    a.stats = Some(chats::Stats {
        completion_tokens: 212,
        prompt_tokens: 38,
        cached_tokens: 0,
        tok_per_s: 80.3,
        first_token_s: 0.21,
        total_s: 2.9,
        finish: "stop".into(),
    });
    c.messages.push(a);
    c.model = "Qwen3.5-4B-Q4_K_M".to_string();
    c
}

pub fn new_chat_id() -> String {
    new_id()
}

impl eframe::App for StudioApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        // FORGE_STUDIO_FPS=1: repaint continuously and print the frame rate (a performance probe for long chats)
        if let Some((t, n)) = self.fps.as_mut() {
            *n += 1;
            if t.elapsed().as_secs_f32() >= 2.0 {
                eprintln!("fps {:.1}", *n as f32 / t.elapsed().as_secs_f32());
                *t = Instant::now();
                *n = 0;
            }
            ui.ctx().request_repaint();
        }
        self.pump();
        crate::ui::draw(self, ui);
        self.apply_actions();
        let ctx = ui.ctx().clone();
        self.handle_shot(&ctx);
    }

    fn on_exit(&mut self, _gl: Option<&eframe::glow::Context>) {
        self.flush_settings();
        self.engine.stop();
    }

    fn clear_color(&self, _visuals: &egui::Visuals) -> [f32; 4] {
        crate::theme::BG.to_normalized_gamma_f32()
    }
}
