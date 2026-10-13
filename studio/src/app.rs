//! The application state and what happens to it: loading models, chatting, downloading, scanning. The screens in `ui` only draw this state and queue `Action`s.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::Arc;
use std::time::{Duration, Instant};

use eframe::egui;

use crate::chats::{self, Chat, ChatMeta, Message, Role, Store};
use crate::client::{self, ChatEvent, ChatParams, Endpoint};
use crate::engine::{EngineManager, EngineState, Launch};
use crate::hardware::{self, HardwareInfo};
use crate::hf::{self, Progress, Repo, RepoFile};
use crate::library::{self, Inspect, ModelEntry};
use crate::settings::{self, data_dir, EngineKind, Settings};
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
}

/// The answer being generated right now.
pub struct Generation {
    pub req: u64,
    pub cancel: Arc<AtomicBool>,
    pub started: Instant,
    pub reasoning_started: Option<Instant>,
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
    pub send: Option<String>,
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
    pending_send: Option<String>,
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
            pending_send: opts.send.clone(),
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
        let _ = self.store.save(&self.chat);
        self.chat_list = self.store.list();
    }

    pub fn is_generating(&self) -> bool {
        self.generation.is_some()
    }

    pub fn send(&mut self) {
        let text = self.input.trim().to_string();
        if text.is_empty() || self.generation.is_some() {
            return;
        }
        let Some((ep, model)) = self.active_endpoint() else {
            self.toast("Load a model first (pick one in the bar above, or in My models).");
            return;
        };
        self.chat.messages.push(Message::new(Role::User, text));
        self.input.clear();
        self.chat.retitle();
        self.begin_answer(ep, model);
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
        self.generation_cancel();
        self.save_chat();
    }

    pub fn new_chat(&mut self) {
        self.generation_cancel();
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
            self.chat = Chat::new();
        }
        self.chat_list = self.store.list();
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
        if let (EngineState::Failed { .. }, Some(l)) =
            (self.engine.state(), self.engine.current().cloned())
        {
            if l.mtp.is_some() {
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
                let _ = self.settings.save();
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
        if self.pending_send.is_some()
            && self.active_endpoint().is_some()
            && self.generation.is_none()
        {
            self.input = self.pending_send.take().unwrap_or_default();
            self.send();
        }
        if self.engine.state().is_busy()
            || self.generation.is_some()
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
                Action::Toast(t) => self.toast(t),
            }
        }
    }

    /// Screenshot plumbing for `--shot`.
    pub fn handle_shot(&mut self, ctx: &egui::Context) {
        let ready_for_reply = {
            let waiting = self.shot.as_ref().is_some_and(|s| s.wait_reply);
            !waiting
                || (self.opts.send.is_some()
                    && self.pending_send.is_none()
                    && self.generation.is_none()
                    && self.chat.messages.iter().any(|m| {
                        m.role == Role::Assistant && m.stats.is_some() || m.error.is_some()
                    }))
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
        "## The KV cache\n\nWhen a transformer generates text it computes **keys** and **values** for every token it has seen. Instead of recomputing them for each new token, the engine *keeps* them in memory: that is the *KV cache*. It makes each new token cost one step instead of re-reading the whole conversation, at the price of memory that grows with the context length.\n\nOn a small GPU the cache is often the limit, so engines store it in fewer bits. See [the ForgeInfer notes](https://github.com/) for numbers.\n\n```rust\nstruct KvCache {\n    k: Vec<f16>, // [layers][tokens][kv_heads * head_dim]\n    v: Vec<f16>,\n    len: usize,\n}\n```\n\n| format | memory per 1k tokens | quality loss |\n|---|---|---|\n| fp16 | 100% | none |\n| int8 | 50% | negligible |\n| int4 | 25% | small |\n\n- Longer context needs more memory\n- Quantizing the cache halves it\n  - int8 is almost free\n  - int4 starts to show\n\n> A bigger cache is the price of a longer memory.\n",
    );
    a.reasoning = "The user wants a short explanation, a code sample and a table. Keep it concise and accurate.".to_string();
    a.thought_secs = 2.4;
    a.stats = Some(chats::Stats {
        completion_tokens: 212,
        prompt_tokens: 38,
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
