//! Settings (one JSON file in the app's data folder) and the places the app looks for the engine and for models.

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

use crate::util::home_dir;

/// Which program runs the model.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
pub enum EngineKind {
    /// The ForgeInfer engine (`forge serve`): CUDA GPUs, GGUF files.
    #[default]
    Forge,
    /// A llama.cpp `llama-server` binary: any GPU or CPU, GGUF files.
    LlamaCpp,
    /// Any OpenAI-compatible server that is already running (Ollama, LM Studio, vLLM, a homelab ...).
    External,
}

impl EngineKind {
    pub const ALL: [EngineKind; 3] = [EngineKind::Forge, EngineKind::LlamaCpp, EngineKind::External];

    pub fn label(self) -> &'static str {
        match self {
            EngineKind::Forge => "ForgeInfer engine (CUDA)",
            EngineKind::LlamaCpp => "llama.cpp server",
            EngineKind::External => "Connect to a running server",
        }
    }
}

/// How the KV cache (the attention memory of a conversation) is stored.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
pub enum KvMode {
    #[default]
    Fp16,
    Int8,
    Int4,
}

impl KvMode {
    pub const ALL: [KvMode; 3] = [KvMode::Fp16, KvMode::Int8, KvMode::Int4];

    pub fn label(self) -> &'static str {
        match self {
            KvMode::Fp16 => "16-bit (exact)",
            KvMode::Int8 => "8-bit (half the memory, negligible loss)",
            KvMode::Int4 => "4-bit (a quarter of the memory)",
        }
    }

    pub fn flag(self) -> Option<&'static str> {
        match self {
            KvMode::Fp16 => None,
            KvMode::Int8 => Some("int8"),
            KvMode::Int4 => Some("int4"),
        }
    }
}

/// Whether thinking models reason before they answer.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
pub enum ThinkingMode {
    /// Leave it to the model's own chat template (Qwen3.5 small models answer directly, bigger ones think).
    #[default]
    Default,
    On,
    Off,
}

impl ThinkingMode {
    pub const ALL: [ThinkingMode; 3] = [ThinkingMode::Default, ThinkingMode::On, ThinkingMode::Off];

    pub fn label(self) -> &'static str {
        match self {
            ThinkingMode::Default => "Model default",
            ThinkingMode::On => "Always think first",
            ThinkingMode::Off => "Answer directly",
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(default)]
pub struct GenParams {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: u32,
    pub min_p: f32,
    pub repeat_penalty: f32,
    /// 0 = no limit (the server's default).
    pub max_tokens: u32,
    /// Let thinking models (Qwen3, Qwen3.5) reason before they answer.
    #[serde(deserialize_with = "de_thinking")]
    pub thinking: ThinkingMode,
}

/// Settings files of earlier versions stored a boolean here.
fn de_thinking<'de, D: serde::Deserializer<'de>>(d: D) -> Result<ThinkingMode, D::Error> {
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum Raw {
        Flag(bool),
        Name(String),
    }
    Ok(match Raw::deserialize(d)? {
        Raw::Flag(true) => ThinkingMode::On,
        Raw::Flag(false) => ThinkingMode::Off,
        Raw::Name(n) => match n.as_str() {
            "on" => ThinkingMode::On,
            "off" => ThinkingMode::Off,
            _ => ThinkingMode::Default,
        },
    })
}

impl Default for GenParams {
    fn default() -> Self {
        GenParams { temperature: 0.7, top_p: 0.9, top_k: 40, min_p: 0.0, repeat_penalty: 1.05, max_tokens: 0, thinking: ThinkingMode::Default }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(default)]
pub struct Settings {
    /// Folders scanned for `.gguf` files.
    pub model_dirs: Vec<String>,
    /// Where downloads from Hugging Face go.
    pub download_dir: String,
    pub engine: EngineKind,
    /// Path of the `forge` binary; empty = look next to this app, then in the PATH.
    pub forge_binary: String,
    /// Path of llama.cpp's `llama-server`.
    pub llama_server: String,
    /// Base URL of the server used when the engine is `External`, e.g. `http://192.168.1.x:8080`.
    pub external_url: String,
    pub external_key: String,
    /// Context length the engine allocates when it loads a model.
    pub context_len: u32,
    pub kv_cache: KvMode,
    /// Tokens drafted per step by a model's own prediction head (0 = the engine's default).
    pub spec_depth: u32,
    /// Use a Qwen3.5 model's prediction head for speculative decoding when one is found next to the model (the ForgeInfer engine only).
    pub speculative: bool,
    /// Port of the local server (the default is the one LM Studio uses, so tools that know it just work).
    pub port: u16,
    /// Listen on every network interface instead of this computer only.
    pub serve_on_lan: bool,
    /// Required from clients when set (sent as `Authorization: Bearer ...`).
    pub api_key: String,
    pub gen: GenParams,
    pub system_prompt: String,
    pub last_model: String,
    pub auto_load_last: bool,
    /// Unload the model after this many idle minutes (0 = never): frees the GPU for games and other programs.
    pub idle_eject_minutes: u32,
    /// Extra command line arguments for the engine, one string, split on spaces.
    pub extra_args: String,
    /// Interface zoom (1.0 = normal).
    pub zoom: f32,
}

impl Default for Settings {
    fn default() -> Self {
        let data = data_dir();
        Settings {
            model_dirs: default_model_dirs(&data).iter().map(|p| p.display().to_string()).collect(),
            download_dir: data.join("models").display().to_string(),
            engine: if cfg!(target_os = "macos") { EngineKind::LlamaCpp } else { EngineKind::Forge },
            forge_binary: String::new(),
            llama_server: String::new(),
            external_url: "http://127.0.0.1:11434".to_string(),
            external_key: String::new(),
            context_len: 8192,
            kv_cache: KvMode::Fp16,
            spec_depth: 0,
            speculative: true,
            port: 1234,
            serve_on_lan: false,
            api_key: String::new(),
            gen: GenParams::default(),
            system_prompt: String::new(),
            last_model: String::new(),
            auto_load_last: false,
            idle_eject_minutes: 0,
            extra_args: String::new(),
            zoom: 1.0,
        }
    }
}

/// The app's own folder: settings, chats, downloaded models. `FORGE_STUDIO_HOME` overrides it; a folder named `data` next to the program
/// makes the app portable (everything stays beside the program).
pub fn data_dir() -> PathBuf {
    if let Some(p) = std::env::var_os("FORGE_STUDIO_HOME") {
        return PathBuf::from(p);
    }
    if let Some(dir) = std::env::current_exe().ok().and_then(|e| e.parent().map(Path::to_path_buf)) {
        let portable = dir.join("data");
        if portable.is_dir() {
            return portable;
        }
    }
    if cfg!(windows) {
        if let Some(p) = std::env::var_os("APPDATA") {
            return PathBuf::from(p).join("ForgeStudio");
        }
        home_dir().join("AppData").join("Roaming").join("ForgeStudio")
    } else if cfg!(target_os = "macos") {
        home_dir().join("Library").join("Application Support").join("ForgeStudio")
    } else if let Some(p) = std::env::var_os("XDG_CONFIG_HOME") {
        PathBuf::from(p).join("forge-studio")
    } else {
        home_dir().join(".config").join("forge-studio")
    }
}

/// Where models usually live: the app's own folder and, when they exist, the folders of LM Studio.
pub fn default_model_dirs(data: &Path) -> Vec<PathBuf> {
    let mut v = vec![data.join("models")];
    for p in [home_dir().join(".lmstudio").join("models"), home_dir().join(".cache").join("lm-studio").join("models")] {
        if p.is_dir() {
            v.push(p);
        }
    }
    v
}

pub fn settings_path() -> PathBuf {
    data_dir().join("settings.json")
}

impl Settings {
    pub fn load() -> Settings {
        match std::fs::read_to_string(settings_path()) {
            Ok(text) => match serde_json::from_str(&text) {
                Ok(settings) => settings,
                Err(_) => {
                    // keep what the user had: the next save would overwrite it
                    let _ = std::fs::rename(settings_path(), settings_path().with_extension("json.broken"));
                    Settings::default()
                }
            },
            Err(_) => Settings::default(),
        }
    }

    pub fn save(&self) -> std::io::Result<()> {
        let path = settings_path();
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let tmp = path.with_extension("json.tmp");
        std::fs::write(&tmp, serde_json::to_string_pretty(self).unwrap_or_default())?;
        std::fs::rename(&tmp, &path)
    }

    pub fn dirs(&self) -> Vec<PathBuf> {
        self.model_dirs.iter().filter(|s| !s.trim().is_empty()).map(|s| PathBuf::from(s.trim())).collect()
    }

    /// The URL of the server the chat talks to, without a trailing slash and without `/v1`.
    pub fn external_base(&self) -> String {
        let u = self.external_url.trim().trim_end_matches('/');
        u.strip_suffix("/v1").unwrap_or(u).to_string()
    }
}

/// The `forge` engine binary: the path in the settings, `FORGE_BIN`, next to this app, a build folder of the source tree, then the PATH.
pub fn find_forge(settings: &Settings) -> Option<PathBuf> {
    let exe = if cfg!(windows) { "forge.exe" } else { "forge" };
    let given = settings.forge_binary.trim();
    if !given.is_empty() {
        let p = PathBuf::from(given);
        return p.is_file().then_some(p);
    }
    if let Some(p) = std::env::var_os("FORGE_BIN") {
        let p = PathBuf::from(p);
        if p.is_file() {
            return Some(p);
        }
    }
    if let Ok(me) = std::env::current_exe() {
        if let Some(dir) = me.parent() {
            let mut candidates = vec![dir.join(exe)];
            // running from studio/target/<profile>/ inside the source tree: the engine is built in ../../../target/release
            for up in dir.ancestors().take(5) {
                candidates.push(up.join("target").join("release").join(exe));
            }
            for c in candidates {
                if c.is_file() {
                    return Some(c);
                }
            }
        }
    }
    find_in_path(exe)
}

pub fn find_in_path(name: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    std::env::split_paths(&path).map(|d| d.join(name)).find(|p| p.is_file())
}

/// llama.cpp's server: the path in the settings, then the PATH.
pub fn find_llama_server(settings: &Settings) -> Option<PathBuf> {
    let given = settings.llama_server.trim();
    if !given.is_empty() {
        let p = PathBuf::from(given);
        return p.is_file().then_some(p);
    }
    let name = if cfg!(windows) { "llama-server.exe" } else { "llama-server" };
    find_in_path(name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn settings_survive_a_round_trip_and_missing_fields_get_defaults() {
        let mut s = Settings::default();
        s.gen.temperature = 0.3;
        s.kv_cache = KvMode::Int8;
        let text = serde_json::to_string(&s).unwrap();
        let back: Settings = serde_json::from_str(&text).unwrap();
        assert_eq!(s, back);
        let partial: Settings = serde_json::from_str(r#"{"port": 4321}"#).unwrap();
        assert_eq!(partial.port, 4321);
        assert_eq!(partial.context_len, 8192);
    }

    #[test]
    fn the_old_boolean_thinking_setting_still_loads() {
        let on: Settings = serde_json::from_str(r#"{"gen": {"thinking": true}, "port": 4000}"#).unwrap();
        assert_eq!(on.gen.thinking, ThinkingMode::On);
        assert_eq!(on.port, 4000);
        let off: Settings = serde_json::from_str(r#"{"gen": {"thinking": false}}"#).unwrap();
        assert_eq!(off.gen.thinking, ThinkingMode::Off);
        let named: Settings = serde_json::from_str(r#"{"gen": {"thinking": "default"}}"#).unwrap();
        assert_eq!(named.gen.thinking, ThinkingMode::Default);
    }

    #[test]
    fn the_external_base_drops_v1_and_the_slash() {
        let mut s = Settings::default();
        s.external_url = "http://host:8080/v1/".into();
        assert_eq!(s.external_base(), "http://host:8080");
        s.external_url = "http://host:8080".into();
        assert_eq!(s.external_base(), "http://host:8080");
    }
}
