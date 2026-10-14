//! The local engine process: start `forge serve` (or llama.cpp's server) on a model, watch it come up, collect its log, stop it.

use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::process::{Child, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::client::{list_models, Endpoint};
use crate::settings::{EngineKind, KvMode, DEFAULT_PROMPT_CACHE_MB};
use crate::util::quiet_command;

#[derive(Clone, Debug, PartialEq)]
pub enum EngineState {
    Idle,
    Loading {
        model: String,
        since: Instant,
    },
    Ready {
        model: String,
        port: u16,
        load_secs: f32,
    },
    Failed {
        model: String,
        message: String,
        /// the process ended before the model was ready (an engine that dies while it answers is not a loading problem)
        during_load: bool,
    },
}

impl EngineState {
    pub fn is_busy(&self) -> bool {
        matches!(
            self,
            EngineState::Loading { .. } | EngineState::Ready { .. }
        )
    }
}

/// Everything needed to start one model.
#[derive(Clone, Debug)]
pub struct Launch {
    pub kind: EngineKind,
    pub binary: PathBuf,
    pub model: PathBuf,
    /// The name clients see (the `model` field of the API).
    pub name: String,
    pub port: u16,
    pub host: String,
    pub ctx: u32,
    pub kv: KvMode,
    pub depth: u32,
    /// The prediction head of a Qwen3.5 model (speculative decoding); `forge serve` only.
    pub mtp: Option<PathBuf>,
    /// Prompt-lookup drafting (`forge serve` only; on by default in the engine).
    pub lookup: bool,
    /// Host memory for the chats the engine is not working on (`forge serve` only; 0 = off).
    pub prompt_cache_mb: u32,
    pub api_key: String,
    pub extra: Vec<String>,
}

/// The command line (without the program) for a launch.
pub fn command_args(l: &Launch) -> Vec<String> {
    let s = |x: &str| x.to_string();
    let mut a: Vec<String> = Vec::new();
    match l.kind {
        EngineKind::LlamaCpp => {
            a.extend([
                s("-m"),
                l.model.display().to_string(),
                s("--host"),
                l.host.clone(),
                s("--port"),
                l.port.to_string(),
                s("-c"),
                l.ctx.to_string(),
                s("--alias"),
                l.name.clone(),
                s("-ngl"),
                s("999"),
            ]);
            if l.kv != KvMode::Fp16 {
                a.extend([s("-ctk"), s("q8_0"), s("-ctv"), s("q8_0")]);
            }
            if !l.api_key.is_empty() {
                a.extend([s("--api-key"), l.api_key.clone()]);
            }
        }
        _ => {
            a.extend([
                s("serve"),
                l.model.display().to_string(),
                s("--host"),
                l.host.clone(),
                s("--port"),
                l.port.to_string(),
                s("--ctx"),
                l.ctx.to_string(),
                s("--alias"),
                l.name.clone(),
            ]);
            if let Some(kv) = l.kv.flag() {
                a.extend([s("--kv"), s(kv)]);
            }
            if let Some(m) = &l.mtp {
                a.extend([s("--mtp"), m.display().to_string()]);
            }
            if l.depth > 0 {
                a.extend([s("--depth"), l.depth.to_string()]);
            }
            if !l.lookup {
                a.push(s("--no-lookup"));
            }
            if l.prompt_cache_mb != DEFAULT_PROMPT_CACHE_MB {
                a.extend([s("--prompt-cache-mb"), l.prompt_cache_mb.to_string()]);
            }
            if !l.api_key.is_empty() {
                a.extend([s("--api-key"), l.api_key.clone()]);
            }
        }
    }
    a.extend(l.extra.iter().cloned());
    a
}

const LOG_CAP: usize = 4000;

#[derive(Default)]
struct Shared {
    state: Option<EngineState>,
    log: Vec<String>,
    /// Bumped on every start and stop: a watcher of an older process must not touch the state.
    generation: u64,
}

pub struct EngineManager {
    shared: Arc<Mutex<Shared>>,
    child: Option<Child>,
    launch: Option<Launch>,
    /// when the process was seen to have ended: its last lines of output are still on their way through the reader threads
    exited: Option<(Instant, std::process::ExitStatus)>,
}

impl Default for EngineManager {
    fn default() -> Self {
        EngineManager {
            shared: Arc::new(Mutex::new(Shared {
                state: Some(EngineState::Idle),
                ..Default::default()
            })),
            child: None,
            launch: None,
            exited: None,
        }
    }
}

fn push_log(shared: &Arc<Mutex<Shared>>, line: String) {
    if let Ok(mut s) = shared.lock() {
        s.log.push(line);
        if s.log.len() > LOG_CAP {
            let drop_n = s.log.len() - LOG_CAP;
            s.log.drain(..drop_n);
        }
    }
}

fn pump(
    reader: impl std::io::Read + Send + 'static,
    shared: Arc<Mutex<Shared>>,
    wake: Arc<dyn Fn() + Send + Sync>,
) {
    std::thread::spawn(move || {
        for line in BufReader::new(reader).lines() {
            let Ok(line) = line else { break };
            push_log(&shared, line);
            wake();
        }
    });
}

impl EngineManager {
    pub fn state(&self) -> EngineState {
        self.shared
            .lock()
            .ok()
            .and_then(|s| s.state.clone())
            .unwrap_or(EngineState::Idle)
    }

    fn set_state(&self, st: EngineState) {
        if let Ok(mut s) = self.shared.lock() {
            s.state = Some(st);
        }
    }

    pub fn log(&self) -> Vec<String> {
        self.shared
            .lock()
            .map(|s| s.log.clone())
            .unwrap_or_default()
    }

    pub fn log_len(&self) -> usize {
        self.shared.lock().map(|s| s.log.len()).unwrap_or(0)
    }

    pub fn current(&self) -> Option<&Launch> {
        self.launch.as_ref()
    }

    pub fn app_log(&self, line: impl Into<String>) {
        push_log(&self.shared, line.into());
    }

    /// Stop whatever runs, then start `launch`. Returns an error text when the process cannot even be spawned.
    pub fn start(
        &mut self,
        launch: Launch,
        wake: Arc<dyn Fn() + Send + Sync>,
    ) -> Result<(), String> {
        self.stop();
        let generation = {
            let mut s = self
                .shared
                .lock()
                .map_err(|_| "engine state is poisoned".to_string())?;
            s.generation += 1;
            s.log.clear();
            s.generation
        };
        let args = command_args(&launch);
        push_log(
            &self.shared,
            format!("$ {} {}", launch.binary.display(), args.join(" ")),
        );
        let mut cmd = quiet_command(&launch.binary);
        cmd.args(&args)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .stdin(Stdio::null());
        if let Some(dir) = launch.binary.parent() {
            cmd.current_dir(dir);
        }
        crate::util::die_with_parent_before(&mut cmd);
        let mut child = cmd
            .spawn()
            .map_err(|e| format!("cannot start {}: {e}", launch.binary.display()))?;
        crate::util::die_with_parent_after(&child);
        if let Some(out) = child.stdout.take() {
            pump(out, self.shared.clone(), wake.clone());
        }
        if let Some(err) = child.stderr.take() {
            pump(err, self.shared.clone(), wake.clone());
        }
        let since = Instant::now();
        self.set_state(EngineState::Loading {
            model: launch.name.clone(),
            since,
        });
        self.child = Some(child);
        self.exited = None;
        self.launch = Some(launch.clone());

        // watcher: ask the server for its model list until it answers
        let shared = self.shared.clone();
        let ep = Endpoint::local(launch.port, &launch.api_key);
        let name = launch.name.clone();
        let port = launch.port;
        std::thread::spawn(move || loop {
            std::thread::sleep(Duration::from_millis(350));
            let current = shared.lock().map(|s| (s.generation, s.state.clone())).ok();
            match current {
                Some((g, Some(EngineState::Loading { .. }))) if g == generation => {}
                _ => return,
            }
            if list_models(&ep).is_ok() {
                if let Ok(mut s) = shared.lock() {
                    if s.generation == generation {
                        s.state = Some(EngineState::Ready {
                            model: name.clone(),
                            port,
                            load_secs: since.elapsed().as_secs_f32(),
                        });
                    }
                }
                wake();
                return;
            }
        });
        Ok(())
    }

    /// Check whether the process ended by itself; call this every frame.
    pub fn poll(&mut self) {
        let Some(child) = self.child.as_mut() else {
            return;
        };
        if self.exited.is_none() {
            if let Ok(Some(status)) = child.try_wait() {
                self.exited = Some((Instant::now(), status));
            }
        }
        // the reader threads still carry the last lines (the reason of a failure is usually the last one): look again a moment later
        if let Some((at, status)) = self
            .exited
            .filter(|(at, _)| at.elapsed() >= Duration::from_millis(400))
        {
            let _ = at;
            self.exited = None;
            self.child = None;
            let log = self.log();
            let tail: Vec<&String> = log
                .iter()
                .rev()
                // the first line of the log is the command line (`$ forge serve ...`): that is not a reason
                .filter(|l| !l.trim().is_empty() && !l.starts_with("$ "))
                .take(3)
                .collect();
            let message = if tail.is_empty() {
                format!("the engine exited ({status})")
            } else {
                tail.iter()
                    .rev()
                    .map(|s| s.as_str())
                    .collect::<Vec<_>>()
                    .join(" | ")
            };
            let model = self
                .launch
                .as_ref()
                .map(|l| l.name.clone())
                .unwrap_or_default();
            let was_loading = matches!(self.state(), EngineState::Loading { .. });
            if status.success() && !was_loading {
                self.set_state(EngineState::Idle);
            } else {
                self.set_state(EngineState::Failed {
                    model,
                    message,
                    during_load: was_loading,
                });
            }
            if let Ok(mut s) = self.shared.lock() {
                s.generation += 1;
            }
        }
    }

    pub fn stop(&mut self) {
        self.exited = None;
        if let Ok(mut s) = self.shared.lock() {
            s.generation += 1;
        }
        if let Some(mut child) = self.child.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
        self.set_state(EngineState::Idle);
    }
}

impl Drop for EngineManager {
    fn drop(&mut self) {
        self.stop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn launch(kind: EngineKind) -> Launch {
        Launch {
            kind,
            binary: PathBuf::from("forge"),
            model: PathBuf::from("m.gguf"),
            name: "m".into(),
            port: 1234,
            host: "127.0.0.1".into(),
            ctx: 8192,
            kv: KvMode::Int8,
            depth: 3,
            mtp: Some(PathBuf::from("head.safetensors")),
            lookup: true,
            prompt_cache_mb: DEFAULT_PROMPT_CACHE_MB,
            api_key: "k".into(),
            extra: vec!["--threads".into(), "6".into()],
        }
    }

    #[test]
    fn forge_command_line() {
        let a = command_args(&launch(EngineKind::Forge));
        assert_eq!(a.join(" "), "serve m.gguf --host 127.0.0.1 --port 1234 --ctx 8192 --alias m --kv int8 --mtp head.safetensors --depth 3 --api-key k --threads 6");
    }

    #[test]
    fn the_lookup_can_be_switched_off() {
        let mut l = launch(EngineKind::Forge);
        l.lookup = false;
        let a = command_args(&l).join(" ");
        assert!(a.contains("--depth 3 --no-lookup --api-key k"), "{a}");
    }

    #[test]
    fn the_prompt_cache_size_is_passed_on_when_it_is_not_the_engines_default() {
        let mut l = launch(EngineKind::Forge);
        l.prompt_cache_mb = 0;
        let a = command_args(&l).join(" ");
        assert!(a.contains("--depth 3 --prompt-cache-mb 0 --api-key k"), "{a}");
        l.prompt_cache_mb = 4096;
        assert!(command_args(&l).join(" ").contains("--prompt-cache-mb 4096"));
        // llama.cpp has no such cache
        let mut other = launch(EngineKind::LlamaCpp);
        other.prompt_cache_mb = 0;
        assert!(!command_args(&other).join(" ").contains("prompt-cache"));
    }

    #[test]
    fn llama_cpp_command_line() {
        let a = command_args(&launch(EngineKind::LlamaCpp));
        assert_eq!(a.join(" "), "-m m.gguf --host 127.0.0.1 --port 1234 -c 8192 --alias m -ngl 999 -ctk q8_0 -ctv q8_0 --api-key k --threads 6");
    }

    #[test]
    fn a_missing_binary_is_an_error_not_a_panic() {
        let mut m = EngineManager::default();
        let mut l = launch(EngineKind::Forge);
        l.binary = PathBuf::from("/definitely/not/here/forge-xyz");
        assert!(m.start(l, Arc::new(|| {})).is_err());
        assert_eq!(m.state(), EngineState::Idle);
    }
}
