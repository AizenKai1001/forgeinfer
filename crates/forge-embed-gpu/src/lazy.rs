//! Choosing where an embedding model runs, and loading it only when the first text arrives.
//!
//! A chat server that has an embedding model next to its chat model should not pay for it until somebody asks for an embedding: the CPU copy of nomic-embed-text is 550 MB of memory and the GPU copy 0.4 GB of
//! video memory, which on an 8 GB card is what an automatic context would have used. `LazyEncoder` reads the model's metadata at start (a few milliseconds) and loads the weights on the first request, on the GPU when
//! it has room for them and on the CPU when it has not.

use crate::GpuEmbedder;
use forge_embed::{read_info, Config, Embedder, Embedding, Encode, Info};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

/// Where an embedding model should run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Place {
    Cpu,
    /// On the GPU, and an error when that is not possible.
    Gpu,
    /// On the GPU when it works and has room (`FORGE_EMBED_MIN_FREE_MB`, 700 by default), else on the CPU.
    Auto,
}

impl Place {
    pub fn parse(s: &str) -> Option<Place> {
        match s {
            "cpu" => Some(Place::Cpu),
            "gpu" => Some(Place::Gpu),
            "auto" => Some(Place::Auto),
            _ => None,
        }
    }
}

/// Video memory the GPU copy needs to be worth trying: its weights (275 MB), its working buffers (about 130 MB) and some room for the driver.
fn min_free_bytes() -> usize {
    std::env::var("FORGE_EMBED_MIN_FREE_MB").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(700) << 20
}

/// Load an embedding model now, where `place` says (see `Place`). `ordinal` is the GPU's number.
pub fn load_encoder(path: &Path, place: Place, ordinal: usize) -> Result<Arc<dyn Encode>, String> {
    let model = Embedder::load(path).map_err(|e| e.to_string())?;
    if place == Place::Cpu {
        return Ok(Arc::new(model));
    }
    if place == Place::Auto {
        // a card without room for it (an automatic context filled it) or without a working driver: the CPU, quietly
        let room = forge_cuda::Device::open(ordinal).and_then(|d| d.mem_info()).map(|(free, _)| free);
        match room {
            Ok(free) if free >= min_free_bytes() => {}
            Ok(free) => {
                eprintln!("[embed] only {} MB of video memory are free: the embedding model runs on the CPU", free >> 20);
                return Ok(Arc::new(model));
            }
            Err(e) => {
                eprintln!("[embed] no GPU for the embedding model ({e}): it runs on the CPU");
                return Ok(Arc::new(model));
            }
        }
    }
    match GpuEmbedder::new(model.into_parts(), ordinal) {
        Ok(g) => Ok(Arc::new(g)),
        Err(e) if place == Place::Gpu => Err(format!("the embedding model does not run on GPU {ordinal}: {e}")),
        Err(e) => {
            eprintln!("[embed] no room on the GPU for the embedding model ({e}): it runs on the CPU");
            Embedder::load(path).map(|m| Arc::new(m) as Arc<dyn Encode>).map_err(|e| e.to_string())
        }
    }
}

/// An embedding model whose weights are loaded by the first request (its configuration is known at once).
pub struct LazyEncoder {
    path: PathBuf,
    place: Place,
    ordinal: usize,
    info: Info,
    loaded: Mutex<Option<Arc<dyn Encode>>>,
}

impl LazyEncoder {
    pub fn new(path: impl Into<PathBuf>, place: Place, ordinal: usize) -> Result<LazyEncoder, String> {
        let path = path.into();
        let info = read_info(&path).map_err(|e| e.to_string())?;
        Ok(LazyEncoder { path, place, ordinal, info, loaded: Mutex::new(None) })
    }

    /// The loaded model: loads it now when this is the first call (the other callers wait for that), and tries again next time when the load failed.
    fn get(&self) -> Result<Arc<dyn Encode>, String> {
        let mut slot = self.loaded.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(e) = slot.as_ref() {
            return Ok(e.clone());
        }
        let t = std::time::Instant::now();
        let e = load_encoder(&self.path, self.place, self.ordinal)?;
        eprintln!("[serve] embedding model {} loaded on first use in {:.1} s ({})", e.name(), t.elapsed().as_secs_f64(), e.backend());
        *slot = Some(e.clone());
        Ok(e)
    }
}

impl Encode for LazyEncoder {
    fn embed_many(&self, texts: &[&str]) -> Result<Vec<Embedding>, String> {
        self.get()?.embed_many(texts)
    }
    fn config(&self) -> &Config {
        &self.info.cfg
    }
    fn name(&self) -> &str {
        &self.info.name
    }
    fn backend(&self) -> String {
        match self.loaded.lock().unwrap_or_else(|p| p.into_inner()).as_ref() {
            Some(e) => e.backend(),
            None => match self.place {
                Place::Cpu => "CPU, loads on first use".to_string(),
                Place::Gpu => "GPU, loads on first use".to_string(),
                Place::Auto => "loads on first use: on the GPU when it has room, else on the CPU".to_string(),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn places_are_parsed() {
        assert_eq!(Place::parse("cpu"), Some(Place::Cpu));
        assert_eq!(Place::parse("gpu"), Some(Place::Gpu));
        assert_eq!(Place::parse("auto"), Some(Place::Auto));
        assert_eq!(Place::parse("tpu"), None);
    }

    #[test]
    fn a_missing_model_is_an_error_at_once() {
        assert!(LazyEncoder::new("no/such/model.gguf", Place::Auto, 0).is_err());
    }
}
