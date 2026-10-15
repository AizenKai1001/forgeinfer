//! The model library: find `.gguf` files (the app's folder, LM Studio's, Ollama's blobs), read what their headers say, and take a model apart for the inspector.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use forge_gguf::{Gguf, Value};

use crate::util::home_dir;

#[derive(Clone, Debug, PartialEq)]
pub struct ModelEntry {
    pub path: PathBuf,
    /// A readable name: the model's own name, else the file name.
    pub name: String,
    pub size: u64,
    pub arch: String,
    pub params: u64,
    /// `Q4_K_M`, `Q8_0`, `F16` ...
    pub quant: String,
    /// Context length the model was trained for.
    pub ctx_train: u64,
    pub layers: u64,
    /// `tokenizer.ggml.model`: `gpt2` is byte-level BPE (Qwen, Llama 3, SmolLM), `llama` is SentencePiece.
    pub tokenizer: String,
    pub source: String,
    /// A multimodal projector, not a language model.
    pub projector: bool,
    /// The header could not be read.
    pub error: Option<String>,
    pub modified: u64,
}

/// Architectures the ForgeInfer engine can run today.
pub const FORGE_ARCHS: [&str; 4] = ["llama", "qwen2", "qwen3", "qwen35"];

/// Embedding models the engine serves on `/v1/embeddings` (no chat: they turn text into vectors): `nomic-bert` (nomic-embed-text) and `bert` (bge, MiniLM, e5, gte ...).
pub const EMBEDDING_ARCHS: [&str; 2] = ["nomic-bert", "bert"];

impl ModelEntry {
    /// An embedding model: Studio shows an embedding playground for it instead of a chat. A `bert` file counts only with the WordPiece vocabulary the engine reads (bge-m3 and the multilingual
    /// e5 models are BERT-shaped too, but use a SentencePiece vocabulary).
    pub fn is_embedding(&self) -> bool {
        !self.projector && self.error.is_none() && EMBEDDING_ARCHS.contains(&self.arch.as_str()) && (self.arch != "bert" || self.tokenizer == "bert")
    }

    pub fn forge_supported(&self) -> bool {
        self.forge_issue().is_none()
    }

    /// Why the ForgeInfer engine cannot run this file, if it cannot.
    pub fn forge_issue(&self) -> Option<String> {
        if self.error.is_some() {
            return Some("the file could not be read".to_string());
        }
        if self.projector {
            return Some("a vision projector goes with a model, it is not a model".to_string());
        }
        if self.is_embedding() {
            // WordPiece tokenizer, read by the engine's embedding encoder
            return None;
        }
        if self.arch == "bert" {
            return Some(format!(
                "this BERT model has a \"{}\" vocabulary; only the WordPiece ones (bge-*-en, MiniLM, e5-*-v2, gte, nomic-embed-text) are supported yet",
                if self.tokenizer.is_empty() { "unknown" } else { &self.tokenizer }
            ));
        }
        if !FORGE_ARCHS.contains(&self.arch.as_str()) {
            return Some(format!(
                "the \"{}\" architecture is not supported yet (llama, qwen2, qwen3 and qwen3.5 are, and embedding models of the nomic-embed-text and BERT families: bge, MiniLM, e5, gte)",
                self.arch
            ));
        }
        if self.tokenizer != "gpt2" {
            return Some(format!(
                "its tokenizer type is \"{}\"; only byte-level BPE (gpt2) is supported yet",
                if self.tokenizer.is_empty() {
                    "none"
                } else {
                    &self.tokenizer
                }
            ));
        }
        None
    }
}

/// The embedding model LM Studio ships (nomic-embed-text-v1.5), used where it is when LM Studio is installed: `~/.lmstudio/.internal/bundled-models/nomic-ai/nomic-embed-text-v1.5-GGUF/*.gguf`.
pub fn lmstudio_embedding_model() -> Option<PathBuf> {
    let dir = home_dir().join(".lmstudio").join(".internal").join("bundled-models").join("nomic-ai").join("nomic-embed-text-v1.5-GGUF");
    std::fs::read_dir(dir).ok()?.filter_map(|e| e.ok()).map(|e| e.path()).find(|p| crate::util::is_gguf_file(p))
}

fn ftype_name(n: u64) -> Option<&'static str> {
    Some(match n {
        0 => "F32",
        1 => "F16",
        2 => "Q4_0",
        3 => "Q4_1",
        7 => "Q8_0",
        8 => "Q5_0",
        9 => "Q5_1",
        10 => "Q2_K",
        11 => "Q3_K_S",
        12 => "Q3_K_M",
        13 => "Q3_K_L",
        14 => "Q4_K_S",
        15 => "Q4_K_M",
        16 => "Q5_K_S",
        17 => "Q5_K_M",
        18 => "Q6_K",
        19 => "IQ2_XXS",
        20 => "IQ2_XS",
        21 => "Q2_K_S",
        22 => "IQ3_XS",
        23 => "IQ3_XXS",
        24 => "IQ1_S",
        25 => "IQ4_NL",
        26 => "IQ3_S",
        27 => "IQ3_M",
        28 => "IQ2_S",
        29 => "IQ2_M",
        30 => "IQ4_XS",
        31 => "IQ1_M",
        32 => "BF16",
        _ => return None,
    })
}

/// The quantization of a file: its `general.file_type` when it has one, else the format that holds most of the bytes.
fn quant_of(g: &Gguf) -> String {
    if let Some(name) = g.get_u64("general.file_type").and_then(ftype_name) {
        return name.to_string();
    }
    dominant_dtype(g)
}

fn dominant_dtype(g: &Gguf) -> String {
    let mut by: BTreeMap<String, u64> = BTreeMap::new();
    for t in &g.tensors {
        if t.dims.len() >= 2 {
            *by.entry(t.dtype.name()).or_default() += t.checked_n_bytes().unwrap_or(0);
        }
    }
    by.into_iter()
        .max_by_key(|(_, b)| *b)
        .map(|(n, _)| n)
        .unwrap_or_else(|| "?".to_string())
}

fn arch_u64(g: &Gguf, key: &str) -> u64 {
    let arch = g.architecture().unwrap_or("");
    g.get_u64(&format!("{arch}.{key}")).unwrap_or(0)
}

fn modified_secs(p: &Path) -> u64 {
    std::fs::metadata(p)
        .and_then(|m| m.modified())
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Read one file's header into an entry.
pub fn read_entry(path: &Path, display: Option<String>, source: &str) -> ModelEntry {
    let size = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    let file_stem = path
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "model".to_string());
    let mut e = ModelEntry {
        path: path.to_path_buf(),
        name: display.unwrap_or(file_stem.clone()),
        size,
        arch: "?".to_string(),
        params: 0,
        quant: "?".to_string(),
        ctx_train: 0,
        layers: 0,
        tokenizer: String::new(),
        source: source.to_string(),
        projector: file_stem.to_lowercase().contains("mmproj"),
        error: None,
        modified: modified_secs(path),
    };
    match Gguf::open(path) {
        Ok(g) => {
            e.arch = g.architecture().unwrap_or("?").to_string();
            if e.arch == "clip" {
                e.projector = true;
            }
            e.quant = quant_of(&g);
            e.ctx_train = arch_u64(&g, "context_length");
            e.layers = arch_u64(&g, "block_count");
            e.tokenizer = g.get_str("tokenizer.ggml.model").unwrap_or("").to_string();
            e.params = g
                .get_u64("general.parameter_count")
                .filter(|&n| n > 0)
                .unwrap_or_else(|| g.tensors.iter().map(|t| t.n_elements() as u64).sum());
        }
        Err(err) => e.error = Some(err.to_string()),
    }
    e
}

fn walk(dir: &Path, depth: usize, out: &mut Vec<PathBuf>) {
    if depth > 6 {
        return;
    }
    let Ok(rd) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in rd.flatten() {
        let p = entry.path();
        let Ok(ft) = entry.file_type() else { continue };
        if ft.is_dir() {
            walk(&p, depth + 1, out);
        } else if p
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| e.eq_ignore_ascii_case("gguf"))
        {
            out.push(p);
        }
    }
}

fn source_label(dir: &Path) -> String {
    let s = dir.display().to_string().to_lowercase();
    if s.contains("lm-studio") || s.contains("lmstudio") {
        "LM Studio".to_string()
    } else if s.contains("forgestudio") || s.contains("forge-studio") {
        "Forge Studio".to_string()
    } else {
        dir.file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| "folder".to_string())
    }
}

/// Ollama keeps models as hash-named blobs; its manifests say which blob is the model of which name.
pub fn ollama_models_dir() -> PathBuf {
    std::env::var_os("OLLAMA_MODELS")
        .map(PathBuf::from)
        .unwrap_or_else(|| home_dir().join(".ollama").join("models"))
}

pub fn ollama_blobs() -> Vec<(String, PathBuf)> {
    let root = ollama_models_dir();
    let mut out = Vec::new();
    let manifests = root.join("manifests");
    let mut files = Vec::new();
    fn walk_all(dir: &Path, depth: usize, out: &mut Vec<PathBuf>) {
        if depth > 6 {
            return;
        }
        if let Ok(rd) = std::fs::read_dir(dir) {
            for e in rd.flatten() {
                let p = e.path();
                if p.is_dir() {
                    walk_all(&p, depth + 1, out);
                } else {
                    out.push(p);
                }
            }
        }
    }
    walk_all(&manifests, 0, &mut files);
    for f in files {
        let Ok(text) = std::fs::read_to_string(&f) else {
            continue;
        };
        let Ok(v) = serde_json::from_str::<serde_json::Value>(&text) else {
            continue;
        };
        let Some(layers) = v["layers"].as_array() else {
            continue;
        };
        for l in layers {
            if l["mediaType"].as_str() == Some("application/vnd.ollama.image.model") {
                if let Some(d) = l["digest"].as_str() {
                    let blob = root.join("blobs").join(d.replace(':', "-"));
                    if blob.is_file() {
                        // .../library/<name>/<tag>
                        let tag = f
                            .file_name()
                            .map(|s| s.to_string_lossy().to_string())
                            .unwrap_or_default();
                        let name = f
                            .parent()
                            .and_then(|p| p.file_name())
                            .map(|s| s.to_string_lossy().to_string())
                            .unwrap_or_default();
                        out.push((format!("{name}:{tag}"), blob));
                    }
                }
            }
        }
    }
    out
}

/// Every model in `dirs` (and in Ollama's store when `with_ollama`), newest first.
pub fn scan(dirs: &[PathBuf], with_ollama: bool) -> Vec<ModelEntry> {
    let mut jobs: Vec<(PathBuf, Option<String>, String)> = Vec::new();
    for d in dirs {
        let mut files = Vec::new();
        walk(d, 0, &mut files);
        let label = source_label(d);
        for f in files {
            if f.extension()
                .is_some_and(|e| e.eq_ignore_ascii_case("part"))
            {
                continue;
            }
            jobs.push((f, None, label.clone()));
        }
    }
    if with_ollama {
        for (name, blob) in ollama_blobs() {
            jobs.push((blob, Some(name), "Ollama".to_string()));
        }
    }
    jobs.sort_by(|a, b| a.0.cmp(&b.0));
    jobs.dedup_by(|a, b| a.0 == b.0);
    let mut entries: Vec<ModelEntry> = Vec::with_capacity(jobs.len());
    let workers = 4.min(jobs.len().max(1));
    let chunk = jobs.len().div_ceil(workers).max(1);
    std::thread::scope(|s| {
        let handles: Vec<_> = jobs
            .chunks(chunk)
            .map(|c| {
                s.spawn(move || {
                    c.iter()
                        .map(|(p, n, src)| read_entry(p, n.clone(), src))
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        for h in handles {
            if let Ok(v) = h.join() {
                entries.extend(v);
            }
        }
    });
    // vision projectors go last: they are not models to chat with
    entries.sort_by(|a, b| {
        a.projector
            .cmp(&b.projector)
            .then_with(|| b.modified.cmp(&a.modified))
            .then_with(|| a.name.cmp(&b.name))
    });
    entries
}

// ------------------------------------------------------------------------------------------------------------------ inspector

#[derive(Clone, Debug)]
pub struct TensorRow {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub bytes: u64,
}

#[derive(Clone, Debug)]
pub struct LayerRow {
    pub index: usize,
    pub kind: String,
    pub bytes: u64,
    /// The main weight format of the layer.
    pub dtype: String,
}

#[derive(Clone, Debug)]
pub struct Inspect {
    pub entry: ModelEntry,
    pub meta: Vec<(String, String)>,
    pub tensors: Vec<TensorRow>,
    pub layers: Vec<LayerRow>,
    /// (format, bytes, tensor count), biggest first.
    pub formats: Vec<(String, u64, usize)>,
    /// (what the tensors are, bytes), biggest first.
    pub roles: Vec<(String, u64)>,
    pub vocab: usize,
}

fn value_text(v: &Value) -> String {
    match v {
        Value::Str(s) => {
            let t: String = s.chars().take(160).collect();
            t.replace('\n', " ")
        }
        other => {
            if let Some(a) = other.as_array() {
                format!("[{} values]", a.len())
            } else if let Some(b) = other.as_bool() {
                b.to_string()
            } else if let Some(n) = other.as_u64() {
                n.to_string()
            } else if let Some(n) = other.as_i64() {
                n.to_string()
            } else if let Some(f) = other.as_f64() {
                format!("{f}")
            } else {
                "?".to_string()
            }
        }
    }
}

fn layer_of(name: &str) -> Option<usize> {
    let rest = name.strip_prefix("blk.")?;
    let end = rest.find('.')?;
    rest[..end].parse().ok()
}

fn role_of(name: &str) -> &'static str {
    let n = name.to_lowercase();
    if n.starts_with("token_embd") {
        "token embeddings"
    } else if n.starts_with("output_norm") || n.contains("norm") {
        "norms"
    } else if n.starts_with("output") {
        "output head"
    } else if n.contains("ssm") {
        "recurrent (DeltaNet) mixing"
    } else if n.contains("_exps") || n.contains("ffn_gate_inp") {
        "MoE experts"
    } else if n.contains("ffn") {
        "feed-forward"
    } else if n.contains("attn") {
        "attention"
    } else {
        "other"
    }
}

pub fn inspect(path: &Path, display: Option<String>, source: &str) -> Result<Inspect, String> {
    let g = Gguf::open(path).map_err(|e| e.to_string())?;
    let entry = read_entry(path, display, source);
    let mut meta = Vec::new();
    for (k, v) in &g.metadata {
        if k.starts_with("tokenizer.ggml.")
            && k != "tokenizer.ggml.model"
            && k != "tokenizer.ggml.pre"
            && !k.ends_with("_token_id")
        {
            continue;
        }
        if k.ends_with(".chat_template") {
            meta.push((
                k.clone(),
                format!("{} characters", v.as_str().map_or(0, str::len)),
            ));
            continue;
        }
        meta.push((k.clone(), value_text(v)));
    }
    let vocab = g
        .get("tokenizer.ggml.tokens")
        .and_then(|v| v.as_array())
        .map_or(0, |a| a.len());
    let mut tensors = Vec::with_capacity(g.tensors.len());
    let mut by_layer: BTreeMap<usize, (Vec<String>, u64, BTreeMap<String, u64>)> = BTreeMap::new();
    let mut formats: BTreeMap<String, (u64, usize)> = BTreeMap::new();
    let mut roles: BTreeMap<&'static str, u64> = BTreeMap::new();
    for t in &g.tensors {
        let bytes = t.checked_n_bytes().unwrap_or(0);
        let dtype = t.dtype.name();
        tensors.push(TensorRow {
            name: t.name.clone(),
            dtype: dtype.clone(),
            shape: t.shape(),
            bytes,
        });
        let f = formats.entry(dtype.clone()).or_default();
        f.0 += bytes;
        f.1 += 1;
        *roles.entry(role_of(&t.name)).or_default() += bytes;
        if let Some(i) = layer_of(&t.name) {
            let e = by_layer.entry(i).or_default();
            e.0.push(t.name.clone());
            e.1 += bytes;
            if t.dims.len() >= 2 {
                *e.2.entry(dtype).or_default() += bytes;
            }
        }
    }
    let layers = by_layer
        .into_iter()
        .map(|(index, (names, bytes, dts))| {
            let has = |s: &str| names.iter().any(|n| n.contains(s));
            let kind = if has("ssm") {
                "DeltaNet (recurrent)"
            } else if has("_exps") {
                "Attention + MoE"
            } else if has("attn_q") || has("attn_qkv") {
                "Attention"
            } else {
                "Other"
            };
            let dtype = dts
                .into_iter()
                .max_by_key(|(_, b)| *b)
                .map(|(n, _)| n)
                .unwrap_or_default();
            LayerRow {
                index,
                kind: kind.to_string(),
                bytes,
                dtype,
            }
        })
        .collect();
    let mut formats: Vec<(String, u64, usize)> =
        formats.into_iter().map(|(k, (b, n))| (k, b, n)).collect();
    formats.sort_by(|a, b| b.1.cmp(&a.1));
    let mut roles: Vec<(String, u64)> =
        roles.into_iter().map(|(k, b)| (k.to_string(), b)).collect();
    roles.sort_by(|a, b| b.1.cmp(&a.1));
    Ok(Inspect {
        entry,
        meta,
        tensors,
        layers,
        formats,
        roles,
        vocab,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn put_str(b: &mut Vec<u8>, s: &str) {
        b.extend((s.len() as u64).to_le_bytes());
        b.extend(s.as_bytes());
    }

    /// A tiny valid GGUF v3 file: arch "llama", 2 layers, one F32 tensor per kind.
    fn tiny_gguf(path: &Path, arch: &str) {
        tiny_gguf_with_tokenizer(path, arch, "gpt2");
    }

    fn tiny_gguf_with_tokenizer(path: &Path, arch: &str, tokenizer: &str) {
        let mut b: Vec<u8> = Vec::new();
        b.extend(b"GGUF");
        b.extend(3u32.to_le_bytes());
        let tensors: [(&str, [u64; 2]); 3] = [
            ("token_embd.weight", [4, 3]),
            ("blk.0.attn_q.weight", [4, 4]),
            ("blk.1.ffn_up.weight", [4, 2]),
        ];
        b.extend((tensors.len() as u64).to_le_bytes());
        b.extend(5u64.to_le_bytes());
        put_str(&mut b, "tokenizer.ggml.model");
        b.extend(8u32.to_le_bytes());
        put_str(&mut b, tokenizer);
        put_str(&mut b, "general.architecture");
        b.extend(8u32.to_le_bytes());
        put_str(&mut b, arch);
        put_str(&mut b, "general.name");
        b.extend(8u32.to_le_bytes());
        put_str(&mut b, "Tiny Test");
        put_str(&mut b, &format!("{arch}.block_count"));
        b.extend(4u32.to_le_bytes());
        b.extend(2u32.to_le_bytes());
        put_str(&mut b, &format!("{arch}.context_length"));
        b.extend(4u32.to_le_bytes());
        b.extend(2048u32.to_le_bytes());
        let mut offset = 0u64;
        for (name, dims) in tensors {
            put_str(&mut b, name);
            b.extend(2u32.to_le_bytes());
            for d in dims {
                b.extend(d.to_le_bytes());
            }
            b.extend(0u32.to_le_bytes()); // F32
            b.extend(offset.to_le_bytes());
            offset += dims[0] * dims[1] * 4;
            offset = offset.div_ceil(32) * 32;
        }
        while b.len() % 32 != 0 {
            b.push(0);
        }
        b.extend(vec![0u8; offset as usize]);
        std::fs::write(path, b).unwrap();
    }

    fn temp_dir(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!(
            "forge-studio-test-{name}-{}",
            crate::util::new_id()
        ));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn a_header_becomes_an_entry() {
        let dir = temp_dir("lib");
        let f = dir.join("sub").join("tiny.gguf");
        std::fs::create_dir_all(f.parent().unwrap()).unwrap();
        tiny_gguf(&f, "llama");
        let found = scan(&[dir.clone()], false);
        assert_eq!(found.len(), 1);
        let e = &found[0];
        assert_eq!(e.arch, "llama");
        assert_eq!(e.layers, 2);
        assert_eq!(e.ctx_train, 2048);
        assert_eq!(e.params, 12 + 16 + 8);
        assert_eq!(e.quant, "F32");
        assert!(e.forge_supported());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn an_embedding_model_is_supported_and_is_not_a_chat_model() {
        let dir = temp_dir("embed");
        let f = dir.join("nomic.gguf");
        tiny_gguf(&f, "nomic-bert");
        let g = dir.join("chat.gguf");
        tiny_gguf(&g, "qwen3");
        let found = scan(&[dir.clone()], false);
        let nomic = found.iter().find(|e| e.arch == "nomic-bert").unwrap();
        assert!(nomic.is_embedding());
        assert!(nomic.forge_supported());
        let chat = found.iter().find(|e| e.arch == "qwen3").unwrap();
        assert!(!chat.is_embedding());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_bert_model_is_an_embedding_model_only_with_a_wordpiece_vocabulary() {
        let dir = temp_dir("bert");
        tiny_gguf_with_tokenizer(&dir.join("bge-small.gguf"), "bert", "bert");
        std::fs::create_dir_all(dir.join("m3")).unwrap();
        tiny_gguf_with_tokenizer(&dir.join("m3").join("bge-m3.gguf"), "bert", "t5");
        let found = scan(&[dir.clone()], false);
        let small = found.iter().find(|e| e.path.ends_with("bge-small.gguf")).unwrap();
        assert!(small.is_embedding() && small.forge_supported(), "{:?}", small.forge_issue());
        let m3 = found.iter().find(|e| e.path.ends_with("bge-m3.gguf")).unwrap();
        assert!(!m3.is_embedding() && !m3.forge_supported());
        assert!(m3.forge_issue().unwrap().contains("WordPiece"), "{:?}", m3.forge_issue());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_inspector_groups_layers_and_roles() {
        let dir = temp_dir("inspect");
        let f = dir.join("tiny.gguf");
        tiny_gguf(&f, "qwen3");
        let i = inspect(&f, None, "test").unwrap();
        assert_eq!(i.layers.len(), 2);
        assert_eq!(i.layers[0].kind, "Attention");
        assert_eq!(i.tensors.len(), 3);
        assert!(i.roles.iter().any(|(r, _)| r == "token embeddings"));
        assert_eq!(i.formats[0].0, "F32");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_broken_file_is_an_entry_with_an_error() {
        let dir = temp_dir("broken");
        let f = dir.join("junk.gguf");
        std::fs::write(&f, b"not a gguf file at all").unwrap();
        let found = scan(&[dir.clone()], false);
        assert_eq!(found.len(), 1);
        assert!(found[0].error.is_some());
        assert!(!found[0].forge_supported());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
