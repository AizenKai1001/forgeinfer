//! The multi-token-prediction head of Qwen3.5 models. The GGUF files do not contain it, the original Hugging Face checkpoints do: this module finds
//! a head file next to a model and can fetch just the `mtp.*` tensors of a checkpoint (HTTP range requests, no multi-gigabyte download) into one small
//! safetensors file. With the head, the engine drafts a few tokens ahead and verifies them in one pass over the weights: 1.5 to 2 times faster answers.

use serde_json::{json, Map, Value};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::time::Duration;

use crate::hf::Progress;

const API: &str = "https://huggingface.co";
const CHUNK: u64 = 8 << 20;

/// `Qwen3.5-9B-Q4_K_M` -> `Qwen3.5-9B`: the file name without the quantization (and the `UD` of Unsloth's dynamic files).
pub fn base_name(stem: &str) -> String {
    let q = crate::hf::quant_label(stem);
    let mut base = stem.to_string();
    if !q.is_empty() {
        if let Some(pos) = stem.to_uppercase().find(&q) {
            base = stem[..pos].to_string();
        }
    }
    let mut b = base.trim_end_matches(['-', '_', '.', ' ']).to_string();
    for tail in ["-UD", "_UD", "-ud"] {
        if let Some(x) = b.strip_suffix(tail) {
            b = x.to_string();
        }
    }
    b
}

/// The Hugging Face repository of the original checkpoint of a Qwen3.5 model, from its file name (`Qwen3.5-9B-Q4_K_M` -> `Qwen/Qwen3.5-9B`).
pub fn repo_of(stem: &str) -> Option<String> {
    let base = base_name(stem);
    let base = base.trim_end_matches("-GGUF").to_string();
    let low = base.to_lowercase();
    if low.starts_with("qwen3.5-") && !low.contains("mmproj") {
        Some(format!("Qwen/{base}"))
    } else {
        None
    }
}

/// A head file for a model: `<base>-mtp.safetensors` or `<base>.mtp.safetensors` beside the model, in an `mtp` folder beside it, or in one of `dirs`
/// (and their `mtp` folders). (A plain `mtp.safetensors` beside the model is found by the engine itself.)
pub fn find_head(model: &Path, dirs: &[PathBuf]) -> Option<PathBuf> {
    let stem = model.file_stem()?.to_string_lossy().to_string();
    let base = base_name(&stem);
    let mut places: Vec<PathBuf> = Vec::new();
    if let Some(d) = model.parent() {
        places.push(d.to_path_buf());
        places.push(d.join("mtp"));
    }
    for d in dirs {
        places.push(d.clone());
        places.push(d.join("mtp"));
    }
    for d in places {
        for name in [format!("{base}-mtp.safetensors"), format!("{base}.mtp.safetensors"), format!("{stem}-mtp.safetensors")] {
            let p = d.join(&name);
            if p.is_file() {
                return Some(p);
            }
        }
    }
    None
}

fn agent() -> ureq::Agent {
    let config = ureq::Agent::config_builder().timeout_connect(Some(Duration::from_secs(10))).http_status_as_error(false).user_agent("forge-studio/0.1").build();
    ureq::Agent::new_with_config(config)
}

fn get_range(url: &str, start: u64, end: u64) -> Result<Vec<u8>, String> {
    let mut last = String::new();
    for attempt in 0..4 {
        let r = agent().get(url).header("Range", format!("bytes={start}-{end}")).config().timeout_global(Some(Duration::from_secs(300))).build().call();
        match r {
            Ok(mut resp) if resp.status().as_u16() == 206 || resp.status().as_u16() == 200 => {
                let mut data = Vec::new();
                match std::io::Read::read_to_end(&mut resp.body_mut().as_reader(), &mut data) {
                    Ok(_) => {
                        if resp.status().as_u16() == 200 && data.len() as u64 > end - start + 1 {
                            // the server ignored the range: keep the part asked for
                            let s = start as usize;
                            let e = (end + 1) as usize;
                            if e <= data.len() {
                                return Ok(data[s..e].to_vec());
                            }
                        }
                        if data.len() as u64 == end - start + 1 {
                            return Ok(data);
                        }
                        last = format!("short read: {} of {} bytes", data.len(), end - start + 1);
                    }
                    Err(e) => last = e.to_string(),
                }
            }
            Ok(resp) => {
                last = format!("HTTP {}", resp.status().as_u16());
                if resp.status().as_u16() == 404 {
                    break;
                }
            }
            Err(e) => last = e.to_string(),
        }
        std::thread::sleep(Duration::from_secs(2 + 2 * attempt));
    }
    Err(format!("cannot fetch {url}: {last}"))
}

fn get_text(url: &str) -> Result<Option<String>, String> {
    let mut resp = agent().get(url).config().timeout_global(Some(Duration::from_secs(60))).build().call().map_err(|e| format!("cannot reach Hugging Face: {e}"))?;
    let status = resp.status().as_u16();
    if status == 404 {
        return Ok(None);
    }
    if !(200..300).contains(&status) {
        return Err(format!("Hugging Face answered HTTP {status}"));
    }
    resp.body_mut().read_to_string().map(Some).map_err(|e| e.to_string())
}

struct Piece {
    name: String,
    dtype: String,
    shape: Value,
    url: String,
    base: u64,
    b0: u64,
    b1: u64,
}

/// Assemble a safetensors file from tensors held in memory: `(name, dtype, shape, bytes)`, sorted by name.
pub fn write_safetensors(out: &Path, tensors: &[(String, String, Value, Vec<u8>)]) -> std::io::Result<()> {
    let mut header = Map::new();
    let mut offset = 0u64;
    for (name, dtype, shape, bytes) in tensors {
        header.insert(name.clone(), json!({"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + bytes.len() as u64]}));
        offset += bytes.len() as u64;
    }
    header.insert("__metadata__".to_string(), json!({"format": "pt"}));
    let mut text = serde_json::to_string(&Value::Object(header)).unwrap_or_default();
    while text.len() % 8 != 0 {
        text.push(' ');
    }
    let tmp = out.with_extension("safetensors.part");
    let mut f = std::io::BufWriter::new(std::fs::File::create(&tmp)?);
    f.write_all(&(text.len() as u64).to_le_bytes())?;
    f.write_all(text.as_bytes())?;
    for (_, _, _, bytes) in tensors {
        f.write_all(bytes)?;
    }
    f.flush()?;
    drop(f);
    std::fs::rename(&tmp, out)
}

/// Download the `mtp.*` tensors of `repo` into the safetensors file `out`. Blocks; run it on a thread. `progress.total` is set once the sizes are known.
pub fn fetch_head(repo: &str, out: &Path, progress: &Progress) -> Result<(), String> {
    if let Some(dir) = out.parent() {
        std::fs::create_dir_all(dir).map_err(|e| format!("cannot create {}: {e}", dir.display()))?;
    }
    let index = get_text(&format!("{API}/{repo}/resolve/main/model.safetensors.index.json"))?;
    // which shard files hold mtp tensors (None: the checkpoint is one file)
    let files: Vec<String> = match index {
        Some(text) => {
            let v: Value = serde_json::from_str(&text).map_err(|e| format!("unexpected index: {e}"))?;
            let mut f: Vec<String> = v["weight_map"].as_object().map(|m| m.iter().filter(|(k, _)| k.starts_with("mtp.")).map(|(_, f)| f.as_str().unwrap_or("").to_string()).collect()).unwrap_or_default();
            f.sort();
            f.dedup();
            f
        }
        None => vec!["model.safetensors".to_string()],
    };
    if files.is_empty() {
        return Err(format!("{repo} has no prediction head (no mtp.* tensors)"));
    }
    let mut pieces: Vec<Piece> = Vec::new();
    for f in &files {
        let url = format!("{API}/{repo}/resolve/main/{f}");
        let n = u64::from_le_bytes(get_range(&url, 0, 7)?.try_into().map_err(|_| "bad safetensors header".to_string())?);
        if n == 0 || n > 64 << 20 {
            return Err(format!("{f}: implausible header size {n}"));
        }
        let header: Value = serde_json::from_slice(&get_range(&url, 8, 8 + n - 1)?).map_err(|e| format!("{f}: {e}"))?;
        for (k, t) in header.as_object().into_iter().flatten() {
            if !k.starts_with("mtp.") {
                continue;
            }
            let (b0, b1) = (t["data_offsets"][0].as_u64().unwrap_or(0), t["data_offsets"][1].as_u64().unwrap_or(0));
            pieces.push(Piece { name: k.clone(), dtype: t["dtype"].as_str().unwrap_or("BF16").to_string(), shape: t["shape"].clone(), url: url.clone(), base: 8 + n, b0, b1 });
        }
    }
    if pieces.is_empty() {
        return Err(format!("{repo} has no prediction head (no mtp.* tensors)"));
    }
    pieces.sort_by(|a, b| a.name.cmp(&b.name));
    progress.total.store(pieces.iter().map(|p| p.b1 - p.b0).sum(), Ordering::Relaxed);
    progress.done.store(0, Ordering::Relaxed);
    let mut tensors = Vec::with_capacity(pieces.len());
    for p in &pieces {
        let mut bytes = Vec::with_capacity((p.b1 - p.b0) as usize);
        let mut at = p.b0;
        while at < p.b1 {
            if progress.cancel.load(Ordering::Relaxed) {
                return Err("cancelled".to_string());
            }
            let end = (at + CHUNK).min(p.b1);
            let part = get_range(&p.url, p.base + at, p.base + end - 1)?;
            progress.done.fetch_add(part.len() as u64, Ordering::Relaxed);
            bytes.extend_from_slice(&part);
            at = end;
        }
        tensors.push((p.name.clone(), p.dtype.clone(), p.shape.clone(), bytes));
    }
    write_safetensors(out, &tensors).map_err(|e| format!("cannot write {}: {e}", out.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_of_heads_and_repositories() {
        assert_eq!(base_name("Qwen3.5-9B-Q4_K_M"), "Qwen3.5-9B");
        assert_eq!(base_name("Qwen3.5-27B-UD-Q2_K_XL"), "Qwen3.5-27B");
        assert_eq!(base_name("Qwen3.5-4B-Q8_0"), "Qwen3.5-4B");
        assert_eq!(repo_of("Qwen3.5-9B-Q4_K_M").as_deref(), Some("Qwen/Qwen3.5-9B"));
        assert_eq!(repo_of("Qwen3-4B-Q4_K_M"), None);
        assert_eq!(repo_of("Qwen3.5-4B-mmproj-F16"), None);
    }

    #[test]
    fn a_head_beside_or_in_an_mtp_folder_is_found() {
        let dir = std::env::temp_dir().join(format!("forge-studio-mtp-{}", crate::util::new_id()));
        std::fs::create_dir_all(dir.join("mtp")).unwrap();
        let model = dir.join("Qwen3.5-9B-Q4_K_M.gguf");
        std::fs::write(&model, b"x").unwrap();
        assert_eq!(find_head(&model, &[]), None);
        std::fs::write(dir.join("mtp").join("Qwen3.5-9B-mtp.safetensors"), b"x").unwrap();
        assert_eq!(find_head(&model, &[]), Some(dir.join("mtp").join("Qwen3.5-9B-mtp.safetensors")));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_written_safetensors_file_reads_back() {
        let dir = std::env::temp_dir().join(format!("forge-studio-st-{}", crate::util::new_id()));
        std::fs::create_dir_all(&dir).unwrap();
        let out = dir.join("head.safetensors");
        write_safetensors(&out, &[("mtp.a".to_string(), "F32".to_string(), json!([2]), vec![0, 0, 128, 63, 0, 0, 0, 64]), ("mtp.b".to_string(), "BF16".to_string(), json!([1]), vec![1, 2])]).unwrap();
        let data = std::fs::read(&out).unwrap();
        let n = u64::from_le_bytes(data[..8].try_into().unwrap()) as usize;
        assert_eq!(n % 8, 0);
        let header: Value = serde_json::from_slice(&data[8..8 + n]).unwrap();
        assert_eq!(header["mtp.a"]["data_offsets"], json!([0, 8]));
        assert_eq!(header["mtp.b"]["data_offsets"], json!([8, 10]));
        assert_eq!(&data[8 + n..], &[0, 0, 128, 63, 0, 0, 0, 64, 1, 2]);
        // the engine's own reader accepts it
        let st = forge_gguf::SafeTensors::open(&out).unwrap();
        assert!(st.tensors.iter().any(|t| t.name == "mtp.a"));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
