//! Hugging Face: search for GGUF repositories, list their files, download one (resumable).

use serde_json::Value;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

const API: &str = "https://huggingface.co";

#[derive(Clone, Debug, PartialEq)]
pub struct Repo {
    pub id: String,
    pub downloads: u64,
    pub likes: u64,
    pub tags: Vec<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RepoFile {
    pub path: String,
    pub size: u64,
}

impl RepoFile {
    pub fn file_name(&self) -> &str {
        self.path.rsplit('/').next().unwrap_or(&self.path)
    }

    pub fn is_projector(&self) -> bool {
        self.file_name().to_lowercase().contains("mmproj")
    }

    /// `Some((part, of))` for `name-00002-of-00005.gguf`.
    pub fn shard(&self) -> Option<(u32, u32)> {
        let n = self.file_name().to_lowercase();
        let stem = n.strip_suffix(".gguf")?;
        let idx = stem.rfind("-of-")?;
        let total: u32 = stem[idx + 4..].parse().ok()?;
        let head = &stem[..idx];
        let part: u32 = head.rsplit('-').next()?.parse().ok()?;
        Some((part, total))
    }

    pub fn quant(&self) -> String {
        quant_label(self.file_name())
    }
}

fn agent() -> ureq::Agent {
    let config = ureq::Agent::config_builder().timeout_connect(Some(Duration::from_secs(10))).http_status_as_error(false).user_agent("forge-studio/0.1").build();
    ureq::Agent::new_with_config(config)
}

fn get_json(url: &str) -> Result<Value, String> {
    let mut resp = agent().get(url).config().timeout_global(Some(Duration::from_secs(25))).build().call().map_err(|e| format!("cannot reach Hugging Face: {e}"))?;
    let status = resp.status().as_u16();
    let body = resp.body_mut().read_to_string().map_err(|e| e.to_string())?;
    if !(200..300).contains(&status) {
        return Err(format!("Hugging Face answered HTTP {status}"));
    }
    serde_json::from_str(&body).map_err(|e| format!("unexpected answer: {e}"))
}

pub fn urlencode(s: &str) -> String {
    let mut out = String::new();
    for b in s.bytes() {
        if b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.' | b'~') {
            out.push(b as char);
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

fn urlencode_path(p: &str) -> String {
    p.split('/').map(urlencode).collect::<Vec<_>>().join("/")
}

pub fn search(query: &str) -> Result<Vec<Repo>, String> {
    let q = query.trim();
    let url = format!("{API}/api/models?search={}&filter=gguf&sort=downloads&direction=-1&limit=40", urlencode(q));
    let v = get_json(&url)?;
    Ok(parse_search(&v))
}

pub fn parse_search(v: &Value) -> Vec<Repo> {
    v.as_array()
        .map(|a| {
            a.iter()
                .filter_map(|m| {
                    let id = m["id"].as_str().or(m["modelId"].as_str())?.to_string();
                    Some(Repo {
                        id,
                        downloads: m["downloads"].as_u64().unwrap_or(0),
                        likes: m["likes"].as_u64().unwrap_or(0),
                        tags: m["tags"].as_array().map(|t| t.iter().filter_map(|x| x.as_str().map(str::to_string)).collect()).unwrap_or_default(),
                    })
                })
                .collect()
        })
        .unwrap_or_default()
}

pub fn files(repo: &str) -> Result<Vec<RepoFile>, String> {
    let url = format!("{API}/api/models/{}/tree/main?recursive=true", urlencode_path(repo));
    let v = get_json(&url)?;
    Ok(parse_files(&v))
}

pub fn parse_files(v: &Value) -> Vec<RepoFile> {
    let mut out: Vec<RepoFile> = v
        .as_array()
        .map(|a| {
            a.iter()
                .filter(|f| f["type"].as_str() == Some("file"))
                .filter_map(|f| {
                    let path = f["path"].as_str()?.to_string();
                    if !path.to_lowercase().ends_with(".gguf") {
                        return None;
                    }
                    let size = f["lfs"]["size"].as_u64().or(f["size"].as_u64()).unwrap_or(0);
                    Some(RepoFile { path, size })
                })
                .collect()
        })
        .unwrap_or_default();
    out.sort_by(|a, b| a.size.cmp(&b.size).then_with(|| a.path.cmp(&b.path)));
    out
}

const QUANTS: [&str; 33] = [
    "IQ1_S", "IQ1_M", "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ2_M", "IQ3_XXS", "IQ3_XS", "IQ3_S", "IQ3_M", "IQ4_XS", "IQ4_NL", "Q2_K_S", "Q2_K", "Q3_K_S", "Q3_K_M", "Q3_K_L", "Q4_0", "Q4_1", "Q4_K_S", "Q4_K_M", "Q4_K_L", "Q5_0",
    "Q5_1", "Q5_K_S", "Q5_K_M", "Q5_K_L", "Q6_K_L", "Q6_K", "Q8_0", "BF16", "F16", "F32",
];

/// `Q4_K_M`, `IQ3_XS`, `F16` ... read from a file name; empty when there is none.
pub fn quant_label(file: &str) -> String {
    let up = file.to_uppercase().replace('-', "_").replace('.', "_");
    let mut best: Option<&str> = None;
    for q in QUANTS {
        if let Some(pos) = up.find(q) {
            let before_ok = pos == 0 || !up.as_bytes()[pos - 1].is_ascii_alphanumeric();
            let after = up.as_bytes().get(pos + q.len()).copied();
            let after_ok = after.map_or(true, |c| !c.is_ascii_alphanumeric());
            if before_ok && after_ok && best.map_or(true, |b| q.len() > b.len()) {
                best = Some(q);
            }
        }
    }
    best.unwrap_or("").to_string()
}

/// A rough guide for the list: how good the quantization is.
pub fn quant_note(q: &str) -> &'static str {
    match q {
        "Q4_K_M" => "recommended: best size to quality",
        "Q5_K_M" | "Q5_K_S" | "Q6_K" | "Q6_K_L" | "Q5_K_L" => "high quality",
        "Q8_0" => "near lossless, large",
        "BF16" | "F16" | "F32" => "full precision, very large",
        "Q4_K_S" | "Q4_0" | "Q4_1" | "Q4_K_L" | "IQ4_XS" | "IQ4_NL" => "good, a little smaller",
        "Q3_K_M" | "Q3_K_L" | "Q3_K_S" | "IQ3_M" | "IQ3_S" | "IQ3_XS" | "IQ3_XXS" => "small, some quality loss",
        "Q2_K" | "Q2_K_S" | "IQ2_M" | "IQ2_S" | "IQ2_XS" | "IQ2_XXS" | "IQ1_S" | "IQ1_M" => "tiny, noticeable quality loss",
        _ => "",
    }
}

/// Where a download ends up: `<dir>/<owner>/<repo>/<file>` (the layout LM Studio uses, so both programs find it).
pub fn destination(dir: &Path, repo: &str, file: &RepoFile) -> PathBuf {
    let mut p = dir.to_path_buf();
    for part in repo.split('/') {
        p.push(part);
    }
    p.push(file.file_name());
    p
}

/// Shared between the download thread and the screen.
#[derive(Default)]
pub struct Progress {
    pub done: AtomicU64,
    pub total: AtomicU64,
    pub cancel: AtomicBool,
}

/// Download `repo`/`file` to `dest`, resuming a `.part` file. Blocks; run it on a thread.
pub fn download(repo: &str, file: &RepoFile, dest: &Path, progress: &Arc<Progress>, wake: &dyn Fn()) -> Result<PathBuf, String> {
    if let Some(dir) = dest.parent() {
        std::fs::create_dir_all(dir).map_err(|e| format!("cannot create {}: {e}", dir.display()))?;
    }
    let part = dest.with_extension("gguf.part");
    let have = std::fs::metadata(&part).map(|m| m.len()).unwrap_or(0);
    let url = format!("{API}/{}/resolve/main/{}", urlencode_path(repo), urlencode_path(&file.path));
    let mut req = agent().get(&url);
    if have > 0 {
        req = req.header("Range", format!("bytes={have}-"));
    }
    let resp = req.call().map_err(|e| format!("cannot reach Hugging Face: {e}"))?;
    let status = resp.status().as_u16();
    if !(status == 200 || status == 206) {
        return Err(format!("Hugging Face answered HTTP {status} (a gated model needs a login this app does not have yet)"));
    }
    let resumed = status == 206 && have > 0;
    let start = if resumed { have } else { 0 };
    let remaining = resp.body().content_length().unwrap_or(file.size.saturating_sub(start));
    progress.total.store(start + remaining, Ordering::Relaxed);
    progress.done.store(start, Ordering::Relaxed);
    let mut out = std::fs::OpenOptions::new().create(true).write(true).append(resumed).truncate(!resumed).open(&part).map_err(|e| format!("cannot write {}: {e}", part.display()))?;
    let mut reader = resp.into_body().into_reader();
    let mut buf = vec![0u8; 256 * 1024];
    let mut tick = 0u32;
    loop {
        if progress.cancel.load(Ordering::Relaxed) {
            return Err("cancelled".to_string());
        }
        let n = reader.read(&mut buf).map_err(|e| format!("the download broke: {e}"))?;
        if n == 0 {
            break;
        }
        out.write_all(&buf[..n]).map_err(|e| format!("cannot write the file: {e}"))?;
        progress.done.fetch_add(n as u64, Ordering::Relaxed);
        tick += 1;
        if tick % 8 == 0 {
            wake();
        }
    }
    out.flush().map_err(|e| e.to_string())?;
    drop(out);
    let size = std::fs::metadata(&part).map(|m| m.len()).unwrap_or(0);
    let expected = progress.total.load(Ordering::Relaxed);
    if expected > 0 && size < expected {
        return Err(format!("incomplete: {size} of {expected} bytes (start it again to resume)"));
    }
    std::fs::rename(&part, dest).map_err(|e| format!("cannot move the finished file: {e}"))?;
    Ok(dest.to_path_buf())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn quant_names_come_out_of_file_names() {
        assert_eq!(quant_label("Qwen3.5-4B-Q4_K_M.gguf"), "Q4_K_M");
        assert_eq!(quant_label("model-IQ3_XXS.gguf"), "IQ3_XXS");
        assert_eq!(quant_label("Qwen3.5-27B-UD-Q2_K_XL.gguf"), "Q2_K");
        assert_eq!(quant_label("llama-3.2-1b-instruct-f16.gguf"), "F16");
        assert_eq!(quant_label("mmproj-F16.gguf"), "F16");
        assert_eq!(quant_label("weird.gguf"), "");
    }

    #[test]
    fn shards_and_projectors_are_recognised() {
        let f = RepoFile { path: "dir/model-00002-of-00005.gguf".into(), size: 1 };
        assert_eq!(f.shard(), Some((2, 5)));
        assert_eq!(f.file_name(), "model-00002-of-00005.gguf");
        assert!(RepoFile { path: "mmproj-F16.gguf".into(), size: 1 }.is_projector());
        assert_eq!(RepoFile { path: "a.gguf".into(), size: 1 }.shard(), None);
    }

    #[test]
    fn api_answers_are_parsed() {
        let s = parse_search(&json!([{"id": "bartowski/x-GGUF", "downloads": 12, "likes": 3, "tags": ["gguf", "text-generation"]}, {"nope": 1}]));
        assert_eq!(s.len(), 1);
        assert_eq!(s[0].downloads, 12);
        let f = parse_files(&json!([
            {"type": "file", "path": "b.gguf", "size": 200, "lfs": {"size": 5000}},
            {"type": "file", "path": "README.md", "size": 10},
            {"type": "directory", "path": "sub"},
            {"type": "file", "path": "a.gguf", "size": 100}
        ]));
        assert_eq!(f, vec![RepoFile { path: "a.gguf".into(), size: 100 }, RepoFile { path: "b.gguf".into(), size: 5000 }]);
    }

    #[test]
    fn urls_are_encoded() {
        assert_eq!(urlencode_path("dir/My Model (1).gguf"), "dir/My%20Model%20%281%29.gguf");
        let d = destination(Path::new("/m"), "owner/repo", &RepoFile { path: "x/y.gguf".into(), size: 0 });
        assert_eq!(d, Path::new("/m").join("owner").join("repo").join("y.gguf"));
    }
}

#[cfg(test)]
mod live_tests {
    use super::*;

    /// Needs the network: `cargo test --lib live_ -- --ignored --nocapture`. A 1.2 MB public test model.
    #[test]
    #[ignore]
    fn live_search_files_download_and_resume() {
        let found = search("tiny-llamas").expect("search");
        assert!(found.iter().any(|r| r.id == "ggml-org/tiny-llamas"), "{found:?}");
        let files = files("ggml-org/tiny-llamas").expect("files");
        let f = files.iter().find(|f| f.path == "stories260K.gguf").expect("file listed").clone();
        assert_eq!(f.size, 1_185_376);
        let dir = std::env::temp_dir().join(format!("forge-studio-dl-{}", crate::util::new_id()));
        let dest = destination(&dir, "ggml-org/tiny-llamas", &f);
        let progress = Arc::new(Progress::default());
        let path = download("ggml-org/tiny-llamas", &f, &dest, &progress, &|| {}).expect("download");
        assert_eq!(std::fs::metadata(&path).unwrap().len(), 1_185_376);
        assert_eq!(progress.done.load(Ordering::Relaxed), 1_185_376);
        // resume: keep the first 400,000 bytes as a .part file, fetch the rest
        let whole = std::fs::read(&path).unwrap();
        std::fs::remove_file(&path).unwrap();
        let part = path.with_extension("gguf.part");
        std::fs::write(&part, &whole[..400_000]).unwrap();
        let progress2 = Arc::new(Progress::default());
        let again = download("ggml-org/tiny-llamas", &f, &dest, &progress2, &|| {}).expect("resume");
        assert_eq!(std::fs::read(&again).unwrap(), whole);
        assert_eq!(progress2.total.load(Ordering::Relaxed), 1_185_376);
        // and the file is a model the library can read
        let e = crate::library::read_entry(&again, None, "test");
        assert_eq!(e.arch, "llama");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
