//! Small helpers shared by the modules: number formatting, ids, paths, opening a folder.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

/// `1.5 GB`, `812 MB`, `3.2 KB`.
pub fn human_bytes(n: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KB", "MB", "GB", "TB"];
    let mut v = n as f64;
    let mut u = 0;
    while v >= 1000.0 && u + 1 < UNITS.len() {
        v /= 1000.0;
        u += 1;
    }
    if u == 0 {
        format!("{n} B")
    } else if v >= 100.0 {
        format!("{v:.0} {}", UNITS[u])
    } else if v >= 10.0 {
        format!("{v:.1} {}", UNITS[u])
    } else {
        format!("{v:.2} {}", UNITS[u])
    }
}

/// Memory sizes as Windows and the GPU vendors show them: binary units under the name GB (`31.9 GB`, `8.0 GB`).
pub fn human_gib(n: u64) -> String {
    let g = n as f64 / (1024.0 * 1024.0 * 1024.0);
    if g >= 10.0 {
        format!("{g:.0} GB")
    } else if g >= 1.0 {
        format!("{g:.1} GB")
    } else {
        format!("{:.0} MB", n as f64 / (1024.0 * 1024.0))
    }
}

/// `752M`, `27B`, `1.2K`.
pub fn human_count(n: u64) -> String {
    let v = n as f64;
    if n >= 1_000_000_000_000 {
        format!("{:.1}T", v / 1e12)
    } else if n >= 10_000_000_000 {
        format!("{:.0}B", v / 1e9)
    } else if n >= 1_000_000_000 {
        format!("{:.1}B", v / 1e9)
    } else if n >= 10_000_000 {
        format!("{:.0}M", v / 1e6)
    } else if n >= 1_000_000 {
        format!("{:.1}M", v / 1e6)
    } else if n >= 1000 {
        format!("{:.1}K", v / 1e3)
    } else {
        n.to_string()
    }
}

pub fn unix_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// A short unique id (time in microseconds plus a counter), sortable by creation time.
pub fn new_id() -> String {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let micros = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_micros() as u64)
        .unwrap_or(0);
    format!(
        "{micros:x}{:02x}",
        COUNTER.fetch_add(1, Ordering::Relaxed) % 256
    )
}

pub fn home_dir() -> PathBuf {
    let var = if cfg!(windows) { "USERPROFILE" } else { "HOME" };
    std::env::var_os(var)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."))
}

/// The first `n` characters of `s`, with an ellipsis when it was cut.
pub fn truncate(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        s.to_string()
    } else {
        let mut t: String = s.chars().take(n.saturating_sub(1)).collect();
        t.push('\u{2026}');
        t
    }
}

/// Run a command without a console window flashing up on Windows.
pub fn quiet_command(program: impl AsRef<std::ffi::OsStr>) -> Command {
    #[allow(unused_mut)]
    let mut c = Command::new(program);
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        c.creation_flags(0x0800_0000);
    }
    c
}

/// Make a child process die with this app, so a crash never leaves an engine holding the GPU: a job object on Windows, PR_SET_PDEATHSIG on Linux.
/// Call `die_with_parent_before` on the command before it is spawned and `die_with_parent_after` on the child right after.
#[allow(unused_variables)]
pub fn die_with_parent_before(cmd: &mut Command) {
    #[cfg(target_os = "linux")]
    {
        use std::os::unix::process::CommandExt;
        extern "C" {
            fn prctl(option: i32, arg2: u64, arg3: u64, arg4: u64, arg5: u64) -> i32;
        }
        // SAFETY: runs in the forked child before exec; prctl is async-signal-safe and only sets this process's own death signal (PR_SET_PDEATHSIG = 1, SIGKILL = 9).
        unsafe {
            cmd.pre_exec(|| {
                prctl(1, 9, 0, 0, 0);
                Ok(())
            });
        }
    }
}

#[allow(unused_variables)]
pub fn die_with_parent_after(child: &std::process::Child) {
    #[cfg(windows)]
    {
        use std::os::windows::io::AsRawHandle;
        use std::sync::OnceLock;
        type Handle = *mut core::ffi::c_void;
        #[repr(C)]
        struct BasicLimit {
            per_process_user_time: i64,
            per_job_user_time: i64,
            limit_flags: u32,
            min_ws: usize,
            max_ws: usize,
            active_process_limit: u32,
            affinity: usize,
            priority_class: u32,
            scheduling_class: u32,
        }
        #[repr(C)]
        struct IoCounters([u64; 6]);
        #[repr(C)]
        struct ExtendedLimit {
            basic: BasicLimit,
            io: IoCounters,
            process_memory: usize,
            job_memory: usize,
            peak_process: usize,
            peak_job: usize,
        }
        extern "system" {
            fn CreateJobObjectW(attributes: *mut core::ffi::c_void, name: *const u16) -> Handle;
            fn SetInformationJobObject(
                job: Handle,
                class: i32,
                info: *mut core::ffi::c_void,
                len: u32,
            ) -> i32;
            fn AssignProcessToJobObject(job: Handle, process: Handle) -> i32;
        }
        static JOB: OnceLock<usize> = OnceLock::new();
        let job = *JOB.get_or_init(|| {
            // SAFETY: plain Win32 calls with properly sized, zero-initialised structures; the handle is kept for the life of the process, so the system closes it (and ends the children) when this app ends.
            unsafe {
                let h = CreateJobObjectW(core::ptr::null_mut(), core::ptr::null());
                if h.is_null() {
                    return 0;
                }
                let mut info: ExtendedLimit = core::mem::zeroed();
                info.basic.limit_flags = 0x2000; // JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
                SetInformationJobObject(
                    h,
                    9,
                    &mut info as *mut ExtendedLimit as *mut core::ffi::c_void,
                    core::mem::size_of::<ExtendedLimit>() as u32,
                );
                h as usize
            }
        });
        if job != 0 {
            // SAFETY: both handles are valid; the child handle is owned by `child`.
            unsafe {
                AssignProcessToJobObject(job as Handle, child.as_raw_handle() as Handle);
            }
        }
    }
}

/// Show a folder (or a file's folder) in the system file manager.
pub fn reveal(path: &Path) {
    let target = if path.is_dir() {
        path.to_path_buf()
    } else {
        path.parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| path.to_path_buf())
    };
    let result = if cfg!(windows) {
        Command::new("explorer").arg(&target).spawn()
    } else if cfg!(target_os = "macos") {
        Command::new("open").arg(&target).spawn()
    } else {
        Command::new("xdg-open").arg(&target).spawn()
    };
    let _ = result;
}

/// A TCP port that nothing listens on, searching upwards from `start`.
pub fn free_port(start: u16) -> u16 {
    for p in start..start.saturating_add(200) {
        if std::net::TcpListener::bind(("127.0.0.1", p)).is_ok() {
            return p;
        }
    }
    start
}

/// Seconds as `1.2 s` / `3 min 4 s`.
pub fn human_secs(s: f64) -> String {
    if s < 60.0 {
        format!("{s:.1} s")
    } else {
        format!("{} min {} s", (s / 60.0) as u64, (s % 60.0) as u64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sizes_and_counts_read_well() {
        assert_eq!(human_bytes(0), "0 B");
        assert_eq!(human_bytes(1_500_000_000), "1.50 GB");
        assert_eq!(human_bytes(532_517_120), "533 MB");
        assert_eq!(human_count(752_000_000), "752M");
        assert_eq!(human_count(27_000_000_000), "27B");
        assert_eq!(human_count(4_205_000_000), "4.2B");
    }

    #[test]
    fn truncation_keeps_whole_characters() {
        assert_eq!(truncate("héllo wörld", 5), "héll\u{2026}");
        assert_eq!(truncate("short", 10), "short");
    }

    #[test]
    fn ids_are_unique() {
        let a = new_id();
        let b = new_id();
        assert_ne!(a, b);
    }
}

/// The text of a file for attaching to a message: UTF-8 (a byte-order mark is dropped), no NUL bytes, at most `max` bytes.
pub fn read_text_file(path: &std::path::Path, max: u64) -> Result<String, String> {
    let name = path
        .file_name()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_default();
    let meta = std::fs::metadata(path).map_err(|e| format!("{name}: {e}"))?;
    if meta.len() > max {
        return Err(format!(
            "{name} is {} and the limit is {}: attach a smaller file.",
            human_bytes(meta.len()),
            human_bytes(max)
        ));
    }
    let bytes = std::fs::read(path).map_err(|e| format!("{name}: {e}"))?;
    if bytes.contains(&0) {
        return Err(format!("{name} is not a text file."));
    }
    let text = String::from_utf8(bytes).map_err(|_| format!("{name} is not UTF-8 text."))?;
    Ok(text.trim_start_matches('\u{feff}').to_string())
}
