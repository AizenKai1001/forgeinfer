//! What the computer has: CPU, memory, GPUs. Used to say which models fit.

use crate::util::quiet_command;

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Gpu {
    pub name: String,
    pub vram_total: u64,
    pub vram_used: u64,
    pub driver: String,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct HardwareInfo {
    pub os: String,
    pub cpu: String,
    pub threads: usize,
    pub ram_total: u64,
    pub ram_free: u64,
    pub gpus: Vec<Gpu>,
    /// The memory the GPU and the CPU share (Apple silicon).
    pub unified: bool,
}

impl HardwareInfo {
    /// The biggest video memory of any GPU.
    pub fn best_vram(&self) -> u64 {
        self.gpus.iter().map(|g| g.vram_total).max().unwrap_or(0)
    }
}

fn run(program: &str, args: &[&str]) -> Option<String> {
    let out = quiet_command(program).args(args).output().ok()?;
    if !out.status.success() {
        return None;
    }
    Some(String::from_utf8_lossy(&out.stdout).to_string())
}

pub fn parse_nvidia_smi(text: &str) -> Vec<Gpu> {
    text.lines()
        .filter_map(|l| {
            let f: Vec<&str> = l.split(',').map(str::trim).collect();
            if f.len() < 4 {
                return None;
            }
            let mib = |s: &str| s.parse::<f64>().ok().map(|v| (v * 1024.0 * 1024.0) as u64);
            Some(Gpu {
                name: f[0].to_string(),
                vram_total: mib(f[1])?,
                vram_used: mib(f[2]).unwrap_or(0),
                driver: f[3].to_string(),
            })
        })
        .collect()
}

pub fn parse_cpuinfo(text: &str) -> Option<String> {
    text.lines()
        .find(|l| {
            l.starts_with("model name") || l.starts_with("Hardware") || l.starts_with("Model")
        })
        .and_then(|l| l.split(':').nth(1))
        .map(|s| s.trim().to_string())
}

pub fn parse_meminfo(text: &str) -> (u64, u64) {
    let kb = |key: &str| {
        text.lines()
            .find(|l| l.starts_with(key))
            .and_then(|l| l.split_whitespace().nth(1))
            .and_then(|v| v.parse::<u64>().ok())
            .map(|v| v * 1024)
            .unwrap_or(0)
    };
    (kb("MemTotal:"), kb("MemAvailable:"))
}

#[cfg(windows)]
fn memory() -> (u64, u64) {
    #[repr(C)]
    struct MemoryStatusEx {
        length: u32,
        memory_load: u32,
        total_phys: u64,
        avail_phys: u64,
        total_page: u64,
        avail_page: u64,
        total_virtual: u64,
        avail_virtual: u64,
        avail_ext: u64,
    }
    extern "system" {
        fn GlobalMemoryStatusEx(buffer: *mut MemoryStatusEx) -> i32;
    }
    let mut s = MemoryStatusEx {
        length: std::mem::size_of::<MemoryStatusEx>() as u32,
        memory_load: 0,
        total_phys: 0,
        avail_phys: 0,
        total_page: 0,
        avail_page: 0,
        total_virtual: 0,
        avail_virtual: 0,
        avail_ext: 0,
    };
    // SAFETY: `s` is a properly sized, initialised MEMORYSTATUSEX with its length field set, as the API requires.
    let ok = unsafe { GlobalMemoryStatusEx(&mut s) };
    if ok != 0 {
        (s.total_phys, s.avail_phys)
    } else {
        (0, 0)
    }
}

#[cfg(not(windows))]
fn memory() -> (u64, u64) {
    if cfg!(target_os = "macos") {
        let total = run("sysctl", &["-n", "hw.memsize"])
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(0);
        return (total, 0);
    }
    std::fs::read_to_string("/proc/meminfo")
        .map(|t| parse_meminfo(&t))
        .unwrap_or((0, 0))
}

fn cpu_name() -> String {
    if cfg!(windows) {
        if let Some(t) = run(
            "reg",
            &[
                "query",
                "HKLM\\HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0",
                "/v",
                "ProcessorNameString",
            ],
        ) {
            if let Some(l) = t.lines().find(|l| l.contains("ProcessorNameString")) {
                if let Some(idx) = l.find("REG_SZ") {
                    return l[idx + 6..].trim().to_string();
                }
            }
        }
        return std::env::var("PROCESSOR_IDENTIFIER").unwrap_or_else(|_| "unknown CPU".to_string());
    }
    if cfg!(target_os = "macos") {
        return run("sysctl", &["-n", "machdep.cpu.brand_string"])
            .map(|s| s.trim().to_string())
            .unwrap_or_else(|| "unknown CPU".to_string());
    }
    std::fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|t| parse_cpuinfo(&t))
        .unwrap_or_else(|| "unknown CPU".to_string())
}

/// The parts that change while the app runs: the GPUs' memory in use and the free system memory.
pub fn detect_live() -> (Vec<Gpu>, u64) {
    let gpus = run(
        "nvidia-smi",
        &[
            "--query-gpu=name,memory.total,memory.used,driver_version",
            "--format=csv,noheader,nounits",
        ],
    )
    .map(|t| parse_nvidia_smi(&t))
    .unwrap_or_default();
    (gpus, memory().1)
}

pub fn detect() -> HardwareInfo {
    let (ram_total, ram_free) = memory();
    let mut gpus = run(
        "nvidia-smi",
        &[
            "--query-gpu=name,memory.total,memory.used,driver_version",
            "--format=csv,noheader,nounits",
        ],
    )
    .map(|t| parse_nvidia_smi(&t))
    .unwrap_or_default();
    let unified = cfg!(target_os = "macos") && std::env::consts::ARCH == "aarch64";
    if unified && gpus.is_empty() {
        gpus.push(Gpu {
            name: "Apple GPU (shares the computer's memory)".to_string(),
            vram_total: ram_total * 3 / 4,
            vram_used: 0,
            driver: String::new(),
        });
    }
    HardwareInfo {
        os: format!("{} {}", std::env::consts::OS, std::env::consts::ARCH),
        cpu: cpu_name(),
        threads: std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1),
        ram_total,
        ram_free,
        gpus,
        unified,
    }
}

/// How a model of `bytes` would run here.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fit {
    /// Everything in video memory.
    Gpu,
    /// Part of it in system memory (works, slower).
    Split,
    /// More than this computer holds.
    TooBig,
    Unknown,
}

pub fn fit(hw: &HardwareInfo, bytes: u64) -> Fit {
    if hw.ram_total == 0 && hw.gpus.is_empty() {
        return Fit::Unknown;
    }
    let vram = hw.best_vram();
    // leave room for the KV cache and the runtime
    if vram > 0 && bytes + bytes / 8 + 400_000_000 <= vram {
        Fit::Gpu
    } else if bytes + bytes / 10 <= vram + hw.ram_total.saturating_sub(4_000_000_000) {
        Fit::Split
    } else {
        Fit::TooBig
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nvidia_smi_lines_are_parsed() {
        let g = parse_nvidia_smi("NVIDIA GeForce RTX 4060, 8188, 828, 581.15\nNVIDIA GeForce RTX 2060 SUPER, 8192, 100, 550.1\n");
        assert_eq!(g.len(), 2);
        assert_eq!(g[0].name, "NVIDIA GeForce RTX 4060");
        assert_eq!(g[0].vram_total, 8188 * 1024 * 1024);
        assert_eq!(g[1].driver, "550.1");
        assert!(parse_nvidia_smi("garbage").is_empty());
    }

    #[test]
    fn linux_files_are_parsed() {
        assert_eq!(
            parse_cpuinfo("processor : 0\nmodel name\t: Intel(R) Core(TM) i5-12600K\n").unwrap(),
            "Intel(R) Core(TM) i5-12600K"
        );
        assert_eq!(
            parse_meminfo(
                "MemTotal:       32768000 kB\nMemFree: 1 kB\nMemAvailable:   16384000 kB\n"
            ),
            (32768000 * 1024, 16384000 * 1024)
        );
    }

    #[test]
    fn fit_follows_the_memory() {
        let hw = HardwareInfo {
            ram_total: 32_000_000_000,
            gpus: vec![Gpu {
                name: "g".into(),
                vram_total: 8_000_000_000,
                vram_used: 0,
                driver: String::new(),
            }],
            ..Default::default()
        };
        assert_eq!(fit(&hw, 2_700_000_000), Fit::Gpu);
        assert_eq!(fit(&hw, 15_600_000_000), Fit::Split);
        assert_eq!(fit(&hw, 60_000_000_000), Fit::TooBig);
        assert_eq!(fit(&HardwareInfo::default(), 1), Fit::Unknown);
    }
}
