// Compiles kernels/*.cu with nvcc into one cubin per supported architecture plus PTX (compute_75) as the forward-compatible fallback (decision D11).
// nvcc is looked up in $NVCC, $CUDA_PATH/bin, ~/cuda-13.2.0/bin and the PATH. Without nvcc the build still succeeds with EMPTY images (the GPU backend then reports that kernels are missing),
// so that the CPU-only crates stay buildable on machines without a CUDA toolkit.
// FORGE_CUDA_ARCHS=89 (or 86,89,...) adds a native cubin for each listed architecture (Ada: RTX 40xx, L4); `image_for` prefers it on exactly that compute capability. Unset = no extra images.
// Windows: nvcc needs MSVC (cl.exe) as its host compiler even for -cubin/-ptx; see windows_ccbin.

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Windows, for when cl.exe is not on the PATH (no developer prompt) and NVCC_CCBIN is unset: the folder of cl.exe of the newest Visual Studio 2022 (vswhere, version 17.x; CUDA 12.x rejects the
/// MSVC 14.5x of Visual Studio 2026). nvcc reads NVCC_CCBIN itself and sets up the compiler's environment.
fn windows_ccbin() -> Option<PathBuf> {
    let pf86 = env::var_os("ProgramFiles(x86)")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("C:/Program Files (x86)"));
    let vswhere = pf86
        .join("Microsoft Visual Studio")
        .join("Installer")
        .join("vswhere.exe");
    let out = Command::new(vswhere)
        .args([
            "-products",
            "*",
            "-version",
            "[17.0,18.0)",
            "-requires",
            "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
            "-latest",
            "-property",
            "installationPath",
        ])
        .output()
        .ok()?;
    let inst = PathBuf::from(String::from_utf8_lossy(&out.stdout).lines().next()?.trim());
    let vc = inst.join("VC");
    let ver = std::fs::read_to_string(
        vc.join("Auxiliary")
            .join("Build")
            .join("Microsoft.VCToolsVersion.default.txt"),
    )
    .ok()?;
    let dir = vc
        .join("Tools")
        .join("MSVC")
        .join(ver.trim())
        .join("bin")
        .join("Hostx64")
        .join("x64");
    dir.join("cl.exe").exists().then_some(dir)
}

/// The extra architectures of FORGE_CUDA_ARCHS (75 and 80 are always built).
fn extra_archs() -> Vec<u32> {
    let mut v: Vec<u32> = env::var("FORGE_CUDA_ARCHS")
        .unwrap_or_default()
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .filter(|&a| a != 75 && a != 80 && a >= 50)
        .collect();
    v.sort();
    v.dedup();
    v
}

/// `extra_cubins.rs` (included by lib.rs): the native cubins of the extra architectures as (major, minor, image).
fn write_extra_cubins(out: &Path, archs: &[u32]) {
    let mut s = String::from("pub static EXTRA_CUBINS: &[(i32, i32, &[u8])] = &[\n");
    for a in archs {
        s += &format!(
            "    ({}, {}, include_bytes!(concat!(env!(\"OUT_DIR\"), \"/forge_sm{a}.cubin\"))),\n",
            a / 10,
            a % 10
        );
    }
    s += "];\n";
    std::fs::write(out.join("extra_cubins.rs"), s).unwrap();
}

fn find_nvcc() -> Option<PathBuf> {
    if let Ok(p) = env::var("NVCC") {
        return Some(PathBuf::from(p));
    }
    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Ok(p) = env::var("CUDA_PATH") {
        candidates.push(Path::new(&p).join("bin").join(if cfg!(windows) {
            "nvcc.exe"
        } else {
            "nvcc"
        }));
    }
    if let Ok(h) = env::var("HOME") {
        candidates.push(Path::new(&h).join("cuda-13.2.0/bin/nvcc"));
    }
    candidates.push(PathBuf::from("/usr/local/cuda/bin/nvcc"));
    for c in candidates {
        if c.exists() {
            return Some(c);
        }
    }
    // PATH
    let probe = Command::new(if cfg!(windows) { "nvcc.exe" } else { "nvcc" })
        .arg("--version")
        .output();
    if probe.map(|o| o.status.success()).unwrap_or(false) {
        return Some(PathBuf::from(if cfg!(windows) {
            "nvcc.exe"
        } else {
            "nvcc"
        }));
    }
    None
}

fn main() {
    println!("cargo:rerun-if-changed=kernels");
    println!("cargo:rerun-if-env-changed=NVCC");
    println!("cargo:rerun-if-env-changed=FORGE_NO_SM80");
    println!("cargo:rerun-if-env-changed=FORGE_DEV_ARCH_ONLY");
    println!("cargo:rerun-if-env-changed=FORGE_CUDA_ARCHS");
    println!("cargo:rerun-if-env-changed=NVCC_CCBIN");
    let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    let archs = extra_archs();
    write_extra_cubins(&out, &archs);
    let src = Path::new(&env::var("CARGO_MANIFEST_DIR").unwrap()).join("kernels/all.cu");
    let cubin = out.join("forge_sm75.cubin");
    let ptx = out.join("forge_compute75.ptx");
    let cubin80 = out.join("forge_sm80.cubin");
    let ptx80 = out.join("forge_compute80.ptx");
    let nvcc = match find_nvcc() {
        Some(n) => n,
        None => {
            println!("cargo:warning=nvcc not found: forge-kernels is built WITHOUT GPU kernels");
            std::fs::write(&cubin, b"").unwrap();
            std::fs::write(&ptx, b"").unwrap();
            std::fs::write(&cubin80, b"").unwrap();
            std::fs::write(&ptx80, b"").unwrap();
            for a in &archs {
                std::fs::write(out.join(format!("forge_sm{a}.cubin")), b"").unwrap();
            }
            return;
        }
    };
    if cfg!(windows)
        && env::var_os("NVCC_CCBIN").is_none()
        && Command::new("cl.exe").output().is_err()
    {
        match windows_ccbin() {
            // nvcc reads NVCC_CCBIN (edition 2021: set_var is safe; the build script has one thread)
            Some(dir) => env::set_var("NVCC_CCBIN", dir),
            None => println!("cargo:warning=nvcc needs the MSVC host compiler (cl.exe) and none was found: run tools/build_windows.ps1, build from a VS 2022 developer prompt, or set NVCC_CCBIN to the folder of cl.exe"),
        }
    }
    let common = [
        "-O3",
        "-lineinfo",
        "--use_fast_math",
        "-std=c++17",
        "-Xptxas",
        "-v",
    ];
    // FORGE_DEV_ARCH_ONLY=1 (development builds for one GPU): only the native cubins of FORGE_CUDA_ARCHS are compiled, the portable images stay empty (one nvcc run of about 90 s instead of five)
    if env::var("FORGE_DEV_ARCH_ONLY").map(|v| v == "1").unwrap_or(false) && !archs.is_empty() {
        for f in [&cubin, &ptx, &cubin80, &ptx80] {
            std::fs::write(f, b"").unwrap();
        }
        for a in &archs {
            let status = Command::new(&nvcc)
                .arg("-cubin")
                .arg(format!("-arch=sm_{a}"))
                .args(common)
                .arg("-o")
                .arg(out.join(format!("forge_sm{a}.cubin")))
                .arg(&src)
                .status()
                .expect("failed to run nvcc");
            assert!(status.success(), "nvcc failed (cubin sm_{a})");
        }
        return;
    }
    let status = Command::new(&nvcc)
        .args(["-cubin", "-arch=sm_75"])
        .args(common)
        .arg("-o")
        .arg(&cubin)
        .arg(&src)
        .status()
        .expect("failed to run nvcc");
    assert!(status.success(), "nvcc failed (cubin)");
    let status = Command::new(&nvcc)
        .args(["-ptx", "-arch=compute_75"])
        .args(common)
        .arg("-o")
        .arg(&ptx)
        .arg(&src)
        .status()
        .expect("failed to run nvcc");
    assert!(status.success(), "nvcc failed (ptx)");
    // the native cubins of FORGE_CUDA_ARCHS (same source and flags)
    for a in &archs {
        let status = Command::new(&nvcc)
            .arg("-cubin")
            .arg(format!("-arch=sm_{a}"))
            .args(common)
            .arg("-o")
            .arg(out.join(format!("forge_sm{a}.cubin")))
            .arg(&src)
            .status()
            .expect("failed to run nvcc");
        assert!(status.success(), "nvcc failed (cubin sm_{a})");
    }
    // sm_80 (Ampere, and every 8.x device: a cubin runs on the same major version with a higher minor) and PTX for compute_80 (Hopper and newer JIT it): the same source, with the Ampere GEMMs (gemm_s80.cu)
    // using cp.async and mma.m16n8k16 instead of their emulation. FORGE_NO_SM80=1 skips these two nvcc runs (faster builds on the Turing homelab when they are not needed).
    if env::var("FORGE_NO_SM80").map(|v| v == "1").unwrap_or(false) {
        std::fs::write(&cubin80, b"").unwrap();
        std::fs::write(&ptx80, b"").unwrap();
        return;
    }
    let status = Command::new(&nvcc)
        .args(["-cubin", "-arch=sm_80"])
        .args(common)
        .arg("-o")
        .arg(&cubin80)
        .arg(&src)
        .status()
        .expect("failed to run nvcc");
    assert!(status.success(), "nvcc failed (cubin sm_80)");
    let status = Command::new(&nvcc)
        .args(["-ptx", "-arch=compute_80"])
        .args(common)
        .arg("-o")
        .arg(&ptx80)
        .arg(&src)
        .status()
        .expect("failed to run nvcc");
    assert!(status.success(), "nvcc failed (ptx compute_80)");
}
