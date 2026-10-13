//! The compiled kernel images, embedded at build time (see build.rs). `CUBIN_SM75` runs on compute capability 7.5 only, `CUBIN_SM80` on every 8.x device (Ampere, Ada); the PTX images are JIT-compiled by the
//! driver for newer GPUs (`PTX_COMPUTE80` for 9.x and up: it has the Ampere-tuned GEMMs, `PTX_COMPUTE75` is the fallback). `EXTRA_CUBINS` holds the native cubins of FORGE_CUDA_ARCHS (empty by default).

pub static CUBIN_SM75: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/forge_sm75.cubin"));
pub static PTX_COMPUTE75: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/forge_compute75.ptx"));
pub static CUBIN_SM80: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/forge_sm80.cubin"));
pub static PTX_COMPUTE80: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/forge_compute80.ptx"));
// (major, minor, image) of every architecture in FORGE_CUDA_ARCHS at build time, e.g. 89 for Ada (RTX 40xx, L4)
include!(concat!(env!("OUT_DIR"), "/extra_cubins.rs"));

/// The image to load on a device of compute capability `major.minor`, and whether it is PTX (JIT) instead of a ready cubin.
pub fn image_for(major: i32, minor: i32) -> Option<(&'static [u8], bool)> {
    // a native cubin built for exactly this device wins; FORGE_NO_EXTRA_CUBIN=1 skips it (A/B against the portable images below with the same binary)
    if let Some(e) = EXTRA_CUBINS
        .iter()
        .find(|e| e.0 == major && e.1 == minor && !e.2.is_empty())
    {
        if std::env::var("FORGE_NO_EXTRA_CUBIN")
            .map(|v| v == "0")
            .unwrap_or(true)
        {
            eprintln!("[kernels] native sm_{major}{minor} cubin (FORGE_CUDA_ARCHS); FORGE_NO_EXTRA_CUBIN=1 selects the portable image");
            return Some((e.2, false));
        }
        eprintln!("[kernels] FORGE_NO_EXTRA_CUBIN: the native sm_{major}{minor} cubin is skipped");
    }
    if major == 7 && minor == 5 && !CUBIN_SM75.is_empty() {
        return Some((CUBIN_SM75, false));
    }
    if major == 8 && !CUBIN_SM80.is_empty() {
        return Some((CUBIN_SM80, false));
    }
    if major >= 9 && !PTX_COMPUTE80.is_empty() {
        return Some((PTX_COMPUTE80, true));
    }
    if (major, minor) >= (7, 5) && !PTX_COMPUTE75.is_empty() {
        return Some((PTX_COMPUTE75, true));
    }
    None
}

/// Does the image chosen for this device contain the Ampere GEMMs with real cp.async / m16n8k16 (`*_s80` kernels; below sm_80 they exist only as an emulation for tests)?
pub fn has_s80(major: i32) -> bool {
    major >= 8 && !CUBIN_SM80.is_empty()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_extra_cubins_are_the_architectures_of_forge_cuda_archs() {
        // what build.rs saw (cargo reruns it when FORGE_CUDA_ARCHS changes); 75 and 80 are always built and never extra
        let want: Vec<(i32, i32)> = {
            let mut v: Vec<u32> = option_env!("FORGE_CUDA_ARCHS")
                .unwrap_or("")
                .split(',')
                .filter_map(|s| s.trim().parse().ok())
                .filter(|&a| a != 75 && a != 80 && a >= 50)
                .collect();
            v.sort();
            v.dedup();
            v.iter()
                .map(|a| ((a / 10) as i32, (a % 10) as i32))
                .collect()
        };
        let got: Vec<(i32, i32)> = EXTRA_CUBINS.iter().map(|e| (e.0, e.1)).collect();
        assert_eq!(got, want);
        // with nvcc every image is a cubin (ELF), never PTX text
        for e in EXTRA_CUBINS.iter().filter(|e| !e.2.is_empty()) {
            assert_eq!(
                &e.2[..4],
                b"\x7fELF",
                "sm_{}{} image is not a cubin",
                e.0,
                e.1
            );
        }
    }

    #[test]
    fn devices_get_the_images_they_got_before_unless_an_extra_cubin_matches() {
        let extra = |ma: i32, mi: i32| {
            EXTRA_CUBINS
                .iter()
                .any(|e| e.0 == ma && e.1 == mi && !e.2.is_empty())
        };
        if !CUBIN_SM75.is_empty() && !extra(7, 5) {
            assert_eq!(
                image_for(7, 5).map(|i| i.0.as_ptr()),
                Some(CUBIN_SM75.as_ptr())
            );
        }
        for mi in [0, 6, 9] {
            if !CUBIN_SM80.is_empty() && !extra(8, mi) {
                assert_eq!(
                    image_for(8, mi).map(|i| (i.0.as_ptr(), i.1)),
                    Some((CUBIN_SM80.as_ptr(), false))
                );
            }
            if extra(8, mi) && std::env::var_os("FORGE_NO_EXTRA_CUBIN").is_none() {
                let e = EXTRA_CUBINS.iter().find(|e| e.0 == 8 && e.1 == mi).unwrap();
                assert_eq!(
                    image_for(8, mi).map(|i| (i.0.as_ptr(), i.1)),
                    Some((e.2.as_ptr(), false))
                );
            }
        }
        if !PTX_COMPUTE80.is_empty() && !extra(9, 0) {
            assert_eq!(
                image_for(9, 0).map(|i| (i.0.as_ptr(), i.1)),
                Some((PTX_COMPUTE80.as_ptr(), true))
            );
        }
        if !extra(6, 1) {
            assert!(image_for(6, 1).is_none(), "Pascal has no image");
        }
    }
}
