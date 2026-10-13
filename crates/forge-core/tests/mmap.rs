//! `Mmap::open` gives the bytes of the file (a real mapping on Unix and Windows), for an empty file, a file that is not a multiple of the page size, and several maps of the same file at once.

use forge_core::Mmap;
use std::path::PathBuf;

fn scratch(name: &str) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("forge_core_mmap");
    std::fs::create_dir_all(&dir).unwrap();
    dir.join(name)
}

fn bytes(n: usize, seed: u64) -> Vec<u8> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (s >> 33) as u8
        })
        .collect()
}

#[test]
fn a_mapped_file_reads_back_its_bytes() {
    for (i, n) in [1usize, 4095, 4096, 65537, 3 << 20].into_iter().enumerate() {
        let path = scratch(&format!("file_{i}.bin"));
        let want = bytes(n, i as u64 + 1);
        std::fs::write(&path, &want).unwrap();
        let m = Mmap::open(&path).unwrap();
        assert_eq!(m.len(), n);
        assert!(m[..] == want[..], "contents differ for {n} bytes");
        // a second map of the same file while the first one is alive, then both are dropped (unmapped)
        let m2 = Mmap::open(&path).unwrap();
        assert!(m2[..] == m[..]);
        // Windows refuses to truncate a file that has a mapped view: proof that this is a mapping and not a copy
        #[cfg(windows)]
        assert!(
            std::fs::write(&path, b"x").is_err(),
            "a mapped file could be truncated: not a real mapping?"
        );
        drop(m);
        assert_eq!(m2[n - 1], want[n - 1]);
        drop(m2);
        // the file can be rewritten after the maps are gone
        std::fs::write(&path, b"x").unwrap();
        assert_eq!(&Mmap::open(&path).unwrap()[..], b"x");
    }
}

#[test]
fn an_empty_file_maps_to_an_empty_slice() {
    let path = scratch("empty.bin");
    std::fs::write(&path, b"").unwrap();
    let m = Mmap::open(&path).unwrap();
    assert_eq!(m.len(), 0);
    assert!(m.is_empty());
}

#[test]
fn a_missing_file_is_an_error_that_names_it() {
    let path = scratch("does_not_exist.bin");
    let _ = std::fs::remove_file(&path);
    let e = Mmap::open(&path)
        .err()
        .expect("opening a missing file must fail");
    assert!(e.to_string().contains("does_not_exist.bin"), "{e}");
}
