//! Repacking of GGUF quantized blocks into the planar layouts of the quantized decode kernels (kernels/decode_q.cu; tools/ref_qdot.py is the executable specification). Each plane holds one
//! field class of all blocks in order, so a lane-task is one 16-byte load from the widest plane; the total bytes per block equal the native block.
//!
//!   Q8_0  p0 int8 q (32 B/block)                          p1 fp16 d
//!   Q4_0  p0 q (16 B/block, byte j = weight j | weight j + 16 << 4)       p1 fp16 d
//!   Q5_0  p0 q (16 B/block)   p1 u32 5th-bit word, transposed: bit (8 i + p) = old bit (4 p + i)       p2 fp16 d
//!   Q4_K  p0 qs (128 B/super-block)                       p1 header (d, dmin, 12 scale bytes: 16 B)
//!   Q5_K  p0 qs (128 B)   p1 8 x u32 transposed 5th-bit words (task t = 2 c + half: bit (8 i + p) = bit (2 c + p / 4) of qh[16 half + 4 (p % 4) + i])       p2 header (16 B)
//!   Q6_K  p0 ql (128 B)   p1 qh (64 B)   p2 sc' (16 B: sc'[4 (2 h + u) + g] = sc[8 h + u + 2 g])   p3 fp16 d

use forge_core::{Error, Result};
use forge_gguf::DType;

/// Weight format of the quantized decode kernels (the numbers are the FMT_* constants of decode_q.cu).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum QFmt {
    Q8_0 = 0,
    Q4_0 = 1,
    Q5_0 = 2,
    Q4_K = 3,
    Q5_K = 4,
    Q6_K = 5,
    /// host memory only: the native blocks in one plane, read by the CPU kernels and the dequantizing prefill kernel
    Q3_K = 6,
    Q2_K = 7,
}

impl QFmt {
    pub fn from_dtype(d: DType) -> Option<QFmt> {
        match d {
            DType::Q8_0 => Some(QFmt::Q8_0),
            DType::Q4_0 => Some(QFmt::Q4_0),
            DType::Q5_0 => Some(QFmt::Q5_0),
            DType::Q4_K => Some(QFmt::Q4_K),
            DType::Q5_K => Some(QFmt::Q5_K),
            DType::Q6_K => Some(QFmt::Q6_K),
            DType::Q3_K => Some(QFmt::Q3_K),
            DType::Q2_K => Some(QFmt::Q2_K),
            _ => None,
        }
    }

    pub fn dtype(self) -> DType {
        match self {
            QFmt::Q8_0 => DType::Q8_0,
            QFmt::Q4_0 => DType::Q4_0,
            QFmt::Q5_0 => DType::Q5_0,
            QFmt::Q4_K => DType::Q4_K,
            QFmt::Q5_K => DType::Q5_K,
            QFmt::Q6_K => DType::Q6_K,
            QFmt::Q3_K => DType::Q3_K,
            QFmt::Q2_K => DType::Q2_K,
        }
    }

    /// (values per block, native bytes per block)
    pub fn block(self) -> (usize, usize) {
        match self {
            QFmt::Q8_0 => (32, 34),
            QFmt::Q4_0 => (32, 18),
            QFmt::Q5_0 => (32, 22),
            QFmt::Q4_K => (256, 144),
            QFmt::Q5_K => (256, 176),
            QFmt::Q6_K => (256, 210),
            QFmt::Q3_K => (256, 110),
            QFmt::Q2_K => (256, 84),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            QFmt::Q8_0 => "Q8_0",
            QFmt::Q4_0 => "Q4_0",
            QFmt::Q5_0 => "Q5_0",
            QFmt::Q4_K => "Q4_K",
            QFmt::Q5_K => "Q5_K",
            QFmt::Q6_K => "Q6_K",
            QFmt::Q3_K => "Q3_K",
            QFmt::Q2_K => "Q2_K",
        }
    }
}

/// The one-thread version, the specification the parallel `repack` is tested against.
#[cfg(test)]
fn repack_reference(fmt: QFmt, native: &[u8]) -> Result<Vec<Vec<u8>>> {
    let (_, bb) = fmt.block();
    if native.len() % bb != 0 {
        return Err(Error::format(format!(
            "{}: {} bytes is not a whole number of {bb}-byte blocks",
            fmt.name(),
            native.len()
        )));
    }
    let nb = native.len() / bb;
    let blocks = native.chunks_exact(bb);
    Ok(match fmt {
        QFmt::Q8_0 => {
            let (mut q, mut d) = (Vec::with_capacity(nb * 32), Vec::with_capacity(nb * 2));
            for b in blocks {
                d.extend_from_slice(&b[0..2]);
                q.extend_from_slice(&b[2..34]);
            }
            vec![q, d]
        }
        QFmt::Q4_0 => {
            let (mut q, mut d) = (Vec::with_capacity(nb * 16), Vec::with_capacity(nb * 2));
            for b in blocks {
                d.extend_from_slice(&b[0..2]);
                q.extend_from_slice(&b[2..18]);
            }
            vec![q, d]
        }
        QFmt::Q5_0 => {
            let (mut q, mut h, mut d) = (
                Vec::with_capacity(nb * 16),
                Vec::with_capacity(nb * 4),
                Vec::with_capacity(nb * 2),
            );
            for b in blocks {
                d.extend_from_slice(&b[0..2]);
                let old = u32::from_le_bytes([b[2], b[3], b[4], b[5]]);
                let mut new = 0u32;
                for i in 0..4 {
                    for p in 0..8 {
                        if (old >> (4 * p + i)) & 1 != 0 {
                            new |= 1 << (8 * i + p);
                        }
                    }
                }
                h.extend_from_slice(&new.to_le_bytes());
                q.extend_from_slice(&b[6..22]);
            }
            vec![q, h, d]
        }
        QFmt::Q4_K => {
            let (mut q, mut hdr) = (Vec::with_capacity(nb * 128), Vec::with_capacity(nb * 16));
            for b in blocks {
                hdr.extend_from_slice(&b[0..16]);
                q.extend_from_slice(&b[16..144]);
            }
            vec![q, hdr]
        }
        QFmt::Q5_K => {
            let (mut q, mut h, mut hdr) = (
                Vec::with_capacity(nb * 128),
                Vec::with_capacity(nb * 32),
                Vec::with_capacity(nb * 16),
            );
            for b in blocks {
                hdr.extend_from_slice(&b[0..16]);
                let qh = &b[16..48];
                for t in 0..8 {
                    let (c, hf) = (t / 2, t % 2);
                    let mut word = 0u32;
                    for p in 0..8 {
                        for i in 0..4 {
                            if (qh[16 * hf + 4 * (p % 4) + i] >> (2 * c + p / 4)) & 1 != 0 {
                                word |= 1 << (8 * i + p);
                            }
                        }
                    }
                    h.extend_from_slice(&word.to_le_bytes());
                }
                q.extend_from_slice(&b[48..176]);
            }
            vec![q, h, hdr]
        }
        QFmt::Q6_K => {
            let (mut ql, mut qh, mut sc, mut d) = (
                Vec::with_capacity(nb * 128),
                Vec::with_capacity(nb * 64),
                Vec::with_capacity(nb * 16),
                Vec::with_capacity(nb * 2),
            );
            for b in blocks {
                ql.extend_from_slice(&b[0..128]);
                qh.extend_from_slice(&b[128..192]);
                let s = &b[192..208];
                for h in 0..2 {
                    for u in 0..2 {
                        for g in 0..4 {
                            sc.push(s[8 * h + u + 2 * g]);
                        }
                    }
                }
                d.extend_from_slice(&b[208..210]);
            }
            vec![ql, qh, sc, d]
        }
        QFmt::Q3_K | QFmt::Q2_K => vec![native.to_vec()],
    })
}

/// Bytes per block of every plane of a format, in plane order (see the table at the top).
fn plane_widths(fmt: QFmt) -> &'static [usize] {
    match fmt {
        QFmt::Q8_0 => &[32, 2],
        QFmt::Q4_0 => &[16, 2],
        QFmt::Q5_0 => &[16, 4, 2],
        QFmt::Q4_K => &[128, 16],
        QFmt::Q5_K => &[128, 32, 16],
        QFmt::Q6_K => &[128, 64, 16, 2],
        QFmt::Q3_K | QFmt::Q2_K => &[],
    }
}

/// Repack a run of whole blocks into the matching run of every plane (`outs[p]` holds `plane_widths[p]` bytes per block).
fn repack_blocks(fmt: QFmt, blocks: &[u8], outs: &mut [&mut [u8]]) {
    let (_, bb) = fmt.block();
    for (i, b) in blocks.chunks_exact(bb).enumerate() {
        match fmt {
            QFmt::Q8_0 => {
                outs[0][32 * i..32 * i + 32].copy_from_slice(&b[2..34]);
                outs[1][2 * i..2 * i + 2].copy_from_slice(&b[0..2]);
            }
            QFmt::Q4_0 => {
                outs[0][16 * i..16 * i + 16].copy_from_slice(&b[2..18]);
                outs[1][2 * i..2 * i + 2].copy_from_slice(&b[0..2]);
            }
            QFmt::Q5_0 => {
                let old = u32::from_le_bytes([b[2], b[3], b[4], b[5]]);
                let mut new = 0u32;
                for k in 0..4 {
                    for q in 0..8 {
                        if (old >> (4 * q + k)) & 1 != 0 {
                            new |= 1 << (8 * k + q);
                        }
                    }
                }
                outs[0][16 * i..16 * i + 16].copy_from_slice(&b[6..22]);
                outs[1][4 * i..4 * i + 4].copy_from_slice(&new.to_le_bytes());
                outs[2][2 * i..2 * i + 2].copy_from_slice(&b[0..2]);
            }
            QFmt::Q4_K => {
                outs[0][128 * i..128 * i + 128].copy_from_slice(&b[16..144]);
                outs[1][16 * i..16 * i + 16].copy_from_slice(&b[0..16]);
            }
            QFmt::Q5_K => {
                let qh = &b[16..48];
                for t in 0..8 {
                    let (c, hf) = (t / 2, t % 2);
                    let mut word = 0u32;
                    for q in 0..8 {
                        for k in 0..4 {
                            if (qh[16 * hf + 4 * (q % 4) + k] >> (2 * c + q / 4)) & 1 != 0 {
                                word |= 1 << (8 * k + q);
                            }
                        }
                    }
                    outs[1][32 * i + 4 * t..32 * i + 4 * t + 4].copy_from_slice(&word.to_le_bytes());
                }
                outs[0][128 * i..128 * i + 128].copy_from_slice(&b[48..176]);
                outs[2][16 * i..16 * i + 16].copy_from_slice(&b[0..16]);
            }
            QFmt::Q6_K => {
                outs[0][128 * i..128 * i + 128].copy_from_slice(&b[0..128]);
                outs[1][64 * i..64 * i + 64].copy_from_slice(&b[128..192]);
                let sc = &b[192..208];
                let mut n = 0;
                for h in 0..2 {
                    for u in 0..2 {
                        for g in 0..4 {
                            outs[2][16 * i + n] = sc[8 * h + u + 2 * g];
                            n += 1;
                        }
                    }
                }
                outs[3][2 * i..2 * i + 2].copy_from_slice(&b[208..210]);
            }
            QFmt::Q3_K | QFmt::Q2_K => unreachable!("kept whole"),
        }
    }
}

/// Repack `native` (a whole number of blocks) into the planes of `fmt`. Big matrices are cut into runs of blocks that go through all the cores (a 9B model has 38 million blocks: on one thread this was
/// 7.5 of the 9.1 seconds of loading it).
pub fn repack(fmt: QFmt, native: &[u8]) -> Result<Vec<Vec<u8>>> {
    let (_, bb) = fmt.block();
    if native.len() % bb != 0 {
        return Err(Error::format(format!(
            "{}: {} bytes is not a whole number of {bb}-byte blocks",
            fmt.name(),
            native.len()
        )));
    }
    let widths = plane_widths(fmt);
    if widths.is_empty() {
        return Ok(vec![native.to_vec()]);
    }
    let nb = native.len() / bb;
    let mut planes: Vec<Vec<u8>> = widths.iter().map(|w| vec![0u8; nb * w]).collect();
    let threads = if nb < 8192 {
        1
    } else {
        std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1).clamp(1, 16)
    };
    let per = nb.div_ceil(threads);
    // every plane cut into the same runs of blocks
    let mut runs: Vec<Vec<&mut [u8]>> = (0..threads).map(|_| Vec::new()).collect();
    for (pi, plane) in planes.iter_mut().enumerate() {
        for (ti, chunk) in plane.chunks_mut(per * widths[pi]).enumerate() {
            runs[ti].push(chunk);
        }
    }
    if threads == 1 {
        let mut outs = runs.pop().unwrap();
        repack_blocks(fmt, native, &mut outs);
    } else {
        std::thread::scope(|sc| {
            for (ti, mut outs) in runs.into_iter().enumerate() {
                let (lo, hi) = (ti * per, ((ti + 1) * per).min(nb));
                if lo >= hi {
                    continue;
                }
                let blocks = &native[lo * bb..hi * bb];
                sc.spawn(move || repack_blocks(fmt, blocks, &mut outs));
            }
        });
    }
    Ok(planes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_parallel_repack_equals_the_one_thread_version_for_every_format() {
        let mut x = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = move || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        for fmt in [QFmt::Q8_0, QFmt::Q4_0, QFmt::Q5_0, QFmt::Q4_K, QFmt::Q5_K, QFmt::Q6_K, QFmt::Q3_K, QFmt::Q2_K] {
            let (_, bb) = fmt.block();
            // 5 blocks (one thread), 20,001 (cut into runs; the last run is shorter) and an odd count that does not divide by the thread count
            for nb in [5usize, 20_001, 33_333] {
                let native: Vec<u8> = (0..nb * bb).map(|_| (next() >> 33) as u8).collect();
                let want = repack_reference(fmt, &native).unwrap();
                let got = repack(fmt, &native).unwrap();
                assert_eq!(want.len(), got.len(), "{} planes", fmt.name());
                for (p, (a, b)) in want.iter().zip(got.iter()).enumerate() {
                    assert!(a == b, "{} plane {p} differs for {nb} blocks", fmt.name());
                }
            }
        }
        // a size that is not a whole number of blocks is still an error
        assert!(repack(QFmt::Q4_K, &[0u8; 143]).is_err());
    }
}
