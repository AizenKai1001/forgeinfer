//! The ForgeInfer quantizer: f32 weights -> GGUF block formats (Q8_0, Q4_0, Q5_0, Q4_K, Q5_K, Q6_K), in exactly the byte layouts `forge-gguf` decodes (and llama.cpp reads).
//!
//! What is ours is the choice of the scales: every sub-block's scale (and minimum) is the least-squares fit of the block's values *weighted by how much the model cares about each input
//! column* (`Weighting::Importance`: the mean square of the activation that multiplies that column, from a calibration text), then the 6-bit scales of the K-quants are refined jointly with the
//! integer codes. Without calibration data the weights are uniform (plain mean squared error) or follow llama.cpp's magnitude heuristic.
//!
//! A matrix is [rows][cols] with `cols` a multiple of the block size (32, or 256 for the K-quants); every row is encoded on its own, rows in parallel.

pub mod gptq;
mod kquant;
mod simple;

pub use gptq::{gptq_matrix, inverse_cholesky_upper, GptqOpts};

use forge_core::{Error, Result};
use forge_gguf::DType;

/// How the squared error of each weight counts when a block's scale is fitted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Weighting {
    /// every weight counts the same (plain mean squared error)
    Uniform,
    /// llama.cpp's heuristic without calibration data: sqrt(mean x^2 of the 256-block) + |x|
    Magnitude,
    /// the importance of the weight's input column (mean square of the activations that multiply it)
    Importance,
    /// llama.cpp's use of an importance matrix: importance * sqrt(sigma^2 + x^2)
    ImportanceMagnitude,
}

#[derive(Clone, Debug)]
pub struct EncodeOpts {
    pub weighting: Weighting,
    /// refine the quantized 6-bit scales and minimums of the K-quants jointly with the codes (slower, a little better)
    pub refine: bool,
    /// worker threads (0: all cores)
    pub threads: usize,
}

impl Default for EncodeOpts {
    fn default() -> Self {
        EncodeOpts {
            weighting: Weighting::Uniform,
            refine: true,
            threads: 0,
        }
    }
}

/// Bytes of one encoded row of `cols` values.
pub fn row_bytes(dtype: DType, cols: usize) -> usize {
    dtype.bytes_for(cols)
}

/// Encode one row. `imp` (the importance of each column, same length as `x`) is only used by the importance weightings.
pub fn encode_row(
    dtype: DType,
    x: &[f32],
    imp: Option<&[f32]>,
    opts: &EncodeOpts,
    out: &mut [u8],
) -> Result<()> {
    match dtype {
        DType::Q8_0 => simple::q8_0(x, out),
        DType::Q4_0 => simple::q4_0(x, imp, opts, out),
        DType::Q5_0 => simple::q5_0(x, imp, opts, out),
        DType::Q4_K => kquant::q4_k(x, imp, opts, out, 15),
        DType::Q5_K => kquant::q4_k(x, imp, opts, out, 31),
        DType::Q6_K => kquant::q6_k(x, imp, opts, out),
        other => {
            return Err(Error::unsupported(format!(
                "quantization to {}",
                other.name()
            )))
        }
    }
    Ok(())
}

/// Encode a matrix [rows][cols] (row-major f32) in `dtype`; `imp` has `cols` entries when the weighting needs it.
pub fn encode_matrix(
    dtype: DType,
    rows: usize,
    cols: usize,
    w: &[f32],
    imp: Option<&[f32]>,
    opts: &EncodeOpts,
) -> Result<Vec<u8>> {
    let bv = dtype.block_values();
    if bv == 0 || !cols.is_multiple_of(bv) {
        return Err(Error::format(format!(
            "{}: {cols} columns are not a multiple of the block size {bv}",
            dtype.name()
        )));
    }
    if w.len() != rows * cols {
        return Err(Error::format(format!(
            "matrix of {rows} x {cols} given {} values",
            w.len()
        )));
    }
    if let Some(i) = imp {
        if i.len() != cols {
            return Err(Error::format(format!(
                "importance has {} entries for {cols} columns",
                i.len()
            )));
        }
    }
    if matches!(
        opts.weighting,
        Weighting::Importance | Weighting::ImportanceMagnitude
    ) && imp.is_none()
    {
        return Err(Error::format(
            "importance weighting needs the importance of the columns",
        ));
    }
    let rb = row_bytes(dtype, cols);
    let mut out = vec![0u8; rows * rb];
    let threads = if opts.threads == 0 {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
    } else {
        opts.threads
    }
    .min(rows.max(1));
    let per = rows.div_ceil(threads);
    let errs: Vec<Option<Error>> = std::thread::scope(|s| {
        let handles: Vec<_> = out
            .chunks_mut(per * rb)
            .zip(w.chunks(per * cols))
            .map(|(o, x)| {
                s.spawn(move || {
                    for (orow, xrow) in o.chunks_mut(rb).zip(x.chunks(cols)) {
                        if let Err(e) = encode_row(dtype, xrow, imp, opts, orow) {
                            return Some(e);
                        }
                    }
                    None
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    if let Some(e) = errs.into_iter().flatten().next() {
        return Err(e);
    }
    Ok(out)
}

/// Root-mean-square error of the encoding relative to the root-mean-square of the values (diagnostic).
pub fn relative_rmse(
    dtype: DType,
    rows: usize,
    cols: usize,
    w: &[f32],
    encoded: &[u8],
) -> Result<f64> {
    let mut back = vec![0f32; rows * cols];
    forge_gguf::quant::dequantize_into(dtype, encoded, &mut back)?;
    let (mut e, mut n) = (0f64, 0f64);
    for (a, b) in w.iter().zip(&back) {
        e += ((*a - *b) as f64).powi(2);
        n += (*a as f64).powi(2);
    }
    Ok((e / n.max(1e-30)).sqrt())
}

/// Per-element weights of one block of values `x` (columns `c0..c0+x.len()` of the row) for the fit, according to the weighting. `av_x` is llama.cpp's per-256-block magnitude term.
pub(crate) fn element_weights(
    opts: &EncodeOpts,
    x: &[f32],
    imp: Option<&[f32]>,
    av_x: f32,
    out: &mut [f32],
) {
    match opts.weighting {
        Weighting::Uniform => out.iter_mut().for_each(|w| *w = 1.0),
        Weighting::Magnitude => {
            for (w, v) in out.iter_mut().zip(x) {
                *w = av_x + v.abs();
            }
        }
        Weighting::Importance => {
            let imp = imp.expect("importance");
            for (w, i) in out.iter_mut().zip(imp) {
                *w = i.max(0.0);
            }
        }
        Weighting::ImportanceMagnitude => {
            let imp = imp.expect("importance");
            for ((w, i), v) in out.iter_mut().zip(imp).zip(x) {
                *w = i.max(0.0) * (av_x * av_x + v * v).sqrt();
            }
        }
    }
}

/// llama.cpp's `av_x`: sqrt(2 * mean(x^2)) of the block.
pub(crate) fn av_x(x: &[f32]) -> f32 {
    let s: f32 = x.iter().map(|v| v * v).sum();
    (2.0 * s / x.len() as f32).sqrt()
}
