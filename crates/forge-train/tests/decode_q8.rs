//! Q8 weights in the KV-cache decoder (`Decoder::enable_q8`): the logits of a Q8 step against the fp16 step (the trainer's own arithmetic, bitwise) on the same prefix, for models shaped like Forge-1
//! (GQA 3:1, qk-norm, ReLU^2, lambdas, value embeddings, tied head) and for the other structure (head dimension 128, SwiGLU, skips, untied head), through the prompt path (dequantized matrices into
//! the tensor-core GEMMs) and the single-token path (fused Q8 kernels, captured as a graph); the freed fp32 / fp16 weights of `release`; the default decoder keeps its bits.

use forge_cuda::Device;
use forge_train::{Decoder, TrainConfig, Trainer};

/// The tests of this file take turns on the GPU (a CUDA graph capture in one thread and a synchronous copy in another one invalidate each other).
static GPU: std::sync::Mutex<()> = std::sync::Mutex::new(());

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
    fn unit(&mut self) -> f32 {
        self.next() as f32 / (1u64 << 31) as f32
    }
}

/// (largest |diff| / largest |reference logit|, mean |diff| / mean |reference logit|, cosine)
fn rel_err(got: &[f32], want: &[f32]) -> (f64, f64, f64) {
    let (mut mx, mut sd, mut sa, mut amax) = (0f64, 0f64, 0f64, 0f64);
    let (mut ab, mut aa, mut bb) = (0f64, 0f64, 0f64);
    for (g, w) in got.iter().zip(want) {
        let d = (*g - *w).abs() as f64;
        mx = mx.max(d);
        sd += d;
        sa += w.abs() as f64;
        amax = amax.max(w.abs() as f64);
        ab += *g as f64 * *w as f64;
        aa += (*g as f64).powi(2);
        bb += (*w as f64).powi(2);
    }
    (
        mx / amax.max(1e-30),
        sd / sa.max(1e-30),
        ab / (aa.sqrt() * bb.sqrt()).max(1e-30),
    )
}

fn forge1_cfg() -> TrainConfig {
    // the production structure in small: 6 layers, d 192, 3 query heads on 1 KV head (head_dim 64), ReLU^2 MLP of 768, tied head, qk-norm, lambdas, value embeddings on layers 0, 1, 4, 5
    TrainConfig {
        n_layer: 6,
        n_embd: 192,
        n_head: 3,
        n_kv: 1,
        head_dim: 64,
        n_ff: 768,
        vocab: 2048,
        seq_len: 320,
        batch: 1,
        qk_norm: true,
        ce_chunk: 64,
        relu2: true,
        lambdas: true,
        ve_layers: vec![0, 1, 4, 5],
        ..TrainConfig::default()
    }
}

fn other_cfg() -> TrainConfig {
    // the other attention template (head dimension 128), the SwiGLU MLP (gate | up in one matrix), separate head, skips
    TrainConfig {
        n_layer: 4,
        n_embd: 256,
        n_head: 4,
        n_kv: 2,
        head_dim: 128,
        n_ff: 512,
        vocab: 1024,
        seq_len: 192,
        batch: 1,
        qk_norm: true,
        ce_chunk: 64,
        untied: true,
        relu2: false,
        lambdas: true,
        skips: true,
        ve_layers: vec![0, 3],
        ..TrainConfig::default()
    }
}

/// A random trainer with the scalars moved away from their initial values (so that every mixing term matters) and weights of a usual size.
fn trainer(cfg: TrainConfig, seed: u64) -> Trainer {
    let mut tr = Trainer::new_inference(cfg, 0).unwrap();
    tr.init_random(seed).unwrap();
    let mut named = tr.params_named().unwrap();
    let mut rng = Lcg(seed + 1);
    for k in ["lam", "velam"] {
        if let Some(v) = named.get_mut(k) {
            for x in v.iter_mut() {
                *x += 0.4 * (rng.unit() - 0.5);
            }
        }
    }
    if let Some(v) = named.get_mut("head") {
        // the separate head starts at zero
        for x in v.iter_mut() {
            *x = 0.05 * (rng.unit() - 0.5);
        }
    }
    tr.set_params(&named).unwrap();
    tr
}

/// Feed `toks` to a fp16 and a Q8 decoder (the first `prompt` tokens at once, then one at a time) and return the per-position errors; the Q8 decoder is `q`.
fn compare(
    name: &str,
    tr: &Trainer,
    fp: &mut Decoder,
    q: &mut Decoder,
    toks: &[u32],
    prompt: usize,
    limit: (f64, f64),
) {
    let mut worst = (0f64, 0f64, 1f64);
    let mut sum = (0f64, 0f64);
    let mut n = 0usize;
    let mut top1 = 0usize;
    let mut record = |a: &[f32], b: &[f32]| {
        let (mx, mean, cos) = rel_err(a, b);
        worst.0 = worst.0.max(mx);
        worst.1 = worst.1.max(mean);
        worst.2 = worst.2.min(cos);
        sum.0 += mx;
        sum.1 += mean;
        n += 1;
        let arg = |l: &[f32]| (0..l.len()).max_by(|&i, &j| l[i].total_cmp(&l[j])).unwrap();
        top1 += (arg(a) == arg(b)) as usize;
    };
    let a = fp.feed(tr, &toks[..prompt]).unwrap();
    let b = q.feed(tr, &toks[..prompt]).unwrap();
    record(&b, &a);
    for i in prompt..toks.len() {
        let a = fp.feed(tr, &toks[i..i + 1]).unwrap();
        let b = q.feed(tr, &toks[i..i + 1]).unwrap();
        record(&b, &a);
    }
    eprintln!(
        "{name}: {n} positions, Q8 against fp16: largest |diff| / largest |logit| mean {:.2e} worst {:.2e}, mean |diff| / mean |logit| mean {:.2e} worst {:.2e}, worst cosine {:.7}, top-1 agreement {}/{}",
        sum.0 / n as f64,
        worst.0,
        sum.1 / n as f64,
        worst.1,
        worst.2,
        top1,
        n
    );
    assert!(
        worst.0 < limit.0,
        "{name}: relative logits error {} above {}",
        worst.0,
        limit.0
    );
    assert!(
        sum.1 / (n as f64) < limit.1,
        "{name}: mean relative logits error {} above {}",
        sum.1 / n as f64,
        limit.1
    );
    assert!(worst.2 > 0.999, "{name}: cosine {}", worst.2);
}

#[test]
fn q8_logits_follow_the_fp16_decoder_on_a_forge1_shaped_model() {
    let _gpu = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if Device::count().unwrap_or(0) == 0 {
        eprintln!("skipped: no CUDA device");
        return;
    }
    let mut tr = trainer(forge1_cfg(), 11);
    let mut rng = Lcg(3);
    let toks: Vec<u32> = (0..300).map(|_| rng.next() % 2048).collect();
    let mut fp = Decoder::new(&tr, 32).unwrap();
    let mut q = Decoder::new(&tr, 32).unwrap();
    let info = q.enable_q8(&mut tr, false).unwrap();
    assert!(q.is_q8() && !fp.is_q8());
    eprintln!(
        "Q8 matrices {:.2} MB, quantized in {:.1} ms",
        info.matrix_bytes as f64 / 1e6,
        info.quantize_ms
    );
    // fp16 weights: 2 bytes per weight; Q8: 34 bytes per 32 (the matrices of the layers and the tied embedding / head)
    let matrices: usize = (6 * (192 * 320 + 192 * 192 + 768 * 192 + 192 * 768)) + 2048 * 192;
    assert_eq!(info.matrix_bytes, matrices / 16 * 17);
    compare(
        "forge1-shaped",
        &tr,
        &mut fp,
        &mut q,
        &toks,
        130,
        (0.05, 0.02),
    );
    // the same decoder through the prompt path only (dequantized matrices into the GEMMs) against the single-token Q8 kernels: the same Q8 weights, only the order of the sums and fp16 rounding differ
    let mut qa = Decoder::new(&tr, 32).unwrap();
    qa.enable_q8(&mut tr, false).unwrap();
    let mut qb = Decoder::new(&tr, 32).unwrap();
    qb.enable_q8(&mut tr, false).unwrap();
    let whole = qa.feed(&tr, &toks[..100]).unwrap();
    let mut by_step = Vec::new();
    for i in 0..100 {
        by_step = qb.feed(&tr, &toks[i..i + 1]).unwrap();
    }
    let (mx, mean, cos) = rel_err(&whole, &by_step);
    eprintln!("Q8 prompt path against Q8 single-token path: largest {mx:.2e}, mean {mean:.2e}, cosine {cos:.8}");
    assert!(mx < 0.02 && cos > 0.9999);
    // graph or not: the same kernels, the same bits
    let mut qc = Decoder::new(&tr, 32).unwrap();
    qc.enable_q8(&mut tr, false).unwrap();
    qc.use_graph = false;
    let mut again = Vec::new();
    for i in 0..100 {
        again = qc.feed(&tr, &toks[i..i + 1]).unwrap();
    }
    assert_eq!(
        again.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        by_step.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        "the captured graph and the direct launches must give the same bits"
    );
    // prefix reuse after the truncation keeps working with Q8 weights
    qb.truncate(60);
    let l = qb.feed(&tr, &toks[60..100]).unwrap();
    let (mx2, _, cos2) = rel_err(&l, &by_step);
    assert!(
        mx2 < 0.02 && cos2 > 0.9999,
        "after prefix reuse: {mx2} {cos2}"
    );
}

#[test]
fn q8_logits_follow_the_fp16_decoder_with_head_dim_128_swiglu_skips_and_a_separate_head() {
    let _gpu = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if Device::count().unwrap_or(0) == 0 {
        eprintln!("skipped: no CUDA device");
        return;
    }
    let mut tr = trainer(other_cfg(), 5);
    let mut rng = Lcg(9);
    let toks: Vec<u32> = (0..150).map(|_| rng.next() % 1024).collect();
    let mut fp = Decoder::new(&tr, 32).unwrap();
    let mut q = Decoder::new(&tr, 32).unwrap();
    q.enable_q8(&mut tr, false).unwrap();
    compare(
        "hd128-swiglu-skips",
        &tr,
        &mut fp,
        &mut q,
        &toks,
        41,
        (0.05, 0.02),
    );
}

#[test]
fn q8_logits_follow_the_fp16_decoder_with_sliding_window_layers() {
    let _gpu = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if Device::count().unwrap_or(0) == 0 {
        eprintln!("skipped: no CUDA device");
        return;
    }
    // the window changes the attention only: the Q8 matrices work as they do in the dense model
    let cfg = TrainConfig {
        n_layer: 4,
        n_embd: 128,
        n_head: 2,
        n_kv: 1,
        head_dim: 64,
        n_ff: 256,
        vocab: 512,
        seq_len: 128,
        batch: 1,
        qk_norm: true,
        ce_chunk: 64,
        relu2: true,
        lambdas: true,
        ve_layers: vec![0, 3],
        window: 16,
        window_layers: TrainConfig::auto_window_layers(4),
        ..TrainConfig::default()
    };
    let mut tr = trainer(cfg, 23);
    let mut rng = Lcg(31);
    let toks: Vec<u32> = (0..100).map(|_| rng.next() % 512).collect();
    let mut fp = Decoder::new(&tr, 32).unwrap();
    let mut q = Decoder::new(&tr, 32).unwrap();
    q.enable_q8(&mut tr, false).unwrap();
    compare("window16", &tr, &mut fp, &mut q, &toks, 30, (0.06, 0.03));
}

#[test]
fn releasing_the_fp32_and_fp16_weights_keeps_the_q8_logits_and_frees_the_memory() {
    let _gpu = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if Device::count().unwrap_or(0) == 0 {
        eprintln!("skipped: no CUDA device");
        return;
    }
    for cfg in [forge1_cfg(), other_cfg()] {
        let vocab = cfg.vocab as u32;
        let mut tr = trainer(cfg, 17);
        let mut rng = Lcg(21);
        let toks: Vec<u32> = (0..120).map(|_| rng.next() % vocab).collect();
        let mut kept = Decoder::new(&tr, 32).unwrap();
        kept.enable_q8(&mut tr, false).unwrap();
        let mut a = kept.feed(&tr, &toks[..50]).unwrap();
        for i in 50..120 {
            a = kept.feed(&tr, &toks[i..i + 1]).unwrap();
        }
        // a second decoder of the same trainer: its Q8 copy and then the release
        let mut gone = Decoder::new(&tr, 32).unwrap();
        let info = gone.enable_q8(&mut tr, true).unwrap();
        assert_eq!(info.freed_bytes, tr.n_params * 6);
        assert!(info.small_bytes > 0 && info.small_bytes < tr.n_params * 6 / 4);
        eprintln!(
            "released {:.2} MB; the decoder keeps {:.2} MB of Q8 matrices and {:.2} MB of norms, scalars and value-embedding tables",
            info.freed_bytes as f64 / 1e6,
            info.matrix_bytes as f64 / 1e6,
            info.small_bytes as f64 / 1e6
        );
        let mut b = gone.feed(&tr, &toks[..50]).unwrap();
        for i in 50..120 {
            b = gone.feed(&tr, &toks[i..i + 1]).unwrap();
        }
        assert_eq!(
            a.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            b.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            "the released trainer's Q8 decoder must give the bits of the one that kept the fp16 weights"
        );
        assert!(
            tr.logits_at(&toks, 10).is_err(),
            "the trainer cannot run the model without its weights"
        );
        assert_eq!(gone.q8_weight_bytes(), info.matrix_bytes + info.small_bytes);
    }
}

#[test]
fn the_default_decoder_is_unchanged() {
    let _gpu = GPU.lock().unwrap_or_else(|e| e.into_inner());
    if Device::count().unwrap_or(0) == 0 {
        eprintln!("skipped: no CUDA device");
        return;
    }
    // a decoder without enable_q8 is the old one: bitwise the trainer's forward pass
    let mut tr = trainer(forge1_cfg(), 11);
    let mut rng = Lcg(3);
    let toks: Vec<u32> = (0..140).map(|_| rng.next() % 2048).collect();
    let mut d = Decoder::new(&tr, 32).unwrap();
    let mut l = d.feed(&tr, &toks[..100]).unwrap();
    for i in 100..140 {
        l = d.feed(&tr, &toks[i..i + 1]).unwrap();
    }
    let want = tr.logits_at(&toks, 140).unwrap();
    assert_eq!(
        l.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        want.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
    );
}
