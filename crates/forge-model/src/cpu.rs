//! The f32 CPU REFERENCE forward pass (one token at a time with a KV cache). Deliberately simple: it is the oracle the CUDA kernels are debugged against and the fallback when no GPU exists.
//! tools/ref_forward_numpy.py is the executable specification; the block per layer is
//!   h = rms_norm(x) ; q,k,v = Wq h, Wk h, Wv h ; [Qwen3: rms_norm per head on q and k] ; RoPE (rotate half) ; attention (GQA, causal) ; x += Wo o
//!   h2 = rms_norm(x) ; x += Wd (silu(Wg h2) * Wu h2)           and finally logits = (rms_norm(x) * w_out) . embedding^T  (or lm_head)
//! Qwen3.5 adds: gated attention (each head's q projection carries a gate, output * sigmoid(gate)), partial RoPE, and Gated DeltaNet layers in place of attention for three layers out of four
//! (tools/ref_qwen35_numpy.py is the specification; `delta_step` below is the recurrence for one token).

use crate::config::DeltaCfg;
use crate::weights::{DeltaLayer, Layer, Mat, Model};

pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len() / 8 * 8;
    let mut acc = [0f32; 8];
    for (ca, cb) in a[..n].chunks_exact(8).zip(b[..n].chunks_exact(8)) {
        for k in 0..8 {
            acc[k] += ca[k] * cb[k];
        }
    }
    let mut s = ((acc[0] + acc[4]) + (acc[1] + acc[5])) + ((acc[2] + acc[6]) + (acc[3] + acc[7]));
    for k in n..a.len() {
        s += a[k] * b[k];
    }
    s
}

/// out = W x for a row-major [rows][cols] matrix, optionally split over threads.
pub fn matvec(m: &Mat, x: &[f32], out: &mut [f32], threads: usize) {
    debug_assert_eq!(x.len(), m.cols);
    debug_assert_eq!(out.len(), m.rows);
    let cols = m.cols;
    if threads <= 1 || m.rows < 4096 {
        for (r, o) in out.iter_mut().enumerate() {
            *o = dot(&m.w[r * cols..(r + 1) * cols], x);
        }
        return;
    }
    let chunk = m.rows.div_ceil(threads);
    std::thread::scope(|s| {
        for (ci, oc) in out.chunks_mut(chunk).enumerate() {
            let w = &m.w;
            s.spawn(move || {
                for (i, o) in oc.iter_mut().enumerate() {
                    let r = ci * chunk + i;
                    *o = dot(&w[r * cols..(r + 1) * cols], x);
                }
            });
        }
    });
}

pub fn rms_norm(x: &[f32], w: &[f32], eps: f32, out: &mut [f32]) {
    let ss = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let r = 1.0 / (ss + eps).sqrt();
    for i in 0..x.len() {
        out[i] = x[i] * r * w[i];
    }
}

#[inline]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[inline]
fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else {
        (1.0 + x.exp()).ln()
    }
}

/// x / sqrt(sum(x^2) + 1e-6), the epsilon inside the root
fn l2norm(x: &[f32], out: &mut [f32]) {
    let ss: f32 = x.iter().map(|v| v * v).sum();
    let r = 1.0 / (ss + 1e-6).sqrt();
    for i in 0..x.len() {
        out[i] = x[i] * r;
    }
}

/// A generation state: the KV cache and the position.
pub struct Session<'m> {
    m: &'m Model,
    /// per attention layer: [ctx][n_head_kv][head_dim] (empty for Gated DeltaNet layers)
    k: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
    /// per Gated DeltaNet layer: the recurrent state [n_v][head_k][head_v] and the convolution window [conv_dim][conv - 1] (oldest first)
    state: Vec<Vec<f32>>,
    conv_state: Vec<Vec<f32>>,
    pub pos: usize,
    /// the final-normed hidden state of the last step that produced logits (the input of the MTP head)
    pub last_hidden: Vec<f32>,
    max_ctx: usize,
    cos: Vec<f32>,
    sin: Vec<f32>,
    pub threads: usize,
}

impl<'m> Session<'m> {
    pub fn new(m: &'m Model, max_ctx: usize) -> Session<'m> {
        let c = &m.cfg;
        let max_ctx = max_ctx.min(c.max_ctx.max(1));
        let half = c.rotary_dim / 2;
        let mut cos = vec![0f32; max_ctx * half];
        let mut sin = vec![0f32; max_ctx * half];
        for p in 0..max_ctx {
            for i in 0..half {
                // angle = p * theta^(-2i/d), in f64 and rounded once
                let inv = (c.rope_theta as f64).powf(-((2 * i) as f64) / c.rotary_dim as f64);
                let a = p as f64 * inv;
                cos[p * half + i] = a.cos() as f32;
                sin[p * half + i] = a.sin() as f32;
            }
        }
        Session {
            m,
            // one more cache after the layers: the MTP head's attention layer
            k: (0..=c.n_layer)
                .map(|l| {
                    vec![
                        0f32;
                        if l < c.n_layer && c.is_delta(l) || (l == c.n_layer && m.mtp.is_none()) {
                            0
                        } else {
                            max_ctx * c.kv_dim()
                        }
                    ]
                })
                .collect(),
            v: (0..=c.n_layer)
                .map(|l| {
                    vec![
                        0f32;
                        if l < c.n_layer && c.is_delta(l) || (l == c.n_layer && m.mtp.is_none()) {
                            0
                        } else {
                            max_ctx * c.kv_dim()
                        }
                    ]
                })
                .collect(),
            state: (0..c.n_layer)
                .map(|l| {
                    c.delta
                        .as_ref()
                        .filter(|_| c.is_delta(l))
                        .map(|d| vec![0f32; d.n_v * d.head_k * d.head_v])
                        .unwrap_or_default()
                })
                .collect(),
            conv_state: (0..c.n_layer)
                .map(|l| {
                    c.delta
                        .as_ref()
                        .filter(|_| c.is_delta(l))
                        .map(|d| vec![0f32; d.conv_dim() * (d.conv - 1)])
                        .unwrap_or_default()
                })
                .collect(),
            pos: 0,
            last_hidden: Vec::new(),
            max_ctx,
            cos,
            sin,
            threads: std::thread::available_parallelism()
                .map(|n| n.get().min(8))
                .unwrap_or(1),
        }
    }

    pub fn reset(&mut self) {
        self.pos = 0;
        for v in self.state.iter_mut().chain(self.conv_state.iter_mut()) {
            v.iter_mut().for_each(|x| *x = 0.0);
        }
    }

    pub fn max_ctx(&self) -> usize {
        self.max_ctx
    }

    fn rope(&self, x: &mut [f32], heads: usize, pos: usize) {
        let hd = self.m.cfg.head_dim;
        let half = self.m.cfg.rotary_dim / 2;
        let (c, s) = (
            &self.cos[pos * half..(pos + 1) * half],
            &self.sin[pos * half..(pos + 1) * half],
        );
        for h in 0..heads {
            // only the first rotary_dim dimensions rotate (all of them unless the model uses partial RoPE)
            let v = &mut x[h * hd..(h + 1) * hd];
            for i in 0..half {
                let (x1, x2) = (v[i], v[i + half]);
                v[i] = x1 * c[i] - x2 * s[i];
                v[i + half] = x2 * c[i] + x1 * s[i];
            }
        }
    }

    /// One Gated DeltaNet layer for one token: projections, causal convolution on the carried window, gates, the delta-rule update of the recurrent state, the gated RMSNorm and the
    /// output projection. `h` is the normalised input, the result (the layer's contribution to the residual stream) goes to `out`.
    fn delta_step(&mut self, li: usize, d: &DeltaCfg, dl: &DeltaLayer, h: &[f32], out: &mut [f32]) {
        let m = self.m;
        let c = &m.cfg;
        let (nv, dk, dv, kc) = (d.n_v, d.head_k, d.head_v, d.conv);
        let (key_dim, val_dim, cdim) = (d.key_dim(), d.value_dim(), d.conv_dim());
        let mut mixed = vec![0f32; cdim];
        let mut z = vec![0f32; val_dim];
        let mut a = vec![0f32; nv];
        let mut b = vec![0f32; nv];
        matvec(&dl.w_qkv, h, &mut mixed, self.threads);
        matvec(&dl.w_z, h, &mut z, self.threads);
        matvec(&dl.w_a, h, &mut a, 1);
        matvec(&dl.w_b, h, &mut b, 1);
        // causal depthwise convolution over the last `conv` columns (w[conv-1] multiplies the current token), then SiLU
        let mut conv = vec![0f32; cdim];
        {
            let cs = &mut self.conv_state[li];
            for ch in 0..cdim {
                let w = &dl.conv[ch * kc..(ch + 1) * kc];
                let st = &mut cs[ch * (kc - 1)..(ch + 1) * (kc - 1)];
                let mut acc = w[kc - 1] * mixed[ch];
                for j in 0..kc - 1 {
                    acc += w[j] * st[j];
                }
                for j in 0..kc - 2 {
                    st[j] = st[j + 1];
                }
                st[kc - 2] = mixed[ch];
                conv[ch] = silu(acc);
            }
        }
        let (q, rest) = conv.split_at(key_dim);
        let (k, v) = rest.split_at(key_dim);
        let state = &mut self.state[li];
        let mut gated = vec![0f32; val_dim];
        let mut qn = vec![0f32; dk];
        let mut kn = vec![0f32; dk];
        let mut kv = vec![0f32; dv];
        let mut o = vec![0f32; dv];
        for hv in 0..nv {
            let kh = d.key_head(hv);
            l2norm(&q[kh * dk..(kh + 1) * dk], &mut qn);
            l2norm(&k[kh * dk..(kh + 1) * dk], &mut kn);
            let qscale = 1.0 / (dk as f32).sqrt();
            qn.iter_mut().for_each(|x| *x *= qscale);
            let beta = sigmoid(b[hv]);
            let decay = (dl.a_neg[hv] * softplus(a[hv] + dl.dt_bias[hv])).exp();
            let s = &mut state[hv * dk * dv..(hv + 1) * dk * dv];
            s.iter_mut().for_each(|x| *x *= decay);
            kv.iter_mut().for_each(|x| *x = 0.0);
            for i in 0..dk {
                let row = &s[i * dv..(i + 1) * dv];
                for j in 0..dv {
                    kv[j] += row[j] * kn[i];
                }
            }
            let vh = &v[hv * dv..(hv + 1) * dv];
            for i in 0..dk {
                let row = &mut s[i * dv..(i + 1) * dv];
                for j in 0..dv {
                    row[j] += kn[i] * ((vh[j] - kv[j]) * beta);
                }
            }
            o.iter_mut().for_each(|x| *x = 0.0);
            for i in 0..dk {
                let row = &s[i * dv..(i + 1) * dv];
                for j in 0..dv {
                    o[j] += row[j] * qn[i];
                }
            }
            // gated RMSNorm: rms(o) * w * silu(z)
            let ss = o.iter().map(|x| x * x).sum::<f32>() / dv as f32;
            let r = 1.0 / (ss + c.rms_eps).sqrt();
            for j in 0..dv {
                gated[hv * dv + j] = o[j] * r * dl.norm[j] * silu(z[hv * dv + j]);
            }
        }
        matvec(&dl.w_out, &gated, out, self.threads);
    }

    /// The attention part of a layer for one token: projections of the normalised input `h` (the query projection carries a gate per head when the model has one), per-head norms, RoPE, the KV
    /// cache of `slot` (a layer index; the MTP head uses the slot after the last layer), attention over positions 0..=pos, the output gate and the output projection; the result goes to `out`.
    fn attn_layer(&mut self, l: &Layer, slot: usize, h: &[f32], out: &mut [f32], pos: usize) {
        let m = self.m;
        let c = &m.cfg;
        let (hd, nh, nkv) = (c.head_dim, c.n_head, c.n_head_kv);
        let group = nh / nkv;
        let scale = 1.0 / (hd as f32).sqrt();
        let mut qfull = vec![0f32; c.q_rows()];
        let mut q = vec![0f32; c.q_dim()];
        let mut gate = vec![0f32; c.q_dim()];
        let mut kk = vec![0f32; c.kv_dim()];
        let mut vv = vec![0f32; c.kv_dim()];
        let mut o = vec![0f32; c.q_dim()];
        let mut scores = vec![0f32; pos + 1];
        let proj = out;
        {
            matvec(&l.wq, h, &mut qfull, 1);
            if c.attn_gate {
                // per head: hd query values, then hd gate values
                for hh in 0..nh {
                    q[hh * hd..(hh + 1) * hd]
                        .copy_from_slice(&qfull[hh * 2 * hd..hh * 2 * hd + hd]);
                    gate[hh * hd..(hh + 1) * hd]
                        .copy_from_slice(&qfull[hh * 2 * hd + hd..(hh + 1) * 2 * hd]);
                }
            } else {
                q.copy_from_slice(&qfull);
            }
            matvec(&l.wk, h, &mut kk, 1);
            matvec(&l.wv, h, &mut vv, 1);
            if let (Some(qn), Some(kn)) = (&l.q_norm, &l.k_norm) {
                for hh in 0..nh {
                    let seg = q[hh * hd..(hh + 1) * hd].to_vec();
                    rms_norm(&seg, qn, c.rms_eps, &mut q[hh * hd..(hh + 1) * hd]);
                }
                for hh in 0..nkv {
                    let seg = kk[hh * hd..(hh + 1) * hd].to_vec();
                    rms_norm(&seg, kn, c.rms_eps, &mut kk[hh * hd..(hh + 1) * hd]);
                }
            }
            self.rope(&mut q, nh, pos);
            self.rope(&mut kk, nkv, pos);
            let kvd = c.kv_dim();
            self.k[slot][pos * kvd..(pos + 1) * kvd].copy_from_slice(&kk);
            self.v[slot][pos * kvd..(pos + 1) * kvd].copy_from_slice(&vv);
            for hh in 0..nh {
                let kvh = hh / group;
                let qh = &q[hh * hd..(hh + 1) * hd];
                let mut mx = f32::NEG_INFINITY;
                for s in 0..=pos {
                    let kh = &self.k[slot][s * kvd + kvh * hd..s * kvd + (kvh + 1) * hd];
                    let sc = dot(qh, kh) * scale;
                    scores[s] = sc;
                    mx = mx.max(sc);
                }
                let mut sum = 0f32;
                for s in 0..=pos {
                    scores[s] = (scores[s] - mx).exp();
                    sum += scores[s];
                }
                let oh = &mut o[hh * hd..(hh + 1) * hd];
                oh.iter_mut().for_each(|v| *v = 0.0);
                for s in 0..=pos {
                    let p = scores[s] / sum;
                    let vh = &self.v[slot][s * kvd + kvh * hd..s * kvd + (kvh + 1) * hd];
                    for d in 0..hd {
                        oh[d] += p * vh[d];
                    }
                }
            }
            if c.attn_gate {
                for i in 0..c.q_dim() {
                    o[i] *= sigmoid(gate[i]);
                }
            }
            matvec(&l.wo, &o, proj, 1);
        }
    }

    /// Feed one token; returns the final hidden state after the output norm (to be multiplied with the head) when `want_logits`, i.e. the logits.
    pub fn step(&mut self, token: u32, want_logits: bool) -> Option<Vec<f32>> {
        let m = self.m;
        let c = &m.cfg;
        assert!(
            self.pos < self.max_ctx,
            "context of {} tokens exhausted",
            self.max_ctx
        );
        let pos = self.pos;
        let e = c.n_embd;
        let mut x = m.tok_emb.w[token as usize * e..(token as usize + 1) * e].to_vec();
        let mut h = vec![0f32; e];
        let mut proj = vec![0f32; e];
        let mut g = vec![0f32; c.n_ff];
        let mut u = vec![0f32; c.n_ff];
        for (li, l) in m.layers.iter().enumerate() {
            rms_norm(&x, &l.attn_norm, c.rms_eps, &mut h);
            if let (Some(dl), Some(d)) = (&l.delta, &c.delta) {
                self.delta_step(li, d, dl, &h, &mut proj);
                for i in 0..e {
                    x[i] += proj[i];
                }
            } else {
                self.attn_layer(l, li, &h, &mut proj, pos);
                for i in 0..e {
                    x[i] += proj[i];
                }
            }
            rms_norm(&x, &l.ffn_norm, c.rms_eps, &mut h);
            matvec(&l.w_gate, &h, &mut g, 1);
            matvec(&l.w_up, &h, &mut u, 1);
            for i in 0..c.n_ff {
                g[i] = silu(g[i]) * u[i];
            }
            matvec(&l.w_down, &g, &mut proj, 1);
            for i in 0..e {
                x[i] += proj[i];
            }
        }
        self.pos += 1;
        if !want_logits {
            return None;
        }
        let mut hn = vec![0f32; e];
        rms_norm(&x, &m.out_norm, c.rms_eps, &mut hn);
        let head = m.lm_head.as_ref().unwrap_or(&m.tok_emb);
        let mut logits = vec![0f32; c.vocab];
        matvec(head, &hn, &mut logits, self.threads);
        self.last_hidden = hn;
        Some(logits)
    }

    /// The multi-token-prediction head for position `pos`: from the final-normed hidden state of the main model at `pos` (`Session::last_hidden` after the step that produced the logits for
    /// `pos`) and the token at `pos + 1`, the logits for the token at `pos + 2`. The head has its own attention layer and KV cache over the earlier positions it has been called for:
    /// call it for every position in order (an earlier position that was skipped is simply missing from its attention).
    pub fn mtp_step(&mut self, hidden: &[f32], next_token: u32, pos: usize) -> Vec<f32> {
        let m = self.m;
        let w = m.mtp.as_ref().expect("this model has no MTP head");
        let c = &m.cfg;
        let e = c.n_embd;
        let mut cat = vec![0f32; 2 * e];
        {
            let (ce, ch) = cat.split_at_mut(e);
            rms_norm(
                &m.tok_emb.w[next_token as usize * e..(next_token as usize + 1) * e],
                &w.norm_e,
                c.rms_eps,
                ce,
            );
            rms_norm(hidden, &w.norm_h, c.rms_eps, ch);
        }
        let mut x = vec![0f32; e];
        matvec(&w.fc, &cat, &mut x, self.threads);
        let mut h = vec![0f32; e];
        let mut proj = vec![0f32; e];
        rms_norm(&x, &w.layer.attn_norm, c.rms_eps, &mut h);
        let slot = c.n_layer;
        self.attn_layer(&w.layer, slot, &h, &mut proj, pos);
        for i in 0..e {
            x[i] += proj[i];
        }
        rms_norm(&x, &w.layer.ffn_norm, c.rms_eps, &mut h);
        let mut g = vec![0f32; c.n_ff];
        let mut u = vec![0f32; c.n_ff];
        matvec(&w.layer.w_gate, &h, &mut g, 1);
        matvec(&w.layer.w_up, &h, &mut u, 1);
        for i in 0..c.n_ff {
            g[i] = silu(g[i]) * u[i];
        }
        matvec(&w.layer.w_down, &g, &mut proj, 1);
        for i in 0..e {
            x[i] += proj[i];
        }
        rms_norm(&x, &w.out_norm, c.rms_eps, &mut h);
        let head = m.lm_head.as_ref().unwrap_or(&m.tok_emb);
        let mut logits = vec![0f32; c.vocab];
        matvec(head, &h, &mut logits, self.threads);
        logits
    }

    /// Feed a prompt; returns the logits after its last token.
    pub fn prefill(&mut self, tokens: &[u32]) -> Vec<f32> {
        let mut last = None;
        for (i, &t) in tokens.iter().enumerate() {
            last = self.step(t, i + 1 == tokens.len());
        }
        last.expect("empty prompt")
    }
}

pub fn argmax(v: &[f32]) -> u32 {
    let mut best = 0;
    for (i, &x) in v.iter().enumerate() {
        if x > v[best] {
            best = i;
        }
    }
    best as u32
}
