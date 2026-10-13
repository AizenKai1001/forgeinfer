"""Forge-1 trainer in JAX (TPU, GPU or CPU): the model, the optimizers (Muon + AdamW with learning-rate groups), the learning-rate schedule, the data windows and the checkpoint format of the Rust/CUDA trainer
(crates/forge-train), so that a run can start on one and continue on the other. The model is validated against the PyTorch goldens of tools/gen_train_goldens.py (tpu/test_forge_jax.py).

    python forge_jax.py train --data DIR --out DIR --layers 30 --dim 576 --heads 9 --kv 3 --ffn 2304 --seq 2048 --tokens-per-step 65536 --steps 45000 --relu2 --lambdas ... (the flags of `forge train`)

Differences from the CUDA trainer, on purpose: bf16 instead of fp16 operands (no loss scaling), XLA's attention instead of our flash kernels, jax.random instead of our LCG for the initial weights and the data windows.
Not implemented: the hashed n-gram memory (--ngram), which Forge-1 v1 does not use.
"""
import argparse
import json
import math
import os
import sys
import threading
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from queue import Queue


def _pin_wanted(now, pin):
    """now = ['0.7.2', 'tpu', '8'] (version, backend, devices) of the preinstalled JAX or None: pin only on a TPU whose JAX is not the tested version."""
    return bool(now) and now[1] == "tpu" and now[0] != pin


def _pin_jax_for_tpu():
    """Kaggle's TPU image force-installs libtpu 0.0.17 (built 2025-06-12) next to jax 0.7.2, and the Pallas of jax 0.7.2 refuses a libtpu built before 2025-08-01 ("Pallas TPU requires a libtpu version that's
    at most a month old"), so the splash-attention kernel cannot run there and the run falls back to XLA's attention (about 40 % fewer tokens in the same time). The trainer was tested with jax 0.11.2 (CPU
    tests, and a deviceless TPU v5e compile with libtpu 0.0.48). Only on Kaggle and only when its JAX sees a TPU and is not that version: install jax[tpu]==PIN into /tmp with pip --target, check in a
    subprocess that it sees the same TPU devices, and run the whole trainer in a child process with that directory first on PYTHONPATH. Anything that goes wrong leaves the preinstalled JAX in place, and a
    pinned child that fails within 15 minutes (an unusable stack) is followed by a normal start with the preinstalled JAX. Off with --no-pin, FORGE_NO_PIN=1 or JAX_PLATFORMS=cpu."""
    if os.environ.get("FORGE_JAX_PINNED") or os.environ.get("FORGE_NO_PIN") or "--no-pin" in sys.argv or os.environ.get("JAX_PLATFORMS") == "cpu" or not os.path.isdir("/kaggle/working"):
        return
    import subprocess

    pin = os.environ.get("FORGE_JAX_PIN", "0.11.2")
    t0 = time.time()

    def probe(env):
        r = subprocess.run([sys.executable, "-c", "import jax; d = jax.devices(); print('JAXPROBE', jax.__version__, jax.default_backend(), len(d))"], capture_output=True, text=True, timeout=900, env=env)
        marks = [ln.split("JAXPROBE", 1)[1].split() for ln in (r.stdout + r.stderr).splitlines() if "JAXPROBE" in ln]
        return marks[-1] if marks else None

    try:
        now = probe(dict(os.environ))
        print(f"[pin] preinstalled JAX (version, backend, devices): {now}", flush=True)
        if not _pin_wanted(now, pin):
            return
        target = f"/tmp/forge_jax_pin_{pin.replace('.', '_')}"
        r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--target", target, f"jax[tpu]=={pin}", "-f", "https://storage.googleapis.com/jax-releases/libtpu_releases.html"], capture_output=True, text=True, timeout=1500)
        print(f"[pin] pip install jax[tpu]=={pin} into {target}: exit code {r.returncode} after {time.time() - t0:.0f} s {(r.stdout + r.stderr).strip()[-300:]}", flush=True)
        if r.returncode != 0:
            return
        env = dict(os.environ, PYTHONPATH=target + (os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else ""), FORGE_JAX_PINNED="1", PYTHONUNBUFFERED="1")
        env.pop("TPU_LIBRARY_PATH", None)  # a library path set for the image's own libtpu would override the one that comes with the pinned jax
        got = probe(env)
        print(f"[pin] pinned JAX (version, backend, devices): {got}", flush=True)
        if not got or got[0] != pin or got[1] != "tpu" or got[2] != now[2]:
            print("[pin] the pinned JAX does not see the same TPU: the preinstalled JAX is used", flush=True)
            return
    except Exception as e:  # noqa: BLE001 (an optimisation: whatever goes wrong, train with the preinstalled JAX)
        print(f"[pin] failed ({type(e).__name__}: {str(e)[:200]}): the preinstalled JAX is used", flush=True)
        return
    print(f"[pin] running the trainer with jax {pin} ({time.time() - t0:.0f} s of preparation)", flush=True)
    t1 = time.time()
    rc = subprocess.call([sys.executable, "-u"] + sys.argv, env=env)
    if rc == 0 or time.time() - t1 > 900:
        sys.exit(rc)
    print(f"[pin] the pinned trainer failed after {time.time() - t1:.0f} s (exit code {rc}): continuing with the preinstalled JAX", flush=True)
    try:  # a checkpoint that the failed child wrote is not thrown away
        if os.path.exists(os.path.join(sys.argv[sys.argv.index("--out") + 1], "ckpt_latest.bin")) and "--resume" not in sys.argv:
            sys.argv.append("--resume")
    except (ValueError, IndexError):
        pass


if __name__ == "__main__":
    _pin_jax_for_tpu()

import numpy as np

import jax
import jax.numpy as jnp

F32, BF16 = jnp.float32, jnp.bfloat16


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# configuration and parameter layout (the order and the names of crates/forge-train `param_layout`)
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Cfg:
    n_layer: int = 4
    n_embd: int = 256
    n_head: int = 4
    n_kv: int = 4
    hd: int = 64
    n_ff: int = 1024
    vocab: int = 32768
    seq: int = 2048
    qk_norm: bool = True
    theta: float = 10000.0
    eps: float = 1e-5
    untied: bool = False
    relu2: bool = False
    softcap: float = 0.0
    lambdas: bool = False
    skips: bool = False
    ve_layers: tuple = ()

    @property
    def mixing(self):
        return self.lambdas or self.skips

    @property
    def q_dim(self):
        return self.n_head * self.hd

    @property
    def kv_dim(self):
        return self.n_kv * self.hd

    @property
    def n_enc(self):
        return self.n_layer // 2


def layout(c: Cfg):
    """[(name, shape)] in the order of the flat parameter array of the Rust trainer."""
    e, qd, kvd, ff, v = c.n_embd, c.q_dim, c.kv_dim, c.n_ff, c.vocab
    out = [("emb", (v, e))]
    for l in range(c.n_layer):
        out += [(f"l{l}.wq", (qd, e)), (f"l{l}.wk", (kvd, e)), (f"l{l}.wv", (kvd, e)), (f"l{l}.wo", (e, qd))]
        if not c.relu2:
            out.append((f"l{l}.w_gate", (ff, e)))
        out += [(f"l{l}.w_up", (ff, e)), (f"l{l}.w_down", (e, ff))]
    for l in range(c.n_layer):
        out += [(f"l{l}.attn_norm", (e,)), (f"l{l}.ffn_norm", (e,))]
        if c.qk_norm:
            out += [(f"l{l}.q_norm", (c.hd,)), (f"l{l}.k_norm", (c.hd,))]
    out.append(("out_norm", (e,)))
    if c.untied:
        out.append(("head", (v, e)))
    for l in c.ve_layers:
        out.append((f"ve{l}", (v, kvd)))
    if c.mixing:
        out.append(("lam", (c.n_layer, 3)))
    if c.ve_layers:
        out.append(("velam", (len(c.ve_layers), 2)))
    return out


def is_muon_matrix(name, shape):
    return len(shape) == 2 and name != "emb" and name != "head" and not name.startswith("ve") and not name.startswith("ng") and name not in ("lam", "velam")


def group_of(name):
    """AdamW learning-rate group of a parameter that Muon does not own: emb, norm, head, table, scalar."""
    if name == "emb":
        return "emb"
    if name == "head":
        return "head"
    if name.startswith("ve"):
        return "table" if name != "velam" else "scalar"
    if name == "lam":
        return "scalar"
    return "norm"


def init_params(c: Cfg, seed: int):
    key = jax.random.PRNGKey(seed)
    std = 0.02
    resid = std / math.sqrt(2 * c.n_layer)
    params = {}
    for i, (name, shape) in enumerate(layout(c)):
        k = jax.random.fold_in(key, i)
        if name.endswith("_norm"):
            params[name] = jnp.ones(shape, F32)
        elif name == "head":
            params[name] = jnp.zeros(shape, F32)
        elif name == "lam":
            lam = np.zeros(shape, np.float32)
            lam[:, 0] = 1.0
            params[name] = jnp.asarray(lam)
        elif name == "velam":
            params[name] = jnp.ones(shape, F32)
        else:
            s = resid if (name.endswith(".wo") or name.endswith(".w_down")) else std
            params[name] = jax.random.normal(k, shape, F32) * s
    return params


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# the model
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


def rms(x, w, eps):
    x32 = x.astype(F32)
    return x32 * jax.lax.rsqrt(jnp.mean(x32 * x32, axis=-1, keepdims=True) + eps) * w


def rope_tables(c: Cfg, t):
    pos = np.arange(t, dtype=np.float64)
    inv = c.theta ** (-np.arange(0, c.hd, 2, dtype=np.float64) / c.hd)
    ang = pos[:, None] * inv[None, :]
    return jnp.asarray(np.cos(ang), F32), jnp.asarray(np.sin(ang), F32)


def rope(x, cos, sin):  # x [B, T, H, hd]
    h = x.shape[-1] // 2
    x1, x2 = x[..., :h], x[..., h:]
    cc, ss = cos[None, :, None, :], sin[None, :, None, :]
    return jnp.concatenate([x1 * cc - x2 * ss, x2 * cc + x1 * ss], axis=-1)


def mm(a, w, cdt):
    """a [..., in] times w [out, in]^T with `cdt` operands and an fp32 result."""
    return jnp.einsum("...i,oi->...o", a.astype(cdt), w.astype(cdt), preferred_element_type=F32)


ATTN = {"impl": None}  # None: XLA's attention (attention_xla); on a TPU main() may put a faster kernel here (select_attention)


def attention_xla(c: Cfg, q, k, v, cdt):
    """Causal attention with grouped keys/values in plain XLA ops: q [B, T, H, hd], k and v [B, T, KV, hd] (fp32) -> [B, T, H * hd] (fp32). The [T, T] scores of every head are written to memory."""
    b, t = q.shape[:2]
    rep = c.n_head // c.n_kv
    q = q.reshape(b, t, c.n_kv, rep, c.hd).astype(cdt)
    att = jnp.einsum("btgrd,bsgd->bgrts", q, k.astype(cdt), preferred_element_type=F32) / math.sqrt(c.hd)
    mask = jnp.tril(jnp.ones((t, t), bool))
    att = jnp.where(mask[None, None, None], att, -jnp.inf)
    att = jax.nn.softmax(att, axis=-1)
    return jnp.einsum("bgrts,bsgd->btgrd", att.astype(cdt), v.astype(cdt), preferred_element_type=F32).reshape(b, t, c.q_dim)


def _shard_map(f, mesh, in_specs, out_specs):
    """shard_map across JAX versions (jax.shard_map with check_vma, or the experimental one with check_rep)."""
    sm = getattr(jax, "shard_map", None)
    if sm is not None:
        try:
            return sm(f, mesh=mesh, in_specs=in_specs, out_specs=out_specs, check_vma=False)
        except TypeError:
            return sm(f, mesh=mesh, in_specs=in_specs, out_specs=out_specs, check_rep=False)
    from jax.experimental.shard_map import shard_map

    return shard_map(f, mesh=mesh, in_specs=in_specs, out_specs=out_specs, check_rep=False)


def make_splash_attention(c: Cfg, mesh, block=512, fused_bwd=False, interpret=False):
    """The same attention as a Pallas splash-attention kernel (TPU): the scores stay in the chip's fast memory, blocks above the diagonal are skipped. The batch is split over the mesh axis 'data'
    (a Pallas kernel cannot be partitioned automatically). Returns attn(c, q, k, v, cdt) with the signature of attention_xla."""
    from jax.sharding import PartitionSpec as P
    from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel as sk
    from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask as sm

    t = c.seq
    blk = min(block, t)
    assert t % blk == 0 and blk % 128 == 0, (t, blk)
    mask = sm.MultiHeadMask([sm.CausalMask((t, t)) for _ in range(c.n_head)])
    fwd = dict(block_q=blk, block_kv=blk, block_kv_compute=blk, block_q_dkv=blk, block_kv_dkv=blk, block_kv_dkv_compute=blk)
    sizes = sk.BlockSizes(**fwd, use_fused_bwd_kernel=True) if fused_bwd else sk.BlockSizes(**fwd, block_q_dq=blk, block_kv_dq=blk)
    kernel = sk.make_splash_mha_single_device(mask, block_sizes=sizes, **({"interpret": True} if interpret else {}))
    spec = P("data")
    mapped = _shard_map(lambda q, k, v: jax.vmap(kernel)(q, k, v), mesh, (spec, spec, spec), spec)

    def attn(c_, q, k, v, cdt):
        b, t_ = q.shape[:2]
        qh = (q * (1.0 / math.sqrt(c_.hd))).astype(cdt).transpose(0, 2, 1, 3)  # [B, H, T, hd]; the kernel does not scale the scores
        o = mapped(qh, k.astype(cdt).transpose(0, 2, 1, 3), v.astype(cdt).transpose(0, 2, 1, 3))
        return o.transpose(0, 2, 1, 3).reshape(b, t_, c_.q_dim).astype(F32)

    return attn


def _rel_errors(ref, out):
    """Relative differences of two (value, (dq, dk, dv)) results of value_and_grad."""
    (rv, rg), (ov, og) = ref, out
    errs = [abs(float(ov) - float(rv)) / (abs(float(rv)) + 1e-6)]
    for a_, b_ in zip(og, rg):
        errs.append(float(jnp.linalg.norm((a_ - b_).astype(F32)) / (jnp.linalg.norm(b_.astype(F32)) + 1e-9)))
    return [round(e, 4) for e in errs]


def select_attention(c: Cfg, mesh, cdt, per_micro, force=False, candidates=((512, False), (512, True), (256, False), (1024, False)), tol=0.05, log=print):
    """On a TPU: build the splash kernel with a few block sizes, compare each with the XLA attention on random data of the real shape (value and gradients), time forward + backward of all of them and
    return the fastest valid one (None = keep XLA's; the splash kernel must be at least 10 % faster unless `force`). Any failure of a candidate only removes that candidate."""
    from jax.sharding import NamedSharding, PartitionSpec as P

    sh = NamedSharding(mesh, P("data"))
    b, t = per_micro, c.seq
    keys = jax.random.split(jax.random.PRNGKey(7), 4)
    q = jax.device_put(jax.random.normal(keys[0], (b, t, c.n_head, c.hd), F32), sh)
    k = jax.device_put(jax.random.normal(keys[1], (b, t, c.n_kv, c.hd), F32), sh)
    v = jax.device_put(jax.random.normal(keys[2], (b, t, c.n_kv, c.hd), F32), sh)
    w = jax.device_put(jax.random.normal(keys[3], (b, t, c.q_dim), F32), sh)

    def make(impl):
        return jax.jit(jax.value_and_grad(lambda q_, k_, v_: jnp.sum(impl(c, q_, k_, v_, cdt) * w), argnums=(0, 1, 2)))

    def run(fn, reps=3):
        jax.block_until_ready(fn(q, k, v))
        t0 = time.time()
        for _ in range(reps):
            out = fn(q, k, v)
        jax.block_until_ready(out)
        return out, (time.time() - t0) / reps

    ref, t_ref = run(make(attention_xla))
    log(f"[attn] XLA attention, forward + backward of {b} sequences per device: {t_ref * 1e3:.1f} ms (x{c.n_layer} layers = {t_ref * c.n_layer:.2f} s per micro-step)")
    best_t, best_impl, best_name = float("inf"), None, ""
    for blk, fused in candidates:
        name = f"splash block {blk}" + (" fused backward" if fused else "")
        try:
            impl = make_splash_attention(c, mesh, blk, fused)
            out, dt = run(make(impl))
            errs = _rel_errors(ref, out)
        except Exception as e:  # noqa: BLE001 (a kernel that does not compile or run on this JAX / TPU only drops out)
            text = str(e).strip().splitlines()
            log(f"[attn] {name}: failed ({type(e).__name__}: {text[0][:200] if text else ''})")
            continue
        ok = max(errs) < tol
        log(f"[attn] {name}: {dt * 1e3:.1f} ms (x{c.n_layer} = {dt * c.n_layer:.2f} s), relative errors value/dq/dk/dv {errs}: {'ok' if ok else 'MISMATCH'}")
        if ok and dt < best_t:
            best_t, best_impl, best_name = dt, impl, name
    if best_impl is None:
        log("[attn] no splash candidate usable: XLA attention")
        return None
    if force or best_t < 0.9 * t_ref:
        log(f"[attn] using {best_name} ({t_ref / best_t:.2f}x faster than XLA's)")
        return best_impl
    log(f"[attn] best splash candidate ({best_name}, {best_t * 1e3:.1f} ms) is not clearly faster than XLA's ({t_ref * 1e3:.1f} ms): XLA attention")
    return None


def layer_fn(c: Cfg, cdt, x, x0, skip, lam_l, velam_l, tokens_ve, lw, cos, sin):
    """One transformer layer. x: the input stream (fp32), x0: the embedding output, skip: the output of the mirrored layer or None, lam_l: (a, b, c), lw: the layer's weights."""
    b, t, e = x.shape
    if c.mixing:
        xin = (lam_l[0] if c.lambdas else 1.0) * x
        if c.lambdas:
            xin = xin + lam_l[1] * x0
        if c.skips and skip is not None:
            xin = xin + lam_l[2] * skip
        x = xin
    h = rms(x, lw["attn_norm"], c.eps)
    q = mm(h, lw["wq"], cdt).reshape(b, t, c.n_head, c.hd)
    k = mm(h, lw["wk"], cdt).reshape(b, t, c.n_kv, c.hd)
    v = mm(h, lw["wv"], cdt).reshape(b, t, c.n_kv, c.hd)
    if c.qk_norm:
        q = rms(q, lw["q_norm"], c.eps)
        k = rms(k, lw["k_norm"], c.eps)
    q, k = rope(q, cos, sin), rope(k, cos, sin)
    if tokens_ve is not None:
        v = velam_l[0] * v + velam_l[1] * tokens_ve.reshape(b, t, c.n_kv, c.hd).astype(F32)
    o = (ATTN["impl"] or attention_xla)(c, q, k, v, cdt)
    x = x + mm(o, lw["wo"], cdt)
    h2 = rms(x, lw["ffn_norm"], c.eps)
    if c.relu2:
        act = jnp.square(jax.nn.relu(mm(h2, lw["w_up"], cdt)))
    else:
        act = jax.nn.silu(mm(h2, lw["w_gate"], cdt)) * mm(h2, lw["w_up"], cdt)
    return x + mm(act, lw["w_down"], cdt)


def hidden(c: Cfg, params, tokens, cdt=BF16, remat=True):
    """The final hidden states [B, T, E] (after the output norm) for tokens [B, T]."""
    b, t = tokens.shape
    cos, sin = rope_tables(c, t)
    x0 = params["emb"][tokens]
    outs = []
    ve_pos = {l: i for i, l in enumerate(c.ve_layers)}
    layer = jax.checkpoint(partial(layer_fn, c, cdt), policy=jax.checkpoint_policies.dots_with_no_batch_dims_saveable) if remat else partial(layer_fn, c, cdt)
    for l in range(c.n_layer):
        xprev = x0 if l == 0 else outs[l - 1]
        lw = {n: params[f"l{l}.{n}"] for n in ("wq", "wk", "wv", "wo", "w_up", "w_down", "attn_norm", "ffn_norm") + (("w_gate",) if not c.relu2 else ()) + (("q_norm", "k_norm") if c.qk_norm else ())}
        lam_l = params["lam"][l] if c.mixing else None
        skip = outs[c.n_layer - 1 - l] if (c.skips and l >= c.n_layer - c.n_enc) else None
        if l in ve_pos:
            tok_ve = params[f"ve{l}"][tokens].astype(cdt)  # gather the fp32 rows, then cast: the backward scatter-add then sums repeated tokens in fp32 (a bf16 table summed them in bf16: up to 3 % error on frequent tokens)
            velam_l = params["velam"][ve_pos[l]]
        else:
            tok_ve, velam_l = None, None
        outs.append(layer(xprev, x0, skip, lam_l, velam_l, tok_ve, lw, cos, sin))
    return rms(outs[-1], params["out_norm"], c.eps)


def loss_fn(c: Cfg, params, tokens, targets, ignore, cdt=BF16, ce_chunk=256, remat=True):
    """Mean cross entropy over the positions whose target is not ignored. tokens, targets: [B, T] int32; ignore: [B, T] bool. The logits are made `ce_chunk` positions at a time (and not kept for the backward pass)."""
    hf = hidden(c, params, tokens, cdt, remat)
    b, t, e = hf.shape
    ch = min(ce_chunk, t)
    assert t % ch == 0, (t, ch)
    nc = t // ch
    head = (params["head"] if c.untied else params["emb"]).astype(cdt)

    def chunk(args):
        h_c, t_c, i_c = args  # [B, ch, E], [B, ch], [B, ch]
        z = jnp.einsum("bne,ve->bnv", h_c.astype(cdt), head, preferred_element_type=F32)
        if c.softcap > 0:
            z = c.softcap * jnp.tanh(z / c.softcap)
        lse = jax.scipy.special.logsumexp(z, axis=-1)
        zt = jnp.take_along_axis(z, t_c[..., None], axis=-1)[..., 0]
        return jnp.sum(jnp.where(i_c, 0.0, lse - zt))

    f = jax.checkpoint(chunk) if remat else chunk
    parts = (hf.reshape(b, nc, ch, e).swapaxes(0, 1), targets.reshape(b, nc, ch).swapaxes(0, 1), ignore.reshape(b, nc, ch).swapaxes(0, 1))
    total = jnp.sum(jax.lax.map(f, parts))
    valid = jnp.maximum(jnp.sum(~ignore), 1)
    return total / valid


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# optimizers
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

NS_COEF = (3.4445, -4.7750, 2.0315)


def newton_schulz(x, iters=5, dtype=BF16):
    """Five iterations of the quintic Newton-Schulz orthogonalisation on the (Frobenius-normalised) matrix x; a tall matrix is iterated through its transpose (the same as the Rust trainer)."""
    a_, b_, c_ = NS_COEF
    tall = x.shape[0] > x.shape[1]
    if tall:
        x = x.T
    x = x.astype(dtype)
    for _ in range(iters):
        a = jnp.einsum("ik,jk->ij", x, x, preferred_element_type=F32).astype(dtype)
        bb = (b_ * a.astype(F32) + c_ * jnp.einsum("ik,kj->ij", a, a, preferred_element_type=F32)).astype(dtype)
        x = (a_ * x.astype(F32) + jnp.einsum("ik,kj->ij", bb, x, preferred_element_type=F32)).astype(dtype)
    x = x.astype(F32)
    return x.T if tall else x


def muon_update(p, g, mom, lr, wd, mu, ns_dtype=BF16):
    mom = mu * mom + g
    u = g + mu * mom
    x = u / (jnp.sqrt(jnp.sum(u * u)) + 1e-7)
    d = newton_schulz(x, 5, ns_dtype)
    scale = 0.2 * math.sqrt(max(p.shape))
    return p * (1.0 - lr * wd) - lr * scale * d, mom


def adamw_update(p, g, m, v, lr, b1, b2, eps, wd, bc1, bc2):
    m = b1 * m + (1.0 - b1) * g
    v = b2 * v + (1.0 - b2) * g * g
    p = p * (1.0 - lr * wd)
    p = p - lr * (m / bc1) / (jnp.sqrt(v / bc2) + eps)
    return p, m, v


@dataclass(frozen=True)
class Hyper:
    peak_lr: float = 1.5e-3
    muon_lr: float = 1.5e-3
    warmup: int = 100
    min_frac: float = 0.05
    decay_frac: float = 0.1
    wd: float = 0.1
    muon_wd: float = 0.05
    mu: float = 0.95
    beta2: float = 0.95
    clip: float = 1.0
    mult_emb: float = 1.0
    mult_head: float = 1.0
    mult_table: float = 1.0
    mult_scalar: float = 1.0
    use_muon: bool = True


def lr_at(step, steps, peak, warmup, min_frac, decay_frac):
    if step < warmup:
        return peak * (step + 1) / warmup
    decay_start = int((1.0 - decay_frac) * steps)
    if step < decay_start:
        return peak
    t = (step - decay_start) / max(steps - decay_start, 1)
    return peak * (1.0 - (1.0 - min_frac) * min(t, 1.0))


def plan_steps(steps_max, done, elapsed, step_secs, overhead_per_step, time_limit, reserve=90.0):
    """The total number of steps (at most steps_max) that the time limit allows: the steps done so far plus those that fit into the remaining time at the measured speed (`reserve` seconds are kept for the last evaluation and checkpoint)."""
    n_more = int(max(time_limit - elapsed - reserve, 0.0) / max(step_secs + overhead_per_step, 1e-9))
    return min(steps_max, done + n_more)


def apply_update(c: Cfg, h: Hyper, params, grads, mstate, vstate, step_count, lr, lr_muon):
    """One optimizer step on the gradients (clipped to a global norm); step_count is the 1-based step for the bias correction. Returns new params, m, v and the gradient norm.
    The Muon matrices of one shape (all the wq of the layers, all the w_up ...) are stacked and updated together: one batched Newton-Schulz instead of one per matrix (a much smaller program for the compiler)."""
    gnorm = jnp.sqrt(sum(jnp.sum(g * g) for g in grads.values()))
    clip = jnp.where((h.clip > 0) & (gnorm > h.clip), h.clip / gnorm, 1.0)
    t = step_count.astype(F32)
    bc1 = 1.0 - 0.9**t
    bc2 = 1.0 - h.beta2**t
    mults = {"emb": h.mult_emb, "head": h.mult_head, "table": h.mult_table, "scalar": h.mult_scalar, "norm": 1.0}
    newp, newm, newv = {}, {}, {}
    muon_groups = {}
    for name, shape in layout(c):
        if h.use_muon and is_muon_matrix(name, shape):
            muon_groups.setdefault(shape, []).append(name)
            continue
        g = grads[name] * clip
        decay = h.wd if (name == "emb" or (is_muon_matrix(name, shape) and not h.use_muon)) else 0.0
        mult = mults[group_of(name)] if not is_muon_matrix(name, shape) else 1.0
        newp[name], newm[name], newv[name] = adamw_update(params[name], g, mstate[name], vstate[name], lr * mult, 0.9, h.beta2, 1e-8, decay, bc1, bc2)
    for shape, names in muon_groups.items():
        stack = lambda tree: jnp.stack([tree[n] for n in names])
        p_new, m_new = jax.vmap(lambda p, g, m: muon_update(p, g * clip, m, lr_muon, h.muon_wd, h.mu))(stack(params), stack(grads), stack(mstate))
        for i, n in enumerate(names):
            newp[n], newm[n], newv[n] = p_new[i], m_new[i], vstate[n]
    return newp, newm, newv, gnorm


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# checkpoints in the format of the Rust trainer: one JSON line, then the parameters, the first moments (Muon: the momentum) and the second moments as raw little-endian f32 in layout order
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


def header_of(c: Cfg, n_params, step, loss_scale=32768.0, plan_steps=None):  # the JAX trainer needs no loss scaling; the Rust trainer (fp16) that resumes from this file starts its dynamic scaler at the value written here
    cfg = {
        "n_layer": c.n_layer, "n_embd": c.n_embd, "n_head": c.n_head, "n_kv": c.n_kv, "head_dim": c.hd, "n_ff": c.n_ff, "vocab": c.vocab, "seq_len": c.seq, "qk_norm": c.qk_norm,
        "rope_theta": c.theta, "eps": c.eps, "untied": c.untied, "relu2": c.relu2, "softcap": c.softcap, "lambdas": c.lambdas, "skips": c.skips, "ve_layers": list(c.ve_layers), "ngram": [],
        "ngram_bits": 0, "ngram_dim": 0,
    }
    head = {"magic": "forgeckpt1", "n_params": n_params, "step": step, "loss_scale": loss_scale, "cfg": cfg}
    if plan_steps:
        head["plan_steps"] = int(plan_steps)  # the planned total number of steps: a resumed JAX run keeps the learning-rate schedule (the Rust loader reads the header by key and ignores it)
    return json.dumps(head, separators=(",", ":"))


def flat(c: Cfg, tree):
    return np.concatenate([np.asarray(tree[n], np.float32).reshape(-1) for n, _ in layout(c)])


def unflat(c: Cfg, arr):
    out, off = {}, 0
    for n, s in layout(c):
        k = int(np.prod(s))
        out[n] = jnp.asarray(arr[off : off + k].reshape(s))
        off += k
    assert off == len(arr), (off, len(arr))
    return out


def save_checkpoint(path, c: Cfg, params, m, v, step, plan_steps=None):
    n = sum(int(np.prod(s)) for _, s in layout(c))
    tmp = str(path) + ".tmp"
    with open(tmp, "wb") as f:
        f.write((header_of(c, n, step, plan_steps=plan_steps) + "\n").encode())
        for tree in (params, m, v):
            f.write(flat(c, jax.device_get(tree)).astype("<f4").tobytes())
    os.replace(tmp, path)


def load_checkpoint(path, c: Cfg, with_state=True):
    with open(path, "rb") as f:
        head = json.loads(f.readline())
        n = head["n_params"]
        assert n == sum(int(np.prod(s)) for _, s in layout(c)), "checkpoint does not match this model"
        params = unflat(c, np.frombuffer(f.read(4 * n), dtype="<f4"))
        if not with_state:
            return params, None, None, head["step"]
        m = unflat(c, np.frombuffer(f.read(4 * n), dtype="<f4"))
        v = unflat(c, np.frombuffer(f.read(4 * n), dtype="<f4"))
    return params, m, v, head["step"]


def checkpoint_plan(path):
    """The planned total number of steps stored in a checkpoint header (None for files of the Rust trainer or older JAX files)."""
    with open(path, "rb") as f:
        return json.loads(f.readline()).get("plan_steps")


def cfg_from_checkpoint(path):
    with open(path, "rb") as f:
        h = json.loads(f.readline())["cfg"]
    return Cfg(n_layer=h["n_layer"], n_embd=h["n_embd"], n_head=h["n_head"], n_kv=h["n_kv"], hd=h["head_dim"], n_ff=h["n_ff"], vocab=h["vocab"], seq=h["seq_len"], qk_norm=h["qk_norm"], theta=h["rope_theta"],
               eps=h["eps"], untied=h["untied"], relu2=h["relu2"], softcap=h["softcap"], lambdas=h["lambdas"], skips=h["skips"], ve_layers=tuple(h["ve_layers"]))


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# data: uint16 token shards (bit 15 = "not a target" in chat data), random windows, a validation tail in the last shard
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


class Data:
    def __init__(self, directory, val_tokens=20_000_000, flags=True):
        files = sorted(Path(directory).glob("*.u16"))
        assert files, f"no .u16 shards in {directory}"
        self.files = files
        self.maps = [np.memmap(f, dtype="<u2", mode="r") for f in files]
        sizes = [len(m) for m in self.maps]
        v = min(val_tokens, sizes[-1] // 4)
        self.train_sizes = sizes[:-1] + [sizes[-1] - v]
        self.val_start = self.train_sizes[-1]
        self.val_tokens = v
        self.total_train = sum(self.train_sizes)
        self.flags = flags
        self.cum = np.cumsum([0] + self.train_sizes)

    def _raw(self, h, length):
        """Window of `length` tokens at a position derived from h (a position that would straddle the end of a shard is taken from the start of that shard, as in the Rust loader)."""
        p = int(h % (self.total_train - length))
        s = int(np.searchsorted(self.cum, p, side="right") - 1)
        off = p - int(self.cum[s])
        if off + length > self.train_sizes[s]:
            off = 0
        return np.asarray(self.maps[s][off : off + length])

    def train_batch(self, rng, batch, seq):
        hs = rng.integers(0, 1 << 62, size=batch)
        return self._split(np.stack([self._raw(h, seq + 1) for h in hs]))

    def val_batch(self, k, batch, seq):
        """The validation windows of the Rust trainer: window k * batch + s starts at val_start + ((k*batch+s) * 7919 * 4093) mod span."""
        span = max(self.val_tokens - (seq + 1), 1)
        rows = []
        for s in range(batch):
            idx = k * batch + s
            p = self.val_start + (idx * 7919 * 4093) % span
            rows.append(np.asarray(self.maps[-1][p : p + seq + 1]))
        return self._split(np.stack(rows))

    def _split(self, win):
        win = win.astype(np.int32)
        ignore = (win & 0x8000) != 0 if self.flags else np.zeros_like(win, bool)
        ids = win & 0x7FFF if self.flags else win
        return ids[:, :-1], ids[:, 1:], ignore[:, 1:]


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# training
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


def warm_files(files, chunk=64 << 20):
    """Read the shards once, front to back (a background thread): the random windows of the training then come from the operating system's page cache instead of a cold, possibly remote, disk. Errors are ignored."""
    t0, n = time.time(), 0
    try:
        for f in files:
            with open(f, "rb", buffering=0) as fh:
                while True:
                    blk = fh.read(chunk)
                    if not blk:
                        break
                    n += len(blk)
    except Exception:  # noqa: BLE001
        pass
    print(f"[train] data warmed: {n / 1e9:.1f} GB read in {time.time() - t0:.0f} s", flush=True)


def make_mesh():
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    devs = np.array(jax.devices())
    mesh = Mesh(devs, ("data",))
    # batches have the shape [micro-steps, sequences, tokens]: the sequences are split over the devices; the evaluation batches [sequences, tokens] likewise
    return mesh, NamedSharding(mesh, P(None, "data")), NamedSharding(mesh, P("data")), NamedSharding(mesh, P())


def build_train_step(c: Cfg, h: Hyper, cdt, ce_chunk, remat):
    """One optimizer step over `tokens` [accum, B, T]: the gradients of the micro-batches are averaged (a scan, one micro-batch at a time in memory), then the update."""

    def step_fn(params, m, v, step_count, tokens, targets, ignore, lr, lr_muon):
        accum = tokens.shape[0]

        def micro(carry, xs):
            gsum, lsum = carry
            t_, y_, i_ = xs
            loss, g = jax.value_and_grad(lambda p: loss_fn(c, p, t_, y_, i_, cdt, ce_chunk, remat))(params)
            return (jax.tree_util.tree_map(jnp.add, gsum, g), lsum + loss), None

        zeros = jax.tree_util.tree_map(jnp.zeros_like, params)
        (grads, lsum), _ = jax.lax.scan(micro, (zeros, jnp.zeros((), F32)), (tokens, targets, ignore))
        grads = jax.tree_util.tree_map(lambda g: g / accum, grads)
        loss = lsum / accum
        step_count = step_count + 1
        params, m, v, gnorm = apply_update(c, h, params, grads, m, v, step_count, lr, lr_muon)
        return params, m, v, step_count, loss, gnorm

    return jax.jit(step_fn, donate_argnums=(0, 1, 2, 3))


def build_eval(c: Cfg, cdt, ce_chunk):
    return jax.jit(lambda params, tokens, targets, ignore: loss_fn(c, params, tokens, targets, ignore, cdt, ce_chunk, False))


def parse_args(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train"])
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--kv", type=int, default=0)
    ap.add_argument("--head-dim", type=int, default=64)
    ap.add_argument("--ffn", type=int, default=1024)
    ap.add_argument("--vocab", type=int, default=32768)
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--tokens-per-step", type=int, default=131072)
    ap.add_argument("--micro", type=int, default=4, help="sequences per device per micro-step; the micro-steps of an optimizer step run one after the other and their gradients are averaged")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--decay-frac", type=float, default=0.2)
    ap.add_argument("--min-lr-frac", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--muon-lr", type=float, default=None)
    ap.add_argument("--muon-wd", type=float, default=0.05)
    ap.add_argument("--muon-mu", type=float, default=0.95)
    ap.add_argument("--optim", default="adamw")
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--beta2", type=float, default=0.95)
    ap.add_argument("--qk-norm", action="store_true")
    ap.add_argument("--rope-theta", type=float, default=10000.0)
    ap.add_argument("--untied", action="store_true")
    ap.add_argument("--relu2", action="store_true")
    ap.add_argument("--softcap", type=float, default=0.0)
    ap.add_argument("--lambdas", action="store_true")
    ap.add_argument("--skips", action="store_true")
    ap.add_argument("--value-emb", default="")
    ap.add_argument("--lr-emb-mult", type=float, default=1.0)
    ap.add_argument("--lr-head-mult", type=float, default=1.0)
    ap.add_argument("--lr-table-mult", type=float, default=1.0)
    ap.add_argument("--lr-scalar-mult", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--eval-batches", type=int, default=8)
    ap.add_argument("--val-tokens", type=int, default=20_000_000)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--init-from", default="")
    ap.add_argument("--time-limit-s", type=float, default=0.0, help="stop after this many seconds (a checkpoint is written); unless --no-auto-steps the number of steps is then planned so that the learning-rate decay ends inside the limit")
    ap.add_argument("--no-auto-steps", action="store_true", help="keep --steps even when the time limit would stop the run before the end of the schedule")
    ap.add_argument("--attn-test-fail", default="", choices=["", "step", "eval"], help=argparse.SUPPRESS)  # tests: an attention that fails in the training step / only in the evaluation
    ap.add_argument("--no-warm-data", action="store_true", help="do not read the shards once in the background at the start (page-cache warm-up)")
    ap.add_argument("--print-every", type=int, default=10, help="print every n-th step line (log.jsonl keeps every step; evaluations, the first 20 steps and the last step are always printed)")
    ap.add_argument("--watchdog-s", type=float, default=2400.0, help="exit with code 97 when no step, evaluation or checkpoint completes for this many seconds (a hung device would otherwise hold a Kaggle session until it is killed); 0 = off")
    ap.add_argument("--no-pin", action="store_true", help="on Kaggle TPU VMs do not install the tested jax version (see _pin_jax_for_tpu)")
    ap.add_argument("--plan-reserve-s", type=float, default=90.0, help="seconds of the time limit kept for the last evaluation and checkpoint when the steps are planned")
    ap.add_argument("--attn", choices=["auto", "xla", "splash"], default="auto", help="attention implementation: auto = on a TPU the splash kernel when it is checked and faster, otherwise XLA's")
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--ce-chunk", type=int, default=256, help="positions per chunk of the cross entropy")
    ap.add_argument("--no-remat", action="store_true")
    return ap.parse_args(argv)


def main(argv=None):
    a = parse_args(argv if argv is not None else sys.argv[1:])
    steps_max = a.steps  # the command line's number of steps is the upper bound of the plan
    auto_steps = a.time_limit_s > 0 and not a.no_auto_steps
    c = Cfg(n_layer=a.layers, n_embd=a.dim, n_head=a.heads, n_kv=a.kv or a.heads, hd=a.head_dim, n_ff=a.ffn, vocab=a.vocab, seq=a.seq, qk_norm=a.qk_norm, theta=a.rope_theta, untied=a.untied, relu2=a.relu2,
            softcap=a.softcap, lambdas=a.lambdas, skips=a.skips, ve_layers=tuple(int(x) for x in a.value_emb.split(",") if x))
    muon_lr = a.muon_lr if a.muon_lr is not None else a.lr
    h = Hyper(peak_lr=a.lr, muon_lr=muon_lr, warmup=a.warmup, min_frac=a.min_lr_frac, decay_frac=a.decay_frac, wd=a.wd, muon_wd=a.muon_wd, mu=a.muon_mu, beta2=a.beta2, clip=a.clip, mult_emb=a.lr_emb_mult,
              mult_head=a.lr_head_mult, mult_table=a.lr_table_mult, mult_scalar=a.lr_scalar_mult, use_muon=(a.optim == "muon"))
    cdt = BF16 if a.dtype == "bf16" else F32
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    mesh, batch_sh, eval_sh, repl_sh = make_mesh()
    ndev = len(jax.devices())
    progress = {"t": time.time(), "what": "start-up"}

    def watchdog():
        while True:
            time.sleep(30)
            idle = time.time() - progress["t"]
            if idle > a.watchdog_s:
                print(f"[train] watchdog: nothing completed for {idle:.0f} s (last: {progress['what']}): exiting with code 97 so that a hung device or reader does not hold the session", flush=True)
                os._exit(97)

    if a.watchdog_s > 0:
        threading.Thread(target=watchdog, daemon=True).start()
    data = Data(a.data, a.val_tokens, flags=a.vocab <= 32768)
    if not a.no_warm_data:
        threading.Thread(target=warm_files, args=(data.files,), daemon=True).start()
    seq = a.seq
    # the global batch of one optimizer step: tokens_per_step / seq sequences, a multiple of devices x micro; the micro-steps accumulate gradients
    micro = max(1, a.micro)
    per_micro = ndev * micro
    seqs_per_step = max(a.tokens_per_step // seq, per_micro)
    seqs_per_step = (seqs_per_step + per_micro - 1) // per_micro * per_micro
    accum = seqs_per_step // per_micro
    tokens_per_step = seqs_per_step * seq
    print(f"[train] {jax.default_backend()} x {ndev}: {seqs_per_step} sequences of {seq} tokens per step ({tokens_per_step} tokens): {accum} micro-steps of {per_micro} sequences ({micro} per device); data {data.total_train / 1e9:.2f} G training tokens, {data.val_tokens / 1e6:.1f} M validation tokens", flush=True)
    n_params = sum(int(np.prod(s)) for _, s in layout(c))
    print(f"[train] model {a.layers} layers, dim {a.dim}, {a.heads}/{c.n_kv} heads, ffn {a.ffn}, vocab {a.vocab}: {n_params / 1e6:.1f} M parameters, optimizer {a.optim}, structure relu2={a.relu2} lambdas={a.lambdas} skips={a.skips} ve={list(c.ve_layers)}", flush=True)
    start = 0
    if a.resume and (out / "ckpt_latest.bin").exists():
        params, m, v, start = load_checkpoint(out / "ckpt_latest.bin", c)
        print(f"[train] resumed from step {start}", flush=True)
        plan = checkpoint_plan(out / "ckpt_latest.bin")
        if auto_steps and plan and start >= int((1.0 - h.decay_frac) * plan):
            # the checkpoint lies inside the learning-rate decay of its run: the run must keep the schedule it was planned with (a different total would jump the learning rate back up)
            a.steps = min(plan, steps_max)
            print(f"[train] the checkpoint is inside the decay of a run planned for {plan} steps: keeping that schedule ({a.steps} steps)", flush=True)
    elif a.init_from:
        params, _, _, _ = load_checkpoint(a.init_from, c, with_state=False)
        m = {n: jnp.zeros(s, F32) for n, s in layout(c)}
        v = {n: jnp.zeros(s, F32) for n, s in layout(c)}
        print(f"[train] starting from the parameters of {a.init_from}", flush=True)
    else:
        params = init_params(c, a.seed)
        m = {n: jnp.zeros(s, F32) for n, s in layout(c)}
        v = {n: jnp.zeros(s, F32) for n, s in layout(c)}
    params, m, v = (jax.device_put(t, repl_sh) for t in (params, m, v))
    step_count = jax.device_put(jnp.asarray(start, jnp.int32), repl_sh)
    if a.attn != "xla" and jax.default_backend() == "tpu":
        try:
            ATTN["impl"] = select_attention(c, mesh, cdt, per_micro, force=(a.attn == "splash"))
        except Exception as e:  # noqa: BLE001 (the selection is an optimisation: whatever goes wrong, train with XLA's attention)
            print(f"[attn] selection failed ({type(e).__name__}: {str(e).strip()[:200]}): XLA attention", flush=True)
            ATTN["impl"] = None
    if a.attn_test_fail:
        def failing_attention(c_, q, k, v_, cdt_, mode=a.attn_test_fail, ndev_=ndev):
            if mode == "step" or q.shape[0] == ndev_:
                raise RuntimeError("test failure of the attention kernel")
            return attention_xla(c_, q, k, v_, cdt_)

        ATTN["impl"] = failing_attention
    print(f"[train] attention: {'splash kernel' if ATTN['impl'] else 'XLA ops'}", flush=True)
    progress.update(t=time.time(), what="attention selection")
    train_step = build_train_step(c, h, cdt, a.ce_chunk, not a.no_remat)
    evaluate = build_eval(c, cdt, a.ce_chunk)
    rng = np.random.default_rng(a.seed * 1000003 + start)
    log = open(out / "log.jsonl", "a")
    produced = {"n": 0}

    def next_batch():
        produced["n"] += 1
        if produced["n"] == int(os.environ.get("FORGE_TEST_PRODUCER_FAIL", "0") or 0):
            raise OSError("test failure of the batch reader")  # tests only
        t, y, ig = data.train_batch(rng, seqs_per_step, seq)
        return tuple(jax.device_put(x.reshape(accum, per_micro, seq), batch_sh) for x in (t, y, ig))

    # a background thread prepares the next batch while the devices work on the current one
    q = Queue(maxsize=3)

    def producer():
        try:
            while True:
                q.put(next_batch())
        except BaseException as e:  # noqa: BLE001 (hand the error to the training loop: a dead producer must not leave it waiting forever)
            q.put(e)

    threading.Thread(target=producer, daemon=True).start()
    t_start = time.time()
    # with a time limit the total number of steps follows the measured speed (the schedule must reach its decay phase before the limit): re-planned every 100 steps until the decay starts
    plain_dts, eval_dts, save_dts = [], [], []
    step = start
    while step < a.steps:
        t0 = time.time()
        lr = lr_at(step, a.steps, h.peak_lr, h.warmup, h.min_frac, h.decay_frac)
        lr_muon = lr * h.muon_lr / h.peak_lr
        item = q.get(timeout=900)  # raises queue.Empty if no batch arrives for 15 minutes (a hung reader stops the run instead of holding the session)
        if isinstance(item, BaseException):
            raise RuntimeError("the batch producer thread failed") from item
        tokens, targets, ignore = item
        wait = time.time() - t0
        if os.environ.get("FORGE_TEST_HANG_AT_STEP") == str(step + 1):
            time.sleep(10**6)  # tests only: a hung device
        try:
            params, m, v, step_count, loss, gnorm = train_step(params, m, v, step_count, tokens, targets, ignore, jnp.float32(lr), jnp.float32(lr_muon))
            loss_v, gnorm_v = float(loss), float(gnorm)
        except Exception as e:  # noqa: BLE001
            if ATTN["impl"] is None or step != start:
                raise
            # the splash kernel passed its self-test but the whole step does not compile or run with it: nothing was consumed (the failure is at compile time), start again with XLA's attention
            print(f"[attn] the splash kernel failed in the full training step ({type(e).__name__}: {str(e).strip()[:400]}): XLA attention", flush=True)
            ATTN["impl"] = None
            train_step = build_train_step(c, h, cdt, a.ce_chunk, not a.no_remat)
            evaluate = build_eval(c, cdt, a.ce_chunk)
            params, m, v, step_count, loss, gnorm = train_step(params, m, v, step_count, tokens, targets, ignore, jnp.float32(lr), jnp.float32(lr_muon))
            loss_v, gnorm_v = float(loss), float(gnorm)
        dt = time.time() - t0
        plain_dts.append(dt)
        line = f"step {step + 1:5}/{a.steps} loss {loss_v:.4f} lr {lr:.2e} gnorm {gnorm_v:.3f} scale 1 {tokens_per_step / dt:.0f} tok/s ({dt:.2f} s)" + (f" data wait {wait:.2f} s" if wait > 0.02 else "")
        rec = {"step": step + 1, "loss": loss_v, "lr": lr, "grad_norm": gnorm_v, "tokens_per_s": tokens_per_step / dt}
        if (step + 1) % a.eval_every == 0 or step + 1 == a.steps:
            t_e = time.time()
            vl = 0.0
            for k in range(a.eval_batches):
                t_, y_, i_ = data.val_batch(k, 2, seq)  # two windows per batch as in the CUDA trainer; repeated so that every device gets a row (the mean is unchanged)
                reps = (ndev + 1) // 2
                t_, y_, i_ = (np.tile(x, (reps, 1))[:ndev] if ndev > 2 else x for x in (t_, y_, i_))
                try:
                    vl += float(evaluate(params, *(jax.device_put(x, eval_sh) for x in (t_, y_, i_))))
                except Exception as e:  # noqa: BLE001
                    if ATTN["impl"] is None:
                        raise
                    print(f"[attn] the splash kernel failed in the evaluation ({type(e).__name__}: {str(e).strip()[:400]}): XLA attention for the evaluation", flush=True)
                    ATTN["impl"] = None  # the training step is already compiled with its attention; only the evaluation is traced again
                    evaluate = build_eval(c, cdt, a.ce_chunk)
                    vl += float(evaluate(params, *(jax.device_put(x, eval_sh) for x in (t_, y_, i_))))
            vl /= a.eval_batches
            eval_dts.append(time.time() - t_e)
            line += f"  val {vl:.4f}"
            rec["val_loss"] = vl
        if (step + 1) % a.print_every == 0 or step < start + 20 or " val " in line or step + 1 == a.steps:
            print(line, flush=True)
        log.write(json.dumps(rec) + "\n")
        log.flush()
        progress.update(t=time.time(), what=f"step {step + 1}")
        if a.time_limit_s > 0 and step + 1 == int((1.0 - h.decay_frac) * a.steps) and not (out / "ckpt_predecay.bin").exists():
            # the last state before the learning-rate decay: a later session can start the decay again from here (for example with another data mix) if the run has to be repeated
            save_checkpoint(out / "ckpt_predecay.bin", c, params, m, v, step + 1, plan_steps=a.steps)
            print(f"[train] pre-decay checkpoint at step {step + 1}", flush=True)
            progress.update(t=time.time(), what="pre-decay checkpoint")
        if (step + 1) % a.save_every == 0 or step + 1 == a.steps or (a.time_limit_s > 0 and time.time() - t_start > a.time_limit_s):
            t_s = time.time()
            save_checkpoint(out / "ckpt_latest.bin", c, params, m, v, step + 1, plan_steps=a.steps)
            save_dts.append(time.time() - t_s)
            progress.update(t=time.time(), what="checkpoint")
            print(f"[train] checkpoint at step {step + 1} ({time.time() - t_start:.0f} s elapsed, {save_dts[-1]:.0f} s to write)", flush=True)
            if a.time_limit_s > 0 and time.time() - t_start > a.time_limit_s:
                print(f"[train] time limit of {a.time_limit_s:.0f} s reached: stop", flush=True)
                break
        step += 1
        if auto_steps and step % 100 == 0 and len(plain_dts) >= 60 and int((1.0 - h.decay_frac) * a.steps) > step + 25:
            step_secs = float(np.median(plain_dts[5:][-200:]))
            overhead = (float(np.median(eval_dts)) / a.eval_every if eval_dts else 0.0) + ((float(np.median(save_dts)) if save_dts else 25.0) / a.save_every)
            new_steps = plan_steps(steps_max, step, time.time() - t_start, step_secs, overhead, a.time_limit_s, a.plan_reserve_s)
            if new_steps != a.steps and int((1.0 - h.decay_frac) * new_steps) > step + 25 and abs(new_steps - a.steps) >= max(10, a.steps // 100):
                print(f"[train] time plan: {step_secs:.3f} s per step (+{overhead:.3f} s evaluation and checkpoints) -> {new_steps} steps fit into {a.time_limit_s:.0f} s (was {a.steps}, at most {steps_max})", flush=True)
                a.steps = new_steps
    return 0


if __name__ == "__main__":
    sys.exit(main())
