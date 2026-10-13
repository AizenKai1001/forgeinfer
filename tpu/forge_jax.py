"""Forge-1 trainer in JAX (TPU, GPU or CPU): the model, the optimizers (Muon + AdamW with learning-rate groups), the learning-rate schedule, the data windows and the checkpoint format of the Rust/CUDA trainer
(crates/forge-train), so that a run can start on one and continue on the other. The model is validated against the PyTorch goldens of tools/gen_train_goldens.py (tpu/test_forge_jax.py).

    python forge_jax.py train --data DIR --out DIR --layers 30 --dim 576 --heads 9 --kv 3 --ffn 2304 --seq 2048 --tokens-per-step 65536 --steps 45000 --relu2 --lambdas ... (the flags of `forge train`)

Differences from the CUDA trainer, on purpose: bf16 instead of fp16 operands (no loss scaling), XLA's attention instead of our flash kernels, jax.random instead of our LCG for the initial weights and the data windows.
Not implemented: the hashed n-gram memory (--ngram), which Forge-1 v1 does not use.
"""
import argparse
import faulthandler
import json
import math
import os
import shutil
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from queue import Full, Queue


def _pin_wanted(now, pin):
    """now = ['0.7.2', 'tpu', '8'] (version, backend, devices) of the preinstalled JAX or None: pin only on a TPU whose JAX is not the tested version."""
    return bool(now) and now[1] == "tpu" and now[0] != pin


def _shrink_time_limit(argv, spent):
    """argv with its --time-limit-s (both spellings) reduced by `spent` seconds (at least 1 s remain): a child process that is started after a preparation of `spent` seconds must not count its limit from zero again."""
    out = list(argv)
    for i, t in enumerate(out):
        try:
            if t == "--time-limit-s" and i + 1 < len(out):
                out[i + 1] = f"{max(float(out[i + 1]) - spent, 1.0):.0f}"
            elif t.startswith("--time-limit-s="):
                out[i] = f"--time-limit-s={max(float(t.split('=', 1)[1]) - spent, 1.0):.0f}"
        except ValueError:
            pass
    return out


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
    rc = subprocess.call([sys.executable, "-u"] + _shrink_time_limit(sys.argv, time.time() - t0), env=env)  # the preparation (pip install, probes) is part of the time limit: the child must not count it again
    if rc == 0 or time.time() - t1 > 900:
        sys.exit(rc)
    print(f"[pin] the pinned trainer failed after {time.time() - t1:.0f} s (exit code {rc}): continuing with the preinstalled JAX", flush=True)
    sys.argv[:] = _shrink_time_limit(sys.argv, time.time() - t0)
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
from jax.ad_checkpoint import checkpoint_name  # (jax.ad_checkpoint is no attribute of jax until it has been imported: jax 0.11.2)

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
    window: int = 0  # sliding-window attention (crates/forge-train): in the layers of `window_layers` query i sees the keys j with i - window < j <= i; 0 = every layer is dense causal
    window_layers: tuple = ()

    def window_of(self, l, t=None):
        """The attention window of layer `l` for sequences of `t` positions (default: seq), None = dense causal. As the Rust trainer: a window of at least t positions is the dense case."""
        t = self.seq if t is None else t
        if self.window > 0 and l in self.window_layers and self.window < t:
            return self.window
        return None

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


def auto_window_layers(n_layer):
    """`--window-layers auto` of the Rust trainer (TrainConfig::auto_window_layers): three of every four layers windowed; every fourth layer (3, 7, 11, ...) and the last layer stay dense."""
    return tuple(l for l in range(n_layer) if l % 4 != 3 and l + 1 != n_layer)


def parse_window_layers(spec, window, n_layer):
    """The layers of `--window-layers` (as the Rust CLI): auto (or no flag) = auto_window_layers when --window is on, else none; otherwise the layer numbers separated by commas. ValueError for a layer outside the model."""
    if spec in ("", "auto"):
        return auto_window_layers(n_layer) if window > 0 else ()
    layers = tuple(int(x) for x in spec.split(",") if x)
    if any(l < 0 or l >= n_layer for l in layers):
        raise ValueError(f"--window-layers {spec}: the layers must be layer indices of the model (0 to {n_layer - 1})")
    return layers


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


def rms(x, w, eps, name=None):
    """x / rms(x) * w over the last dimension (fp32). name (default None: nothing is added to the program): the checkpoint name of the mean square plus eps, a [.., 1] array, for a remat policy that saves it (the `norm`
    part; the value before the rsqrt, because the derivative of rsqrt needs it as well as the result: a saved result alone would still make the backward pass reduce over x again)."""
    x32 = x.astype(F32)
    ms = jnp.mean(x32 * x32, axis=-1, keepdims=True) + eps
    if name is not None:
        ms = checkpoint_name(ms, name)
    return x32 * jax.lax.rsqrt(ms) * w


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


def mm(a, w, cdt, name=None):
    """a [..., in] times w [out, in]^T with `cdt` operands and an fp32 result. name (default None: nothing is added to the program): the checkpoint name of the bf16 cast of `a` for a remat policy that saves it. Every call
    that uses `a` casts and names it itself (and not once for all of them): the backward pass then adds the fp32 cotangents of the casts, as it does without a name (a shared bf16 value would add them in bf16)."""
    if name is not None:
        return jnp.einsum("...i,oi->...o", checkpoint_name(a.astype(cdt), name), w.astype(cdt), preferred_element_type=F32)
    return jnp.einsum("...i,oi->...o", a.astype(cdt), w.astype(cdt), preferred_element_type=F32)


ATTN = {"impl": None}  # None: XLA's attention (attention_xla); on a TPU main() may put a faster kernel here (select_attention)


def attention_xla(c: Cfg, q, k, v, cdt, window=None):
    """Causal attention with grouped keys/values in plain XLA ops: q [B, T, H, hd], k and v [B, T, KV, hd] (fp32) -> [B, T, H * hd] (fp32). The [T, T] scores of every head are written to memory.
    window (a layer of the sliding-window attention): query i sees only the keys j with i - window < j <= i, the mask of the Rust trainer's flash kernels."""
    b, t = q.shape[:2]
    rep = c.n_head // c.n_kv
    q = q.reshape(b, t, c.n_kv, rep, c.hd).astype(cdt)
    att = jnp.einsum("btgrd,bsgd->bgrts", q, k.astype(cdt), preferred_element_type=F32) / math.sqrt(c.hd)
    mask = jnp.tril(jnp.ones((t, t), bool))
    if window is not None:
        pos = jnp.arange(t)
        mask = mask & ((pos[:, None] - pos[None, :]) < window)
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


ATTN_RES_NAME = "forge_attn_res"  # the checkpoint name of the splash kernel's residuals (output and logsumexp) for --remat-policy dots+attn
# the extra saves of a remat policy (--remat-policy dots+PART+PART...): token -> checkpoint name. h, h2, act: the bf16 operands of the layer's matmuls (the outputs of attn_norm and of ffn_norm, the relu-squared
# activation; layer_fn names them), qkv: the prepared bf16 q, k, v that feed the splash kernel (make_splash_attention names them), attn: the kernel's output and logsumexp, norm: the mean square (plus eps) of the two
# normalisations of the residual stream (attn_norm, ffn_norm), qknorm: that of the q and k norms (small fp32 arrays; the backward pass then does not make another pass over x, q and k for them). acts = h + h2 + act.
SAVE_NAMES = {"attn": ATTN_RES_NAME, "h": "forge_h", "h2": "forge_h2", "act": "forge_act", "qkv": "forge_qkv", "norm": "forge_norm", "qknorm": "forge_qknorm"}
SAVE_GROUPS = {"acts": ("h", "h2", "act")}
LAYER_PARTS = ("h", "h2", "act", "norm", "qknorm")  # the parts that layer_fn names itself ("attn" and "qkv" are named inside the splash attention)
REMAT_POLICIES = ("dots", "dots+attn", "dots+qkv", "dots+attn+qkv", "dots+acts", "dots+attn+acts", "dots+acts+qkv", "dots+attn+acts+qkv")  # the documented values; any '+' combination of the parts is accepted


def parse_remat(spec):
    """A remat policy spec 'dots[+PART...][@N]' -> (parts, last_n). parts: the tuple of the extra saves (SAVE_NAMES keys, 'acts' expanded, canonical order); last_n: the number of layers, counted from the last one, that carry
    them (0 = every layer; written @N after the parts, the form that --remat-save-layers N gives the policy). ValueError for anything else."""
    head, sep, tail = str(spec).partition("@")
    items = head.split("+")
    ok = items[0] == "dots" and all(p in SAVE_NAMES or p in SAVE_GROUPS for p in items[1:])
    try:
        last_n = int(tail) if sep else 0
    except ValueError:
        ok, last_n = False, 0
    if not ok or last_n < 0 or (sep and last_n == 0):
        raise ValueError(f"--remat-policy {spec}: must be one of {', '.join(REMAT_POLICIES)} (or dots plus any of the parts {', '.join(list(SAVE_NAMES) + list(SAVE_GROUPS))} joined by +, optionally followed by @N = only in the last N layers)")
    got = {q for p in items[1:] for q in SAVE_GROUPS.get(p, (p,))}
    return tuple(p for p in SAVE_NAMES if p in got), last_n


def remat_policy_arg(s):
    """argparse `type` of --remat-policy: the spec itself when parse_remat accepts it."""
    try:
        parse_remat(s)
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e).split(": ", 1)[-1])
    return s


def remat_extras(spec, layer, n_layer):
    """The extra saves (parts) that layer `layer` of `n_layer` carries under the policy spec: all of them in every layer, or in the last N layers when the spec says @N."""
    parts, last_n = parse_remat(spec)
    return parts if last_n <= 0 or layer >= n_layer - last_n else ()


def splash_names(spec):
    """The keyword arguments of make_splash_attention / select_attention that a policy needs: the checkpoint names of the kernel's residuals (attn) and of its prepared operands (qkv), None for a part that is not asked for."""
    parts, _ = parse_remat(spec)
    return {"residual_name": SAVE_NAMES["attn"] if "attn" in parts else None, "qkv_name": SAVE_NAMES["qkv"] if "qkv" in parts else None}


def remat_policy_of(name):
    """The `jax.checkpoint` policy of a layer. dots (the default, the program of every earlier run): the outputs of the matmuls without batch dimensions are saved, everything else is recomputed in the backward pass,
    the splash-attention forward kernel included (a Pallas call is not a dot). dots+attn: in addition the values named ATTN_RES_NAME are saved, the output and the logsumexp of the splash kernel (a splash attention
    built with residual_name=ATTN_RES_NAME): the backward pass then only runs the backward kernels. With XLA's attention nothing carries that name and the program is the one of `dots`.
    dots+acts, dots+qkv and their combinations (parse_remat) also save the named bf16 matmul operands h, h2, act (layer_fn) and the prepared q, k, v of the splash kernel (make_splash_attention): the same numbers (a
    saved bf16 value is the one that the recomputation would make), more memory, less recomputation. The names are only put into the program when the policy asks for them. The policy returned is the one for a layer
    that carries the extras (see remat_extras); a layer without them uses `dots`."""
    dots = jax.checkpoint_policies.dots_with_no_batch_dims_saveable
    parts, _ = parse_remat(name)
    if not parts:
        return dots
    return jax.checkpoint_policies.save_from_both_policies(dots, jax.checkpoint_policies.save_only_these_names(*(SAVE_NAMES[p] for p in parts)))


def named_operands(name, *arrays):
    """The arrays marked with the checkpoint `name` for a remat policy (checkpoint_name: the identity in every computation); name None = the arrays unchanged, nothing added to the program."""
    return arrays if not name else tuple(checkpoint_name(a, name) for a in arrays)


def make_splash_attention(c: Cfg, mesh, block=512, fused_bwd=False, interpret=False, residual_name=None, qkv_name=None):
    """The same attention as a Pallas splash-attention kernel (TPU): the scores stay in the chip's fast memory, blocks above the diagonal are skipped. The batch is split over the mesh axis 'data'
    (a Pallas kernel cannot be partitioned automatically). Returns attn(c, q, k, v, cdt, local=False) with the signature of attention_xla; local=True is the attention of a sliding-window layer (c.window), served by a
    second kernel with the same block sizes that is only built when the model has such a layer. In the splash masks True = the query sees the key: the window mask is LocalMask((T, T), window_size=(window - 1, 0)),
    the keys j with i - (window - 1) <= j <= i, i.e. i - window < j <= i (the causal mask and the window in one; the kernel skips the blocks outside the band).
    residual_name (default None = unnamed, the program of every earlier run): a checkpoint name for the kernel's forward residuals (output and logsumexp), which a remat policy can save (remat_policy_of).
    qkv_name (default None = unnamed): a checkpoint name for the three prepared bf16 operands of the kernel (q scaled, k, v in [B, H, T, hd]), which a remat policy can save (the +qkv policies)."""
    from jax.sharding import PartitionSpec as P
    from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel as sk
    from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask as sm

    t = c.seq
    blk = min(block, t)
    assert t % blk == 0 and blk % 128 == 0, (t, blk)
    fwd = dict(block_q=blk, block_kv=blk, block_kv_compute=blk, block_q_dkv=blk, block_kv_dkv=blk, block_kv_dkv_compute=blk)
    sizes = sk.BlockSizes(**fwd, use_fused_bwd_kernel=True) if fused_bwd else sk.BlockSizes(**fwd, block_q_dq=blk, block_kv_dq=blk)
    spec = P("data")

    def build(one_mask):
        mask = sm.MultiHeadMask([one_mask for _ in range(c.n_head)])
        extra = ({"interpret": True} if interpret else {}) | ({"residual_checkpoint_name": residual_name} if residual_name else {})
        kernel = sk.make_splash_mha_single_device(mask, block_sizes=sizes, **extra)
        return _shard_map(lambda q, k, v: jax.vmap(kernel)(q, k, v), mesh, (spec, spec, spec), spec)

    mapped = build(sm.CausalMask((t, t)))
    has_local = any(c.window_of(l) for l in range(c.n_layer))
    mapped_local = build(sm.LocalMask((t, t), window_size=(c.window - 1, 0), offset=0)) if has_local else None

    def attn(c_, q, k, v, cdt, local=False):
        b, t_ = q.shape[:2]
        if local and mapped_local is None:
            raise ValueError("this splash attention was built for a model without sliding-window layers")
        qh = (q * (1.0 / math.sqrt(c_.hd))).astype(cdt).transpose(0, 2, 1, 3)  # [B, H, T, hd]; the kernel does not scale the scores
        kh, vh = k.astype(cdt).transpose(0, 2, 1, 3), v.astype(cdt).transpose(0, 2, 1, 3)
        qh, kh, vh = named_operands(qkv_name, qh, kh, vh)  # the identity (nothing is added to the program) unless a remat policy asks to save the prepared operands
        o = (mapped_local if local else mapped)(qh, kh, vh)
        return o.transpose(0, 2, 1, 3).reshape(b, t_, c_.q_dim).astype(F32)

    def prepared(c_, qh, kh, vh, local=False):
        """The same attention on inputs that are already in the kernel's form (--attn-prep lane): qh [B, H, T, hd] scaled by 1/sqrt(hd), kh and vh [B, KV, T, hd], all in the compute dtype. Returns [B, T, H * hd]
        in the kernel's dtype (attn's result is this converted to float32: the matmul that consumes it converts back, a round trip that changes no value)."""
        if local and mapped_local is None:
            raise ValueError("this splash attention was built for a model without sliding-window layers")
        qh, kh, vh = named_operands(qkv_name, qh, kh, vh)  # the identity unless a remat policy asks to save the prepared operands
        o = (mapped_local if local else mapped)(qh, kh, vh)
        return o.transpose(0, 2, 1, 3).reshape(qh.shape[0], qh.shape[2], c_.q_dim)

    attn.prepared = prepared
    return attn


def _xla_window(window, c, q, k, v, cdt):
    """attention_xla of a sliding-window layer with the signature of an attention implementation (the reference of the windowed splash kernel)."""
    return attention_xla(c, q, k, v, cdt, window)


def _rel_errors(ref, out):
    """Relative differences of two (value, (dq, dk, dv)) results of value_and_grad."""
    (rv, rg), (ov, og) = ref, out
    errs = [abs(float(ov) - float(rv)) / (abs(float(rv)) + 1e-6)]
    for a_, b_ in zip(og, rg):
        errs.append(float(jnp.linalg.norm((a_ - b_).astype(F32)) / (jnp.linalg.norm(b_.astype(F32)) + 1e-9)))
    return [round(e, 4) for e in errs]


def select_attention(c: Cfg, mesh, cdt, per_micro, force=False, candidates=((512, False), (512, True), (256, False), (1024, False)), tol=0.05, log=print, residual_name=None, qkv_name=None):
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

    n_local = sum(1 for l in range(c.n_layer) if c.window_of(l))  # sliding-window layers: a second kernel, checked and timed against XLA's windowed attention with the same window
    ref, t_ref = run(make(attention_xla))
    log(f"[attn] XLA attention, forward + backward of {b} sequences per device: {t_ref * 1e3:.1f} ms (x{c.n_layer} layers = {t_ref * c.n_layer:.2f} s per micro-step)")
    ref_l, t_ref_l = None, 0.0
    if n_local:
        ref_l, t_ref_l = run(make(partial(_xla_window, c.window)))
        log(f"[attn] XLA attention of the {n_local} sliding-window layers (window {c.window}): {t_ref_l * 1e3:.1f} ms per layer")

    def per_layer(dense, local):  # the mean time of a layer over the dense and the windowed ones
        return dense if not n_local else ((c.n_layer - n_local) * dense + n_local * local) / c.n_layer

    t_ref_mix = per_layer(t_ref, t_ref_l)
    best_t, best_impl, best_name = float("inf"), None, ""
    for blk, fused in candidates:
        name = f"splash block {blk}" + (" fused backward" if fused else "")
        try:
            impl = make_splash_attention(c, mesh, blk, fused, residual_name=residual_name, qkv_name=qkv_name)
            out, dt = run(make(impl))
            errs = _rel_errors(ref, out)
            dt_l, errs_l = 0.0, []
            if n_local:  # BOTH kernels must agree with XLA: a candidate whose windowed kernel fails or differs drops out as a whole
                out_l, dt_l = run(make(lambda c_, q_, k_, v_, cdt_, impl=impl: impl(c_, q_, k_, v_, cdt_, local=True)))
                errs_l = _rel_errors(ref_l, out_l)
        except Exception as e:  # noqa: BLE001 (a kernel that does not compile or run on this JAX / TPU only drops out)
            text = str(e).strip().splitlines()
            log(f"[attn] {name}: failed ({type(e).__name__}: {text[0][:200] if text else ''})")
            continue
        mix = per_layer(dt, dt_l)
        ok = max(errs + errs_l) < tol
        extra = f"; sliding-window layers {dt_l * 1e3:.1f} ms, relative errors {errs_l}" if n_local else ""
        log(f"[attn] {name}: {dt * 1e3:.1f} ms (x{c.n_layer} = {mix * c.n_layer:.2f} s), relative errors value/dq/dk/dv {errs}{extra}: {'ok' if ok else 'MISMATCH'}")
        if ok and mix < best_t:
            best_t, best_impl, best_name = mix, impl, name
    if best_impl is None:
        log("[attn] no splash candidate usable: XLA attention")
        return None
    if force or best_t < 0.9 * t_ref_mix:
        log(f"[attn] using {best_name} ({t_ref_mix / best_t:.2f}x faster than XLA's)")
        return best_impl
    log(f"[attn] best splash candidate ({best_name}, {best_t * 1e3:.1f} ms) is not clearly faster than XLA's ({t_ref_mix * 1e3:.1f} ms): XLA attention")
    return None


ATTN_PREPS = ("plain", "lane")
# --layer-opts: named options of the layer program, a comma separated list (default empty = the program of every earlier run). Each one is a different formulation of the same layer, found with the compile lab (tpu/hlo_cost.py
# --matmuls); the ones that change the order of a floating point sum say so in the option's line of --help and are measured in tpu/test_layers2.py.
LAYER_OPTS = {
    "ffnbf16": "the bf16 copies of the feed-forward weights (w_up, w_down) are made once per layer, in their own fusions (an optimization barrier keeps the cast out of the matmul fusions), instead of inside every matmul that "
               "reads them: the same numbers. Compile lab (v5e-8, 365M, estimated step): in the lane programs the matmuls on the f32 weights lose their prefetch into the fast memory in most layers (a w_down matmul 0.98 ms instead of "
               "0.42 ms); -9.6 ms with --attn-prep lane --muon-shard --muon-lean, -3 ms with --attn-prep lane alone, +3.9 ms on the plain program; the attention weights gain nothing from it",
    "woheads": "the attention output goes into the output projection as [B, heads, T, hd] (the layout the kernel makes) and contracts over heads and hd, instead of through a transposed copy [B, T, heads x hd]; the same function, "
               "the sum over the 1024 inputs is made per head first (fp32 accumulation order differs); lab: -0.5 to -1 ms, within its noise",
}


def parse_layer_opts(spec):
    """'a,b' -> the frozenset of option names of --layer-opts; ValueError for an unknown name."""
    names = [x.strip() for x in str(spec or "").split(",") if x.strip()]
    bad = [x for x in names if x not in LAYER_OPTS]
    if bad:
        raise ValueError(f"--layer-opts {spec}: unknown option {', '.join(bad)} (choices: {', '.join(LAYER_OPTS)})")
    return frozenset(names)


FFN_WEIGHTS = ("w_up", "w_down", "w_gate")  # the feed-forward weights of a layer (w_gate only in a model without relu2)


def layer_opts_arg(s):
    """argparse `type` of --layer-opts: the spec itself when parse_layer_opts accepts it."""
    try:
        parse_layer_opts(s)
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e).split(": ", 1)[-1])
    return s


def attend(c: Cfg, cdt, q, k, v, window):
    """The attention of one layer on q [B, T, H, hd], k and v [B, T, KV, hd] (fp32): the splash kernel when one was chosen (ATTN["impl"]), else XLA's. window: the sliding window of the layer (None = dense causal)."""
    impl = ATTN["impl"]
    if window is None:
        return (impl or attention_xla)(c, q, k, v, cdt)
    if impl is None:
        return attention_xla(c, q, k, v, cdt, window)
    return impl(c, q, k, v, cdt, local=True)  # the windowed kernel of the splash attention (its window is c.window)


def attn_block_plain(c: Cfg, cdt, h, lw, cos, sin, velam_l, tokens_ve, window, fuse_qkv, name_h=None, name_norm=None):
    """From the normalised layer input h [B, T, E] to the attention output [B, T, H * hd] (fp32): the projections, the qk-norm, RoPE and the value-embedding mix on arrays [B, T, heads, hd], the attention. The program of every earlier run. name_h: the checkpoint name of the bf16 cast of h (each projection casts and names it itself, see mm) for a remat policy that saves it."""
    b, t = h.shape[:2]
    if fuse_qkv:
        qkv = mm(h, jnp.concatenate([lw["wq"].astype(cdt), lw["wk"].astype(cdt), lw["wv"].astype(cdt)], axis=0), cdt, name_h)  # [b, t, q_dim + 2 kv_dim]; the gradient of the concatenation is the three slices
        q, k, v = (a_.reshape(b, t, n, c.hd) for a_, n in zip(jnp.split(qkv, [c.q_dim, c.q_dim + c.kv_dim], axis=-1), (c.n_head, c.n_kv, c.n_kv)))
    else:
        q = mm(h, lw["wq"], cdt, name_h).reshape(b, t, c.n_head, c.hd)
        k = mm(h, lw["wk"], cdt, name_h).reshape(b, t, c.n_kv, c.hd)
        v = mm(h, lw["wv"], cdt, name_h).reshape(b, t, c.n_kv, c.hd)
    if c.qk_norm:
        q = rms(q, lw["q_norm"], c.eps, name_norm)
        k = rms(k, lw["k_norm"], c.eps, name_norm)
    q, k = rope(q, cos, sin), rope(k, cos, sin)
    if tokens_ve is not None:
        v = velam_l[0] * v + velam_l[1] * tokens_ve.reshape(b, t, c.n_kv, c.hd).astype(F32)
    return attend(c, cdt, q, k, v, window)


def rope_tables_lane(cos, sin):
    """The RoPE tables of rope_tables ([T, hd/2]) for the lane layout: [hd/2, T], T is the minor (lane) dimension there."""
    return cos.T, sin.T


def norm_rope_lane(x, w, cos_t, sin_t, eps, name=None):
    """rms() with the weight w [hd] (None = no qk-norm) followed by rope() on x [B, n, hd, T] (fp32): the two functions of the plain path, written per half of the head with the same arithmetic per element (x * rsqrt(mean(x * x) + eps) * w,
    then x1 * cos - x2 * sin and x2 * cos + x1 * sin), on an array whose last dimension is T (a multiple of 128 lanes) and whose head dimension lies on the sublanes: the mean is a reduction over sublanes, the two halves of a head are
    slices at multiples of 8 rows, nothing needs a lane shuffle. cos_t and sin_t [hd/2, T]. name: the checkpoint name of the mean square (see rms). Returns fp32 [B, n, hd, T]."""
    h = x.shape[2] // 2
    x1, x2 = x[:, :, :h], x[:, :, h:]
    if w is not None:
        ms = jnp.mean(x * x, axis=2, keepdims=True) + eps
        if name is not None:
            ms = checkpoint_name(ms, name)
        inv = jax.lax.rsqrt(ms)
        x1, x2 = x1 * inv * w[None, None, :h, None], x2 * inv * w[None, None, h:, None]
    cc, ss = cos_t[None, None], sin_t[None, None]
    return jnp.concatenate([x1 * cc - x2 * ss, x2 * cc + x1 * ss], axis=2)


def kernel_heads(a, cdt):
    """a [B, n, hd, T] -> [B, n, T, hd] in cdt: the transposition into the layout of the attention kernel's operands, done as a matmul with the identity matrix. Why not a transpose: XLA's layout assignment wants a transpose to be free
    (a bitcast) and so gives the whole chain of elementwise operations before it the kernel's layout, hd minor; a 64-wide minor dimension is padded to 128 lanes by the TPU (twice the memory traffic of every pass over q, k and v,
    four times for the half heads of RoPE). A matmul has fixed operand layouts, so the chain before it keeps T as its minor dimension, and the matrix unit transposes. Exact: every output element is one product with 1 plus exact zeros
    (a non-finite input value would turn its whole row into NaN through the zeros, which only happens in a run that is lost anyway)."""
    return jnp.einsum("bndt,de->bnte", a.astype(cdt), jnp.asarray(np.eye(a.shape[2]), cdt), preferred_element_type=cdt, precision=jax.lax.Precision.HIGHEST)


def attn_block_lane(c: Cfg, cdt, h, lw, cos, sin, velam_l, tokens_ve, window, fuse_qkv, name_h=None, name_norm=None):
    """attn_block_plain with q, k and v kept as [B, heads, hd, T] (T on the 128 lanes of the TPU, the head dimension on the sublanes) from the projection to the end of the preparation, so that the qk-norm, RoPE, the value-embedding
    mix, the scaling and the cast to bf16 run without arrays whose minor dimension is the 64-wide head dimension (padded to 128 lanes: twice the memory traffic, four times for the half heads of RoPE); one transposition into the
    kernel's [B, heads, T, hd] follows, in bf16 (kernel_heads). The same function as attn_block_plain; cos and sin are the tables of rope_tables_lane. With XLA's attention (or any implementation without a `prepared` entry) q, k and v
    are transposed back to fp32 [B, T, heads, hd] at the end and the attention is called as in the plain path."""
    b, t = h.shape[:2]
    def proj(w):  # [b, features, t]; every projection converts h itself (and names its own cast when a remat policy saves h), as mm does: the three gradients of h are then converted to fp32 one by one and summed in fp32
        hc = h.astype(cdt)
        if name_h is not None:
            hc = checkpoint_name(hc, name_h)
        return jnp.einsum("oe,bte->bot", w.astype(cdt), hc, preferred_element_type=F32)

    if fuse_qkv:
        q, k, v = jnp.split(proj(jnp.concatenate([lw["wq"], lw["wk"], lw["wv"]], axis=0)), [c.q_dim, c.q_dim + c.kv_dim], axis=1)
    else:
        q, k, v = proj(lw["wq"]), proj(lw["wk"]), proj(lw["wv"])
    q, k, v = q.reshape(b, c.n_head, c.hd, t), k.reshape(b, c.n_kv, c.hd, t), v.reshape(b, c.n_kv, c.hd, t)
    q = norm_rope_lane(q, lw["q_norm"] if c.qk_norm else None, cos, sin, c.eps, name_norm)
    k = norm_rope_lane(k, lw["k_norm"] if c.qk_norm else None, cos, sin, c.eps, name_norm)
    if tokens_ve is not None:
        v = velam_l[0] * v + velam_l[1] * tokens_ve.transpose(0, 2, 1).reshape(b, c.n_kv, c.hd, t).astype(F32)
    prepared = getattr(ATTN["impl"], "prepared", None)
    if prepared is None:  # (the barrier: XLA:CPU of jax 0.11.2 leaves a bf16 dot whose operand is a fused concatenate + convert outside any fusion, and its thunk runtime cannot run BF16 x BF16 = F32; only this path, the tests and the fallback, is affected)
        q, k, v = jax.lax.optimization_barrier((q.transpose(0, 3, 1, 2), k.transpose(0, 3, 1, 2), v.transpose(0, 3, 1, 2)))
        return attend(c, cdt, q, k, v, window)
    return prepared(c, kernel_heads(q * (1.0 / math.sqrt(c.hd)), cdt), kernel_heads(k, cdt), kernel_heads(v, cdt), local=window is not None)  # the kernel does not scale the scores: q goes in scaled


def layer_fn(c: Cfg, cdt, x, x0, skip, lam_l, velam_l, tokens_ve, lw, cos, sin, window=None, fuse_qkv=False, attn_prep="plain", save=(), opts=frozenset()):
    """One transformer layer. x: the input stream (fp32), x0: the embedding output, skip: the output of the mirrored layer or None, lam_l: (a, b, c), lw: the layer's weights.
    window: the sliding window of this layer (None = dense causal attention). fuse_qkv: the three projections as ONE matmul on [wq; wk; wv] (the parameters stay three arrays; the same numbers).
    attn_prep: how q, k and v are prepared for the attention, see attn_block_lane (--attn-prep); with "lane" cos and sin are the tables of rope_tables_lane.
    save: the operands of this layer's matmuls ('h', 'h2', 'act') whose bf16 cast carries a checkpoint name for the remat policy (remat_policy_of; the same numbers, see mm), and 'norm' / 'qknorm': the mean square of the layer's
    two residual-stream normalisations / of the q and k norms. Empty (the default): nothing is named and the program is the one without the option. opts: the --layer-opts that act inside the layer (woheads; see LAYER_OPTS)."""
    nh, nh2, nact, nnorm, nqk = (SAVE_NAMES[p] if p in save else None for p in ("h", "h2", "act", "norm", "qknorm"))
    b, t, e = x.shape
    if c.mixing:
        xin = (lam_l[0] if c.lambdas else 1.0) * x
        if c.lambdas:
            xin = xin + lam_l[1] * x0
        if c.skips and skip is not None:
            xin = xin + lam_l[2] * skip
        x = xin
    h = rms(x, lw["attn_norm"], c.eps, nnorm)
    o = (attn_block_lane if attn_prep == "lane" else attn_block_plain)(c, cdt, h, lw, cos, sin, velam_l, tokens_ve, window, fuse_qkv, nh, nqk)
    if "woheads" in opts:  # the output projection contracts over (heads, hd) of the kernel's own layout [B, H, T, hd]
        o4 = o.reshape(b, t, c.n_head, c.hd).transpose(0, 2, 1, 3).astype(cdt)
        x = x + jnp.einsum("bhtd,ehd->bte", o4, lw["wo"].astype(cdt).reshape(lw["wo"].shape[0], c.n_head, c.hd), preferred_element_type=F32)
    else:
        x = x + mm(o, lw["wo"], cdt)
    h2 = rms(x, lw["ffn_norm"], c.eps, nnorm)
    if c.relu2:
        act = jnp.square(jax.nn.relu(mm(h2, lw["w_up"], cdt, nh2)))
    else:
        act = jax.nn.silu(mm(h2, lw["w_gate"], cdt, nh2)) * mm(h2, lw["w_up"], cdt, nh2)
    return x + mm(act, lw["w_down"], cdt, nact)


def hidden(c: Cfg, params, tokens, cdt=BF16, remat=True, remat_policy="dots", fuse_qkv=False, attn_prep="plain", layer_opts=""):
    """The final hidden states [B, T, E] (after the output norm) for tokens [B, T]. remat_policy: what the layers' backward pass saves instead of recomputing, see remat_policy_of. attn_prep: see layer_fn.
    layer_opts: the --layer-opts of the layer program (LAYER_OPTS)."""
    b, t = tokens.shape
    opts = parse_layer_opts(layer_opts)
    cos, sin = rope_tables(c, t)
    if attn_prep == "lane":
        cos, sin = rope_tables_lane(cos, sin)
    x0 = params["emb"][tokens]
    outs = []
    ve_pos = {l: i for i, l in enumerate(c.ve_layers)}
    layers = {}  # (sliding window (None = dense), extra saves of the remat policy) -> the layer function: a windowed layer is a different program

    def layer_of(w, extras=()):
        if (w, extras) not in layers:
            named = {"save": tuple(p for p in extras if p in LAYER_PARTS)} if remat and extras else {}  # the matmul operands that this layer names for the remat policy (none: the program of the default)
            f = (partial(layer_fn, c, cdt, fuse_qkv=fuse_qkv, attn_prep=attn_prep, opts=opts, **named) if w is None
                 else partial(layer_fn, c, cdt, window=w, fuse_qkv=fuse_qkv, attn_prep=attn_prep, opts=opts, **named))
            layers[(w, extras)] = jax.checkpoint(f, policy=remat_policy_of("dots" if not extras else "dots+" + "+".join(extras))) if remat else f
        return layers[(w, extras)]

    for l in range(c.n_layer):
        layer = layer_of(c.window_of(l, t), remat_extras(remat_policy, l, c.n_layer) if remat else ())
        xprev = x0 if l == 0 else outs[l - 1]
        lw = {n: params[f"l{l}.{n}"] for n in ("wq", "wk", "wv", "wo", "w_up", "w_down", "attn_norm", "ffn_norm") + (("w_gate",) if not c.relu2 else ()) + (("q_norm", "k_norm") if c.qk_norm else ())}
        if "ffnbf16" in opts:  # the feed-forward weights as bf16 arrays made here, once (the barrier: not folded back into every matmul that reads them); the gradient is converted back as before
            lw = {n: (jax.lax.optimization_barrier(w_.astype(cdt)) if n in FFN_WEIGHTS else w_) for n, w_ in lw.items()}
        lam_l = params["lam"][l] if c.mixing else None
        skip = outs[c.n_layer - 1 - l] if (c.skips and l >= c.n_layer - c.n_enc) else None
        if l in ve_pos:
            tok_ve = params[f"ve{l}"][tokens].astype(cdt)  # gather the fp32 rows, then cast: the backward scatter-add then sums repeated tokens in fp32 (a bf16 table summed them in bf16: up to 3 % error on frequent tokens)
            velam_l = params["velam"][ve_pos[l]]
        else:
            tok_ve, velam_l = None, None
        outs.append(layer(xprev, x0, skip, lam_l, velam_l, tok_ve, lw, cos, sin))
    return rms(outs[-1], params["out_norm"], c.eps)


def loss_fn(c: Cfg, params, tokens, targets, ignore, cdt=BF16, ce_chunk=256, remat=True, remat_policy="dots", fuse_qkv=False, attn_prep="plain", layer_opts=""):
    """Mean cross entropy over the positions whose target is not ignored. tokens, targets: [B, T] int32; ignore: [B, T] bool. The logits are made `ce_chunk` positions at a time (and not kept for the backward pass)."""
    hf = hidden(c, params, tokens, cdt, remat, remat_policy, fuse_qkv, attn_prep, layer_opts)
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


def newton_schulz(x, iters=5, dtype=BF16, barrier=False):
    """Five iterations of the quintic Newton-Schulz orthogonalisation on the (Frobenius-normalised) matrix x; a tall matrix is iterated through its transpose (the same as the Rust trainer).
    barrier (--ns-barrier, default off): an optimization barrier after every iteration, the same values; it stops XLA from folding the elementwise chain of several iterations into the consumer matmul."""
    a_, b_, c_ = NS_COEF
    tall = x.shape[0] > x.shape[1]
    if tall:
        x = x.T
    x = x.astype(dtype)
    for _ in range(iters):
        a = jnp.einsum("ik,jk->ij", x, x, preferred_element_type=F32).astype(dtype)
        bb = (b_ * a.astype(F32) + c_ * jnp.einsum("ik,kj->ij", a, a, preferred_element_type=F32)).astype(dtype)
        x = (a_ * x.astype(F32) + jnp.einsum("ik,kj->ij", bb, x, preferred_element_type=F32)).astype(dtype)
        if barrier:
            x = jax.lax.optimization_barrier(x)
    x = x.astype(F32)
    return x.T if tall else x


def muon_update(p, g, mom, lr, wd, mu, ns_dtype=BF16, barrier=False):
    mom = mu * mom + g
    u = g + mu * mom
    x = u / (jnp.sqrt(jnp.sum(u * u)) + 1e-7)
    d = newton_schulz(x, 5, ns_dtype, barrier)
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
    muon_lean: bool = False  # --muon-lean: the Muon update per layer (see muon_group_lean); the same numbers, fewer passes over the fp32 state
    ns_barrier: bool = False  # --ns-barrier: an optimization barrier after every Newton-Schulz iteration (newton_schulz); the same numbers


def lr_at(step, steps, peak, warmup, min_frac, decay_frac):
    if step < warmup:
        return peak * (step + 1) / warmup
    decay_start = int((1.0 - decay_frac) * steps)
    if step < decay_start:
        return peak
    t = (step - decay_start) / max(steps - decay_start, 1)
    return peak * (1.0 - (1.0 - min_frac) * min(t, 1.0))


def cool_share(step, steps, decay_frac, blend_steps, max_share, start=-1):
    """The share of the cool-down pool in the batch of (0-based) step `step`: 0 before `start` (default: the first step of the learning-rate decay, the same int((1 - decay_frac) * steps) as lr_at), then a linear
    ramp over `blend_steps` steps up to `max_share` (blend_steps 0 = a hard switch). A pure function of the step, so a resumed run (or a session chain) continues the ramp where it stopped."""
    first = int((1.0 - decay_frac) * steps) if start < 0 else start
    if step < first:
        return 0.0
    if blend_steps <= 0:
        return max_share
    return max_share * min(1.0, (step - first) / blend_steps)


def cool_blend_len(steps, first, blend_steps, blend_frac):
    """Length of the ramp of the cool-down share in steps: `blend_frac` of the steps from the first cool-down step to the end of the run when that is given (it follows a time-planned number of steps), else `blend_steps`."""
    return int(blend_frac * max(steps - first, 0)) if blend_frac > 0 else blend_steps


def plan_steps(steps_max, done, elapsed, step_secs, overhead_per_step, time_limit, reserve=90.0, wall_per_step=0.0):
    """The total number of steps (at most steps_max) that the time limit allows: the steps done so far plus those that fit into the remaining time at the measured speed (`reserve` seconds are kept for the last evaluation and checkpoint).
    A step costs the larger of (median step time + the median evaluation and checkpoint share) and `wall_per_step`, the MEAN wall time per step since the start of the session: the medians describe a quiet moment, the mean
    contains every stall of the host or its disk (in the 365M run on Kaggle five stalls of 4 to 6 minutes cost 28 minutes, which the medians never saw)."""
    cost = max(step_secs + overhead_per_step, wall_per_step, 1e-9)
    n_more = int(max(time_limit - elapsed - reserve, 0.0) / cost)
    return min(steps_max, done + n_more)


def replan_steps(cur_steps, steps_max, done, elapsed, step_secs, overhead_per_step, time_limit, reserve, decay_frac, wall_per_step=0.0):
    """The new planned total of a run with a time limit, or None to keep the current plan. The plan only moves while the learning-rate decay has not started (afterwards a new total would jump the learning rate), while
    the decay would still start at least 25 steps ahead, and when it changes by at least a third of a percent (a hysteresis: not a new schedule every 100 steps)."""
    if int((1.0 - decay_frac) * cur_steps) <= done + 25:
        return None
    new = plan_steps(steps_max, done, elapsed, step_secs, overhead_per_step, time_limit, reserve, wall_per_step)
    if new == cur_steps or int((1.0 - decay_frac) * new) <= done + 25 or abs(new - cur_steps) < max(10, cur_steps // 300):
        return None
    return new


def muon_stack_update(h: Hyper, p, g, m, clip, lr_muon, mesh=None):
    """The Muon update of a stack [n, r, c] of same-shaped matrices (vmapped over the stack). With a `mesh` (the flag --muon-shard) only the Newton-Schulz orthogonalisation, the expensive part (five iterations of three
    matrix products per matrix, computed on every device when everything is replicated), is split over the axis "data": every device orthogonalises its share of the matrices and the directions are gathered back to
    every device in bf16 (the iterations run in bf16, so these are the same values: a quarter of the traffic of gathering the new weights and momenta in fp32). The momentum, the normalisation and the weight update
    stay elementwise and replicated, as the optimizer state is. A stack whose size is not a multiple of the number of devices is padded with zero matrices (the Newton-Schulz result of a zero matrix is zero) and
    cut back. The per-matrix math is the same as without a mesh (that path is unchanged)."""
    if mesh is None:
        return jax.vmap(lambda p_, g_, m_: muon_update(p_, g_ * clip, m_, lr_muon, h.muon_wd, h.mu, BF16, h.ns_barrier))(p, g, m)
    from jax.sharding import NamedSharding, PartitionSpec as P

    g = g * clip
    m_new = h.mu * m + g
    u = g + h.mu * m_new
    x = u / (jnp.sqrt(jnp.sum(u * u, axis=(1, 2), keepdims=True)) + 1e-7)
    n = p.shape[0]
    pad = (-n) % mesh.devices.size
    if pad:
        x = jnp.concatenate([x, jnp.zeros((pad,) + x.shape[1:], x.dtype)])
    x = jax.lax.with_sharding_constraint(x, NamedSharding(mesh, P("data", None, None)))
    d = jax.vmap(lambda x_: newton_schulz(x_, 5, BF16, True))(x) if h.ns_barrier else jax.vmap(newton_schulz)(x)  # float32 values that are exactly bf16 numbers (the iterations run in bf16)
    d = jax.lax.with_sharding_constraint(d.astype(BF16), NamedSharding(mesh, P()))[:n].astype(F32)
    scale = 0.2 * math.sqrt(max(p.shape[1:]))
    return p * (1.0 - lr_muon * h.muon_wd) - lr_muon * scale * d, m_new


def newton_schulz_lean(x, iters=5, barrier=False):
    """newton_schulz of a matrix that is already in bf16, returning bf16 in the same orientation: the float32 that newton_schulz returns holds exactly these bf16 numbers (its last operation is the rounding to
    bf16), so nothing is lost by not converting. A tall matrix is iterated through its transpose, as there."""
    a_, b_, c_ = NS_COEF
    tall = x.shape[0] > x.shape[1]
    if tall:
        x = x.T
    for _ in range(iters):
        a = jnp.einsum("ik,jk->ij", x, x, preferred_element_type=F32).astype(BF16)
        bb = (b_ * a.astype(F32) + c_ * jnp.einsum("ik,kj->ij", a, a, preferred_element_type=F32)).astype(BF16)
        x = (a_ * x.astype(F32) + jnp.einsum("ik,kj->ij", bb, x, preferred_element_type=F32)).astype(BF16)
        if barrier:
            x = jax.lax.optimization_barrier(x)
    return x.T if tall else x


def muon_group_lean(h: Hyper, shape, names, params, grads, mstate, clip, lr_muon, mesh=None):
    """The Muon update of the matrices `names` (all of one `shape`) without stacking the fp32 state. The stacked formulation (muon_stack_update) copies the parameters, gradients and momenta of the layers into
    three fp32 stacks, updates the stacks and cuts the result back into per-layer arrays: about 70 bytes of memory traffic per parameter, a third of it layout copies. Here the momentum, the normalisation and the
    weight update are elementwise per layer (the same formulas on the same values; XLA fuses them straight into the per-layer inputs and outputs) and only the bf16 input of the Newton-Schulz iterations and its bf16
    result are stacked (the matrices of the same shape go through one batched iteration, as before).
    With a `mesh` (--muon-shard) each device runs the iterations of its share of the stack inside a shard_map (the compiler's sharding propagation of the stacked formulation turned the momenta and the new weights into
    float32 collectives) and the bf16 results are gathered; a stack whose size is not a multiple of the number of devices is padded with zero matrices (their result is zero) as in muon_stack_update.
    Returns the new parameters and momenta by name."""
    from jax.sharding import NamedSharding, PartitionSpec as P

    xs, newm = [], {}
    for n in names:
        g = grads[n] * clip
        m_new = h.mu * mstate[n] + g
        u = h.mu * m_new + g  # the same value as g + mu * m_new; written with the product first so that the CPU compiler contracts it into the same fused multiply-add as the stacked formulation (bit-identical)
        xs.append((u / (jnp.sqrt(jnp.sum((u * u)[None], axis=(1, 2))[0]) + 1e-7)).astype(BF16))  # the norm as a reduction over axes (1, 2) of a [1, r, c] array: the very reduction of the stacked formulation
        newm[n] = m_new
    stack = jnp.stack(xs)
    ns = lambda s_: jax.vmap(lambda x_: newton_schulz_lean(x_, 5, h.ns_barrier))(s_)
    if mesh is None:
        d = ns(stack)
    else:
        pad = (-len(names)) % mesh.devices.size
        if pad:
            stack = jnp.concatenate([stack, jnp.zeros((pad,) + stack.shape[1:], stack.dtype)])
        stack = jax.lax.with_sharding_constraint(stack, NamedSharding(mesh, P()))  # the stack of the (replicated) inputs stays replicated: without this the compiler shards the concatenation itself and moves every layer between the devices
        d = _shard_map(ns, mesh, (P("data", None, None),), P("data", None, None))(stack)
        d = jax.lax.with_sharding_constraint(d, NamedSharding(mesh, P()))
    scale = 0.2 * math.sqrt(max(shape))
    return {n: params[n] * (1.0 - lr_muon * h.muon_wd) - lr_muon * scale * d[i].astype(F32) for i, n in enumerate(names)}, newm


def apply_update(c: Cfg, h: Hyper, params, grads, mstate, vstate, step_count, lr, lr_muon, muon_mesh=None):
    """One optimizer step on the gradients (clipped to a global norm); step_count is the 1-based step for the bias correction. Returns new params, m, v and the gradient norm.
    The Muon matrices of one shape (all the wq of the layers, all the w_up ...) are stacked and updated together: one batched Newton-Schulz instead of one per matrix (a much smaller program for the compiler).
    `muon_mesh` (default None = every device computes every matrix) splits the stacks over the devices of that mesh, see muon_stack_update. With h.muon_lean (--muon-lean) the update is made per layer and only the bf16
    Newton-Schulz input and output are stacked (muon_group_lean): the same numbers."""
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
        if h.muon_lean:
            p_new, m_new = muon_group_lean(h, shape, names, params, grads, mstate, clip, lr_muon, muon_mesh)
            for n in names:
                newp[n], newm[n], newv[n] = p_new[n], m_new[n], vstate[n]
            continue
        stack = lambda tree: jnp.stack([tree[n] for n in names])
        p_new, m_new = muon_stack_update(h, stack(params), stack(grads), stack(mstate), clip, lr_muon, muon_mesh)
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
        "ngram_bits": 0, "ngram_dim": 0, "window": c.window, "window_layers": list(c.window_layers),  # the last two keys as the Rust writer (absent in older files = window 0)
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


def save_checkpoint(path, c: Cfg, params, m, v, step, plan_steps=None, sync_every=0):
    """Writes header + params + m + v (flat little-endian f32 in layout order) to <path>.tmp and renames it: a kill during the write leaves the old file intact. One array at a time goes through host memory (the former
    concatenation of a whole tree held 3 copies of 4 N bytes next to each other: 12 N bytes of host RAM at the peak); the bytes are the same.
    sync_every > 0 (the background writer of --async-io): the data are forced to the disk every that many bytes and before the rename, so that never more than that much dirty data is waiting in the page cache (4.4 GB
    of dirty pages at once made the other writes of the 365M session on Kaggle stall for 4 to 6 minutes, see tools/session_breakdown.py); the file is the same."""
    n = sum(int(np.prod(s)) for _, s in layout(c))
    tmp = str(path) + ".tmp"
    with open(tmp, "wb") as f:
        f.write((header_of(c, n, step, plan_steps=plan_steps) + "\n").encode())
        written = synced = 0
        for tree in (params, m, v):
            for name, _ in layout(c):
                buf = np.ascontiguousarray(np.asarray(tree[name]), dtype="<f4").reshape(-1)
                f.write(buf)
                written += buf.nbytes
                if sync_every and written - synced >= sync_every:
                    f.flush()
                    os.fsync(f.fileno())
                    synced = written
        if sync_every:
            f.flush()
            os.fsync(f.fileno())
    os.replace(tmp, path)


def tree_finite(tree):
    """True when every value of every array of the tree is finite (one reduction per array on the device)."""
    return all(bool(jnp.all(jnp.isfinite(x))) for x in tree.values())


def out_space_needed(n_params, with_predecay):
    """Peak bytes that the checkpoints of a run occupy in its output folder: ckpt_latest.bin (12 bytes per parameter) while its replacement ckpt_latest.bin.tmp is written (a second 12 bytes), plus ckpt_predecay.bin
    (12 bytes) in a run with a time limit. Kaggle allows 20 GB in /kaggle/working: 36 N bytes, so at most 555 M parameters with the pre-decay file and 833 M without."""
    return 12 * n_params * (3 if with_predecay else 2)


def check_complete(path, with_state=True):
    """Raises ValueError when the checkpoint file is shorter than its header promises (a copy or a download that broke off)."""
    with open(path, "rb") as f:
        line = f.readline()
        n = json.loads(line)["n_params"]
    need = len(line) + (12 if with_state else 4) * n
    got = os.path.getsize(path)
    if got < need:
        raise ValueError(f"{path} is incomplete: {got} bytes, but its header promises {need} ({'parameters and both moments' if with_state else 'the parameters'}); it is a broken copy or a weights-only file")


def load_checkpoint(path, c: Cfg, with_state=True):
    check_complete(path, with_state)
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
               eps=h["eps"], untied=h["untied"], relu2=h["relu2"], softcap=h["softcap"], lambdas=h["lambdas"], skips=h["skips"], ve_layers=tuple(h["ve_layers"]),
               window=int(h.get("window", 0)), window_layers=tuple(int(x) for x in h.get("window_layers", [])))


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# data: uint16 token shards (bit 15 = "not a target" in chat data), random windows, a validation tail in the last shard
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


class Data:
    def __init__(self, directory, val_tokens=20_000_000, flags=True, cool_directory=None, cool_pattern="*.u16"):
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
        self.cool = CoolPool(cool_directory, flags, cool_pattern) if cool_directory else None  # the cool-down pool (see cool_share); the validation always stays the tail of the LAST shard of `directory`

    def _raw(self, h, length):
        """Window of `length` tokens at a position derived from h (a position that would straddle the end of a shard is taken from the start of that shard, as in the Rust loader)."""
        p = int(h % (self.total_train - length))
        s = int(np.searchsorted(self.cum, p, side="right") - 1)
        off = p - int(self.cum[s])
        if off + length > self.train_sizes[s]:
            off = 0
        return np.asarray(self.maps[s][off : off + length])

    def train_batch(self, rng, batch, seq, cool=0.0):
        """`batch` windows. With a cool-down pool a share `cool` of them comes from it (stochastic rounding of the count, so the mean share is exact); with cool = 0 the random stream is that of the
        trainer without a cool-down pool."""
        hs = rng.integers(0, 1 << 62, size=batch)
        n_cool = 0
        if self.cool is not None and cool > 0.0:
            x = min(cool, 1.0) * batch
            n_cool = int(x) + int(rng.random() < x - int(x))
        pick = set(rng.permutation(batch)[:n_cool].tolist()) if n_cool else set()
        return self._split(np.stack([(self.cool if i in pick else self)._raw(h, seq + 1) for i, h in enumerate(hs)]))

    def cool_val_batch(self, k, batch, seq):
        """Validation windows of the held-out cool-down file, spaced like those of val_batch (from the start of the file)."""
        held = self.cool.heldout
        span = max(len(held) - (seq + 1), 1)
        ps = [((k * batch + s) * 7919 * 4093) % span for s in range(batch)]
        return self._split(np.stack([np.asarray(held[p : p + seq + 1]) for p in ps]))

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


class CoolPool(Data):
    """The cool-down shards of one directory (the files that match `pattern`, default *.u16; a directory that also holds the main shards needs a pattern of its own, e.g. *.cool): every shard is training data
    (no validation tail); files whose name ends in _val (cool_val.u16) are never trained on. The held-out file is <directory>/heldout/cool_val.u16, <directory>/cool_val.u16 or <directory>/cool_heldout.val
    (optional): the trainer then logs `val_cool` next to `val`."""

    def __init__(self, directory, flags=True, pattern="*.u16", min_len=0):
        d = Path(directory)
        files = sorted(f for f in d.glob(pattern) if not f.stem.endswith("_val"))
        assert files, f"no cool-down shards ({pattern} files whose name does not end in _val) in {directory}"
        if min_len:  # a shard shorter than a window would make np.stack fail in the middle of the cool-down phase, hours into the run: such shards are left out
            short = [f for f in files if f.stat().st_size // 2 < min_len]
            for f in short:
                print(f"[train] WARNING: cool-down shard {f.name} has fewer than {min_len} tokens: not used", flush=True)
            files = [f for f in files if f not in short]
            assert files, f"no cool-down shard with at least {min_len} tokens in {directory}"
        self.files = files
        self.maps = [np.memmap(f, dtype="<u2", mode="r") for f in files]
        self.train_sizes = [len(m) for m in self.maps]
        self.val_start, self.val_tokens = 0, 0
        self.total_train = sum(self.train_sizes)
        self.flags = flags
        self.cum = np.cumsum([0] + self.train_sizes)
        self.cool = None
        held = [p for p in (d / "heldout" / "cool_val.u16", d / "cool_val.u16", d / "cool_heldout.val") if p.exists()]
        self.heldout = np.memmap(held[0], dtype="<u2", mode="r") if held else None


def validate_shards(data, seq, vocab, per_shard=4, span=4096):
    """Start-up check of the main shards (and the pool, when there is one) with a window of seq + 1 tokens: every training region must hold a whole window (a shorter one makes np.stack fail inside the run), and a few
    samples per shard must carry token ids below `vocab` (a gather with an id beyond the table silently returns the last row in JAX, and with flags a 15-bit alias would train on garbage). Raises ValueError. Returns
    (tokens sampled, the share of them with bit 15 set), the share being a loss mask in chat data and a sign of a wrong --vocab or tokenizer in pre-training data."""
    seen = flagged = 0
    for pool in (data, data.cool):
        if pool is None:
            continue
        for f, mp, size in zip(pool.files, pool.maps, pool.train_sizes):
            if size < seq + 1:
                raise ValueError(f"shard {Path(f).name} holds only {size} training tokens, fewer than a window of {seq + 1}")
            for k in range(per_shard):
                s = int((size - span) * k / max(per_shard - 1, 1)) if size > span else 0
                win = np.asarray(mp[s : min(s + span, size)]).astype(np.int32)
                ids = win & 0x7FFF if pool.flags else win
                if int(ids.max()) >= vocab:
                    raise ValueError(f"shard {Path(f).name} holds the token id {int(ids.max())} at about token {s}, but --vocab is {vocab}: the shards were made with another tokenizer or a larger vocabulary")
                seen += len(win)
                flagged += int(((win & 0x8000) != 0).sum()) if pool.flags else 0
    return seen, (flagged / seen if seen else 0.0)


def open_cool_pool(spec, pattern="*.u16", flags=True, min_len=0):
    """The cool-down pool of --cool-data ('@code' = the folder of this script, where a Kaggle session finds the shards of the forge-jax dataset), or None. The pool is an extra: a missing or unreadable one is
    reported and the run goes on without it (an unattended session is worth more than the cool-down)."""
    if not spec:
        return None
    try:
        return CoolPool(Path(__file__).resolve().parent if spec == "@code" else spec, flags, pattern, min_len)
    except Exception as e:  # noqa: BLE001
        print(f"[train] WARNING: no cool-down pool ({type(e).__name__}: {str(e).strip()[:200]}): training on --data only", flush=True)
        return None


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


def build_train_step(c: Cfg, h: Hyper, cdt, ce_chunk, remat, muon_mesh=None, remat_policy="dots", fuse_qkv=False, attn_prep="plain", layer_opts=""):
    """One optimizer step over `tokens` [accum, B, T]: the gradients of the micro-batches are averaged (a scan, one micro-batch at a time in memory), then the update. `muon_mesh`: see apply_update (--muon-shard).
    `remat_policy`: see remat_policy_of (--remat-policy). `fuse_qkv`: see layer_fn (--fuse-qkv). `attn_prep`: see layer_fn (--attn-prep). `layer_opts`: see LAYER_OPTS (--layer-opts)."""

    def step_fn(params, m, v, step_count, tokens, targets, ignore, lr, lr_muon):
        accum = tokens.shape[0]

        def micro_grad(t_, y_, i_):
            return jax.value_and_grad(lambda p: loss_fn(c, p, t_, y_, i_, cdt, ce_chunk, remat, remat_policy, fuse_qkv, attn_prep, layer_opts))(params)

        if accum == 1:
            # one micro-step: its gradient IS the average. No zero-initialised accumulator (a full-size array of zeros per parameter, 0 + g == g), no scan over one element and no division by 1 (g / 1 == g): the same
            # numbers (a -0.0 in a gradient used to come out as +0.0, which compares equal and changes nothing downstream), fewer passes over the arrays
            loss, grads = micro_grad(tokens[0], targets[0], ignore[0])
        else:
            def micro(carry, xs):
                gsum, lsum = carry
                loss, g = micro_grad(*xs)
                return (jax.tree_util.tree_map(jnp.add, gsum, g), lsum + loss), None

            zeros = jax.tree_util.tree_map(jnp.zeros_like, params)
            (grads, lsum), _ = jax.lax.scan(micro, (zeros, jnp.zeros((), F32)), (tokens, targets, ignore))
            grads = jax.tree_util.tree_map(lambda g: g / accum, grads)
            loss = lsum / accum
        step_count = step_count + 1
        params, m, v, gnorm = apply_update(c, h, params, grads, m, v, step_count, lr, lr_muon, muon_mesh)
        return params, m, v, step_count, loss, gnorm

    return jax.jit(step_fn, donate_argnums=(0, 1, 2, 3))


def build_eval(c: Cfg, cdt, ce_chunk):
    return jax.jit(lambda params, tokens, targets, ignore: loss_fn(c, params, tokens, targets, ignore, cdt, ce_chunk, False))


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# the host side of the training loop: a kernel pick without the self-test, the two learning-rate scalars, the background writers (--async-io)
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


def parse_attn_pick(spec):
    """'512,1' -> (512, True): the block size of the splash kernel and whether it uses the fused backward kernel ('512,0' = separate dq and dkv kernels). ValueError for anything else."""
    try:
        blk, fused = (int(x) for x in spec.split(","))
    except ValueError:
        raise ValueError(f"--attn-pick wants BLOCK,FUSED such as 512,1 (got {spec!r})") from None
    if blk < 128 or blk % 128 or fused not in (0, 1):
        raise ValueError(f"--attn-pick wants a block size that is a multiple of 128 and FUSED 0 or 1 (got {spec!r})")
    return blk, bool(fused)


def attention_from_pick(c: Cfg, mesh, spec, log=print, **names):
    """The splash kernel of --attn-pick without the self-test of select_attention (which compiles XLA's attention and every candidate kernel: 128 s of the 365M session on a v5e-8, 74 s of them for the four kernels).
    The pick comes from a benchmark of the same session (tpu/bench_step.py, `attn_pick` of its base row), which has run that selection on this TPU. Returns the attention (make_splash_attention; `names` = splash_names of the
    remat policy) or None when the pick cannot be built: the caller then selects normally. Whether the kernel works inside the whole step is shown by the first step, whose failure falls back to XLA's attention as always."""
    try:
        blk, fused = parse_attn_pick(spec)
        impl = make_splash_attention(c, mesh, blk, fused, **names)
    except Exception as e:  # noqa: BLE001 (an optimisation: whatever goes wrong, select as usual)
        log(f"[attn] --attn-pick {spec} cannot be used ({type(e).__name__}: {str(e).strip()[:200]}): the normal selection")
        return None
    log(f"[attn] --attn-pick {spec}: splash block {blk}{' fused backward' if fused else ''} without the self-test")
    return impl


class DeviceScalars:
    """The float32 scalars that the training step takes (the two learning rates) as arrays replicated over the mesh. jnp.float32(x) built two new arrays on device 0 at every step, which the step then had to copy to the
    other devices; here the same device array is handed over again for as long as the value stays the same (the constant part of the schedule, 92 percent of the steps of the 365M session: no transfer at all) and a new
    one is made, already in its final sharding, when it changes. The value is rounded to float32 as jnp.float32 does it, and the compiled program is the same (tests: test_async_loop.py)."""

    def __init__(self, sharding):
        self.sharding = sharding
        self.cache = {}

    def get(self, key, x):
        f = np.float32(x)
        hit = self.cache.get(key)
        if hit is None or hit[0] != f:
            hit = (f, jax.device_put(f, self.sharding))
            self.cache[key] = hit
        return hit[1]


def host_copy(x):
    """A numpy array with the values of the device array x that owns its memory. On the CPU backend np.asarray gives a VIEW of the device buffer, which the next training step donates (overwrites): a snapshot that is
    written later needs a real copy; where np.asarray already made a copy (a TPU), that one is used as it is."""
    h = np.asarray(x)
    return h if (h.flags.owndata and h.base is None) else np.array(h)


class AsyncLog:
    """log.jsonl written by a background thread (--async-io): write() only queues the line, so that a disk that blocks for minutes holds the thread and not the training loop. Same interface as the file that it replaces."""

    def __init__(self, path):
        self.f = open(path, "a")
        self.q = Queue()
        self.failed = False
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        delay = float(os.environ.get("FORGE_TEST_SLOW_LOG", "0") or 0)  # tests only: a disk that is slow
        while True:
            text = self.q.get()
            if text is None:
                return
            try:
                if delay > 0:
                    time.sleep(delay)
                self.f.write(text)
                self.f.flush()
            except OSError as e:
                if not self.failed:
                    self.failed = True
                    print(f"[train] WARNING: writing log.jsonl failed ({type(e).__name__}: {str(e)[:200]}): its lines are dropped, the training goes on", flush=True)

    def write(self, text):
        self.q.put(text)

    def flush(self):
        pass

    def close(self, timeout=600.0):
        """Waits (at most `timeout` seconds) until every queued line is written, then closes the file."""
        self.q.put(None)
        self.thread.join(timeout)
        if not self.thread.is_alive():
            self.f.close()


class CheckpointWriter:
    """--async-io: the periodic checkpoints are written by a background thread. The training loop copies the state to host memory (host_copy: the next step donates the device buffers) and goes on; the thread writes
    <path>.tmp, forces it to the disk every SYNC_BYTES and renames it (save_checkpoint with sync_every), so that a half-written file never replaces a good one, as before. One job at a time."""

    SYNC_BYTES = 256 << 20
    WAIT_LIMIT_S = 1800.0  # the longest wait for a running job

    def __init__(self):
        self.thread = None
        self.lock = threading.Lock()
        self.done = None  # (label, step, status, seconds, message) of the last finished job, until collect() takes it

    def busy(self):
        return self.thread is not None and self.thread.is_alive()

    def start(self, path, c, trees, step, plan_steps, label):
        """Writes the (params, m, v) host copies `trees` as the checkpoint of `step` in a new thread."""
        assert not self.busy(), "one checkpoint write at a time"

        def job():
            t0, status, msg = time.time(), "ok", ""
            try:
                delay = float(os.environ.get("FORGE_TEST_SLOW_WRITE", "0") or 0)  # tests only: a disk that is slow
                if delay > 0:
                    time.sleep(delay)
                save_checkpoint(path, c, *trees, step, plan_steps=plan_steps, sync_every=self.SYNC_BYTES)
            except BaseException as e:  # noqa: BLE001 (the old file stays; the loop reports it)
                status, msg = "failed", f"{type(e).__name__}: {str(e)[:200]}"
                try:
                    os.remove(str(path) + ".tmp")
                except OSError:
                    pass
            with self.lock:
                self.done = (label, step, status, time.time() - t0, msg)

        self.thread = threading.Thread(target=job, daemon=True)
        self.thread.start()

    def collect(self):
        """The result of the last finished job, once (None while there is none): (label, step, 'ok' | 'failed', seconds, message)."""
        with self.lock:
            r, self.done = self.done, None
        return r

    def wait(self, log=print, every=60.0, limit=None):
        """Waits for the running job (at most `limit` seconds, default WAIT_LIMIT_S); a line every `every` seconds keeps the runner's silence timer from taking a slow disk for a hung trainer."""
        t0, limit = time.time(), self.WAIT_LIMIT_S if limit is None else limit
        while self.busy() and time.time() - t0 < limit:
            self.thread.join(timeout=every)
            if self.busy():
                log(f"[train] waiting for the checkpoint write of the background thread ({time.time() - t0:.0f} s so far)")


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
    ap.add_argument("--window", type=int, default=0, help="sliding-window attention (as `forge train --window`): in the layers of --window-layers query i sees only the keys j with i - window < j <= i; 0 = off (every layer dense causal)")
    ap.add_argument("--window-layers", default="auto", help="auto = three of every four layers windowed, every fourth layer (3, 7, 11, ...) and the last layer dense (the rule of the Rust trainer); or layer numbers separated by commas; only used with --window")
    ap.add_argument("--lr-emb-mult", type=float, default=1.0)
    ap.add_argument("--lr-head-mult", type=float, default=1.0)
    ap.add_argument("--lr-table-mult", type=float, default=1.0)
    ap.add_argument("--lr-scalar-mult", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--eval-batches", type=int, default=8)
    ap.add_argument("--val-tokens", type=int, default=20_000_000)
    ap.add_argument("--cool-data", default="", help="directory with the cool-down shards (cool_NNN.u16; an optional heldout/cool_val.u16 adds a `val_cool` line): from the start of the learning-rate decay a growing share of every batch comes from here, the rest from --data (whose last shard still holds the validation tail)")
    ap.add_argument("--cool-glob", default="*.u16", help="file pattern of the cool-down shards in --cool-data (use *.cool when the directory also holds main shards; '@code' as --cool-data = the folder of this script)")
    ap.add_argument("--cool-blend-steps", type=int, default=0, help="steps of the linear ramp of the cool-down share from 0 to --cool-share (0 = a hard switch)")
    ap.add_argument("--cool-blend-frac", type=float, default=0.0, help="the ramp as a fraction of the steps from the first cool-down step to the end of the run (overrides --cool-blend-steps; follows a time-planned number of steps)")
    ap.add_argument("--cool-share", type=float, default=0.6, help="the share of the cool-down data in a batch once the ramp is over")
    ap.add_argument("--cool-start", type=int, default=-1, help="first step with cool-down data (default: the first step of the learning-rate decay, int((1 - decay_frac) * steps))")
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
    ap.add_argument("--stall-trace-s", type=float, default=45.0, help="when no step, evaluation or checkpoint completes for this many seconds, print the stack of every thread once (where the loop is waiting: a blocked write, the device, the reader); 0 = off. Armed by the first step (the start-up has minutes without a line), "
                    "works with the watchdog thread (--watchdog-s > 0)")
    ap.add_argument("--no-pin", action="store_true", help="on Kaggle TPU VMs do not install the tested jax version (see _pin_jax_for_tpu)")
    ap.add_argument("--plan-reserve-s", type=float, default=-1.0, help="seconds of the time limit kept free when the steps are planned: the last evaluation and checkpoint, and the stalls that come after the plan is fixed at the start of the decay "
                    "(default: 600, but at most 10 percent of the time limit; the 365M run on Kaggle lost the end of its decay to a 90 s reserve)")
    ap.add_argument("--no-preflight", action="store_true", help="skip the start-up checks of the data (window length, token ids below the vocabulary) and of the disk space for the checkpoints")
    ap.add_argument("--async-steps", type=int, default=0, metavar="N", help="read the loss of a step N steps late: step n+1 .. n+N are dispatched before the host waits for the loss of step n, so that the device never waits for the host's bookkeeping "
                    "(formatting, json, the finite check) and the dispatch of the next step happens while the previous one runs. 0 (default) = the loop of every earlier run. The numbers, the printed lines and the checkpoints are the same; "
                    "before an evaluation, a checkpoint, the end of the plan and the time limit the pipeline is emptied, and the first step is always read at once (a failing kernel is caught there). A NaN or infinite loss "
                    "is found N steps later (the run stops with exit code 98 and writes nothing, as before). The 'seconds' of a printed step are the time between the reads of two steps. At most 16")
    ap.add_argument("--async-io", action="store_true", help="never let the training loop wait for the disk: log.jsonl is written by a background thread, and the periodic checkpoints (not the last one) are copied to host memory and "
                    "written by a background thread that forces the data to the disk every 256 MB (so that no 4.4 GB of dirty pages build up) before it renames the file. If the previous write is still running when the next periodic "
                    "checkpoint is due, that one is skipped (the pre-decay and the final checkpoint wait). The files are the same")
    ap.add_argument("--eval-early-until", type=int, default=0, metavar="STEP", help="until this step evaluate every 100 steps whatever --eval-every says (the early evaluations catch a broken run), later only every --eval-every steps; 0 = off")
    ap.add_argument("--attn", choices=["auto", "xla", "splash"], default="auto", help="attention implementation: auto = on a TPU the splash kernel when it is checked and faster, otherwise XLA's")
    ap.add_argument("--attn-pick", default="", metavar="BLOCK,FUSED", help="on a TPU: use this splash kernel (e.g. 512,1 = block 512 with the fused backward, 512,0 = separate backward kernels) without the self-test of --attn auto, which compiles XLA's attention and every "
                    "candidate kernel (128 s on a v5e-8). The runner takes the pick from the `attn_pick` of its autotune benchmark, which ran that selection on the same TPU a few minutes earlier. A pick that cannot be built falls back to "
                    "the normal selection, a kernel that fails in the first step to XLA's attention. Ignored with --attn xla and off a TPU")
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--ce-chunk", type=int, default=256, help="positions per chunk of the cross entropy")
    ap.add_argument("--no-remat", action="store_true")
    ap.add_argument("--fuse-qkv", action="store_true", help="compute the q, k and v projections of a layer as ONE matmul on the concatenation [wq; wk; wv] (a wider matmul for the MXU; the parameters, the checkpoint layout and the numbers stay "
                    "the same, the concatenation is made inside the layer); a speed option (default off)")
    ap.add_argument("--attn-prep", choices=list(ATTN_PREPS), default="plain", help="how q, k and v are prepared for the attention: plain (default) = the qk-norm, RoPE and the value-embedding mix on arrays [batch, T, heads, 64] (the program "
                    "of every earlier run); lane = the same function on arrays [batch, heads, 64, T] (T, a multiple of 128, on the lanes of the TPU), scaled and cast to bf16 and transposed into the splash kernel's [batch, heads, T, 64] "
                    "once, by a matmul with the identity matrix (a last dimension of 64 is padded to 128 lanes on a TPU: twice the memory traffic for every pass over an array of that shape, four times for the halves of RoPE). "
                    "A speed option for the TPU; with XLA's attention it changes nothing but the order of the operations")
    ap.add_argument("--layer-opts", type=layer_opts_arg, default="", metavar="OPT[,OPT...]", help="named formulations of the layer program that compute the same function in a form the TPU compiler likes better (default none = the program of "
                    "every earlier run; found with tpu/hlo_cost.py --matmuls in the compile lab). Options: " + "; ".join(f"{k} = {v}" for k, v in LAYER_OPTS.items()))
    ap.add_argument("--remat-policy", type=remat_policy_arg, default="dots", metavar="POLICY", help="what a layer's backward pass saves instead of recomputing: dots (default) = the matmul outputs only, everything else is recomputed, the splash-attention forward "
                    "included (about 0.8 ms per layer on a v5e); dots+attn = also the output and the logsumexp of the splash kernel (the same numbers, the forward kernel is not run twice). Without the splash kernel (XLA attention) the attn and "
                    "qkv parts are the same program as dots. More parts, joined by + in any combination: qkv = the prepared bf16 q, k, v that feed the splash kernel, so that the qk-norm / RoPE / scale / cast / transpose chain is not "
                    "recomputed in the backward pass (the big one in the compile lab's estimate of a v5e-8 step: -19 %); acts = the bf16 operands of the layer's matmuls, the outputs of attn_norm and ffn_norm and the relu-squared "
                    "activation (h, h2, act can also be named one by one; no gain in the lab: h and h2 cost more than they save). Every part is the value that the recomputation would make: the same numbers, more HBM, less recomputation. "
                    "Extra HBM per chip at micro 4 for 24 layers (compile lab): attn 0.8 GiB, qkv 0.9 GiB, acts 1.8 GiB (the 64-wide minor dimension of q, k, v and of the attention output is padded to 128 lanes). norm / qknorm = the mean squares "
                    "of the residual-stream norms / of the q and k norms (a few hundred KB per layer; the backward pass then makes no extra pass over x, q and k for them; qknorm: within the lab's noise (about +-2 ms), norm: +8 to +12 ms, "
                    "the fusions of the backward pass change). Choices: " + ", ".join(REMAT_POLICIES))
    ap.add_argument("--remat-save-layers", type=int, default=0, metavar="N", help="with a --remat-policy that saves extras (attn, qkv, acts): only the LAST N layers carry them (0 = all layers; the others use dots), to fit a memory budget. "
                    "The memory grows in proportion to N, the gain in the compiler's estimate more slowly (half of the layers gave about a third of the gain): use all layers when they fit")
    ap.add_argument("--muon-shard", action="store_true", help="split the stacked Muon update over the devices: each device orthogonalises its share of the matrices (the Newton-Schulz iterations are otherwise computed redundantly on every device) and the "
                    "results are gathered; the same math per matrix, a speed option (default off)")
    ap.add_argument("--muon-lean", action="store_true", help="the Muon update per layer instead of on fp32 stacks of the layers (muon_group_lean): only the bf16 input and output of the Newton-Schulz iterations are stacked, the "
                    "momentum, normalisation and weight update are elementwise per layer; the same numbers, fewer passes and copies; with --muon-shard the iterations are split with a shard_map (default off)")
    ap.add_argument("--ns-barrier", action="store_true", help="an optimization barrier after every Newton-Schulz iteration: the same numbers; XLA then does not fold the elementwise chain of the earlier iterations into the last matmul (default off)")
    a = ap.parse_args(argv)
    if not 0 <= a.async_steps <= 16:
        ap.error("--async-steps: between 0 (off) and 16 (the learning-rate plan may move up to 25 steps ahead of the decay: a longer pipeline could dispatch steps with a stale plan)")
    if a.attn_pick:
        try:
            parse_attn_pick(a.attn_pick)
        except ValueError as e:
            ap.error(str(e))
    if a.remat_save_layers < 0:
        ap.error("--remat-save-layers: must be 0 (every layer) or the number of layers that carry the extras")
    if a.remat_save_layers > 0 and parse_remat(a.remat_policy)[0]:
        if "@" in a.remat_policy:
            ap.error("--remat-save-layers and an @N in --remat-policy name the number of layers twice")
        a.remat_policy += f"@{a.remat_save_layers}"  # the layer count travels inside the policy spec (parse_remat): every caller that hands a.remat_policy on (training, benchmark, compile lab) gets it
    return a


def main(argv=None):
    a = parse_args(argv if argv is not None else sys.argv[1:])
    steps_max = a.steps  # the command line's number of steps is the upper bound of the plan
    auto_steps = a.time_limit_s > 0 and not a.no_auto_steps
    try:
        window_layers = parse_window_layers(a.window_layers, a.window, a.layers)
    except ValueError as e:
        print(f"[train] ERROR: {e}", flush=True)
        return 2
    c = Cfg(n_layer=a.layers, n_embd=a.dim, n_head=a.heads, n_kv=a.kv or a.heads, hd=a.head_dim, n_ff=a.ffn, vocab=a.vocab, seq=a.seq, qk_norm=a.qk_norm, theta=a.rope_theta, untied=a.untied, relu2=a.relu2,
            softcap=a.softcap, lambdas=a.lambdas, skips=a.skips, ve_layers=tuple(int(x) for x in a.value_emb.split(",") if x), window=a.window, window_layers=window_layers)
    muon_lr = a.muon_lr if a.muon_lr is not None else a.lr
    h = Hyper(peak_lr=a.lr, muon_lr=muon_lr, warmup=a.warmup, min_frac=a.min_lr_frac, decay_frac=a.decay_frac, wd=a.wd, muon_wd=a.muon_wd, mu=a.muon_mu, beta2=a.beta2, clip=a.clip, mult_emb=a.lr_emb_mult,
              mult_head=a.lr_head_mult, mult_table=a.lr_table_mult, mult_scalar=a.lr_scalar_mult, use_muon=(a.optim == "muon"), muon_lean=a.muon_lean, ns_barrier=a.ns_barrier)
    cdt = BF16 if a.dtype == "bf16" else F32
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    mesh, batch_sh, eval_sh, repl_sh = make_mesh()
    ndev = len(jax.devices())
    muon_mesh = mesh if (a.muon_shard and h.use_muon) else None
    if muon_mesh is not None:
        print(f"[train] Muon update sharded over {ndev} devices (--muon-shard)", flush=True)
    if (a.muon_lean or a.ns_barrier) and h.use_muon:
        print(f"[train] Muon update options:{' per-layer update, only the bf16 Newton-Schulz input and output stacked (--muon-lean)' if a.muon_lean else ''}{' barrier after every Newton-Schulz iteration (--ns-barrier)' if a.ns_barrier else ''}", flush=True)
    if a.remat_policy != "dots":
        print(f"[train] remat policy {a.remat_policy}" + (" (ignored: --no-remat)" if a.no_remat else ""), flush=True)
    if a.fuse_qkv:
        print("[train] q, k and v projections fused into one matmul (--fuse-qkv)", flush=True)
    if a.attn_prep != "plain":
        print(f"[train] attention input preparation: {a.attn_prep} (--attn-prep)", flush=True)
    if a.layer_opts:
        print(f"[train] layer program options: {a.layer_opts} (--layer-opts)", flush=True)
    progress = {"t": time.time(), "what": "start-up", "armed": False}  # "armed": the stall trace waits for the first step (the start-up has minutes without a line: the data, the kernel selection, the compilation)

    def watchdog():
        traced = None
        while True:
            time.sleep(min(30.0, max(1.0, a.watchdog_s / 4), max(1.0, (a.stall_trace_s or 90.0) / 3)))
            idle = time.time() - progress["t"]
            if a.stall_trace_s > 0 and progress["armed"] and idle > a.stall_trace_s and traced != progress["t"]:
                # once per stall: where every thread is waiting (the 365M session lost 5.8 percent to five stalls of 4 to 6 minutes that its log could not place)
                traced = progress["t"]
                print(f"[train] STALL: nothing completed for {idle:.0f} s (last: {progress['what']}); the stack of every thread follows", flush=True)
                try:
                    faulthandler.dump_traceback(file=sys.stdout, all_threads=True)
                except Exception:  # noqa: BLE001 (a stdout without a file descriptor)
                    pass
            if idle > a.watchdog_s:
                print(f"[train] watchdog: nothing completed for {idle:.0f} s (last: {progress['what']}): exiting with code 97 so that a hung device or reader does not hold the session", flush=True)
                os._exit(97)

    if a.watchdog_s > 0:
        threading.Thread(target=watchdog, daemon=True).start()
    data = Data(a.data, a.val_tokens, flags=a.vocab <= 32768)
    data.cool = open_cool_pool(a.cool_data, a.cool_glob, flags=a.vocab <= 32768, min_len=a.seq + 1)
    if not a.no_preflight:
        try:
            seen, flagged = validate_shards(data, a.seq, a.vocab)
        except ValueError as e:
            print(f"[train] ERROR: the data do not fit this run: {e} (--no-preflight skips this check)", flush=True)
            return 2
        print(f"[train] data check: {seen / 1e6:.1f} M sampled tokens, all ids below the vocabulary of {a.vocab}; {100 * flagged:.1f} % of them carry bit 15 (the loss mask of chat data: about 0 in pre-training data)", flush=True)
    if not a.no_warm_data:
        threading.Thread(target=warm_files, args=(data.files + (data.cool.files if data.cool is not None else []),), daemon=True).start()
    seq = a.seq
    # the global batch of one optimizer step: tokens_per_step / seq sequences, a multiple of devices x micro; the micro-steps accumulate gradients
    micro = max(1, a.micro)
    per_micro = ndev * micro
    seqs_per_step = max(a.tokens_per_step // seq, per_micro)
    seqs_per_step = (seqs_per_step + per_micro - 1) // per_micro * per_micro
    accum = seqs_per_step // per_micro
    tokens_per_step = seqs_per_step * seq
    print(f"[train] {jax.default_backend()} x {ndev}: {seqs_per_step} sequences of {seq} tokens per step ({tokens_per_step} tokens): {accum} micro-steps of {per_micro} sequences ({micro} per device); data {data.total_train / 1e9:.2f} G training tokens, {data.val_tokens / 1e6:.1f} M validation tokens", flush=True)
    if tokens_per_step != a.tokens_per_step:
        print(f"[train] WARNING: {a.tokens_per_step} tokens per step were asked for, but {ndev} devices x {micro} sequences per device x {seq} tokens only allow multiples of {per_micro * seq}: this run trains with {tokens_per_step} tokens per step "
              f"(another --micro changes that: a different micro-batch after an out-of-memory retry would change the batch size, the number of tokens seen and the best learning rate)", flush=True)
    n_params = sum(int(np.prod(s)) for _, s in layout(c))
    if not a.no_preflight:
        need = out_space_needed(n_params, a.time_limit_s > 0)
        quota = float(os.environ.get("FORGE_OUT_QUOTA_GB", "20" if str(out.resolve()).replace("\\", "/").startswith("/kaggle/working") else "0")) * 1e9  # Kaggle allows 20 GB in /kaggle/working
        have = quota if quota > 0 else shutil.disk_usage(out).free + sum(p.stat().st_size for p in out.glob("ckpt_*") if p.is_file())
        print(f"[train] disk: the checkpoints need up to {need / 1e9:.1f} GB in {out} ({'quota' if quota > 0 else 'free space'} {have / 1e9:.1f} GB)", flush=True)
        if need > have:
            print(f"[train] ERROR: {n_params / 1e6:.0f} M parameters need up to {need / 1e9:.1f} GB for ckpt_latest.bin (12 bytes per parameter), its .tmp copy while it is written and ckpt_predecay.bin, but only {have / 1e9:.1f} GB are "
                  f"available: the first checkpoint write would fail hours into the run (--no-preflight skips this check)", flush=True)
            return 2
    for stale in out.glob("ckpt_*.tmp"):  # the half-written file of a killed earlier session
        try:
            stale.unlink()
        except OSError:
            pass
    print(f"[train] model {a.layers} layers, dim {a.dim}, {a.heads}/{c.n_kv} heads, ffn {a.ffn}, vocab {a.vocab}: {n_params / 1e6:.1f} M parameters, optimizer {a.optim}, structure relu2={a.relu2} lambdas={a.lambdas} skips={a.skips} ve={list(c.ve_layers)}", flush=True)
    if any(c.window_of(l) for l in range(c.n_layer)):
        print(f"[train] sliding-window attention: window {c.window} in layers {list(c.window_layers)} (query i sees the keys i - {c.window} < j <= i), the other layers are dense causal", flush=True)
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
    if a.attn_pick and (a.attn == "xla" or jax.default_backend() != "tpu"):
        print(f"[attn] --attn-pick {a.attn_pick} ignored ({'--attn xla' if a.attn == 'xla' else 'not a TPU'})", flush=True)
    if a.attn != "xla" and jax.default_backend() == "tpu":
        picked = attention_from_pick(c, mesh, a.attn_pick, **splash_names(a.remat_policy)) if a.attn_pick else None
        if picked is not None:
            ATTN["impl"] = picked
        else:
            try:
                ATTN["impl"] = select_attention(c, mesh, cdt, per_micro, force=(a.attn == "splash"), **splash_names(a.remat_policy))
            except Exception as e:  # noqa: BLE001 (the selection is an optimisation: whatever goes wrong, train with XLA's attention)
                print(f"[attn] selection failed ({type(e).__name__}: {str(e).strip()[:200]}): XLA attention", flush=True)
                ATTN["impl"] = None
    if a.attn_test_fail:
        def failing_attention(c_, q, k, v_, cdt_, local=False, mode=a.attn_test_fail, ndev_=ndev):
            if mode == "step" or q.shape[0] == ndev_:
                raise RuntimeError("test failure of the attention kernel")
            return attention_xla(c_, q, k, v_, cdt_, c_.window if local else None)

        ATTN["impl"] = failing_attention
    print(f"[train] attention: {'splash kernel' if ATTN['impl'] else 'XLA ops'}", flush=True)
    def share_at(step):
        """The cool-down share of the batch of `step` under the current plan (a.steps moves while a time plan is being made)."""
        first = int((1.0 - h.decay_frac) * a.steps) if a.cool_start < 0 else a.cool_start
        return cool_share(step, a.steps, h.decay_frac, cool_blend_len(a.steps, first, a.cool_blend_steps, a.cool_blend_frac), a.cool_share, a.cool_start)

    if data.cool is not None:
        first = int((1.0 - h.decay_frac) * a.steps) if a.cool_start < 0 else a.cool_start
        blend = cool_blend_len(a.steps, first, a.cool_blend_steps, a.cool_blend_frac)
        ramp = f"a hard switch to {a.cool_share:.2f}" if blend <= 0 else f"a linear ramp over {blend} steps up to {a.cool_share:.2f}"
        print(f"[train] cool-down data: {len(data.cool.files)} shards, {data.cool.total_train / 1e6:.0f} M tokens, held-out file: {'yes' if data.cool.heldout is not None else 'no'}; cool-down share 0 before step {first}"
              f"{' (moves with a time-planned number of steps)' if a.cool_start < 0 and a.time_limit_s > 0 else ''}, then {ramp}", flush=True)
    progress.update(t=time.time(), what="attention selection")
    train_step = build_train_step(c, h, cdt, a.ce_chunk, not a.no_remat, muon_mesh, a.remat_policy, a.fuse_qkv, a.attn_prep, a.layer_opts)
    evaluate = build_eval(c, cdt, a.ce_chunk)
    rng = np.random.default_rng(a.seed * 1000003 + start)
    log = AsyncLog(out / "log.jsonl") if a.async_io else open(out / "log.jsonl", "a")  # --async-io: a background thread writes the lines
    ckpt_job = CheckpointWriter() if a.async_io else None
    produced = {"n": 0}

    def safe_save(path, at_step):
        """Writes a checkpoint of the current state: 'ok', 'nonfinite' (some parameter or moment is NaN or infinite: nothing is written, the older checkpoint stays the last good one) or 'failed' (the disk refused: the half
        file is removed and the run goes on; the next save tries again)."""
        if not (tree_finite(params) and tree_finite(m) and tree_finite(v)):
            print(f"[train] ERROR: the parameters or moments are not finite at step {at_step}: {Path(path).name} NOT written (the older checkpoint stays the last good one); exit code 98", flush=True)
            return "nonfinite"
        try:
            save_checkpoint(path, c, params, m, v, at_step, plan_steps=a.steps)
            return "ok"
        except OSError as e:
            print(f"[train] ERROR: writing {Path(path).name} failed at step {at_step} ({type(e).__name__}: {str(e)[:200]}): the old file stays, training goes on", flush=True)
            try:
                os.remove(str(path) + ".tmp")
            except OSError:
                pass
            return "failed"

    def report_job():
        """Says how a background checkpoint write ended (once, when it has finished) and returns its result (label, step, 'ok' | 'failed', seconds, message) or None."""
        r = ckpt_job.collect() if ckpt_job is not None else None
        if r is not None:
            label, st, status, secs, msg = r
            if status == "ok":
                print(f"[train] checkpoint at step {st} written by the background thread ({label}, {secs:.0f} s, {time.time() - t_start:.0f} s elapsed)", flush=True)
            else:
                print(f"[train] ERROR: writing {label} failed at step {st} ({msg}): the old file stays, training goes on", flush=True)
            progress.update(t=time.time(), what="checkpoint written")
        return r

    def save_state(path, at_step, background=False, skip_if_busy=False):
        """The checkpoint of the current state: 'ok' (written, safe_save), 'started' (--async-io: the state is copied to host memory and a background thread writes the file), 'skipped' (skip_if_busy and the thread is still
        writing the previous checkpoint), 'nonfinite' or 'failed' (see safe_save). A write in the foreground waits for the background thread first: one file at a time."""
        def wait_for_thread():  # False when the thread is still writing after the longest wait: a second writer must never start (two writers on one .tmp file could rename garbage over a good checkpoint)
            if ckpt_job is not None and ckpt_job.busy():
                ckpt_job.wait(lambda s: print(s, flush=True))
                report_job()
                if ckpt_job.busy():
                    print(f"[train] ERROR: the background checkpoint write is still running after {ckpt_job.WAIT_LIMIT_S:.0f} s: {Path(path).name} NOT written, no second writer is started", flush=True)
                    return False
            return True

        if not (background and ckpt_job is not None):
            return safe_save(path, at_step) if wait_for_thread() else "failed"
        if skip_if_busy and ckpt_job.busy():
            return "skipped"
        if not (tree_finite(params) and tree_finite(m) and tree_finite(v)):
            print(f"[train] ERROR: the parameters or moments are not finite at step {at_step}: {Path(path).name} NOT written (the older checkpoint stays the last good one); exit code 98", flush=True)
            return "nonfinite"
        if not wait_for_thread():
            return "failed"
        trees = tuple({n: host_copy(x) for n, x in tree.items()} for tree in (params, m, v))  # the next step donates the device buffers: the thread writes this copy
        ckpt_job.start(path, c, trees, at_step, a.steps, Path(path).name)
        return "started"

    def next_batch():
        produced["n"] += 1
        if produced["n"] == int(os.environ.get("FORGE_TEST_PRODUCER_FAIL", "0") or 0):
            raise OSError("test failure of the batch reader")  # tests only
        # the share of the cool-down pool for the step this batch belongs to (the producer runs a few steps ahead of the training loop: the batch index is start + the number of batches made so far)
        share = share_at(start + produced["n"] - 1) if data.cool is not None else 0.0
        t, y, ig = data.train_batch(rng, seqs_per_step, seq, share)
        return tuple(jax.device_put(x.reshape(accum, per_micro, seq), batch_sh) for x in (t, y, ig))

    # a background thread prepares the next batch while the devices work on the current one
    q = Queue(maxsize=3)
    producer_stop = threading.Event()

    def put_batch(item):
        while not producer_stop.is_set():  # gives up when the run is over: nobody takes batches any more
            try:
                q.put(item, timeout=0.2)
                return
            except Full:
                pass

    def producer():
        try:
            while not producer_stop.is_set():
                put_batch(next_batch())
        except BaseException as e:  # noqa: BLE001 (hand the error to the training loop: a dead producer must not leave it waiting forever)
            put_batch(e)

    # not left to the interpreter: a daemon thread that is inside jax.device_put when the process ends makes the exit abort ("terminate called ... exception not rethrown"), see the end of main
    producer_thread = threading.Thread(target=producer, daemon=True)
    producer_thread.start()
    t_start = time.time()
    # with a time limit the total number of steps follows the measured speed (the schedule must reach its decay phase before the limit): re-planned every 100 steps until the decay starts
    plain_dts, eval_dts, save_dts = [], [], []
    plan_reserve = a.plan_reserve_s if a.plan_reserve_s >= 0 else min(600.0, 0.1 * a.time_limit_s)
    t_ref, step_ref = None, 0  # the wall clock after the first steps (the compilation is over): the mean wall time per step since then contains every stall (see plan_steps)
    scalars = DeviceScalars(repl_sh)  # the two learning rates of a step as replicated device scalars, made again only when the value changes
    # The pipeline of --async-steps: `sent` steps are dispatched, `step` of them are read and booked; the others wait in `pending` (a step that is dispatched but whose loss the host has not read yet). `params` is always the state after
    # step `sent`. A step that needs exactly its own state (an evaluation, a checkpoint, the end of the plan, the time limit) sets `hold`: nothing is dispatched until the pipeline is empty again. With --async-steps 0 every step
    # is read at once after its dispatch: the loop of every earlier run.
    lag = a.async_steps
    pending = deque()
    sent = step = start
    hold = False
    t_read = None  # when the loss of the last step reached the host: the time of a step is counted from the later of its own start and this
    final_bg_step = None  # the step of a background checkpoint that was started when the time limit ended the run
    host_work_s = float(os.environ.get("FORGE_TEST_HOST_WORK_MS", "0") or 0) / 1000.0  # tests only: host work after every step (the pipeline must hide it behind the device)

    def eval_due(s):
        return s % a.eval_every == 0 or s == a.steps or (s <= a.eval_early_until and s % 100 == 0)

    def time_over():
        """The time limit is over, and the pipeline is empty (only then is `params` the state after the step that was just booked)."""
        return a.time_limit_s > 0 and not pending and time.time() - t_start > a.time_limit_s

    def val_mean(get_batch):
        """The mean loss over --eval-batches windows. All the evaluations are dispatched first and read afterwards (one wait instead of one per window; the values and their order are those of the sequential loop)."""
        outs = []
        for k in range(a.eval_batches):
            t_, y_, i_ = get_batch(k, 2, seq)  # two windows per batch as in the CUDA trainer; repeated so that every device gets a row (the mean is unchanged)
            reps = (ndev + 1) // 2
            t_, y_, i_ = (np.tile(x, (reps, 1))[:ndev] if ndev > 2 else x for x in (t_, y_, i_))
            outs.append(evaluate(params, *(jax.device_put(x, eval_sh) for x in (t_, y_, i_))))
        total = 0.0
        for o in outs:
            total += float(o)
        return total / a.eval_batches

    try:
        while step < a.steps:
            if not hold and sent < a.steps:
                t0 = time.time()
                lr = lr_at(sent, a.steps, h.peak_lr, h.warmup, h.min_frac, h.decay_frac)
                lr_muon = lr * h.muon_lr / h.peak_lr
                item = q.get(timeout=900)  # raises queue.Empty if no batch arrives for 15 minutes (a hung reader stops the run instead of holding the session)
                if isinstance(item, BaseException):
                    raise RuntimeError("the batch producer thread failed") from item
                tokens, targets, ignore = item
                wait = time.time() - t0
                if os.environ.get("FORGE_TEST_HANG_AT_STEP") == str(sent + 1):
                    time.sleep(10**6)  # tests only: a hung device
                lr_a, lr_muon_a = scalars.get("lr", lr), scalars.get("lr_muon", lr_muon)
                loss_v = gnorm_v = t_done = None
                try:
                    params, m, v, step_count, loss, gnorm = train_step(params, m, v, step_count, tokens, targets, ignore, lr_a, lr_muon_a)
                    if sent == start:
                        loss_v, gnorm_v = float(loss), float(gnorm)  # the first step is always read at once, whatever --async-steps says: a kernel that fails to compile or run fails here
                        t_done = time.time()
                except Exception as e:  # noqa: BLE001
                    if ATTN["impl"] is None or sent != start:
                        raise
                    # the splash kernel passed its self-test but the whole step does not compile or run with it: nothing was consumed (the failure is at compile time), start again with XLA's attention
                    print(f"[attn] the splash kernel failed in the full training step ({type(e).__name__}: {str(e).strip()[:400]}): XLA attention", flush=True)
                    ATTN["impl"] = None
                    train_step = build_train_step(c, h, cdt, a.ce_chunk, not a.no_remat, muon_mesh, a.remat_policy, a.fuse_qkv, a.attn_prep, a.layer_opts)
                    evaluate = build_eval(c, cdt, a.ce_chunk)
                    params, m, v, step_count, loss, gnorm = train_step(params, m, v, step_count, tokens, targets, ignore, lr_a, lr_muon_a)
                    loss_v, gnorm_v = float(loss), float(gnorm)
                    t_done = time.time()
                sent += 1
                pending.append((t0, wait, lr, loss, gnorm, loss_v, gnorm_v, t_done))
                if os.environ.get("FORGE_TEST_POISON_AT_STEP") == str(sent):
                    params = {k: (x * jnp.nan if k == "l0.attn_norm" else x) for k, x in params.items()}  # tests only: a finite loss followed by poisoned parameters
                hold = (eval_due(sent) or sent % a.save_every == 0 or sent >= a.steps or (a.time_limit_s > 0 and sent == int((1.0 - h.decay_frac) * a.steps))
                        or (a.time_limit_s > 0 and bool(plain_dts) and time.time() - t_start + (len(pending) + 1) * plain_dts[-1] > a.time_limit_s))  # the last term: the limit is about to be reached, so the pipeline is emptied
            stop = False
            while pending and (hold or sent >= a.steps or len(pending) > lag):
                t0, wait, lr, loss, gnorm, loss_v, gnorm_v, t_done = pending.popleft()
                if loss_v is None:
                    loss_v, gnorm_v = float(loss), float(gnorm)
                    t_done = time.time()
                if not (math.isfinite(loss_v) and math.isfinite(gnorm_v)):
                    # the parameters are already poisoned (NaN gradients pass the clipping and the update): nothing is written, so that ckpt_latest.bin keeps the last good state for a --resume (with --async-steps the
                    # steps behind this one were dispatched already: the run stops at the first bad loss that is read, and no checkpoint is written in between: every save empties the pipeline and checks the state itself)
                    print(f"[train] ERROR: loss {loss_v} and gradient norm {gnorm_v} at step {step + 1} are not finite: stopping WITHOUT a checkpoint (ckpt_latest.bin keeps the last good state); exit code 98", flush=True)
                    return 98
                dt = t_done - (t0 if t_read is None else max(t0, t_read))
                t_read = t_done
                plain_dts.append(dt)
                line = f"step {step + 1:5}/{a.steps} loss {loss_v:.4f} lr {lr:.2e} gnorm {gnorm_v:.3f} scale 1 {tokens_per_step / dt:.0f} tok/s ({dt:.2f} s)" + (f" data wait {wait:.2f} s" if wait > 0.02 else "")
                rec = {"step": step + 1, "loss": loss_v, "lr": lr, "grad_norm": gnorm_v, "tokens_per_s": tokens_per_step / dt}
                if data.cool is not None:
                    share_now = share_at(step)
                    line += f" cool {share_now:.2f}"
                    rec["cool_share"] = share_now
                if eval_due(step + 1):
                    assert not pending, "an evaluation needs the state after its own step"
                    t_e = time.time()
                    try:
                        vl = val_mean(data.val_batch)
                    except Exception as e:  # noqa: BLE001
                        if ATTN["impl"] is None:
                            raise
                        print(f"[attn] the splash kernel failed in the evaluation ({type(e).__name__}: {str(e).strip()[:400]}): XLA attention for the evaluation", flush=True)
                        ATTN["impl"] = None  # the training step is already compiled with its attention; only the evaluation is traced again
                        evaluate = build_eval(c, cdt, a.ce_chunk)
                        vl = val_mean(data.val_batch)
                    if data.cool is not None and data.cool.heldout is not None:
                        rec["val_cool"] = val_mean(data.cool_val_batch)  # the held-out cool-down file (the evaluation function is final here: the call above has already handled a failing attention kernel)
                    eval_dts.append(time.time() - t_e)
                    line += f"  val {vl:.4f}" + (f"  val_cool {rec['val_cool']:.4f}" if "val_cool" in rec else "")
                    rec["val_loss"] = vl
                if (step + 1) % a.print_every == 0 or step < start + 20 or " val " in line or step + 1 == a.steps:
                    print(line, flush=True)
                log.write(json.dumps(rec) + "\n")
                log.flush()
                progress.update(t=time.time(), what=f"step {step + 1}", armed=True)
                if host_work_s > 0:
                    time.sleep(host_work_s)
                report_job()  # the outcome of a background checkpoint write that has finished meanwhile
                if a.time_limit_s > 0 and step + 1 == int((1.0 - h.decay_frac) * a.steps) and not (out / "ckpt_predecay.bin").exists():
                    # the last state before the learning-rate decay: a later session can start the decay again from here (for example with another data mix) if the run has to be repeated
                    assert not pending, "a checkpoint needs the state after its own step"
                    status = save_state(out / "ckpt_predecay.bin", step + 1, background=True)
                    if status == "nonfinite":
                        return 98
                    if status in ("ok", "started"):
                        print(f"[train] pre-decay checkpoint at step {step + 1}" + (" (written by a background thread)" if status == "started" else ""), flush=True)
                    progress.update(t=time.time(), what="pre-decay checkpoint")
                if (step + 1) % a.save_every == 0 or step + 1 == a.steps or time_over():
                    assert not pending, "a checkpoint needs the state after its own step"
                    t_s = time.time()
                    # a write in the background is never used for what may be the last checkpoint: the end of the plan, or a time limit that is over or about to be
                    near_end = step + 1 == a.steps or time_over() or (a.time_limit_s > 0 and time.time() - t_start + 120.0 > a.time_limit_s)
                    status = save_state(out / "ckpt_latest.bin", step + 1, background=not near_end, skip_if_busy=True)
                    if status == "nonfinite":
                        return 98
                    final = step + 1 == a.steps or time_over()
                    if status == "skipped" and final:
                        status = save_state(out / "ckpt_latest.bin", step + 1)  # the clock ran out just now: this one is the last, write it in the foreground
                    if status == "started" and final:
                        final_bg_step = step + 1
                    if status == "failed" and final:
                        print("[train] ERROR: the final checkpoint could not be written: exit code 96", flush=True)
                        return 96
                    progress.update(t=time.time(), what="checkpoint")
                    if status == "ok":
                        save_dts.append(time.time() - t_s)
                        print(f"[train] checkpoint at step {step + 1} ({time.time() - t_start:.0f} s elapsed, {save_dts[-1]:.0f} s to write)", flush=True)
                    elif status == "started":
                        save_dts.append(time.time() - t_s)
                        print(f"[train] checkpoint at step {step + 1}: the state is copied to host memory ({save_dts[-1]:.1f} s), a background thread writes the file ({time.time() - t_start:.0f} s elapsed)", flush=True)
                    elif status == "skipped":
                        print(f"[train] checkpoint at step {step + 1} skipped: the background thread is still writing the previous one", flush=True)
                    if time_over():
                        print(f"[train] time limit of {a.time_limit_s:.0f} s reached: stop", flush=True)
                        if step + 1 < a.steps:
                            print(f"[train] WARNING: stopped at step {step + 1} of {a.steps} planned: the learning-rate decay is NOT finished (learning rate {lr:.2e}, the schedule ends at "
                                  f"{lr_at(a.steps - 1, a.steps, h.peak_lr, h.warmup, h.min_frac, h.decay_frac):.2e}); a later session can finish it from ckpt_latest.bin with --resume", flush=True)
                        stop = True
                        break
                step += 1
                if t_ref is None and len(plain_dts) >= 5:
                    t_ref, step_ref = time.time(), step
                if auto_steps and step % 100 == 0 and len(plain_dts) >= 60:
                    step_secs = float(np.median(plain_dts[5:][-200:]))
                    overhead = (float(np.median(eval_dts)) / a.eval_every if eval_dts else 0.0) + ((float(np.median(save_dts)) if save_dts else 25.0) / a.save_every)
                    wall = (time.time() - t_ref) / (step - step_ref) if t_ref is not None and step - step_ref >= 200 else 0.0
                    new_steps = replan_steps(a.steps, steps_max, step, time.time() - t_start, step_secs, overhead, a.time_limit_s, plan_reserve, h.decay_frac, wall)
                    if new_steps is not None:
                        print(f"[train] time plan: {step_secs:.3f} s per step (+{overhead:.3f} s evaluation and checkpoints; {wall:.3f} s per step measured over the whole run) -> {new_steps} steps fit into {a.time_limit_s:.0f} s (was {a.steps}, at most {steps_max})", flush=True)
                        a.steps = new_steps
            if stop:
                break
            if not pending:
                hold = False
        if ckpt_job is not None:  # the last background write must be on the disk before the run ends (a failed final one is exit code 96)
            ckpt_job.wait(lambda s: print(s, flush=True))
            r = report_job()
            if r is not None and r[2] == "failed" and r[1] == final_bg_step:
                print("[train] ERROR: the final checkpoint could not be written: exit code 96", flush=True)
                return 96
    finally:
        if ckpt_job is not None:  # also after an error: a write that is running is given the time to finish (its file is complete or absent, never half)
            ckpt_job.wait(lambda s: print(s, flush=True))
            report_job()
        if isinstance(log, AsyncLog):
            log.close()
        producer_stop.set()  # the batch thread stops after the batch it is making (a read and a device_put); the process must not end while it is inside the runtime
        producer_thread.join(timeout=60)
        if producer_thread.is_alive():
            print("[train] WARNING: the batch thread is still busy 60 s after the end of the run: the process may abort at exit", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
