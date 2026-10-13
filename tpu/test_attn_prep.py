"""Checks of --attn-prep lane (tpu/forge_jax.py): q, k and v prepared for the attention on arrays [batch, heads, head dim, T] (T on the lanes) and transposed once, in bf16, into the kernel's layout. It must be the SAME
function as the plain preparation (qk-norm, RoPE, value-embedding mix, scale, cast on [batch, T, heads, head dim]); the default must stay the program of every earlier run.

    JAX_PLATFORMS=cpu python test_attn_prep.py          (from this folder; tiny models on fake CPU devices, a few minutes)
"""
import hashlib
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax
import jax.numpy as jnp

import forge_jax as fj

HERE = os.path.dirname(os.path.abspath(__file__))
F32, BF16 = jnp.float32, jnp.bfloat16


def tiny_cfg(**kw):
    """GQA (4 heads on 2 key/value heads), head dim 64 as the real model, qk-norm, relu2, lambdas, value embeddings in two layers, a window in layer 0."""
    base = dict(n_layer=3, n_embd=256, n_head=4, n_kv=2, hd=64, n_ff=512, vocab=256, seq=64, relu2=True, lambdas=True, qk_norm=True, ve_layers=(0, 2), window=16, window_layers=(0,))
    base.update(kw)
    return fj.Cfg(**base)


def perturbed_params(c, seed=1):
    """init_params with every matrix and table moved away from the small initial values (norm weights and the lambdas stay at 1 and the mixing values, but get a deterministic spread below)."""
    p = fj.init_params(c, seed)
    out = {}
    for i, (n, a) in enumerate(p.items()):
        noise = jax.random.normal(jax.random.PRNGKey(100 + i), a.shape)
        if n.endswith("_norm"):
            out[n] = a * (1.0 + 0.2 * noise)  # different weights per lane: a mix-up of the lanes of the norm weight would show
        elif n in ("lam", "velam"):
            out[n] = a + 0.1 * noise
        else:
            out[n] = a + 0.05 * noise
    return out


def batch(c, b=4, seed=0):
    rng = np.random.default_rng(seed)
    tok = jnp.asarray(rng.integers(0, c.vocab, (b, c.seq)), jnp.int32)
    tgt = jnp.asarray(rng.integers(0, c.vocab, (b, c.seq)), jnp.int32)
    return tok, tgt, jnp.zeros((b, c.seq), bool)


def rel(a, b):
    """norm(a - b) / norm(b)"""
    return float(jnp.linalg.norm((a - b).astype(F32)) / (jnp.linalg.norm(b.astype(F32)) + 1e-30))


def global_rel(ga, gb):
    """relative difference of two gradient dicts as ONE vector"""
    return float(np.sqrt(sum(float(jnp.sum((ga[n] - gb[n]) ** 2)) for n in gb) / sum(float(jnp.sum(gb[n] ** 2)) for n in gb)))


def check_flag():
    base = ["train", "--data", "d", "--out", "o"]
    assert fj.parse_args(base).attn_prep == "plain"  # off by default
    assert fj.parse_args(base + ["--attn-prep", "lane"]).attn_prep == "lane"
    assert fj.parse_args(base + ["--attn-prep", "plain"]).attn_prep == "plain"
    try:
        fj.parse_args(base + ["--attn-prep", "everything"])
        raise AssertionError("an unknown --attn-prep was accepted")
    except SystemExit:
        pass
    assert fj.ATTN_PREPS == ("plain", "lane")
    print("flag: default plain, lane accepted, unknown refused: ok")


def check_kernel_heads_exact():
    """The transposition as a matmul with the identity is exact: every element comes out as it went in (bf16 and fp32 operands), also with zeros of both signs, and its gradient is the transposition back."""
    rng = np.random.default_rng(3)
    for cdt in (BF16, F32):
        a = jnp.asarray(rng.standard_normal((2, 3, 64, 256)) * 10.0 ** rng.integers(-6, 6, (2, 3, 64, 256)), F32)
        a = a.at[0, 0, 0, :4].set(jnp.asarray([0.0, -0.0, 1e-30, -1e30]))
        want = a.astype(cdt).transpose(0, 1, 3, 2)
        got = fj.kernel_heads(a, cdt)
        assert got.dtype == cdt and got.shape == (2, 3, 256, 64)
        assert np.array_equal(np.asarray(got.astype(F32)), np.asarray(want.astype(F32))), cdt  # equal values (a zero compares equal to a zero of the other sign)
        w = jnp.asarray(rng.standard_normal((2, 3, 256, 64)), F32)
        g = jax.grad(lambda x: jnp.sum(fj.kernel_heads(x, cdt).astype(F32) * w))(a)
        g_ref = jax.grad(lambda x: jnp.sum(x.astype(cdt).transpose(0, 1, 3, 2).astype(F32) * w))(a)
        assert np.array_equal(np.asarray(g), np.asarray(g_ref)), cdt
    print("kernel_heads (the identity matmul): the exact transposition, bf16 and fp32, values and gradient: ok")


def check_norm_rope_lane():
    """norm_rope_lane on [B, n, hd, T] gives the numbers of rms() followed by rope() on [B, T, n, hd]: in fp32 (and with a norm weight that differs per lane), also without the qk-norm."""
    c = tiny_cfg()
    rng = np.random.default_rng(5)
    b, n, t, hd = 2, 4, c.seq, c.hd
    x = jnp.asarray(rng.standard_normal((b, t, n, hd)) * 3.0, F32)
    w = jnp.asarray(1.0 + 0.3 * rng.standard_normal(hd), F32)
    cos, sin = fj.rope_tables(c, t)
    cos_t, sin_t = fj.rope_tables_lane(cos, sin)
    assert cos_t.shape == (hd // 2, t) and sin_t.shape == (hd // 2, t)
    for weight in (w, None):
        ref = fj.rope(fj.rms(x, weight, c.eps) if weight is not None else x, cos, sin)  # [b, t, n, hd]
        got = fj.norm_rope_lane(x.transpose(0, 2, 3, 1), weight, cos_t, sin_t, c.eps).transpose(0, 3, 1, 2)
        err = float(jnp.max(jnp.abs(got - ref)) / jnp.max(jnp.abs(ref)))
        assert err < 1e-6, (weight is None, err)
    print("norm_rope_lane = rms + rope of the plain path (fp32, with and without the qk-norm): ok")


def check_layer_outputs():
    """layer_fn with attn_prep lane against plain on the same inputs: fp32 within 1e-5, bf16 at the level of bf16 rounding (far below the distance of bf16 to fp32); all combinations of fuse_qkv, window, value embedding."""
    c = tiny_cfg(n_layer=1, ve_layers=(0,), window=16, window_layers=(0,))
    p = perturbed_params(c)
    rng = np.random.default_rng(2)
    b, t, e = 2, c.seq, c.n_embd
    x = jnp.asarray(rng.standard_normal((b, t, e)), F32)
    x0 = jnp.asarray(rng.standard_normal((b, t, e)), F32)
    lam = jnp.asarray([0.9, 0.2, 0.0], F32)
    velam = jnp.asarray([0.8, 0.4], F32)
    ve = jnp.asarray(rng.standard_normal((b, t, c.kv_dim)), BF16)
    lw = {n: p[f"l0.{n}"] for n in ("wq", "wk", "wv", "wo", "w_up", "w_down", "attn_norm", "ffn_norm", "q_norm", "k_norm")}
    cos, sin = fj.rope_tables(c, t)
    tabs = fj.rope_tables_lane(cos, sin)
    msgs = []
    for fuse in (False, True):
        for window in (None, 16):
            for use_ve in (False, True):
                outs = {}
                for cdt in (F32, BF16):
                    for prep, (cs, sn) in (("plain", (cos, sin)), ("lane", tabs)):
                        f = jax.jit(lambda x_, cs=cs, sn=sn, cdt=cdt, prep=prep: fj.layer_fn(c, cdt, x_, x0, None, lam, velam if use_ve else None, ve if use_ve else None, lw, cs, sn, window, fuse, prep))
                        outs[(cdt, prep)] = f(x)
                d32 = float(jnp.max(jnp.abs(outs[(F32, "lane")] - outs[(F32, "plain")])) / jnp.max(jnp.abs(outs[(F32, "plain")])))
                d16 = rel(outs[(BF16, "lane")], outs[(BF16, "plain")])
                dref = rel(outs[(BF16, "plain")], outs[(F32, "plain")])  # what bf16 itself costs
                assert d32 < 1e-5, (fuse, window, use_ve, d32)
                assert d16 < 0.5 * dref, (fuse, window, use_ve, d16, dref)
                msgs.append((d32, d16, dref))
    print(f"layer outputs, lane against plain over fuse_qkv x window x value embedding: fp32 worst {max(m[0] for m in msgs):.1e} (limit 1e-5); bf16 worst {max(m[1] for m in msgs):.1e} against {min(m[2] for m in msgs):.1e} between bf16 and fp32: ok")


def check_gradients():
    """jax.grad of the loss of a small model (GQA, qk-norm, relu2, lambdas, value embeddings, a window layer) with respect to EVERY parameter: lane against plain, with and without remat and fuse_qkv. fp32: every array within 1e-5 and
    the loss equal; bf16: the loss within 1e-5, the gradient differs far less than bf16 differs from fp32, and the lane gradient is as close to the fp32 one as the plain gradient is."""
    c = tiny_cfg()
    p = perturbed_params(c)
    tok, tgt, ign = batch(c)

    def vg(cdt, prep, remat, fuse):
        return jax.jit(jax.value_and_grad(lambda p_: fj.loss_fn(c, p_, tok, tgt, ign, cdt, 32, remat, "dots", fuse, prep)))(p)

    ref32 = vg(F32, "plain", False, False)
    rows = []
    for remat in (False, True):
        for fuse in (False, True):
            a, b = vg(F32, "plain", remat, fuse), vg(F32, "lane", remat, fuse)
            worst = max(rel(b[1][n], a[1][n]) for n in p)
            assert abs(float(b[0]) - float(a[0])) <= 1e-6 * abs(float(a[0])), (remat, fuse, float(a[0]), float(b[0]))
            assert worst < 1e-5, (remat, fuse, max(p, key=lambda n: rel(b[1][n], a[1][n])), worst)
            assert all(float(jnp.linalg.norm(b[1][f"l{l}.{w}"])) > 0 for l in range(3) for w in ("wq", "wk", "wv", "q_norm", "k_norm")), "a parameter received no gradient"
            a16, b16 = vg(BF16, "plain", remat, fuse), vg(BF16, "lane", remat, fuse)
            d_between, d_plain, d_lane = global_rel(b16[1], a16[1]), global_rel(a16[1], ref32[1]), global_rel(b16[1], ref32[1])
            assert abs(float(b16[0]) - float(a16[0])) <= 1e-5 * abs(float(a16[0])), (remat, fuse, float(a16[0]), float(b16[0]))
            assert d_between < 0.5 * d_plain and d_lane < 1.05 * d_plain, (remat, fuse, d_between, d_plain, d_lane)
            rows.append((worst, d_between, d_plain, d_lane))
    print(f"gradients of {len(p)} arrays, lane against plain over remat x fuse_qkv: fp32 worst array {max(r[0] for r in rows):.1e} (limit 1e-5), loss equal; bf16 lane-plain {max(r[1] for r in rows):.1e}, "
          f"plain-fp32 {np.mean([r[2] for r in rows]):.1e}, lane-fp32 {np.mean([r[3] for r in rows]):.1e}: ok")


def _program_hash(attn_prep=None):
    c = fj.Cfg(n_layer=3, n_embd=64, n_head=2, n_kv=2, hd=32, n_ff=128, vocab=512, seq=64, relu2=True, lambdas=True, ve_layers=(0, 1))
    h = fj.Hyper(peak_lr=3e-3, muon_lr=3e-3, mult_table=20.0, mult_scalar=10.0, use_muon=True)
    p = fj.init_params(c, 1)
    m = {n: jnp.zeros_like(a) for n, a in p.items()}
    v = {n: jnp.zeros_like(a) for n, a in p.items()}
    t1 = jnp.zeros((1, 2, 64), jnp.int32)
    ig = jnp.zeros((1, 2, 64), bool)
    step = fj.build_train_step(c, h, F32, 32, False) if attn_prep is None else fj.build_train_step(c, h, F32, 32, False, None, "dots", False, attn_prep)
    text = step.lower(p, m, v, jnp.asarray(0, jnp.int32), t1, t1, ig, jnp.float32(3e-3), jnp.float32(3e-3)).as_text()
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def check_default_program_unchanged():
    """The default (and --attn-prep plain) lowers to the very program of before the option existed: the hash of the lowered tiny training step is the one test_forge_jax.py pins; lane is another program."""
    d0, plain, lane = _program_hash(None), _program_hash("plain"), _program_hash("lane")
    assert d0 == plain, "--attn-prep plain must be the program of the default"
    assert lane != d0, "--attn-prep lane must change the program"
    if jax.__version__ == "0.11.2":
        assert d0 == "dc6c181a9996e0c1", d0  # the hash of check_window_off_is_identical / check_fuse_qkv
    print(f"default program unchanged ({d0}), lane differs ({lane}): ok")


def _splash_lane_body():
    """Runs in a process with two CPU devices: the lane preparation in front of the splash kernels (Pallas interpreter, dense and windowed layers, both backward variants) against the plain preparation in front of the same kernels."""
    from jax.sharding import Mesh

    devs = jax.devices()
    assert len(devs) == 2, devs
    mesh = Mesh(np.array(devs), ("data",))
    c = tiny_cfg(n_layer=3, n_embd=128, n_head=2, n_kv=1, seq=256, n_ff=128, window=128, window_layers=(0,))
    assert [c.window_of(l) for l in range(3)] == [128, None, None]
    p = perturbed_params(c)
    tok, tgt, ign = batch(c, b=2)
    for fused in (False, True):
        splash = fj.make_splash_attention(c, mesh, 128, fused, interpret=True)
        calls = {"prepared": 0, "plain": 0}
        orig = splash.prepared

        def counting(*a, **k):
            calls["prepared"] += 1
            return orig(*a, **k)

        def counting_plain(*a, **k):
            calls["plain"] += 1
            return splash(*a, **k)

        counting_plain.prepared = counting  # the lane path asks for `prepared`, the plain path calls the object itself
        fj.ATTN["impl"] = counting_plain
        for cdt in (F32, BF16):
            res = {}
            for prep in ("plain", "lane"):
                loss_grad = jax.value_and_grad(lambda p_: fj.loss_fn(c, p_, tok, tgt, ign, cdt, 64, True, "dots", False, prep))
                res[prep] = jax.jit(loss_grad)(p)
            (l0, g0), (l1, g1) = res["plain"], res["lane"]
            worst = max(rel(g1[n], g0[n]) for n in p)
            lim = 1e-5 if cdt == F32 else 2e-2
            print(f"lane in front of the splash kernels (interpreter), fused backward={fused}, {cdt.__name__}: loss {float(l0):.6f} vs {float(l1):.6f}, worst gradient difference {worst:.2e} (limit {lim:.0e})", flush=True)
            assert abs(float(l1) - float(l0)) <= (1e-6 if cdt == F32 else 2e-4) * abs(float(l0)) and worst < lim, (fused, cdt, float(l0), float(l1), worst)
        assert calls["prepared"] > 0 and calls["plain"] > 0, calls  # both entries were really used (traced once per layer function and remat)
    fj.ATTN["impl"] = None
    print("SPLASH LANE OK")


def check_splash_prepared():
    env = dict(os.environ, XLA_FLAGS="--xla_force_host_platform_device_count=2", JAX_PLATFORMS="cpu", PYTHONUTF8="1")
    r = subprocess.run([sys.executable, os.path.join(HERE, "test_attn_prep.py"), "splash-lane-body"], capture_output=True, text=True, env=env, timeout=1500)
    print(r.stdout[-1500:], r.stderr[-1500:] if r.returncode else "")
    assert r.returncode == 0 and "SPLASH LANE OK" in r.stdout, r.stderr[-3000:]
    print("lane in front of the splash kernels (Pallas interpreter, 2 devices): same loss and gradients: ok")


def _tiny_run(tmp, extra, steps="14"):
    import test_forge_jax as tf

    out, args, env = tf._tiny_run_args(tmp, ["--steps", steps, "--eval-every", "7", "--save-every", "100"] + extra)
    r = subprocess.run([sys.executable, os.path.join(HERE, "forge_jax.py")] + args, capture_output=True, text=True, env=env, timeout=600)
    assert r.returncode == 0, r.stderr[-2000:] + r.stdout[-1000:]
    losses = [json.loads(ln)["loss"] for ln in open(os.path.join(out, "log.jsonl")) if ln.strip()]
    return r.stdout, losses


def check_training_run():
    """A tiny training run (2 fake CPU devices, fp32) with --attn-prep lane follows the plain run (the same function: the losses agree to a few 1e-5), says so in its log, and a run whose attention kernel fails goes on with XLA's attention."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as t1, tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as t2:
        _, plain = _tiny_run(t1, [])
        out, lane = _tiny_run(t2, ["--attn-prep", "lane"])
    assert "attention input preparation: lane" in out, out[-800:]
    assert len(plain) == len(lane) == 14 and lane[-1] < lane[0] - 0.3, (plain, lane)
    worst = max(abs(a - b) / abs(a) for a, b in zip(plain, lane))
    assert worst < 2e-4, (worst, plain, lane)
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as t3:
        out3, lane3 = _tiny_run(t3, ["--attn-prep", "lane", "--attn-test-fail", "step"], steps="12")
        assert "failed in the full training step" in out3 and "step    12/12" in out3, out3[-800:]
    print(f"training run: lane follows plain over 14 steps (worst relative difference of the loss {worst:.1e}), loss {lane[0]:.3f} -> {lane[-1]:.3f}; a failing attention kernel falls back to XLA's with lane: ok")


def check_bench_variants():
    """bench_step.py: the two new variants come after the existing ones with the flags the task names, the CPU self-test runs attn-lane (a worker process on a tiny model with 8 fake devices) and reports the option."""
    import bench_step as bs

    names = [v[0] for v in bs.variants()]
    flags = dict((v[0], v[1]) for v in bs.variants())
    assert names[0] == "base" and names.index("attn-lane") + 1 == names.index("attn-lane+muon-shard") and names.index("attn-lane") > names.index("muon-shard+libtpu-async"), names  # after every variant of the first rounds; later options may follow them
    assert flags["attn-lane"] == "--attn-prep lane" and flags["attn-lane+muon-shard"] == "--attn-prep lane --muon-shard", flags
    assert "attn-lane" in [v[0] for v in bs.variants(True)]
    env = dict(os.environ, JAX_PLATFORMS="cpu", PYTHONUTF8="1", XLA_FLAGS="--xla_force_host_platform_device_count=8")
    r = subprocess.run([sys.executable, os.path.join(HERE, "bench_step.py"), "--worker", "--name", "attn-lane", "--args=" + bs.ARGS_TINY + " --attn-prep lane", "--steps", "2", "--warmup", "1"], capture_output=True, text=True, env=env, timeout=600)
    res = bs.parse_worker_output(r.stdout)
    assert res is not None and res["status"] == "ok" and res["attn_prep"] == "lane" and res["tokens_per_s"] > 0, (res, r.stderr[-1500:])
    print("bench_step: attn-lane and attn-lane+muon-shard follow the variants of the first rounds with the task's flags, a tiny CPU worker runs attn-lane: ok")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "splash-lane-body":
        _splash_lane_body()
        sys.exit(0)
    check_flag()
    check_kernel_heads_exact()
    check_norm_rope_lane()
    check_layer_outputs()
    check_gradients()
    check_default_program_unchanged()
    check_splash_prepared()
    check_training_run()
    check_bench_variants()
    print("ATTN PREP OK")
