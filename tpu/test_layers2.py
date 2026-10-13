"""Checks of the layer program options that the compile lab found (tpu/forge_jax.py): --layer-opts ffnbf16 (the bf16 copies of the feed-forward weights made once per layer), --layer-opts woheads (the output projection on the attention
kernel's own layout [B, heads, T, hd]) and the remat parts `norm` / `qknorm` (--remat-policy dots+norm+qknorm: the mean squares of the normalisations are kept instead of recomputed in the backward pass). They must compute the SAME
function (ffnbf16 and the remat parts: bit for bit; woheads: the sum over the 1024 inputs of the output projection is made per head, a different fp32 accumulation order), and the default must stay the program of every earlier run.

    JAX_PLATFORMS=cpu python test_layers2.py          (from this folder; tiny models on fake CPU devices, a few minutes)
"""
import contextlib
import io
import os
import re
import shlex
import subprocess
import sys

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax
import jax.numpy as jnp

import forge_jax as fj
import test_attn_prep as T

HERE = os.path.dirname(os.path.abspath(__file__))
F32, BF16 = jnp.float32, jnp.bfloat16
BASE_ARGS = ["train", "--data", "d", "--out", "o"]


def same(a, b):
    return bool(np.array_equal(np.asarray(a), np.asarray(b)))


def vg(c, p, cdt, remat, policy="dots", fuse=False, prep="plain", opts="", b=4):
    tok, tgt, ign = T.batch(c, b=b)
    return jax.jit(jax.value_and_grad(lambda p_: fj.loss_fn(c, p_, tok, tgt, ign, cdt, 32, remat, policy, fuse, prep, opts)))(p)


def check_flags():
    assert fj.parse_args(BASE_ARGS).layer_opts == ""  # off by default
    assert fj.parse_args(BASE_ARGS + ["--layer-opts", "ffnbf16,woheads"]).layer_opts == "ffnbf16,woheads"
    assert fj.parse_layer_opts("") == frozenset() and fj.parse_layer_opts(None) == frozenset() and fj.parse_layer_opts("ffnbf16, woheads") == frozenset({"ffnbf16", "woheads"})
    for bad in ("nope", "ffnbf16,nope"):
        try:
            fj.parse_args(BASE_ARGS + ["--layer-opts", bad])
            raise AssertionError(f"accepted --layer-opts {bad}")
        except SystemExit:
            pass
        try:
            fj.parse_layer_opts(bad)
            raise AssertionError(bad)
        except ValueError:
            pass
    assert set(fj.LAYER_OPTS) == {"ffnbf16", "woheads"}
    # the remat parts: the grammar knows them, in the canonical order, with @N, and every part has its own checkpoint name
    assert fj.parse_remat("dots+qknorm+norm") == (("norm", "qknorm"), 0) and fj.parse_remat("dots+norm@3") == (("norm",), 3) and fj.parse_remat("dots+attn+acts+qkv+norm+qknorm")[0] == ("attn", "h", "h2", "act", "qkv", "norm", "qknorm")
    assert fj.parse_args(BASE_ARGS + ["--remat-policy", "dots+norm+qknorm", "--remat-save-layers", "2"]).remat_policy == "dots+norm+qknorm@2"
    assert len(set(fj.SAVE_NAMES.values())) == len(fj.SAVE_NAMES)
    assert fj.splash_names("dots+norm+qknorm") == {"residual_name": None, "qkv_name": None}  # these two parts are named in the layer, not in the kernel
    assert fj.remat_extras("dots+norm", 0, 4) == ("norm",) and fj.remat_extras("dots+qknorm@1", 2, 4) == () and fj.remat_extras("dots+qknorm@1", 3, 4) == ("qknorm",)
    print("flags: --layer-opts off by default, ffnbf16 and woheads accepted, unknown refused; the remat parts norm and qknorm in the grammar: ok")


def _hash(remat, policy="dots", prep="plain", fuse=False, opts=None):
    """sha256 prefix of the lowered training step of a tiny model (the idea of test_forge_jax._window_hlo_hash)."""
    import hashlib

    c = fj.Cfg(n_layer=3, n_embd=64, n_head=2, n_kv=2, hd=32, n_ff=128, vocab=512, seq=64, relu2=True, lambdas=True, qk_norm=True, ve_layers=(0, 1))
    h = fj.Hyper(peak_lr=3e-3, muon_lr=3e-3, mult_table=20.0, mult_scalar=10.0, use_muon=True)
    p = fj.init_params(c, 1)
    m = {n: jnp.zeros_like(a) for n, a in p.items()}
    v = {n: jnp.zeros_like(a) for n, a in p.items()}
    tok = jnp.zeros((1, 2, 64), jnp.int32)
    ign = jnp.zeros((1, 2, 64), bool)
    step = fj.build_train_step(c, h, F32, 32, remat, None, policy, fuse, prep) if opts is None else fj.build_train_step(c, h, F32, 32, remat, None, policy, fuse, prep, opts)
    text = step.lower(p, m, v, jnp.asarray(0, jnp.int32), tok, tok, ign, jnp.float32(3e-3), jnp.float32(3e-3)).as_text()
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def check_default_program_unchanged():
    """No option, or an empty --layer-opts, is the very program of before the options existed: the hashes below are those of the lowered tiny training step of main at ca253e0 (jax 0.11.2); every option changes the program."""
    gold = {("off", "plain"): "dc6c181a9996e0c1", ("on", "plain"): "333e8c45ff7c5bd0", ("on", "lane"): "538dd72e634b6a26", ("off", "lane"): "31f491960cf17766", ("on", "plain+fuse"): "d5aedc061f9887ed"}
    got = {("off", "plain"): _hash(False), ("on", "plain"): _hash(True), ("on", "lane"): _hash(True, prep="lane"), ("off", "lane"): _hash(False, prep="lane"), ("on", "plain+fuse"): _hash(True, fuse=True)}
    assert _hash(True, opts="") == got[("on", "plain")] and _hash(True, "dots", "lane", False, "") == got[("on", "lane")], "an empty --layer-opts must be the program of the default"
    if jax.__version__ == "0.11.2":
        assert got == gold, (got, gold)
    for opts, kw in (("ffnbf16", {}), ("woheads", {}), ("ffnbf16,woheads", {"prep": "lane"})):
        assert _hash(True, opts=opts, **kw) != (got[("on", "lane")] if kw else got[("on", "plain")]), opts
    for pol in ("dots+norm", "dots+qknorm", "dots+norm+qknorm"):
        assert _hash(True, pol) != got[("on", "plain")], pol
        assert _hash(False, pol) == got[("off", "plain")], pol  # without remat a policy does nothing
    print(f"default program unchanged ({got[('on', 'plain')]} with remat, {got[('off', 'plain')]} without; lane {got[('on', 'lane')]}), every option changes it: ok")


def check_ffnbf16_exact():
    """ffnbf16 = the same numbers, bit for bit: loss and every gradient, bf16 and fp32 operands, with and without remat, plain and lane, fused and separate q, k, v projections."""
    c = T.tiny_cfg()
    p = T.perturbed_params(c)
    n = 0
    for cdt in (BF16, F32):
        for remat in (False, True):
            for prep in ("plain", "lane"):
                for fuse in (False, True):
                    a, b = vg(c, p, cdt, remat, "dots", fuse, prep), vg(c, p, cdt, remat, "dots", fuse, prep, "ffnbf16")
                    assert same(a[0], b[0]), (cdt, remat, prep, fuse, float(a[0]), float(b[0]))
                    bad = [k for k in p if not same(a[1][k], b[1][k])]
                    assert not bad, (cdt, remat, prep, fuse, bad)
                    n += 1
    assert float(jnp.linalg.norm(b[1]["l0.w_down"])) > 0
    print(f"ffnbf16: loss and all {len(p)} gradients bit-identical in {n} configurations (bf16 / fp32 x remat x plain / lane x fuse_qkv): ok")


def check_remat_parts_exact():
    """The parts norm and qknorm save values that the backward pass would recompute: the same numbers bit for bit (plain and lane, bf16 and fp32, with the qk-norm and a model without it), the named values kept are exactly
    the mean squares (2 per layer for each part), and none are named without the part."""
    c = T.tiny_cfg(window=0, window_layers=())
    p = T.perturbed_params(c)

    def named(policy, prep, cc=c):
        tok, tgt, ign = T.batch(cc)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            jax.ad_checkpoint.print_saved_residuals(lambda p_: fj.loss_fn(cc, p_, tok, tgt, ign, BF16, 32, True, policy, False, prep), p)
        return [m for m in re.findall(r"named '(forge_[a-z0-9_]+)'", buf.getvalue())]

    for prep in ("plain", "lane"):
        assert named("dots", prep) == []
        for cdt in (BF16, F32):
            ref = vg(c, p, cdt, True, "dots", False, prep)
            for pol, want in (("dots+norm", {"forge_norm": 2 * c.n_layer}), ("dots+qknorm", {"forge_qknorm": 2 * c.n_layer}), ("dots+norm+qknorm", {"forge_norm": 2 * c.n_layer, "forge_qknorm": 2 * c.n_layer}), ("dots+norm+qknorm@2", {"forge_norm": 4, "forge_qknorm": 4})):
                out = vg(c, p, cdt, True, pol, False, prep)
                assert same(out[0], ref[0]) and all(same(out[1][k], ref[1][k]) for k in p), (prep, cdt, pol)
                if cdt is BF16:
                    got = named(pol, prep)
                    assert {k: got.count(k) for k in set(got)} == want, (prep, pol, got)
    cn = T.tiny_cfg(window=0, window_layers=(), qk_norm=False)  # no qk-norm: only the two residual-stream norms have a mean square to keep
    pn = T.perturbed_params(cn)
    for prep in ("plain", "lane"):
        a, b = vg(cn, pn, F32, True, "dots", False, prep), vg(cn, pn, F32, True, "dots+norm+qknorm", False, prep)
        assert same(a[0], b[0]) and all(same(a[1][k], b[1][k]) for k in pn), prep
    print("remat parts norm, qknorm: bit-identical loss and gradients (plain / lane, bf16 / fp32, with and without the qk-norm), exactly the mean squares are kept (2 per layer and part): ok")


def check_woheads_close():
    """woheads = the same function: the output projection of the attention on [B, heads, T, hd]. fp32: loss and every gradient array within 1e-5 of the default; bf16: within the rounding of bf16 (the loss to 5e-5, the
    gradient far below the distance of bf16 to fp32), also with remat, in the lane layout, with a window layer and value embeddings."""
    c = T.tiny_cfg()
    p = T.perturbed_params(c)
    rows = []
    for prep in ("plain", "lane"):
        for remat in (False, True):
            a, b = vg(c, p, F32, remat, "dots", False, prep), vg(c, p, F32, remat, "dots", False, prep, "woheads")
            worst = max(T.rel(b[1][k], a[1][k]) for k in p)
            assert abs(float(b[0]) - float(a[0])) <= 1e-6 * abs(float(a[0])) and worst < 1e-5, (prep, remat, float(a[0]), float(b[0]), worst)
            a16, b16 = vg(c, p, BF16, remat, "dots", False, prep), vg(c, p, BF16, remat, "dots", False, prep, "woheads")
            ref32 = vg(c, p, F32, False)
            d_between, d_ref = T.global_rel(b16[1], a16[1]), T.global_rel(a16[1], ref32[1])
            assert abs(float(b16[0]) - float(a16[0])) <= 5e-5 * abs(float(a16[0])) and d_between < 0.5 * d_ref, (prep, remat, d_between, d_ref)  # (bf16 and fp32 differ by 2e-4 in the loss)
            rows.append((worst, d_between, d_ref))
    # the option is the same for GQA with other head counts and for a model whose heads x hd differ from the width
    c2 = T.tiny_cfg(n_head=8, n_kv=2, hd=32, n_embd=128, ve_layers=(1,), window_layers=(1,))
    p2 = T.perturbed_params(c2)
    a, b = vg(c2, p2, F32, True), vg(c2, p2, F32, True, "dots", False, "plain", "woheads")
    assert max(T.rel(b[1][k], a[1][k]) for k in p2) < 1e-5
    print(f"woheads: fp32 worst gradient array {max(r[0] for r in rows):.1e} (limit 1e-5), bf16 option-vs-default {max(r[1] for r in rows):.1e} against {min(r[2] for r in rows):.1e} between bf16 and fp32: ok")


def check_all_together():
    """Every option together (plain and lane, remat with all the parts): fp32 within 1e-5 of the default; one optimizer step of the real train step: the parameters after it equal for the bit-exact options, within 1e-3 for woheads."""
    c = T.tiny_cfg()
    p = T.perturbed_params(c)
    ref = vg(c, p, F32, True, "dots", False, "lane")
    out = vg(c, p, F32, True, "dots+norm+qknorm", False, "lane", "ffnbf16,woheads")
    assert abs(float(out[0]) - float(ref[0])) <= 1e-6 * abs(float(ref[0])) and max(T.rel(out[1][k], ref[1][k]) for k in p) < 1e-5
    h = fj.Hyper(peak_lr=3e-3, muon_lr=3e-3, mult_table=20.0, mult_scalar=10.0, use_muon=True)
    tok, tgt, ign = T.batch(c, b=2)
    tok, tgt, ign = tok[None], tgt[None], ign[None]

    def one_step(policy, opts):
        step = fj.build_train_step(c, h, F32, 32, True, None, policy, False, "lane", opts)
        m = {n: jnp.zeros_like(a) for n, a in p.items()}
        v = {n: jnp.zeros_like(a) for n, a in p.items()}
        res = step({n: jnp.array(a) for n, a in p.items()}, m, v, jnp.asarray(0, jnp.int32), tok, tgt, ign, jnp.float32(3e-3), jnp.float32(3e-3))  # (the step donates its arguments: fresh copies)
        return res[0], float(res[4])

    p0, l0 = one_step("dots", "")
    p1, l1 = one_step("dots+norm+qknorm", "ffnbf16")
    assert l0 == l1 and all(same(p0[k], p1[k]) for k in p), "ffnbf16 and the remat parts must give the very same step"
    p2, l2 = one_step("dots+norm+qknorm", "ffnbf16,woheads")
    worst = max(T.rel(p2[k], p0[k]) for k in p)
    # (the Newton-Schulz iterations of Muon run in bf16: a gradient that differs at the 1e-6 level moves the orthogonalised update by about 1 percent, which is 1e-4 of the parameters after one step)
    assert abs(l2 - l0) <= 1e-6 * abs(l0) and worst < 1e-3, (l0, l2, worst)
    print(f"all options together: fp32 loss and gradients within 1e-5; one real train step: ffnbf16 + remat parts bit-identical, with woheads the parameters differ by {worst:.1e} (Muon's bf16 iterations amplify the 1e-6 gradient differences): ok")


def check_bench_variants():
    """The variants of tpu/bench_step.py that use the new options parse with the trainer's own argument parser, and the tiny CPU list still has the base variant first."""
    import bench_step as bs

    names = [v[0] for v in bs.variants(False)]
    new = [v[0] for v in bs.variants(False) if "--layer-opts" in v[1] or "norm" in v[1]]
    assert {"layer-ffnbf16", "layer-woheads", "remat-qknorm", "remat-norm", "lane+shard+lean+ffnbf16+woheads"} <= set(new), new
    for name, flags, env, note in bs.variants(False) + bs.variants(True):
        a = fj.parse_args(BASE_ARGS + shlex.split(bs.ARGS_365M) + shlex.split(flags))
        fj.parse_layer_opts(a.layer_opts)
        fj.parse_remat(a.remat_policy)
    assert [v[0] for v in bs.variants(True)][0] == "base" and len(set(names)) == len(names) and names.index("layer-ffnbf16") > names.index("lane+shard+lean+attn")  # after every earlier variant
    assert {"layer-ffnbf16", "layer-woheads", "remat-qknorm", "lane+shard+lean+ffnbf16+woheads"} <= {v[0] for v in bs.variants(True)}  # part of the CPU self-test of the benchmark
    print(f"bench variants: {len(new)} for the new options ({', '.join(new)}), all flags parse: ok")


def _splash_body():
    """Runs in a process with two CPU devices: the options in front of the splash kernels (Pallas interpreter, dense and windowed layers, both backward variants, lane preparation): ffnbf16 and the remat parts bit for bit, woheads within 1e-5."""
    from jax.sharding import Mesh

    devs = jax.devices()
    assert len(devs) == 2, devs
    mesh = Mesh(np.array(devs), ("data",))
    c = T.tiny_cfg(n_layer=3, n_embd=128, n_head=2, n_kv=1, seq=256, n_ff=128, window=128, window_layers=(0,))
    p = T.perturbed_params(c)
    tok, tgt, ign = T.batch(c, b=2)
    for fused in (False, True):
        fj.ATTN["impl"] = fj.make_splash_attention(c, mesh, 128, fused, interpret=True)

        def run(policy, opts, cdt):
            return jax.jit(jax.value_and_grad(lambda p_: fj.loss_fn(c, p_, tok, tgt, ign, cdt, 64, True, policy, False, "lane", opts)))(p)

        for cdt in (BF16, F32):
            ref = run("dots", "", cdt)
            for policy, opts in (("dots", "ffnbf16"), ("dots+norm+qknorm", "")):
                out = run(policy, opts, cdt)
                assert same(out[0], ref[0]) and all(same(out[1][k], ref[1][k]) for k in p), (fused, cdt, policy, opts)
            out = run("dots+norm+qknorm", "ffnbf16,woheads", cdt)
            worst = max(T.rel(out[1][k], ref[1][k]) for k in p)
            lim = 1e-5 if cdt is F32 else 2e-2
            print(f"options in front of the splash kernels (interpreter), fused backward={fused}, {cdt.__name__}: bit-identical ffnbf16 / norm parts; all options: loss {float(ref[0]):.6f} vs {float(out[0]):.6f}, worst gradient {worst:.2e} (limit {lim:.0e})", flush=True)
            assert abs(float(out[0]) - float(ref[0])) <= (1e-6 if cdt is F32 else 2e-4) * abs(float(ref[0])) and worst < lim, (fused, cdt, worst)
    fj.ATTN["impl"] = None
    print("SPLASH OPTIONS OK")


def check_splash_options():
    env = dict(os.environ, XLA_FLAGS="--xla_force_host_platform_device_count=2", JAX_PLATFORMS="cpu", PYTHONUTF8="1")
    r = subprocess.run([sys.executable, os.path.join(HERE, "test_layers2.py"), "splash-body"], capture_output=True, text=True, env=env, timeout=1500)
    print(r.stdout[-1500:], r.stderr[-1500:] if r.returncode else "")
    assert r.returncode == 0 and "SPLASH OPTIONS OK" in r.stdout, r.stderr[-3000:]
    print("layer options in front of the splash kernels (Pallas interpreter, 2 devices): ok")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "splash-body":
        _splash_body()
        return 0
    for f in (check_flags, check_default_program_unchanged, check_ffnbf16_exact, check_remat_parts_exact, check_woheads_close, check_all_together, check_bench_variants, check_splash_options):
        f()
    print("ALL LAYERS2 CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
