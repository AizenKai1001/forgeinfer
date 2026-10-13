"""Checks of the optimizer options of forge_jax.py on the CPU: --muon-lean and --ns-barrier must give BIT-IDENTICAL parameters, momenta, second moments and gradient norm to the stacked formulation they replace
(with and without --muon-shard on 8 fake devices), and the default program must stay the one of main.

    python test_opt_tune.py            (about 2 minutes; the 8-device part runs in a child process)
"""
import hashlib
import os
import re
import subprocess
import sys

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax
import jax.numpy as jnp

import forge_jax as fj

HERE = os.path.dirname(os.path.abspath(__file__))

# every parameter group present: emb, head (untied), ve tables, norms, lam / velam, Muon matrices of three shapes (square wq / wo, wide wk / wv / w_down, tall w_up / w_gate)
CFGS = {
    "relu2": dict(n_layer=3, n_embd=64, n_head=2, n_kv=1, hd=32, n_ff=128, vocab=512, seq=64, qk_norm=True, relu2=True, lambdas=True, untied=True, ve_layers=(0, 2)),
    "gate": dict(n_layer=4, n_embd=64, n_head=2, n_kv=1, hd=32, n_ff=128, vocab=512, seq=64, qk_norm=True, relu2=False, lambdas=True, skips=True, ve_layers=(1, 3)),
}


def bits(a):
    return np.asarray(a, np.float32).view(np.uint32)


def same(a, b):
    return bool(np.array_equal(bits(a), bits(b)))


def random_state(c, seed):
    """Parameters, gradients, first and second moments of random values with the right scales (the second moment positive)."""
    rng = np.random.default_rng(seed)
    p = fj.init_params(c, seed)
    mk = lambda scale, positive=False: {n: jnp.asarray((np.abs(rng.standard_normal(a.shape)) if positive else rng.standard_normal(a.shape)).astype(np.float32) * scale) for n, a in p.items()}
    return p, mk(0.05), mk(0.01), mk(1e-4, True)


def run_updates(c, h, mesh, steps=3, clip=1.0):
    """`steps` chained optimizer steps on random gradients; returns the final parameters, moments, gradient norms."""
    hh = fj.Hyper(**{**h.__dict__, "clip": clip})
    p, _, m, v = random_state(c, 7)
    upd = jax.jit(lambda p_, g_, m_, v_, s: fj.apply_update(c, hh, p_, g_, m_, v_, s, jnp.float32(3e-3), jnp.float32(3e-3), mesh))
    norms = []
    for i in range(steps):
        _, g, _, _ = random_state(c, 100 + i)
        p, m, v, gn = upd(p, g, m, v, jnp.asarray(i + 1, jnp.int32))
        norms.append(float(gn))
    return p, m, v, norms


def compare(ref, got, what):
    (p0, m0, v0, n0), (p1, m1, v1, n1) = ref, got
    bad = [(t, k) for t, (a, b) in zip("pmv", ((p0, p1), (m0, m1), (v0, v1))) for k in a if not same(a[k], b[k])]
    assert not bad and n0 == n1, f"{what}: not bit-identical: {bad[:6]} norms {n0} vs {n1}"
    assert all(np.isfinite(np.asarray(a)).all() for a in p1.values())


def check_apply_update_identical(mesh=None, tag=""):
    base_h = dict(peak_lr=3e-3, muon_lr=3e-3, mult_table=20.0, mult_scalar=10.0, mult_emb=3.0, use_muon=True)
    for name, kw in CFGS.items():
        c = fj.Cfg(**kw)
        for clip in (1.0, 1e-3):  # the gradient norm is about 100: both clip values are active; 1e9 is not
            ref = run_updates(c, fj.Hyper(**base_h), mesh, clip=clip)
            for flags in ({"muon_lean": True}, {"ns_barrier": True}, {"muon_lean": True, "ns_barrier": True}):
                got = run_updates(c, fj.Hyper(**base_h, **flags), mesh, clip=clip)
                compare(ref, got, f"{tag}{name} clip {clip} {flags}")
        ref = run_updates(c, fj.Hyper(**base_h), mesh, clip=1e9)  # no clipping at all
        compare(ref, run_updates(c, fj.Hyper(**base_h, muon_lean=True), mesh, clip=1e9), f"{tag}{name} unclipped")
        print(f"{tag}{name}: --muon-lean, --ns-barrier and both are bit-identical to the stacked update after 3 steps (clip 1.0, 1e-3, off): ok", flush=True)
    # AdamW only (no Muon): the options are not used and must not matter
    c = fj.Cfg(**CFGS["relu2"])
    ref = run_updates(c, fj.Hyper(peak_lr=3e-3, use_muon=False), mesh)
    compare(ref, run_updates(c, fj.Hyper(peak_lr=3e-3, use_muon=False, muon_lean=True, ns_barrier=True), mesh), f"{tag}adamw")


def lowered_default_hash():
    c = fj.Cfg(n_layer=3, n_embd=64, n_head=2, n_kv=2, hd=32, n_ff=128, vocab=512, seq=64, relu2=True, lambdas=True, ve_layers=(0, 1))
    h = fj.Hyper(peak_lr=3e-3, muon_lr=3e-3, mult_table=20.0, mult_scalar=10.0, use_muon=True)
    p = fj.init_params(c, 1)
    m = {n: jnp.zeros_like(a) for n, a in p.items()}
    v = {n: jnp.zeros_like(a) for n, a in p.items()}
    tok = jnp.zeros((1, 2, 64), jnp.int32)
    step = fj.build_train_step(c, h, jnp.float32, 32, False)
    return hashlib.sha256(step.lower(p, m, v, jnp.asarray(0, jnp.int32), tok, tok, jnp.zeros((1, 2, 64), bool), jnp.float32(3e-3), jnp.float32(3e-3)).as_text().encode()).hexdigest()[:16]


def check_default_program():
    """The options are off by default: the lowered training step is the program of main (the hash of check_window_off_is_identical), and the flags default to off."""
    a = fj.parse_args(["train", "--data", "x", "--out", "y"])
    assert a.muon_lean is False and a.ns_barrier is False
    b = fj.parse_args(["train", "--data", "x", "--out", "y", "--muon-lean", "--ns-barrier"])
    assert b.muon_lean is True and b.ns_barrier is True
    if jax.__version__ == "0.11.2":
        assert lowered_default_hash() == "dc6c181a9996e0c1", lowered_default_hash()
    print("default program unchanged (hash dc6c181a9996e0c1), flags default off: ok")


def check_bench_variants():
    """The benchmark variants of the optimizer options come AFTER the earlier ones (a short TPU session reaches the old rows first), carry exactly their flags, and the tiny CPU list contains them."""
    import bench_step as bs

    names = [v[0] for v in bs.variants()]
    flags = {v[0]: v[1] for v in bs.variants()}
    new = ["muon-lean", "muon-lean+muon-shard", "ns-barrier", "ns-barrier+muon-shard", "muon-lean+ns-barrier", "muon-lean+ns-barrier+muon-shard"]
    assert names[:16] == ["base", "muon-shard", "window512", "muon-shard+window512", "remat-attn", "fuse-qkv", "hd128", "all", "breakdown-base", "breakdown-all", "no-remat", "no-remat+muon-shard", "micro8", "micro8+no-remat",
                          "libtpu-async", "muon-shard+libtpu-async"], names
    for n in new:
        assert n in names and names.index(n) > names.index("muon-shard+libtpu-async"), n
        assert sorted(flags[n].split()) == sorted(("--" + n.replace("+", " --")).split()), (n, flags[n])
    assert {"muon-lean", "muon-lean+muon-shard", "ns-barrier"} <= {v[0] for v in bs.variants(True)}
    print("bench variants of the optimizer options: after the existing rows, their flags, in the tiny list: ok")


def _concat_types(text):
    """Result types of the concatenate ops of a lowered module, e.g. ['tensor<24x4096x1024xf32>']."""
    return re.findall(r"stablehlo\.concatenate.*?->\s*(tensor<[^>]*>)", text)


def check_no_fp32_stacks():
    """The point of --muon-lean in the program text: the stacked formulation concatenates the fp32 parameters, gradients and momenta of the layers into [layers, rows, cols] stacks, the lean one only the bf16
    Newton-Schulz input; no rank-3 float32 concatenation is left, and the number of float32 elementwise stack-sized values drops."""
    c = fj.Cfg(**CFGS["relu2"])
    p, g, m, v = random_state(c, 3)

    def lowered(**flags):
        h = fj.Hyper(peak_lr=3e-3, muon_lr=3e-3, use_muon=True, **flags)
        return jax.jit(lambda *a: fj.apply_update(c, h, *a, jnp.float32(3e-3), jnp.float32(3e-3), None)).lower(p, g, m, v, jnp.asarray(1, jnp.int32)).as_text()

    old, new = _concat_types(lowered()), _concat_types(lowered(muon_lean=True))
    f32_old = [t for t in old if t.count("x") >= 3 and t.endswith("f32>")]
    f32_new = [t for t in new if t.count("x") >= 3 and t.endswith("f32>")]
    bf16_new = [t for t in new if t.endswith("bf16>")]
    assert len(f32_old) >= 3 * 3 and not f32_new and len(bf16_new) >= 3, (old, new)
    print(f"muon-lean: {len(f32_old)} rank-3 float32 concatenations of the stacked update -> {len(f32_new)}; {len(bf16_new)} bf16 stacks (the Newton-Schulz inputs): ok")


def check_train_step_identical():
    """The whole training step (forward, backward, update) with --muon-lean / --ns-barrier is bit-identical to the one without, after 3 steps with one and with two micro-steps."""
    c = fj.Cfg(**CFGS["relu2"])
    base = dict(peak_lr=3e-3, muon_lr=3e-3, mult_table=20.0, mult_scalar=10.0, use_muon=True)
    for accum in (1, 2):
        out = []
        for flags in ({}, {"muon_lean": True, "ns_barrier": True}):
            h = fj.Hyper(**base, **flags)
            step = fj.build_train_step(c, h, jnp.float32, 32, False)
            p = fj.init_params(c, 1)
            m = {n: jnp.zeros_like(a) for n, a in p.items()}
            v = {n: jnp.zeros_like(a) for n, a in p.items()}
            sc = jnp.asarray(0, jnp.int32)
            rng = np.random.default_rng(5)
            for _ in range(3):
                b = rng.integers(0, 512, size=(accum * 2, 65)).astype(np.int32)
                t, y, ig = b[:, :-1].reshape(accum, 2, 64), b[:, 1:].reshape(accum, 2, 64), np.zeros((accum, 2, 64), bool)
                p, m, v, sc, loss, gn = step(p, m, v, sc, t, y, ig, jnp.float32(3e-3), jnp.float32(3e-3))
            out.append((p, m, v, float(loss), float(gn)))
        (p0, m0, v0, l0, g0), (p1, m1, v1, l1, g1) = out
        assert all(same(t0[k], t1[k]) for t0, t1 in ((p0, p1), (m0, m1), (v0, v1)) for k in p0) and l0 == l1 and g0 == g1, (accum, l0, l1, g0, g1)
    print("training step with --muon-lean --ns-barrier: parameters, moments, loss and gradient norm bit-identical after 3 steps (1 and 2 micro-steps): ok")


def _shard_body():
    """In a process with 8 CPU devices: the same checks with the Muon update split over the devices (--muon-shard): the lean update with the shard_map against the stacked update with sharding constraints, bit for bit."""
    from jax.sharding import Mesh

    assert len(jax.devices()) == 8, jax.devices()
    mesh = Mesh(np.array(jax.devices()), ("data",))
    check_apply_update_identical(mesh, "muon-shard ")
    c = fj.Cfg(**CFGS["relu2"])
    base = dict(peak_lr=3e-3, muon_lr=3e-3, mult_table=20.0, use_muon=True)
    plain = run_updates(c, fj.Hyper(**base), None)
    lean_sharded = run_updates(c, fj.Hyper(**base, muon_lean=True), mesh)
    worst = max(np.linalg.norm(np.asarray(a[k], np.float64) - np.asarray(b[k], np.float64)) / max(np.linalg.norm(np.asarray(b[k], np.float64)), 1e-30) for a, b in zip(lean_sharded[:3], plain[:3]) for k in a)
    print(f"info: sharded lean update against the replicated stacked update: worst relative difference {worst:.2e} (the sharded Newton-Schulz batches are smaller; not required to be bit-identical)")
    assert worst < 1e-5, worst
    # the program text: the stacked formulation constrains FLOAT32 stacks (the compiler's sharding propagation then shards the elementwise update and gathers float32 values), the lean one splits only the
    # Newton-Schulz iterations in a shard_map (sdy.manual_computation) and constrains bf16 stacks only (a bf16 all-gather of the results)
    p, g, m, v = random_state(c, 3)

    def constraints(**flags):
        h = fj.Hyper(**base, **flags)
        f = jax.jit(lambda *a: fj.apply_update(c, h, *a, jnp.float32(3e-3), jnp.float32(3e-3), mesh))
        text = f.lower(p, g, m, v, jnp.asarray(1, jnp.int32)).as_text()
        return [ln for ln in text.split("\n") if "sdy.sharding_constraint" in ln], text.count("sdy.manual_computation")

    old_c, old_manual = constraints()
    new_c, new_manual = constraints(muon_lean=True)
    assert any("xf32>" in ln for ln in old_c) and old_manual == 0, old_c
    assert new_c and not any("xf32>" in ln for ln in new_c) and all("xbf16>" in ln for ln in new_c) and len(new_c) == 2 * new_manual, (new_c, new_manual)  # per group: the replicated input stack and the gathered result
    print(f"sharded lean update: {new_manual} shard_maps (one per shape group), {len(new_c)} bf16 constraints (the replicated input stack and the gather of every group), no float32 constraint (the stacked update has {sum('xf32>' in ln for ln in old_c)}): ok")
    print("OPT TUNE SHARD OK")


def check_shard():
    env = dict(os.environ, XLA_FLAGS="--xla_force_host_platform_device_count=8", JAX_PLATFORMS="cpu", PYTHONUTF8="1")
    r = subprocess.run([sys.executable, os.path.join(HERE, "test_opt_tune.py"), "shard-body"], capture_output=True, text=True, env=env, timeout=1500)
    print(r.stdout[-2500:], r.stderr[-2500:] if r.returncode else "")
    assert r.returncode == 0 and "OPT TUNE SHARD OK" in r.stdout, r.stderr[-3000:]


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "shard-body":
        _shard_body()
        sys.exit(0)
    check_default_program()
    check_bench_variants()
    check_apply_update_identical()
    check_no_fp32_stacks()
    check_train_step_identical()
    check_shard()
    print("ALL OPT TUNE CHECKS PASSED")
