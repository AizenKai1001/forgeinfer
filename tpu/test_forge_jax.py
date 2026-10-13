"""Checks of forge_jax.py on the CPU against the PyTorch goldens of tools/gen_train_goldens.py (tests/golden/train_tiny) and the Muon goldens (tests/golden/muon).

    python test_forge_jax.py
"""
import json
import math
import os
import struct
import sys

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax
import jax.numpy as jnp

import forge_jax as fj

GOLD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests", "golden")


def read_f32(path):
    return np.fromfile(path, dtype="<f4")


def stats(a, b):
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    rel = np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30)
    cos = float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-30))
    return rel, cos


def load_variant(name):
    d = os.path.join(GOLD, "train_tiny", name)
    man = json.load(open(os.path.join(d, "manifest.json")))
    k = man["config"]
    c = fj.Cfg(n_layer=k["n_layer"], n_embd=k["n_embd"], n_head=k["n_head"], n_kv=k["n_kv"], hd=k["hd"], n_ff=k["n_ff"], vocab=k["vocab"], seq=k["T"], qk_norm=k["qk_norm"], theta=k["theta"],
               untied=k.get("untied", False), relu2=k.get("relu2", False), softcap=k.get("softcap", 0.0), lambdas=k.get("lambdas", False), skips=k.get("skips", False), ve_layers=tuple(k.get("ve_layers", [])))
    params, grads = {}, {}
    for p in man["params"]:
        n = p["name"]
        params[n] = jnp.asarray(read_f32(os.path.join(d, f"{n}.w.f32")).reshape(p["shape"]))
        grads[n] = read_f32(os.path.join(d, f"{n}.g.f32")).reshape(p["shape"])
    tokens = np.fromfile(os.path.join(d, "tokens.u32"), dtype="<u4").reshape(k["B"], k["T"] + 1).astype(np.int32)
    return c, man, params, grads, tokens, d


def check_variant(name):
    c, man, params, gold_grads, tokens, d = load_variant(name)
    assert [n for n, _ in fj.layout(c)] == [p["name"] for p in man["params"]] or set(n for n, _ in fj.layout(c)) == set(p["name"] for p in man["params"]), "parameter names differ"
    x, y = tokens[:, :-1], tokens[:, 1:]
    ig = np.zeros_like(y, bool)
    f = lambda p: fj.loss_fn(c, p, x, y, ig, cdt=jnp.float32, ce_chunk=32, remat=False)
    loss, grads = jax.value_and_grad(f)(params)
    want = man["loss"]
    worst_rel, worst_cos = 0.0, 1.0
    for n, g in gold_grads.items():
        rel, cos = stats(grads[n], g)
        worst_rel, worst_cos = max(worst_rel, rel), min(worst_cos, cos)
    print(f"{name:8s} loss {float(loss):.6f} (PyTorch {want:.6f}); gradients of {len(gold_grads)} tensors: worst relative error {worst_rel:.2e}, worst cosine {worst_cos:.7f}")
    assert abs(float(loss) - want) < 1e-5, (float(loss), want)
    assert worst_cos > 0.99999 and worst_rel < 1e-3, (worst_rel, worst_cos)
    # the same model with bf16 operands and with remat: close to the float32 result
    loss16 = fj.loss_fn(c, params, x, y, ig, cdt=jnp.bfloat16, ce_chunk=32, remat=True)
    assert abs(float(loss16) - want) < 0.03, (float(loss16), want)
    # five AdamW steps as in the golden (lr 1e-2, betas (0.9, 0.95), eps 1e-8, weight decay 0.1 on the embedding and the layer matrices)
    h = fj.Hyper(peak_lr=1e-2, beta2=0.95, wd=0.1, clip=0.0, use_muon=False)
    m = {n: jnp.zeros_like(p) for n, p in params.items()}
    v = {n: jnp.zeros_like(p) for n, p in params.items()}
    p = dict(params)
    losses = []
    for i in range(5):
        l, g = jax.value_and_grad(f)(p)
        losses.append(float(l))
        p, m, v, _ = fj.apply_update(c, h, p, g, m, v, jnp.asarray(i + 1, jnp.int32), 1e-2, 1e-2)
    wl = man["adamw_losses_before_each_step"]
    worst = 1.0
    for n, _ in fj.layout(c):
        want_w = read_f32(os.path.join(d, f"{n}.w5.f32"))
        _, cos = stats(p[n], want_w)
        worst = min(worst, cos)
    print(f"         AdamW: losses {[round(a, 4) for a in losses]} (PyTorch {[round(a, 4) for a in wl]}), worst cosine of the weights after 5 steps {worst:.7f}")
    assert all(abs(a - b) < 2e-4 for a, b in zip(losses, wl)), (losses, wl)
    assert worst > 0.99999


def check_muon():
    d = os.path.join(GOLD, "muon")
    for name in ("wide", "tall", "square", "kv"):
        raw = open(os.path.join(d, f"{name}.f32"), "rb").read()
        r, cc = struct.unpack("<II", raw[:8])
        arr = np.frombuffer(raw[8:], dtype="<f4")
        g = arr[: r * cc].reshape(r, cc)
        want = arr[r * cc :].reshape(r, cc)
        x = jnp.asarray(g) / (jnp.linalg.norm(g) + 1e-7)
        for dt, tol in ((jnp.float32, 1e-3), (jnp.bfloat16, 0.06)):
            got = fj.newton_schulz(x, 5, dt)
            rel, cos = stats(got, want)
            print(f"muon {name:6s} {r}x{cc} {str(dt.__name__):9s}: relative error {rel:.2e}, cosine {cos:.6f}")
            assert rel < tol, (name, dt, rel)


def check_stacked_muon():
    """The batched Muon update of same-shaped matrices equals the matrix-by-matrix update."""
    c = fj.Cfg(n_layer=3, n_embd=64, n_head=2, n_kv=1, hd=32, n_ff=128, vocab=512, seq=64, relu2=True)
    h = fj.Hyper(peak_lr=3e-3, muon_lr=3e-3, use_muon=True)
    p = fj.init_params(c, 4)
    rng = np.random.default_rng(1)
    g = {n: jnp.asarray(rng.standard_normal(a.shape).astype(np.float32)) * 0.05 for n, a in p.items()}
    m = {n: jnp.asarray(rng.standard_normal(a.shape).astype(np.float32)) * 0.01 for n, a in p.items()}
    v = {n: jnp.zeros_like(a) for n, a in p.items()}
    newp, newm, _, gn = fj.apply_update(c, h, p, g, m, v, jnp.asarray(1, jnp.int32), 3e-3, 3e-3)
    clip = min(1.0, h.clip / float(gn))
    for n, shape in fj.layout(c):
        if fj.is_muon_matrix(n, shape):
            want_p, want_m = fj.muon_update(p[n], g[n] * clip, m[n], 3e-3, h.muon_wd, h.mu)
            assert np.allclose(np.asarray(newp[n]), np.asarray(want_p), atol=1e-6, rtol=1e-5), n
            assert np.allclose(np.asarray(newm[n]), np.asarray(want_m), atol=1e-6, rtol=1e-5), n
    print("stacked Muon update equals the per-matrix update: ok")


def _muon_shard_body():
    """Runs in a process with 8 CPU devices: the training step with the Muon update sharded over the devices (--muon-shard) against the replicated one, after 3 steps; stack sizes with and without padding."""
    mesh, batch_sh, _, repl_sh = fj.make_mesh()
    assert len(jax.devices()) == 8, jax.devices()
    h = fj.Hyper(peak_lr=3e-3, muon_lr=3e-3, mult_table=20.0, mult_scalar=10.0, use_muon=True)
    rng = np.random.default_rng(0)
    base = rng.integers(0, 512, size=(16, 65)).astype(np.int32)
    batch = [jax.device_put(x, batch_sh) for x in (base[None, :, :-1], base[None, :, 1:], np.zeros((1, 16, 64), bool))]
    for n_layer in (3, 4):  # 3 layers: stacks of 3 / 6 matrices (padded to 8); 4 layers: wq+wo 8 (no padding), w_up 4 (padded)
        c = fj.Cfg(n_layer=n_layer, n_embd=64, n_head=2, n_kv=2, hd=32, n_ff=128, vocab=512, seq=64, relu2=True, lambdas=True, ve_layers=(0, 1))
        sizes = {}
        for name, shape in fj.layout(c):
            if fj.is_muon_matrix(name, shape):
                sizes[shape] = sizes.get(shape, 0) + 1
        assert any(s % 8 for s in sizes.values()), sizes  # the padding path is used
        results = {}
        texts = {}
        for flag in (False, True):
            p = fj.init_params(c, 1)
            m = {n: jnp.zeros_like(a) for n, a in p.items()}
            v = {n: jnp.zeros_like(a) for n, a in p.items()}
            p, m, v = (jax.device_put(t, repl_sh) for t in (p, m, v))
            sc = jax.device_put(jnp.asarray(0, jnp.int32), repl_sh)
            step = fj.build_train_step(c, h, jnp.float32, 32, False, mesh if flag else None)
            args = (batch[0], batch[1], batch[2], jnp.float32(3e-3), jnp.float32(3e-3))
            texts[flag] = step.lower(p, m, v, sc, *args).as_text()
            for _ in range(3):
                p, m, v, sc, loss, gn = step(p, m, v, sc, *args)
            results[flag] = (p, m, v, float(loss))
        worst = 0.0
        for k in range(3):
            for n in results[False][k]:
                a, b = np.asarray(results[True][k][n], np.float64), np.asarray(results[False][k][n], np.float64)
                worst = max(worst, np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))
        print(f"muon shard {n_layer} layers, stack sizes {sorted(sizes.values())}: worst relative difference of parameters and moments after 3 steps {worst:.2e}, loss {results[True][3]:.5f} vs {results[False][3]:.5f}", flush=True)
        assert worst < 1e-5, worst
        assert abs(results[True][3] - results[False][3]) < 1e-5
        assert 'sharding_constraint' in texts[True] and 'sharding_constraint' not in texts[False]  # the split is in the sharded program only
    z = fj.newton_schulz(jnp.zeros((16, 32), jnp.float32), 5, jnp.bfloat16)
    assert bool(jnp.all(z == 0)) and bool(jnp.all(jnp.isfinite(z)))
    zp, zm = fj.muon_update(jnp.zeros((16, 32)), jnp.zeros((16, 32)), jnp.zeros((16, 32)), 3e-3, 0.05, 0.95)
    assert bool(jnp.all(zp == 0)) and bool(jnp.all(zm == 0))
    print("MUON SHARD OK")


def check_muon_shard():
    import subprocess

    a = fj.parse_args(["train", "--data", "x", "--out", "y"])
    assert a.muon_shard is False and fj.parse_args(["train", "--data", "x", "--out", "y", "--muon-shard"]).muon_shard is True
    here = os.path.dirname(os.path.abspath(__file__))
    env = dict(os.environ, XLA_FLAGS="--xla_force_host_platform_device_count=8", JAX_PLATFORMS="cpu", PYTHONUTF8="1")
    r = subprocess.run([sys.executable, os.path.join(here, "test_forge_jax.py"), "muon-shard-body"], capture_output=True, text=True, env=env, timeout=1500)
    print(r.stdout[-1200:], r.stderr[-1500:] if r.returncode else "")
    assert r.returncode == 0 and "MUON SHARD OK" in r.stdout, r.stderr[-3000:]
    print("sharded Muon update (8 fake devices, padded stacks, zero matrices): ok")


def check_bench_step():
    """tpu/bench_step.py end to end on CPU: a tiny model on 8 fake devices, every variant in its own process, three deliberately failing variants (exception, out-of-memory, a process that dies) are reported without stopping the rest,
    the table is sorted by tokens per second and bench.json is written; a spent deadline skips the variants."""
    import subprocess
    import tempfile

    import bench_step as bs

    assert bs.classify("RESOURCE_EXHAUSTED: Out of memory while trying to allocate") == "oom" and bs.classify("ValueError: x") == "error"
    assert len(bs.failure_text("a\n" * 50 + "b" * 900)) <= 300 and "Unknown flag" in bs.failure_text("noise\nUnknown flag in LIBTPU_INIT_ARGS: --x\nmore")
    assert bs.parse_worker_output('warning\n{"variant": "x", "status": "ok"}\ntrailing') == {"variant": "x", "status": "ok"} and bs.parse_worker_output("nothing") is None
    names = [v[0] for v in bs.variants()]
    assert names[0] == "base" and {"muon-shard", "no-remat", "no-remat+muon-shard", "micro8", "micro8+no-remat", "libtpu-async"} <= set(names), names
    assert {"window512", "muon-shard+window512"} <= set(names) and "window16" in [v[0] for v in bs.variants(True)] and dict((v[0], v[1]) for v in bs.variants())["window512"] == "--window 512 --window-layers auto", names
    # the order of the TPU session: the options that may matter first (a 25-minute deadline must reach the new ones), the unlikely-to-fit no-remat and micro8 rows after them
    assert names[:8] == ["base", "muon-shard", "window512", "muon-shard+window512", "remat-attn", "fuse-qkv", "hd128", "all"] and names.index("no-remat") > names.index("all") and names.index("micro8") > names.index("all"), names
    flags = dict((v[0], v[1]) for v in bs.variants())
    assert flags["remat-attn"] == "--remat-policy dots+attn" and flags["fuse-qkv"] == "--fuse-qkv" and flags["hd128"] == "--heads 8 --kv 2 --head-dim 128", flags
    assert flags["all"] == "--muon-shard --window 512 --window-layers auto --remat-policy dots+attn --fuse-qkv", flags
    assert {"remat-attn", "fuse-qkv", "hd128", "all"} <= set(v[0] for v in bs.variants(True))
    # the remat policies that spend spare HBM come after every earlier variant (and are part of the CPU self-test)
    assert names.index("remat-qkv") < names.index("remat-attn-qkv") < names.index("remat-qkv+muon-shard") < names.index("remat-attn-qkv+muon-shard") and names.index("remat-qkv") > names.index("muon-shard+libtpu-async"), names
    assert flags["remat-qkv"] == "--remat-policy dots+qkv" and flags["remat-attn-qkv"] == "--remat-policy dots+attn+qkv", flags
    assert flags["remat-qkv+muon-shard"] == "--muon-shard --remat-policy dots+qkv" and flags["remat-attn-qkv+muon-shard"] == "--muon-shard --remat-policy dots+attn+qkv", flags
    assert {"remat-qkv", "remat-attn-qkv", "remat-qkv+muon-shard", "remat-attn-qkv+muon-shard"} <= set(v[0] for v in bs.variants(True))
    # the whole table runs in priority order (the best candidates of the compile lab first: a 25-minute session gets through about a dozen variants); the table itself keeps its historical order
    po = [v[0] for v in bs.priority_order(bs.variants())]
    assert po[:6] == ["base", "muon-shard", "attn-lane", "attn-lane+muon-shard", "lane+shard+lean", "lane+shard+lean+ffnbf16"] and sorted(po) == sorted(names), po
    assert names.index("window512") < names.index("remat-attn") and po.index("window512") < po.index("fuse-qkv")
    here = os.path.dirname(os.path.abspath(__file__))
    env = dict(os.environ, JAX_PLATFORMS="cpu", PYTHONUTF8="1")
    env.pop("XLA_FLAGS", None)  # the benchmark sets the 8 fake devices itself with --tiny
    with tempfile.TemporaryDirectory() as tmp:
        r = subprocess.run([sys.executable, os.path.join(here, "bench_step.py"), "--tiny", "--steps", "3", "--out", tmp], capture_output=True, text=True, env=env, timeout=900)
        assert r.returncode == 0, r.stderr[-2000:]
        doc = json.load(open(os.path.join(tmp, "bench.json")))
        by = {row["variant"]: row for row in doc["table"]}
        for n in ("base", "muon-shard", "no-remat", "micro8", "libtpu-async", "window16", "remat-attn", "fuse-qkv", "hd128", "all", "remat-qkv", "remat-attn-qkv", "remat-qkv+muon-shard", "remat-attn-qkv+muon-shard"):
            row = by[n]
            assert row["status"] == "ok" and row["devices"] == 8 and row["tokens_per_s"] > 0 and row["median_s"] >= row["min_s"] > 0 and row["gain_pct"] is not None, row
        bd = by["breakdown-base"]  # the time breakdown: forward, backward and optimizer as their own programs
        assert bd["status"] == "ok" and {"forward_ms", "backward_ms", "optimizer_ms", "sum_ms"} <= set(bd["breakdown"]) and all(bd["breakdown"][k] > 0 for k in ("forward_ms", "optimizer_ms")), bd
        assert "optimizer" in bd["breakdown"]["text"]
        assert by["base"]["gain_pct"] == 0.0 and by["muon-shard"]["muon_shard"] is True and by["no-remat"]["remat"] is False and by["micro8"]["tokens_per_step"] == 8 * 8 * 64, by
        assert by["window16"]["window"] == 16 and by["window16"]["window_layers"] == [0, 1] and by["base"]["window"] == 0 and by["base"]["window_layers"] == [], by["window16"]  # 3 layers: 0 and 1 windowed, the last dense
        assert by["remat-qkv"]["remat_policy"] == "dots+qkv" and by["remat-attn-qkv"]["remat_policy"] == "dots+attn+qkv" and by["remat-qkv+muon-shard"]["muon_shard"] is True and by["remat-qkv+muon-shard"]["remat_policy"] == "dots+qkv" and by["remat-attn-qkv+muon-shard"]["muon_shard"] is True, by["remat-qkv"]
        assert by["remat-attn"]["remat_policy"] == "dots+attn" and by["base"]["remat_policy"] == "dots" and by["fuse-qkv"]["fuse_qkv"] is True and by["base"]["fuse_qkv"] is False, by["remat-attn"]
        assert "--head-dim 64" in by["hd128"]["args"] and by["all"]["muon_shard"] is True and by["all"]["window"] == 16 and by["all"]["remat_policy"] == "dots+attn" and by["all"]["fuse_qkv"] is True, by["all"]
        assert by["selftest-error"]["status"] == "error" and "deliberate" in by["selftest-error"]["message"]
        assert by["selftest-oom"]["status"] == "oom" and by["selftest-crash"]["status"] == "error" and "Unknown flag" in by["selftest-crash"]["message"], by
        ok_rows = [row["tokens_per_s"] for row in doc["table"] if row["status"] == "ok"]
        assert ok_rows == sorted(ok_rows, reverse=True) and all(row["status"] != "ok" for row in doc["table"][len(ok_rows):])
        assert "selftest-crash" in r.stdout and "tokens/s" in r.stdout
        print(f"bench_step: {len(ok_rows)} variants ok, 3 deliberate failures reported (error, oom, crash), table sorted, bench.json written: ok")
    with tempfile.TemporaryDirectory() as tmp:
        r = subprocess.run([sys.executable, os.path.join(here, "bench_step.py"), "--tiny", "--deadline-s", "10", "--out", tmp], capture_output=True, text=True, env=env, timeout=300)
        assert r.returncode == 0, r.stderr[-1500:]
        doc = json.load(open(os.path.join(tmp, "bench.json")))
        assert doc["table"] and all(row["status"] == "skipped" for row in doc["table"]), doc["table"]
        print("bench_step: a spent deadline skips the variants: ok")


def check_kaggle_bench_runner():
    """cloud/kaggle_tpu_bench.py in a fake Kaggle environment (input and output directories from FORGE_KAGGLE_IN / FORGE_KAGGLE_OUT, a tiny model on fake CPU devices, no pip): the happy path, no accelerator, too little time, missing code."""
    import shutil
    import subprocess
    import tempfile

    here = os.path.dirname(os.path.abspath(__file__))
    runner = os.path.join(here, "..", "cloud", "kaggle_tpu_bench.py")

    def run(settings, tiny, copy_bench=True):
        with tempfile.TemporaryDirectory() as tmp:
            inp, out = os.path.join(tmp, "input"), os.path.join(tmp, "working")
            os.makedirs(os.path.join(inp, "forge-jax"))
            os.makedirs(out)
            shutil.copy(os.path.join(here, "forge_jax.py"), os.path.join(inp, "forge-jax", "forge_jax.py"))
            if copy_bench:
                shutil.copy(os.path.join(here, "bench_step.py"), os.path.join(inp, "forge-jax", "bench_step.py"))
            if settings is not None:
                os.makedirs(os.path.join(inp, "forge-tpujob-bench"))
                json.dump(settings, open(os.path.join(inp, "forge-tpujob-bench", "settings.json"), "w"))
            env = dict(os.environ, FORGE_KAGGLE_IN=inp, FORGE_KAGGLE_OUT=out, FORGE_KAGGLE_NO_PIP="1", JAX_PLATFORMS="cpu", PYTHONUTF8="1")
            env.pop("XLA_FLAGS", None)
            if tiny:
                env["FORGE_BENCH_TINY"] = "1"
            r = subprocess.run([sys.executable, runner], capture_output=True, text=True, env=env, timeout=900)
            files = {f: open(os.path.join(out, f)).read() for f in os.listdir(out) if os.path.isfile(os.path.join(out, f))}
            return r, files

    r, files = run({"only": "base,muon-shard", "total_s": 600}, tiny=True)
    assert r.returncode == 0, r.stderr[-2000:]
    assert files["exit.txt"].strip() == "0" and "bench.log" in files and "env.txt" in files, sorted(files)
    doc = json.loads(files["bench.json"])
    assert [row["variant"] for row in doc["table"] if row["status"] == "ok"] and {row["variant"] for row in doc["table"]} == {"base", "muon-shard"}, doc["table"]
    assert "the base variant was measured" in files["env.txt"] and "tokens/s" in files["bench.log"]
    r, files = run(None, tiny=False)
    assert r.returncode == 0 and files["exit.txt"].strip() == "no accelerator" and "bench.json" not in files, (r.stderr[-1000:], files)
    r, files = run({"total_s": 100}, tiny=True)
    assert files["exit.txt"].strip() == "too little time" and "bench.json" not in files, files
    r, files = run({}, tiny=True, copy_bench=False)
    assert files["exit.txt"].strip() == "no benchmark code", files
    print("kaggle bench runner (fake Kaggle environment): the run, no accelerator, too little time, missing code: ok")


def check_checkpoint_roundtrip():
    c = fj.Cfg(n_layer=3, n_embd=64, n_head=2, n_kv=1, hd=64, n_ff=128, vocab=256, seq=64, relu2=True, lambdas=True, ve_layers=(0, 2))
    p = fj.init_params(c, 3)
    m = {n: jnp.full(s, 0.5, jnp.float32) for n, s in fj.layout(c)}
    v = {n: jnp.full(s, 0.25, jnp.float32) for n, s in fj.layout(c)}
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_tmp_ckpt.bin")
    fj.save_checkpoint(path, c, p, m, v, 17)
    c2 = fj.cfg_from_checkpoint(path)
    assert c2 == c, (c2, c)
    p2, m2, v2, step = fj.load_checkpoint(path, c2)
    assert step == 17 and all(np.array_equal(np.asarray(p[n]), np.asarray(p2[n])) for n in p) and float(m2["emb"][0, 0]) == 0.5 and float(v2["emb"][0, 0]) == 0.25
    os.remove(path)
    print("checkpoint round trip: ok (header, parameters, moments)")


def check_window_checkpoint():
    """The header carries the window keys as the Rust writer does (`window`, `window_layers` at the end of cfg), a round trip returns the same Cfg, and a header of an older file without the keys loads as window 0."""
    import tempfile

    c = fj.Cfg(n_layer=5, n_embd=64, n_head=2, n_kv=1, hd=64, n_ff=128, vocab=256, seq=64, relu2=True, lambdas=True, ve_layers=(0, 2), window=32, window_layers=fj.auto_window_layers(5))
    assert c.window_layers == (0, 1, 2), c.window_layers
    p = fj.init_params(c, 3)
    m = {n: jnp.full(s, 0.5, jnp.float32) for n, s in fj.layout(c)}
    v = {n: jnp.full(s, 0.25, jnp.float32) for n, s in fj.layout(c)}
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "ckpt.bin")
        fj.save_checkpoint(path, c, p, m, v, 9)
        with open(path, "rb") as f:
            head = json.loads(f.readline())
        assert head["cfg"]["window"] == 32 and head["cfg"]["window_layers"] == [0, 1, 2], head["cfg"]
        assert list(head["cfg"])[-2:] == ["window", "window_layers"]  # the position of the keys in the Rust writer's cfg object (after emb_norm; the JAX header has no keys for the options it lacks)
        c2 = fj.cfg_from_checkpoint(path)
        assert c2 == c and isinstance(c2.window_layers, tuple), (c2, c)
        p2, m2, v2, step = fj.load_checkpoint(path, c2)
        assert step == 9 and all(np.array_equal(np.asarray(p[n]), np.asarray(p2[n])) for n in p)
        # a dense model writes window 0 and an empty list, as the Rust trainer does
        cd = fj.Cfg(**{**c.__dict__, "window": 0, "window_layers": ()})
        hd = json.loads(fj.header_of(cd, 1, 0))["cfg"]
        assert hd["window"] == 0 and hd["window_layers"] == [], hd
        # an older header (before this change) has neither key: window 0, no window layers
        old = json.loads(fj.header_of(c, sum(int(np.prod(s)) for _, s in fj.layout(c)), 4))
        del old["cfg"]["window"], old["cfg"]["window_layers"]
        path_old = os.path.join(tmp, "old.bin")
        with open(path, "rb") as f:
            f.readline()
            body = f.read()
        with open(path_old, "wb") as f:
            f.write((json.dumps(old, separators=(",", ":")) + "\n").encode() + body)
        c3 = fj.cfg_from_checkpoint(path_old)
        assert c3.window == 0 and c3.window_layers == () and all(c3.window_of(l) is None for l in range(c3.n_layer)), c3
        assert c3 == fj.Cfg(**{**c.__dict__, "window": 0, "window_layers": ()})
        _, _, _, step3 = fj.load_checkpoint(path_old, c3)
        assert step3 == 4
    print("window checkpoint: header keys, round trip, an old header loads as window 0: ok")


def _brute_force_attention(c, q, k, v, window):
    """Causal attention of one head at a time in float64 with a plain double loop over the allowed keys: query i sees the keys j with j <= i and (window is None or i - j < window)."""
    b, t = q.shape[:2]
    rep = c.n_head // c.n_kv
    out = np.zeros((b, t, c.n_head, c.hd))
    for bi in range(b):
        for h in range(c.n_head):
            g = h // rep
            for i in range(t):
                js = [j for j in range(i + 1) if window is None or i - j < window]
                s = np.array([q[bi, i, h] @ k[bi, j, g] for j in js]) / math.sqrt(c.hd)
                e = np.exp(s - s.max())
                out[bi, i, h] = (e / e.sum()) @ np.stack([v[bi, j, g] for j in js])
    return out.reshape(b, t, c.q_dim)


def check_window_attention():
    """XLA's windowed attention equals a brute-force masked softmax (grouped keys/values), the mask is the Rust trainer's (i - window < j <= i), a window of at least T is the dense case, and window 1 sees only itself."""
    c = fj.Cfg(n_layer=1, n_embd=64, n_head=4, n_kv=2, hd=16, n_ff=64, vocab=64, seq=40)
    rng = np.random.default_rng(11)
    b, t = 2, c.seq
    q = rng.standard_normal((b, t, c.n_head, c.hd)).astype(np.float32)
    k = rng.standard_normal((b, t, c.n_kv, c.hd)).astype(np.float32)
    v = rng.standard_normal((b, t, c.n_kv, c.hd)).astype(np.float32)
    for window in (None, 1, 2, 7, 16, 39, 40, 100):
        got = np.asarray(fj.attention_xla(c, jnp.asarray(q), jnp.asarray(k), jnp.asarray(v), jnp.float32, window), np.float64)
        want = _brute_force_attention(c, q.astype(np.float64), k.astype(np.float64), v.astype(np.float64), window)
        err = np.abs(got - want).max()
        assert err < 2e-5, (window, err)
    dense = np.asarray(fj.attention_xla(c, jnp.asarray(q), jnp.asarray(k), jnp.asarray(v), jnp.float32))
    for window in (40, 41, 100):  # window >= T is the dense case
        assert np.allclose(np.asarray(fj.attention_xla(c, jnp.asarray(q), jnp.asarray(k), jnp.asarray(v), jnp.float32, window)), dense, atol=1e-6)
    one = np.asarray(fj.attention_xla(c, jnp.asarray(q), jnp.asarray(k), jnp.asarray(v), jnp.float32, 1)).reshape(b, t, c.n_head, c.hd)
    assert np.allclose(one, np.repeat(v, c.n_head // c.n_kv, axis=2), atol=1e-6)  # one key: itself, probability 1
    # the key at distance exactly `window` is NOT seen, the key at distance window - 1 is (changing it changes the output of the last query)
    window = 5
    base = np.asarray(fj.attention_xla(c, jnp.asarray(q), jnp.asarray(k), jnp.asarray(v), jnp.float32, window))[:, -1]
    for dist, seen in ((window, False), (window - 1, True)):
        v2 = v.copy()
        v2[:, t - 1 - dist] += 1.0
        moved = np.asarray(fj.attention_xla(c, jnp.asarray(q), jnp.asarray(k), jnp.asarray(v2), jnp.float32, window))[:, -1]
        assert (np.abs(moved - base).max() > 1e-3) == seen, (dist, seen)
    print("window attention: equals a brute-force masked softmax (windows 1 to 100, grouped keys/values), i - window < j <= i, window >= T is dense: ok")


def check_window_layer_rule():
    """`--window-layers auto` is the rule of crates/forge-train (auto_window_layers: l % 4 != 3 and l + 1 != n_layer), the layer list parsing is the Rust CLI's, and a layer is windowed only if the window is shorter than the sequence."""
    assert fj.auto_window_layers(1) == () and fj.auto_window_layers(2) == (0,) and fj.auto_window_layers(4) == (0, 1, 2) and fj.auto_window_layers(5) == (0, 1, 2)
    assert fj.auto_window_layers(12) == (0, 1, 2, 4, 5, 6, 8, 9, 10)  # 3, 7, 11 dense (11 is also the last layer)
    assert fj.auto_window_layers(13) == (0, 1, 2, 4, 5, 6, 8, 9, 10)  # layer 12 is the last one
    assert fj.auto_window_layers(14) == (0, 1, 2, 4, 5, 6, 8, 9, 10, 12)
    n30 = fj.auto_window_layers(30)  # the 150M model: 22 windowed layers, 7 + the last one dense
    assert len(n30) == 22 and all(l % 4 != 3 and l != 29 for l in n30) and set(range(30)) - set(n30) == {3, 7, 11, 15, 19, 23, 27, 29}
    assert fj.parse_window_layers("auto", 512, 8) == (0, 1, 2, 4, 5, 6) and fj.parse_window_layers("", 512, 8) == (0, 1, 2, 4, 5, 6)
    assert fj.parse_window_layers("auto", 0, 8) == () and fj.parse_window_layers("0,3,5", 512, 8) == (0, 3, 5) and fj.parse_window_layers("0,3", 0, 8) == (0, 3)
    for bad in ("8", "0,9", "-1"):
        try:
            fj.parse_window_layers(bad, 512, 8)
            raise AssertionError(bad)
        except ValueError:
            pass
    a = fj.parse_args(["train", "--data", "d", "--out", "o"])
    assert a.window == 0 and a.window_layers == "auto"  # off by default
    a = fj.parse_args(["train", "--data", "d", "--out", "o", "--window", "512", "--window-layers", "0,1"])
    assert a.window == 512 and a.window_layers == "0,1"
    c = fj.Cfg(n_layer=8, seq=2048, window=512, window_layers=fj.auto_window_layers(8))
    assert [c.window_of(l) for l in range(8)] == [512, 512, 512, None, 512, 512, 512, None]
    assert all(fj.Cfg(**{**c.__dict__, "window": 2048}).window_of(l) is None for l in range(8))  # window >= T: dense (the Rust trainer: window_of(l).filter(|&w| w < t))
    assert fj.Cfg(**{**c.__dict__, "window": 2047}).window_of(0) == 2047 and c.window_of(0, 512) is None and c.window_of(0, 513) == 512
    assert all(fj.Cfg(**{**c.__dict__, "window": 0}).window_of(l) is None for l in range(8))  # window 0: every layer dense whatever the list says
    print("window layers: auto = the Rust rule, list parsing, window >= T is dense, window 0 is off: ok")


def _window_hlo_hash(window, window_layers):
    """SHA-256 prefix of the lowered training step of a tiny model on one device (the idea of scratch/hlo_hash.py)."""
    import hashlib

    c = fj.Cfg(n_layer=3, n_embd=64, n_head=2, n_kv=2, hd=32, n_ff=128, vocab=512, seq=64, relu2=True, lambdas=True, ve_layers=(0, 1), window=window, window_layers=window_layers)
    h = fj.Hyper(peak_lr=3e-3, muon_lr=3e-3, mult_table=20.0, mult_scalar=10.0, use_muon=True)
    p = fj.init_params(c, 1)
    m = {n: jnp.zeros_like(a) for n, a in p.items()}
    v = {n: jnp.zeros_like(a) for n, a in p.items()}
    tok = jnp.zeros((1, 2, 64), jnp.int32)
    ign = jnp.zeros((1, 2, 64), bool)
    step = fj.build_train_step(c, h, jnp.float32, 32, False)
    text = step.lower(p, m, v, jnp.asarray(0, jnp.int32), tok, tok, ign, jnp.float32(3e-3), jnp.float32(3e-3)).as_text()
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def check_window_off_is_identical():
    """--window 0 changes nothing: the lowered training step is the same program whatever the layer list says, and (jax 0.11.2) the same as the one of the trainer before the window option existed (hash of the commit before it)."""
    off = _window_hlo_hash(0, ())
    assert _window_hlo_hash(0, (0, 1)) == off, "a layer list without a window must not change the program"
    assert _window_hlo_hash(64, (0, 1)) == off, "a window of the sequence length is the dense case: the same program"
    on = _window_hlo_hash(16, (0, 1))
    assert on != off, "a window of 16 must change the program"
    if jax.__version__ == "0.11.2":
        # sha256 of step.lower(...).as_text() with one micro-step. Before the accum == 1 shortcut of build_train_step (main at 0abc35a, before the window option, up to 9b15b7d) it was 696cb7f65e3bff5f, the program with
        # the zero accumulator, the scan over one micro-step and the division by 1 (check_accum1_step keeps that reference step and proves the new one bit-identical)
        assert off == "dc6c181a9996e0c1", off
    print(f"window off: identical program ({off}), windowed program differs ({on}): ok")


def check_window_model():
    """The model with windowed layers: it differs from the dense one, window >= T equals dense, and a token further back than the reach of the windowed layers does not change the last position of a model whose every layer is windowed."""
    c = fj.Cfg(n_layer=2, n_embd=64, n_head=2, n_kv=1, hd=32, n_ff=128, vocab=256, seq=48, relu2=True, lambdas=True)
    p = fj.init_params(c, 5)
    p = {n: (a + 0.05 * jax.random.normal(jax.random.PRNGKey(i), a.shape) if n != "lam" else a) for i, (n, a) in enumerate(p.items())}
    tok = jnp.asarray(np.random.default_rng(1).integers(0, 256, (2, 48)), jnp.int32)
    dense = np.asarray(fj.hidden(c, p, tok, jnp.float32, False))
    full = fj.Cfg(**{**c.__dict__, "window": 48, "window_layers": (0, 1)})
    assert np.array_equal(np.asarray(fj.hidden(full, p, tok, jnp.float32, False)), dense)
    local = fj.Cfg(**{**c.__dict__, "window": 4, "window_layers": (0, 1)})
    got = np.asarray(fj.hidden(local, p, tok, jnp.float32, False))
    assert np.abs(got - dense).max() > 1e-3
    assert np.array_equal(np.asarray(fj.hidden(local, p, tok, jnp.float32, True)).shape, got.shape)  # with remat too
    # two windowed layers of window 4 reach back 2 * 3 = 6 positions: a token 7 positions before the last one cannot matter, one 6 positions before can
    for back, matters in ((7, False), (6, True)):
        tok2 = tok.at[:, 47 - back].set((tok[:, 47 - back] + 1) % 256)
        moved = np.asarray(fj.hidden(local, p, tok2, jnp.float32, False))
        assert (np.abs(moved[:, -1] - got[:, -1]).max() > 1e-5) == matters, (back, matters)
    # in the dense model the same token matters
    tok2 = tok.at[:, 40].set((tok[:, 40] + 1) % 256)
    assert np.abs(np.asarray(fj.hidden(c, p, tok2, jnp.float32, False))[:, -1] - dense[:, -1]).max() > 1e-5
    # only some layers windowed (layer 0): the reach is not limited by layer 1, a dense layer
    some = fj.Cfg(**{**c.__dict__, "window": 4, "window_layers": (0,)})
    tok2 = tok.at[:, 40].set((tok[:, 40] + 1) % 256)
    assert np.abs(np.asarray(fj.hidden(some, p, tok2, jnp.float32, False))[:, -1] - np.asarray(fj.hidden(some, p, tok, jnp.float32, False))[:, -1]).max() > 1e-5
    print("window model: differs from dense, window >= T is dense, the receptive field of the windowed layers is exactly layers x (window - 1): ok")


def check_window_end_to_end():
    """A tiny training run on 2 fake devices with --window 16 --window-layers auto (2 layers: layer 0 windowed, layer 1 dense): the loss goes down, the log names the window, the checkpoint carries the window keys and a resume continues."""
    import subprocess
    import tempfile

    here = os.path.dirname(os.path.abspath(__file__))
    script = os.path.join(here, "forge_jax.py")
    with tempfile.TemporaryDirectory() as tmp:
        out, args, env = _tiny_run_args(tmp, ["--steps", "30", "--window", "16", "--window-layers", "auto"])
        r = subprocess.run([sys.executable, script] + args, capture_output=True, text=True, env=env, timeout=600)
        assert r.returncode == 0, r.stderr[-2000:]
        assert "sliding-window attention: window 16 in layers [0]" in r.stdout, r.stdout[-1500:]
        vals = [float(l.split("val ")[1].split()[0]) for l in r.stdout.splitlines() if " val " in l]
        losses = [float(l.split(" loss ")[1].split()[0]) for l in r.stdout.splitlines() if l.startswith("step ")]
        assert len(vals) >= 3 and vals[-1] < vals[0] - 0.3 and losses[-1] < losses[0] - 0.5, (vals, losses)
        ck = os.path.join(out, "ckpt_latest.bin")
        with open(ck, "rb") as f:
            cfg = json.loads(f.readline())["cfg"]
        assert cfg["window"] == 16 and cfg["window_layers"] == [0], cfg
        c = fj.cfg_from_checkpoint(ck)
        assert c.window == 16 and c.window_layers == (0,) and [c.window_of(l) for l in range(2)] == [16, None]
        args2 = [x if x != "30" else "40" for x in args]
        r2 = subprocess.run([sys.executable, script] + args2 + ["--resume"], capture_output=True, text=True, env=env, timeout=600)
        assert r2.returncode == 0 and "resumed from step 30" in r2.stdout and "step    40/40" in r2.stdout, r2.stdout[-1200:]
        # a layer list outside the model is refused with exit code 2
        out2, args3, env = _tiny_run_args(tempfile.mkdtemp(dir=tmp), ["--steps", "3", "--window", "16", "--window-layers", "0,5"])
        r3 = subprocess.run([sys.executable, script] + args3, capture_output=True, text=True, env=env, timeout=300)
        assert r3.returncode == 2 and "layer indices of the model" in r3.stdout, (r3.returncode, r3.stdout[-500:])
        print(f"window end to end (2 devices, --window 16 --window-layers auto): validation {vals[0]:.3f} -> {vals[-1]:.3f}, checkpoint keys, resume, bad layer list refused: ok")


def check_training_runs():
    """A few optimizer steps with Muon on a tiny model: the loss goes down, the step is deterministic and jit-compatible."""
    c = fj.Cfg(n_layer=2, n_embd=64, n_head=2, n_kv=1, hd=32, n_ff=128, vocab=512, seq=64, relu2=True, lambdas=True, ve_layers=(0, 1))
    h = fj.Hyper(peak_lr=3e-3, muon_lr=3e-3, mult_table=20.0, mult_scalar=10.0, use_muon=True)
    p = fj.init_params(c, 1)
    m = {n: jnp.zeros_like(a) for n, a in p.items()}
    v = {n: jnp.zeros_like(a) for n, a in p.items()}
    step = fj.build_train_step(c, h, jnp.float32, 32, False)
    rng = np.random.default_rng(0)
    base = rng.integers(0, 512, size=(8, 65)).astype(np.int32)  # a fixed batch: the loss must fall when it is fitted
    x, y, ig = base[None, :, :-1], base[None, :, 1:], np.zeros((1, 8, 64), bool)  # one micro-step of 8 sequences
    sc = jnp.asarray(0, jnp.int32)
    losses = []
    for i in range(25):
        p, m, v, sc, loss, gn = step(p, m, v, sc, x, y, ig, jnp.float32(3e-3), jnp.float32(3e-3))
        losses.append(float(loss))
    print("training steps (Muon + AdamW groups):", [round(a, 3) for a in losses[::6]], "... grad norm", round(float(gn), 3))
    assert losses[-1] < losses[0] - 0.5, losses


def check_end_to_end():
    """The training command on 8 fake CPU devices: data parallel, gradient accumulation, evaluation, checkpoint, resume."""
    import subprocess
    import tempfile

    here = os.path.dirname(os.path.abspath(__file__))
    with tempfile.TemporaryDirectory() as tmp:
        data = os.path.join(tmp, "data")
        os.makedirs(data)
        rng = np.random.default_rng(5)
        # a learnable stream: a repeating pattern of 97 tokens with some noise
        base = (np.arange(400_000) % 97).astype(np.int64)
        base = np.where(rng.random(400_000) < 0.05, rng.integers(0, 512, 400_000), base)
        base.astype("<u2").tofile(os.path.join(data, "a_000.u16"))
        base[:100_000].astype("<u2").tofile(os.path.join(data, "zzz_val.u16"))
        out = os.path.join(tmp, "run")
        common = ["--data", data, "--out", out, "--layers", "2", "--dim", "64", "--heads", "2", "--kv", "1", "--head-dim", "32", "--ffn", "128", "--vocab", "512", "--seq", "64", "--tokens-per-step", "2048", "--micro", "2",
                  "--lr", "3e-3", "--optim", "muon", "--muon-lr", "3e-3", "--warmup", "5", "--qk-norm", "--relu2", "--lambdas", "--value-emb", "0,1", "--lr-table-mult", "20", "--eval-every", "10", "--eval-batches", "2",
                  "--save-every", "10", "--val-tokens", "30000", "--dtype", "f32", "--ce-chunk", "32"]
        env = dict(os.environ, XLA_FLAGS="--xla_force_host_platform_device_count=8", JAX_PLATFORMS="cpu", PYTHONUTF8="1")
        r = subprocess.run([sys.executable, os.path.join(here, "forge_jax.py"), "train", "--steps", "30"] + common, capture_output=True, text=True, env=env, timeout=600)
        print(r.stdout[-1500:], r.stderr[-800:] if r.returncode else "")
        assert r.returncode == 0, r.stderr[-2000:]
        assert "cpu x 8" in r.stdout and "micro-steps" in r.stdout
        vals = [float(l.split("val ")[1]) for l in r.stdout.splitlines() if " val " in l]
        assert vals[-1] < vals[0] - 0.3, vals
        assert os.path.getsize(os.path.join(out, "ckpt_latest.bin")) > 1000
        r2 = subprocess.run([sys.executable, os.path.join(here, "forge_jax.py"), "train", "--steps", "40", "--resume"] + common, capture_output=True, text=True, env=env, timeout=600)
        assert r2.returncode == 0, r2.stderr[-2000:]
        assert "resumed from step 30" in r2.stdout and "step    40/40" in r2.stdout, r2.stdout[-800:]
        print("end to end (8 devices, accumulation, evaluation, checkpoint, resume): ok; validation losses", [round(v, 3) for v in vals])


def check_plan_steps():
    """The step planning of runs with a time limit."""
    # fast enough: the whole schedule fits
    assert fj.plan_steps(45000, 11000, 6000.0, 0.5, 0.02, 29000.0) == 45000
    # slower: (29000 - 6000 - 90) s / (0.8 + 0.02) s = 27939 more steps
    assert fj.plan_steps(45000, 11000, 6000.0, 0.8, 0.02, 29000.0) == 11000 + 27939
    # no time left: the steps done so far
    assert fj.plan_steps(45000, 11000, 29000.0, 0.8, 0.02, 29000.0) == 11000
    # a measured mean wall time per step (stalls included) above the quiet-moment estimate wins, a smaller one changes nothing
    assert fj.plan_steps(45000, 11000, 6000.0, 0.5, 0.02, 29000.0, 90.0, 0.8) == 11000 + int((29000 - 6000 - 90) / 0.8)
    assert fj.plan_steps(45000, 11000, 6000.0, 0.8, 0.02, 29000.0, 90.0, 0.5) == 11000 + 27939
    # replan_steps: the new total when it moves by a third of a percent or more, None below that, None once the decay (it starts at 63000 for a plan of 70000) is near or has started
    assert fj.replan_steps(70000, 100000, 40000, 16000.0, 0.38, 0.0, 28000.0, 100.0, 0.1) == 40000 + int((28000 - 16000 - 100) / 0.38)
    assert fj.replan_steps(71300, 100000, 40000, 16000.0, 0.38, 0.0, 28000.0, 100.0, 0.1) is None
    assert fj.replan_steps(70000, 100000, 62990, 24000.0, 0.38, 0.0, 28000.0, 100.0, 0.1) is None
    assert fj.replan_steps(70000, 100000, 64000, 24000.0, 0.38, 0.0, 28000.0, 100.0, 0.1) is None
    print("step planning: ok")


def check_plan_with_stalls():
    """The 365M run on Kaggle (2026-10-11): 0.3848 s per step including the normal evaluation and checkpoint share, five stalls of 4 to 6 minutes, a limit of 28000 s. The planning before the fix (medians only, a 90 s reserve, a
    hysteresis of one percent) stopped it at step 68524 of 69796 in the middle of the decay; the planning now must end the plan inside the limit."""
    limit, steps_max, decay, per_step = 28000.0, 100000, 0.1, 0.3848
    stalls = {32000: 338.0, 36000: 351.0, 40000: 354.0, 52000: 237.0, 66000: 330.0}  # seconds lost after the step

    def run(new_style):
        t, step, plan, ref = 33.0, 0, steps_max, None
        while step < plan:
            t += per_step
            step += 1
            t += stalls.get(step, 0.0)
            if t > limit:
                return step, plan, t, False
            if ref is None and step >= 5:
                ref = (t, step)
            if step % 100 == 0 and step >= 60:
                wall = (t - ref[0]) / (step - ref[1]) if step - ref[1] >= 200 else 0.0
                if new_style:
                    new = fj.replan_steps(plan, steps_max, step, t, 0.379, 0.005, limit, 600.0, decay, wall)
                else:  # the planning before the fix
                    new = fj.plan_steps(steps_max, step, t, 0.379, 0.005, limit, 90.0)
                    if not (int((1 - decay) * plan) > step + 25 and new != plan and int((1 - decay) * new) > step + 25 and abs(new - plan) >= max(10, plan // 100)):
                        new = None
                if new is not None:
                    plan = new
        return step, plan, t, True

    step, plan, t, ok = run(False)
    assert not ok and step < plan, (step, plan, t)  # the old planning stops before the end of the decay
    old_step, old_plan = step, plan
    step, plan, t, ok = run(True)
    assert ok and step == plan and t < limit - 200, (step, plan, t)
    print(f"planning with stalls: before the fix stopped at step {old_step} of {old_plan}; now a plan of {plan} steps ends at {t:.0f} s of {limit:.0f} s: ok")


def _splash_body():
    """Runs in a process with two CPU devices: the splash kernel (Pallas interpreter) against the XLA attention, value and gradients, grouped keys/values, the batch split over the devices."""
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    c = fj.Cfg(n_layer=2, n_embd=256, n_head=4, n_kv=2, hd=64, n_ff=128, vocab=256, seq=256)
    devs = jax.devices()
    assert len(devs) == 2, devs
    mesh = Mesh(np.array(devs), ("data",))
    sh = NamedSharding(mesh, P("data"))
    keys = jax.random.split(jax.random.PRNGKey(3), 4)
    b, t = 4, c.seq
    q = jax.device_put(jax.random.normal(keys[0], (b, t, c.n_head, c.hd), jnp.float32), sh)
    k = jax.device_put(jax.random.normal(keys[1], (b, t, c.n_kv, c.hd), jnp.float32), sh)
    v = jax.device_put(jax.random.normal(keys[2], (b, t, c.n_kv, c.hd), jnp.float32), sh)
    w = jax.device_put(jax.random.normal(keys[3], (b, t, c.q_dim), jnp.float32), sh)

    def make(impl):
        return jax.jit(jax.value_and_grad(lambda q_, k_, v_: jnp.sum(impl(c, q_, k_, v_, jnp.bfloat16) * w), argnums=(0, 1, 2)))

    ref = make(fj.attention_xla)(q, k, v)
    for blk, fused in ((128, False), (128, True)):
        out = make(fj.make_splash_attention(c, mesh, blk, fused, interpret=True))(q, k, v)
        errs = fj._rel_errors(ref, out)
        print(f"splash attention block {blk} fused={fused}: relative errors value/dq/dk/dv {errs}", flush=True)
        assert max(errs) < 0.05, errs
    # the sliding-window kernel (LocalMask) against XLA windowed attention: a window inside a block (100), exactly one block (128) and a window longer than one block (200)
    for window, blk, fused in ((100, 128, False), (100, 128, True), (128, 128, False), (200, 128, False)):
        cw = fj.Cfg(**{**c.__dict__, "window": window, "window_layers": (0,)})
        assert cw.window_of(0) == window and cw.window_of(1) is None
        refw = make(lambda c_, q_, k_, v_, cdt_, window=window: fj.attention_xla(c_, q_, k_, v_, cdt_, window))(q, k, v)
        splash = fj.make_splash_attention(cw, mesh, blk, fused, interpret=True)
        outw = make(lambda c_, q_, k_, v_, cdt_: splash(c_, q_, k_, v_, cdt_, local=True))(q, k, v)
        errs = fj._rel_errors(refw, outw)
        print(f"splash attention, window {window}, block {blk} fused={fused}: relative errors value/dq/dk/dv {errs}", flush=True)
        assert max(errs) < 0.05, errs
        # the dense kernel of the same object is still the causal attention, and the windowed result differs from it (the window really cuts keys)
        dense = make(splash)(q, k, v)
        assert max(fj._rel_errors(ref, dense)) < 0.05
        assert max(fj._rel_errors(ref, outw)) > 0.05
    # an attention built for a model without windowed layers refuses the windowed call (the training step then falls back to XLA's attention)
    try:
        fj.make_splash_attention(c, mesh, 128, False, interpret=True)(c, q, k, v, jnp.bfloat16, local=True)
        raise AssertionError("a window-less splash attention served a windowed layer")
    except ValueError:
        pass
    print("SPLASH OK")


def check_splash_attention():
    import subprocess

    here = os.path.dirname(os.path.abspath(__file__))
    env = dict(os.environ, XLA_FLAGS="--xla_force_host_platform_device_count=2", JAX_PLATFORMS="cpu", PYTHONUTF8="1")
    r = subprocess.run([sys.executable, os.path.join(here, "test_forge_jax.py"), "splash-body"], capture_output=True, text=True, env=env, timeout=1500)
    print(r.stdout[-1200:], r.stderr[-1500:] if r.returncode else "")
    assert r.returncode == 0 and "SPLASH OK" in r.stdout, r.stderr[-3000:]
    print("splash attention (Pallas interpreter, 2 devices): ok")


def _count_eqns(jaxpr, prim):
    """Number of equations of primitive `prim` in a jaxpr and in all the jaxprs nested in it (shard_map, remat, custom_vjp, scan ...)."""
    n = 0
    for e in jaxpr.eqns:
        n += e.primitive.name == prim
        for val in e.params.values():
            for sub in (val if isinstance(val, (list, tuple)) else [val]):
                j = getattr(sub, "jaxpr", sub)
                if hasattr(j, "eqns"):
                    n += _count_eqns(j, prim)
    return n


def _pallas_calls_after_dce(fn, *args):
    """Pallas calls (the splash-attention kernels: in the Pallas interpreter they are still `pallas_call` equations of the jaxpr, while the CPU HLO turns them into loops) of the jaxpr of fn(*args) after dead-code elimination,
    which is what the compiler receives: a recomputed forward kernel whose result nothing uses is gone."""
    from jax._src.interpreters import partial_eval as pe

    jp = jax.make_jaxpr(fn)(*args)
    dced, _ = pe.dce_jaxpr(jp.jaxpr, [True] * len(jp.jaxpr.outvars))
    return _count_eqns(dced, "pallas_call")


def _remat_body():
    """Runs in a process with two CPU devices: --remat-policy dots+attn against dots with the splash kernels (Pallas interpreter) of a model with dense and windowed layers: loss and every gradient, the number of forward kernels in the
    backward pass, the saved residuals; and with XLA's attention the two policies are the same program."""
    import contextlib
    import hashlib
    import io
    from jax.sharding import Mesh

    devs = jax.devices()
    assert len(devs) == 2, devs
    mesh = Mesh(np.array(devs), ("data",))
    c = fj.Cfg(n_layer=3, n_embd=128, n_head=2, n_kv=1, hd=64, n_ff=128, vocab=256, seq=256, relu2=True, lambdas=True, qk_norm=True, ve_layers=(0, 2), window=128, window_layers=(0,))
    assert [c.window_of(l) for l in range(3)] == [128, None, None]  # layer 0 uses the windowed kernel, layers 1 and 2 the dense one
    p = fj.init_params(c, 1)
    p = {n: (a + 0.05 * jax.random.normal(jax.random.PRNGKey(i), a.shape) if n not in ("lam", "velam") and not n.endswith("_norm") else a) for i, (n, a) in enumerate(p.items())}
    rng = np.random.default_rng(0)
    tok = jnp.asarray(rng.integers(0, 256, (2, 256)), jnp.int32)
    tgt = jnp.asarray(rng.integers(0, 256, (2, 256)), jnp.int32)
    ign = jnp.zeros((2, 256), bool)

    def run(policy, remat=True):
        loss_grad = jax.value_and_grad(lambda p_: fj.loss_fn(c, p_, tok, tgt, ign, jnp.bfloat16, 64, remat, policy))
        return jax.jit(loss_grad)(p), loss_grad

    for fused in (False, True):
        results = {}
        for policy in ("dots", "dots+attn"):
            fj.ATTN["impl"] = fj.make_splash_attention(c, mesh, 128, fused, interpret=True, residual_name=fj.ATTN_RES_NAME if policy == "dots+attn" else None)
            (loss, grads), loss_grad = run(policy)
            n_calls = _pallas_calls_after_dce(loss_grad, p)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):  # the public report of what the backward pass keeps: "bf16[2,2,256,64] output of shard_map from ..."
                jax.ad_checkpoint.print_saved_residuals(lambda p_: fj.loss_fn(c, p_, tok, tgt, ign, jnp.bfloat16, 64, True, policy), p)
            saved = [ln.split(" output of shard_map")[0] for ln in buf.getvalue().splitlines() if " output of shard_map" in ln]
            results[policy] = (loss, grads, n_calls, saved)
        (_, _), loss_grad_nr = run("dots", remat=False)
        n_norem = _pallas_calls_after_dce(loss_grad_nr, p)
        (l0, g0, n0, s0), (l1, g1, n1, s1) = results["dots"], results["dots+attn"]
        rel = abs(float(l1) - float(l0)) / abs(float(l0))
        worst = 0.0
        for n in g0:
            d = float(jnp.linalg.norm(g1[n] - g0[n]) / (jnp.linalg.norm(g0[n]) + 1e-30))
            worst = max(worst, d)
        print(f"remat policy, splash kernels (interpreter), fused backward={fused}: loss {float(l0):.6f} vs {float(l1):.6f}, relative difference {rel:.2e}; worst gradient difference {worst:.2e} over {len(g0)} arrays; "
              f"Pallas calls in the lowered gradient: dots {n0}, dots+attn {n1}, no remat {n_norem}; saved attention residuals: dots {len(s0)}, dots+attn {len(s1)}", flush=True)
        assert rel < 1e-5 and worst < 1e-5, (rel, worst)
        assert float(jnp.linalg.norm(g0["l0.wq"])) > 0 and float(jnp.linalg.norm(g0["l2.wv"])) > 0  # the gradients are not trivially zero
        assert n0 == n_norem + c.n_layer, (n0, n_norem)  # dots: every layer runs its forward kernel a second time in the backward pass
        assert n1 == n_norem, (n1, n_norem)  # dots+attn: the forward kernels run once (as without any recomputation)
        assert len(s0) == 0 and len(s1) == 2 * c.n_layer, (s0, s1)  # per layer the output [B, H, T, hd] and the logsumexp [B, H, T] of the kernel
        assert sorted(set(s1)) == ["bf16[2,2,256,64]", "f32[2,2,256]"], s1  # the output in the compute dtype and the logsumexp in fp32
    # XLA's attention (nothing carries the checkpoint name): both policies lower to the same program, and give the same numbers
    fj.ATTN["impl"] = None
    texts, outs = {}, {}
    cd = fj.Cfg(**{**c.__dict__, "window": 0, "window_layers": ()})  # (the text of a lowering names repeated helper functions tril, tril_0 ... in the order of the process: compared on the dense model)
    for policy in ("dots", "dots+attn"):
        lg = jax.jit(jax.value_and_grad(lambda p_: fj.loss_fn(cd, p_, tok, tgt, ign, jnp.bfloat16, 64, True, policy)))
        texts[policy] = hashlib.sha256(lg.lower(p).as_text().encode()).hexdigest()
        outs[policy] = jax.jit(jax.value_and_grad(lambda p_: fj.loss_fn(c, p_, tok, tgt, ign, jnp.bfloat16, 64, True, policy)))(p)  # and the windowed model with XLA's attention
    assert texts["dots"] == texts["dots+attn"], texts
    assert float(outs["dots"][0]) == float(outs["dots+attn"][0]) and all(bool(jnp.all(outs["dots"][1][n] == outs["dots+attn"][1][n])) for n in p)
    try:
        fj.remat_policy_of("everything")
        raise AssertionError("an unknown remat policy was accepted")
    except ValueError:
        pass
    print("REMAT OK")


def check_remat_policy():
    import subprocess

    here = os.path.dirname(os.path.abspath(__file__))
    a = fj.parse_args(["train", "--data", "d", "--out", "o"])
    assert a.remat_policy == "dots" and not a.no_remat  # off by default
    assert fj.parse_args(["train", "--data", "d", "--out", "o", "--remat-policy", "dots+attn"]).remat_policy == "dots+attn"
    assert fj.remat_policy_of("dots") is jax.checkpoint_policies.dots_with_no_batch_dims_saveable  # the default is the very policy function that was used before the option existed
    env = dict(os.environ, XLA_FLAGS="--xla_force_host_platform_device_count=2", JAX_PLATFORMS="cpu", PYTHONUTF8="1")
    r = subprocess.run([sys.executable, os.path.join(here, "test_forge_jax.py"), "remat-body"], capture_output=True, text=True, env=env, timeout=1500)
    print(r.stdout[-2500:], r.stderr[-1500:] if r.returncode else "")
    assert r.returncode == 0 and "REMAT OK" in r.stdout, r.stderr[-3000:]
    print("remat policy dots+attn (Pallas interpreter, 2 devices): same loss and gradients, forward kernel not recomputed: ok")


def _remat_acts_body():
    """Runs in a process with two CPU devices: the remat policies that save more than dots (acts = the bf16 operands of the matmuls h, h2, act; qkv = the prepared q, k, v of the splash kernel; attn; every combination,
    in all layers or only in the last N) against dots, with the splash kernels (Pallas interpreter) of a model with dense and windowed layers and bf16 operands: the loss and every gradient bit for bit the same, the named
    values that the backward pass keeps (and no others), the forward kernels run once exactly when attn is saved."""
    import contextlib
    import io
    import re
    from jax.sharding import Mesh

    devs = jax.devices()
    assert len(devs) == 2, devs
    mesh = Mesh(np.array(devs), ("data",))
    c = fj.Cfg(n_layer=3, n_embd=128, n_head=2, n_kv=1, hd=64, n_ff=128, vocab=256, seq=256, relu2=True, lambdas=True, qk_norm=True, ve_layers=(0, 2), window=128, window_layers=(0,))
    p = fj.init_params(c, 1)
    p = {n: (a + 0.05 * jax.random.normal(jax.random.PRNGKey(i), a.shape) if n not in ("lam", "velam") and not n.endswith("_norm") else a) for i, (n, a) in enumerate(p.items())}
    rng = np.random.default_rng(0)
    tok = jnp.asarray(rng.integers(0, 256, (2, 256)), jnp.int32)
    tgt = jnp.asarray(rng.integers(0, 256, (2, 256)), jnp.int32)
    ign = jnp.zeros((2, 256), bool)

    def fn(policy):
        return lambda p_: fj.loss_fn(c, p_, tok, tgt, ign, jnp.bfloat16, 64, True, policy)

    def named_saved(policy):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            jax.ad_checkpoint.print_saved_residuals(fn(policy), p)
        return [m for m in re.findall(r"named '(forge_[a-z0-9_]+)'", buf.getvalue())]

    def build(policy, fused):
        fj.ATTN["impl"] = fj.make_splash_attention(c, mesh, 128, fused, interpret=True, **fj.splash_names(policy))

    for fused in (False, True):
        build("dots", fused)
        ref = jax.jit(jax.value_and_grad(fn("dots")))(p)
        n_norem = _pallas_calls_after_dce(jax.value_and_grad(lambda p_: fj.loss_fn(c, p_, tok, tgt, ign, jnp.bfloat16, 64, False)), p)
        assert named_saved("dots") == []
        for policy, want_named, n_layers_attn in (("dots+qkv", {"forge_qkv": 9}, 0), ("dots+acts", {"forge_h": 9, "forge_h2": 3, "forge_act": 3}, 0), ("dots+attn+acts+qkv", {"forge_h": 9, "forge_h2": 3, "forge_act": 3, "forge_qkv": 9}, 3),
                                                  ("dots+attn+acts+qkv@2", {"forge_h": 6, "forge_h2": 2, "forge_act": 2, "forge_qkv": 6}, 2), ("dots+h2+qkv@1", {"forge_h2": 1, "forge_qkv": 3}, 0)):
            build(policy, fused)
            out = jax.jit(jax.value_and_grad(fn(policy)))(p)
            same = float(out[0]) == float(ref[0]) and all(bool(jnp.all(out[1][n] == ref[1][n])) for n in p)
            worst = max(float(jnp.max(jnp.abs(out[1][n] - ref[1][n]))) for n in p)
            got = named_saved(policy)
            counts = {k: got.count(k) for k in set(got)}
            n_calls = _pallas_calls_after_dce(jax.value_and_grad(fn(policy)), p)
            print(f"remat policy {policy}, splash kernels (interpreter), fused backward={fused}: loss {float(out[0]):.6f}, bit-identical to dots: {same} (worst difference {worst:.1e}); named values kept: {counts}; Pallas calls {n_calls} (no remat {n_norem})", flush=True)
            assert same, (policy, worst)
            assert counts == want_named, (policy, counts, want_named)
            assert n_calls == n_norem + c.n_layer - n_layers_attn, (policy, n_calls, n_norem)  # the forward kernel runs twice in a layer that does not save its residuals
    # without a splash kernel (XLA's attention) the attn and qkv parts name nothing, the acts parts still do; the numbers are the same
    fj.ATTN["impl"] = None
    ref = jax.jit(jax.value_and_grad(fn("dots")))(p)
    for policy, want in (("dots+attn+qkv", {}), ("dots+attn+acts+qkv", {"forge_h": 9, "forge_h2": 3, "forge_act": 3})):
        out = jax.jit(jax.value_and_grad(fn(policy)))(p)
        got = named_saved(policy)
        assert {k: got.count(k) for k in set(got)} == want, (policy, got)
        assert float(out[0]) == float(ref[0]) and all(bool(jnp.all(out[1][n] == ref[1][n])) for n in p), policy
    print("REMAT ACTS OK")


def check_remat_acts():
    """The remat policies that save the matmul operands (acts) and the prepared q, k, v (qkv): the policy grammar and --remat-save-layers, the default program unchanged (and without names), bit-identical loss and gradients
    with XLA's attention, then the same with the splash kernels in a process with two CPU devices (_remat_acts_body)."""
    import hashlib
    import subprocess

    # the grammar: dots plus any parts, optionally @N; the documented values are all accepted
    assert fj.parse_remat("dots") == ((), 0) and fj.parse_remat("dots+attn") == (("attn",), 0) and fj.parse_remat("dots+acts") == (("h", "h2", "act"), 0)
    assert fj.parse_remat("dots+qkv+acts+attn@12") == (("attn", "h", "h2", "act", "qkv"), 12) and fj.parse_remat("dots+h2+act") == (("h2", "act"), 0) and fj.parse_remat("dots+acts+h") == (("h", "h2", "act"), 0)
    assert all(fj.parse_remat(s) for s in fj.REMAT_POLICIES)
    for bad in ("", "acts", "attn", "dots+", "dots+x", "dots+attn@0", "dots+attn@-1", "dots+attn@x", "dots@3x", "everything"):
        try:
            fj.parse_remat(bad)
            raise AssertionError(f"accepted: {bad!r}")
        except ValueError:
            pass
    assert [fj.remat_extras("dots+acts@2", l, 4) for l in range(4)] == [(), (), ("h", "h2", "act"), ("h", "h2", "act")] and fj.remat_extras("dots+acts", 0, 4) == ("h", "h2", "act") and fj.remat_extras("dots", 3, 4) == ()
    assert fj.remat_extras("dots+acts@10", 0, 4) == ("h", "h2", "act")  # more layers than the model has: every layer
    assert fj.splash_names("dots") == {"residual_name": None, "qkv_name": None} and fj.splash_names("dots+attn") == {"residual_name": fj.ATTN_RES_NAME, "qkv_name": None}
    assert fj.splash_names("dots+acts+qkv@3") == {"residual_name": None, "qkv_name": fj.SAVE_NAMES["qkv"]} and len(set(fj.SAVE_NAMES.values())) == len(fj.SAVE_NAMES)
    assert fj.remat_policy_of("dots") is jax.checkpoint_policies.dots_with_no_batch_dims_saveable and fj.remat_policy_of("dots+acts") is not fj.remat_policy_of("dots")
    # the flags: --remat-save-layers goes into the policy spec, only for a policy that has parts to save
    base = ["train", "--data", "d", "--out", "o"]
    assert fj.parse_args(base).remat_policy == "dots" and fj.parse_args(base).remat_save_layers == 0
    assert fj.parse_args(base + ["--remat-policy", "dots+attn+acts"]).remat_policy == "dots+attn+acts"
    assert fj.parse_args(base + ["--remat-policy", "dots+attn+acts", "--remat-save-layers", "12"]).remat_policy == "dots+attn+acts@12"
    assert fj.parse_args(base + ["--remat-policy", "dots", "--remat-save-layers", "12"]).remat_policy == "dots"
    assert fj.parse_args(base + ["--remat-policy", "dots+qkv@5"]).remat_policy == "dots+qkv@5"
    for bad in (["--remat-policy", "dots+nope"], ["--remat-policy", "dots+qkv@3", "--remat-save-layers", "2"], ["--remat-policy", "dots+qkv", "--remat-save-layers", "-1"]):
        try:
            fj.parse_args(base + bad)
            raise AssertionError(f"accepted: {bad}")
        except SystemExit:
            pass

    c = fj.Cfg(n_layer=4, n_embd=128, n_head=4, n_kv=2, hd=32, n_ff=256, vocab=256, seq=64, relu2=True, lambdas=True, qk_norm=True, ve_layers=(0, 3))
    p = fj.init_params(c, 1)
    p = {n: (a + 0.05 * jax.random.normal(jax.random.PRNGKey(i), a.shape) if n not in ("lam", "velam") and not n.endswith("_norm") else a) for i, (n, a) in enumerate(p.items())}
    rng = np.random.default_rng(0)
    tok = jnp.asarray(rng.integers(0, 256, (4, 64)), jnp.int32)
    tgt = jnp.asarray(rng.integers(0, 256, (4, 64)), jnp.int32)
    ign = jnp.zeros((4, 64), bool)

    def vg(policy, cdt, cfg=c):
        return jax.jit(jax.value_and_grad(lambda p_: fj.loss_fn(cfg, p_, tok, tgt, ign, cdt, 32, True, policy)))(p)

    # names only when asked: the default program has none; the others name the casts of the matmul operands, one per consumer (the three projections of h) so that the backward pass adds fp32 cotangents as before
    def names_in(policy):
        import re

        jp = str(jax.make_jaxpr(jax.value_and_grad(lambda p_: fj.loss_fn(c, p_, tok, tgt, ign, jnp.bfloat16, 32, True, policy)))(p))
        found = re.findall(r"name=(forge_[a-z0-9_]+)", jp)
        return {n: found.count(n) for n in set(found)}

    assert names_in("dots") == {} and names_in("dots+attn") == {} and names_in("dots+qkv") == {}
    assert names_in("dots+acts") == {"forge_h": 12, "forge_h2": 4, "forge_act": 4} and names_in("dots+h2") == {"forge_h2": 4} and names_in("dots+acts@2") == {"forge_h": 6, "forge_h2": 2, "forge_act": 2}, names_in("dots+acts")
    msgs = []
    for cdt in (jnp.bfloat16, jnp.float32):
        ref = vg("dots", cdt)
        for policy in ("dots+acts", "dots+h", "dots+h2", "dots+act", "dots+attn+acts+qkv", "dots+acts@2"):
            out = vg(policy, cdt)
            same = float(out[0]) == float(ref[0]) and all(bool(jnp.all(out[1][n] == ref[1][n])) for n in p)
            worst = max(float(jnp.max(jnp.abs(out[1][n] - ref[1][n]))) for n in p)
            assert same, (cdt.__name__, policy, worst)
            msgs.append(f"{cdt.__name__[:2]} {policy}")
    # the default program is the program of before the option (the hash of the lowered training step of a tiny model with remat and bf16 operands); with XLA's attention dots+attn and dots+qkv name nothing: the same text
    def program(policy):
        h = fj.Hyper(peak_lr=3e-3, muon_lr=3e-3, mult_table=20.0, mult_scalar=10.0, use_muon=True)
        c1 = fj.Cfg(n_layer=3, n_embd=64, n_head=2, n_kv=2, hd=32, n_ff=128, vocab=512, seq=64, relu2=True, lambdas=True, qk_norm=True, ve_layers=(0, 1))
        p1 = fj.init_params(c1, 1)
        m = {n: jnp.zeros_like(a) for n, a in p1.items()}
        v = {n: jnp.zeros_like(a) for n, a in p1.items()}
        t1 = jnp.zeros((1, 2, 64), jnp.int32)
        step = fj.build_train_step(c1, h, jnp.bfloat16, 32, True) if policy is None else fj.build_train_step(c1, h, jnp.bfloat16, 32, True, None, policy)
        return hashlib.sha256(step.lower(p1, m, v, jnp.asarray(0, jnp.int32), t1, t1, jnp.zeros((1, 2, 64), bool), jnp.float32(3e-3), jnp.float32(3e-3)).as_text().encode()).hexdigest()[:16]

    d0 = program(None)
    assert program("dots") == d0 and program("dots+attn") == d0 and program("dots+qkv") == d0, "without a splash kernel these policies name nothing: the same program as dots"
    if jax.__version__ == "0.11.2":
        assert d0 == "828bdab5abdaa529", d0  # sha256 of the lowered step with remat and bf16 operands of main before this option (b72510b)
    print(f"remat acts: grammar, flags, names only when asked, default program unchanged ({d0}), bit-identical loss and gradients with XLA's attention ({len(msgs)} policy x dtype cases): ok")
    here = os.path.dirname(os.path.abspath(__file__))
    env = dict(os.environ, XLA_FLAGS="--xla_force_host_platform_device_count=2", JAX_PLATFORMS="cpu", PYTHONUTF8="1")
    r = subprocess.run([sys.executable, os.path.join(here, "test_forge_jax.py"), "remat-acts-body"], capture_output=True, text=True, env=env, timeout=1500)
    print(r.stdout[-3500:], r.stderr[-1500:] if r.returncode else "")
    assert r.returncode == 0 and "REMAT ACTS OK" in r.stdout, r.stderr[-3000:]
    print("remat policies with acts / qkv (Pallas interpreter, 2 devices): same loss and gradients bit for bit, the named values kept: ok")
    # through the trainer's command line (--remat-save-layers is merged into the policy): the loss of every step and the checkpoint after 12 Muon steps are bit for bit those of the default policy
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        got = {}
        for tag, extra in (("dots", []), ("extras", ["--remat-policy", "dots+attn+acts+qkv", "--remat-save-layers", "1"])):
            out, args, env = _tiny_run_args(tempfile.mkdtemp(dir=tmp), ["--steps", "12"] + extra)
            r = subprocess.run([sys.executable, os.path.join(here, "forge_jax.py")] + args, capture_output=True, text=True, env=env, timeout=600)
            assert r.returncode == 0, r.stderr[-1500:]
            assert ("remat policy dots+attn+acts+qkv@1" in r.stdout) == bool(extra), r.stdout[-600:]
            losses = [json.loads(line)["loss"] for line in open(os.path.join(out, "log.jsonl"))]
            got[tag] = (losses, hashlib.sha256(open(os.path.join(out, "ckpt_latest.bin"), "rb").read()).hexdigest())
        assert len(got["dots"][0]) == 12 and got["dots"] == got["extras"], got
        print(f"remat policy dots+attn+acts+qkv with --remat-save-layers 1 through the trainer: the 12 losses and the checkpoint ({got['dots'][1][:12]}) are bit for bit those of dots: ok")


def check_fuse_qkv():
    """--fuse-qkv: one matmul on [wq; wk; wv] gives the same loss and gradients (the gradient flows back through the concatenation into the three parameters), the default program does not change, the flag is the
    flag of the trainer; and what XLA's CPU compiler does with the three separate projections (it says nothing about the TPU compiler)."""
    import hashlib
    import re

    c = fj.Cfg(n_layer=3, n_embd=128, n_head=4, n_kv=2, hd=32, n_ff=256, vocab=256, seq=64, relu2=True, lambdas=True, qk_norm=True, ve_layers=(0, 2), window=16, window_layers=(0,))
    p = fj.init_params(c, 1)
    p = {n: (a + 0.05 * jax.random.normal(jax.random.PRNGKey(i), a.shape) if n not in ("lam", "velam") and not n.endswith("_norm") else a) for i, (n, a) in enumerate(p.items())}
    rng = np.random.default_rng(0)
    tok = jnp.asarray(rng.integers(0, 256, (4, 64)), jnp.int32)
    tgt = jnp.asarray(rng.integers(0, 256, (4, 64)), jnp.int32)
    ign = jnp.zeros((4, 64), bool)
    assert fj.parse_args(["train", "--data", "d", "--out", "o"]).fuse_qkv is False and fj.parse_args(["train", "--data", "d", "--out", "o", "--fuse-qkv"]).fuse_qkv is True
    def vg(cdt, remat, fuse):
        return jax.jit(jax.value_and_grad(lambda p_: fj.loss_fn(c, p_, tok, tgt, ign, cdt, 32, remat, "dots", fuse)))(p)

    def global_rel(a, b):  # relative difference of the whole gradient (all arrays together)
        return float(np.sqrt(sum(float(jnp.sum((a[1][n] - b[1][n]) ** 2)) for n in p) / sum(float(jnp.sum(b[1][n] ** 2)) for n in p)))

    msgs = []
    for remat in (False, True):  # float32 operands: the same numbers up to the order of the summation
        ref, new = vg(jnp.float32, remat, False), vg(jnp.float32, remat, True)
        rel = abs(float(new[0]) - float(ref[0])) / abs(float(ref[0]))
        worst = max(float(jnp.linalg.norm(new[1][n] - ref[1][n]) / (jnp.linalg.norm(ref[1][n]) + 1e-30)) for n in p)
        msgs.append(f"f32 remat={remat}: loss {rel:.1e}, worst gradient {worst:.1e}")
        assert rel < 1e-5 and worst < 1e-5, (remat, rel, worst)
        assert all(float(jnp.linalg.norm(new[1][f"l{l}.{w}"])) > 0 for l in range(3) for w in ("wq", "wk", "wv"))  # the three parameters all receive a gradient
    # bfloat16 operands: the forward pass is the same (the loss agrees), the gradients differ at the level of bf16 rounding, because the unfused backward pass rounds the three input-gradient contributions of q, k and v
    # to bf16 one by one and the fused one adds them inside one fp32 accumulator: the fused gradient is as close to the fp32 gradient as the unfused one, and the two differ by much less than either differs from fp32
    ref16, new16, ref32 = vg(jnp.bfloat16, True, False), vg(jnp.bfloat16, True, True), vg(jnp.float32, True, False)
    rel = abs(float(new16[0]) - float(ref16[0])) / abs(float(ref16[0]))
    d_fuse, d_ref, d_between = global_rel(new16, ref32), global_rel(ref16, ref32), global_rel(new16, ref16)
    msgs.append(f"bf16: loss {rel:.1e}; gradient distance to fp32: unfused {d_ref:.1e}, fused {d_fuse:.1e}; fused to unfused {d_between:.1e}")
    assert rel < 1e-5 and d_fuse < 1.1 * d_ref and d_between < 0.5 * d_ref, (rel, d_fuse, d_ref, d_between)
    print("fuse-qkv, differences to the separate projections:", "; ".join(msgs))

    def text(fuse, compiled=False):
        h = fj.Hyper(peak_lr=3e-3, muon_lr=3e-3, mult_table=20.0, mult_scalar=10.0, use_muon=True)
        c1 = fj.Cfg(n_layer=3, n_embd=64, n_head=2, n_kv=2, hd=32, n_ff=128, vocab=512, seq=64, relu2=True, lambdas=True, ve_layers=(0, 1))
        p1 = fj.init_params(c1, 1)
        m = {n: jnp.zeros_like(a) for n, a in p1.items()}
        v = {n: jnp.zeros_like(a) for n, a in p1.items()}
        t1 = jnp.zeros((1, 2, 64), jnp.int32)
        ig = jnp.zeros((1, 2, 64), bool)
        step = fj.build_train_step(c1, h, jnp.float32, 32, False) if fuse is None else fj.build_train_step(c1, h, jnp.float32, 32, False, None, "dots", fuse)
        low = step.lower(p1, m, v, jnp.asarray(0, jnp.int32), t1, t1, ig, jnp.float32(3e-3), jnp.float32(3e-3))
        return low.compile().as_text() if compiled else low.as_text()

    d0, d1, on = (hashlib.sha256(text(f).encode()).hexdigest()[:16] for f in (None, False, True))
    assert d0 == d1, "fuse_qkv=False must be the program of the default"
    assert on != d0, "fuse_qkv=True must change the program"
    if jax.__version__ == "0.11.2":
        assert d0 == "dc6c181a9996e0c1", d0  # the same hash as check_window_off_is_identical (696cb7f65e3bff5f was the program of main before the accum == 1 shortcut: the option itself changes nothing)
    n_off, n_on = (len(re.findall(r"stablehlo\.dot_general", text(f))) for f in (False, True))
    assert n_off - n_on == 6 * 3, (n_off, n_on)  # per layer 2 fewer matmuls in the forward pass, 2 fewer for the input gradient and 2 fewer for the weight gradient (q, k, v are one matmul instead of three)
    c_off, c_on = (len(re.findall(r"= \S+ (?:dot|convolution)\(", text(f, True))) for f in (False, True))
    print(f"fuse-qkv: default program unchanged ({d0}); lowered dot_general ops {n_off} -> {n_on}; dots in the CPU-compiled program {c_off} -> {c_on} "
          f"({'XLA CPU does not merge the three projections by itself' if c_on < c_off else 'XLA CPU already merges them'}): ok")


def _reference_train_step(c, h, cdt, ce_chunk, remat):
    """build_train_step as it was before the accum == 1 shortcut (a scan over the micro-steps from a zero accumulator, a division by accum): the reference of the bit-for-bit comparison."""

    def step_fn(params, m, v, step_count, tokens, targets, ignore, lr, lr_muon):
        accum = tokens.shape[0]

        def micro(carry, xs):
            gsum, lsum = carry
            t_, y_, i_ = xs
            loss, g = jax.value_and_grad(lambda p: fj.loss_fn(c, p, t_, y_, i_, cdt, ce_chunk, remat))(params)
            return (jax.tree_util.tree_map(jnp.add, gsum, g), lsum + loss), None

        zeros = jax.tree_util.tree_map(jnp.zeros_like, params)
        (grads, lsum), _ = jax.lax.scan(micro, (zeros, jnp.zeros((), jnp.float32)), (tokens, targets, ignore))
        grads = jax.tree_util.tree_map(lambda g: g / accum, grads)
        loss = lsum / accum
        step_count = step_count + 1
        params, m, v, gnorm = fj.apply_update(c, h, params, grads, m, v, step_count, lr, lr_muon, None)
        return params, m, v, step_count, loss, gnorm

    return jax.jit(step_fn, donate_argnums=(0, 1, 2, 3))


def check_accum1_step():
    """With one micro-step per optimizer step the step has no zero accumulator, scan or division: params, moments, loss and gradient norm after 3 steps are bit-for-bit those of the old formulation (and the old
    formulation is the program of main: the hash of before); with several micro-steps the program is the unchanged old one."""
    import hashlib

    c = fj.Cfg(n_layer=3, n_embd=64, n_head=2, n_kv=2, hd=32, n_ff=128, vocab=512, seq=64, relu2=True, lambdas=True, ve_layers=(0, 1))
    h = fj.Hyper(peak_lr=3e-3, muon_lr=3e-3, mult_table=20.0, mult_scalar=10.0, use_muon=True)
    rng = np.random.default_rng(0)

    def fresh():
        p = fj.init_params(c, 1)
        return p, {n: jnp.zeros_like(a) for n, a in p.items()}, {n: jnp.zeros_like(a) for n, a in p.items()}

    def batch(accum):
        base = rng.integers(0, 512, size=(accum * 2, 65)).astype(np.int32)
        return base[None, :, :-1].reshape(accum, 2, 64), base[None, :, 1:].reshape(accum, 2, 64), np.zeros((accum, 2, 64), bool)

    def lowered(step, accum):
        p, m, v = fresh()
        t, y, ig = batch(accum)
        return step.lower(p, m, v, jnp.asarray(0, jnp.int32), t, y, ig, jnp.float32(3e-3), jnp.float32(3e-3)).as_text()

    new_step = fj.build_train_step(c, h, jnp.float32, 32, False)
    old_step = _reference_train_step(c, h, jnp.float32, 32, False)
    t_old, t_new = lowered(old_step, 1), lowered(new_step, 1)
    h_old, h_new = (hashlib.sha256(t.encode()).hexdigest()[:16] for t in (t_old, t_new))
    if jax.__version__ == "0.11.2":
        assert h_old == "696cb7f65e3bff5f", h_old  # the reference IS the program of main before this change
    assert h_old != h_new and t_new.count("stablehlo.while") == t_old.count("stablehlo.while") - 1 and len(t_new) < len(t_old), (h_old, h_new)
    assert lowered(old_step, 2) == lowered(new_step, 2), "with two micro-steps the program must be the old one"
    for accum, use_bits in ((1, True), (2, True)):
        states = []
        for step in (old_step, new_step):
            p, m, v = fresh()
            sc = jnp.asarray(0, jnp.int32)
            rng2 = np.random.default_rng(5)
            for i in range(3):
                base = rng2.integers(0, 512, size=(accum * 2, 65)).astype(np.int32)
                t, y, ig = base[:, :-1].reshape(accum, 2, 64), base[:, 1:].reshape(accum, 2, 64), np.zeros((accum, 2, 64), bool)
                p, m, v, sc, loss, gn = step(p, m, v, sc, t, y, ig, jnp.float32(3e-3), jnp.float32(3e-3))
            states.append((p, m, v, float(loss), float(gn)))
        (p0, m0, v0, l0, g0), (p1, m1, v1, l1, g1) = states
        bits = lambda a: np.asarray(a).view(np.uint32)
        same = all(np.array_equal(bits(t0[n]), bits(t1[n])) for t0, t1 in ((p0, p1), (m0, m1), (v0, v1)) for n in p0)
        assert same and l0 == l1 and g0 == g1, (accum, l0, l1, g0, g1)
        assert l0 > 0 and sum(float(jnp.sum(jnp.abs(a))) for a in p0.values()) > 0
    print(f"accum 1: params, moments, loss and gradient norm bit-identical after 3 steps; the program changes ({h_old} -> {h_new}, one while loop and the zero accumulator fewer); accum 2: the program is unchanged: ok")


def check_auto_steps():
    """A run whose time limit is far shorter than its schedule plans its steps so that the decay ends inside the limit (the last learning rate is the minimum)."""
    import subprocess
    import tempfile

    here = os.path.dirname(os.path.abspath(__file__))
    with tempfile.TemporaryDirectory() as tmp:
        data = os.path.join(tmp, "data")
        os.makedirs(data)
        rng = np.random.default_rng(6)
        base = np.where(rng.random(300_000) < 0.05, rng.integers(0, 512, 300_000), (np.arange(300_000) % 97)).astype(np.int64)
        base.astype("<u2").tofile(os.path.join(data, "a_000.u16"))
        base[:100_000].astype("<u2").tofile(os.path.join(data, "zzz_val.u16"))
        out = os.path.join(tmp, "run")
        args = ["train", "--data", data, "--out", out, "--layers", "2", "--dim", "64", "--heads", "2", "--kv", "1", "--head-dim", "32", "--ffn", "128", "--vocab", "512", "--seq", "64", "--tokens-per-step", "1024", "--micro", "2",
                "--lr", "3e-3", "--optim", "muon", "--muon-lr", "3e-3", "--warmup", "5", "--qk-norm", "--relu2", "--eval-every", "50", "--eval-batches", "1", "--save-every", "50", "--val-tokens", "30000", "--dtype", "f32",
                "--ce-chunk", "32", "--steps", "1000000", "--decay-frac", "0.2", "--min-lr-frac", "0.1", "--time-limit-s", "45", "--plan-reserve-s", "6"]
        env = dict(os.environ, XLA_FLAGS="--xla_force_host_platform_device_count=2", JAX_PLATFORMS="cpu", PYTHONUTF8="1")
        r = subprocess.run([sys.executable, os.path.join(here, "forge_jax.py")] + args, capture_output=True, text=True, env=env, timeout=600)
        assert r.returncode == 0, r.stderr[-2000:]
        assert "time plan:" in r.stdout and "time limit of" not in r.stdout, r.stdout[-1500:]
        last = [l for l in r.stdout.splitlines() if l.startswith("step ")][-1]
        total = int(last.split("/")[1].split()[0])
        done = int(last.split()[1].split("/")[0])
        assert done == total and total < 1_000_000, last
        lr_last = float(last.split(" lr ")[1].split()[0])
        assert lr_last < 4e-4, lr_last  # the schedule ended near its floor (3e-4)
        print(f"auto steps: planned {total} steps for a 45 s limit; last learning rate {lr_last:.2e} (floor 3.00e-04): ok")


def check_attention_fallback():
    """An attention kernel that fails (in the training step, or only in the evaluation) must not stop the run: it continues with XLA's attention."""
    import subprocess
    import tempfile

    here = os.path.dirname(os.path.abspath(__file__))
    for mode, needle in (("step", "failed in the full training step"), ("eval", "failed in the evaluation")):
        with tempfile.TemporaryDirectory() as tmp:
            data = os.path.join(tmp, "data")
            os.makedirs(data)
            rng = np.random.default_rng(7)
            base = np.where(rng.random(200_000) < 0.05, rng.integers(0, 512, 200_000), (np.arange(200_000) % 97)).astype(np.int64)
            base.astype("<u2").tofile(os.path.join(data, "a_000.u16"))
            base[:80_000].astype("<u2").tofile(os.path.join(data, "zzz_val.u16"))
            args = ["train", "--data", data, "--out", os.path.join(tmp, "run"), "--layers", "2", "--dim", "64", "--heads", "2", "--kv", "1", "--head-dim", "32", "--ffn", "128", "--vocab", "512", "--seq", "64",
                    "--tokens-per-step", "512", "--micro", "2", "--lr", "3e-3", "--optim", "muon", "--muon-lr", "3e-3", "--warmup", "3", "--qk-norm", "--relu2", "--steps", "12", "--eval-every", "6", "--eval-batches", "1",
                    "--save-every", "100", "--val-tokens", "30000", "--dtype", "f32", "--ce-chunk", "32", "--attn-test-fail", mode]
            env = dict(os.environ, XLA_FLAGS="--xla_force_host_platform_device_count=2", JAX_PLATFORMS="cpu", PYTHONUTF8="1")
            r = subprocess.run([sys.executable, os.path.join(here, "forge_jax.py")] + args, capture_output=True, text=True, env=env, timeout=600)
            assert r.returncode == 0, r.stderr[-2000:]
            assert needle in r.stdout and "step    12/12" in r.stdout and " val " in r.stdout, r.stdout[-1500:]
            print(f"attention fallback ({mode}): the run finished after '{needle}': ok")


def _tiny_run_args(tmp, extra):
    """Data and the argument list of a tiny 2-device CPU run (the data are a learnable pattern with noise)."""
    data = os.path.join(tmp, "data")
    os.makedirs(data, exist_ok=True)
    rng = np.random.default_rng(8)
    base = np.where(rng.random(200_000) < 0.05, rng.integers(0, 512, 200_000), (np.arange(200_000) % 97)).astype(np.int64)
    base.astype("<u2").tofile(os.path.join(data, "a_000.u16"))
    base[:80_000].astype("<u2").tofile(os.path.join(data, "zzz_val.u16"))
    out = os.path.join(tmp, "run")
    args = ["train", "--data", data, "--out", out, "--layers", "2", "--dim", "64", "--heads", "2", "--kv", "1", "--head-dim", "32", "--ffn", "128", "--vocab", "512", "--seq", "64", "--tokens-per-step", "512", "--micro", "2",
            "--lr", "3e-3", "--optim", "muon", "--muon-lr", "3e-3", "--warmup", "3", "--qk-norm", "--relu2", "--eval-every", "10", "--eval-batches", "1", "--save-every", "10", "--val-tokens", "30000", "--dtype", "f32", "--ce-chunk", "32"] + extra
    env = dict(os.environ, XLA_FLAGS="--xla_force_host_platform_device_count=2", JAX_PLATFORMS="cpu", PYTHONUTF8="1")
    return out, args, env


def check_pin_wanted():
    assert fj._pin_wanted(["0.7.2", "tpu", "8"], "0.11.2")
    assert not fj._pin_wanted(["0.11.2", "tpu", "8"], "0.11.2")
    assert not fj._pin_wanted(["0.7.2", "gpu", "2"], "0.11.2")
    assert not fj._pin_wanted(["0.7.2", "cpu", "1"], "0.11.2")
    assert not fj._pin_wanted(None, "0.11.2")
    print("jax pin decision: ok")


def check_safety_nets():
    """A batch reader that fails must stop the run at once, a hung step must end in the watchdog's exit code 97, the printed step lines are throttled, and a resume inside the decay keeps the planned schedule."""
    import subprocess
    import tempfile
    import time as _t

    here = os.path.dirname(os.path.abspath(__file__))
    script = os.path.join(here, "forge_jax.py")
    with tempfile.TemporaryDirectory() as tmp:
        out, args, env = _tiny_run_args(tmp, ["--steps", "30"])
        t0 = _t.time()
        r = subprocess.run([sys.executable, script] + args, capture_output=True, text=True, env=dict(env, FORGE_TEST_PRODUCER_FAIL="4"), timeout=300)
        assert r.returncode != 0 and "the batch producer thread failed" in r.stderr, (r.returncode, r.stderr[-800:])
        assert _t.time() - t0 < 120, _t.time() - t0
        print(f"producer failure: the run stops (exit code {r.returncode}) instead of hanging: ok")
    with tempfile.TemporaryDirectory() as tmp:
        out, args, env = _tiny_run_args(tmp, ["--steps", "30", "--watchdog-s", "20"])
        t0 = _t.time()
        r = subprocess.run([sys.executable, script] + args, capture_output=True, text=True, env=dict(env, FORGE_TEST_HANG_AT_STEP="8"), timeout=300)
        assert r.returncode == 97 and "watchdog" in r.stdout, (r.returncode, r.stdout[-600:])
        print(f"watchdog: a hung step ends with exit code 97 after {_t.time() - t0:.0f} s: ok")
    with tempfile.TemporaryDirectory() as tmp:
        out, args, env = _tiny_run_args(tmp, ["--steps", "40", "--decay-frac", "0.5", "--min-lr-frac", "0.1", "--time-limit-s", "100000", "--plan-reserve-s", "1", "--print-every", "10"])
        r = subprocess.run([sys.executable, script] + args, capture_output=True, text=True, env=env, timeout=300)
        assert r.returncode == 0, r.stderr[-1500:]
        shown = [ln for ln in r.stdout.splitlines() if ln.startswith("step ")]
        assert len(shown) < 40 and shown[-1].lstrip("step ").startswith("40/40"), (len(shown), shown[-1])  # throttled, the last step is shown
        assert os.path.exists(os.path.join(out, "ckpt_predecay.bin")) and "pre-decay checkpoint at step 20" in r.stdout, r.stdout[-800:]
        assert fj.checkpoint_plan(os.path.join(out, "ckpt_latest.bin")) == 40
        # the checkpoint (step 40) lies inside the decay of a run planned for 40 steps: a resume with a larger --steps must keep that schedule and stop at once
        args2 = [x if x != "40" else "80" for x in args]
        r2 = subprocess.run([sys.executable, script] + args2 + ["--resume"], capture_output=True, text=True, env=env, timeout=300)
        assert r2.returncode == 0 and "keeping that schedule (40 steps)" in r2.stdout and "step    41/" not in r2.stdout, r2.stdout[-800:]
        # that resume has nothing left to do, so the batch thread has just been started when the process ends: it must be stopped first (a daemon thread inside jax.device_put at the end of the process aborts it with
        # "terminate called after throwing an instance of ''": 6 of 8 runs before the fix); repeated because the race is intermittent
        for _ in range(3):
            r3 = subprocess.run([sys.executable, script] + args2 + ["--resume"], capture_output=True, text=True, env=env, timeout=300)
            assert r3.returncode == 0, (r3.returncode, r3.stderr[-400:])
        print(f"printed lines {len(shown)} of 40, pre-decay checkpoint, plan stored in the header, resume inside the decay keeps the schedule, clean exit of a resume with nothing to do (4 times): ok")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "splash-body":
        _splash_body()
        sys.exit(0)
    if len(sys.argv) > 1 and sys.argv[1] == "remat-body":
        _remat_body()
        sys.exit(0)
    if len(sys.argv) > 1 and sys.argv[1] == "remat-acts-body":
        _remat_acts_body()
        sys.exit(0)
    if len(sys.argv) > 1 and sys.argv[1] == "muon-shard-body":
        _muon_shard_body()
        sys.exit(0)
    for name in ("plain", "qknorm", "untied", "relu2", "softcap", "lambdas", "skips", "valemb"):
        check_variant(name)
    check_muon()
    check_stacked_muon()
    check_muon_shard()
    check_bench_step()
    check_kaggle_bench_runner()
    check_checkpoint_roundtrip()
    check_window_checkpoint()
    check_window_attention()
    check_window_layer_rule()
    check_window_off_is_identical()
    check_window_model()
    check_training_runs()
    check_plan_steps()
    check_plan_with_stalls()
    check_end_to_end()
    check_auto_steps()
    check_attention_fallback()
    check_window_end_to_end()
    check_pin_wanted()
    check_safety_nets()
    check_splash_attention()
    check_remat_policy()
    check_remat_acts()
    check_fuse_qkv()
    check_accum1_step()
    print("all checks passed")
