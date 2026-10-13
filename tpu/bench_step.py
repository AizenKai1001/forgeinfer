"""Benchmark of the JAX training step for ONE short TPU session: which of a few speed options makes the step faster.

    python bench_step.py [--out DIR] [--args "MODEL ARGS"] [--deadline-s 1500] [--only base,muon-shard] [--tiny]

The orchestrator (this process) never touches jax: every variant runs in its own worker process (`--worker`), because a TPU can be opened by one process at a time, because environment variables
(LIBTPU_INIT_ARGS) apply at start-up, and because an out-of-memory error or a runtime that rejects a flag must not take the other variants with it. A worker builds the model of the argument string (default: the
365M model of cloud/tpu_settings/m365_ve8.json without its training-only flags), makes random tokens (one micro-step of `micro` sequences per device), does 2 warm-up and 12 timed steps with block_until_ready and prints
ONE json line (variant, status ok / oom / error / timeout / skipped, median and minimum step seconds, tokens per second, memory statistics when jax gives them). The orchestrator prints a table sorted by tokens per
second with the gain over `base` and writes DIR/bench.json (after every variant, so a killed session leaves its partial results). The whole run stops starting new variants when --deadline-s is used up.

Tokens per second are the number to compare: the micro-batch 8 variants run 131,072 tokens per step (8 sequences per device, one micro-step), the others 65,536 (a bigger batch changes the training dynamics, so a gain
there has to be weighed against that). Nothing here says anything about speed until it has run on a TPU.
"""
import argparse
import json
import os
import re
import shlex
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

# the model of cloud/tpu_settings/m365_ve8.json: the flags that shape the model and the step (the training-only ones - steps, data, evaluation, checkpoints, time limit - are left out)
ARGS_365M = ("--layers 24 --dim 1024 --heads 16 --kv 4 --ffn 4096 --lr 1.0e-3 --muon-lr 1.0e-3 --value-emb 0,1,2,3,20,21,22,23 --lr-table-mult 100 --lr-emb-mult 3 --seq 2048 "
             "--qk-norm --optim muon --muon-wd 0.05 --relu2 --lambdas --lr-scalar-mult 10")
ARGS_TINY = "--layers 3 --dim 64 --heads 2 --kv 1 --head-dim 32 --ffn 128 --vocab 512 --seq 64 --qk-norm --optim muon --relu2 --lambdas --value-emb 0,2 --dtype f32 --ce-chunk 32"

# async collectives and compute/collective overlap as MaxText sets them; UNVERIFIED on this libtpu: a runtime that does not know a flag may refuse to start, which the table then shows as an error
LIBTPU_ASYNC = ("--xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true "
                "--xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_gather=true")

OOM = re.compile(r"RESOURCE_EXHAUSTED|out of memory|Ran out of memory", re.IGNORECASE)


def variants(tiny=False):
    """[(name, extra trainer flags, extra environment, note)] in the order they run (the ones most likely to matter first, `base` always first; the no-remat and micro8 rows, unlikely to fit in memory, come after the new options so that a 25-minute deadline reaches those)."""
    v = [
        ("base", "", {}, "as the training kernel runs now"),
        ("muon-shard", "--muon-shard", {}, "Newton-Schulz split over the chips"),
        ("window512", "--window 512 --window-layers auto", {}, "sliding window 512 in three of every four layers (second splash kernel); a different model: equal quality was found only in our own proxy runs"),
        ("muon-shard+window512", "--muon-shard --window 512 --window-layers auto", {}, "both"),
        ("remat-attn", "--remat-policy dots+attn", {}, "the splash kernel's output and logsumexp are saved in the backward pass instead of recomputing the forward kernel (about 400 MB more; the same program as base when the attention is XLA's)"),
        ("fuse-qkv", "--fuse-qkv", {}, "q, k and v as one matmul on the concatenated weights"),
        ("hd128", "--heads 8 --kv 2 --head-dim 128", {}, "the same parameter count with 128-wide heads (fills the v5e's 128-wide MXU): a speed question only, the model quality is another matter"),
        ("all", "--muon-shard --window 512 --window-layers auto --remat-policy dots+attn --fuse-qkv", {}, "muon-shard, window 512, remat-attn and fuse-qkv together (the window makes it a different model)"),
        ("breakdown-base", "", {}, "where the time goes (base): the forward, the backward and the optimizer as separate programs"),
        ("breakdown-all", "--muon-shard --window 512 --window-layers auto --remat-policy dots+attn --fuse-qkv", {}, "where the time goes (all options)"),
        ("no-remat", "--no-remat", {}, "no recomputation in the backward pass (more memory)"),
        ("no-remat+muon-shard", "--no-remat --muon-shard", {}, ""),
        ("micro8", "--micro 8", {}, "8 sequences per chip: 131,072 tokens per step"),
        ("micro8+no-remat", "--micro 8 --no-remat", {}, "131,072 tokens per step"),
        ("libtpu-async", "", {"LIBTPU_INIT_ARGS": LIBTPU_ASYNC}, "unverified on this libtpu: async collective fusion flags"),
        ("muon-shard+libtpu-async", "--muon-shard", {"LIBTPU_INIT_ARGS": LIBTPU_ASYNC}, "unverified on this libtpu: async collective fusion flags"),
        # the remat policies of tpu/forge_jax.py that save more in the forward pass instead of recomputing it in the backward pass: the same numbers as base, bit for bit (the peak GiB per chip are the compile lab's, micro 4)
        ("remat-qkv", "--remat-policy dots+qkv", {}, "the prepared bf16 q, k, v of the splash kernel are saved: the qk-norm / RoPE / scale / cast / transpose chain is not recomputed in the backward pass (lab peak 12.4 GiB, base 11.4)"),
        ("remat-attn-qkv", "--remat-policy dots+attn+qkv", {}, "remat-qkv and remat-attn together: also the kernel's output and logsumexp are saved, the forward kernel runs once (lab peak 13.1 GiB)"),
        ("remat-qkv+muon-shard", "--muon-shard --remat-policy dots+qkv", {}, "remat-qkv with the sharded Muon update (lab peak 11.5 GiB, muon-shard alone 10.7)"),
        ("remat-attn-qkv+muon-shard", "--muon-shard --remat-policy dots+attn+qkv", {}, "remat-attn-qkv with the sharded Muon update (lab peak 13.6 GiB: the splash residuals cost about 2 GiB here, 0.8 without muon-shard)"),
        ("attn-lane", "--attn-prep lane", {}, "q, k and v prepared for the attention (qk-norm, RoPE, value-embedding mix, scale, cast) on lane-dense arrays [batch, T, heads x 64] and transposed once, in bf16 (the same function)"),
        ("attn-lane+muon-shard", "--attn-prep lane --muon-shard", {}, "both"),
        ("muon-lean", "--muon-lean", {}, "Muon update per layer: only the bf16 Newton-Schulz input and output are stacked (same numbers, fewer fp32 copies)"),
        ("muon-lean+muon-shard", "--muon-lean --muon-shard", {}, "per-layer Muon update with the Newton-Schulz split over the chips by a shard_map"),
        ("ns-barrier", "--ns-barrier", {}, "an optimization barrier after every Newton-Schulz iteration (same numbers)"),
        ("ns-barrier+muon-shard", "--ns-barrier --muon-shard", {}, "barrier + Newton-Schulz split over the chips"),
        ("muon-lean+ns-barrier", "--muon-lean --ns-barrier", {}, "per-layer Muon update and a barrier after every Newton-Schulz iteration"),
        ("muon-lean+ns-barrier+muon-shard", "--muon-lean --ns-barrier --muon-shard", {}, "all three optimizer options"),
        # the combinations of the options that compute the same function as base, the compile lab's best candidates (2026-10-11, estimates against base 364.8 ms: lane+shard 237.9, + lean and barrier 219.6)
        ("lane+shard+lean", "--attn-prep lane --muon-shard --muon-lean --ns-barrier", {}, "attn-lane, muon-shard and the lean Muon update with barriers: the compile lab's best (estimate 219.6 ms against base 364.8; 11.5 GiB)"),
        ("lane+shard+lean+attn", "--attn-prep lane --muon-shard --muon-lean --ns-barrier --remat-policy dots+attn", {}, "the same with the splash kernel's output and logsumexp saved (lab: 13.5 GiB)"),
        # the layer program options of the compile lab, round 2 (tpu/hlo_cost.py --matmuls, 2026-10-12; estimates against lane+shard+lean 219.6 ms, 11.49 GiB): the same function as base (woheads: another fp32 summation order)
        ("layer-ffnbf16", "--layer-opts ffnbf16", {}, "the bf16 copies of the feed-forward weights made once per layer in their own fusions instead of inside every matmul (bit-identical numbers; lab on lane+shard+lean: -9.6 ms, +0.3 GiB; on the plain base program it does not help)"),
        ("layer-woheads", "--layer-opts woheads", {}, "the output projection on the attention kernel's [batch, heads, T, 64] layout, no transposed copy (fp32 sums in another order; lab: about -1 ms, within the lab's noise)"),
        ("remat-qknorm", "--remat-policy dots+qknorm", {}, "the mean squares of the q and k norms are kept for the backward pass instead of recomputed (bit-identical; lab: within noise, the remat region -3 ms is paid back by other fusions)"),
        ("remat-norm", "--remat-policy dots+norm", {}, "the mean squares of the two residual-stream norms are kept (bit-identical; the lab's estimate gets WORSE by 8 to 12 ms: the fusions of the backward pass change)"),
        ("lane+shard+lean+ffnbf16", "--attn-prep lane --muon-shard --muon-lean --ns-barrier --layer-opts ffnbf16", {}, "the lab's best of round 2: lane+shard+lean with the bf16 feed-forward weight copies (estimate 210.0 ms against 219.6; 11.8 GiB)"),
        ("lane+shard+lean+ffnbf16+woheads", "--attn-prep lane --muon-shard --muon-lean --ns-barrier --layer-opts ffnbf16,woheads", {}, "with the output projection on the kernel layout too (estimate 209.5 ms; 11.8 GiB)"),
    ]
    if tiny:  # CPU self-test: the options that exist on a CPU, and three deliberately failing variants (an exception, an out-of-memory error, a process that dies)
        v = [x for x in v if x[0] in ("base", "muon-shard", "no-remat", "micro8", "libtpu-async", "remat-attn", "fuse-qkv", "hd128", "all", "breakdown-base", "remat-qkv", "remat-attn-qkv", "remat-qkv+muon-shard", "remat-attn-qkv+muon-shard", "attn-lane", "muon-lean", "muon-lean+muon-shard", "ns-barrier", "lane+shard+lean", "layer-ffnbf16", "layer-woheads", "remat-qknorm", "lane+shard+lean+ffnbf16+woheads")]
        v = [(n, "--heads 1 --kv 1 --head-dim 64", e, note) if n == "hd128" else (n, f.replace("--window 512", "--window 16"), e, note) for n, f, e, note in v]  # the tiny model has 2 heads of 32: one head of 64 is its wider-head variant; a window of 512 would be the dense case on 64 positions
        v += [("window16", "--window 16 --window-layers auto", {}, "sliding window 16 (a CPU self-test of the flags)")]
        v += [("selftest-error", "", {"FORGE_BENCH_TEST_FAIL": "error"}, "fails on purpose"), ("selftest-oom", "", {"FORGE_BENCH_TEST_FAIL": "oom"}, "fails on purpose"),
              ("selftest-crash", "", {"FORGE_BENCH_TEST_FAIL": "crash"}, "fails on purpose")]
    return v


# The order in which the built-in table runs when the whole table is asked for (no --only, no --variants-json, no --tiny): a 25-minute session gets through about a dozen variants, so the ones that matter most for the next
# training run come first (the compile lab's best candidates 2026-10-11, then the model-changing window and head-size rows for information), the ones that the lab says do not fit or help come last.
PRIORITY = ("base", "muon-shard", "attn-lane", "attn-lane+muon-shard", "lane+shard+lean", "lane+shard+lean+ffnbf16", "lane+shard+lean+attn", "remat-qkv+muon-shard", "muon-lean+ns-barrier+muon-shard", "window512", "fuse-qkv", "libtpu-async",
            "muon-shard+libtpu-async", "hd128", "remat-attn", "breakdown-base")


def priority_order(plan):
    """The plan (a list of variant tuples) with the PRIORITY names first, in that order, and every other variant after them in its table order (a stable sort: the names not in PRIORITY keep their order)."""
    return sorted(plan, key=lambda v: PRIORITY.index(v[0]) if v[0] in PRIORITY else len(PRIORITY))


def classify(text):
    """'oom' when the text mentions an out-of-memory error, else 'error'."""
    return "oom" if OOM.search(text or "") else "error"


def failure_text(text, limit=300):
    """The first 300 characters of the interesting part of a failure: the first line that names an error, else the last lines."""
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    for i, ln in enumerate(lines):
        if re.search(r"error|exception|unknown|unrecognized|invalid|exhausted|out of memory|fatal|check failed|flag", ln, re.IGNORECASE):
            return " | ".join(lines[i:i + 4])[:limit]
    return " | ".join(lines[-4:])[-limit:] if lines else "(no output)"


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# worker: one variant in one process
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


def memory_stats(jax):
    try:
        st = jax.devices()[0].memory_stats()
    except Exception:  # noqa: BLE001
        return None
    if not st:
        return None
    return {k: int(st[k]) for k in ("bytes_in_use", "peak_bytes_in_use", "bytes_limit") if k in st}


def timed(jax, f, *args, n=10):
    """Median seconds of f(*args) over n runs after two warm-up runs (the first one compiles), each ended by block_until_ready."""
    for _ in range(2):
        jax.block_until_ready(f(*args))
    ts = []
    for _ in range(n):
        t0 = time.time()
        jax.block_until_ready(f(*args))
        ts.append(time.time() - t0)
    return statistics.median(ts)


def breakdown(fj, jax, jnp, c, h, cdt, a_, muon_mesh, params, m_, v_, t_, y_, ig_, lr, lr_muon, n):
    """Where the step time goes: the forward pass alone, forward + backward (the gradients, with their all-reduce), and the optimizer update alone, each as its own program on the real shapes. The sum is a little
    different from the one-program step (no overlap between the parts, no fusion across them); the point is the SHARES: attention and matmuls (forward, backward) against the optimizer."""
    remat = not a_.no_remat

    def loss(p, t, y, i):
        return fj.loss_fn(c, p, t, y, i, cdt, a_.ce_chunk, remat, a_.remat_policy, a_.fuse_qkv, a_.attn_prep, a_.layer_opts)

    fwd, vg = jax.jit(loss), jax.jit(jax.value_and_grad(loss))
    t_fwd = timed(jax, fwd, params, t_, y_, ig_, n=n)
    _, g = vg(params, t_, y_, ig_)
    t_vg = timed(jax, vg, params, t_, y_, ig_, n=n)
    sc = jnp.asarray(1, jnp.int32)
    upd = jax.jit(lambda p, g_, m, v, s: fj.apply_update(c, h, p, g_, m, v, s, lr, lr_muon, muon_mesh))  # no donation: the timing reuses its inputs
    t_upd = timed(jax, upd, params, g, m_, v_, sc, n=n)
    d = {"forward_ms": round(t_fwd * 1e3, 1), "backward_ms": round((t_vg - t_fwd) * 1e3, 1), "forward_backward_ms": round(t_vg * 1e3, 1), "optimizer_ms": round(t_upd * 1e3, 1), "sum_ms": round((t_vg + t_upd) * 1e3, 1)}
    d["text"] = f"forward {d['forward_ms']} ms, backward {d['backward_ms']} ms, optimizer {d['optimizer_ms']} ms (sum {d['sum_ms']} ms)"
    return d


def worker(a):
    info = {"variant": a.name, "args": a.args}
    t_start = time.time()
    try:
        fail = os.environ.get("FORGE_BENCH_TEST_FAIL", "")
        if fail == "crash":
            print("Unknown flag in LIBTPU_INIT_ARGS: --xla_test_flag (test)", file=sys.stderr, flush=True)
            os._exit(3)
        if fail == "error":
            raise RuntimeError("deliberate test failure of the variant")
        if fail == "oom":
            raise RuntimeError("RESOURCE_EXHAUSTED: Ran out of memory in memory space hbm (deliberate test failure)")
        import numpy as np

        sys.path.insert(0, HERE)
        import forge_jax as fj
        import jax
        import jax.numpy as jnp

        a_ = fj.parse_args(["train", "--data", "-", "--out", "-"] + shlex.split(a.args))
        c = fj.Cfg(n_layer=a_.layers, n_embd=a_.dim, n_head=a_.heads, n_kv=a_.kv or a_.heads, hd=a_.head_dim, n_ff=a_.ffn, vocab=a_.vocab, seq=a_.seq, qk_norm=a_.qk_norm, theta=a_.rope_theta, untied=a_.untied,
                   relu2=a_.relu2, softcap=a_.softcap, lambdas=a_.lambdas, skips=a_.skips, ve_layers=tuple(int(x) for x in a_.value_emb.split(",") if x),
                   window=a_.window, window_layers=fj.parse_window_layers(a_.window_layers, a_.window, a_.layers))
        muon_lr = a_.muon_lr if a_.muon_lr is not None else a_.lr
        h = fj.Hyper(peak_lr=a_.lr, muon_lr=muon_lr, warmup=a_.warmup, min_frac=a_.min_lr_frac, decay_frac=a_.decay_frac, wd=a_.wd, muon_wd=a_.muon_wd, mu=a_.muon_mu, beta2=a_.beta2, clip=a_.clip,
                     mult_emb=a_.lr_emb_mult, mult_head=a_.lr_head_mult, mult_table=a_.lr_table_mult, mult_scalar=a_.lr_scalar_mult, use_muon=(a_.optim == "muon"), muon_lean=a_.muon_lean, ns_barrier=a_.ns_barrier)
        cdt = fj.BF16 if a_.dtype == "bf16" else fj.F32
        mesh, batch_sh, _, repl_sh = fj.make_mesh()
        ndev = len(jax.devices())
        muon_mesh = mesh if (a_.muon_shard and h.use_muon) else None
        per_micro = ndev * max(1, a_.micro)
        info.update(backend=jax.default_backend(), devices=ndev, device_kind=jax.devices()[0].device_kind, jax=jax.__version__, sequences=per_micro, seq=c.seq, tokens_per_step=per_micro * c.seq,
                    window=c.window, window_layers=list(c.window_layers), muon_shard=muon_mesh is not None, remat=not a_.no_remat, remat_policy=a_.remat_policy, fuse_qkv=a_.fuse_qkv, attn_prep=a_.attn_prep, layer_opts=a_.layer_opts, micro=a_.micro, libtpu_init_args=os.environ.get("LIBTPU_INIT_ARGS", ""))
        attn_name = "XLA ops"
        if a_.attn != "xla" and jax.default_backend() == "tpu":
            seen = []
            pick = a.attn_pick
            cands = ((int(pick.split(",")[0]), pick.split(",")[1] == "1"),) if pick else ((512, False), (512, True), (256, False), (1024, False))
            # the pick of the first variant was chosen and checked against XLA's attention there: the others build that kernel without the self-test (it compiles XLA's attention and the kernel again: about 40 s a variant).
            # Not for a model with sliding-window layers: base has none, so the second (windowed) kernel of such a variant was never checked, and select_attention checks both
            windowed = any(c.window_of(l) for l in range(c.n_layer))
            picked = fj.attention_from_pick(c, mesh, pick, log=seen.append, **fj.splash_names(a_.remat_policy)) if pick and not windowed else None
            if picked is not None:
                fj.ATTN["impl"] = picked
                blk, fused = fj.parse_attn_pick(pick)
                attn_name = f"splash block {blk}{' fused backward' if fused else ''}"
                info["attn_pick"] = pick
            else:
                try:
                    fj.ATTN["impl"] = fj.select_attention(c, mesh, cdt, per_micro, force=bool(pick) or a_.attn == "splash", candidates=cands, log=seen.append,
                                                      **fj.splash_names(a_.remat_policy))
                except Exception as e:  # noqa: BLE001
                    seen.append(f"[attn] selection failed ({type(e).__name__}): XLA attention")
                    fj.ATTN["impl"] = None
                m = [re.search(r"using (splash block (\d+)( fused backward)?)", s) for s in seen]
                m = [x for x in m if x]
                if fj.ATTN["impl"] is not None and m:
                    attn_name = m[-1].group(1)
                    info["attn_pick"] = f"{m[-1].group(2)},{1 if m[-1].group(3) else 0}"
                else:
                    attn_name = "XLA ops"
                    info["attn_pick"] = "xla"
        info["attention"] = attn_name
        params = fj.init_params(c, 1)
        m_ = {n: jnp.zeros(s, fj.F32) for n, s in fj.layout(c)}
        v_ = {n: jnp.zeros(s, fj.F32) for n, s in fj.layout(c)}
        params, m_, v_ = (jax.device_put(t, repl_sh) for t in (params, m_, v_))
        step_count = jax.device_put(jnp.asarray(0, jnp.int32), repl_sh)
        rng = np.random.default_rng(0)
        seqs = rng.integers(0, c.vocab, size=(per_micro, c.seq + 1)).astype(np.int32)
        tokens, targets = (jax.device_put(x[None], batch_sh) for x in (seqs[:, :-1], seqs[:, 1:]))
        ignore = jax.device_put(np.zeros((1, per_micro, c.seq), bool), batch_sh)
        step = fj.build_train_step(c, h, cdt, a_.ce_chunk, not a_.no_remat, muon_mesh, a_.remat_policy, a_.fuse_qkv, a_.attn_prep, a_.layer_opts)
        lr, lr_muon = jnp.float32(1e-4), jnp.float32(1e-4)
        times = []
        for i in range(a.warmup + a.steps):
            t0 = time.time()
            params, m_, v_, step_count, loss, gnorm = step(params, m_, v_, step_count, tokens, targets, ignore, lr, lr_muon)
            jax.block_until_ready((params, loss))
            dt = time.time() - t0
            if i == 0:
                info["first_step_s"] = round(dt, 2)  # compilation + the first step
                info["first_loss"] = float(loss)  # the loss of the first step depends only on the initial parameters and the data: variants that compute the same function agree on it (the in-session autotune checks that)
            if i >= a.warmup:
                times.append(dt)
            if not np.isfinite(float(loss)):
                raise FloatingPointError(f"the loss is not finite at step {i}")
        med = statistics.median(times)
        info.update(status="ok", median_s=med, min_s=min(times), tokens_per_s=info["tokens_per_step"] / med, final_loss=float(loss), memory=memory_stats(jax))
        if a.name.startswith("breakdown"):
            info["breakdown"] = breakdown(fj, jax, jnp, c, h, cdt, a_, muon_mesh, params, m_, v_, tokens[0], targets[0], ignore[0], lr, lr_muon, a.steps)
    except Exception as e:  # noqa: BLE001 (the whole point: report it instead of dying)
        msg = f"{type(e).__name__}: {e}"
        info.update(status=classify(msg), message=failure_text(msg))
    info["worker_s"] = round(time.time() - t_start, 1)
    print(json.dumps(info), flush=True)
    return 0


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# orchestrator
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


def parse_worker_output(out):
    """The last line of the worker's output that is a json object with a "variant", or None."""
    for line in reversed((out or "").splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if isinstance(d, dict) and "variant" in d:
                return d
    return None


def run_variant(name, args, extra, env_extra, timeout, a, attn_pick, env_base):
    cmd = [sys.executable, "-u", os.path.abspath(__file__), "--worker", "--name", name, "--args=" + (args + " " + extra).strip(), "--steps", str(a.steps), "--warmup", str(a.warmup)]
    if attn_pick:
        cmd += ["--attn-pick", attn_pick]
    env = dict(env_base)
    for k, val in env_extra.items():
        env[k] = (env[k] + " " + val) if k == "LIBTPU_INIT_ARGS" and env.get(k) else val
    t0 = time.time()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
    except subprocess.TimeoutExpired as e:
        txt = ((e.stdout or b"").decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")) + ((e.stderr or b"").decode(errors="replace") if isinstance(e.stderr, bytes) else (e.stderr or ""))
        return {"variant": name, "status": "timeout", "message": f"no result after {timeout:.0f} s " + failure_text(txt, 200), "worker_s": round(time.time() - t0, 1)}
    res = parse_worker_output(r.stdout)
    if res is None:  # the process died before it could report (a runtime that refuses a flag, a hard out-of-memory abort)
        both = (r.stdout or "") + "\n" + (r.stderr or "")
        res = {"variant": name, "status": classify(both), "message": f"exit code {r.returncode}: " + failure_text(both, 270), "worker_s": round(time.time() - t0, 1)}
    return res


def table(results, base_name="base"):
    """Rows sorted by tokens per second (failed variants last, in run order) as printable text, and the list of dicts with the gain over the base."""
    base = next((r for r in results if r["variant"] == base_name and r.get("status") == "ok"), None)
    rows = []
    for r in results:
        r = dict(r)
        r["gain_pct"] = round((r["tokens_per_s"] / base["tokens_per_s"] - 1.0) * 100.0, 1) if base and r.get("status") == "ok" else None
        rows.append(r)
    ok = sorted([r for r in rows if r.get("status") == "ok"], key=lambda r: -r["tokens_per_s"])
    bad = [r for r in rows if r.get("status") != "ok"]
    lines = [f"{'variant':28s} {'status':8s} {'median s':>9s} {'min s':>8s} {'tokens/s':>10s} {'gain':>8s} {'peak GiB':>9s}  note"]
    for r in ok + bad:
        if r.get("status") == "ok":
            mem = (r.get("memory") or {}).get("peak_bytes_in_use")
            lines.append(f"{r['variant']:28s} {'ok':8s} {r['median_s']:9.4f} {r['min_s']:8.4f} {r['tokens_per_s']:10.0f} {('%+.1f%%' % r['gain_pct']) if r['gain_pct'] is not None else 'n/a':>8s} "
                         f"{(mem / 2**30 if mem else float('nan')):9.1f}  {r.get('attention', '')}" + (f"; {r['note']}" if r.get("note") else "") + (f"; {r['breakdown']['text']}" if r.get("breakdown") else ""))
        else:
            lines.append(f"{r['variant']:28s} {r['status']:8s} {'-':>9s} {'-':>8s} {'-':>10s} {'-':>8s} {'-':>9s}  {r.get('message', '')[:300]}")
    return "\n".join(lines), ok + bad


def orchestrate(a):
    t0 = time.time()
    out = a.out
    os.makedirs(out, exist_ok=True)
    args = a.args or (ARGS_TINY if a.tiny else ARGS_365M)
    env_base = dict(os.environ, PYTHONUTF8="1")
    if a.tiny:
        env_base.update(JAX_PLATFORMS="cpu", XLA_FLAGS=("--xla_force_host_platform_device_count=8 " + env_base.get("XLA_FLAGS", "")).strip())
    wanted = [x for x in a.only.split(",") if x] if a.only else None
    plan = variants(a.tiny)
    if a.variants_json:  # a list of {"name", "flags", "env", "note"} instead of the built-in table (the in-session autotune of cloud/kaggle_tpu_train.py); `base` runs first, even when the list does not name it
        with open(a.variants_json) as f:
            given = [(str(v["name"]), str(v.get("flags", "")), dict(v.get("env") or {}), str(v.get("note", ""))) for v in json.load(f)]
        plan = [v for v in given if v[0] == "base"] or [("base", "", {}, "as the training kernel runs now")]
        plan += [v for v in given if v[0] != "base"]
    plan = [v for v in plan if wanted is None or v[0] in wanted or v[0] == "base"]
    if wanted is None and not a.variants_json and not a.tiny:
        plan = priority_order(plan)
    results, attn_pick = [], ""
    print(f"[bench] {len(plan)} variants, model args: {args}, deadline {a.deadline_s:.0f} s, at most {a.per_variant_s:.0f} s per variant", flush=True)

    def save():
        text, rows = table(results)
        doc = {"model_args": args, "steps": a.steps, "warmup": a.warmup, "elapsed_s": round(time.time() - t0, 1), "deadline_s": a.deadline_s, "tiny": a.tiny, "table": rows}
        with open(os.path.join(out, "bench.json"), "w") as f:
            json.dump(doc, f, indent=1)
        return text

    for name, extra, env_extra, note in plan:
        left = a.deadline_s - (time.time() - t0)
        if left < 45:
            results.append({"variant": name, "status": "skipped", "message": f"deadline: {left:.0f} s left", "note": note})
            print(f"[bench] {name}: skipped (deadline)", flush=True)
            continue
        pick_args = ("--attn xla" if attn_pick == "xla" else "")
        print(f"[bench] {name}: running ({left:.0f} s of the deadline left) ...", flush=True)
        res = run_variant(name, args, (extra + " " + pick_args).strip(), env_extra, min(a.per_variant_s, left - 5), a, attn_pick if attn_pick != "xla" else "", env_base)
        res["note"] = note
        res["extra_flags"] = extra
        res["env"] = env_extra
        if res.get("status") == "ok" and not attn_pick and res.get("attn_pick"):
            attn_pick = res["attn_pick"]  # the attention the first variant chose is used by all the others (the selection costs minutes of compilation)
        results.append(res)
        print(f"[bench] {name}: {res['status']}" + (f", {res['tokens_per_s']:.0f} tokens/s (median step {res['median_s']:.4f} s, first step {res.get('first_step_s')} s)" if res.get("status") == "ok" else f" {res.get('message', '')[:200]}"), flush=True)
        save()
    text = save()
    print("\n" + text, flush=True)
    if a.tiny:
        print("[bench] CPU self-test with a tiny model: these numbers say nothing about a TPU", flush=True)
    print(f"\n[bench] done in {time.time() - t0:.0f} s; results in {os.path.join(out, 'bench.json')}", flush=True)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--name", default="base", help=argparse.SUPPRESS)
    ap.add_argument("--attn-pick", default="", help=argparse.SUPPRESS)
    ap.add_argument("--args", default="", help="the model arguments (default: the 365M model of m365_ve8.json without its training-only flags; with --tiny a tiny model)")
    ap.add_argument("--out", default=".")
    ap.add_argument("--steps", type=int, default=12, help="timed steps")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--deadline-s", type=float, default=1500.0, help="no new variant is started when this many seconds of the whole run are used")
    ap.add_argument("--per-variant-s", type=float, default=420.0, help="a worker that has not reported after this long is killed (status timeout)")
    ap.add_argument("--only", default="", help="comma separated variant names (base always runs)")
    ap.add_argument("--variants-json", default="", help="a json file with a list of {name, flags, env, note} that replaces the built-in variants (base runs first)")
    ap.add_argument("--tiny", action="store_true", help="CPU self-test: a tiny model on 8 fake devices, with deliberately failing variants")
    a = ap.parse_args(argv)
    return worker(a) if a.worker else orchestrate(a)


if __name__ == "__main__":
    sys.exit(main())
