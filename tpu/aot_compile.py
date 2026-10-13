"""Deviceless TPU compile of the Forge-1 training step: does every speed variant of tpu/bench_step.py compile for a TPU v5e-8, how much memory per chip does it need, and what does XLA's own cost model say.

    python aot_compile.py [--args "MODEL ARGS"] [--only base,muon-shard,...] [--out DIR] [--topology v5e:2x4] [--block 512] [--hlo-dir DIR] [--per-variant-s 900]

No TPU is needed, only jax[tpu] (libtpu): `jax.experimental.topologies.get_topology_desc` describes the chips, the real train step (`forge_jax.build_train_step`, the same program the Kaggle session runs) is lowered
with abstract arguments (shapes and shardings, no data) and compiled by libtpu's own TPU compiler on the host CPU (a few minutes and a few GB of RAM per variant; the splash-attention Pallas kernels go through the
Mosaic compiler as well). What this proves: the variant lowers and compiles for the v5e (a flag that libtpu does not know, a sharding or a kernel that the TPU compiler rejects shows up as an error here, not after hours
in the Kaggle queue) and its per-chip memory (argument + temporary + output - aliased buffers) against the 15.75 GB that a v5e chip offers. What it does NOT prove: speed. XLA's cost model numbers (flops, bytes accessed, an
estimated time when libtpu gives one) are an estimate of a model, and a Pallas kernel is a black box to it; only a measured step (bench_step.py on a TPU) counts.
Every variant runs in its own worker process (LIBTPU_INIT_ARGS applies at start-up, the memory of a compile must be given back, a crash must not take the others with it) with the flags of bench_step.variants();
the attention is the one the real run picks on the v5e (splash, block 512, fused backward), `--attn xla` in a variant's flags selects XLA's. Results: DIR/aot.json (after every variant) and a table.
Memory statistics come from `compiled.memory_analysis()`, the op counts from the optimized HLO text (`compiled.as_text()`).
"""
import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bench_step as bs  # noqa: E402  (no jax at import time: the variants and the model arguments of the benchmark)
import hlo_cost  # noqa: E402  (the compiler's own cycle estimates in the HLO text)

HBM_USABLE = 15.75 * 2**30  # bytes a v5e chip offers (16 GB less the runtime's share), the figure the earlier review measured against
PEAK_FLOPS, PEAK_BYTES_S = 197e12, 819e9  # one v5e chip: bf16 matrix peak and HBM bandwidth (the spec sheet)
OPS = {"all-reduce": r"= \S+ all-reduce(-start)?\(", "reduce-scatter": r"= \S+ reduce-scatter\(", "all-gather": r"= \S+ all-gather(-start)?\(", "all-to-all": r"= \S+ all-to-all\(",
       "pallas-calls": r"custom_call_target=\"tpu_custom_call\"", "fusions": r"= \S+ fusion\(", "while-loops": r"= \S+ while\("}


def stat(obj, *names):
    for n in names:
        v = getattr(obj, n, None)
        if v is None and isinstance(obj, dict):
            v = obj.get(n)
        if v is not None:
            return v
    return None


def worker(a):
    info = {"variant": a.name, "args": a.args}
    t_start = time.time()
    try:
        os.environ.setdefault("TPU_SKIP_MDS_QUERY", "1")
        import numpy as np

        import forge_jax as fj
        import jax
        import jax.numpy as jnp
        from jax.experimental import topologies
        from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

        a_ = fj.parse_args(["train", "--data", "-", "--out", "-"] + shlex.split(a.args))
        c = fj.Cfg(n_layer=a_.layers, n_embd=a_.dim, n_head=a_.heads, n_kv=a_.kv or a_.heads, hd=a_.head_dim, n_ff=a_.ffn, vocab=a_.vocab, seq=a_.seq, qk_norm=a_.qk_norm, theta=a_.rope_theta, untied=a_.untied,
                   relu2=a_.relu2, softcap=a_.softcap, lambdas=a_.lambdas, skips=a_.skips, ve_layers=tuple(int(x) for x in a_.value_emb.split(",") if x),
                   window=a_.window, window_layers=fj.parse_window_layers(a_.window_layers, a_.window, a_.layers))
        muon_lr = a_.muon_lr if a_.muon_lr is not None else a_.lr
        h = fj.Hyper(peak_lr=a_.lr, muon_lr=muon_lr, warmup=a_.warmup, min_frac=a_.min_lr_frac, decay_frac=a_.decay_frac, wd=a_.wd, muon_wd=a_.muon_wd, mu=a_.muon_mu, beta2=a_.beta2, clip=a_.clip,
                     mult_emb=a_.lr_emb_mult, mult_head=a_.lr_head_mult, mult_table=a_.lr_table_mult, mult_scalar=a_.lr_scalar_mult, use_muon=(a_.optim == "muon"), muon_lean=a_.muon_lean, ns_barrier=a_.ns_barrier)
        cdt = fj.BF16 if a_.dtype == "bf16" else fj.F32
        topo = topologies.get_topology_desc(topology_name=a.topology, platform="tpu")
        devs = np.array(topo.devices)
        mesh = Mesh(devs, ("data",))
        repl, batch = NamedSharding(mesh, P()), NamedSharding(mesh, P(None, "data"))
        ndev = len(devs)
        per_micro = ndev * max(1, a_.micro)
        muon_mesh = mesh if (a_.muon_shard and h.use_muon) else None
        info.update(devices=ndev, device_kind=devs[0].device_kind, jax=jax.__version__, sequences=per_micro, tokens_per_step=per_micro * c.seq, micro=a_.micro, muon_shard=muon_mesh is not None,
                    remat=not a_.no_remat, remat_policy=a_.remat_policy, fuse_qkv=a_.fuse_qkv, attn_prep=a_.attn_prep, layer_opts=a_.layer_opts, window=c.window, libtpu_init_args=os.environ.get("LIBTPU_INIT_ARGS", ""))
        if a_.attn == "xla":
            fj.ATTN["impl"] = None
            info["attention"] = "XLA ops"
        else:
            fj.ATTN["impl"] = fj.make_splash_attention(c, mesh, block=a.block, fused_bwd=not a.no_fused_bwd, **fj.splash_names(a_.remat_policy))
            info["attention"] = f"splash block {a.block}" + ("" if a.no_fused_bwd else " fused backward")
        n_params = sum(int(np.prod(s)) for _, s in fj.layout(c))
        info["n_params"] = n_params
        sds = jax.ShapeDtypeStruct
        params = {n: sds(s, fj.F32, sharding=repl) for n, s in fj.layout(c)}
        m_, v_ = dict(params), dict(params)
        step = fj.build_train_step(c, h, cdt, a_.ce_chunk, not a_.no_remat, muon_mesh, a_.remat_policy, a_.fuse_qkv, a_.attn_prep, a_.layer_opts)
        args = (params, m_, v_, sds((), jnp.int32, sharding=repl), sds((1, per_micro, c.seq), jnp.int32, sharding=batch), sds((1, per_micro, c.seq), jnp.int32, sharding=batch),
                sds((1, per_micro, c.seq), jnp.bool_, sharding=batch), sds((), jnp.float32, sharding=repl), sds((), jnp.float32, sharding=repl))
        t0 = time.time()
        lowered = step.lower(*args)
        info["lower_s"] = round(time.time() - t0, 1)
        t0 = time.time()
        compiled = lowered.compile()
        info["compile_s"] = round(time.time() - t0, 1)
        ma = compiled.memory_analysis()
        mem = {k: stat(ma, k) for k in ("argument_size_in_bytes", "output_size_in_bytes", "alias_size_in_bytes", "temp_size_in_bytes", "generated_code_size_in_bytes")}
        if all(isinstance(mem[k], (int, float)) for k in ("argument_size_in_bytes", "output_size_in_bytes", "alias_size_in_bytes", "temp_size_in_bytes")):
            mem["peak_bytes"] = int(mem["argument_size_in_bytes"] + mem["output_size_in_bytes"] - mem["alias_size_in_bytes"] + mem["temp_size_in_bytes"])
            mem["fits"] = mem["peak_bytes"] <= HBM_USABLE
        info["memory"] = mem
        try:
            ca = compiled.cost_analysis()
            ca = ca[0] if isinstance(ca, (list, tuple)) and ca else ca
            info["cost"] = {k: float(v) for k, v in (ca or {}).items() if isinstance(v, (int, float)) and (k in ("flops", "bytes accessed", "optimal_seconds", "transcendentals") or "second" in k)}
            info["cost_keys"] = sorted((ca or {}).keys())[:40]
            if info["cost"].get("flops") and info["cost"].get("bytes accessed"):
                # lower bounds of the step time on one chip: the matrix work at the peak rate, the memory traffic at the peak bandwidth (XLA counts every operand and result of every fusion; a Pallas kernel is not counted at all)
                info["bound_flops_s"] = round(info["cost"]["flops"] / PEAK_FLOPS, 4)
                info["bound_bytes_s"] = round(info["cost"]["bytes accessed"] / PEAK_BYTES_S, 4)
        except Exception as e:  # noqa: BLE001
            info["cost"] = {"error": f"{type(e).__name__}: {str(e)[:120]}"}
        text = compiled.as_text()
        info["hlo_ops"] = {k: len(re.findall(p, text)) for k, p in OPS.items()}
        est = hlo_cost.analyze(text)
        info["est_step_ms"] = round(est["total_ms"], 1)
        info["est_regions"] = {k: round(v["ms"], 1) for k, v in est["regions"].items()}
        try:  # the matrix-unit work against the peak rate (python tpu/hlo_cost.py FILE --matmuls N lists the worst fusions); a failure here must not lose the compile result
            mm_rows = hlo_cost.matmul_fusions(text)
            info["matmul_est_ms"], info["matmul_ideal_ms"], info["matmul_fusions"] = round(sum(r["est_ms"] for r in mm_rows), 1), round(sum(r["ideal_ms"] for r in mm_rows), 1), len(mm_rows)
        except Exception as e:  # noqa: BLE001
            info["matmul_error"] = f"{type(e).__name__}: {str(e)[:100]}"
        info["hlo_chars"] = len(text)
        if a.hlo_dir:
            os.makedirs(a.hlo_dir, exist_ok=True)
            with open(os.path.join(a.hlo_dir, a.name + ".hlo.txt"), "w") as f:
                f.write(text)
        info["status"] = "ok"
    except Exception as e:  # noqa: BLE001 (the point of the tool: report it instead of dying)
        msg = f"{type(e).__name__}: {e}"
        info.update(status=bs.classify(msg), message=bs.failure_text(msg, 400))
    info["worker_s"] = round(time.time() - t_start, 1)
    print(json.dumps(info), flush=True)
    return 0


def run_variant(name, args, extra, env_extra, a):
    cmd = [sys.executable, "-u", os.path.abspath(__file__), "--worker", "--name", name, "--args=" + (args + " " + extra).strip(), "--topology", a.topology, "--block", str(a.block)]
    cmd += (["--no-fused-bwd"] if a.no_fused_bwd else []) + (["--hlo-dir", a.hlo_dir] if a.hlo_dir else [])
    env = dict(os.environ, PYTHONUTF8="1", TPU_SKIP_MDS_QUERY="1", ALLOW_MULTIPLE_LIBTPU_LOAD="1")  # the last one: libtpu's lockfile otherwise makes two lab runs at the same time abort (a compile needs no device)
    for k, val in env_extra.items():
        env[k] = (env[k] + " " + val) if k == "LIBTPU_INIT_ARGS" and env.get(k) else val
    t0 = time.time()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=a.per_variant_s, env=env)
    except subprocess.TimeoutExpired:
        return {"variant": name, "status": "timeout", "message": f"no result after {a.per_variant_s:.0f} s", "worker_s": round(time.time() - t0, 1)}
    res = bs.parse_worker_output(r.stdout)
    if res is None:
        both = (r.stdout or "") + "\n" + (r.stderr or "")
        res = {"variant": name, "status": bs.classify(both), "message": f"exit code {r.returncode}: " + bs.failure_text(both, 400), "worker_s": round(time.time() - t0, 1)}
    return res


def table(rows):
    base = next((r for r in rows if r["variant"] == "base" and r.get("status") == "ok" and r.get("est_step_ms")), None)
    for r in rows:
        r["est_vs_base_pct"] = round((r["est_step_ms"] / base["est_step_ms"] - 1.0) * 100.0, 1) if base and r.get("status") == "ok" and r.get("est_step_ms") else None
    lines = [f"{'variant':26s} {'status':8s} {'compile s':>9s} {'peak GiB':>9s} {'fits':>5s} {'arg GiB':>8s} {'temp GiB':>8s} {'Tflop':>8s} {'GB moved':>9s} {'t flops':>8s} {'t bytes':>8s} {'est ms':>7s} {'vs base':>8s}  ops / note"]
    for r in rows:
        if r.get("status") != "ok":
            lines.append(f"{r['variant']:26s} {r['status']:8s} {'-':>9s} {'-':>9s} {'-':>5s} {'-':>8s} {'-':>8s} {'-':>8s} {'-':>9s} {'-':>8s} {'-':>8s} {'-':>7s} {'-':>8s}  {r.get('message', '')[:260]}")
            continue
        m, cost, ops = r.get("memory") or {}, r.get("cost") or {}, r.get("hlo_ops") or {}
        g = lambda v: f"{v / 2**30:.2f}" if isinstance(v, (int, float)) else "n/a"  # noqa: E731
        lines.append(f"{r['variant']:26s} {'ok':8s} {r.get('compile_s', 0):9.1f} {g(m.get('peak_bytes')):>9s} {('yes' if m.get('fits') else 'NO') if 'fits' in m else 'n/a':>5s} {g(m.get('argument_size_in_bytes')):>8s} "
                     f"{g(m.get('temp_size_in_bytes')):>8s} {cost.get('flops', float('nan')) / 1e12:8.1f} {cost.get('bytes accessed', float('nan')) / 1e9:9.1f} {r.get('bound_flops_s', float('nan')):8.3f} {r.get('bound_bytes_s', float('nan')):8.3f} "
                     f"{r.get('est_step_ms', float('nan')):7.1f} {('%+.1f%%' % r['est_vs_base_pct']) if r.get('est_vs_base_pct') is not None else 'n/a':>8s}  "
                     + " ".join(f"{k}={v}" for k, v in ops.items() if v) + (f"; {r['note']}" if r.get("note") else ""))
    return "\n".join(lines)


def orchestrate(a):
    t0 = time.time()
    os.makedirs(a.out, exist_ok=True)
    args = a.args or bs.ARGS_365M
    wanted = [x for x in a.only.split(",") if x] if a.only else None
    plan = [v for v in bs.variants(False) if not v[0].startswith("breakdown") and (wanted is None or v[0] in wanted)]
    results = []
    print(f"[aot] {len(plan)} variants, topology {a.topology}, model args: {args}", flush=True)
    for name, extra, env_extra, note in plan:
        print(f"[aot] {name}: compiling ...", flush=True)
        res = run_variant(name, args, extra, env_extra, a)
        res["note"], res["extra_flags"], res["env"] = note, extra, env_extra
        results.append(res)
        print(f"[aot] {name}: {res['status']}" + (f", compile {res.get('compile_s')} s, peak {(res.get('memory') or {}).get('peak_bytes', 0) / 2**30:.2f} GiB" if res.get("status") == "ok" else f" {res.get('message', '')[:200]}"), flush=True)
        with open(os.path.join(a.out, "aot.json"), "w") as f:
            json.dump({"model_args": args, "topology": a.topology, "elapsed_s": round(time.time() - t0, 1), "table": results}, f, indent=1)
    print("\n" + table(results), flush=True)
    print("[aot] est ms = the compiler's own cycle estimate of the step (tpu/hlo_cost.py; 0.365 s for base against 0.379 s measured on a v5e-8); it does not look inside Pallas kernels, so the window and the block-size variants "
          "cannot show a gain there; a change of the XLA-compiled part (muon-shard, fuse-qkv, layouts, fusions) is what it ranks", flush=True)
    print(f"\n[aot] done in {time.time() - t0:.0f} s; results in {os.path.join(a.out, 'aot.json')}", flush=True)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--name", default="base", help=argparse.SUPPRESS)
    ap.add_argument("--args", default="", help="the model arguments (default: the 365M model of bench_step.ARGS_365M)")
    ap.add_argument("--out", default=".")
    ap.add_argument("--only", default="", help="comma separated variant names of bench_step.variants()")
    ap.add_argument("--topology", default="v5e:2x4", help="the TPU slice to compile for (v5e:2x4 = the 8 chips of a Kaggle v5e-8)")
    ap.add_argument("--block", type=int, default=512, help="splash block size (the real run picked 512 with the fused backward kernel)")
    ap.add_argument("--no-fused-bwd", action="store_true")
    ap.add_argument("--hlo-dir", default="", help="also write the optimized HLO text of every variant here")
    ap.add_argument("--per-variant-s", type=float, default=900.0)
    a = ap.parse_args(argv)
    return worker(a) if a.worker else orchestrate(a)


if __name__ == "__main__":
    sys.exit(main())
