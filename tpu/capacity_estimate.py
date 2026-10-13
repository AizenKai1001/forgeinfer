"""Per-chip memory estimate of the data-parallel training step: compiles forge_jax.build_train_step for 8 fake CPU devices with abstract inputs (nothing is allocated) and prints XLA's memory_analysis.
It is the CPU compiler's buffer assignment with XLA attention, a conservative proxy and not a TPU measurement (docs/TPU_PREMORTEM.md, "Capacity").

    python capacity_estimate.py 365M 2          (model 150M, 365M, 600M or 1B; sequences per chip per micro-step; a third argument turns remat off)
"""
import os, sys, time
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, jax, jax.numpy as jnp
import forge_jax as fj

F32 = jnp.float32


def analyse(name, c, micro, remat=True):
    h = fj.Hyper(use_muon=True)
    mesh, batch_sh, eval_sh, repl = fj.make_mesh()
    nd = 8
    n = sum(int(np.prod(s)) for _, s in fj.layout(c))
    ps = {k: jax.ShapeDtypeStruct(s, F32, sharding=repl) for k, s in fj.layout(c)}
    step = fj.build_train_step(c, h, jnp.bfloat16, 256, remat)
    tok = jax.ShapeDtypeStruct((1, nd * micro, c.seq), jnp.int32, sharding=batch_sh)
    ig = jax.ShapeDtypeStruct((1, nd * micro, c.seq), jnp.bool_, sharding=batch_sh)
    sc = jax.ShapeDtypeStruct((), jnp.int32, sharding=repl)
    lr = jax.ShapeDtypeStruct((), F32, sharding=repl)
    t0 = time.time()
    comp = step.lower(ps, ps, ps, sc, tok, tok, ig, lr, lr).compile()
    ma = comp.memory_analysis()
    g = 1e9
    peak = (ma.argument_size_in_bytes + ma.output_size_in_bytes - ma.alias_size_in_bytes + ma.temp_size_in_bytes) / g
    print(f"{name}: {n/1e6:.0f} M params micro {micro} remat {remat}: args {ma.argument_size_in_bytes/g:.2f} GB, out {ma.output_size_in_bytes/g:.2f}, alias {ma.alias_size_in_bytes/g:.2f}, temp {ma.temp_size_in_bytes/g:.2f}; peak ~ {peak:.2f} GB (12N = {12*n/g:.2f}) [{time.time()-t0:.0f} s]", flush=True)


CFGS = {
    "150M": fj.Cfg(n_layer=30, n_embd=576, n_head=9, n_kv=3, hd=64, n_ff=2304, seq=2048, qk_norm=True, relu2=True, lambdas=True, ve_layers=(0, 1, 2, 3, 26, 27, 28, 29)),
    "365M": fj.Cfg(n_layer=24, n_embd=1024, n_head=16, n_kv=4, hd=64, n_ff=4096, seq=2048, qk_norm=True, relu2=True, lambdas=True, ve_layers=(0, 1, 2, 3, 20, 21, 22, 23)),
    "600M": fj.Cfg(n_layer=28, n_embd=1280, n_head=20, n_kv=4, hd=64, n_ff=5120, seq=2048, qk_norm=True, relu2=True, lambdas=True, ve_layers=(0, 1, 2, 3, 24, 25, 26, 27)),
    "1B": fj.Cfg(n_layer=32, n_embd=1664, n_head=26, n_kv=2, hd=64, n_ff=6656, seq=2048, qk_norm=True, relu2=True, lambdas=True, ve_layers=(0, 1, 2, 3, 28, 29, 30, 31)),
}

if __name__ == "__main__":
    name, micro = sys.argv[1], int(sys.argv[2])
    analyse(name, CFGS[name], micro, remat=(len(sys.argv) < 4))
