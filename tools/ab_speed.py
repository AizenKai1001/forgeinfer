"""Prompt-reading speed of two configurations of the same binary (environment settings), interleaved within the same minutes: the sustained clock of a GPU drifts by several percent between hours, so only
numbers measured next to each other can be compared.
    python tools/ab_speed.py MODEL_NAME[,MODEL_NAME..] TOKENS[,TOKENS..] "ENV_A" "ENV_B" [rounds]
ENV is a list like "FORGE_MMQ=0,FORGE_F16ACC=1" (empty: the defaults). Each cell is `forge prefill-bench MODEL.gguf --native --tokens N --reps 4` (best of 4 repetitions, tokens/s), run `rounds` times
(default 3) alternately with A and B; the table shows the best of each and the ratio B / A.
Environment: FORGE_BIN (target/release/forge), BENCH_MODEL_DIR (MODEL_NAME.gguf in it)."""
import os
import re
import subprocess
import sys

if len(sys.argv) < 5:
    sys.exit(__doc__)
binary = os.path.abspath(os.environ.get("FORGE_BIN", "target/release/forge"))
models_dir = os.environ.get("BENCH_MODEL_DIR", "models")
models, sizes, env_a, env_b = sys.argv[1].split(","), [int(x) for x in sys.argv[2].split(",")], sys.argv[3], sys.argv[4]
rounds = int(sys.argv[5]) if len(sys.argv) > 5 else 3


def env_of(spec):
    e = dict(os.environ)
    for kv in filter(None, spec.split(",")):
        k, v = kv.split("=", 1)
        e[k] = v
    return e


def run(model, n, spec):
    r = subprocess.run([binary, "prefill-bench", f"{models_dir}/{model}.gguf", "--native", "--tokens", str(n), "--reps", "4"], capture_output=True, text=True, env=env_of(spec), timeout=250)
    m = re.search(r"best (\d+) tokens/s", r.stdout)
    return int(m.group(1)) if m else -1


for model in models:
    for n in sizes:
        a, b = [], []
        for _ in range(rounds):
            a.append(run(model, n, env_a))
            b.append(run(model, n, env_b))
        ba, bb = max(a), max(b)
        print(f"{model:28s} {n:5d} tokens   A [{env_a or 'defaults'}] {ba:6d} (runs {a})   B [{env_b or 'defaults'}] {bb:6d} (runs {b})   B/A {bb / ba:5.3f}", flush=True)
