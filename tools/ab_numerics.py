"""Numerical A/B of two configurations of the same model (environment settings of one binary): KL, top-1 agreement and perplexity change of configuration B against the logits of configuration A on the
same texts, from `forge ppl` (the prompt GEMMs run in both: ppl reads whole windows).
    python tools/ab_numerics.py MODEL.gguf "ENV_A" "ENV_B" TEXT_FILE[,TEXT_FILE..] [ctx] [max_tokens]
ENV is a list like "FORGE_MMQ=0,FORGE_F16ACC=1" (empty: the defaults). Configuration A writes the top-64 log-probabilities of every position (--dump-ref), configuration B is compared with them (--ref):
`kl_top64` is the mean KL divergence in nats over the top 64 tokens of A, `top1_agree` how often both pick the same token, `dnll` the change of the mean negative log-likelihood of the text.
Environment: FORGE_BIN (target/release/forge); the dumps go to ./ab_refs/."""
import os
import subprocess
import sys

if len(sys.argv) < 5:
    sys.exit(__doc__)
binary = os.path.abspath(os.environ.get("FORGE_BIN", "target/release/forge"))
model, env_a, env_b, texts = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4].split(",")
ctx = sys.argv[5] if len(sys.argv) > 5 else "1024"
mx = sys.argv[6] if len(sys.argv) > 6 else "4096"
os.makedirs("ab_refs", exist_ok=True)
refs = ",".join(f"ab_refs/{os.path.basename(model)}.{os.path.basename(t)}.{ctx}.ref" for t in texts)
common = ["ppl", model, "--native", "--text", ",".join(texts), "--ctx", ctx, "--max-tokens", mx]


def env_of(spec):
    e = dict(os.environ)
    for kv in filter(None, spec.split(",")):
        k, v = kv.split("=", 1)
        e[k] = v
    return e


print(f"== A: {env_a or '(defaults)'}: dump", flush=True)
a = subprocess.run([binary] + common + ["--dump-ref", refs], capture_output=True, text=True, env=env_of(env_a))
print("\n".join(l for l in (a.stdout + a.stderr).splitlines() if "RESULT" in l or "error" in l.lower()), flush=True)
print(f"== B: {env_b or '(defaults)'}: compare", flush=True)
b = subprocess.run([binary] + common + ["--ref", refs], capture_output=True, text=True, env=env_of(env_b))
print("\n".join(l for l in (b.stdout + b.stderr).splitlines() if "RESULT" in l or "VS-REFERENCE" in l or "error" in l.lower()), flush=True)
