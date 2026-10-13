"""The five chat workloads of pld_bench.py (a code edit, a JSON array, a summary with quotes, a grammar fix, free prose) against forge serve, llama.cpp's llama-server and the llama-server that ships inside LM Studio:
greedy, thinking off, the servers' own timings; or, with --prefill, the speed of reading a prompt.
    python tools/bench_engines.py MODEL_NAME PRESET [PRESET ...]
    python tools/bench_engines.py --prefill TOKENS MODEL_NAME PRESET [PRESET ...]
Presets: forge-plain (--no-speculate), forge-lookup (prompt lookup only), forge-head (lookup + the model's MTP head), llama (llama-server as it comes), llama-ngram-simple, llama-ngram-mod, llama-ngram-map-k,
llama-ngram-short (ngram-simple with a 4-token lookup and 16 drafted tokens), llama-mtp (--spec-type draft-mtp), lmstudio (LM Studio's own llama-server, started from its runtime folder).
Prints tokens per second per workload, how often the text equals the first preset's, the drafts a llama-server reports, the time until the server answers and the video memory it holds (nvidia-smi, above what was in use before).
Environment: FORGE_BIN (target/release/forge), LLAMA_SERVER (llama-server), LMSTUDIO_BACKEND (the folder of LM Studio's llama.cpp runtime that holds llama-server; its CUDA libraries are expected in LMSTUDIO_VENDOR),
BENCH_MODEL_DIR (models: MODEL_NAME.gguf in it, heads as mtp/BASE-mtp.safetensors), BENCH_LOGDIR (.)."""
import json
import os
import subprocess
import sys
import time
import urllib.request

FORGE = os.environ.get("FORGE_BIN", "target/release/forge")
LLAMA = os.environ.get("LLAMA_SERVER", "llama-server")
LMS_DIR = os.environ.get("LMSTUDIO_BACKEND", "")
LMS_VENDOR = os.environ.get("LMSTUDIO_VENDOR", "")
MODELS = os.environ.get("BENCH_MODEL_DIR", "models")
LOGDIR = os.environ.get("BENCH_LOGDIR", ".")
PORT = 18560
SENT = "The forge glowed red while the smith turned the iron and listed the numbers of the day: "

args = sys.argv[1:]
prefill = None
if args and args[0] == "--prefill":
    prefill, args = int(args[1]), args[2:]
name, presets = args[0], args[1:]
base = name.replace("-Q4_K_M", "")
HEAD = f"{MODELS}/mtp/{base}-mtp.safetensors"

# the prompts are the ones of pld_bench.py (read from there so that both stay the same)
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "pld_bench.py"), encoding="utf-8").read()
ns = {}
exec(src[src.index("CODE = '''"):src.index("def run(pld):")], ns)
WORK = ns["WORK"]

PRESETS = {
    "lmstudio": ("lms", []),
    "llama": ("llama", []),
    "llama-ngram-simple": ("llama", ["--spec-type", "ngram-simple"]),
    "llama-ngram-mod": ("llama", ["--spec-type", "ngram-mod"]),
    "llama-ngram-map-k": ("llama", ["--spec-type", "ngram-map-k"]),
    "llama-ngram-short": ("llama", ["--spec-type", "ngram-simple", "--spec-ngram-simple-size-n", "4", "--spec-ngram-simple-size-m", "16"]),
    "llama-mtp": ("llama", ["--spec-type", "draft-mtp"]),
    "forge-plain": ("forge", ["--no-speculate"]),
    "forge-lookup": ("forge", []),
    "forge-head": ("forge", ["--mtp", HEAD, "--depth", "3"]),
}


def vram_used():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20).stdout
        return int(out.strip().splitlines()[0])
    except Exception:
        return 0


def post(path, body):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=200))


def start(preset, ctx):
    kind, extra = PRESETS[preset]
    env = None
    if kind in ("llama", "lms"):
        exe = LLAMA if kind == "llama" else os.path.join(LMS_DIR, "llama-server")
        cmd = [exe, "-m", f"{MODELS}/{name}.gguf", "--port", str(PORT), "-c", str(ctx), "-np", "1", "-ngl", "999", "--jinja"] + extra
        if kind == "lms":
            add = os.pathsep.join(p for p in (LMS_VENDOR, LMS_DIR) if p)
            env = dict(os.environ, PATH=add + os.pathsep + os.environ.get("PATH", ""), LD_LIBRARY_PATH=add + os.pathsep + os.environ.get("LD_LIBRARY_PATH", ""))
    else:
        cmd = [FORGE, "serve", f"{MODELS}/{name}.gguf", "--port", str(PORT), "--ctx", str(ctx)] + extra
    log = open(f"{LOGDIR}/bench_{preset}.log", "w")
    return subprocess.Popen(cmd, stdout=log, stderr=log, env=env), log


def wait_ready(p):
    for _ in range(240):
        if p.poll() is not None:
            return False
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models", timeout=1)
            return True
        except Exception:
            time.sleep(0.5)
    return False


def stop(p, log):
    p.terminate()
    try:
        p.wait(15)
    except Exception:
        p.kill()
    time.sleep(2)
    log.close()


def run(preset):
    base_vram = vram_used()
    t_start = time.time()
    p, log = start(preset, 4096)
    try:
        ready = wait_ready(p)
        ready_s = time.time() - t_start
        if not ready:
            tail = open(f"{LOGDIR}/bench_{preset}.log", encoding="utf-8", errors="replace").read().strip().splitlines()[-6:]
            return {"error": "server did not start: " + " | ".join(tail)[:400]}
        rows = []
        for label, prompt, mx in [("warm-up", "Say hello in one short sentence.", 24)] + WORK:
            r = post("/v1/chat/completions", {"model": "x", "messages": [{"role": "user", "content": prompt}], "max_tokens": mx, "temperature": 0, "seed": 7, "chat_template_kwargs": {"enable_thinking": False}})
            t = r["timings"]
            if label != "warm-up":
                rows.append({"label": label, "n": t["predicted_n"], "pn": t.get("prompt_n", 0), "tps": t["predicted_per_second"], "draft_n": t.get("draft_n"), "draft_ok": t.get("draft_n_accepted"), "text": r["choices"][0]["message"]["content"]})
        return {"rows": rows, "ready_s": ready_s, "vram_mib": vram_used() - base_vram}
    finally:
        stop(p, log)


def run_prefill(preset):
    p, log = start(preset, max(4096, prefill + 512))
    try:
        if not wait_ready(p):
            return None
        runs = []
        for i in range(6):
            # the number at the front keeps a server's prompt cache from helping
            prompt = f"Run {1000 + i * 7 + len(name)}. " + SENT * (prefill // 18) + "1, 2, 3, 4, 5,"
            t = post("/v1/completions", {"model": "x", "prompt": prompt, "max_tokens": 8, "temperature": 0})["timings"]
            if i:
                runs.append((t["prompt_per_second"], t["prompt_n"]))
        runs.sort()
        return runs[len(runs) // 2]
    finally:
        stop(p, log)


if prefill:
    print(f"{name}: a prompt of about {prefill} tokens, prompt processing in tokens/s (median of 5)")
    for pr in presets:
        r = run_prefill(pr)
        print(f"  {pr:20s} " + (f"{r[0]:8.0f}  ({r[1]} tokens)" if r else "server did not start"), flush=True)
    sys.exit(0)

results = {}
for pr in presets:
    results[pr] = run(pr)
    print(f"[{pr}] done", flush=True)
json.dump(results, open(f"{LOGDIR}/bench_{name}.json", "w"))

print()
print(f"{name}   greedy, thinking off, the servers' own timings (tokens/s of generation)")
print(f"{'preset':20s}" + "".join(f"{w[0]:>13s}" for w in WORK))
first = None
for pr in presets:
    r = results[pr]
    if "error" in r:
        print(f"{pr:20s} {r['error']}")
        continue
    first = first or r
    print(f"{pr:20s}" + "".join(f"{row['tps']:13.1f}" for row in r["rows"]))
print()
for pr in presets:
    r = results[pr]
    if "error" in r:
        continue
    drafts = "; ".join(f"{row['label']}: {row['draft_ok']}/{row['draft_n']} drafts accepted" for row in r["rows"] if row["draft_n"] and pr.startswith("llama"))
    same = ""
    if r is not first:
        eq = [a["text"] == b["text"] for a, b in zip(first["rows"], r["rows"])]
        same = f"  text identical to {presets[0]} in {sum(eq)} of {len(eq)}"
    print(f"{pr:20s}{same}  {drafts}")
print()
for pr in presets:
    r = results[pr]
    if "error" not in r:
        print(f"{pr:20s} ready after {r['ready_s']:.1f} s, video memory in use at the end {r['vram_mib']} MiB (above what was in use before)")
