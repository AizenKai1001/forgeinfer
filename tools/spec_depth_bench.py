"""Decode speed of `forge serve` after a long prompt, with and without the Qwen3.5 draft head (speculation).

    python3 tools/spec_depth_bench.py MODEL.gguf HEAD.safetensors "8,16,32,64" [port]

The depths are thousands of tokens (about 64 tokens per paragraph of filler). For each mode one server is started with a context for the deepest prompt; the prompts share their beginning, so the prompt cache reads only
what is new. After every prompt the model is asked for a story of about 250 words (greedy, 256 tokens at most) and the server's own `timings` give the decode speed. Environment: FORGE_BIN (default target/release/forge),
FORGE_ARGS (extra `forge serve` arguments, for example "--kv k8v4"), FORGE_SPEC_DEPTH (default 3 for the speculative mode), FORGE_BENCH_TASK (`story`, the default, or `copy`: the model is asked to repeat the first
three entries of the text, which prompt lookup and the head predict well), FORGE_BENCH_NOGOV=1 (a third mode: the speculative one with FORGE_GOVERN=0, without the speed governor), FORGE_SERVER_LOG (a file that
keeps the server's speculation statistics)."""
import json
import os
import subprocess
import sys
import time
import urllib.request

if len(sys.argv) < 4:
    sys.exit(__doc__)
model, head, depths = sys.argv[1], sys.argv[2], [int(x) for x in sys.argv[3].split(",")]
port = int(sys.argv[4]) if len(sys.argv) > 4 else 18921
forge = os.environ.get("FORGE_BIN", "target/release/forge")
filler = "The harbour town of Aldermere keeps careful records of every ship that docks, noting the cargo, the captain and the weather of the day. "
TASK = os.environ.get("FORGE_BENCH_TASK", "story")
ASK = {
    "story": "Write a story of about 250 words about the lighthouse keeper's cat Biscuit and the ships of Aldermere.",
    "copy": "Copy the first three entries of the text above exactly, word for word, and write nothing else.",
}[TASK]


def document(k_tokens):
    paras = int(k_tokens * 1000 / 64)
    return "".join((filler * 2) + f"Entry {i}. " for i in range(paras))


def run(mode):
    ctx = max(depths) * 1000 + 3000
    cmd = [forge, "serve", model, "--port", str(port), "--ctx", str(ctx)] + os.environ.get("FORGE_ARGS", "").split()
    env = dict(os.environ)
    if mode == "plain":
        cmd.append("--no-speculate")
    else:
        cmd += ["--mtp", head]
        env["FORGE_SPEC_DEPTH"] = os.environ.get("FORGE_SPEC_DEPTH", "3")
        if mode == "speculative-nogov":
            env["FORGE_GOVERN"] = "0"
    log = os.environ.get("FORGE_SERVER_LOG")
    if log:
        env["FORGE_SPEC_STATS"] = "1"
        errf = open(log, "a")
        errf.write(f"==== {mode}" + chr(10))
        errf.flush()
    else:
        errf = subprocess.DEVNULL
    p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=errf, env=env)
    rows = []
    try:
        for _ in range(240):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=1)
                break
            except Exception:
                time.sleep(0.5)
        for k in depths:
            msg = [{"role": "user", "content": document(k) + chr(10) * 2 + ASK}]
            body = {"model": "x", "messages": msg, "max_tokens": 256, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
            req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
            t0 = time.time()
            r = json.load(urllib.request.urlopen(req, timeout=900))
            wall = time.time() - t0
            tm = r.get("timings") or {}
            u = r.get("usage") or {}
            rows.append((k, u.get("prompt_tokens"), u.get("completion_tokens"), tm.get("predicted_per_second"), tm.get("prompt_per_second"), wall))
            print(f"{mode:18s} depth {k:3d}K  prompt {u.get('prompt_tokens')}  generated {u.get('completion_tokens')}  decode {tm.get('predicted_per_second')} tok/s  (prompt {tm.get('prompt_per_second')} tok/s, wall {wall:.1f} s)", flush=True)
    finally:
        p.terminate()
        try:
            p.wait(20)
        except Exception:
            p.kill()
    return rows


plain = run("plain")
spec = run("speculative")
nogov = run("speculative-nogov") if os.environ.get("FORGE_BENCH_NOGOV") == "1" else None
print(chr(10) + f"task: {TASK}" + chr(10) + "depth  plain tok/s  speculative tok/s  gain" + ("  without the governor" if nogov else ""))
for i, (a, b) in enumerate(zip(plain, spec)):
    if a[3] and b[3]:
        extra = f"  {nogov[i][3]:.1f} (x{nogov[i][3] / a[3]:.2f})" if nogov and nogov[i][3] else ""
        print(f"{a[0]:4d}K  {a[3]:10.1f}  {b[3]:16.1f}  x{b[3] / a[3]:.2f}{extra}")
