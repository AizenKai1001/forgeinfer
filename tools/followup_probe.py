"""How long does a chat follow-up take after a long conversation? One conversation (no swap with another): a long document and a question, then short follow-up questions. For every turn the new prompt
tokens, the time to read them, the time to answer, and the whole request. At a long context the one-block attention kernel makes both slow (FORGE_ATTN_SPLIT=0 shows it, 1 the split kernels).

    python tools/followup_probe.py MODEL.gguf [paragraphs] [port]

paragraphs: about 64 tokens each (100 -> 6,300 tokens). Environment: FORGE_BIN (the engine program, default target/release/forge), FORGE_ARGS (extra arguments for `forge serve`), PROBE_LOG (a file for the
engine's messages)."""
import json
import os
import subprocess
import sys
import time
import urllib.request

model = sys.argv[1]
paras = int(sys.argv[2]) if len(sys.argv) > 2 else 100
port = int(sys.argv[3]) if len(sys.argv) > 3 else 18971
forge = os.environ.get("FORGE_BIN", "target/release/forge")
filler = "The harbour town of Aldermere keeps careful records of every ship that docks, noting the cargo, the captain and the weather of the day. "
doc = "".join(filler * 2 + f"Entry A{i}. " for i in range(paras))
cmd = [forge, "serve", model, "--port", str(port), "--ctx", str(paras * 66 + 2048), "--no-speculate"] + os.environ.get("FORGE_ARGS", "").split()
log = open(os.environ["PROBE_LOG"], "a") if os.environ.get("PROBE_LOG") else subprocess.DEVNULL
p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=log, env=dict(os.environ))
try:
    for _ in range(240):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=1)
            break
        except Exception:
            time.sleep(0.5)
    history = []
    for i, q in enumerate(["How many entries does the record have?", "Name the town.", "What is noted for every ship?", "What else?", "Thanks."]):
        history.append({"role": "user", "content": (doc + "\n\nQuestion: " + q) if i == 0 else q})
        body = {"model": "x", "messages": history, "max_tokens": 24, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
        req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        t0 = time.time()
        r = json.load(urllib.request.urlopen(req, timeout=600))
        dt = time.time() - t0
        answer = r["choices"][0]["message"]["content"].strip()
        history.append({"role": "assistant", "content": answer})
        u, t = r["usage"], r.get("timings") or {}
        new = u["prompt_tokens"] - u["prompt_tokens_details"]["cached_tokens"]
        print(f"turn {i + 1}: prompt {u['prompt_tokens']:5d} tokens, new {new:5d}, prompt {t.get('prompt_ms', 0):7.1f} ms, generation {t.get('predicted_ms', 0):6.1f} ms for {u['completion_tokens']} tokens, request {dt * 1e3:7.1f} ms | {answer[:48]!r}")
finally:
    p.terminate()
    try:
        p.wait(15)
    except Exception:
        p.kill()
