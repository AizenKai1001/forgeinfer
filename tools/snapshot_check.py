"""A hybrid model keeps its snapshot when the conversation on the GPU is mostly a long answer: a short prompt, an answer much longer than the prompt (the conversation is over 1,024 tokens), then a follow-up. The
engine resumes the follow-up from the snapshot at the end of the first prompt, so the second request must report the first prompt as cached (the prompt cache once moved that snapshot away: the follow-up
read everything again). Needs a hybrid model (Qwen3.5); a dense one resumes from any shared prefix.
    python tools/snapshot_check.py MODEL.gguf [port]   (FORGE_BIN, FORGE_ARGS, PROBE_LOG)
Exit code 1 when the second request did not reuse the first prompt."""
import json
import os
import subprocess
import sys
import time
import urllib.request

model = sys.argv[1]
port = int(sys.argv[2]) if len(sys.argv) > 2 else 18991
forge = os.environ.get("FORGE_BIN", "target/release/forge")
filler = "The harbour town of Aldermere keeps careful records of every ship that docks, noting the cargo, the captain and the weather of the day. "
doc = "".join(filler + f"Entry {i}. " for i in range(8))
cmd = [forge, "serve", model, "--port", str(port), "--ctx", "4096", "--no-speculate"] + os.environ.get("FORGE_ARGS", "").split()
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
    first_prompt = 0
    ok = True
    for i, q in enumerate([doc + "\n\nCount from 1 to 400, separated by commas, and write nothing else.", "Now say the last number you wrote, and nothing else.", "Thanks."]):
        history.append({"role": "user", "content": q})
        body = {"model": "x", "messages": history, "max_tokens": 1600, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
        req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        t0 = time.time()
        r = json.load(urllib.request.urlopen(req, timeout=600))
        answer = r["choices"][0]["message"]["content"].strip()
        history.append({"role": "assistant", "content": answer})
        u, t = r["usage"], r.get("timings") or {}
        cached = u["prompt_tokens_details"]["cached_tokens"]
        if i == 0:
            first_prompt = u["prompt_tokens"]
        if i == 1 and cached < first_prompt - 16:
            ok = False
        print(f"turn {i + 1}: prompt {u['prompt_tokens']:5d} tokens, cached {cached:5d}, generated {u['completion_tokens']:5d}, prompt {t.get('prompt_ms', 0):7.1f} ms, request {(time.time() - t0) * 1e3:7.1f} ms | {answer[-40:]!r}")
finally:
    p.terminate()
    try:
        p.wait(15)
    except Exception:
        p.kill()
print("SNAPSHOT KEPT" if ok else "SNAPSHOT LOST: the follow-up read the first prompt again")
sys.exit(0 if ok else 1)
