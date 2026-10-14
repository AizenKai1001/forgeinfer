"""End-to-end check of `forge serve` on a long prompt: one sentence ("the lighthouse keeper's cat is called Biscuit and eats smoked herring") is planted in the middle of N paragraphs of filler and the
model is asked about it. The check passes when the answer repeats the fact. It exercises the whole prompt path at that length (chunked prefill, the attention and Gated DeltaNet kernels, the KV cache, the
decode attention at that depth) and prints the server's own timings.

    python tools/needle_check.py MODEL.gguf [paragraphs] [port]

paragraphs: about 64 tokens each (120 -> 7,600 tokens, 250 -> 16,000, 500 -> 32,000, 1000 -> 64,000, 2000 -> 129,000). Environment: FORGE_BIN (the engine program, default target/release/forge),
FORGE_ARGS (extra arguments for `forge serve`, for example "--kv int8" for 100,000 tokens and more on an 8 GB card, "--kv int4" for 200,000 and more), FORGE_CTX (the --ctx of the server: a number or `auto`; default the size of the prompt), FORGE_TIMEOUT (seconds to wait for the answer, default 240).
Exit code 1 when the planted fact is not in the answer."""
import json
import os
import subprocess
import sys
import time
import urllib.request

if len(sys.argv) < 2:
    sys.exit(__doc__)
model = sys.argv[1]
paras = int(sys.argv[2]) if len(sys.argv) > 2 else 120
port = int(sys.argv[3]) if len(sys.argv) > 3 else 18911
forge = os.environ.get("FORGE_BIN", "target/release/forge")
timeout = int(os.environ.get("FORGE_TIMEOUT", "240"))
filler = "The harbour town of Aldermere keeps careful records of every ship that docks, noting the cargo, the captain and the weather of the day. "
fact = "Remember this: the lighthouse keeper's cat is called Biscuit and she only eats smoked herring."
parts = [(filler * 2) + f"Entry {i}. " for i in range(paras)]
parts.insert(len(parts) // 2, fact + " ")
text = "".join(parts)
msg = [{"role": "user", "content": text + "\n\nQuestion: what is the lighthouse keeper's cat called, and what does she eat? Answer in one sentence."}]

cmd = [forge, "serve", model, "--port", str(port), "--ctx", os.environ.get("FORGE_CTX", str(paras * 66 + 1024)), "--no-speculate"] + os.environ.get("FORGE_ARGS", "").split()
p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
try:
    for _ in range(240):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=1)
            break
        except Exception:
            time.sleep(0.5)
    body = {"model": "x", "messages": msg, "max_tokens": 80, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time()
    r = json.load(urllib.request.urlopen(req, timeout=timeout))
    answer = r["choices"][0]["message"]["content"]
    print("answer:", answer.strip())
    print("usage:", r.get("usage"), "\ntimings:", r.get("timings"), f"\nwall {time.time() - t0:.1f} s")
    ok = "biscuit" in answer.lower() and "herring" in answer.lower()
    print("PLANTED FACT FOUND" if ok else "PLANTED FACT NOT FOUND")
    sys.exit(0 if ok else 1)
finally:
    p.terminate()
    try:
        p.wait(15)
    except Exception:
        p.kill()
