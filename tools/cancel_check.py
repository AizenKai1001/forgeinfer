"""End-to-end check that closing the connection during a long prompt cancels the work: a streamed request with `return_progress` and a long prompt is dropped after its second progress chunk, then a short request is sent at
once. The engine handles one request at a time, so if the first one were read to the end the short one would wait for the rest of the prompt; if the cancellation works it waits for the piece in flight only (1,024 tokens).

    python tools/cancel_check.py MODEL.gguf [paragraphs] [port]

paragraphs: about 64 tokens each (150 -> 9,500 tokens). Environment: FORGE_BIN (the engine program, default target/release/forge), FORGE_ARGS (extra arguments for `forge serve`). Exit code 1 when the short request waits
longer than a quarter of the time the whole prompt takes (or one second more than a piece takes)."""
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request

model = sys.argv[1]
paras = int(sys.argv[2]) if len(sys.argv) > 2 else 150
port = int(sys.argv[3]) if len(sys.argv) > 3 else 18931
forge = os.environ.get("FORGE_BIN", "target/release/forge")
filler = "The harbour town of Aldermere keeps careful records of every ship that docks, noting the cargo, the captain and the weather of the day. "
long_text = "".join((filler * 2) + f"Entry {i}. " for i in range(paras))
cmd = [forge, "serve", model, "--port", str(port), "--ctx", str(paras * 66 + 1024), "--no-speculate"] + os.environ.get("FORGE_ARGS", "").split()
p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def post(body, stream=False):
    return urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})


def chat(text, **extra):
    return {"model": "x", "messages": [{"role": "user", "content": text}], "max_tokens": 8, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}, **extra}


try:
    for _ in range(240):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=1)
            break
        except Exception:
            time.sleep(0.5)
    # 1. the whole long prompt once, to know what it costs (and to warm the engine); a different prompt afterwards so that nothing of it is cached
    t0 = time.time()
    urllib.request.urlopen(post(chat(long_text + "\n\nQuestion one?")), timeout=240).read()
    whole = time.time() - t0
    # 2. the same kind of prompt again (different tail, so the cache holds only the shared prefix: use a different first line to start from nothing), dropped after the second progress chunk
    body = chat("Different start. " + long_text + "\n\nQuestion two?", stream=True, return_progress=True)
    s = socket.create_connection(("127.0.0.1", port), timeout=60)
    raw = json.dumps(body).encode()
    s.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nConnection: close\r\nContent-Length: " + str(len(raw)).encode() + b"\r\n\r\n" + raw)
    seen, buf = 0, b""
    while seen < 3:  # the start event and two pieces
        d = s.recv(65536)
        if not d:
            break
        buf += d
        seen = buf.count(b"prompt_progress")
    s.close()
    t1 = time.time()
    # 3. a short request: how long does it wait?
    urllib.request.urlopen(post(chat("Say hi.")), timeout=240).read()
    waited = time.time() - t1
    print(f"the whole long prompt takes {whole:.2f} s; after dropping the connection ({seen} progress chunks seen) a short request took {waited:.2f} s")
    ok = waited < 0.25 * whole + 1.0
    print("CANCEL OK" if ok else "CANCEL TOO SLOW")
    sys.exit(0 if ok else 1)
finally:
    p.terminate()
    try:
        p.wait(15)
    except Exception:
        p.kill()
