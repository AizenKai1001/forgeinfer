#!/usr/bin/env python3
"""Chat-path timing of one build of this repository: the prompt time of fresh prompts of growing length, and the first-token time of the turns of a short conversation (the history is reused, so
a turn prefills only its own new tokens). Starts its own server on a spare port and stops it again; never touches another server.

usage (on the homelab):  mrow_chat_probe.py LABEL BIN MODEL TOKENIZER CHAT [serve flags ...]
  e.g.  FORGE_MROW_MAX=0 mrow_chat_probe.py old target/release/forge Qwen3.5-4B-Q4_K_M.gguf tokenizer.json qwen35 --mtp mtp.safetensors
The FORGE_* variables of the environment pass through to the server (FORGE_MROW_MAX=0 is the old path); PROBE_SHOW=1 prints the start of every answer."""
import http.client
import json
import os
import socket
import subprocess
import sys
import time

PORT = 18177
label, binp, model, tok, chat = sys.argv[1:6]
extra = sys.argv[6:]


def port_free():
    s = socket.socket()
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # connections still in TIME_WAIT from an earlier run do not count
    try:
        s.bind(("127.0.0.1", PORT))
        return True
    except OSError:
        return False
    finally:
        s.close()


def post(body, timeout=300):
    c = http.client.HTTPConnection("127.0.0.1", PORT, timeout=timeout)
    c.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
    r = c.getresponse()
    data = r.read()
    c.close()
    return json.loads(data)


def stream(messages, max_tokens=24):
    body = {"model": "q4", "messages": messages, "stream": True, "stream_options": {"include_usage": True}, "temperature": 0.0, "max_tokens": max_tokens}
    c = http.client.HTTPConnection("127.0.0.1", PORT, timeout=300)
    t0 = time.perf_counter()
    c.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
    r = c.getresponse()
    content, t_first, timings, buf = "", None, None, b""
    while True:
        chunk = r.read1(4096)
        if not chunk:
            break
        buf += chunk
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            line = line.strip()
            if not line.startswith(b"data:") or line.endswith(b"[DONE]"):
                continue
            j = json.loads(line[5:])
            if j.get("timings"):
                timings = j["timings"]
            ch = (j.get("choices") or [None])[0]
            if not ch:
                continue
            d = ch.get("delta") or {}
            if (d.get("content") or d.get("reasoning_content")) and t_first is None:
                t_first = time.perf_counter() - t0
            content += d.get("content") or ""
    c.close()
    return content, t_first, timings or {}


if not port_free():
    sys.exit("port %d is busy" % PORT)
log = open(os.path.expanduser("~/forgeinfer/agents/mrow/chat_probe_%s.log" % label), "wb")
proc = subprocess.Popen(
    [binp, "serve", model, "--native", "--tokenizer", tok, "--chat", chat, "--ctx", "8192", "--alias", "q4", "--port", str(PORT)] + extra,
    stdout=log,
    stderr=log,
    stdin=subprocess.DEVNULL,
)
try:
    for _ in range(180):
        if proc.poll() is not None:
            sys.exit("the server exited early, see chat_probe_%s.log" % label)
        if not port_free():
            break
        time.sleep(1)
    else:
        sys.exit("the server did not come up")
    time.sleep(1)
    post({"model": "q4", "messages": [{"role": "user", "content": "warm up"}], "max_tokens": 2, "temperature": 0})  # graphs, autotuner
    rows = []
    for nw in (0, 1, 2, 4, 8, 16, 32):
        best = None
        for k in range(3):
            text = "Please read number %d carefully. " % (nw * 10 + k) + " ".join(["The quick brown fox jumps over the lazy dog."] * nw)
            d = post({"model": "q4", "messages": [{"role": "user", "content": text}], "max_tokens": 1, "temperature": 0})
            t = d["timings"]
            if best is None or t["prompt_ms"] < best["prompt_ms"]:
                best = t
        rows.append("%d tok %.0f ms" % (best["prompt_n"], best["prompt_ms"]))
    print(label, "fresh prompts (prompt time):", "; ".join(rows))
    # a conversation: the history is reused, a turn prefills only the new tokens
    hist = [{"role": "system", "content": "You are a helpful assistant."}]
    for user in ["What is the capital of France? Answer in one short sentence.", "And what is its population, roughly?", "Thanks. Now write a haiku about the sea.", "Another one, about the mountains please.", "Last one, about the desert."]:
        hist.append({"role": "user", "content": user})
        content, t_first, t = stream(hist)
        print(
            "  turn: first token %.0f ms, prompt %d new + %d reused (%.0f ms), %.0f tok/s%s"
            % ((t_first or 0) * 1000, t.get("prompt_n", 0), t.get("cache_n", 0), t.get("prompt_ms", 0), t.get("predicted_per_second", 0), "" if not t.get("draft_n") else ", draft head right %.0f %%" % (100 * t["draft_n_accepted"] / t["draft_n"]))
        )
        if os.environ.get("PROBE_SHOW"):
            print("    ->", repr(content[:100]))
        hist.append({"role": "assistant", "content": content})
finally:
    proc.terminate()
    try:
        proc.wait(30)
    except subprocess.TimeoutExpired:
        proc.kill()
