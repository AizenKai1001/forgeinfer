#!/usr/bin/env python3
"""Serving benchmark: streams a long answer and reports engine-side and client-side decode speed, time to first token and prefill speed.
  serve_bench.py [port] [max_tokens]"""
import http.client
import json
import sys
import time

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 18081
MAXT = int(sys.argv[2]) if len(sys.argv) > 2 else 256


def run(messages, **kw):
    body = json.dumps(dict(messages=messages, stream=True, stream_options={"include_usage": True}, max_tokens=MAXT, **kw))
    c = http.client.HTTPConnection("127.0.0.1", PORT, timeout=300)
    t0 = time.time()
    c.request("POST", "/v1/chat/completions", body, {"Content-Type": "application/json"})
    r = c.getresponse()
    first = None
    last = None
    n = 0
    usage = None
    timings = None
    buf = b""
    while True:
        piece = r.read1(65536)
        if not piece:
            break
        buf += piece
        while b"\n\n" in buf:
            ev, buf = buf.split(b"\n\n", 1)
            for line in ev.split(b"\n"):
                if line.startswith(b"data: ") and line[6:] != b"[DONE]":
                    d = json.loads(line[6:])
                    if d.get("usage"):
                        usage, timings = d["usage"], d["timings"]
                    elif d["choices"]:
                        dl = d["choices"][0]["delta"]
                        if dl.get("content") or dl.get("reasoning_content"):
                            now = time.time()
                            if first is None:
                                first = now
                            last = now
                            n += 1
    c.close()
    return dict(ttft=(first - t0) * 1e3 if first else None, client_tps=(n - 1) / (last - first) if n > 1 else 0, usage=usage, timings=timings)


long = [{"role": "user", "content": "Write a detailed essay about how rivers shape landscapes, with many examples."}]
filler = "The quick brown fox jumps over the lazy dog. " * 120
prefill = [{"role": "user", "content": filler + " Summarize the above in one word."}]
import os
QUICK = os.environ.get("BENCH_QUICK") == "1"
CONFIGS = [("greedy", dict(temperature=0)), ("sampled t=0.7 top_k=40 top_p=0.95", dict(temperature=0.7, top_k=40, top_p=0.95)), ("sampled t=0.7 min_p=0.05 rep=1.1", dict(temperature=0.7, top_k=0, min_p=0.05, repetition_penalty=1.1))]
for name, kw in (CONFIGS[:2] if QUICK else CONFIGS):
    best = None
    for _ in range(2 if QUICK else 3):
        r = run(long, **kw)
        if best is None or r["timings"]["predicted_per_second"] > best["timings"]["predicted_per_second"]:
            best = r
    t = best["timings"]
    spec = ""
    if t.get("draft_n"):
        spec = f", drafts {t['draft_n']} accepted {t['draft_n_accepted']} ({100 * t['draft_n_accepted'] / t['draft_n']:.0f} %, {best['usage']['completion_tokens'] / t['draft_n']:.2f} tokens per step)"
    print(f"{name:42s} decode {t['predicted_per_second']:7.1f} tok/s engine, {best['client_tps']:7.1f} tok/s client, first token {best['ttft']:6.1f} ms, {best['usage']['completion_tokens']} tokens{spec}")
if QUICK:
    sys.exit(0)
body = dict(messages=prefill, max_tokens=2, temperature=0)
c = http.client.HTTPConnection("127.0.0.1", PORT, timeout=300)
for i in range(3):
    # change the first character so the prefix cache cannot help
    body["messages"] = [{"role": "user", "content": chr(65 + i) + filler + " Summarize the above in one word."}]
    c.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
    d = json.loads(c.getresponse().read())
    t = d["timings"]
    print(f"prefill {d['usage']['prompt_tokens']} tokens: {t['prompt_ms']:.0f} ms = {t['prompt_per_second']:.0f} tok/s")
    c.close()
    c = http.client.HTTPConnection("127.0.0.1", PORT, timeout=300)
