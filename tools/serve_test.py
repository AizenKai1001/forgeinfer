#!/usr/bin/env python3
"""HTTP-level tests of `forge serve` (stdlib only; run on the homelab).

  serve_test.py basic [port]     cancellation by hang-up, concurrency, determinism, queue limit
  serve_test.py tools [port]     reasoning_content and tool calls (a Qwen3 model), streaming and not, with a tool-result turn
"""
import http.client
import json
import socket
import sys
import threading
import time

PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 18080


def post(path, body, timeout=120):
    c = http.client.HTTPConnection("127.0.0.1", PORT, timeout=timeout)
    c.request("POST", path, json.dumps(body), {"Content-Type": "application/json"})
    r = c.getresponse()
    data = r.read()
    c.close()
    return r.status, dict(r.getheaders()), data


def get(path):
    c = http.client.HTTPConnection("127.0.0.1", PORT, timeout=30)
    c.request("GET", path)
    r = c.getresponse()
    data = r.read()
    c.close()
    return r.status, json.loads(data)


def chat(messages, **kw):
    kw.setdefault("chat_template_kwargs", {"enable_thinking": False})   # thinking models would spend the small token budgets of these checks on reasoning
    status, _, data = post("/v1/chat/completions", dict(messages=messages, **kw))
    return status, json.loads(data)


def stream(messages, stop_after=None, **kw):
    kw.setdefault("chat_template_kwargs", {"enable_thinking": False})
    """returns (chunks, finished_cleanly); stop_after: hang up after that many chunks"""
    body = json.dumps(dict(messages=messages, stream=True, **kw))
    c = http.client.HTTPConnection("127.0.0.1", PORT, timeout=120)
    c.request("POST", "/v1/chat/completions", body, {"Content-Type": "application/json"})
    sock = c.sock
    r = c.getresponse()
    chunks = []
    done = False
    buf = b""
    while True:
        piece = r.read1(4096) if hasattr(r, "read1") else r.read(1)
        if not piece:
            break
        buf += piece
        while b"\n\n" in buf:
            ev, buf = buf.split(b"\n\n", 1)
            for line in ev.split(b"\n"):
                if line.startswith(b"data: "):
                    payload = line[6:]
                    if payload == b"[DONE]":
                        done = True
                    else:
                        chunks.append(json.loads(payload))
        if stop_after is not None and len(chunks) >= stop_after:
            sock.shutdown(socket.SHUT_RDWR)
            sock.close()
            return chunks, False
    c.close()
    return chunks, done


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (("  " + detail) if detail else ""))
    if not ok:
        check.failed += 1


check.failed = 0


def basic():
    q = [{"role": "user", "content": "Tell me a long story about a dragon."}]
    # hang-up cancels the generation: a request right after must not wait for 1500 tokens
    st0 = get("/props")[1]
    chunks, ok = stream(q, stop_after=4, max_tokens=1500, temperature=0.8)
    t0 = time.time()
    s, d = chat([{"role": "user", "content": "Say hi."}], max_tokens=8, temperature=0)
    dt = time.time() - t0
    check("cancel on hang-up: next request answered promptly", s == 200 and dt < 1.5, f"{dt * 1000:.0f} ms")
    time.sleep(0.3)
    st1 = get("/props")[1]
    produced = st1["completion_tokens"] - st0["completion_tokens"] - d["usage"]["completion_tokens"]
    check("cancelled generation stopped early", produced < 200, f"{produced} tokens produced for the hung-up request")

    # determinism
    # (the first request prefills the prompt with the batched tensor-core path, later ones evaluate only the last token with the decode path: near-ties at the very first token can differ
    # between those two, so compare two requests that are served the same way)
    chat(q, max_tokens=30, temperature=0)
    a = chat(q, max_tokens=30, temperature=0)[1]["choices"][0]["message"]["content"]
    b = chat(q, max_tokens=30, temperature=0)[1]["choices"][0]["message"]["content"]
    check("greedy decoding is deterministic", a == b and len(a) > 0)
    a = chat(q, max_tokens=30, temperature=0.9, seed=7)[1]["choices"][0]["message"]["content"]
    b = chat(q, max_tokens=30, temperature=0.9, seed=7)[1]["choices"][0]["message"]["content"]
    check("same seed, same sample", a == b)

    # streaming equals non-streaming
    full = chat(q, max_tokens=30, temperature=0)[1]["choices"][0]["message"]["content"]
    chunks, done = stream(q, max_tokens=30, temperature=0)
    text = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks if c["choices"])
    check("stream text equals non-stream text", text == full and done, repr(text[:60]))
    check("first chunk carries the role", chunks[0]["choices"][0]["delta"].get("role") == "assistant")

    # concurrency: several requests at once all complete correctly (queued behind one engine)
    results = [None] * 4

    def worker(i):
        results[i] = chat([{"role": "user", "content": f"Reply with the single word number{i}."}], max_tokens=10, temperature=0)

    th = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    t0 = time.time()
    [t.start() for t in th]
    [t.join() for t in th]
    check("4 concurrent requests all answered", all(r and r[0] == 200 and r[1]["choices"][0]["message"]["content"] for r in results), f"{(time.time() - t0) * 1000:.0f} ms")

    # sampling parameters accepted
    s, d = chat(q, max_tokens=12, temperature=0.7, top_p=0.9, top_k=50, min_p=0.05, repetition_penalty=1.1, stop=["\n"])
    check("sampling parameters accepted", s == 200)

    # request validation
    s, _, data = post("/v1/chat/completions", {"messages": []})
    check("empty messages rejected", s == 400, data.decode()[:90])
    s, _, data = post("/v1/chat/completions", {"messages": [{"role": "user", "content": "x"}], "n": 2})
    check("n=2 rejected", s == 400)
    s, _, data = post("/v1/chat/completions", {"messages": [{"role": "robot", "content": "x"}]})
    check("unknown role rejected", s == 400)

    # unicode survives
    s, d = chat([{"role": "user", "content": "Repeat exactly: café 日本語 😀"}], max_tokens=24, temperature=0)
    check("unicode request handled", s == 200, repr(d["choices"][0]["message"]["content"]))


def queue_limit():
    """needs a server started with --queue 1: with one running and one waiting, the third is refused with 429"""
    q = [{"role": "user", "content": "Write a very long essay about the sea."}]
    threads = []
    codes = []

    def worker():
        s, h, data = post("/v1/chat/completions", dict(messages=q, max_tokens=400, temperature=0.8))
        codes.append((s, h.get("Retry-After")))

    for _ in range(4):
        t = threading.Thread(target=worker)
        t.start()
        threads.append(t)
        time.sleep(0.15)
    [t.join() for t in threads]
    check("full queue answers 429 with Retry-After", any(c[0] == 429 and c[1] for c in codes), str(codes))


def tools():
    weather = {"type": "function", "function": {"name": "get_weather", "description": "Get the current weather for a city", "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "City name"}}, "required": ["city"]}}}
    q = [{"role": "user", "content": "What's the weather in Paris right now? Use the tool."}]
    t0 = time.time()
    s, d = chat(q, tools=[weather], max_tokens=700, temperature=0.6, top_p=0.95, top_k=20, chat_template_kwargs={"enable_thinking": True})
    m = d["choices"][0]["message"]
    print("   message:", json.dumps(m)[:400])
    print("   finish:", d["choices"][0]["finish_reason"], "usage:", d["usage"], f"{time.time() - t0:.1f} s")
    check("non-stream: reasoning_content present", bool(m.get("reasoning_content")))
    calls = m.get("tool_calls") or []
    check("non-stream: a tool call came back", len(calls) >= 1)
    if calls:
        c0 = calls[0]
        args = json.loads(c0["function"]["arguments"])
        check("non-stream: function name and JSON arguments", c0["function"]["name"] == "get_weather" and "city" in args, str(args))
        check("non-stream: finish_reason tool_calls", d["choices"][0]["finish_reason"] == "tool_calls")
        # tool result turn
        q2 = q + [m | {"content": m.get("content") or ""}, {"role": "tool", "tool_call_id": c0["id"], "content": json.dumps({"temp_c": 17, "sky": "cloudy"})}]
        s, d2 = chat(q2, tools=[weather], max_tokens=500, temperature=0.6, top_p=0.95, top_k=20, chat_template_kwargs={"enable_thinking": True})
        m2 = d2["choices"][0]["message"]
        print("   answer:", json.dumps(m2)[:400])
        check("tool-result turn answered", s == 200 and bool(m2.get("content")) and ("17" in m2["content"] or "cloud" in m2["content"].lower()))
    # streaming
    chunks, done = stream(q, tools=[weather], max_tokens=700, temperature=0.6, top_p=0.95, top_k=20, chat_template_kwargs={"enable_thinking": True})
    reasoning = "".join(c["choices"][0]["delta"].get("reasoning_content", "") for c in chunks if c["choices"])
    starts = [tc for c in chunks if c["choices"] for tc in c["choices"][0]["delta"].get("tool_calls", []) if tc.get("id")]
    frag = "".join(tc["function"].get("arguments", "") for c in chunks if c["choices"] for tc in c["choices"][0]["delta"].get("tool_calls", []))
    fin = [c["choices"][0]["finish_reason"] for c in chunks if c["choices"] and c["choices"][0].get("finish_reason")]
    check("stream: reasoning deltas", len(reasoning) > 0, f"{len(reasoning)} chars")
    check("stream: tool call start with id and name", len(starts) >= 1 and starts[0]["function"]["name"] == "get_weather")
    try:
        ok = "city" in json.loads(frag)
    except Exception:
        ok = False
    check("stream: argument fragments join to JSON", ok, frag)
    check("stream: finish reason and [DONE]", fin == ["tool_calls"] and done, str(fin))
    # no thinking
    s, d = chat([{"role": "user", "content": "Say hello."}], max_tokens=60, temperature=0.6, chat_template_kwargs={"enable_thinking": False})
    m = d["choices"][0]["message"]
    print("   no-think message:", json.dumps(m)[:300])
    check("enable_thinking=false: plain content, no reasoning", bool(m.get("content")) and not m.get("reasoning_content"))


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "basic"
    {"basic": basic, "tools": tools, "queue": queue_limit}[mode]()
    print("FAILED: %d" % check.failed if check.failed else "ALL PASSED")
    sys.exit(1 if check.failed else 0)
