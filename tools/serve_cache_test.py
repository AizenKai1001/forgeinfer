#!/usr/bin/env python3
"""A multi-turn conversation against `forge serve`: the cached prompt tokens per turn show whether the prefix (a KV-cache prefix for ordinary models, a state snapshot for hybrid ones) is reused.
  serve_cache_test.py [port]"""
import http.client
import json
import sys

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 18082


def chat(messages, **kw):
    c = http.client.HTTPConnection("127.0.0.1", PORT, timeout=300)
    c.request("POST", "/v1/chat/completions", json.dumps(dict(messages=messages, **kw)), {"Content-Type": "application/json"})
    r = json.loads(c.getresponse().read())
    c.close()
    return r


system = {"role": "system", "content": "You are a concise assistant. Answer in one or two sentences."}
history = [system]
questions = ["What is the capital of France?", "And what is the capital of Italy?", "Name one famous landmark in each of those two cities.", "Which of the two cities is older?"]
ok = True
for i, q in enumerate(questions):
    history.append({"role": "user", "content": q})
    r = chat(history, max_tokens=60, temperature=0, chat_template_kwargs={"enable_thinking": False})
    m = r["choices"][0]["message"]
    u = r["usage"]
    print(f"turn {i + 1}: prompt {u['prompt_tokens']} tokens, cached {u['prompt_tokens_details']['cached_tokens']}, new {r['timings']['prompt_n']}, prefill {r['timings']['prompt_ms']:.0f} ms -> {m['content'][:80]!r}")
    if i > 0 and u["prompt_tokens_details"]["cached_tokens"] < 10:
        ok = False
    history.append({"role": "assistant", "content": m["content"]})
# the same conversation again from scratch with a different first message must not reuse anything wrongly
r = chat([{"role": "system", "content": "You are a pirate."}, {"role": "user", "content": "What is the capital of France?"}], max_tokens=30, temperature=0, chat_template_kwargs={"enable_thinking": False})
print("different conversation: cached", r["usage"]["prompt_tokens_details"]["cached_tokens"], "->", r["choices"][0]["message"]["content"][:60].replace("\n", " "))
# and then back to the first one: the answer for turn 2 must be unchanged by all this
hist2 = [system, {"role": "user", "content": questions[0]}]
a = chat(hist2, max_tokens=60, temperature=0, chat_template_kwargs={"enable_thinking": False})["choices"][0]["message"]["content"]
hist2 += [{"role": "assistant", "content": a}, {"role": "user", "content": questions[1]}]
b = chat(hist2, max_tokens=60, temperature=0, chat_template_kwargs={"enable_thinking": False})["choices"][0]["message"]["content"]
print("replay turn 2:", repr(b[:80]))
print("PASS" if ok else "FAIL: later turns did not reuse the conversation prefix")
sys.exit(0 if ok else 1)
