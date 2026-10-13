#!/usr/bin/env python3
"""End-to-end check of `forge serve` on a training checkpoint: a multi-turn conversation with greedy decoding, every reply STREAMED through /v1/chat/completions (SSE).

    kv_serve_check.py run PORT OUT.json [MAX_TOKENS]     talk to the server on PORT, write the replies, cached-token counts and timings to OUT.json
    kv_serve_check.py compare A.json B.json               the replies of two runs must be identical (e.g. the KV-cache server against one started with FORGE_NO_KV=1)

Turn 2 onwards shares the earlier turns with the cached tokens of the previous request, so the KV-cache server reports cached prompt tokens (prefix reuse) while the old path reports none.
"""
import http.client
import json
import sys
import time


def stream_chat(port, messages, max_tokens):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=600)
    body = dict(messages=messages, max_tokens=max_tokens, temperature=0, stream=True, stream_options={"include_usage": True})
    t0 = time.time()
    c.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
    r = c.getresponse()
    text, usage, timings, chunks, first = "", None, None, 0, None
    while True:
        line = r.fp.readline()
        if not line:
            break
        line = line.strip()
        if not line.startswith(b"data: "):
            continue
        data = line[6:]
        if data == b"[DONE]":
            break
        ev = json.loads(data)
        if ev.get("usage"):
            usage, timings = ev["usage"], ev.get("timings")
        for ch in ev.get("choices", []):
            piece = ch.get("delta", {}).get("content")
            if piece:
                if first is None:
                    first = time.time() - t0
                text += piece
                chunks += 1
    c.close()
    return dict(text=text, usage=usage, timings=timings, chunks=chunks, first_token_s=first, total_s=time.time() - t0)


def run(port, out, max_tokens):
    system = {"role": "system", "content": "You are a helpful assistant."}
    questions = ["What is the capital of France?", "Tell me a short story about a robot who learns to paint.", "Now summarize that story in one sentence.", "Write a Python function that reverses a string."]
    history = [system]
    turns = []
    for q in questions:
        history.append({"role": "user", "content": q})
        r = stream_chat(port, history, max_tokens)
        u = r["usage"] or {}
        cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens")
        tm = r["timings"] or {}
        print(f"prompt {u.get('prompt_tokens')} tokens, cached {cached}, {r['chunks']} streamed chunks, {u.get('completion_tokens')} tokens, prefill {tm.get('prompt_ms', 0):.1f} ms, decode {tm.get('predicted_per_second', 0):.1f} tokens/s -> {r['text'][:70]!r}", flush=True)
        turns.append(r)
        history.append({"role": "assistant", "content": r["text"]})
    json.dump(turns, open(out, "w"), indent=1)


def compare(a, b):
    ta, tb = json.load(open(a)), json.load(open(b))
    ok = len(ta) == len(tb)
    for i, (x, y) in enumerate(zip(ta, tb)):
        same = x["text"] == y["text"]
        ok &= same
        print(f"turn {i + 1}: {'IDENTICAL' if same else 'DIFFERENT'} ({len(x['text'])} / {len(y['text'])} chars); cached tokens {x['usage']['prompt_tokens_details']['cached_tokens']} vs {y['usage']['prompt_tokens_details']['cached_tokens']}; decode {x['timings']['predicted_per_second']:.1f} vs {y['timings']['predicted_per_second']:.1f} tokens/s")
    print("SERVE CHECK PASSED" if ok else "SERVE CHECK FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(int(sys.argv[2]), sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 64)
    else:
        sys.exit(compare(sys.argv[2], sys.argv[3]))
