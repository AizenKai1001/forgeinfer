"""End-to-end check of the prompt cache of `forge serve`: two long documents are discussed in two conversations that take turns (A, B, A, B, A, B), once with the cache off (FORGE_PROMPT_CACHE_MB=0) and once with it on.
Without the cache every return to a conversation reads its whole document again; with it the conversation comes back from host memory. The answers must be the same in both runs (greedy), the cached run must report most of
the prompt as cached on the returns, and it should be faster. Works for dense and hybrid models.

    python tools/cache_check.py MODEL.gguf [paragraphs] [port]

paragraphs: about 64 tokens each (100 -> 6,500 tokens per document). Environment: FORGE_BIN (the engine program, default target/release/forge), FORGE_ARGS (extra arguments for `forge serve`), CACHE_CHECK_SPECULATE=1 (leave speculative decoding on), CACHE_CHECK_BOTH_OFF=1 (run twice with the cache off: shows how much two runs of the same engine differ by themselves). Exit code 1 when the answers
differ or a return was not served from the cache."""
import json
import os
import subprocess
import sys
import time
import urllib.request

model = sys.argv[1]
paras = int(sys.argv[2]) if len(sys.argv) > 2 else 100
port = int(sys.argv[3]) if len(sys.argv) > 3 else 18961
forge = os.environ.get("FORGE_BIN", "target/release/forge")
filler = "The harbour town of Aldermere keeps careful records of every ship that docks, noting the cargo, the captain and the weather of the day. "


def document(tag, facts):
    """filler paragraphs with a few facts of this document planted in them"""
    parts = []
    for i in range(paras):
        parts.append(filler * 2 + f"Entry {tag}{i}. ")
        if i in facts:
            parts.append(f"Fact {tag}{facts[i][0]}: {facts[i][1]}. ")
    return "".join(parts)


DOCS = {
    "A": document("A", {10: ("one", "the lighthouse cat is called Biscuit"), 40: ("two", "the harbour master drinks only green tea"), 80: ("three", "the bell of the north pier was cast in 1887")}),
    "B": document("B", {12: ("one", "the baker of the quay sells blue loaves"), 45: ("two", "the ferry to the island runs on Tuesdays"), 85: ("three", "the old mill grinds salt, not flour")}),
}
QUESTIONS = {
    "A": ["What is the lighthouse cat called?", "What does the harbour master drink?", "When was the bell of the north pier cast?"],
    "B": ["What colour are the loaves of the baker of the quay?", "On which day does the ferry to the island run?", "What does the old mill grind?"],
}


def run(cache_mb):
    env = dict(os.environ)
    env["FORGE_PROMPT_CACHE_MB"] = str(cache_mb)
    # (speculation is off unless CACHE_CHECK_SPECULATE is set: with a model's prediction head the cache holds the head's own KV entries too)
    cmd = [forge, "serve", model, "--port", str(port), "--ctx", str(paras * 66 + 2048)] + ([] if os.environ.get("CACHE_CHECK_SPECULATE") else ["--no-speculate"]) + os.environ.get("FORGE_ARGS", "").split()
    log = open(os.environ["FORGE_CHECK_LOG"], "a") if os.environ.get("FORGE_CHECK_LOG") else subprocess.DEVNULL
    p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=log, env=env)
    out = []
    try:
        for _ in range(240):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=1)
                break
            except Exception:
                time.sleep(0.5)
        history = {"A": [], "B": []}
        asked = {"A": 0, "B": 0}
        for who in "ABABAB":
            q = QUESTIONS[who][asked[who]]
            text = (DOCS[who] + "\n\nQuestion: " + q) if asked[who] == 0 else q
            history[who].append({"role": "user", "content": text})
            body = {"model": "x", "messages": history[who], "max_tokens": 24, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
            req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
            t0 = time.time()
            r = json.load(urllib.request.urlopen(req, timeout=600))
            dt = time.time() - t0
            answer = r["choices"][0]["message"]["content"].strip()
            history[who].append({"role": "assistant", "content": answer})
            asked[who] += 1
            u = r["usage"]
            t = r.get("timings") or {}
            # where the time went: reading the new part of the prompt, generating, and the rest (the queue, and with the cache on the copies of the conversations)
            split = (t.get("prompt_ms", 0.0) / 1e3, t.get("predicted_ms", 0.0) / 1e3)
            out.append((who, asked[who], u["prompt_tokens"], u["prompt_tokens_details"]["cached_tokens"], dt, answer, split))
    finally:
        p.terminate()
        try:
            p.wait(15)
        except Exception:
            p.kill()
    return out


off = run(0)
# (CACHE_CHECK_BOTH_OFF=1 runs the cache off twice: what differs between two runs of the same engine, with speculative decoding and its timing-based guards, is not the cache's doing)
on = run(0 if os.environ.get("CACHE_CHECK_BOTH_OFF") else 2048)
print("turn      prompt tokens | cache off: cached, seconds | cache on: cached, seconds | answers equal")
ok = True
for a, b in zip(off, on):
    same = a[5] == b[5]
    ok &= same
    print(f"{a[0]} turn {a[1]}   {a[2]:6d}          | {a[3]:6d} {a[4]:6.2f} s            | {b[3]:6d} {b[4]:6.2f} s           | {'yes' if same else 'NO: ' + repr(a[5]) + ' / ' + repr(b[5])}")
    print(f"           (prompt {a[6][0]:.2f} s + generation {a[6][1]:.2f} s + other {a[4] - a[6][0] - a[6][1]:.2f} s)   |   (prompt {b[6][0]:.2f} s + generation {b[6][1]:.2f} s + other {b[4] - b[6][0] - b[6][1]:.2f} s)")
# the returns (turn 2 and 3 of a conversation) must come back from the cache
for a, b in zip(off, on):
    if a[1] >= 2 and b[3] < 0.8 * b[2]:
        ok = False
        print(f"{b[0]} turn {b[1]}: only {b[3]} of {b[2]} prompt tokens cached with the cache on")
t_off = sum(x[4] for x in off[2:])
t_on = sum(x[4] for x in on[2:])
print(f"time of the four returns: {t_off:.1f} s without the cache, {t_on:.1f} s with it")
print("CACHE OK" if ok else "CACHE CHECK FAILED")
sys.exit(0 if ok else 1)
