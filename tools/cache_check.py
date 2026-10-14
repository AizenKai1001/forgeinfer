"""End-to-end check of the prompt cache of `forge serve`: two long documents are discussed in two conversations that take turns (A, B, A, B, A, B). Three runs of the engine:
  cold    the cache off, the chats take turns: every return to a chat reads its whole document again (what the engine did before the cache),
  in place the cache off, the chats one after the other (A, A, A, B, B, B): every turn continues the conversation that is on the GPU (the engine's ordinary reuse of the previous turn),
  cached  the cache on, the chats take turns: a return brings the conversation back from host memory.
The answers of `cached` must be exactly those of `in place` (greedy): a conversation that comes back from the cache is what the GPU held. The returns must report most of their prompt as cached, and should be
faster than `cold`. The answers of `cold` are printed too; they may differ by a word at a near tie, because a read from scratch writes the entries of the previous answer with the prompt kernels where the
decode kernels wrote them. Works for dense and hybrid models.

    python tools/cache_check.py MODEL.gguf [paragraphs] [port]

paragraphs: about 64 tokens each (100 -> 6,500 tokens per document). Environment: FORGE_BIN (the engine program, default target/release/forge), FORGE_ARGS (extra arguments for `forge serve`),
FORGE_CHECK_LOG (a file for the engine's messages), CACHE_CHECK_SPECULATE=1 (leave speculative decoding on). Exit code 1 when the answers differ from `in place` or a return was not served from the cache."""
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


def run(cache_mb, order="ABABAB"):
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
        for who in order:
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


cold = {(r[0], r[1]): r for r in run(0, "ABABAB")}
inplace = {(r[0], r[1]): r for r in run(0, "AAABBB")}
cached = [r for r in run(2048, "ABABAB")]
print("turn      prompt tokens | cold: cached, seconds | cache on: cached, seconds | answers equal to in place / to cold")
ok = True
for r in cached:
    key = (r[0], r[1])
    c, ip = cold[key], inplace[key]
    same_ip = r[5] == ip[5]
    same_cold = r[5] == c[5]
    ok &= same_ip
    print(f"{r[0]} turn {r[1]}   {r[2]:6d}          | {c[3]:6d} {c[4]:6.2f} s       | {r[3]:6d} {r[4]:6.2f} s       | {'yes' if same_ip else 'NO: ' + repr(ip[5]) + ' / ' + repr(r[5])} / {'yes' if same_cold else 'no'}")
    print(f"           (prompt {r[6][0]:.2f} s + generation {r[6][1]:.2f} s + other {r[4] - r[6][0] - r[6][1]:.2f} s)")
    # the returns (turn 2 and 3 of a chat) must come back from the cache
    if r[1] >= 2 and r[3] < 0.8 * r[2]:
        ok = False
        print(f"{r[0]} turn {r[1]}: only {r[3]} of {r[2]} prompt tokens cached with the cache on")
t_cold = sum(c[4] for k, c in cold.items() if k[1] >= 2)
t_on = sum(r[4] for r in cached if r[1] >= 2)
print(f"time of the four returns: {t_cold:.1f} s read from scratch, {t_on:.1f} s from the cache")
print("CACHE OK" if ok else "CACHE CHECK FAILED")
sys.exit(0 if ok else 1)
