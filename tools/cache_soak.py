"""Random use of the prompt cache of `forge serve`: several long chats visited in a random order (seeded). Three runs of the engine:
  cold     the cache off, the random order: every return to a chat reads its whole document again,
  in place the cache off, the chats one after the other (all turns of chat A, then of B ...): every turn continues the conversation that is on the GPU, the engine's ordinary reuse of the previous turn,
  cached   the cache on (2,048 MB), the random order.
What it checks, besides the speed:
  - no request fails (no HTTP error, no timeout) and the engine writes no "[cache] failed" line,
  - every turn of the cached run is exactly the answer of the in-place run (greedy), as long as the chat's returns came from the cache: a chat that was evicted is read from scratch, which may differ by a word
    at a near tie (the entries of the previous answer are written by the prompt kernels instead of the decode kernels), so only the turns before the first return that was read again are compared,
  - on Linux, the memory the engine process holds at the end of the cached run is not more than the cache budget above what it held at the end of the cold run.

    python tools/cache_soak.py MODEL.gguf [chats] [paragraphs] [turns] [seed] [port]

chats 3, paragraphs 100 (6,400 tokens per document: four chats do not fit a 2,048 MB cache of a dense 4B, which makes it evict), turns 24, seed 1. Environment: FORGE_BIN, FORGE_ARGS (extra arguments for
`forge serve`), CACHE_CHECK_SPECULATE=1 (speculative decoding stays on), FORGE_CHECK_LOG (a file for the engine's messages). Exit code 1 on a failure."""
import json
import os
import random
import subprocess
import sys
import time
import urllib.request

model = sys.argv[1]
n_chats = int(sys.argv[2]) if len(sys.argv) > 2 else 3
paras = int(sys.argv[3]) if len(sys.argv) > 3 else 100
turns = int(sys.argv[4]) if len(sys.argv) > 4 else 24
seed = int(sys.argv[5]) if len(sys.argv) > 5 else 1
port = int(sys.argv[6]) if len(sys.argv) > 6 else 18971
forge = os.environ.get("FORGE_BIN", "target/release/forge")
budget_mb = 2048
topics = ["lighthouse", "market", "harbour master", "ferry", "bell tower", "salt mill", "customs house", "fish auction", "ropewalk", "boatyard", "chandlery", "tide table"]


def document(k):
    """a document of its own (the entries differ from chat to chat, so no prefix is shared)"""
    return "".join(f"Chat {k} entry {i}. The {topics[(i + k) % len(topics)]} of town {k} was inspected on day {i}; the clerk wrote that the {topics[(i * 5 + k) % len(topics)]} needed {i * 3 + k} repairs, and the council agreed to pay {100 + i * 13 + k} coins by the end of the season. " for i in range(paras))


DOCS = [document(k) for k in range(n_chats)]
QUESTIONS = ["How many repairs did the first entry need?", "Which building was inspected on day 7?", "How many coins were agreed on day 12?", "Name one building that is mentioned.", "What did the clerk write on day 3?", "Thank you."]
rng = random.Random(seed)
order = [rng.randrange(n_chats) for _ in range(turns)]


def rss_mb(pid):
    try:
        for line in open(f"/proc/{pid}/status"):
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    except OSError:
        pass
    return None


def run(cache_mb, visits):
    env = dict(os.environ)
    env["FORGE_PROMPT_CACHE_MB"] = str(cache_mb)
    cmd = [forge, "serve", model, "--port", str(port), "--ctx", str(paras * 66 + 2048)] + ([] if os.environ.get("CACHE_CHECK_SPECULATE") else ["--no-speculate"]) + os.environ.get("FORGE_ARGS", "").split()
    log_path = os.environ.get("FORGE_CHECK_LOG")
    log = open(log_path, "a") if log_path else subprocess.DEVNULL
    p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=log, env=env)
    out, failures, rss = [], [], None
    try:
        for _ in range(240):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=1)
                break
            except Exception:
                time.sleep(0.5)
        history = [[] for _ in range(n_chats)]
        asked = [0] * n_chats
        for step, k in enumerate(visits):
            q = QUESTIONS[asked[k] % len(QUESTIONS)]
            text = (DOCS[k] + "\n\nQuestion: " + q) if asked[k] == 0 else q
            history[k].append({"role": "user", "content": text})
            body = {"model": "x", "messages": history[k], "max_tokens": 16, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
            req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
            t0 = time.time()
            try:
                r = json.load(urllib.request.urlopen(req, timeout=600))
            except Exception as e:
                failures.append(f"step {step} chat {k}: {e}")
                history[k].pop()
                continue
            dt = time.time() - t0
            answer = r["choices"][0]["message"]["content"].strip()
            history[k].append({"role": "assistant", "content": answer})
            asked[k] += 1
            u = r["usage"]
            out.append((step, k, asked[k], u["prompt_tokens"], u["prompt_tokens_details"]["cached_tokens"], dt, answer))
        rss = rss_mb(p.pid)
    finally:
        p.terminate()
        try:
            p.wait(15)
        except Exception:
            p.kill()
    return out, failures, rss


cold, cold_fail, rss_cold = run(0, order)
ref, ref_fail, _ = run(0, [k for k in range(n_chats) for _ in range(order.count(k))])
on, on_fail, rss_on = run(budget_mb, order)
ok = not (cold_fail or ref_fail or on_fail)
by = lambda rows: {(r[1], r[2]): r for r in rows}
refd = by(ref)
print(f"chats {n_chats}, {paras} paragraphs each, {turns} turns in the order {''.join('ABCDEFGH'[k] for k in order)}")
print("step chat turn  prompt | cold: cached  seconds | cache on: cached  seconds | answer against the in-place run")
first_miss = {}
for b in on:
    if b[2] >= 2 and b[4] < 0.5 * b[3] and b[1] not in first_miss:
        first_miss[b[1]] = b[2]
exact = compared = 0
for a, b in zip(cold, on):
    r = refd.get((b[1], b[2]))
    if r is None:
        verdict = "no reference"
    elif b[2] >= first_miss.get(b[1], 10 ** 9):
        verdict = "(read again after a miss: not compared)"
    else:
        compared += 1
        if r[6] == b[6]:
            exact += 1
            verdict = "same"
        else:
            ok = False
            verdict = "DIFFERENT: " + repr(r[6][:40]) + " / " + repr(b[6][:40])
    print(f"{a[0]:4d} {'ABCDEFGH'[a[1]]}  {a[2]:4d} {a[3]:7d} |        {a[4]:6d} {a[5]:7.2f} |          {b[4]:6d} {b[5]:7.2f} | {verdict}")
misses = sum(1 for b in on if b[2] >= 2 and b[4] < 0.5 * b[3])
print(f"answers equal to the in-place run: {exact} of {compared} compared; {misses} returns read less than half of their prompt from the cache (evicted by the budget, or never stored)")
print(f"time {sum(x[5] for x in cold):.1f} s without the cache, {sum(x[5] for x in on):.1f} s with it")
for f in cold_fail + ref_fail + on_fail:
    print("FAILED REQUEST:", f)
if rss_cold is not None and rss_on is not None:
    print(f"process memory at the end: {rss_cold:.0f} MiB without the cache, {rss_on:.0f} MiB with it")
    if rss_on > rss_cold + budget_mb + 512:
        ok = False
        print("the cached run holds more memory than the cache budget explains")
log_path = os.environ.get("FORGE_CHECK_LOG")
if log_path and os.path.exists(log_path):
    bad = [l for l in open(log_path, errors="replace") if "[cache] failed" in l or "could not be kept" in l or "panicked" in l]
    for l in bad[:5]:
        print("ENGINE:", l.strip())
    if bad:
        ok = False
print("SOAK OK" if ok else "SOAK FAILED")
sys.exit(0 if ok else 1)
