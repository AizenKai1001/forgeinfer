"""How many of several facts planted in a long prompt does the model still find? A multi-needle version of tools/needle_check.py: N facts of the form "The access code of research notes 07 is 481-922." are spread over a long text of filler
paragraphs and one question asks for all of them ("research notes 03: ...", one per line); the answer is scored code by code. Run it with different engine settings (FORGE_FAKEQ, FORGE_MMQ, ...) in the environment to see what a numerics change
does to retrieval at length: the perplexity of a text can look fine while a few confident predictions that depend on one earlier line are lost.

    python tools/retrieval_check.py MODEL.gguf [paragraphs] [facts] [seeds] [port]

paragraphs: about 64 tokens each (250 -> 16,000 tokens); facts: how many codes (default 16); seeds: how many different prompts (default 3). With the last argument `lines` the text is not filler with a few facts but a
list of thousands of similar lines ("The access code of research notes 4821 is 481-922."; paragraphs x 3 lines, 250 -> about 15,000 tokens) and the questions pick lines of it: a much harder test, since the right line has to be told from
hundreds of near-copies. Environment: FORGE_BIN (the engine program, default target/release/forge), FORGE_ARGS (extra
arguments for `forge serve`), FORGE_TIMEOUT (seconds per request, default 600). Prints the share of codes found; exit code 0."""
import json
import os
import random
import re
import subprocess
import sys
import time
import urllib.request

if len(sys.argv) < 2:
    sys.exit(__doc__)
model = sys.argv[1]
paras = int(sys.argv[2]) if len(sys.argv) > 2 else 250
facts = int(sys.argv[3]) if len(sys.argv) > 3 else 16
seeds = int(sys.argv[4]) if len(sys.argv) > 4 else 3
port = int(sys.argv[5]) if len(sys.argv) > 5 else 18941
lines_mode = len(sys.argv) > 6 and sys.argv[6] == "lines"
forge = os.environ.get("FORGE_BIN", "target/release/forge")
timeout = int(os.environ.get("FORGE_TIMEOUT", "600"))
filler = "The harbour town of Aldermere keeps careful records of every ship that docks, noting the cargo, the captain and the weather of the day. "
cmd = [forge, "serve", model, "--port", str(port), "--ctx", str(paras * 66 + 2048), "--no-speculate"] + os.environ.get("FORGE_ARGS", "").split()
p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def build_lines(seed):
    rnd = random.Random(seed)
    n = paras * 3
    ids = rnd.sample(range(1000, 10000), n)
    codes = {v: f"{rnd.randint(100, 999)}-{rnd.randint(100, 999)}" for v in ids}
    text = "".join(f"The access code of research notes {v} is {codes[v]}.\n" for v in ids)
    ask_ids = sorted(rnd.sample(ids, facts))
    ask = ", ".join(str(v) for v in ask_ids)
    q = f"\nQuestion: give the access codes of the vaults {ask}. Answer with one line per research notes in the form 'research notes NNNN: CODE' and nothing else."
    return text + q, {v: codes[v] for v in ask_ids}


def build(seed):
    if lines_mode:
        return build_lines(seed)
    rnd = random.Random(seed)
    vaults = sorted(rnd.sample(range(1, 100), facts))
    codes = {v: f"{rnd.randint(100, 999)}-{rnd.randint(100, 999)}" for v in vaults}
    slots = sorted(rnd.sample(range(5, paras - 5), facts))
    placed = vaults[:]
    rnd.shuffle(placed)  # the order in the text is not the order of the question
    at = dict(zip(slots, placed))
    parts = []
    for i in range(paras):
        parts.append((filler * 2) + f"Entry {i}. ")
        if i in at:
            parts.append(f"The access code of research notes {at[i]:02d} is {codes[at[i]]}. ")
    ask = ", ".join(f"{v:02d}" for v in vaults)
    q = f"\n\nQuestion: give the access codes of the vaults {ask}. Answer with one line per research notes in the form 'research notes NN: CODE' and nothing else."
    return "".join(parts) + q, codes


try:
    for _ in range(240):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=1)
            break
        except Exception:
            time.sleep(0.5)
    found = total = 0
    for seed in range(1, seeds + 1):
        prompt, codes = build(seed)
        body = {"model": "x", "messages": [{"role": "user", "content": prompt}], "max_tokens": 16 * facts + 40, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
        req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        t0 = time.time()
        r = json.load(urllib.request.urlopen(req, timeout=timeout))
        answer = r["choices"][0]["message"]["content"]
        got = {int(m.group(1)): m.group(2) for m in re.finditer(r"research notes\s+0*(\d+)\s*:\s*(\d{3}-\d{3})", answer)}
        ok = sum(1 for v, c in codes.items() if got.get(v) == c)
        wrong = [f"{v}: wanted {c}, got {got.get(v)}" for v, c in codes.items() if got.get(v) != c]
        found += ok
        total += len(codes)
        print(f"seed {seed}: {ok} of {len(codes)} codes found ({r['usage']['prompt_tokens']} prompt tokens, {time.time() - t0:.0f} s)" + ("; wrong: " + "; ".join(wrong[:6]) if wrong else ""), flush=True)
    print(f"RETRIEVAL {found} of {total} codes found = {100.0 * found / total:.1f} percent")
finally:
    p.terminate()
    try:
        p.wait(15)
    except Exception:
        p.kill()
