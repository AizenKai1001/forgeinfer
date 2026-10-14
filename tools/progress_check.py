"""End-to-end check of `return_progress` against a real engine: a long prompt is streamed with the field set, the progress chunks and the first token are timed, then the same prompt again (all in the cache: nothing to
report) and once without the field (no progress chunks).

    python tools/progress_check.py MODEL.gguf [paragraphs] [port]

paragraphs: about 64 tokens each (150 -> 9,500 tokens). Environment: FORGE_BIN (the engine program, default target/release/forge), FORGE_ARGS (extra arguments for `forge serve`). Exit code 1 when a check fails."""
import json
import os
import subprocess
import sys
import time
import urllib.request

model = sys.argv[1]
paras = int(sys.argv[2]) if len(sys.argv) > 2 else 120
port = int(sys.argv[3]) if len(sys.argv) > 3 else 18921
forge = os.environ.get("FORGE_BIN", "target/release/forge")
filler = "The harbour town of Aldermere keeps careful records of every ship that docks, noting the cargo, the captain and the weather of the day. "
text = "".join((filler * 2) + f"Entry {i}. " for i in range(paras))
messages = [{"role": "user", "content": text + "\n\nQuestion: in one short sentence, what does the town of Aldermere keep records of?"}]
cmd = [forge, "serve", model, "--port", str(port), "--ctx", str(paras * 66 + 1024), "--no-speculate"] + os.environ.get("FORGE_ARGS", "").split()
p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def stream(progress):
    body = {"model": "x", "messages": messages, "max_tokens": 24, "temperature": 0, "stream": True, "chat_template_kwargs": {"enable_thinking": False}}
    if progress is not None:
        body["return_progress"] = progress
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time()
    events, first = [], None
    with urllib.request.urlopen(req, timeout=240) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line.endswith("[DONE]"):
                continue
            j = json.loads(line[5:])
            now = time.time() - t0
            if "prompt_progress" in j:
                pp = j["prompt_progress"]
                events.append((round(now, 2), pp["processed"], pp["total"], pp["cache"], round(pp["time_ms"])))
                assert j["choices"][0]["delta"] == {}, j
            delta = (j.get("choices") or [{}])[0].get("delta") or {}
            if first is None and (delta.get("content") or delta.get("reasoning_content")):
                first = round(now, 2)
    return events, first


try:
    for _ in range(240):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=1)
            break
        except Exception:
            time.sleep(0.5)
    ev, first = stream(True)
    print(f"cold, return_progress=true: {len(ev)} progress chunks, first token at {first} s")
    for e in ev:
        print("   at %.2f s: %d of %d tokens (cache %d), engine time %d ms" % e)
    assert ev, "no progress chunks for a long prompt"
    assert [e[1] for e in ev] == sorted(e[1] for e in ev) and all(e[1] < e[2] for e in ev)
    assert ev[0][1] == ev[0][3] and ev[0][4] == 0, "the first event is the start (nothing new in the cache yet)"
    ev2, first2 = stream(True)
    print(f"warm (all cached), return_progress=true: {len(ev2)} progress chunks, first token at {first2} s")
    assert not ev2, "a prompt that is in the cache has nothing to report"
    ev3, first3 = stream(None)
    print(f"field absent: {len(ev3)} progress chunks, first token at {first3} s")
    assert not ev3
    print("PROGRESS OK")
finally:
    p.terminate()
    try:
        p.wait(15)
    except Exception:
        p.kill()
