"""Prompt-lookup drafting on and off: speed and equality of greedy output on workloads that copy (code edit, JSON, quotes, grammar fix) and one that does not (prose).
    python tools/pld_bench.py MODEL_NAME [HEAD_FILE] [temperature]
Starts forge serve twice (FORGE_PLD=0 / 1; with HEAD_FILE the model drafts with its MTP head, depth 3), prints tokens/s per workload, the engine's [spec] line, and whether the answers are identical."""
import json
import os
import subprocess
import sys
import time
import urllib.request

FORGE = os.environ.get("BENCH_FORGE", "target/release/forge")
LOGDIR = os.environ.get("BENCH_LOGDIR", ".")
name = sys.argv[1]
head = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] != "-" else None
temp = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
PORT = 18540
MODELS = os.environ.get("BENCH_MODEL_DIR", "models")

CODE = '''import json
import os


def load_config(path):
    with open(path) as f:
        return json.load(f)


def save_config(path, cfg):
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)


def merge(a, b):
    out = dict(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge(out[k], v)
        else:
            out[k] = v
    return out


def find_files(root, ext):
    result = []
    for base, dirs, files in os.walk(root):
        for name in files:
            if name.endswith(ext):
                result.append(os.path.join(base, name))
    return result


class Cache:
    def __init__(self, size):
        self.size = size
        self.data = {}

    def get(self, key):
        return self.data.get(key)

    def put(self, key, value):
        if len(self.data) >= self.size:
            self.data.pop(next(iter(self.data)))
        self.data[key] = value
'''
ARTICLE = """The bicycle has a longer and stranger history than most people expect. In 1817 the German inventor Karl von Drais built a wooden running machine with two wheels and a steerable front wheel, which riders pushed along with their feet. It was called the draisine, and for a few years it was a fashionable curiosity in Europe. The pedal arrived in the 1860s, when French makers attached cranks directly to the front wheel, and the velocipede, nicknamed the boneshaker because of its iron tires and stiff frame, became popular in Paris and later in London. By the 1870s the front wheel had grown enormous in order to go faster, producing the high-wheeler, or penny-farthing, which was fast but dangerous: a rider who hit a stone could be thrown over the handlebars. The safety bicycle of the 1880s solved the problem with two wheels of equal size and a chain drive to the rear wheel, and the invention of the pneumatic tire by John Boyd Dunlop in 1888 made riding comfortable. Within a decade bicycles were everywhere, and historians often argue that they changed society as much as any machine of the century: they gave ordinary people, and especially women, a new freedom to travel, they pushed governments to pave roads, and they helped create the mass production methods later used for cars."""
BAD = """Yesterday me and my freinds went to the old lighthouse on the hill, wich we has been planing to visit for a long time. The wether was perfect, and there were nobody else around, so we could take all the photographs we wanted. When we reached the top, we seen that the door was open, and the keeper, a kind old man who was wearing a blue sweater, invite us in for a cup of tea. He told us storys about the ships that had past the coast over the years, and about the night when a great storm almost destroyed the hole building. We stayed for two hours, and when we left he gived each of us a small brass compass as a present. I will never forgot that afternoon."""
WORK = [
    ("code edit", f"Add type hints and a one-line docstring to every function and method in this file. Output the complete updated file in a single code block.\n\n```python\n{CODE}```", 700),
    ("json", "Generate a JSON array with 10 objects. Each object has the fields id (number), name, email, active (boolean) and tags (an array of three short strings). Output only the JSON.", 500),
    ("quotes", f"Summarize the following text in five bullet points and quote one short phrase from the text in each bullet.\n\n{ARTICLE}", 300),
    ("grammar fix", f"Rewrite the following paragraph fixing all grammar and spelling errors, keeping everything else identical. Output only the corrected paragraph.\n\n{BAD}", 260),
    ("prose", "Explain how a hash map works and why lookups are fast, in about 250 words.", 300),
]


def run(pld):
    log = open(f"{LOGDIR}/pld_{pld}.log", "w")
    env = dict(os.environ, FORGE_SPEC_STATS="1", FORGE_PLD=str(pld))
    args = [FORGE, "serve", f"{MODELS}/{name}.gguf", "--port", str(PORT), "--ctx", "4096"]
    if head:
        args += ["--mtp", head, "--depth", "3"]
    p = subprocess.Popen(args, stdout=log, stderr=log, env=env)
    res = []
    try:
        for _ in range(240):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models", timeout=1)
                break
            except Exception:
                time.sleep(0.5)
        for label, prompt, mx in WORK:
            body = {"model": "x", "messages": [{"role": "user", "content": prompt}], "max_tokens": mx, "temperature": temp, "top_p": 0.95, "top_k": 40, "seed": 7, "chat_template_kwargs": {"enable_thinking": False}}
            req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
            r = json.load(urllib.request.urlopen(req, timeout=280))
            tm = r["timings"]
            res.append((label, tm["predicted_n"], tm["predicted_per_second"], r["choices"][0]["message"]["content"]))
    finally:
        p.terminate()
        time.sleep(1.5)
        log.close()
    return res


a = run(0)
b = run(1)
import json as _j
_j.dump({'off': a, 'on': b}, open(f'{LOGDIR}/pld_out.json', 'w'))
spec_lines = {0: [], 1: []}
for pld in (0, 1):
    for line in open(f"{LOGDIR}/pld_{pld}.log", encoding="utf-8", errors="replace"):
        if line.startswith("[spec]"):
            spec_lines[pld].append(line.strip())
print(f"{name} {'with head' if head else 'no head'} temperature {temp}")
print(f"{'workload':12s} {'off tok/s':>10s} {'on tok/s':>10s} {'gain':>7s}  identical")
for (label, n0, s0, t0), (_, n1, s1, t1) in zip(a, b):
    same = t0 == t1
    k = next((i for i, (x, y) in enumerate(zip(t0, t1)) if x != y), None)
    print(f"{label:12s} {s0:10.1f} {s1:10.1f} {s1 / s0:6.2f}x  {'yes' if same else f'no (differs at char {k} of {len(t0)})'}")
for i, line in enumerate(spec_lines[1]):
    print(f"  on  [{WORK[i][0]}] {line[:230]}" if i < len(WORK) else line)
for i, line in enumerate(spec_lines[0]):
    print(f"  off [{WORK[i][0]}] {line[:230]}" if i < len(WORK) else line)
