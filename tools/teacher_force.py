"""Does forge pick the same next token as another engine, given that engine's own answer as context? (teacher forcing)
    python tools/teacher_force.py MODEL_NAME RESULT_JSON PRESET
RESULT_JSON is the bench_MODEL_NAME.json that tools/bench_engines.py writes (in BENCH_LOGDIR), PRESET the engine whose answers are replayed (llama, lmstudio, ...). For each of the five chat workloads of
pld_bench.py the Qwen3.5 chat prompt (thinking off) is rendered and checked against the prompt length that server reported; prompt + answer go through `forge ppl --dump-ref`; at every position of the
answer the top-1 token of forge is compared with the token the other engine produced. Prints the agreement, and for the disagreements how far (in nats) the other engine's token was behind forge's top choice.
Environment: FORGE_BIN (target/release/forge), BENCH_MODEL_DIR (models), BENCH_LOGDIR (.)."""
import json
import os
import statistics
import struct
import subprocess
import sys

name, src_json, key = sys.argv[1], sys.argv[2], sys.argv[3]
FORGE = os.environ.get("FORGE_BIN", "target/release/forge")
MODELS = os.environ.get("BENCH_MODEL_DIR", "models")
LOGDIR = os.environ.get("BENCH_LOGDIR", ".")
HERE = os.path.dirname(os.path.abspath(__file__))
model = f"{MODELS}/{name}.gguf"

src = open(os.path.join(HERE, "pld_bench.py"), encoding="utf-8").read()
ns = {}
exec(src[src.index("CODE = '''"):src.index("def run(pld):")], ns)
WORK = ns["WORK"]
rows = json.load(open(src_json))[key]["rows"]


def tokenize(text):
    out = subprocess.run([FORGE, "tokenize", model, text], capture_output=True, text=True, timeout=120).stdout
    return json.loads(out.strip().splitlines()[0])


def read_dump(path):
    b = open(path, "rb").read()
    assert b[:8] == b"FQREF002"
    k, n = struct.unpack("<II", b[8:16])
    stride = k * 8 + 4
    out = []
    for i in range(n):
        r = b[16 + i * stride: 16 + (i + 1) * stride]
        out.append((struct.unpack(f"<{k}I", r[: 4 * k]), struct.unpack(f"<{k}f", r[4 * k: 8 * k]), struct.unpack("<f", r[8 * k:])[0]))
    return out


tot_n = tot_same = 0
gaps = []
print(f"{name}: forge's top-1 against the token {key} produced, at every position of its own answer")
for (label, prompt, mx), row in zip(WORK, rows):
    head = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    p_ids = tokenize(head)
    if len(p_ids) != row["pn"]:
        print(f"  {label}: the rendered prompt has {len(p_ids)} tokens, the server counted {row['pn']}: template mismatch, skipped")
        continue
    full = head + row["text"]
    ids = tokenize(full)
    assert ids[: len(p_ids)] == p_ids
    base = os.path.join(LOGDIR, "tf_" + label.replace(" ", "_"))
    open(base + ".txt", "w", encoding="utf-8", newline="").write(full)
    r = subprocess.run([FORGE, "ppl", model, "--native", "--ctx", str(len(ids)), "--max-tokens", str(len(ids)), "--text", base + ".txt", "--dump-ref", base + ".ref"], capture_output=True, text=True, timeout=250)
    if r.returncode != 0:
        print(f"  {label}: forge ppl failed: {r.stderr[-300:]}")
        continue
    dump = read_dump(base + ".ref")
    n = same = 0
    g = []
    for i in range(len(p_ids) - 1, len(ids) - 1):
        top_ids, logp, tgt = dump[i]
        n += 1
        if top_ids[0] == ids[i + 1]:
            same += 1
        else:
            g.append(logp[0] - tgt)
    tot_n += n
    tot_same += same
    gaps += g
    med = f"{statistics.median(g):.2f}" if g else "-"
    print(f"  {label:12s} {n:5d} answer tokens, same top-1 {same:5d} ({100 * same / n:5.2f} %), {n - same:3d} differ, median gap of the other token {med} nats, largest {max(g) if g else 0:.2f}")
print(f"  total        {tot_n:5d} answer tokens, same top-1 {tot_same:5d} ({100 * tot_same / max(tot_n, 1):5.2f} %), median gap {statistics.median(gaps) if gaps else 0:.2f} nats, mean {statistics.mean(gaps) if gaps else 0:.2f}")
