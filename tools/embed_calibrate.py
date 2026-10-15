"""Where on the cosine scale does an embedding model put related, unrelated and off-topic passages? (To pick Studio's default minimum score per model: docs/EMBEDDINGS.md has the table.)

Natural prose: the project's own documentation (docs/STUDIO.md, ENGINE.md, EMBEDDINGS.md, ARCHITECTURE.md, RUNBOOK.md) cut into passages like Studio does (1,800 characters, 350 overlap); 14 questions
whose answer is in the text (a passage is relevant when it holds a marker phrase), 6 questions about something else entirely. Needs `forge serve MODEL.gguf --port PORT` running.

    python tools/embed_calibrate.py PORT LABEL [--doc-prefix P] [--query-prefix P]

The marker phrases are looked for in the text as it is today: a question whose phrase has been reworded away from the docs says so and is left out.
"""
import json
import math
import sys
import urllib.request
from pathlib import Path

CHUNK, OVERLAP = 1800, 350


def split(text):
    out, i = [], 0
    while i < len(text):
        end = min(len(text), i + CHUNK)
        if end < len(text):
            cut = max(text.rfind("\n\n", i, end), text.rfind("\n", i + CHUNK // 2, end), text.rfind(" ", i + CHUNK // 2, end))
            if cut > i + CHUNK // 2:
                end = cut
        out.append(text[i:end].strip())
        if end >= len(text):
            break
        i = max(end - OVERLAP, i + 1)
    return [p for p in out if p]


QUESTIONS = [
    ("Which kinds of files can I attach to a chat?", ["OpenDocument spreadsheet", "EPUB book"]),
    ("How does Studio read a scanned PDF?", ["text recognition that comes with Windows"]),
    ("When is a file sent whole instead of searched?", ["70 percent of what is left of the context"]),
    ("How does the chat summarize a long paper?", ["writes a short summary of each part"]),
    ("What happens when a question matches nothing well?", ["two nearest passages when they score at least 0.4"]),
    ("Where does Studio keep my settings and chats?", ["%APPDATA%\\ForgeStudio"]),
    ("What does the server do against web pages that post to localhost?", ["DNS rebinding"]),
    ("Which request limits does the server enforce?", ["limited to 8 MiB"]),
    ("How do I see the metadata and the tensor table of a GGUF file?", ["metadata and tensor table"]),
    ("What does the prompt lookup feature do?", ["proposes the next tokens from there"]),
    ("What do I do when the engine reports no CUDA device?", ["CUDA_ERROR_NO_DEVICE"]),
    ("Which API route returns embeddings?", ["POST /v1/embeddings"]),
    ("How can I run Studio with the llama.cpp engine?", ["llama-server"]),
    ("Why do answers get slow when the video memory is almost full?", ["pages video memory"]),
]
OFF_TOPIC = [
    "How do I bake sourdough bread at home?",
    "Who won the football world cup in 2018?",
    "What is the capital of Australia?",
    "Explain the plot of Romeo and Juliet.",
    "How many calories are in a banana?",
    "What is the boiling point of water on a mountain?",
]


def embed(port, texts):
    out = []
    for i in range(0, len(texts), 16):
        req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/embeddings", data=json.dumps({"model": "embedding", "input": texts[i : i + 16]}).encode(), headers={"Content-Type": "application/json"})
        d = json.load(urllib.request.urlopen(req, timeout=600))["data"]
        out += [x["embedding"] for x in sorted(d, key=lambda x: x["index"])]
    return out


def unit(v):
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, max(0, int(round(p * (len(xs) - 1)))))]


def main():
    port, label = sys.argv[1], sys.argv[2]
    dp = qp = ""
    if "--doc-prefix" in sys.argv:
        dp = sys.argv[sys.argv.index("--doc-prefix") + 1]
    if "--query-prefix" in sys.argv:
        qp = sys.argv[sys.argv.index("--query-prefix") + 1]
    docs = Path(__file__).resolve().parent.parent / "docs"
    passages = []
    for name in ("ARCHITECTURE.md", "EMBEDDINGS.md", "ENGINE.md", "RUNBOOK.md", "STUDIO.md"):
        if (docs / name).exists():
            passages += split((docs / name).read_text(encoding="utf-8"))
    pv = [unit(v) for v in embed(port, [dp + p for p in passages])]
    qs = [q for q, _ in QUESTIONS] + OFF_TOPIC
    qv = [unit(v) for v in embed(port, [qp + q for q in qs])]
    related_best, unrelated_top, off_top = [], [], []
    all_unrelated = []
    for qi, (q, marks) in enumerate(QUESTIONS):
        scores = [sum(a * b for a, b in zip(qv[qi], p)) for p in pv]
        rel = [i for i, p in enumerate(passages) if any(m in p for m in marks)]
        if not rel:
            print("  (no passage holds the marker of:", q, ")")
            continue
        related_best.append(max(scores[i] for i in rel))
        non = [scores[i] for i in range(len(passages)) if i not in rel]
        unrelated_top.append(max(non))
        all_unrelated += non
    for oi in range(len(OFF_TOPIC)):
        qi = len(QUESTIONS) + oi
        scores = [sum(a * b for a, b in zip(qv[qi], p)) for p in pv]
        off_top.append(max(scores))
        all_unrelated += scores
    print(f"== {label}: {len(passages)} passages, {len(related_best)} answerable questions, {len(OFF_TOPIC)} off-topic")
    print(f"   related (best relevant passage): min {min(related_best):.3f}  p25 {pct(related_best, .25):.3f}  median {pct(related_best, .5):.3f}  max {max(related_best):.3f}")
    print(f"   non-relevant (top per answerable question): median {pct(unrelated_top, .5):.3f}  p90 {pct(unrelated_top, .9):.3f}  max {max(unrelated_top):.3f}")
    print(f"   off-topic (top per question): median {pct(off_top, .5):.3f}  max {max(off_top):.3f}")
    print(f"   every non-relevant pair: median {pct(all_unrelated, .5):.3f}  p95 {pct(all_unrelated, .95):.3f}  p99 {pct(all_unrelated, .99):.3f}")
    # the best cut: the threshold that keeps most related passages and drops the other questions' top non-relevant ones
    best = None
    for t in [x / 100 for x in range(0, 100)]:
        keep = sum(1 for r in related_best if r >= t) / len(related_best)
        leak = (sum(1 for u in unrelated_top if u >= t) + sum(1 for u in off_top if u >= t)) / (len(unrelated_top) + len(off_top))
        score = keep - leak
        if best is None or score > best[0] + 1e-9:
            best = (score, t, keep, leak)
    print(f"   best single cut {best[1]:.2f}: keeps {best[2]*100:.0f}% of the related, lets {best[3]*100:.0f}% of the questions' non-relevant top passages through")
    for t in (0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85):
        keep = sum(1 for r in related_best if r >= t) / len(related_best)
        leak = (sum(1 for u in unrelated_top if u >= t) + sum(1 for u in off_top if u >= t)) / (len(unrelated_top) + len(off_top))
        print(f"     threshold {t:.2f}: related kept {keep*100:3.0f}%   non-relevant leaking {leak*100:3.0f}%")


if __name__ == "__main__":
    main()
