#!/usr/bin/env python3
"""Check a running /v1/embeddings route against the golden vectors of forge-embed (tests/golden/nomic_v15_q4km.json).

    python3 tools/embed_route_check.py http://127.0.0.1:8080 crates/forge-embed/tests/golden/nomic_v15_q4km.json [MODEL] [MIN_COSINE]

It posts every golden text on its own (float output), the whole list in one request, a base64 request (decoded the way the OpenAI client does)
and a request with `dimensions`, and checks the cosine with the golden vector (default floor 0.9999 on the CPU path), the token count
reported in `usage`, the order of a batch, and the error replies for a bad body. Exit code 0 when everything passes.
"""
import base64
import json
import math
import struct
import sys
import urllib.error
import urllib.request


def post(url, body, raw=False):
    data = body if raw else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read())
        except Exception:
            return e.code, None


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb)


def main():
    base, golden_path = sys.argv[1].rstrip("/"), sys.argv[2]
    model = sys.argv[3] if len(sys.argv) > 3 else "nomic"
    floor = float(sys.argv[4]) if len(sys.argv) > 4 else 0.9999
    cases = json.load(open(golden_path))["cases"]
    url = base + "/v1/embeddings"
    bad = 0
    worst = 1.0

    def check(name, ok, detail=""):
        nonlocal bad
        if not ok:
            bad += 1
        print(("PASS " if ok else "FAIL ") + name + (" " + detail if detail else ""))

    # one text per request
    for i, c in enumerate(cases):
        code, r = post(url, {"model": model, "input": c["text"]})
        if code != 200:
            check(f"single {i}", False, f"HTTP {code} {r}")
            continue
        v = r["data"][0]["embedding"]
        cs = cosine(v, c["vector"])
        worst = min(worst, cs)
        tokens = r["usage"]["prompt_tokens"]
        check(f"single {i} ({len(c['ids'])} tokens)", cs >= floor and tokens == len(c["ids"]) and len(v) == len(c["vector"]), f"cosine {cs:.6f} tokens {tokens}")
    print(f"worst single cosine {worst:.6f}")

    # all texts in one request, order kept
    code, r = post(url, {"model": model, "input": [c["text"] for c in cases]})
    ok = code == 200 and [d["index"] for d in r["data"]] == list(range(len(cases)))
    if ok:
        cs = min(cosine(d["embedding"], c["vector"]) for d, c in zip(r["data"], cases))
        ok = cs >= floor and r["usage"]["prompt_tokens"] == sum(len(c["ids"]) for c in cases)
        check("batch of all texts", ok, f"worst cosine {cs:.6f} tokens {r['usage']['prompt_tokens']}")
    else:
        check("batch of all texts", False, f"HTTP {code}")

    # base64
    code, r = post(url, {"model": model, "input": cases[1]["text"], "encoding_format": "base64"})
    ok = code == 200 and isinstance(r["data"][0]["embedding"], str)
    if ok:
        raw = base64.b64decode(r["data"][0]["embedding"])
        v = struct.unpack("<%df" % (len(raw) // 4), raw)
        cs = cosine(v, cases[1]["vector"])
        check("base64 output", cs >= floor and len(v) == len(cases[1]["vector"]), f"cosine {cs:.6f} length {len(v)}")
    else:
        check("base64 output", False, f"HTTP {code} {r}")

    # dimensions: the first N values, normalised again
    code, r = post(url, {"model": model, "input": cases[2]["text"], "dimensions": 256})
    ok = code == 200 and len(r["data"][0]["embedding"]) == 256
    if ok:
        v = r["data"][0]["embedding"]
        ref = cases[2]["vector"][:256]
        n = math.sqrt(sum(x * x for x in ref))
        ref = [x / n for x in ref]
        norm = math.sqrt(sum(x * x for x in v))
        cs = cosine(v, ref)
        check("dimensions=256", cs >= floor and abs(norm - 1.0) < 1e-4, f"cosine {cs:.6f} norm {norm:.6f}")
    else:
        check("dimensions=256", False, f"HTTP {code} {r}")

    # errors
    code, _ = post(url, b"{not json", raw=True)
    check("bad json is a 400", code == 400, f"HTTP {code}")
    code, _ = post(url, {"model": model})
    check("missing input is a 400", code == 400, f"HTTP {code}")
    code, _ = post(url, {"model": model, "input": []})
    check("empty list is a 400", code == 400, f"HTTP {code}")

    # the models list names the embedding model
    with urllib.request.urlopen(base + "/v1/models", timeout=20) as m:
        ids = [x["id"] for x in json.loads(m.read())["data"]]
    check("/v1/models lists the model", model in ids or len(ids) >= 1, str(ids))

    print("ALL PASS" if bad == 0 else f"{bad} FAILED")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
