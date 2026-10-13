#!/usr/bin/env python3
"""An independent evaluator for forgeckpt1 checkpoints: the JAX model of tpu/forge_jax.py scores the ladder files with the SAME token sequences that `forge eval-mc` / `forge eval-hellaswag` build, and prints the same
output lines. It exists to cross-check the Rust evaluators on a real Forge-1 checkpoint without a GPU (CPU JAX), and to evaluate on a TPU / JAX-GPU machine where the Rust binary does not run.

Two steps, because the tokenizer library and JAX may live in different Python environments (on the PC: the system Python has `tokenizers`, tpu/venv has `jax`):

    python tools/f1_eval_jax.py prep  --tokenizer tokenizer_f1.json --tasks hellaswag:1000:hs.jsonl,arc_easy:500:arc.jsonl --out seqs.json     # needs `tokenizers`
        task spec NAME:LIMIT:FILE (kind from the name: hellaswag -> the HellaSwag record format, everything else -> the eval-mc format); the encoding rules are those of crates/forge-cli/src/eval_cmd.rs
    python tools/f1_eval_jax.py score --ckpt ckpt.bin --seqs seqs.json [--batch-tokens 2048]                                       # needs jax + tpu/forge_jax.py

The output lines have the format of the Rust commands (`arc_easy: 500 examples  acc 39.00 %  acc_norm 36.80 %   (chance 25.0 %, 41 s)`), the HellaSwag line adds the per-token mean (llm.c style).
"""
import argparse
import json
import os
import sys
import time


def hs_preprocess(text):
    t = text.strip().replace(" [title]", ". ")
    out, depth = [], 0
    for c in t:
        if c == "[":
            depth += 1
        elif c == "]" and depth > 0:
            depth -= 1
        elif depth == 0:
            out.append(c)
    return "".join(out).replace("  ", " ")


def capitalize(s):
    return s[:1].upper() + s[1:].lower() if s else s


def cut_left(ids, clen, t_len):
    if len(ids) > t_len + 1:
        cut = len(ids) - (t_len + 1)
        ids, clen = ids[cut:], max(clen - cut, 1)
    return ids, clen


def prep(a):
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(a.tokenizer)
    enc = lambda s: tok.encode(s, add_special_tokens=False).ids
    out = {"tokenizer": a.tokenizer, "tasks": {}}
    for spec in a.tasks.split(","):
        name, limit, path = spec.split(":", 2)
        limit = int(limit)
        examples = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                j = json.loads(line)
                seqs = []
                if name == "hellaswag":
                    ctx = hs_preprocess(f"{j['activity_label']}: {j['ctx_a']} {capitalize(j['ctx_b'])}")
                    ctx_ids = enc(ctx)
                    for e in j["endings"]:
                        ending = hs_preprocess(e)
                        ids, clen = cut_left(ctx_ids + enc(" " + ending), len(ctx_ids), 256)
                        seqs.append((ids, clen, len(ending)))
                    label = int(j["label"])
                else:
                    for c in j["choices"]:
                        ctx_ids = enc(c["ctx"])
                        whole = enc(c["ctx"] + c["cont"])
                        ids = whole if len(whole) > len(ctx_ids) and whole[: len(ctx_ids)] == ctx_ids else ctx_ids + enc(c["cont"])
                        ids, clen = cut_left(ids, max(len(ctx_ids), 1), a.seq)
                        seqs.append((ids, clen, max(len(c["cont"].lstrip()), 1)))
                    label = int(j["label"])
                examples.append({"seqs": seqs, "label": label})
                if len(examples) >= limit:
                    break
        out["tasks"][name] = examples
        print(f"{name}: {len(examples)} examples, {sum(len(e['seqs']) for e in examples)} sequences, longest {max(len(s[0]) for e in examples for s in e['seqs'])} tokens", flush=True)
    with open(a.out, "w") as f:
        json.dump(out, f)


def score(a):
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tpu"))
    import jax
    import jax.numpy as jnp
    import numpy as np

    import forge_jax as fj

    c = fj.cfg_from_checkpoint(a.ckpt)
    params, _, _, step = fj.load_checkpoint(a.ckpt, c, with_state=False)
    print(f"checkpoint step {step}: {c.n_layer} layers, d {c.n_embd}, relu2={c.relu2} lambdas={c.lambdas} ve={list(c.ve_layers)} softcap={c.softcap}; JAX {jax.__version__} on {jax.devices()[0]}", flush=True)

    @jax.jit
    def logp(params, tokens, targets):
        h = fj.hidden(c, params, tokens, jnp.float32, False)
        z = jnp.einsum("bne,ve->bnv", h, params["head"] if c.untied else params["emb"])
        if c.softcap > 0:
            z = c.softcap * jnp.tanh(z / c.softcap)
        return jnp.take_along_axis(z, targets[..., None], axis=-1)[..., 0] - jax.scipy.special.logsumexp(z, axis=-1)

    data = json.load(open(a.seqs))
    for name, examples in data["tasks"].items():
        flat = [(e, k, tuple(s[0]), s[1], s[2]) for e, ex in enumerate(examples) for k, s in enumerate(ex["seqs"])]
        order = sorted(range(len(flat)), key=lambda i: -len(flat[i][2]))
        scores = {}
        t0 = time.time()
        i = 0
        while i < len(order):
            longest = len(flat[order[i]][2]) - 1  # inputs are ids[:-1]
            T = 64
            while T < longest:
                T *= 2
            B = max(1, a.batch_tokens // T)
            idx = order[i:i + B]
            tok = np.zeros((B, T), np.int32)
            tgt = np.zeros((B, T), np.int32)
            for r, k in enumerate(idx):
                ids = flat[k][2]
                tok[r, :len(ids) - 1] = ids[:-1]
                tgt[r, :len(ids) - 1] = ids[1:]
            lp = np.asarray(logp(params, jnp.asarray(tok), jnp.asarray(tgt)))
            for r, k in enumerate(idx):
                e, kk, ids, clen, chars = flat[k]
                scores[(e, kk)] = (float(lp[r, clen - 1:len(ids) - 1].sum()), len(ids) - clen, chars)
            i += B
        n = len(examples)
        acc = acc_norm = acc_tok = 0
        chance = 0.0
        for e, ex in enumerate(examples):
            m = len(ex["seqs"])
            chance += 1.0 / m
            ll = [scores[(e, k)][0] for k in range(m)]
            acc += max(range(m), key=lambda k: ll[k]) == ex["label"]
            acc_norm += max(range(m), key=lambda k: ll[k] / scores[(e, k)][2]) == ex["label"]
            acc_tok += max(range(m), key=lambda k: ll[k] / max(1, scores[(e, k)][1])) == ex["label"]
        dt = time.time() - t0
        if name == "hellaswag":
            print(f"HellaSwag validation, {n} examples: acc {100 * acc / n:.2f} %  acc_norm {100 * acc_norm / n:.2f} %  acc (per-token mean, llm.c style) {100 * acc_tok / n:.2f} %   ({dt:.0f} s; chance 25 %)  [JAX, step {step}]", flush=True)
        else:
            print(f"{name}: {n} examples  acc {100 * acc / n:.2f} %  acc_norm {100 * acc_norm / n:.2f} %   (chance {100 * chance / n:.1f} %, {dt:.0f} s)  [JAX, step {step}]", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prep")
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--tasks", required=True, help="NAME:LIMIT:FILE,... (NAME hellaswag = HellaSwag records, any other name = eval-mc records)")
    p.add_argument("--seq", type=int, default=256, help="the --seq of eval-mc (left cut beyond seq + 1 tokens)")
    p.add_argument("--out", required=True)
    s = sub.add_parser("score")
    s.add_argument("--ckpt", required=True)
    s.add_argument("--seqs", required=True)
    s.add_argument("--batch-tokens", type=int, default=2048)
    a = ap.parse_args()
    if a.cmd == "prep":
        prep(a)
    else:
        score(a)  # limit the CPU threads from outside (taskset / process affinity); JAX sizes its pool by the number of cores
    return 0


if __name__ == "__main__":
    sys.exit(main())
