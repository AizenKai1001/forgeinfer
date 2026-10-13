#!/usr/bin/env python3
"""Score a Hugging Face causal LM (GPT-2, Pythia, SmolLM2, ...) on the Forge-1 evaluation-ladder files with exactly the conventions of `forge eval-mc` / `eval-hellaswag` / `eval-lambada`
(the lm-evaluation-harness prompts, acc = highest total log-likelihood of the continuation, acc_norm = divided by the characters of the continuation). Two uses:

  1. validation of the data files: a known model (GPT-2 124M) must land near its published harness scores, which a wrong label or a broken prompt would not;
  2. same-pipeline reference rows: other small models on the very same examples (same shuffled subsample, same prompts), instead of numbers from papers with other prompts and shot counts.

    python3 tools/f1_eval_hf_ref.py MODEL [--data DIR] [--tasks hellaswag,arc_easy,...] [--limit N] [--device cpu|cuda] [--threads 3] [--batch 16] [--out FILE.json]

    python3 tools/f1_eval_hf_ref.py MODEL --ppl [--ppl-windows 64] [--ppl-seq 512]        bits per byte on the held-out domain files of the ladder (same text windows as `forge eval-ppl`)

Needs torch + transformers (the homelab venv ~/forgeinfer/venv has the CPU builds). MODEL is a Hugging Face id or a local directory. Tokenization follows the harness: the continuation is the tail of
the encoded context + continuation (whole minus context length); sequences are cut from the left at the model's maximum length.
"""
import argparse
import json
import math
import os
import sys
import time

TASKS = ["winogrande", "openbookqa", "arc_easy", "arc_challenge", "piqa", "sciq", "hellaswag", "lambada_openai", "mmlu_cloze", "mmlu_letter"]


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


def load_examples(path, limit, kind):
    ex = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            j = json.loads(line)
            if kind == "hellaswag":
                ctx = hs_preprocess(f"{j['activity_label']}: {j['ctx_a']} {j['ctx_b'][:1].upper() + j['ctx_b'][1:].lower()}")
                ex.append(([(ctx, " " + hs_preprocess(e)) for e in j["endings"]], int(j["label"]), [len(hs_preprocess(e)) for e in j["endings"]]))
            elif kind == "lambada":
                ex.append(([(j["ctx"], j["cont"])], 0, [max(1, len(j["cont"].lstrip()))]))
            else:
                ex.append(([(c["ctx"], c["cont"]) for c in j["choices"]], int(j["label"]), [max(1, len(c["cont"].lstrip())) for c in j["choices"]]))
            if len(ex) >= limit:
                break
    return ex


def run_ppl(a, man, tok, model, max_len, pad, torch):
    """bits per byte of this model on the windows `forge eval-ppl` scores: window k of a domain file starts at span * k // windows (seq + 1 tokens); the Forge-1 tokens are decoded to text, which this model's
    own tokenizer reads again. bits per byte = (sum of -log2 p over all tokens but the first) / (bytes of the text after the first token); document separators are dropped from the text."""
    import array
    import math as _m

    from tokenizers import Tokenizer

    f1 = Tokenizer.from_file(os.path.join(a.data, man["tokenizer"]["file"]))
    out = {"model": a.model, "windows": a.ppl_windows, "seq": a.ppl_seq, "domains": {}}
    for d in [x for x in a.ppl_domains.split(",") if x]:
        info = man["ppl"]["files"].get(d)
        if not info:
            continue
        arr = array.array("H")
        with open(os.path.join(a.data, info["file"]), "rb") as f:
            arr.frombytes(f.read())
        span = len(arr) - (a.ppl_seq + 1)
        bits = nbytes = ntok = 0.0
        t0 = time.time()
        windows = []
        truncated = 0
        for k in range(a.ppl_windows):
            start = span * k // a.ppl_windows
            text = f1.decode([int(t) & 0x7FFF for t in arr[start:start + a.ppl_seq + 1]], skip_special_tokens=True)
            ids_all = tok(text, add_special_tokens=False)["input_ids"]
            ids = ids_all[:max_len]
            if len(ids) < 8:
                continue
            first = len(tok.decode(ids[:1]).encode("utf-8"))
            if len(ids_all) > max_len:  # the model cannot read the whole window (GPT-2: 1024 tokens): count only the bytes of the tokens it did read
                truncated += 1
                windows.append((ids, len(tok.decode(ids).encode("utf-8")) - first))
            else:
                windows.append((ids, len(text.encode("utf-8")) - first))
        windows.sort(key=lambda w: -len(w[0]))
        with torch.no_grad():
            for i in range(0, len(windows), a.batch):
                batch = windows[i:i + a.batch]
                L = max(len(w[0]) for w in batch)
                x = torch.full((len(batch), L), pad, dtype=torch.long)
                m = torch.zeros((len(batch), L), dtype=torch.long)
                for j, (ids, _) in enumerate(batch):
                    x[j, :len(ids)] = torch.tensor(ids)
                    m[j, :len(ids)] = 1
                lg = model(input_ids=x.to(a.device), attention_mask=m.to(a.device)).logits.float()
                lp = torch.log_softmax(lg[:, :-1], dim=-1)
                tgt = x[:, 1:].to(lp.device)
                ll = lp.gather(2, tgt[:, :, None])[:, :, 0] * m[:, 1:].to(lp.device)
                for j, (ids, nb) in enumerate(batch):
                    bits += -float(ll[j].sum()) / _m.log(2)
                    nbytes += nb
                    ntok += len(ids) - 1
        r = {"bits_per_byte": bits / nbytes, "nats_per_token": bits * _m.log(2) / ntok, "bytes_per_token": nbytes / ntok, "windows": len(windows), "truncated_windows": truncated, "seconds": time.time() - t0}
        out["domains"][d] = r
        print(f"{d}: {r['bits_per_byte']:.4f} bits per byte  ({r['nats_per_token']:.4f} nats per token of this model's tokenizer, {r['bytes_per_token']:.3f} bytes per token, {len(windows)} windows, {truncated} cut at the model's maximum length, {r['seconds']:.0f} s)", flush=True)
        if a.out:
            with open(a.out, "w", encoding="utf-8") as f:
                json.dump(out, f, indent=1)
    if out["domains"]:
        print(f"mean over {len(out['domains'])} domains: {sum(v['bits_per_byte'] for v in out['domains'].values()) / len(out['domains']):.4f} bits per byte")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("--data", default="/mnt/data/forgeinfer-data/eval")
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--limit", type=int, default=500, help="examples per task (the files are shuffled: the first N are a random subsample)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--batch", type=int, default=16, help="sequences per forward pass")
    ap.add_argument("--max-tokens", type=int, default=4096, help="padded tokens per forward pass (memory bound)")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--tokenizer-file", help="a tokenizer.json to use instead of the model's own (a Forge model exported with `forge export-hf`: pass its tokenizer, e.g. tokenizer_f1.json)")
    ap.add_argument("--out")
    ap.add_argument("--ppl", action="store_true", help="instead of the benchmarks: bits per byte on the ppl/*.u16 domain files (token ids of tokenizer_f1.json decoded back to text, rescored with this model's tokenizer)")
    ap.add_argument("--ppl-windows", type=int, default=64)
    ap.add_argument("--ppl-seq", type=int, default=512)
    ap.add_argument("--ppl-domains", default="web,cosmo,fmath,owmath,wiki,books,python")
    a = ap.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(a.threads)
    man = json.load(open(os.path.join(a.data, "MANIFEST.json"), encoding="utf-8"))
    if a.tokenizer_file:
        from transformers import PreTrainedTokenizerFast

        tok = PreTrainedTokenizerFast(tokenizer_file=a.tokenizer_file)
    else:
        tok = AutoTokenizer.from_pretrained(a.model)
    try:
        model = AutoModelForCausalLM.from_pretrained(a.model, dtype=getattr(torch, a.dtype))
    except TypeError:  # older transformers
        model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=getattr(torch, a.dtype))
    model = model.to(a.device).eval()
    max_len = getattr(model.config, "max_position_embeddings", None) or getattr(model.config, "n_positions", None) or 1024
    pad = tok.eos_token_id if tok.eos_token_id is not None else 0
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model {a.model}: {n_params / 1e6:.1f} M parameters, max length {max_len}, device {a.device}, {a.threads} threads", flush=True)

    def encode(ctx, cont):
        c_ids = tok(ctx, add_special_tokens=False)["input_ids"]
        w_ids = tok(ctx + cont, add_special_tokens=False)["input_ids"]
        return w_ids, max(1, len(c_ids)) if len(w_ids) > len(c_ids) else len(w_ids) - 1

    @torch.no_grad()
    def logprobs(batch):
        """per sequence: (log-prob of every position after the context, greedy hit flags) for ids[clen:] given ids[:clen]"""
        L = max(len(ids) for ids, _ in batch)
        x = torch.full((len(batch), L), pad, dtype=torch.long)
        m = torch.zeros((len(batch), L), dtype=torch.long)
        for i, (ids, _) in enumerate(batch):
            x[i, :len(ids)] = torch.tensor(ids)
            m[i, :len(ids)] = 1
        lg = model(input_ids=x.to(a.device), attention_mask=m.to(a.device)).logits.float()
        lp = torch.log_softmax(lg, dim=-1)
        out = []
        for i, (ids, clen) in enumerate(batch):
            tgt = torch.tensor(ids[clen:], device=lp.device)
            rows = lp[i, clen - 1:len(ids) - 1]
            ll = rows.gather(1, tgt[:, None])[:, 0]
            hit = bool((rows.argmax(-1) == tgt).all())
            out.append((float(ll.sum()), hit, len(tgt)))
        return out

    if a.ppl:
        return run_ppl(a, man, tok, model, max_len, pad, torch)

    results = {"model": a.model, "params_m": round(n_params / 1e6, 1), "limit": a.limit, "tasks": {}}
    for t in [x for x in a.tasks.split(",") if x]:
        if t not in man["tasks"]:
            print(f"skip unknown task {t}")
            continue
        m = man["tasks"][t]
        kind = m["kind"]
        ex = load_examples(os.path.join(a.data, m["file"]), a.limit, kind)
        seqs = []  # (example, choice, ids, clen)
        for e, (pairs, _, _) in enumerate(ex):
            for c, (ctx, cont) in enumerate(pairs):
                ids, clen = encode(ctx, cont)
                if len(ids) > max_len:
                    cut = len(ids) - max_len
                    ids, clen = ids[cut:], max(1, clen - cut)
                seqs.append((e, c, ids, clen))
        order = sorted(range(len(seqs)), key=lambda i: -len(seqs[i][2]))
        score = {}
        t0 = time.time()
        i = 0
        while i < len(order):
            j = i
            longest = len(seqs[order[i]][2])
            while j < len(order) and (j - i) < a.batch and longest * (j - i + 1) <= a.max_tokens:
                j += 1
            j = max(j, i + 1)
            batch = [(seqs[k][2], seqs[k][3]) for k in order[i:j]]
            for k, r in zip(order[i:j], logprobs(batch)):
                score[(seqs[k][0], seqs[k][1])] = r
            i = j
        n = len(ex)
        if kind == "lambada":
            hits = sum(score[(e, 0)][1] for e in range(n))
            nll = sum(-score[(e, 0)][0] for e in range(n)) / n
            r = {"n": n, "acc": 100 * hits / n, "perplexity": math.exp(nll), "seconds": time.time() - t0}
            print(f"{t}: {n} examples  acc {r['acc']:.2f} %  perplexity {r['perplexity']:.2f}   ({r['seconds']:.0f} s)", flush=True)
        else:
            acc = acc_norm = acc_tok = 0
            for e, (pairs, label, chars) in enumerate(ex):
                ll = [score[(e, c)][0] for c in range(len(pairs))]
                ntok = [score[(e, c)][2] for c in range(len(pairs))]
                acc += max(range(len(ll)), key=lambda c: ll[c]) == label
                acc_norm += max(range(len(ll)), key=lambda c: ll[c] / chars[c]) == label
                acc_tok += max(range(len(ll)), key=lambda c: ll[c] / max(1, ntok[c])) == label
            r = {"n": n, "acc": 100 * acc / n, "acc_norm": 100 * acc_norm / n, "acc_per_token": 100 * acc_tok / n, "seconds": time.time() - t0}
            print(f"{t}: {n} examples  acc {r['acc']:.2f} %  acc_norm {r['acc_norm']:.2f} %  (per-token-normalised acc {r['acc_per_token']:.2f} %)   ({r['seconds']:.0f} s)", flush=True)
        results["tasks"][t] = r
        if a.out:
            with open(a.out, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
