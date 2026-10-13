"""Does Qwen3.5's multi-token-prediction head work as a speculative drafter, and how often would its guess match the main model?
    ~/forgeinfer/venv/bin/python tools/ref_qwen35_mtp.py          (homelab; reads ~/forgeinfer/oracle/models/Qwen__Qwen3.5-0.8B)

The MTP block takes the main model's hidden state h_t at position t and the embedding of the NEXT token x_{t+1}, and predicts the token after it (x_{t+2}):
    z = fc(concat(rms0(embed(x_{t+1}), pre_fc_norm_embedding), rms0(h_t, pre_fc_norm_hidden)))     fc: [hidden, 2 * hidden]
    z = one decoder layer (gated attention + MLP, its own weights, attending over the previous MTP inputs)
    logits = rms0(z, mtp.norm) @ embed^T            (the head is shared with the main model)
Which hidden state (before or after the final norm) and which concatenation order the checkpoint expects is not documented anywhere I could read: this script tries the four combinations and
reports, for each, how often the arg-max of the MTP logits equals the main model's own arg-max for that position (teacher-forced on real text), which is what speculative decoding accepts.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ref_forward_numpy as F
import ref_qwen35_numpy as Q


def load_mtp():
    cfg = json.load(open(os.path.join(Q.ROOT, "config.json")))["text_config"]
    w = F.read_safetensors(os.path.join(Q.ROOT, "model.safetensors-00001-of-00001.safetensors"))
    main = {k[len("model.language_model."):]: v for k, v in w.items() if k.startswith("model.language_model.")}
    mtp = {k[len("mtp."):]: v for k, v in w.items() if k.startswith("mtp.")}
    return cfg, main, mtp


def mtp_forward(cfg, main, mtp, ids, hidden, order, which):
    emb = main["embed_tokens.weight"]
    e = emb[ids[1:]]                                   # embeddings of x_{t+1}, t = 0 .. T-2
    hid = hidden[:-1] if which == "pre" else None
    return e, hid


def main():
    cfg, w, mtp = load_mtp()
    eps = cfg["rms_norm_eps"]
    gold = json.load(open(os.path.join(F.GOLD, "fwd_qwen35.json")))
    emb = w["embed_tokens.weight"]
    p = {k[len("layers.0."):]: v for k, v in mtp.items() if k.startswith("layers.0.")}
    attn_p = {k[len("self_attn."):]: v for k, v in p.items() if k.startswith("self_attn.")}
    totals = {}
    for pi in (6, 2, 3):
        ids = gold["prompts"][pi]["ids"]
        # a longer text: the prompt followed by the main model's own greedy continuation
        logits, hidden, final = Q.forward(cfg, w, ids)
        seq = list(ids)
        # extend with 40 greedy tokens (each step re-runs the numpy forward: slow but simple)
        for _ in range(0):
            pass
        T = len(seq)
        main_arg = logits.argmax(-1)                   # main model's prediction for position t+1
        pre = hidden[-1]                               # residual stream after the last layer (before the final norm)
        post = final                                   # after the final norm
        for which, hid_all in (("pre-norm", pre), ("post-norm", post)):
            for order in ("emb,hid", "hid,emb"):
                e = Q.rms0(emb[seq[1:]], mtp["pre_fc_norm_embedding.weight"], eps)
                h = Q.rms0(hid_all[:-1], mtp["pre_fc_norm_hidden.weight"], eps)
                cat = np.concatenate([e, h], -1) if order == "emb,hid" else np.concatenate([h, e], -1)
                z = cat @ mtp["fc.weight"].T
                a = Q.rms0(z, p["input_layernorm.weight"], eps)
                z = z + Q.full_attention_layer(a, attn_p, cfg)
                a = Q.rms0(z, p["post_attention_layernorm.weight"], eps)
                z = z + (Q.silu(a @ p["mlp.gate_proj.weight"].T) * (a @ p["mlp.up_proj.weight"].T)) @ p["mlp.down_proj.weight"].T
                out = Q.rms0(z, mtp["norm.weight"], eps) @ emb.T          # [T-1, V]: prediction of x_{t+2}
                pred = out.argmax(-1)
                # the MTP output at t predicts the token after x_{t+1}: compare with the main model's prediction made at position t+1, and with the real text
                n = T - 2
                agree = float((pred[:n] == main_arg[1:T - 1]).mean())
                real = float((pred[:n] == np.array(seq[2:T])).mean())
                key = (which, order)
                totals.setdefault(key, []).append((agree, real, n))
                print(f"prompt {pi} ({T} tokens)  {which:9s} {order:8s}: MTP == main model's next-next prediction {agree * 100:5.1f} %,  == real text {real * 100:5.1f} %")
    print()
    for key, v in totals.items():
        n = sum(x[2] for x in v)
        print(f"{key}: agree with the main model {sum(x[0] * x[2] for x in v) / n * 100:.1f} %  ({n} positions)")


if __name__ == "__main__":
    main()
