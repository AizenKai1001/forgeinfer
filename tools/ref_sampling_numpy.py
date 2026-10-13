"""An executable SPECIFICATION of next-token sampling and of exact speculative decoding, with Monte-Carlo proofs on a tiny vocabulary.
    python3 tools/ref_sampling_numpy.py
 1. the sampler chain (repetition penalty -> temperature -> top-k -> top-p -> min-p -> softmax -> draw) and its edge cases;
 2. speculative decoding (Leviathan et al. / Chen et al.): draft k tokens from a cheap distribution q, verify them with the target distribution p in ONE pass, accept token x with probability
    min(1, p(x)/q(x)), on the first rejection draw from norm(max(0, p - q)), and after k acceptances draw a bonus token from p. The output distribution is exactly p: checked by simulation.
"""
import numpy as np


def softmax(z):
    z = z - np.max(z)
    e = np.exp(z)
    return e / e.sum()


def apply_repetition_penalty(logits, recent, penalty):
    out = logits.copy()
    for t in set(recent):
        out[t] = out[t] / penalty if out[t] > 0 else out[t] * penalty          # the llama.cpp / CTRL rule: shrink positive logits, push negative ones further down
    return out


def filter_probs(logits, temperature=1.0, top_k=0, top_p=1.0, min_p=0.0):
    """Returns the probability vector the draw is made from. temperature 0 means greedy (a one-hot)."""
    if temperature <= 0:
        p = np.zeros_like(logits)
        p[int(np.argmax(logits))] = 1.0
        return p
    p = softmax(logits / temperature)
    order = np.argsort(-p, kind="stable")
    keep = np.ones(len(p), dtype=bool)
    if top_k > 0:
        keep[order[top_k:]] = False
    if top_p < 1.0:
        c = np.cumsum(p[order])
        cut = int(np.searchsorted(c, top_p, side="left")) + 1                  # smallest prefix whose mass reaches top_p (always keeps at least one token)
        keep[order[cut:]] = False
    if min_p > 0.0:
        keep &= p >= min_p * p.max()
    q = np.where(keep, p, 0.0)
    return q / q.sum()


def speculative_step(p_fn, q_fn, ctx, k, rng):
    """One round: returns the list of tokens appended (1..k+1). p_fn(ctx) and q_fn(ctx) give the target / draft distributions for the next token after `ctx`."""
    draft, qs = [], []
    c = list(ctx)
    for _ in range(k):
        q = q_fn(c)
        x = int(rng.choice(len(q), p=q))
        draft.append(x); qs.append(q); c.append(x)
    out = []
    c = list(ctx)
    for x, q in zip(draft, qs):
        p = p_fn(c)                                                             # in a real engine all k+1 target distributions come from one batched forward pass
        if rng.random() < min(1.0, p[x] / q[x]):
            out.append(x); c.append(x)
        else:
            resid = np.maximum(p - q, 0.0)
            resid /= resid.sum()
            out.append(int(rng.choice(len(p), p=resid)))
            return out
    out.append(int(rng.choice(len(p_fn(c)), p=p_fn(c))))                        # bonus token
    return out


def main():
    rng = np.random.default_rng(0)
    # --- sampler edge cases
    logits = np.array([2.0, 1.0, 0.5, -1.0, -3.0])
    assert filter_probs(logits, 0.0).argmax() == 0 and filter_probs(logits, 0.0).sum() == 1.0
    p = filter_probs(logits, 1.0, top_k=2)
    assert np.count_nonzero(p) == 2 and abs(p.sum() - 1) < 1e-12
    p = filter_probs(logits, 1.0, top_p=0.5)
    assert np.count_nonzero(p) >= 1 and p[0] > 0
    p = filter_probs(logits, 1.0, min_p=0.3)                                  # p = [.607 .223 .135 .030 .004]: threshold .3 * .607 = .182 keeps two tokens
    assert np.count_nonzero(p) == 2, p
    r = apply_repetition_penalty(np.array([2.0, -2.0, 1.0]), [0, 1], 2.0)
    assert list(r) == [1.0, -4.0, 1.0]
    print("sampler chain edge cases OK")

    # --- speculative decoding is exact: a 5-token vocabulary, context-dependent target p and a deliberately different, cheaper draft q
    V = 5

    def p_fn(ctx):
        return softmax(np.array([np.sin(1.3 * len(ctx) + i) * 2.0 + (i == (ctx[-1] if ctx else 0)) for i in range(V)]))

    def q_fn(ctx):
        return softmax(np.array([np.cos(0.7 * len(ctx) + 2 * i) + 0.3 * (i == (ctx[-1] if ctx else 0)) for i in range(V)]))

    ctx = [1, 3]
    n = 200_000
    counts = np.zeros(V)
    for _ in range(n):
        counts[speculative_step(p_fn, q_fn, ctx, 3, rng)[0]] += 1                # the FIRST token of every round must be distributed exactly like p(. | ctx)
    emp, want = counts / n, p_fn(ctx)
    print("first token, target  :", np.round(want, 4))
    print("first token, spec dec:", np.round(emp, 4), " max deviation", float(np.abs(emp - want).max()))
    assert np.abs(emp - want).max() < 0.005
    # the second token of rounds that accepted the first must also follow p(. | ctx + first): check the joint (first, second) distribution
    pairs = np.zeros((V, V)); total = 0
    for _ in range(n):
        out = speculative_step(p_fn, q_fn, ctx, 3, rng)
        if len(out) >= 2:
            pairs[out[0], out[1]] += 1; total += 1
    joint = pairs / total
    cond = joint / joint.sum(axis=1, keepdims=True)
    err = max(float(np.abs(cond[a] - p_fn(ctx + [a])).max()) for a in range(V) if pairs[a].sum() > 2000)
    print("conditional second token vs target: max deviation", round(err, 4))
    assert err < 0.02
    # acceptance statistics: the expected number of accepted draft tokens per round is sum_x min(p, q) summed over steps (a property of the pair, not of the engine)
    acc = [sum(min(a, b) for a, b in zip(p_fn(ctx), q_fn(ctx)))]
    print("single-step acceptance probability for this p, q:", round(acc[0], 3))
    print("OK")


if __name__ == "__main__":
    main()
