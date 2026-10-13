"""Executable spec of the Muon update used by forge-train: the orthogonalisation (Newton-Schulz with Jordan's quintic coefficients, or Polar Express), NorMuon, cautious weight decay.

    python tools/ref_muon.py            print the singular values of the output for a few shapes (checks the documented [0.7, 1.3] band) and write tests/golden/muon/*.f32 and tests/golden/muon_step/

tests/golden/muon/*.f32 hold (rows, cols as little-endian u32) then the float32 input G, then the float64-computed direction X5 after five iterations (float32). The Rust test
(crates/forge-train/tests/muon.rs) feeds G to the GPU path (fp16 operands, fp32 accumulation) and compares.

tests/golden/muon_step/ (crates/forge-train/tests/muon_opts.rs): the Muon matrices of a tiny model (manifest.json lists names and shapes), their starting weights (NAME.w0.f32), the gradients of
three optimizer steps (NAME.g1.f32 .. g3) and, for every variant of the update (a directory each), the weights after the third step (VARIANT/NAME.w3.f32), all computed in float64 by muon_step below.

Polar Express (Amsel, Persson, Musco, Gower 2025, arXiv 2505.16932): optimal_quintic / optimal_composition are the authors' reference code (github.com/NoahAmsel/PolarExpress, polar_express.py).
NorMuon (Li et al. 2025, arXiv 2510.05491; reference code github.com/zichongli5/NorMuon and modded-nanogpt record 41). Cautious weight decay: Chen et al. 2025, arXiv 2510.12402.
"""
import json
import os
import struct
from math import inf, sqrt

import numpy as np
from numpy.polynomial import Polynomial

A, B, C = 3.4445, -4.7750, 2.0315
JORDAN = (A, B, C)


def newton_schulz(g, iters=5, dtype=np.float64):
    x = g.astype(dtype)
    x = x / (np.linalg.norm(x) + 1e-7)
    tall = x.shape[0] > x.shape[1]
    if tall:
        x = x.T
    for _ in range(iters):
        a = x @ x.T
        b = B * a + C * (a @ a)
        x = A * x + b @ x
    return x.T if tall else x


# ---- Polar Express coefficients (the authors' code) ----

def optimal_quintic(l, u):
    """The odd quintic a x + b x^3 + c x^5 closest to 1 on [l, u] in the max norm (simplified Remez)."""
    assert 0 <= l <= u
    if 1 - 5e-6 <= l / u:
        return (15 / 8) / u, (-10 / 8) / (u ** 3), (3 / 8) / (u ** 5)
    q = (3 * l + u) / 4
    r = (l + 3 * u) / 4
    E, old_E = inf, None
    while not old_E or abs(old_E - E) > 1e-15:
        old_E = E
        LHS = np.array([[l, l ** 3, l ** 5, 1], [q, q ** 3, q ** 5, -1], [r, r ** 3, r ** 5, 1], [u, u ** 3, u ** 5, -1]])
        a, b, c, E = np.linalg.solve(LHS, np.ones(4))
        q, r = np.sqrt((-3 * b + np.array([-1, 1]) * sqrt(9 * b ** 2 - 20 * a * c)) / (10 * c))
    return float(a), float(b), float(c)


def optimal_composition(l, num_iters, safety_factor_eps=0.0, cushion=0.0):
    u = 1
    safety_factor = 1 + safety_factor_eps
    coefficients = []
    for it in range(num_iters):
        oc = optimal_quintic(max(l, cushion * u), u)
        p = Polynomial.identity() * Polynomial(oc)(Polynomial.identity() ** 2)
        if cushion * u > l:
            p *= 2 / (p(l) + p(u))
        if it < num_iters - 1:
            p = p(Polynomial.identity() / safety_factor)
        coefficients.append(tuple(float(x) for x in p.coef[1::2]))
        l = p(l)
        u = 2 - l
    return coefficients


def polar_express_coeffs(iters):
    """The coefficients forge-train uses (polar_express_coeffs in crates/forge-train/src/lib.rs): l = 1e-3, safety 2e-2, cushion 0.02 (the five-iteration set is modded-nanogpt record 38's)."""
    return optimal_composition(1e-3, iters, 2e-2, 0.02)


def orthogonalise(u, coefs, margin, eps):
    """x = u / (|u|_F margin + eps), then x <- a x + (b A + c A A) x with A = x x^T for each (a, b, c) (on the transpose for a tall matrix)."""
    x = u / (np.linalg.norm(u) * margin + eps)
    tall = x.shape[0] > x.shape[1]
    if tall:
        x = x.T
    for a, b, c in coefs:
        Am = x @ x.T
        Bm = b * Am + c * (Am @ Am)
        x = a * x + Bm @ x
    return x.T if tall else x


def muon_step(W, G, state, lr, wd, mu, polar=False, iters=5, normuon=None, beta2=0.95, cautious=False):
    """One Muon update of the matrix W (rows x cols, float64) with the gradient G. state: 'mom' (momentum, starts at 0) and 'v' (NorMuon's running lane mean squares, starts at 0).
    normuon: None, 'long' (lanes = rows unless the matrix is wide, then columns) or 'rows'."""
    rows, cols = W.shape
    state["mom"] = mu * state["mom"] + G
    u = G + mu * state["mom"]  # Nesterov
    if polar:
        x = orthogonalise(u, polar_express_coeffs(iters), 1.02, 1e-6)
    else:
        x = orthogonalise(u, [JORDAN] * iters, 1.0, 1e-7)
    scale = 0.2 * sqrt(max(rows, cols))  # Moonshot scaling: the update RMS of an AdamW step
    if normuon is not None:
        by_row = normuon == "rows" or rows >= cols
        lane_pow = (x * x).mean(axis=1) if by_row else (x * x).mean(axis=0)
        if "v" not in state:
            state["v"] = np.zeros_like(lane_pow)
        state["v"] = beta2 * state["v"] + (1 - beta2) * lane_pow
        gain = 1.0 / np.sqrt(np.maximum(state["v"], 1e-10))
        length = cols if by_row else rows
        pre = sqrt(length * lane_pow.sum())
        post = sqrt(length * (lane_pow * gain * gain).sum())
        x = x * (gain[:, None] if by_row else gain[None, :]) * (pre / max(post, 1e-10))
    upd = scale * x
    decay = lr * wd * W
    if cautious:
        decay = decay * (upd * W >= 0)
    return W - decay - lr * upd


# the Muon matrices of the tiny model of crates/forge-train/tests/muon_opts.rs (1 layer, dim 64, 2 heads of 64, 1 kv head, SwiGLU ffn 192): tall, square and wide ones
STEP_MATS = [("l0.wq", 128, 64), ("l0.wk", 64, 64), ("l0.wv", 64, 64), ("l0.wo", 64, 128), ("l0.w_gate", 192, 64), ("l0.w_up", 192, 64), ("l0.w_down", 64, 192)]
STEP_HP = {"lr": 0.02, "wd": 2.0, "mu": 0.95}  # a large decay so that the cautious mask shows in the result
STEP_VARIANTS = {
    "default": {},
    "iters4": {"iters": 4},
    "polar5": {"polar": True, "iters": 5},
    "polar4": {"polar": True, "iters": 4},
    "normuon_long": {"normuon": "long"},
    "normuon_rows": {"normuon": "rows"},
    "cautious": {"cautious": True},
    "all": {"polar": True, "iters": 5, "normuon": "long", "cautious": True},
}


def write_step_goldens(root):
    rng = np.random.default_rng(11)
    out = os.path.join(root, "muon_step")
    os.makedirs(out, exist_ok=True)
    W0, Gs = {}, {}
    for name, r, c in STEP_MATS:
        W0[name] = (rng.standard_normal((r, c)) * 0.05).astype(np.float32)
        Gs[name] = []
        for s in range(3):
            g = rng.standard_normal((r, c))
            uu, sv, vt = np.linalg.svd(g, full_matrices=False)
            sv = sv * (0.9 ** np.arange(len(sv)))
            # rows of different sizes too (the per-neuron statistics NorMuon equalises)
            g = ((uu * sv) @ vt) * np.exp(rng.standard_normal((r, 1)) * 0.5)
            Gs[name].append(g.astype(np.float32))
        W0[name].astype("<f4").tofile(os.path.join(out, f"{name}.w0.f32"))
        for s in range(3):
            Gs[name][s].astype("<f4").tofile(os.path.join(out, f"{name}.g{s + 1}.f32"))
    results = {}
    for var, opts in STEP_VARIANTS.items():
        d = os.path.join(out, var)
        os.makedirs(d, exist_ok=True)
        results[var] = {}
        for name, r, c in STEP_MATS:
            W = W0[name].astype(np.float64)
            state = {"mom": np.zeros_like(W)}
            ws = []
            for s in range(3):
                W = muon_step(W, Gs[name][s].astype(np.float64), state, STEP_HP["lr"], STEP_HP["wd"], STEP_HP["mu"], **opts)
                ws.append(W)
            W.astype("<f4").tofile(os.path.join(d, f"{name}.w3.f32"))
            results[var][name] = ws
    # how far apart the variants are (relative L2 distance of the 3-step weight change to the default's): the GPU test's tolerance must be well below these
    print("distance of each variant's 3-step weight change from the default update (relative L2, worst / best matrix):")
    for var in STEP_VARIANTS:
        ds = []
        for name, r, c in STEP_MATS:
            d0 = results["default"][name][2] - W0[name]
            dv = results[var][name][2] - W0[name]
            ds.append(np.linalg.norm(dv - d0) / np.linalg.norm(d0))
        print(f"  {var:13s} max {max(ds):.3f} min {min(ds):.3f}")
    manifest = {"mats": [{"name": n, "rows": r, "cols": c} for n, r, c in STEP_MATS], "hp": STEP_HP, "variants": STEP_VARIANTS,
                "polar_express": {str(k): polar_express_coeffs(k) for k in range(3, 8)}}
    with open(os.path.join(out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)


def main():
    rng = np.random.default_rng(7)
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests", "golden")
    out_dir = os.path.join(root, "muon")
    os.makedirs(out_dir, exist_ok=True)
    for name, (r, c) in {"wide": (64, 192), "tall": (192, 64), "square": (64, 64), "kv": (32, 64)}.items():
        g = rng.standard_normal((r, c)).astype(np.float32)
        # a realistic gradient is far from isotropic: give it a decaying spectrum
        u, s, vt = np.linalg.svd(g.astype(np.float64), full_matrices=False)
        s = s * (0.93 ** np.arange(len(s)))
        g = ((u * s) @ vt).astype(np.float32)
        x = newton_schulz(g, 5, np.float64)
        sv = np.linalg.svd(x, compute_uv=False)
        print(f"{name:7s} {r}x{c}: output singular values min {sv.min():.3f} max {sv.max():.3f}  (input condition number {np.linalg.cond(g):.1f})")
        for label, coefs, margin, eps in [("polar5", polar_express_coeffs(5), 1.02, 1e-6), ("polar4", polar_express_coeffs(4), 1.02, 1e-6), ("jordan4", [JORDAN] * 4, 1.0, 1e-7)]:
            sv2 = np.linalg.svd(orthogonalise(g.astype(np.float64), coefs, margin, eps), compute_uv=False)
            print(f"          {label:8s} singular values min {sv2.min():.3f} max {sv2.max():.3f}")
        with open(os.path.join(out_dir, f"{name}.f32"), "wb") as f:
            f.write(struct.pack("<II", r, c))
            f.write(g.astype("<f4").tobytes())
            f.write(x.astype("<f4").tobytes())
    write_step_goldens(root)


if __name__ == "__main__":
    main()
