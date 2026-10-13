"""An executable SPECIFICATION of the FlashAttention-2 algorithm (forward and backward, causal, grouped-query), tiled exactly as a CUDA kernel would tile it, checked against naive
attention in float64. The Turing (sm_75) kernels are written against this file: online softmax, the saved log-sum-exp, the D = rowsum(dO * O) trick, the loop orders.
    python3 tools/ref_flash_numpy.py
Layout per head: Q [T, d], K [T, d], V [T, d]; with grouped-query attention query head h reads kv head h // rep and the kv gradients are summed over the rep query heads.
"""
import numpy as np


def naive(Q, K, V, dO):
    T, d = Q.shape
    S = Q @ K.T / np.sqrt(d) + np.triu(np.full((T, T), -np.inf), 1)
    S = S - S.max(-1, keepdims=True)
    P = np.exp(S)
    P /= P.sum(-1, keepdims=True)
    O = P @ V
    dV = P.T @ dO
    dP = dO @ V.T
    dS = P * (dP - (dP * P).sum(-1, keepdims=True))
    return O, dS @ K / np.sqrt(d), dS.T @ Q / np.sqrt(d), dV


def flash_fwd(Q, K, V, Br, Bc):
    """Returns O and L = logsumexp of each row (what the backward pass needs instead of the T x T probability matrix)."""
    T, d = Q.shape
    O = np.zeros((T, d))
    L = np.zeros(T)
    scale = 1.0 / np.sqrt(d)
    for i0 in range(0, T, Br):
        i1 = min(i0 + Br, T)
        q = Q[i0:i1]
        m = np.full(i1 - i0, -np.inf)
        l = np.zeros(i1 - i0)
        acc = np.zeros((i1 - i0, d))
        for j0 in range(0, i1, Bc):                                   # causal: key blocks beyond the last query row of this block are never visited
            j1 = min(j0 + Bc, T)
            s = q @ K[j0:j1].T * scale
            if j1 > i0 + 1:                                           # diagonal block: mask keys after the query position
                rows = np.arange(i0, i1)[:, None]
                cols = np.arange(j0, j1)[None, :]
                s = np.where(cols > rows, -np.inf, s)
            m_new = np.maximum(m, s.max(-1))
            p = np.exp(s - m_new[:, None])
            corr = np.exp(m - m_new)                                  # rescale what was accumulated under the old maximum
            l = l * corr + p.sum(-1)
            acc = acc * corr[:, None] + p @ V[j0:j1]
            m = m_new
        O[i0:i1] = acc / l[:, None]
        L[i0:i1] = m + np.log(l)
    return O, L


def flash_bwd(Q, K, V, O, L, dO, Br, Bc):
    T, d = Q.shape
    scale = 1.0 / np.sqrt(d)
    D = (dO * O).sum(-1)                                              # D_i = rowsum(dO_i * O_i), computed once
    dQ, dK, dV = np.zeros_like(Q), np.zeros_like(K), np.zeros_like(V)
    for j0 in range(0, T, Bc):                                        # outer loop over key blocks (dK, dV stay in registers/shared memory), dQ is accumulated across blocks
        j1 = min(j0 + Bc, T)
        k, v = K[j0:j1], V[j0:j1]
        dk, dv = np.zeros_like(k), np.zeros_like(v)
        for i0 in range((j0 // Br) * Br, T, Br):                      # causal: only query blocks that can see this key block
            i1 = min(i0 + Br, T)
            q, do = Q[i0:i1], dO[i0:i1]
            s = q @ k.T * scale
            rows = np.arange(i0, i1)[:, None]
            cols = np.arange(j0, j1)[None, :]
            s = np.where(cols > rows, -np.inf, s)
            p = np.exp(s - L[i0:i1, None])
            dv += p.T @ do
            dp = do @ v.T
            ds = p * (dp - D[i0:i1, None])
            dQ[i0:i1] += ds @ k * scale
            dk += ds.T @ q * scale
        dK[j0:j1], dV[j0:j1] = dk, dv
    return dQ, dK, dV


def main():
    rng = np.random.default_rng(3)
    worst = 0.0
    for T, d, Br, Bc in ((37, 16, 8, 8), (64, 32, 16, 32), (50, 8, 16, 16), (129, 64, 32, 16), (7, 8, 16, 16)):
        Q, K, V, dO = (rng.normal(size=(T, d)) for _ in range(4))
        O0, dQ0, dK0, dV0 = naive(Q, K, V, dO)
        O, L = flash_fwd(Q, K, V, Br, Bc)
        dQ, dK, dV = flash_bwd(Q, K, V, O, L, dO, Br, Bc)
        errs = [np.abs(a - b).max() for a, b in ((O, O0), (dQ, dQ0), (dK, dK0), (dV, dV0))]
        worst = max(worst, max(errs))
        print(f"T={T:3d} d={d:2d} tiles {Br}x{Bc}: max error O {errs[0]:.1e} dQ {errs[1]:.1e} dK {errs[2]:.1e} dV {errs[3]:.1e}")
    # grouped-query: 4 query heads share 2 kv heads; dK/dV are summed over the query heads of a group
    T, d, nh, nkv = 45, 16, 4, 2
    rep = nh // nkv
    Q = rng.normal(size=(nh, T, d)); K = rng.normal(size=(nkv, T, d)); V = rng.normal(size=(nkv, T, d)); dO = rng.normal(size=(nh, T, d))
    dK_ref, dV_ref = np.zeros_like(K), np.zeros_like(V)
    dK_fl, dV_fl = np.zeros_like(K), np.zeros_like(V)
    for h in range(nh):
        g = h // rep
        O0, dQ0, dK0, dV0 = naive(Q[h], K[g], V[g], dO[h])
        dK_ref[g] += dK0; dV_ref[g] += dV0
        O, L = flash_fwd(Q[h], K[g], V[g], 16, 16)
        dQ, dK, dV = flash_bwd(Q[h], K[g], V[g], O, L, dO[h], 16, 16)
        dK_fl[g] += dK; dV_fl[g] += dV
        worst = max(worst, np.abs(dQ - dQ0).max())
    print(f"GQA 4 heads / 2 kv heads: dK error {np.abs(dK_fl - dK_ref).max():.1e}, dV error {np.abs(dV_fl - dV_ref).max():.1e}")
    worst = max(worst, np.abs(dK_fl - dK_ref).max(), np.abs(dV_fl - dV_ref).max())
    print("OK" if worst < 1e-10 else f"FAIL (worst {worst:.1e})")
    return 0 if worst < 1e-10 else 1


if __name__ == "__main__":
    raise SystemExit(main())
