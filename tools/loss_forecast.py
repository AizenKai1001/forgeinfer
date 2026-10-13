"""Fit a power law to the production run's training loss and forecast the constant-LR stage (steps up to 40,500) and a longer run.

    python fit.py log.jsonl

Training loss per optimizer step (65,536 tokens each) is block-averaged (100 steps); L(t) = E + A * t^-alpha is fitted on blocks from step FIRST on
(least squares in log-spaced weights, scipy curve_fit). Prints the fit, the residual scatter, and forecasts of the stable-phase loss at several steps.
The decay (WSD, last 10 percent of the steps) is NOT in the fit: it lowers the loss further by an amount that must be measured (branch A of the plan).
"""
import json
import sys

import numpy as np
from scipy.optimize import curve_fit

path = sys.argv[1]
FIRST = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
steps, loss = [], []
for line in open(path):
    j = json.loads(line)
    steps.append(j["step"])
    loss.append(j["loss"])
steps = np.array(steps, float)
loss = np.array(loss, float)
B = 100
n = len(steps) // B
bs = steps[: n * B].reshape(n, B).mean(1)
bl = loss[: n * B].reshape(n, B).mean(1)
m = bs >= FIRST
x, y = bs[m], bl[m]


def f(t, E, A, a):
    return E + A * np.power(t, -a)


best = None
for E0 in (1.5, 2.0, 2.4):
    try:
        p, _ = curve_fit(f, x, y, p0=(E0, 20.0, 0.4), maxfev=20000, bounds=([0.5, 0.0, 0.05], [3.0, 1e4, 2.0]))
    except Exception:  # noqa: BLE001
        continue
    r = y - f(x, *p)
    s = float(np.sqrt((r**2).mean()))
    if best is None or s < best[0]:
        best = (s, p)
s, p = best
E, A, a = p
print(f"blocks of {B} steps from step {FIRST}: {m.sum()} points; last block mean at step {bs[-1]:.0f} = {bl[-1]:.4f}")
print(f"fit: L(t) = {E:.3f} + {A:.2f} * t^-{a:.3f}   (rms residual {s:.4f})")
for t in (10000, 15000, 20000, 25000, 30000, 40500, 45000, 60000, 81000, 90000):
    print(f"  step {t:6d} ({t * 65536 / 1e9:5.2f} B tokens): stable-phase train loss {f(t, *p):.4f}")
# the same on the validation-like rolling mean of the last 1000 steps
for FIRST2 in (1000, 4000, 6000):
    m2 = bs >= FIRST2
    try:
        p2, _ = curve_fit(f, bs[m2], bl[m2], p0=(2.0, 20.0, 0.4), maxfev=20000, bounds=([0.5, 0.0, 0.05], [3.0, 1e4, 2.0]))
        print(f"  fit from step {FIRST2}: E={p2[0]:.3f} A={p2[1]:.1f} alpha={p2[2]:.3f}  ->  step 40500: {f(40500, *p2):.4f}, 90000: {f(90000, *p2):.4f}")
    except Exception as e:  # noqa: BLE001
        print("  fit from", FIRST2, "failed", e)
