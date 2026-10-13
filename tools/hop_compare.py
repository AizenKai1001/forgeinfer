"""Compare trainer logs of runs that share data, model and learning-rate schedule at matching steps (validation and mean training loss, speed).

    python tools/hop_compare.py REF.log ARM1.log [ARM2.log ...] [--every 100] [--names ref,a,b]

A log is the text of `forge train` (lines like `step   500/45000 loss 3.5321 lr 1.50e-3 gnorm ... 13123 tok/s (5.00 s)  val 3.4021`); the Kaggle runner's arm logs and the trainer's own train.log both work.
Prints, for every step where the reference has a validation value, the validation of every run and its difference to the reference, then the mean training loss of the last 100 steps at the same steps and the mean speed
(steps 200 on). Validation values are only comparable when the runs evaluated the same windows (the same --eval-batches and --micro product).
"""
import argparse
import re
import statistics

LINE = re.compile(r"^step\s+(\d+)/\d+ loss ([\d.]+)(.*)$")
TOKS = re.compile(r"(\d+) tok/s")
VAL = re.compile(r"\bval ([\d.]+)")


def parse(path):
    loss, val, tps = {}, {}, {}
    for line in open(path, encoding="utf-8", errors="replace"):
        m = LINE.match(line.strip())
        if not m:
            continue
        s = int(m.group(1))
        loss[s] = float(m.group(2))
        t = TOKS.search(m.group(3))
        if t:
            tps[s] = int(t.group(1))
        v = VAL.search(m.group(3))
        if v:
            val[s] = float(v.group(1))
    return loss, val, tps


def mean_last(loss, step, n=100):
    xs = [loss[s] for s in range(step - n + 1, step + 1) if s in loss]
    return statistics.fmean(xs) if len(xs) >= n // 2 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--every", type=int, default=100)
    ap.add_argument("--names", default="")
    ap.add_argument("--from-step", type=int, default=1200, help="paired comparison of the 100-step mean training loss from this step on")
    a = ap.parse_args()
    names = a.names.split(",") if a.names else [p.replace("\\", "/").split("/")[-1].replace(".log", "") for p in a.logs]
    runs = [parse(p) for p in a.logs]
    ref_loss, ref_val, _ = runs[0]
    steps = [s for s in sorted(ref_val) if s % a.every == 0]
    print("validation at matching steps (difference to " + names[0] + " in brackets)")
    print("step   " + "  ".join("%-22s" % n for n in names))
    for s in steps:
        cells = []
        for i, (loss, val, _) in enumerate(runs):
            if s in val:
                cells.append("%.4f" % val[s] + ("" if i == 0 else " (%+.4f)" % (val[s] - ref_val[s])))
            else:
                cells.append("-")
        print("%-6d " % s + "  ".join("%-22s" % c for c in cells))
    print()
    print("mean training loss of the last 100 steps")
    for s in steps[::2]:
        cells = []
        for i, (loss, val, _) in enumerate(runs):
            m = mean_last(loss, s)
            r = mean_last(ref_loss, s)
            cells.append("-" if m is None else ("%.4f" % m + ("" if i == 0 or r is None else " (%+.4f)" % (m - r))))
        print("%-6d " % s + "  ".join("%-22s" % c for c in cells))
    print()
    last_common = min(max(r[0]) for r in runs)
    grid = list(range(a.from_step, last_common + 1, 100))
    print("paired difference of the 100-step mean training loss to %s, steps %d to %d (mean +- sd of the differences, n points); negative = better than the reference" % (names[0], a.from_step, last_common))
    for n, (loss, val, tps) in list(zip(names, runs))[1:]:
        d = []
        for s in grid:
            m, r = mean_last(loss, s), mean_last(ref_loss, s)
            if m is not None and r is not None:
                d.append(m - r)
        if len(d) >= 2:
            print("  %-26s %+.4f +- %.4f (n %d)" % (n, statistics.fmean(d), statistics.stdev(d), len(d)))
    print()
    for n, (loss, val, tps) in zip(names, runs):
        xs = [v for s, v in tps.items() if s >= 200]
        if xs:
            print("%-28s mean speed from step 200: %7.0f tok/s over %d steps (last step %d)" % (n, statistics.fmean(xs), len(xs), max(loss)))


if __name__ == "__main__":
    main()
