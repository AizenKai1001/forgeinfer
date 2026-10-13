"""The TPU compiler's own estimate of a training step, read from the optimized HLO text of a (deviceless) compile.

    python hlo_cost.py FILE.hlo.txt [--top 25] [--ghz 1.5]

libtpu annotates every scheduled instruction with `"estimated_cycles":"N"` in its backend_config (the cost model that the scheduler and the fusion passes use). Summing those over the ENTRY computation gives an
estimated step time without a TPU: for the 365M step the sum was 0.365 s against 0.379 s measured on the Kaggle v5e-8 (a 4 percent difference; one data point, so it ranks changes, it does not promise seconds).
A Pallas kernel (splash attention) carries its own estimate. Overlap between asynchronous copies and compute is not modelled (copy-start / copy-done carry no cycles), collectives are estimated like any op.
Instructions are grouped by what their `op_name` metadata says about their place in the program (no change of the model code is needed): forward, backward (the transposed program), the recomputation inside the
backward pass (remat), the Muon update (vmap over the stacked layers), the splash kernels and the reduction after the dK/dV kernel, the rest. `analyze(text)` returns the numbers, `render(result)` a table.
"""
import argparse
import collections
import re
import sys

GHZ = 1.5  # v5e core clock: 4 MXUs x 128 x 128 x 2 flops x 1.5 GHz = 197 Tflops, the spec sheet's bf16 peak
CYC = re.compile(r'"estimated_cycles":"(\d+)"')
OP = re.compile(r"^  (?:ROOT )?%(\S+) = (.+?) ([a-z][a-z0-9\-_.]*)\(")
OPNAME = re.compile(r'op_name="([^"]*)"')


def region(opcode, op_name):
    """The group of one instruction, from its opcode and its op_name path (JAX's name stack: jit(step_fn)/transpose(jvp(...))/checkpoint/rematted_computation/...)."""
    n = op_name or ""
    if "splash_mha_dkv" in n:
        return "attention: dK/dV reduction over the head group" if opcode == "reduce" else "attention: backward kernels"
    if "splash_mha" in n:
        return "attention: kernels"
    if "vmap(ik,kj->ij)" in n or "vmap(ik,jk->ij)" in n or "vmap(ij,jk->ik)" in n:
        return "muon: Newton-Schulz matmuls"
    if n.startswith("jit(step_fn)/vmap(") or n.startswith("vmap("):
        return "muon: update of the stacked matrices"
    if "rematted_computation" in n:
        return "backward: recomputation (remat)"
    if "transpose(jvp" in n:
        return "backward: gradients"
    if "jvp(" in n:
        return "forward"
    if not n:
        return "no name (copies, layout changes, constants)"
    return "optimizer and the rest (AdamW, clipping, gradient norm)"


def analyze(text, ghz=GHZ):
    lines = text.split("\n")
    start = next((i for i, ln in enumerate(lines) if ln.startswith("ENTRY")), None)
    if start is None:
        raise ValueError("no ENTRY computation in the HLO text")
    total = 0
    n_ops = 0
    by_region = collections.Counter()
    by_opcode = collections.Counter()
    by_region_n = collections.Counter()
    top = []
    for ln in lines[start + 1:]:  # the ENTRY computation is the last one of the text; its instructions are indented two spaces (a stray "}" at column 0 inside a long attribute must not end the loop)
        m = OP.match(ln)
        c = CYC.search(ln)
        if not m or not c:
            continue
        cyc = int(c.group(1))
        on = OPNAME.search(ln)
        name = on.group(1) if on else ""
        reg = region(m.group(3), name)
        total += cyc
        n_ops += 1
        by_region[reg] += cyc
        by_region_n[reg] += 1
        by_opcode[m.group(3)] += cyc
        top.append((cyc, m.group(3), m.group(2)[:60], name[-80:]))
    top.sort(reverse=True)
    to_ms = 1e3 / (ghz * 1e9)
    return {"total_ms": total * to_ms, "ops": n_ops,
            "regions": {k: {"ms": v * to_ms, "ops": by_region_n[k]} for k, v in by_region.most_common()},
            "opcodes": {k: v * to_ms for k, v in by_opcode.most_common(8)},
            "top": [(c * to_ms, o, s, n) for c, o, s, n in top[:40]]}


def render(res, top=0):
    out = [f"estimated step: {res['total_ms']:.1f} ms ({res['ops']} instructions with a cost estimate)"]
    for k, v in res["regions"].items():
        out.append(f"  {v['ms']:8.1f} ms {100 * v['ms'] / res['total_ms']:5.1f} %  {v['ops']:5d} ops  {k}")
    out.append("  by opcode: " + ", ".join(f"{k} {v:.0f} ms" for k, v in res["opcodes"].items()))
    for ms, opc, shape, name in res["top"][:top]:
        out.append(f"    {ms:7.3f} ms {opc:10s} {shape:60s} {name}")
    return "\n".join(out)


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# the matrix-unit work: one row per fusion of the ENTRY computation that contains a convolution (XLA:TPU's form of a dot), with its FLOPs, the time they would take at the peak rate, the estimate, the layouts
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

PEAK_FLOPS = 197e12  # one v5e chip, bf16 (4 MXUs x 128 x 128 x 2 flops x 1.5 GHz)
PEAK_BYTES_S = 819e9  # one v5e chip's HBM bandwidth
COMP_HEAD = re.compile(r"^(?:ENTRY )?%(\S+) \(")
INSTR = re.compile(r"^  (?:ROOT )?%(\S+) = (.+?) ([a-z][a-z0-9\-_.]*)\(")
ARRAY = re.compile(r"\b(pred|[suf]\d+|bf16|c64|c128|f8\w*)\[([\d,]*)\](?:\{([\d,]*)(?::([^}]*))?\})?")
CALLS = re.compile(r"(?:calls|to_apply)=%([^\s,}]+)")
BYTES_OF = {"pred": 1, "s8": 1, "u8": 1, "s16": 2, "u16": 2, "f16": 2, "bf16": 2, "s32": 4, "u32": 4, "f32": 4, "s64": 8, "u64": 8, "f64": 8, "c64": 8, "c128": 16}


def parse_array(text):
    """The first array type in `text` ('bf16[4,2048,1024]{2,1,0:T(8,128)(2,1)S(1)}') as a dict: dtype, dims, minor-to-major order (None without a layout), tiling ('T(8,128)(2,1)' or ''), memory space ('S(1)' or ''),
    elements, bytes and padded bytes (the tiles pad the second-minor dimension to the tile's rows and the minor dimension to its 128 lanes). None when there is no array type in the text (a tuple or a token)."""
    m = ARRAY.search(text)
    if m is None:
        return None
    dims = tuple(int(x) for x in m.group(2).split(",") if x)
    m2m = tuple(int(x) for x in m.group(3).split(",") if x) if m.group(3) is not None else None
    extra = m.group(4) or ""
    tile = "".join(re.findall(r"T\([\d,]+\)(?:\(\d+,\d+\))?", extra))
    space = "".join(re.findall(r"S\(\d+\)", extra))
    n = 1
    for d in dims:
        n *= d
    item = BYTES_OF.get(m.group(1), 4)
    padded = n
    first = re.match(r"T\((\d+(?:,\d+)*)\)((?:\(\d+,\d+\))*)", tile)
    if first and m2m and dims:
        t = [int(x) for x in first.group(1).split(",")]
        sub = re.findall(r"\((\d+),(\d+)\)", first.group(2))
        rows_mult = int(sub[0][0]) if sub else 1
        phys = [dims[i] for i in reversed(m2m)]  # major to minor
        if len(t) == 1:
            phys[-1] = -(-phys[-1] // t[0]) * t[0]
        elif len(phys) >= 2:
            phys[-1] = -(-phys[-1] // t[-1]) * t[-1]
            phys[-2] = -(-phys[-2] // (t[-2] * rows_mult)) * (t[-2] * rows_mult)
        padded = 1
        for d in phys:
            padded *= d
    return {"dtype": m.group(1), "dims": dims, "m2m": m2m, "tile": tile, "space": space, "elems": n, "bytes": n * item, "padded_bytes": padded * item}


def short_of(a):
    """'bf16[4,2048,1024]{2,1,0}' (no tiling; S1 appended for the fast memory space S(1)): the compact form of a parse_array dict for the tables."""
    lay = "{" + ",".join(str(i) for i in a["m2m"]) + "}" if a["m2m"] is not None else ""
    return f"{a['dtype']}[{','.join(str(d) for d in a['dims'])}]{lay}" + ("S1" if "S(1)" in a["space"] else "")


def short_array(text):
    """short_of of the first array type in `text` (the text itself, cut, when there is none)."""
    a = parse_array(text)
    return short_of(a) if a else text[:40]


def parse_computations(text):
    """{name: [(instruction name, result type text, opcode, line)]} of every computation of an optimized HLO text, and the name of the ENTRY computation."""
    comps, cur, entry = {}, None, None
    for ln in text.split("\n"):
        h = COMP_HEAD.match(ln)
        if h:
            cur = h.group(1)
            comps[cur] = []
            if ln.startswith("ENTRY"):
                entry = cur
            continue
        if cur is None:
            continue
        m = INSTR.match(ln)
        if m:
            comps[cur].append((m.group(1), m.group(2), m.group(3), ln))
    return comps, entry


def window_sizes(line):
    """The window attribute of a convolution line as {'size': [..], 'stride': [..], 'pad': [(lo, hi)..], 'lhs_dilate': [..], 'rhs_dilate': [..]} (a missing field: None)."""
    m = re.search(r"window=\{([^}]*)\}", line)
    out = {"size": None, "stride": None, "pad": None, "lhs_dilate": None, "rhs_dilate": None}
    if not m:
        return out
    for part in m.group(1).split():
        k, _, v = part.partition("=")
        if k == "pad":
            out["pad"] = [tuple(int(y) for y in x.split("_")) for x in v.split("x")]
        elif k in out:
            out[k] = [int(x) for x in v.split("x")]
    return out


def valid_pairs(in_size, out_size, size, stride=1, pad=(0, 0), lhs_dilate=1, rhs_dilate=1):
    """The number of (output position, kernel position) pairs of one spatial dimension of a convolution that read a real input element (not padding, not a hole of the dilated input): what XLA's own flop count uses."""
    n, limit = 0, (in_size - 1) * lhs_dilate + 1
    for o in range(out_size):
        for k in range(size):
            p = o * stride + k * rhs_dilate - pad[0]
            if 0 <= p < limit and p % lhs_dilate == 0:
                n += 1
    return n


def convolution_flops(line, shapes):
    """The FLOPs (2 per multiply-add) of one `convolution` instruction line; `shapes` maps the names of the instructions of its computation to their type text. Returns (flops, info) where info has the contraction
    size k, the output elements, the dim_labels and the operand arrays; (0, None) when the line cannot be understood. Every output element sums over the rhs input features and the valid taps of the window; the batch
    of the einsum is a spatial dimension of the convolution (window size 1), the sums over a batch (weight gradients) a window of the batch's size."""
    m = re.search(r"convolution\(%([^,\s)]+),\s*%([^,\s)]+)\)", line)
    dl = re.search(r"dim_labels=([0-9bfio]+)_([0-9bfio]+)->([0-9bfio]+)", line)
    r = INSTR.match(line)
    if not (m and dl and r):
        return 0, None
    lhs, rhs, out = (parse_array(shapes.get(m.group(1), "")), parse_array(shapes.get(m.group(2), "")), parse_array(r.group(2)))
    if lhs is None or rhs is None or out is None:
        return 0, None
    ll, rl, ol = dl.group(1), dl.group(2), dl.group(3)
    try:
        out_b, out_f, rhs_i = out["dims"][ol.index("b")], out["dims"][ol.index("f")], rhs["dims"][rl.index("i")]
    except (ValueError, IndexError):
        return 0, None
    w = window_sizes(line)
    pairs, nsp = 1, sum(1 for ch in ol if ch.isdigit())
    for j in range(nsp):
        pick = lambda lst, d=1: (lst[j] if lst and j < len(lst) else d)  # noqa: E731
        pairs *= valid_pairs(lhs["dims"][ll.index(str(j))], out["dims"][ol.index(str(j))], pick(w["size"]), pick(w["stride"]), pick(w["pad"], (0, 0)), pick(w["lhs_dilate"]), pick(w["rhs_dilate"]))
    flops = 2 * out_b * out_f * rhs_i * pairs
    return flops, {"k": flops // (2 * out["elems"]) if out["elems"] else 0, "out_elems": out["elems"], "dim_labels": ll + "_" + rl + "->" + ol, "lhs": lhs, "rhs": rhs, "out": out}


def contains_matmul(comps, name, seen=None):
    """True when the computation `name`, or one it calls, has a convolution or a dot."""
    seen = set() if seen is None else seen
    if name in seen or name not in comps:
        return False
    seen.add(name)
    for _, _, opc, ln in comps[name]:
        if opc in ("convolution", "dot"):
            return True
        for c in CALLS.findall(ln):
            if contains_matmul(comps, c, seen):
                return True
    return False


def matmul_flops_in(comps, name, seen=None):
    """[(flops, info, line)] of every convolution or dot in the computation `name` and in what it calls."""
    seen = set() if seen is None else seen
    if name in seen or name not in comps:
        return []
    seen.add(name)
    shapes = {n: t for n, t, _, _ in comps[name]}
    out = []
    for _, _, opc, ln in comps[name]:
        if opc == "convolution":
            f, info = convolution_flops(ln, shapes)
            out.append((f, info, ln))
        elif opc == "dot":
            f, info = dot_flops(ln, shapes)
            out.append((f, info, ln))
        for c in CALLS.findall(ln):
            out += matmul_flops_in(comps, c, seen)
    return out


def dot_flops(line, shapes):
    """The FLOPs of one `dot` instruction line: 2 x the result elements x the contracted size (lhs_contracting_dims)."""
    m = re.search(r"dot\(%([^,\s)]+),\s*%([^,\s)]+)\)", line)
    cd = re.search(r"lhs_contracting_dims=\{([\d,]*)\}", line)
    r = INSTR.match(line)
    if not (m and cd and r):
        return 0, None
    lhs, rhs, out = (parse_array(shapes.get(m.group(1), "")), parse_array(shapes.get(m.group(2), "")), parse_array(r.group(2)))
    if lhs is None or rhs is None or out is None:
        return 0, None
    k = 1
    for i in (int(x) for x in cd.group(1).split(",") if x):
        k *= lhs["dims"][i]
    return 2 * out["elems"] * k, {"k": k, "out_elems": out["elems"], "dim_labels": "dot", "lhs": lhs, "rhs": rhs, "out": out}


def matmul_fusions(text, ghz=GHZ, peak=PEAK_FLOPS, hbm=PEAK_BYTES_S):
    """One dict per instruction of the ENTRY computation that does matrix-unit work (a fusion that contains a convolution or a dot, or a bare one): name, fusion kind, op_name, region, the estimate (ms), the FLOPs, the
    time at the peak rate (ideal_ms), the time lost (est_ms - ideal_ms), the efficiency (ideal / estimate), the bytes that the fusion reads and writes (operands and results of the ENTRY instruction, as stored: tiling padding
    included) and their time at the HBM bandwidth (hbm_ms), the layouts of the operands and the result, and the convolution's own operand and result types. Sorted by lost time, the worst first."""
    comps, entry = parse_computations(text)
    if entry is None:
        raise ValueError("no ENTRY computation in the HLO text")
    shapes = {n: t for n, t, _, _ in comps[entry]}
    rows = []
    for name, rtype, opc, ln in comps[entry]:
        c = CYC.search(ln)
        if not c:
            continue
        calls = CALLS.findall(ln)
        if opc in ("convolution", "dot"):  # a matrix operation outside any fusion: its operands are instructions of the ENTRY computation
            f, info = (convolution_flops if opc == "convolution" else dot_flops)(ln, shapes)
            found = [(f, info, ln)]
        elif opc == "fusion" and calls and any(contains_matmul(comps, x) for x in calls):
            found = [t for x in calls for t in matmul_flops_in(comps, x)]
        else:
            continue
        flops = sum(f for f, _, _ in found)
        est = int(c.group(1)) * 1e3 / (ghz * 1e9)
        ideal = flops / peak * 1e3
        on = OPNAME.search(ln)
        opn = on.group(1) if on else ""
        ops = re.search(r"\(([^)]*)\)", ln[ln.index(opc + "(") + len(opc):])
        operands = [re.sub(r"/\*[^*]*\*/", "", o).strip().lstrip("%") for o in ops.group(1).split(",")] if ops else []  # (the printer puts /*index=5*/ comments between the operands)
        op_arrays = [parse_array(shapes.get(o, "")) for o in operands]
        res_arrays = [parse_array(x.group(0)) for x in ARRAY.finditer(rtype)]
        nbytes = sum(a["padded_bytes"] for a in op_arrays + res_arrays if a)
        kind = re.search(r"kind=(\w+)", ln)
        rows.append({"name": name, "kind": kind.group(1) if kind else opc, "op_name": opn, "region": region(opc, opn), "est_ms": est, "flops": flops, "ideal_ms": ideal, "lost_ms": est - ideal,
                     "eff": (ideal / est) if est > 0 else 0.0, "bytes": nbytes, "hbm_ms": nbytes / hbm * 1e3, "operands": [short_array(shapes.get(o, "")) for o in operands], "operand_bytes": [a["padded_bytes"] if a else 0 for a in op_arrays],
                     "result": [short_array(x.group(0)) for x in ARRAY.finditer(rtype)],
                     "conv": [{"labels": i["dim_labels"], "lhs": short_of(i["lhs"]), "rhs": short_of(i["rhs"]), "out": short_of(i["out"]), "k": i["k"]} for _, i, _ in found if i]})
    rows.sort(key=lambda r: -r["lost_ms"])
    return rows


def einsum_of(op_name):
    """The einsum string and the direction ('fwd', 'bwd', 'remat') that a matmul's op_name path carries: '...i,oi->...o/dot_general' -> ('...i,oi->...o', 'bwd')."""
    parts = op_name.split("/")
    ein = next((p for p in reversed(parts) if "->" in p), "")
    ein = re.sub(r"^(?:jvp|transpose|vmap)\(+", "", ein).rstrip(")")
    d = "remat" if "rematted_computation" in op_name else ("bwd" if "transpose(jvp" in op_name else "fwd")
    return ein, d


def family_key(r):
    """What a matmul fusion is, for grouping: the direction and einsum of its op_name and the shapes of its (first) convolution."""
    ein, d = einsum_of(r["op_name"])
    c = r["conv"][0] if r["conv"] else {"lhs": "?", "rhs": "?", "out": "?"}
    lay = lambda s: s.split("{", 1)[0]  # noqa: E731
    return (d, ein, lay(c["lhs"]), lay(c["rhs"]), lay(c["out"]))


def families(rows):
    """The matmul fusions grouped by family_key: [(key, count, est_ms, ideal_ms, lost_ms)], the most lost time first."""
    acc = collections.defaultdict(lambda: [0, 0.0, 0.0])
    for r in rows:
        a = acc[family_key(r)]
        a[0] += 1
        a[1] += r["est_ms"]
        a[2] += r["ideal_ms"]
    out = [(k, v[0], v[1], v[2], v[1] - v[2]) for k, v in acc.items()]
    out.sort(key=lambda x: -x[4])
    return out


def render_matmuls(rows, top=25, fams=12):
    """A table of the matmul fusions that lose the most time against the peak rate, then the families (same kind and shapes) that lose the most in sum."""
    est, ideal = sum(r["est_ms"] for r in rows), sum(r["ideal_ms"] for r in rows)
    flops = sum(r["flops"] for r in rows)
    out = [f"matmul fusions: {len(rows)}, {flops / 1e12:.2f} Tflop = {ideal:.1f} ms at {PEAK_FLOPS / 1e12:.0f} Tflops, estimated {est:.1f} ms ({100 * ideal / est if est else 0:.0f} % of the peak rate), lost {est - ideal:.1f} ms"]
    out.append(f"  the {min(top, len(rows))} fusions that lose the most:")
    out.append(f"  {'est ms':>7s} {'ideal':>7s} {'lost':>7s} {'eff':>4s} {'hbm ms':>7s}  fusion / direction einsum / convolution lhs x rhs -> out [dim_labels] (layouts minor-to-major) / operands of 1 MiB and more (S1 = in the fast memory)")
    for r in rows[:top]:
        ein, d = einsum_of(r["op_name"])
        c = r["conv"][0] if r["conv"] else None
        shapes = f"{c['lhs']} x {c['rhs']} -> {c['out']}  [{c['labels']}]" if c else ""
        big = " ".join(t for t, nb in zip(r["operands"], r["operand_bytes"]) if nb >= 1 << 20)  # the operands of 1 MiB and more as stored, S1 = in the fast memory
        out.append(f"  {r['est_ms']:7.3f} {r['ideal_ms']:7.3f} {r['lost_ms']:7.3f} {100 * r['eff']:3.0f}% {r['hbm_ms']:7.3f}  {r['name']} ({r['kind']}) {d} {ein}  {shapes}  operands: {big}")
    out.append(f"  the {fams} families that lose the most (same direction, einsum and shapes):")
    out.append(f"  {'count':>5s} {'est ms':>7s} {'ideal':>7s} {'lost':>7s} {'eff':>4s}  direction einsum: lhs x rhs -> out")
    for k, n, e, i, lost in families(rows)[:fams]:
        out.append(f"  {n:5d} {e:7.2f} {i:7.2f} {lost:7.2f} {100 * i / e if e else 0:3.0f}%  {k[0]} {k[1]}: {k[2]} x {k[3]} -> {k[4]}")
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--top", type=int, default=0)
    ap.add_argument("--ghz", type=float, default=GHZ)
    ap.add_argument("--matmuls", type=int, default=0, metavar="N", help="also a table of the N matmul fusions that lose the most time against the peak rate, and the families of them")
    a = ap.parse_args(argv)
    with open(a.file, encoding="utf-8", errors="replace") as f:
        text = f.read()
    print(render(analyze(text, a.ghz), a.top))
    if a.matmuls:
        print()
        print(render_matmuls(matmul_fusions(text, a.ghz), a.matmuls))
    return 0


if __name__ == "__main__":
    sys.exit(main())
