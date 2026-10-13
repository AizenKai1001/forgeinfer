#!/usr/bin/env bash
# f1_eval.sh - the Forge-1 evaluation ladder: ONE command scores a checkpoint (a forgeckpt1 file of the Rust or the JAX trainer) and writes a results table (JSON + markdown).
#
#   bash tools/f1_eval.sh CKPT [--quick] [--dry-run] [--out DIR] [--data DIR] [--forge BIN] [--tokenizer FILE] [--tasks LIST] [--limit N] [--batch N] [--tight] [--name LABEL]
#   bash tools/f1_eval.sh --help         (all options, the ladder, exit codes)
#
# Needs: a GPU and the `forge` binary (it runs `forge eval-ppl`, `eval-hellaswag`, `eval-mc` and `eval-lambada`), bash, python3 (3.8+, standard library only) and the data directory made by
# tools/f1_eval_data.py (MANIFEST.json + task files). On the homelab the data is in /mnt/data/forgeinfer-data/eval; on a Lightning / Kaggle T4 pack that directory (about 90 MB) and pass --data.
# --dry-run needs no GPU: it validates the checkpoint header, the binary, the manifest and the data files and prints every command that a real run would execute.
#
# The program that does the work is the Python text in the here-document below (so that this single file can be copied to a cloud machine); bash only finds python3 and starts it.
set -u
SELF="${BASH_SOURCE[0]:-$0}"
PY=""
for cand in python3 python; do
  if command -v "$cand" > /dev/null 2>&1 && "$cand" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)' > /dev/null 2>&1; then PY="$cand"; break; fi
done
if [ -z "$PY" ]; then
  echo "f1_eval.sh: no working python3 (3.8 or newer) was found on the PATH (it composes the tables; no packages needed)" >&2
  exit 2
fi
TMP="$(mktemp "${TMPDIR:-/tmp}/f1_eval_XXXXXX")" || { echo "f1_eval.sh: cannot create a temporary file" >&2; exit 2; }
trap 'rm -f "$TMP"' EXIT
cat > "$TMP" <<'PYTHON_PROGRAM_END'
import argparse
import hashlib
import json
import math
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import time
from array import array

DEFAULT_DATA = "/mnt/data/forgeinfer-data/eval"
CORE = ["hellaswag", "arc_easy", "arc_challenge", "piqa", "winogrande", "openbookqa", "sciq"]
ALL = ["ppl"] + CORE + ["lambada_openai", "mmlu_cloze"]  # mmlu_letter (the harness's lettered format, at chance for this size) only on request: --tasks all,mmlu_letter
RUN_ORDER = ["ppl", "winogrande", "openbookqa", "arc_easy", "arc_challenge", "piqa", "sciq", "hellaswag", "lambada_openai", "mmlu_cloze", "mmlu_letter"]
GROUPS = {"all": ALL, "everything": ALL + ["mmlu_letter"], "core": CORE, "lambada": ["lambada_openai"], "mmlu": ["mmlu_cloze", "mmlu_letter"]}
PPL_DOMAINS = ["web", "cosmo", "fmath", "owmath", "wiki", "books", "python"]
EPILOG = """
The ladder (all zero-shot, the lm-evaluation-harness prompts; chance level and the metric used for the headline number are in MANIFEST.json and in the table):
  ppl             nats per token, perplexity and bits per byte on 7 held-out domains (web, cosmo, fmath, owmath, wiki, books, python)   forge eval-ppl
  hellaswag       HellaSwag validation, 10,042 examples, headline acc_norm                                                              forge eval-hellaswag
  arc_easy arc_challenge piqa winogrande openbookqa sciq                                                                                  forge eval-mc
  lambada_openai  LAMBADA (OpenAI version) 5,153 examples: accuracy of the greedy last word and word perplexity                         forge eval-lambada (needs the patch in tools/rust_patches)
  mmlu_cloze      MMLU test in the cloze format (the answer text is scored), a sanity row: about chance at this size; 2000 random questions   forge eval-mc
  mmlu_letter     the same questions in the harness's lettered format (not in the default run: --tasks all,mmlu_letter or --tasks everything)  forge eval-mc
--tasks takes names or groups (all, everything, core, lambada, mmlu), comma separated. --quick uses the per-task subsample sizes of the manifest (about a quarter of the time).
Exit codes: 0 all requested tasks scored, 1 some tasks failed (the table says which), 2 setup / validation error (nothing was run).
"""


def die(msg, code=2):
    print(f"f1_eval.sh: ERROR: {msg}", file=sys.stderr)
    sys.exit(code)


def warn(msg):
    print(f"[WARN]  {msg}")


def ok(msg):
    print(f"[ok]    {msg}")


def sha256_file(path, limit=None):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


# ----------------------------------------------------------------------------- checkpoint, binary, data
def read_header(path):
    try:
        with open(path, "rb") as f:
            line = f.readline(1 << 16)
    except OSError as e:
        die(f"cannot read the checkpoint {path}: {e}")
    try:
        h = json.loads(line.decode("utf-8"))
    except Exception:  # noqa: BLE001
        die(f"{path} is not a forgeckpt1 file (the first line is not a JSON header). Pass the checkpoint file written by `forge train` (ckpt_latest.bin or ckpt_<step>.bin), not a model directory")
    if h.get("magic") != "forgeckpt1":
        die(f"{path}: the header says magic={h.get('magic')!r}, expected 'forgeckpt1'")
    need = len(line) + 4 * int(h.get("n_params", 0))
    size = os.path.getsize(path)
    if size < need:
        die(f"{path} is truncated: {size} bytes, but the header ({h.get('n_params')} parameters) needs at least {need}. Was the copy interrupted?")
    return h, size


def find_forge(arg):
    home = os.path.expanduser("~")
    if arg and not (os.path.isfile(arg) and os.access(arg, os.X_OK)):
        die(f"--forge {arg}: not an executable file")
    if not arg and os.environ.get("FORGE_EVAL_BIN") and not (os.path.isfile(os.environ["FORGE_EVAL_BIN"]) and os.access(os.environ["FORGE_EVAL_BIN"], os.X_OK)):
        die(f"$FORGE_EVAL_BIN={os.environ['FORGE_EVAL_BIN']}: not an executable file")
    cands = [arg, os.environ.get("FORGE_EVAL_BIN"), "./forge", os.path.join(home, "forgeinfer/target-dev3/release/forge"), "./target/release/forge", os.path.join(home, "forgeinfer/bin/forge-f1_v1"), shutil.which("forge")]
    tried = []
    for c in cands:
        if not c:
            continue
        tried.append(c)
        if os.path.isfile(c) and os.access(c, os.X_OK):
            return os.path.abspath(c)
    die("the `forge` binary was not found. Tried: " + (", ".join(tried) or "(nothing)") + ". Pass --forge PATH or set FORGE_EVAL_BIN (homelab: ~/forgeinfer/target-dev3/release/forge)")


def find_tokenizer(arg, data):
    home = os.path.expanduser("~")
    if arg and not os.path.isfile(arg):
        die(f"--tokenizer {arg}: no such file")
    for c in [arg, os.environ.get("F1_TOKENIZER"), os.path.join(data, "tokenizer_f1.json"), os.path.join(home, "forgeinfer/data/tokenizer_f1.json"), "./tokenizer_f1.json"]:
        if c and os.path.isfile(c):
            return os.path.abspath(c)
    die("the tokenizer (tokenizer_f1.json) was not found. Pass --tokenizer FILE (homelab: ~/forgeinfer/data/tokenizer_f1.json)")


def run_quiet(cmd, timeout=60):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.returncode, (r.stdout or "") + (r.stderr or "")
    except Exception as e:  # noqa: BLE001
        return 127, str(e)


def gpu_info():
    if not shutil.which("nvidia-smi"):
        return None
    rc, out = run_quiet(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,utilization.gpu", "--format=csv,noheader,nounits"], 20)
    if rc != 0 or not out.strip():
        return None
    name, tot, used, util = [x.strip() for x in out.strip().splitlines()[0].split(",")]
    return {"name": name, "memory_total_mib": int(tot), "memory_used_mib": int(used), "utilization_pct": int(util)}


def pick_batch(seq, fwd):
    """rows per forward pass: about fwd / seq, but forge's cross-entropy chunk needs batch x seq <= 2048 or a multiple of 2048"""
    b = max(1, fwd // seq)
    while b > 1 and b * seq > 2048 and (b * seq) % 2048:
        b -= 1
    return b


# ----------------------------------------------------------------------------- bits per byte
def byte_table(tok_path):
    """bytes of the text that every token id stands for (byte-level BPE: one character of the token string per byte; special tokens are not text)"""
    d = json.load(open(tok_path, encoding="utf-8"))
    vocab = d["model"]["vocab"]
    special = {a["id"] for a in d.get("added_tokens", []) if a.get("special")}
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(0xA1, 0xAC + 1)) + list(range(0xAE, 0xFF + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    mapped = {chr(c) for c in cs}
    size = max(vocab.values()) + 1
    table = [0] * size
    for tok, i in vocab.items():
        if i in special:
            continue
        table[i] = len(tok) if all(ch in mapped for ch in tok) else len(tok.encode("utf-8"))
    return table


def window_bytes(path, table, windows, seq):
    """bytes of the text of the tokens that `forge eval-ppl` scores: window k starts at span * k // windows and scores the seq tokens after its first token"""
    arr = array("H")
    with open(path, "rb") as f:
        arr.frombytes(f.read())
    if sys.byteorder == "big":
        arr.byteswap()
    span = len(arr) - (seq + 1)
    if span < 0:
        return 0, 0
    total = 0
    for k in range(windows):
        start = span * k // windows
        total += sum(table[t & 0x7FFF] if (t & 0x7FFF) < len(table) else 0 for t in arr[start + 1:start + seq + 1])
    return total, windows * seq


# ----------------------------------------------------------------------------- parsing of the forge outputs
RE_MC = re.compile(r"^(?P<name>\S+): (?P<n>\d+) examples\s+acc (?P<acc>[\d.]+) %\s+acc_norm (?P<norm>[\d.]+) %\s+\(chance (?P<chance>[\d.]+) %, (?P<sec>\d+) s\)", re.M)
RE_HS = re.compile(r"HellaSwag validation, (?P<n>\d+) examples: acc (?P<acc>[\d.]+) %\s+acc_norm (?P<norm>[\d.]+) %\s+acc \(per-token mean, llm\.c style\) (?P<tok>[\d.]+) %\s+\((?P<sec>\d+) s", re.M)
RE_LAM = re.compile(r"^(?P<name>\S+): (?P<n>\d+) examples\s+acc (?P<acc>[\d.]+) %\s+perplexity (?P<ppl>[\d.]+)\s+\(mean (?P<nats>[\d.]+) nats per word, (?P<tpw>[\d.]+) target tokens per word, (?P<sec>\d+) s", re.M)
RE_SELF = re.compile(r"selfcheck: .*= (?P<diff>[\d.]+) nats")
RE_PPL = re.compile(r"^(?P<name>\S+): (?P<nats>[\d.]+) nats per token \(perplexity (?P<ppl>[\d.]+)\) over (?P<count>\d+) tokens", re.M)


def stderr_pct(p, n):
    p = p / 100.0
    return 100.0 * math.sqrt(max(p * (1 - p), 0.0) / max(n - 1, 1))


def run_command(cmd, log_path, timeout):
    """run a command, echo and log its output, return (exit code, text, seconds)"""
    t0 = time.time()
    lines = []
    try:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace")
    except OSError as e:
        return 127, f"cannot start the command: {e}", 0.0
    deadline = t0 + timeout
    try:
        import threading

        def pump():
            for ln in p.stdout:
                lines.append(ln)
                print("    | " + ln.rstrip())

        th = threading.Thread(target=pump, daemon=True)
        th.start()
        while th.is_alive():
            th.join(1.0)
            if time.time() > deadline:
                p.kill()
                th.join(5.0)
                lines.append(f"[f1_eval] killed after {timeout} s (--timeout)\n")
                p.wait()
                break
        rc = p.wait()
    except KeyboardInterrupt:
        p.kill()
        p.wait()
        lines.append("[f1_eval] interrupted\n")
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("".join(lines))
        raise
    text = "".join(lines)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("$ " + " ".join(cmd) + "\n" + text)
    return rc, text, time.time() - t0


# ----------------------------------------------------------------------------- reporting
def fmt(v, d=1):
    return "-" if v is None else f"{v:.{d}f}"


def render_markdown(res, ref):
    c = res["checkpoint"]
    cfg = c["cfg"]
    L = [f"# Forge-1 evaluation: {res['name']}", ""]
    L.append(f"- checkpoint `{c['path']}`: step {c['step']}, {c['n_params'] / 1e6:.1f} M parameters ({cfg.get('n_layer')} layers, d {cfg.get('n_embd')}, {cfg.get('n_head')} heads / {cfg.get('n_kv')} kv, ffn {cfg.get('n_ff')}, vocabulary {cfg.get('vocab')})")
    L.append(f"- mode **{res['mode']}**, started {res['created']}, {res['seconds_total']:.0f} s in total; host {res['env']['host']}, GPU {res['env'].get('gpu') or 'n/a'}")
    L.append(f"- binary `{res['env']['forge']['path']}` (sha256 {res['env']['forge']['sha256'][:12]}), data manifest of {res['env']['manifest_updated']} (seed {res['env']['manifest_seed']})")
    if res.get("val_loss_from_log"):
        v = res["val_loss_from_log"]
        L.append(f"- validation loss in the trainer log at step {v['step']}: **{v['value']:.4f}** nats per token (the trainer's own held-out set, `{v['file']}`)")
    L += ["", "## Benchmarks (zero-shot)", ""]
    prev = res.get("previous", {})
    head = ["task", "examples", "headline metric", "score %", "± se", "acc %", "acc_norm %", "chance %", "time s"]
    if prev:
        head.append("delta vs previous")
    L.append("| " + " | ".join(head) + " |")
    L.append("|" + "---|" * len(head))
    scores = []
    for t in CORE + ["lambada_openai", "mmlu_cloze", "mmlu_letter"]:
        r = res["tasks"].get(t)
        if r is None:
            continue
        if r["status"] != "ok":
            L.append(f"| {r['title']} | - | - | **{r['status'][:60]}** | | | | | |" + (" |" if prev else ""))
            continue
        extra = ""
        if prev:
            pv = prev.get(t)
            extra = " " + (f"{r['headline'] - pv:+.1f}" if pv is not None else "-") + " |"
        acc_norm = r.get("acc_norm")
        L.append(f"| {r['title']} | {r['n']}{' (subsample)' if r['subsample'] else ''} | {r['headline_metric']} | **{r['headline']:.1f}** | {r['headline_se']:.1f} | {fmt(r.get('acc'))} | {fmt(acc_norm)} | {r['chance']:.1f} | {r['seconds']:.0f} |" + extra)
        if t == "lambada_openai":
            L.append(f"| LAMBADA word perplexity | {r['n']} | perplexity | **{r['perplexity']:.1f}** | | | | | |" + (" |" if prev else ""))
    if res["averages"].get("core_headline") is not None:
        L.append("")
        L.append(f"Average of the headline scores over {', '.join(res['averages']['core_tasks'])}: **{res['averages']['core_headline']:.2f} %** (chance level of the same tasks: {res['averages']['core_chance']:.2f} %).")
    L += ["", "Metrics: acc = highest total log-likelihood of the continuation; acc_norm = highest log-likelihood divided by the number of characters of the continuation (the lm-evaluation-harness definitions). "
          "The headline metric is the one most published tables use for that task (see the research note); WinoGrande has no acc_norm in the harness (the column shows the same quantity). "
          "se = binomial standard error of the headline score.", ""]
    ppl = res.get("ppl")
    if ppl and ppl.get("domains"):
        L += ["## Held-out perplexity (the validation tails of the training sources)", "",
              f"{ppl['windows']} windows of {ppl['seq']} + 1 tokens per domain, evenly spread; bits per byte = nats per token / ln 2 / bytes per token (special tokens count as 0 bytes), comparable across tokenizers.", "",
              "| domain | nats / token | perplexity | bytes / token | bits / byte | tokens scored |", "|---|---|---|---|---|---|"]
        for d, r in ppl["domains"].items():
            L.append(f"| {d} | {r['nats']:.4f} | {r['ppl']:.1f} | {fmt(r.get('bytes_per_token'), 3)} | {fmt(r.get('bpb'), 4)} | {r['count']} |")
        L.append(f"| **mean** | **{ppl['mean_nats']:.4f}** | | | **{fmt(ppl.get('mean_bpb'), 4)}** | |")
        L.append("")
    if ref and ref.get("models"):
        L += render_reference(res, ref)
    L += ["## Commands", "", "```"] + [f"{r['command']}" for r in ([res["ppl"]] if res.get("ppl") else []) + [res["tasks"][t] for t in RUN_ORDER if t in res["tasks"]] if r.get("command")] + ["```", ""]
    failed = [t for t, r in res["tasks"].items() if r["status"] != "ok"]
    if failed:
        L += ["## Failures", ""] + [f"- {t}: {res['tasks'][t]['status']}" + (f": {res['tasks'][t]['error']}" if res["tasks"][t].get("error") else "") for t in failed] + [""]
    return "\n".join(L) + "\n"


def render_reference(res, ref):
    cols = [("hellaswag", "HellaSwag"), ("arc_easy", "ARC-E"), ("arc_challenge", "ARC-C"), ("piqa", "PIQA"), ("winogrande", "WinoG"), ("openbookqa", "OBQA"), ("sciq", "SciQ"), ("lambada_openai", "LAMBADA acc"), ("mmlu_cloze", "MMLU cloze")]
    L = ["## Compared with published models", "",
         "Headline metrics (acc_norm for HellaSwag / ARC-C / OBQA, acc for the others), zero-shot. A value in brackets is the other metric of that task (the source gives only that one); a star marks a number whose "
         "metric the source does not state; '-' = not published. " + ref.get("source_note", ""), "",
         "| model | params | training tokens | " + " | ".join(n for _, n in cols) + " | LAMBADA ppl | source |", "|---|---|---|" + "---|" * (len(cols) + 2)]
    mine = ["**" + res["name"] + " (this run)**", f"{res['checkpoint']['n_params'] / 1e6:.0f} M", "-"]
    for t, _ in cols:
        r = res["tasks"].get(t)
        mine.append(f"**{r['headline']:.1f}**" if r and r["status"] == "ok" else "-")
    lam = res["tasks"].get("lambada_openai")
    mine.append(f"**{lam['perplexity']:.1f}**" if lam and lam["status"] == "ok" else "-")
    mine.append("this run")
    L.append("| " + " | ".join(mine) + " |")
    heads = {"hellaswag": "acc_norm", "arc_easy": "acc", "arc_challenge": "acc_norm", "piqa": "acc", "winogrande": "acc", "openbookqa": "acc_norm", "sciq": "acc", "lambada_openai": "acc", "mmlu_cloze": "acc_norm"}
    for m in ref["models"]:
        row = [m["name"], f"{m['params_m']:g} M" if m.get("params_m") else "-", f"{m['train_tokens_b']:g} B" if m.get("train_tokens_b") else "-"]
        for t, _ in cols:
            s = (m.get("scores") or {}).get(t) or {}
            h = heads[t]
            if h in s:
                row.append(f"{s[h]:.1f}")
            elif "x" in s:
                row.append(f"{s['x']:.1f}*")
            else:
                alt = [k for k in ("acc_norm", "acc") if k in s]
                row.append(f"({s[alt[0]]:.1f})" if alt else "-")
        lam_s = (m.get("scores") or {}).get("lambada_openai") or {}
        row.append(f"{lam_s['ppl']:.1f}" if "ppl" in lam_s else "-")
        row.append(m.get("source_short", ""))
        L.append("| " + " | ".join(row) + " |")
    L.append("")
    for k, v in (ref.get("legend") or {}).items():
        L.append(f"- **{k}**: {v}")
    L.append("")
    return L


def load_ref(arg, data):
    here = os.path.dirname(os.path.abspath(os.environ.get("F1_EVAL_SELF", sys.argv[0])))
    for c in [arg, os.path.join(here, "f1_eval_reference.json"), os.path.join(data, "reference_scores.json")]:
        if c and os.path.isfile(c):
            try:
                return json.load(open(c, encoding="utf-8")), c
            except Exception as e:  # noqa: BLE001
                warn(f"reference file {c} unreadable: {e}")
    return None, None


def last_val_loss(ckpt, step):
    """the trainer's own validation loss at (or just before) the checkpoint step, from train.log / log.jsonl next to the checkpoint"""
    d = os.path.dirname(os.path.abspath(ckpt))
    best = None
    p = os.path.join(d, "log.jsonl")
    if os.path.isfile(p):
        for line in open(p, encoding="utf-8", errors="replace"):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if "val_loss" in r and r.get("step", 1 << 60) <= step:
                if best is None or r["step"] >= best["step"]:
                    best = {"step": r["step"], "value": float(r["val_loss"]), "file": "log.jsonl"}
    p = os.path.join(d, "train.log")
    if best is None and os.path.isfile(p):
        for line in open(p, encoding="utf-8", errors="replace"):
            m = re.match(r"step\s+(\d+)/\d+ .* val ([\d.]+)\s*$", line)
            if m and int(m.group(1)) <= step and (best is None or int(m.group(1)) >= best["step"]):
                best = {"step": int(m.group(1)), "value": float(m.group(2)), "file": "train.log"}
    return best


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(prog="f1_eval.sh", description="Score a Forge-1 checkpoint on the whole evaluation ladder and write results.json + results.md.", epilog=EPILOG, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", help="forgeckpt1 file (ckpt_latest.bin or ckpt_<step>.bin of either trainer)")
    ap.add_argument("--out", help="results directory (default: ./f1_eval_results/<name>_<time>)")
    ap.add_argument("--data", default=os.environ.get("F1_EVAL_DATA", DEFAULT_DATA), help=f"directory made by tools/f1_eval_data.py (default $F1_EVAL_DATA or {DEFAULT_DATA})")
    ap.add_argument("--forge", help="forge binary (default $FORGE_EVAL_BIN, ./forge, ~/forgeinfer/target-dev3/release/forge, ./target/release/forge, the v1 binary, forge on the PATH)")
    ap.add_argument("--tokenizer", help="tokenizer_f1.json (default DATA/tokenizer_f1.json or ~/forgeinfer/data/tokenizer_f1.json)")
    ap.add_argument("--tasks", default="all", help="comma separated task names or groups: all everything core lambada mmlu ppl hellaswag arc_easy arc_challenge piqa winogrande openbookqa sciq lambada_openai mmlu_cloze mmlu_letter")
    ap.add_argument("--quick", action="store_true", help="subsample every task (per-task sizes in the manifest) and use fewer perplexity windows")
    ap.add_argument("--limit", type=int, help="examples per task, overriding the mode (the files are shuffled, so the first N are a random subsample)")
    ap.add_argument("--mmlu-full", action="store_true", help="score all MMLU questions (about 13.9 k) instead of the 2000-question default")
    ap.add_argument("--forward-tokens", type=int, default=4096, help="padded tokens per forward pass (batch x seq) of eval-mc / eval-hellaswag / eval-ppl; a power of two, at least 2048 or exactly a "
                    "multiple of 2048 (the trainer's cross-entropy chunk; default 4096)")
    ap.add_argument("--ppl-windows", type=int, help="windows per domain (default 256, quick 64)")
    ap.add_argument("--ppl-seq", type=int, default=512, help="window length of eval-ppl (default 512)")
    ap.add_argument("--tight", action="store_true", help="little GPU memory (the trainer is running on the same 8 GB card): 1024 padded tokens per forward pass")
    ap.add_argument("--name", help="label of the run in the tables (default: <checkpoint folder>/<file> step N)")
    ap.add_argument("--dry-run", action="store_true", help="validate everything and print the commands; run nothing (works without a GPU)")
    ap.add_argument("--verify", action="store_true", help="also check the sha256 of every selected data file against the manifest")
    ap.add_argument("--resume", action="store_true", help="keep the tasks that already have status ok in OUT/results.json and run only the rest")
    ap.add_argument("--timeout", type=int, default=2400, help="seconds per forge command before it is killed (default 2400)")
    ap.add_argument("--ref", help="reference-scores JSON for the comparison table (default tools/f1_eval_reference.json next to this script)")
    ap.add_argument("--previous", help="an earlier results.json: the benchmark table gets a delta column")
    ap.add_argument("--assume-tps", type=float, default=50000.0, help="padded tokens per second assumed for the time estimate of --dry-run (default 50000; measured values land in results.json)")
    a = ap.parse_args()

    # ---- checkpoint
    ckpt = os.path.abspath(os.path.expanduser(a.checkpoint))
    if not os.path.isfile(ckpt):
        die(f"checkpoint not found: {ckpt}")
    header, ck_bytes = read_header(ckpt)
    cfg = header.get("cfg", {})
    ok(f"checkpoint {ckpt}: step {header.get('step')}, {header.get('n_params')} parameters, {cfg.get('n_layer')} layers x d {cfg.get('n_embd')}, vocabulary {cfg.get('vocab')}, {ck_bytes / 1e6:.0f} MB")
    # ---- data + manifest
    data = os.path.abspath(os.path.expanduser(a.data))
    man_path = os.path.join(data, "MANIFEST.json")
    if not os.path.isfile(man_path):
        die(f"no MANIFEST.json in {data}. Build the data with `python3 tools/f1_eval_data.py --out {data}` (homelab venv) or pass --data DIR of a copy of /mnt/data/forgeinfer-data/eval")
    manifest = json.load(open(man_path, encoding="utf-8"))
    tok_path = find_tokenizer(a.tokenizer, data)
    tok_sha = sha256_file(tok_path)
    other_tok = bool(manifest.get("tokenizer", {}).get("sha256")) and manifest["tokenizer"]["sha256"] != tok_sha
    ok(f"tokenizer {tok_path} (sha256 {tok_sha[:12]}), data {data} (manifest of {manifest.get('updated')}, seed {manifest.get('seed')})")
    if int(cfg.get("vocab", 0)) != int(manifest.get("tokenizer", {}).get("vocab", cfg.get("vocab", 0))):
        warn(f"the checkpoint vocabulary ({cfg.get('vocab')}) differs from the tokenizer's ({manifest['tokenizer']['vocab']}): scores would be meaningless")
    # ---- tasks
    wanted = []
    for part in [x.strip() for x in a.tasks.split(",") if x.strip()]:
        wanted += GROUPS.get(part, [part])
    wanted = [t for t in dict.fromkeys(wanted)]
    for t in wanted:
        if t != "ppl" and t not in manifest["tasks"]:
            die(f"unknown task {t!r}; the ladder is: {', '.join(ALL + ['mmlu_letter'])} (groups: all, everything, core, lambada, mmlu)")
    if "ppl" in wanted and not manifest.get("ppl", {}).get("files"):
        die("the manifest has no per-domain perplexity files (rerun tools/f1_eval_data.py)")
    if other_tok:
        if "ppl" in wanted:
            die(f"the tokenizer {tok_path} (sha256 {tok_sha[:16]}) is not the one the perplexity files were encoded with (sha256 {manifest['tokenizer']['sha256'][:16]}): drop `ppl` from --tasks to score the benchmarks of a model with another tokenizer")
        warn(f"the tokenizer {tok_path} differs from the one the data was prepared with: fine for the text benchmarks, and the 'seq' values of the manifest are only estimates for it")
    # ---- binary
    forge = find_forge(a.forge)
    fsha = sha256_file(forge)
    rc, out = run_quiet([forge, "tokenize", tok_path, "The capital of France is"], 30)
    if rc != 0 or not out.strip().startswith("["):
        die(f"the binary {forge} does not run (`forge tokenize` failed with exit code {rc}): {out.strip()[:300]}")
    ok(f"binary {forge} (sha256 {fsha[:12]}); `forge tokenize` works")
    lam_ok = True
    if "lambada_openai" in wanted:
        rc, out = run_quiet([forge, "eval-lambada"], 30)
        lam_ok = "eval-lambada" in out and "usage:\n  forge run" not in out
        if lam_ok:
            ok("the binary has the eval-lambada subcommand")
        else:
            warn("this binary has no `eval-lambada` subcommand: LAMBADA will be reported as unavailable. Apply tools/rust_patches/eval_lambada (README there) and rebuild the dev binary")
    # ---- GPU
    gpu = gpu_info()
    if a.dry_run:
        print("[info]  dry run: the GPU is not needed" + (f" (found: {gpu['name']}, {gpu['memory_total_mib'] - gpu['memory_used_mib']} MiB free)" if gpu else " (none found)"))
    elif gpu is None:
        die("no GPU: nvidia-smi is missing or reports nothing. The evaluators run on the GPU (use --dry-run to check the setup without one)")
    else:
        free = gpu["memory_total_mib"] - gpu["memory_used_mib"]
        ok(f"GPU {gpu['name']}: {free} MiB free of {gpu['memory_total_mib']} MiB, {gpu['utilization_pct']} % busy")
        if gpu["utilization_pct"] > 50 and not a.tight:
            warn("the GPU is busy (the trainer?): consider --tight (small batches) or run inside a GPU window (tools/pause_training.sh); timings are not representative while it is shared")
        if free < 1500:
            warn(f"only {free} MiB of GPU memory are free: use --tight, or the evaluators will fail with an out-of-memory error")
    # ---- plan
    mode = "quick" if a.quick else "full"
    fwd = 1024 if a.tight else a.forward_tokens
    if fwd < 256 or (fwd > 2048 and fwd % 2048) or fwd & (fwd - 1):
        die(f"--forward-tokens {fwd}: must be a power of two, between 256 and 2048 or a multiple of 2048 (4096, 8192, ...); forge's cross-entropy needs batch x seq <= 2048 or a multiple of 2048")
    hs_batch = max(4, fwd // 256)
    if a.ppl_seq % 64:
        die(f"--ppl-seq {a.ppl_seq}: must be a multiple of 64 (try 256, 512, 1024)")
    ppl_batch = pick_batch(a.ppl_seq, fwd)
    ppl_windows = a.ppl_windows or (64 if a.quick else 256)
    plan = []  # (task, limit, seq, batch, command list, padded tokens)
    for t in [x for x in RUN_ORDER if x in wanted]:
        if t == "ppl":
            files = [os.path.join(data, v["file"]) for d, v in manifest["ppl"]["files"].items()]
            plan.append(("ppl", None, a.ppl_seq, ppl_batch, [forge, "eval-ppl", ckpt, "--data"] + files + ["--windows", str(ppl_windows), "--seq", str(a.ppl_seq), "--batch", str(ppl_batch)],
                         ppl_windows * (a.ppl_seq + 1) * len(files)))
            continue
        m = manifest["tasks"][t]
        limit = a.limit if a.limit else (m["quick"] if a.quick else (m["examples"] if (t not in ("mmlu_cloze", "mmlu_letter") or a.mmlu_full) else m.get("full_limit", m["examples"])))
        limit = min(limit, m["examples"])
        f = os.path.join(data, m["file"])
        if m["kind"] == "hellaswag":
            cmd = [forge, "eval-hellaswag", ckpt, "--tokenizer", tok_path, "--data", f, "--limit", str(limit), "--batch", str(hs_batch)]
            seq, b, padded = 256, hs_batch, limit * m["choices"] * 257
        elif m["kind"] == "lambada":
            cmd = [forge, "eval-lambada", ckpt, "--tokenizer", tok_path, "--data", f, "--limit", str(limit), "--seq", str(m["seq"])]
            seq, b, padded = m["seq"], 1, limit * 3 * 128
        else:
            if m["seq"] % 64:
                die(f"{t}: seq {m['seq']} in the manifest is not a multiple of 64 (rebuild the data with the current tools/f1_eval_data.py)")
            mc_batch = pick_batch(m["seq"], fwd)
            cmd = [forge, "eval-mc", ckpt, "--tokenizer", tok_path, "--data", f, "--limit", str(limit), "--seq", str(m["seq"]), "--batch", str(mc_batch)]
            seq, b, padded = m["seq"], mc_batch, limit * m["choices"] * (m["seq"] + 1)
        plan.append((t, limit, seq, b, cmd, padded))
    # ---- data files
    problems = 0
    for t, limit, seq, b, cmd, padded in plan:
        if t == "ppl":
            for d, v in manifest["ppl"]["files"].items():
                p = os.path.join(data, v["file"])
                if not os.path.isfile(p) or os.path.getsize(p) != v["bytes"]:
                    print(f"[ERROR] ppl file {p} missing or of the wrong size (manifest: {v['bytes']} bytes)")
                    problems += 1
                elif a.verify and sha256_file(p) != v["sha256"]:
                    print(f"[ERROR] ppl file {p}: sha256 does not match the manifest")
                    problems += 1
            continue
        m = manifest["tasks"][t]
        p = os.path.join(data, m["file"])
        if not os.path.isfile(p) or os.path.getsize(p) != m["bytes"]:
            print(f"[ERROR] {t}: {p} missing or of the wrong size (manifest: {m['bytes']} bytes)")
            problems += 1
        elif a.verify and sha256_file(p) != m["sha256"]:
            print(f"[ERROR] {t}: {p}: sha256 does not match the manifest")
            problems += 1
    if problems:
        die(f"{problems} data file problem(s); rebuild with tools/f1_eval_data.py or copy the complete data directory")
    ok(f"{len(plan)} task group(s) selected, data files present" + (" and checksums verified" if a.verify else " (sizes checked; --verify checks the sha256)"))
    est = sum(p[5] for p in plan) / a.assume_tps
    name = a.name or f"{os.path.basename(os.path.dirname(ckpt))}/{os.path.basename(ckpt)} step {header.get('step')}"
    out_dir = os.path.abspath(a.out) if a.out else os.path.abspath(os.path.join("f1_eval_results", re.sub(r"[^A-Za-z0-9_.-]+", "_", name) + time.strftime("_%Y%m%d_%H%M%S")))
    print(f"[info]  mode {mode}; results go to {out_dir}; estimated GPU time {est / 60:.1f} min at an assumed {a.assume_tps:.0f} padded tokens/s (the first run measures the real speed)")
    print("")
    print("Commands:")
    for t, limit, seq, b, cmd, padded in plan:
        print(f"  [{t}] " + " ".join(cmd))
    if "lambada_openai" in wanted and not lam_ok:
        print("  [lambada_openai] SKIPPED: the binary has no eval-lambada")
    if a.dry_run:
        print("\ndry run finished: nothing was executed")
        return 0

    # ---- run
    os.makedirs(os.path.join(out_dir, "logs"), exist_ok=True)
    prev = None
    res_path = os.path.join(out_dir, "results.json")
    if a.resume and os.path.isfile(res_path):
        prev = json.load(open(res_path, encoding="utf-8"))
        print(f"[info]  --resume: keeping {sum(1 for r in prev['tasks'].values() if r['status'] == 'ok')} finished task(s) of {res_path}")
    ref, ref_path = load_ref(a.ref, data)
    table = byte_table(tok_path)
    res = {"schema": 1, "name": name, "created": time.strftime("%Y-%m-%d %H:%M:%S"), "mode": mode, "checkpoint": {"path": ckpt, "bytes": ck_bytes, "step": header.get("step"), "n_params": header.get("n_params"), "cfg": cfg},
           "env": {"host": socket.gethostname(), "gpu": (gpu or {}).get("name"), "forge": {"path": forge, "sha256": fsha}, "python": platform.python_version(), "data": data,
                   "manifest_updated": manifest.get("updated"), "manifest_seed": manifest.get("seed"), "tokenizer_sha256": tok_sha, "args": " ".join(sys.argv[1:])},
           "tasks": (prev or {}).get("tasks", {}), "ppl": (prev or {}).get("ppl"), "val_loss_from_log": last_val_loss(ckpt, int(header.get("step", 0)))}
    if a.previous and os.path.isfile(a.previous):
        pj = json.load(open(a.previous, encoding="utf-8"))
        res["previous"] = {t: r["headline"] for t, r in pj.get("tasks", {}).items() if r.get("status") == "ok"}
    t_start = time.time()
    failures = 0
    unavailable = 0

    def save():
        res["seconds_total"] = time.time() - t_start
        ok_tasks = [t for t in CORE if res["tasks"].get(t, {}).get("status") == "ok"]
        res["averages"] = {"core_tasks": ok_tasks, "core_headline": (sum(res["tasks"][t]["headline"] for t in ok_tasks) / len(ok_tasks)) if ok_tasks else None,
                           "core_chance": (sum(res["tasks"][t]["chance"] for t in ok_tasks) / len(ok_tasks)) if ok_tasks else None}
        with open(res_path, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=1)
        with open(os.path.join(out_dir, "results.md"), "w", encoding="utf-8") as f:
            f.write(render_markdown(res, ref))

    try:
        for t, limit, seq, b, cmd, padded in plan:
            if t == "ppl":
                if prev and (prev.get("ppl") or {}).get("domains") and prev["ppl"].get("status") == "ok" and prev["ppl"].get("windows") == ppl_windows and prev["ppl"].get("seq") == a.ppl_seq:
                    continue
                print(f"\n== ppl: {len(PPL_DOMAINS)} domains, {ppl_windows} windows x {a.ppl_seq} tokens")
                rc, text, sec = run_command(cmd, os.path.join(out_dir, "logs", "ppl.log"), a.timeout)
                dom = {}
                for m in RE_PPL.finditer(text):
                    d = {"nats": float(m["nats"]), "ppl": float(m["ppl"]), "count": int(m["count"])}
                    path = os.path.join(data, manifest["ppl"]["files"][m["name"]]["file"]) if m["name"] in manifest["ppl"]["files"] else None
                    if path and os.path.isfile(path):
                        nbytes, ntok = window_bytes(path, table, ppl_windows, a.ppl_seq)
                        if nbytes and ntok:
                            d["bytes_per_token"] = nbytes / ntok
                            d["bpb"] = d["nats"] / math.log(2) / d["bytes_per_token"]
                    dom[m["name"]] = d
                status = "ok" if rc == 0 and len(dom) == len(manifest["ppl"]["files"]) else f"failed (exit {rc}, {len(dom)} of {len(manifest['ppl']['files'])} domains parsed)"
                res["ppl"] = {"status": status, "windows": ppl_windows, "seq": a.ppl_seq, "batch": b, "seconds": sec, "command": " ".join(cmd), "domains": dom,
                              "mean_nats": sum(v["nats"] for v in dom.values()) / len(dom) if dom else None,
                              "mean_bpb": (sum(v["bpb"] for v in dom.values() if "bpb" in v) / max(1, sum(1 for v in dom.values() if "bpb" in v))) if dom else None}
                if status != "ok":
                    failures += 1
                    res["ppl"]["error"] = text.strip().splitlines()[-1][:300] if text.strip() else "no output"
                save()
                continue
            m = manifest["tasks"][t]
            if prev and prev["tasks"].get(t, {}).get("status") == "ok" and prev["tasks"][t].get("limit") == limit:
                continue
            r = {"title": m["title"], "status": "ok", "command": " ".join(cmd), "limit": limit, "seq": seq, "batch": b, "chance": m["chance"], "headline_metric": m["headline"], "subsample": limit < m["examples"], "dropped_long": m.get("dropped_long", 0)}
            if t == "lambada_openai" and not lam_ok:
                r.update(status="unavailable: the binary has no eval-lambada subcommand (apply tools/rust_patches/eval_lambada)", headline=0.0)
                res["tasks"][t] = r
                unavailable += 1
                save()
                continue
            print(f"\n== {t}: {limit} examples (seq {seq}, batch {b})")
            rc, text, sec = run_command(cmd, os.path.join(out_dir, "logs", f"{t}.log"), a.timeout)
            mm = RE_HS.search(text) if m["kind"] == "hellaswag" else (RE_LAM.search(text) if m["kind"] == "lambada" else RE_MC.search(text))
            if rc != 0 or not mm:
                tail = [ln for ln in text.strip().splitlines() if ln.strip()][-3:]
                r.update(status=f"failed (exit {rc})", error=" / ".join(tail)[:400], headline=0.0)
                failures += 1
            else:
                n = int(mm["n"])
                r.update(n=n, seconds=sec, padded_tokens=padded, padded_tokens_per_s=padded / sec if sec > 0 else None)
                if m["kind"] == "lambada":
                    r.update(acc=float(mm["acc"]), perplexity=float(mm["ppl"]), nats_per_word=float(mm["nats"]), target_tokens_per_word=float(mm["tpw"]), headline=float(mm["acc"]))
                    sc = RE_SELF.search(text)
                    if sc:
                        r["selfcheck_max_diff_nats"] = float(sc["diff"])
                        if float(sc["diff"]) > 0.02:
                            r["warning"] = f"selfcheck difference {sc['diff']} nats > 0.02: eval-lambada and token_nll disagree"
                            warn(r["warning"])
                else:
                    r.update(acc=float(mm["acc"]), acc_norm=float(mm["norm"]))
                    if m["kind"] == "hellaswag":
                        r["acc_per_token_mean"] = float(mm["tok"])
                    r["headline"] = r["acc_norm"] if m["headline"] == "acc_norm" else r["acc"]
                    if t == "winogrande":
                        r["acc_norm"] = None  # the harness has no acc_norm for the partial-scoring WinoGrande
                r["headline_se"] = stderr_pct(r["headline"], n)
                print(f"    -> {m['title']}: {r['headline_metric']} {r['headline']:.2f} % +- {r['headline_se']:.2f}   ({sec:.0f} s)")
            res["tasks"][t] = r
            save()
    except KeyboardInterrupt:
        print("\ninterrupted: partial results are saved")
        save()
        return 1
    save()
    print("\n" + "=" * 100)
    print(render_markdown(res, ref))
    print(f"results: {res_path}\n         {os.path.join(out_dir, 'results.md')}")
    if unavailable:
        warn(f"{unavailable} rung(s) unavailable with this binary (LAMBADA needs the eval-lambada subcommand); the rest of the ladder was scored")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
PYTHON_PROGRAM_END
export F1_EVAL_SELF="$SELF"
export PYTHONUNBUFFERED=1
"$PY" "$TMP" "$@"
exit $?
