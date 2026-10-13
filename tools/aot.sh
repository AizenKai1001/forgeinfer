#!/bin/bash
# Deviceless TPU compile lab on the homelab: what does the TPU compiler make of the CURRENT work tree's training step, without a TPU?
#   tools/aot.sh LAB [aot_compile.py arguments]        e.g.   tools/aot.sh exp1 --only base,remat-attn          tools/aot.sh exp2 --only base --args "<model arguments>"
# Copies tpu/forge_jax.py, bench_step.py, aot_compile.py and hlo_cost.py of the work tree you run it from (a worktree of a helper included) to ~/forgeinfer/aot_lab/LAB/ on the homelab and runs
# tpu/aot_compile.py there (nice 15) with the jax[tpu] 0.11.2 + libtpu 0.0.48 virtualenv ~/forgeinfer/tpuc (created once: python3 -m venv ~/forgeinfer/tpuc && ~/forgeinfer/tpuc/bin/pip install 'jax[tpu]==0.11.2').
# Per variant: does it compile for a v5e-8, peak memory per chip against the 15.75 GB of a chip, and the compiler's own estimate of the step (tpu/hlo_cost.py: 0.365 s for the base step, 0.379 s measured on a real
# v5e-8; it does not look inside Pallas kernels). Results stay in ~/forgeinfer/aot_lab/LAB/out/ (aot.json and one *.hlo.txt per variant: fetch with `tools/hl.sh -n 'cat ~/forgeinfer/aot_lab/LAB/out/base.hlo.txt' > f.txt`,
# read with `python tpu/hlo_cost.py f.txt --top 30`). About 40 to 60 s and 4 to 6 GB of RAM per variant; a run of all 14 variants is ~12 minutes (put it in the background; a tool call is capped at 5 minutes).
# The homelab is shared: one lab run at a time per person (look at `uptime` first), and remove a lab when done: tools/hl.sh -n 'rm -r ~/forgeinfer/aot_lab/LAB'.
set -e
LAB="${1:?usage: tools/aot.sh LAB [aot_compile.py arguments]}"
shift
case "$LAB" in *[!A-Za-z0-9_.-]*|"") echo "LAB must be letters, digits, . _ -"; exit 2;; esac
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SSH="ssh -i $HOME/.ssh/id_ed25519 -o BatchMode=yes -o IdentitiesOnly=yes -o ConnectTimeout=10 user@homelab"
tar -C "$ROOT/tpu" -cf - forge_jax.py bench_step.py aot_compile.py hlo_cost.py | $SSH "mkdir -p ~/forgeinfer/aot_lab/$LAB && tar -C ~/forgeinfer/aot_lab/$LAB -xf -"
QARGS=$(printf '%q ' "$@")
$SSH "cd ~/forgeinfer/aot_lab/$LAB && TPU_SKIP_MDS_QUERY=1 ALLOW_MULTIPLE_LIBTPU_LOAD=1 nice -n 15 ~/forgeinfer/tpuc/bin/python aot_compile.py --out out --hlo-dir out $QARGS 2>&1 | grep -v -E 'Failed to find host bounds|InitGoogle|unexpected worker hostname|Source Location Trace|libtpu_init_utils'"
