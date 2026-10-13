#!/bin/bash
# Runs on the homelab INSIDE a GPU window of the production run (tools/pause_training.sh stops the trainer after a checkpoint and restarts it afterwards): (1) the evaluation ladder on the newest checkpoint
# (all tasks, quick mode, LAMBADA included), (2) the interleaved GEMM benchmark (tools/bench_gemm.sh) and (3) a whole-step A/B of the new GEMM kernels: 25 steps from the checkpoint's parameters with the
# kernel switches at their defaults and with FORGE_GEMM_E2=0 FORGE_GEMM_W144=0 (the speed of steps 6-25 from log.jsonl). Everything is logged to ~/forgeinfer/eval_results/window_NAME.log.
#   NAME=s10100 bash ~/forgeinfer/src/tools/pause_training.sh bash ~/forgeinfer/src/tools/window_checks.sh      (with the LAUNCH/RUN/FORGE_BIN environment of the production run; see the Run Book)
set -u
NAME="${NAME:-window}"
CKPT="${CKPT:-$HOME/forgeinfer/runs/f1_v1/ckpt_latest.bin}"
EVAL_BIN="${EVAL_BIN:-$HOME/forgeinfer/agents/lead-main/target/release/forge}"
STEP_BIN="${STEP_BIN:-$HOME/forgeinfer/agents/reviewer/bin/forge-fixed}"
BENCH_BIN="${BENCH_BIN:-$HOME/forgeinfer/agents/f16acc/bin/forge}"
OUT="$HOME/forgeinfer/eval_results"
mkdir -p "$OUT"
LOG="$OUT/window_$NAME.log"
{
echo "=== window $NAME $(date '+%Y-%m-%d %I:%M:%S %p'), checkpoint $CKPT"
echo "--- 1. evaluation ladder ($EVAL_BIN)"
(cd "$HOME/forgeinfer/src" && bash tools/f1_eval.sh "$CKPT" --quick --forge "$EVAL_BIN" --tasks all --name "$NAME" --out "$OUT") 2>&1 | tail -n 60
echo "--- 2. GEMM benchmark ($BENCH_BIN)"
bash "$HOME/forgeinfer/src/tools/bench_gemm.sh" "$BENCH_BIN" 2>&1 | tail -n 80
echo "--- 3. whole-step A/B ($STEP_BIN), 25 steps each from the parameters of the checkpoint"
for variant in default old; do
  d="$HOME/forgeinfer/agents/window_step_$variant"
  mkdir -p "$d"
  [ -e "$d/log.jsonl" ] && rm "$d/log.jsonl"
  if [ "$variant" = old ]; then export FORGE_GEMM_E2=0 FORGE_GEMM_W144=0; else unset FORGE_GEMM_E2 FORGE_GEMM_W144; fi
  timeout 280 "$STEP_BIN" train --data /mnt/data/forgeinfer-data/tokens_f1 --out "$d" --init-from "$CKPT" --layers 30 --dim 576 --heads 9 --kv 3 --ffn 2304 --seq 2048 --micro 2 --tokens-per-step 65536 --steps 25 --warmup 5 \
    --eval-every 100000 --save-every 100000 --qk-norm --optim muon --lr 1.5e-3 --muon-lr 1.5e-3 --muon-wd 0.05 --relu2 --lambdas --lr-scalar-mult 10 --value-emb 0,1,28,29 --lr-table-mult 20 --lr-emb-mult 3 > "$d/train.log" 2>&1
  python3 - "$d/log.jsonl" "$variant" <<'PYEOF'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
t = [r["tokens_per_s"] for r in rows if 6 <= r["step"] <= 25 and "tokens_per_s" in r]
print(f"  variant {sys.argv[2]}: steps {len(rows)}, mean tokens/s of steps 6-25 = {sum(t)/max(len(t),1):.0f} (min {min(t) if t else 0:.0f}, max {max(t) if t else 0:.0f})")
PYEOF
  [ -e "$d/ckpt_latest.bin" ] && rm "$d/ckpt_latest.bin"
done
echo "=== end $(date '+%I:%M:%S %p')"
} > "$LOG" 2>&1
