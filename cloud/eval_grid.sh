#!/bin/bash
# Evaluate a grid of fine-tunes on Modal in parallel: the 20-prompt chat checklist (automatic score out of 40) and the benchmark ladder (forgetting against the base), a few cents each.
#   bash cloud/eval_grid.sh PREFIX LR [LR ...]          example: bash cloud/eval_grid.sh sft_tpu1_human_2250 2e-4 5e-4 1e-3
# Needs the runs /runs/PREFIX_LR/ckpt_latest.bin in the forge-runs volume (made by cloud/modal_grid.sh). Output: scratch/eval_PREFIX_LR.log, scratch/checklist_PREFIX_LR_chk.txt (answers), scratch/ladder_PREFIX_LR.json,
# then a comparison table (tools/ladder_table.py) and the checklist scores (tools/sft_data/score_checklist.py).
PREFIX="${1:?run name prefix, e.g. sft_tpu1_human_2250}"; shift
LRS="${*:?at least one learning rate}"
export MSYS_NO_PATHCONV=1 PYTHONUTF8=1 PYTHONIOENCODING=utf-8
MODAL=<dev>/research/neural-ai-from-scratch-20260919/reasoner5b/cloud/.venv/Scripts/modal.exe
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO" || exit 1
mkdir -p scratch
for LR in $LRS; do
  N="${PREFIX}_${LR}"
  ( "$MODAL" run cloud/modal_forge.py::checklist --name "$N" --label chk --approved-usd 0.6 > "scratch/eval_${N}_chk.log" 2>&1 ) &
  ( "$MODAL" run cloud/modal_forge.py::ladder --name "$N" --label lad --approved-usd 1 > "scratch/eval_${N}_lad.log" 2>&1 ) &
done
wait
ARGS=()
FILES=()
for LR in $LRS; do
  N="${PREFIX}_${LR}"
  "$MODAL" volume get forge-runs "/$N/ladder_lad/results.json" "scratch/ladder_${N}.json" --force > /dev/null 2>&1
  ARGS+=("lr ${LR}=scratch/ladder_${N}.json")
  [ -f "scratch/checklist_${N}_chk.txt" ] && FILES+=("scratch/checklist_${N}_chk.txt")
done
echo "=== benchmark ladder (quick mode)"
python -I -X utf8 tools/ladder_table.py "${ARGS[@]}" --domains
echo "=== chat checklist (automatic score; read the answers too)"
[ ${#FILES[@]} -gt 0 ] && python -I -X utf8 tools/sft_data/score_checklist.py "${FILES[@]}"
echo "eval grid finished $(date '+%I:%M %p')"
