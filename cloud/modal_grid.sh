#!/bin/bash
# Fine-tune grid on Modal: the same base fine-tuned in PARALLEL at several learning rates (one A100 container each, so the wall time is that of one run, the cost the sum).
#   bash cloud/modal_grid.sh BASE_NAME [MIX] [STEPS] [LR ...]        defaults: human, 2250 steps, LRs 3e-4 6e-4 1e-3
#   STRUCT="..." bash cloud/modal_grid.sh tpu1 human 2250 3e-4 6e-4 1e-3      (the structure flags of the base, see cloud/modal_sft.sh; the TPU model has value embeddings in 8 layers)
# Cost at 87 k tokens/s: about 1.3 USD per human-mix run of 2,250 steps, about 2.3 USD per mix2 run of 4,000 steps (the guard of cloud/modal_sft.sh refuses a run whose worst case is above 3.5 USD).
# Logs: scratch/modal_sft_BASE_MIX_STEPS_LR.log ; results: /runs/sft_BASE_MIX_STEPS_LR/ in the forge-runs volume. Afterwards (one call each, about 3 minutes, a few cents):
#   modal run cloud/modal_forge.py::checklist --name sft_BASE_MIX_STEPS_LR --label NAME --approved-usd 0.6      (prints the automatic score out of 40)
#   modal run cloud/modal_forge.py::ladder --name sft_BASE_MIX_STEPS_LR --label NAME --approved-usd 1          (benchmarks; forgetting against the base)
BASE="${1:?base folder name in the forge-runs volume}"; shift
MIX="${1:-human}"; [ $# -gt 0 ] && shift
STEPS="${1:-2250}"; [ $# -gt 0 ] && shift
LRS="${*:-3e-4 6e-4 1e-3}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$REPO/scratch"
echo "grid on $BASE: mix $MIX, $STEPS steps, learning rates $LRS ($(date '+%I:%M %p'))"
for LR in $LRS; do
  LOG="$REPO/scratch/modal_sft_${BASE}_${MIX}_${STEPS}_${LR}.log"
  ( date '+start %I:%M %p' > "$LOG"; bash "$REPO/cloud/modal_sft.sh" "$BASE" "$MIX" "$STEPS" "$LR" >> "$LOG" 2>&1; echo "exit $?" >> "$LOG"; date '+end %I:%M %p' >> "$LOG" ) &
done
wait
echo "grid finished $(date '+%I:%M %p'); logs in $REPO/scratch/modal_sft_${BASE}_${MIX}_${STEPS}_*.log"
