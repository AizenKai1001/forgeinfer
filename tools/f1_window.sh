#!/bin/bash
# The work of a GPU window for Forge-1 v1 (run it through pause_training.sh, which stops the trainer right after a checkpoint and relaunches it afterwards):
# the tracking evaluation at full batch sizes (domain perplexities, benchmarks, four text samples) and the chat fine-tune smoke test on the masked chat data.
export FULL=1 RUN=f1_v1
echo "== tracking $(date '+%I:%M %p')"
bash ~/forgeinfer/src/tools/f1_track.sh 2>&1 | tail -n 22
echo "== chat fine-tune smoke test $(date '+%I:%M %p')"
# the chat data carries the "not a target" mark in bit 15: only the dev build reads it (FORGE_BIN of the relaunch environment points at the old v1 binary: a masked token id there is out of range)
FORGE_BIN=$HOME/forgeinfer/target-dev3/release/forge SFT_STEPS=40 F1_ARGS="--relu2 --lambdas --lr-scalar-mult 10 --value-emb 0,1,28,29 --lr-table-mult 20 --lr-emb-mult 3" bash ~/forgeinfer/src/tools/f1_sft_smoke.sh 2>&1 | tail -n 14
echo "== window done $(date '+%I:%M %p')"
