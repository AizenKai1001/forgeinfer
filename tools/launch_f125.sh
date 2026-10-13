#!/bin/bash
# Start (or resume) the long F125 pre-training run on the homelab, detached, with the optimizer and learning rates given in the environment.
#   RUN=f125_muon_v1 OPTIM_ARGS="--optim muon --lr 2e-3 --muon-lr 4e-3" tools/launch_f125.sh
# Defaults: 30 layers, d 576, 9/3 heads, ffn 1536, seq 2048, micro-batch 2, 65,536 tokens per optimizer step, qk-norm, WSD with warmup 100 (decay only when --steps is reached: the run is meant
# to be stopped from a checkpoint and cooled down in a separate short run), checkpoints every 100 steps in ~/forgeinfer/runs/$RUN (ckpt_latest.bin, log.jsonl, train.log). Re-running resumes.
set -u
RUN="${RUN:-f125_v1}"
OUT=~/forgeinfer/runs/$RUN
mkdir -p "$OUT"
RESUME=""
if [ -f "$OUT/ckpt_latest.bin" ]; then RESUME="--resume"; fi
cd ~/forgeinfer/src
nohup nice -n 5 ./target/release/forge train --data ~/forgeinfer/data/tokens --out "$OUT" \
  --layers 30 --dim 576 --heads 9 --kv 3 --ffn 1536 --seq 2048 --micro 2 --tokens-per-step ${TOKENS_PER_STEP:-65536} --steps ${STEPS:-40000} \
  --warmup ${WARMUP:-100} --decay-frac ${DECAY_FRAC:-0.1} --min-lr-frac ${MIN_LR_FRAC:-0.05} --eval-every 100 --eval-batches 8 --save-every ${SAVE_EVERY:-100} --qk-norm $RESUME ${OPTIM_ARGS:-} \
  >> "$OUT/train.log" 2>&1 < /dev/null &
echo "started pid $! in $OUT"
