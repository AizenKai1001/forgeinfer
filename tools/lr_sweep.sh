#!/bin/bash
# Learning-rate sweep on the ~31M proxy model (12 layers, d 384, 6/2 heads, ffn 1024, vocab 32768, seq 2048), run on the homelab, one run after the other.
#   tools/lr_sweep.sh 1e-3 2e-3 4e-3
# Each run: 500 steps of 65,536 tokens with warmup 30 and a linear decay over the last 20 %; prints the final validation loss. Logs in ~/forgeinfer/runs/sweep_<lr>/.
set -u
FORGE=~/forgeinfer/src/target/release/forge
DATA=~/forgeinfer/data/tokens
for lr in "$@"; do
  out=~/forgeinfer/runs/sweep_$lr
  echo "== lr $lr"
  nice -n 10 "$FORGE" train --data "$DATA" --out "$out" --layers 12 --dim 384 --heads 6 --kv 2 --ffn 1024 --seq 2048 --micro 4 --tokens-per-step 65536 --steps ${STEPS:-500} \
    --lr "$lr" --warmup 30 --eval-every 100 --eval-batches 8 --save-every 100000 --qk-norm > "$out.log" 2>&1
  grep " val " "$out.log" | tail -n 3
  tail -n 1 "$out.log"
done
