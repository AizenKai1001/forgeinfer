#!/bin/bash
# Proxy experiments on the ~31M model (12 layers, d 384, 6/2 heads, ffn 1024, vocab 32768, seq 2048), one after the other, on the homelab GPU.
#   STEPS=500 tools/ab_sweep.sh "adamw2e-3|--lr 2e-3" "muon4e-3|--lr 2e-3 --optim muon --muon-lr 4e-3 --muon-wd 0.05"
# Each argument is "name|extra forge-train flags". Every run: 65,536 tokens per step, warmup 30, linear decay over the last 20 %; logs in ~/forgeinfer/runs/ab_<name>/ and ab_<name>.log.
set -u
FORGE=~/forgeinfer/src/target/release/forge
DATA=~/forgeinfer/data/tokens
for spec in "$@"; do
  name="${spec%%|*}"
  extra="${spec#*|}"
  out=~/forgeinfer/runs/ab_$name
  echo "== $name: $extra"
  nice -n 10 "$FORGE" train --data "$DATA" --out "$out" --layers 12 --dim 384 --heads 6 --kv 2 --ffn 1024 --seq 2048 --micro 4 --tokens-per-step 65536 --steps ${STEPS:-500} \
    --warmup ${WARMUP:-30} --eval-every 100 --eval-batches 8 --save-every 100000 --qk-norm $extra > "$out.log" 2>&1
  grep " val " "$out.log" | tail -n 5
  tail -n 1 "$out.log"
done
