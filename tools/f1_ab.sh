#!/bin/bash
# Forge-1 structure A/B on the ~31M proxy (12 layers, d 384, 6/2 heads, ffn 1024, vocab 32768, seq 2048), one run after the other on the homelab GPU. Equal tokens for every arm (500 steps x 65,536 tokens), the same
# data stream for the same seed (window positions depend on the seed only), validation = the tail of the last shard. Each argument is "name|extra forge-train flags"; finished runs (a final 'val' line in the log)
# are skipped, so an interrupted sequence can simply be started again. One line per arm goes to $SUMMARY.
#   FORGE=... DATA=... STEPS=500 tools/f1_ab.sh "base|" "relu2|--relu2 --ffn 1536" "untied|--untied"
#   (run it inside a GPU window: tools/pause_training.sh stops the F125 run after a checkpoint and resumes it afterwards)
set -u
FORGE=${FORGE:-$HOME/forgeinfer/target-dev/release/forge}
DATA=${DATA:-$HOME/forgeinfer/data/tokens_f1_web}
BASE=${BASE:---lr 2e-3 --optim muon --muon-lr 2e-3 --muon-wd 0.05 --qk-norm}
RUNS=${RUNS:-$HOME/forgeinfer/runs}
SUMMARY=${SUMMARY:-$RUNS/f1_ab_summary.txt}
STEPS=${STEPS:-500}
for spec in "$@"; do
  name="${spec%%|*}"
  extra="${spec#*|}"
  out=$RUNS/f1ab_$name
  if grep -q " val " "$out.log" 2>/dev/null && grep -q "step *$STEPS/$STEPS" "$out.log" 2>/dev/null; then
    echo "== $name: already done"
    continue
  fi
  echo "== $name: $extra"
  rm -f "$out/log.jsonl"
  nice -n 10 "$FORGE" train --data "$DATA" --out "$out" --layers 12 --dim 384 --heads 6 --kv 2 --ffn 1024 --seq 2048 --micro 4 --tokens-per-step 65536 --steps $STEPS \
    --warmup ${WARMUP:-30} --eval-every 100 --eval-batches 8 --save-every 100000 $BASE $extra > "$out.log" 2>&1
  rm -f "$out/ckpt_latest.bin"
  v=$(grep " val " "$out.log" | tail -n 1 | sed 's/.*val //')
  v4=$(grep " val " "$out.log" | tail -n 2 | head -n 1 | sed 's/.*val //')
  tps=$(grep "step *$STEPS/$STEPS" "$out.log" | grep -o '[0-9]* tok/s')
  np=$(grep -o '[0-9.]* M parameters' "$out.log" | head -n 1)
  echo "$name | val@$STEPS $v | val@$((STEPS - 100)) $v4 | $tps | $np | $extra" | tee -a "$SUMMARY"
done
