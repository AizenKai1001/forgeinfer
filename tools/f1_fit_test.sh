#!/bin/bash
# Does the Forge-1 v1 model fit the 8 GB card, and how fast is it? 15 steps of the 125M shape (30 layers, d 576) with the candidate structures; peak GPU memory from nvidia-smi.
#   bash tools/pause_training.sh bash tools/f1_fit_test.sh          (the F125 run holds 5.8 GB: the test needs the whole card)
F=${FORGE_BIN:-$HOME/forgeinfer/target-dev3/release/forge}
D=${DATA:-/mnt/data/forgeinfer-data/tokens_f1}
BASE="--layers 30 --dim 576 --heads 9 --kv 3 --seq 2048 --micro ${MICRO:-2} --tokens-per-step 65536 --steps 15 --warmup 5 --qk-norm --optim muon --lr 1.5e-3 --muon-lr 1.5e-3 --muon-wd 0.05 --eval-every 100000 --save-every 100000 --val-tokens 2000000"
run() {
  name=$1
  shift
  (while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >> /tmp/f1_fit_mem_$name.txt; sleep 2; done) &
  mon=$!
  nice -n 5 $F train --data $D --out /tmp/f1_fit_$name $BASE "$@" > /tmp/f1_fit_$name.log 2>&1
  kill $mon 2>/dev/null
  echo "== $name: $* -> $(grep -c '^step' /tmp/f1_fit_$name.log) steps, $(grep '^step' /tmp/f1_fit_$name.log | tail -n 1 | grep -o '[0-9]* tok/s'), peak $(sort -n /tmp/f1_fit_mem_$name.txt | tail -n 1) MiB; $(grep -i 'error\|out of memory' /tmp/f1_fit_$name.log | head -n 1)"
}
run plain --ffn 1536
run relu2_lambdas --relu2 --ffn 2304 --lambdas --lr-scalar-mult 10
run full --relu2 --ffn 2304 --lambdas --lr-scalar-mult 10 --value-emb 0,1,28,29 --lr-table-mult 5 --ngram 2,3 --ngram-bits 17 --ngram-dim 128
