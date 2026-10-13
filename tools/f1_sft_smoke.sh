#!/bin/bash
# Mechanics of the chat fine-tune on the latest checkpoint of a Forge-1 run: --init-from (parameters only, fresh optimizer, step 0), the ChatML data, a few dozen steps, then a chat with the result. Meant for a GPU window.
#   (the structure flags must be those of the run)
RUN=${RUN:-f1_v1}
F=${FORGE_BIN:-$HOME/forgeinfer/target-dev3/release/forge}   # the dev build: the masked chat data needs its loader (the v1 binary reads bit 15 as part of the token id)
SRC=$HOME/forgeinfer/runs/$RUN/ckpt_latest.bin
OUT=/tmp/sft_smoke_$RUN
mkdir -p $OUT
cp $SRC $OUT/start.bin
nice -n 5 $F train --data ${SFT_DATA:-/mnt/data/forgeinfer-data/tokens_sft_masked} --out $OUT/run --init-from $OUT/start.bin --layers 30 --dim 576 --heads 9 --kv 3 --ffn 2304 --seq 2048 --micro 2 --tokens-per-step 32768 --steps ${SFT_STEPS:-40} \
  --warmup 5 --eval-every 10 --eval-batches 4 --save-every ${SFT_STEPS:-40} --qk-norm --optim muon --lr 3e-4 --muon-lr 3e-4 --muon-wd 0.01 --val-tokens 4000000 ${F1_ARGS:-} > $OUT/train.log 2>&1
grep -E "starting from|^step +(1|10|20|30|40)/" $OUT/train.log | cut -c1-150
nice -n 5 $F chat $OUT/run/ckpt_latest.bin --tokenizer $HOME/forgeinfer/data/tokenizer_f1.json --chat "Hello! Please tell me a fun fact about the ocean." --tokens 40 --temp 0.7 --seq 256 2>&1 | tail -n 3 | cut -c1-300
