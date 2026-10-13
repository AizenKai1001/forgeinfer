#!/bin/bash
# Start (or resume) a long Forge-1 pre-training run on the homelab, detached. The model structure comes from F1_ARGS, everything else has the defaults of the F125 run; the data is the whole Forge-1 mixture
# (6.75 B tokens, 7 sources, tokens_f1/zzz_val.u16 holds the validation tail).
#   RUN=f1_v1 F1_ARGS="--relu2 --ffn 2304 --lambdas --lr-scalar-mult 10" tools/launch_f1.sh
# Re-running resumes from ckpt_latest.bin. Checkpoints every SAVE_EVERY (default 100) steps in ~/forgeinfer/runs/$RUN (ckpt_latest.bin, log.jsonl, train.log).
set -u
RUN="${RUN:-f1_v1}"
OUT=~/forgeinfer/runs/$RUN
mkdir -p "$OUT"
RESUME=""
if [ -f "$OUT/ckpt_latest.bin" ]; then RESUME="--resume"; fi
# flags that belong to the run itself and must survive every restart (benchmark windows, binary swaps, branches), for example "--window 512 --window-layers auto": one line in $OUT/extra_args.txt
EXTRA=""
if [ -f "$OUT/extra_args.txt" ]; then EXTRA=$(cat "$OUT/extra_args.txt"); fi
BIN="${FORGE_BIN:-$HOME/forgeinfer/target-dev3/release/forge}"
cd ~/forgeinfer/src
nohup nice -n 5 "$BIN" train --data "${DATA:-/mnt/data/forgeinfer-data/tokens_f1}" --out "$OUT" \
  --layers ${LAYERS:-30} --dim ${DIM:-576} --heads ${HEADS:-9} --kv ${KV:-3} --ffn ${FFN:-1536} --seq 2048 --micro ${MICRO:-2} --tokens-per-step ${TOKENS_PER_STEP:-65536} --steps ${STEPS:-45000} \
  --warmup ${WARMUP:-100} --decay-frac ${DECAY_FRAC:-0.1} --min-lr-frac ${MIN_LR_FRAC:-0.05} --eval-every 100 --eval-batches ${EVAL_BATCHES:-8} --save-every ${SAVE_EVERY:-100} --qk-norm $RESUME \
  --optim muon --lr ${LR:-1.5e-3} --muon-lr ${MUON_LR:-1.5e-3} --muon-wd 0.05 ${F1_ARGS:-} $EXTRA \
  >> "$OUT/train.log" 2>&1 < /dev/null &
echo "started pid $! in $OUT"
