#!/bin/bash
# Chat fine-tune of a Forge-1 checkpoint on the SFT mix v2 (/mnt/data/forgeinfer-data/tokens_sft_mix2; build, filters and weaknesses: research note "ForgeInfer - SFT Data Mix").
# Needs the GPU (a window of pause_training.sh, a cloud GPU or an idle card): 14.4 k tokens/s for the 125M model on the RTX 2060 SUPER means about 2.7 h per epoch of the mix.
#   bash tools/f1_sft_mix2.sh                       default: 2 epochs (4000 steps of 65,536 tokens), peak learning rate 3e-4, 12 % pre-training replay, checkpoint copies every 500 steps
#   SFT_STEPS=300 SFT_LR=1.5e-4 OUT=... bash tools/f1_sft_mix2.sh       a short learning-rate probe (compare the val lines and the tracking evaluation)
# Variables: SFT_WARMUP (default 100; use 20 for a 300-step probe), CKPT (start checkpoint, default the latest of RUN), RUN, OUT (run directory), SFT_DATA (data directory), SFT_STEPS, SFT_LR, SFT_TPS (tokens per step), F1_STRUCT (the structure flags of the base run), FORGE_BIN.
# The structure flags MUST be those of the base run (a checkpoint only loads into the same structure); the learning-rate multipliers are the ones of the pre-training run (as in tools/f1_sft_smoke.sh).
RUN=${RUN:-f1_v1}
CKPT=${CKPT:-$HOME/forgeinfer/runs/$RUN/ckpt_latest.bin}
F=${FORGE_BIN:-$HOME/forgeinfer/target-dev3/release/forge}   # the dev build: the masked data needs its loader (the v1 binary reads bit 15 as part of the token id)
OUT=${OUT:-$HOME/forgeinfer/runs/sft_mix2_$RUN}
DATA=${SFT_DATA:-/mnt/data/forgeinfer-data/tokens_sft_mix2_replay}
STEPS=${SFT_STEPS:-4000}
LR=${SFT_LR:-3e-4}
TPS=${SFT_TPS:-65536}
STRUCT=${F1_STRUCT:---layers 30 --dim 576 --heads 9 --kv 3 --ffn 2304 --qk-norm --relu2 --lambdas --lr-scalar-mult 10 --value-emb 0,1,28,29 --lr-table-mult 20 --lr-emb-mult 3}
[ -f "$HOME/forgeinfer/runs/$RUN/extra_args.txt" ] && STRUCT="$STRUCT $(cat "$HOME/forgeinfer/runs/$RUN/extra_args.txt")"   # a base trained with window attention is fine-tuned with it (and a dense base without)
TOK=${TOK:-$HOME/forgeinfer/data/tokenizer_f1.json}
mkdir -p "$OUT"
cp "$CKPT" "$OUT/start.bin"
echo "start $(date '+%I:%M %p'): $CKPT -> $OUT, data $DATA, $STEPS steps x $TPS tokens, lr $LR" | tee -a "$OUT/notes.txt"
# --val-tokens 4000000: any value >= T (about 2.0 M, see MANIFEST.txt) gives exactly the held-out region of zzz_val.u16 (the file is 4 x T long: the trainer reserves min(--val-tokens, tokens / 4) of its tail)
nice -n 5 "$F" train --data "$DATA" --out "$OUT/run" --init-from "$OUT/start.bin" $STRUCT --seq 2048 --micro 2 --tokens-per-step "$TPS" --steps "$STEPS" \
  --warmup ${SFT_WARMUP:-100} --decay-frac 0.85 --min-lr-frac 0.05 --optim muon --lr "$LR" --muon-lr "$LR" --muon-wd 0.01 --val-tokens 4000000 --eval-every 250 --eval-batches 48 --save-every 500 --seed 1 > "$OUT/train.log" 2>&1 &
PID=$!
# the trainer overwrites ckpt_latest.bin at every save: keep a copy of every save (atomic checkpoints, so a copy after the log line is complete)
last=0
while kill -0 $PID 2>/dev/null; do
  sleep 60
  n=$(grep -o "checkpoint at step [0-9]*" "$OUT/train.log" | tail -n 1 | grep -o "[0-9]*$")
  if [ -n "$n" ] && [ "$n" != "$last" ]; then cp "$OUT/run/ckpt_latest.bin" "$OUT/ckpt_step_$n.bin" && last=$n; fi
done
wait $PID
n=$(grep -o "checkpoint at step [0-9]*" "$OUT/train.log" | tail -n 1 | grep -o "[0-9]*$")
[ -n "$n" ] && [ "$n" != "$last" ] && cp "$OUT/run/ckpt_latest.bin" "$OUT/ckpt_step_$n.bin"
echo "done $(date '+%I:%M %p')" | tee -a "$OUT/notes.txt"
grep -E "starting from|^step +(1|250|500|1000|2000|3000|4000)/|val " "$OUT/train.log" | cut -c1-160 | tail -n 40
