#!/bin/bash
# Progress of a Forge-1 run next to the training process on the homelab GPU: a copy of the latest checkpoint is evaluated on the per-domain validation sets (nats per token) and a few benchmarks (subsets of the
# multiple-choice sets), and one block is appended to ~/forgeinfer/runs/$RUN/track.txt.
#   next to the running trainer (about 1.3 GB of GPU memory: tight, the card has 8 GB and the run holds 6.5 GB): RUN=f1_v1 bash tools/f1_track.sh
#   in a GPU window (the trainer is paused right after a checkpoint, then relaunched; full batch sizes): see the run book; FULL=1 selects the larger settings
#   (windows: WINDOWS=96, benchmark examples: LIMIT=500)
RUN=${RUN:-f1_v1}
F=${FORGE_EVAL_BIN:-$HOME/forgeinfer/target-dev3/release/forge}
T=$HOME/forgeinfer/data/tokenizer_f1.json
D=$HOME/forgeinfer/data
OUT=$HOME/forgeinfer/runs/$RUN/track.txt
if [ -n "${FULL:-}" ]; then PPL_SEQ=512; PPL_BATCH=4; MC_BATCH=8; HS_BATCH=8; fi
SNAP=/tmp/track_${RUN}.bin
cp $HOME/forgeinfer/runs/$RUN/ckpt_latest.bin $SNAP
STEP=$(head -c 600 $SNAP | head -n 1 | grep -o '"step":[0-9]*' | grep -o '[0-9]*')
{
  echo "== $(date '+%Y-%m-%d %I:%M %p'): $RUN at step $STEP ($(( STEP * 65536 / 1000000 )) M tokens)"
  nice -n 15 $F eval-ppl $SNAP --data $D/val_domains/web.u16 $D/val_domains/cosmo.u16 $D/val_domains/fmath.u16 $D/val_domains/owmath.u16 $D/val_domains/wiki.u16 $D/val_domains/books.u16 $D/val_domains/python.u16 --windows ${WINDOWS:-96} --seq ${PPL_SEQ:-256} --batch ${PPL_BATCH:-2} 2>&1 | tail -n 7
  for t in arc_easy piqa sciq winogrande; do
    nice -n 15 $F eval-mc $SNAP --tokenizer $T --data $D/eval/mc/$t.jsonl --limit ${LIMIT:-500} --batch ${MC_BATCH:-2} 2>&1 | tail -n 1
  done
  nice -n 15 $F eval-hellaswag $SNAP --tokenizer $T --data $D/hellaswag_val.jsonl --limit ${HS_LIMIT:-1000} --batch ${HS_BATCH:-4} 2>&1 | tail -n 1
  # what the model writes now: four plain continuations, temperature 0.7 (the same seed every time, so the samples are comparable between steps)
  for p in "The capital of France is" "Once upon a time" "def fibonacci(n):" "The main idea of the theory of evolution is"; do
    printf 'sample: '
    nice -n 15 $F chat $SNAP --tokenizer $T --prompt "$p" --tokens 48 --temp 0.7 --top-p 0.9 --seq 256 --seed 1 2>&1 | tail -n 6 | tr '\n' ' ' | cut -c1-420
    echo
  done
} >> $OUT 2>&1
rm -f $SNAP
tail -n 20 $OUT
