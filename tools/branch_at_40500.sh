#!/bin/bash
# Runs on the homelab. At step 40,500 of the production run (the start of the learning-rate decay of a 45,000 step schedule) the learning rate is still constant, so the run can BRANCH (warmup-stable-decay):
#   A  the cool-down branch: decay 4,500 steps with the cool-down data blend (the 3 B token base model)         -> ~/forgeinfer/runs/${RUN}_A
#   C  the control: the same decay without --cool-data (what is the mix worth?)                  (CONTROL=1)    -> ~/forgeinfer/runs/${RUN}_C
#   B  the long run: the main run resumed from the same checkpoint with --steps ${B_STEPS} (default 90,000)  (LONG=1)  -> ~/forgeinfer/runs/${RUN}
# The branches run one after the other on the one GPU. After every branch the evaluation ladder scores its final checkpoint.
#   nohup bash branch_at_40500.sh > ~/forgeinfer/branch.log 2>&1 &         DRY=1 bash branch_at_40500.sh      (prints the plan and the commands, starts nothing)
# Settings (environment): RUN (f1_v1), FORGE_BIN (production binary), EVAL_BIN (a build that has eval-lambada), AT (40500), CONTROL (1), LONG (1), B_STEPS (90000), COOL (the cool-down directory), EVAL_FULL (1 = the
# full benchmark sets instead of --quick). The trainer of $RUN is found by its --out directory (other `forge train` processes of other agents are never touched).
set -u
RUN="${RUN:-f1_v1}"
AT="${AT:-40500}"
FORGE_BIN="${FORGE_BIN:-$HOME/forgeinfer/bin/forge-f1_merged}"
EVAL_BIN="${EVAL_BIN:-$HOME/forgeinfer/agents/lead-main/target/release/forge}"
[ -x "$EVAL_BIN" ] || EVAL_BIN="$FORGE_BIN"
COOL="${COOL:-/mnt/data/forgeinfer-data/tokens_cool}"
DATA="${DATA:-/mnt/data/forgeinfer-data/tokens_f1}"
CONTROL="${CONTROL:-1}"
LONG="${LONG:-1}"
B_STEPS="${B_STEPS:-90000}"
EVAL_FULL="${EVAL_FULL:-0}"
DRY="${DRY:-}"
RUNS="$HOME/forgeinfer/runs"
SNAP=/mnt/data/forgeinfer-data/ckpt_f1_v1
LOG="$RUNS/$RUN/train.log"
# the model and optimiser flags of the production run (everything except --steps, --out and the data blend)
COMMON="--layers 30 --dim 576 --heads 9 --kv 3 --ffn 2304 --seq 2048 --micro 2 --tokens-per-step 65536 --warmup 100 --decay-frac 0.1 --min-lr-frac 0.05 --eval-every 100 --eval-batches 32 --save-every 100 --qk-norm --optim muon --lr 1.5e-3 --muon-lr 1.5e-3 --muon-wd 0.05 --relu2 --lambdas --lr-scalar-mult 10 --value-emb 0,1,28,29 --lr-table-mult 20 --lr-emb-mult 3"
EXTRA_FILE="$HOME/forgeinfer/runs/$RUN/extra_args.txt"; [ -f "$EXTRA_FILE" ] && COMMON="$COMMON $(cat "$EXTRA_FILE")"   # per-run flags (for example the window attention of the run) belong to the branches as well
say() { echo "[$(date '+%Y-%m-%d %I:%M:%S %p')] $*"; }
trainer_pid() { ps -eo pid,args | grep -E "/forge[^ /]* train .*--out $RUNS/$RUN( |\$)" | grep -v grep | awk '{print $1}' | head -n 1; }
# Is header step $2 the checkpoint written at loop step $1? The log names the LOOP step ("checkpoint at step 40500"), the header holds the number of APPLIED updates: the loop step minus the
# updates skipped for non-finite gradients since the trainer was last started (forge train resumes at the header step and replays those steps). So accept $1 - 99 .. $1.
accept_header() { [ -n "$2" ] && [ "$2" -le "$1" ] && [ "$2" -gt $(( $1 - 100 )) ]; }
ladder() {  # $1 = checkpoint, $2 = name
  local mode="--quick"; [ "$EVAL_FULL" = "1" ] && mode=""
  say "ladder on $2"
  [ -n "$DRY" ] && { echo "  bash $HOME/forgeinfer/src/tools/f1_eval.sh $1 $mode --forge $EVAL_BIN --tasks all --name $2 --out $HOME/forgeinfer/eval_results"; return; }
  (cd "$HOME/forgeinfer/src" && bash tools/f1_eval.sh "$1" $mode --forge "$EVAL_BIN" --tasks all --name "$2" --out "$HOME/forgeinfer/eval_results") > "$HOME/forgeinfer/eval_results/$2.log" 2>&1
}
branch() {  # $1 = out dir name, $2 = extra flags, $3 = steps
  local out="$RUNS/$1"
  say "branch $1: $FORGE_BIN train ... --steps $3 $2 (from the checkpoint of step $AT)"
  [ -n "$DRY" ] && return 0
  mkdir -p "$out"
  cp "$SNAP/ckpt_s$AT.bin" "$out/ckpt_latest.bin"
  nice -n 5 "$FORGE_BIN" train --data "$DATA" --out "$out" $COMMON --steps "$3" --resume $2 >> "$out/train.log" 2>&1
  say "branch $1 ended with exit code $?"
}
say "plan: wait for the checkpoint of step $AT of $RUN; snapshot; stop the trainer; A (cool-down, 45000 steps)$( [ "$CONTROL" = 1 ] && echo '; C (control)')$( [ "$LONG" = 1 ] && echo "; B (main run resumed to $B_STEPS steps)")"
if [ -z "$DRY" ]; then
  until grep -q "checkpoint at step $AT " "$LOG"; do sleep 5; done
fi
pid=$(trainer_pid)
say "step $AT reached: trainer pid ${pid:-none}"
if [ -z "$DRY" ]; then
  # the header is checked BEFORE the trainer is stopped: if this is not the checkpoint of step $AT, nothing is stopped and the main run keeps going
  hdr=$(head -c 600 "$RUNS/$RUN/ckpt_latest.bin" | head -n 1 | grep -o '"step": *[0-9]*' | grep -o '[0-9]*$')
  accept_header "$AT" "$hdr" || { say "ckpt_latest.bin says step ${hdr:-?}, not the checkpoint of step $AT: nothing stopped, abort (the main run continues)"; exit 1; }
  [ "$hdr" = "$AT" ] || say "WARNING: the checkpoint of step $AT holds $hdr applied updates ($(( AT - hdr )) skipped since the trainer was started): the branches resume at step $hdr and replay $(( AT - hdr )) steps (harmless: the decay and the cool-down share follow the step number)"
  [ -n "$pid" ] && { kill "$pid"; for i in $(seq 1 90); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done; }
  kill -0 "$pid" 2>/dev/null && { say "the trainer did not stop: abort"; exit 1; }
  ionice -c2 -n7 cp "$RUNS/$RUN/ckpt_latest.bin" "$SNAP/.tmp_at"
  hdr2=$(head -c 600 "$SNAP/.tmp_at" | head -n 1 | grep -o '"step": *[0-9]*' | grep -o '[0-9]*$')
  [ "$hdr2" = "$hdr" ] || { say "the copy says step $hdr2, not $hdr: abort (the trainer is stopped: restart it with tools/launch_f1.sh)"; exit 1; }
  mv "$SNAP/.tmp_at" "$SNAP/ckpt_s$AT.bin"
  say "snapshot saved: $SNAP/ckpt_s$AT.bin (header step $hdr)"
fi
branch "${RUN}_A" "--cool-data $COOL --cool-blend-steps 1125 --cool-share 0.6" 45000
ladder "$RUNS/${RUN}_A/ckpt_latest.bin" "${RUN}_A"
if [ "$CONTROL" = 1 ]; then
  branch "${RUN}_C" "" 45000
  ladder "$RUNS/${RUN}_C/ckpt_latest.bin" "${RUN}_C"
fi
if [ "$LONG" = 1 ]; then
  say "branch B: the main run $RUN resumes from its checkpoint of step $AT with --steps $B_STEPS"
  if [ -z "$DRY" ]; then
    cd "$HOME/forgeinfer/src" || exit 1
    RUN="$RUN" FFN=2304 STEPS="$B_STEPS" EVAL_BATCHES=32 FORGE_BIN="$FORGE_BIN" F1_ARGS="--relu2 --lambdas --lr-scalar-mult 10 --value-emb 0,1,28,29 --lr-table-mult 20 --lr-emb-mult 3" bash tools/launch_f1.sh
  fi
fi
say "done"
