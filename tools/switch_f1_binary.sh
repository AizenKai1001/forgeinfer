#!/bin/bash
# Runs on the homelab. Swap the binary of the production run (f1_v1) at the next checkpoint: wait for a new "checkpoint at step N" line in train.log (the trainer writes ckpt_latest.bin atomically every 100 steps),
# stop the trainer right after it, start NEWBIN with --resume through tools/launch_f1.sh (same flags as the run), print the first lines. Nothing but the seconds of start-up is lost; the step counter, the
# learning-rate schedule and the data order continue from the checkpoint.
#   bash switch_f1_binary.sh NEWBIN            DRY=1 bash switch_f1_binary.sh NEWBIN   (prints what it would do)
# To go back: bash switch_f1_binary.sh ~/forgeinfer/bin/forge-f1_v1
set -u
NEW="${1:?usage: switch_f1_binary.sh NEWBIN}"
RUN="${RUN:-f1_v1}"
LOG="$HOME/forgeinfer/runs/$RUN/train.log"
F1_ARGS="${F1_ARGS:---relu2 --lambdas --lr-scalar-mult 10 --value-emb 0,1,28,29 --lr-table-mult 20 --lr-emb-mult 3}"
DRY="${DRY:-}"
[ -x "$NEW" ] || { echo "not an executable file: $NEW"; exit 1; }
# the trainer of THIS run (--out $HOME/forgeinfer/runs/$RUN): other `forge ... train` processes on the homelab (agents' test runs, other runs) must neither be stopped nor taken for it
trainer_pid() { ps -eo pid,args | grep -E "/forge[^ /]* train .*--out $HOME/forgeinfer/runs/$RUN( |\$)" | grep -v grep | awk '{print $1}' | head -n 1; }
before=$(grep -c "checkpoint at step" "$LOG")
echo "[$(date '+%I:%M:%S %p')] waiting for the next checkpoint (now $before checkpoint lines in $LOG)"
if [ -z "$DRY" ]; then
  until [ "$(grep -c 'checkpoint at step' "$LOG")" -gt "$before" ]; do sleep 1; done
fi
line=$(grep "checkpoint at step" "$LOG" | tail -n 1)
pid=$(trainer_pid)
echo "[$(date '+%I:%M:%S %p')] $line -> stopping the trainer (pid ${pid:-none}) and starting $NEW"
[ -n "$DRY" ] && exit 0
if [ -n "$pid" ]; then
  kill "$pid"
  for i in $(seq 1 60); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
  kill -0 "$pid" 2>/dev/null && { echo "the trainer did not stop in 60 s: not starting a second one"; exit 1; }
fi
cd "$HOME/forgeinfer/src" || exit 1
RUN="$RUN" FFN="${FFN:-2304}" STEPS="${STEPS:-45000}" FORGE_BIN="$NEW" F1_ARGS="$F1_ARGS" bash tools/launch_f1.sh
for i in $(seq 1 12); do sleep 10; done
tail -n 6 "$LOG" | cut -c1-170
[ -n "$(trainer_pid)" ] || echo "WARNING: no trainer of $RUN is running after the switch (see $LOG)"
