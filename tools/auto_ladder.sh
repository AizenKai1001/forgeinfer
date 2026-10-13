#!/bin/bash
# Runs on the homelab. Watches the snapshots of the production run (/mnt/data/forgeinfer-data/ckpt_f1_v1/ckpt_s<N>.bin, one every 2500 steps) and scores every new one (N >= FIRST) with the evaluation ladder
# (all tasks, quick mode) inside a short GPU window of the production run (tools/pause_training.sh: the trainer stops right after a checkpoint and is restarted afterwards: about 5 minutes per snapshot).
# Results: ~/forgeinfer/eval_results/auto/s<N>.md (the whole table) and one line per snapshot in ~/forgeinfer/eval_results/auto/summary.txt.
#   nohup bash ~/forgeinfer/auto_ladder.sh > ~/forgeinfer/auto_ladder.log 2>&1 &
# Settings: FIRST (12500), SNAP, EVAL_BIN (a build that has eval-lambada), FORGE_BIN (the production binary used to restart the run), RUN (f1_v1), POLL_S (300), STOP_FILE (~/forgeinfer/STOP_AUTO_LADDER: create it to end the loop).
# It leaves a snapshot alone while the step-40,500 branch script is waiting for its checkpoint (within 400 steps of AT) and while the main run is not training.
set -u
FIRST="${FIRST:-12500}"
RUN="${RUN:-f1_v1}"
SNAP="${SNAP:-/mnt/data/forgeinfer-data/ckpt_f1_v1}"
EVAL_BIN="${EVAL_BIN:-$HOME/forgeinfer/agents/lead-main/target/release/forge}"
FORGE_BIN="${FORGE_BIN:-$HOME/forgeinfer/bin/forge-f1_merged}"
POLL_S="${POLL_S:-300}"
AT="${AT:-40500}"
STOP_FILE="${STOP_FILE:-$HOME/forgeinfer/STOP_AUTO_LADDER}"
AUTO="$HOME/forgeinfer/eval_results/auto"
LOG="$HOME/forgeinfer/runs/$RUN/train.log"
mkdir -p "$AUTO"
say() { echo "[$(date '+%Y-%m-%d %I:%M:%S %p')] $*"; }
say "auto ladder started: snapshots of $SNAP from step $FIRST on"
while [ ! -e "$STOP_FILE" ]; do
  for f in $(ls "$SNAP"/ckpt_s*.bin 2>/dev/null | sort -V); do
    n=$(basename "$f" .bin); n=${n#ckpt_s}
    case "$n" in ''|*[!0-9]*) continue;; esac
    [ "$n" -ge "$FIRST" ] || continue
    [ -f "$AUTO/s$n.md" ] && continue
    cur=$(tail -n 1 "$LOG" | grep -o 'step *[0-9]*' | grep -o '[0-9]*' | head -n 1)
    if [ -n "$cur" ] && [ "$cur" -ge $((AT - 400)) ] && [ "$cur" -le $((AT + 100)) ]; then say "step $cur is close to $AT: snapshot $n waits"; continue; fi
    if [ -z "$cur" ]; then say "the log of $RUN shows no step: snapshot $n waits"; continue; fi
    say "snapshot $n: starting the window (the run is at step $cur)"
    ( cd "$HOME/forgeinfer/src" && env LAUNCH=tools/launch_f1.sh RUN="$RUN" FFN=2304 STEPS=45000 FORGE_BIN="$FORGE_BIN" F1_ARGS='--relu2 --lambdas --lr-scalar-mult 10 --value-emb 0,1,28,29 --lr-table-mult 20 --lr-emb-mult 3' SAVE_EVERY=100 \
        bash tools/pause_training.sh bash tools/f1_eval.sh "$f" --quick --forge "$EVAL_BIN" --tasks all --name "s$n" --out "$AUTO/out_s$n" ) > "$AUTO/window_s$n.log" 2>&1
    if [ -f "$AUTO/out_s$n/results.md" ]; then
      cp "$AUTO/out_s$n/results.md" "$AUTO/s$n.md"
      row=$(grep -E "^\| \*\*s$n \(this run\)" "$AUTO/s$n.md" | head -n 1)
      mean=$(grep -E '^\| \*\*mean\*\*' "$AUTO/s$n.md" | head -n 1)
      echo "s$n | $row | ppl $mean" >> "$AUTO/summary.txt"
      say "snapshot $n scored"
    else
      say "snapshot $n: no results (see $AUTO/window_s$n.log); will not retry"
      echo "failed" > "$AUTO/s$n.md"
    fi
    sleep 30
  done
  sleep "$POLL_S"
done
say "stop file found: end"
