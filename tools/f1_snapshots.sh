#!/bin/bash
# Runs on the homelab: keeps a copy of the production run's checkpoint every EVERY steps (default 2500) in /mnt/data/forgeinfer-data/ckpt_f1_v1/ (the trainer only keeps one rolling ckpt_latest.bin,
# written atomically every 100 steps: a diverged or corrupted state would replace the last good one). The copy is taken with idle IO priority; a snapshot is only kept when its size matches the source.
#   nohup bash f1_snapshots.sh [every=2500] [run_dir] [out_dir] > ~/forgeinfer/snapshots.log 2>&1 &
# The snapshots serve the evaluation ladder over training time, forks (a WSD decay from any step gives a finished model of that size of training) and the control experiment of the cool-down at step 40,500.
EVERY="${1:-2500}"
RUN="${2:-$HOME/forgeinfer/runs/f1_v1}"
OUT="${3:-/mnt/data/forgeinfer-data/ckpt_f1_v1}"
mkdir -p "$OUT"
step_of() { head -c 600 "$1" | head -n 1 | grep -o '"step": *[0-9]*' | grep -o '[0-9]*$'; }
echo "$(date '+%F %I:%M %p') snapshot loop started: every $EVERY steps, $RUN -> $OUT"
while true; do
  if [ -f "$RUN/ckpt_latest.bin" ]; then
    s=$(step_of "$RUN/ckpt_latest.bin")
    if [ -n "$s" ] && [ "$s" -gt 0 ] && [ $((s % EVERY)) -eq 0 ] && [ ! -f "$OUT/ckpt_s$s.bin" ]; then
      tmp="$OUT/.tmp_s$s"
      ionice -c3 nice -n 19 cp "$RUN/ckpt_latest.bin" "$tmp"
      want=$(stat -c %s "$RUN/ckpt_latest.bin")
      got=$(stat -c %s "$tmp")
      s2=$(step_of "$tmp")
      if [ "$got" = "$want" ] && [ "$s2" = "$s" ]; then
        mv "$tmp" "$OUT/ckpt_s$s.bin"
        echo "$(date '+%F %I:%M %p') saved step $s ($got bytes)"
      else
        echo "$(date '+%F %I:%M %p') copy of step $s failed (size $got of $want, header step $s2): removed"
        rm "$tmp"
      fi
    fi
  fi
  sleep 60
done
