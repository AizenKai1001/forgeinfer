#!/bin/bash
# Run a command with the homelab training processes paused (SIGSTOP), so the GPU is idle for clean timings; they are continued afterwards, also if the command fails.
#   tools/hl.sh "bash tools/gpu_window.sh ./target/release/forge gpu-bench MODEL --tokens 128"
# The trainer keeps its GPU memory (5.8 GB) while stopped: only models/workloads that fit in the rest of the 8 GB can be measured. Keep the window short (a few minutes).
PIDS=$(ps -eo pid,args | grep -E '/forge[^ /]* train ' | grep -v grep | awk '{print $1}')
for p in $PIDS; do kill -STOP "$p"; done
trap 'for p in $PIDS; do kill -CONT "$p"; done' EXIT
sleep 1
"$@"
