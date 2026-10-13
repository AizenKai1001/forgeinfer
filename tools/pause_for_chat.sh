#!/bin/bash
# Runs on the homelab. pause_for_chat.sh CKPT: stops the f1_v1 training right after a checkpoint (the trainer saves every 100 steps, so almost nothing is lost) and starts the chat engine on CKPT,
# because the 8 GB card cannot hold trainer (6.3 GB) + chat engine (1.1 GB) + another program that holds 1 GB (Blender of another session did on 2026-10-10).
# Resume the training later (it continues from ckpt_latest.bin):   cd ~/forgeinfer/src && RUN=f1_v1 bash tools/launch_f1.sh      (stop the chat engine first if the card is full: bash ~/forgeinfer/start_chat.sh stop)
RUN=${RUN:-f1_v1}
LOG=$HOME/forgeinfer/runs/$RUN/train.log
CKPT=${1:?checkpoint for the chat engine}
PID=$(ps -eo pid,args | grep -E "/forge[^ /]* train .*--out $HOME/forgeinfer/runs/$RUN( |\$)" | grep -v grep | awk '{print $1}')
if [ -n "$PID" ]; then
  s=0
  for i in $(seq 1 400); do
    s=$(tail -n 1 "$LOG" | grep -o 'step *[0-9]*' | grep -o '[0-9]*')
    m=$(( s % 100 ))
    if [ "$m" = "1" ] || [ "$m" = "2" ]; then break; fi
    sleep 3
  done
  kill $PID
  sleep 5
  echo "trainer $PID stopped at step $s ($(date '+%I:%M %p'))"
else
  echo "no trainer of $RUN is running"
fi
bash "$HOME/forgeinfer/start_chat.sh" "$CKPT"
