#!/bin/bash
# How many tokens of the vocabulary the draft head should score (FORGE_DRAFT_ROWS): decode speed and acceptance of the Qwen3.5 4B and 9B server for several values. Needs the whole GPU (see tools/pause_training.sh).
M=$HOME/forgeinfer/oracle/models
cd $HOME/forgeinfer/src
PORT=18090
SIZES=${SIZES:-4B 9B}
ROWS=${ROWS:-16384 32768 65536 131072 248320}
for q in $SIZES; do
  F=unsloth__Qwen3.5-$q-GGUF/Qwen3.5-$q-Q4_K_M.gguf
  H=$M/Qwen__Qwen3.5-$q/mtp.safetensors
  for r in $ROWS; do
    echo "######## Qwen3.5-$q, draft head over the first $r tokens"
    FORGE_DRAFT_ROWS=$r bash tools/serve_up.sh $PORT $F Qwen__Qwen3.5-$q/tokenizer.json qwen35-$q --native --ctx 8192 --mtp $H > /dev/null 2>&1
    for i in $(seq 1 150); do curl -s -o /dev/null http://127.0.0.1:$PORT/health && break; sleep 1; done
    ss -ltnp 2>/dev/null | grep ":$PORT " | sed -E 's/.*pid=([0-9]+).*/\1/' > $HOME/forgeinfer/serve_$PORT.pid
    BENCH_QUICK=1 python3 tools/serve_bench.py $PORT 256 2>&1 | tail -n 2
    kill $(cat $HOME/forgeinfer/serve_$PORT.pid) 2>/dev/null
    sleep 4
  done
done
