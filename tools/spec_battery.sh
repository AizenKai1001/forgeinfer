#!/bin/bash
# Speculative decoding on the Qwen3.5 models: the same server benchmark with and without the MTP head. Needs the whole GPU (see tools/pause_training.sh).
#   tools/hl.sh -n "bash ~/forgeinfer/src/tools/pause_training.sh bash ~/forgeinfer/src/tools/spec_battery.sh 4B 9B"
M=$HOME/forgeinfer/oracle/models
cd $HOME/forgeinfer/src
SIZES=${*:-4B 9B}
PORT=18090
wait_ready() {
  for i in $(seq 1 150); do
    curl -s -o /dev/null http://127.0.0.1:$PORT/health && return 0
    sleep 1
  done
  return 1
}
for q in $SIZES; do
  F=unsloth__Qwen3.5-$q-GGUF/Qwen3.5-$q-Q4_K_M.gguf
  H=$M/Qwen__Qwen3.5-$q/mtp.safetensors
  [ -f $M/$F ] && [ -f $H ] || { echo "Qwen3.5-$q: files missing"; continue; }
  echo "################ Qwen3.5-$q: speculative output against plain decoding"
  FORGE_SPEC_SIZE=$q cargo test --release -p forge-engine --test speculative -- --nocapture 2>&1 | grep -E "^prompt|tokens in|^sampled|stop string|test result|panicked|assert"
  for mode in plain speculative; do
    echo "################ Qwen3.5-$q Q4_K_M, $mode"
    if [ $mode = plain ]; then EXTRA="--no-speculate"; else EXTRA="--mtp $H"; fi
    bash tools/serve_up.sh $PORT $F Qwen__Qwen3.5-$q/tokenizer.json qwen35-$q --native --ctx 8192 $EXTRA > /dev/null 2>&1
    if wait_ready; then
      ss -ltnp 2>/dev/null | grep ":$PORT " | sed -E 's/.*pid=([0-9]+).*/\1/' > $HOME/forgeinfer/serve_$PORT.pid
      grep -E "ready" $HOME/forgeinfer/serve_$PORT.log | cut -c1-200
      python3 tools/serve_bench.py $PORT 256 2>&1 | tail -n 7
    else
      echo "the server did not start:"; tail -n 5 $HOME/forgeinfer/serve_$PORT.log
    fi
    kill $(cat $HOME/forgeinfer/serve_$PORT.pid) 2>/dev/null
    sleep 4
  done
done
