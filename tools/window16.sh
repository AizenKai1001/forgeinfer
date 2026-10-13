#!/bin/bash
# GPU window 16: the final prefill table on the production build (defaults: fast dequantization, fused GEMMs up to 448 tokens): 0.8B, 4B, 9B at 256 / 512 / 1024 / 2048 / 4096 tokens, forge against llama.cpp b11521 on the same files.
M=$HOME/forgeinfer/oracle/models
cd $HOME/forgeinfer/src
F=./target/release/forge
source $HOME/forgeinfer/oracle/env.sh 2>/dev/null
for q in 0.8B 4B 9B; do
  F2=$M/unsloth__Qwen3.5-$q-GGUF/Qwen3.5-$q-Q4_K_M.gguf
  for n in 256 512 1024 2048 4096; do
    echo "-- $q, $n tokens: forge"
    $F prefill-bench $F2 --native --tokens $n --reps 4 2>&1 | grep -E "best"
  done
  echo "-- $q: llama.cpp"
  llama-bench -m $F2 -ngl 99 -p 256,512,1024,2048,4096 -n 0 -r 3 2>&1 | grep -E "pp[0-9]+" | cut -c1-120
done
