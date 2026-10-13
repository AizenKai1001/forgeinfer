#!/bin/bash
# GPU window 11: the fused dequantizing GEMM (K-quant weights read in place) against dequantize-then-GEMM (FORGE_NO_FUSED=1), prefill of 512 / 1024 / 4096 tokens on the 0.8B, 4B and 9B, and llama.cpp's numbers for the same.
M=$HOME/forgeinfer/oracle/models
cd $HOME/forgeinfer/src
F=./target/release/forge
source $HOME/forgeinfer/oracle/env.sh 2>/dev/null
for q in 0.8B 4B 9B; do
  F2=$M/unsloth__Qwen3.5-$q-GGUF/Qwen3.5-$q-Q4_K_M.gguf
  for n in 512 1024 4096; do
    echo "-- $q, $n tokens: dequantize first / fused"
    FORGE_NO_FUSED=1 $F prefill-bench $F2 --native --tokens $n --reps 4 2>&1 | grep -E "best"
    $F prefill-bench $F2 --native --tokens $n --reps 4 2>&1 | grep -E "best"
  done
  echo "-- $q: llama.cpp"
  llama-bench -m $F2 -ngl 99 -p 512,1024,4096 -n 0 -r 2 2>&1 | grep -E "pp[0-9]+" | cut -c1-120
done
