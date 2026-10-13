#!/bin/bash
# GPU window 15: where the fused dequantizing GEMM beats dequantize-then-GEMM (with the fast dequantization kernels): prefill chunks of 64 ... 768 tokens on the 4B and 9B (dev build).
M=$HOME/forgeinfer/oracle/models
cd $HOME/forgeinfer/src
F=${FORGE:-$HOME/forgeinfer/target-dev/release/forge}
for q in 4B 9B; do
  F2=$M/unsloth__Qwen3.5-$q-GGUF/Qwen3.5-$q-Q4_K_M.gguf
  for n in 64 128 256 384 512 768; do
    echo "-- $q, $n tokens: dequantize first / fused"
    FORGE_NO_FUSED=1 $F prefill-bench $F2 --native --tokens $n --reps 6 2>&1 | grep -E "best"
    FORGE_FUSED_MAX=100000 $F prefill-bench $F2 --native --tokens $n --reps 6 2>&1 | grep -E "best"
  done
done
