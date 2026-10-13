#!/bin/bash
# GPU window 13: the fused dequantizing GEMM against dequantize-then-GEMM on the dev build (fused forced on at every size with FORGE_FUSED_MAX): prefill of 512 / 1024 / 4096 tokens on the 4B and 9B,
# and the default (fused up to 768 tokens) for the sizes in between.
M=$HOME/forgeinfer/oracle/models
cd $HOME/forgeinfer/src
F=${FORGE:-$HOME/forgeinfer/target-dev/release/forge}
for q in 4B 9B; do
  F2=$M/unsloth__Qwen3.5-$q-GGUF/Qwen3.5-$q-Q4_K_M.gguf
  for n in 512 1024 4096; do
    echo "-- $q, $n tokens: dequantize first / fused (forced) / default"
    FORGE_NO_FUSED=1 $F prefill-bench $F2 --native --tokens $n --reps 4 2>&1 | grep -E "best"
    FORGE_FUSED_MAX=100000 $F prefill-bench $F2 --native --tokens $n --reps 4 2>&1 | grep -E "best"
    $F prefill-bench $F2 --native --tokens $n --reps 4 2>&1 | grep -E "best"
  done
done
