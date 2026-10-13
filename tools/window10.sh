#!/bin/bash
# GPU window 10: how much of the prefill time is the weight dequantization pass? Prefill of 512 / 1024 / 4096 tokens on the 4B and 9B with the pass (default), with it skipped (FORGE_SKIP_DEQUANT=1: garbage
# results, only the time counts: the ceiling of a GEMM that reads the quantized weights directly) and overlapped (FORGE_PIPE=1).
M=$HOME/forgeinfer/oracle/models
cd $HOME/forgeinfer/src
F=./target/release/forge
for q in 4B 9B; do
  F2=$M/unsloth__Qwen3.5-$q-GGUF/Qwen3.5-$q-Q4_K_M.gguf
  for n in 512 1024 4096; do
    echo "-- $q, $n tokens: serial / no dequantization / overlapped"
    $F prefill-bench $F2 --native --tokens $n --reps 4 2>&1 | grep -E "best"
    FORGE_SKIP_DEQUANT=1 $F prefill-bench $F2 --native --tokens $n --reps 4 2>&1 | grep -E "best"
    FORGE_PIPE=1 $F prefill-bench $F2 --native --tokens $n --reps 4 2>&1 | grep -E "best"
  done
done
