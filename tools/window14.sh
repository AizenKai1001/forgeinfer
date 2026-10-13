#!/bin/bash
# GPU window 14: where the fused dequantizing GEMM gains or loses against dequantize-then-GEMM: Nsight Systems kernel tables of the 9B prefill (1024 tokens) in both modes (dev build).
S=$HOME/forgeinfer/src/tools
M=$HOME/forgeinfer/oracle/models
F=$HOME/forgeinfer/target-dev/release/forge
G=$M/unsloth__Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf
echo "################ dequantize first"
FORGE_NO_FUSED=1 bash $S/nsys_cmd.sh $F prefill-bench $G --native --tokens 1024 --reps 2 2>&1 | grep -E "total|gemm|dequant|gdn|flash" | head -n 12
echo "################ fused"
bash $S/nsys_cmd.sh $F prefill-bench $G --native --tokens 1024 --reps 2 2>&1 | grep -E "total|gemm|dequant|gdn|flash" | head -n 14
