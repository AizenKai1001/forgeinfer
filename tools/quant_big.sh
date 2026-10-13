#!/bin/bash
# Quantize a big model with our GPTQ (CPU only, run it niced): into exactly the per-tensor formats of unsloth's Q4_K_M file, optionally with changes (--override), second moments read from disk.
#   tools/quant_big.sh 4B  OUT_NAME [extra forge-quantize flags...]     e.g.   tools/quant_big.sh 4B forgeinfer__q4b_gptq
#   tools/quant_big.sh 9B  forgeinfer__q9b_gptq --override "token_embd=Q5_K"
SIZE=$1
OUT=$2
shift 2
M=$HOME/forgeinfer/oracle/models
E=$HOME/forgeinfer/data/eval
F=$HOME/forgeinfer/src/target/release/forge
HF=/mnt/data/forgeinfer-hf/Qwen__Qwen3.5-$SIZE
G=$M/unsloth__Qwen3.5-$SIZE-GGUF/Qwen3.5-$SIZE-Q4_K_M.gguf
if [ "$SIZE" = "4B" ]; then H="$E/h4/hess4.*.bin"; else H="$E/h9/hess9.*.bin"; fi
nice -n 15 $F quantize $HF $M/$OUT --like $G --gptq "$H" --gptq-damp ${DAMP:-0.01} "$@" 2>&1 | grep -E "quantize\]|quant\]|error|panick"
