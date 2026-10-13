#!/bin/bash
# Greedy continuation of a prompt by forge and by llama.cpp b11521 on the same GGUF (a quick check that a model family is implemented right: both should say the same thing for a while).
#   bash tools/gpu_window.sh bash tools/compare_greedy.sh <model.gguf> <tokenizer.json> "prompt" [tokens]
M=$1; TOK=$2; P=$3; N=${4:-32}
source ~/forgeinfer/oracle/env.sh 2>/dev/null
cd ~/forgeinfer/src
echo "== forge"
./target/release/forge run "$M" --gpu --native --tokenizer "$TOK" --prompt "$P" --tokens $N --temp 0 2>&1 | grep -v "^\[" | head -n 12
echo "== llama.cpp"
llama-completion -m "$M" -p "$P" -n $N --temp 0 -ngl 99 -no-cnv --no-display-prompt 2>/dev/null | head -n 12
