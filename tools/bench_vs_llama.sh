#!/bin/bash
# Decode (tg128) and prefill (pp512) of one GGUF: forge (gpu-bench / prefill-bench --native) and llama.cpp b11521 (llama-bench) back to back, trainer paused.
#   bash tools/gpu_window.sh bash tools/bench_vs_llama.sh <model.gguf> [tokens]
M=$1; N=${2:-128}
source ~/forgeinfer/oracle/env.sh 2>/dev/null
cd ~/forgeinfer/src
echo "== forge decode"
./target/release/forge gpu-bench "$M" --native --tokens $N --ctx 1024 2>&1 | grep -E "best|run 4"
echo "== forge prefill"
./target/release/forge prefill-bench "$M" --native --tokens 512 2>&1 | grep -E "best"
echo "== llama.cpp"
llama-bench -m "$M" -ngl 99 -p 512 -n $N -r 3 2>&1 | grep -E "pp512|tg$N|model" | head -4
