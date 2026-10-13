#!/bin/bash
# Build the multi-row matvec micro-benchmark (tools/mrow_kbench.cu) in the work tree it is run from (homelab: tools/agent_hl.sh NAME "tools/mrow_kbench.sh").
# Prints the registers and spills of the benchmark kernels; the binary is ./mrow_kbench. Extra arguments go to nvcc.
set -e
cd "$(dirname "$0")/.."
export PATH="$HOME/cuda-13.2.0/bin:$PATH"
LOG=$(mktemp)
if ! nvcc -O3 -lineinfo --use_fast_math -std=c++17 -arch=sm_75 -Xptxas -v -I crates/forge-kernels/kernels tools/mrow_kbench.cu -o mrow_kbench "$@" > "$LOG" 2>&1; then
    grep -E "error|Error" "$LOG" | head -40
    exit 1
fi
awk '/Compiling entry function/ { name=$0; show = (name ~ /bench_mm/) } show && /Compiling entry function|spill|Used [0-9]+ registers/ { print }' "$LOG" | sed -E "s/ptxas info    : //" | sed -E "s/Compiling entry function '_Z8bench_mmILi([0-9]+)ELi([0-9]+)ELi([0-9]+)ELi([0-9]+)ELi([0-9]+)ELi([0-9]+)E.*/bench_mm F=\1 T=\2 R=\3 NW=\4 MINB=\5 EPI=\6/"
