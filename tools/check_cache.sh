#!/bin/bash
# Cross-check of the residual-stream cache in `forge hessian`: the 0.8B (Q8_0 weights, small enough to run next to the trainer) with 6 layers per pass, once with the cache and once from the
# embedding every time (--no-cache). The second moments must agree up to the rounding of the differently chunked sums.
M=$HOME/forgeinfer/oracle/models
E=$HOME/forgeinfer/data/eval
cd $HOME/forgeinfer/src
F=./target/release/forge
D=$HOME/forgeinfer/tmp/hc
mkdir -p $D
T=${TOKENS:-16384}
HF=$M/Qwen__Qwen3.5-0.8B
echo "-- with the cache"
$F hessian $HF --q8 --layers-per-pass 6 --text $E/calib_multi.txt --max-tokens $T --out $D/cached 2>&1 | grep -E "hessian\]|error|panick"
echo "-- without (--no-cache)"
$F hessian $HF --q8 --layers-per-pass 6 --no-cache --text $E/calib_multi.txt --max-tokens $T --out $D/uncached 2>&1 | grep -E "hessian\]|error|panick"
$F cmp-hessians "$D/cached.*.bin" "$D/uncached.*.bin"
