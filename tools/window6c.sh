#!/bin/bash
# GPU window 6c: the second moments of layers 4.. of the 4B (window 6a's second group ran out of memory: the previous group's buffers were still allocated; fixed). 4 layers per pass, passes cut off after the group.
E=$HOME/forgeinfer/data/eval
cd $HOME/forgeinfer/src
F=./target/release/forge
HF4=/mnt/data/forgeinfer-hf/Qwen__Qwen3.5-4B
echo "################ second moments of layers 4.. of the 4B"
$F hessian $HF4 --q8 --layers-per-pass 4 --first-layer 4 --text $E/calib_multi4.txt --max-tokens 196608 --out $E/h4/hess4 2>&1 | grep -E "hessian\]|error|panick"
