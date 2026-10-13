#!/bin/bash
# GPU window 6a: the 4B for the quantizer experiments: importance statistics and second moments (4 layers at a time: the Q8_0 4B + 4 layers of second moments fit the card) on a larger multi-domain calibration,
# the reference (Q8_0 weights: the fp16 copy does not fit the card) and unsloth's Q4_K_M, on held-out text of five domains.
S=$HOME/forgeinfer/src/tools
M=$HOME/forgeinfer/oracle/models
E=$HOME/forgeinfer/data/eval
cd $HOME/forgeinfer/src
F=./target/release/forge
HF4=/mnt/data/forgeinfer-hf/Qwen__Qwen3.5-4B
G4=$M/unsloth__Qwen3.5-4B-GGUF/Qwen3.5-4B-Q4_K_M.gguf
mkdir -p $E/h4
N=${N:-16384}
python3 tools/mix_calib.py $E 1.6 calib_multi4.txt
echo "################ importance statistics"
$F imatrix $HF4 --q8 --text $E/calib_multi4.txt --max-tokens 196608 --out $E/im4.bin 2>&1 | grep -E "imatrix\]|error|panick"
echo "################ second moments (4 layers at a time, one process)"
$F hessian $HF4 --q8 --layers-per-pass 4 --text $E/calib_multi4.txt --max-tokens 196608 --out $E/h4/hess4 2>&1 | grep -E "hessian\]|error|panick"
EVAL=""; REFS=""
for d in en code zh de chat; do EVAL="$EVAL,$E/eval_$d.txt"; REFS="$REFS,$E/ref4_$d.bin"; done
EVAL=${EVAL#,}; REFS=${REFS#,}
echo "################ reference: 4B with Q8_0 weights"
$F ppl $HF4 --q8 --text $EVAL --max-tokens $N --dump-ref $REFS 2>&1 | grep -E "RESULT|error|panick"
echo "################ unsloth Q4_K_M"
$F ppl $G4 --native --tokenizer $HF4/tokenizer.json --text $EVAL --max-tokens $N --ref $REFS 2>&1 | grep -E "RESULT|VS-REF|error|panick"
echo "################ does a 3x larger calibration help? (0.8B, GPTQ, same formats as unsloth)"
HF=$M/Qwen__Qwen3.5-0.8B
G=$M/unsloth__Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf
EVAL3=""; REFS3=""
for d in en code zh de chat; do EVAL3="$EVAL3,$E/eval_$d.txt"; REFS3="$REFS3,$E/ref3_$d.bin"; done
EVAL3=${EVAL3#,}; REFS3=${REFS3#,}
python3 tools/mix_calib.py $E 3.0 calib_multi_x3.txt
$F hessian $HF --text $E/calib_multi_x3.txt --max-tokens 458752 --out $E/hess_x3.bin 2>&1 | grep -E "hessian\]|error|panick"
$F ppl $HF --quant-like $G --gptq $E/hess_x3.bin --gptq-damp 0.01 --text $EVAL3 --max-tokens 24576 --ref $REFS3 2>&1 | grep -E "RESULT|VS-REF|error|panick"
