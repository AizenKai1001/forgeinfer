#!/bin/bash
# GPU window 7: the 9B for the quantizer experiments. The 9B does not fit the card in fp16 or Q8_0, so (a) the activation statistics come from our own plain-rounding Q4_K_M of the same checkpoint (a directory
# written by `forge quantize`, Hugging Face value-head order), one layer at a time with the passes cut off after that layer; (b) the reference is unsloth's Q5_K_M (6.6 GB, fits) and everything is measured
# against it: unsloth's Q4_K_M and our plain-rounding Q4_K_M, as baselines.
S=$HOME/forgeinfer/src/tools
M=$HOME/forgeinfer/oracle/models
E=$HOME/forgeinfer/data/eval
cd $HOME/forgeinfer/src
F=./target/release/forge
Q9=$M/forgeinfer__q9b_rtn
G9=$M/unsloth__Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf
R9=$M/unsloth__Qwen3.5-9B-GGUF/Qwen3.5-9B-Q5_K_M.gguf
TOK=$M/Qwen__Qwen3.5-9B/tokenizer.json
mkdir -p /mnt/data/forgeinfer-data/h9
[ -e $E/h9 ] || ln -s /mnt/data/forgeinfer-data/h9 $E/h9
N=${N:-12288}
python3 tools/mix_calib.py $E 1.0 calib_multi9.txt
echo "################ importance statistics from our Q4_K_M"
$F imatrix $Q9 --native --text $E/calib_multi9.txt --max-tokens 98304 --out $E/im9.bin 2>&1 | grep -E "imatrix\]|error|panick"
echo "################ second moments, one layer per pass, each pass starts from the residual stream the previous one left"
$F hessian $Q9 --native --layers-per-pass 1 --text $E/calib_multi9.txt --max-tokens 98304 --out $E/h9/hess9 2>&1 | grep -E "hessian\]|error|panick" | tail -n 40
EVAL=""; REFS=""
for d in en code zh de chat; do EVAL="$EVAL,$E/eval_$d.txt"; REFS="$REFS,$E/ref9_$d.bin"; done
EVAL=${EVAL#,}; REFS=${REFS#,}
echo "################ reference: unsloth Q5_K_M"
$F ppl $R9 --native --tokenizer $TOK --text $EVAL --max-tokens $N --dump-ref $REFS 2>&1 | grep -E "RESULT|error|panick"
echo "################ unsloth Q4_K_M"
$F ppl $G9 --native --tokenizer $TOK --text $EVAL --max-tokens $N --ref $REFS 2>&1 | grep -E "RESULT|VS-REF|error|panick"
echo "################ ours, plain rounding (the calibration model)"
$F ppl $Q9 --native --tokenizer $TOK --text $EVAL --max-tokens $N --ref $REFS 2>&1 | grep -E "RESULT|VS-REF|error|panick"
