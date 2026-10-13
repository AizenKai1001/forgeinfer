#!/bin/bash
# GPU window 12: the 9B quantized by us (GPTQ, unsloth's Q4_K_M formats) against unsloth's file, both measured against unsloth's Q5_K_M (the reference that fits the card) on five domains; then the
# finished 4B variants (a directory counts as finished when it has a DONE file or the queue log says so).
S=$HOME/forgeinfer/src/tools
M=$HOME/forgeinfer/oracle/models
E=$HOME/forgeinfer/data/eval
cd $HOME/forgeinfer/src
F=./target/release/forge
TOK=$M/Qwen__Qwen3.5-9B/tokenizer.json
N=${N:-12288}
EVAL=""; REFS=""
for d in en code zh de chat; do EVAL="$EVAL,$E/eval_$d.txt"; REFS="$REFS,$E/ref9_$d.bin"; done
EVAL=${EVAL#,}; REFS=${REFS#,}
echo "################ 9B ours, GPTQ in unsloth's formats"
$F ppl $M/forgeinfer__q9b_gptq --native --tokenizer $TOK --text $EVAL --max-tokens $N --ref $REFS 2>&1 | grep -E "RESULT|VS-REF|error|panick"
HF4=/mnt/data/forgeinfer-hf/Qwen__Qwen3.5-4B
EVAL=""; REFS=""
for d in en code zh de chat; do EVAL="$EVAL,$E/eval_$d.txt"; REFS="$REFS,$E/ref4_$d.bin"; done
EVAL=${EVAL#,}; REFS=${REFS#,}
for v in forgeinfer__q4b_gptq_h5 forgeinfer__q4b_gptq_small; do
  if [ -f $M/$v/DONE ]; then
    echo "################ 4B $v"
    ls -la $M/$v/model.safetensors | awk '{print "size", $5}'
    $F ppl $M/$v --native --tokenizer $HF4/tokenizer.json --text $EVAL --max-tokens 16384 --ref $REFS 2>&1 | grep -E "RESULT|VS-REF|error|panick"
  fi
done
