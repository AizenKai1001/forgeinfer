#!/bin/bash
# GPU window 8: (1) the 4B quantized by us (GPTQ, unsloth's Q4_K_M formats) against unsloth's file, both measured against the Q8_0-weights reference on five domains; (2) everything of window 6b
# (fp16-accumulate prefill GEMMs, prefill against llama.cpp, server tests with speculation, MTP acceptance of our file, HellaSwag of the F125 checkpoint).
S=$HOME/forgeinfer/src/tools
M=$HOME/forgeinfer/oracle/models
E=$HOME/forgeinfer/data/eval
cd $HOME/forgeinfer/src
F=./target/release/forge
HF4=/mnt/data/forgeinfer-hf/Qwen__Qwen3.5-4B
N=${N:-16384}
EVAL=""; REFS=""
for d in en code zh de chat; do EVAL="$EVAL,$E/eval_$d.txt"; REFS="$REFS,$E/ref4_$d.bin"; done
EVAL=${EVAL#,}; REFS=${REFS#,}
for v in forgeinfer__q4b_gptq forgeinfer__q4b_gptq_h5 forgeinfer__q4b_gptq_small; do
  if [ -f $M/$v/DONE ] || [ "$v" = forgeinfer__q4b_gptq ]; then
    echo "################ $v"
    ls -la $M/$v/model.safetensors | awk '{print "size", $5}'
    $F ppl $M/$v --native --tokenizer $HF4/tokenizer.json --text $EVAL --max-tokens $N --ref $REFS 2>&1 | grep -E "RESULT|VS-REF|error|panick"
  fi
done
bash $S/window6b.sh
