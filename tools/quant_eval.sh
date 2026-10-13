#!/bin/bash
# Quality of the quantized Qwen3.5-0.8B files against the fp16 reference (bf16 checkpoint weights): perplexity of held-out English and code, KL divergence of the next-token distribution, top-1 agreement.
# Needs the GPU (see tools/pause_training.sh).   N = tokens per text (default 65536)
M=$HOME/forgeinfer/oracle/models
E=$HOME/forgeinfer/data/eval
cd $HOME/forgeinfer/src
F=./target/release/forge
TOK=$M/Qwen__Qwen3.5-0.8B/tokenizer.json
N=${N:-65536}
for text in en code; do
  echo "######## text: $text"
  echo "--- reference: bf16 checkpoint as fp16"
  $F ppl $M/Qwen__Qwen3.5-0.8B --text $E/eval_$text.txt --max-tokens $N --dump-ref $E/ref_08b_$text.bin 2>&1 | grep -E "RESULT|error|panick"
  for f in "$@"; do
    echo "--- $f"
    $F ppl $f --native --tokenizer $TOK --text $E/eval_$text.txt --max-tokens $N --ref $E/ref_08b_$text.bin 2>&1 | grep -E "RESULT|VS-REF|error|panick"
  done
done
