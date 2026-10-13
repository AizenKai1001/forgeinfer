#!/bin/bash
# Draft acceptance (teacher forced, `forge mtp-agree`) of different versions of the 0.8B on the same held-out text: the more faithful the quantization, the more of the head's guesses are the model's own choice.
M=$HOME/forgeinfer/oracle/models
E=$HOME/forgeinfer/data/eval
cd $HOME/forgeinfer/src
F=./target/release/forge
HF=$M/Qwen__Qwen3.5-0.8B
T=${TEXT:-$E/eval_en.txt}
N=${N:-24576}
echo "text: $T, $N tokens"
for v in "unsloth__Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf" "forgeinfer__q08b_gptq" "unsloth__Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q8_0.gguf"; do
  nice -n 5 $F mtp-agree $M/$v --native --tokenizer $HF/tokenizer.json --mtp $HF --text $T --max-tokens $N 2>&1 | grep -E "mtp-agree|error|panick" | cut -c1-220
done
nice -n 5 $F mtp-agree $HF --tokenizer $HF/tokenizer.json --text $T --max-tokens $N 2>&1 | grep -E "mtp-agree|error|panick" | cut -c1-220
