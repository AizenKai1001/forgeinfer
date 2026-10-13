#!/bin/bash
# Experiment 2: error-compensating quantization (GPTQ) against importance-weighted rounding at the same per-tensor formats, Qwen3.5-0.8B (reference dumps of experiment 1 are reused).
M=$HOME/forgeinfer/oracle/models
E=$HOME/forgeinfer/data/eval
cd $HOME/forgeinfer/src
F=./target/release/forge
HF=$M/Qwen__Qwen3.5-0.8B
G=$M/unsloth__Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf
N=${N:-32768}
echo "######## second moments of the layer inputs (calibration: English + code, 65536 tokens)"
$F hessian $HF --text $E/calib_mix.txt --max-tokens 65536 --out $E/hess_08b.bin 2>&1 | grep -E "hessian|error|panick"
ls -la $E/hess_08b.bin | awk '{print $5/1e6 " MB"}'
for text in en code; do
  echo "######## text: $text ($N tokens)"
  echo "--- ours, importance weighting (no error compensation)"
  $F ppl $HF --quant-like $G --weighting importance --imatrix $E/im_08b.bin --text $E/eval_$text.txt --max-tokens $N --ref $E/ref_$text.bin 2>&1 | grep -E "RESULT|VS-REF|error|panick"
  for damp in 0.01 0.1; do
    echo "--- ours, GPTQ (damping $damp)"
    $F ppl $HF --quant-like $G --gptq $E/hess_08b.bin --gptq-damp $damp --text $E/eval_$text.txt --max-tokens $N --ref $E/ref_$text.bin 2>&1 | grep -E "quant\]|RESULT|VS-REF|error|panick"
  done
done
