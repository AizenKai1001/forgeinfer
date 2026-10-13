#!/bin/bash
# Experiment 1: our quantizer against unsloth's Q4_K_M at the SAME per-tensor formats (so the file size is identical and only the quality of the encoder differs), Qwen3.5-0.8B.
# Quality = KL(reference || model) over the reference's top 64 next tokens, top-1 agreement, change of the log-likelihood and perplexity, on held-out English and Python code.
# The reference is the bf16 checkpoint run in fp16 (no quantization). Needs the whole GPU (see tools/pause_training.sh).   N = tokens per text (default 32768)
M=$HOME/forgeinfer/oracle/models
E=$HOME/forgeinfer/data/eval
cd $HOME/forgeinfer/src
F=./target/release/forge
HF=$M/Qwen__Qwen3.5-0.8B
G=$M/unsloth__Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf
TOK=$HF/tokenizer.json
N=${N:-32768}
head -c 450000 $E/calib_en.txt > $E/calib_mix.txt
printf '\n\n' >> $E/calib_mix.txt
head -c 250000 $E/calib_code.txt >> $E/calib_mix.txt
echo "######## importance statistics (calibration: English + code, 65536 tokens)"
$F imatrix $HF --text $E/calib_mix.txt --max-tokens 65536 --out $E/im_08b.bin 2>&1 | grep -E "imatrix|error|panick"
for text in en code; do
  echo "######## text: $text ($N tokens)"
  echo "--- reference (bf16 checkpoint as fp16)"
  $F ppl $HF --text $E/eval_$text.txt --max-tokens $N --dump-ref $E/ref_$text.bin 2>&1 | grep -E "RESULT|error|panick"
  echo "--- unsloth Q4_K_M (their encoder, their calibration)"
  $F ppl $G --native --tokenizer $TOK --text $E/eval_$text.txt --max-tokens $N --ref $E/ref_$text.bin 2>&1 | grep -E "RESULT|VS-REF|error|panick"
  for w in uniform magnitude importance impmag; do
    echo "--- ours, same formats, weighting $w"
    extra=""
    if [ $w = importance ] || [ $w = impmag ]; then extra="--imatrix $E/im_08b.bin"; fi
    $F ppl $HF --quant-like $G --weighting $w $extra --text $E/eval_$text.txt --max-tokens $N --ref $E/ref_$text.bin 2>&1 | grep -E "quant\]|RESULT|VS-REF|error|panick"
  done
done

bash $HOME/forgeinfer/src/tools/ncu_t2.sh
