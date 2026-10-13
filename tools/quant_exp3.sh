#!/bin/bash
# Experiment 3: robustness across domains. Calibration on a MIX (English, Python, Chinese, German, math, chat), evaluation on each domain separately: unsloth's Q4_K_M, importance-weighted rounding and
# GPTQ (both calibrated on the mix) at identical per-tensor formats, Qwen3.5-0.8B. Needs the whole GPU (see tools/pause_training.sh).   N = tokens per evaluation text (default 24576)
# SKIP_REF=1: the reference dumps ref3_*.bin of an earlier run are reused (and unsloth's file is not measured again)
M=$HOME/forgeinfer/oracle/models
E=$HOME/forgeinfer/data/eval
cd $HOME/forgeinfer/src
F=./target/release/forge
HF=$M/Qwen__Qwen3.5-0.8B
G=$M/unsloth__Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf
N=${N:-24576}
DOMS="en code zh de math chat"
EVAL=""; REFS=""
for d in $DOMS; do
  [ -f $E/eval_$d.txt ] || continue
  EVAL="$EVAL,$E/eval_$d.txt"; REFS="$REFS,$E/ref3_$d.bin"
done
EVAL=${EVAL#,}; REFS=${REFS#,}
python3 tools/mix_calib.py
echo "######## calibration statistics on the multi-domain mix"
$F imatrix $HF --text $E/calib_multi.txt --max-tokens 163840 --out $E/im_multi.bin 2>&1 | grep -E "imatrix\]|error|panick"
$F hessian $HF --text $E/calib_multi.txt --max-tokens 163840 --out $E/hess_multi.bin 2>&1 | grep -E "hessian\]|error|panick"
if [ -z "$SKIP_REF" ]; then
  echo "######## reference (bf16 checkpoint as fp16)"
  $F ppl $HF --text $EVAL --max-tokens $N --dump-ref $REFS 2>&1 | grep -E "RESULT|error|panick"
  echo "######## unsloth Q4_K_M"
  $F ppl $G --native --tokenizer $HF/tokenizer.json --text $EVAL --max-tokens $N --ref $REFS 2>&1 | grep -E "RESULT|VS-REF|error|panick"
fi
echo "######## ours: importance-weighted rounding (multi-domain imatrix)"
$F ppl $HF --quant-like $G --weighting importance --imatrix $E/im_multi.bin --text $EVAL --max-tokens $N --ref $REFS 2>&1 | grep -E "RESULT|VS-REF|error|panick"
for damp in 0.01 0.1; do
  echo "######## ours: GPTQ (multi-domain Hessians, damping $damp)"
  $F ppl $HF --quant-like $G --gptq $E/hess_multi.bin --gptq-damp $damp --text $EVAL --max-tokens $N --ref $REFS 2>&1 | grep -E "RESULT|VS-REF|error|panick"
done
