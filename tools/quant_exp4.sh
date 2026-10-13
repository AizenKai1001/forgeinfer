#!/bin/bash
# Experiment 4: size / quality frontier of per-tensor format recipes with GPTQ (multi-domain Hessians), Qwen3.5-0.8B. Starting point: unsloth's Q4_K_M formats (the template); each recipe changes some tensors.
# Needs the reference dumps ref3_*.bin of experiment 3 (same N) and the whole GPU.   RECIPES_AT="name|override;override@name2|override..."
M=$HOME/forgeinfer/oracle/models
E=$HOME/forgeinfer/data/eval
cd $HOME/forgeinfer/src
F=./target/release/forge
HF=$M/Qwen__Qwen3.5-0.8B
G=$M/unsloth__Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf
N=${N:-24576}
DOMS=${DOMS:-"en code zh de chat"}
EVAL=""; REFS=""
for d in $DOMS; do
  [ -f $E/ref3_$d.bin ] || continue
  EVAL="$EVAL,$E/eval_$d.txt"; REFS="$REFS,$E/ref3_$d.bin"
done
EVAL=${EVAL#,}; REFS=${REFS#,}
RECIPES_AT=${RECIPES_AT:-"base|@head-Q5_K|token_embd=Q5_K@head-Q4_K|token_embd=Q4_K@down-Q4_K|ffn_down=Q4_K@head-Q5_K+down-Q4_K|token_embd=Q5_K;ffn_down=Q4_K@qkv+ssm_out-Q4_K|attn_qkv=Q4_K;ssm_out=Q4_K@head-Q5_K+all-Q4_K|token_embd=Q5_K;ffn_down=Q4_K;attn_qkv=Q4_K;ssm_out=Q4_K;attn_v=Q4_K"}
IFS='@' read -ra recs <<< "$RECIPES_AT"
for r in "${recs[@]}"; do
  name=${r%%|*}; ov=${r#*|}
  echo "######## recipe $name  (override: ${ov:-none})"
  $F ppl $HF --quant-like $G --gptq $E/hess_multi.bin --gptq-damp 0.1 --override "$ov" --text $EVAL --max-tokens $N --ref $REFS 2>&1 | grep -E "quant\] [0-9]+ matrices|RESULT|VS-REF|error|panick"
done
