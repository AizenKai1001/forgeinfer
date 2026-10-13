#!/bin/bash
# The CPU quantization queue (one `forge quantize` at a time: it uses every core): wait for whatever is running, then
#   the 9B in unsloth's Q4_K_M formats (GPTQ), then the two smaller 4B recipes (Q5_K head; Q5_K head + more tensors at Q4_K).
cd $HOME/forgeinfer/src
wait_quiet() { while ps -eo args | grep -E "release/forge quantize" | grep -v grep > /dev/null; do sleep 10; done; }
wait_quiet
echo "== 9B main: $(date '+%I:%M:%S %p')"
bash tools/quant_big.sh 9B forgeinfer__q9b_gptq
echo "== 4B B, head Q5_K: $(date '+%I:%M:%S %p')"
bash tools/quant_big.sh 4B forgeinfer__q4b_gptq_h5 --override "token_embd=Q5_K"
echo "== 4B C, head Q5_K and ffn_down / attn_qkv / ssm_out / attn_v at Q4_K: $(date '+%I:%M:%S %p')"
bash tools/quant_big.sh 4B forgeinfer__q4b_gptq_small --override "token_embd=Q5_K;ffn_down=Q4_K;attn_qkv=Q4_K;ssm_out=Q4_K;attn_v=Q4_K"
echo "== done: $(date '+%I:%M:%S %p')"
