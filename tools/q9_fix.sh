#!/bin/bash
# The 9B with a plainly rounded input embedding (the head's second moments are the wrong objective for a lookup table): the directory (only token_embd is quantized again), two GGUFs (same formats; head at Q5_K),
# and llama.cpp's KL against the code base that tools/llama_kl.sh left in /mnt/data/forgeinfer-data/kld_base.bin.
cd ~/forgeinfer/src
M=$HOME/forgeinfer/oracle/models
E=$HOME/forgeinfer/data/eval
HF9=/mnt/data/forgeinfer-hf/Qwen__Qwen3.5-9B
G9=$M/unsloth__Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf
F="nice -n 15 ./target/release/forge"
$F quantize $HF9 $M/forgeinfer__q9b_gptq2 --like $G9 --gptq "$E/h9/hess9.*.bin" --gptq-damp 0.01 --reuse $M/forgeinfer__q9b_gptq --redo token_embd 2>&1 | grep -E "quantize\]|error|panick" | tail -n 2
$F quantize $HF9 $HOME/forgeinfer/tmp/ours_9b_gptq2.gguf --like $G9 --gguf --from-dir $M/forgeinfer__q9b_gptq2 --gptq "$E/h9/hess9.*.bin" --gptq-damp 0.01 2>&1 | grep -E "gguf\]|error|panick" | tail -n 2
$F quantize $HF9 $HOME/forgeinfer/tmp/ours_9b_gptq2_h5.gguf --like $G9 --gguf --from-dir $M/forgeinfer__q9b_gptq2 --gptq "$E/h9/hess9.*.bin" --gptq-damp 0.01 --override "^output.weight=Q5_K" 2>&1 | grep -E "gguf\]|error|panick" | tail -n 2
THREADS=3 bash tools/llama_kl_more.sh $E/eval_code.txt 8 $HOME/forgeinfer/tmp/ours_9b_gptq2.gguf $HOME/forgeinfer/tmp/ours_9b_gptq2_h5.gguf
