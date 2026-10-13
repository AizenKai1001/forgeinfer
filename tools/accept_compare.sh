#!/bin/bash
# Acceptance of the MTP drafts for different quantizations of the same 0.8B model (the draft head is the bf16 model's own): the served greedy/sampled 256-token runs of tools/serve_bench.py.
S=$HOME/forgeinfer/src/tools
M=$HOME/forgeinfer/oracle/models
cd $HOME/forgeinfer/src
PORT=${PORT:-18094}
for variant in "unsloth__Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf|Qwen__Qwen3.5-0.8B/tokenizer.json|unsloth Q4_K_M" "forgeinfer__q08b_gptq|forgeinfer__q08b_gptq/tokenizer.json|ours GPTQ (damp 0.1)" "unsloth__Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q8_0.gguf|Qwen__Qwen3.5-0.8B/tokenizer.json|unsloth Q8_0"; do
  mdl=${variant%%|*}; rest=${variant#*|}; tk=${rest%%|*}; nm=${rest#*|}
  echo "-- $nm"
  bash $S/serve_up.sh $PORT $mdl $tk q08 --native --ctx 4096 --mtp $M/Qwen__Qwen3.5-0.8B > /dev/null 2>&1
  for i in $(seq 1 60); do curl -s -o /dev/null http://127.0.0.1:$PORT/health && break; sleep 1; done
  BENCH_QUICK=1 python3 tools/serve_bench.py $PORT 256 2>&1 | tail -n 2 | cut -c1-200
  kill $(cat $HOME/forgeinfer/serve_$PORT.pid) 2>/dev/null
  for i in $(seq 1 30); do ss -ltn | grep -q ":$PORT " || break; sleep 1; done
done
