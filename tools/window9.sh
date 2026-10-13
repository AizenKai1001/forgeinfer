#!/bin/bash
# GPU window 9: (1) the weight pipeline (dequantization of the next matrix on a second stream) against the serial path, prefill of 512 / 1024 / 4096 tokens on the 4B and 9B; (2) what window 8 could not do:
# the acceptance of the MTP drafts for unsloth's and our 0.8B file, and HellaSwag of the F125 checkpoint.
S=$HOME/forgeinfer/src/tools
M=$HOME/forgeinfer/oracle/models
cd $HOME/forgeinfer/src
F=./target/release/forge
echo "################ prefill: weight pipeline on against off (FORGE_NO_PIPE=1)"
for q in 4B 9B; do
  F2=$M/unsloth__Qwen3.5-$q-GGUF/Qwen3.5-$q-Q4_K_M.gguf
  for n in 512 1024 4096; do
    echo "-- $q, $n tokens, serial dequantization"
    FORGE_NO_PIPE=1 $F prefill-bench $F2 --native --tokens $n --reps 4 2>&1 | grep -E "best"
    echo "-- $q, $n tokens, pipelined"
    $F prefill-bench $F2 --native --tokens $n --reps 4 2>&1 | grep -E "best"
  done
done
echo "################ acceptance of the MTP drafts: unsloth Q4_K_M against our GPTQ Q4_K_M of the same model (0.8B)"
for variant in "unsloth__Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf|Qwen__Qwen3.5-0.8B/tokenizer.json|unsloth" "forgeinfer__q08b_gptq|forgeinfer__q08b_gptq/tokenizer.json|ours-GPTQ"; do
  mdl=${variant%%|*}; rest=${variant#*|}; tk=${rest%%|*}; nm=${rest#*|}
  echo "-- $nm"
  bash $S/serve_up.sh 18093 $mdl $tk q08 --native --ctx 4096 --mtp $M/Qwen__Qwen3.5-0.8B/mtp.safetensors > /dev/null 2>&1
  for i in $(seq 1 60); do curl -s -o /dev/null http://127.0.0.1:18093/health && break; sleep 1; done
  BENCH_QUICK=1 python3 tools/serve_bench.py 18093 256 2>&1 | tail -n 2
  kill $(cat $HOME/forgeinfer/serve_18093.pid) 2>/dev/null
  for i in $(seq 1 30); do ss -ltn | grep -q ":18093 " || break; sleep 1; done
done
echo "################ HellaSwag of the F125 checkpoint (own tokenizer, own trainer)"
cp ~/forgeinfer/runs/f125_muon_v1/ckpt_latest.bin ~/forgeinfer/tmp/f125_eval.bin
$F eval-hellaswag ~/forgeinfer/tmp/f125_eval.bin --tokenizer ~/forgeinfer/data/tokenizer32k.json --data ~/forgeinfer/data/hellaswag_val.jsonl --limit 4000 2>&1 | tail -n 4
grep -E "val" ~/forgeinfer/runs/f125_muon_v1/train.log | tail -n 3
echo "################ leftovers (should be none)"
ps -eo pid,etime,args | grep -E "[r]elease/forge serve" | cut -c1-120
