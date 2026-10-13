#!/bin/bash
# GPU window 6b: prompt processing against llama.cpp at several prompt lengths and where its time goes, and the HTTP tests of the server with speculative decoding.
S=$HOME/forgeinfer/src/tools
M=$HOME/forgeinfer/oracle/models
cd $HOME/forgeinfer/src
F=./target/release/forge
echo "################ fused dequantizing GEMMs against dequantize-then-GEMM: agreement and speed (0.8B), then perplexity effect"
cargo test --release -p forge-gpu --test prefill_variants -- --nocapture 2>&1 | grep -E "fused =|logits:|test result|panicked|assert"
E=$HOME/forgeinfer/data/eval
G=$M/unsloth__Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf
for a in 1 0; do
  echo "-- unsloth 0.8B Q4_K_M, FORGE_NO_FUSED=$a"
  if [ $a = 1 ]; then export FORGE_NO_FUSED=1; else unset FORGE_NO_FUSED; fi
  $F ppl $G --native --tokenizer $M/Qwen__Qwen3.5-0.8B/tokenizer.json --text $E/eval_en.txt,$E/eval_code.txt --max-tokens 24576 --ref $E/ref3_en.bin,$E/ref3_code.bin 2>&1 | grep -E "RESULT|VS-REF"
done
unset FORGE_NO_FUSED
echo "################ prefill against llama.cpp at several prompt lengths"
source $HOME/forgeinfer/oracle/env.sh 2>/dev/null
for q in 4B 9B; do
  F2=$M/unsloth__Qwen3.5-$q-GGUF/Qwen3.5-$q-Q4_K_M.gguf
  for n in 512 1024 2048 4096; do
    echo "-- $q, $n tokens: forge"
    $F prefill-bench $F2 --native --tokens $n --reps 3 2>&1 | grep -E "best"
  done
  echo "-- $q: llama.cpp"
  llama-bench -m $F2 -ngl 99 -p 512,1024,2048,4096 -n 0 -r 2 2>&1 | grep -E "pp[0-9]+" | cut -c1-120
done
echo "################ where the prefill time goes (9B, 1024 tokens)"
bash $S/nsys_cmd.sh $F prefill-bench $M/unsloth__Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf --native --tokens 1024 --reps 2 2>&1 | grep -vE "^Generated|prof_cmd" | head -30
echo "################ the server with speculative decoding: HTTP tests (4B)"
bash $S/serve_up.sh 18092 unsloth__Qwen3.5-4B-GGUF/Qwen3.5-4B-Q4_K_M.gguf Qwen__Qwen3.5-4B/tokenizer.json qwen35-4B --native --ctx 8192 --mtp $M/Qwen__Qwen3.5-4B/mtp.safetensors > /dev/null 2>&1
for i in $(seq 1 60); do curl -s -o /dev/null http://127.0.0.1:18092/health && break; sleep 1; done
python3 tools/serve_test.py basic 18092 2>&1 | tail -n 12
python3 tools/serve_test.py tools 18092 2>&1 | tail -n 12
python3 tools/serve_cache_test.py 18092 2>&1 | tail -n 8
curl -s http://127.0.0.1:18092/props | head -c 600; echo
kill $(cat $HOME/forgeinfer/serve_18092.pid) 2>/dev/null
echo "################ acceptance of the MTP drafts: unsloth Q4_K_M against our GPTQ Q4_K_M of the same model (0.8B)"
for variant in "unsloth__Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf|Qwen__Qwen3.5-0.8B/tokenizer.json|unsloth" "forgeinfer__q08b_gptq|forgeinfer__q08b_gptq/tokenizer.json|ours-GPTQ"; do
  mdl=${variant%%|*}; rest=${variant#*|}; tk=${rest%%|*}; nm=${rest#*|}
  echo "-- $nm"
  bash $S/serve_up.sh 18093 $mdl $tk q08 --native --ctx 4096 --mtp $M/Qwen__Qwen3.5-0.8B/mtp.safetensors > /dev/null 2>&1
  for i in $(seq 1 60); do curl -s -o /dev/null http://127.0.0.1:18093/health && break; sleep 1; done
  BENCH_QUICK=1 python3 tools/serve_bench.py 18093 256 2>&1 | tail -n 2
  kill $(cat $HOME/forgeinfer/serve_18093.pid) 2>/dev/null
  sleep 3
done
echo "################ HellaSwag of the F125 checkpoint (own tokenizer, own trainer)"
cp ~/forgeinfer/runs/f125_muon_v1/ckpt_latest.bin ~/forgeinfer/tmp/f125_eval.bin
$F eval-hellaswag ~/forgeinfer/tmp/f125_eval.bin --tokenizer ~/forgeinfer/data/tokenizer32k.json --data ~/forgeinfer/data/hellaswag_val.jsonl --limit 4000 2>&1 | tail -n 4
grep -E "val" ~/forgeinfer/runs/f125_muon_v1/train.log | tail -n 3
