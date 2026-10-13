#!/bin/bash
# GPU window 4: (1) quantizer robustness across domains, (2) a model quantized by `forge quantize` loaded by the engine, (3) speculative output equality after the kernel changes, (4) prefill chunk length.
S=$HOME/forgeinfer/src/tools
M=$HOME/forgeinfer/oracle/models
E=$HOME/forgeinfer/data/eval
cd $HOME/forgeinfer/src
F=./target/release/forge
HF=$M/Qwen__Qwen3.5-0.8B
G=$M/unsloth__Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf
echo "################ 1. multi-domain quantizer experiment"
bash $S/quant_exp3.sh 2>&1
echo "################ 2. forge quantize, then load it"
Q=$M/forgeinfer__q08b_gptq
$F quantize $HF $Q --like $G --gptq $E/hess_multi.bin --gptq-damp 0.1 2>&1 | grep -E "quantize\]|quant\]|error|panick"
ls -la $Q | awk '{print $5, $9}' | head -12
$F ppl $Q --native --text $E/eval_en.txt,$E/eval_code.txt --max-tokens 24576 --ref $E/ref3_en.bin,$E/ref3_code.bin 2>&1 | grep -E "RESULT|VS-REF|error|panick"
bash $S/serve_up.sh 18091 forgeinfer__q08b_gptq forgeinfer__q08b_gptq/tokenizer.json q08b --native --ctx 4096 > /dev/null 2>&1
for i in $(seq 1 60); do curl -s -o /dev/null http://127.0.0.1:18091/health && break; sleep 1; done
curl -s http://127.0.0.1:18091/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"q08b","messages":[{"role":"user","content":"Name three primary colors, one short sentence."}],"max_tokens":60,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}' | python3 -c "import sys,json; d=json.load(sys.stdin); print('SERVED:', d['choices'][0]['message']['content'].strip()[:200]); print(d['timings'])"
kill $(cat $HOME/forgeinfer/serve_18091.pid) 2>/dev/null
sleep 3
echo "################ 3. speculative output against plain decoding (4B, 9B)"
for q in 4B 9B; do
  FORGE_SPEC_SIZE=$q cargo test --release -p forge-engine --test speculative -- --nocapture 2>&1 | grep -E "^prompt|tokens in|^sampled|stop string|test result|panicked|assert"
done
echo "################ 4. prefill chunk length"
for q in 4B 9B; do
  F2=$M/unsloth__Qwen3.5-$q-GGUF/Qwen3.5-$q-Q4_K_M.gguf
  for c in 1024 2048; do
    echo "-- $q, chunks of $c tokens"
    FORGE_PREFILL_CHUNK=$c $F prefill-bench $F2 --native --tokens 4096 --reps 3 2>&1 | grep -E "best"
  done
done
