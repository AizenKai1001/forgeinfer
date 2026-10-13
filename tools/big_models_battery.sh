#!/bin/bash
# The tests that need the whole GPU (the Qwen3.5 4B and 9B): speed against llama.cpp, a conversation through the server with state snapshots, long contexts.
# Run it on the homelab with the trainer stopped (see tools/pause_training.sh).
M=$HOME/forgeinfer/oracle/models
cd $HOME/forgeinfer/src
source $HOME/forgeinfer/oracle/env.sh 2>/dev/null
for q in 4B 9B; do
  F=$M/unsloth__Qwen3.5-$q-GGUF/Qwen3.5-$q-Q4_K_M.gguf
  [ -f $F ] || continue
  echo "################ Qwen3.5-$q Q4_K_M"
  bash tools/bench_vs_llama.sh $F 2>&1 | grep -E "best|pp512|tg128"
  echo "-- server: a conversation (cached prompt tokens per turn)"
  bash tools/serve_up.sh 18090 unsloth__Qwen3.5-$q-GGUF/Qwen3.5-$q-Q4_K_M.gguf Qwen__Qwen3.5-$q/tokenizer.json qwen35-$q --native --ctx 8192 > /dev/null 2>&1
  python3 tools/serve_cache_test.py 18090 2>&1 | tail -n 8
  python3 tools/serve_bench.py 18090 192 2>&1 | tail -n 6
  kill $(cat $HOME/forgeinfer/serve_18090.pid) 2>/dev/null
  sleep 2
  echo "-- long context"
  for n in 8192 32768; do
    ./target/release/forge prefill-bench $F --native --tokens $n --reps 1 2>&1 | grep -E "run 0|depth"
  done
done
