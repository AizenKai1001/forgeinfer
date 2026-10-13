#!/bin/bash
# Where the time of a speculation cycle goes: the component timings of tests/verify2.rs::speculative_cycle_timing for the 4B and 9B, then the per-kernel breakdown (Nsight Systems) of the 9B.
# Needs the whole GPU:  tools/hl.sh -n "bash ~/forgeinfer/src/tools/pause_training.sh bash ~/forgeinfer/src/tools/spec_profile.sh"
M=$HOME/forgeinfer/oracle/models
cd $HOME/forgeinfer/src
T=$(ls -t target/release/deps/verify2-* | grep -v '\.d$' | head -n 1)
for q in ${*:-4B 9B}; do
  echo "######## Qwen3.5-$q"
  FORGE_VERIFY_MODEL=$M/unsloth__Qwen3.5-$q-GGUF/Qwen3.5-$q-Q4_K_M.gguf FORGE_VERIFY_HEAD=$M/Qwen__Qwen3.5-$q/mtp.safetensors $T speculative_cycle_timing --nocapture 2>&1 | grep -E "single-token|a cycle|panicked"
done
q=$(echo ${*:-4B 9B} | awk '{print $NF}')
echo "######## kernel breakdown, Qwen3.5-$q (all phases of the timing test together)"
FORGE_VERIFY_MODEL=$M/unsloth__Qwen3.5-$q-GGUF/Qwen3.5-$q-Q4_K_M.gguf FORGE_VERIFY_HEAD=$M/Qwen__Qwen3.5-$q/mtp.safetensors bash tools/nsys_cmd.sh $T speculative_cycle_timing --nocapture
