#!/bin/bash
# Per-kernel profile of one kind of step of the Qwen3.5 speculation machinery, each in its own process so that the tables do not mix:
#   tools/hl.sh -n "bash ~/forgeinfer/src/tools/pause_training.sh bash ~/forgeinfer/src/tools/spec_profile2.sh 9B"
# single: 136 single-token steps; verify: 121 two-token steps with rollback (+1 direct run); draft: 124 draft steps and 63 KV-only steps.
M=$HOME/forgeinfer/oracle/models
cd $HOME/forgeinfer/src
T=$(ls -t target/release/deps/verify2-* | grep -v '\.d$' | head -n 1)
q=${1:-9B}
for phase in single verify draft; do
  echo "######## Qwen3.5-$q: phase $phase"
  FORGE_TIMING_PHASE=$phase FORGE_VERIFY_MODEL=$M/unsloth__Qwen3.5-$q-GGUF/Qwen3.5-$q-Q4_K_M.gguf FORGE_VERIFY_HEAD=$M/Qwen__Qwen3.5-$q/mtp.safetensors bash tools/nsys_cmd.sh $T speculative_cycle_timing --nocapture 2>&1 | grep -vE "^Generated|prof_cmd"
done
