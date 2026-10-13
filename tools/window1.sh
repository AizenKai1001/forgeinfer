#!/bin/bash
# One GPU window: the per-phase kernel profiles of the 9B speculation machinery, then the quantization quality baselines of the 0.8B.
M=$HOME/forgeinfer/oracle/models
bash $HOME/forgeinfer/src/tools/spec_profile2.sh 9B
bash $HOME/forgeinfer/src/tools/quant_eval.sh $M/unsloth__Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q8_0.gguf $M/unsloth__Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf
