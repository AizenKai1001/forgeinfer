#!/bin/bash
# GPU window 3: (1) the swizzled long-row kernels (timings + Nsight Compute), (2) GPTQ against importance-weighted rounding on the 0.8B, (3) how many tokens the draft head should score.
S=$HOME/forgeinfer/src/tools
echo "################ 1. cycle timings"
bash $S/spec_profile.sh 2>&1 | grep -E "^####|single-token|a cycle"
bash $S/ncu_t2.sh 2>&1 | grep -E "^----|gpu__time|dram__|l1tex__throughput|sm__warps|stalled_(long|short|mio)"
echo "################ 2. GPTQ"
bash $S/quant_exp2.sh 2>&1
echo "################ 3. draft head size"
ROWS="32768 65536 131072 248320" bash $S/draft_sweep.sh 2>&1
