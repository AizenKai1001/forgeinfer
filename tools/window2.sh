#!/bin/bash
# GPU window 2: (1) cycle timings of the speculation machinery on the 4B and 9B, (2) speculative vs plain decoding through the server, (3) the quantizer experiment and Nsight Compute of the matvec kernels.
S=$HOME/forgeinfer/src/tools
echo "################ 1. cycle timings"
bash $S/spec_profile.sh 2>&1 | grep -vE "^Generated|prof_cmd"
echo "################ 2. speculative decoding vs plain decoding"
bash $S/spec_battery.sh 4B 9B 2>&1
echo "################ 3. quantizer experiment + ncu"
bash $S/quant_exp1.sh 2>&1
