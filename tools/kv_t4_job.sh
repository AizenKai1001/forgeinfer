#!/bin/bash
# The decode-kv measurement job on a Lightning T4 (run from the extracted job folder: ./forge, ./f1v1_params.bin = header + parameters of the production checkpoint, ./kv_serve_check.py,
# ./kv_serve_e2e.sh; the tokenizer is in the studio). 1: parity on the real checkpoint (greedy, 4 prompts x 200 tokens) + old vs new tokens/s + the long-context cost; 2: ablations, run alternately
# with the default because the T4 clock drifts under its 70 W cap (matrix-vector staging, CUDA graph, the row path = the trainer's big GEMMs at M = 1, attention variants at a long context);
# 3: per-kernel profile of the fused step; 4: forge serve end to end (streamed chat, KV cache vs FORGE_NO_KV=1).
set -u
cd "$(dirname "$0")"
chmod +x ./forge
TOK=${TOK:-../tokenizer_f1.json}
CK=./f1v1_params.bin
P1="The history of the Roman Empire"
nvidia-smi --query-gpu=name,clocks.max.sm,clocks.max.mem,memory.total,driver_version --format=csv,noheader
echo "== 1. parity + timings"
./forge gen-check $CK --tokenizer $TOK --tokens 200 --ctx 2048 --bench --old-long 6 2>&1 | tee out1.txt
short() { ./forge gen-check $CK --tokenizer $TOK --tokens 192 --ctx 2048 --no-ref --prompts "$P1" 2>&1 | grep -E "cache only" | sed 's/.*decode/decode/'; }
long() { ./forge gen-check $CK --tokenizer $TOK --tokens 8 --ctx 2048 --no-ref --prompts "$P1" --bench --old-long 0 2>&1 | grep -E "long context" | sed 's/; old path.*//' | sed 's/.*KV-cache/KV-cache/'; }
echo "== 2a. matrix-vector kernels: warp-specialised (default) / batches of 8 tiles per warp (GEMV=3) / one tile at a time (GEMV=1), alternating"
for r in 1 2; do
    echo "default: $(short)"
    echo "GEMV=3:  $(FORGE_KV_GEMV=3 short)"
    echo "GEMV=1:  $(FORGE_KV_GEMV=1 short)"
done
echo "== 2b. CUDA graph off / row path"
echo "no graph: $(FORGE_KV_NOGRAPH=1 short)"
echo "row path: $(FORGE_KV_GENERIC=1 short)"
echo "default:  $(short)"
echo "== 2c. attention at a long context (1968 cached tokens)"
for a in auto fused split flash; do
    echo "$a: $(FORGE_KV_ATTN=$a long)"
done
echo "== 3. kernel profile (fused step, no graph, 64 tokens)"
FORGE_KPROF=1 FORGE_KV_NOGRAPH=1 ./forge gen-check $CK --tokenizer $TOK --tokens 64 --ctx 2048 --no-ref --prompts "$P1" 2>&1 | grep -E "^ *[0-9.]+ %|kernel time" | head -14
echo "== 4. serve end to end"
OUT=./serve_out bash ./kv_serve_e2e.sh ./forge $CK $TOK 2048 64 18777 2>&1 | tail -6
echo "== summary"
grep -E "IDENTICAL|DIFFER|short context|long context|PASSED|FAILED" out1.txt | cut -c1-260
