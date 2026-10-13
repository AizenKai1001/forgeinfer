#!/bin/bash
# Training GEMMs on Turing (branch f16acc): one-command interleaved A/B of the trainer's GEMM kernels against their variants, on THIS machine's GPU, for every Forge-1 production shape at micro 2
# (4096 tokens): forward qkv / attention output (+ fp32 residual) / MLP up (ReLU^2 epilogue) / MLP down (+ residual), data gradients D1 (ReLU^2 backward) / D2 / D3 / D4, the head (a chunk of 1024 rows).
#   part 1  `forge kbench --only e2`: e2 = the staged epilogue (FORGE_GEMM_E2, on by default for fp16 outputs and the residual epilogue), w144 = 128 x 144 tiles (FORGE_GEMM_W144, on by default for
#           the fp32 data gradients), e2g3 = column groups of 3 tiles in the tile order (experiment); every variant is checked bit for bit against the old kernel ('same').
#   part 2  `forge kbench --only f16acc`: fp16 accumulation (FORGE_F16ACC=P, off) and the diagnostics (8f32 control, 8skel without MMAs, *noepi without the epilogue).
# Meant for the homelab RTX 2060 SUPER while the production run is paused (on a shared GPU only the ratios mean something). About 450 MB of GPU memory, well under 2 minutes.
#   bash tools/bench_gemm.sh [FORGE_BINARY] [reps, default 30]
BIN="${1:-$HOME/forgeinfer/agents/f16acc/bin/forge}"
REPS="${2:-30}"
if [ ! -x "$BIN" ]; then
    echo "no forge binary at $BIN (build branch f16acc and pass its path)" >&2
    exit 1
fi
echo "=== $(date '+%I:%M:%S %p')  $BIN"
nvidia-smi --query-gpu=name,memory.used,utilization.gpu,power.draw,power.limit,clocks.sm --format=csv
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
timeout 55 "$BIN" kbench --only e2 --tokens 4096 --reps "$REPS" --e2v e2,w144,e2g3 | sed -e 's/ | /\n    /g'
timeout 55 "$BIN" kbench --only f16acc --tokens 4096 --reps "$REPS" --f16p 2,8,8f32,8skel,8noepi,8f32noepi | sed -e 's/ | /\n    /g'
echo "=== $(date '+%I:%M:%S %p')"
nvidia-smi --query-gpu=power.draw,clocks.sm,temperature.gpu --format=csv,noheader
