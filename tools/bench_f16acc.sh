#!/bin/bash
# fp16-accumulate forward GEMMs (branch f16acc, FORGE_F16ACC=P): interleaved `forge kbench --only f16acc` on THIS machine's GPU:
#   1) the tensor-core rate with fp32 and with fp16 accumulators (register operands; GeForce Turing: fp16 accumulation runs at 2x),
#   2) every forward NT GEMM of Forge-1 at micro 2 (4096 tokens: qkv 4096x960x576, attention output 4096x576x576 + fp32 residual, MLP up 4096x2304x576 with the ReLU^2 epilogue,
#      MLP down 4096x576x2304 + fp32 residual), the trainer's fp32-accumulate kernel launch by launch interleaved with each variant:
#        2, 4, 8, 16      the fp16-accumulate kernels the trainer uses with FORGE_F16ACC=P (flush into fp32 every P k8 MMA steps)
#        8f32             the same code with fp32 accumulators (control: bit-identical results, rel diff 0)
#        8skel            the P = 8 kernel without any MMA (garbage output): the floor set by loads, ldmatrix, barriers and the epilogue
#        8noepi, 8f32noepi, 8skelnoepi   the same three without the epilogue's loads and stores (how much of the time is the epilogue)
#      Each line: old ms -> variant ms, TFLOPs, speed ratio, relative RMS difference of the outputs (fp16 outputs: u; residual outputs: C - R).
# Meant for the homelab RTX 2060 SUPER while the production run is paused (on a shared GPU only the ratios mean something). About 300 MB of GPU memory; well under 2 minutes.
#   bash tools/bench_f16acc.sh [FORGE_BINARY] [variant list] [reps, default 30]
BIN="${1:-$HOME/forgeinfer/agents/f16acc/bin/forge}"
PS="${2:-2,4,8,16,8f32,8skel,8noepi,8f32noepi,8skelnoepi}"
REPS="${3:-30}"
if [ ! -x "$BIN" ]; then
    echo "no forge binary at $BIN (build branch f16acc and pass its path)" >&2
    exit 1
fi
echo "=== $(date '+%I:%M:%S %p')  $BIN"
nvidia-smi --query-gpu=name,memory.used,utilization.gpu,power.draw,power.limit,clocks.sm --format=csv
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
timeout 115 "$BIN" kbench --only f16acc --tokens 4096 --reps "$REPS" --f16p "$PS" | sed -e 's/ | /\n    /g'
echo "=== exit ${PIPESTATUS[0]}  $(date '+%I:%M:%S %p')"
nvidia-smi --query-gpu=power.draw,clocks.sm,temperature.gpu --format=csv,noheader
