#!/bin/bash
# Profile a CUDA program on the homelab with Nsight Compute. The driver restricts GPU performance counters to admin users, so this runs ncu through sudo (read-only counters, no system change;
# reports go to /tmp/forge_ncu so that no root-owned files land in the work tree).
#   tools/ncu.sh <kernel-name-regex> <count> <program> [args...]     e.g.  tools/ncu.sh matvec_half 3 ./bench/gpu_chain
# Prints, per profiled kernel launch: duration, DRAM throughput %, SM throughput %, achieved occupancy %, registers per thread, and the top warp stall reasons.
KEY="$HOME/.ssh/id_ed25519"
HOST="user@homelab"
K="$1"; N="$2"; shift 2
METRICS="gpu__time_duration.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active,launch__registers_per_thread,smsp__average_warp_latency_issue_stalled_long_scoreboard.ratio,smsp__average_warp_latency_issue_stalled_barrier.ratio,smsp__average_warp_latency_issue_stalled_memory_throttle.ratio,smsp__average_warp_latency_issue_stalled_math_pipe_throttle.ratio"
exec ssh -i "$KEY" -o BatchMode=yes -o IdentitiesOnly=yes -o ConnectTimeout=10 "$HOST" ". ~/.cargo/env 2>/dev/null; cd ~/forgeinfer/src; mkdir -p /tmp/forge_ncu; sudo -n \$HOME/cuda-13.2.0/bin/ncu --kernel-name regex:$K --launch-count $N --metrics $METRICS $* 2>&1 | grep -v '^==PROF== \(Connected\|Disconnected\)'"
