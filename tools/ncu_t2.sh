#!/bin/bash
# Nsight Compute on the one-token and two-token streaming matvec kernels of the Qwen3.5-9B decode (inside a GPU window, see tools/pause_training.sh): duration, DRAM throughput, occupancy, stalls.
M=$HOME/forgeinfer/oracle/models
cd $HOME/forgeinfer/src
T=$(ls -t target/release/deps/verify2-* | grep -v '\.d$' | head -n 1)
MET="gpu__time_duration.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,l1tex__throughput.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active,launch__registers_per_thread,launch__grid_size,launch__block_size,launch__occupancy_limit_shared_mem,launch__occupancy_limit_registers,launch__shared_mem_per_block_dynamic,smsp__inst_executed.sum,smsp__warps_active.avg.per_cycle_active,"
MET+="smsp__warp_issue_stalled_barrier_per_warp_active.pct,smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct,smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct,smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct,smsp__warp_issue_stalled_wait_per_warp_active.pct,smsp__warp_issue_stalled_lg_throttle_per_warp_active.pct,smsp__warp_issue_stalled_not_selected_per_warp_active.pct,smsp__warp_issue_stalled_selected_per_warp_active.pct,smsp__warp_issue_stalled_dispatch_stall_per_warp_active.pct,smsp__warp_issue_stalled_no_instruction_per_warp_active.pct,smsp__warp_issue_stalled_branch_resolving_per_warp_active.pct"
run() {   # phase kernel-regex launches-to-skip
  echo "---- phase $1, kernel $2 (launch $3)"
  FORGE_TIMING_PHASE=$1 FORGE_VERIFY_MODEL=$M/unsloth__Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf FORGE_VERIFY_HEAD=$M/Qwen__Qwen3.5-9B/mtp.safetensors \
    sudo -n --preserve-env=FORGE_TIMING_PHASE,FORGE_VERIFY_MODEL,FORGE_VERIFY_HEAD $HOME/cuda-13.2.0/bin/ncu --clock-control none --kernel-name regex:"$2" --launch-skip $3 --launch-count 1 --metrics $MET $T speculative_cycle_timing --nocapture 2>&1 \
    | grep -E 'Duration|pct|\.sum|per_cycle|launch__' | grep -v '^run '
}
run single "mvqb_res_q6k_kernel" 40
run verify "mvt2_res_q6k_kernel" 40
run single "mvqb_res_q4k_kernel" 40
run verify "mvt2_res_q4k_kernel" 40
run verify "mvt2_out_q6k_kernel" 40
run single "headq_q6k_kernel_8" 40
