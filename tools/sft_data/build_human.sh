#!/bin/bash
# build_human.sh: the whole build of tokens_sft_human (the human-written-only chat mix) on the homelab, step by step (see the research note "ForgeInfer - SFT Data Mix (human-written only)").
# Every step is a job of scripts/run_job_human.sh (nice 15, idle IO, one thread, logged to sft_work_human/logs/NAME.log) and is paused by scripts/guard_human.sh while the production run slows down.
# Run the steps one at a time (they are listed, not chained: look at each log, two jobs may run in parallel but never more than two python processes):
#   bash build_human.sh steps          print the commands
#   bash build_human.sh run NAME       start one step in the background
# The scripts live in sft_work_human/scripts (copied from tools/sft_data/ of the repository with put.sh); cwd of every job is sft_work_human.
W=/mnt/data/forgeinfer-data/sft_work_human
PY=$HOME/forgeinfer/venv/bin/python
S=$W/scripts
declare -A STEP
STEP[fetch]="$PY $S/fetch_human.py small shp se"
STEP[bench_index]="$PY $S/human_bench_index.py"
STEP[pools_a]="$PY $S/human_pools.py oasst2 dolly no_robots aya gsm8k hendrycks aqua apps mbpp rosetta squad coqa nq trivia cnn dialogsum wp csn shp"
STEP[pools_se]="$PY $S/human_pools.py se"
STEP[audit_pools]="$PY $S/human_audit.py pools"
STEP[audit_fn]="$PY $S/human_audit.py fnames --max-df 12"
STEP[plan]="$PY $S/human_assemble.py plan"
STEP[encode]="$PY $S/human_assemble.py encode"
STEP[audit_build]="$PY $S/human_audit.py build $W/build"
STEP[verify_build]="$PY $S/sft_verify.py $W/build --build $W/build --n 0"
STEP[publish]="$PY $S/human_assemble.py publish"
STEP[replay]="$PY $S/human_assemble.py replay"
STEP[verify]="$PY $S/sft_verify.py /mnt/data/forgeinfer-data/tokens_sft_human --build $W/build --n 0"
STEP[contam_scan]="$PY $S/human_contam.py scan"
STEP[contam_collect]="$PY $S/human_contam.py collect"
STEP[contam_report]="$PY $S/human_contam.py report"
STEP[manifest]="$PY $S/human_manifest.py"
ORDER="fetch bench_index pools_a pools_se audit_pools audit_fn plan encode audit_build verify_build publish replay verify contam_scan contam_collect contam_report manifest"
case "$1" in
  steps) for n in $ORDER; do echo "$n: ${STEP[$n]}"; done ;;
  run) n=$2; [ -z "${STEP[$n]}" ] && { echo "unknown step $n"; exit 1; }
       nohup setsid bash $S/run_job_human.sh $n ${STEP[$n]} > /dev/null 2>&1 &
       echo "started $n (log: $W/logs/$n.log)" ;;
  *) echo "usage: bash build_human.sh steps | run NAME"; exit 1 ;;
esac
