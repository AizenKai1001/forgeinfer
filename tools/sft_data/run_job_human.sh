#!/bin/bash
# run_job_human.sh NAME CMD [ARGS...]: run one job of the human-only SFT build politely: nice 15, idle IO class, one thread per library, output to sft_work_human/logs/NAME.log, its pid
# registered in ops/pids.txt so that guard_human.sh can pause it.  The command is exec'd (the pid stays the pid of the job).  Start it in the background:
#     nohup bash run_job_human.sh NAME python script.py ... > /dev/null 2>&1 &
# The command line of every job contains the work directory (the guard only touches processes whose command line contains 'sft_work_human').
W=/mnt/data/forgeinfer-data/sft_work_human
NAME=$1
shift
mkdir -p $W/logs $W/ops
echo $$ >> $W/ops/pids.txt
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=1 TOKENIZERS_PARALLELISM=true HF_HOME=$W/hf_home HF_HUB_DISABLE_TELEMETRY=1 PYTHONUNBUFFERED=1
cd $W
echo "$(date '+%F %T') start $NAME: $*" >> $W/logs/jobs.log
exec nice -n 15 ionice -c3 "$@" > $W/logs/$NAME.log 2>&1
