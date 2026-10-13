#!/bin/bash
# More files against a KL base that tools/llama_kl.sh already wrote (same text, same chunk count): tools/llama_kl_more.sh TEXT CHUNKS TEST1.gguf [TEST2.gguf ...]
source $HOME/forgeinfer/oracle/env.sh
export CUDA_VISIBLE_DEVICES=-1
TEXT=$1
CHUNKS=$2
shift 2
T=${THREADS:-6}
BASE=${KLD_BASE:-/mnt/data/forgeinfer-data/kld_base.bin}
for m in "$@"; do
  echo "== $(basename $m)"
  nice -n 15 llama-perplexity -m $m -f $TEXT -c 512 --chunks $CHUNKS -ngl 0 -t $T --kl-divergence-base $BASE --kl-divergence 2>&1 | grep -E "Mean +KLD|Median +KLD|99.0% +KLD|Same top p|Mean +PPL|Mean +Δp" | cut -c1-110
done
