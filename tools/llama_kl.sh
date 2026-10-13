#!/bin/bash
# An independent check of the quantizer with llama.cpp's own perplexity tool (the oracle, CPU only so it can run next to the GPU jobs): the KL divergence of GGUF files against a reference GGUF on the same
# text, computed by llama.cpp's kernels and tokenizer, not ours.
#   tools/llama_kl.sh TEXT CHUNKS REF.gguf TEST1.gguf [TEST2.gguf ...]        (chunks of 512 tokens; THREADS=6 by default)
source $HOME/forgeinfer/oracle/env.sh
export CUDA_VISIBLE_DEVICES=-1   # even with -ngl 0 the CUDA backend takes GPU memory for big batches: keep it off the card (the GPU jobs of the other windows need it)
TEXT=$1
CHUNKS=$2
REF=$3
shift 3
T=${THREADS:-6}
BASE=${KLD_BASE:-/mnt/data/forgeinfer-data/kld_base.bin}
nice -n 15 llama-perplexity -m $REF -f $TEXT -c 512 --chunks $CHUNKS -ngl 0 -t $T --kl-divergence-base $BASE > /tmp/llama_kl_base.log 2>&1
tail -n 2 /tmp/llama_kl_base.log | cut -c1-160
for m in "$@"; do
  echo "== $(basename $m)"
  nice -n 15 llama-perplexity -m $m -f $TEXT -c 512 --chunks $CHUNKS -ngl 0 -t $T --kl-divergence-base $BASE --kl-divergence 2>&1 | grep -E "Mean +KLD|Median +KLD|99.0% +KLD|Same top p|Mean +PPL|Mean +Δp" | cut -c1-110
done
