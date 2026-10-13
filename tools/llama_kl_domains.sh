#!/bin/bash
# llama.cpp's KL check (tools/llama_kl.sh) for each evaluation text: reference GGUF, then the test GGUFs.
#   tools/llama_kl_domains.sh CHUNKS REF.gguf TEST1.gguf [TEST2.gguf ...]
E=$HOME/forgeinfer/data/eval
CHUNKS=$1
shift
for d in ${DOMAINS:-en code zh de chat}; do
  echo "######## $d"
  bash $HOME/forgeinfer/src/tools/llama_kl.sh $E/eval_$d${SUFFIX}.txt $CHUNKS "$@"
done
