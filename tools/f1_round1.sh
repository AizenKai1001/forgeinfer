#!/bin/bash
# Forge-1 round 1: every structure option on its own against the plain proxy (same data stream, same hyperparameters), a second seed of the baseline for the noise floor, then all options together.
# Equal parameters where the option replaces something (ReLU squared with ffn 1536 has the parameters of SwiGLU with ffn 1024); the options that add tables (value embeddings) add parameters, listed in the summary.
#   tools/hl.sh -n "bash ~/forgeinfer/src/tools/pause_training.sh bash ~/forgeinfer/src/tools/f1_round1.sh"
cd ~/forgeinfer/src
exec bash tools/f1_ab.sh \
  "base|" \
  "base_s2|--seed 2" \
  "relu2|--relu2 --ffn 1536" \
  "untied|--untied" \
  "lambdas|--lambdas" \
  "skips|--skips" \
  "ve4|--value-emb 0,1,10,11" \
  "softcap20|--softcap 20" \
  "all|--untied --relu2 --ffn 1536 --softcap 20 --lambdas --skips --value-emb 0,1,10,11"
