#!/bin/bash
# Chat fine-tune of a base checkpoint on Modal (A100-40GB, about 1 hour and 2.5 USD for the mix2 recipe, 0.6 hours for the human-only mix).
#   bash cloud/modal_sft.sh BASE_NAME MIX [STEPS] [LR]
#     BASE_NAME  a folder of the forge-runs volume that holds ckpt_latest.bin (put it there first:  modal volume put forge-runs LOCAL.bin /BASE_NAME/ckpt_latest.bin)
#     MIX        human | mix2   (the data directories tokens_sft_human_replay / tokens_sft_mix2_replay of the forge-data volume)
#     STEPS      default 2250 (human, about 2 epochs) or 4000 (mix2);   LR default 3e-4
#   The structure flags must be those of the base model: set STRUCT for a base that is not the 150M v1 model (the TPU model has value embeddings in 8 layers:
#   STRUCT="--layers 30 --dim 576 --heads 9 --kv 3 --ffn 2304 --qk-norm --relu2 --lambdas --lr-scalar-mult 10 --value-emb 0,1,2,3,26,27,28,29 --lr-table-mult 100 --lr-emb-mult 3").
#   WARMUP (default 100; 20 for a 300-step probe). Result: /runs/sft_BASE_MIX_STEPS_LR/ckpt_latest.bin and train_stdout.log in the volume (modal volume get forge-runs sft_BASE_MIX_STEPS_LR/ckpt_latest.bin LOCAL.bin).
set -u
BASE="${1:?base folder name}"; MIX="${2:?human or mix2}"
case "$MIX" in human) STEPS_D=2250; DIR=tokens_sft_human_replay;; mix2) STEPS_D=4000; DIR=tokens_sft_mix2_replay;; *) echo "MIX is human or mix2"; exit 2;; esac
STEPS="${3:-$STEPS_D}"; LR="${4:-3e-4}"
export MSYS_NO_PATHCONV=1 PYTHONUTF8=1 PYTHONIOENCODING=utf-8
MODAL="<dev>/research/neural-ai-from-scratch-20260919/reasoner5b/cloud/.venv/Scripts/modal.exe"
STRUCT="${STRUCT:---layers 30 --dim 576 --heads 9 --kv 3 --ffn 2304 --qk-norm --relu2 --lambdas --lr-scalar-mult 10 --value-emb 0,1,28,29 --lr-table-mult 20 --lr-emb-mult 3}"
ARGS="--init-from /runs/$BASE/ckpt_latest.bin $STRUCT --seq 2048 --micro 8 --tokens-per-step 65536 --steps $STEPS --warmup ${WARMUP:-100} --decay-frac 0.85 --min-lr-frac 0.05 --optim muon --lr $LR --muon-lr $LR --muon-wd 0.01 --val-tokens 4000000 --eval-every 250 --eval-batches 12 --save-every 500 --seed 1"
cd "$(dirname "$0")/.." || exit 1
"$MODAL" run --detach cloud/modal_forge.py::train --name "sft_${BASE}_${MIX}_${STEPS}_${LR}" --gpu A100-40GB --args="$ARGS" --data "dir:$DIR" --max-hours 1.25 --approved-usd 3.5
