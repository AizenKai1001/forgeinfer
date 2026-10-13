#!/bin/bash
# The human-written chat fine-tune with one more shard of safety conversations (about 28 minutes and 1.3 USD on an A100-40GB), in the background (modal run --detach).
#   bash cloud/modal_sft_safety.sh RUN_NAME SAFETY_SHARD [STEPS] [LR] [BASE]
#     RUN_NAME      folder of the forge-runs volume for the result, e.g. sft_tpu1_humansafe2_2250_2e-4
#     SAFETY_SHARD  path of the safety shard inside the forge-data volume, e.g. tokens_safety2/sft_safety2.u16 (tools/sft_data/safety_shard.py + build_shard.sh, then  modal volume put forge-data ...)
#     STEPS 2250, LR 2e-4, BASE tpu1 (the pre-trained model: a folder of forge-runs that holds ckpt_latest.bin)
#   STRUCT must be the structure of BASE (default: the first TPU base, 150M with value embeddings in layers 0,1,28,29).
#   Afterwards:  modal run cloud/modal_forge.py::checklist --name RUN_NAME --label chk --approved-usd 0.6  ,  ::probe ... --prompts-file scratch/safety_probe.json --label safety  ,  ::ladder ... --label lad
set -u
NAME="${1:?run name}"; SHARD="${2:?safety shard path in the data volume}"
STEPS="${3:-2250}"; LR="${4:-2e-4}"; BASE="${5:-tpu1}"
export MSYS_NO_PATHCONV=1 PYTHONUTF8=1 PYTHONIOENCODING=utf-8
MODAL="<dev>/research/neural-ai-from-scratch-20260919/reasoner5b/cloud/.venv/Scripts/modal.exe"
STRUCT="${STRUCT:---layers 30 --dim 576 --heads 9 --kv 3 --ffn 2304 --qk-norm --relu2 --lambdas --lr-scalar-mult 10 --value-emb 0,1,28,29 --lr-table-mult 20 --lr-emb-mult 3}"
ARGS="--init-from /runs/$BASE/ckpt_latest.bin $STRUCT --seq 2048 --micro 8 --tokens-per-step 65536 --steps $STEPS --warmup ${WARMUP:-100} --decay-frac 0.85 --min-lr-frac 0.05 --optim muon --lr $LR --muon-lr $LR --muon-wd 0.01 --val-tokens 4000000 --eval-every 250 --eval-batches 12 --save-every 500 --seed 1"
DATA="tokens_sft_human_replay/a_replay.u16,tokens_sft_human_replay/sft_000.u16,tokens_sft_human_replay/sft_001.u16,$SHARD,tokens_sft_human_replay/zzz_val.u16"
cd "$(dirname "$0")/.." || exit 1
"$MODAL" run --detach cloud/modal_forge.py::train --name "$NAME" --gpu A100-40GB --args="$ARGS" --data "$DATA" --max-hours 1.25 --approved-usd 3.5
