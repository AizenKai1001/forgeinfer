#!/bin/bash
# A short continuation fine-tune ("patch") of an already chat-tuned checkpoint on Modal (A100-40GB, about 0.76 s per step of 65,536 tokens: 700 steps = 9 minutes = about 0.45 USD), in the background (modal run --detach).
#   bash cloud/modal_sft_patch.sh RUN_NAME BASE STEPS LR "DATA,LIST,..." [WARMUP]
#     RUN_NAME  result folder in the forge-runs volume
#     BASE      a folder of forge-runs that holds ckpt_latest.bin (the chat-tuned model to patch)
#     DATA      comma-separated shard paths in the forge-data volume; the LAST one must be the validation shard (zzz_val)
#   STRUCT must be the structure of BASE (default: the first TPU base, 150M with value embeddings in layers 0,1,28,29). Environment: MICRO (micro-batch, default 8; the 365M model needs 4 or less), MAXH (hour limit, default 0.5), APPROVED (the cost guard in USD, default 1.5; the run is refused if MAXH at the GPU price is above it).
#   Example (human mix replay + skills + safety on top of the human-written chat model):
#     bash cloud/modal_sft_patch.sh sft_patch1 sft_tpu1_human_2250_2e-4 700 1.5e-4 "tokens_sft_human_replay/a_replay.u16,tokens_sft_human_replay/sft_000.u16,tokens_sft_human_replay/sft_001.u16,tokens_skills/sft_skills.u16,tokens_safety2/sft_safety2.u16,tokens_sft_human_replay/zzz_val.u16"
set -u
NAME="${1:?run name}"; BASE="${2:?base folder}"; STEPS="${3:?steps}"; LR="${4:?learning rate}"; DATA="${5:?data list}"; WARMUP="${6:-40}"
export MSYS_NO_PATHCONV=1 PYTHONUTF8=1 PYTHONIOENCODING=utf-8
MODAL="<dev>/research/neural-ai-from-scratch-20260919/reasoner5b/cloud/.venv/Scripts/modal.exe"
STRUCT="${STRUCT:---layers 30 --dim 576 --heads 9 --kv 3 --ffn 2304 --qk-norm --relu2 --lambdas --lr-scalar-mult 10 --value-emb 0,1,28,29 --lr-table-mult 20 --lr-emb-mult 3}"
ARGS="--init-from /runs/$BASE/ckpt_latest.bin $STRUCT --seq 2048 --micro ${MICRO:-8} --tokens-per-step 65536 --steps $STEPS --warmup $WARMUP --decay-frac 0.85 --min-lr-frac 0.05 --optim muon --lr $LR --muon-lr $LR --muon-wd 0.01 --val-tokens 4000000 --eval-every 100 --eval-batches 12 --save-every 350 --seed 2"
cd "$(dirname "$0")/.." || exit 1
"$MODAL" run --detach cloud/modal_forge.py::train --name "$NAME" --gpu A100-40GB --args="$ARGS" --data "$DATA" --max-hours ${MAXH:-0.5} --approved-usd ${APPROVED:-1.5}
