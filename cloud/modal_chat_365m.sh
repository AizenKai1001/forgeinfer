#!/bin/bash
# The single chat-tuning run of the 365M base on Modal (A100-40GB), the plan of the Run Book ("365M plan and budget", 2026-10-11): human-written replay mix + schoolbook arithmetic v3 + skills, no safety set (a safety patch comes afterwards only if money remains).
#   bash cloud/modal_chat_365m.sh [RUN_NAME] [STEPS] [LR] [BASE]       defaults: sft_tpu2_chat_2300_1.5e-4   2300   1.5e-4   tpu2
# BASE is a folder of the forge-runs volume that holds ckpt_latest.bin (tools/tpu_arrival.sh puts the 365M base there as /tpu2; a weights-only file is enough for --init-from). Start it only after the base was judged usable
# (log summary and ladders of tools/tpu_arrival.sh). Cost: measured 1.35 s per step of 65,536 tokens on 2026-10-11 (the estimate was 1.8) = about 52 minutes = about 2.1 USD; the guard APPROVED (default 3.9 USD) refuses the run if MAXH (default 1.6 h) at the GPU price
# (2.42 USD/h for the A100-40GB in cloud/modal_forge.py: 3.87 USD) is above it; with the earlier default of 3.5 USD the first launch was refused.
# Micro-batch 4 (the step has 32 sequences of 2,048 tokens; 8 may fit in 40 GB but was not tried). The structure flags are those of the Kaggle job (<dev>/kaggle/stage/forge-tpujob-b/settings.json); a mismatch makes the trainer stop at once with "checkpoint does not match this model".
NAME="${1:-sft_tpu2_chat_2300_1.5e-4}"; STEPS="${2:-2300}"; LR="${3:-1.5e-4}"; BASE="${4:-tpu2}"
export STRUCT="--layers 24 --dim 1024 --heads 16 --kv 4 --ffn 4096 --qk-norm --relu2 --lambdas --lr-scalar-mult 10 --value-emb 0,1,2,3,20,21,22,23 --lr-table-mult 100 --lr-emb-mult 3"
export MICRO="${MICRO:-4}" MAXH="${MAXH:-1.6}" APPROVED="${APPROVED:-3.9}"
R=tokens_sft_human_replay
DATA="$R/a_replay.u16,$R/sft_000.u16,$R/sft_001.u16,tokens_365/p_arith3.u16,tokens_365/p_skills.u16,$R/zzz_val.u16"
cd "$(dirname "$0")/.." || exit 1
exec bash cloud/modal_sft_patch.sh "$NAME" "$BASE" "$STEPS" "$LR" "$DATA" 60
