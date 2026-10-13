#!/bin/bash
# Relay check and window-switch test of the production run on Modal (needs 30 USD to be in the workspace; costs about 1 USD).
#
#   bash cloud/modal_relay_check.sh STEP          (STEP = a snapshot of the production run, for example 15000: ckpt_s15000.bin on the homelab)
#
# 1. copies the snapshot from the homelab to <repo>-data/relay/ (1.8 GB) and puts it twice into the Modal volume forge-runs (relay_dense, relay_window),
# 2. starts two detached training runs of 7 minutes each on an A100-40GB with the PRODUCTION flags of the run f1_v1 (micro 8 instead of 2: the same 65,536 tokens per step),
#    one dense (does Modal continue the run like the homelab does?) and one with --window 512 --window-layers auto (what does switching the running model to window attention cost, and does it recover?),
# 3. prints the commands that fetch the logs (volume forge-runs, relay_*/train_stdout.log) and compare them with the homelab's log of the same steps (tools/hop_compare.py).
# Nothing is billed until `modal run` is executed (each call prints its worst-case cost first and refuses without --approved-usd).
set -u
STEP="${1:?give the snapshot step, for example 15000}"
export MSYS_NO_PATHCONV=1 PYTHONUTF8=1 PYTHONIOENCODING=utf-8
MODAL="<dev>/research/neural-ai-from-scratch-20260919/reasoner5b/cloud/.venv/Scripts/modal.exe"
KEY="$HOME/.ssh/id_ed25519"
LOCAL="<dev>/forgeinfer-data/relay"
mkdir -p "$LOCAL"
SNAP="/mnt/data/forgeinfer-data/ckpt_f1_v1/ckpt_s${STEP}.bin"
scp -i "$KEY" "user@homelab:$SNAP" "$LOCAL/ckpt_s${STEP}.bin" || exit 1
for name in relay_dense relay_window; do
  "$MODAL" volume put forge-runs "$LOCAL/ckpt_s${STEP}.bin" "/$name/ckpt_latest.bin" || exit 1
done
# the production flags (see the Run Book); --micro 8 for the A100, no periodic evaluation, a 7 minute time limit; the checkpoint header carries the step, so --resume continues at STEP
COMMON="--layers 30 --dim 576 --heads 9 --kv 3 --ffn 2304 --seq 2048 --micro 8 --tokens-per-step 65536 --steps 45000 --warmup 100 --decay-frac 0.1 --min-lr-frac 0.05 --eval-every 100000 --eval-batches 8 --save-every 100000 --qk-norm --optim muon --lr 1.5e-3 --muon-lr 1.5e-3 --muon-wd 0.05 --relu2 --lambdas --lr-scalar-mult 10 --value-emb 0,1,28,29 --lr-table-mult 20 --lr-emb-mult 3 --time-limit-s 420"
cd "$(dirname "$0")/.." || exit 1
"$MODAL" run --detach cloud/modal_forge.py::train --name relay_dense --gpu A100-40GB --args="$COMMON" --max-hours 0.4 --approved-usd 1.5 &
"$MODAL" run --detach cloud/modal_forge.py::train --name relay_window --gpu A100-40GB --args="$COMMON --window 512 --window-layers auto" --max-hours 0.4 --approved-usd 1.5 &
wait
echo "fetch the logs:  $MODAL volume get forge-runs relay_dense/train_stdout.log <repo>-data/relay/dense.log   (same for relay_window)"
echo "compare:         python tools/hop_compare.py <homelab train.log copy> <repo>-data/relay/dense.log <repo>-data/relay/window.log --names homelab,modal_dense,modal_window --every 100"
