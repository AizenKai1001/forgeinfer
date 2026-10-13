#!/bin/bash
# tpu_arrival.sh [KERNEL] [TAG]: what to do the moment a Kaggle TPU session has ended (run from the repo root on the PC; DRY=1 only prints the commands).
#   1. download the kernel output FIRST (before any push to that kernel): logs and small files, then ckpt_latest.bin (the pre-decay checkpoint comes later, in step 3b)
#   2. print the log summary (attention line, speed, steps, validation, errors) and a JAX check of the checkpoint on the PC's CPU
#   3. put the checkpoint on Modal (volume forge-runs, folder /TAG) and run the evaluation ladders on a Modal T4 (about 3 minutes, a few cents each): full sets and the leak-free subsets (eval_pack_clean)
#   3b. the pre-decay checkpoint: download, upload, ladder (the like-for-like comparison with the un-annealed snapshots of the Rust run); NO_PRE=1 skips it
#   4. print the commands for the fine-tunes (decided by hand: the grid over learning rates, then mix2 at the winner)
# Needs: <dev>/kaggle/kg.sh (Kaggle CLI wrapper), the Modal client of the reasoner5b venv, a free credit or a spend limit on Modal.
KERNEL=${1:-YOUR_KAGGLE_USER/forge-tpu}
TAG=${2:-tpu1}
export MSYS_NO_PATHCONV=1 PYTHONUTF8=1 PYTHONIOENCODING=utf-8
MODAL=<dev>/research/neural-ai-from-scratch-20260919/reasoner5b/cloud/.venv/Scripts/modal.exe
KG=<dev>/kaggle/kg.sh
OUT=<dev>/kaggle/out/${KERNEL#*/}_$(date +%m%d_%H%M)
run() { echo "+ $*"; if [ -z "${DRY:-}" ]; then "$@"; fi; }
# The Kaggle client reads a whole file into memory before it writes it, and a dropped connection ("IncompleteRead") loses everything and leaves an empty file (2026-10-11: 3.2 of 4.4 GB, the second try worked).
# So: try up to 4 times and go on only with a file that has the size its own header promises.
fetch_ckpt() {   # fetch_ckpt FILE_PATTERN FILE_NAME
  local i f
  for i in 1 2 3 4; do
    run bash $KG kernels output "$KERNEL" -p "$OUT" --file-pattern "$1" --force
    [ -n "${DRY:-}" ] && return 0
    f=$(find "$OUT" -maxdepth 3 -name "$2" 2>/dev/null | head -n 1)
    if [ -n "$f" ] && python tools/ckpt_params_only.py "$f" --verify; then return 0; fi
    echo "download of $2 incomplete (attempt $i of 4)"
  done
  return 1
}
echo "=== 1. download $KERNEL -> $OUT $(date '+%I:%M %p')"
mkdir -p "$OUT"
run bash $KG kernels output "$KERNEL" -p "$OUT" --file-pattern '^(?!.*ckpt_.*[.]bin$).*'
fetch_ckpt 'ckpt_latest[.]bin$' ckpt_latest.bin || { echo "the checkpoint could not be downloaded completely: stop here"; exit 1; }
ls -la "$OUT" | head -n 30
echo "=== 2. log summary $(date '+%I:%M %p')"
echo "files:"; find "$OUT" -maxdepth 3 -type f -printf "%s %p\n" 2>/dev/null | head -n 20
cat "$OUT"/exit.txt 2>/dev/null
LOG=$(find "$OUT" -maxdepth 3 \( -name train.log -o -name "forge-tpu*.log" \) 2>/dev/null | head -n 1)
if [ -n "$LOG" ]; then
  grep -E "\[attn\]|\[train\] (data|model|structure|time|plan|checkpoint)|WARNING|Traceback|Error|error|watchdog" "$LOG" | head -n 25 | cut -c1-220
  echo "--- last lines"; tail -n 6 "$LOG" | cut -c1-220
fi
CK=$(find "$OUT" -maxdepth 3 -name ckpt_latest.bin 2>/dev/null | head -n 1)
echo "checkpoint: ${CK:-none} ($(ls -la "${CK:-/nonexistent}" 2>/dev/null | awk '{print $5}') bytes)"
echo "paired comparison with the Rust trainer (same recipe, first steps):  python tools/hop_compare.py <hop1 log.jsonl> $(find "$OUT" -maxdepth 3 -name log.jsonl | head -n 1) --every 500 --from-step 1000"
[ -n "${DRY:-}" ] || [ -n "$CK" ] || { echo "no checkpoint: stop here"; exit 1; }
echo "=== 2b. JAX check of the checkpoint on the PC's CPU: the same 64 windows of 512 tokens of the web file as the Rust ladder (the Rust ladder prints ppl.domains.web.nats; a fp16 overflow in the Rust engine of a bf16-trained model would show up as a difference) $(date '+%I:%M %p')"
if [ -n "$CK" ] && [ -x tpu/venv/Scripts/python.exe ] && [ -f scratch/eval_pack_clean_stage/cleanpresent/ppl/web.u16 ]; then
  run powershell.exe -NoProfile -File tools/run_low.ps1 tpu/venv/Scripts/python.exe tpu/eval_ppl_jax.py "$CK" --data scratch/eval_pack_clean_stage/cleanpresent/ppl/web.u16 --windows 64 --seq 512 --batch 4 --dtype f32   # below-normal priority: the PC may be running a game
fi
echo "=== 3. checkpoint to Modal, ladders $(date '+%I:%M %p')"
# the upload is the slow step: send the weights only (header + parameters, 4 bytes per parameter instead of 12; the 365M model: 1.46 GB instead of 4.4 GB). Everything after the pre-training reads only that prefix (fine-tune --init-from,
# ladders, probes, the chat engine: checked on 2026-10-11 with the 150M model, identical greedy answer). FULL_UPLOAD=1 sends the whole file (needed only to resume the pre-training)
UP="$CK"
if [ -z "${FULL_UPLOAD:-}" ] && [ -n "$CK" ]; then
  UP="${CK%.bin}_params.bin"
  run python tools/ckpt_params_only.py "$CK" "$UP" --check
fi
run "$MODAL" volume put forge-runs "$UP" "/$TAG/ckpt_latest.bin" --force
run "$MODAL" run cloud/modal_forge.py::ladder --name "$TAG" --label "$TAG" --approved-usd 1
run "$MODAL" run cloud/modal_forge.py::ladder --name "$TAG" --label "${TAG}clean" --extra="--data /data/eval_pack_clean --tasks core,lambada_openai,mmlu_cloze" --approved-usd 1
for L in "$TAG" "${TAG}clean"; do
  run "$MODAL" volume get forge-runs "/$TAG/ladder_$L/results.json" "scratch/ladder_${TAG}_$L.json" --force
done
if [ -z "${NO_PRE:-}" ]; then   # the snapshot before the learning-rate decay: the like-for-like comparison with the un-annealed snapshots of the Rust run
  echo "=== 3b. pre-decay checkpoint $(date '+%I:%M %p')"
  PRE=""
  if fetch_ckpt 'ckpt_predecay[.]bin$' ckpt_predecay.bin; then PRE=$(find "$OUT" -maxdepth 3 -name ckpt_predecay.bin 2>/dev/null | head -n 1); else echo "the pre-decay checkpoint could not be downloaded completely: skipped"; fi
  if [ -n "$PRE" ]; then
    UPRE="$PRE"
    if [ -z "${FULL_UPLOAD:-}" ]; then
      UPRE="${PRE%.bin}_params.bin"
      run python tools/ckpt_params_only.py "$PRE" "$UPRE" --check
    fi
    run "$MODAL" volume put forge-runs "$UPRE" "/${TAG}pre/ckpt_latest.bin" --force
    run "$MODAL" run cloud/modal_forge.py::ladder --name "${TAG}pre" --label "${TAG}pre" --approved-usd 1
    run "$MODAL" volume get forge-runs "/${TAG}pre/ladder_${TAG}pre/results.json" "scratch/ladder_${TAG}pre.json" --force
  fi
fi
echo "=== 4. fine-tunes (by hand): the structure of the TPU model has value embeddings in 8 layers; the grid runs the human mix at three learning rates in parallel (about 1.3 USD each)"
echo "  (the learning-rate optimum of the first TPU base was 2e-4: 1e-4 underfit, 5e-4 and 1e-3 overfit; for a new base try 1.5e-4 2e-4 3e-4; set ARRIVAL_STRUCT to print the structure of another model, e.g. the 365M one: --layers 24 --dim 1024 --heads 16 --kv 4 --ffn 4096 ... --value-emb 0,1,2,3,20,21,22,23)"
echo "  STRUCT=\"${ARRIVAL_STRUCT:---layers 30 --dim 576 --heads 9 --kv 3 --ffn 2304 --qk-norm --relu2 --lambdas --lr-scalar-mult 10 --value-emb 0,1,2,3,26,27,28,29 --lr-table-mult 100 --lr-emb-mult 3}\" bash cloud/modal_grid.sh $TAG human 2250 1.5e-4 2e-4 3e-4"
echo '  then for each run:  modal run cloud/modal_forge.py::checklist --name sft_'"$TAG"'_human_2250_LR --label NAME --approved-usd 0.6   and   ::ladder --name sft_'"$TAG"'_human_2250_LR --label NAME --approved-usd 1'
echo '  then mix2 only at the winning rate:  STRUCT=... bash cloud/modal_sft.sh '"$TAG"' mix2 4000 LR'
echo "done $(date '+%I:%M %p')"
