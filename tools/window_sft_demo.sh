#!/bin/bash
# Runs on the homelab INSIDE a GPU window of the production run (tools/pause_training.sh ...): a short chat fine-tune of a snapshot of the base model on the SFT mix v2 (tools/f1_sft_mix2.sh with 300 steps), then a chat
# with eight fixed prompts through the KV-cache engine (forge chat). Everything goes to ~/forgeinfer/eval_results/sft_demo_NAME.log. The base snapshot is read only; the SFT run lives in ~/forgeinfer/runs/sft_demo_NAME.
#   NAME=s10000 CKPT=/mnt/data/forgeinfer-data/ckpt_f1_v1/ckpt_s10000.bin bash ~/forgeinfer/src/tools/pause_training.sh bash ~/forgeinfer/window_sft_demo.sh
set -u
NAME="${NAME:-demo}"
CKPT="${CKPT:-/mnt/data/forgeinfer-data/ckpt_f1_v1/ckpt_s10000.bin}"
BIN="${BIN:-$HOME/forgeinfer/agents/reviewer/bin/forge-fixed}"
STEPS="${SFT_STEPS:-300}"
OUT="$HOME/forgeinfer/runs/sft_demo_$NAME"
LOG="$HOME/forgeinfer/eval_results/sft_demo_$NAME.log"
TOK="$HOME/forgeinfer/data/tokenizer_f1.json"
mkdir -p "$HOME/forgeinfer/eval_results"
{
echo "=== SFT demo $NAME $(date '+%Y-%m-%d %I:%M:%S %p'): base $CKPT, $STEPS steps, binary $BIN"
RUN=f1_v1 CKPT="$CKPT" OUT="$OUT" FORGE_BIN="$BIN" SFT_STEPS="$STEPS" SFT_LR=3e-4 SFT_WARMUP=20 bash "$HOME/forgeinfer/src/tools/f1_sft_mix2.sh" 2>&1 | tail -n 30
CK="$OUT/run/ckpt_latest.bin"
echo "=== chat with the fine-tuned checkpoint $(date '+%I:%M:%S %p')"
for p in "Who are you?" "What is the capital of France?" "Write a haiku about the sea." "Explain photosynthesis in two sentences." "What is 17 + 25?" "Write a Python function that reverses a string." "Give me three tips for studying." "Tell me a joke."; do
  echo "--- PROMPT: $p"
  timeout 120 "$BIN" chat "$CK" --tokenizer "$TOK" --chat "$p" --tokens 120 --temp 0.7 --seq 512 2>&1 | tail -n 12
done
echo "=== end $(date '+%I:%M:%S %p')"
} > "$LOG" 2>&1
