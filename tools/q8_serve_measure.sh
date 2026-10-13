#!/bin/bash
# Load time, GPU memory and decode / prefill speed of `forge serve` on a training checkpoint, fp16 against Q8 weights (run on the homelab; own port, own process, stopped at the end).
#   tools/q8_serve_measure.sh FORGE_BIN MODEL.bin [--q8] [PORT=18095] [CTX=2048]
# Prints the [serve] line of the engine (seconds until ready, MB of weights), the memory nvidia-smi attributes to the server process, and tools/serve_bench.py (decode tokens/s greedy and sampled,
# first-token time, prefill tokens/s of a ~1,300-token prompt).
BIN="${1:?usage: q8_serve_measure.sh FORGE_BIN MODEL.bin [--q8] [PORT] [CTX]}"
MODEL="${2:?model}"
MODE="${3:-}"
case "$MODE" in --q8) ;; *) MODE="";; esac
PORT="${4:-18095}"
CTX="${5:-2048}"
HERE="$(cd "$(dirname "$0")" && pwd)"
TOK="${TOK:-$HOME/forgeinfer/data/tokenizer_f1.json}"
LOG="$(mktemp /tmp/q8_serve_XXXXXX.log)"
"$BIN" serve "$MODEL" --tokenizer "$TOK" --host 127.0.0.1 --port "$PORT" --ctx "$CTX" --alias m --chat chatml $MODE > "$LOG" 2>&1 &
PID=$!
trap 'kill $PID 2>/dev/null; wait $PID 2>/dev/null' EXIT
for i in $(seq 1 240); do
  if curl -s "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; then break; fi
  if ! kill -0 $PID 2>/dev/null; then echo "the server died:"; tail -5 "$LOG"; exit 1; fi
  sleep 0.5
done
grep -E '\[serve\]|\[engine\]' "$LOG"
echo "GPU memory of the server process: $(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep "^$PID," | head -1)"
python3 "$HERE/serve_bench.py" "$PORT" 200
echo "GPU memory after the requests: $(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep "^$PID," | head -1)"
