#!/bin/bash
# Runs on the homelab. Starts a chat engine for a Forge-1 checkpoint (forge serve, OpenAI-compatible, on 127.0.0.1:8190 only, with an API key) and the web chat page in front of it (tools/chat_proxy.py on port 8191,
# reachable on the LAN and the tailnet: http://192.168.1.x:8191/). The engine needs about 1.1 GB of GPU memory next to the training run (6.3 GB of the 8 GB) and loads the checkpoint in a few seconds.
#   bash ~/forgeinfer/start_chat.sh [CKPT]          stop: bash ~/forgeinfer/start_chat.sh stop
# The key is in ~/forgeinfer/chat.key (created once, mode 600; only the proxy reads it). The conversation lives in the browser tab; nothing is logged but the engine's own log (~/forgeinfer/chat_engine.log).
set -u
H="$HOME/forgeinfer"
if [ "${1:-}" = stop ]; then
  pkill -f "forge[^ ]* serve .* --port 8190" 2>/dev/null
  pkill -f "chat_proxy.py --port 8191" 2>/dev/null
  echo stopped
  exit 0
fi
CKPT="${1:-${CKPT:-/mnt/data/forgeinfer-data/ckpt_f1_v1/sft_demo_s10000_step300.bin}}"
BIN="${BIN:-$H/bin/forge-chat}"   # a stable copy of the engine build (cp ~/forgeinfer/src/target/release/forge ~/forgeinfer/bin/forge-chat after an engine change); it used to be the reviewer helper's build under agents/, which is deleted when the helper is cleaned up
TOK="$H/data/tokenizer_f1.json"
[ -f "$H/chat.key" ] || { (umask 077; openssl rand -hex 12 > "$H/chat.key"); }
KEY=$(cat "$H/chat.key")
pkill -f "forge[^ ]* serve .* --port 8190" 2>/dev/null
pkill -f "chat_proxy.py --port 8191" 2>/dev/null
sleep 1
# Q8 weights (forge serve --q8: +58 percent decode speed, 40 to 60 percent less GPU memory, perplexity +0.0002) when the models file says "q8": true; the engine without --q8 is the fallback when it does not come up
Q8=""
if [ -f "$H/chat_models.json" ] && python3 -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get("q8") else 1)' "$H/chat_models.json" 2>/dev/null; then Q8="--q8"; fi
launch_engine() {
  nohup nice -n 10 "$BIN" serve "$CKPT" --tokenizer "$TOK" --host 127.0.0.1 --port 8190 --ctx 1024 --alias forge-1 --chat chatml --api-key "$KEY" --temp 0.7 --top-k 40 --top-p 0.95 --repeat-penalty 1.1 $1 > "$H/chat_engine.log" 2>&1 &
}
launch_engine "$Q8"
MODELS=""
[ -f "$H/chat_models.json" ] && MODELS="--models-file $H/chat_models.json --engine-bin $BIN --tokenizer $TOK --current-ckpt $CKPT"
nohup python3 "$H/chat_proxy.py" --port 8191 --engine http://127.0.0.1:8190 --key-file "$H/chat.key" --run-dir "${RUN_DIR:-$H/runs/f1_v1}" $MODELS > "$H/chat_proxy.log" 2>&1 &
sleep 8
if [ -n "$Q8" ] && ! curl -s -m 3 -H "Authorization: Bearer $KEY" http://127.0.0.1:8190/v1/models > /dev/null; then
  echo "the Q8 engine did not come up: starting it without --q8"
  pkill -f "forge[^ ]* serve .* --port 8190" 2>/dev/null
  sleep 1
  launch_engine ""
  sleep 6
fi
echo "engine: $(tail -n 2 "$H/chat_engine.log" | tr '\n' ' ' | cut -c1-200)"
echo "page: http://$(hostname -I | awk '{print $1}'):8191/   training status: http://$(hostname -I | awk '{print $1}'):8191/status"
