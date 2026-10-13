#!/bin/bash
# End-to-end: `forge serve` on a training checkpoint with the KV cache, then with FORGE_NO_KV=1 (the old path), the same streamed multi-turn greedy conversation against both; the replies must be identical.
#   tools/kv_serve_e2e.sh FORGE_BINARY CHECKPOINT TOKENIZER [CTX=1024] [MAX_TOKENS=64] [PORT=18777]      (OUT=dir for the logs, default /tmp/kvserve)
set -u
FORGE=$1; CK=$2; TOK=$3; CTX=${4:-1024}; MAXT=${5:-64}; PORT=${6:-18777}
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${OUT:-/tmp/kvserve}
mkdir -p "$OUT"
for mode in kv nokv; do
    if [ "$mode" = nokv ]; then export FORGE_NO_KV=1; else unset FORGE_NO_KV; fi
    "$FORGE" serve "$CK" --tokenizer "$TOK" --ctx "$CTX" --port "$PORT" > "$OUT/serve_$mode.log" 2>&1 &
    PID=$!
    for i in $(seq 1 180); do
        grep -q "ready" "$OUT/serve_$mode.log" && break
        kill -0 $PID 2>/dev/null || break
        sleep 1
    done
    echo "== $mode: $(grep -m1 ready "$OUT/serve_$mode.log")"
    python3 "$HERE/kv_serve_check.py" run "$PORT" "$OUT/$mode.json" "$MAXT"
    kill $PID
    wait $PID 2>/dev/null
done
python3 "$HERE/kv_serve_check.py" compare "$OUT/kv.json" "$OUT/nokv.json"
