#!/bin/bash
# Smoke test of `forge serve` over HTTP (run on the homelab). Usage: serve_smoke.sh [port] [model-name]
PORT=${1:-18080}
B=http://127.0.0.1:$PORT
hr() { echo; echo "=== $1"; }

hr "GET /health"
curl -s -i $B/health | head -n 12

hr "GET /v1/models"
curl -s $B/v1/models; echo

hr "GET /props"
curl -s $B/props; echo

hr "GET /nope (404 JSON)"
curl -s -i $B/ember/status | head -n 12

hr "POST /apply-template"
curl -s $B/apply-template -d '{"messages":[{"role":"user","content":"Hello there"}]}'; echo

hr "POST /tokenize"
curl -s $B/tokenize -d '{"content":"Hello, world"}'; echo

hr "chat, non-streaming, greedy"
curl -s $B/v1/chat/completions -d '{"messages":[{"role":"user","content":"What is the capital of France? Answer in one short sentence."}],"max_tokens":40,"temperature":0}'; echo

hr "chat, streaming, greedy, usage"
curl -s -N $B/v1/chat/completions -d '{"messages":[{"role":"user","content":"Count from one to five."}],"max_tokens":40,"temperature":0,"stream":true,"stream_options":{"include_usage":true}}'; echo

hr "text completion"
curl -s $B/v1/completions -d '{"prompt":"The capital of France is","max_tokens":12,"temperature":0}'; echo

hr "stop string"
curl -s $B/v1/chat/completions -d '{"messages":[{"role":"user","content":"Count from one to ten with commas."}],"max_tokens":60,"temperature":0,"stop":["5"]}'; echo

hr "bad JSON"
curl -s -i $B/v1/chat/completions -d '{"messages": [' | head -n 14

hr "context overflow"
python3 - <<'PYEOF' > /tmp/forge_big_prompt.json
import json
print(json.dumps({"messages": [{"role": "user", "content": "word " * 5000}], "max_tokens": 8}))
PYEOF
curl -s -i $B/v1/chat/completions -d @/tmp/forge_big_prompt.json | head -n 14

hr "prefix reuse: same conversation twice (second should report cached tokens)"
for i in 1 2; do
  curl -s $B/v1/chat/completions -d '{"messages":[{"role":"system","content":"You are a helpful assistant."},{"role":"user","content":"Name three colors."}],"max_tokens":16,"temperature":0}' | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['usage'], d['timings'])"
done
