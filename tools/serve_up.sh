#!/bin/bash
# Start `forge serve` detached on the homelab: serve_up.sh <port> <model-subpath> <tokenizer-subpath> <alias> [extra forge serve flags...]
# Paths are relative to ~/forgeinfer/oracle/models. Logs to ~/forgeinfer/serve_<port>.log, pid in ~/forgeinfer/serve_<port>.pid
PORT=$1; MODEL=$2; TOK=$3; ALIAS=$4; shift 4
M=$HOME/forgeinfer/oracle/models
cd $HOME/forgeinfer/src
setsid nohup nice -n 15 ./target/release/forge serve $M/$MODEL --tokenizer $M/$TOK --port $PORT --alias $ALIAS "$@" < /dev/null > $HOME/forgeinfer/serve_$PORT.log 2>&1 &
for i in $(seq 1 120); do ss -ltn 2>/dev/null | grep -q ":$PORT " && break; sleep 1; done   # (a 4B model needs more than 5 s before it listens: the pid file was empty and the window could not stop the server)
ss -ltnp 2>/dev/null | grep ":$PORT " | sed -E 's/.*pid=([0-9]+).*/\1/' > $HOME/forgeinfer/serve_$PORT.pid
cat $HOME/forgeinfer/serve_$PORT.log
echo "pid $(cat $HOME/forgeinfer/serve_$PORT.pid)"
