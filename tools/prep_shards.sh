#!/bin/bash
# On the homelab: download FineWeb-Edu shards, encode them to uint16 token files with our tokenizer, delete the raw text (the disk is small). One shard at a time, politely (nice 15).
#   tools/prep_shards.sh 1 2 3        (shards 0..13 exist)
# Needs: ~/forgeinfer/data/tokenizer32k.json, the release build of the forge CLI, the oracle venv (pyarrow) for the download tool.
set -u
DATA=~/forgeinfer/data
SRC=~/forgeinfer/src
for n in "$@"; do
  nn=$(printf "%03d" "$n")
  if [ -s "$DATA/tokens/fwedu_$nn.u16" ]; then echo "shard $n already encoded"; continue; fi
  echo "[$(date +%H:%M:%S)] shard $n: download"
  nice -n 15 ~/forgeinfer/venv/bin/python "$SRC/tools/data_fetch.py" fwedu "$n" || { echo "download failed"; exit 1; }
  echo "[$(date +%H:%M:%S)] shard $n: encode"
  nice -n 15 "$SRC/target/release/forge" tok-encode "$DATA/tokenizer32k.json" "$DATA/text/fwedu_$nn.txt" --out "$DATA/tokens/fwedu_$nn.u16" --threads 6 2>&1 | tail -n 2
  rm "$DATA/text/fwedu_$nn.txt"
  echo "[$(date +%H:%M:%S)] shard $n done"
done
