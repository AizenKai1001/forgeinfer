#!/bin/bash
# One shard of every new source, for the tokenizer sample and a look at the data (homelab, niced; text and raw shards go to the big drive).
cd ~/forgeinfer/src
for a in "cosmo 0" "fmath 0" "owmath 0" "wiki 0" "gut 0" "pycode 1"; do
  echo "[$(date +%H:%M:%S)] $a"
  nice -n 15 ~/forgeinfer/venv/bin/python tools/data_fetch.py $a 2>&1 | tail -n 3
done
echo "[$(date +%H:%M:%S)] done"
