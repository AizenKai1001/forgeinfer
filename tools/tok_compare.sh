#!/bin/bash
# Tokens per byte of two tokenizers on a 60 MB sample of every text source (homelab): tools/tok_compare.sh OLD.json NEW.json
cd ~/forgeinfer/src
D=/mnt/data/forgeinfer-data
for s in fwedu cosmo fmath owmath wiki gut pycode; do
  python3 tools/mix_text.py $D/tok_cmp.txt $s=60 > /dev/null
  a=$(./target/release/forge tok-encode $1 $D/tok_cmp.txt --out $D/tok_cmp.u16 --threads 6 2>&1 | tail -n 1 | grep -o '[0-9.]* tokens per byte')
  b=$(./target/release/forge tok-encode $2 $D/tok_cmp.txt --out $D/tok_cmp.u16 --threads 6 2>&1 | tail -n 1 | grep -o '[0-9.]* tokens per byte')
  echo "$s: old $a, new $b"
done
