#!/bin/bash
# Forge-1 training data (homelab, niced): fetch each shard, encode it with the Forge-1 tokenizer to a uint16 shard, delete the text. One shard at a time. The sampling of `forge train` is uniform over all
# tokens, so the number of shards per source sets the mixture.   Plan (about 6.5 B tokens): FineWeb-Edu 4 shards (49 %), Cosmopedia 3 (13 %), FineMath 2 (10 %), OpenWebMath 2 (4 %), Wikipedia 2 (5 %),
# Gutenberg books 3 (6 %), Python code 3 (13 %).    Then the validation shard from shards no training shard overlaps: tokens_f1/zzz_val.u16 (a proportional mixture of 7 sources).
#   tools/f1_data.sh            tools/f1_data.sh val
cd ~/forgeinfer/src
T=~/forgeinfer/data/tokenizer_f1.json
D=/mnt/data/forgeinfer-data
OUT=$D/tokens_f1
mkdir -p $OUT
PY="nice -n 15 $HOME/forgeinfer/venv/bin/python"
F="nice -n 15 ./target/release/forge"
if [ "${1:-}" != "val" ]; then
  for spec in "fwedu 0" "fwedu 1" "fwedu 2" "fwedu 3" "cosmo 0" "cosmo 1" "cosmo 2" "fmath 0" "fmath 1" "owmath 0" "owmath 1" "wiki 0" "wiki 1" "gut 0" "gut 1" "gut 2" "pycode 0" "pycode 1" "pycode 2"; do
    set -- $spec
    nn=$(printf "%03d" $2)
    if [ -s $OUT/$1_$nn.u16 ]; then echo "$1 $2 already encoded"; continue; fi
    echo "[$(date +%H:%M:%S)] $1 $2: fetch"
    $PY tools/data_fetch.py $1 $2 | tail -n 1
    echo "[$(date +%H:%M:%S)] $1 $2: encode"
    $F tok-encode $T $D/text/$1_$nn.txt --out $OUT/$1_$nn.u16 --threads 6 2>&1 | tail -n 1
    rm $D/text/$1_$nn.txt
  done
else
  V=$D/text_val
  mkdir -p $V
  for spec in "fwedu 13" "cosmo 103" "fmath 127" "owmath 113" "wiki 40" "gut 51" "pycode 53"; do
    set -- $spec
    [ -s $V/$1_$(printf "%03d" $2).txt ] || $PY tools/data_fetch.py $1 $2 --out $V | tail -n 1
  done
  python3 tools/mix_text.py $D/val_mix.txt fwedu=60,cosmo=20,fmath=14,owmath=6,wiki=10,gut=10,pycode=20 --dir $V
  $F tok-encode $T $D/val_mix.txt --out $OUT/zzz_val.u16 --threads 6 2>&1 | tail -n 1
fi
echo "[$(date +%H:%M:%S)] done"
