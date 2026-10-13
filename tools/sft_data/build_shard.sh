#!/bin/bash
# build_shard.sh CORPUS.txt OUT.u16 : turn a NUL-separated ChatML corpus (tools/sft_data/safety_shard.py) into a training shard with the loss mask (bit 15 = not a target), on the homelab CPU (nice 15, one thread),
# and copy the finished shard back to the PC. Needs the key-based ssh channel of the homelab (see the memory note homelab-control-channel). Read-only for everything but /tmp/shard_* on the homelab.
#   bash tools/sft_data/build_shard.sh scratch/safety_v2_x4.txt scratch/safety_v2_x4.u16
set -eu
CORPUS="${1:?corpus.txt}"; OUT="${2:?out.u16}"
KEY="$HOME/.ssh/id_ed25519"; HOST="user@homelab"
SSHO="-i $KEY -o BatchMode=yes -o IdentitiesOnly=yes -o ConnectTimeout=10"
TAG="shard_$$"
scp $SSHO -q "$CORPUS" "$HOST:/tmp/$TAG.txt"
ssh $SSHO "$HOST" "bash -s" <<EOF
set -eu
F=\$HOME/forgeinfer/target-dev3/release/forge
TOK=\$HOME/forgeinfer/data/tokenizer_f1.json
nice -n 15 \$F tok-encode \$TOK /tmp/$TAG.txt --out /tmp/$TAG.raw.u16 --threads 1
nice -n 15 \$F chat-mask \$TOK /tmp/$TAG.raw.u16 --out /tmp/$TAG.u16
ls -la /tmp/$TAG.raw.u16 /tmp/$TAG.u16
EOF
scp $SSHO -q "$HOST:/tmp/$TAG.u16" "$OUT"
ssh $SSHO "$HOST" "rm -f /tmp/$TAG.txt /tmp/$TAG.raw.u16 /tmp/$TAG.u16"
ls -la "$OUT"
