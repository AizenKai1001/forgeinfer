#!/bin/bash
# Copy the cool-down shards from the homelab (tokens_cool, built by the cooldown-mix agent) into the staging folder of the Kaggle dataset forge-jax, under the names the JAX trainer reads with
#   --cool-data @code --cool-glob "*.cool"      (cool_NNN.cool shards, cool_heldout.val = the held-out file)
# The reads use O_DIRECT and idle IO priority so that the page cache of the running training (the warmed token pool) is not evicted and the disk is not disturbed.
#   bash tools/fetch_cool_shards.sh [first_shard=0] [last_shard=11]
KEY="$HOME/.ssh/id_ed25519"
HOST="user@homelab"
SRC="/mnt/data/forgeinfer-data/tokens_cool"
DST="<dev>/kaggle/stage/forge-jax"
FIRST=${1:-0}
LAST=${2:-11}
SSH="ssh -i $KEY -o BatchMode=yes -o IdentitiesOnly=yes -o ConnectTimeout=10 $HOST"
mkdir -p "$DST"
for i in $(seq $FIRST $LAST); do
  n=$(printf "%03d" "$i")
  want=$($SSH "stat -c %s $SRC/cool_$n.u16")
  $SSH "ionice -c3 nice -n 19 dd if=$SRC/cool_$n.u16 bs=4M iflag=direct status=none" > "$DST/cool_$n.cool"
  got=$(stat -c %s "$DST/cool_$n.cool")
  echo "cool_$n: $got of $want bytes"
  if [ "$got" != "$want" ]; then echo "SIZE MISMATCH for cool_$n"; exit 1; fi
done
want=$($SSH "stat -c %s $SRC/heldout/cool_val.u16")
$SSH "ionice -c3 nice -n 19 dd if=$SRC/heldout/cool_val.u16 bs=4M iflag=direct status=none" > "$DST/cool_heldout.val"
got=$(stat -c %s "$DST/cool_heldout.val")
echo "cool_heldout.val: $got of $want bytes"
[ "$got" = "$want" ] || { echo "SIZE MISMATCH for the held-out file"; exit 1; }
echo done
