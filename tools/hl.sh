#!/bin/bash
# Run a command on the homelab (homelab) inside the ForgeInfer work tree, politely (nice 15), with the Rust and CUDA toolchains on the PATH.
#   tools/hl.sh "cargo build --release"        tools/hl.sh -n "nvidia-smi"   (-n = no nice, no cd: for quick look-ups)
#   tools/hl.sh -m 6 "cargo test ..."   (-m GB = cap the address space of the command: a runaway test dies alone instead of thrashing the whole machine; not for CUDA programs)
# The homelab hosts real services (Jellyfin, the Realm backend, n8n ...): keep jobs short, look at `uptime` and `nvidia-smi` before heavy GPU runs.
KEY="$HOME/.ssh/id_ed25519"
HOST="user@homelab"
NICE="nice -n 15"
CD="cd ~/forgeinfer/src 2>/dev/null || cd ~/forgeinfer;"
LIM=""
if [ "$1" = "-m" ]; then LIM="ulimit -v $(( $2 * 1048576 ));"; shift 2; fi
if [ "$1" = "-n" ]; then NICE=""; CD=""; shift; fi
ENVSETUP='. ~/.cargo/env 2>/dev/null; export PATH="$HOME/cuda-13.2.0/bin:$PATH"; export LD_LIBRARY_PATH="$HOME/cuda-13.2.0/lib64:$LD_LIBRARY_PATH";'
exec ssh -i "$KEY" -o BatchMode=yes -o IdentitiesOnly=yes -o ConnectTimeout=10 "$HOST" "$ENVSETUP $CD $NICE bash -c $(printf '%q' "$LIM $*")"
