#!/bin/bash
# Wait (on the homelab) until the GPU is idle enough for a clean benchmark: no other compute process and utilization below 10 %, for at most N seconds (default 120).
# The homelab also runs Jellyfin / tdarr transcodes; a benchmark that overlaps one is noisy. Exit status 1 if the GPU never became idle.
#   tools/gpuwait.sh [seconds] && tools/hl.sh "./bench/gpu_chain"
LIMIT="${1:-120}"
KEY="$HOME/.ssh/id_ed25519"
HOST="user@homelab"
exec ssh -i "$KEY" -o BatchMode=yes -o IdentitiesOnly=yes -o ConnectTimeout=10 "$HOST" "for i in \$(seq 1 $LIMIT); do apps=\$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .); util=\$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -n 1); if [ \"\$apps\" = 0 ] && [ \"\$util\" -lt 10 ]; then echo \"GPU idle after \$i s (apps=\$apps util=\$util%)\"; exit 0; fi; sleep 1; done; echo \"GPU still busy (apps=\$apps util=\$util%)\"; exit 1"
