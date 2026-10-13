#!/bin/bash
# Wait until a Kaggle kernel has finished (COMPLETE, ERROR or CANCEL) or until MAX_MIN minutes passed, then print its status and exit (run it in the background: the exit wakes the session).
#   tools/wait_kaggle.sh YOUR_KAGGLE_USER/forge-ab2 [max_minutes=240]
REF="${1:?kernel ref, e.g. YOUR_KAGGLE_USER/forge-ab2}"
MAX_MIN="${2:-240}"
END=$(( $(date +%s) + MAX_MIN * 60 ))
while [ "$(date +%s)" -lt "$END" ]; do
  st=$(bash <dev>/kaggle/kg.sh kernels status "$REF" 2>&1 | tail -1)
  case "$st" in
    *COMPLETE*|*ERROR*|*CANCEL*) echo "$(date '+%I:%M %p') $st"; exit 0;;
  esac
  sleep 60
done
echo "$(date '+%I:%M %p') still running after $MAX_MIN minutes: $st"
