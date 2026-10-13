#!/bin/bash
# Regression test of tools/branch_at_40500.sh. Reads the script only: starts, stops and copies nothing.
#  1. accept_header takes the checkpoint written at loop step AT although its header lags by the updates skipped (non-finite gradients) since the trainer was started (a tiny run: log
#     "checkpoint at step 300", header "step":281 after 19 skipped updates), and refuses the neighbouring checkpoints (AT - 100, AT + 100) and garbage;
#  2. the header is checked BEFORE the trainer is stopped. The first version stopped the trainer and then aborted on any header other than AT: one skipped update before step 40,500 left
#     the main run down and started no branch.
#   bash tools/test_branch_at_40500.sh      (prints PASS or the failures; exit code 0 / 1)
set -u
here=$(cd "$(dirname "$0")" && pwd)
s="$here/branch_at_40500.sh"
fail=0
eval "$(grep -E '^accept_header\(\) ' "$s")"
check() {
  local got
  if accept_header "$1" "$2" 2>/dev/null; then got=yes; else got=no; fi
  [ "$got" = "$3" ] || { echo "FAIL accept_header $1 '$2': $got, want $3"; fail=1; }
}
check 40500 40500 yes
check 40500 40499 yes
check 40500 40401 yes
check 300 281 yes
check 40500 40400 no
check 40500 40501 no
check 40500 40600 no
check 40500 "" no
check 40500 abc no
first_check=$(grep -n 'accept_header "$AT" "$hdr"' "$s" | head -n 1 | cut -d: -f1)
first_kill=$(grep -n 'kill "$pid"' "$s" | head -n 1 | cut -d: -f1)
if [ -z "$first_check" ] || [ -z "$first_kill" ] || [ "$first_check" -ge "$first_kill" ]; then
  echo "FAIL the header must be checked before the trainer is stopped (check at line ${first_check:-none}, stop at line ${first_kill:-none})"
  fail=1
fi
[ "$fail" = 0 ] && echo PASS
exit $fail
