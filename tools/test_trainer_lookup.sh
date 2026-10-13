#!/bin/bash
# Regression test of the trainer lookup in tools/switch_f1_binary.sh (trainer_pid) and tools/pause_training.sh (run_trainer_pids): while another `forge ... train` process runs on the machine
# (an agent's test run, another run), only the trainer of $RUN (--out $HOME/forgeinfer/runs/$RUN) may be found. The old lookup (the regex '/forge[^ /]* train ' alone) also returned the other
# process: pause_training.sh then stopped it too and its relaunch() skipped the relaunch while it ran; switch_f1_binary.sh stopped the lowest pid, which can be the other process.
# Starts two fake trainers (a script named forge-fake that sleeps; the other one first, so it has the lower pid), checks both lookups, stops the fakes. Nothing else is touched.
#   bash tools/test_trainer_lookup.sh      (prints PASS or the failures; exit code 0 / 1)
set -u
# the scripts under test run on the homelab (Linux, procps ps); Git Bash's ps has no -o
ps -eo pid,args > /dev/null 2>&1 || { echo "SKIP: this ps has no -o (run the test on the homelab)"; exit 0; }
here=$(cd "$(dirname "$0")" && pwd)
tmp=$(mktemp -d)
printf '#!/bin/bash\nfor i in $(seq 1 60); do sleep 1; done\n' > "$tmp/forge-fake"
chmod +x "$tmp/forge-fake"
RUN="lookup_test_$$"
"$tmp/forge-fake" train --data x --out "$tmp/runs/other" --steps 1 &
other=$!
"$tmp/forge-fake" train --data x --out "$HOME/forgeinfer/runs/$RUN" --steps 1 &
mine=$!
sleep 0.5
fail=0
old=$(ps -eo pid,args | grep -E '/forge[^ /]* train ' | grep -v grep | awk '{print $1}' | grep -cE "^($other|$mine)$")
echo "the old lookup finds $old of the 2 fake trainers"
eval "$(grep -E '^trainer_pid\(\) ' "$here/switch_f1_binary.sh")"
got=$(trainer_pid)
[ "$got" = "$mine" ] || { echo "FAIL switch_f1_binary.sh trainer_pid: got '$got', want $mine (the other fake trainer is $other)"; fail=1; }
eval "$(grep -E '^run_trainer_pids\(\) ' "$here/pause_training.sh")"
got=$(run_trainer_pids | tr '\n' ' ' | sed 's/ $//')
[ "$got" = "$mine" ] || { echo "FAIL pause_training.sh run_trainer_pids: got '$got', want $mine (the other fake trainer is $other)"; fail=1; }
# a run whose name is a prefix of another run's name must not match it
RUN_SAVED=$RUN
RUN="lookup_test"
[ -z "$(run_trainer_pids)" ] || { echo "FAIL run_trainer_pids matches the run $RUN_SAVED for RUN=$RUN"; fail=1; }
kill "$other" "$mine" 2>/dev/null
wait 2>/dev/null
rm "$tmp/forge-fake"
rmdir "$tmp"
[ "$fail" = 0 ] && echo PASS
exit $fail
