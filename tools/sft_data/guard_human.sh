#!/bin/bash
# guard_human.sh: keeps the jobs of the human-only SFT build from slowing the production training run (same rule as contam/guard.py, in bash so that it is not a third python process).
# Every 10 s: median of the last 16 step times of ~/forgeinfer/runs/f1_v1/train.log.  Above LIMIT (4.2 s) for more than 60 s -> SIGSTOP to the jobs listed in ops/pids.txt (only processes whose command line
# contains 'sft_work_human'); the jobs stay paused until the median is below RESUME (4.05 s) for 30 s, or for at most 120 s when the slowdown does not go away while they are paused (then it was not them).
# Everything is logged to ops/guard.log.  Start:  nohup bash guard_human.sh > /dev/null 2>&1 &   Stop: kill the guard pid in ops/guard.pid (it also exits an hour after the last job has finished).
LOG=${TRAIN_LOG:-$HOME/forgeinfer/runs/f1_v1/train.log}
OPS=/mnt/data/forgeinfer-data/sft_work_human/ops
PIDS=$OPS/pids.txt
GLOG=$OPS/guard.log
LIMIT=4.2
RESUME=4.05
echo $$ > $OPS/guard.pid
say() { echo "$(date '+%F %T') $*" >> "$GLOG"; }
median() {
  if [ ! -r "$LOG" ]; then echo NA; return; fi
  age=$(( $(date +%s) - $(stat -c %Y "$LOG") ))
  if [ "$age" -gt 120 ]; then echo NA; return; fi
  tail -c 6000 "$LOG" | grep -o '([0-9]*\.[0-9]* s)' | tail -16 | tr -d '() s' | sort -n | awk '{a[NR]=$1} END {if (NR>=8) print a[int((NR+1)/2)]; else print "NA"}'
}
my_pids() {
  for p in $(cat "$PIDS" 2>/dev/null); do
    if [ -r /proc/$p/cmdline ] && tr '\0' ' ' < /proc/$p/cmdline | grep -q "sft_work_human"; then echo $p; fi
  done
}
sig_all() { for p in $(my_pids); do kill -$1 $p 2>/dev/null; for c in $(pgrep -P $p 2>/dev/null); do kill -$1 $c 2>/dev/null; done; done; }  # the job and its direct children (forge tok-encode, forge chat-mask)
above_since=0
paused_at=0
ok_since=0
say "guard started (pid $$)"
idle_since=$(date +%s)
while true; do
  now=$(date +%s)
  med=$(median)
  pids=$(my_pids | tr '\n' ' ')
  if [ -n "$pids" ]; then idle_since=$now; fi
  if [ "$paused_at" -eq 0 ]; then
    if [ "$med" != "NA" ] && [ -n "$pids" ] && awk -v m="$med" -v l="$LIMIT" 'BEGIN{exit !(m>l)}'; then
      [ "$above_since" -eq 0 ] && above_since=$now
      if [ $((now - above_since)) -gt 60 ]; then
        sig_all STOP; paused_at=$now; ok_since=0; above_since=0
        say "PAUSED jobs [$pids]: median step time $med s > $LIMIT s for 60 s"
      fi
    else
      above_since=0
    fi
  else
    if [ "$med" != "NA" ] && awk -v m="$med" -v r="$RESUME" 'BEGIN{exit !(m<r)}'; then
      [ "$ok_since" -eq 0 ] && ok_since=$now
      if [ $((now - ok_since)) -gt 30 ]; then
        sig_all CONT; say "RESUMED jobs [$pids]: step time back to $med s ($((now - paused_at)) s pause)"; paused_at=0; ok_since=0
      fi
    else
      ok_since=0
      if [ $((now - paused_at)) -gt 120 ] && { [ "$med" = "NA" ] || awk -v m="$med" -v l="$LIMIT" 'BEGIN{exit !(m>l)}'; }; then
        sig_all CONT; say "RESUMED jobs [$pids] after 120 s: the slowdown (median $med) did not go away while paused, so it is not caused by them"; paused_at=0; above_since=$now
      fi
    fi
  fi
  if [ -z "$pids" ] && [ "$paused_at" -eq 0 ] && [ $((now - idle_since)) -gt 3600 ]; then say "no jobs for an hour: guard exits"; exit 0; fi
  sleep 10
done
