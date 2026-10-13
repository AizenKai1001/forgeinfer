#!/bin/bash
# contam_loop_human.sh [MAX_ROUNDS]: remove the conversations that tools/contamination_check.py finds in the FINISHED token files of tokens_sft_human, until a scan finds none.
# One round = plan (the blacklist blacklist_contam.txt is read) -> encode -> publish -> replay -> scan -> collect (verified 16-token windows and sentence-like short texts, every occurrence) -> report -> collect_spans
# (the documents with a STRONG hit at 10 tokens); a round that adds nothing to the blacklist ends the loop with the files of that round as the final ones.
# A hit window that occurs several times in a file only reports its first position (collect finds the other occurrences by itself); the removed conversations stay in the pool as 'dead' records during the
# deduplication (human_assemble.py), so that their near duplicates do not take their place.
# Run as one job:  nohup setsid bash run_job_human.sh contam_loop bash /mnt/data/forgeinfer-data/sft_work_human/scripts/contam_loop_human.sh 6
W=/mnt/data/forgeinfer-data/sft_work_human
PY=$HOME/forgeinfer/venv/bin/python
S=$W/scripts
MAX=${1:-5}
cd $W || exit 1
for r in $(seq 1 $MAX); do
  echo "=== round $r: plan, encode, publish, replay, scan, collect, report, collect_spans ($(date '+%I:%M %p'))"
  $PY $S/human_assemble.py plan > logs/loop_plan_$r.log 2>&1 || { echo "plan failed"; exit 1; }
  $PY $S/human_assemble.py encode > logs/loop_encode_$r.log 2>&1 || { echo "encode failed"; exit 1; }
  $PY $S/human_assemble.py publish > logs/loop_publish_$r.log 2>&1 || { echo "publish failed"; exit 1; }
  $PY $S/human_assemble.py replay > logs/loop_replay_$r.log 2>&1 || { echo "replay failed"; exit 1; }
  rm $W/contam_human/scan/human__* 2>/dev/null   # the scan skips a file whose json says it was scanned with the same size: always rescan
  $PY $S/human_contam.py scan > logs/loop_scan_$r.log 2>&1 || { echo "scan failed"; exit 1; }
  $PY $S/human_contam.py collect > logs/loop_collect_$r.log 2>&1 || { echo "collect failed"; exit 1; }
  $PY $S/human_contam.py report > logs/loop_report_$r.log 2>&1 || { echo "report failed"; exit 1; }
  $PY $S/human_contam.py collect_spans > logs/loop_spans_$r.log 2>&1 || { echo "collect_spans failed"; exit 1; }
  n=$($PY -c "import json; print(json.load(open('$W/contam_human.json'))['final']['conversations_flagged_this_round'])")
  m=$($PY -c "import json; print(json.load(open('$W/contam_human_spans.json'))[-1]['new_ids'])")
  echo "round $r: $n conversations flagged by windows / short texts, $m new ones with a strong 10-token hit"
  if [ "$n" = "0" ] && [ "$m" = "0" ]; then echo "clean after $r round(s)"; exit 0; fi
done
echo "stopped after $MAX rounds with hits left"
