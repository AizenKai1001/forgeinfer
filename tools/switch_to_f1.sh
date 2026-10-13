#!/bin/bash
# Hand the homelab GPU from the running F125 control run to a Forge-1 run: wait until F125 has just written a checkpoint, stop it (nothing is lost; `RUN=f125_muon_v1 tools/launch_f125.sh` resumes it later), and start the
# Forge-1 run with the settings in the environment (see tools/launch_f1.sh). The binary is copied first, so that later builds cannot touch the file a days-long run executes.
#   RUN=f1_v1 FFN=2304 STEPS=45000 F1_ARGS="--relu2 --lambdas ..." bash tools/switch_to_f1.sh
OLD=${OLD_RUN:-f125_muon_v1}
LOG=$HOME/forgeinfer/runs/$OLD/train.log
mkdir -p $HOME/forgeinfer/bin
SRC=${FORGE_SRC_BIN:-$HOME/forgeinfer/target-dev3/release/forge}
DST=$HOME/forgeinfer/bin/forge-${RUN:-f1_v1}
cp "$SRC" "$DST"
PID=$(ps -eo pid,args | grep 'release/forge train' | grep -v grep | awk '{print $1}' | head -n 1)
if [ -n "$PID" ]; then
  for i in $(seq 1 400); do
    s=$(tail -n 1 $LOG | grep -o 'step *[0-9]*' | grep -o '[0-9]*')
    m=$(( s % ${SAVE_EVERY:-100} ))
    if [ "$m" = "1" ] || [ "$m" = "2" ]; then break; fi
    sleep 3
  done
  kill $PID
  sleep 5
  echo "stopped $OLD at step $s"
fi
cd $HOME/forgeinfer/src
FORGE_BIN=$DST bash tools/launch_f1.sh
