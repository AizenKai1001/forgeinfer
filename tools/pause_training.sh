#!/bin/bash
# Stop the F125 training run right after a checkpoint (so that nothing is lost), run a command that needs the whole GPU, and resume the run afterwards (also if the command fails).
#   tools/hl.sh -n "bash ~/forgeinfer/src/tools/pause_training.sh bash ~/forgeinfer/src/tools/big_models_battery.sh"
RUN=${RUN:-f125_muon_v1}
LOG=$HOME/forgeinfer/runs/$RUN/train.log
# the trainer of THIS run (--out $HOME/forgeinfer/runs/$RUN): other `forge ... train` processes on the homelab (agents' test runs, other runs) are neither stopped nor taken for it by relaunch()
run_trainer_pids() { ps -eo pid,args | grep -E "/forge[^ /]* train .*--out $HOME/forgeinfer/runs/$RUN( |\$)" | grep -v grep | awk '{print $1}'; }
PID=$(run_trainer_pids)
if [ -n "$PID" ]; then
  # wait for the first steps after a checkpoint (the step number is 1 or 2 modulo the checkpoint interval SAVE_EVERY, default 100)
  for i in $(seq 1 400); do
    s=$(tail -n 1 $LOG | grep -o 'step *[0-9]*' | grep -o '[0-9]*')
    m=$(( s % ${SAVE_EVERY:-100} ))
    if [ "$m" = "1" ] || [ "$m" = "2" ]; then break; fi
    sleep 3
  done
  kill $PID
  sleep 3
  echo "trainer stopped at step $s"
fi
# Relaunch the trainer afterwards (the F125 run, or the one named by LAUNCH / RUN); if it dies at once (a leftover process of the window still holds GPU memory: 2026-10-09 an orphaned server kept it down for 18 minutes) try again every 30 s.
relaunch() {
  cd $HOME/forgeinfer/src
  for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do
    if [ -n "$(run_trainer_pids)" ]; then break; fi
    if [ -n "${LAUNCH:-}" ]; then
      RUN=$RUN bash $LAUNCH     # for example LAUNCH=tools/launch_f1.sh RUN=f1_v1 F1_ARGS="..." (the environment carries the settings)
    else
      RUN=$RUN OPTIM_ARGS="--optim muon --lr 1.5e-3 --muon-lr 1.5e-3 --muon-wd 0.05" bash tools/launch_f125.sh
    fi
    sleep 30
  done
}
trap relaunch EXIT
"$@"
