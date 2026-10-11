#!/bin/bash
# Remaining Kuhn runs with Laya as the policy. Memory-gated, watchdog on.
cd "$(dirname "$0")"
export HF_HOME=$PWD/.hf OMP_NUM_THREADS=4 MIN_FREE=40 KILL_FREE=20 MAX_MB=3800
source ./guard.sh
./guard.sh watch >> logs_guard.txt 2>&1 &
WD=$!; trap 'kill $WD 2>/dev/null' EXIT
P=./.venv/bin/python
run(){ wait_for_memory; $P -u train.py "$@" || echo "FAILED $*"; }
for s in 0 1 2; do run --env kuhn --brain laya --pool 1 --seed $s --updates 150 --eval_every 10 --lr 1e-4; done
for s in 0 1;   do run --env kuhn --brain laya --pool 0 --seed $s --updates 150 --eval_every 10 --lr 1e-4; done
echo PIPELINE DONE
