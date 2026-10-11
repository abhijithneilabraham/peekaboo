#!/bin/bash
# Laya as the hide-and-seek policy, same v2 setup as the MLP baselines (half scripted sparring partners).
cd "$(dirname "$0")"

export HF_HOME=$PWD/.hf OMP_NUM_THREADS=4 MIN_FREE=40 KILL_FREE=20 MAX_MB=4500
source ./guard.sh
./guard.sh watch >> logs_guard.txt 2>&1 &
WD=$!; trap 'kill $WD 2>/dev/null' EXIT
P=./.venv/bin/python
run(){ wait_for_memory; $P -u train.py "$@" || echo "FAILED $*"; }
run --env grid --brain laya --pool 1 --seed 0 --updates 100 --batch 32 --eval_every 20 --script_frac 0.5 --lr 1e-3 --tag _v2_lr1e-3
echo PIPELINE4 DONE
