#!/bin/bash
# Laya-as-policy experiments. One process at a time, memory-gated, watchdog on.
cd "$(dirname "$0")"
export HF_HOME=$PWD/.hf OMP_NUM_THREADS=4 MIN_FREE=40 KILL_FREE=20 MAX_MB=3800
source ./guard.sh
./guard.sh watch >> logs_guard.txt 2>&1 &
WD=$!; trap 'kill $WD 2>/dev/null' EXIT
P=./.venv/bin/python
run(){ wait_for_memory; $P -u train.py "$@" || echo "FAILED $*"; }
# cheap baselines: tabular and from-scratch network, naive self-play vs opponent pool
for s in 0 1 2 3 4; do for pool in 0 1; do
  run --env kuhn --brain table --pool $pool --seed $s --updates 300 --eval_every 10
  run --env kuhn --brain mlp   --pool $pool --seed $s --updates 300 --eval_every 10
  run --env grid --brain mlp   --pool $pool --seed $s --updates 300 --batch 16 --eval_every 10
done; done
# Laya with no training
run --env kuhn --brain laya0
run --env grid --brain laya0
# Laya's decision head trained by RL
for s in 0 1 2; do run --env kuhn --brain laya --pool 1 --seed $s --updates 150 --eval_every 10 --lr 1e-4; done
for s in 0 1;   do run --env kuhn --brain laya --pool 0 --seed $s --updates 150 --eval_every 10 --lr 1e-4; done
for s in 0 1;   do run --env grid --brain laya --pool 1 --seed $s --updates 100 --batch 16 --eval_every 20 --lr 1e-4; done
run --env grid --brain laya --pool 0 --seed 0 --updates 100 --batch 16 --eval_every 20 --lr 1e-4
echo PIPELINE DONE
