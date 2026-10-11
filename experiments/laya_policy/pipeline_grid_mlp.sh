#!/bin/bash
# Grid baselines v2: longer training, half the batches against scripted opponents, dense early evals.
cd "$(dirname "$0")"
export HF_HOME=$PWD/.hf OMP_NUM_THREADS=1
P=./.venv/bin/python
for s in 0 1 2 3 4; do
  for pool in 0 1; do
    $P -u train.py --env grid --brain mlp --pool $pool --seed $s --updates 3000 --batch 32 --eval_every 100 --dense_until 200 --script_frac 0.5 --tag _v2 > /dev/null 2>&1 &
  done; wait
done
echo GRID MLP DONE
