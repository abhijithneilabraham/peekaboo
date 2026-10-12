#!/bin/bash
source .venv/bin/activate; export HF_HOME=$PWD/.hf OMP_NUM_THREADS=4
source guard.sh; wait_for_memory
run(){ python vtrain.py --env kuhn --updates 300 "$@" 2>&1 | grep done; }
for s in 0 1 2 3 4; do
  run --brain adapter --lr 1e-4 --seed $s --kl 1 --tag _L_kl1
  run --brain adapter --lr 1e-4 --seed $s --kl 3 --tag _L_kl3
  run --brain adapter --lr 1e-4 --seed $s --kl 1 --magnet 30 --tag _L_kl1m30
  run --brain adapter --lr 1e-4 --seed $s --kl 3 --magnet 30 --tag _L_kl3m30
  run --brain table --seed $s --tag _L_plain
  run --brain table --seed $s --kl 1 --tag _L_kl1
  run --brain table --seed $s --kl 1 --magnet 30 --tag _L_kl1m30
done
echo ALLDONE
