#!/bin/bash
source .venv/bin/activate; export HF_HOME=$PWD/.hf OMP_NUM_THREADS=4
source guard.sh; wait_for_memory
run(){ python vtrain.py --env kuhn "$@" 2>&1 | grep done; }
for s in 0 1 2; do
  run --brain engine  --lr 1e-5 --seed $s --tag _base
  run --brain adapter --lr 1e-4 --seed $s --kl 0.3 --tag _kl0.3
  run --brain adapter --lr 1e-4 --seed $s --kl 1 --tag _kl1
  run --brain engine  --lr 1e-5 --seed $s --ent 0.1 --tag _ent0.1
  run --brain engine  --lr 1e-5 --seed $s --contrast 5 --tag _c5
  run --brain adapter --lr 1e-4 --seed $s --contrast 5 --kl 0.3 --tag _c5kl0.3
  run --brain engine  --lr 1e-5 --seed $s --contrast 5 --ent 0.1 --tag _c5ent0.1
  run --brain adapter --lr 1e-4 --seed $s --contrast 5 --kl 1 --ent 0.1 --tag _c5kl1ent0.1
done
