#!/bin/bash
source .venv/bin/activate; export HF_HOME=$PWD/.hf OMP_NUM_THREADS=4
source guard.sh; wait_for_memory
for b in adapter engine; do for lr in 3e-4 1e-4 3e-5 1e-5; do for s in 0 1; do
  python vtrain.py --env kuhn --brain $b --lr $lr --seed $s --tag _lr$lr | tail -1
done; done; done
