#!/bin/bash
source .venv/bin/activate; export HF_HOME=$PWD/.hf OMP_NUM_THREADS=4
source guard.sh; wait_for_memory
C(){ python compare.py "$@" 2>&1 | grep -E "_s[0-9]|Traceback|Error" | cut -c1-260; }
for s in 0 1 2; do
  C --method feedback --seed $s
  C --method sup_ts --seed $s
  C --method linear_fb --seed $s
  C --method feedback --kl 1 --seed $s --tag _kl1
done
echo ALLDONE
