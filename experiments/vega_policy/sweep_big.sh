#!/bin/bash
source .venv/bin/activate; export HF_HOME=$PWD/.hf OMP_NUM_THREADS=4
source guard.sh; wait_for_memory
R(){ python bench_rl.py --items decisionbench_items_100.json --n_test 40 "$@" 2>&1 | grep -E "feedback temperature|Traceback|Error" | cut -c1-220; echo "done $*"; }
for s in 0 1 2; do
  R --method zero --n_train 60 --seed $s --tag _big
  for n in 5 15 30 60; do
    R --method bandit --kl 0.1 --n_train $n --seed $s --tag _big_n${n}_kl0.1
    R --method bandit --kl 1   --n_train $n --seed $s --tag _big_n${n}_kl1
    R --method sup    --kl 0.1 --n_train $n --seed $s --tag _big_n${n}_sup
  done
  R --method bandit --kl 0.1 --n_train 60 --epochs 1 --seed $s --tag _big_n60_kl0.1_online
  R --method bandit --kl 0.1 --split tasks --seed $s --tag _big_tasks_kl0.1
done
echo ALLDONE
