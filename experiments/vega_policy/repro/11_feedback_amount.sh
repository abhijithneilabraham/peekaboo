#!/bin/bash
# Section 5.5, "how much feedback": 5/15/30/60 feedback rows per task, fixed 1,720-row test set
# (40 rows per task from the 100-per-task pool). Ours (KL 0.1 and 1) and the supervised adapter at each size.
source "$(dirname "$0")/_common.sh"
R(){ python bench_rl.py --items decisionbench_items_100.json --n_test 40 "$@" | grep -E "feedback temperature|skip" || true; }
for s in $SEEDS; do
  R --method zero --n_train 60 --seed $s --tag _big
  for n in 5 15 30 60; do
    R --method bandit --kl 0.1 --n_train $n --seed $s --tag _big_n${n}_kl0.1
    R --method bandit --kl 1   --n_train $n --seed $s --tag _big_n${n}_kl1
    R --method sup    --kl 0.1 --n_train $n --seed $s --tag _big_n${n}_sup
  done
done
