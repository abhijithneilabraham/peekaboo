#!/bin/bash
# Section 5.5, "new tasks": feedback on 33 tasks, test on 10 tasks never seen in training.
source "$(dirname "$0")/_common.sh"
for s in $SEEDS; do
  python bench_rl.py --items decisionbench_items_100.json --method bandit --kl 0.1 --split tasks --seed $s \
    --tag _big_tasks_kl0.1 | grep -E "feedback temperature|skip" || true
done
