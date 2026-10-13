#!/bin/bash
# Section 5.5, "online": every feedback row seen once (one epoch), 60 rows per task. Compare with
# _big_n60_kl0.1 from 11_feedback_amount.sh.
source "$(dirname "$0")/_common.sh"
for s in $SEEDS; do
  python bench_rl.py --items decisionbench_items_100.json --n_test 40 --method bandit --kl 0.1 --n_train 60 \
    --epochs 1 --seed $s --tag _big_n60_kl0.1_online | grep -E "feedback temperature|skip" || true
done
