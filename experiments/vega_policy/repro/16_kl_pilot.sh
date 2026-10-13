#!/bin/bash
# The pilot runs where the KL weight was chosen (beta in {0, 0.1, 1}), on the 1,075-row slice:
# 'rows' = 15 train / 10 test rows per task; 'tasks' = 33 train tasks / 10 held-out tasks.
# The paper discloses that this pilot touched the comparison slice.
source "$(dirname "$0")/_common.sh"
for sp in rows tasks; do for s in $SEEDS; do
  python bench_rl.py --method zero   --split $sp --seed $s | tail -1 | cut -c1-200
  for kl in 0 0.1 1; do python bench_rl.py --method bandit --split $sp --seed $s --kl $kl --tag _kl$kl | tail -1 | cut -c1-200; done
  python bench_rl.py --method sup    --split $sp --seed $s --kl 0.1 --tag _kl0.1 | tail -1 | cut -c1-200
done; done
python dbgroup.py "$RESULTS/db_*_s[0-9]_kl*.json"
