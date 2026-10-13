#!/bin/bash
# Section 7, 300 updates x 5 seeds: Vega adapter + KL anchor against the lookup-table controls
# (table + the same KL anchor to uniform, and plain table). This is the control that showed the anchor,
# not Vega's knowledge, gives the gain in poker.
source "$(dirname "$0")/_common.sh"
run(){ python vtrain.py --env kuhn --updates 300 "$@" | grep done || true; }
for s in 0 1 2 3 4; do
  run --brain adapter --lr 1e-4 --seed $s --kl 1 --tag _L_kl1
  run --brain adapter --lr 1e-4 --seed $s --kl 3 --tag _L_kl3
  run --brain adapter --lr 1e-4 --seed $s --kl 1 --magnet 30 --tag _L_kl1m30
  run --brain adapter --lr 1e-4 --seed $s --kl 3 --magnet 30 --tag _L_kl3m30
  run --brain table --seed $s --tag _L_plain
  run --brain table --seed $s --kl 1 --tag _L_kl1
  run --brain table --seed $s --kl 1 --magnet 30 --tag _L_kl1m30
done
python group.py "$RESULTS/kuhn_*_L_*.json"
