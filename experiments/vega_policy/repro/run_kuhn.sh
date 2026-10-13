#!/bin/bash
# Every Kuhn poker experiment in order (scripts 20-25).
set -e; D="$(dirname "$0")"
for s in 20_kuhn_features 21_kuhn_zero_shot 22_kuhn_ppo_lr 23_kuhn_kl_anchor 24_kuhn_long_runs 25_kuhn_fit_probe; do
  bash "$D/$s.sh"
done
