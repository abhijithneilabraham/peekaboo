#!/bin/bash
# Section 7: plain self-play PPO (no anchor) over learning rates, adapter and full engine. Shows the collapse
# to "always bet or call".
source "$(dirname "$0")/_common.sh"
for b in adapter engine; do for lr in 3e-4 1e-4 3e-5 1e-5; do for s in 0 1; do
  python vtrain.py --env kuhn --brain $b --lr $lr --seed $s --tag _lr$lr | tail -1
done; done; done
python group.py "$RESULTS/kuhn_*_lr*.json"
