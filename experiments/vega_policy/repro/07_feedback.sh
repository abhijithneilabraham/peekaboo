#!/bin/bash
# Section 5.1, OURS: a fresh gated adapter trained by REINFORCE on right/wrong feedback with a KL anchor
# (beta 0.1) to the released model, then a temperature fitted from feedback alone. Gold labels are never read
# except through the 0/1 reward.
source "$(dirname "$0")/_common.sh"
for s in $SEEDS; do python compare.py --method feedback --seed $s; done
