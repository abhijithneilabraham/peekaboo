#!/bin/bash
# Section 5.1, ablation: our method with the KL weight we did not select (beta 1).
source "$(dirname "$0")/_common.sh"
for s in $SEEDS; do python compare.py --method feedback --kl 1 --seed $s --tag _kl1; done
