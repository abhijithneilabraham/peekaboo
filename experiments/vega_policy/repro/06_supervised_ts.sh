#!/bin/bash
# Section 5.1, baseline "supervised adapter + temperature scaling": a fresh gated adapter trained with cross-entropy
# on gold labels, temperature fitted on held-out labelled rows.
source "$(dirname "$0")/_common.sh"
for s in $SEEDS; do python compare.py --method sup_ts --seed $s; done
