#!/bin/bash
# Sanity check before any result is trusted: the engine replay on cached features must match Vega.decide()
# (prints max |dp| over the 20 rows decided officially during extraction). Also scores the readout variants
# that need no training (task contrast, adapters off, no halting). Output: results*/decisionbench_variants.json
source "$(dirname "$0")/_common.sh"
python bench_eval.py 1.5 2 3
