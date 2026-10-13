#!/bin/bash
# Section 5.1, baseline "vegaml TTT": Vega.fit() test-time-training head per task (gold labels, 'auto' mode).
# Deterministic, so one run.
source "$(dirname "$0")/_common.sh"
python compare.py --method ttt_auto
