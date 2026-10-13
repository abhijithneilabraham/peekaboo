#!/bin/bash
# Section 5.1, control: the same feedback training and calibration on a bilinear scorer over the same frozen
# features, without Vega's engine. Tests whether the engine is needed.
source "$(dirname "$0")/_common.sh"
for s in $SEEDS; do python compare.py --method linear_fb --seed $s; done
