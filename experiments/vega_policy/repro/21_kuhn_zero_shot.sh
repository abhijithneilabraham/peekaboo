#!/bin/bash
# Section 7: Vega with no training, and with feature contrast x3/5/10/20 around the per-question mean.
source "$(dirname "$0")/_common.sh"
python vtrain.py --env kuhn --brain vega0 | grep done || true
for c in 3 5 10 20; do python vtrain.py --env kuhn --brain vega0 --contrast $c --tag _c$c | grep done || true; done
python group.py "$RESULTS/kuhn_vega0_*.json"
