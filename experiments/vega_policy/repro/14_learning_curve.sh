#!/bin/bash
# Summary of scripts 11-13 (mean over seeds, paired CI vs released) and the learning-curve figure.
# Output: results*/feedback_runs.txt and reports/figures/vega_feedback_curve[_4b].png
source "$(dirname "$0")/_common.sh"
python dbgroup.py "$RESULTS/db_*_big*.json" | tee "$RESULTS/feedback_runs.txt"
python figure.py
