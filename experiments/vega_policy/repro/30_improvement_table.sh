#!/bin/bash
# The end result: Vega's README benchmark table with "+ ours" next to each released size (0.8B, and 4B once
# VEGA_SIZE=4b has been run), Jev copied from the README, and the paired accuracy gain with its CI.
# Reads results/compare and results_4b/compare. Output: results/readme_table.md
source "$(dirname "$0")/_common.sh"
python readme_table.py
