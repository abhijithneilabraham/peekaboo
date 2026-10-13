#!/bin/bash
# Section 5.1-5.3 tables from scripts 04-09: main table, paired bootstrap CIs vs released and vs TTT,
# rows TTT covers vs not, per-domain accuracy. Output: results*/compare/tables.md
source "$(dirname "$0")/_common.sh"
python compare_table.py | tee "$RESULTS/compare/tables.md"
