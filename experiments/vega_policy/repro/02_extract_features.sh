#!/bin/bash
# The only step that runs the language model. Rebuilds Vega's README slice of DecisionBench (25 rows per task,
# seed 20261011, 1,075 rows) and the adaptation pool (100 rows per task, 4,280 rows, a superset of the slice),
# and caches the frozen backbone's pooled features. Every later script replays the engine on this cache.
# Resumable: run it again to continue. Approximate time: 0.8B ~3 s/row on a laptop CPU (~3.5 h);
# 4B needs a GPU (~9 GB of weights). Output: bank_decisionbench[_4b].pt
source "$(dirname "$0")/_common.sh"
DB_PER_TASK=25 python bench_extract.py
DB_PER_TASK=100 python bench_extract.py
