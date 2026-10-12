#!/bin/bash
source .venv/bin/activate; export HF_HOME=$PWD/.hf OMP_NUM_THREADS=4
source guard.sh; wait_for_memory
for s in 0 1 2; do python compare.py --method feedback --seed $s --tag _v2 2>&1 | grep -E "_s[0-9]|Trace|Error"; done
python compare.py --method zero --tag _v2 2>&1 | grep -E "_s[0-9]|Trace"
echo ALLDONE
