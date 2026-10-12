#!/bin/bash
source .venv/bin/activate; export HF_HOME=$PWD/.hf OMP_NUM_THREADS=4 HF_HUB_DISABLE_XET=1
source guard.sh; wait_for_memory
MAX_MB=5200 KILL_FREE=12 ./guard.sh watch >> logs_guard.txt 2>&1 & G=$!
DB_PER_TASK=${DB_PER_TASK:-25} python bench_extract.py
kill $G; echo ALLDONE
