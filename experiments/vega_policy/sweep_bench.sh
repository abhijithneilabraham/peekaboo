#!/bin/bash
source .venv/bin/activate; export HF_HOME=$PWD/.hf OMP_NUM_THREADS=4
source guard.sh; wait_for_memory
python bench_eval.py 1.5 2 3 2>&1 | grep -v -i "fetch\|warn"
for sp in rows tasks; do for s in 0 1 2; do
  python bench_rl.py --method zero   --split $sp --seed $s 2>&1 | tail -1 | cut -c1-200
  python bench_rl.py --method bandit --split $sp --seed $s --kl 0   --tag _kl0   2>&1 | tail -1 | cut -c1-200
  python bench_rl.py --method bandit --split $sp --seed $s --kl 0.1 --tag _kl0.1 2>&1 | tail -1 | cut -c1-200
  python bench_rl.py --method bandit --split $sp --seed $s --kl 1   --tag _kl1   2>&1 | tail -1 | cut -c1-200
  python bench_rl.py --method sup    --split $sp --seed $s --kl 0.1 --tag _kl0.1 2>&1 | tail -1 | cut -c1-200
done; done
echo ALLDONE
