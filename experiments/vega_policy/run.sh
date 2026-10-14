#!/bin/bash
# Reproduces "Vega learns from feedback" (reports/Vega learns from feedback.md) and builds the final table:
# Vega's README benchmark table with released Vega vs Vega + our feedback adapter.
#
#   bash run.sh                      # Vega 0.8B: everything, ending with results/readme_table.md
#   VEGA_SIZE=4b bash run.sh         # Vega 4B (feature extraction needs a GPU); fills the 4B columns of the table
#   bash run.sh kuhn                 # also the Kuhn poker experiments (paper section 7)
#
# Settings: VEGA_SIZE=0.8b|4b (default 0.8b), SEEDS="0 1 2", VEGA_DEVICE=cuda|cpu (default: cuda if available),
# NO_VENV=1 to install into the current Python (Kaggle, Colab). Every run whose result file exists is skipped,
# so the script can be stopped and started again.
set -eo pipefail
cd "$(dirname "$0")"
export VEGA_SIZE=${VEGA_SIZE:-0.8b} SEEDS=${SEEDS:-"0 1 2"}
export HF_HOME=${HF_HOME:-$PWD/.hf} HF_HUB_DISABLE_XET=1 OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} PYTHONUNBUFFERED=1
step(){ echo; echo "==== $*  (size $VEGA_SIZE, $(date +%H:%M:%S))"; }

# ── environment ──
if [ -z "$NO_VENV" ]; then [ -d .venv ] || python3 -m venv .venv; source .venv/bin/activate; fi
python -c "import vegaml" 2>/dev/null || pip install -r requirements.txt
RESULTS=$(python -c "import paths; print(paths.RESULTS)"); mkdir -p "$RESULTS/compare"

# On macOS (the 8 GB laptop the paper used): wait for free memory, and kill a run that grows too large.
if [ "$(uname)" = Darwin ]; then
  source guard.sh; wait_for_memory
  MAX_MB=${MAX_MB:-5200} KILL_FREE=${KILL_FREE:-12} ./guard.sh watch >> logs_guard.txt 2>&1 & GUARD=$!
  trap 'kill $GUARD 2>/dev/null' EXIT
fi
quiet(){ grep -E "$1|skip|Error|Traceback" || true; }

# ── 1. features: the only step that runs the language model ──
# Vega's README slice of DecisionBench (25 rows per task, seed 20261011, 1,075 rows = the test set) and the
# adaptation pool (100 rows per task, 4,280 rows, a superset). 0.8B: ~3 s/row on a laptop CPU. 4B: use a GPU.
step "1. extract backbone features (section 2.2-2.3)"
DB_PER_TASK=25 python bench_extract.py
DB_PER_TASK=100 python bench_extract.py

# ── 2. the engine replay must match Vega.decide() before any result is trusted ──
step "2. replay check and no-training variants (section 2.2, 6)"
python bench_eval.py 1.5 2 3

# ── 3. direct comparison on the exact 1,075 README rows ──
step "3. direct comparison (section 4, 5.1)"
python compare.py --method zero     | quiet _s0                                     # released model
python compare.py --method ttt_auto | quiet _s0                                     # vegaml fit() TTT head
for s in $SEEDS; do
  python compare.py --method sup_ts    --seed $s | quiet _s$s                       # supervised adapter + temp. scaling
  python compare.py --method feedback  --seed $s | quiet _s$s                       # OURS
  python compare.py --method feedback  --seed $s --kl 1 --tag _kl1 | quiet _s$s     # ablation: KL weight 1
  python compare.py --method linear_fb --seed $s | quiet _s$s                       # control without Vega's engine
done
python compare_table.py | tee "$RESULTS/compare/tables.md"

# ── 4. how much feedback, one pass, unseen tasks (1,720-row test set from the pool) ──
step "4. feedback amount, one pass, unseen tasks (section 5.5)"
R(){ python bench_rl.py --items decisionbench_items_100.json "$@" | quiet "feedback temperature"; }
for s in $SEEDS; do
  R --n_test 40 --method zero --n_train 60 --seed $s --tag _big
  for n in 5 15 30 60; do
    R --n_test 40 --method bandit --kl 0.1 --n_train $n --seed $s --tag _big_n${n}_kl0.1
    R --n_test 40 --method bandit --kl 1   --n_train $n --seed $s --tag _big_n${n}_kl1
    R --n_test 40 --method sup    --kl 0.1 --n_train $n --seed $s --tag _big_n${n}_sup
  done
  R --n_test 40 --method bandit --kl 0.1 --n_train 60 --epochs 1 --seed $s --tag _big_n60_kl0.1_online
  R --method bandit --kl 0.1 --split tasks --seed $s --tag _big_tasks_kl0.1
done
python dbgroup.py "$RESULTS/db_*_big*.json" | tee "$RESULTS/feedback_runs.txt"
python figure.py

# ── 5. cost ──
step "5. latency and adapter size (section 5.6)"
python latency.py

# ── 6. the pilot where the KL weight was chosen (disclosed in section 8: it used the comparison slice) ──
step "6. KL weight pilot (section 8)"
for sp in rows tasks; do for s in $SEEDS; do
  python bench_rl.py --method zero --split $sp --seed $s | tail -1 | cut -c1-200
  for kl in 0 0.1 1; do python bench_rl.py --method bandit --split $sp --seed $s --kl $kl --tag _kl$kl | quiet "feedback temperature"; done
  python bench_rl.py --method sup --split $sp --seed $s --kl 0.1 --tag _kl0.1 | quiet "feedback temperature"
done; done

# ── 7. Kuhn poker (optional: bash run.sh kuhn) ──
if [ "$1" = kuhn ]; then
  step "7. Kuhn poker (section 7)"
  python extract.py kuhn
  K(){ python vtrain.py --env kuhn "$@" | quiet done; }
  K --brain vega0                                                     # zero-shot
  for c in 3 5 10 20; do K --brain vega0 --contrast $c --tag _c$c; done
  for b in adapter engine; do for lr in 3e-4 1e-4 3e-5 1e-5; do for s in 0 1; do
    K --brain $b --lr $lr --seed $s --tag _lr$lr                      # plain PPO: collapses
  done; done; done
  for s in $SEEDS; do                                                 # 150 updates: KL anchor vs other fixes
    K --brain engine  --lr 1e-5 --seed $s --tag _base
    K --brain adapter --lr 1e-4 --seed $s --kl 0.3 --tag _kl0.3
    K --brain adapter --lr 1e-4 --seed $s --kl 1 --tag _kl1
    K --brain engine  --lr 1e-5 --seed $s --ent 0.1 --tag _ent0.1
    K --brain engine  --lr 1e-5 --seed $s --contrast 5 --tag _c5
    K --brain adapter --lr 1e-4 --seed $s --contrast 5 --kl 0.3 --tag _c5kl0.3
    K --brain engine  --lr 1e-5 --seed $s --contrast 5 --ent 0.1 --tag _c5ent0.1
    K --brain adapter --lr 1e-4 --seed $s --contrast 5 --kl 1 --ent 0.1 --tag _c5kl1ent0.1
  done
  for s in 0 1 2 3 4; do                                              # 300 updates vs lookup-table controls
    K --updates 300 --brain adapter --lr 1e-4 --seed $s --kl 1 --tag _L_kl1
    K --updates 300 --brain adapter --lr 1e-4 --seed $s --kl 3 --tag _L_kl3
    K --updates 300 --brain adapter --lr 1e-4 --seed $s --kl 1 --magnet 30 --tag _L_kl1m30
    K --updates 300 --brain adapter --lr 1e-4 --seed $s --kl 3 --magnet 30 --tag _L_kl3m30
    K --updates 300 --brain table --seed $s --tag _L_plain
    K --updates 300 --brain table --seed $s --kl 1 --tag _L_kl1
    K --updates 300 --brain table --seed $s --kl 1 --magnet 30 --tag _L_kl1m30
  done
  python fitprobe.py                                                  # the engine can represent Nash
  python group.py | tee "$RESULTS/kuhn_runs.txt"
fi

# ── the final table ──
step "final: README-format table, released vs + ours (section 5.2)"
python readme_table.py
echo; echo "Done. Table: results/readme_table.md   Details: $RESULTS/compare/tables.md"
