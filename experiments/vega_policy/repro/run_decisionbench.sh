#!/bin/bash
# Every DecisionBench experiment in order (scripts 02-16). Each step skips runs whose result file exists,
# so this can be stopped and restarted.   VEGA_SIZE=4b ./repro/run_decisionbench.sh
set -e; D="$(dirname "$0")"
for s in 02_extract_features 03_check_replay 04_released 05_ttt 06_supervised_ts 07_feedback 08_feedback_kl1 \
         09_no_engine_control 10_tables 11_feedback_amount 12_one_pass 13_unseen_tasks 14_learning_curve \
         15_latency 16_kl_pilot 30_improvement_table; do
  bash "$D/$s.sh"
done
