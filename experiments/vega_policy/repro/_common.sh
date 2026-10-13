# Shared setup. Every script in this folder sources this file; do not run it directly.
#   VEGA_SIZE=0.8b|4b   checkpoint (default 0.8b). 4b uses bank_*_4b.pt and results_4b/, so sizes never mix.
#   SEEDS="0 1 2"       seeds for every training run
#   VEGA_DEVICE=cuda    backbone device for feature extraction (default: cuda if available, else cpu)
set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if [ -f .venv/bin/activate ]; then source .venv/bin/activate; fi
export VEGA_SIZE=${VEGA_SIZE:-0.8b} SEEDS=${SEEDS:-"0 1 2"}
export HF_HOME=${HF_HOME:-$PWD/.hf} HF_HUB_DISABLE_XET=1 OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} PYTHONUNBUFFERED=1
RESULTS=$(python -c "import paths; print(paths.RESULTS)")
mkdir -p "$RESULTS/compare"
echo "== $(basename "$0")  size=$VEGA_SIZE  seeds=$SEEDS  results=$RESULTS"
# On macOS (the 8 GB laptop the paper used): wait for free memory, and kill a run that grows too large.
if [ "$(uname)" = Darwin ]; then
  source guard.sh; wait_for_memory
  MAX_MB=${MAX_MB:-5200} KILL_FREE=${KILL_FREE:-12} ./guard.sh watch >> logs_guard.txt 2>&1 & GUARD=$!
  trap 'kill $GUARD 2>/dev/null' EXIT
fi
