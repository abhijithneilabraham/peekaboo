#!/bin/bash
# Python environment in experiments/vega_policy/.venv.
# NO_VENV=1 installs into the current environment instead (Kaggle, Colab).
set -e; cd "$(dirname "$0")/.."
if [ -z "$NO_VENV" ]; then python3 -m venv .venv && source .venv/bin/activate; fi
pip install -r repro/requirements.txt
