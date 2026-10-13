#!/bin/bash
# Section 7 diagnosis: can each parameterisation (linear probe, full engine, adapter) fit a Nash strategy
# with supervised training? It can (error ~0.001), so the RL collapse is an optimisation problem.
source "$(dirname "$0")/_common.sh"
python fitprobe.py
