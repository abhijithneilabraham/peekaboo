#!/bin/bash
# Section 5.6: engine-only decision latency (CPU, batch 1, cached features), released vs + one feedback adapter,
# and the adapter's parameter count. Backbone time is not included. Output: results*/compare/latency.json
source "$(dirname "$0")/_common.sh"
python latency.py
