#!/bin/bash
# Section 5.1, row "released model": the checkpoint as published, on the exact 1,075-row slice. No training.
source "$(dirname "$0")/_common.sh"
python compare.py --method zero
