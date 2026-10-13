#!/bin/bash
# Section 7 (Kuhn poker): encode all 28 information sets (3- and 4-card decks) with the frozen backbone.
# Needs ../laya_policy (game code). Output: bank_kuhn[_4b].pt
source "$(dirname "$0")/_common.sh"
python extract.py kuhn
