#!/bin/bash
# Section 7, 150 updates: the KL anchor to the released model (adapter, beta 0.3 and 1) against entropy bonus
# and contrast fixes. The paper's "Vega adapter + PPO + KL anchor" row is _kl1.
source "$(dirname "$0")/_common.sh"
run(){ python vtrain.py --env kuhn "$@" | grep done || true; }
for s in $SEEDS; do
  run --brain engine  --lr 1e-5 --seed $s --tag _base
  run --brain adapter --lr 1e-4 --seed $s --kl 0.3 --tag _kl0.3
  run --brain adapter --lr 1e-4 --seed $s --kl 1 --tag _kl1
  run --brain engine  --lr 1e-5 --seed $s --ent 0.1 --tag _ent0.1
  run --brain engine  --lr 1e-5 --seed $s --contrast 5 --tag _c5
  run --brain adapter --lr 1e-4 --seed $s --contrast 5 --kl 0.3 --tag _c5kl0.3
  run --brain engine  --lr 1e-5 --seed $s --contrast 5 --ent 0.1 --tag _c5ent0.1
  run --brain adapter --lr 1e-4 --seed $s --contrast 5 --kl 1 --ent 0.1 --tag _c5kl1ent0.1
done
python group.py
