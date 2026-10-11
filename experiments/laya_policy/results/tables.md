## Kuhn poker (3 cards): exploitability, lower is better (Nash = 0, uniform random = 0.458)

| Policy | Opponents | Seeds | Hands played | Start | At 9,600 hands |  Final | Best during training | Time-averaged strategy (at 9,600 / final) | 4-card deck, never trained on (final / time-avg) |
|---|---|---|---|---|---|---|---|---|---|
| Laya head + RL | naive self-play | 2 | 9600 | 0.348 | +0.311 ± 0.129 | 0.311 ± 0.129 | 0.163 | +0.208 ± 0.025 / 0.208 | 0.504 / 0.341 |
| Laya head + RL | pool of past selves | 3 | 9600 | 0.348 | +0.328 ± 0.183 | 0.328 ± 0.183 | 0.141 | +0.155 ± 0.044 / 0.155 | 0.629 / 0.387 |
| Laya zero-shot | pool of past selves | 1 | 0 | 0.348 | +0.348 | 0.348 ± 0.000 | 0.348 | +0.348 / 0.348 | 0.323 / 0.323 |
| small MLP (from scratch) | naive self-play | 5 | 19200 | 0.467 | +0.394 ± 0.158 | 0.464 ± 0.117 | 0.133 | +0.277 ± 0.110 / 0.320 | rank-mapped by hand: 0.404 |
| small MLP (from scratch) | pool of past selves | 5 | 19200 | 0.467 | +0.339 ± 0.158 | 0.306 ± 0.165 | 0.149 | +0.189 ± 0.075 / 0.222 | rank-mapped by hand: 0.281 |
| tabular (from scratch) | naive self-play | 5 | 19200 | 0.458 | +0.298 ± 0.079 | 0.204 ± 0.061 | 0.101 | +0.097 ± 0.022 / 0.113 | rank-mapped by hand: 0.149 |
| tabular (from scratch) | pool of past selves | 5 | 19200 | 0.458 | +0.225 ± 0.059 | 0.305 ± 0.092 | 0.104 | +0.092 ± 0.013 / 0.120 | rank-mapped by hand: 0.196 |

## Text hide-and-seek: score per episode against scripted opponents (−1 to +1, higher is better)

Scripted reference: a fleeing hider scores +0.17 against a chasing seeker; standing still scores −0.61.

| Policy | Opponents | Seeds | Episodes | hider vs chasing seeker | seeker vs fleeing hider | hider | seeker | distinct situations encoded |
|---|---|---|---|---|---|---|---|---|
| Laya head + RL (lr 0.0001) | pool | 1 | 3200 | -0.60 → -0.58 | -0.91 → -0.84 | -0.32 → -0.33 | -0.94 → -0.93 | 1252 |
| Laya head + RL (lr 0.001) | pool | 1 | 1920 | -0.60 → -0.53 | -0.91 → -0.89 | -0.32 → -0.26 | -0.94 → -0.97 | 1196 |
| Laya zero-shot | pool | 1 | 0 | -0.60 → -0.60 | -0.91 → -0.91 | -0.32 → -0.32 | -0.94 → -0.94 | 432 |
| small MLP (from scratch) | naive self-play | 5 | 96000 | -0.52 → +0.37 ± 0.06 (at 3200 eps: -0.247 ± 0.087) | -0.89 → -0.27 ± 0.05 (at 3200 eps: -0.873 ± 0.051) | -0.27 → +0.37 ± 0.04 (at 3200 eps: +0.005 ± 0.111) | -0.97 → -0.54 ± 0.03 (at 3200 eps: -0.936 ± 0.053) | — |
| small MLP (from scratch) | pool | 5 | 96000 | -0.52 → +0.39 ± 0.11 (at 3200 eps: -0.290 ± 0.067) | -0.89 → -0.28 ± 0.05 (at 3200 eps: -0.862 ± 0.034) | -0.27 → +0.38 ± 0.02 (at 3200 eps: -0.047 ± 0.102) | -0.97 → -0.56 ± 0.03 (at 3200 eps: -0.926 ± 0.024) | — |
| small MLP (from scratch) (v1: 4800 self-play episodes only) | naive self-play | 5 | | -0.46 | -0.80 | -0.22 | -0.88 | |
| small MLP (from scratch) (v1: 4800 self-play episodes only) | pool | 5 | | -0.50 | -0.66 | -0.14 | -0.80 | |
