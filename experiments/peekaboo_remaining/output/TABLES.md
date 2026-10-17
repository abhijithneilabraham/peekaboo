# Peekaboo against Vega and Laya, on their own benchmarks

"base" = the model as released, measured in this run. "+ Peekaboo" = the same model plus our adapter, trained only from right / wrong feedback (mean ± s.d. over 3 seeds). "README" = the number Vega or Laya publish. Bold = best in the row.

## 0. Everything: every model, method and ablation on every benchmark

Accuracy (mean ± s.d. over 3 seeds), then calibration error, then the paired accuracy gain over the base model in points with a 95% bootstrap CI. JevBench: per-suite Peekaboo learns by 5-fold cross-fitting; pooled Peekaboo never sees a JevBench row. "switch" = the add-on is used only where held-out right / wrong feedback says it beats the base model.

### accuracy

| model | method | learns from | public | jevbench | decisionbench | typed | emotion | agnews |
|---|---|---|---|---|---|---|---|---|
| Vega 0.8B | base model | none | 0.556 | 0.550 | 0.509 | 0.665 | 0.568 | 0.901 |
| Vega 0.8B | VegaML's TTT head | answer key | 0.533 | · | 0.531 | 0.721 | 0.638 | 0.897 |
| Vega 0.8B | answer-key adapter (Laya: answer-key head) | answer key | 0.584 ± 0.005 | · | 0.628 ± 0.008 | 0.734 ± 0.000 | 0.630 ± 0.011 | 0.907 ± 0.001 |
| Vega 0.8B | **Peekaboo** | right / wrong | 0.590 ± 0.013 | 0.517 ± 0.018 | 0.613 ± 0.011 | 0.725 ± 0.012 | 0.593 ± 0.009 | 0.907 ± 0.002 |
| Vega 0.8B | Peekaboo + switch | right / wrong | 0.578 ± 0.020 | 0.544 ± 0.004 | 0.613 ± 0.011 | 0.711 ± 0.032 | 0.593 ± 0.009 | 0.907 ± 0.002 |
| Vega 0.8B | Peekaboo without the honesty fix | right / wrong | 0.590 ± 0.013 | 0.517 ± 0.018 | 0.613 ± 0.011 | 0.725 ± 0.012 | 0.593 ± 0.009 | 0.907 ± 0.002 |
| Vega 0.8B | Peekaboo without the leash (KL 0) | right / wrong | 0.568 ± 0.021 | 0.509 ± 0.025 | 0.612 ± 0.007 | 0.698 ± 0.020 | 0.529 ± 0.020 | 0.908 ± 0.001 |
| Vega 0.8B | **pooled Peekaboo** (one add-on, all suites' feedback) | right / wrong | 0.596 ± 0.006 | 0.548 ± 0.011 | 0.606 ± 0.011 | 0.720 ± 0.015 | 0.567 ± 0.040 | 0.911 ± 0.005 |
| Vega 0.8B | pooled Peekaboo + switch | right / wrong | 0.568 ± 0.016 | 0.548 ± 0.011 | 0.606 ± 0.011 | 0.708 ± 0.031 | 0.567 ± 0.040 | 0.911 ± 0.005 |
| Vega 0.8B | pooled Peekaboo without the honesty fix | right / wrong | 0.596 ± 0.006 | 0.548 ± 0.011 | 0.606 ± 0.011 | 0.720 ± 0.015 | 0.567 ± 0.040 | 0.911 ± 0.005 |
| Vega 4B | base model | none | 0.715 | 0.667 | 0.631 | 0.730 | 0.579 | 0.927 |
| Vega 4B | VegaML's TTT head | answer key | 0.677 | · | 0.639 | 0.753 | 0.651 | 0.919 |
| Vega 4B | answer-key adapter (Laya: answer-key head) | answer key | 0.705 ± 0.005 | · | 0.683 ± 0.011 | 0.774 ± 0.006 | 0.592 ± 0.022 | 0.921 ± 0.003 |
| Vega 4B | **Peekaboo** | right / wrong | 0.716 ± 0.007 | 0.711 ± 0.004 | 0.697 ± 0.006 | 0.764 ± 0.006 | 0.556 ± 0.009 | 0.928 ± 0.002 |
| Vega 4B | Peekaboo + switch | right / wrong | 0.713 ± 0.003 | 0.675 ± 0.006 | 0.676 ± 0.032 | 0.764 ± 0.006 | 0.556 ± 0.009 | 0.927 ± 0.000 |
| Vega 4B | Peekaboo without the honesty fix | right / wrong | 0.716 ± 0.007 | 0.711 ± 0.004 | 0.697 ± 0.006 | 0.764 ± 0.006 | 0.556 ± 0.009 | 0.928 ± 0.002 |
| Vega 4B | Peekaboo without the leash (KL 0) | right / wrong | 0.707 ± 0.008 | 0.704 ± 0.011 | 0.681 ± 0.002 | 0.741 ± 0.007 | 0.521 ± 0.022 | 0.930 ± 0.002 |
| Vega 4B | **pooled Peekaboo** (one add-on, all suites' feedback) | right / wrong | 0.720 ± 0.005 | 0.687 ± 0.002 | 0.682 ± 0.008 | 0.744 ± 0.018 | 0.544 ± 0.009 | 0.932 ± 0.003 |
| Vega 4B | pooled Peekaboo + switch | right / wrong | 0.715 ± 0.000 | 0.687 ± 0.002 | 0.651 ± 0.029 | 0.741 ± 0.017 | 0.571 ± 0.011 | 0.927 ± 0.000 |
| Vega 4B | pooled Peekaboo without the honesty fix | right / wrong | 0.720 ± 0.005 | 0.687 ± 0.002 | 0.682 ± 0.008 | 0.744 ± 0.018 | 0.544 ± 0.009 | 0.932 ± 0.003 |
| Laya | base model | none | 0.542 | 0.589 | 0.401 | 0.361 | 0.597 | 0.930 |
| Laya | answer-key adapter (Laya: answer-key head) | answer key | 0.535 ± 0.004 | · | 0.389 ± 0.009 | 0.544 ± 0.005 | 0.540 ± 0.032 | 0.930 ± 0.000 |
| Laya | laya-typed-decisions (Laya's released fine-tune) | answer key (typed-decisions) | 0.551 | 0.537 | 0.438 | 0.762 | 0.604 | 0.930 |
| Laya | **Peekaboo** | right / wrong | 0.543 ± 0.001 | 0.584 ± 0.000 | 0.405 ± 0.002 | 0.493 ± 0.008 | 0.596 ± 0.001 | 0.930 ± 0.000 |
| Laya | Peekaboo + switch | right / wrong | 0.542 ± 0.000 | 0.586 ± 0.002 | 0.401 ± 0.000 | 0.493 ± 0.008 | 0.597 ± 0.000 | 0.930 ± 0.000 |
| Laya | Peekaboo without the honesty fix | right / wrong | 0.543 ± 0.001 | 0.584 ± 0.000 | 0.405 ± 0.002 | 0.493 ± 0.008 | 0.596 ± 0.001 | 0.930 ± 0.000 |
| Laya | Peekaboo without the leash (KL 0) | right / wrong | 0.539 ± 0.006 | 0.584 ± 0.000 | 0.403 ± 0.001 | 0.463 ± 0.009 | 0.586 ± 0.009 | 0.930 ± 0.001 |
| Laya | **pooled Peekaboo** (one add-on, all suites' feedback) | right / wrong | 0.538 ± 0.002 | 0.569 ± 0.007 | 0.398 ± 0.006 | 0.475 ± 0.004 | 0.594 ± 0.002 | 0.930 ± 0.000 |
| Laya | pooled Peekaboo + switch | right / wrong | 0.542 ± 0.000 | 0.569 ± 0.007 | 0.401 ± 0.000 | 0.475 ± 0.004 | 0.597 ± 0.000 | 0.930 ± 0.000 |
| Laya | pooled Peekaboo without the honesty fix | right / wrong | 0.538 ± 0.002 | 0.569 ± 0.007 | 0.398 ± 0.006 | 0.475 ± 0.004 | 0.594 ± 0.002 | 0.930 ± 0.000 |

### calibration error (ECE, lower is better)

| model | method | learns from | public | jevbench | decisionbench | typed | emotion | agnews |
|---|---|---|---|---|---|---|---|---|
| Vega 0.8B | base model | none | 0.154 | 0.173 | 0.105 | 0.063 | 0.033 | 0.021 |
| Vega 0.8B | VegaML's TTT head | answer key | 0.208 | · | 0.113 | 0.139 | 0.039 | 0.043 |
| Vega 0.8B | answer-key adapter (Laya: answer-key head) | answer key | 0.182 ± 0.003 | · | 0.071 ± 0.014 | 0.079 ± 0.008 | 0.059 ± 0.013 | 0.022 ± 0.004 |
| Vega 0.8B | **Peekaboo** | right / wrong | 0.203 ± 0.007 | 0.092 ± 0.018 | 0.055 ± 0.015 | 0.077 ± 0.019 | 0.070 ± 0.025 | 0.016 ± 0.004 |
| Vega 0.8B | Peekaboo + switch | right / wrong | 0.184 ± 0.021 | 0.152 ± 0.003 | 0.055 ± 0.015 | 0.063 ± 0.001 | 0.070 ± 0.025 | 0.016 ± 0.004 |
| Vega 0.8B | Peekaboo without the honesty fix | right / wrong | 0.328 ± 0.018 | 0.267 ± 0.008 | 0.241 ± 0.010 | 0.198 ± 0.019 | 0.251 ± 0.063 | 0.075 ± 0.003 |
| Vega 0.8B | Peekaboo without the leash (KL 0) | right / wrong | 0.213 ± 0.016 | 0.097 ± 0.014 | 0.082 ± 0.024 | 0.052 ± 0.006 | 0.048 ± 0.009 | 0.018 ± 0.003 |
| Vega 0.8B | **pooled Peekaboo** (one add-on, all suites' feedback) | right / wrong | 0.193 ± 0.024 | 0.108 ± 0.034 | 0.066 ± 0.009 | 0.062 ± 0.031 | 0.068 ± 0.023 | 0.019 ± 0.005 |
| Vega 0.8B | pooled Peekaboo + switch | right / wrong | 0.157 ± 0.004 | 0.108 ± 0.034 | 0.066 ± 0.009 | 0.048 ± 0.011 | 0.068 ± 0.023 | 0.019 ± 0.005 |
| Vega 0.8B | pooled Peekaboo without the honesty fix | right / wrong | 0.267 ± 0.038 | 0.308 ± 0.022 | 0.232 ± 0.051 | 0.154 ± 0.018 | 0.311 ± 0.110 | 0.073 ± 0.014 |
| Vega 4B | base model | none | 0.141 | 0.158 | 0.099 | 0.050 | 0.054 | 0.027 |
| Vega 4B | VegaML's TTT head | answer key | 0.173 | · | 0.112 | 0.119 | 0.043 | 0.041 |
| Vega 4B | answer-key adapter (Laya: answer-key head) | answer key | 0.181 ± 0.006 | · | 0.051 ± 0.009 | 0.056 ± 0.007 | 0.038 ± 0.010 | 0.014 ± 0.001 |
| Vega 4B | **Peekaboo** | right / wrong | 0.172 ± 0.003 | 0.098 ± 0.008 | 0.049 ± 0.009 | 0.044 ± 0.013 | 0.035 ± 0.012 | 0.018 ± 0.003 |
| Vega 4B | Peekaboo + switch | right / wrong | 0.150 ± 0.013 | 0.128 ± 0.005 | 0.064 ± 0.026 | 0.044 ± 0.013 | 0.035 ± 0.012 | 0.027 ± 0.000 |
| Vega 4B | Peekaboo without the honesty fix | right / wrong | 0.235 ± 0.003 | 0.167 ± 0.012 | 0.223 ± 0.009 | 0.171 ± 0.004 | 0.311 ± 0.026 | 0.049 ± 0.016 |
| Vega 4B | Peekaboo without the leash (KL 0) | right / wrong | 0.182 ± 0.011 | 0.099 ± 0.005 | 0.075 ± 0.027 | 0.049 ± 0.016 | 0.052 ± 0.004 | 0.016 ± 0.002 |
| Vega 4B | **pooled Peekaboo** (one add-on, all suites' feedback) | right / wrong | 0.176 ± 0.015 | 0.134 ± 0.038 | 0.057 ± 0.019 | 0.050 ± 0.005 | 0.044 ± 0.009 | 0.016 ± 0.006 |
| Vega 4B | pooled Peekaboo + switch | right / wrong | 0.141 ± 0.000 | 0.134 ± 0.038 | 0.084 ± 0.021 | 0.052 ± 0.002 | 0.051 ± 0.004 | 0.027 ± 0.000 |
| Vega 4B | pooled Peekaboo without the honesty fix | right / wrong | 0.227 ± 0.020 | 0.261 ± 0.024 | 0.253 ± 0.027 | 0.202 ± 0.013 | 0.321 ± 0.006 | 0.056 ± 0.005 |
| Laya | base model | none | 0.212 | 0.113 | 0.174 | 0.183 | 0.306 | 0.012 |
| Laya | answer-key adapter (Laya: answer-key head) | answer key | 0.143 ± 0.001 | · | 0.066 ± 0.006 | 0.035 ± 0.008 | 0.072 ± 0.016 | 0.010 ± 0.002 |
| Laya | laya-typed-decisions (Laya's released fine-tune) | answer key (typed-decisions) | 0.166 | 0.101 | 0.045 | 0.206 | 0.185 | 0.106 |
| Laya | **Peekaboo** | right / wrong | 0.157 ± 0.004 | 0.063 ± 0.012 | 0.071 ± 0.006 | 0.052 ± 0.005 | 0.086 ± 0.014 | 0.018 ± 0.004 |
| Laya | Peekaboo + switch | right / wrong | 0.212 ± 0.000 | 0.084 ± 0.007 | 0.174 ± 0.000 | 0.052 ± 0.005 | 0.306 ± 0.000 | 0.015 ± 0.005 |
| Laya | Peekaboo without the honesty fix | right / wrong | 0.319 ± 0.009 | 0.228 ± 0.002 | 0.329 ± 0.027 | 0.351 ± 0.014 | 0.347 ± 0.012 | 0.059 ± 0.000 |
| Laya | Peekaboo without the leash (KL 0) | right / wrong | 0.178 ± 0.015 | 0.066 ± 0.015 | 0.065 ± 0.008 | 0.056 ± 0.013 | 0.088 ± 0.018 | 0.018 ± 0.002 |
| Laya | **pooled Peekaboo** (one add-on, all suites' feedback) | right / wrong | 0.164 ± 0.007 | 0.096 ± 0.019 | 0.067 ± 0.015 | 0.052 ± 0.016 | 0.087 ± 0.006 | 0.016 ± 0.003 |
| Laya | pooled Peekaboo + switch | right / wrong | 0.212 ± 0.000 | 0.096 ± 0.019 | 0.174 ± 0.000 | 0.052 ± 0.016 | 0.306 ± 0.000 | 0.012 ± 0.000 |
| Laya | pooled Peekaboo without the honesty fix | right / wrong | 0.328 ± 0.033 | 0.271 ± 0.029 | 0.349 ± 0.054 | 0.328 ± 0.036 | 0.364 ± 0.019 | 0.056 ± 0.009 |

### gain over the base model

| model | method | learns from | public | jevbench | decisionbench | typed | emotion | agnews |
|---|---|---|---|---|---|---|---|---|
| Vega 0.8B | base model | none | — | — | — | — | — | — |
| Vega 0.8B | VegaML's TTT head | answer key | -2.3 [-4.3, -0.3] | · | +2.2 [+0.0, +4.5] | +5.5 [+3.1, +7.9] | +7.0 [+4.2, +9.8] | -0.3 [-1.6, +0.9] |
| Vega 0.8B | answer-key adapter (Laya: answer-key head) | answer key | +2.7 [+0.8, +4.5] | · | +11.9 [+9.2, +14.7] | +6.9 [+5.0, +8.9] | +6.1 [+3.8, +8.6] | +0.6 [-0.4, +1.7] |
| Vega 0.8B | **Peekaboo** | right / wrong | +3.4 [+1.7, +4.9] | -3.3 [-7.1, +0.4] | +10.4 [+8.0, +12.8] | +5.9 [+4.2, +7.7] | +2.5 [+0.4, +4.6] | +0.7 [-0.0, +1.4] |
| Vega 0.8B | Peekaboo + switch | right / wrong | +2.2 [+1.1, +3.2] | -0.6 [-1.9, +0.7] | +10.4 [+8.0, +12.8] | +4.5 [+3.4, +5.7] | +2.5 [+0.4, +4.6] | +0.7 [-0.0, +1.4] |
| Vega 0.8B | Peekaboo without the honesty fix | right / wrong | +3.4 [+1.7, +4.9] | -3.3 [-7.1, +0.4] | +10.4 [+8.0, +12.8] | +5.9 [+4.2, +7.7] | +2.5 [+0.4, +4.6] | +0.7 [-0.0, +1.4] |
| Vega 0.8B | Peekaboo without the leash (KL 0) | right / wrong | +1.2 [-0.5, +2.8] | -4.0 [-7.9, -0.1] | +10.3 [+7.7, +13.0] | +3.3 [+1.4, +5.2] | -3.9 [-6.5, -1.4] | +0.7 [-0.1, +1.6] |
| Vega 0.8B | **pooled Peekaboo** (one add-on, all suites' feedback) | right / wrong | +4.0 [+2.6, +5.4] | -0.1 [-3.9, +3.6] | +9.7 [+7.5, +12.0] | +5.4 [+3.8, +7.0] | -0.2 [-2.4, +2.2] | +1.1 [+0.2, +2.0] |
| Vega 0.8B | pooled Peekaboo + switch | right / wrong | +1.2 [+0.6, +1.7] | -0.1 [-3.9, +3.6] | +9.7 [+7.5, +12.0] | +4.2 [+3.1, +5.4] | -0.2 [-2.4, +2.2] | +1.1 [+0.2, +2.0] |
| Vega 0.8B | pooled Peekaboo without the honesty fix | right / wrong | +4.0 [+2.6, +5.4] | -0.1 [-3.9, +3.6] | +9.7 [+7.5, +12.0] | +5.4 [+3.8, +7.0] | -0.2 [-2.4, +2.2] | +1.1 [+0.2, +2.0] |
| Vega 4B | base model | none | — | — | — | — | — | — |
| Vega 4B | VegaML's TTT head | answer key | -3.9 [-5.7, -2.2] | · | +0.8 [-1.0, +2.7] | +2.4 [+0.4, +4.5] | +7.3 [+4.7, +10.0] | -0.8 [-1.9, +0.4] |
| Vega 4B | answer-key adapter (Laya: answer-key head) | answer key | -1.0 [-2.5, +0.4] | · | +5.2 [+2.8, +7.9] | +4.5 [+3.1, +5.8] | +1.3 [-1.0, +3.5] | -0.5 [-1.5, +0.4] |
| Vega 4B | **Peekaboo** | right / wrong | +0.1 [-1.3, +1.3] | +4.5 [+0.3, +9.0] | +6.7 [+4.6, +8.8] | +3.4 [+2.0, +4.9] | -2.3 [-4.1, -0.4] | +0.1 [-0.4, +0.5] |
| Vega 4B | Peekaboo + switch | right / wrong | -0.2 [-0.7, +0.3] | +0.9 [-0.6, +2.3] | +4.5 [+3.1, +5.9] | +3.4 [+2.0, +4.9] | -2.3 [-4.1, -0.4] | +0.0 [+0.0, +0.0] |
| Vega 4B | Peekaboo without the honesty fix | right / wrong | +0.1 [-1.3, +1.3] | +4.5 [+0.3, +9.0] | +6.7 [+4.6, +8.8] | +3.4 [+2.0, +4.9] | -2.3 [-4.1, -0.4] | +0.1 [-0.4, +0.5] |
| Vega 4B | Peekaboo without the leash (KL 0) | right / wrong | -0.8 [-2.1, +0.4] | +3.8 [-0.7, +8.5] | +5.0 [+2.8, +7.3] | +1.1 [-0.4, +2.7] | -5.8 [-7.8, -3.7] | +0.3 [-0.1, +0.8] |
| Vega 4B | **pooled Peekaboo** (one add-on, all suites' feedback) | right / wrong | +0.4 [-0.8, +1.6] | +2.0 [-1.9, +6.2] | +5.1 [+3.3, +7.0] | +1.4 [-0.2, +3.0] | -3.5 [-5.2, -1.8] | +0.5 [-0.1, +1.2] |
| Vega 4B | pooled Peekaboo + switch | right / wrong | +0.0 [+0.0, +0.0] | +2.0 [-1.9, +6.2] | +2.1 [+1.3, +2.8] | +1.2 [+0.7, +1.8] | -0.8 [-1.3, -0.2] | +0.0 [+0.0, +0.0] |
| Vega 4B | pooled Peekaboo without the honesty fix | right / wrong | +0.4 [-0.8, +1.6] | +2.0 [-1.9, +6.2] | +5.1 [+3.3, +7.0] | +1.4 [-0.2, +3.0] | -3.5 [-5.2, -1.8] | +0.5 [-0.1, +1.2] |
| Laya | base model | none | — | — | — | — | — | — |
| Laya | answer-key adapter (Laya: answer-key head) | answer key | -0.7 [-1.6, +0.2] | · | -1.2 [-3.4, +1.1] | +18.3 [+15.6, +20.8] | -5.6 [-7.1, -4.1] | +0.0 [-0.2, +0.3] |
| Laya | laya-typed-decisions (Laya's released fine-tune) | answer key (typed-decisions) | +0.9 [-0.3, +2.2] | -5.2 [-10.0, +0.0] | +3.7 [+1.3, +6.2] | +40.2 [+37.5, +42.7] | +0.7 [-0.5, +1.9] | +0.0 [-0.6, +0.6] |
| Laya | **Peekaboo** | right / wrong | +0.1 [-0.3, +0.5] | -0.4 [-1.3, +0.0] | +0.4 [+0.0, +0.8] | +13.2 [+11.0, +15.4] | -0.1 [-0.4, +0.2] | +0.0 [-0.0, +0.1] |
| Laya | Peekaboo + switch | right / wrong | +0.0 [+0.0, +0.0] | -0.3 [-0.9, +0.0] | +0.0 [+0.0, +0.0] | +13.2 [+11.0, +15.4] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.1] |
| Laya | Peekaboo without the honesty fix | right / wrong | +0.1 [-0.3, +0.5] | -0.4 [-1.3, +0.0] | +0.4 [+0.0, +0.8] | +13.2 [+11.0, +15.4] | -0.1 [-0.4, +0.2] | +0.0 [-0.0, +0.1] |
| Laya | Peekaboo without the leash (KL 0) | right / wrong | -0.3 [-0.9, +0.3] | -0.4 [-1.3, +0.0] | +0.2 [-0.2, +0.6] | +10.2 [+8.0, +12.4] | -1.1 [-1.7, -0.4] | +0.1 [-0.1, +0.2] |
| Laya | **pooled Peekaboo** (one add-on, all suites' feedback) | right / wrong | -0.4 [-1.0, +0.2] | -2.0 [-4.9, +0.7] | -0.2 [-1.4, +0.9] | +11.4 [+9.4, +13.5] | -0.2 [-0.6, +0.1] | +0.0 [+0.0, +0.1] |
| Laya | pooled Peekaboo + switch | right / wrong | +0.0 [+0.0, +0.0] | -2.0 [-4.9, +0.7] | +0.0 [+0.0, +0.0] | +11.4 [+9.4, +13.5] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] |
| Laya | pooled Peekaboo without the honesty fix | right / wrong | -0.4 [-1.0, +0.2] | -2.0 [-4.9, +0.7] | -0.2 [-1.4, +0.9] | +11.4 [+9.4, +13.5] | -0.2 [-0.6, +0.1] | +0.0 [+0.0, +0.1] |

## 1. Vega's README benchmark table, with Peekaboo

| benchmark | Vega 0.8B (README) | base Vega 0.8B | **Vega 0.8B + Peekaboo** | Vega 4B (README) | base Vega 4B | **Vega 4B + Peekaboo** | base Laya | laya-typed-decisions | **Laya + Peekaboo** | Jev (README) |
|---|---|---|---|---|---|---|---|---|---|---|
| **Public tasks the README lists (accuracy)** | | | | | | | | | | |
| unfair_tos | 0.737 | 0.737 | 0.532 ± 0.083 | **0.877** | **0.877** | 0.760 ± 0.079 | **0.877** | 0.842 | 0.830 ± 0.054 | 0.667 |
| tweet_stance_abortion | 0.364 | 0.364 | 0.436 ± 0.156 | **0.545** | 0.527 | 0.424 ± 0.163 | 0.327 | 0.309 | 0.327 ± 0.000 | 0.436 |
| prompt_injections | 0.746 | 0.746 | 0.867 ± 0.009 | 0.855 | 0.855 | **0.885 ± 0.043** | 0.654 | 0.600 | 0.721 ± 0.017 | 0.764 |
| tweet_sentiment | 0.424 | 0.441 | 0.542 ± 0.050 | **0.593** | 0.576 | 0.492 ± 0.063 | 0.559 | 0.491 | 0.576 ± 0.014 | 0.525 |
| tweet_offensive | 0.780 | 0.780 | 0.808 ± 0.021 | 0.780 | 0.780 | **0.836 ± 0.008** | 0.695 | 0.712 | 0.672 ± 0.016 | 0.729 |
| enron_spam | **1.000** | **1.000** | 0.994 ± 0.008 | **1.000** | **1.000** | **1.000 ± 0.000** | 0.917 | 0.900 | 0.939 ± 0.008 | 0.950 |
| jailbreak_cls | 0.983 | 0.983 | 0.983 ± 0.014 | **1.000** | **1.000** | 0.989 ± 0.008 | 0.983 | 0.983 | 0.983 ± 0.000 | 0.966 |
| glue_rte | 0.633 | 0.633 | 0.600 ± 0.014 | **0.883** | **0.883** | 0.850 ± 0.014 | 0.833 | 0.817 | 0.833 ± 0.000 | 0.850 |
| emotion | 0.550 | 0.550 | 0.517 ± 0.049 | **0.650** | **0.650** | 0.572 ± 0.021 | 0.567 | 0.567 | 0.567 ± 0.000 | 0.617 |
| toxic_chat_toxicity | 0.950 | 0.950 | 0.911 ± 0.021 | **1.000** | **1.000** | 0.983 ± 0.000 | 0.967 | 0.967 | 0.978 ± 0.016 | 0.983 |
| superglue_cb | 0.667 | 0.667 | 0.770 ± 0.046 | 0.822 | 0.822 | 0.800 ± 0.018 | 0.867 | 0.867 | **0.881 ± 0.010** | 0.800 |
| clinc150_oos | 0.683 | 0.667 | 0.622 ± 0.031 | **0.900** | **0.900** | 0.800 ± 0.024 | 0.000 | 0.317 | 0.000 ± 0.000 | 0.883 |
| **Slices** | | | | | | | | | | |
| JevBench family:temporal_numeric | **0.467** | **0.467** | 0.333 ± 0.000 | 0.400 | 0.400 | 0.356 ± 0.031 | 0.333 | 0.200 | 0.333 ± 0.000 | 0.200 |
| DecisionBench domain:privacy | 0.880 | 0.880 | 0.880 ± 0.000 | **1.000** | **1.000** | **1.000 ± 0.000** | 0.080 | 0.160 | 0.093 ± 0.019 | 0.960 |
| DecisionBench domain:public_sector | 0.760 | 0.760 | 0.893 ± 0.019 | **0.960** | **0.960** | **0.960 ± 0.000** | 0.360 | 0.400 | 0.373 ± 0.019 | 0.880 |
| **Public menu, 43 tasks, macro average** | | | | | | | | | | |
| accuracy | 0.556 | 0.556 | 0.590 ± 0.013 | 0.717 | 0.715 | 0.716 ± 0.007 | 0.542 | 0.551 | 0.543 ± 0.001 | **0.836** |
| skill, chance corrected | 0.388 | 0.388 | 0.426 ± 0.014 | 0.594 | 0.593 | 0.593 ± 0.009 | 0.375 | 0.380 | 0.375 ± 0.002 | **0.755** |
| macro F1 | 0.486 | 0.485 | 0.524 ± 0.012 | 0.649 | 0.648 | 0.655 ± 0.006 | 0.472 | 0.477 | 0.481 ± 0.008 | **0.796** |
| expected calibration error | 0.151 | 0.154 | 0.203 ± 0.007 | 0.148 | 0.141 | 0.172 ± 0.003 | 0.212 | 0.166 | 0.157 ± 0.004 | **0.115** |
| negative log likelihood | 1.099 | 1.100 | 1.156 ± 0.031 | **0.768** | **0.768** | 0.821 ± 0.012 | 1.651 | 1.170 | 1.118 ± 0.005 | 0.878 |
| area under risk-coverage | 0.362 | 0.363 | 0.333 ± 0.019 | 0.169 | 0.169 | 0.177 ± 0.004 | 0.365 | 0.348 | 0.366 ± 0.004 | **0.077** |
| latency per decision, p50 ms | 74 | 101 | 115 | 106 | **37** | 89 | 80 | 91 | 80 | 162 |
| **JevBench, 231 items** | | | | | | | | | | |
| accuracy | 0.545 | 0.550 | 0.517 ± 0.018 | 0.662 | 0.667 | 0.711 ± 0.004 | 0.589 | 0.537 | 0.584 ± 0.000 | **0.866** |
| accuracy, hard tier | 0.414 | 0.414 | 0.390 ± 0.015 | 0.513 | 0.522 | 0.522 ± 0.000 | 0.351 | 0.270 | 0.351 ± 0.000 | **0.730** |
| paraphrase consistency | 0.694 | 0.694 | 0.796 ± 0.047 | 0.722 | 0.722 | 0.824 ± 0.026 | 0.694 | 0.750 | 0.667 ± 0.000 | **0.972** |
| expected calibration error | 0.178 | 0.173 | 0.092 ± 0.018 | 0.169 | 0.158 | 0.098 ± 0.008 | 0.113 | 0.101 | **0.063 ± 0.012** | 0.067 |
| **DecisionBench, 1,075 items** | | | | | | | | | | |
| accuracy | 0.509 | 0.509 | 0.613 ± 0.011 | 0.633 | 0.631 | 0.697 ± 0.006 | 0.401 | 0.438 | 0.405 ± 0.002 | **0.752** |
| expected calibration error | 0.103 | 0.105 | 0.055 ± 0.015 | 0.098 | 0.099 | 0.049 ± 0.009 | 0.174 | **0.045** | 0.071 ± 0.006 | 0.122 |
| negative log likelihood | 1.628 | 1.630 | 1.305 ± 0.029 | 1.420 | 1.422 | **1.101 ± 0.008** | 2.062 | 1.566 | 1.575 ± 0.008 | 3.196 |
| KL to the gold distribution, 10 rows | 1.012 | 1.018 | 1.032 ± 0.089 | 0.854 | 0.852 | 0.850 ± 0.180 | 1.110 | 0.930 | **0.773 ± 0.009** | 6.099 |
| coverage, rows answered | **1.000** | **1.000** | **1.000 ± 0.000** | **1.000** | **1.000** | **1.000 ± 0.000** | 0.954 | 0.990 | 0.954 ± 0.000 | 0.997 |
| latency per decision, p50 ms | 85 | 104 | 83 | 245 | 46 | 101 | **33** | 36 | **33** | 163 |

Vega's DecisionBench rows: the finished runs in experiments/vega_policy/results*/compare (its latency is the README's). Laya is measured in this run on every row. laya-typed-decisions is Laya's released answer-key fine-tune (on typed-decisions).

## 2. Laya's README benchmark: typed-decisions (2,000 test decisions; feedback / labels from the 6,000 training decisions)

| model | learns from | accuracy (README) | accuracy | soft accuracy | ECE | NLL | gain vs base |
|---|---|---|---|---|---|---|---|
| base Laya | none | 0.362 | 0.361 | 0.326 | 0.183 | 1.361 | — |
| laya-typed-decisions (Laya's own fine-tune) | answer key | 0.766 | 0.762 | 0.471 | 0.206 | 0.706 | +40.2 [+37.5, +42.7] |
| **Laya + Peekaboo** | right / wrong |  | 0.493 ± 0.008 | 0.368 ± 0.002 | 0.052 ± 0.005 | 1.124 ± 0.012 | +13.2 [+11.0, +15.4] |
| base Vega 0.8B | none |  | 0.665 | 0.501 | 0.063 | 0.754 | — |
| Vega 0.8B + VegaML's TTT head | answer key |  | 0.721 | 0.559 | 0.139 | 0.908 | +5.5 [+3.1, +7.9] |
| Vega 0.8B + answer-key adapter | answer key |  | 0.734 ± 0.000 | 0.559 ± 0.001 | 0.079 ± 0.008 | 0.693 ± 0.037 | +6.9 [+5.0, +8.9] |
| **Vega 0.8B + Peekaboo** | right / wrong |  | 0.725 ± 0.012 | 0.525 ± 0.028 | 0.077 ± 0.019 | 0.728 ± 0.064 | +5.9 [+4.2, +7.7] |
| base Vega 4B | none |  | 0.730 | 0.555 | 0.050 | 0.630 | — |
| Vega 4B + VegaML's TTT head | answer key |  | 0.753 | 0.577 | 0.119 | 0.763 | +2.4 [+0.4, +4.5] |
| Vega 4B + answer-key adapter | answer key |  | 0.774 ± 0.006 | 0.578 ± 0.003 | 0.056 ± 0.007 | 0.563 ± 0.006 | +4.5 [+3.1, +5.8] |
| **Vega 4B + Peekaboo** | right / wrong |  | 0.764 ± 0.006 | 0.565 ± 0.002 | 0.044 ± 0.013 | 0.599 ± 0.015 | +3.4 [+2.0, +4.9] |
| Jev 1.13.0 (dataset card) | none | 0.727 | | | | | |

Vega's README also reports typed-decisions (0.763 / 0.803 with its task adapter) on six other workflows whose rows are not published, so it cannot be rerun; the rows above are the public benchmark Laya reports.

## 3. The 43 public tasks: Peekaboo against VegaML's own ways to improve Vega

| model | method | learns from | accuracy | skill | ECE | NLL | gain vs base |
|---|---|---|---|---|---|---|---|
| Vega 0.8B | base | none | 0.556 | 0.388 | 0.154 | 1.100 | — |
| Vega 0.8B | VegaML's TTT head (the notebook's 50 labelled rows per task) | answer key | 0.533 | 0.347 | 0.208 | 1.468 | -2.3 [-4.3, -0.3] |
| Vega 0.8B | answer-key adapter | answer key | 0.584 ± 0.005 | 0.408 ± 0.006 | 0.182 ± 0.003 | 1.078 ± 0.010 | +2.7 [+0.8, +4.5] |
| Vega 0.8B | **Peekaboo** | right / wrong | 0.590 ± 0.013 | 0.426 ± 0.014 | 0.203 ± 0.007 | 1.156 ± 0.031 | +3.4 [+1.7, +4.9] |
| Vega 4B | base | none | 0.715 | 0.593 | 0.141 | 0.768 | — |
| Vega 4B | VegaML's TTT head (the notebook's 50 labelled rows per task) | answer key | 0.677 | 0.535 | 0.173 | 1.208 | -3.9 [-5.7, -2.2] |
| Vega 4B | answer-key adapter | answer key | 0.705 ± 0.005 | 0.578 ± 0.006 | 0.181 ± 0.006 | 0.865 ± 0.003 | -1.0 [-2.5, +0.4] |
| Vega 4B | **Peekaboo** | right / wrong | 0.716 ± 0.007 | 0.593 ± 0.009 | 0.172 ± 0.003 | 0.821 ± 0.012 | +0.1 [-1.3, +1.3] |

## 4. Peekaboo head to head (accuracy points, paired 95% CI; positive = Peekaboo better)

| benchmark | model | Peekaboo vs base model | Peekaboo vs their improvement (answer key) |
|---|---|---|---|
| public | Vega 0.8B | +3.4 [+1.7, +4.9] | TTT: +5.7 [+3.8, +7.5] · adapter: +0.6 [-0.6, +1.9] |
| typed | Vega 0.8B | +5.9 [+4.2, +7.7] | TTT: +0.4 [-1.5, +2.5] · adapter: -0.9 [-1.9, +0.1] |
| jevbench | Vega 0.8B | -3.3 [-7.1, +0.4] | — (Vega has none) |
| public | Vega 4B | +0.1 [-1.3, +1.3] | TTT: +3.9 [+2.2, +5.6] · adapter: +1.1 [+0.0, +2.1] |
| typed | Vega 4B | +3.4 [+2.0, +4.9] | TTT: +1.0 [-0.7, +2.7] · adapter: -1.1 [-1.9, -0.2] |
| jevbench | Vega 4B | +4.5 [+0.3, +9.0] | — (Vega has none) |
| public | Laya | +0.1 [-0.3, +0.5] | laya-typed-decisions: -0.8 [-2.1, +0.5] |
| jevbench | Laya | -0.4 [-1.3, +0.0] | laya-typed-decisions: +4.8 [-0.4, +9.5] |
| decisionbench | Laya | +0.4 [+0.0, +0.8] | laya-typed-decisions: -3.3 [-5.8, -0.9] |
| typed | Laya | +13.2 [+11.0, +15.4] | laya-typed-decisions: -26.9 [-28.9, -24.8] |

DecisionBench (Vega 0.8B / 4B) is in the paper: +10.2 / +6.0 over base, +8.0 / +5.8 over TTT, -1.6 / +1.2 against the answer-key adapter.

## 5. Checks

| model | suite | replay vs official answers, max abs. probability difference | rows the model could not answer |
|---|---|---|---|
| 0.8b | public | 1.5e-16 | 0 |
| 0.8b | jevbench | 1.4e-16 | 0 |
| 0.8b | typed | 1.5e-16 | 0 |
| 4b | public | 1.5e-16 | 0 |
| 4b | jevbench | 1.5e-16 | 0 |
| 4b | typed | 1.4e-16 | 0 |
| laya | public | 5.0e-05 | 110 |
| laya | jevbench | 5.0e-05 | 0 |
| laya | decisionbench | 5.0e-05 | 195 |
| laya | typed | 5.0e-05 | 0 |
| laya_td | public | — | 0 |
| laya_td | jevbench | — | 0 |
| laya_td | decisionbench | — | 36 |
| laya_td | typed | — | 0 |
