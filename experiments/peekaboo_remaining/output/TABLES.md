# Peekaboo against Vega and Laya, on their own benchmarks

"base" = the model as released, measured in this run. "+ Peekaboo" = the same model plus our adapter, trained only from right / wrong feedback (mean ± s.d. over 3 seeds). "README" = the number Vega or Laya publish. Bold = best in the row.

## 0. Everything: every model, method and ablation on every benchmark

Accuracy (mean ± s.d. over 3 seeds), then calibration error, then the paired accuracy gain over the base model in points with a 95% bootstrap CI. JevBench: per-suite Peekaboo learns by 5-fold cross-fitting; pooled Peekaboo never sees a JevBench row. "switch" = the add-on is used only where held-out right / wrong feedback says it beats the base model.

### accuracy

| model | method | learns from | public | jevbench | decisionbench | typed | emotion | agnews |
|---|---|---|---|---|---|---|---|---|
| Vega 0.8B | base model | none | 0.556 | 0.550 | 0.509 | 0.665 | 0.568 | 0.901 |
| Vega 0.8B | VegaML's TTT head | answer key | 0.533 | · | 0.531 | 0.721 | 0.638 | 0.897 |
| Vega 0.8B | answer-key adapter (Laya: answer-key head) | answer key | 0.584 ± 0.005 | · | 0.628 ± 0.008 | 0.734 ± 0.000 | 0.630 ± 0.011 | 0.906 |
| Vega 0.8B | **Peekaboo** | right / wrong | 0.590 ± 0.009 | 0.518 ± 0.018 | 0.613 ± 0.011 | 0.734 ± 0.007 | 0.593 ± 0.009 | · |
| Vega 0.8B | Peekaboo + switch | right / wrong | · | · | 0.613 ± 0.011 | · | 0.593 ± 0.009 | · |
| Vega 0.8B | Peekaboo without the honesty fix | right / wrong | · | · | 0.613 ± 0.011 | · | 0.593 ± 0.009 | · |
| Vega 0.8B | Peekaboo without the leash (KL 0) | right / wrong | 0.568 ± 0.021 | 0.509 ± 0.025 | 0.612 ± 0.007 | 0.698 ± 0.020 | 0.529 ± 0.020 | · |
| Vega 4B | base model | none | 0.715 | 0.667 | 0.631 | 0.730 | · | · |
| Vega 4B | VegaML's TTT head | answer key | 0.677 | · | 0.639 | · | · | · |
| Vega 4B | answer-key adapter (Laya: answer-key head) | answer key | 0.705 ± 0.005 | · | 0.683 ± 0.011 | · | · | · |
| Vega 4B | **Peekaboo** | right / wrong | 0.716 ± 0.007 | 0.711 ± 0.004 | 0.697 ± 0.006 | · | · | · |
| Vega 4B | Peekaboo + switch | right / wrong | 0.713 ± 0.003 | 0.675 ± 0.006 | 0.676 ± 0.032 | · | · | · |
| Vega 4B | Peekaboo without the honesty fix | right / wrong | 0.716 ± 0.007 | 0.711 ± 0.004 | 0.697 ± 0.006 | · | · | · |
| Vega 4B | Peekaboo without the leash (KL 0) | right / wrong | 0.707 ± 0.008 | 0.704 ± 0.011 | · | · | · | · |
| Laya | base model | none | 0.542 | 0.589 | 0.401 | 0.361 | 0.597 | 0.930 |
| Laya | laya-typed-decisions (Laya's released fine-tune) | answer key (typed-decisions) | 0.551 | 0.537 | 0.438 | 0.762 | 0.604 | 0.930 |
| Laya | **Peekaboo** | right / wrong | · | 0.586 ± 0.002 | · | · | · | · |

### calibration error (ECE, lower is better)

| model | method | learns from | public | jevbench | decisionbench | typed | emotion | agnews |
|---|---|---|---|---|---|---|---|---|
| Vega 0.8B | base model | none | 0.154 | 0.173 | 0.105 | 0.063 | 0.033 | 0.021 |
| Vega 0.8B | VegaML's TTT head | answer key | 0.208 | · | 0.113 | 0.139 | 0.039 | 0.043 |
| Vega 0.8B | answer-key adapter (Laya: answer-key head) | answer key | 0.182 ± 0.003 | · | 0.071 ± 0.014 | 0.079 ± 0.008 | 0.059 ± 0.013 | 0.021 |
| Vega 0.8B | **Peekaboo** | right / wrong | 0.188 ± 0.009 | 0.094 ± 0.011 | 0.055 ± 0.015 | 0.074 ± 0.007 | 0.070 ± 0.025 | · |
| Vega 0.8B | Peekaboo + switch | right / wrong | · | · | 0.055 ± 0.015 | · | 0.070 ± 0.025 | · |
| Vega 0.8B | Peekaboo without the honesty fix | right / wrong | · | · | 0.241 ± 0.010 | · | 0.251 ± 0.063 | · |
| Vega 0.8B | Peekaboo without the leash (KL 0) | right / wrong | 0.213 ± 0.016 | 0.097 ± 0.014 | 0.082 ± 0.024 | 0.052 ± 0.006 | 0.048 ± 0.009 | · |
| Vega 4B | base model | none | 0.141 | 0.158 | 0.099 | 0.050 | · | · |
| Vega 4B | VegaML's TTT head | answer key | 0.173 | · | 0.112 | · | · | · |
| Vega 4B | answer-key adapter (Laya: answer-key head) | answer key | 0.181 ± 0.006 | · | 0.051 ± 0.009 | · | · | · |
| Vega 4B | **Peekaboo** | right / wrong | 0.172 ± 0.003 | 0.098 ± 0.008 | 0.049 ± 0.009 | · | · | · |
| Vega 4B | Peekaboo + switch | right / wrong | 0.150 ± 0.013 | 0.128 ± 0.005 | 0.064 ± 0.026 | · | · | · |
| Vega 4B | Peekaboo without the honesty fix | right / wrong | 0.235 ± 0.003 | 0.167 ± 0.012 | 0.223 ± 0.009 | · | · | · |
| Vega 4B | Peekaboo without the leash (KL 0) | right / wrong | 0.182 ± 0.011 | 0.099 ± 0.005 | · | · | · | · |
| Laya | base model | none | 0.212 | 0.113 | 0.174 | 0.183 | 0.306 | 0.012 |
| Laya | laya-typed-decisions (Laya's released fine-tune) | answer key (typed-decisions) | 0.166 | 0.101 | 0.045 | 0.206 | 0.185 | 0.106 |
| Laya | **Peekaboo** | right / wrong | · | 0.077 ± 0.010 | · | · | · | · |

### gain over the base model

| model | method | learns from | public | jevbench | decisionbench | typed | emotion | agnews |
|---|---|---|---|---|---|---|---|---|
| Vega 0.8B | base model | none | — | — | — | — | — | — |
| Vega 0.8B | VegaML's TTT head | answer key | -2.3 [-4.3, -0.3] | · | +2.2 [+0.0, +4.5] | +5.5 [+3.1, +7.9] | +7.0 [+4.2, +9.8] | -0.3 [-1.6, +0.9] |
| Vega 0.8B | answer-key adapter (Laya: answer-key head) | answer key | +2.7 [+0.8, +4.5] | · | +11.9 [+9.2, +14.7] | +6.9 [+5.0, +8.9] | +6.1 [+3.8, +8.6] | +0.5 [-0.7, +1.7] |
| Vega 0.8B | **Peekaboo** | right / wrong | +3.4 [+1.7, +4.9] | -3.2 [-7.4, +1.0] | +10.4 [+8.0, +12.8] | +6.8 [+5.1, +8.6] | +2.5 [+0.4, +4.6] | · |
| Vega 0.8B | Peekaboo + switch | right / wrong | · | · | +10.4 [+8.0, +12.8] | · | +2.5 [+0.4, +4.6] | · |
| Vega 0.8B | Peekaboo without the honesty fix | right / wrong | · | · | +10.4 [+8.0, +12.8] | · | +2.5 [+0.4, +4.6] | · |
| Vega 0.8B | Peekaboo without the leash (KL 0) | right / wrong | +1.2 [-0.5, +2.8] | -4.0 [-7.9, -0.1] | +10.3 [+7.7, +13.0] | +3.3 [+1.4, +5.2] | -3.9 [-6.5, -1.4] | · |
| Vega 4B | base model | none | — | — | — | — | · | · |
| Vega 4B | VegaML's TTT head | answer key | -3.9 [-5.7, -2.2] | · | +0.8 [-1.0, +2.7] | · | · | · |
| Vega 4B | answer-key adapter (Laya: answer-key head) | answer key | -1.0 [-2.5, +0.4] | · | +5.2 [+2.8, +7.9] | · | · | · |
| Vega 4B | **Peekaboo** | right / wrong | +0.1 [-1.3, +1.3] | +4.5 [+0.3, +9.0] | +6.7 [+4.6, +8.8] | · | · | · |
| Vega 4B | Peekaboo + switch | right / wrong | -0.2 [-0.7, +0.3] | +0.9 [-0.6, +2.3] | +4.5 [+3.1, +5.9] | · | · | · |
| Vega 4B | Peekaboo without the honesty fix | right / wrong | +0.1 [-1.3, +1.3] | +4.5 [+0.3, +9.0] | +6.7 [+4.6, +8.8] | · | · | · |
| Vega 4B | Peekaboo without the leash (KL 0) | right / wrong | -0.8 [-2.1, +0.4] | +3.8 [-0.7, +8.5] | · | · | · | · |
| Laya | base model | none | — | — | — | — | — | — |
| Laya | laya-typed-decisions (Laya's released fine-tune) | answer key (typed-decisions) | +0.9 [-0.3, +2.2] | -5.2 [-10.0, +0.0] | +3.7 [+1.3, +6.2] | +40.2 [+37.5, +42.7] | +0.7 [-0.5, +1.9] | +0.0 [-0.6, +0.6] |
| Laya | **Peekaboo** | right / wrong | · | -0.3 [-0.9, +0.0] | · | · | · | · |

## 1. Vega's README benchmark table, with Peekaboo

| benchmark | Vega 0.8B (README) | base Vega 0.8B | **Vega 0.8B + Peekaboo** | Vega 4B (README) | base Vega 4B | **Vega 4B + Peekaboo** | base Laya | laya-typed-decisions | **Laya + Peekaboo** | Jev (README) |
|---|---|---|---|---|---|---|---|---|---|---|
| **Public tasks the README lists (accuracy)** | | | | | | | | | | |
| unfair_tos | 0.737 | 0.737 | 0.690 ± 0.074 | **0.877** | **0.877** | 0.760 ± 0.079 | **0.877** | 0.842 | not run | 0.667 |
| tweet_stance_abortion | 0.364 | 0.364 | 0.533 ± 0.076 | **0.545** | 0.527 | 0.424 ± 0.163 | 0.327 | 0.309 | not run | 0.436 |
| prompt_injections | 0.746 | 0.746 | **0.891 ± 0.030** | 0.855 | 0.855 | 0.885 ± 0.043 | 0.654 | 0.600 | not run | 0.764 |
| tweet_sentiment | 0.424 | 0.441 | 0.486 ± 0.008 | **0.593** | 0.576 | 0.492 ± 0.063 | 0.559 | 0.491 | not run | 0.525 |
| tweet_offensive | 0.780 | 0.780 | 0.785 ± 0.021 | 0.780 | 0.780 | **0.836 ± 0.008** | 0.695 | 0.712 | not run | 0.729 |
| enron_spam | **1.000** | **1.000** | 0.989 ± 0.008 | **1.000** | **1.000** | **1.000 ± 0.000** | 0.917 | 0.900 | not run | 0.950 |
| jailbreak_cls | 0.983 | 0.983 | 0.971 ± 0.021 | **1.000** | **1.000** | 0.989 ± 0.008 | 0.983 | 0.983 | not run | 0.966 |
| glue_rte | 0.633 | 0.633 | 0.606 ± 0.008 | **0.883** | **0.883** | 0.850 ± 0.014 | 0.833 | 0.817 | not run | 0.850 |
| emotion | 0.550 | 0.550 | 0.522 ± 0.069 | **0.650** | **0.650** | 0.572 ± 0.021 | 0.567 | 0.567 | not run | 0.617 |
| toxic_chat_toxicity | 0.950 | 0.950 | 0.922 ± 0.021 | **1.000** | **1.000** | 0.983 ± 0.000 | 0.967 | 0.967 | not run | 0.983 |
| superglue_cb | 0.667 | 0.667 | 0.637 ± 0.082 | 0.822 | 0.822 | 0.800 ± 0.018 | **0.867** | **0.867** | not run | 0.800 |
| clinc150_oos | 0.683 | 0.667 | 0.561 ± 0.061 | **0.900** | **0.900** | 0.800 ± 0.024 | 0.000 | 0.317 | not run | 0.883 |
| **Slices** | | | | | | | | | | |
| JevBench family:temporal_numeric | **0.467** | **0.467** | 0.378 ± 0.031 | 0.400 | 0.400 | 0.356 ± 0.031 | 0.333 | 0.200 | 0.333 ± 0.000 | 0.200 |
| DecisionBench domain:privacy | 0.880 | 0.880 | 0.880 ± 0.000 | **1.000** | **1.000** | **1.000 ± 0.000** | 0.080 | 0.160 | not run | 0.960 |
| DecisionBench domain:public_sector | 0.760 | 0.760 | 0.893 ± 0.019 | **0.960** | **0.960** | **0.960 ± 0.000** | 0.360 | 0.400 | not run | 0.880 |
| **Public menu, 43 tasks, macro average** | | | | | | | | | | |
| accuracy | 0.556 | 0.556 | 0.590 ± 0.009 | 0.717 | 0.715 | 0.716 ± 0.007 | 0.542 | 0.551 | not run | **0.836** |
| skill, chance corrected | 0.388 | 0.388 | 0.423 ± 0.013 | 0.594 | 0.593 | 0.593 ± 0.009 | 0.375 | 0.380 | not run | **0.755** |
| macro F1 | 0.486 | 0.485 | 0.519 ± 0.005 | 0.649 | 0.648 | 0.655 ± 0.006 | 0.472 | 0.477 | not run | **0.796** |
| expected calibration error | 0.151 | 0.154 | 0.188 ± 0.009 | 0.148 | 0.141 | 0.172 ± 0.003 | 0.212 | 0.166 | not run | **0.115** |
| negative log likelihood | 1.099 | 1.100 | 1.128 ± 0.034 | **0.768** | **0.768** | 0.821 ± 0.012 | 1.651 | 1.170 | not run | 0.878 |
| area under risk-coverage | 0.362 | 0.363 | 0.323 ± 0.018 | 0.169 | 0.169 | 0.177 ± 0.004 | 0.367 | 0.348 | not run | **0.077** |
| latency per decision, p50 ms | 74 | 101 | 203 | 106 | 82 | 89 | 43 | **42** | not run | 162 |
| **JevBench, 231 items** | | | | | | | | | | |
| accuracy | 0.545 | 0.550 | 0.518 ± 0.018 | 0.662 | 0.667 | 0.711 ± 0.004 | 0.589 | 0.537 | 0.586 ± 0.002 | **0.866** |
| accuracy, hard tier | 0.414 | 0.414 | 0.423 ± 0.013 | 0.513 | 0.522 | 0.522 ± 0.000 | 0.351 | 0.270 | 0.351 ± 0.000 | **0.730** |
| paraphrase consistency | 0.694 | 0.694 | 0.750 ± 0.045 | 0.722 | 0.722 | 0.824 ± 0.026 | 0.694 | 0.750 | 0.676 ± 0.013 | **0.972** |
| expected calibration error | 0.178 | 0.173 | 0.094 ± 0.011 | 0.169 | 0.158 | 0.098 ± 0.008 | 0.113 | 0.101 | 0.077 ± 0.010 | **0.067** |
| **DecisionBench, 1,075 items** | | | | | | | | | | |
| accuracy | 0.509 | 0.509 | 0.613 ± 0.011 | 0.633 | 0.631 | 0.697 ± 0.006 | 0.401 | 0.438 | not run | **0.752** |
| expected calibration error | 0.103 | 0.105 | 0.055 ± 0.015 | 0.098 | 0.099 | 0.049 ± 0.009 | 0.174 | **0.045** | not run | 0.122 |
| negative log likelihood | 1.628 | 1.630 | 1.305 ± 0.029 | 1.420 | 1.422 | **1.101 ± 0.008** | 2.062 | 1.566 | not run | 3.196 |
| KL to the gold distribution, 10 rows | 1.012 | 1.018 | 1.032 ± 0.089 | 0.854 | 0.852 | **0.850 ± 0.180** | 1.110 | 0.930 | not run | 6.099 |
| coverage, rows answered | **1.000** | **1.000** | **1.000 ± 0.000** | **1.000** | **1.000** | **1.000 ± 0.000** | 0.954 | 0.990 | not run | 0.997 |
| latency per decision, p50 ms | 85 | 98 | 83 | 245 | 99 | 101 | **33** | 36 | not run | 163 |

Vega's DecisionBench rows: the finished runs in experiments/vega_policy/results*/compare (its latency is the README's). Laya is measured in this run on every row. laya-typed-decisions is Laya's released answer-key fine-tune (on typed-decisions).

## 2. Laya's README benchmark: typed-decisions (2,000 test decisions; feedback / labels from the 6,000 training decisions)

| model | learns from | accuracy (README) | accuracy | soft accuracy | ECE | NLL | gain vs base |
|---|---|---|---|---|---|---|---|
| base Laya | none | 0.362 | 0.361 | 0.326 | 0.183 | 1.361 | — |
| laya-typed-decisions (Laya's own fine-tune) | answer key | 0.766 | 0.762 | 0.471 | 0.206 | 0.706 | +40.2 [+37.5, +42.7] |
| **Laya + Peekaboo** | right / wrong |  | not run | not run | not run | not run | not run |
| base Vega 0.8B | none |  | 0.665 | 0.501 | 0.063 | 0.754 | — |
| Vega 0.8B + VegaML's TTT head | answer key |  | 0.721 | 0.559 | 0.139 | 0.908 | +5.5 [+3.1, +7.9] |
| Vega 0.8B + answer-key adapter | answer key |  | 0.734 ± 0.000 | 0.559 ± 0.001 | 0.079 ± 0.008 | 0.693 ± 0.037 | +6.9 [+5.0, +8.9] |
| **Vega 0.8B + Peekaboo** | right / wrong |  | 0.734 ± 0.007 | 0.549 ± 0.004 | 0.074 ± 0.007 | 0.692 ± 0.008 | +6.8 [+5.1, +8.6] |
| base Vega 4B | none |  | 0.730 | 0.555 | 0.050 | 0.630 | — |
| Vega 4B + VegaML's TTT head | answer key |  | not run | not run | not run | not run | not run |
| Vega 4B + answer-key adapter | answer key |  | not run | not run | not run | not run | not run |
| **Vega 4B + Peekaboo** | right / wrong |  | not run | not run | not run | not run | not run |
| Jev 1.13.0 (dataset card) | none | 0.727 | | | | | |

Vega's README also reports typed-decisions (0.763 / 0.803 with its task adapter) on six other workflows whose rows are not published, so it cannot be rerun; the rows above are the public benchmark Laya reports.

## 3. The 43 public tasks: Peekaboo against VegaML's own ways to improve Vega

| model | method | learns from | accuracy | skill | ECE | NLL | gain vs base |
|---|---|---|---|---|---|---|---|
| Vega 0.8B | base | none | 0.556 | 0.388 | 0.154 | 1.100 | — |
| Vega 0.8B | VegaML's TTT head (the notebook's 50 labelled rows per task) | answer key | 0.533 | 0.347 | 0.208 | 1.468 | -2.3 [-4.3, -0.3] |
| Vega 0.8B | answer-key adapter | answer key | 0.584 ± 0.005 | 0.408 ± 0.006 | 0.182 ± 0.003 | 1.078 ± 0.010 | +2.7 [+0.8, +4.5] |
| Vega 0.8B | **Peekaboo** | right / wrong | 0.590 ± 0.009 | 0.423 ± 0.013 | 0.188 ± 0.009 | 1.128 ± 0.034 | +3.4 [+1.7, +4.9] |
| Vega 4B | base | none | 0.715 | 0.593 | 0.141 | 0.768 | — |
| Vega 4B | VegaML's TTT head (the notebook's 50 labelled rows per task) | answer key | 0.677 | 0.535 | 0.173 | 1.208 | -3.9 [-5.7, -2.2] |
| Vega 4B | answer-key adapter | answer key | 0.705 ± 0.005 | 0.578 ± 0.006 | 0.181 ± 0.006 | 0.865 ± 0.003 | -1.0 [-2.5, +0.4] |
| Vega 4B | **Peekaboo** | right / wrong | 0.716 ± 0.007 | 0.593 ± 0.009 | 0.172 ± 0.003 | 0.821 ± 0.012 | +0.1 [-1.3, +1.3] |

## 4. Peekaboo head to head (accuracy points, paired 95% CI; positive = Peekaboo better)

| benchmark | model | Peekaboo vs base model | Peekaboo vs their improvement (answer key) |
|---|---|---|---|
| public | Vega 0.8B | +3.4 [+1.7, +4.9] | TTT: +5.6 [+3.7, +7.5] · adapter: +0.6 [-0.6, +1.9] |
| typed | Vega 0.8B | +6.8 [+5.1, +8.6] | TTT: +1.3 [-0.5, +3.3] · adapter: -0.0 [-1.0, +0.9] |
| jevbench | Vega 0.8B | -3.2 [-7.4, +1.0] | — (Vega has none) |
| public | Vega 4B | +0.1 [-1.3, +1.3] | TTT: +3.9 [+2.2, +5.6] · adapter: +1.1 [+0.0, +2.1] |
| typed | Vega 4B | not run | TTT: not run · adapter: not run |
| jevbench | Vega 4B | +4.5 [+0.3, +9.0] | — (Vega has none) |
| public | Laya | not run | laya-typed-decisions: not run |
| jevbench | Laya | -0.3 [-0.9, +0.0] | laya-typed-decisions: +4.9 [-0.3, +9.7] |
| decisionbench | Laya | not run | laya-typed-decisions: not run |
| typed | Laya | not run | laya-typed-decisions: not run |

DecisionBench (Vega 0.8B / 4B) is in the paper: +10.2 / +6.0 over base, +8.0 / +5.8 over TTT, -1.6 / +1.2 against the answer-key adapter.

## 5. Checks

| model | suite | replay vs official answers, max abs. probability difference | rows the model could not answer |
|---|---|---|---|
| 0.8b | public | 1.5e-16 | 0 |
| 0.8b | jevbench | 1.4e-16 | 0 |
| 0.8b | typed | 1.5e-16 | 0 |
| 4b | public | 1.5e-16 | 0 |
| 4b | jevbench | 1.5e-16 | 0 |
| 4b | typed | — | 0 |
| laya | public | 5.0e-05 | 110 |
| laya | jevbench | 5.0e-05 | 0 |
| laya | decisionbench | 5.0e-05 | 195 |
| laya | typed | 5.0e-05 | 0 |
| laya_td | public | — | 0 |
| laya_td | jevbench | — | 0 |
| laya_td | decisionbench | — | 36 |
| laya_td | typed | — | 0 |
