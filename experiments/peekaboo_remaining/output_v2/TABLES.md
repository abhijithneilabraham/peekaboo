# Peekaboo against Vega and Laya, on their own benchmarks

"base" = the model as released, measured in this run. "+ Peekaboo" = the same model plus our adapter, trained only from right / wrong feedback (mean ± s.d. over 3 seeds). "README" = the number Vega or Laya publish. Bold = best in the row.

## 0. Everything: every model, method and ablation on every benchmark

Accuracy (mean ± s.d. over 3 seeds), then calibration error, then the paired accuracy gain over the base model in points with a 95% bootstrap CI. JevBench: per-suite Peekaboo learns by 5-fold cross-fitting; pooled Peekaboo never sees a JevBench row. "switch" = the add-on is used only where held-out right / wrong feedback says it beats the base model.

### accuracy

| model | method | learns from | public | jevbench | decisionbench | typed | emotion | agnews |
|---|---|---|---|---|---|---|---|---|

### calibration error (ECE, lower is better)

| model | method | learns from | public | jevbench | decisionbench | typed | emotion | agnews |
|---|---|---|---|---|---|---|---|---|

### gain over the base model

| model | method | learns from | public | jevbench | decisionbench | typed | emotion | agnews |
|---|---|---|---|---|---|---|---|---|

## 1. Vega's README benchmark table, with Peekaboo

| benchmark | Vega 0.8B (README) | base Vega 0.8B | **Vega 0.8B + Peekaboo** | Vega 4B (README) | base Vega 4B | **Vega 4B + Peekaboo** | base Laya | laya-typed-decisions | **Laya + Peekaboo** | Jev (README) |
|---|---|---|---|---|---|---|---|---|---|---|
| **Public tasks the README lists (accuracy)** | | | | | | | | | | |
| unfair_tos | 0.737 | not run | not run | **0.877** | not run | not run | not run | not run | not run | 0.667 |
| tweet_stance_abortion | 0.364 | not run | not run | **0.545** | not run | not run | not run | not run | not run | 0.436 |
| prompt_injections | 0.746 | not run | not run | **0.855** | not run | not run | not run | not run | not run | 0.764 |
| tweet_sentiment | 0.424 | not run | not run | **0.593** | not run | not run | not run | not run | not run | 0.525 |
| tweet_offensive | **0.780** | not run | not run | **0.780** | not run | not run | not run | not run | not run | 0.729 |
| enron_spam | **1.000** | not run | not run | **1.000** | not run | not run | not run | not run | not run | 0.950 |
| jailbreak_cls | 0.983 | not run | not run | **1.000** | not run | not run | not run | not run | not run | 0.966 |
| glue_rte | 0.633 | not run | not run | **0.883** | not run | not run | not run | not run | not run | 0.850 |
| emotion | 0.550 | not run | not run | **0.650** | not run | not run | not run | not run | not run | 0.617 |
| toxic_chat_toxicity | 0.950 | not run | not run | **1.000** | not run | not run | not run | not run | not run | 0.983 |
| superglue_cb | 0.667 | not run | not run | **0.822** | not run | not run | not run | not run | not run | 0.800 |
| clinc150_oos | 0.683 | not run | not run | **0.900** | not run | not run | not run | not run | not run | 0.883 |
| **Slices** | | | | | | | | | | |
| JevBench family:temporal_numeric | **0.467** | not run | not run | 0.400 | not run | not run | not run | not run | not run | 0.200 |
| DecisionBench domain:privacy | 0.880 | 0.880 | 0.920 ± 0.000 | **1.000** | **1.000** | **1.000 ± 0.000** | not run | not run | not run | 0.960 |
| DecisionBench domain:public_sector | 0.760 | 0.760 | 0.933 ± 0.038 | **0.960** | **0.960** | 0.947 ± 0.019 | not run | not run | not run | 0.880 |
| **Public menu, 43 tasks, macro average** | | | | | | | | | | |
| accuracy | 0.556 | not run | not run | 0.717 | not run | not run | not run | not run | not run | **0.836** |
| skill, chance corrected | 0.388 | not run | not run | 0.594 | not run | not run | not run | not run | not run | **0.755** |
| macro F1 | 0.486 | not run | not run | 0.649 | not run | not run | not run | not run | not run | **0.796** |
| expected calibration error | 0.151 | not run | not run | 0.148 | not run | not run | not run | not run | not run | **0.115** |
| negative log likelihood | 1.099 | not run | not run | **0.768** | not run | not run | not run | not run | not run | 0.878 |
| area under risk-coverage | 0.362 | not run | not run | 0.169 | not run | not run | not run | not run | not run | **0.077** |
| latency per decision, p50 ms | **74** | not run | not run | 106 | not run | not run | not run | not run | not run | 162 |
| **JevBench, 231 items** | | | | | | | | | | |
| accuracy | 0.545 | not run | not run | 0.662 | not run | not run | not run | not run | not run | **0.866** |
| accuracy, hard tier | 0.414 | not run | not run | 0.513 | not run | not run | not run | not run | not run | **0.730** |
| paraphrase consistency | 0.694 | not run | not run | 0.722 | not run | not run | not run | not run | not run | **0.972** |
| expected calibration error | 0.178 | not run | not run | 0.169 | not run | not run | not run | not run | not run | **0.067** |
| **DecisionBench, 1,075 items** | | | | | | | | | | |
| accuracy | 0.509 | 0.509 | 0.611 ± 0.015 | 0.633 | 0.635 | 0.695 ± 0.002 | not run | not run | not run | **0.752** |
| expected calibration error | 0.103 | 0.103 | 0.053 ± 0.009 | 0.098 | 0.097 | **0.045 ± 0.006** | not run | not run | not run | 0.122 |
| negative log likelihood | 1.628 | 1.628 | 1.291 ± 0.024 | 1.420 | 1.425 | **1.100 ± 0.016** | not run | not run | not run | 3.196 |
| KL to the gold distribution, 10 rows | 1.012 | 1.012 | 1.101 ± 0.332 | 0.854 | 0.850 | **0.793 ± 0.032** | not run | not run | not run | 6.099 |
| coverage, rows answered | **1.000** | **1.000** | **1.000 ± 0.000** | **1.000** | **1.000** | **1.000 ± 0.000** | not run | not run | not run | 0.997 |
| latency per decision, p50 ms | **85** | not run | not run | 245 | not run | not run | not run | not run | not run | 163 |

Vega's DecisionBench rows: the finished runs in experiments/vega_policy/results*/compare (its latency is the README's). Laya is measured in this run on every row. laya-typed-decisions is Laya's released answer-key fine-tune (on typed-decisions).

## 2. Laya's README benchmark: typed-decisions (2,000 test decisions; feedback / labels from the 6,000 training decisions)

| model | learns from | accuracy (README) | accuracy | soft accuracy | ECE | NLL | gain vs base |
|---|---|---|---|---|---|---|---|
| base Laya | none | 0.362 | not run | not run | not run | not run | — |
| laya-typed-decisions (Laya's own fine-tune) | answer key | 0.766 | not run | not run | not run | not run | not run |
| **Laya + Peekaboo** | right / wrong |  | not run | not run | not run | not run | not run |
| **Laya + Peekaboo, encoder unfrozen** | right / wrong |  | not run | not run | not run | not run | not run |
| base Vega 0.8B | none |  | not run | not run | not run | not run | — |
| Vega 0.8B + VegaML's TTT head | answer key |  | not run | not run | not run | not run | not run |
| Vega 0.8B + answer-key adapter | answer key |  | not run | not run | not run | not run | not run |
| **Vega 0.8B + Peekaboo** | right / wrong |  | not run | not run | not run | not run | not run |
| base Vega 4B | none |  | not run | not run | not run | not run | — |
| Vega 4B + VegaML's TTT head | answer key |  | not run | not run | not run | not run | not run |
| Vega 4B + answer-key adapter | answer key |  | not run | not run | not run | not run | not run |
| **Vega 4B + Peekaboo** | right / wrong |  | not run | not run | not run | not run | not run |
| Jev 1.13.0 (dataset card) | none | 0.727 | | | | | |

Vega's README also reports typed-decisions (0.763 / 0.803 with its task adapter) on six other workflows whose rows are not published, so it cannot be rerun; the rows above are the public benchmark Laya reports.

## 3. The 43 public tasks: Peekaboo against VegaML's own ways to improve Vega

| model | method | learns from | accuracy | skill | ECE | NLL | gain vs base |
|---|---|---|---|---|---|---|---|
| Vega 0.8B | base | none | not run | not run | not run | not run | — |
| Vega 0.8B | VegaML's TTT head (the notebook's 50 labelled rows per task) | answer key | not run | not run | not run | not run | not run |
| Vega 0.8B | answer-key adapter | answer key | not run | not run | not run | not run | not run |
| Vega 0.8B | **Peekaboo** | right / wrong | not run | not run | not run | not run | not run |
| Vega 4B | base | none | not run | not run | not run | not run | — |
| Vega 4B | VegaML's TTT head (the notebook's 50 labelled rows per task) | answer key | not run | not run | not run | not run | not run |
| Vega 4B | answer-key adapter | answer key | not run | not run | not run | not run | not run |
| Vega 4B | **Peekaboo** | right / wrong | not run | not run | not run | not run | not run |

## 4. Peekaboo head to head (accuracy points, paired 95% CI; positive = Peekaboo better)

| benchmark | model | Peekaboo vs base model | Peekaboo vs their improvement (answer key) |
|---|---|---|---|
| public | Vega 0.8B | not run | TTT: not run · adapter: not run |
| typed | Vega 0.8B | not run | TTT: not run · adapter: not run |
| jevbench | Vega 0.8B | not run | — (Vega has none) |
| public | Vega 4B | not run | TTT: not run · adapter: not run |
| typed | Vega 4B | not run | TTT: not run · adapter: not run |
| jevbench | Vega 4B | not run | — (Vega has none) |
| public | Laya | not run | laya-typed-decisions: not run |
| jevbench | Laya | not run | laya-typed-decisions: not run |
| decisionbench | Laya | not run | laya-typed-decisions: not run |
| typed | Laya | not run | laya-typed-decisions: not run |

DecisionBench (Vega 0.8B / 4B) is in the paper: +10.2 / +6.0 over base, +8.0 / +5.8 over TTT, -1.6 / +1.2 against the answer-key adapter.

## 5. Checks

| model | suite | replay vs official answers, max abs. probability difference | rows the model could not answer |
|---|---|---|---|
| 0.8b | public | — | not run |
| 0.8b | jevbench | — | not run |
| 0.8b | typed | — | not run |
| 4b | public | — | not run |
| 4b | jevbench | — | not run |
| 4b | typed | — | not run |
| laya | public | — | not run |
| laya | jevbench | — | not run |
| laya | decisionbench | — | not run |
| laya | typed | — | not run |
| laya_td | public | — | not run |
| laya_td | jevbench | — | not run |
| laya_td | decisionbench | — | not run |
| laya_td | typed | — | not run |
