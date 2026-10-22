# Peekaboo against Vega and Laya, on their own benchmarks

"base" = the model as released, measured in this run. "+ Peekaboo" = the same model plus our adapter, trained only from right / wrong feedback (mean ± s.d. over 3 seeds). "README" = the number Vega or Laya publish. Bold = best in the row.

## 0. Everything: every model, method and ablation on every benchmark

Accuracy (mean ± s.d. over 3 seeds), then calibration error, then the paired accuracy gain over the base model in points with a 95% bootstrap CI. JevBench: per-suite Peekaboo learns by 5-fold cross-fitting; pooled Peekaboo never sees a JevBench row. "switch" = the add-on is used only where held-out right / wrong feedback says it beats the base model.

### accuracy

| model | method | learns from | public | jevbench | decisionbench | typed | emotion | agnews |
|---|---|---|---|---|---|---|---|---|
| Vega 0.8B | base model | none | 0.556 | 0.545 | 0.509 | 0.666 | 0.568 | 0.901 |
| Vega 0.8B | base model, shipped adapters off (typed-decisions) | none | · | · | · | 0.377 | · | · |
| Vega 0.8B | VegaML's TTT head (vegaml 0.8.0 rule: head where it beats chance) | answer key | 0.516 | · | 0.530 | 0.719 | 0.691 | 0.900 |
| Vega 0.8B | VegaML's TTT head (vegaml 0.12.3 rule: head where it beats chance and the engine) | answer key | 0.576 | · | 0.532 | 0.719 | 0.691 | 0.901 |
| Vega 0.8B | answer-key adapter, same recipe (Laya: answer-key head) | answer key | 0.590 ± 0.003 | 0.537 ± 0.016 | 0.632 ± 0.003 | 0.737 ± 0.001 | 0.679 ± 0.006 | 0.901 ± 0.003 |
| Vega 0.8B | answer-key adapter, early-stopped on calibration labels | answer key | 0.603 ± 0.009 | 0.538 ± 0.004 | 0.641 ± 0.008 | 0.740 ± 0.002 | 0.676 ± 0.005 | 0.917 ± 0.002 |
| Vega 0.8B | answer-key adapter, one pass | answer key | 0.592 ± 0.002 | 0.540 ± 0.002 | 0.608 ± 0.006 | 0.734 ± 0.005 | 0.635 ± 0.002 | 0.917 ± 0.002 |
| Vega 0.8B | **Peekaboo** (8 passes) | right / wrong | 0.593 ± 0.010 | 0.522 ± 0.008 | 0.622 ± 0.002 | 0.728 ± 0.006 | 0.627 ± 0.004 | 0.912 ± 0.003 |
| Vega 0.8B | **Peekaboo, one pass** (each row rated once) | right / wrong | 0.575 ± 0.007 | 0.540 ± 0.002 | 0.576 ± 0.009 | 0.708 ± 0.002 | 0.618 ± 0.005 | 0.914 ± 0.002 |
| Vega 0.8B | Peekaboo + switch (best answers compared) | right / wrong | 0.593 ± 0.010 | 0.545 ± 0.000 | 0.622 ± 0.002 | 0.728 ± 0.006 | 0.627 ± 0.004 | 0.912 ± 0.003 |
| Vega 0.8B | Peekaboo without the honesty fix | right / wrong | 0.593 ± 0.010 | 0.522 ± 0.008 | 0.622 ± 0.002 | 0.728 ± 0.006 | 0.627 ± 0.004 | 0.912 ± 0.003 |
| Vega 0.8B | Peekaboo without the KL term | right / wrong | 0.582 | 0.517 ± 0.011 | 0.609 ± 0.003 | 0.709 | 0.622 | 0.916 |
| Vega 0.8B | logged feedback: the released model's best answers, rated once | right / wrong (logged) | · | · | 0.583 ± 0.007 | 0.726 | 0.603 | · |
| Vega 0.8B | logged feedback: the released model's sampled answers, rated once (clipped IPS) | right / wrong (logged) | · | · | 0.592 ± 0.009 | 0.722 | 0.634 | · |
| Vega 0.8B | Peekaboo, 20% of ratings flipped | right / wrong (noisy) | · | · | 0.600 ± 0.002 | 0.725 | 0.615 | · |
| Vega 0.8B | Peekaboo, half of the answers rated | right / wrong (missing) | · | · | 0.611 ± 0.001 | 0.719 | 0.633 | · |
| Vega 0.8B | Peekaboo, every wrong answer reported, 20% of right ones | right / wrong (complaints) | · | · | 0.609 ± 0.003 | 0.736 | 0.635 | · |
| Vega 0.8B | answer-key adapter, shipped adapters off | answer key | · | · | · | 0.693 ± 0.010 | · | · |
| Vega 0.8B | Peekaboo, shipped adapters off | right / wrong | · | · | · | 0.507 ± 0.023 | · | · |
| Vega 0.8B | Peekaboo, one pass, shipped adapters off | right / wrong | · | · | · | 0.498 ± 0.006 | · | · |
| Vega 4B | base model | none | 0.718 | 0.667 | 0.636 | 0.729 | 0.575 | 0.928 |
| Vega 4B | base model, shipped adapters off (typed-decisions) | none | · | · | · | 0.440 | · | · |
| Vega 4B | VegaML's TTT head (vegaml 0.8.0 rule: head where it beats chance) | answer key | 0.678 | · | 0.640 | 0.757 | 0.697 | 0.915 |
| Vega 4B | VegaML's TTT head (vegaml 0.12.3 rule: head where it beats chance and the engine) | answer key | 0.724 | · | 0.654 | 0.755 | 0.697 | 0.915 |
| Vega 4B | answer-key adapter, same recipe (Laya: answer-key head) | answer key | 0.706 ± 0.001 | 0.706 ± 0.009 | 0.687 ± 0.005 | 0.769 ± 0.000 | 0.640 ± 0.001 | 0.924 ± 0.003 |
| Vega 4B | answer-key adapter, early-stopped on calibration labels | answer key | 0.726 ± 0.004 | 0.677 ± 0.008 | 0.688 ± 0.004 | 0.760 ± 0.007 | 0.663 ± 0.003 | 0.931 ± 0.003 |
| Vega 4B | answer-key adapter, one pass | answer key | 0.723 ± 0.003 | 0.667 ± 0.000 | 0.683 ± 0.007 | 0.775 ± 0.007 | 0.631 ± 0.003 | 0.931 ± 0.003 |
| Vega 4B | **Peekaboo** (8 passes) | right / wrong | 0.706 ± 0.004 | 0.696 ± 0.004 | 0.694 ± 0.004 | 0.746 ± 0.008 | 0.629 ± 0.007 | 0.930 ± 0.002 |
| Vega 4B | **Peekaboo, one pass** (each row rated once) | right / wrong | 0.718 ± 0.014 | 0.664 ± 0.002 | 0.669 ± 0.003 | 0.749 ± 0.003 | 0.607 ± 0.003 | 0.930 ± 0.002 |
| Vega 4B | Peekaboo + switch (best answers compared) | right / wrong | 0.706 ± 0.004 | 0.668 ± 0.002 | 0.694 ± 0.004 | 0.738 ± 0.013 | 0.629 ± 0.007 | 0.930 ± 0.002 |
| Vega 4B | Peekaboo without the honesty fix | right / wrong | 0.706 ± 0.004 | 0.696 ± 0.004 | 0.694 ± 0.004 | 0.746 ± 0.008 | 0.629 ± 0.007 | 0.930 ± 0.002 |
| Vega 4B | Peekaboo without the KL term | right / wrong | 0.713 | 0.693 | 0.672 ± 0.002 | 0.742 | 0.519 | 0.928 |
| Vega 4B | logged feedback: the released model's best answers, rated once | right / wrong (logged) | · | · | 0.655 ± 0.003 | 0.769 | 0.611 | · |
| Vega 4B | logged feedback: the released model's sampled answers, rated once (clipped IPS) | right / wrong (logged) | · | · | 0.680 ± 0.002 | 0.748 | 0.620 | · |
| Vega 4B | Peekaboo, 20% of ratings flipped | right / wrong (noisy) | · | · | 0.679 ± 0.006 | 0.756 | 0.622 | · |
| Vega 4B | Peekaboo, half of the answers rated | right / wrong (missing) | · | · | 0.684 ± 0.005 | 0.764 | 0.575 | · |
| Vega 4B | Peekaboo, every wrong answer reported, 20% of right ones | right / wrong (complaints) | · | · | 0.694 ± 0.003 | 0.770 | 0.602 | · |
| Vega 4B | answer-key adapter, shipped adapters off | answer key | · | · | · | 0.748 ± 0.005 | · | · |
| Vega 4B | Peekaboo, shipped adapters off | right / wrong | · | · | · | 0.685 ± 0.022 | · | · |
| Vega 4B | Peekaboo, one pass, shipped adapters off | right / wrong | · | · | · | 0.661 ± 0.003 | · | · |
| Laya | base model | none | 0.540 | 0.576 | 0.406 | 0.362 | 0.599 | 0.930 |
| Laya | answer-key adapter, one pass | answer key | 0.541 | 0.576 | 0.403 | 0.412 ± 0.009 | · | · |
| Laya | laya-typed-decisions (Laya's released fine-tune, typed-decisions only) | answer key (typed-decisions) | 0.550 | 0.541 | 0.436 | 0.761 | 0.604 | 0.930 |
| Laya | **Peekaboo, one pass** (each row rated once) | right / wrong | 0.541 | 0.576 | 0.406 | 0.364 ± 0.002 | · | · |
| Laya | **Peekaboo, encoder unfrozen** (8 passes) | right / wrong | 0.566 ± 0.001 | 0.558 ± 0.025 | 0.545 ± 0.010 | 0.685 | 0.820 | · |
| Laya | Peekaboo, encoder unfrozen, one pass | right / wrong | 0.562 ± 0.004 | 0.573 ± 0.004 | 0.482 ± 0.018 | 0.588 | 0.698 | · |
| Laya | answer-key fine-tune, encoder unfrozen, same recipe | answer key | 0.587 ± 0.007 | 0.535 ± 0.020 | 0.560 ± 0.012 | 0.750 | 0.899 | · |
| Laya | answer-key fine-tune, encoder unfrozen, early-stopped | answer key | 0.584 ± 0.002 | 0.595 ± 0.004 | 0.554 ± 0.003 | 0.750 | 0.905 | · |
| Laya | Peekaboo, encoder unfrozen + switch | right / wrong | 0.566 ± 0.001 | 0.569 ± 0.019 | 0.545 ± 0.010 | 0.685 | 0.820 | · |
| Laya | Peekaboo, encoder unfrozen, without the honesty fix | right / wrong | 0.566 ± 0.001 | 0.558 ± 0.025 | 0.545 ± 0.010 | 0.685 | 0.820 | · |

### calibration error (ECE, lower is better)

| model | method | learns from | public | jevbench | decisionbench | typed | emotion | agnews |
|---|---|---|---|---|---|---|---|---|
| Vega 0.8B | base model | none | 0.150 | 0.184 | 0.101 | 0.063 | 0.031 | 0.021 |
| Vega 0.8B | base model, shipped adapters off (typed-decisions) | none | · | · | · | 0.234 | · | · |
| Vega 0.8B | VegaML's TTT head (vegaml 0.8.0 rule: head where it beats chance) | answer key | 0.219 | · | 0.110 | 0.142 | 0.041 | 0.040 |
| Vega 0.8B | VegaML's TTT head (vegaml 0.12.3 rule: head where it beats chance and the engine) | answer key | 0.163 | · | 0.107 | 0.069 | 0.041 | 0.021 |
| Vega 0.8B | answer-key adapter, same recipe (Laya: answer-key head) | answer key | 0.174 ± 0.002 | 0.086 ± 0.005 | 0.075 ± 0.011 | 0.076 ± 0.012 | 0.033 ± 0.003 | 0.024 ± 0.006 |
| Vega 0.8B | answer-key adapter, early-stopped on calibration labels | answer key | 0.159 ± 0.013 | 0.074 ± 0.014 | 0.052 ± 0.016 | 0.085 ± 0.007 | 0.034 ± 0.002 | 0.012 ± 0.001 |
| Vega 0.8B | answer-key adapter, one pass | answer key | 0.148 ± 0.003 | 0.074 ± 0.011 | 0.057 ± 0.009 | 0.072 ± 0.009 | 0.037 ± 0.002 | 0.012 ± 0.001 |
| Vega 0.8B | **Peekaboo** (8 passes) | right / wrong | 0.187 ± 0.018 | 0.093 ± 0.005 | 0.053 ± 0.014 | 0.055 ± 0.009 | 0.052 ± 0.016 | 0.023 ± 0.007 |
| Vega 0.8B | **Peekaboo, one pass** (each row rated once) | right / wrong | 0.177 ± 0.010 | 0.079 ± 0.004 | 0.059 ± 0.008 | 0.059 ± 0.003 | 0.046 ± 0.012 | 0.012 ± 0.004 |
| Vega 0.8B | Peekaboo + switch (best answers compared) | right / wrong | 0.187 ± 0.018 | 0.184 ± 0.000 | 0.053 ± 0.014 | 0.055 ± 0.009 | 0.052 ± 0.016 | 0.023 ± 0.007 |
| Vega 0.8B | Peekaboo without the honesty fix | right / wrong | 0.308 ± 0.020 | 0.275 ± 0.010 | 0.257 ± 0.025 | 0.200 ± 0.014 | 0.293 ± 0.036 | 0.073 ± 0.002 |
| Vega 0.8B | Peekaboo without the KL term | right / wrong | 0.209 | 0.086 ± 0.006 | 0.077 ± 0.026 | 0.058 | 0.066 | 0.017 |
| Vega 0.8B | logged feedback: the released model's best answers, rated once | right / wrong (logged) | · | · | 0.047 ± 0.003 | 0.092 | 0.046 | · |
| Vega 0.8B | logged feedback: the released model's sampled answers, rated once (clipped IPS) | right / wrong (logged) | · | · | 0.067 ± 0.016 | 0.050 | 0.032 | · |
| Vega 0.8B | Peekaboo, 20% of ratings flipped | right / wrong (noisy) | · | · | 0.106 ± 0.006 | 0.111 | 0.123 | · |
| Vega 0.8B | Peekaboo, half of the answers rated | right / wrong (missing) | · | · | 0.059 ± 0.007 | 0.066 | 0.031 | · |
| Vega 0.8B | Peekaboo, every wrong answer reported, 20% of right ones | right / wrong (complaints) | · | · | 0.262 ± 0.009 | 0.167 | 0.366 | · |
| Vega 0.8B | answer-key adapter, shipped adapters off | answer key | · | · | · | 0.034 ± 0.009 | · | · |
| Vega 0.8B | Peekaboo, shipped adapters off | right / wrong | · | · | · | 0.069 ± 0.026 | · | · |
| Vega 0.8B | Peekaboo, one pass, shipped adapters off | right / wrong | · | · | · | 0.094 ± 0.042 | · | · |
| Vega 4B | base model | none | 0.146 | 0.162 | 0.095 | 0.051 | 0.056 | 0.025 |
| Vega 4B | base model, shipped adapters off (typed-decisions) | none | · | · | · | 0.199 | · | · |
| Vega 4B | VegaML's TTT head (vegaml 0.8.0 rule: head where it beats chance) | answer key | 0.173 | · | 0.111 | 0.113 | 0.033 | 0.031 |
| Vega 4B | VegaML's TTT head (vegaml 0.12.3 rule: head where it beats chance and the engine) | answer key | 0.148 | · | 0.093 | 0.043 | 0.033 | 0.031 |
| Vega 4B | answer-key adapter, same recipe (Laya: answer-key head) | answer key | 0.173 ± 0.005 | 0.096 ± 0.024 | 0.062 ± 0.008 | 0.056 ± 0.012 | 0.054 ± 0.014 | 0.011 ± 0.001 |
| Vega 4B | answer-key adapter, early-stopped on calibration labels | answer key | 0.138 ± 0.011 | 0.070 ± 0.003 | 0.059 ± 0.016 | 0.074 ± 0.009 | 0.042 ± 0.003 | 0.011 ± 0.003 |
| Vega 4B | answer-key adapter, one pass | answer key | 0.131 ± 0.002 | 0.064 ± 0.010 | 0.048 ± 0.009 | 0.055 ± 0.007 | 0.037 ± 0.008 | 0.010 ± 0.002 |
| Vega 4B | **Peekaboo** (8 passes) | right / wrong | 0.167 ± 0.005 | 0.076 ± 0.010 | 0.047 ± 0.007 | 0.051 ± 0.007 | 0.049 ± 0.008 | 0.012 ± 0.003 |
| Vega 4B | **Peekaboo, one pass** (each row rated once) | right / wrong | 0.143 ± 0.006 | 0.064 ± 0.012 | 0.040 ± 0.006 | 0.055 ± 0.015 | 0.028 ± 0.003 | 0.013 ± 0.001 |
| Vega 4B | Peekaboo + switch (best answers compared) | right / wrong | 0.167 ± 0.005 | 0.156 ± 0.009 | 0.047 ± 0.007 | 0.051 ± 0.001 | 0.049 ± 0.008 | 0.016 ± 0.007 |
| Vega 4B | Peekaboo without the honesty fix | right / wrong | 0.244 ± 0.011 | 0.192 ± 0.003 | 0.237 ± 0.013 | 0.185 ± 0.007 | 0.283 ± 0.006 | 0.046 ± 0.015 |
| Vega 4B | Peekaboo without the KL term | right / wrong | 0.191 | 0.072 | 0.088 ± 0.012 | 0.045 | 0.038 | 0.012 |
| Vega 4B | logged feedback: the released model's best answers, rated once | right / wrong (logged) | · | · | 0.040 ± 0.007 | 0.065 | 0.070 | · |
| Vega 4B | logged feedback: the released model's sampled answers, rated once (clipped IPS) | right / wrong (logged) | · | · | 0.043 ± 0.011 | 0.042 | 0.028 | · |
| Vega 4B | Peekaboo, 20% of ratings flipped | right / wrong (noisy) | · | · | 0.102 ± 0.023 | 0.121 | 0.057 | · |
| Vega 4B | Peekaboo, half of the answers rated | right / wrong (missing) | · | · | 0.055 ± 0.014 | 0.029 | 0.038 | · |
| Vega 4B | Peekaboo, every wrong answer reported, 20% of right ones | right / wrong (complaints) | · | · | 0.256 ± 0.035 | 0.180 | 0.344 | · |
| Vega 4B | answer-key adapter, shipped adapters off | answer key | · | · | · | 0.028 ± 0.004 | · | · |
| Vega 4B | Peekaboo, shipped adapters off | right / wrong | · | · | · | 0.053 ± 0.013 | · | · |
| Vega 4B | Peekaboo, one pass, shipped adapters off | right / wrong | · | · | · | 0.078 ± 0.022 | · | · |
| Laya | base model | none | 0.217 | 0.125 | 0.165 | 0.182 | 0.303 | 0.015 |
| Laya | answer-key adapter, one pass | answer key | 0.154 | 0.083 | 0.064 | 0.048 ± 0.006 | · | · |
| Laya | laya-typed-decisions (Laya's released fine-tune, typed-decisions only) | answer key (typed-decisions) | 0.168 | 0.102 | 0.048 | 0.205 | 0.185 | 0.106 |
| Laya | **Peekaboo, one pass** (each row rated once) | right / wrong | 0.163 | 0.107 | 0.075 | 0.034 ± 0.013 | · | · |
| Laya | **Peekaboo, encoder unfrozen** (8 passes) | right / wrong | 0.157 ± 0.006 | 0.115 ± 0.007 | 0.052 ± 0.000 | 0.077 | 0.094 | · |
| Laya | Peekaboo, encoder unfrozen, one pass | right / wrong | 0.159 ± 0.001 | 0.110 ± 0.016 | 0.070 ± 0.012 | 0.033 | 0.090 | · |
| Laya | answer-key fine-tune, encoder unfrozen, same recipe | answer key | 0.179 ± 0.009 | 0.148 ± 0.015 | 0.057 ± 0.001 | 0.045 | 0.035 | · |
| Laya | answer-key fine-tune, encoder unfrozen, early-stopped | answer key | 0.169 ± 0.004 | 0.109 ± 0.010 | 0.046 ± 0.008 | 0.057 | 0.029 | · |
| Laya | Peekaboo, encoder unfrozen + switch | right / wrong | 0.157 ± 0.006 | 0.114 ± 0.011 | 0.052 ± 0.000 | 0.077 | 0.094 | · |
| Laya | Peekaboo, encoder unfrozen, without the honesty fix | right / wrong | 0.324 ± 0.017 | 0.298 ± 0.022 | 0.252 ± 0.081 | 0.257 | 0.092 | · |

### gain over the base model

| model | method | learns from | public | jevbench | decisionbench | typed | emotion | agnews |
|---|---|---|---|---|---|---|---|---|
| Vega 0.8B | base model | none | — | — | — | — | — | — |
| Vega 0.8B | base model, shipped adapters off (typed-decisions) | none | · | · | · | -28.9 [-31.4, -26.6] | · | · |
| Vega 0.8B | VegaML's TTT head (vegaml 0.8.0 rule: head where it beats chance) | answer key | -4.1 [-6.1, -2.0] | · | +2.1 [-0.1, +4.4] | +5.3 [+2.8, +7.7] | +12.3 [+9.8, +14.9] | -0.2 [-1.5, +1.2] |
| Vega 0.8B | VegaML's TTT head (vegaml 0.12.3 rule: head where it beats chance and the engine) | answer key | +2.0 [+0.7, +3.3] | · | +2.3 [+0.3, +4.4] | +5.2 [+3.8, +6.7] | +12.3 [+9.8, +14.9] | +0.0 [+0.0, +0.0] |
| Vega 0.8B | answer-key adapter, same recipe (Laya: answer-key head) | answer key | +3.4 [+1.5, +5.1] | -0.9 [-6.5, +4.5] | +12.3 [+9.4, +15.1] | +7.0 [+5.2, +8.9] | +11.1 [+8.8, +13.3] | +0.0 [-1.0, +1.1] |
| Vega 0.8B | answer-key adapter, early-stopped on calibration labels | answer key | +4.7 [+3.0, +6.3] | -0.7 [-1.7, +0.1] | +13.2 [+10.6, +15.7] | +7.4 [+5.7, +9.3] | +10.8 [+8.5, +13.1] | +1.6 [+0.7, +2.5] |
| Vega 0.8B | answer-key adapter, one pass | answer key | +3.6 [+2.1, +5.0] | -0.6 [-1.7, +0.0] | +10.0 [+7.6, +12.3] | +6.8 [+5.0, +8.6] | +6.6 [+4.6, +8.8] | +1.6 [+0.7, +2.5] |
| Vega 0.8B | **Peekaboo** (8 passes) | right / wrong | +3.7 [+2.1, +5.2] | -2.3 [-6.6, +2.0] | +11.3 [+8.7, +13.8] | +6.2 [+4.4, +7.9] | +5.9 [+4.0, +7.7] | +1.1 [+0.2, +1.9] |
| Vega 0.8B | **Peekaboo, one pass** (each row rated once) | right / wrong | +1.9 [+0.6, +3.2] | -0.6 [-1.6, +0.1] | +6.7 [+4.7, +8.7] | +4.1 [+2.5, +5.9] | +5.0 [+3.1, +6.7] | +1.3 [+0.5, +2.1] |
| Vega 0.8B | Peekaboo + switch (best answers compared) | right / wrong | +3.7 [+2.1, +5.2] | +0.0 [+0.0, +0.0] | +11.3 [+8.7, +13.8] | +6.2 [+4.4, +7.9] | +5.9 [+4.0, +7.7] | +1.1 [+0.2, +1.9] |
| Vega 0.8B | Peekaboo without the honesty fix | right / wrong | +3.7 [+2.1, +5.2] | -2.3 [-6.6, +2.0] | +11.3 [+8.7, +13.8] | +6.2 [+4.4, +7.9] | +5.9 [+4.0, +7.7] | +1.1 [+0.2, +1.9] |
| Vega 0.8B | Peekaboo without the KL term | right / wrong | +2.5 [+0.7, +4.5] | -2.9 [-7.4, +1.9] | +10.0 [+7.4, +12.6] | +4.2 [+2.1, +6.6] | +5.4 [+3.1, +7.7] | +1.5 [+0.5, +2.6] |
| Vega 0.8B | logged feedback: the released model's best answers, rated once | right / wrong (logged) | · | · | +7.4 [+4.8, +10.0] | +6.0 [+4.0, +8.0] | +3.5 [+0.7, +6.2] | · |
| Vega 0.8B | logged feedback: the released model's sampled answers, rated once (clipped IPS) | right / wrong (logged) | · | · | +8.3 [+6.1, +10.7] | +5.6 [+3.5, +7.8] | +6.6 [+4.6, +8.6] | · |
| Vega 0.8B | Peekaboo, 20% of ratings flipped | right / wrong (noisy) | · | · | +9.1 [+6.9, +11.2] | +5.9 [+4.2, +7.4] | +4.7 [+3.1, +6.3] | · |
| Vega 0.8B | Peekaboo, half of the answers rated | right / wrong (missing) | · | · | +10.3 [+7.9, +12.6] | +5.2 [+3.2, +7.2] | +6.5 [+4.2, +8.9] | · |
| Vega 0.8B | Peekaboo, every wrong answer reported, 20% of right ones | right / wrong (complaints) | · | · | +10.0 [+7.7, +12.5] | +7.0 [+4.9, +9.1] | +6.7 [+4.5, +9.0] | · |
| Vega 0.8B | answer-key adapter, shipped adapters off | answer key | · | · | · | +2.7 [+0.7, +4.8] | · | · |
| Vega 0.8B | Peekaboo, shipped adapters off | right / wrong | · | · | · | -15.9 [-18.2, -13.7] | · | · |
| Vega 0.8B | Peekaboo, one pass, shipped adapters off | right / wrong | · | · | · | -16.8 [-19.3, -14.2] | · | · |
| Vega 4B | base model | none | — | — | — | — | — | — |
| Vega 4B | base model, shipped adapters off (typed-decisions) | none | · | · | · | -28.9 [-31.7, -26.1] | · | · |
| Vega 4B | VegaML's TTT head (vegaml 0.8.0 rule: head where it beats chance) | answer key | -3.9 [-5.8, -2.2] | · | +0.4 [-1.4, +2.2] | +2.8 [+0.8, +5.0] | +12.2 [+10.0, +14.5] | -1.3 [-2.5, -0.2] |
| Vega 4B | VegaML's TTT head (vegaml 0.12.3 rule: head where it beats chance and the engine) | answer key | +0.6 [-0.5, +1.8] | · | +1.8 [+0.1, +3.4] | +2.6 [+1.4, +3.9] | +12.2 [+10.0, +14.5] | -1.3 [-2.5, -0.2] |
| Vega 4B | answer-key adapter, same recipe (Laya: answer-key head) | answer key | -1.2 [-2.7, +0.2] | +3.9 [-1.2, +9.4] | +5.1 [+2.6, +7.7] | +4.0 [+2.7, +5.3] | +6.5 [+4.3, +8.7] | -0.3 [-1.0, +0.3] |
| Vega 4B | answer-key adapter, early-stopped on calibration labels | answer key | +0.8 [-0.3, +2.0] | +1.0 [-2.0, +4.2] | +5.1 [+2.8, +7.6] | +3.0 [+1.7, +4.4] | +8.8 [+6.9, +10.8] | +0.3 [-0.2, +0.9] |
| Vega 4B | answer-key adapter, one pass | answer key | +0.5 [-0.6, +1.6] | -0.0 [-0.6, +0.7] | +4.7 [+2.8, +6.5] | +4.6 [+3.2, +6.0] | +5.7 [+3.8, +7.6] | +0.3 [-0.1, +0.7] |
| Vega 4B | **Peekaboo** (8 passes) | right / wrong | -1.1 [-2.5, +0.1] | +2.9 [-1.4, +7.6] | +5.7 [+3.8, +7.8] | +1.7 [+0.3, +3.1] | +5.4 [+3.7, +7.1] | +0.2 [-0.2, +0.6] |
| Vega 4B | **Peekaboo, one pass** (each row rated once) | right / wrong | +0.0 [-1.3, +1.1] | -0.3 [-1.3, +0.3] | +3.2 [+1.7, +4.8] | +2.0 [+0.9, +3.2] | +3.2 [+1.7, +4.7] | +0.2 [-0.1, +0.5] |
| Vega 4B | Peekaboo + switch (best answers compared) | right / wrong | -1.1 [-2.5, +0.1] | +0.1 [-0.6, +0.9] | +5.7 [+3.8, +7.8] | +0.9 [+0.4, +1.5] | +5.4 [+3.7, +7.1] | +0.2 [-0.2, +0.6] |
| Vega 4B | Peekaboo without the honesty fix | right / wrong | -1.1 [-2.5, +0.1] | +2.9 [-1.4, +7.6] | +5.7 [+3.8, +7.8] | +1.7 [+0.3, +3.1] | +5.4 [+3.7, +7.1] | +0.2 [-0.2, +0.6] |
| Vega 4B | Peekaboo without the KL term | right / wrong | -0.5 [-2.0, +1.0] | +2.6 [-2.6, +8.2] | +3.5 [+1.5, +5.7] | +1.3 [-0.4, +3.0] | -5.6 [-8.0, -3.2] | +0.0 [+0.0, +0.0] |
| Vega 4B | logged feedback: the released model's best answers, rated once | right / wrong (logged) | · | · | +1.8 [-0.4, +4.3] | +4.0 [+2.6, +5.5] | +3.6 [+1.2, +5.9] | · |
| Vega 4B | logged feedback: the released model's sampled answers, rated once (clipped IPS) | right / wrong (logged) | · | · | +4.3 [+2.4, +6.3] | +1.8 [+0.1, +3.6] | +4.5 [+2.5, +6.6] | · |
| Vega 4B | Peekaboo, 20% of ratings flipped | right / wrong (noisy) | · | · | +4.2 [+2.5, +6.1] | +2.8 [+1.2, +4.2] | +4.7 [+2.7, +6.8] | · |
| Vega 4B | Peekaboo, half of the answers rated | right / wrong (missing) | · | · | +4.7 [+2.8, +7.0] | +3.5 [+1.8, +5.2] | +0.0 [+0.0, +0.0] | · |
| Vega 4B | Peekaboo, every wrong answer reported, 20% of right ones | right / wrong (complaints) | · | · | +5.7 [+3.9, +7.6] | +4.2 [+2.5, +6.0] | +2.7 [+0.9, +4.5] | · |
| Vega 4B | answer-key adapter, shipped adapters off | answer key | · | · | · | +1.9 [+0.2, +3.5] | · | · |
| Vega 4B | Peekaboo, shipped adapters off | right / wrong | · | · | · | -4.4 [-6.1, -2.6] | · | · |
| Vega 4B | Peekaboo, one pass, shipped adapters off | right / wrong | · | · | · | -6.8 [-8.8, -4.8] | · | · |
| Laya | base model | none | — | — | — | — | — | — |
| Laya | answer-key adapter, one pass | answer key | +0.1 [-0.2, +0.3] | +0.0 [+0.0, +0.0] | -0.3 [-0.7, +0.0] | +5.0 [+2.4, +7.3] | · | · |
| Laya | laya-typed-decisions (Laya's released fine-tune, typed-decisions only) | answer key (typed-decisions) | +1.0 [-0.2, +2.3] | -3.5 [-8.2, +2.2] | +3.1 [+0.7, +5.5] | +40.0 [+37.3, +42.5] | +0.5 [-0.6, +1.7] | +0.0 [-0.7, +0.6] |
| Laya | **Peekaboo, one pass** (each row rated once) | right / wrong | +0.1 [-0.1, +0.2] | +0.0 [+0.0, +0.0] | +0.1 [+0.0, +0.3] | +0.2 [-0.0, +0.4] | · | · |
| Laya | **Peekaboo, encoder unfrozen** (8 passes) | right / wrong | +2.6 [+1.2, +3.9] | -1.7 [-6.9, +3.3] | +14.0 [+11.2, +16.8] | +32.3 [+29.5, +34.8] | +22.2 [+19.9, +24.5] | · |
| Laya | Peekaboo, encoder unfrozen, one pass | right / wrong | +2.1 [+0.9, +3.3] | -0.3 [-3.3, +2.9] | +7.7 [+5.3, +10.1] | +22.6 [+20.1, +25.3] | +9.9 [+7.7, +12.2] | · |
| Laya | answer-key fine-tune, encoder unfrozen, same recipe | answer key | +4.6 [+3.2, +6.1] | -4.0 [-9.2, +1.3] | +15.4 [+12.5, +18.5] | +38.8 [+36.0, +41.4] | +30.0 [+27.5, +32.6] | · |
| Laya | answer-key fine-tune, encoder unfrozen, early-stopped | answer key | +4.3 [+2.9, +5.8] | +1.9 [-2.0, +5.9] | +14.8 [+12.0, +17.8] | +38.8 [+35.9, +41.3] | +30.6 [+28.1, +33.1] | · |
| Laya | Peekaboo, encoder unfrozen + switch | right / wrong | +2.6 [+1.2, +3.9] | -0.7 [-4.9, +3.5] | +14.0 [+11.2, +16.8] | +32.3 [+29.5, +34.8] | +22.2 [+19.9, +24.5] | · |
| Laya | Peekaboo, encoder unfrozen, without the honesty fix | right / wrong | +2.6 [+1.2, +3.9] | -1.7 [-6.9, +3.3] | +14.0 [+11.2, +16.8] | +32.3 [+29.5, +34.8] | +22.2 [+19.9, +24.5] | · |

## 1. Vega's README benchmark table, with Peekaboo

| benchmark | Vega 0.8B (README) | base Vega 0.8B | **Vega 0.8B + Peekaboo** | Vega 4B (README) | base Vega 4B | **Vega 4B + Peekaboo** | base Laya | laya-typed-decisions | **Laya + Peekaboo** | Jev (README) |
|---|---|---|---|---|---|---|---|---|---|---|
| **Public tasks the README lists (accuracy)** | | | | | | | | | | |
| unfair_tos | 0.737 | 0.737 | 0.567 ± 0.138 | **0.877** | **0.877** | 0.725 ± 0.117 | **0.877** | 0.842 | not run | 0.667 |
| tweet_stance_abortion | 0.364 | 0.364 | **0.576 ± 0.060** | 0.545 | 0.545 | 0.309 ± 0.026 | 0.309 | 0.309 | not run | 0.436 |
| prompt_injections | 0.746 | 0.746 | 0.849 ± 0.031 | 0.855 | 0.855 | **0.891 ± 0.030** | 0.654 | 0.600 | not run | 0.764 |
| tweet_sentiment | 0.424 | 0.441 | 0.497 ± 0.040 | **0.593** | 0.576 | 0.565 ± 0.049 | 0.559 | 0.491 | not run | 0.525 |
| tweet_offensive | 0.780 | 0.797 | **0.802 ± 0.021** | 0.780 | 0.780 | 0.780 ± 0.014 | 0.695 | 0.712 | not run | 0.729 |
| enron_spam | **1.000** | **1.000** | 0.989 ± 0.008 | **1.000** | **1.000** | **1.000 ± 0.000** | 0.917 | 0.900 | not run | 0.950 |
| jailbreak_cls | 0.983 | 0.983 | 0.971 ± 0.021 | **1.000** | **1.000** | 0.983 ± 0.000 | 0.983 | 0.983 | not run | 0.966 |
| glue_rte | 0.633 | 0.617 | 0.617 ± 0.024 | **0.883** | 0.867 | 0.856 ± 0.016 | 0.833 | 0.817 | not run | 0.850 |
| emotion | 0.550 | 0.550 | 0.528 ± 0.042 | **0.650** | 0.633 | 0.522 ± 0.028 | 0.567 | 0.567 | not run | 0.617 |
| toxic_chat_toxicity | 0.950 | 0.950 | 0.900 ± 0.027 | **1.000** | **1.000** | 0.983 ± 0.000 | 0.967 | 0.967 | not run | 0.983 |
| superglue_cb | 0.667 | 0.667 | 0.748 ± 0.042 | 0.822 | 0.822 | 0.778 ± 0.018 | **0.867** | 0.844 | not run | 0.800 |
| clinc150_oos | 0.683 | 0.683 | 0.617 ± 0.014 | 0.900 | **0.917** | 0.778 ± 0.042 | 0.000 | 0.317 | not run | 0.883 |
| **Slices** | | | | | | | | | | |
| JevBench family:temporal_numeric | **0.467** | **0.467** | 0.378 ± 0.031 | 0.400 | 0.400 | 0.333 ± 0.054 | 0.333 | 0.200 | not run | 0.200 |
| DecisionBench domain:privacy | 0.880 | 0.880 | 0.893 ± 0.019 | **1.000** | **1.000** | **1.000 ± 0.000** | 0.080 | 0.160 | not run | 0.960 |
| DecisionBench domain:public_sector | 0.760 | 0.720 | 0.933 ± 0.019 | **0.960** | **0.960** | 0.920 ± 0.000 | 0.360 | 0.400 | not run | 0.880 |
| **Public menu, 43 tasks, macro average** | | | | | | | | | | |
| accuracy | 0.556 | 0.556 | 0.593 ± 0.010 | 0.717 | 0.718 | 0.706 ± 0.004 | 0.540 | 0.550 | not run | **0.836** |
| skill, chance corrected | 0.388 | 0.387 | 0.427 ± 0.015 | 0.594 | 0.596 | 0.582 ± 0.008 | 0.375 | 0.378 | not run | **0.755** |
| macro F1 | 0.486 | 0.486 | 0.528 ± 0.007 | 0.649 | 0.649 | 0.644 ± 0.005 | 0.471 | 0.477 | not run | **0.796** |
| expected calibration error | 0.151 | 0.150 | 0.187 ± 0.018 | 0.148 | 0.146 | 0.167 ± 0.005 | 0.217 | 0.168 | not run | **0.115** |
| negative log likelihood | 1.099 | 1.100 | 1.114 ± 0.045 | **0.768** | **0.768** | 0.836 ± 0.017 | 1.651 | 1.170 | not run | 0.878 |
| area under risk-coverage | 0.362 | 0.363 | 0.322 ± 0.009 | 0.169 | 0.169 | 0.180 ± 0.007 | 0.366 | 0.349 | not run | **0.077** |
| latency per decision, p50 ms | **74** | 78 | 93 | 106 | 99 | 118 | 86 | 89 | not run | 162 |
| **JevBench, 231 items** | | | | | | | | | | |
| accuracy | 0.545 | 0.545 | 0.522 ± 0.008 | 0.662 | 0.667 | 0.696 ± 0.004 | 0.576 | 0.541 | not run | **0.866** |
| accuracy, hard tier | 0.414 | 0.423 | 0.384 ± 0.018 | 0.513 | 0.522 | 0.492 ± 0.015 | 0.333 | 0.279 | not run | **0.730** |
| paraphrase consistency | 0.694 | 0.722 | 0.796 ± 0.026 | 0.722 | 0.722 | 0.852 ± 0.035 | 0.667 | 0.750 | not run | **0.972** |
| expected calibration error | 0.178 | 0.184 | 0.093 ± 0.005 | 0.169 | 0.162 | 0.076 ± 0.010 | 0.125 | 0.102 | not run | **0.067** |
| **DecisionBench, 1,075 items** | | | | | | | | | | |
| accuracy | 0.509 | 0.509 | 0.622 ± 0.002 | 0.633 | 0.636 | 0.694 ± 0.004 | 0.406 | 0.436 | not run | **0.752** |
| expected calibration error | 0.103 | 0.101 | 0.053 ± 0.014 | 0.098 | 0.095 | **0.047 ± 0.007** | 0.165 | 0.048 | not run | 0.122 |
| negative log likelihood | 1.628 | 1.631 | 1.316 ± 0.008 | 1.420 | 1.423 | **1.095 ± 0.018** | 2.062 | 1.566 | not run | 3.196 |
| KL to the gold distribution, 10 rows | 1.012 | 1.024 | 1.311 ± 0.502 | **0.854** | 0.855 | **0.854 ± 0.190** | 1.104 | 0.929 | not run | 6.099 |
| coverage, rows answered | **1.000** | **1.000** | **1.000 ± 0.000** | **1.000** | **1.000** | **1.000 ± 0.000** | 0.954 | 0.990 | not run | 0.997 |
| latency per decision, p50 ms | 85 | 107 | 107 | 245 | 170 | 175 | **47** | 66 | not run | 163 |

Vega's DecisionBench rows: the finished runs in experiments/vega_policy/results*/compare (its latency is the README's). Laya is measured in this run on every row. laya-typed-decisions is Laya's released answer-key fine-tune (on typed-decisions).

## 2. Laya's README benchmark: typed-decisions (2,000 test decisions; feedback / labels from the 6,000 training decisions)

| model | learns from | accuracy (README) | accuracy | soft accuracy | ECE | NLL | gain vs base |
|---|---|---|---|---|---|---|---|
| base Laya | none | 0.362 | 0.362 | 0.326 | 0.182 | 1.361 | — |
| laya-typed-decisions (Laya's own fine-tune) | answer key | 0.766 | 0.761 | 0.471 | 0.205 | 0.706 | +40.0 [+37.3, +42.5] |
| **Laya + Peekaboo** | right / wrong |  | not run | not run | not run | not run | not run |
| **Laya + Peekaboo, encoder unfrozen** | right / wrong |  | 0.685 | 0.471 | 0.077 | 0.827 | +32.3 [+29.5, +34.8] |
| base Vega 0.8B | none |  | 0.666 | 0.501 | 0.063 | 0.753 | — |
| Vega 0.8B + VegaML's TTT head | answer key |  | 0.719 | 0.558 | 0.142 | 0.906 | +5.3 [+2.8, +7.7] |
| Vega 0.8B + answer-key adapter | answer key |  | 0.737 ± 0.001 | 0.557 ± 0.002 | 0.076 ± 0.012 | 0.685 ± 0.037 | +7.0 [+5.2, +8.9] |
| **Vega 0.8B + Peekaboo** | right / wrong |  | 0.728 ± 0.006 | 0.539 ± 0.009 | 0.055 ± 0.009 | 0.690 ± 0.033 | +6.2 [+4.4, +7.9] |
| base Vega 4B | none |  | 0.729 | 0.555 | 0.051 | 0.631 | — |
| Vega 4B + VegaML's TTT head | answer key |  | 0.757 | 0.577 | 0.113 | 0.756 | +2.8 [+0.8, +5.0] |
| Vega 4B + answer-key adapter | answer key |  | 0.769 ± 0.000 | 0.576 ± 0.005 | 0.056 ± 0.012 | 0.567 ± 0.008 | +4.0 [+2.7, +5.3] |
| **Vega 4B + Peekaboo** | right / wrong |  | 0.746 ± 0.008 | 0.558 ± 0.011 | 0.051 ± 0.007 | 0.637 ± 0.048 | +1.7 [+0.3, +3.1] |
| Jev 1.13.0 (dataset card) | none | 0.727 | | | | | |

Vega's README also reports typed-decisions (0.763 / 0.803 with its task adapter) on six other workflows whose rows are not published, so it cannot be rerun; the rows above are the public benchmark Laya reports.

## 3. The 43 public tasks: Peekaboo against VegaML's own ways to improve Vega

| model | method | learns from | accuracy | skill | ECE | NLL | gain vs base |
|---|---|---|---|---|---|---|---|
| Vega 0.8B | base | none | 0.556 | 0.387 | 0.150 | 1.100 | — |
| Vega 0.8B | VegaML's TTT head (the notebook's 50 labelled rows per task) | answer key | 0.516 | 0.322 | 0.219 | 1.692 | -4.1 [-6.1, -2.0] |
| Vega 0.8B | answer-key adapter | answer key | 0.590 ± 0.003 | 0.417 ± 0.003 | 0.174 ± 0.002 | 1.073 ± 0.024 | +3.4 [+1.5, +5.1] |
| Vega 0.8B | **Peekaboo** | right / wrong | 0.593 ± 0.010 | 0.427 ± 0.015 | 0.187 ± 0.018 | 1.114 ± 0.045 | +3.7 [+2.1, +5.2] |
| Vega 4B | base | none | 0.718 | 0.596 | 0.146 | 0.768 | — |
| Vega 4B | VegaML's TTT head (the notebook's 50 labelled rows per task) | answer key | 0.678 | 0.536 | 0.173 | 1.170 | -3.9 [-5.8, -2.2] |
| Vega 4B | answer-key adapter | answer key | 0.706 ± 0.001 | 0.577 ± 0.004 | 0.173 ± 0.005 | 0.840 ± 0.018 | -1.2 [-2.7, +0.2] |
| Vega 4B | **Peekaboo** | right / wrong | 0.706 ± 0.004 | 0.582 ± 0.008 | 0.167 ± 0.005 | 0.836 ± 0.017 | -1.1 [-2.5, +0.1] |

## 4. Peekaboo head to head (accuracy points, paired 95% CI; positive = Peekaboo better)

| benchmark | model | Peekaboo vs base model | Peekaboo vs their improvement (answer key) |
|---|---|---|---|
| public | Vega 0.8B | +3.7 [+2.1, +5.2] | TTT: +7.7 [+5.8, +9.6] · adapter: +0.3 [-1.0, +1.6] |
| typed | Vega 0.8B | +6.2 [+4.4, +7.9] | TTT: +0.9 [-1.0, +2.8] · adapter: -0.9 [-1.8, +0.1] |
| jevbench | Vega 0.8B | -2.3 [-6.6, +2.0] | — (Vega has none) |
| public | Vega 4B | -1.1 [-2.5, +0.1] | TTT: +2.8 [+1.0, +4.5] · adapter: +0.0 [-1.0, +1.0] |
| typed | Vega 4B | +1.7 [+0.3, +3.1] | TTT: -1.1 [-2.9, +0.6] · adapter: -2.3 [-3.2, -1.4] |
| jevbench | Vega 4B | +2.9 [-1.4, +7.6] | — (Vega has none) |
| public | Laya | not run | laya-typed-decisions: not run |
| jevbench | Laya | not run | laya-typed-decisions: not run |
| decisionbench | Laya | not run | laya-typed-decisions: not run |
| typed | Laya | not run | laya-typed-decisions: not run |
| public | Laya, encoder unfrozen | +2.6 [+1.2, +3.9] | laya-typed-decisions: +1.6 [+0.1, +3.1] |
| jevbench | Laya, encoder unfrozen | -1.7 [-6.9, +3.3] | laya-typed-decisions: +1.7 [-4.0, +7.4] |
| decisionbench | Laya, encoder unfrozen | +14.0 [+11.2, +16.8] | laya-typed-decisions: +10.9 [+8.0, +13.7] |
| typed | Laya, encoder unfrozen | +32.3 [+29.5, +34.8] | laya-typed-decisions: -7.6 [-9.6, -5.6] |
| emotion | Laya, encoder unfrozen | +22.2 [+19.9, +24.5] | laya-typed-decisions: +21.6 [+19.4, +24.0] |

DecisionBench (Vega 0.8B / 4B) is in the paper: +10.2 / +6.0 over base, +8.0 / +5.8 over TTT, -1.6 / +1.2 against the answer-key adapter.

## 5. Checks

| model | suite | replay vs official answers, max abs. probability difference | rows the model could not answer |
|---|---|---|---|
| 0.8b | public | 1.6e-16 | 0 |
| 0.8b | jevbench | 1.4e-16 | 0 |
| 0.8b | typed | 1.5e-16 | 0 |
| 4b | public | 1.6e-16 | 0 |
| 4b | jevbench | 1.4e-16 | 0 |
| 4b | typed | 1.4e-16 | 0 |
| laya | public | 5.0e-05 | 110 |
| laya | jevbench | 5.0e-05 | 0 |
| laya | decisionbench | 5.0e-05 | 195 |
| laya | typed | 5.0e-05 | 0 |
| laya_td | public | — | 0 |
| laya_td | jevbench | — | 0 |
| laya_td | decisionbench | — | 36 |
| laya_td | typed | — | 0 |
