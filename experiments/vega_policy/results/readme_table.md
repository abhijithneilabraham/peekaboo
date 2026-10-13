| benchmark | Vega 0.8B | **0.8B + ours** | Vega 4B (README) | **4B + ours** | Jev (README) |
|---|---|---|---|---|---|
| DecisionBench domain:privacy, 25 items | 0.880 | 0.920 | **1.000** | not run | 0.960 |
| DecisionBench domain:public_sector, 25 items | 0.760 | 0.933 ± 0.038 | **0.960** | not run | 0.880 |
| DecisionBench, accuracy | 0.509 | 0.611 ± 0.015 | 0.633 | not run | **0.752** |
| DecisionBench, calibration error (ECE) | 0.103 | **0.053 ± 0.009** | 0.098 | not run | 0.122 |
| DecisionBench, negative log likelihood | 1.628 | **1.291 ± 0.024** | 1.420 | not run | 3.196 |
| DecisionBench, KL to gold distribution (10 rows) | 1.012 | 1.101 ± 0.332 | **0.854** | not run | 6.099 |
| DecisionBench, rows answered | **1.000** | **1.000** | **1.000** | not run | 0.997 |

Paired accuracy gain of ours over the released model, same 1,075 rows, bootstrap 95% CI:
- 0.8B: +10.2 [+7.9, +12.5] points accuracy, 3 seeds
