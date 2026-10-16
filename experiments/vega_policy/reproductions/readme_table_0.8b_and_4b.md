| benchmark | Vega 0.8B | **0.8B + ours** | Vega 4B | **4B + ours** | Jev (README) |
|---|---|---|---|---|---|
| DecisionBench domain:privacy, 25 items | 0.880 | 0.880 ± 0.033 | **1.000** | **1.000** | 0.960 |
| DecisionBench domain:public_sector, 25 items | 0.720 | 0.880 ± 0.065 | **0.960** | 0.947 ± 0.019 | 0.880 |
| DecisionBench, accuracy | 0.506 | 0.611 ± 0.004 | 0.635 | 0.695 ± 0.002 | **0.752** |
| DecisionBench, calibration error (ECE) | 0.107 | 0.056 ± 0.007 | 0.097 | **0.045 ± 0.006** | 0.122 |
| DecisionBench, negative log likelihood | 1.635 | 1.312 ± 0.042 | 1.425 | **1.100 ± 0.016** | 3.196 |
| DecisionBench, KL to gold distribution (10 rows) | 1.011 | 0.992 ± 0.249 | 0.850 | **0.793 ± 0.032** | 6.099 |
| DecisionBench, rows answered | **1.000** | **1.000** | **1.000** | **1.000** | 0.997 |

Paired accuracy gain of ours over the released model, same 1,075 rows, bootstrap 95% CI:
- 0.8B: +10.4 [+8.0, +13.0] points accuracy, 3 seeds
- 4B: +6.0 [+4.1, +7.9] points accuracy, 3 seeds
