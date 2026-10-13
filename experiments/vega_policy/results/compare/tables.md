| method | learns from | seeds | accuracy | skill | NLL | ECE | KL to gold |
|---|---|---|---|---|---|---|---|
| Released Vega 0.8B | none | 1 | 0.509 | 0.362 | 1.628 | 0.098 | 1.012 |
| vegaml fit() TTT head, 'auto' | gold labels | 1 | 0.531 | 0.391 | 1.628 | 0.111 | 1.263 |
| Adapter, supervised + temp. scaling | gold labels | 3 | 0.627 ± 0.010 | 0.515 ± 0.013 | 1.183 ± 0.012 | 0.074 ± 0.017 | 0.786 ± 0.025 |
| No-engine control (bilinear), feedback | right/wrong only | 3 | 0.262 ± 0.023 | 0.041 ± 0.030 | 1.949 ± 0.020 | 0.115 ± 0.021 | 0.794 ± 0.079 |
| Ours, KL 1 | right/wrong only | 3 | 0.547 ± 0.002 | 0.412 ± 0.003 | 1.430 ± 0.007 | 0.076 ± 0.005 | 0.758 ± 0.012 |
| Ours: feedback RL + feedback calibration | right/wrong only | 3 | 0.611 ± 0.015 | 0.495 ± 0.019 | 1.291 ± 0.024 | 0.054 ± 0.008 | 1.101 ± 0.332 |

Paired accuracy gain, points [95% bootstrap CI], same 1,075 rows:
  vegaml fit() TTT head, 'auto'              vs released: +2.2 [-0.1, +4.4]
  Adapter, supervised + temp. scaling        vs released: +11.8 [+8.9, +14.6]   vs TTT: +9.6 [+6.9, +12.3]
  No-engine control (bilinear), feedback     vs released: -24.7 [-27.9, -21.5]   vs TTT: -26.9 [-30.0, -23.9]
  Ours, KL 1                                 vs released: +3.8 [+2.4, +5.2]   vs TTT: +1.6 [-0.7, +4.0]
  Ours: feedback RL + feedback calibration   vs released: +10.2 [+7.9, +12.6]   vs TTT: +8.0 [+5.6, +10.5]

TTT heads kept 20/43 tasks and answer 394/1075 rows; the rest fall back to the engine.
| rows | n | Released Vega 0.8B | vegaml fit() TTT head, 'auto' | Adapter, supervised + temp. scaling | No-engine control (bilinear), feedback | Ours, KL 1 | Ours: feedback RL + feedback calibration |
|---|---|---|---|---|---|---|---|
| TTT head answers | 394 | 0.558 | 0.619 | 0.661 | 0.409 | 0.597 | 0.646 |
| TTT falls back to engine | 681 | 0.480 | 0.480 | 0.607 | 0.177 | 0.518 | 0.591 |

| domain | Released Vega 0.8B | vegaml fit() TTT head, 'auto' | Adapter, supervised + temp. scaling | No-engine control (bilinear), feedback | Ours, KL 1 | Ours: feedback RL + feedback calibration |
|---|---|---|---|---|---|---|
| agent_control | 0.472 | 0.616 | 0.589 | 0.269 | 0.520 | 0.581 |
| browser_automation | 0.320 | 0.320 | 0.393 | 0.067 | 0.340 | 0.407 |
| code_search | 0.720 | 0.720 | 0.660 | 0.307 | 0.713 | 0.767 |
| coding_agents | 0.880 | 0.867 | 0.907 | 0.587 | 0.880 | 0.911 |
| databases | 0.400 | 0.400 | 0.360 | 0.160 | 0.453 | 0.440 |
| ecommerce | 0.360 | 0.440 | 0.413 | 0.240 | 0.427 | 0.387 |
| evidence_reasoning | 0.520 | 0.480 | 0.613 | 0.267 | 0.613 | 0.520 |
| finance | 0.300 | 0.300 | 0.520 | 0.320 | 0.420 | 0.440 |
| financial | 0.400 | 0.400 | 0.440 | 0.027 | 0.400 | 0.400 |
| games | 0.240 | 0.253 | 0.378 | 0.116 | 0.258 | 0.364 |
| general_assistant | 0.760 | 0.760 | 0.720 | 0.000 | 0.760 | 0.760 |
| home_automation | 0.440 | 0.440 | 0.813 | 0.227 | 0.467 | 0.760 |
| knowledge_graphs | 0.320 | 0.320 | 0.480 | 0.133 | 0.333 | 0.507 |
| legal | 0.600 | 0.720 | 0.673 | 0.373 | 0.640 | 0.693 |
| logic | 0.440 | 0.240 | 0.360 | 0.347 | 0.360 | 0.360 |
| multi_hop_search | 0.560 | 0.440 | 0.673 | 0.500 | 0.593 | 0.587 |
| online_safety | 0.480 | 0.720 | 0.840 | 0.307 | 0.560 | 0.860 |
| privacy | 0.880 | 0.880 | 0.947 | 0.253 | 0.893 | 0.920 |
| public_sector | 0.760 | 0.760 | 0.880 | 0.307 | 0.813 | 0.933 |
| robotics | 0.280 | 0.280 | 0.620 | 0.160 | 0.293 | 0.447 |
| search | 0.560 | 0.560 | 0.733 | 0.200 | 0.733 | 0.680 |
| social_media | 0.560 | 0.680 | 0.760 | 0.093 | 0.693 | 0.773 |
| software_agents | 0.840 | 0.840 | 0.880 | 0.133 | 0.840 | 0.880 |
| software_engineering | 0.440 | 0.440 | 0.627 | 0.213 | 0.480 | 0.680 |
| tax_documents | 0.040 | 0.040 | 0.187 | 0.027 | 0.040 | 0.080 |
| technical | 0.400 | 0.280 | 0.400 | 0.093 | 0.413 | 0.347 |
| web | 0.760 | 0.600 | 0.773 | 0.507 | 0.760 | 0.720 |
| web_content | 0.760 | 0.840 | 1.000 | 0.507 | 0.920 | 1.000 |
