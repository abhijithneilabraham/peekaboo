| method | learns from | seeds | accuracy | skill | NLL | ECE | KL to gold |
|---|---|---|---|---|---|---|---|
| Released Vega 4B | none | 1 | 0.635 | 0.526 | 1.425 | 0.095 | 0.850 |
| vegaml fit() TTT head, 'auto' | gold labels | 1 | 0.637 | 0.529 | 1.435 | 0.113 | 1.206 |
| Adapter, supervised + temp. scaling | gold labels | 3 | 0.683 ± 0.015 | 0.588 ± 0.019 | 1.043 ± 0.020 | 0.051 ± 0.011 | 0.622 ± 0.017 |
| No-engine control (bilinear), feedback | right/wrong only | 3 | 0.242 ± 0.021 | 0.021 ± 0.023 | 1.965 ± 0.019 | 0.143 ± 0.009 | 0.726 ± 0.038 |
| Ours, KL 1 | right/wrong only | 3 | 0.656 ± 0.004 | 0.553 ± 0.005 | 1.142 ± 0.005 | 0.032 ± 0.001 | 0.720 ± 0.003 |
| Ours: feedback RL + feedback calibration | right/wrong only | 3 | 0.695 ± 0.002 | 0.604 ± 0.003 | 1.100 ± 0.016 | 0.035 ± 0.002 | 0.793 ± 0.032 |

Paired accuracy gain, points [95% bootstrap CI], same 1,075 rows:
  vegaml fit() TTT head, 'auto'              vs released: +0.2 [-1.7, +2.0]
  Adapter, supervised + temp. scaling        vs released: +4.7 [+2.3, +7.3]   vs TTT: +4.6 [+2.1, +6.8]
  No-engine control (bilinear), feedback     vs released: -39.3 [-42.4, -36.0]   vs TTT: -39.5 [-42.6, -36.2]
  Ours, KL 1                                 vs released: +2.0 [+1.0, +3.2]   vs TTT: +1.9 [-0.1, +3.7]
  Ours: feedback RL + feedback calibration   vs released: +6.0 [+4.0, +7.9]   vs TTT: +5.8 [+3.7, +7.8]

TTT heads kept 20/43 tasks and answer 394/1075 rows; the rest fall back to the engine.
| rows | n | Released Vega 4B | vegaml fit() TTT head, 'auto' | Adapter, supervised + temp. scaling | No-engine control (bilinear), feedback | Ours, KL 1 | Ours: feedback RL + feedback calibration |
|---|---|---|---|---|---|---|---|
| TTT head answers | 394 | 0.662 | 0.668 | 0.714 | 0.341 | 0.685 | 0.727 |
| TTT falls back to engine | 681 | 0.620 | 0.620 | 0.665 | 0.186 | 0.639 | 0.676 |

| domain | Released Vega 4B | vegaml fit() TTT head, 'auto' | Adapter, supervised + temp. scaling | No-engine control (bilinear), feedback | Ours, KL 1 | Ours: feedback RL + feedback calibration |
|---|---|---|---|---|---|---|
| agent_control | 0.624 | 0.664 | 0.675 | 0.272 | 0.637 | 0.651 |
| browser_automation | 0.540 | 0.540 | 0.487 | 0.073 | 0.560 | 0.553 |
| code_search | 0.760 | 0.760 | 0.687 | 0.193 | 0.760 | 0.767 |
| coding_agents | 0.787 | 0.893 | 0.933 | 0.351 | 0.849 | 0.933 |
| databases | 0.400 | 0.400 | 0.373 | 0.173 | 0.400 | 0.480 |
| ecommerce | 0.520 | 0.480 | 0.507 | 0.307 | 0.493 | 0.427 |
| evidence_reasoning | 0.800 | 0.680 | 0.773 | 0.227 | 0.800 | 0.840 |
| finance | 0.420 | 0.420 | 0.567 | 0.260 | 0.453 | 0.540 |
| financial | 0.760 | 0.760 | 0.707 | 0.000 | 0.720 | 0.680 |
| games | 0.400 | 0.400 | 0.444 | 0.187 | 0.436 | 0.418 |
| general_assistant | 0.760 | 0.760 | 0.813 | 0.013 | 0.787 | 0.773 |
| home_automation | 0.640 | 0.640 | 0.880 | 0.213 | 0.640 | 0.893 |
| knowledge_graphs | 0.800 | 0.800 | 0.787 | 0.240 | 0.800 | 0.867 |
| legal | 0.860 | 0.840 | 0.813 | 0.393 | 0.873 | 0.847 |
| logic | 0.600 | 0.360 | 0.520 | 0.427 | 0.587 | 0.533 |
| multi_hop_search | 0.500 | 0.360 | 0.653 | 0.453 | 0.520 | 0.647 |
| online_safety | 0.600 | 0.720 | 0.853 | 0.453 | 0.660 | 0.860 |
| privacy | 1.000 | 1.000 | 0.973 | 0.160 | 1.000 | 1.000 |
| public_sector | 0.960 | 0.960 | 0.907 | 0.160 | 0.960 | 0.947 |
| robotics | 0.540 | 0.540 | 0.540 | 0.133 | 0.607 | 0.620 |
| search | 0.720 | 0.720 | 0.773 | 0.200 | 0.733 | 0.813 |
| social_media | 0.840 | 0.720 | 0.787 | 0.133 | 0.840 | 0.880 |
| software_agents | 0.920 | 0.920 | 0.920 | 0.120 | 0.920 | 0.920 |
| software_engineering | 0.520 | 0.480 | 0.667 | 0.187 | 0.520 | 0.587 |
| tax_documents | 0.120 | 0.120 | 0.133 | 0.027 | 0.120 | 0.120 |
| technical | 0.280 | 0.320 | 0.373 | 0.093 | 0.293 | 0.373 |
| web | 0.760 | 0.800 | 0.760 | 0.493 | 0.840 | 0.773 |
| web_content | 0.800 | 0.920 | 1.000 | 0.360 | 0.840 | 1.000 |
