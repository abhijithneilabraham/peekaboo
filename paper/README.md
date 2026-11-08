# The paper

`main.tex` is the current version (v2). The first version, its figure script and its PDF are at commit `cd52a9f`.

## Regenerating everything from the results

    VEGA_OUT=experiments/peekaboo_remaining/output_v2 python experiments/peekaboo_remaining/analysis.py   # analysis.json: intervals, controls, baselines
    VEGA_OUT=experiments/peekaboo_remaining/output_v2 python paper/make_figures.py                        # figures/*.pdf, tables_*.tex, numbers.tex
    cd paper && tectonic --keep-intermediates main.tex                                                                          # main.pdf

Main result estimates use macros from `numbers.tex`; result tables and figures are generated from `analysis.json` and per-row records. The seed-coverage table is generated from the available run files. Fixed configuration values and historical pilot disclosures are maintained in the text. Generated values still require consistency checks against the records.

The paper is an empirical study of simulated correctness feedback on text decision tasks. The primary comparison uses one verdict per training row; calibration and optional model selection use additional held-out feedback. It does not establish live-user effectiveness, multimodal improvement or retention, or full independent reproduction of the current grid. Historical pilot selection involved DecisionBench test rows and is disclosed in the appendix.

## What to upload to arXiv

`main.tex`, `abstract.tex`, `intro_findings.tex`, `results.tex`, `limitations.tex`, `conclusion.tex`, `appendix.tex`,
`numbers.tex`, `tables_*.tex`, `refs.bib`, `main.bbl` (retained by `--keep-intermediates`) and `figures/`.
