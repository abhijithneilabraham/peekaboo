# The paper

`main.tex` is the current version (v2, after the review in `reports/critique.zip`); `main_v1.tex` is the first version it replaced.

## Regenerating everything from the results

    VEGA_OUT=experiments/peekaboo_remaining/output_v2 python experiments/peekaboo_remaining/analysis.py   # analysis.json: intervals, controls, baselines
    VEGA_OUT=experiments/peekaboo_remaining/output_v2 python paper/make_figures.py                        # figures/*.pdf, tables_*.tex, numbers.tex
    cd paper && tectonic main.tex                                                                          # main.pdf

Every number in the text is a macro from `numbers.tex`, every table is a `tables_*.tex` file and every figure is in `figures/`,
all produced by `make_figures.py` from `analysis.json` and the per-row records. Nothing is typed in by hand.

## What to upload to arXiv

`main.tex`, `abstract.tex`, `intro_findings.tex`, `results.tex`, `limitations.tex`, `conclusion.tex`, `appendix.tex`,
`numbers.tex`, `tables_*.tex`, `refs.bib`, `main.bbl` (after one compile) and `figures/`.
