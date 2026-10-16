# Independent reproductions

Runs of this folder's experiments on another machine (Windows 11, RTX 3080 Laptop 16 GB), with the old `run.sh`
(commit ffe9cf3). Both covered DecisionBench only; the public menu and JevBench came later with `reproduce.py`.

| file | what |
|---|---|
| `SUMMARY_0.8B.txt` | Vega 0.8B rerun: the headline reproduces (+10.4 points against the paper's +10.2) |
| `SUMMARY_4B.txt` | Vega 4B, first run: +6.0 points over the released 4B; the raw results are in `../results_4b/` |
| `readme_table_0.8b_and_4b.md` | the README-format table from the 4B run, both sizes |
| `4b/` | the run logs (including the crashed attempt) and the patch for the NaN crash it found |

The crash fix (the no-engine control masked padding with -1e4, which 4B logits can exceed) is applied in `adapt.py`.
