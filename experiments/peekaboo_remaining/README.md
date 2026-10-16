# Remaining Peekaboo experiments: head to head with Vega and Laya

Only comparisons on Vega's and Laya's own benchmarks, against their own models and their own ways to improve them.

```bash
cd experiments/peekaboo_remaining
python run.py            # resumable; several steps at once on one GPU
python run.py --check    # 0 complete, 1 stopped with steps not done, 2 running
```

| benchmark (theirs) | their model, and their improvement (needs the answer key) | ours (right / wrong only) |
|---|---|---|
| 43 public tasks (Vega README) | base Vega · VegaML's TTT head · Vega answer-key adapter | Vega + Peekaboo |
| JevBench (Vega README) | base Vega | Vega + Peekaboo |
| typed-decisions (Laya README) | base Laya · laya-typed-decisions (Laya's own fine-tune, released) | Laya + Peekaboo |
| typed-decisions, same rows | base Vega · VegaML's TTT head · Vega answer-key adapter | Vega + Peekaboo |
| DecisionBench (Vega README) | already done: `experiments/vega_policy/results*/compare` | |

Vega 0.8B and 4B, 3 seeds. `output/TABLES.md` puts each benchmark in its owner's README layout with base,
their improvement and + Peekaboo side by side, plus paired 95% confidence intervals. `output/STATUS.md` lists
what is left.

**Hardware:** one GPU with at least 24 GB (the 4B backbone); 48 GB (L40S, A6000) runs 4 steps at once, about
2–3 hours. 80 GB disk, 32 GB RAM, 8 CPU cores. RunPod PyTorch template.

**With Claude Code:** open Claude Code in the repository and type `/run-peekaboo-remaining`. It follows
`CLAUDE.md`: starts the run, polls it, fixes failures, restarts, and reports when complete.

Vega's README also reports typed-decisions with its task adapter (0.763 / 0.803), but on six workflows whose rows
are not published, so that number cannot be rerun; the typed-decisions rows here are the public benchmark Laya
reports.
