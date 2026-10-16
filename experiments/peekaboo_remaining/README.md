# Remaining Peekaboo experiments

One command runs everything that is not in the repo yet:

```bash
cd experiments/peekaboo_remaining
python run.py            # resumable; several steps at once on one GPU
python run.py --check    # 0 complete, 1 stopped with steps not done, 2 running
```

| what | base model | + Peekaboo | baselines |
|---|---|---|---|
| Vega 0.8B and 4B: 43 public tasks + JevBench | ✓ | ✓ | VegaML's TTT head, answer-key adapter, Peekaboo KL 1, no-engine control |
| Vega 0.8B and 4B: DecisionBench, realistic feedback | ✓ | ✓ clean, noisy (10/20/30%), half / quarter rated, complaints only | logged-feedback (off-policy) |
| Laya: DecisionBench, 43 public tasks, JevBench | ✓ | ✓ | Laya + RLCD (its own fine-tuning, answer key) |
| Laya: DecisionBench, realistic feedback | | ✓ the same variants | logged-feedback (off-policy) |

3 seeds each. Output goes to `output/`. `output/TABLES.md` has VegaML's README table with base / + Peekaboo
columns for Vega 0.8B, Vega 4B and Laya, every method per suite, and the realistic-feedback table.
`output/STATUS.md` lists what is left.

**Hardware:** one GPU with at least 24 GB (the 4B backbone); 48 GB (L40S, A6000) runs 4 steps at once,
about 3–4 hours. 80 GB disk, 32 GB RAM, 8 CPU cores. Use the RunPod PyTorch template.

**Running it with Claude Code:** open Claude Code in the repository and type `/run-peekaboo-remaining`. It
follows `CLAUDE.md`: it starts the run, polls it, fixes failures, restarts, and reports when complete.

The code reuses `experiments/vega_policy/` (Vega, VegaML's vendored benchmark harness, the methods) and adds
`laya_peekaboo.py` (Laya) and `table.py`.
