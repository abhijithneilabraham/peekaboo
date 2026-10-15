# Reproducing the Vega feedback results: instructions for the agent running them

Goal: complete results, for both Vega 0.8B and Vega 4B, in `reproduction/TABLES.md`.

## The loop

1. `python reproduce.py` from this folder. It runs everything. On the first run it installs `requirements.txt`
   into the current Python if `vegaml` 0.8.0 is missing. Several hours on one GPU.
2. If it exits 0, you are done: `reproduction/TABLES.md` holds the tables.
3. If it exits 1, read `reproduction/STATUS.md`. It lists every failed, blocked or missing step, the command,
   and the end of its log (`reproduction/logs/<step>.log` holds the full log).
4. Find and fix the cause, then run `python reproduce.py` again. Finished steps are skipped, so only the failed
   or missing ones run. Repeat until `python reproduce.py --check` exits 0.

A step that fails is already retried 3 times by the script, so a step listed as failed needs a fix, not
just another try. Blocked steps only wait for a failed step they depend on.

## What you may change

Fix the **environment** and anything that keeps code from **running**:
- packages: versions, a missing package, a CUDA build of torch that matches the driver
- downloads: Hugging Face rate limits (set `HF_TOKEN`), network errors, a dataset that needs a retry, disk space
- memory: free the GPU of other processes; on out-of-memory for the 4B backbone, close other GPU programs
  first; `VEGA_DEVICE=cpu` for one step only as a last resort (slow)
- platform problems (Windows paths, encodings, symlinks) and genuine crashes in this folder's code
- if a public task cannot be loaded at all after retries (dataset removed upstream), it is recorded as skipped
  in `reproduction/items_public.json`; report it, do not substitute another task

## What you must not change

The method, its settings and the evaluation: learning rate, epochs, rank, KL weights, seeds, temperature grid,
the splits (which rows are test, pool or calibration), the metrics, the task list, `harness/` (VegaML's own
benchmark code, vendored unmodified), and any number in a result file. Do not edit results by hand and do not
delete result files to make a step look done. If a fix would change a result, stop and report it instead.

## Reporting

When `--check` exits 0, send back `reproduction/TABLES.md`, `reproduction/STATUS.md`, the `reproduction/results*/`
folders (JSON only; skip `*.pt`) and the `reproduction/logs/` folder, plus a short note on every fix you made.
