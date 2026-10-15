# Reproducing the Vega feedback results: instructions for the agent running them

You are running the experiments of `reports/Vega learns from feedback.md` on this machine, unattended. The goal is
complete results for Vega 0.8B **and** Vega 4B in `experiments/vega_policy/reproduction/TABLES.md`. Keep going
until that is done: when something fails, find the cause, fix it, and start the run again yourself. Do not stop
to ask the user unless a fix would need something only they can give (a token, a paid service, more hardware).

## The loop

The full run takes hours (most of it reading the 4B backbone over ~9,000 rows), longer than one tool call, so it
runs detached and you poll it. All commands are from `experiments/vega_policy/`.

1. **Start it detached** (skip if `python reproduce.py --check` exits 2, which means a run is already alive):
   - bash / Git Bash: `mkdir -p reproduction && nohup python reproduce.py >> reproduction/run.log 2>&1 &`
   - PowerShell: `New-Item -ItemType Directory -Force reproduction; Start-Process python -ArgumentList reproduce.py -NoNewWindow -RedirectStandardOutput reproduction/run.log -RedirectStandardError reproduction/run.err`
2. **Poll** with `python reproduce.py --check` every 10 to 20 minutes (a wait loop such as
   `until ! python reproduce.py --check >/dev/null; [ $? -ne 2 ]; do sleep 60; done` capped at your tool's time
   limit, then poll again). Exit codes:
   - `2`: still running. `reproduction/run.log` shows the current step. Wait and poll again.
   - `0`: complete. Go to Reporting.
   - `1`: the run ended with steps not done. Read `reproduction/STATUS.md`: each failed step, its command and
     the end of its log (`reproduction/logs/<step>.log` has all of it). Fix the cause, then go to step 1.
3. To test one fix fast, run the failing step's own command from STATUS.md directly (with `VEGA_OUT=reproduction`
   and `VEGA_SIZE=0.8b` or `4b` set as in the log), then restart the full run. Finished steps are always skipped.

The script retries every failed step 3 times by itself, so a step listed as failed needs a fix, not another try.
A blocked step only waits for a failed step it depends on. If the same step fails the same way after two of your
fixes, write down what you tried in `reproduction/NOTES.md` and move on to the other failures before returning.

## What you may change

Anything about the **environment**, and anything that keeps code from **running**:
- packages: versions, a missing package, the torch build that matches the GPU driver
- downloads: Hugging Face rate limits (`HF_TOKEN`), network errors, disk space, cache locations (`HF_HOME`)
- memory: close other GPU programs; the 4B backbone needs ~10 GB of GPU memory. `VEGA_DEVICE=cpu` only as a
  last resort for one small step (the 4B on a CPU takes days)
- platform problems (Windows paths, encodings, symlinks, line endings) and genuine crashes in this folder's code
- a public task whose dataset cannot be loaded at all is recorded as skipped in `reproduction/items_public.json`;
  note it in NOTES.md, do not substitute another task

## What you must not change

The method, its settings and the evaluation: learning rates, epochs, rank, KL weights, seeds, the temperature grid,
the splits (which rows are test, pool or calibration), the metrics, the task list, `harness/` (VegaML's own benchmark
code, vendored unmodified), and any number in a result file. Do not edit or delete result files to make a step
look done. If the only fix would change a result, stop and write it up in NOTES.md instead.

## Reporting

When `--check` exits 0: zip `reproduction/` without the `*.pt` files and the `items_*.json` files (dataset text).
Tell the user where the zip is, paste the first table of `reproduction/TABLES.md`, and list every fix you made
(also in NOTES.md).
