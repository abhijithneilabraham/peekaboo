# The remaining Peekaboo experiments: instructions for the agent running them

You are running the remaining Peekaboo experiments (see README.md in this folder) on this machine, unattended. The
goal is complete results for Vega 0.8B, Vega 4B **and** Laya in `experiments/peekaboo_remaining/output/TABLES.md`.
Keep going until that is done: when something fails, find the cause, fix it, and start the run again yourself. Do
not stop to ask the user unless a fix needs something only they can give (a token, a paid service, more hardware).

## The loop

The run takes hours, longer than one tool call, so it runs detached and you poll it. All commands are from
`experiments/peekaboo_remaining/`.

1. **Start it detached** (skip if `python run.py --check` exits 2, which means a run is already alive):
   - bash: `mkdir -p output && nohup python run.py >> output/run.log 2>&1 &`
   - PowerShell: `New-Item -ItemType Directory -Force output; Start-Process python -ArgumentList run.py -NoNewWindow -RedirectStandardOutput output/run.log -RedirectStandardError output/run.err`
2. **Poll** with `python run.py --check` every 10 to 20 minutes (a wait loop such as
   `until python run.py --check >/dev/null; [ $? -ne 2 ]; do sleep 60; done`, capped at your tool's time limit,
   then poll again). Exit codes:
   - `2`: still running. `output/run.log` shows which steps started and finished. Wait and poll again.
   - `0`: complete. Go to Reporting.
   - `1`: the run ended with steps not done. Read `output/STATUS.md`: each failed step, its command, and the end of
     its log (`output/logs/<step>.log` has all of it). Fix the cause, then go to step 1.
3. To test one fix fast, run the failing step's command from STATUS.md yourself, from the folder it names, with
   `VEGA_OUT` set to the absolute path of `output/` and `VEGA_SIZE` as shown. Then restart the full run; finished
   steps are always skipped.

The script retries every failed step 3 times by itself, so a step listed as failed needs a fix, not another try.
A blocked step only waits for a failed step it depends on. If the same step fails the same way after two of your
fixes, write what you tried in `output/NOTES.md` and move on to the other failures before returning to it.

## What you may change

Anything about the **environment**, and anything that keeps code from **running**:
- packages: versions, a missing package, the torch build that matches the GPU driver
- downloads: Hugging Face rate limits (`HF_TOKEN`), network errors, disk space, cache locations (`HF_HOME`)
- memory: close other GPU programs; lower parallelism with `python run.py --jobs 2` (or 1) after an out-of-memory
  error. `VEGA_DEVICE=cpu` only as a last resort for one small step (the 4B on a CPU takes days)
- genuine crashes in this repository's code (`experiments/peekaboo_remaining/`, `experiments/vega_policy/`),
  without changing what the code computes
- a public task whose dataset cannot be loaded at all is recorded as skipped in `output/items_public.json`; note it
  in NOTES.md, do not substitute another task

## What you must not change

The methods, their settings and the evaluation: learning rates, epochs, ranks, KL weights, seeds, the temperature grid, the splits (which rows are test, pool or calibration), the metrics, the task list,
`experiments/vega_policy/harness/` (VegaML's own benchmark code, vendored unmodified), and any number in a result
file. Do not edit or delete result files to make a step look done. If the only fix would change a result, stop and
write it up in NOTES.md instead.

## Reporting

When `--check` exits 0: zip `output/` without the `*.pt` files and the `items_*.json` files (dataset text). Tell
the user where the zip is, paste sections 1, 2 and 4 of `output/TABLES.md`, and list every fix you made (also in NOTES.md).
