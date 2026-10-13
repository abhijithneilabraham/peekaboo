# Reproducing "Vega learns from feedback"

One script per experiment. Each runs from anywhere and only skips a run if its result file already
exists, so you can stop a script and start it again. Paper: `reports/Vega learns from feedback.md`.

```bash
bash repro/01_setup.sh                       # once: .venv + pinned packages (NO_VENV=1 on Kaggle/Colab)
bash repro/02_extract_features.sh            # the only step that runs the language model
bash repro/07_feedback.sh                    # any experiment, in any order after 02
VEGA_SIZE=4b bash repro/run_decisionbench.sh # everything, on Vega 4B
```

## Settings (environment variables)

| variable | default | meaning |
|---|---|---|
| `VEGA_SIZE` | `0.8b` | `0.8b` or `4b`. 4B writes to `bank_decisionbench_4b.pt` and `results_4b/`, so sizes never mix. |
| `SEEDS` | `0 1 2` | seeds for every training run |
| `VEGA_DEVICE` | cuda if available | backbone device for feature extraction (02, 20) |
| `HF_TOKEN` | — | optional; faster Hub downloads |

## Scripts

DecisionBench uses the exact 1,075-row slice from Vega's README as the test set. The adaptation pool is
3,205 other rows from the same 43 tasks.

| script | paper | what it runs |
|---|---|---|
| `01_setup.sh` | — | Python environment (`requirements.txt`) |
| `02_extract_features.sh` | §2.2–2.3 | rebuilds the slice and the pool, and caches the frozen backbone's features |
| `03_check_replay.sh` | §2.2, §6 | checks the engine replay against `Vega.decide()`, plus readout variants with no training |
| `04_released.sh` | §5.1 | the released model |
| `05_ttt.sh` | §4, §5.1 | baseline: vegaml `fit()` TTT head, 'auto' |
| `06_supervised_ts.sh` | §4, §5.1 | baseline: supervised adapter + temperature scaling |
| `07_feedback.sh` | §3, §5.1 | **ours**: feedback RL + KL anchor + feedback calibration |
| `08_feedback_kl1.sh` | §5.1 | ablation: KL weight 1 |
| `09_no_engine_control.sh` | §5.1 | control: same training without Vega's engine |
| `10_tables.sh` | §5.1–5.4 | tables and paired bootstrap CIs → `results*/compare/tables.md` |
| `11_feedback_amount.sh` | §5.5 | 5 / 15 / 30 / 60 feedback rows per task |
| `12_one_pass.sh` | §5.5 | online: each feedback row seen once |
| `13_unseen_tasks.sh` | §5.5 | train on 33 tasks, test on 10 unseen tasks |
| `14_learning_curve.sh` | §5.5 | summary + figure of 11–13 |
| `15_latency.sh` | §5.6 | engine latency and adapter size |
| `16_kl_pilot.sh` | §8 | the pilot runs where the KL weight was chosen |
| `20_kuhn_features.sh` | §7 | encodes the Kuhn poker information sets |
| `21_kuhn_zero_shot.sh` | §7 | no training, and feature contrast |
| `22_kuhn_ppo_lr.sh` | §7 | plain PPO over learning rates (the collapse) |
| `23_kuhn_kl_anchor.sh` | §7 | 150 updates: KL anchor vs entropy and contrast fixes |
| `24_kuhn_long_runs.sh` | §7 | 300 updates × 5 seeds, against lookup-table controls |
| `25_kuhn_fit_probe.sh` | §7 | supervised fit probe (the engine can represent Nash) |
| `30_improvement_table.sh` | §5.2 | **the final table**: README layout, released vs + ours for each size → `results/readme_table.md` |
| `run_decisionbench.sh` / `run_kuhn.sh` | | runs 02–16 / 20–25 in order |

The poker scripts need `experiments/laya_policy/` (game code), which is in this repo.

## Running on Vega 4B

Only feature extraction needs a GPU. Everything after it trains the engine on cached features, on the
CPU. The 4B backbone (`Qwen/Qwen3.5-4B`, ~9 GB) downloads on first use.

```bash
VEGA_SIZE=4b VEGA_DEVICE=cuda bash repro/02_extract_features.sh
VEGA_SIZE=4b bash repro/run_decisionbench.sh         # skips extraction, which is already done
```

Rough cost: extraction is ~4,280 rows of 4B forward passes, about 1–3 h on one T4. The training runs
take minutes each on a CPU. The 4B cache is ~2 GB on disk. Then `bash repro/30_improvement_table.sh` fills the
"Vega 4B" and "4B + ours" columns of the final table.

On a macOS machine, every script waits for free memory first and runs a watchdog (`guard.sh`). The
watchdog kills a run that grows too large. Other systems skip both.
