---
name: reproduce-vega
description: Run the Vega feedback-adapter reproduction (Vega 0.8B and 4B, all VegaML benchmarks) unattended, fixing failures and restarting until the results are complete.
---

Read `experiments/vega_policy/CLAUDE.md` and follow it exactly: start `reproduce.py` detached, poll it with
`python reproduce.py --check`, and whenever it ends with steps not done, read `reproduction/STATUS.md`, fix the
cause, and start it again. Do not stop until `--check` exits 0, then report as CLAUDE.md says.
