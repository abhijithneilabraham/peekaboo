---
name: run-peekaboo-remaining
description: Run the remaining Peekaboo experiments (Vega 0.8B and 4B on the public tasks and JevBench, realistic feedback, and Laya) unattended, fixing failures and restarting until the results are complete.
---

Read `experiments/peekaboo_remaining/CLAUDE.md` and follow it exactly: start `python run.py` detached from that
folder, poll it with `python run.py --check`, and whenever it ends with steps not done, read `output/STATUS.md`,
fix the cause, and start it again. Do not stop until `--check` exits 0, then report as CLAUDE.md says.
