# Learning from right and wrong
### Calibrated, label-free adaptation of the Vega decision model from deployment feedback

*abhijithneilabraham · October 2026 · code and raw results: `experiments/vega_policy/`*

---

## Abstract

Small local decision models such as Vega (a frozen 0.8B Qwen backbone read by a 14.3M-parameter
"physics" engine) answer typed questions about a situation with calibrated probabilities, privately and
in milliseconds. Once deployed, such a model rarely receives gold labels. What it does receive is
feedback on the answer it gave: an accepted suggestion, a reverted action, a thumbs-down. We treat this
as a contextual bandit and ask whether Vega can improve from that signal alone.

The method has two parts:
1. A fresh gated low-rank adapter on the released engine is trained by REINFORCE on the model's own
   sampled answers, with a KL anchor to the released model.
2. A single temperature is fitted from the same right/wrong signal, using the fact that a calibrated
   model's sampled answer is correct with exactly the probability it assigned.

We evaluate on exactly the 1,075 DecisionBench rows that Vega's README reports. The adaptation pool is
3,205 other rows of the same 43 tasks, and labels are never read except through a 0/1 reward. Results
over 3 seeds:
- Accuracy rises from **0.509 to 0.611** (+10.2 points, 95% CI [+7.9, +12.6]).
- NLL falls from 1.628 to **1.291**, and expected calibration error from 0.098 to **0.054**, the best of
  any method we ran.
- Vega's own adaptation (`fit()`, a TTT head trained on all 3,205 *gold labels*) reaches 0.531. Our
  label-free method beats it by **+8.0 points [+5.6, +10.5]**.
- A supervised adapter with standard temperature scaling, which reads every label, reaches 0.627. Our
  method is statistically indistinguishable from it (−1.6 points [−3.5, +0.3]) and better calibrated.

Three controls locate the gain:
- Without the feedback calibration step, the same policy has ECE 0.24–0.29.
- Without Vega's engine (the same feedback on a bilinear scorer over the same features), accuracy is 0.26.
- With a ten-times-stronger KL anchor, the gain shrinks to +3.8 points.

On a larger 1,720-row test set the gain grows smoothly with feedback, from +3.1 points at 5 rows per
task to +11.3 at 60. A single online pass, in which each row is seen once, gives +6.9. Feedback from 33
tasks transfers to 10 unseen tasks with +4.2 points. The adapter adds 409k parameters and about 0.2 ms
per decision. We also report a smaller self-play result in Kuhn poker and several null results.

---

## 1. Motivation

A decision model deployed inside an agent, a moderation queue or a support flow makes thousands of
small decisions per day. Examples:
- Is this tool call risky?
- Does this request need a human?
- Which category does this belong to?

Labelling those decisions is the expensive step that teams skip. Feedback on them is almost free:
- the user undoes the action;
- the reviewer overrides the call;
- the customer replies "that's not what I asked".

A model that improves from that stream, *without becoming overconfident*, is directly useful to
anyone running one. That is especially true of a model that runs locally with nothing leaving the
machine.

Calibration is the hard part. Vega's README makes calibration its headline advantage: on DecisionBench
it trails the hosted reference by 24 accuracy points but has a better likelihood by a factor of two.
Fine-tuning usually destroys calibration (we see the same: supervised adapters reach ECE 0.15–0.33
before rescaling), and the standard repair, temperature scaling, needs labelled data. We needed a
method that improves accuracy *and* keeps calibration using only the signal a deployment produces.

## 2. Setting

**Vega.** `vegaml` 0.8.0 with the public checkpoint `nandakishorm/vega-08b-public-intents`:
- a frozen Qwen3.5-0.8B backbone pooled at layers 13 and 19;
- a 14.3M-parameter engine in which each option is a potential well and the answer is where a particle
  settles;
- a shipped thermal calibration, which sets a per-question temperature from the trajectory;
- two shipped gated adapters, routed automatically.

Everything shipped is kept exactly as released.

**Benchmark and the exact test set.** DecisionBench (`Hanno-Labs/decision-bench`) has 43 tasks over
three primitives in 28 domains, such as agent control, privacy, legal, robotics and tax documents.
Vega's README reports a 1,075-row slice built by its benchmark notebook (25 rows per task,
seed 20261011). We reimplemented that sampler verbatim and reproduce the slice exactly. Our offline
replay of the engine on cached backbone features matches the official `Vega.decide()` probabilities to
0.0 on every checked row, and reproduces the README's numbers (accuracy 0.509, NLL 1.628, KL to gold
1.012). **The 1,075 README rows are our test set.**

**Adaptation pool.** Running the same sampler at 100 rows per task gives 4,280 rows, a strict superset
of the slice. The 3,205 rows outside the slice form the pool. 20% of each task's pool (641 rows) is
held out for calibration, leaving 2,564 rows to learn from. Every method sees the same rows.

**Feedback.** For a pool row the learner samples an option from its own distribution and receives
r = 1 if it is the gold option, else 0. Nothing else about the label is revealed. This is simulated
feedback derived from gold labels. Real deployment feedback is noisier, a limitation we return to in
§8.

## 3. Method

**Policy.** π(a|s) is Vega's own calibrated readout, softmax(−E_a / τ(s)). The energies E come from
the engine with a new adapter, and τ(s) is the shipped thermal temperature, held fixed.

**What is trained.** We attach one gated LoRA adapter (rank 8) to the engine's input layers, through
vegaml's own `attach_adapter`. That is 409k parameters, 2.9% of the engine. Its gate starts open, and
the shipped adapters keep routing and calibrating exactly as released.

**Update.** The loss is REINFORCE with a running-mean baseline plus a KL anchor to the released model:

  L = −(r − b) · log π(a|s) + β · KL(π(·|s) ‖ π₀(·|s)),   β = 0.1

We use 8 passes over the pool, batch 32, and Adam at lr 1e-3. The anchor keeps the policy from
collapsing onto a few options. Without it, NLL reaches 7.5. In Kuhn poker (§7), the same
collapse-and-anchor pattern was the first thing we found.

**Feedback calibration.** RL sharpens the distribution: after training, the raw policy has ECE 0.24–0.29.
The repair uses one observation. If a model is calibrated, the option it samples is correct with
probability exactly p(a). Each logged pair (a, r) is therefore a Bernoulli observation of p(a). On fresh
interactions with the 641 held-out rows, we sample a, observe r, and choose the temperature T that
maximises

  Σ r · log p_T(a) + (1 − r) · log(1 − p_T(a)).

That is one scalar, fitted by grid search, with no labels read. It is the bandit-feedback counterpart
of temperature scaling. The fitted T is 5.4–6.9 for β = 0.1 and about 1.9 for β = 1.

## 4. Baselines

| method | reads from the pool | description |
|---|---|---|
| Released Vega 0.8B | nothing | the model as published |
| vegaml `fit()` TTT, `auto` | **gold labels** of all 3,205 rows | Vega's own adaptation: a PCA + logistic head on the same frozen features, one per task. It is kept when 5-fold CV accuracy beats chance, as `fit()` does; otherwise the engine answers. To be generous to it, a test row whose options the head never saw goes to the engine instead of getting probability 0 for that option |
| Supervised adapter + temperature scaling | **gold labels** | the same adapter trained with cross-entropy and the same KL anchor, then standard temperature scaling on the 641 held-out labelled rows. This is the strongest labelled recipe we ran |
| No-engine control | right/wrong only | a bilinear state × option scorer over the same frozen features, with the same feedback, entropy bonus and feedback calibration |
| Ours, β = 1 | right/wrong only | ablation: a 10× stronger anchor |
| **Ours** | right/wrong only | §3 |

## 5. Results

### 5.1 Head to head on Vega's exact test set

1,075 rows. Training methods use 3 seeds (mean ± s.d.); the released model and TTT are deterministic.
"Skill" is chance-corrected accuracy. KL to gold is on the 10 rows with soft gold labels, so it is
noisy.

| method | learns from | accuracy | skill | NLL ↓ | ECE ↓ | KL to gold ↓ |
|---|---|---|---|---|---|---|
| Released Vega 0.8B | — | 0.509 | 0.362 | 1.628 | 0.098 | 1.012 |
| vegaml `fit()` TTT, `auto` | gold labels | 0.531 | 0.391 | 1.628 | 0.111 | 1.263 |
| Supervised adapter + temp. scaling | gold labels | **0.627 ± 0.010** | **0.515** | **1.183** | 0.074 ± 0.017 | 0.786 |
| No-engine control | right/wrong | 0.262 ± 0.023 | 0.041 | 1.949 | 0.115 | 0.794 |
| Ours, β = 1 | right/wrong | 0.547 ± 0.002 | 0.412 | 1.430 | 0.076 | 0.758 |
| **Ours** | right/wrong | **0.611 ± 0.015** | 0.495 | 1.291 ± 0.024 | **0.054 ± 0.008** | 1.101 ± 0.332 |

**Paired accuracy differences.** These are per-row on the same 1,075 rows, averaged over seeds, with a
95% bootstrap CI:

| comparison | points |
|---|---|
| Ours vs released Vega | **+10.2 [+7.9, +12.6]** |
| Ours vs vegaml TTT (gold labels) | **+8.0 [+5.6, +10.5]** |
| Ours vs supervised + temp. scaling (gold labels) | −1.6 [−3.5, +0.3] |
| TTT vs released Vega | +2.2 [−0.1, +4.4] |
| Supervised + temp. scaling vs released Vega | +11.8 [+8.9, +14.6] |

### 5.2 Next to the README's own DecisionBench table

The first three rows are copied from Vega's README (same 1,075 rows). The README's ECE uses its own
estimator; ours gives 0.098 for the released 0.8B, where the README prints 0.103.

| system | accuracy | NLL ↓ | ECE ↓ | needs |
|---|---|---|---|---|
| Vega 0.8B (README) | 0.509 | 1.628 | 0.103 | — |
| Vega 4B (README) | 0.633 | 1.420 | 0.098 | 9.3 GB backbone, about 1.4× the latency |
| Jev, hosted (README) | 0.752 | 3.196 | 0.122 | API, data leaves the machine |
| **Vega 0.8B + ours** | **0.611** | **1.291** | **0.054** | right/wrong feedback, 409k extra parameters, about +0.2 ms |

With feedback alone, the 0.8B model closes 82% of the accuracy gap to the 4B. It has the best
likelihood and calibration in the table, while keeping the 0.8B's footprint. The hosted model remains
far more accurate.

### 5.3 Why it wins over vegaml's own adaptation

**TTT can only adapt where labels are stable.** A TTT head is a classifier over a fixed label set. Many
DecisionBench tasks give every row its own options (an ordinal scale worded differently each time, a
list of candidate records). `fit()` kept heads for 20 of 43 tasks, and those heads cover only 394 of the
1,075 test rows. The other 681 rows get nothing from it. Our adapter changes the engine, which reads
the option text, so it applies to every row:

| test rows | n | released | TTT | ours | supervised + TS |
|---|---|---|---|---|---|
| a TTT head answers | 394 | 0.558 | 0.619 | **0.646** | 0.661 |
| TTT falls back to the engine | 681 | 0.480 | 0.480 | **0.591** | 0.607 |

Ours wins on both parts of the test set, and by 11 points where TTT cannot act.

**TTT also hurts some domains.** Its CV-gated heads lower logic from 0.44 to 0.24, web from 0.76 to
0.60, and multi-hop search from 0.56 to 0.44. Ours moves the same three domains by −8, −4 and +3 points.

**Calibration survives because of the feedback temperature.** Without that step, our policy is as
miscalibrated as any fine-tune (ECE 0.24–0.29, NLL 3.1–4.1). With it, ECE is 0.054. The supervised
recipe needs held-out *labels* to do the same, and ends at 0.074.

**The engine matters.** The same feedback, calibration and features, without Vega's engine, give 0.26.
That is worse than the released model, because a policy learning from scratch with 2,564 bandit
rewards over 43 tasks has nothing to start from. Vega's zero-shot energies are what make the bandit
problem easy enough to learn from 0/1 rewards. This matches our Kuhn-poker control (§7), where the
regulariser, not the prior, explained the gain. Here the prior is essential.

### 5.4 Largest domain gains

From 3-seed means:
- home automation 0.44 → 0.76
- online safety 0.48 → 0.86
- knowledge graphs 0.32 → 0.51
- software engineering 0.44 → 0.68
- web content 0.76 → 1.00
- robotics 0.28 → 0.45
- public sector 0.76 → 0.93

Losses:
- logic 0.44 → 0.36
- evidence reasoning 0.52 → 0.52 (no change; supervised reaches 0.61)

Tax documents stays near zero for every method except supervised (0.19).

### 5.5 How much feedback, online use, unseen tasks

The second experiment uses a larger, fixed test set: 40 rows per task, 1,720 rows, drawn from the
4,280-row pool and disjoint from the training rows. It has 3 seeds and paired CIs. In this sweep the
feedback temperature was fitted on the training rows rather than a held-out slice, which is slightly
less clean than §5.1.

![learning curve](figures/vega_feedback_curve.png)

| feedback rows per task | 5 | 15 | 30 | 60 |
|---|---|---|---|---|
| ours: accuracy gain, points | +3.1 [+1.4, +4.7] | +5.0 [+3.2, +6.7] | +8.4 [+6.6, +10.3] | **+11.3 [+9.4, +13.1]** |
| ours: ECE (released 0.119) | 0.105 | 0.118 | 0.081 | **0.066** |
| supervised, no calibration: gain | +1.4 | +4.2 | +9.2 | +11.7 |
| supervised, no calibration: ECE | 0.249 | 0.275 | 0.202 | 0.155 |
| ours, β = 1: gain | +0.9 | +2.1 | +2.9 | +4.4 |

**Learning curve.** Feedback-only learning stays within about a point of label-supervised learning
at 15–60 rows per task, and is ahead at 5. With the feedback temperature, calibration is better than the
released model from 30 rows per task on.

**Online.** One pass, each of the 2,560 rows seen once, as in a live deployment: +6.9 points
[+5.3, +8.5], ECE 0.054. In the 8-pass runs of §5.1, the average reward on the model's own sampled
answers rose from 0.48 in the first pass to 0.67 in the last. Sampled answers are right less often than
greedy ones, so early exploration has a real cost to the people being answered (see §8).

**Unseen tasks.** We learn from all rows of 33 tasks and test on 10 tasks never seen (1,000 rows):
- accuracy 0.378 → 0.420 (+4.2 [+1.9, +6.6]);
- NLL 2.100 → 1.850;
- ECE 0.105 → 0.065.

On the small 25-row-per-task pool this transfer was absent (+0.4). It needs volume.

### 5.6 Cost

| | released engine | engine + feedback adapter |
|---|---|---|
| engine time per decision, CPU, batch 1, p50 / p95 | 6.3 / 7.4 ms | 6.5 / 8.6 ms |
| parameters | 14.34M | +0.41M |

The backbone read (about 74 ms on a T4, per the README) is unchanged and dominates. Training on
cached features takes 4–15 minutes on a laptop CPU, in under 1.2 GB of memory.

## 6. Things that did not work

| idea | result on the 1,075-row slice |
|---|---|
| Contrast amplification: move each state's features away from its task's mean by ×1.5, 2 or 3, with no training | 0.504 / 0.504 / 0.511 accuracy vs 0.509: null |
| Shipped adapters off | accuracy 0.501, NLL 1.567, ECE 0.069: better calibrated, slightly less accurate |
| Let the physics run 24 steps without halting | accuracy 0.475, ECE 0.057, KL to gold 0.70: calibration up, accuracy down |
| Feedback without the KL anchor (pilot, 15 rows/task of the slice) | +1.9 points, NLL 7.5: collapses onto a few options |
| A stronger anchor, β = 1 | stable and calibrated, but only a third of the gain |

Contrast amplification did help in poker (§7), where states differ by one word. DecisionBench states
differ substantially already.

## 7. Where this started: Vega as a self-play policy in Kuhn poker

Our first experiment used Vega as the policy in text Kuhn poker, the setting of our earlier Laya paper,
with the same PPO loop and opponent pool, at 64 hands per update.

Exploitability is measured by the same script for every row: the mean over the last 5 evaluations of the
current policy. That is not the time-averaged strategy figure quoted in the Laya paper. Lower is better.

| | updates | seeds | exploitability, 3-card | unseen 4-card deck |
|---|---|---|---|---|
| Vega zero-shot | 0 | — | 0.429 | 0.431 |
| Vega + contrast ×5, no training | 0 | — | 0.340 | 0.404 |
| Vega + PPO, any lr from 1e-5 to 3e-4 | 150 | 2 | 0.333 (collapses to always bet/call) | 0.25 |
| **Vega adapter + PPO + KL anchor** | 150 | 3 | **0.246 ± 0.016** | **0.308** |
| Laya head + PPO (earlier paper's runs) | 150 | 3 | 0.275 ± 0.058 | 0.631 |
| Vega adapter + PPO + KL anchor | 300 | 5 | 0.225 ± 0.007 | 0.318 |
| Lookup table + PPO + the same KL to uniform | 300 | 5 | 0.173 ± 0.005 | n/a |
| Lookup table + PPO, no KL | 300 | 5 | 0.292 ± 0.087 | n/a |

Plain RL on Vega collapsed within 10 updates. The diagnosis was that features for different cards are
99.6% cosine-similar, so every update moves all cards together. Yet the engine *can* represent the Nash
strategy: supervised, it fits it to 0.001. The KL anchor fixed the collapse and beat trained Laya at the
same budget. It also ran in about 30 s and 480 MB per run, against hours for Laya.

The control matters, though. At 300 updates, a lookup table with the same KL regulariser did better still (0.173 vs 0.225), so in poker
the gain came from the regulariser, not from Vega's prior. DecisionBench is the opposite case. There
the no-engine control fails and the prior is what makes 0/1 feedback learnable. That contrast is why
we moved to it.

## 8. Limitations

- **Simulated feedback.** Rewards are derived from gold labels, so they are noise-free and unbiased. Real
  feedback is sparse, delayed, and biased: users rarely confirm correct answers and often miss wrong ones.
  The method needs only a 0/1 signal, but its robustness to label noise and selective feedback is untested.
- **One benchmark, one backbone size.** Vega 4B (9.3 GB) does not fit our 8 GB machine. We ran nothing
  on the hosted reference; its numbers are copied from the README.
- **In-distribution tasks.** §5.1 tests on rows of tasks the model received feedback on. Cross-task
  transfer exists (+4.2) but is smaller and needs volume.
- **Exploration is real.** The learner samples answers, so during learning some decisions are deliberately
  non-greedy. In the online run the first pass matched the released model's accuracy. A deployment would
  want conservative exploration (e.g. only on low-stakes questions, or only within Vega's conformal
  prediction set). That is the natural next step and is not done here.
- **Baseline tuning.** TTT uses vegaml's defaults (k = 128, l2 = 1e-2). The supervised adapter shares our
  hyperparameters and was not tuned separately. Ours was not tuned on the comparison either, but there
  is a disclosure to make. β = 0.1 was picked from {0, 0.1, 1} in pilot runs that trained and tested on a
  split *of the same 1,075-row slice*, so the slice informed that one choice. The other settings
  (8 passes, lr 1e-3, rank 8) were set before any DecisionBench result. β = 1 is reported alongside as
  the alternative we did not pick.
- **KL to gold** is measured on 10 rows and has a large variance (ours 1.10 ± 0.33).

## 9. Reproduce

The code is in `experiments/vega_policy/`, and everything runs on CPU with a memory watchdog.

| script | what it does |
|---|---|
| `bench_extract.py` | rebuilds the README's 1,075-row slice and caches backbone features; `DB_PER_TASK=100` builds the 4,280-row pool. About 3 s per row on a laptop CPU |
| `bench_eval.py` | offline replay (verified against `Vega.decide()`), zero-parameter variants |
| `compare.py --method {zero, ttt_auto, sup_ts, feedback, linear_fb} --seed N` | the head-to-head of §5.1 |
| `compare_table.py` | the head-to-head tables |
| `bench_rl.py` | the scaling, online and unseen-task runs |
| `dbgroup.py` | summarises the scaling, online and unseen-task runs |
| `figure.py` | draws the learning curve |
| `latency.py` | §5.6 |
| `vtrain.py` | the Kuhn poker experiments |
| `group.py` | summarises the Kuhn poker runs |

Raw results are in `results/`, and the head-to-head is in `results/compare/`.
