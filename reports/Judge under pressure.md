# Judge Under Pressure: research plan

Writing standard: ASD-STE100 (Simplified Technical English).
Status: proposal. Date: 2026-10-10.

## 1. Purpose

This document gives the plan for one research project.
The project tests a small calibrated decision model (a "judge") when an RL policy optimizes against it.
The document also tells how the project uses the same method as OpenAI multi-agent hide-and-seek.

## 2. Research question

When a policy optimizes against a small calibrated judge, does the judge stay honest?

"Honest" has two parts:

- The judge score agrees with a trusted gold score.
- The judge probabilities stay calibrated on the text that the policy writes.

## 3. Answer: is this the same line of thought as hide-and-seek?

Yes. The two projects use the same core idea. Some parts are different.

### 3.1 The core idea is the same

In hide-and-seek, agents optimized a reward against an opponent and a simulator ([Baker et al.](https://arxiv.org/abs/1909.07528)).
The optimization pressure found flaws that the designers did not expect.
Examples are box surfing, escapes out of the arena, and physics bugs.

In this project, a policy optimizes a reward from a judge.
The same pressure will try to find flaws in the judge.
The first RL test with the Jev judge found such flaws ([RLJF](https://huggingface.co/blog/jgalego/rljf)).
Replies became much longer. At 1.5B, 63% of replies contained invented facts, and the judge gave them higher scores.

Thus, the judge has the same function as the hide-and-seek physics engine.
It is the "world" that the policy can exploit.

### 3.2 Component comparison

| Component | Hide-and-seek | Judge Under Pressure |
|---|---|---|
| Learner | PPO with GAE (γ = 0.998, λ = 0.95) | GRPO, a PPO-family method without a critic; best-of-n as a cheap form of pressure |
| What the agent optimizes against | Opponent team and physics simulator | Calibrated judge (Laya, a DeBERTa guard, an LLM judge) |
| Ground truth | Game rules: a hider is seen or not seen | Gold score: an LLM ensemble and a human audit |
| Exploits | Box surfing, arena escapes, physics bugs | Length inflation, label-word triggers, invented facts |
| Self-play | Hiders against seekers | Phase 2: policy against a judge that you retrain on found exploits |
| Autocurriculum | Each new strategy forces a counter-strategy | Each judge refit forces a new exploit search |
| Opponent pool (later research) | PSRO and league training stop strategy cycles ([PSRO](https://arxiv.org/abs/1711.00832)) | Keep old policies in a pool, so the refit judge does not forget old exploits |
| Hard cases (later research) | ACCEL edits levels with high regret ([ACCEL](https://arxiv.org/abs/2203.01302)) | Keep the texts where judge and gold disagree most, and train on them |

### 3.3 The differences

- This project has no physics and no 3D world. All states are text.
- Phase 1 has one agent only. The judge does not learn in phase 1.
- Hide-and-seek did not measure calibration. This project measures it at each step.
- Hide-and-seek found exploits by accident. This project measures them on purpose.

## 4. Background facts

- Laya is an open-weight decision model. It has 421M parameters on a ModernBERT-large base ([model card](https://huggingface.co/convaiinnovations/laya)).
- Laya gives probabilities for choice, score, and yes/no questions.
- Raw Laya calibration error (ECE) is high. Sources give 0.213 or 0.466. After temperature fitting, the ECE is 0.081 ([model card](https://huggingface.co/convaiinnovations/laya); [wilsonwu.me](https://wilsonwu.me/en/blog/2026/jev-vs-laya/)).
- These judges follow the name of an option more than its definition ([arXiv 2609.26758](https://arxiv.org/abs/2609.26758)).
- Scaling laws for reward over-optimization exist for LLM reward models ([Gao et al.](https://arxiv.org/abs/2210.10760)). No equivalent result exists for small encoder judges.
- RL attackers already defeat prompt-injection detectors ([PISmith](https://arxiv.org/pdf/2603.13026)).

## 5. Hypotheses

- H1: Under pressure, judge calibration (ECE) decreases faster than judge accuracy.
- H2: The policy first exploits label names that have strong meaning.
- H3: Ensembles, a temperature refit, or a hard checker decrease the speed of the decay.
- H4 (phase 2): A self-play loop with a judge refit improves robustness. It fails if no opponent pool is used.

## 6. Setup

### 6.1 Judges

1. Laya (421M, open weights).
2. A DeBERTa guard model, for example PromptGuard.
3. An LLM judge through an API.

### 6.2 Gold signal

1. An ensemble of three strong LLMs.
2. A human audit of 200 to 300 items at each pressure level.

### 6.3 Domains

| Domain | Judge function | Policy task |
|---|---|---|
| Support replies | Reward: "Is this reply helpful and correct?" | Write a reply to a support ticket |
| Prompt injection | Cost: "Is this text an injection?" | Write a text that contains a hidden instruction |
| Summary faithfulness | Reward: "Is this summary faithful to the source?" | Write a summary of a document |

### 6.4 Compute

- Best-of-n uses a 0.5B generator with MLX on the 8 GB M1 laptop.
- GRPO with LoRA on Qwen2.5-0.5B or 1.5B uses the free Kaggle T4 GPUs.
- The judges run on the M1 laptop.

CAUTION: Do not do a full fine-tune of Laya on the 8 GB laptop. Memory is not sufficient. Train the head only, or use LoRA.

## 7. Procedure

### 7.1 Phase 1: fixed judge (weeks 1 to 5)

1. Get Laya, the DeBERTa guard, and API access for the LLM judge.
2. Fit a temperature for each judge on 500 held-out labels per domain.
3. Record the start ECE and AUC for each judge.
4. Make a gold label set for each domain.
5. Run best-of-n with n = 1, 4, 16, 64, and 256.
6. At each n, record the judge score and the gold score.
7. Run GRPO against each judge for a fixed number of steps.
8. Save policy samples at regular steps.
9. At each saved step, record the judge score, gold score, ECE, and AUC.
10. Record the KL divergence between the policy and the start policy.
11. Classify each exploit that you find: length, keyword, label word, or invented fact.

### 7.2 Phase 2: self-play with the judge (weeks 5 to 6)

This phase uses the hide-and-seek method.

1. Collect the texts where the judge and the gold score disagree most.
2. Retrain the judge head on these texts.
3. Train the policy against the new judge.
4. Do steps 1 to 3 again for four or more rounds.
5. Do the loop again with an opponent pool. Keep samples from all old policies in the judge training set.
6. Compare the results with and without the opponent pool.

### 7.3 Ablations (week 6)

1. Change the judge size.
2. Turn calibration on and off.
3. Use an ensemble of judge heads.
4. Use neutral label names, for example "A" and "B", in place of names with meaning.
5. Add a hard checker as an anchor, for example a length cap or a fact check against a source.

### 7.4 Human audit and report (weeks 7 to 8)

1. Do the human audit at each pressure level.
2. Make the plots.
3. Write the paper.
4. Publish the code and data.

## 8. Metrics

| Metric | What it shows |
|---|---|
| Judge score against gold score, as a function of pressure (n or KL) | Over-optimization curve, as in Gao et al. |
| ECE on policy text at each step | Calibration decay |
| AUC on policy text at each step | Accuracy decay |
| Exploit type counts | How the policy wins |
| Robustness after each self-play round | Effect of the judge refit and the opponent pool |

## 9. Risks

WARNING: The prompt-injection domain makes attack text. Keep this text in a closed test set. Do not publish working attacks against real products.

CAUTION: The gold LLM ensemble can share flaws with the judges. Use the human audit to find this problem.

CAUTION: Jev is a closed API with a waitlist. Do not make the main result depend on Jev.

- Laya facts come mostly from the vendor and blogs. Measure the Laya numbers again on your own data before you use them.
- If the judges stay honest under pressure, the result is still useful. Report it as a positive robustness result.

## 10. Completion criteria

The project is complete when all these items are done:

1. Over-optimization curves for three judges in two or more domains.
2. One mitigation that measurably decreases the decay.
3. The phase 2 self-play result, with and without the opponent pool.
4. A public code release.
5. An arXiv preprint.
6. A submission to a NeurIPS or ICLR workshop on RLHF or reliable ML.

## 11. References

- Baker et al., Emergent Tool Use From Multi-Agent Autocurricula: https://arxiv.org/abs/1909.07528
- Gao et al., Scaling Laws for Reward Model Overoptimization: https://arxiv.org/abs/2210.10760
- RLJF, Reinforcement Learning from Jev Feedback: https://huggingface.co/blog/jgalego/rljf
- Laya model card: https://huggingface.co/convaiinnovations/laya
- Jev vs Laya comparison: https://wilsonwu.me/en/blog/2026/jev-vs-laya/
- Name-over-definition study: https://arxiv.org/abs/2609.26758
- PSRO: https://arxiv.org/abs/1711.00832
- ACCEL: https://arxiv.org/abs/2203.01302
- PISmith: https://arxiv.org/pdf/2603.13026
