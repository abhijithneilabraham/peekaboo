# Decision Models (cross-encoders, small judges, reward models, NLI classifiers, Jev/Laya) + Reinforcement Learning: Integration Patterns

Scope note: Notes cover 9 integration patterns plus a background section on Jev/Laya. Facts about Jev/Laya come from 2026 web write-ups, which are mostly vendor docs, blogs and aggregators. Only one peer-style study exists (arXiv 2609.26758), and one community experiment (RLJF on the Hugging Face blog). Classic RL/RLHF citations are primary arXiv papers. Where a number is not from a fetched page, it is phrased conservatively, and I flag uncertainty. "M1-8GB" means an 8 GB Apple M1 laptop running a ~421M-parameter encoder such as Laya (ModernBERT-large).

## Background: what Jev and Laya are (needed for every pattern below)

### Takeaway
Jev (closed API) and Laya (Apache-2.0, 421M ModernBERT-large) are non-generative "System One" decision models. They return typed answers with probabilities, and both are trained with "RLCD", an RL objective built on strictly proper scoring rules. This makes them natural drop-in reward, gate or feature functions for RL. However, their calibration out of the box is weak (Laya ECE 0.466 before temperature fitting). They also have a documented semantic failure: they follow option names rather than the definitions given for those names.

### Cited Findings
- Jev was released 15 Sep 2026 by TypeSafe AI. It answers typed questions about text/JSON and returns only structured values (choice, score, yes/no), each with a calibrated probability. It is marketed as the first "System One model". — [Sanity glossary](https://www.sanity.io/glossary/jev-typesafe-ai-model); [OpenRouter blog](https://openrouter.ai/blog/insights/what-is-jev/)
- Jev's three primitives: choice (up to 255 options, full distribution + confidence), score (ordered rubric), and "noul" yes/no (a single probability). Latency is 70–500 ms. Price is $0.042 per 1M input tokens; output is free. — [Layer3Labs](https://www.layer3labs.io/guides/jev-explained)
- RLCD = "Reinforcement Learning for Calibrated Decisions". Layer3Labs calls it "a training objective rather than a guarantee of factual accuracy". Jev's TypeSafe-internal benchmark showed 67.8% vs 74.1% for an unnamed comparator. "No independent third-party benchmark exists as of September 22, 2026." — [Layer3Labs](https://www.layer3labs.io/guides/jev-explained)
- RLCD trains with RL against strictly proper scoring rules: log and spherical scores, plus the ranked probability score for ordinal questions. Expected reward is maximised only by reporting honest probabilities. — [Michał Chromiak blog](https://mchromiak.github.io/articles/2026/Sep/17/Typed-Decision-Models-Jev-and-Laya-in-Agentic-AI/) (via search snippet)
- Jev's documented limits: no arithmetic, counting, date logic or multi-step deduction. It is vulnerable to prompt injection, accepts text only, and has a ~32k-token request budget. — [Layer3Labs](https://www.layer3labs.io/guides/jev-explained); [wilsonwu.me](https://wilsonwu.me/en/blog/2026/jev-vs-laya/)
- Laya (Convai Innovations, Nandakishor M) was open-sourced 18 Sep 2026. Checkpoints:
  - English: ModernBERT-large, 421M, 512 context.
  - Multilingual: mmBERT-base, 322M, 1024 context, 100+ languages.
  - Typed-decisions: fine-tuned on the benchmark's training split.
  - License is Apache 2.0. — [aiweekly](https://aiweekly.co/alerts/convai-ships-laya-a-non-autoregressive-multilingual-system-1-decision-model-at); [wilsonwu.me](https://wilsonwu.me/en/blog/2026/jev-vs-laya/); [HF model card](https://huggingface.co/convaiinnovations/laya)
- Laya has one output head per question type. It is trained with log-loss/Brier-style proper scoring, with calibration temperatures fitted after training. Memory footprint is ~1 GB. It runs on CPU. Fine-tuning takes ~4 h on 2×T4 (free Kaggle tier). — [themenonlab](https://themenonlab.blog/blog/laya-local-system-1-decision-model)
- Head-to-head results (third-party, as reported by wilsonwu.me):

  | Metric | Jev | Laya |
  |---|---|---|
  | Typed-decisions accuracy | 0.727 (zero-shot) | 0.766 (fine-tuned); base 0.34–0.36 |
  | ECE | 0.246 | 0.466 out of box; 0.081 after temperature fitting on held-out data |
  | p50 latency | 236–276 ms | ~33–40 ms on T4 |

  — [wilsonwu.me](https://wilsonwu.me/en/blog/2026/jev-vs-laya/)
- Runware reports these points:
  - Laya's general English checkpoint scored 36.2%, below a 46.1% majority-class baseline.
  - Laya docs warn that "shipped probabilities can be overconfident" and recommend calibrating before the model controls consequential actions.
  - Choice options share a fixed input budget; take care above ~20 options.
  - In a phishing test (by Felix Sanz), Jev scored 62.6% on a direct yes/no question vs 81.3% for Claude Haiku 4.5. It reached 95% when the task was decomposed into five specific signals.

  — [Runware](https://runware.ai/blog/jev-laya-and-the-emerging-role-of-decision-models)
- Name-over-definition failure: decision models (Jev + two open-weight models) follow the option *name*, not the rubric bound to it.
  - Swapping name/definition pairs raised decision-flip rates by up to 70.4 pp (yes/no names vs a 0/1 control).
  - Mean AUC fell from 93.8% to 23.2%.
  - The type-error rate stayed at 0% throughout.

  — [Sun et al., arXiv 2609.26758](https://arxiv.org/abs/2609.26758)

### Inferences
- For RL use, treat Laya's raw probabilities as uncalibrated until you refit temperature on in-domain held-out labels. The 0.466 → 0.081 ECE swing shows how much this matters.
- Use neutral option labels (A/B, 0/1) plus definitions, or names that already match their meaning. Otherwise a policy can be rewarded or gated by the label name rather than the rubric (per 2609.26758).
- Decomposing one hard judgment into several narrow yes/no signals substantially improved accuracy in the phishing example. That suggests reward/gate designs built from multiple atomic questions, not a single "is this good?" question.

### Gaps
- No independent peer-reviewed evaluation of RLCD exists. Jev's architecture and parameter count are undisclosed.
- I could not confirm whether the ~86 decisions/s and ~33 ms Laya figures were measured on Apple silicon. wilsonwu cites T4 for 33–40 ms. Themenonlab cites ~0.18 s for short inputs and ~2.5 s for ~6,300 tokens on an "Apple GPU" (chip unspecified).

---

## 1. Decision models as reward models / critics (RLHF, RLAIF, PRM vs ORM, small classifier RM vs LLM-as-judge, over-optimization, ensembles)

### Takeaway
Learned classifier reward models are the backbone of RLHF. Optimizing against them reliably produces over-optimization, which is predictable (Gao et al.) and partly mitigated by KL penalties and conservative ensembles. The only Jev-as-reward experiment (RLJF) reproduces the classic hacking modes: length inflation and confident fabrication. Fixes there were length caps and grounding.

### Cited Findings
- Learning a reward model from human pairwise preferences and optimizing it with RL (deep RL from human preferences) is the origin of RLHF. — [Christiano et al. 2017](https://arxiv.org/abs/1706.03741)
- InstructGPT: a reward model trained on labeler comparisons, then PPO with a KL penalty to the SFT policy. Labelers preferred the 1.3B InstructGPT outputs over the 175B GPT-3 outputs. — [Ouyang et al. 2022](https://arxiv.org/abs/2203.02155)
- RLAIF replaces human preference labels with LLM labels. Lee et al. report RLAIF achieving performance comparable to RLHF on summarization and dialogue. — [Lee et al. 2023](https://arxiv.org/abs/2309.00267); Constitutional AI is the earlier AI-feedback approach — [Bai et al. 2022](https://arxiv.org/abs/2212.08073)
- Gao, Schulman & Hilton (ICML 2023) on reward over-optimization:
  - Gold reward under RL follows R = d0 + d1·√KL − d2·KL, so it rises, peaks, then declines while proxy reward keeps rising.
  - The coefficients scale smoothly with reward-model parameter count.

  — [Gao et al. 2022/2023](https://arxiv.org/abs/2210.10760); [ICML listing](https://mlanthology.org/icml/2023/gao2023icml-scaling/)
- Over-optimization also occurs in direct alignment algorithms (DPO etc.), without an explicit RM. — [Rafailov et al., NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/e45caa3d5273d105b8d045e748636957-Paper-Conference.pdf)
- Ensembles (Coste et al.): worst-case optimization (WCO) and uncertainty-weighted optimization (UWO) over RM ensembles.
  - For best-of-n they gave up to 70% improvement and "practically eliminated" over-optimization.
  - For PPO they always reduced over-optimization, and combined with a modest KL penalty they prevented it. Tested with 25% label noise.

  — [Coste et al. 2023](https://arxiv.org/abs/2310.02743)
- Ensembles help but do not eliminate reward hacking, because members tend to share error patterns (pretraining-seed diversity matters). — [Eisenstein et al. 2023 "Helping or Herding?"](https://arxiv.org/abs/2312.09244). A cheaper alternative is a shared-backbone ensemble with multiple heads — [Zhang et al. 2024, Scalable Ensembling](https://arxiv.org/pdf/2406.01013)
- Process vs outcome reward models: process supervision (step-level labels, PRM800K) outperformed outcome supervision on MATH. — [Lightman et al. 2023 "Let's Verify Step by Step"](https://arxiv.org/abs/2305.20050); earlier comparison on GSM8K — [Uesato et al. 2022](https://arxiv.org/abs/2211.14275); automatic process labels without humans — [Math-Shepherd, Wang et al. 2023](https://arxiv.org/abs/2312.08935)
- Benchmarking RMs, including small classifier RMs vs generative judges: [RewardBench, Lambert et al. 2024](https://arxiv.org/abs/2403.13787). Generative verifiers (an LLM judge that produces next-token "Yes" probabilities) are an alternative to classifier heads — [Zhang et al. 2024 GenRM](https://arxiv.org/abs/2408.15240)
- **RLJF (Reinforcement Learning from Jev Feedback), Hugging Face community blog, 2026:**
  - Reward = `good_score/3 − P(rude)`. The inputs are a Jev 4-point score question and a Jev yes/no question; logits over the allowed answers become probabilities, with no text parsing.
  - Policy: Qwen2.5 0.5B/1.5B/7B with LoRA, trained with GRPO (8 samples per prompt, 150 steps).
  - Hacking:
    - Without length caps, reply length tripled (30 → 90 tokens).
    - RLJF policies fabricated specific menu paths and prices: 63% fabrication at 1.5B. RLHF policies hedged instead. Judges rewarded false confidence.
    - Grounding with product docs cut fabrication to 2–4%.
  - At 7B, the Jev-trained policy scored +0.81 on the Skywork RM vs +0.58 for RLHF.
  - Caveats: a single task, 2–4 runs per configuration, no human validation.

  — [HF blog, jgalego/rljf](https://huggingface.co/blog/jgalego/rljf)
- Reward hacking in large models (2026 survey covering mechanisms and emergent misalignment). — [arXiv 2604.13602](https://arxiv.org/pdf/2604.13602)

### Inferences
- **What it improves:** cheap, dense, low-latency reward. With GRPO's 8+ samples per prompt, the reward call count is large, so per-call cost matters. Laya runs locally (~1 GB). Jev costs $0.042/M input tokens ([Layer3Labs](https://www.layer3labs.io/guides/jev-explained)), so 1M reward calls at ~500 tokens ≈ $21.
- **Failure modes:**
  - Goodhart over-optimization (Gao).
  - Length and confidence bias (RLJF).
  - Name-over-definition rubric drift (2609.26758).
  - Shift as the policy's outputs move away from the RM's training distribution.
- **Mitigations grounded in the literature:**
  - KL penalty and early stopping at the predicted gold-reward peak.
  - Conservative ensembles (WCO/UWO). A multi-head Laya ensemble on one backbone is cheap.
  - Decomposed atomic questions.
  - Grounding checks via an NLI-style "is this claim supported by the docs" yes/no question.
- **M1-8GB feasibility:**
  - Inference: Laya as a reward model for a small policy is feasible.
  - Training: a full PPO/GRPO loop on a ≥0.5B *policy* and the reward model together is tight in 8 GB. Policy LoRA at 0.5B is plausible but slow. That is my estimate, not from a source.
  - Classic RL environments (tabular/small MLP policies) with Laya as the reward function are clearly feasible.

### Gaps
- No published study measures Laya's calibration *under policy-induced distribution shift*, i.e. ECE on policy samples late in RL training.
- The "Revisiting scaling laws for reward optimization" paper ([arXiv 2609.38526](https://arxiv.org/html/2609.38526)) appeared in search, but I did not read it. Its findings are not captured here.

---

## 2. Decision models as preference judges and success detectors for agents (VLM/LLM success detectors, foundation models as reward, Eureka)

### Takeaway
Foundation models are now used routinely as zero-shot reward/success signals. Each form below has a characteristic trade-off: binary success detectors are robust but sparse, CLIP-similarity rewards are dense but noisy, and LLM-written reward code is powerful but needs evolutionary search with environment feedback.

### Cited Findings
- **Success detectors:** fine-tuned VLMs (Flamingo-based) used as binary success detectors. The paper reports better generalization than bespoke detectors but notes failure under distribution shift. — [Du et al. 2023, "Vision-Language Models as Success Detectors"](https://arxiv.org/abs/2303.07280)
- **LLM as proxy reward** for negotiation/game tasks from a few examples or a description. — [Kwon et al. 2023, "Reward Design with Language Models"](https://arxiv.org/abs/2303.00001)
- **VLM-RM:** CLIP image-text similarity as a zero-shot reward (e.g., humanoid "kneeling"). The paper reports that larger VLMs give better rewards. — [Rocamonde et al. 2023](https://arxiv.org/abs/2310.12921)
- **Motif:** reward derived from LLM preferences over NetHack event captions; it outperformed training on game score alone in some settings. — [Klissarov et al. 2023](https://arxiv.org/abs/2310.00166)
- **Eureka:** a GPT-4-written reward function code plus evolutionary refinement from training statistics. It outperformed expert human-engineered rewards on most of 29 tasks. — [Ma et al. 2023](https://arxiv.org/abs/2310.12931)
- **ELLM:** LLM-suggested goals as exploration rewards, with goal achievement scored by embedding similarity. — [Du et al. 2023](https://arxiv.org/abs/2302.06692)
- LangChain added Jev as a judge in LangSmith Evals. — [Runware](https://runware.ai/blog/jev-laya-and-the-emerging-role-of-decision-models)

### Inferences
- A Laya `noul` ("did the agent complete X?") over a textual trajectory or state description is a direct analogue of a success detector. Its probability can serve as a shaped reward or a terminal flag.
- Calibration matters most here. Thresholding P(success) at 0.5 with ECE 0.466 (out of box) will mislabel many episodes.
- Cost:
  - VLM detectors need a GPU.
  - A text-only Laya detector runs on M1-8GB but needs a text rendering of the state.
  - Jev is text-only too ([Layer3Labs](https://www.layer3labs.io/guides/jev-explained)).

### Gaps
- No published experiment uses Jev or Laya as a success detector in an RL environment (e.g., ALFWorld, ScienceWorld, TextWorld).

---

## 3. Decision models as safety shields / constraint checkers (shielded RL, CMDPs, guard models)

### Takeaway
Shielding blocks unsafe actions before execution, and constrained MDPs optimize reward subject to cost limits. Classifier guards (Llama Guard, Prompt Guard) and decision models can act as learned shields or cost functions. However, they give probabilistic, not formal, guarantees, and are themselves attackable.

### Cited Findings
- **Formal shielding:** a shield computed from a temporal-logic safety spec monitors and corrects the RL agent's actions. — [Alshiekh et al. 2018 "Safe RL via Shielding"](https://arxiv.org/abs/1708.08611)
- **Constrained Policy Optimization** for CMDPs. — [Achiam et al. 2017](https://arxiv.org/abs/1705.10528)
- **Safe RLHF:** separate reward and cost models (helpfulness vs harmlessness) optimized as a constrained problem via a Lagrangian. — [Dai et al. 2023](https://arxiv.org/abs/2310.12773)
- **Llama Guard:** a Llama-2-7B classifier for input/output safety taxonomy classification. — [Inan et al. 2023](https://arxiv.org/abs/2312.06674)
- **Decision models as guards:**
  - Laya was evaluated at ~0.698 accuracy on prompt-injection detection, and both models' docs warn decision models alone are insufficient defenses. — [wilsonwu.me](https://wilsonwu.me/en/blog/2026/jev-vs-laya/)
  - Jev itself is "vulnerable to adversarial prompt injection". — [Layer3Labs](https://www.layer3labs.io/guides/jev-explained)
  - LangChain demonstrated Jev for "model routing and tool-risk checks". — [Runware](https://runware.ai/blog/jev-laya-and-the-emerging-role-of-decision-models)

### Inferences
- **Pattern:** `if P(unsafe | state, proposed_action) > τ then block / substitute / escalate`. Choose τ via conformal calibration (see section 9) to bound the miss rate on calibration data. Alternatively, feed P(unsafe) as the CMDP cost signal with a Lagrangian (as in Safe RLHF / CPO).
- **Failure modes:**
  - Adversarial inputs: the policy itself can learn to evade a learned shield if the shield's verdict affects reward. This is a reward-hacking analogue.
  - Distribution shift.
  - Name-label confusion (2609.26758).
  - At ~70% injection-detection accuracy, Laya should be one layer, not the shield.
- **M1-8GB:** feasible. Laya at ~33 ms (T4) / sub-second on Apple GPU per check is cheap enough to gate every action in a text agent. Llama Guard 7B would not fit comfortably at fp16 in 8 GB (my estimate; 4-bit quantized may).

### Gaps
- No benchmark compares formal shields vs learned-classifier shields on the same agent tasks with calibrated thresholds.
- I did not verify Prompt Guard (86M) specs in this session.

---

## 4. Decision models as action-space pruning / affordance models (SayCan, action elimination)

### Takeaway
Scoring which actions are feasible/relevant and pruning the rest reliably speeds up learning in large discrete action spaces. A typed `choice`/`noul` decision model is a natural affordance or action-elimination scorer for text and tool environments.

### Cited Findings
- **SayCan:** the LLM scores how useful a skill is ("Say"). A learned value function scores whether the skill can succeed from the current state ("Can"). The product selects the action. — [Ahn et al. 2022](https://arxiv.org/abs/2204.01691)
- **Action Elimination Network (AE-DQN):** a contextual-bandit eliminator learns which actions are invalid in a state, which speeds up learning in Zork's large text action space. — [Zahavy et al. 2018](https://arxiv.org/abs/1809.02121)
- **Laya choice limits:**
  - Laya: options share the input budget; take care above ~20 options. — [Runware](https://runware.ai/blog/jev-laya-and-the-emerging-role-of-decision-models)
  - Jev: up to 255 options. — [Layer3Labs](https://www.layer3labs.io/guides/jev-explained)

### Inferences
- **Pattern A (affordance):** for each candidate action, ask Laya `noul: "Is action a applicable/useful in this state?"`. Multiply by policy/Q-values (SayCan-style), or mask actions with P < τ (AE-DQN-style).
- **Pattern B (prior):** use the choice distribution as a policy prior / KL anchor. This is cheaper than an LLM prior.
- **Risk:** over-pruning removes the optimal action, causing irrecoverable bias. Keep ε-exploration over pruned actions, or prune only with a conformal guarantee.
- **M1-8GB:** feasible. With N candidate actions you need N `noul` calls, or a single `choice` call if N ≤ ~20.

### Gaps
- No papers test encoder-classifier affordance scorers vs LLM scorers inside SayCan-style loops at matched latency.

---

## 5. Decision models as routers / gating policies (RouteLLM, MoE, System-1/System-2 agents)

### Takeaway
Routing between a cheap fast path and an expensive slow path is the best-documented commercial use of decision models. RL can learn the routing policy, with the decision model's calibrated confidence as the state/feature that decides when to deliberate.

### Cited Findings
- **RouteLLM:** routers trained on human-preference data plus augmentation choose between a strong and a weak LLM. Costs fell "by over 2 times in certain cases—without compromising the quality of responses". — [Ong et al. 2024](https://arxiv.org/abs/2406.18665)
- **SwiftSage:** a fast small fine-tuned model (Swift) plus a slow LLM planner (Sage), with heuristics deciding when to switch. It outperformed ReAct/SayCan-style baselines on ScienceWorld. — [Lin et al. 2023](https://arxiv.org/abs/2305.17390)
- **Talker-Reasoner:** a dual-system agent architecture (fast conversational Talker, slow Reasoner). — [Christakopoulou et al. 2024](https://arxiv.org/abs/2410.08328)
- **Expert Iteration "Thinking Fast and Slow":** a fast policy network is distilled from slow tree search, which in turn uses the network. — [Anthony et al. 2017](https://arxiv.org/abs/1705.08439)
- **Sparsely-gated MoE:** a learned gate routes each input to experts. — [Shazeer et al. 2017](https://arxiv.org/abs/1701.06538)
- **Decision models as routers:**
  - Both suit "agent routing, ticket triage". LangChain demoed Jev routing. — [Runware](https://runware.ai/blog/jev-laya-and-the-emerging-role-of-decision-models)
  - Laya ships its own checkpoint router (script/language detection). — [themenonlab](https://themenonlab.blog/blog/laya-local-system-1-decision-model)

### Inferences
- **Pattern:** state = (task features, decision-model answer distribution and its entropy/max-prob). Action = {act on System-1 answer, call LLM, ask human}. Reward = task success − λ·cost. This is a contextual bandit and trains in seconds on M1.
- Calibration matters directly: a router that thresholds confidence only works if confidence tracks accuracy. Jev's ECE 0.246 or uncalibrated Laya at 0.466 would mis-route.
- Failure modes:
  - Router collapse to always-cheap when the cost penalty is too high.
  - Shift when the upstream LLM changes.

### Gaps
- No public results on RL-trained routers that use Jev/Laya confidences as features.

---

## 6. Decision models as environment generators / verifiers and curriculum judges (OMNI, task verifiers)

### Takeaway
Models of "interestingness" and automated verifiers let open-ended systems generate and filter tasks. A cheap classifier can stand in for the expensive LLM interestingness/success judge in the inner loop, once its calibration has been checked.

### Cited Findings
- **OMNI:** "Open-endedness via Models of human Notions of Interestingness". An LLM judges which tasks are interesting and learnable, focusing the curriculum. It was tested in Crafter and a larger task space. — [Zhang et al. 2023](https://arxiv.org/abs/2306.01711)
- **OMNI-EPIC:** an LLM generates environments *and* reward/success-check code. An interestingness model archives novel tasks. — [Faldor et al. 2024](https://arxiv.org/abs/2405.15568)
- **Eureka:** generated reward code is evaluated in simulation and iterated (verifier-in-the-loop). — [Ma et al. 2023](https://arxiv.org/abs/2310.12931)

### Inferences
- **Pattern:**
  - An LLM proposes task descriptions.
  - Laya answers `noul: "novel relative to archive?"`, `noul: "solvable by current agent?"` and `score: "difficulty 1–5"`. Learning-progress-based sampling then selects tasks.
  - The LLM is called only for generation; the judging loop is local and fast.
- Laya's weakest primitive is `score` on ordinal tasks (0.372 on 5-level sentiment, per [wilsonwu.me](https://wilsonwu.me/en/blog/2026/jev-vs-laya/)), so difficulty scoring needs fine-tuning.
- **Risk:** a curriculum judge whose biases are exploited, producing a curriculum that collapses to tasks the judge mislabels "interesting".

### Gaps
- No published work uses encoder classifiers instead of LLMs as the OMNI interestingness model.

---

## 7. Decision models as state abstraction / feature extractors (question-answer state representations)

### Takeaway
Representing state as a vector of answers to semantic questions gives compact, interpretable observations that tabular or tiny policies can learn on. Decision models make this cheap. Calibrated probabilities (rather than argmax labels) preserve uncertainty for the downstream policy.

### Cited Findings
- **Language as an abstraction** for hierarchical RL (instructions as the interface between high- and low-level policies). — [Jiang et al. 2019](https://arxiv.org/abs/1906.07343)
- **Zero-shot NLI classification:** frame labels as entailment hypotheses ("This text is about X"). This is the classic way to answer arbitrary semantic questions with a cross-encoder. — [Yin et al. 2019](https://arxiv.org/abs/1909.00161)
- **Decision models answer many questions per call:** Laya's 10-question batch took 158.6 ms (English) / 72.3 ms (multilingual) on T4. — [Runware](https://runware.ai/blog/jev-laya-and-the-emerging-role-of-decision-models)
- **JevSEO:** 20 deterministic checks plus ~25 semantic Jev queries in a single request; 2.3 s per page, ~$0.0001 per page. This is effectively a 45-dim semantic feature vector. — [Layer3Labs](https://www.layer3labs.io/guides/jev-explained)

### Inferences
- **Pattern:**
  - Define K yes/no questions (e.g., "user frustrated?", "task constraint satisfied?", "tool error present?").
  - s = [P(yes_1)…P(yes_K)] ∈ [0,1]^K.
  - Discretize into bins for tabular Q-learning, or feed to a small MLP/linear policy.
  - With K ≈ 10–30, the policy trains on CPU in seconds to minutes. The decision model is frozen, so only inference cost applies (~one batched call per step).
- **Risks:**
  - Aliasing: important state info not covered by any question.
  - Correlated feature errors.
  - Non-stationarity, if the decision model is later fine-tuned.
- Being explicit about the 2609.26758 name bias is important here, since features are defined by question wording.
- **M1-8GB:** the most clearly feasible pattern in this note.

### Gaps
- I found no paper that specifically benchmarks "calibrated QA-probability vectors as RL observations" against learned embeddings. This appears to be an open, unpublished design point. Concept-bottleneck RL work is related, but I did not verify a specific citation in this session.

---

## 8. Decision models as human-feedback amplifiers (TAMER, COACH, sparse labels → classifier)

### Takeaway
Interactive RL from human evaluative feedback (TAMER, COACH) and preference-based RL (PEBBLE) show that a model learned from sparse human labels can generalize the human's judgment. A fine-tuned decision model is a modern version of that learned "human reward model".

### Cited Findings
- **TAMER:** learns a model of the human's reinforcement signal and acts greedily on it. Deep TAMER extends this to high-dimensional (Atari) inputs, using a few minutes of human feedback. — [Warnell et al. 2017 Deep TAMER](https://arxiv.org/abs/1709.10163)
- **COACH:** treats human feedback as a policy-dependent advantage signal. — [MacGlashan et al. 2017](https://arxiv.org/abs/1701.06049)
- **PEBBLE:** preference-based RL with unsupervised pretraining and relabeling, improving feedback efficiency. — [Lee et al. 2021](https://arxiv.org/abs/2106.05091)
- **Uncertainty-driven exploration in preference-based RL** uses reward-model ensemble disagreement as an intrinsic bonus. — [Liang et al. 2022 RUNE](https://arxiv.org/abs/2205.12401)
- **Laya fine-tuning:** Laya can be fine-tuned with a provided notebook (~4 h on 2×T4). Fine-tuning moved the benchmark from 0.362 to 0.766. — [themenonlab](https://themenonlab.blog/blog/laya-local-system-1-decision-model)

### Inferences
- **Pattern:**
  1. Collect a few hundred human yes/no labels on agent behavior.
  2. Fine-tune Laya's head, or LoRA (cheaper than a full fine-tune).
  3. Temperature-fit on held-out labels.
  4. Use it as the reward/critic.
  5. Query humans only where the calibrated P is near 0.5 or the ensemble disagrees (active learning).
- **M1-8GB:** fine-tuning feasibility is my estimate, not sourced. A full fine-tune of 421M in fp32 with Adam needs roughly 421M × 16 bytes ≈ 6.7 GB before activations, which is likely too tight. Head-only or LoRA training should fit. Free cloud T4s are the documented path.

### Gaps
- No source reports Laya fine-tuning on Apple silicon.

---

## 9. Calibration techniques and why calibration matters for RL

### Takeaway
Modern neural classifiers are miscalibrated. Temperature scaling fixes much of this cheaply on in-distribution data. Conformal prediction gives distribution-free coverage guarantees for "act vs abstain/ask" decisions. Both are needed when decision-model probabilities drive risk-sensitive actions, abstention, or exploration bonuses. Neither survives distribution shift without recalibration.

### Cited Findings
- **Temperature scaling:** modern deep networks are poorly calibrated, and single-parameter temperature scaling is an effective post-hoc fix. — [Guo et al. 2017](https://arxiv.org/abs/1706.04599)
- **Laya's calibration in practice:** ECE 0.466 out of box → 0.081 after temperature fitting (Jev: 0.246). — [wilsonwu.me](https://wilsonwu.me/en/blog/2026/jev-vs-laya/). Laya docs warn that "shipped probabilities can be overconfident" — [Runware](https://runware.ai/blog/jev-laya-and-the-emerging-role-of-decision-models)
- **Multilingual caveat:** confidence can remain high while Laya fails on some non-English inputs. — [Runware](https://runware.ai/blog/jev-laya-and-the-emerging-role-of-decision-models)
- **KnowNo:** conformal prediction over an LLM planner's multiple-choice option scores. If the prediction set has more than one option, the robot asks for help. This gives statistical task-completion guarantees while minimizing help, with no fine-tuning. — [Ren et al. 2023 (CoRL)](https://arxiv.org/abs/2307.01928)
- **ENCP:** episode-normalized conformal prediction for vision-and-language navigation (2026), extending conformal methods to sequential agent settings. — [arXiv 2609.17499](https://arxiv.org/html/2609.17499v1)
- **Proper scoring rules** (log, Brier, spherical, RPS) are the basis of RLCD's claim that honest probabilities maximize reward. — [Chromiak](https://mchromiak.github.io/articles/2026/Sep/17/Typed-Decision-Models-Jev-and-Laya-in-Agentic-AI/); [themenonlab](https://themenonlab.blog/blog/laya-local-system-1-decision-model)
- **Uncertainty as exploration:** reward-model uncertainty as an exploration bonus. — [RUNE](https://arxiv.org/abs/2205.12401); uncertainty-weighted conservative optimization — [Coste et al.](https://arxiv.org/abs/2310.02743)

### Inferences
- **Why it matters for RL:**
  1. Expected-reward shaping with P(success) is unbiased only if P is calibrated.
  2. Thresholded gates/shields need calibrated P to control false-negative rates.
  3. Exploration bonuses and risk-sensitive objectives need *uncertainty*, which a single softmax does not separate from noise. Use ensembles or multi-head models for epistemic uncertainty.
- **Recommended recipe:**
  1. Fit the temperature on in-domain labeled data.
  2. Wrap gating decisions in split-conformal sets (KnowNo-style) with a target error α.
  3. Re-calibrate periodically on samples from the *current* policy, because RL induces covariate shift. Gao/Coste show that proxy-gold divergence grows with optimization.
- All of this is CPU-cheap (fitting one scalar; computing a quantile) and fully feasible on M1-8GB.

### Gaps
- No source measures how RLCD-trained calibration holds up under adversarial or policy-induced shift.
- Conformal guarantees assume exchangeability, which RL trajectories violate. ENCP addresses part of this; I did not read it in full.

---

## Compute & cost summary (cross-cutting)

### Takeaway
Every pattern above is feasible at inference time on an 8 GB M1 with Laya (~1 GB RAM). The binding constraints are the *policy* size (LLM policies under RLHF/GRPO) and full fine-tuning of the encoder, not the decision model.

### Cited Findings
- **Laya resources:**
  - ~1 GB memory (MLX port, M3 Max). — [wilsonwu.me](https://wilsonwu.me/en/blog/2026/jev-vs-laya/)
  - CPU-capable, `pip install laya`, ~0.18 s short / ~2.5 s at ~6.3k tokens on Apple GPU. — [themenonlab](https://themenonlab.blog/blog/laya-local-system-1-decision-model)
  - Runs inside a browser tab. — [Medium demo](https://medium.com/@visrow/jev-vs-laya-live-demo-i-ran-a-421-million-parameter-ai-decision-model-inside-a-browser-tab-no-84b86bed1f10)
  - A Node/ONNX port exists. — [receptron/laya](https://github.com/receptron/laya)
- **Prices:**
  - Jev: $0.042/M input tokens, output free. — [Layer3Labs](https://www.layer3labs.io/guides/jev-explained)
  - Hosted Laya: $0.020/M input tokens (output free until Oct 12, 2026). — [Runware](https://runware.ai/blog/jev-laya-and-the-emerging-role-of-decision-models)
- **RLJF training scale:** Qwen2.5 0.5–7B LoRA policies, GRPO, 150 steps. — [HF RLJF](https://huggingface.co/blog/jgalego/rljf)

### Inferences
- Feasibility on M1-8GB, by pattern (my assessment):

  | Pattern | Feasibility on M1-8GB |
  |---|---|
  | (7) QA-state features + tabular/small policy | Easily feasible |
  | (5) Router bandit | Easily feasible |
  | (3) Shield / cost model | Easily feasible |
  | (4) Action pruning (≤20 actions) | Easily feasible |
  | (2) Text success detector | Easily feasible |
  | (9) Calibration / conformal | Easily feasible |
  | (8) Head/LoRA fine-tune from human labels | Feasible but slow |
  | (1) RLHF/GRPO with an LLM policy + Laya reward | Marginal (≤0.5B policy with LoRA) |
  | (6) Curriculum with LLM generation | Requires API calls for the generator |

### Gaps
- No measured Laya throughput on an M1 (8 GB) specifically. All Apple numbers are from unspecified or M3 Max hardware.
