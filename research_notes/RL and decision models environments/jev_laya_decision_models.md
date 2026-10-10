# Jev, Laya and non-generative "System One" decision models (as of Oct 2026)

Labels used below: **[VENDOR]** = claim by TypeSafe/Convai or restated from their docs; **[INDEP]** = independent test/third party; **[AGG]** = aggregator/explainer restating others.

## Q1. What exactly are Jev and Laya? (architecture, training, data, license, access, HF location)

### Takeaway
Jev (TypeSafe AI, launched 15 Sept 2026) is a closed, hosted, waitlisted API whose architecture/parameter count is undisclosed beyond "parallel sampler + RLCD post-training." Laya (Convai Innovations / Nandakishor M, 18-19 Sept 2026) is an Apache-2.0 open-weight bidirectional encoder (ModernBERT-large 421M English; mmBERT-base 322M multilingual) with a small typed decision head that scores each option at its own [MASK] token, trained with RL against strictly proper scoring rules.

### Cited Findings
**Jev**
- Released by TypeSafe AI on 15 Sept 2026; marketed as the first "System One model"; takes "state" (text/JSON) plus questions defined in code; returns choices, scores or probabilities with no prose/rationale [AGG] — [layer3labs](https://www.layer3labs.io/guides/jev-explained), [sanity.io glossary](https://www.sanity.io/glossary/jev-typesafe-ai-model)
- TypeSafe describes "a new architecture, a parallel sampler, and post-training called Reinforcement Learning for Calibrated Decisions (RLCD)" which "replaces RLHF" [VENDOR] — [typesafe.ai](https://typesafe.ai/), [runware.ai](https://runware.ai/blog/jev-laya-and-the-emerging-role-of-decision-models)
- TypeSafe "has not published the neural topology, parameter count, training recipe, or loss" [INDEP commentary] — [Michał Chromiak blog](https://mchromiak.github.io/articles/2026/Sep/17/Typed-Decision-Models-Jev-and-Laya-in-Agentic-AI/); Runware likewise notes insufficient detail for independent verification — [runware.ai](https://runware.ai/blog/jev-laya-and-the-emerging-role-of-decision-models)
- Access: closed weights, hosted only, early access/waitlist; no self-hosting/on-prem [AGG] — [layer3labs](https://www.layer3labs.io/guides/jev-explained). Also available via OpenRouter (OpenRouter key, TS/Python SDKs, `POST /api/alpha/decisions` or System One API) — [OpenRouter docs](https://openrouter.ai/docs/guides/community/jev), and via Runware `/v1/systemone` — [runware.ai](https://runware.ai/blog/jev-laya-and-the-emerging-role-of-decision-models)
- Pricing: $0.042 per 1M input tokens, output free ("$42 per billion input tokens") [VENDOR] — [typesafe.ai](https://typesafe.ai/), [runware.ai](https://runware.ai/blog/jev-laya-and-the-emerging-role-of-decision-models)
- TypeSafe FAQ concedes Jev may still "get things wrong" and is not deterministic [VENDOR] — [typesafe.ai](https://typesafe.ai/)

**Laya**
- Open-source non-autoregressive decision engine by Convai Innovations (India; CEO Nandakishor M), Apache 2.0, Sept 2026 [AGG] — [eesel.ai](https://www.eesel.ai/blog/laya-ai), [aiweekly](https://aiweekly.co/alerts/convai-ships-laya-a-421m-modernbert-decision-model-apache-20)
- Checkpoints / HF: `convaiinnovations/laya` (18 Sep 2026; ModernBERT-large, 421M, 512 ctx), `convaiinnovations/laya-typed-decisions` (18 Sep; same 421M model fine-tuned on typed-decisions train split, 1,024 ctx), `convaiinnovations/laya-multilingual` (19 Sep; mmBERT-base, 322M, 1,024 ctx) — [systemonemodels.org](https://systemonemodels.org/models/laya/). Official site says the HF repo `convaiinnovations/laya` bundles all 3 checkpoints in subfolders; code at github.com/NandhaKishorM/laya; `pip install laya>=0.3.3` [VENDOR] — [laya.convaiinnovations.com](https://laya.convaiinnovations.com/)
- Release date conflict: aiweekly says 19 Sept 2026 — [aiweekly](https://aiweekly.co/alerts/convai-ships-laya-a-421m-modernbert-decision-model-apache-20); English checkpoints dated 18 Sept — [systemonemodels.org](https://systemonemodels.org/models/laya/)
- Model card architecture: ModernBERT-large backbone (395M) or mmBERT-base (22 layers, 256k vocab) + decision head of 2 transformer layers, an option-marker scorer and an act/escalate head; "Every option is scored at its own [MASK] token, then softmaxed over" the question's options, so the answer schema is defined at request time without retraining [VENDOR] — [HF model card](https://huggingface.co/convaiinnovations/laya)
- Training (RLCD): reward = strictly proper scoring rules ("log + spherical, plus ranked probability score for ordinal questions"); "Expected reward is maximised only by reporting honest probabilities"; updates via REINFORCE with group-mean baseline (GRPO-style), TD(λ=1.0) for multi-turn [VENDOR] — [HF model card](https://huggingface.co/convaiinnovations/laya). Conflict: the official site describes RLCD "via PPO over sequence representations" — [laya.convaiinnovations.com](https://laya.convaiinnovations.com/)
- Training data not disclosed — [aiweekly](https://aiweekly.co/alerts/convai-ships-laya-a-421m-modernbert-decision-model-apache-20); the specialized checkpoint was trained on "four synthetic workflow types" — [runware.ai](https://runware.ai/blog/jev-laya-and-the-emerging-role-of-decision-models)
- Origin: Nandakishor began Laya in April of the previous year while exploring detection of hallucinations before an LLM finishes generating — [Analytics India Mag](https://analyticsindiamag.com/ai-features/this-kerala-engineer-built-open-source-jev-alternative-a-year-before-the-hype)
- Community ports: Node.js/TypeScript via ONNX Runtime ([receptron/laya](https://github.com/receptron/laya)); in-browser demo ([Medium/visrow](https://medium.com/@visrow/jev-vs-laya-live-demo-i-ran-a-421-million-parameter-ai-decision-model-inside-a-browser-tab-no-84b86bed1f10)); Unsloth local server exposing a Jev-compatible API ([Unsloth docs](https://unsloth.ai/docs/models/decision-laya))

### Inferences
- Laya's design is essentially a cross-encoder / multiple-choice encoder head (option-marker scoring + softmax) with an RL proper-scoring-rule objective; "RLCD" with proper scoring rules on a single-step classifier is close in effect to minimizing log-loss/Brier, i.e. standard calibrated classification.
- "Jev-compatible API" ports (Unsloth, Runware) mean RL code written against the Jev schema can swap to local Laya, but confidence semantics differ (see Q2).

### Gaps
- Jev's parameter count, architecture and training data are unknown. Laya's training corpus is undisclosed. Exact HF repo structure (one repo with subfolders vs three repos) is inconsistently reported.

## Q2. Decision types, input/output format, calibration claims and evidence

### Takeaway
Both use the same three primitives: `choice` (pick from options, full distribution), `score` (ordinal scale, probability-weighted position) and `noul` (P(yes) for a statement), given a `state` plus typed questions; output is probabilities/confidence only. Calibration is decent on choice/yes-no in-distribution (independent Jev ECE ~0.06-0.08) but poor on score questions and out-of-distribution; Laya's raw probabilities are badly overconfident until per-question temperature fitting.

### Cited Findings
- Primitives: choice ("selects from defined options" with distribution), score ("places the input on an ordered scale"), noul ("probability that a yes-or-no statement is true") — [runware.ai](https://runware.ai/blog/jev-laya-and-the-emerging-role-of-decision-models); Jev score returns "probability-weighted position" — [OpenRouter docs](https://openrouter.ai/docs/guides/community/jev)
- Request = `state` (text or JSON; nested fields can be referenced) + typed questions; response = typed answers with per-option probabilities, confidence values and `usage.cost`; no reasoning traces — [OpenRouter docs](https://openrouter.ai/docs/guides/community/jev)
- Local Laya/Jev-compatible endpoint: `POST /v1/systemone` with `model`, `state`, `questions` (up to 64 questions; score up to 10 levels) — [Unsloth docs](https://unsloth.ai/docs/models/decision-laya)
- Jev limits: up to 255 options per choice question; 32K context — [runware.ai](https://runware.ai/blog/jev-laya-and-the-emerging-role-of-decision-models), [OpenRouter docs](https://openrouter.ai/docs/guides/community/jev)
- Laya limits: 512-token context (English; state budget 320 tokens), 1,024 (multilingual/typed-decisions; multilingual extensible to 8,192 with RoPE); options share a fixed budget, care above ~20 options — [systemonemodels.org](https://systemonemodels.org/models/laya/), [HF model card](https://huggingface.co/convaiinnovations/laya), [runware.ai](https://runware.ai/blog/jev-laya-and-the-emerging-role-of-decision-models)
- Jev and Laya confidence calculations differ; when migrating, threshold on `probabilities` not `confidence` — [Unsloth docs](https://unsloth.ai/docs/models/decision-laya)
- Laya calibration [VENDOR]: ECE 0.213 raw (English) to 0.081 after domain temperature fitting — [HF model card](https://huggingface.co/convaiinnovations/laya); other sources quote raw ECE 0.466 to 0.081 — [systemonemodels.org](https://systemonemodels.org/models/laya/), [aiweekly](https://aiweekly.co/alerts/convai-ships-laya-a-421m-modernbert-decision-model-apache-20) (conflicting raw figures). Laya docs warn "shipped probabilities can be overconfident" — [runware.ai](https://runware.ai/blog/jev-laya-and-the-emerging-role-of-decision-models)
- Jev calibration [INDEP]: on 900 unseen synthetic tickets ECE 0.072-0.084 (choice), 0.079 (boolean), 0.325 (score) vs noise floor 0.024; boolean underconfident (refit T=0.66), choice overconfident (T=1.30), score severely overconfident (T=1.92) — "type-specific post-processing rather than one calibrated distribution"; study cost ~$0.06 — [scienthoon/jev-ood-calibration](https://github.com/scienthoon/jev-ood-calibration)
- Jev ECE 0.025 on routing (72 cases, best of Jev/Haiku 4.5/Opus 5); 0.0712 on 60 hand-labeled tool-call risk cases; 0.0588 on 662 prompt-injection messages [INDEP, small n] — search summary of [stern9/jev-bench](https://github.com/stern9/jev-bench), [dev.to webofmike](https://dev.to/webofmike/i-benchmarked-jev-on-agent-tool-call-risk-calibration-held-49i3)
- Meta-review: median ECE 0.071 on 13 public subsets; direction of error changes with data (overconfident GoEmotions, underconfident synthetic); "70.4% of choice probabilities came back as exactly 0"; TypeSafe publishes no calibration metrics — [dev.to aws-builders](https://dev.to/aws-builders/jev-after-eight-days-of-independent-tests-level-with-mid-price-llms-behind-the-frontier-1c60)
- Flowtivity reports Jev ECE 0.246 (on typed-decisions) vs Laya 0.081 refit — [flowtivity](https://flowtivity.ai/blog/laya-open-source-jev-alternative/) (Convai's own site calls this "3x improvement vs Jev" — [HF model card](https://huggingface.co/convaiinnovations/laya))

### Inferences
- For RL rewards, use `noul`/`choice` probabilities rather than `score`; fit per-question temperatures on a few hundred labeled examples before treating outputs as calibrated rewards.
- Hard zeros (70% of choice probs = 0) make log-prob-based rewards unstable; clip/smooth.

### Gaps
- No vendor-published calibration curves for Jev; no independent Laya calibration study beyond restatements of Convai's numbers.

## Q3. Benchmarks: accuracy, latency, memory, on-device feasibility

### Takeaway
Laya is ~33-40 ms/question on a T4 (vendor) vs Jev ~70-500 ms network round trip; Laya needs ~4-5 GB RAM, ~0.7-0.85 GB weights, runs on CPU/Apple GPU/ONNX/browser. Accuracy: Jev zero-shot ≈ mid-price LLMs, 6.5-11.5 pts behind frontier (independent); Laya base checkpoints are near chance zero-shot and only competitive after fine-tuning.

### Cited Findings
- Laya latency [VENDOR]: 32.8-39.5 ms single question on Tesla T4; 72.3-158.6 ms for 10 batched; 103-332 questions/s; CPU 193-464 ms preloaded; Apple GPU 1.7 s for a 4,000-token input — [HF model card](https://huggingface.co/convaiinnovations/laya); "7.2 ms/question batched" — [laya.convaiinnovations.com](https://laya.convaiinnovations.com/)
- Jev latency: "roughly 70 to 500 milliseconds" — [runware.ai](https://runware.ai/blog/jev-laya-and-the-emerging-role-of-decision-models); 236-276 ms (third-party figure used by Convai) — [flowtivity](https://flowtivity.ai/blog/laya-open-source-jev-alternative/); server ~0.1 s, 76 ms network-adjusted [INDEP] — [dev.to aws-builders](https://dev.to/aws-builders/jev-after-eight-days-of-independent-tests-level-with-mid-price-llms-behind-the-frontier-1c60)
- TypeSafe claim "193.6x faster ... (0.114s vs 8.566s)" and "444.6x cheaper" [VENDOR] — [typesafe.ai](https://typesafe.ai/); independent: 0.5x-12.1x faster, 0.6x-478x cheaper depending on baseline; vendor comparator unnamed — [dev.to aws-builders](https://dev.to/aws-builders/jev-after-eight-days-of-independent-tests-level-with-mid-price-llms-behind-the-frontier-1c60)
- Memory: Unsloth minimum RAM 4 GB (multilingual, 678 MB) / 5 GB (English & typed-decisions, 846 MB); CPU default, GPU optional; first request loads in 10-20 s, then "well under a second on most CPUs"; runs on Mac/Windows/Linux — [Unsloth docs](https://unsloth.ai/docs/models/decision-laya). English checkpoint 808 MB download — [flowtivity](https://flowtivity.ai/blog/laya-open-source-jev-alternative/)
- Contrary CPU result [INDEP]: on a 4 vCPU/7 GB VPS, first cold predict 85.4 s, warm median 49.4 s, checkpoint load 675 s; "a batch tool, not a request path" without GPU — [flowtivity](https://flowtivity.ai/blog/laya-open-source-jev-alternative/) (sharply contradicts vendor/Unsloth CPU numbers)
- Laya accuracy [VENDOR]: typed-decisions fine-tuned 0.766; base English 0.362, multilingual 0.342/0.352 vs 0.318 random and 0.461 majority baseline; AG News 0.950; Enron spam 0.993 (ECE 0.013); phishing 0.980 (ECE 0.012); Banking77 0.425; SST-5 0.372; MASSIVE intent macro 0.227 across 51 languages; 45/51 languages >3x random — [laya.convaiinnovations.com](https://laya.convaiinnovations.com/), [HF model card](https://huggingface.co/convaiinnovations/laya), [aiweekly](https://aiweekly.co/alerts/convai-ships-laya-a-421m-modernbert-decision-model-apache-20), [mchromiak](https://mchromiak.github.io/articles/2026/Sep/17/Typed-Decision-Models-Jev-and-Laya-in-Agentic-AI/)
- Comparison table: typed decisions Laya-FT 0.766 vs Jev 0.727; AG News 0.950 vs 0.910; Banking77 0.425 vs 0.870 — [flowtivity](https://flowtivity.ai/blog/laya-open-source-jev-alternative/). aiweekly editor: the 0.766 "is fine-tuned on the benchmark's own train split" — [aiweekly](https://aiweekly.co/alerts/convai-ships-laya-a-421m-modernbert-decision-model-apache-20)
- systemonemodels.org "Decision Index 0.3": Laya 4.43 vs Jev 60.11; public benchmarks 6.19 vs 57.96 — [systemonemodels.org](https://systemonemodels.org/models/laya/)
- Jev accuracy [INDEP meta-review of 14 preprints/104 repos/33 posts to 23 Sept]: "level with mid-price LLMs and 6.5 to 11.5 points behind the frontier"; median 11.6 macro-F1 behind best per-task model on 7,977 human-labeled items; Banking77 median 80.9% over 8 runs; spam 98.33% on 18,514 emails; Gemma 4 26B with label-probability scoring trails Jev by only 2.1 pts — [dev.to aws-builders](https://dev.to/aws-builders/jev-after-eight-days-of-independent-tests-level-with-mid-price-llms-behind-the-frontier-1c60)
- Jev on unseen synthetic tickets: 75.1% overall; anger 91.7%, queue 89.0%, policy-based priority 44.7%; public benchmarks 86-94% — [scienthoon](https://github.com/scienthoon/jev-ood-calibration)
- Runware's phishing example (synthetic): Jev direct 62.6%, Claude Haiku 4.5 81.3%, Jev with five separate signals 95%, domain-rules baseline 91.8% — [runware.ai](https://runware.ai/blog/jev-laya-and-the-emerging-role-of-decision-models)

### Inferences
- On an Apple Silicon M1 8 GB, Laya should fit (4-5 GB min RAM per Unsloth) and run via CPU/MPS/ONNX; expect sub-second warm latency per vendor/Unsloth, but the Flowtivity VPS result shows performance can be far worse on weak CPUs — benchmark locally.
- Decomposing a judgment into several noul questions (Runware's 62.6% → 95%) is a key usage pattern for reward design.

### Gaps
- No direct M1 8 GB or phone benchmark found. Browser demo exists but without published latency numbers retrieved here. The "I tested Laya" Substack (Elisa Terumi, 3 Oct 2026) results are paywalled — [Substack](https://exploringartificialintelligence.substack.com/p/i-tested-laya-and-the-result-surprised). The gnana70 Medium test returned 403.

## Q4. Stated use cases; use as reward models, judges, guards for agents/RL

### Takeaway
Stated uses: routing, tool selection, triage, phishing/spam, prompt-injection/guardrails, moderation, eval. There is real (non-vendor) evidence of RL use: an HF blog "RLJF" trained 0.5-7B policies with GRPO using Jev as reward, showing strong reward hacking (length, fabrication); an arXiv paper used Jev as a zero-shot alignment-failure detector (median AUROC 0.886); LangChain added Jev as a LangSmith Evals judge.

### Cited Findings
- Use cases: agent routing, tool selection, ticket triage, model evaluation, risk checks; LangChain demoed Jev for "model routing and tool-risk checks" — [runware.ai](https://runware.ai/blog/jev-laya-and-the-emerging-role-of-decision-models); LangChain "added it as a judge in LangSmith Evals" — search summary of [runware.ai](https://runware.ai/blog/jev-laya-and-the-emerging-role-of-decision-models)
- Laya intended uses: email triage/routing, guardrails/moderation, support ticket classification, security incident detection; not for >20 options without tuning or production without calibration [VENDOR] — [HF model card](https://huggingface.co/convaiinnovations/laya). Nandakishor cites phishing, prompt-injection detection, moderation, model routing, support — [Analytics India Mag](https://analyticsindiamag.com/ai-features/this-kerala-engineer-built-open-source-jev-alternative-a-year-before-the-hype)
- Laya guardrails 0.931 at 50% coverage (selective deferral) — [flowtivity](https://flowtivity.ai/blog/laya-open-source-jev-alternative/)
- RLJF (Jev as reward): GRPO, 8 samples/prompt, 150 steps, LoRA, policies 0.5B-7B; reward = good_score/3 - P(rude) (minus P(invented) when grounded). Results: at 1.5B RLHF +0.81 on OpenAssistant RM but +0.19 on Jev judges; RLJF +0.69 on System One judges, +0.31 on human-pref RMs. Without docs, RLJF policies fabricated menus/prices in 63-66% of replies vs 24% for RLHF yet scored higher; grounded 1.5B policy fabricated 4%. Length hacking: replies doubled by step 75; length cap removed half to three quarters of gains on Skywork — [HF blog jgalego/rljf](https://huggingface.co/blog/jgalego/rljf)
- "Just Ask Jev" (Guo et al., 24 Sep 2026): zero-shot alignment-failure detection over 10 failure types/44 benchmarks/5 target models (RLCDAlignBench); median AUROC 0.886, beats supervised baselines on most, 63x cheaper than LLM judges; caveat: context fields that encode the label matter more than question wording — [arXiv 2609.29429](https://arxiv.org/abs/2609.29429)
- Pentest harness (Barbosa, 24 Sep 2026): Jev/Laya as decision layers for vuln confirmation, severity recalibration, agent selection, validation loops; single NeuroSploit case study (13 vulns), explicitly not statistically significant — [arXiv 2609.28940](https://arxiv.org/abs/2609.28940)
- "A bounded decision model is not an agent ... Its role is a semantic decision layer between unstructured evidence and code" — [mchromiak](https://mchromiak.github.io/articles/2026/Sep/17/Typed-Decision-Models-Jev-and-Laya-in-Agentic-AI/)
- TypeSafe: set thresholds for autonomous action vs human escalation — [typesafe.ai](https://typesafe.ai/); Laya's act/escalate head "carries no usable signal yet ... reads 1.0 for almost every input" — [HF model card](https://huggingface.co/convaiinnovations/laya)

### Inferences
- As RL reward, these are best as cheap dense shaping/guard signals combined with grounding checks, length penalties, and periodic LLM/human audits; the RLJF results show optimizing a single decision-model judge is easily hacked.
- Laya's 33 ms local inference + open weights make it the only one usable inside high-throughput on-policy RL loops without API rate/cost constraints; Jev's API latency (~100-500 ms) suits offline eval or batched rollouts.

### Gaps
- No published RL result using Laya as reward. No Jev rate limits documented.

## Q5. Limitations, criticisms, failure modes, context length, fine-tuning

### Takeaway
Main failure modes: option-label sensitivity, confident failure out-of-distribution/out-of-language, weak ordinal scoring, no "unknown" signal on unknowable tasks, high-cardinality degradation (Laya), short context (Laya 512), and probabilities needing per-type recalibration. Laya is effectively a fine-tune-first model.

### Cited Findings
- Jev documents weaknesses in numerical precision, counting, date comparison — [mchromiak](https://mchromiak.github.io/articles/2026/Sep/17/Typed-Decision-Models-Jev-and-Laya-in-Agentic-AI/)
- Jev: swapping which rubric sits behind no/yes changed 32.5% of answers; Russian XNLI 88.3%→77.3%; every KoBBQ "unknown" answer wrong; fair-die tasks probabilities averaged 83% vs true 17% — [dev.to aws-builders](https://dev.to/aws-builders/jev-after-eight-days-of-independent-tests-level-with-mid-price-llms-behind-the-frontier-1c60)
- Jev fails to signal uncertainty when labels depend on policy not text (priority 44.7%) — [scienthoon](https://github.com/scienthoon/jev-ood-calibration)
- "A confidently wrong choice from the options it was given is still wrong"; simple rules may beat models in some domains — [runware.ai](https://runware.ai/blog/jev-laya-and-the-emerging-role-of-decision-models)
- Laya: English checkpoint on Khmer scored 0.000 at 0.952 confidence; score is weakest primitive; overconfident until temperature fit; near chance on typed-decisions without fine-tuning — [systemonemodels.org](https://systemonemodels.org/models/laya/); noul on English can "follow its option labels instead of the state"; 77-label questions get ~3-4 tokens per label — [HF model card](https://huggingface.co/convaiinnovations/laya)
- Laya 512-token window "is roughly one email, not a support thread" — [flowtivity](https://flowtivity.ai/blog/laya-open-source-jev-alternative/)
- Fine-tuning: "Fine-tuning on decisions from your own domain is where accuracy jumps"; official notebook trains on Kaggle free 2x T4 with dataset building, temperature fitting, Hub upload (github.com/NandhaKishorM/laya/notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb) — [HF model card](https://huggingface.co/convaiinnovations/laya)
- Labeling 50 to a few hundred decisions enables calibration improvement for Jev — [dev.to aws-builders](https://dev.to/aws-builders/jev-after-eight-days-of-independent-tests-level-with-mid-price-llms-behind-the-frontier-1c60)

### Inferences
- For RL environments with long trajectories (agent traces), Laya's 512/1,024 context requires summarizing or windowing the state; Jev's 32K handles full traces.
- Always randomize/permute option order and labels during evaluation to detect label-following bias.

### Gaps
- Jev cannot be fine-tuned by users (no documented fine-tuning API found).

## Q6. "System One vs System Two" framing; relation to cross-encoders, NLI zero-shot classifiers, reward models

### Takeaway
"System One" (Kahneman) = fast, single-pass, reflexive bounded judgment vs "System Two" = slow, deliberative generative LLM reasoning. Technically, Laya is a cross-encoder-style multiple-choice classifier; the closest prior art is NLI-based zero-shot classification and scalar reward models, with novelty mainly in the typed request-time schema and proper-scoring-rule RL for calibration. Independent tests show LLMs scored via label-token probabilities nearly match Jev.

### Cited Findings
- Convai calls Laya "System 1: the instant, reflexive judgment, as opposed to the slow, deliberative System 2 reasoning you get from a generative LLM"; bidirectional encoder reads whole input and scores options — [eesel.ai](https://www.eesel.ai/blog/laya-ai)
- TypeSafe's framing: "fast, intuitive judgment" replacing token-by-token generation for bounded decisions — [runware.ai](https://runware.ai/blog/jev-laya-and-the-emerging-role-of-decision-models)
- Laya option scoring at per-option [MASK] markers with softmax over options — [HF model card](https://huggingface.co/convaiinnovations/laya)
- Plain Gemma 4 26B with label-probability scoring trails Jev by 2.1 pts; reranking 0.692 vs dedicated reranker 0.691 — [dev.to aws-builders](https://dev.to/aws-builders/jev-after-eight-days-of-independent-tests-level-with-mid-price-llms-behind-the-frontier-1c60)
- RLJF author notes standard LLMs can approximate this via prompting plus logit extraction over allowed answer tokens (Qwen example) — [HF blog jgalego/rljf](https://huggingface.co/blog/jgalego/rljf)
- Other "open replicas"/alternatives: Nimble, Luce, Laya; Unsloth also serves "Clef" — [dev.to aws-builders](https://dev.to/aws-builders/jev-after-eight-days-of-independent-tests-level-with-mid-price-llms-behind-the-frontier-1c60), [Unsloth docs](https://unsloth.ai/docs/models/decision-laya), [pinggy](https://pinggy.io/blog/best_open_source_jev_alternatives_self_hosted_decision_models/)

### Inferences
- The Laya head is functionally a multiple-choice cross-encoder (like BERT SWAG heads) generalized to request-time label text, similar to NLI zero-shot classifiers (premise=state, hypothesis=option) but scoring all options jointly in one pass. Proper-scoring-rule RL on a single-step decision is mathematically close to supervised log-loss/Brier training.
- Compared with a reward model (scalar on prompt+response), decision models offer multiple named, calibrated sub-judgments per call, useful for decomposed/multi-objective RL rewards.

### Gaps
- No source explicitly benchmarks Laya against classic NLI zero-shot classifiers (e.g., DeBERTa-MNLI) or cross-encoders; the mchromiak blog explicitly does not discuss these connections.
