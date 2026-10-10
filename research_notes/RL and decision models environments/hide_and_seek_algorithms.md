# Hide-and-Seek (Baker et al. 2019) Algorithms and Their 2019–2026 Successors

Notes compiled 2026-10-10. Primary source for hide-and-seek facts is the ICLR 2020 paper (arXiv 1909.07528, v1 Sept 2019): https://arxiv.org/abs/1909.07528 (text extracted from the PDF). Classic later papers are cited by their arXiv/venue pages; items marked "(not re-verified this session)" were cited from well-known primary URLs without re-fetching their content in this session.

## Q1. What exactly did the hide-and-seek system use, and what were its costs and weaknesses?

### Takeaway
Hide-and-seek used plain PPO + GAE self-play with shared parameters, entity-centric masked self-attention + LSTM policies, and a centralized "omniscient" critic. Six strategy phases emerged, but only at large scale (132M episodes / 31.7B frames / 34 h just to reach phase 4). Transfer results were weak/mixed, and agents exploited physics bugs.

### Cited Findings
**Environment and reward**
- Two teams; hiders get +1 if all hiders are hidden and −1 if any is seen; seekers get the opposite. −10 penalty for leaving an 18 m square. Episode = 240 timesteps; the first 40% is a preparation phase in which seekers are immobilized and all rewards are zero — [Baker et al. 2019](https://arxiv.org/abs/1909.07528)
- MuJoCo physics; 1–3 hiders, 1–3 seekers, 3–9 movable boxes (≥3 elongated), 2 movable ramps, randomly generated static walls/rooms — [Baker et al.](https://arxiv.org/abs/1909.07528)
- Observations: position/velocity/size of entities, masked if not in line of sight or outside a 135° frontal cone; 30 lidar-like range sensors; team identity; time left in prep phase. Actions: discretized x/y force + z torque, binary grab, binary lock. Only the locking team can unlock — [Baker et al.](https://arxiv.org/abs/1909.07528)
- No explicit incentive to interact with objects; tool use emerges only from the competitive objective — [Baker et al.](https://arxiv.org/abs/1909.07528)

**Optimization**
- Self-play ("a natural curriculum as agents always play opponents of an appropriate level"); PPO + GAE; trained on OpenAI's distributed "rapid" framework — [Baker et al.](https://arxiv.org/abs/1909.07528)
- Centralized training / decentralized execution: each agent acts on its own observation and memory, while the critic is an "omniscient value function" seeing the full unmasked state (following Pinto et al. 2017, MADDPG, COMA) — [Baker et al.](https://arxiv.org/abs/1909.07528)
- All agents share policy parameters; separate parameters per agent also reached all six stages but with lower sample efficiency — [Baker et al.](https://arxiv.org/abs/1909.07528)
- Hyperparameters: buffer 320,000 chunks; minibatch 64,000 chunks of 10 timesteps; lr 3e-4; PPO clip 0.2; gradient clip 5; entropy coefficient 0.01; γ = 0.998; λ = 0.95; max GAE horizon 160; BPTT truncation 10. Each optimization step = 60 Adam SGD substeps; each chunk is used for at most 4 optimization steps to stay roughly on-policy. CPU machines run rollouts and compute advantages — [Baker et al., App. B.5–B.6](https://arxiv.org/abs/1909.07528)

**Architecture (entity-centric attention)**
- Lidar → circular 1D conv, concatenated with self state. Each entity is concatenated with self and embedded by a dense layer whose weights are shared per entity type. Then a residual self-attention block (no position embeddings, y = dense(attn(x)) + x), masked average pooling, dense, LSTM, and separate heads for the 3 action types. Layer norm on all hidden layers sped up training and improved transfer. Unobserved entities are masked in attention and pooling (policy only) — [Baker et al., App. B.7](https://arxiv.org/abs/1909.07528)
- Default model: 1.6M parameters — [Baker et al.](https://arxiv.org/abs/1909.07528)

**Six emergent phases (episode counts)**
- (1) Running and chasing → (2) Fort building by hiders (~25M episodes) → (3) Ramp use by seekers (~75M episodes later) → (4) Ramp defense (~10M later) → (5) Box surfing (by ~380M total episodes) → (6) Surf defense (lock all boxes before building a fort) — [Baker et al.](https://arxiv.org/abs/1909.07528); videos on the [OpenAI blog](https://openai.com/index/emergent-tool-use/) (not re-verified this session)
- Hiders learn division of labor, e.g. each brings its own box. No population-based training or evolved dense rewards were needed — [Baker et al.](https://arxiv.org/abs/1909.07528)
- A new skill does not always change team reward: hider reward still falls after surf defense because 1–2 hider teams often lack time to lock all boxes — [Baker et al.](https://arxiv.org/abs/1909.07528)

**Compute and scale**
- The default (batch 64k) needs 132.3M episodes = 31.7B frames over 34 hours to reach phase 4 (ramp defense). Larger batches cut wall-clock time with little sample-efficiency loss down to 32k; batch sizes of 16k and 8k never converged — [Baker et al., Fig. 4](https://arxiv.org/abs/1909.07528)
- The authors say agents "require an enormous amount of experience... likely because the reward functions are not directly aligned with the resulting behavior", and call reducing sample complexity important future work — [Baker et al., §7](https://arxiv.org/abs/1909.07528)

**Intrinsic-motivation baselines**
- Count-based exploration produced object interaction only when the state was hand-restricted to 2-D box positions. Movement fell sharply with richer state (rotation, velocity, multiple agents), so it needs hand-chosen "interesting" dimensions. RND did slightly better than count-based in the full-state setting. Both were far below self-play at producing meaningful object interaction — [Baker et al., §6.1](https://arxiv.org/abs/1909.07528)

**Intelligence-test transfer suite (weak results)**
- 5 tests. Cognition/memory: Object Counting (supervised probe on frozen LSTM), Lock and Return, Sequential Lock. Manipulation: Construction From Blueprint, Shelter Construction. The final dense layer and layernorm are re-initialized for fine-tuning — [Baker et al., §6.2](https://arxiv.org/abs/1909.07528)
- Hide-and-seek pretraining was only *slightly* better than both count-based and from-scratch baselines on 3/5 tasks (Lock and Return, Sequential Lock, Blueprint). It was slightly worse than count-based on Object Counting and learned slower than random init on Shelter Construction. The authors blame "entangled" skill representations that are hard to fine-tune — [Baker et al.](https://arxiv.org/abs/1909.07528)
- Across phases: memory/navigation transfer rises with phase (phase-6 policy about 20% better than phase 1), Object Counting is transient (best at phase 1), and manipulation transfer is uncorrelated with phase — [Baker et al., App. A.5](https://arxiv.org/abs/1909.07528)

**Exploits**
- Box surfing works because agents can apply force to themselves while airborne. Other exploits: hiders running out of the arena while shielding with boxes, and agents exploiting physics inaccuracies (e.g., ramp launching). The authors flag the need for environment generation that avoids unwanted behaviors — [Baker et al., §7](https://arxiv.org/abs/1909.07528)
- The authors admit the strategy space is "inherently bounded and likely will not surpass the six modes" — [Baker et al.](https://arxiv.org/abs/1909.07528)
- Code was open-sourced as openai/multi-agent-emergence-environments, built on mujoco-worldgen / mujoco-py — [GitHub](https://github.com/openai/multi-agent-emergence-environments) (not re-verified this session)

### Inferences
- The algorithmic core (PPO + GAE + centralized critic + parameter sharing) is exactly what MAPPO later formalized. The novelty was the environment and the scale, not the learner.
- Hide-and-seek's three weaknesses map onto later research lines: a bounded strategy space (→ UED / open-ended environment generation), simulator exploits (→ better physics, adversarial auditing, LLM interestingness filters), and sample inefficiency (→ JAX end-to-end simulation, world models).
- Self-play with only the latest policy risks strategy cycling. Hide-and-seek did not need opponent pools, which is likely because its phases form a near-linear arms race.

### Gaps
- The paper text gives no GPU/CPU counts or dollar cost. Only batch size, episodes, frames and hours are reported. I did not find a reliable primary source for the hardware count.

## Q2. Later MARL algorithmic improvements and what each fixes

### Takeaway
MAPPO showed that hide-and-seek-style PPO with a centralized value is a strong default. Value decomposition (QMIX) targets cooperative credit assignment. PSRO, fictitious self-play and AlphaStar-style leagues fix the cycling and exploitability of naive self-play. PBT tunes hyperparameters and reward weights online.

### Cited Findings
- **MAPPO** (arXiv Mar 2021; NeurIPS 2022 Datasets & Benchmarks): PPO with minimal tuning matches or beats off-policy baselines such as QMIX on MPE, SMAC, Google Research Football and Hanabi. Key implementation factors: value normalization, value-input representation (global/agent-specific state), clipping, batch size, limited sample reuse/epochs, and death masking — [Yu et al.](https://arxiv.org/abs/2103.01955)
- **QMIX** (ICML 2018): a monotonic mixing network combines per-agent Q-values into Q_tot, so decentralized argmax stays consistent with the joint argmax. It addresses cooperative credit assignment beyond VDN's additive sum — [Rashid et al.](https://arxiv.org/abs/1803.11485); VDN: [Sunehag et al. 2017](https://arxiv.org/abs/1706.05296) (not re-verified this session). Limitation: the monotonicity constraint cannot represent some non-monotonic payoffs, which motivated QPLEX and others — [QPLEX](https://arxiv.org/abs/2008.01062) (not re-verified)
- **Neural Fictitious Self-Play** (2016): best responses to the opponents' *average* policy, converging toward Nash in imperfect-information games. This addresses naive self-play's tendency to chase the latest opponent — [Heinrich & Silver](https://arxiv.org/abs/1603.01121) (not re-verified)
- **PSRO** (NeurIPS 2017): a population of policies; build an empirical payoff meta-game, solve it with a meta-solver (Nash, etc.), and train new oracles (RL best responses) against the meta-strategy. It generalizes self-play and fictitious play and reduces overfitting to specific opponents — [Lanctot et al.](https://arxiv.org/abs/1711.00832) (not re-verified)
- **AlphaStar league** (Nature, Oct 2019): main agents, main exploiters and league exploiters, trained with prioritized fictitious self-play over a league of frozen past policies. It reached Grandmaster in StarCraft II (above 99.8% of ranked human players) — [Vinyals et al., Nature](https://www.nature.com/articles/s41586-019-1724-z) (not re-verified)
- **Population-Based Training** (2017): online evolution of hyperparameters and weights across a population — [Jaderberg et al.](https://arxiv.org/abs/1711.09846). Used with evolved internal rewards in Capture-the-Flag (FTW), which Baker et al. explicitly say they did *not* need — [Jaderberg et al. 2019, Science/arXiv](https://arxiv.org/abs/1807.01281); [Baker et al.](https://arxiv.org/abs/1909.07528)
- **JaxMARL** (NeurIPS 2024 D&B; arXiv Nov 2023) provides JAX implementations of IPPO, MAPPO, QMIX, VDN, IQL and others, plus environments. Its JAX pipeline is ~14× faster than prior implementations, and up to 12,500× when many training runs are vectorized — [JaxMARL](https://arxiv.org/abs/2311.10090); [NeurIPS paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/5aee125f052c90e326dcf6f380df94f6-Paper-Datasets_and_Benchmarks_Track.pdf)

### Inferences
- For a hide-and-seek successor, a practical stack is MAPPO/IPPO with an entity-attention encoder, plus a PFSP opponent pool or PSRO-style league to prevent forgetting of earlier counter-strategies, with PBT on the entropy coefficient and learning rate.
- QMIX-style decomposition helps intra-team credit assignment for team rewards (who locked the box?). It is off-policy and value-based, and MAPPO's evidence suggests it is not necessary for performance.

### Gaps
- I found no 2024–2026 head-to-head study that re-runs hide-and-seek with PSRO or league training. The benefit of leagues for this specific environment is untested in the sources I found.
- I did not verify specific model-based MARL papers (e.g., MAMBA-style multi-agent world models) this session.

## Q3. Open-ended and unsupervised environment design (UED)

### Takeaway
The successors to "a fixed environment + self-play autocurriculum" generate the environment distribution itself: POET (co-evolution), PAIRED (regret-based adversary), PLR/Robust PLR and ACCEL (regret-prioritized replay plus mutation), XLand/AdA (procedural task space at massive scale), and Kinetix/Craftax (cheap JAX task spaces). OMNI/OMNI-EPIC and Genie use foundation models to propose "interesting" tasks or generate worlds.

### Cited Findings
- **POET** (Jan 2019): co-evolves environments and agents with transfer between pairs, producing walker terrains that direct training could not solve — [Wang et al.](https://arxiv.org/abs/1901.01753); Enhanced POET: [2003.08536](https://arxiv.org/abs/2003.08536) (not re-verified)
- **PAIRED** (NeurIPS 2020): an adversary designs environments to maximize *regret* (antagonist minus protagonist return), so it avoids unsolvable levels. This introduced the UED formalism — [Dennis et al.](https://arxiv.org/abs/2012.02096) (not re-verified)
- **Prioritized Level Replay** (ICML 2021): replays procedurally generated levels with high learning potential, scored by positive value loss / TD error — [Jiang et al.](https://arxiv.org/abs/2010.03934). **Robust PLR / Dual Curriculum Design** (NeurIPS 2021): train only on replayed levels, which gives minimax-regret guarantees at Nash — [Jiang et al.](https://arxiv.org/abs/2110.02439). **ACCEL** (ICML 2022): adds small edits to high-regret levels to grow complexity — [Parker-Holder et al.](https://arxiv.org/abs/2203.01302) (not re-verified)
- **XLand / Open-Ended Learning Leads to Generally Capable Agents** (July 2021): a procedurally generated multi-agent 3D task universe with PBT and generational training — [DeepMind OEL Team](https://arxiv.org/abs/2107.12808). **AdA** (Jan 2023): a transformer-memory agent with automatic curriculum in XLand 2.0 adapts in-context to new tasks on human-like timescales — [Adaptive Agent Team](https://arxiv.org/abs/2301.07608) (not re-verified)
- **Kinetix** (arXiv Oct 2024; ICLR 2025 Oral): Jax2D hardware-accelerated 2D physics; tens of millions of procedurally generated physics tasks plus 74 handmade levels; a generalist agent zero-shot solves unseen handmade levels — [Matthews et al.](https://arxiv.org/abs/2410.23208); [ICLR oral](https://iclr.cc/virtual/2025/oral/31729); [code](https://github.com/FlairOx/Kinetix/)
- **Craftax** (ICML 2024): JAX rewrite of Crafter plus NetHack-like extension. Craftax-Classic runs ~405k steps/s (257× Crafter), full Craftax ~267k steps/s. A 1B-step PPO run finishes in ~51 min on one RTX 4090, and full Craftax remains unsolved — [Matthews et al.](https://arxiv.org/abs/2402.16801)
- **Multi-Agent Craftax** (arXiv Nov 2025): Craftax-MA and Craftax-Coop (heterogeneous agents, trading). 250M interactions run in under an hour on one L40S — [arXiv 2511.04904](https://arxiv.org/abs/2511.04904)
- **Melting Pot** (ICML 2021) evaluates MARL generalization to novel co-players in social-dilemma "substrates" — [Leibo et al.](https://arxiv.org/abs/2107.06857); Melting Pot 2.0 — [arXiv 2211.13746](https://arxiv.org/abs/2211.13746) (not re-verified)
- **OMNI** (2023) uses an LLM "model of interestingness" to pick learnable *and* interesting tasks — [Zhang et al.](https://arxiv.org/abs/2306.01711). **OMNI-EPIC** (arXiv May 2024; ICLR 2025): an LLM writes environment *and* reward code (reset/step/reward/terminated), with a post-generation interestingness check against an archive. In principle it can produce any simulatable task — [Faldor et al.](https://arxiv.org/abs/2405.15568); [ICLR 2025](https://iclr.cc/virtual/2025/poster/29279)
- **Genie** (Feb 2024, ICML 2024) learns a latent-action world model from unlabeled video — [Bruce et al.](https://arxiv.org/abs/2402.15391) (not re-verified). **Genie 3** (Aug 2025) generates interactive worlds from text in real time at 720p/24 fps with consistency over several minutes and promptable world events. DeepMind ran its SIMA agent toward goals inside Genie 3 worlds — [DeepMind blog](https://deepmind.google/blog/genie-3-a-new-frontier-for-world-models/); [TechCrunch](https://techcrunch.com/2025/08/05/deepmind-thinks-genie-3-world-model-presents-stepping-stone-towards-agi/)

### Inferences
- Regret-based UED (Robust PLR/ACCEL) directly addresses hide-and-seek's admitted limit, a bounded strategy space in one fixed generator. A hide-and-seek redesign could let ACCEL mutate wall/box/ramp layouts by regret while teams self-play.
- LLM interestingness filters (OMNI-EPIC) could also screen out exploit-prone or degenerate generated environments, but this has not been demonstrated for physics exploits specifically.
- Genie-style generated worlds are not yet a substitute for a physics simulator when exact dynamics matter: consistency is only claimed for "several minutes", and there is no public access to a multi-agent RL-training API that I found.

### Gaps
- I found no published multi-agent hide-and-seek built on Kinetix/Jax2D. It would be a natural, cheap re-implementation, but this is unverified.
- Genie 3 is not publicly released for RL training as far as the sources I found show.

## Q4. Hardware-accelerated simulation: what makes this cheap now

### Takeaway
End-to-end JAX (or C-native) environments that keep simulation and learning on one accelerator give 2–4 orders of magnitude speedups. Billion-step runs that once needed clusters now take about an hour on one consumer GPU.

### Cited Findings
- Brax: differentiable JAX rigid-body physics for accelerators — [Freeman et al. 2021](https://arxiv.org/abs/2106.13281) (not re-verified)
- MuJoCo Playground (Jan/Feb 2025; RSS 2025) uses MJX. Most environments train in under 10 minutes on a single GPU, e.g. humanoid and quadruped in ~6 min on an A100 — [playground.mujoco.org](https://playground.mujoco.org/); [arXiv 2502.08844](https://arxiv.org/abs/2502.08844); [RSS 2025](https://roboticsconference.org/2025/program/papers/20/)
- Isaac Lab (arXiv Nov 2025) succeeds Isaac Gym with GPU-parallel physics and photorealistic rendering for multimodal robot learning — [arXiv 2511.04831](https://arxiv.org/abs/2511.04831v1); [GitHub](https://github.com/isaac-sim/IsaacLab). It requires NVIDIA GPUs (Isaac Sim requirement, not re-verified).
- PufferLib 2.0 (RLC 2025): ~1M steps/s — [RLJ paper](https://rlj.cs.umass.edu/2025/papers/RLJ_RLC_2025_151.pdf). PufferLib 3.0: 3–5M steps/s on one machine (up to ~4M on one RTX 5090) — [puffer.ai blog](https://puffer.ai/blog.html). The author claims ~20M steps/s for PufferLib 4.0 — [Suarez on X](https://x.com/jsuarez/article/2041501879222284776) (social-media self-report, not peer-reviewed)
- Craftax: 1B steps in ~51 min on one RTX 4090; JaxMARL: up to 12,500× via vectorized runs — [Craftax](https://arxiv.org/abs/2402.16801); [JaxMARL](https://arxiv.org/abs/2311.10090)
- Apple Silicon: JAX supports the M1 CPU officially, but Apple's jax-metal GPU plugin appears unmaintained (last supported macOS Sonoma). Community alternatives (metaljax, jax-mps) are early-stage with missing StableHLO ops — [JAX issue #8074](https://github.com/jax-ml/jax/issues/8074); [PyMC discourse](https://discourse.pymc.io/t/jax-on-apple-silicon-gpu/12300); [jax-mps](https://vglazer.github.io/jax-mps); [metaljax](https://github.com/eterevsky/metaljax)

### Inferences
- **8 GB M1 laptop feasibility (my assessment, not measured):**
  - Feasible on CPU: JaxMARL MPE/small grid envs with IPPO/MAPPO at small scale; Craftax-Classic at much lower throughput than GPU (CPU vectorization of a few hundred envs); Kinetix small/medium levels at reduced env counts; MuJoCo (classic C, not MJX) single-env or a few-env MAPPO; PufferLib's C environments (CPU-native, designed for high throughput) with small policies; Melting Pot (CPU, slow).
  - Marginal: MJX/Playground (designed for GPU; CPU JAX works but is slow); DreamerV3/TD-MPC2 small configs with PyTorch MPS (long runs).
  - Not feasible: Isaac Lab (needs an NVIDIA GPU), Genie 3 (not public), the original hide-and-seek scale (31.7B frames), V-JEPA 2 pretraining, and XLand/AdA scale.
- A modern hide-and-seek clone in 2D (Jax2D/Kinetix-style physics, entity attention, MAPPO) should reach early phases on one consumer GPU in hours. On an M1 CPU, a gridworld version is realistic. This is inference only; no benchmark exists.

### Gaps
- I found no published M1/MPS benchmark numbers for Craftax, JaxMARL or Kinetix.

## Q5. World models and representation learning for decision making

### Takeaway
World models now cut sample cost substantially. DreamerV3 (Nature 2025) works across 150+ tasks with fixed hyperparameters, TD-MPC2 scales a single model to multitask control, and V-JEPA 2-AC plans zero-shot on real robots from under 62 h of robot video. Multi-agent world models remain much less mature.

### Cited Findings
- DreamerV3 (arXiv Jan 2023; Nature 2025): one algorithm with fixed hyperparameters across 150+ tasks; first to collect Minecraft diamonds from scratch without human data or curricula — [Nature](https://www.nature.com/articles/s41586-025-08744-2); [arXiv](https://arxiv.org/abs/2301.04104)
- TD-MPC2 (ICLR 2024): implicit (decoder-free) world model + MPC; a single 317M-parameter agent across 80+ tasks with fixed hyperparameters — [Hansen et al.](https://arxiv.org/abs/2310.16828) (not re-verified; check the exact task count)
- V-JEPA 2 (June 2025): self-supervised video pretraining, then an action-conditioned predictor (V-JEPA 2-AC, ~300M params) post-trained on under 62 h of DROID robot video. Zero-shot planning on Franka arms in two labs: 65–80% pick-and-place success. Planning takes ~16 s per action versus ~4 min for Cosmos — [arXiv 2506.09985](https://arxiv.org/abs/2506.09985); [Meta blog](https://ai.meta.com/blog/v-jepa-2-world-model-benchmarks/)

### Inferences
- For multi-agent environments, world models reduce environment cost but must model co-player non-stationarity. A common pattern is a learned model per agent conditioned on opponent embeddings, which I did not verify in a specific paper.
- Latent-space planners (JEPA, TD-MPC2) avoid pixel generation and are cheaper than Genie-style generative simulators for control.

### Gaps
- No verified source this session on model-based MARL (e.g., multi-agent Dreamer variants) and their gains.

## Q6. Intrinsic motivation, LLM-guided rewards and environment design, and agentic RL (2025–2026)

### Takeaway
Classic curiosity (count/RND) was weak in hide-and-seek. LLMs now supply priors for rewards (Eureka), skill curricula (Voyager) and task/environment generation (OMNI-EPIC). Separately, GRPO-style critic-free RL is being adapted to multi-turn LLM agents, which needs turn-level credit assignment (GiGPO, Turn-PPO).

### Cited Findings
- RND (2018) — [Burda et al.](https://arxiv.org/abs/1810.12894) (not re-verified); in hide-and-seek, RND was only slightly better than count-based — [Baker et al.](https://arxiv.org/abs/1909.07528)
- Eureka (ICLR 2024): GPT-4 writes and evolves reward code from environment source and training feedback. It outperformed human-expert rewards on most of 29 Isaac Gym tasks — [Ma et al.](https://arxiv.org/abs/2310.12931) (not re-verified; the paper reports 83% of tasks)
- Voyager (May 2023): an LLM agent in Minecraft with an automatic curriculum, a code skill library and iterative prompting — [Wang et al.](https://arxiv.org/abs/2305.16291) (not re-verified)
- OMNI-EPIC generates environments and rewards as code — [Faldor et al.](https://arxiv.org/abs/2405.15568)
- GRPO (DeepSeekMath, Feb 2024) removes the critic by using group-relative advantages over multiple samples per prompt — [Shao et al.](https://arxiv.org/abs/2402.03300) (not re-verified)
- RAGEN / StarPO (Apr 2025): trajectory-level multi-turn agent RL. It identifies an "Echo Trap" collapse (reward-variance cliffs, gradient spikes) and mitigates it with StarPO-S (trajectory filtering, critic, gradient stabilization) — [arXiv 2504.20073](https://arxiv.org/abs/2504.20073)
- GiGPO (NeurIPS 2025): two-level group credit assignment (episode-level plus step-level via anchor states). Reports >12% gains over GRPO on ALFWorld and >9% on WebShop — [paper PDF](https://personal.ntu.edu.sg/boan/papers/NeurIPS25_GiGPO.pdf)
- Turn-PPO (Dec 2025; EACL 2026 Findings): turn-level advantage estimation with a critic. Reports GRPO as unstable in multi-turn settings — [arXiv 2512.17008](https://arxiv.org/abs/2512.17008); [ACL Anthology](https://aclanthology.org/2026.findings-eacl.328.pdf)
- 2026 work continues on credit assignment for agentic RL — e.g., [RC-GRPO, Feb 2026](https://arxiv.org/abs/2602.03025), [TSR, Feb 2026](https://arxiv.org/abs/2602.11767), [survey "From Reasoning to Agentic: Credit Assignment...", Apr 2026](https://arxiv.org/abs/2604.09459), [PGPO, Sep 2026](https://arxiv.org/abs/2609.02236) (titles/dates from search listings only)
- Language-agent open-ended multi-agent coordination benchmark (June 2026) — [arXiv 2606.08340](https://arxiv.org/abs/2606.08340) (search listing only, contents not verified)

### Inferences
- Agentic LLM RL is rediscovering hide-and-seek-era issues: critic-free group baselines work for single-turn tasks, but long-horizon multi-turn tasks again need critics or step-level baselines (GAE-like) for credit assignment.
- LLM agents in multi-agent self-play environments are a plausible next step for a "hide-and-seek for LLMs". The 2026 coordination benchmark suggests this is underway, but I have not verified the details.

### Gaps
- No verified results this session on LLM-driven reward shaping specifically for *multi-agent* competitive environments.

## Q7. What remains unsolved

### Takeaway
Multi-agent credit assignment, non-stationarity and strategy cycling, simulator exploitation, sim-to-real, and especially *measuring* open-endedness all remain open. Hide-and-seek's own transfer suite is an early example of how hard evaluation is.

### Cited Findings
- Evaluation: hide-and-seek's transfer suite gave only marginal gains, with the effect transient or uncorrelated with emergence phase on some tasks — [Baker et al.](https://arxiv.org/abs/1909.07528). Melting Pot frames evaluation as generalization to unseen co-players — [Leibo et al.](https://arxiv.org/abs/2107.06857)
- Simulator exploitation: box surfing, out-of-bounds escapes and physics bugs — [Baker et al.](https://arxiv.org/abs/1909.07528)
- Credit assignment is still an active 2025–2026 research topic in agentic RL — [GiGPO](https://personal.ntu.edu.sg/boan/papers/NeurIPS25_GiGPO.pdf); [Turn-PPO](https://arxiv.org/abs/2512.17008); [credit-assignment survey 2026](https://arxiv.org/abs/2604.09459)
- Open benchmarks are unsolved: full Craftax remains a challenge for current algorithms — [Craftax](https://arxiv.org/abs/2402.16801); Craftax-MA targets long-horizon multi-agent generalization — [arXiv 2511.04904](https://arxiv.org/abs/2511.04904)
- Sim-to-real: V-JEPA 2-AC success is 25–80% depending on task/object, far from reliable — [arXiv 2506.09985](https://arxiv.org/abs/2506.09985); [Meta blog](https://ai.meta.com/blog/v-jepa-2-world-model-benchmarks/)

### Inferences
- **Design recipe for a better hide-and-seek today (synthesis):**
  1. Environment: JAX-native, built on Jax2D/Kinetix or MJX for speed.
  2. Learner: MAPPO with an entity-attention encoder and a centralized critic.
  3. Opponents: a PFSP/league or PSRO opponent pool against cycling.
  4. Curriculum: Robust PLR/ACCEL over level layouts for an unbounded curriculum.
  5. Task generation: an LLM interestingness filter (OMNI-EPIC-style).
  6. Exploit auditing: explicit adversarial exploit-hunting (exploiter agents à la AlphaStar).
  7. Evaluation: held-out co-players (Melting Pot-style) plus a transfer suite, all evaluated across phases.
- Non-stationarity from co-adapting agents is addressed only partially by centralized critics and population methods. No source here claims a general solution.

### Gaps
- No quantitative, community-accepted metric of open-endedness was found in this session's sources. Openness (novelty + learnability) is argued conceptually (OMNI-EPIC's interestingness, UED's regret) but not standardized.
- No verified source on systematic detection of simulator exploits.
