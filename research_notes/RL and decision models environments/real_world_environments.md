# Real-World and Software-World RL Environments: Catalogue, 2025–2026 RL Results, and Gaps

Scope: RL environments and benchmarks for real-world and software decision making (not hide-and-seek clones). Notes are current to about October 2026. Each finding has a citation. Some well-known structural facts (repo URLs, the basic S/A/R of classic gyms) come from the primary repo or paper. Statements about whether an environment runs on an 8 GB Apple M1 are mostly **inferences** from the dependency stack (Docker/VM/GPU/JAX), and the Inferences sections label them that way.

Legend for M1 (8 GB) feasibility:
- **Yes** means pure Python/CPU or JAX-on-CPU, light memory.
- **Partial** means it needs Docker containers or a VM. It is fine for single episodes, but parallel RL or Android emulators will strain 8 GB, and LLM policy training must happen remotely.
- **No** means it needs an NVIDIA GPU (CUDA/PhysX), large clusters, or heavy CFD.

---

## Q1. What environments exist per domain (state, action, reward, fidelity, multi-agent, M1 feasibility)?

### Takeaway
Software-world environments (SWE, terminal, GUI, web, customer service, AIOps, security) moved very fast in 2025–2026 and mostly use Docker sandboxes with programmatic verifiers. Physical and energy gyms (Grid2Op, CityLearn, Sinergym, SustainDC, HydroGym, Gym-TORAX, PDEControlGym) are mature Gymnasium-style simulators, and many of them are CPU-friendly. Robotics GPU simulators (Isaac Lab, ManiSkill3 GPU mode) mostly do not run on Apple Silicon. MuJoCo/MJX does.

### Cited Findings

#### A. Software engineering / DevOps
- **SWE-Gym** (ICML 2025, https://github.com/SWE-Gym/SWE-Gym). Real Python GitHub repos with executable environments and unit tests. State: repo plus issue text plus tool outputs. Action: shell and edit commands. Reward: hidden tests pass. Agents plus fine-tuned verifiers reach 32.0% on SWE-bench Verified and 26.0% on Lite, which was the open-weight SOTA at the time — [SWE-Gym paper](https://nlp.cs.berkeley.edu/pubs/Pan-Wang-Neubig-Jaitly-Ji-Suhr-Zhang_2025_SWEGym_paper.pdf)
- **R2E-Gym** (COLM 2025, https://github.com/R2E-Gym/R2E-Gym). Procedurally generated SWE environments plus hybrid verifiers. A 32B open-weight agent reaches 51% on SWE-bench Verified with hybrid verification, against 43.7% for execution-only and 42.8% for execution-free verification — [arXiv 2504.07164](https://arxiv.org/abs/2504.07164)
- **SWE-RL** (Meta, 2025). RL on open software-evolution data (PR history) with a rule-based similarity reward. Llama3-SWE-RL-70B resolves 41.0% of SWE-bench Verified, trained only on public data — [arXiv 2502.18449](https://arxiv.org/pdf/2502.18449)
- **Terminal-Bench / Terminal-Bench 2.x** (https://www.tbench.ai, leaderboard https://snorkel.ai/leaderboard/terminal-bench-2-1/). Containerized terminal tasks. State: terminal output. Action: shell keystrokes and commands. Reward: task test scripts. Training environments built around it:
  - **Endless Terminals**, which procedurally scales terminal RL environments. Llama-3.2-3B went from 0.0% to 2.2%, Qwen-2.5-7B from 2.2% to 3.4%, and Qwen-3-8B-SFT from 1.1% to 6.7% on TB2.0 — [arXiv 2601.16443](https://arxiv.org/abs/2601.16443)
  - Reported elsewhere: Qwen3-8B trained with GRPO on SETA-Env reached 12% on TB2.0, and a 9B recipe reached 27% — [HF Papers search summary](https://huggingface.co/papers?q=Terminal-Bench+2.0), [T1 paper](https://arxiv.org/pdf/2609.11042), [Environment Evolution for Terminal Agents](https://arxiv.org/pdf/2609.04128). These numbers come from search snippets and have not been verified against the PDFs.
- **AIOpsLab** (Microsoft, https://github.com/microsoft/AIOpsLab). Deploys microservice stacks, injects faults and workloads, and evaluates detection, localization, diagnosis and mitigation. It has 48 problems — [arXiv 2501.06706](https://arxiv.org/pdf/2501.06706)
- **ITBench** (IBM, https://github.com/IBM/ITBench). 94 scenarios across SRE incident resolution, CISO compliance assessment and FinOps cost management — [arXiv 2502.05352](https://ar5iv.labs.arxiv.org/html/2502.05352v1)
- Live-environment AIOps frameworks (AIOpsLab, ITBench) deploy instrumented microservice stacks, inject faults at runtime, and assess the full incident lifecycle — [CUJBench](https://arxiv.org/html/2604.23455v2)
- Newer AIOps work: **Incident-Arena** (2026) — [arXiv 2610.00648](https://arxiv.org/html/2610.00648v1); **SecRespond** for post-compromise incident response — [arXiv 2607.26791](https://arxiv.org/pdf/2607.26791); a telemetry-manipulation attack on AIOps agents ("AI Oops") — [arXiv 2508.06394](https://arxiv.org/html/2508.06394v1)
- **Dependency upgrades / build / deploy**:
  - An LLM multi-agent dependency-upgrade framework reports 71.4% precision on industrial and synthetic repos — [arXiv 2510.03480](https://arxiv.org/html/2510.03480v1)
  - Related benchmarks: CSR-Bench for repo deployment — [ACL 2025](https://aclanthology.org/2025.naacl-long.633.pdf); build-bench for compiling real repos — [arXiv 2509.25248](https://www.arxiv.org/pdf/2509.25248); DeployBench — [arXiv 2606.05238](https://arxiv.org/pdf/2606.05238)
  - VEX-Bench for supply-chain vulnerability exploitability triage (EMNLP 2026) — [Red Hat Research](https://research.redhat.com/blog/2026/09/29/vex-bench-a-benchmark-for-evaluating-llm-agents-on-software-supply-chain-vulnerability-triage/)
- **CI flake triage**: no dedicated RL gym was found. The closest work studies flakiness of LLM-generated DBMS tests — [arXiv 2601.08998](https://arxiv.org/pdf/2601.08998)
- **Compiler optimization**:
  - CompilerGym (Meta, 2021, https://github.com/facebookresearch/CompilerGym). LLVM phase ordering. State: IR features. Action: pass. Reward: code-size or instruction-count reduction.
  - Successor work, **Compiler-R1** (NeurIPS 2025), uses two-stage RL for LLM compiler auto-tuning and gets an 8.46% average IR instruction-count reduction versus `opt -Oz` — [arXiv 2506.15701](https://arxiv.org/pdf/2506.15701), [NeurIPS poster](https://neurips.cc/virtual/2025/poster/115582)
  - Magellan uses AlphaEvolve to discover compiler heuristics — [arXiv 2601.21096](https://arxiv.org/pdf/2601.21096)
- **Database knob tuning**:
  - In an LLM-vs-classical evaluation, GPT-4-Turbo and Qwen2-7B reached near-optimal configurations in 9–14 iterations, against 375 for SMAC — [arXiv 2408.02213](https://arxiv.org/pdf/2408.02213)
  - MCTuner reports up to 19.2% gains — [arXiv 2509.06298](https://arxiv.org/html/2509.06298v1)
  - L2T-Tune is a hybrid of LLM, LHS and TD3 — [arXiv 2511.01602](https://arxiv.org/html/2511.01602v3)
  - No standard open gym exists here. Each paper uses its own Postgres/MySQL plus a benchmark harness.
- **Cloud cost / FinOps**: ITBench's FinOps track is the only established benchmark found ([ITBench](https://ar5iv.labs.arxiv.org/html/2502.05352v1)). A hobby "FinOps Gym" HF Space exists, with little documentation — [HF Space](https://huggingface.co/spaces/gauri-10/finops-gym)

#### B. Utility software / personal computing
- **OSWorld / OSWorld-Verified** (https://os-world.github.io). Real Ubuntu/Windows VMs. State: screenshot and/or accessibility tree. Action: pyautogui mouse and keyboard. Reward: execution-based checker scripts.
  - **ARPO** (GRPO with experience replay) on UI-TARS-1.5 reaches 29.9% on OSWorld, +6.4 points over the base model — [arXiv 2505.16282](https://arxiv.org/pdf/2505.16282)
  - **UI-TARS-2**, trained with multi-turn RL, reports 47.5% on OSWorld — [arXiv 2509.02544](https://arxiv.org/html/2509.02544v1). Secondary aggregators also quote 53% on OSWorld-Verified, and these figures conflict — [Emergent Mind](https://www.emergentmind.com/topics/osworld-verified-and-androidworld-benchmarks)
- **AndroidWorld** (https://github.com/google-research/android_world). Android emulator with programmatic reward from app state.
  - **MobileRL-9B** reaches 80.2% on AndroidWorld and 53.6% on AndroidLab. For Qwen2.5-VL-7B its gains are +44.4% (AndroidWorld) and +32.4% (AndroidLab) — [arXiv 2509.18119](https://arxiv.org/html/2509.18119)
  - MobileGUI-RL does online RL in emulators — [arXiv 2507.05720](https://arxiv.org/pdf/2507.05720)
  - TVWorld covers remote-control TV agents — [arXiv 2601.13142](https://arxiv.org/pdf/2601.13142)
- **WebArena / VisualWebArena / BrowserGym**:
  - WebArena (https://webarena.dev) is self-hosted e-commerce, GitLab, Reddit, CMS and map sites.
  - BrowserGym (https://github.com/ServiceNow/BrowserGym) unifies MiniWoB, WebArena, VisualWebArena and WorkArena.
  - State: DOM/AXTree plus screenshot. Action: browser primitives. Reward: functional checks.
  - OpenEnv integrations (including BrowserEnv) shipped in February 2026 — [Zylos research note](https://zylos.ai/research/2026-07-15-rl-environments-as-a-service-agentic-training-infrastructure/)
- **τ-bench / τ²-bench** (Sierra, https://github.com/sierra-research/tau2-bench). Dual-control customer service across airline, retail and telecom. The agent uses domain tools under a policy document, an LLM-simulated user can also change state, and reward is the final DB-state match.
  - RL recipes: a verl τ²-airline recipe — [verl-recipe PR #121](https://github.com/verl-project/verl-recipe/pull/121); OpenEnv `tau2_env` — [OpenEnv PR #1290](https://github.com/huggingface/OpenEnv/pull/1290)
  - Training data and papers: AReaL SFT→RL data — [HF dataset](https://huggingface.co/datasets/inclusionAI/AReaL-tau2-data); MUA-RL — [arXiv 2508.18669](https://arxiv.org/pdf/2508.18669); Nemotron-Cascade 2 uses it in cascade RL — [arXiv 2603.19220](https://arxiv.org/pdf/2603.19220)
- **Email / calendar / enterprise**:
  - **EmailBench**: 206 scenarios in 16 categories. Sonnet passes only 33.5% even though 94.7% of runs make tool calls without API errors — [arXiv 2609.31906](https://arxiv.org/html/2609.31906)
  - **InboxArena** is an OpenEnv-compliant email triage RL environment. Actions are classify, prioritize, route and draft. Reward is deterministic, in [0,1], with partial credit — [GitHub](https://github.com/Myan17/inbox-arena). It is small and community-made.
  - EnterpriseBench Corecraft — [arXiv 2602.16179](https://arxiv.org/pdf/2602.16179); AgentGym2 for de-idealized real-world environments — [arXiv 2607.05174](https://arxiv.org/pdf/2607.05174); CUBE as a unified benchmark standard — [arXiv 2603.15798](https://arxiv.org/pdf/2603.15798)
- **Notification / attention management, file organization**: no dedicated open RL environment was found (see Gaps).

#### C. Security
- **AgentDojo** (ETH, https://github.com/ethz-spylab/agentdojo). 97 user tasks and 629 security cases across banking, Slack, travel and workspace. State: tool outputs that may contain injections. Reward: utility and attack success — [Emergent Mind summary](https://www.emergentmind.com/topics/agentdojo-benchmark)
- **InjecAgent** (https://github.com/uiuc-kang-lab/InjecAgent) covers indirect prompt injection in tool-integrated agents.
- RL red-teaming on these benchmarks:
  - PISmith — [arXiv 2603.13026](https://arxiv.org/pdf/2603.13026)
  - "Learning to Inject": a 1.5B RL suffix generator compromises frontier systems on AgentDojo — [arXiv 2602.05746](https://arxiv.org/html/2602.05746v1)
  - AutoDojo generates adaptive attacks — [arXiv 2606.15057](https://arxiv.org/html/2606.15057v3)
  - ARGUS is a context-aware defense — [arXiv 2605.03378](https://arxiv.org/pdf/2605.03378)
- **CybORG / CAGE Challenge 4** (https://github.com/cage-challenge/cage-challenge-4). Multi-agent defensive cyber gym with limited inter-agent communication, partial observability and a shared reward — [CAGE 4 site](https://cage-challenge.github.io/cage-challenge-4/). The first integration of LLM and RL agents on CAGE 4 — [arXiv 2509.05311](https://arxiv.org/pdf/2509.05311); "LLMs are Autonomous Cyber Defenders" — [arXiv 2505.04843](https://arxiv.org/pdf/2505.04843); CybORG++ — [arXiv 2410.16324](https://arxiv.org/pdf/2410.16324)
- **CyberBattleSim** (Microsoft, https://github.com/microsoft/CyberBattleSim). An abstract network lateral-movement attacker gym, pure Python.
- **CTF**: NYU CTF Bench (containerized, six categories), Cybench and InterCode-CTF. Agents trained on synthetic trajectories gain up to 13.1 absolute points across the three — [HF Papers summary](https://huggingface.co/papers?q=NYU+CTF)
- **SOC / phishing triage**:
  - Alert-Bench: the agent starts from a detector alert and searches a live SIEM for evidence. One dataset has 770,705 Windows events, 2,774 Sigma rules and 402 triage cases. AIDA gets F1 0.958, against 0.371–0.744 for other approaches — [arXiv 2610.10608](https://arxiv.org/html/2610.10608)
  - Simbian's AI SOC benchmark has 100 kill-chain cases, with top models at 61–67% — [Simbian blog](https://simbian.ai/blog/the-first-ai-soc-llm-benchmark) (vendor source)
  - No standard open phishing-triage RL gym was found. Only small community harnesses exist — [phishing-triage-skill](https://github.com/MuhammadMurtuzaHussain/phishing-triage-skill)

#### D. Physics / science simulation
- **MuJoCo / MJX / MuJoCo Playground** (https://playground.mujoco.org). MJX is built on XLA, so it runs on Nvidia, AMD and Apple Silicon GPUs as well as TPUs. MuJoCo itself runs natively on Apple Silicon — [MuJoCo discussion #1101](https://github.com/google-deepmind/mujoco/discussions/1101), [Bifrost comparison](https://www.bifrost.ai/blog/isaac-lab-vs-mujoco-vs-maniskill-vs-genesis/)
- **mjlab** offers the Isaac Lab API on MuJoCo-Warp — [GitHub](https://github.com/mujocolab/mjlab)
- **Brax** (https://github.com/google/brax) is JAX rigid-body physics.
- **Kinetix** (ICLR 2025 Oral, https://github.com/FLAIROx/Kinetix). Open-ended 2D physics tasks in JAX (Jax2D), with procedurally generated tens of millions of tasks plus 74 handmade levels. A general agent trained on it zero-shot solves unseen human-designed levels — [arXiv 2410.23208](https://arxiv.org/html/2410.23208v1)
- **PDEControlGym** (L4DC 2024, https://github.com/lukebhan/PDEControlGym) is 1D/2D PDE boundary control — [docs](https://pdecontrolgym.readthedocs.io/)
- **controlgym** (L4DC 2024, https://github.com/xiangyuan-zhang/controlgym) has 36 linear industrial control environments and 10 large-scale PDE environments — [PMLR](https://proceedings.mlr.press/v242/zhang24b.html)
- **HydroGym** (Nature 2026, https://github.com/dynamicslab/hydrogym). 42 validated flow-control environments: cylinder wake, cavity, pinball, channel. Agents trained on a simplified channel flow achieve 38% drag reduction on an unseen 3D wing at Re_c=200,000 — [arXiv 2512.17534](https://arxiv.org/html/2512.17534v2), [Nature](https://www.nature.com/articles/s41586-026-10917-6)
- **Gym-TORAX** (2025/2026, https://github.com/antoine-mouchamps/gymtorax). A Gymnasium wrapper on DeepMind's TORAX tokamak transport simulator. It currently has one ITER-scenario environment — [SoftwareX](https://www.sciencedirect.com/science/article/pii/S2665963826000199). CPU baselines (PI, PPO, SAC, MBPO) are available — [rl-tokamak-rampup](https://github.com/leandergrech/rl-tokamak-rampup). DeepMind's TCV magnetic control (Nature 2022) used a non-public simulator.
- **Chemistry / materials**: ChemGymRL — [arXiv 2305.14177](https://arxiv.org/pdf/2305.14177); CrystalGym for materials discovery (2025) — [arXiv 2509.23156](https://arxiv.org/pdf/2509.23156)
- **Weather / climate**:
  - RAIN uses RL to improve numerical weather/climate model parameterizations — [arXiv 2408.16118](https://arxiv.org/html/2408.16118v1)
  - GreenLight-Gym does greenhouse control with weather data — [arXiv 2410.05336](https://arxiv.org/pdf/2410.05336)
  - BoreaRL is multi-objective boreal forest management — [arXiv 2509.19846](https://arxiv.org/pdf/2509.19846)
  - Climate-adaptation RL for flooding in Copenhagen — [arXiv 2409.18574](https://arxiv.org/html/2409.18574v2)
  - No mainstream weather-model RL gym was found.

#### E. Energy and scheduling
- **Grid2Op / L2RPN** (RTE, https://github.com/rte-france/Grid2Op). Sequential power-network operation on real non-linear power-flow simulators. Perturbations cover renewable uncertainty and line contingencies. State: grid state. Action: topology changes, redispatch, curtailment. Reward: survival and operating cost — [CityLearn v2 paper context](https://arxiv.org/pdf/2405.03848)
- **CityLearn v2/v3** (https://github.com/intelligent-environments-lab/CityLearn). Multi-building demand response with batteries, EVs/V2G and occupant feedback. It uses a data-driven LSTM thermal model, so no external simulator is needed — [arXiv 2405.03848](https://arxiv.org/pdf/2405.03848). CityLearn v3 (2026) targets renewable energy communities — [arXiv 2609.21570](https://arxiv.org/pdf/2609.21570). EVLearn adds EV simulation — [arXiv 2404.06521](https://arxiv.org/pdf/2404.06521)
- **Sinergym** (https://github.com/ugr-sail/sinergym). An EnergyPlus-backed building HVAC gym — [arXiv 2412.08293](https://arxiv.org/pdf/2412.08293)
- **SustainDC** (HPE, NeurIPS 2024 D&B, https://github.com/HewlettPackard/dc-rl). Multi-agent: carbon-aware workload shifting, data-centre cooling and battery agents that affect each other — [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2024/file/b6676756f8a935e208f394a1ba47f0bc-Paper-Datasets_and_Benchmarks_Track.pdf). DCcluster-Opt covers geo-distributed data-centre scheduling — [arXiv 2511.00117](https://arxiv.org/pdf/2511.00117)
- **Gym4ReaL** is a 2025 suite for benchmarking real-world RL — [arXiv 2507.00257](https://arxiv.org/pdf/2507.00257)
- **AutoB2G** does agentic simulation plus RL for grid-interactive buildings — [arXiv 2603.26005](https://arxiv.org/pdf/2603.26005)
- **Job-shop scheduling**:
  - **JobShopLab** (https://github.com/proto-lab-ro/jobshoplab) is a Gym environment with transport, buffers, breakdowns, setup times, stochastic processing and multi-objective support — [search summary of project](https://github.com/topics/job-shop-scheduling-problem)
  - graph-jsp-env — [PyPI](https://pypi.org/project/graph-jsp-env/0.3.2)
- **Traffic**:
  - SUMO-RL (https://github.com/LucasAlegre/sumo-rl) and CityFlow (https://github.com/cityflow-project/CityFlow) are the most widely used open-source simulators.
  - PyTSC unifies multi-agent traffic signal control across simulators — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11902778/)
  - Sim2Signal provides sim-to-real TSC benchmarks — [arXiv 2609.01676](https://arxiv.org/pdf/2609.01676)
  - VissimRL — [arXiv 2601.18284](https://arxiv.org/pdf/2601.18284)
- **EV charging**: EV2Gym (https://github.com/StavrosOrf/EV2Gym) is known from its 2024 paper. My searches did not surface 2025–2026 updates for it.

#### F. Robotics / edge / mobile
- **Isaac Lab** does not support Apple Silicon for GPU-parallel simulation — [Bifrost comparison](https://www.bifrost.ai/blog/isaac-lab-vs-mujoco-vs-maniskill-vs-genesis/), [NVIDIA](https://developer.nvidia.com/isaac/lab)
- **ManiSkill3** (https://github.com/haosulab/ManiSkill) runs on SAPIEN with a GPU-parallel mode. My searches did not clearly establish its Apple Silicon support.
- **LeRobot** (https://github.com/huggingface/lerobot) supports real low-cost arms and imitation/RL policies. No 2025–2026 RL results for it were collected in this pass.
- **Android energy / notification control**: no open RL environment was found.

#### G. Customer support / healthcare / finance ops
- Customer support: τ²-bench (above).
- **MedAgentBench** (Stanford, https://stanfordmlgroup.github.io/projects/medagentbench/). A FHIR-compliant virtual EHR with 300 physician-curated tasks, 100 patient profiles and more than 700k data points. Best models score 85.33% on query tasks but 54.00% on action tasks — [alphaXiv](https://www.alphaxiv.org/overview/2501.14654)
- Related healthcare work:
  - MedAgentBench v2 — [PSB 2026](https://psb.stanford.edu/psb-online/proceedings/psb26/chen_eric.pdf)
  - FHIR-AgentBench — [arXiv 2509.19319](https://arxiv.org/pdf/2509.19319)
  - FHIR-AgentEval — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12919212/)
  - PhysicianBench — [arXiv 2605.02240](https://arxiv.org/pdf/2605.02240)
  - HealthAdminBench covers computer-use healthcare administration — [arXiv 2604.09937](https://arxiv.org/pdf/2604.09937)
- Finance:
  - FinQA as an open OpenEnv RL environment, with 290 expert questions over 22 companies — [Snorkel](https://snorkel.ai/blog/building-finqa-an-open-rl-environment-for-financial-reasoning-agents/)
  - Finance Agent Benchmark, with 537 SEC-filing questions — [arXiv 2508.00828](https://arxiv.org/abs/2508.00828)
  - FinToolBench — [alphaXiv](https://www.alphaxiv.org/abs/2603.08262)
  - FORCE-Bench for enterprise finance — [arXiv 2607.19409](https://arxiv.org/pdf/2607.19409)
  - KellyBench for long-horizon sequential betting/decision making — [arXiv 2604.27865](https://arxiv.org/pdf/2604.27865)
  - FinRL-Meta for market trading envs — [NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/file/0bf54b80686d2c4dc0808c2e98d430f7-Paper-Datasets_and_Benchmarks.pdf)

#### H. Infrastructure / hubs
- **Prime Intellect Environments Hub** is a community hub that reports 2,500+ open-source RL environments built on the `verifiers` library. Environments are split into rollout logic, programmatic reward functions and datasets — [Prime Intellect](https://www.primeintellect.ai/), [HF blog](https://huggingface.co/blog/anakin87/environments-hub)
- **Meta/HF OpenEnv** is a Gymnasium-style step/reset/state API over HTTP/Docker. Its governance committee includes Meta-PyTorch, HF, Prime Intellect, Nvidia, Microsoft and others — [Zylos](https://zylos.ai/research/2026-07-15-rl-environments-as-a-service-agentic-training-infrastructure/) (secondary source)
- **Surveys**:
  - "The Landscape of Agentic RL for LLMs" (v5, 2026) — [arXiv 2509.02547](https://arxiv.org/abs/2509.02547)
  - "Agentic Environment Engineering" survey (2026) — [arXiv 2606.12191](https://arxiv.org/pdf/2606.12191)
  - "Environment Scaling for Interactive Agentic Experience Collection" — [arXiv 2511.09586](https://arxiv.org/pdf/2511.09586)
  - Agent-World on real-world environment synthesis — [arXiv 2604.18292](https://arxiv.org/pdf/2604.18292)
  - EnvFactory on executable environment synthesis plus RL — [arXiv 2605.18703](https://arxiv.org/pdf/2605.18703)

### Inferences
- **Summary table** (M1 column is inference):

| Env | State | Action | Reward | Fidelity | Multi-agent | 8 GB M1 |
|---|---|---|---|---|---|---|
| SWE-Gym / R2E-Gym / SWE-bench | repo + issue + tool output | shell/edit | hidden tests | real repos | no | Partial (Docker per task; policy LLM remote) |
| Terminal-Bench | terminal text | shell cmds | test script | real containers | no | Partial |
| AIOpsLab / ITBench | logs, metrics, traces, kubectl | kubectl/shell | diagnosis/mitigation checks | live k8s microservices | no | No/Partial (k8s clusters too heavy for 8 GB) |
| OSWorld | screenshot/a11y tree | mouse/keyboard | checker scripts | real VMs | no | Partial (one VM at most; ARM image issues likely) |
| AndroidWorld | screenshot/UI tree | taps/typing | app state | real emulator | no | Partial (ARM emulator works, 8 GB tight) |
| WebArena / BrowserGym | DOM/AXTree/screenshot | browser ops | functional checks | self-hosted sites | no | Partial (MiniWoB: Yes) |
| τ²-bench | dialogue + tool results | tool calls/messages | DB state match | LLM user sim | dual-control (agent + user) | Yes for env; user sim needs an API LLM |
| AgentDojo / InjecAgent | tool outputs w/ injections | tool calls | utility & ASR | synthetic suites | attacker vs agent | Yes |
| CybORG CAGE 4 | partial net obs | defensive ops | shared reward | abstract emulation | yes | Yes |
| CyberBattleSim | graph obs | exploits/moves | owned nodes | abstract | optional | Yes |
| NYU CTF / Cybench | terminal | shell | flag captured | real containers | no | Partial |
| Grid2Op | grid state | topology/redispatch | survival/cost | real power flow | optional | Yes |
| CityLearn | building/energy obs | battery/HVAC setpoints | cost/carbon/comfort | LSTM surrogate | yes | Yes |
| Sinergym | zone temps/weather | setpoints | energy+comfort | EnergyPlus | no | Yes (EnergyPlus has macOS builds) |
| SustainDC | load/temp/carbon | shift/cool/battery | carbon+energy | surrogate | yes | Yes |
| SUMO-RL / CityFlow | queues/phases | phase select | delay/queue | microscopic | yes | Yes |
| JobShopLab | shop state | dispatch | makespan etc. | discrete-event | no | Yes |
| Kinetix / Brax / MJX | physics state/pixels | torques/thrusters | task | 2D/3D rigid | no | Yes (JAX CPU/Metal, slower) |
| PDEControlGym / controlgym | field state | boundary/actuators | tracking error | numerical PDE | no | Yes |
| HydroGym | flow fields/sensors | jets/rotation | drag/lift | CFD (Firedrake) | no | Partial (small 2D only) |
| Gym-TORAX | plasma profiles | heating/current | profile targets | 1D transport sim | no | Yes (JAX) |
| ManiSkill3 / Isaac Lab | robot state/pixels | joint cmds | task | high (GPU) | no | No (Isaac); ManiSkill CPU mode unclear |
| MedAgentBench | FHIR responses | FHIR GET/POST | task checks | synthetic EHR | no | Partial (Docker FHIR server) |

- These environments come in three shapes:
  - Software environments mostly have **sparse, terminal, programmatically verified** rewards (tests, checkers, DB-state match).
  - Policy-heavy environments (τ²-bench, AgentDojo, SOC/phishing, email) involve **soft judgments**: is this action policy-compliant? Is this email phishing? Is this injected? Is it safe to execute?
  - Energy, physics and scheduling environments have **dense numeric rewards** and small vector state, so a text decision model has little to add except at a supervisory/LLM-planner layer.
- So a calibrated fast text decision model (Laya/Jev style) fits best in the policy-heavy group, as:
  1. a per-step **safety/policy gate** (τ²-bench policy compliance, AgentDojo injection detection before a tool call, a "should this kubectl command run?" check in AIOpsLab)
  2. a **process-reward / partial-credit judge** for long-horizon sparse tasks (OSWorld, Terminal-Bench, SWE), where R2E-Gym already shows that execution-free verifiers add value
  3. a **policy head** for discrete triage actions: email routing, SOC alert escalation, phishing verdicts, CI flake vs real failure.

### Gaps
- I could not verify Apple Silicon support for ManiSkill3, OSWorld's ARM images or AIOpsLab's minimum memory from primary docs. The M1 column is inferred.
- I did not find current status (maintained or archived) for CompilerGym, EV2Gym or CityFlow in 2026.
- I did not verify LeRobot RL training results or Isaac Lab 2025–2026 features in this pass.

---

## Q2. Which 2025–2026 papers report RL (incl. GRPO-style LLM-agent RL) on these, and with what gains?

### Takeaway
RL with verifiable rewards on agentic environments produced large gains in 2025–2026, especially in mobile GUI (MobileRL: +44 points for a 7B model on AndroidWorld), SWE (open 32B at 51% on SWE-bench Verified), and desktop GUI (ARPO +6.4 points, UI-TARS-2 at about 47.5%). Terminal agents remain hard at small scale (≤12–27% on TB2.0). Classic-control domains report physics-RL gains, such as HydroGym's 38% drag reduction transfer.

### Cited Findings
- **SWE**:
  - R2E-Gym: 51% on SWE-bench Verified with 32B and a hybrid verifier — [arXiv 2504.07164](https://arxiv.org/abs/2504.07164)
  - SWE-RL: 41.0% with 70B — [arXiv 2502.18449](https://arxiv.org/pdf/2502.18449)
  - SWE-Gym: 32.0% Verified — [paper](https://nlp.cs.berkeley.edu/pubs/Pan-Wang-Neubig-Jaitly-Ji-Suhr-Zhang_2025_SWEGym_paper.pdf)
- **Terminal**: Endless Terminals gives gains of a few points for 3–8B models on TB2.0 — [arXiv 2601.16443](https://arxiv.org/abs/2601.16443). GRPO Qwen3-8B on SETA-Env reaches 12% and a 9B recipe reaches 27%. These figures are from snippets — [HF Papers](https://huggingface.co/papers?q=Terminal-Bench+2.0)
- **Mobile GUI**:
  - MobileRL-9B: 80.2% AndroidWorld, 53.6% AndroidLab. For GLM-4.1V-9B the gains are +72.5% and +43.5% — [arXiv 2509.18119](https://arxiv.org/html/2509.18119)
  - Generalization of online mobile RL — [OpenReview](https://openreview.net/forum?id=INoDyme6wS)
- **Desktop GUI**:
  - ARPO reaches 29.9% on OSWorld (+6.4) — [arXiv 2505.16282](https://arxiv.org/pdf/2505.16282)
  - UI-TARS-2 reports about 47.5% on OSWorld via multi-turn RL — [arXiv 2509.02544](https://arxiv.org/html/2509.02544v1)
  - Agentic reward modelling for GUI trajectories — [arXiv 2602.00575](https://arxiv.org/pdf/2602.00575)
- **Customer service**:
  - MUA-RL — [arXiv 2508.18669](https://arxiv.org/pdf/2508.18669)
  - SENTINEL, failure-driven RL for tool agents — [arXiv 2606.12908](https://arxiv.org/pdf/2606.12908)
  - Divergence-point preference learning — [arXiv 2606.23112](https://arxiv.org/pdf/2606.23112)
  - AReaL SFT→RL on τ² domains, described as SOTA — [HF](https://huggingface.co/datasets/inclusionAI/AReaL-tau2-data)
  - Exact deltas were not extracted.
- **Compiler**: Compiler-R1 gets 8.46% IR instruction reduction vs `-Oz` — [arXiv 2506.15701](https://arxiv.org/pdf/2506.15701)
- **Security**:
  - RL attackers (Learning to Inject, PISmith) break AgentDojo defenses — [arXiv 2602.05746](https://arxiv.org/html/2602.05746v1), [arXiv 2603.13026](https://arxiv.org/pdf/2603.13026)
  - CTF agents trained on synthetic trajectories gain up to +13.1 points — [HF Papers](https://huggingface.co/papers?q=NYU+CTF)
  - LLM-prompted guidance improves RL agents on CybORG — [arXiv 2509.05311](https://arxiv.org/pdf/2509.05311)
- **Flow control**: HydroGym agents reach 38% drag reduction on an unseen 3D wing — [arXiv 2512.17534](https://arxiv.org/html/2512.17534v2)
- **General recipes and hubs**:
  - INTELLECT-3 trained on Environments Hub environments — [arXiv 2512.16144](https://arxiv.org/pdf/2512.16144)
  - Open-AgentRL (RLAnything, AutoTool; ICML 2026) — [GitHub](https://github.com/Gen-Verse/Open-AgentRL)
  - CuES, a curiosity-driven environment synthesis for agentic RL — [arXiv 2512.01311](https://arxiv.org/pdf/2512.01311)

### Inferences
- The most consistent ingredient behind large gains is a **reliable verifier**: test suites, app-state checkers, DB-state match. Where verification is soft (email quality, policy compliance, SOC verdict), RL results are scarcer. That is where a calibrated judge model can unlock RL.
- R2E-Gym's hybrid verifier (execution plus execution-free) beating either alone suggests calibrated text judges are useful even when tests exist.

### Gaps
- Exact τ²-bench RL gains (pass^k deltas) were not extracted.
- No 2025–2026 LLM-agent RL results were found for Grid2Op, CityLearn, Sinergym or SUMO beyond general LLM+RL hybrids (e.g. AutoB2G).
- No RL-trained (as opposed to evaluated) results were found on AIOpsLab/ITBench.

---

## Q3. Where are the gaps — domains with no good open gym yet (project ideas)?

### Takeaway
Clear gaps exist where decisions are textual, judgment-heavy and frequent, with outcomes that can be partly checked:
- CI flaky-test triage
- notification/attention management
- file organization
- phishing/email triage at scale
- cloud FinOps actions
- DB knob tuning as a standard gym
- on-device Android energy/notification control

These are good fits for a fast calibrated yes/no/score decision model acting as policy, judge or safety gate. They are also lightweight enough to build and run on an 8 GB M1.

### Cited Findings
- No dedicated CI-flake RL gym surfaced. Only flakiness studies exist — [arXiv 2601.08998](https://arxiv.org/pdf/2601.08998)
- Dependency-upgrade work is evaluated on industrial or synthetic repos, without a shared open gym — [arXiv 2510.03480](https://arxiv.org/html/2510.03480v1)
- FinOps: only ITBench's FinOps track and a sparsely documented HF Space exist — [ITBench](https://ar5iv.labs.arxiv.org/html/2502.05352v1), [HF Space](https://huggingface.co/spaces/gauri-10/finops-gym)
- DB knob tuning: each paper uses its own harness (GPTuner, MCTuner, L2T-Tune) — [arXiv 2509.06298](https://arxiv.org/html/2509.06298v1), [arXiv 2511.01602](https://arxiv.org/html/2511.01602v3)
- Email: EmailBench is evaluation-only (33.5% pass for Sonnet), and InboxArena is a small community RL environment — [arXiv 2609.31906](https://arxiv.org/html/2609.31906), [InboxArena](https://github.com/Myan17/inbox-arena)
- SOC/phishing: benchmarks exist (Alert-Bench, Simbian), but there is no open RL gym with a phishing-email stream plus user-report and escalation dynamics — [arXiv 2610.10608](https://arxiv.org/html/2610.10608), [Simbian](https://simbian.ai/blog/the-first-ai-soc-llm-benchmark)
- Tokamak control: Gym-TORAX has only one environment (an ITER scenario) — [SoftwareX](https://www.sciencedirect.com/science/article/pii/S2665963826000199)
- AIOps attack surface: telemetry manipulation can subvert AIOps agents, which motivates safety-gate environments — [arXiv 2508.06394](https://arxiv.org/html/2508.06394v1)

### Inferences (project ideas, ranked by fit to a fast text decision model and to M1 feasibility)
1. **FlakeGym (CI flake triage).**
   - State: CI log excerpt, test history and diff summary.
   - Actions: rerun, quarantine, file bug, revert, or mark real failure.
   - Reward: ground truth from historical reruns (e.g. mined GitHub Actions data) plus a cost model (CI minutes, developer time).
   - Decision model role: a calibrated P(flaky) as the policy. It is text-only and CPU-friendly.
2. **NotifyGym (notification/attention management).**
   - State: notification text, app, time and user context.
   - Actions: deliver now, batch, silence, or summarize.
   - Reward: a simulated user model (LLM or preference data) for interruption cost against missed-urgent cost.
   - No open gym was found. Natural extension: Android on-device energy plus notification control.
3. **PhishTriage-Gym.**
   - State: a stream of reported and unreported emails with headers.
   - Actions: allow, quarantine, escalate, or ask for more evidence (tool calls such as URL detonation stubs).
   - Reward: label correctness, analyst-time cost and attacker-adaptation rounds (red-team RL as in AgentDojo-style attacks).
4. **InjectionGate on AgentDojo/τ²-bench.** Wrap existing environments with a pre-tool-call yes/no "safe to execute?" judge. Measure the utility/ASR trade-off against calibration (ECE). It needs no new simulator.
5. **FinOpsGym.**
   - State: a synthetic cloud billing and usage graph.
   - Actions: rightsize, buy reservation, schedule shutdown, or delete orphan resources.
   - Reward: cost saved minus SLO-violation risk.
   - Decision model role: a safety check ("will this break prod?").
6. **KnobGym.** A standard Postgres-in-Docker knob-tuning gym with a fixed workload suite, a fixed budget and a surrogate mode (no live DB) so it is M1-friendly.
7. **FileOrg-Gym.** A synthetic home directory. Actions: move, rename, tag or delete. Reward from a hidden target taxonomy plus "never delete important files" safety constraints. The decision model scores destructive actions.
8. **DepUpgrade-Gym.** Mine Dependabot/Renovate PRs. Actions: accept, pin, patch code, or defer. Reward: CI pass after merge. This is a smaller, cheaper cousin of SWE-Gym.
9. **LLM-supervisor overlays on energy gyms.** Add a text "operator" layer to Grid2Op/CityLearn/SustainDC for exceptional events (alerts, maintenance notes) where a text decision model approves or vetoes RL actions. The physical sims are CPU-friendly.

- Build these on **OpenEnv or `verifiers`** so they plug into the 2,500+-environment ecosystem and the existing GRPO trainers (verl, AReaL, prime-rl) — [HF blog](https://huggingface.co/blog/anakin87/environments-hub)

### Gaps
- I did not find usage data or a validated user model for notification management. Reward design there would need new data or an LLM user simulator, and the realism of such a simulator is unproven.
- Whether calibrated small decision models measurably improve RL sample efficiency as reward judges in these domains is untested in the literature I found. That makes it an open research question rather than an established result.
