# Pattern-Aware Milestone Critic (PAMC)

**Detecting and intercepting implicit reasoning failure patterns before the milestone critic wastes computation on doomed trajectories.**

---

## The Problem

Multi-step LLM agents decompose complex tasks into milestones (think OS-Themis), but the critic only evaluates after each milestone completes. By then, four silent failure patterns have already degraded the reasoning:

- **Early Pruning** — The agent terminates search prematurely, cutting off promising branches before exploring alternatives
- **Path Lock-in** — Commitment to a failing sub-path, compounding errors with each step
- **Targeted Backtracking** — Circular re-exploration that wastes compute without meaningful progress
- **Knowledge-Guided Prioritization Drift** — The agent drifts from task-specified priorities toward knowledge-biased shortcuts

Current milestone critics evaluate these degraded trajectories anyway, burning compute on evidence audits that can't rescue a trajectory already committed to failure. The RL reward signal arrives too late to guide productive exploration.

---

## What PAMC Does

**Pattern-Aware Milestone Critic (PAMC)** integrates OS-Themis-style milestone decomposition with real-time implicit pattern detection. The critic monitors the reasoning trajectory at each step and computes a **Pattern Risk Score (PRS)** — a 0.0–1.0 measure of how close the current trajectory is to a known failure pattern.

Rather than auditing evidence chains for *all* trajectories, PAMC:

1. **Intercepts** trajectories showing high PRS before they consume further compute
2. **Adjusts** the reward signal for medium-risk trajectories based on pattern type and severity
3. **Audits** evidence chains only for trajectories that survive the pattern filter

The result: the milestone critic works on better-quality reasoning traces, RL signals guide exploration earlier, and compute is allocated where it matters.

---

## Architecture

```
Task Input
    │
    ▼
┌──────────────────────────────────────────────────────┐
│           MILESTONE DECOMPOSER (OS-Themis)           │
│  Decompose task into verifiable, evidence-linked      │
│  milestones with checkpoint boundaries                │
└────────────────────────┬────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────┐
│            TRAJECTORY PATTERN ANALYZER               │
│  Monitor reasoning trace for 4 implicit patterns:    │
│  • Early Pruning (premature search termination)      │
│  • Path Lock-in (commitment to failing sub-path)     │
│  • Targeted Backtracking (circular re-exploration)  │
│  • Knowledge-Guided Prioritization drift             │
└────────────────────────┬────────────────────────────┘
                         │
          ┌──────────────┴──────────────┐
          │  Pattern Risk Score (PRS)  │
          │  0.0 (safe) → 1.0 (failure)│
          └──────────────┬──────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────┐
│          PATTERN-AWARE MILESTONE CRITIC              │
│  Adjust milestone reward based on PRS:               │
│  • Low risk (PRS < 0.3): standard evidence audit    │
│  • Medium risk (0.3–0.7): trigger alternative path   │
│  • High risk (PRS > 0.7): halt + backtrack signal   │
│  Audit evidence chains only for surviving paths     │
└────────────────────────┬────────────────────────────┘
                         │
                         ▼
              ┌───────────────────┐
              │   RL Reward Signal │
              │  (Pattern-Adjusted) │
              └───────────────────┘
```

---

## Key Components

| Component | Description | Responsibility |
|-----------|-------------|----------------|
| **Milestone Decomposer** | OS-Themis-style task decomposition | Break task into verifiable, evidence-linked checkpoints |
| **Trajectory Pattern Analyzer** | Monitors live reasoning trace | Detect 4 implicit failure patterns at each step |
| **Pattern Risk Scorer (PRS)** | Aggregates pattern signals into 0.0–1.0 score | Triage: audit / redirect / halt |
| **Pattern-Aware Critic** | Tiered response to PRS | Adjusts reward and evidence audit strategy |
| **Alternative Path Generator** | Triggered on medium-risk trajectories | Proposes branching alternatives to current path |
| **Evidence Chain Auditor** | OS-Themis-style evidence audit | Only invoked for low-risk surviving paths |
| **RL Training Loop** | PPO-based reward optimization | Train agent to minimize PRS while maximizing task reward |

---

## Installation

```bash
# Clone the repository
git clone https://github.com/your-org/pamc.git
cd pamc

# Create virtual environment
python -m venv venv && source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export OPENAI_API_KEY="your-key-here"   # or ANTHROPIC_API_KEY
export WANDB_API_KEY="your-wandb-key"   # for experiment tracking
```

**requirements.txt core dependencies:**
```
torch>=2.0
transformers>=4.36
rllib>=2.0
networkx>=3.0
numpy>=1.24
scipy>=1.10
wandb>=0.15
```

---

## Quick Start

```python
from pamc import PAMCAgent, MilestoneDecomposer, TrajectoryPatternAnalyzer
from pamc.critic import PatternAwareMilestoneCritic

# Initialize components
decomposer = MilestoneDecomposer.from_pretrained("pamc/milestone-decomposer")
pattern_analyzer = TrajectoryPatternAnalyzer()
critic = PatternAwareMilestoneCritic(pattern_analyzer)

# Build agent with PAMC critic
agent = PAMCAgent(
    llm="gpt-4-turbo",
    decomposer=decomposer,
    critic=critic,
    max_steps=50
)

# Run a task
task = "Analyze the contract clause 4.2b for termination risks and propose amendments"
result = agent.run(task)

print(f"Completed in {result.steps} steps")
print(f"Pattern Risk Score: {result.final_prs:.2f}")
print(f"Milestones verified: {result.milestones_verified}/{result.milestones_total}")
```

---

## Roadmap

- [x] **Step 1:** Project scaffolding and milestone data structures
- [ ] **Step 2:** Trajectory instrumentation layer — log each reasoning step with token-level metadata
- [ ] **Step 3:** Implement implicit pattern detectors (Early Pruning, Path Lock-in, Targeted Backtracking, Knowledge-Guided Drift)
- [ ] **Step 4:** Build Pattern Risk Score (PRS) aggregator
- [ ] **Step 5:** Integrate OS-Themis milestone decomposer
- [ ] **Step 6:** Implement Pattern-Aware Milestone Critic with three-tier response
- [ ] **Step 7:** Add alternative path suggestion module
- [ ] **Step 8:** Implement backtrack signal generation for high-risk trajectories
- [ ] **Step 9:** Build evidence chain auditor (invoked only for surviving trajectories)
- [ ] **Step 10:** RL training loop integration (PPO)
- [ ] **Step 11:** Benchmark evaluation on AndroidWorld and code-generation tasks
- [ ] **Step 12:** Ablation study — measure contribution of each pattern detector

---

## References

1. **OS-Themis** — Multi-agent critic with milestone decomposition and evidence chain auditing  
   *arXiv:2603.19191*  

2. **Implicit Patterns in LLM Binary Analysis** — Four dominant reasoning failure patterns  
   *arXiv:2603.19138*  

3. **Box Maze** — Process-control architecture with boundary enforcement layers  
   *arXiv:2603.19182*  

4. **Uncertainty Estimation Scaling** — Hybrid 2-sample AUROC estimator  
   *arXiv:2603.19118*  

5. **Tree-of-Thought Reasoning** — Explicit search tree reasoning for LLM agents  

6. **PPO / RLHF** — Reinforcement learning from human feedback for reward modeling
