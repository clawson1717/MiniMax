# RAPTOR — Reasoning quality Assessment via Polling, Trajectory, and Orchestration for Reliability

> ⚠️ **Status: Step 2 Complete (Entropy Trajectory Tracker)**
> Full implementation is tracked in `project-raptor-plan.md`.

---

## What Is RAPTOR?

RAPTOR is a **real-time reasoning quality controller** for LLM agents that combines three orthogonal techniques from recent ArXiv papers into a unified feedback loop:

1. **Entropy Trajectory Monitoring** — detects whether a chain-of-thought reasoning chain is *monotone* (entropy decreases at every step), which strongly predicts correctness (+21.9pp per arXiv:2603.18940)
2. **Multi-Agent Disagreement Extraction** — extracts DiscoUQ-style linguistic and embedding geometry signals from agent ensembles to produce calibrated confidence estimates (AUROC 0.802 per arXiv:2603.20975)
3. **Utility-Guided Orchestration** — uses the fused signals to select the next agent action (respond, reroll, verify, escalate, retrieve, stop) by maximizing expected utility (per arXiv:2603.19896)

### The Core Insight

Entropy monotonicity and disagreement entropy are **complementary**:
- Monotonicity measures the *internal coherence* of a single reasoning chain
- Disagreement measures the *external diversity* across agents

When both signals agree (non-monotone trajectory + high disagreement), that's a far stronger reliability alarm than either signal alone. RAPTOR is the first system to combine these signals in real-time.

## Why RAPTOR?

Existing LLM agent frameworks make reasoning decisions in isolation:

| System | Trajectory Shape? | Multi-Agent Disagreement? | Utility-Guided? |
|--------|------------------|--------------------------|----------------|
| Naive prompting | ✗ | ✗ | ✗ |
| Self-consistency (40-chain) | ✗ | ✓ (vote only) | ✗ |
| DiscoUQ | ✗ | ✓ (post-hoc) | ✗ |
| Chain-of-thought + verify | ✓ (heuristic) | ✗ | ✗ |
| **RAPTOR** | **✓ (real-time)** | **✓ (real-time)** | **✓** |

## Installation

```bash
pip install raptor-llm  # TODO: publish to PyPI
```

Or from source:

```bash
git clone https://github.com/YOUR_HANDLE/raptor.git
cd raptor
pip install -e .
```

## Quick Start

```python
from raptor import RAPTOROrchestrator, Config

config = Config(
    monotonicity_threshold=1.0,   # strict: entropy must decrease every step
    disagreement_weight=0.5,
    entropy_weight=0.5,
    max_rerolls=3,
)

orchestrator = RAPTOROrchestrator(config)

# Multi-agent polling
from raptor.agents import poll_agents

agents = [agent1, agent2, agent3, agent4, agent5]
responses = poll_agents(agents, prompt="Solve: 2x + 3 = 7")

# Orchestrate
action = orchestrator.step(prompt, responses)
# action = OrchestrationAction.REROLL  # or RESPOND, VERIFY, ESCALATE, RETRIEVE, STOP

if action == OrchestrationAction.REROLL:
    # Trigger reroll with candidate selection
    best_response = orchestrator.reroll_with_selection(agents, prompt, n_candidates=3)
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      RAPTOR Controller                       │
│                                                             │
│  Entropy Trajectory Tracker   Disagreement Monitor          │
│  [arXiv:2603.18940]           [arXiv:2603.20975]            │
│         │                           │                        │
│         └─────────────┬─────────────┘                        │
│                       ▼                                      │
│              Signal Vector S(t)                              │
│  [monotonicity, slope, disagreement, dispersion, cohesion]   │
│                       │                                      │
│                       ▼                                      │
│            Utility Score Engine                              │
│       U(a|S) = Σ w_k·φ_k(a,S) - cost(a)                     │
│                       │                                      │
│                       ▼                                      │
│         Action: respond | reroll | verify |                  │
│                 escalate | retrieve | stop                   │
└─────────────────────────────────────────────────────────────┘
```

## Signal Vector

At each reasoning step `t`, RAPTOR computes:

```
S(t) = [monotonicity_flag(t),   # 1 if entropy↓ every step, else 0
        entropy_slope(t),      # linear slope of entropy trajectory
        disagreement_score(t), # calibrated confidence from DiscoUQ
        dispersion_score(t),   # embedding cluster spread
        cohesion_score(t),     # intra-cluster closeness
        divergence_depth(t)]   # how deep agents diverged
```

## Actions

| Action | When Triggered | Effect |
|--------|----------------|--------|
| `RESPOND` | High confidence (both signals agree) | Return current answer |
| `REROLL` | Non-monotone + high disagreement | Regenerate reasoning chain |
| `VERIFY` | Non-monotone + moderate disagreement | Run verification step |
| `ESCALATE` | Very low confidence | Switch to stronger model |
| `RETRIEVE` | Missing evidence signals | Trigger RAG retrieval |
| `STOP` | Abort signal from any monitor | Terminate reasoning |

## Project Structure

```
raptor/
├── src/raptor/
│   ├── __init__.py
│   ├── config.py          # Config dataclasses, thresholds, weights
│   ├── entropy_tracker.py # Entropy trajectory monitoring (Step 2)
│   ├── disagreement.py    # DiscoUQ-style disagreement (Step 3)
│   ├── utility.py         # Utility score engine (Step 4)
│   ├── orchestrator.py    # Main RAPTOR controller (Step 5)
│   ├── agents.py          # Agent protocol & polling (Step 6)
│   └── logging.py         # Signal history logging
├── tests/
│   ├── test_entropy_tracker.py
│   ├── test_disagreement.py
│   ├── test_utility.py
│   └── test_orchestrator.py
├── pyproject.toml
└── README.md
```

## Papers This Combines

1. **DiscoUQ** — Structured Disagreement Analysis for Uncertainty Quantification in LLM Agent Ensembles (arXiv:2603.20975)
2. **Entropy Trajectory Shape** — Entropy Trajectory Shape Predicts LLM Reasoning Reliability (arXiv:2603.18940)
3. **Utility-Guided Orchestration** — Utility-Guided Agent Orchestration for Efficient LLM Tool Use (arXiv:2603.19896)

## Status

| Step | Description | Status |
|------|-------------|--------|
| 1 | Scaffold + Plan | ✅ DONE |
| 2 | Entropy Trajectory Tracker | 🔲 |
| 3 | Disagreement Monitor (DiscoUQ-Style) | 🔲 |
| 4 | Utility Score Engine | 🔲 |
| 5 | RAPTOR Orchestrator (Core Loop) | 🔲 |
| 6 | Agent Protocol & Polling | 🔲 |
| 7 | Dashboard / Visualization | 🔲 |
| 8 | Integration Utilities | 🔲 |
| 9 | Experiments & Evaluation | 🔲 |
| 10 | Documentation & Publishing | 🔲 |

## License

MIT
