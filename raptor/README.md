# RAPTOR — Reasoning quality Assessment via Polling, Trajectory, and Orchestration for Reliability

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Status: Alpha](https://img.shields.io/badge/status-alpha-orange.svg)]()

**RAPTOR** is a real-time reasoning quality controller for LLM agents that fuses three complementary uncertainty signals into a unified orchestration loop. It monitors entropy trajectory shape, multi-agent disagreement structure, and utility-guided action selection to decide — in real time — whether to respond, reroll, verify, escalate, retrieve context, or stop.

## Motivation

Three recent research papers independently advanced LLM reasoning reliability:

1. **Entropy Trajectory Shape** ([arXiv:2603.18940](https://arxiv.org/abs/2603.18940)) — Monotonically decreasing per-step entropy predicts correctness (+21.9pp accuracy). But only as a *passive* signal.
2. **DiscoUQ** ([arXiv:2603.20975](https://arxiv.org/abs/2603.20975)) — Structured disagreement analysis across agent ensembles yields calibrated confidence (AUROC 0.802, ECE 0.036). But only *post-hoc*.
3. **Utility-Guided Orchestration** ([arXiv:2603.19896](https://arxiv.org/abs/2603.19896)) — Lightweight utility functions select agent actions efficiently. But relies on *ad-hoc* uncertainty estimates.

**RAPTOR is the first system to close the loop:** combining entropy trajectory monitoring, multi-agent disagreement extraction, and utility-guided orchestration into a unified real-time controller.

### The Core Insight

Entropy monotonicity and disagreement entropy are **complementary**:
- **Monotonicity** measures the *internal coherence* of a single reasoning chain
- **Disagreement** measures the *external diversity* across agents

When both signals agree (non-monotone trajectory + high disagreement), that's a far stronger reliability alarm than either alone.

| System | Trajectory Shape? | Multi-Agent Disagreement? | Utility-Guided? |
|--------|:-:|:-:|:-:|
| Naive prompting | ✗ | ✗ | ✗ |
| Self-consistency (40-chain) | ✗ | ✓ (vote only) | ✗ |
| DiscoUQ | ✗ | ✓ (post-hoc) | ✗ |
| Chain-of-thought + verify | ✓ (heuristic) | ✗ | ✗ |
| **RAPTOR** | **✓ (real-time)** | **✓ (real-time)** | **✓** |

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                         RAPTOR Controller                            │
│                                                                      │
│  ┌───────────────────┐     ┌─────────────────────┐                   │
│  │  Entropy Trajectory│     │  Disagreement Monitor│                  │
│  │  Tracker           │     │  (DiscoUQ-Style)     │                  │
│  │  [arXiv:2603.18940]│     │  [arXiv:2603.20975]  │                  │
│  └────────┬──────────┘     └──────────┬──────────┘                   │
│           │                           │                              │
│           └───────────┬───────────────┘                              │
│                       ▼                                              │
│              Signal Vector S(t)                                      │
│  [monotonicity, slope, disagreement, dispersion, cohesion, depth]    │
│                       │                                              │
│                       ▼                                              │
│            Utility Score Engine  [arXiv:2603.19896]                   │
│       U(a|S) = Σ wₖ·φₖ(a,S) - cost(a)                              │
│                       │                                              │
│                       ▼                                              │
│     ┌──────┬──────┬──────┬──────────┬──────────┬──────┐              │
│     │RESPOND│REROLL│VERIFY│ESCALATE  │RETRIEVE  │ STOP │             │
│     └──────┴──────┴──────┴──────────┴──────────┴──────┘              │
└──────────────────────────────────────────────────────────────────────┘
```

**Signal Vector** — At each reasoning step `t`, RAPTOR computes:
```
S(t) = [monotonicity_flag, entropy_slope, disagreement_score,
        dispersion_score, cohesion_score, divergence_depth]
```

**Actions:**

| Action | Trigger | Effect |
|--------|---------|--------|
| `RESPOND` | High confidence (both signals agree) | Return current answer |
| `REROLL` | Non-monotone + high disagreement | Regenerate reasoning chain |
| `VERIFY` | Non-monotone + moderate disagreement | Run verification step |
| `ESCALATE` | Very low confidence | Switch to stronger model / human |
| `RETRIEVE` | Missing evidence signals | Trigger RAG retrieval |
| `STOP` | Abort signal from any monitor | Terminate reasoning |

## Installation

```bash
pip install raptor-llm
```

Or from source:

```bash
git clone https://github.com/YOUR_HANDLE/raptor.git
cd raptor
pip install -e ".[dev]"
```

**Optional LLM provider dependencies:**

```bash
pip install raptor-llm[openai]     # OpenAI adapter
pip install raptor-llm[anthropic]  # Anthropic adapter
pip install raptor-llm[llm]        # Both providers
```

## Quick Start

### Basic Usage with Mock Agents

```python
from raptor import Config, RAPTOROrchestrator
from raptor.agents import MockReasoningAgent, poll_agents
from raptor.disagreement import AgentResponse

# Create agents (in production, use OpenAIAgent or AnthropicAgent)
agents = [
    MockReasoningAgent(
        agent_id=f"agent-{i}",
        reasoning_steps=["Step 1: Parse equation", "Step 2: Solve for x"],
        final_answer="x = 2",
    )
    for i in range(5)
]

# Configure and create orchestrator
config = Config(max_rerolls=3, max_steps=10)
orchestrator = RAPTOROrchestrator(config)

# Poll agents and get orchestration decision
result = poll_agents(agents, "Solve: 2x + 3 = 7")
decision = orchestrator.step("Solve: 2x + 3 = 7", result.responses)

print(f"Action: {decision.action.value}")
print(f"Utility: {decision.utility_score:.3f}")
print(f"Reason: {decision.reason}")
```

### High-Level Entry Point

```python
from raptor.integration import run_with_raptor
from raptor.agents import MockReasoningAgent
from raptor import Config

agents = [
    MockReasoningAgent(f"agent-{i}", ["Think step by step..."], "42")
    for i in range(5)
]

result = run_with_raptor(
    agents=agents,
    prompt="What is the answer to life, the universe, and everything?",
    config=Config(max_steps=5),
)

print(f"Answer: {result.final_answer}")
print(f"Steps: {result.steps_taken}")
print(f"Final action: {result.final_action.value}")
```

### Using Real LLM Providers

```python
from raptor.integration import run_with_raptor, OpenAIAgent, AnthropicAgent
from raptor import Config

agents = [
    OpenAIAgent(model="gpt-4o", api_key="sk-...", agent_id="gpt4-1"),
    OpenAIAgent(model="gpt-4o", api_key="sk-...", agent_id="gpt4-2"),
    AnthropicAgent(model="claude-sonnet-4-20250514", api_key="sk-ant-...", agent_id="claude-1"),
]

result = run_with_raptor(
    agents=agents,
    prompt="Prove that √2 is irrational.",
    config=Config(max_rerolls=2, max_steps=8),
)
```

### Dashboard (Terminal Visualizer)

Replay a RAPTOR session log in the terminal with Rich-formatted signal panels:

```bash
# Replay a session step-by-step
raptor-dashboard replay raptor_logs/session_20260406_120000_000000.jsonl

# Show a specific step
raptor-dashboard step raptor_logs/session_20260406_120000_000000.jsonl --step 3

# Compact summary table
raptor-dashboard summary raptor_logs/session_20260406_120000_000000.jsonl

# List available logs
raptor-dashboard list raptor_logs/
```

### Running Experiments

```python
from raptor.experiments import (
    generate_gsm8k_synthetic,
    run_experiment,
    compute_metrics,
    ExperimentConfig,
    BaselineMode,
)
from raptor.agents import MockReasoningAgent

tasks = generate_gsm8k_synthetic(50)
agents = [MockReasoningAgent(f"a{i}", ["reasoning"], "42") for i in range(5)]

config = ExperimentConfig(
    baseline_mode=BaselineMode.RAPTOR_FULL,
    n_samples=50,
)

results = run_experiment(config, tasks, agents)
metrics = compute_metrics(results)
print(f"Accuracy: {metrics.accuracy:.2%}, ECE: {metrics.ece:.4f}")
```

## API Overview

### Core Classes

| Class | Module | Description |
|-------|--------|-------------|
| `Config` | `raptor.config` | Top-level configuration (entropy, disagreement, utility sub-configs) |
| `RAPTOROrchestrator` | `raptor.orchestrator` | Main controller — fuses signals, selects actions |
| `EntropyTracker` | `raptor.entropy_tracker` | Monitors per-step entropy trajectory shape |
| `DisagreementMonitor` | `raptor.disagreement` | DiscoUQ-style multi-agent disagreement extraction |
| `UtilityEngine` | `raptor.utility` | Computes utility scores for all 6 actions |

### Agent Protocols

| Protocol/Class | Module | Description |
|-------|--------|-------------|
| `ReasoningAgent` | `raptor.agents` | Abstract sync agent protocol (`generate(prompt) → AgentResponse`) |
| `StreamingReasoningAgent` | `raptor.agents` | Streaming variant (yields `AgentStreamEvent` tokens) |
| `OpenAIAgent` | `raptor.integration` | OpenAI Chat Completions adapter |
| `AnthropicAgent` | `raptor.integration` | Anthropic Messages API adapter |
| `LocalLLMAgent` | `raptor.integration` | Local OpenAI-compatible endpoint adapter |

### Key Functions

| Function | Module | Description |
|----------|--------|-------------|
| `run_with_raptor()` | `raptor.integration` | High-level entry point — full RAPTOR pipeline |
| `poll_agents()` | `raptor.agents` | Concurrent multi-agent polling |
| `reroll()` | `raptor.agents` | Generate multiple candidates from one agent |
| `run_experiment()` | `raptor.experiments` | Evaluation harness over benchmark tasks |
| `compute_metrics()` | `raptor.experiments` | Accuracy, ECE, reroll/step/cost stats |

### Data Structures

| Class | Module | Description |
|-------|--------|-------------|
| `TrajectorySignal` | `raptor.entropy_tracker` | Entropy trajectory metrics per step |
| `DisagreementSignal` | `raptor.disagreement` | Disagreement features + calibrated confidence |
| `AgentResponse` | `raptor.disagreement` | Agent output (reasoning steps + final answer) |
| `SignalVector` | `raptor.orchestrator` | Fused signal vector S(t) |
| `OrchestrationDecision` | `raptor.orchestrator` | Full decision with signals, scores, and reason |
| `RAPTORContext` | `raptor.integration` | Serializable session context |
| `RAPTORResult` | `raptor.integration` | Return type from `run_with_raptor()` |
| `OrchestrationAction` | `raptor.config` | Enum: RESPOND, REROLL, VERIFY, ESCALATE, RETRIEVE, STOP |

## Configuration Reference

RAPTOR is configured via nested dataclasses. See [docs/configuration.md](docs/configuration.md) for the full reference.

```python
from raptor import Config
from raptor.config import EntropyConfig, DisagreementConfig, UtilityConfig

config = Config(
    # Entropy tracking
    entropy=EntropyConfig(
        monotonicity_threshold=1.0,  # Strict: every step must decrease
        slope_window=3,
        token_level=True,
    ),
    # Disagreement monitoring
    disagreement=DisagreementConfig(
        mode="embed",          # "llm" | "embed" | "learn"
        n_agents=5,
        use_linguistic_features=True,
        use_embedding_features=True,
    ),
    # Utility-guided action selection
    utility=UtilityConfig(
        weights={"gain": 0.30, "confidence": 0.25, "cost_penalty": -0.15,
                 "redundancy_penalty": -0.10, "severity": 0.20},
        switch_threshold=0.05,
        hysteresis_steps=1,
    ),
    # Orchestration
    max_rerolls=3,
    max_steps=20,
    log_signal_history=True,
    log_dir="raptor_logs",
)
```

## Project Structure

```
raptor/
├── src/raptor/
│   ├── __init__.py           # Package exports
│   ├── config.py             # Config dataclasses, thresholds, weights
│   ├── entropy_tracker.py    # Entropy trajectory monitoring (arXiv:2603.18940)
│   ├── disagreement.py       # DiscoUQ-style disagreement (arXiv:2603.20975)
│   ├── utility.py            # Utility score engine (arXiv:2603.19896)
│   ├── orchestrator.py       # Main RAPTOR controller loop
│   ├── agents.py             # Agent protocols, polling, reroll
│   ├── dashboard.py          # Rich terminal dashboard + CLI
│   ├── integration.py        # LLM adapters + run_with_raptor()
│   └── experiments.py        # Evaluation harness + synthetic benchmarks
├── tests/                    # 533+ tests
├── docs/                     # Documentation
│   ├── api.md                # Full API reference
│   ├── architecture.md       # Detailed architecture explanation
│   ├── tutorial.md           # Step-by-step tutorial
│   ├── configuration.md      # Configuration reference
│   └── experiments.md        # Running experiments guide
├── examples/                 # Example scripts
│   ├── basic_usage.py        # Simple mock example
│   ├── custom_agent.py       # Implementing a custom ReasoningAgent
│   └── run_experiment.py     # Running experiments with synthetic data
├── pyproject.toml
├── LICENSE
└── README.md
```

## Papers

1. **DiscoUQ** — Structured Disagreement Analysis for Uncertainty Quantification in LLM Agent Ensembles ([arXiv:2603.20975](https://arxiv.org/abs/2603.20975))
2. **Entropy Trajectory Shape** — Entropy Trajectory Shape Predicts LLM Reasoning Reliability ([arXiv:2603.18940](https://arxiv.org/abs/2603.18940))
3. **Utility-Guided Orchestration** — Utility-Guided Agent Orchestration for Efficient LLM Tool Use ([arXiv:2603.19896](https://arxiv.org/abs/2603.19896))

## Contributing

Contributions are welcome! To get started:

```bash
git clone https://github.com/YOUR_HANDLE/raptor.git
cd raptor
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Run tests
python -m pytest tests/ -v

# Run linter
ruff check src/ tests/

# Run type checker
mypy src/raptor/
```

**Guidelines:**
- All new features must include tests
- Maintain 100% pass rate on existing tests
- Follow existing code style (ruff-formatted, type-annotated)
- Update documentation for public API changes

## License

MIT — see [LICENSE](LICENSE) for details.
