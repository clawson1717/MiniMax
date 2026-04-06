# Forecast-Guided Effort Allocation (FGEA)

**A closed-loop reasoning agent that uses trajectory stability — the consistency of forecasted futures — to decide how hard to think at each step.**

---

## Problem Statement

Existing agents use static reasoning effort (always think hard) or react to failures only after-the-fact (think harder when a step fails). Neither approach is anticipatory: both waste compute on trivial steps that require no deep reasoning, and both miss the opportunity to *predict* when a situation is about to become difficult. Forecast-Guided Effort Allocation (FGEA) closes this loop by forecasting multiple candidate futures before each step and using the stability of those forecasts — how consistently they converge on similar outcomes — as a real-time signal for effort allocation.

## Core Innovation

**Trajectory stability as effort allocation signal.** Drawing on TraceR1's anticipatory trajectory forecasting and Ares's per-step effort routing, FGEA introduces a closed feedback loop: the forecaster generates K candidate future trajectories; a stability scorer measures how much those trajectories diverge; the effort router maps the stability score to LOW / MEDIUM / HIGH reasoning effort before committing to an action. When futures are stable, FGEA cuts tokens dramatically. When futures are unstable, FGEA switches to deep reasoning.

## Architecture Overview

FGEA operates as a four-stage closed loop at each reasoning step: (1) the **TrajectoryForecaster** generates K candidate future sequences from the current state using a TraceR1-style anticipatory model; (2) the **StabilityScorer** measures pairwise divergence across all K trajectories and produces a stability score in [0, 1]; (3) the lightweight **EffortRouter** — trained in the Ares style — maps the stability score (and state features) to an effort level; (4) the **EffortCalibratedEngine** executes reasoning at the chosen effort level (direct answer for LOW, standard CoT for MEDIUM, deep CoT + self-consistency for HIGH). The loop repeats until the task is complete.

## Quick Start

```python
from fgea import FGEAConfig, FGEAAgent

config = FGEAConfig(
    forecast_horizon=5,
    num_candidates=8,
    stability_thresholds=(0.3, 0.7),  # (τ_LOW, τ_HIGH)
)
agent = FGEAAgent(config)

response = agent.run(
    "If I flip three fair coins, what is the probability of getting exactly 2 heads?"
)
print(response.output)
print(f"Effort levels used: {response.effort_trace}")   # e.g. [LOW, LOW, MEDIUM]
print(f"Total tokens: {response.total_tokens}")
```

## Installation

```bash
pip install fgea
```

Or install from source:

```bash
git clone https://github.com/your-org/forecast-guided-effort-allocation.git
cd forecast-guided-effort-allocation
pip install -e .
```

## Roadmap

| Step | Description | Status |
|------|-------------|--------|
| 1 | Project scaffold — directory structure, requirements, config schema | 🔲 Not started |
| 2 | State and Trajectory data models | 🔲 Not started |
| 3 | Trajectory Forecaster (TraceR1-style K-candidate forecasting) | 🔲 Not started |
| 4 | Stability Scorer (pairwise divergence, consensus score, reward variance) | 🔲 Not started |
| 5 | Effort Router (lightweight Ares-style classifier) | 🔲 Not started |
| 6 | Effort-Calibrated Reasoning Engine (LOW/MEDIUM/HIGH executors) | 🔲 Not started |
| 7 | Closed-Loop Controller (orchestrates forecast→score→route→execute) | 🔲 Not started |
| 8 | Training Pipeline (Ares-style data generation + router training) | 🔲 Not started |
| 9 | Benchmark Evaluation (vs. static CoT, Ares, TraceR1 baselines) | 🔲 Not started |
| 10 | CLI Interface (run, benchmark, visualize) | 🔲 Not started |
| 11 | Integration Layer (FGEAAgent drop-in interface) | 🔲 Not started |
| 12 | Documentation & README | 🔲 Not started |

## Key Classes

| Class | Module | Purpose |
|-------|--------|---------|
| `FGEAConfig` | `fgea.config` | Typed configuration for all components |
| `State` | `fgea.models.state` | Immutable snapshot of agent state |
| `Trajectory` | `fgea.models.state` | A single forecasted future sequence |
| `TrajectoryForecaster` | `fgea.forecaster` | Generates K candidate futures from current state |
| `StabilityScorer` | `fgea.scorer` | Measures trajectory stability; returns `StabilityResult` |
| `EffortRouter` | `fgea.router` | Lightweight classifier: stability → {LOW, MEDIUM, HIGH} |
| `EffortCalibratedEngine` | `fgea.engine` | Executes reasoning at designated effort level |
| `ClosedLoopController` | `fgea.controller` | Orchestrates full FGEA loop |
| `FGEAAgent` | `fgea.agent` | High-level unified interface |

## Effort Levels

| Level | Reasoning Behavior | Typical Token Usage |
|-------|-------------------|---------------------|
| **LOW** | Direct response, no CoT | ~50–200 tokens |
| **MEDIUM** | Standard chain-of-thought, 1–2 passes | ~300–800 tokens |
| **HIGH** | Deep CoT + self-consistency (N samples) + verification | ~1500–5000 tokens |

## Dependencies

- Python ≥ 3.10
- `numpy`, `scipy`
- `pydantic`
- `torch`
- `transformers`
- `pytest`
- `tqdm`

## References

- **TraceR1** — Anticipatory Trajectory Reasoning via 2-Stage RL (arXiv:2603.16777)
- **Ares** — Per-Step Adaptive Reasoning Effort Selection (arXiv:2603.07915)

## License

MIT
