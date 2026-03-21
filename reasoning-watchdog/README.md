# Reasoning Watchdog — Process-Verified Multi-Agent Reasoning

A lightweight multi-agent reasoning wrapper that verifies each reasoning step before committing to it. Combines Box Maze's process-control layers, OS-Themis-style milestone critics, and hybrid uncertainty signals for self-correcting, drift-aware reasoning.

## Installation

```bash
pip install -e .
```

## Quick Start

```python
from src.watchdog import WatchdogOrchestrator

orchestrator = WatchdogOrchestrator()
result = orchestrator.run("Explain why the sky is blue")
print(result)
```

## Architecture

The watchdog system combines four process-control layers:

1. **Boundary Enforcement** — Keeps reasoning within scope
2. **Milestone Critics** — Decomposes and verifies checkpoints
3. **Uncertainty Monitor** — Detects when confidence is low
4. **Memory Grounding** — Prevents hallucinated premises

## CLI

```bash
# Verify a reasoning chain
python -m src.cli verify "Explain quantum entanglement"

# Monitor with detailed trace
python -m src.cli monitor "Solve this math problem"

# Multi-agent team mode
python -m src.cli team "Compare two theories"
```

## Roadmap

- [ ] **Step 1:** Project Scaffold
- [ ] **Step 2:** Boundary Enforcement Layer
- [ ] **Step 3:** Milestone Critic
- [ ] **Step 4:** Uncertainty Monitor
- [ ] **Step 5:** Memory Grounding Layer
- [ ] **Step 6:** Process-Control Integrator
- [ ] **Step 7:** Multi-Agent Watchdog
- [ ] **Step 8:** Backtracking Controller
- [ ] **Step 9:** CLI Interface
- [ ] **Step 10:** Benchmark Suite
- [ ] **Step 11:** Visualization
- [ ] **Step 12:** Documentation & Demo
