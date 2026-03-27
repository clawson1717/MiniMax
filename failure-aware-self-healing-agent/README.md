# Failure-Aware Self-Healing Agent (FASHA)

An agent framework that doesn't just detect and recover from reasoning failures—it *diagnoses* them. When a failure occurs, the system performs root-cause analysis to understand *why* it happened, selects an appropriate recovery strategy based on the diagnosis, and learns from the episode to prevent similar failures in the future.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     SelfHealingLoop                             │
│  ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐        │
│  │ DETECT  │──▶│ DIAGNOSE │──▶│ RECOVER │──▶│ VERIFY  │        │
│  └─────────┘   └─────────┘   └─────────┘   └─────────┘        │
│       │             │             │             │               │
│       ▼             ▼             ▼             ▼               │
│  ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐        │
│  │Confidence│  │Diagnostic│  │Recovery │  │Outcome  │        │
│  │ Tracker │  │  Engine  │  │ Strategy│  │ Check   │        │
│  └─────────┘   └─────────┘   └─────────┘   └─────────┘        │
│                      │                                         │
│                      ▼                                         │
│               ┌─────────────┐                                  │
│               │   Failure   │                                  │
│               │   Pattern   │                                  │
│               │   Memory    │                                  │
│               └─────────────┘                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Core Components

| Component | File | Purpose |
|-----------|------|---------|
| `FailureEvent` | `src/failure_event.py` | Dataclass for recording failure episodes |
| `DiagnosticEngine` | `src/diagnostic.py` | Root-cause classification |
| `RecoveryStrategy` | `src/recovery.py` | Strategy selection and execution |
| `SelfHealingLoop` | `src/healing_loop.py` | Main orchestrator |
| `FailurePatternMemory` | `src/pattern_memory.py` | Historical learning |
| `ConfidenceTracker` | `src/confidence.py` | Early warning system |
| `MockAgent` | `src/mock_agent.py` | Test interface |

## Root-Cause Categories

1. **Knowledge Gap** — Missing information needed to proceed
2. **Reasoning Error** — Logical flaw in thought process
3. **Context Loss** — Forgot important prior information
4. **Strategy Mismatch** — Wrong approach for the task type
5. **External Failure** — Environment or tool failure

## Recovery Strategies

| Strategy | When to Use |
|----------|-------------|
| `ReRead` | Context loss, surface errors |
| `AskClarification` | Knowledge gaps, ambiguous requirements |
| `Backtrack` | Reasoning errors, wrong strategy |
| `RetryDifferentApproach` | Strategy mismatch |
| `Simplify` | Complexity overwhelm |
| `Escalate` | Non-recoverable failures |

## Implementation Roadmap

- [x] Step 1: Project Scaffold (DONE)
- [x] Step 2: Failure Event Record (DONE)
- [ ] Step 3: Diagnostic Engine
- [ ] Step 4: Recovery Strategy Library
- [ ] Step 5: Self-Healing Loop
- [ ] Step 6: Failure Pattern Memory
- [ ] Step 7: Confidence Tracker
- [ ] Step 8: Mock Agent Interface
- [ ] Step 9: CLI Interface
- [ ] Step 10: Benchmark Suite
- [ ] Step 11: Integration Tests
- [ ] Step 12: Documentation & Demo

## Installation

```bash
cd ClawWork/failure-aware-self-healing-agent
pip install -e .
```

## Usage

### CLI

```bash
# Monitor an agent
python -m src.cli monitor

# Diagnose a failure log
python -m src.cli diagnose --event <json_file>

# View failure patterns
python -m src.cli history --limit 50

# Run benchmarks
python -m src.cli benchmark
```

### Python API

```python
from src.healing_loop import SelfHealingLoop
from src.mock_agent import MockAgent

agent = MockAgent()
loop = SelfHealingLoop(agent)

# Run until healthy or unrecoverable
result = loop.run(max_steps=100)
print(f"Final state: {result['state']}")
```

## Comparison with Related Work

| Project | Focus | FASHA Advantage |
|---------|-------|-----------------|
| ATR | Detection + Pruning | Adds diagnosis + strategy selection |
| Flow-Heal | Recovery | Adds root-cause understanding |
| Watchdog | Monitoring | Adds actionable recovery + learning |

## Status

🚧 Implementation in progress — see `memory/project-failure-aware-self-healing-agent-plan.md`
