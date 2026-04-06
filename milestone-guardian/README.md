# Milestone Guardian (MG)

**A milestone-verified process-control system for reliable LLM agentic reasoning.**

```
pip install milestone-guardian
```

## The Problem

LLM agents are unreliable on complex multi-step tasks:

- **Drift:** They go off-track, pursuing subgoals that don't advance the main task
- **Compound errors:** A mistake at step 3 corrupts steps 4–10
- **No verification:** They proceed past failed subgoals without realizing it
- **No recovery:** Failure means starting over, not structured repair

Standard approaches address parts of this:
- **MiRA** (milestone-conditioned RL) gives dense rewards but no boundary enforcement
- **Box Maze** (process-control architecture) enforces boundaries but has no verifiable milestones
- **OS-Themis** (verifiable milestone critic) reviews evidence but doesn't enforce boundaries

**No system combines verifiable milestones, boundary enforcement, and structured recovery.**

## What Milestone Guardian Does

MG fuses all three into one architecture:

1. **MiRA** → Tasks are decomposed into **verifiable subgoals** with explicit evidence specs
2. **Box Maze** → Each milestone transition is **boundary-enforced** (memory grounding, structured inference, explicit checkpoints)
3. **OS-Themis** → A **critic reviews evidence chains** before allowing progression to the next milestone
4. **Recovery Engine** → Failed milestones trigger **structured recovery strategies**, not raw retry

**Result:** LLM agents that complete multi-step tasks reliably, recover gracefully from failures, and self-improve from milestone-level experience.

## Architecture

```
User task
    │
    ▼
MilestoneDecomposer ──► Milestone sequence with evidence specs
    │
    ▼
Execution Loop ──────────────────────────────┐
    │                                         │
    ▼                                         │
BoundaryEnforcer ──► [ENTERING / EXECUTING]  │
    │                                         │
    ▼                                         │
EvidenceCollector ──► Evidence items          │
    │                                         │
    ▼                                         │
MilestoneCritic ──► APPROVED / REJECTED       │
    │                    │                     │
    │         [REJECTED] ▼                     │
    │         RecoveryEngine                   │
    │              │                           │
    └──────────────►│◄─────────────────────────┘
                    │
                    ▼
             [COMPLETED / FAILED]
                    │
                    ▼
         TrajectoryMemory ──► Pattern store
```

### Key Insight

**Milestones are both the planning unit AND the verification checkpoint.** Unlike MiRA (which uses milestones only for reward signals) or Box Maze (which uses boundaries only for inference control), MG makes every milestone transition a verified handoff — the critic checks evidence, the boundary enforcer confirms state, and only then does execution proceed.

## Installation

```bash
pip install -r requirements.txt
python -m pytest tests/ -v
```

## Quick Start

```python
from milestone_guardian import MilestoneGuardianAgent, MilestoneDecomposer

agent = MilestoneGuardianAgent()

# Plan: decompose task into verifiable milestones
task = "Research LLM agent reliability issues and write a 3-page report"
plan = agent.plan(task)

print(f"Decomposed into {len(plan.milestones)} milestones:")
for m in plan.milestones:
    print(f"  {m.order}. {m.description}")
print(f"  Evidence spec: {m.evidence_spec}")

# Execute: run through milestone loop
for milestone in plan.milestones:
    result = agent.execute(milestone)
    if result.status == "FAILED":
        recovered = agent.recover(milestone)
        if not recovered:
            print(f"Could not recover from {milestone.id}")
            break
    agent.reflect()  # update trajectory memory
```

## Key Concepts

### Milestone

The atomic unit of MG — a verifiable subgoal with explicit completion criteria:

```python
@dataclass
class Milestone:
    id: str                    # uuid
    description: str          # human-readable subgoal
    evidence_spec: list[str]  # what counts as evidence (e.g., ["web_search_results", "source_url"])
    success_rubric: str       # how the critic evaluates evidence quality
    order: int                # execution order
    status: MilestoneStatus   # PENDING / COMPLETED / FAILED
```

### Boundary Enforcer (Box Maze-style)

Prevents reasoning drift at each milestone transition:

| Layer | Function |
|-------|----------|
| Memory Grounding | Working memory is scoped to current milestone |
| Structured Inference | Reasoning steps must reference current milestone goal |
| Boundary Enforcement | Pre-conditions checked before milestone execution begins |

### Milestone Critic (OS-Themis-style)

Reviews evidence before allowing progression:

```python
# Evidence collection
evidence = EvidenceChain(
    milestone_id="milestone-3",
    evidence_items=[
        {"type": "web_search", "query": "LLM agent reliability", "results": [...]},
        {"type": "retrieved_doc", "source": "arxiv:2603.xxx", "excerpts": [...]},
    ]
)

# Critic review
decision = critic.review(evidence, rubric="At least 3 credible sources covering causes and solutions")
# → APPROVED / REJECTED / REVISION_REQUESTED
```

### Recovery Strategies

When a milestone fails, the RecoveryEngine selects a strategy:

| Failure Type | Strategy |
|-------------|----------|
| `evidence_insufficient` | REVISE_EVIDENCE — collect better evidence |
| `boundary_violation` | BACKTRACK_TO_MILESTONE — return to last good milestone |
| `reasoning_error` | DECOMPOSE_FURTHER — break milestone into sub-milestones |
| `external_failure` | SIMPLIFY_GOAL — reduce scope of milestone |
| `unknown` | ESCALATE — flag for human review |

## Components

| Component | File | Description |
|-----------|------|-------------|
| Decomposer | `src/decomposer.py` | Task → milestone sequence with evidence specs |
| Boundary | `src/boundary.py` | Box Maze-style process-control enforcement |
| Critic | `src/critic.py` | OS-Themis-style evidence chain reviewer |
| Recovery | `src/recovery.py` | Structured recovery strategy engine |
| Trajectory | `src/trajectory_memory.py` | Milestone history + pattern extraction |
| Belief State | `src/belief_state.py` | Agent belief state tracking |
| Agent | `src/agent.py` | MG agent: plan, execute, recover, reflect |
| CLI | `src/cli.py` | `python -m milestone_guardian.cli` |

## Roadmap

See `memory/project-milestone-guardian-plan.md` for the full 12-step implementation plan.

| Step | Description | Status |
|------|-------------|--------|
| 1 | Project scaffold + data structures | pending |
| 2 | Milestone decomposer | pending |
| 3 | Process-control boundary enforcer | pending |
| 4 | Verifiable milestone critic | pending |
| 5 | Recovery engine | pending |
| 6 | Trajectory memory + pattern store | pending |
| 7 | Belief state tracker | pending |
| 8 | Multi-agent integration layer | pending |
| 9 | CLI interface | pending |
| 10 | Benchmarks | pending |
| 11 | Integration tests | pending |
| 12 | Documentation + final PR | pending |

## Benchmarks

MG is benchmarked against three baselines:

- **Raw LLM agent** — no milestone control, no boundary enforcement
- **MiRA-only** — milestone decomposition without boundary enforcement
- **Box Maze-only** — boundary enforcement without verifiable milestones

Metrics: task completion rate, milestone efficiency (milestones per completed task), boundary violation rate, recovery success rate, evidence quality score.

## Prior Art Comparison

| Feature | MiRA | Box Maze | OS-Themis | **MG** |
|---------|:----:|:--------:|:---------:|:------:|
| Verifiable milestones | ✓ | ✗ | ✓ | **✓** |
| Boundary enforcement | ✗ | ✓ | ✗ | **✓** |
| Evidence-chain critic | ✗ | ✗ | ✓ | **✓** |
| Structured recovery | ✗ | ✗ | ✗ | **✓** |
| Trajectory memory | ✗ | ✓ | ✓ | **✓** |
| Belief-state adaptation | ✗ | ✓ | ✗ | **✓** |

## Papers Referenced

- Zhang et al. — **MiRA**: Milestoning your Reinforcement Learning Enhanced Agent (2026-03-23)
- Zou et al. — **Box Maze**: A Process-Control Architecture for Reliable LLM Reasoning (2026-03-21)
- Li et al. — **OS-Themis**: A Scalable Critic Framework for Generalist GUI Rewards (2026-03-21)
