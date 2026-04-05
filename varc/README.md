# VARC — Verifiable Agent Runtime Controller

**Runtime trajectory verification for LLM agents. Every reasoning milestone is audited before it cascades.**

---

## The Problem

LLM agents are unreliable in multi-step tasks. A single bad reasoning step — a hallucinated fact, a logical misstep, a boundary violation — can cascade into a complete task failure. Existing agent frameworks treat agent outputs as atomic: either you accept the full response or you reject it and retry. There is no runtime mechanism to verify individual reasoning milestones, prune failed steps, and enforce logical boundaries before failures propagate.

## The Solution

VARC is a process-control substrate that intercepts agent outputs at each reasoning milestone, verifies the milestone against evidence chains, and either **accepts**, **rectifies**, or **prunes** it before the next step runs.

It combines three techniques:

1. **Box Maze** (arXiv:2603.19182) — 3-layer process-control architecture: memory grounding → structured inference → boundary enforcement.
2. **OS-Themis** (arXiv:2603.19191) — Milestone decomposition with evidence chain auditing.
3. **AgentDropoutV2** (arXiv:2602.23258) — Test-time rectify-or-reject pruning.

---

## Roadmap

| Step | Status | Description |
|------|--------|-------------|
| 1 | [DONE] | Project scaffolding — core data structures, package layout, Milestone dataclass, Trajectory graph |
| 2 | PENDING | MilestoneBoundaryDetector — segment agent output stream into verifiable reasoning steps using Implicit Patterns heuristics |
| 3 | PENDING | Box Maze Layer 1: Memory Grounding Checker — conflict detection against grounded memory state |
| 4 | PENDING | Box Maze Layer 2: Structured Inference Checker — logical soundness validation per milestone |
| 5 | PENDING | Box Maze Layer 3: Boundary Enforcement Checker — safety and inference bound validation |
| 6 | PENDING | OS-Themis Evidence Chain Auditor — evidence tracing and confidence scoring per milestone |
| 7 | PENDING | AgentDropoutV2 Rectify-or-Reject Gate — recoverable vs. irreparable milestone routing + retrieval-augmented rectifier |
| 8 | PENDING | Trajectory Graph — milestone acceptance/prune history, cascade visualization, failure attribution |
| 9 | PENDING | CLI + API — command-line interface and Python API for embedding VARC in any agent framework |
| 10 | PENDING | Evaluation suite — benchmark VARC against baseline agents on cascading failure rate, task completion, and audit trail quality |

---

## Core Data Structures (Step 1)

```python
class VerificationStatus(Enum):
    PENDING, PASS, FAIL, RECTIFIED, PRUNED

@dataclass
class EvidenceItem:
    source: str
    claim: str
    confidence: float   # 0.0–1.0
    metadata: dict

@dataclass
class LayerResult:
    layer_name: str
    passed: bool
    reason: str
    confidence: float   # 0.0–1.0

@dataclass
class Milestone:
    id: str
    step_index: int
    content: str
    evidence_chain: list[EvidenceItem]
    verification_status: VerificationStatus
    layer_results: dict[str, LayerResult]
    parent_id: str | None
    children_ids: list[str]

@dataclass
class TrajectoryGraph:
    milestones: dict[str, Milestone]
    root_id: str | None
    accepted_ids: list[str]
    pruned_ids: list[str]
    rectified_ids: list[str]
    # methods: add_milestone, mark_accepted, mark_pruned, mark_rectified,
    #          get_lineage, to_dict
```

---

## Development

```bash
cd varc
python -m pytest tests/ -v
```

Requires Python 3.12+.

---

## Prior Art

| Paper | Technique |
|---|---|
| Box Maze (arXiv:2603.19182) | 3-layer process-control substrate, boundary enforcement |
| OS-Themis (arXiv:2603.19191) | Milestone decomposition, evidence chain auditing |
| AgentDropoutV2 (arXiv:2602.23258) | Test-time rectify-or-reject pruning |
| Implicit Patterns (arXiv:2603.19138) | Reasoning pattern detection for milestone segmentation |
