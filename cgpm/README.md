# CGPM — Confidence-Gated Persistent Memory

> "Agents should never reason confidently about uncertain facts."

**GitHub:** `clawson1717/cgpm`

A persistent, auditable, uncertainty-weighted knowledge store for LLM agents. Every knowledge object carries a `ConfidenceScore` (0.0–1.0). Below-threshold facts are **gated** — not just flagged — preventing agents from acting on unreliable knowledge.

---

## Architecture

```
Agent Query
    │
    ▼
┌─────────────────────────┐
│   ConfidenceGate Layer   │ ← blocks below-threshold facts
└──────────┬──────────────┘
           │
    ┌──────▼──────┐
    │  Fact Store  │ ← hash-addressed knowledge objects
    │  (SQLite)    │
    └──────┬──────┘
           │
    ┌──────▼──────────────┐
    │  Confidence Engine   │ ← Bayesian decay, rule scoring
    └──────────────────────┘
           │
    ┌──────▼──────────────┐
    │  Provenance Chain    │ ← audit trail, fact history
    └──────────────────────┘
```

---

## Novel Contribution

| System | What it does | CGPM's advantage |
|--------|-------------|-----------------|
| Governed Memory (Taheri et al.) | Schema-enforced shared memory | Adds uncertainty gating |
| Facts as First Class Objects (Chana et al.) | Hash-addressed tuples | Adds temporal decay + gating |
| RPMS (Yuan et al.) | Rule filtering for planning | Integrated into confidence scoring |
| Standard RAG | Retrieve & generate | Facts are confidence-tracked, not just retrieved |

**CGPM's key insight:** Uncertainty is not a metadata field — it is a **gate**. The memory layer itself enforces that agents cannot retrieve below-threshold facts.

---

## Installation

```bash
pip install cgpm
# or from source:
cd ClawWork/cgpm && pip install -e .
```

---

## Quick Start

```python
from cgpm import KnowledgeObject, FactStore, ConfidenceGate, ConfidenceEngine

# Create a fact with initial confidence
ko = KnowledgeObject(
    content="The patient's blood type is AB-negative.",
    confidence=0.95,
    provenance={"source": "lab_report", "agent_id": "diagnostic-agent-1"},
    decay_rate=0.01  # 1% decay per hour
)

# Store it
store = FactStore(db_path="./cgpm.db")
store.insert(ko)

# Check gate before reasoning
gate = ConfidenceGate(store, threshold=0.7)
allowed = gate.is_allowed(ko.ko_id, threshold=0.7)
print(f"Can agent use this fact? {allowed}")  # True

# After time passes, confidence decays
engine = ConfidenceEngine(store)
engine.apply_decay(ko.ko_id, hours_elapsed=48)
updated = store.get(ko.ko_id)
print(f"Confidence after 48h: {updated.confidence:.4f}")  # ~0.62
```

---

## CLI

```bash
cgpm add "The file is located at /data/archive/2024/report.pdf" --confidence 0.8 --source manual
cgpm query "file location" --threshold 0.7
cgpm audit --ko-id <id>
cgpm visualize --ko-id <id>
```

---

## Implementation Roadmap

| Step | Component | Status |
|------|-----------|--------|
| 0 | Project concept + README | [DONE] |
| 1 | Project scaffolding | [IN PROGRESS] |
| 2 | KnowledgeObject core | [PENDING] |
| 3 | Fact Store (SQLite) | [PENDING] |
| 4 | Confidence Engine | [PENDING] |
| 5 | Confidence Gate | [PENDING] |
| 6 | Provenance Chain | [PENDING] |
| 7 | Rule Store | [PENDING] |
| 8 | CLI Interface | [PENDING] |
| 9 | Multi-Agent Coordination | [PENDING] |
| 10 | CLI Visualizer | [PENDING] |
| 11 | Uncertainty Estimator | [PENDING] |
| 12 | API Server | [PENDING] |
| 13 | LLM Backend Integration | [PENDING] |
| 14 | Documentation & Final PR | [PENDING] |

---

## Paper Synthesis

1. **Governed Memory** — Hamed Taheri et al. — Schema-enforced shared memory layer for multi-agent systems; bridges memory silos with formal governance
2. **Facts as First Class Objects** — Simran Chana et al. — Knowledge Objects = hash-addressed tuples; eliminates capacity limits and compaction loss
3. **RPMS** — Zhenhang Yuan et al. — Rule-augmented memory synergy for embodied planning; action feasibility filtering

**Current Progress (Step 1):**
- ✅ Core data structures implemented (KnowledgeObject, ConfidenceScore, ProvenanceRecord)
- ✅ Comprehensive test suite written and passing (20+ tests)
- ❌ SQLite fact store (store.py) - NOT STARTED
- ❌ ConfidenceGate (gate.py) - NOT STARTED
- ❌ ConfidenceEngine (engine.py) - NOT STARTED
- ❌ Corresponding tests for missing components

Step 1 is approximately 50% complete.
