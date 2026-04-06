# Governed Executable Subagent Memory (GESM)

**A multi-agent memory architecture where subagents are first-class persistent, governable, belief-state-retrievable memory objects.**

```
pip install gesm
```

## The Problem

Existing agent frameworks treat subagents as **ephemeral**:

- **AgentFactory** (Zhang et al.) stores successful solutions as executable subagents — but they're lost after the session ends
- **Governed Memory** (Taheri et al.) provides schema-enforced shared memory — but with no executable content to govern
- **RPMS** (Yuan et al.) filters episodic memory by belief state — but has no persistent subagent representation

**No system stores, governs, and retrieves executable subagents in a belief-state-conditional way.**

## What GESM Does

GESM fuses all three techniques into one architecture:

1. **AgentFactory** → Subagents are **executable Python objects** stored in shared memory
2. **Governed Memory** → Every subagent is **schema-validated and governance-checked** before admission
3. **RPMS** → Memory retrieval is **belief-state-conditional** — the same query retrieves different subagents depending on whether the agent is in OPERATIONAL, EXPLORING, FAILURE_RECOVERY, or PLANNING mode

**Result:** Agents stop re-deriving solutions. Instead, they commit successful subagents to shared memory — and later agents invoke them directly, like calling a function.

## Architecture

```
Agent A completes task
        ↓
  Subagent(code, schema, belief_tags)
        ↓
  SchemaEnforcer → [PASS] → GovernanceEngine → [APPROVED] → MemoryStore
                         → [FAIL] → Rejected (logged)
        ↓
  Hash-addressed storage (SHA-256)

Agent B queries: "solve {task}" in BELIEF_STATE=EXPLORING
        ↓
  GESMRetriever (rule filtering + belief tag match)
        ↓
  Ranked subagent list (recency × success_count × belief relevance)
        ↓
  Agent.invoke(subagent_id, ctx) → sandboxed execution → result
```

## Installation

```bash
pip install -r requirements.txt
python -m pytest tests/ -v
```

## Quick Start

```python
from gesm import GESMAgent, SubagentMemory

agent = GESMAgent(agent_id="solver-1")

# Agent completes a task and commits the subagent
subagent = agent.commit(
    code="def solve(ctx): return {'result': ctx['input'] * 2}",
    belief_tags=["arithmetic", "multiplication"],
    metadata={"task": "double-input"}
)

# Later, a different agent recalls it
subagents = agent.recall(
    query="double arithmetic",
    belief_state=BeliefState.OPERATIONAL
)

# Invoke it directly
result = agent.invoke(subagents[0].id, ctx={"input": 21})
print(result)  # {'result': 42}
```

## Key Concepts

### SubagentMemory Object

The atomic unit of GESM — a stored executable subagent with full provenance:

```python
@dataclass
class SubagentMemory:
    id: str                      # uuid
    code: str                    # executable Python function
    schema_version: str          # e.g. "v1.0"
    governance_policy: str       # policy ID
    belief_tags: list[str]       # e.g. ["arithmetic", "parsing"]
    hash_addr: str               # SHA-256 of (code + schema + tags)
    created_at: datetime
    success_count: int           # times successfully invoked
```

### Belief States

GESM uses RPMS-style belief states to contextualize retrieval:

| State | Meaning |
|-------|---------|
| `OPERATIONAL` | Agent is reliably solving known task types |
| `EXPLORING` | Agent is attempting novel or unfamiliar tasks |
| `FAILURE_RECOVERY` | Agent is recovering from recent failures |
| `PLANNING` | Agent is in high-level planning mode |

Same query, different belief state → different subagents retrieved.

### Schema Governance

Before a subagent enters memory, it must pass:
1. **Schema validation** — code structure, imports, return type match the schema spec
2. **Governance check** — RBAC permissions, content filters, staleness policies

Unvalidated subagents are rejected and logged to the audit trail.

## Components

| Component | File | Description |
|-----------|------|-------------|
| Subagent | `src/subagent.py` | Executable Python subagent format + sandboxed executor |
| Schema | `src/schema.py` | Schema registry + enforcement layer |
| Belief State | `src/belief_state.py` | RPMS-style belief state machine |
| Memory Store | `src/memory_store.py` | Hash-addressed, capacity-unlimited storage |
| Retriever | `src/retriever.py` | Belief-state-conditional rule retrieval |
| Governance | `src/governance.py` | RBAC + content filtering + audit logging |
| Agent | `src/agent.py` | GESMAgent: commit, recall, invoke |
| CLI | `src/cli.py` | `python -m gesm.cli` |

## Roadmap

See `memory/project-gesm-plan.md` for the full 12-step implementation plan.

| Step | Description | Status |
|------|-------------|--------|
| 1 | Project scaffold + data structures | pending |
| 2 | Executable subagent format + executor | pending |
| 3 | Schema enforcement layer | pending |
| 4 | Belief state tracker | pending |
| 5 | Hash-addressed memory store | pending |
| 6 | Rule retrieval + episodic filter | pending |
| 7 | Governance engine | pending |
| 8 | Multi-agent integration layer | pending |
| 9 | CLI interface | pending |
| 10 | Benchmarks | pending |
| 11 | Integration tests | pending |
| 12 | Documentation + final PR | pending |

## Benchmarks

GESM is benchmarked against three baselines:

- **AgentFactory** — ephemeral subagents (no persistence)
- **Governed Memory** — schema-enforced memory (no executable content)
- **RPMS** — belief-state retrieval (no persistent subagent representation)

Metrics: retrieval latency (P50/P95/P99), memory capacity, belief-state retrieval accuracy, multi-agent handoff success rate.

## Prior Art Comparison

| Feature | AgentFactory | Governed Memory | RPMS | **GESM** |
|---------|:-----------:|:--------------:|:----:|:--------:|
| Executable subagents | ✓ | ✗ | ✗ | **✓** |
| Persistent memory | ✗ | ✓ | ✗ | **✓** |
| Schema governance | ✗ | ✓ | ✗ | **✓** |
| Belief-state retrieval | ✗ | ✗ | ✓ | **✓** |
| Hash-addressed storage | ✗ | ✗ | ✗ | **✓** |
| Audit logging | ✗ | ✓ | ✗ | **✓** |

## Papers Referenced

- Zhang et al. — **AgentFactory**: Continuous Agent Capability Growth via Executable Subagent Preservation (2026-03-19)
- Taheri et al. — **Governed Memory**: Schema-Enforced Shared Memory for Enterprise Multi-Agent Systems (2026-03-19)
- Yuan et al. — **RPMS**: Rule-Augmented Memory Synergy for Embodied Planning (2026-03-19)
