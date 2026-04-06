# SNTR — Self-Healing via Neuro-Symbolic Trajectory Repair

**An agent framework that doesn't navigate around reasoning failures — it formally repairs them using Isabelle/HOL proof search.**

```
pip install sntr
```

## The Problem

Today's self-healing agents recover from failures using heuristic strategies:

- **FASHA** diagnoses failure type and selects a recovery approach (retry, backtrack, simplify) — but doesn't verify the repaired reasoning is *logically correct*
- **Milestone Guardian** uses structured recovery when a milestone critic rejects evidence — but has no formal way to verify repaired reasoning
- **TVC** detects failures and prunes bad trajectories — but doesn't repair them

When an LLM agent makes a *reasoning error* (not a knowledge gap, not an external failure — an actual logical mistake), no existing system can:

1. Identify *which* inference step is wrong
2. Formally prove the *correct* derivation
3. Apply that proof as a trajectory patch
4. Continue execution with a verified reasoning path

**Result:** Agents either give up, try again randomly, or continue with the same flawed reasoning.

## What SNTR Does

SNTR fuses three techniques into a novel self-healing architecture:

### 1. FASHA Diagnostic Engine — Fast Pre-Filter
Before invoking expensive neuro-symbolic repair, SNTR classifies the failure type. Only **REASONING_ERROR** failures (logical flaws, not knowledge gaps) trigger Isabelle repair. Other failures (context loss, external failure, strategy mismatch) use FASHA's heuristic recovery directly. Inspired by FASHA (failure-aware-self-healing-agent).

### 2. Stepwise Neuro-Symbolic Proof Search — Formal Repair
When a reasoning error is detected, SNTR translates the failed claim into a HOL theorem, invokes Isabelle/HOL proof search, extracts the correct logical derivation, and generates a trajectory patch. This is categorically different from "try a different approach" — it finds the *provably correct* derivation. Inspired by Stepwise (Zou et al., 2026).

### 3. HeRL Hindsight Experience Replay — Learning from Repairs
Successfully repaired trajectories are stored in a hindsight experience library. When a new reasoning error occurs, SNTR检索 similar past repairs (via embedding similarity on HOL terms) to warm-start the Isabelle proof search. Over time, the repair engine gets faster and more reliable. Inspired by HeRL (Zhang et al., 2026).

**Result:** An agent that repairs *logically incorrect* reasoning using formal proofs — not heuristics, not retries, not navigation.

## Architecture

```
Reasoning Task
      │
      ▼
┌──────────────────────────────────────────────────────────┐
│                   SNTR Self-Healing Loop                  │
│                                                           │
│  ┌────────────┐                                           │
│  │   DETECT   │  TVC-style confidence tracking            │
│  │ (step PRMs)│  Low-confidence steps trigger warning     │
│  └─────┬──────┘                                           │
│        │ confidence_score < threshold                     │
│        ▼                                                  │
│  ┌────────────┐                                           │
│  │  DIAGNOSE  │  FASHA Diagnostic Engine                  │
│  │  (FASHA)   │  Classifies: REASONING_ERROR /           │
│  └─────┬──────┘  KNOWLEDGE_GAP / CONTEXT_LOSS /           │
│        │         STRATEGY_MISMATCH / EXTERNAL_FAILURE    │
│        │ REASONING_ERROR                                  │
│        ▼                                                  │
│  ┌─────────────────────────────────────────────────────┐  │
│  │                    REPAIR (Stepwise)                │  │
│  │                                                      │  │
│  │  Failed Claim                                         │  │
│  │      │                                                │  │
│  │      ▼                                                │  │
│  │  HOL Translator ──► Isabelle/HOL Theorem            │  │
│  │      │                                                │  │
│  │      ▼                                                │  │
│  │  Neuro-Symbolic Proof Search (Stepwise)              │  │
│  │      │                                                │  │
│  │      ▼                                                │  │
│  │  Logical Derivation ──► Proof Result                 │  │
│  └──────────────┬──────────────────────────────────────┘  │
│                 │ PROOF_FOUND                             │
│                 ▼                                         │
│  ┌────────────────────────┐                                │
│  │   Trajectory Patcher   │  Apply corrected step to      │
│  │   (HeRL Experience)    │  trajectory, store in library  │
│  └───────────┬────────────┘                                │
│              │                                             │
│              ▼                                             │
│    [REPAIR SUCCEEDED]  ──► Continue Execution               │
│              │                                             │
│              │ PROOF_FAILED                                │
│              ▼                                             │
│    Fallback to FASHA Recovery ──► Backtrack/Retry/Simplify │
│                                                           │
│  ┌─────────────────────────────────────────────────────┐  │
│  │           Hindsight Experience Library              │  │
│  │  (failed_step, hol_theorem, proof, corrected_step)  │  │
│  │  Embedding similarity检索 for warm-start            │  │
│  └─────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────┘
```

## Installation

```bash
# Clone the repository
git clone https://github.com/clawson1717/ClawWork
cd ClawWork/self-healing-neurosym-trajectory-repair

# Install Python dependencies
pip install -r requirements.txt

# Install Isabelle2024 (required for neuro-symbolic repair)
# SNTR will auto-provision the Isabelle Docker container on first use
docker pull isabelle/isabelle2024

# Run tests
python -m pytest tests/ -v
```

## Quick Start

```python
from sntr import SNTRAgent, FailureTrajectory

# Initialize SNTR agent
agent = SNTRAgent(
    isabelle_timeout=30,      # seconds per Isabelle proof attempt
    repair_threshold=0.5,    # confidence below which to trigger repair
    experience_cache_size=1000
)

# Load a failed reasoning trajectory
trajectory = FailureTrajectory.from_file("failed_math_reasoning.json")

# Run self-healing loop
result = agent.self_heal(trajectory)

print(f"Status: {result.status}")
# → REPAIR_SUCCEEDED / REPAIR_FAILED / ESCALATED

print(f"Original failed step: {result.original_step}")
# → "Since all humans are mortal and Socrates is a human, Socrates must be eternal"

print(f"Corrected step: {result.corrected_step}")
# → "Since all humans are mortal and Socrates is a human, Socrates must be mortal"

print(f"Isabelle proof: {result.hol_proof}")
# → Formal Isabelle/HOL proof trace

print(f"Experience library hits: {result.experience_hits}")
```

### Standalone Proof Repair

```python
from sntr import IsabelleProofEngine

engine = IsabelleProofEngine(timeout=30)

# Translate a reasoning claim to HOL and prove it
result = engine.prove(
    claim="If all A are B and all B are C, then all A are C",
    context=["All(A, B)", "All(B, C)"]
)

print(f"Proof found: {result.proof_found}")
print(f"HOL term: {result.hol_term}")
print(f"Derivation: {result.derivation}")
```

### Heuristic Recovery (FASHA fallback)

```python
from sntr import FASHAHeuristicRecovery

recovery = FASHAHeuristicRecovery()

# When Isabelle repair fails or isn't applicable
failure = FailureEvent(
    failure_type=FailureType.KNOWLEDGE_GAP,
    failed_step="I need to know the capital of France",
    context={}
)

strategy = recovery.select_strategy(failure)
# → RecoveryStrategy.ASK_CLARIFICATION

result = recovery.execute(strategy, failure)
```

## Key Concepts

### Failure Types & Repair Routes

| Failure Type | Repair Route | Tool |
|--------------|-------------|------|
| `REASONING_ERROR` | Neuro-symbolic repair | Isabelle/HOL → Trajectory Patcher |
| `KNOWLEDGE_GAP` | Heuristic recovery | FASHA AskClarification |
| `CONTEXT_LOSS` | Heuristic recovery | FASHA ReRead |
| `STRATEGY_MISMATCH` | Heuristic recovery | FASHA RetryDifferentApproach |
| `EXTERNAL_FAILURE` | Heuristic recovery | FASHA Simplify |
| `MILESTONE_REJECTED` | Milestone integration | Milestone Guardian + SNTR |

### Trajectory Patch

The atomic unit of repair — a formally verified correction:

```python
@dataclass
class TrajectoryPatch:
    original_step: str                  # The failed reasoning step
    failed_reason: str                   # Why it failed (from diagnostic)
    hol_theorem: str                     # Isabelle/HOL theorem statement
    proof_trace: List[str]              # Step-by-step Isabelle proof
    corrected_step: str                  # The repaired step
    confidence_before: float            # Confidence before repair
    confidence_after: float             # Confidence after repair
    repair_time_ms: int                 # Isabelle proof search time
```

### Hindsight Experience Library

Stores repaired trajectories for warm-start future repairs:

```python
@dataclass
class HindsightExperience:
    id: str                              # UUID
    failed_claim: str                    # Original failed claim
    hol_term: str                        # HOL theorem
    proof: List[str]                     # Isabelle proof steps
    corrected_reasoning: str            # Corrected derivation
    success_count: int                  # Times retrieved and succeeded
    failure_count: int                  # Times retrieved but failed
    embedding: List[float]              # For similarity检索
    created_at: datetime
```

## Components

| Component | File | Description |
|-----------|------|-------------|
| Diagnostic Engine | `src/diagnostic.py` | FASHA-style failure type classifier |
| Isabelle Engine | `src/isabelle_engine.py` | Isabelle/HOL wrapper + neuro-symbolic prover |
| HOL Translator | `src/hol_translator.py` | LLM claim → HOL term translation |
| Trajectory Patcher | `src/patcher.py` | Apply proof-derived corrections to trajectories |
| Hindsight Library | `src/hindsight_library.py` | Experience replay store + retrieval |
| Heuristic Recovery | `src/heuristic_recovery.py` | FASHA fallback recovery strategies |
| SNTR Loop | `src/sntr_loop.py` | Main orchestrator: detect → diagnose → repair → verify |
| Milestone Integration | `src/milestone_integration.py` | SNTR as repair engine for Milestone Guardian |
| Agent | `src/agent.py` | High-level SNTRAgent API |
| CLI | `src/cli.py` | `sntr diagnose`, `sntr repair`, `sntr replay` |

## Roadmap

| Step | Description | Status |
|------|-------------|--------|
| **1** | Project scaffold + core data structures | 🔲 |
| **2** | FASHA diagnostic engine integration | 🔲 |
| **3** | Isabelle/HOL proof engine wrapper | 🔲 |
| **4** | Trajectory patch generator | 🔲 |
| **5** | Proof-aware recovery selector | 🔲 |
| **6** | HeRL-inspired hindsight experience library | 🔲 |
| **7** | SNTR self-healing loop orchestrator | 🔲 |
| **8** | Milestone Guardian integration layer | 🔲 |
| **9** | CLI interface | 🔲 |
| **10** | Benchmark suite | 🔲 |
| **11** | Integration tests + Isabelle environment | 🔲 |
| **12** | Documentation + demo | 🔲 |

## Benchmarks

SNTR is evaluated against FASHA, Milestone Guardian, and TVC on:

| Benchmark | Task | Metric |
|-----------|------|--------|
| **GSM8K** | Math reasoning errors | % repaired without task restart |
| **ProofWriter** | FOL deductive reasoning | Proof repair success rate |
| **FOLIO** | Complex logical deduction | Milestone repair rate |
| **HumanEval** | Algorithm reasoning errors | Corrected trajectory accuracy |
| **HotpotQA** | Multi-step fact synthesis | Reasoning repair → final answer accuracy |
| **Custom** | Isabelle-repairable claims | Isabelle proof success rate |

**Expected advantage:** SNTR should significantly outperform heuristic-only systems on tasks where the reasoning error is *logically determinable* (math, formal logic, deductive reasoning) — exactly the tasks where Isabelle proof search excels.

## Comparison with Related Work

| Feature | FASHA | Milestone Guardian | TVC | VERITAS | **SNTR** |
|---------|:-----:|:------------------:|:---:|:-------:|:--------:|
| Failure diagnosis | ✓ | ✓ | ✓ | ✓ | **✓** |
| Heuristic recovery | ✓ | ✓ | ✓ | ✗ | **✓** |
| Milestone-based recovery | ✗ | ✓ | ✗ | ✗ | **✓** |
| Trajectory patch application | ✗ | limited | pruning | ✗ | **✓** |
| Neuro-symbolic proof repair | ✗ | ✗ | ✗ | ✗ | **✓** |
| Isabelle/HOL verification | ✗ | ✗ | ✗ | ✗ | **✓** |
| Hindsight experience replay | ✗ | ✗ | ✗ | ✗ | **✓** |
| Formal proof of corrected derivation | ✗ | ✗ | ✗ | ✗ | **✓** |

## Papers Referenced

- Zou et al. — **Stepwise**: Neuro-Symbolic Proof Search for Automated Systems Verification (arXiv, 2026-03-23)
- Wenjian Zhang et al. — **HeRL**: Hindsight Experience Guided Reinforcement Learning for LLMs (arXiv, 2026-03-23)
- Li et al. — **OS-Themis**: A Scalable Critic Framework for Generalist GUI Rewards (arXiv, 2026-03-21)
- Qiang Zou et al. — **Box Maze**: A Process-Control Architecture for Reliable LLM Reasoning (arXiv, 2026-03-21)
- Haklay et al. — **Pitfalls in Evaluating Interpretability Agents** (arXiv, 2026-03-23)

## Relationship to Other Projects

- **FASHA** — SNTR adopts FASHA's diagnostic engine as the pre-filter; FASHA-style heuristic recovery is the fallback when Isabelle repair fails or isn't applicable
- **Milestone Guardian** — SNTR acts as a drop-in *repair engine* within Milestone Guardian's recovery loop; when the milestone critic rejects evidence, SNTR formally repairs the reasoning
- **TVC** — SNTR uses TVC's confidence tracking for early failure detection; TVC's pruning is complementary to SNTR's patch application
- **VERITAS** — SNTR's Isabelle verification is categorically different from VERITAS's PRM step scoring: VERITAS *scores* reasoning quality, SNTR *repairs* failed reasoning using formal proof
- **GESM** — Hindsight experiences in SNTR could be stored as GESM-style executable subagents for cross-task reuse
