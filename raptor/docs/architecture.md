# Architecture

This document explains RAPTOR's internal architecture in detail — how the components interact, how signals flow through the system, and the design rationale behind key decisions.

## System Overview

RAPTOR operates as a **feedback loop controller** that sits between your LLM agents and the user. At each step, it:

1. **Polls** multiple agents in parallel for reasoning chains
2. **Monitors** two orthogonal uncertainty signals
3. **Fuses** signals into a unified signal vector
4. **Scores** six possible actions via a utility function
5. **Executes** the highest-utility action
6. **Repeats** until a terminal action (RESPOND, ESCALATE, STOP) is reached

```
User Prompt
    │
    ▼
┌──────────────────────────────────────────────────────────────────┐
│                      RAPTOR Loop                                 │
│                                                                  │
│  ┌─────────┐    ┌───────────────┐    ┌──────────────────┐        │
│  │  Agent   │───▶│  Entropy      │    │  Disagreement    │        │
│  │  Pool    │    │  Tracker      │    │  Monitor         │        │
│  │ (poll)   │───▶│               │    │  (DiscoUQ)       │        │
│  └─────────┘    └───────┬───────┘    └────────┬─────────┘        │
│                         │                     │                  │
│                         └──────────┬──────────┘                  │
│                                    ▼                             │
│                           Signal Vector S(t)                     │
│                                    │                             │
│                                    ▼                             │
│                          Utility Engine                          │
│                       U(a|S) = Σwₖ·φₖ(a,S)                      │
│                                    │                             │
│                                    ▼                             │
│                          Action Selection                        │
│                    (with hysteresis control)                      │
│                                    │                             │
│              ┌─────────────────────┼─────────────────────┐       │
│              ▼         ▼           ▼          ▼           ▼      │
│           RESPOND   REROLL     VERIFY    ESCALATE    RETRIEVE    │
│           (exit)    (loop)     (loop)    (exit)      (loop)      │
│                                                                  │
│                               STOP (exit)                        │
└──────────────────────────────────────────────────────────────────┘
    │
    ▼
RAPTORResult (answer + context + signals)
```

## Component Details

### 1. Agent Pool & Polling (`agents.py`)

The polling layer abstracts over different agent backends. Any object satisfying the `ReasoningAgent` protocol can participate:

```python
class ReasoningAgent(Protocol):
    def generate(self, prompt: str) -> AgentResponse: ...
```

**Polling strategy:**
- `poll_agents()` dispatches `generate()` calls concurrently via `ThreadPoolExecutor`
- Each agent gets its own thread — no serialization bottleneck
- Failed agents are recorded in `PollResult.errors`, not raised (unless `fail_fast=True`)
- Timeout handling: agents that exceed the deadline are cancelled and logged

**Streaming variant:**
- `poll_agents_streaming()` uses `StreamingReasoningAgent.stream()` which yields `AgentStreamEvent` tokens
- Enables real-time entropy computation as tokens arrive (before the full response is ready)
- An `on_event` callback fires for each token from any agent

**Reroll:**
- `reroll(agent, prompt, n_candidates)` generates multiple candidate responses from a single agent
- Used when the orchestrator decides to REROLL — provides diversity within one model

### 2. Entropy Trajectory Tracker (`entropy_tracker.py`)

Based on [arXiv:2603.18940](https://arxiv.org/abs/2603.18940).

**Key insight:** If per-step entropy monotonically decreases during chain-of-thought reasoning, the chain is 68.8% likely to be correct vs. 46.8% for non-monotone chains.

**How it works:**

1. Each token's log-probability distribution is fed to `update(log_probs)`
2. Shannon entropy is computed: `H = -Σ p·log(p)`
3. The tracker maintains the full entropy trajectory `[H₁, H₂, ..., Hₜ]`
4. **Monotonicity check:** `Hᵢ > Hᵢ₊₁` for all consecutive pairs
5. **Slope computation:** Linear regression (OLS) over the trajectory
6. **Confidence score:** Base 0.7 if monotone (0.3 otherwise) + slope contribution up to 0.3

**Signal output:**
```python
TrajectorySignal(
    n_steps=10,
    monotonicity=True,        # Decreasing at every step
    entropy_slope=-0.045,     # Negative = convergent reasoning
    final_entropy=1.23,
    entropies=[...],          # Full trajectory
    confidence_score=0.85,
)
```

**Design choice — synthetic log-probs:** When real log-probabilities aren't available (e.g., from APIs that don't expose them), the orchestrator generates synthetic distributions from token frequency analysis. More focused/repetitive text → concentrated distribution → lower entropy. This preserves the monotonicity signal even without model internals.

### 3. Disagreement Monitor (`disagreement.py`)

Based on [arXiv:2603.20975](https://arxiv.org/abs/2603.20975) (DiscoUQ).

**Key insight:** Structured disagreement analysis (linguistic + embedding geometry) calibrates confidence far better than simple vote-counting, especially in the "weak disagreement" tier where voting fails.

**Two feature sets:**

#### Linguistic Features (DiscoUQ-LLM)
- **Evidence overlap:** Jaccard similarity of keyword sets across all agent pairs. Measures whether agents cite similar evidence.
- **Argument strength:** Heuristic combining reasoning step count and vocabulary diversity. More steps + richer vocabulary → stronger argument.
- **Divergence depth:** First step index where agents' reasoning chains diverge. Late divergence → more agreement on the reasoning path.

#### Embedding Geometry Features (DiscoUQ-Embed)
- **Dispersion:** Mean standard deviation of embedding vectors across agents. High dispersion → agents say very different things.
- **Cohesion:** Mean pairwise cosine similarity between agent embeddings. Low cohesion → the agents are spread out in embedding space.

When pre-computed embeddings aren't available, the monitor generates lightweight TF-IDF hashed embeddings (no external model needed).

**Calibration:** Features are combined via weighted sum → sigmoid to produce a confidence score in [0, 1]. Three modes control which features participate:
- `"llm"` — linguistic features only
- `"embed"` — embedding features only
- `"learn"` — all features (fullest signal)

**Tier classification:**
- **Low disagreement** (confidence > 0.8): Agents strongly agree
- **Medium** (0.5 < confidence ≤ 0.8): Some uncertainty
- **Weak** (confidence ≤ 0.5): This is where RAPTOR provides the most value

### 4. Signal Fusion (`orchestrator.py`)

The orchestrator fuses both signal sources into a single `SignalVector`:

```python
S(t) = SignalVector(
    monotonicity_flag=True,         # From entropy tracker
    entropy_slope=-0.045,           # From entropy tracker
    disagreement_score=0.35,        # 1 − confidence (from disagreement)
    dispersion_score=0.12,          # From disagreement
    cohesion_score=0.78,            # From disagreement
    divergence_depth=5,             # From disagreement
)
```

**Aggregation across agents:**
- Monotonicity = ALL agents must be monotone (conservative)
- Slope, confidence, final entropy = mean across agents
- Entropies = element-wise mean (shorter trajectories padded with last value)

The `disagreement_score` is inverted from `confidence_score` so that higher values consistently indicate *more concern*.

### 5. Utility Engine (`utility.py`)

Based on [arXiv:2603.19896](https://arxiv.org/abs/2603.19896).

For each of the 6 actions, compute:

```
U(a | S) = Σₖ wₖ · φₖ(a, S)
```

**Five feature functions (φₖ):**

| φₖ | What it measures |
|----|------------------|
| `φ_gain` | Expected accuracy improvement for this action given signal state |
| `φ_confidence` | Combined calibrated confidence from both signal sources |
| `φ_cost` | Token/latency cost, scaled by step number (urgency to finish) |
| `φ_redundancy` | Penalty for redundant actions (retrieve after retrieve, exceeded rerolls) |
| `φ_severity` | How bad is getting it wrong in the current signal state |

**Default weights:**
```python
{"gain": 0.30, "confidence": 0.25, "cost_penalty": -0.15,
 "redundancy_penalty": -0.10, "severity": 0.20}
```

Note: `cost_penalty` and `redundancy_penalty` are negative — they penalize expensive or redundant actions.

**Learned mode:** The engine supports online SGD weight updates:
```python
wₖ ← wₖ + lr · (reward − predicted) · φₖ
```

### 6. Hysteresis Control

Two levels of hysteresis prevent action flickering:

1. **Threshold hysteresis** (Utility Engine): The best action must beat the previous action by more than `switch_threshold` (default 0.05) to trigger a switch.

2. **Step-count hysteresis** (Orchestrator): A new action must be proposed for `hysteresis_steps` consecutive calls before the switch takes effect.

### 7. Logging & Replay

Every orchestration step writes a structured JSONL record to `{log_dir}/session_{id}.jsonl`. Records include:
- Full signal vector
- Trajectory and disagreement signal details
- Utility breakdown for the chosen action
- Human-readable reason string
- Timestamp and session context

These logs enable:
- **Post-hoc analysis:** Study what RAPTOR decided and why
- **Dashboard replay:** Step through sessions visually
- **Debugging:** Compare signal evolution across steps

## Data Flow Example

Here's a concrete example of one RAPTOR step:

```
1. User asks: "Prove that √2 is irrational"

2. poll_agents() dispatches to 5 agents in parallel
   → 5 AgentResponse objects with reasoning chains

3. EntropyTracker processes each agent's chain:
   Agent 1: [2.1, 1.8, 1.5, 1.2, 0.9] → monotone ✓, slope=-0.30
   Agent 2: [2.0, 1.6, 1.7, 1.3, 1.0] → non-monotone ✗ (1.6→1.7)
   Agent 3: [2.2, 1.9, 1.6, 1.3, 1.1] → monotone ✓, slope=-0.28
   ...
   Aggregated: monotonicity=False (agent 2 broke it), avg_slope=-0.27

4. DisagreementMonitor analyzes all 5 responses:
   evidence_overlap=0.72, argument_strength=0.65
   dispersion=0.18, cohesion=0.74
   → confidence_score=0.61, tier="medium"

5. Signal fusion:
   S(t) = [mono=False, slope=-0.27, disag=0.39, disp=0.18, coh=0.74, depth=3]

6. Utility Engine scores all actions:
   VERIFY:   U=0.312  ← highest (non-monotone + medium disagreement)
   REROLL:   U=0.245
   RESPOND:  U=0.198
   ESCALATE: U=0.156
   RETRIEVE: U=0.089
   STOP:     U=0.023

7. Action: VERIFY → generate verification prompt, loop back to step 2
```

## Design Rationale

| Decision | Choice | Why |
|----------|--------|-----|
| **Strict monotonicity** | 100% decreasing | Paper shows strict is the best predictor |
| **Hybrid disagreement** | Linguistic + embedding | Covers both semantic and geometric diversity |
| **Synthetic log-probs** | Token-frequency hashing | Works without model internals; preserves monotonicity signal |
| **6-action space** | respond/reroll/verify/escalate/retrieve/stop | Matches utility-guided orchestration paper |
| **Thread-based polling** | ThreadPoolExecutor | Agents are I/O-bound (API calls); threads work well |
| **JSONL logging** | Append-only files | Simple, streamable, easy to parse |
| **Hysteresis** | Two levels (threshold + step-count) | Prevents noisy oscillation between actions |
| **Online SGD** | Optional learned weights | Enables adaptation to specific use cases |
