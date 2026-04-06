# MVUE — Milestone-Verified Uncertainty Estimation

**Pinpoint where reasoning breaks down, then fix exactly that.**

MVUE decomposes LLM reasoning traces into discrete, verifiable *milestones* and applies per-milestone uncertainty estimation using a hybrid 2-sample AUROC estimator. Where existing systems say "this trace is uncertain," MVUE says "milestone 3 of 7 failed — retry just that step."

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Reasoning Trace                           │
│          "First we... then we... therefore..."                   │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│              MilestoneDecomposer (OS-Themis inspired)           │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐            │
│  │Milestone│  │Milestone│  │Milestone│  │Milestone│  ...        │
│  │   M₁    │  │   M₂    │  │   M₃    │  │   M₄    │            │
│  │  "x→y"  │  │  "y→z"  │  │  "z→w"  │  │  "w→q"  │            │
│  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘            │
└───────┼───────────┼───────────┼───────────┼──────────────────┘
        │           │           │           │
        ▼           ▼           ▼           ▼
┌─────────────────────────────────────────────────────────────────┐
│           TwoSampleAUROCEstimator (arXiv:2603.19118)            │
│    Samples: model distribution vs contrastive distribution       │
│              → AUROC ∈ [0, 1] per milestone                     │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐            │
│  │  U(M₁)  │  │  U(M₂)  │  │  U(M₃)  │  │  U(M₄)  │            │
│  │  0.12   │  │  0.87 ✗ │  │  0.23   │  │  0.09   │            │
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘            │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│              BoundaryEnforcer (Box Maze inspired)                │
│   Thresholds → Actions: ACCEPT / RETRY / ESCALATE / REJECT     │
│                                                                 │
│   M₂ uncertainty 0.87 > threshold 0.70 → action: RETRY M₂     │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
               ┌──────────────────────┐
               │  Targeted Outcome    │
               │  "Retry milestone 2" │
               └──────────────────────┘
```

**Per-milestone uncertainty heatmap (diagnostic output):**

```
Milestone:  M₁     M₂     M₃     M₄     M₅
Score:      ▏▏▏   ███████▏▏  ▏▏▏▏   ▏▏
            0.12   0.87 ✗   0.23   0.09

Threshold:  ───────────────────────────────  0.70
```

---

## Installation

> ⚠️ **Not yet implemented.** The MVUE package does not exist yet. Clone this repo and implement from the roadmap, or track progress via GitHub Issues.

```bash
# Clone the repo
git clone https://github.com/your-org/milestone-verified-uncertainty-estimation.git
cd milestone-verified-uncertainty-estimation

# Install dependencies
pip install -e ".[dev]"   # not yet available

# Or install the package only
pip install mvue           # not yet available
```

**Requirements:**
- Python ≥ 3.10
- `numpy`, `scipy`, `pydantic`
- An LLM API key for the milestone decomposer (set `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`)

---

## Quick Start

> ⚠️ **Not yet implemented.** The following is the target API. Implementation tracked in GitHub Issues.

```python
# from mvue.pipeline import MVUEPipeline  # not yet available
# from mvue.enforcer import EnforcementConfig  # not yet available
pass  # See Roadmap above for implementation status
```

---

## Roadmap

| Step | Description | Status |
|------|-------------|--------|
| 1 | Define Milestone dataclass | 📋 Planned |
| 2 | MilestoneDecomposer (rule-based + LLM fallback) | 📋 Planned |
| 3 | TwoSampleAUROCEstimator port (arXiv:2603.19118) | 📋 Planned |
| 4 | MilestoneUncertaintyEstimator wrapper | 📋 Planned |
| 5 | BoundaryEnforcer with RRG Step 3 integration | 📋 Planned |
| 6 | MemoryGroundingStore | 📋 Planned |
| 7 | UncertaintyHeatmap diagnostic renderer | 📋 Planned |
| 8 | End-to-end pipeline + integration test | 📋 Planned |
| 9 | Benchmark evaluation on RRG corpora | 📋 Planned |
| 10 | LLM-assisted milestone decomposition | 📋 Planned |
| 11 | Threshold auto-tuning | 📋 Planned |
| 12 | Compositional uncertainty aggregation | 📋 Planned |
| 13 | API documentation | 📋 Planned |
| 14 | Benchmark results publication | 📋 Planned |
| 15 | Open-source release | 📋 Planned |

> ⚠️ **Note:** No implementation exists yet. README was written as design documentation (paper-ideation project). See GitHub Issues for implementation tracking.

---

## References

1. **OS-Themis** — Multi-agent critic framework with milestone decomposition and evidence chain auditing  
   arXiv:2603.19191

2. **Box Maze** — Process-control architecture with memory grounding, structured inference, and boundary enforcement layers  
   arXiv:2603.19182

3. **Uncertainty Estimation** — Hybrid 2-sample AUROC estimator for reasoning reliability  
   arXiv:2603.19118

4. **Reactive Reasoning Guardrails (RRG Step 3)** — Runtime guardrail pipeline for uncertain reasoning interception  
   (internal: `reactive-reasoning-guardrails` project)

---

## Why MVUE?

| Approach | Uncertainty granularity | Intervention |
|----------|------------------------|--------------|
| Trace-level classifiers | One score per trace | Retry full trace |
| Self-consistency | One score per trace | Retry full trace |
| MC dropout | Per-neuron | No reasoning-level action |
| **MVUE** | **Per milestone** | **Retry exactly the failing step** |

MVUE's core claim: **the milestone is the right unit of reasoning reliability**. When milestone 3 of 10 fails, you should not have to re-run the entire 10-step chain.
