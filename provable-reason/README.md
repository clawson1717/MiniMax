# Provable Reason

> Verified LLM Reasoning: Process Control + Neuro-Symbolic Verification + Uncertainty Quantification

## Problem Solved

LLMs fail in reasoning tasks — they hallucinate steps, break under adversarial prompts, and provide overconfident wrong answers with no signal that anything went wrong. Existing approaches either brute-force with more compute or patch symptoms. **Provable Reason** makes LLM reasoning *verifiable*: the system knows when it doesn't know, calls in formal verification only when needed, and produces machine-checkable proof traces for its answers.

## Core Innovation

Fuses three techniques from distinct ArXiv papers into one verified reasoning system:

1. **Box Maze** (Zou et al., arXiv:2603.19182) — Process-control architecture with memory grounding, structured inference, and boundary enforcement layers; reduces boundary failure rates from ~40% to <1% under adversarial prompting.

2. **Stepwise** (arXiv:2603.19182-era) — Neuro-symbolic proof search combining LLM-generated proof steps with Isabelle/HOL verification; proves up to 77.6% of seL4 theorems. Verifies each Box Maze reasoning step on demand.

3. **Uncertainty Estimation for Reasoning Models** (arXiv:2603.19118) — Hybrid self-consistency + verbalized confidence estimator; 2-sample hybrid improves AUROC by up to +12 over either signal alone. Triggers expensive formal verification only when confidence is low.

**Result:** A reasoning engine that uses structured process control as the fast path, calls in symbolic verification on demand, and uses uncertainty quantification to decide when to do each — achieving high reliability at low average cost.

---

## Architecture

```
Query → [Box Maze Controller]
            ├── Memory Grounding (context + episodic memory)
            ├── Structured Inference (step-by-step reasoning)
            └── Boundary Enforcement (adversarial prompt protection)
                      ↓
            [Uncertainty Estimator]
            ├── Self-consistency score (N=2 sampling)
            └── Verbalized confidence
                      ↓
          ┌────────────┴────────────┐
     High Confidence          Low Confidence
          ↓                         ↓
     Return Answer         [Stepwise Verifier]
                                ├── LLM generates proof step
                                ├── Isabelle/HOL verifies
                                └── Pass → continue / Fail → repair
                                    ↓
                              [Repair Loop]
                                    ↓
                              Verified Answer + Proof Trace
```

---

## 12-Step Implementation Roadmap

| # | Phase | Step | Description |
|---|-------|------|-------------|
| 1 | Foundation | Project Setup & Architecture Skeleton | Initialize package, define Box Maze controller base class, Stepwise verifier interface, UncertaintyEstimator, top-level `ProvableReasoner` class, unit tests |
| 2 | Foundation | Box Maze — Memory Grounding Module | Working memory buffer, episodic memory store, memory grounding routine, memory isolation tests |
| 3 | Foundation | Box Maze — Structured Inference Engine | Step-by-step reasoning loop, boundary detection, early termination, integration with memory grounding |
| 4 | Foundation | Box Maze — Boundary Enforcement | Adversarial prompt detector, boundary failure classification, reset-to-safe-state response, reproduce ~40%→<1% failure reduction |
| 5 | Verification | Stepwise — Isabelle/HOL Client | Isabelle/HOL setup, Python→Isabelle JSON client, translation layer (Box Maze steps → Isar proof steps), proof state tracker |
| 6 | Verification | Stepwise — LLM Proof Step Generation | Proof step generator, beam search over candidates, backtracking on proof failure, Isabelle integration per step |
| 7 | Verification | Closed-Loop: Box Maze + Stepwise | Verification trigger on boundary failure/low UQ, proof-attempt loop, repair mechanism using proof failure info, proof trace export |
| 8 | UQ Layer | Uncertainty Estimator — Dual-Signal Hybrid | Self-consistency scorer (N=2), verbalized confidence parser, hybrid scorer with learnable weights, AUROC evaluation harness |
| 9 | UQ Layer | Calibration — When to Verify | Calibration table (uncertainty → verification probability), adaptive threshold tuning, cost-aware verification scheduler, UQ-trigger integration |
| 10 | Integration | Integrated ProvableReasoner — Full Pipeline | Unified `reason(query, opts)` API, fast/verified/adaptive modes, streaming mode, `ReasoningResult` output format |
| 11 | Integration | Benchmark Evaluation | Math (GSM8K, MATH), code (HumanEval, BigCodeBench), adversarial prompting (AdvGLUE, JailbreakBench), UQ AUROC, mode comparison |
| 12 | Production | Hardening & Deployment | Proof caching, confidence-weighted answer aggregation, reasoning trace visualization dashboard, Gradio demo, pip packaging |

---

## Key Technical Decisions

- **Isabelle/HOL over Lean** — Stepwise paper demonstrated strong seL4 results; HOL is well-suited for both mathematical and systems reasoning proofs
- **N=2 sampling** — Per the uncertainty estimation paper, 2 samples with hybrid scoring captures most AUROC benefit at minimal cost
- **Proof caching** — Isabelle verification is expensive; caching verified proof chains amortizes cost across related queries
- **Adaptive verification** — Verification is triggered on-demand by UQ signals, not on every query; Box Maze provides protection even when verification is skipped

## Differentiation

| Project | Focus | Papers Combined |
|---------|-------|----------------|
| self-healing-rag | Retrieval failures in RAG/KB systems | WriteBack-RAG + FASHA + R-C2 + MiRA |
| GESM | Persistent, governable subagent memory | AgentFactory + Governed Memory + RPMS |
| **Provable Reason** | **Verifiable LLM reasoning with formal proof** | **Box Maze + Stepwise + Uncertainty Estimation** |
