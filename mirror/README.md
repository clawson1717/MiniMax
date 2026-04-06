# MIRROR — Measuring Inference Reliability via Round-trip Observation Reconciliation

A Python library + CLI for stress-testing whether LLM chain-of-thought explanations are actually faithful to the model's reasoning process.

## Why?

When LLMs show their "reasoning," you have no way to know if those explanations reflect what actually happened or are just plausible-sounding post-hoc rationalizations. MIRROR gives you a multi-dimensional faithfulness score using three complementary techniques.

## How It Works

### 1. Cycle Consistency Probing
Borrowed from [R-C2](https://arxiv.org/abs/2603.25720) (cross-modal cycle consistency as verification).

- **Forward:** Input → CoT → Answer
- **Reverse:** Answer + CoT → reconstructed input constraints → compare to original
- **Cross-model:** Different model follows only the CoT → should reach same answer

Low cycle consistency = the explanation doesn't actually encode the reasoning path.

### 2. Multi-Dimensional Judge Scoring
Borrowed from [De Jure](https://arxiv.org/abs/2604.02276) (19-dimension LLM-as-judge evaluation).

Scores CoT along 6 faithfulness dimensions:
- **Logical coherence** — do steps follow from each other?
- **Necessity** — are all steps needed?
- **Sufficiency** — do steps contain enough info?
- **Consistency** — does the explanation contradict itself?
- **Groundedness** — does it reference actual input elements?
- **Counterfactual stability** — does changing the explanation change the answer?

### 3. Functional Interchangeability Testing
Borrowed from [Pitfalls in Evaluating Interpretability Agents](https://arxiv.org/abs/2603.xxxxx) (functional interchangeability over subjective ground truth).

- Build a "CoT executor" that follows the written explanation literally
- If the explanation is faithful, the executor should reach the same answer
- Generate input perturbations → faithful CoT should change predictably

## Quick Start

```bash
pip install mirror-cot

# Test a single explanation
mirror probe --model gpt-4 --input "What is 23 * 47?" --cot "First I multiply 20*47=940, then 3*47=141, total=1081"

# Full benchmark test
mirror test --model claude-3.5 --task gsm8k --samples 100

# Compare models
mirror compare --models gpt-4,claude-3.5,llama-3 --task mmlu --output report.html
```

## Architecture

```
┌─────────────────────────────────────────────────┐
│                   MIRROR CLI                     │
└────────────────────┬────────────────────────────┘
                     │
        ┌────────────▼────────────────┐
        │      Test Orchestrator       │
        └──┬─────────┬─────────┬──────┘
           │         │         │
    ┌──────▼──┐ ┌────▼────┐ ┌──▼──────────┐
    │  Cycle   │ │  Multi- │ │  Functional  │
    │  Consis- │ │  Dim    │ │  Interchange │
    │  tency   │ │  Judge  │ │  -ability    │
    │  Prober  │ │  Scorer │ │  Tester      │
    └──────┬──┘ └────┬────┘ └──┬──────────┘
           │         │         │
        ┌──▼─────────▼─────────▼──┐
        │     Faithfulness Report   │
        │  (per-dimension scores,   │
        │   aggregate metrics,      │
        │   flagged explanations)   │
        └─────────────────────────┘
```

## Implementation Roadmap

### Step 1: Core Framework + Cycle Consistency Prober ✅
- [x] Project scaffold
- [x] LLM abstraction layer (litellm)
- [x] `CycleConsistencyProber` with forward/reverse/cross-model tests
- [x] Basic CLI: `mirror probe`
- [x] Unit tests

### Step 2: Multi-Dimensional Judge Scorer ⬅️ CURRENT
- [ ] `DimensionScorer` with 6 faithfulness dimensions
- [ ] LLM-as-judge prompts (calibrated)
- [ ] Per-dimension + weighted composite scoring
- [ ] JSON report output

### Step 3: Functional Interchangeability Tester
- [ ] `InterchangeabilityTester` with CoT-executor
- [ ] Perturbation generator for counterfactual testing
- [ ] Agreement/disagreement scoring

### Step 4: Test Orchestrator + Benchmarks
- [ ] `TestOrchestrator` coordinating all three probes
- [ ] Built-in benchmarks: GSM8K, MMLU, ARC
- [ ] Batch processing + HTML/Markdown reports

### Step 5: CLI Polish + Advanced Features
- [ ] `mirror compare` and `mirror watch` commands
- [ ] CI integration mode
- [ ] Configurable thresholds

## Tech Stack
- Python 3.11+
- litellm (model abstraction)
- Click (CLI)
- pytest (testing)
- Rich (terminal output)

## Source Papers
- [R-C2: Cycle-Consistent RL Improves Multimodal Reasoning](https://arxiv.org/abs/2603.25720)
- [De Jure: Iterative LLM Self-Refinement for Structured Extraction](https://arxiv.org/abs/2604.02276)
- Pitfalls in Evaluating Interpretability Agents (arXiv, March 2026)

## License
MIT
