# CAD-TRACE (Causal Adversarial Drift-Tracker & Corrective Evaluator)

A reasoning monitor that tracks **Causal Drift** in multi-agent collaboration. It maps every reasoning node to a **Dynamic Interaction Graph (DIG)** and applies an adversarial **TraderBench**-inspired evaluator to sense exactly when a reasoning branch drifts from the original groundwork. It then triggers **DenoiseFlow** corrections for just the drifting branches.

## Concept

CAD-TRACE solves the "Hallucination Spiral" in complex multi-agent chains. By measuring causal distance at every reasoning node using the DIG, we pinpoint exactly when intent was lost. An adversarial senser (inspired by **TraderBench**) verifies the drift, and a **DenoiseFlow** regulator heals just the specific branch of logic that drifted.

### Key Techniques

- **Dynamic Interaction Graphs (DIG):** Causal path tracing for multi-agent reasoning nodes.
- **Adversarial Robustness Evaluation:** TraderBench-inspired noise injection for stress-testing reasoning chains.
- **Sensing-Regulating-Correcting (SRC):** Uncertainty-aware denoising to pinpoint and correct broken branches in the causal graph.

## Architecture

CAD-TRACE implements a four-stage pipeline: **DIG -> Sensing -> Regulating -> Healing**.

```
ReasoningPayload
       |
       v
 LiveDriftTracker (DIG)
       |
       +-- DriftCalculator (cosine distance from root intent)
       |
       v
 UncertaintySenser (ambiguity + drift -> uncertainty flow)
       |
       v
 TruthRegulator (resilience scores, pinpoints drift origin)
       |
       v
 BranchHealer (prunes drifting branches, regenerates)
       |
       v
 CADTraceAgent (unified orchestrator with auto-heal)
```

1. **Reasoning payloads** enter the DIG as nodes with semantic vectors and causal parent edges.
2. The **DriftCalculator** computes cosine distance between each node's vector and its root intent.
3. The **UncertaintySenser** combines ambiguity (keyword heuristics) with drift scores and propagates uncertainty through the graph.
4. The **TruthRegulator** calculates Truth-Resilience scores and identifies the earliest node where drift exceeds the resilience threshold ("patient zero").
5. The **BranchHealer** surgically prunes the drifting subtree and optionally regenerates replacement nodes using a provided regenerator function.

The **CADTraceAgent** orchestrates all stages in a single `process_interaction()` call with optional auto-healing.

## Component Reference

### `src/payload.py` — ReasoningPayload
Pydantic model representing a single reasoning node. Contains `source_id`, `content`, `state_hash` (auto-generated SHA-256), `semantic_vector` (embedding), and `drift_score`.

### `src/adversary.py` — AdversarialSenser
TraderBench-inspired adversarial evaluator. Detects weak points in reasoning (ambiguity markers, logical leaps, high drift) and stress-tests payloads by injecting conflicting hints or noise.

### `src/tracker.py` — LiveDriftTracker
Manages the Dynamic Interaction Graph as a NetworkX DiGraph. Supports adding reasoning nodes with causal parent edges, querying causal paths, computing aggregate drift metrics, and exporting to DOT format.

### `src/drift.py` — DriftCalculator
Computes semantic drift via cosine distance between node vectors and their root intent. Supports single-node drift, cumulative path drift, and batch updates of all drift scores in the tracker.

### `src/sensing.py` — UncertaintySenser
DenoiseFlow-inspired uncertainty detector. Scores ambiguity via keyword density, combines it with drift scores weighted by `alpha`, and propagates uncertainty through the DIG with configurable decay.

### `src/regulating.py` — TruthRegulator
Calculates Truth-Resilience scores (inverse of weighted drift + uncertainty). Pinpoints the drift origin (earliest node below threshold), generates truth reports, and identifies nodes requiring DenoiseFlow correction.

### `src/healing.py` — BranchHealer
Prunes drifting subtrees from the DIG and optionally regenerates replacement nodes using a caller-provided regenerator function. Verifies DAG consistency after healing.

### `src/agent.py` — CADTraceAgent
Unified orchestrator. A single `process_interaction()` call adds a node, updates drift scores, runs adversarial assessment, checks resilience, and triggers auto-healing if the system is drifting.

### `src/benchmark.py` — AdversarialBenchmark
Generates synthetic reasoning chains with deliberate drift points. Measures precision, recall, false-positive rate, drift-origin pinpointing accuracy, and token savings vs. full-restart strategies.

### `src/display.py` — DriftPlotter
Interactive Plotly dashboard. Visualizes the DIG with nodes color-coded by Truth-Resilience (green = healthy, red = drifting). Exports to HTML and generates markdown summary reports.

## Installation

```bash
pip install -r requirements.txt
```

Requirements: `networkx`, `pydantic`, `numpy`, `plotly` (Python 3.10+).

## Usage

### Running the Monitor Agent on a DIG

```python
from src.payload import ReasoningPayload
from src.agent import CADTraceAgent

agent = CADTraceAgent(resilience_threshold=0.6, auto_heal=True)

# Root intent node
root = ReasoningPayload(
    source_id="root",
    content="Solve the causal inference problem.",
    semantic_vector=[1.0] * 10
)
agent.process_interaction(root, root_intent="Solve the causal inference problem.")

# Normal reasoning step
step1 = ReasoningPayload(
    source_id="step-1",
    content="Analyzing variable relationships.",
    semantic_vector=[1.0, 0.95, 1.05, 1.0, 0.98, 1.02, 1.0, 0.97, 1.03, 1.0]
)
agent.process_interaction(step1, parent_ids=["root"])

# Drifting step (vector diverges from root)
step2 = ReasoningPayload(
    source_id="step-2",
    content="Maybe the answer is something else entirely.",
    semantic_vector=[3.0, 4.0, 5.0, 3.5, 4.5, 5.5, 3.0, 4.0, 5.0, 3.5]
)
result = agent.process_interaction(
    step2, parent_ids=["step-1"],
    root_intent="Solve the causal inference problem."
)

print(result["system_status"])   # "healed" or "pruned"
print(agent.get_system_summary())
```

### Running the Benchmark with Visualization

```python
from src.benchmark import AdversarialBenchmark
from src.display import DriftPlotter

# Run benchmark
bench = AdversarialBenchmark(seed=42)
result = bench.run_test_case(length=10, drift_index=5, drift_intensity=0.8)
print(f"Precision: {result['metrics']['precision']:.2f}")
print(f"Recall:    {result['metrics']['recall']:.2f}")
print(f"Savings:   {result['metrics']['token_savings']:.1%}")

# Visualize the DIG state
plotter = DriftPlotter(bench.agent.tracker, bench.agent.regulator)
plotter.save_dashboard("visuals/drift_dashboard.html")
print(plotter.generate_report_summary())
```

## Running Tests

```bash
PYTHONPATH=. python3 -m pytest -q
```

## Roadmap

- [x] **Step 1:** Project Scaffold
- [x] **Step 2:** Causal Interaction Payload
- [x] **Step 3:** Drift-Sensing Adversary
- [x] **Step 4:** Live Causal DIG Tracker
- [x] **Step 5:** Semantic Drift Calculator
- [x] **Step 6:** Uncertainty Flow Senser
- [x] **Step 7:** Regulator of Truth-Resilience
- [x] **Step 8:** Corrective Logic Healer
- [x] **Step 9:** CAD-TRACE Monitor Agent
- [x] **Step 10:** Adversarial Drift Benchmark
- [x] **Step 11:** Plotly Drift Visualizer
- [x] **Step 12:** Documentation & Final PR

## License

MIT
