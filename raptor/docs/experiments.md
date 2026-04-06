# Running Experiments

RAPTOR includes a full experiment harness for benchmarking, ablation studies, and baseline comparisons. The `raptor.experiments` module provides synthetic data generators, pluggable baseline modes, metrics computation, hyperparameter sweeps, and structured reporting.

## Overview

```python
from raptor.experiments import (
    generate_gsm8k_synthetic,
    run_experiment,
    compute_metrics,
    ExperimentConfig,
    ExperimentReport,
    BaselineMode,
)

tasks = generate_gsm8k_synthetic(50)
agents = [MockReasoningAgent(...) for _ in range(5)]
config = ExperimentConfig(baseline_mode=BaselineMode.RAPTOR_FULL)
results = run_experiment(config, tasks, agents)
metrics = compute_metrics(results)
```

---

## Baseline Modes

The `BaselineMode` enum defines four evaluation modes for comparison:

| Mode | Value | Description |
|------|-------|-------------|
| `NAIVE` | `"naive"` | Single agent, direct answer, no RAPTOR pipeline |
| `SELF_CONSISTENCY` | `"self_consistency"` | Majority vote over N agents (no orchestration) |
| `DISCOUQ_ONLY` | `"discouq_only"` | Disagreement monitoring + reroll, no entropy/utility |
| `RAPTOR_FULL` | `"raptor_full"` | Full RAPTOR pipeline: entropy + disagreement + utility |

### What Each Baseline Does

**NAIVE** — Calls `agents[0].generate(prompt)` and returns the answer directly. No polling, no quality signals. This is the "do nothing" baseline.

**SELF_CONSISTENCY** — Polls all agents, then takes a majority vote over their `final_answer` fields. Confidence is the fraction of agents that agreed with the winning answer.

**DISCOUQ_ONLY** — Polls all agents, runs disagreement monitoring via `DisagreementMonitor`, and if the disagreement tier is `"weak"`, performs one round of rerolling on the first agent. Final answer is majority vote.

**RAPTOR_FULL** — Runs the complete `run_with_raptor()` pipeline: poll → signal fusion → utility scoring → action selection → loop. Uses all three signal sources (entropy trajectory, disagreement structure, utility-guided orchestration).

---

## Data Structures

### `BenchmarkTask`

A single evaluation task.

```python
@dataclass
class BenchmarkTask:
    question: str          # The question or prompt
    ground_truth: str      # Known correct answer
    dataset_name: str      # Which benchmark (e.g., "gsm8k_synthetic")
    task_id: str           # Unique identifier
    metadata: dict = {}    # Arbitrary extra data (difficulty, category, etc.)
```

### `EvalResult`

Result of evaluating one task.

```python
@dataclass
class EvalResult:
    task_id: str                           # Links back to BenchmarkTask
    predicted: str                         # Model's predicted answer
    correct: str                           # Ground truth answer
    is_correct: bool                       # Whether prediction matches truth
    signals_history: list[dict] = []       # Signal vectors from orchestration
    n_rerolls: int = 0                     # Number of reroll actions taken
    n_steps: int = 1                       # Number of orchestration steps
    cost_estimate: float = 0.0             # Estimated cost (arbitrary units)
    confidence: float = 0.5               # Calibrated confidence score
```

### `ExperimentConfig`

Configuration for a single experiment run.

```python
@dataclass
class ExperimentConfig:
    raptor_config: Config = Config()                      # RAPTOR system config
    baseline_mode: BaselineMode = BaselineMode.RAPTOR_FULL  # Which baseline to run
    n_agents: int = 5                                     # Number of agents to poll
    dataset: str = "gsm8k"                                # Dataset name (labeling)
    n_samples: int = 100                                  # Max tasks to evaluate (0 = all)
    label: str = ""                                       # Human-readable label
```

---

## Synthetic Data Generators

RAPTOR provides three synthetic data generators for testing without requiring real datasets or API keys. All are deterministic given a seed.

### `generate_gsm8k_synthetic(n, seed=42)`

Generates GSM8K-style grade-school math word problems with known integer answers. Covers five scenario types:

| Scenario | Example |
|----------|---------|
| Shopping | "Sarah buys 5 apples at $3 each with a $4 coupon..." |
| Distance | "A car drives at 40 mph for 3 hours, then at 55 mph for 2 hours..." |
| Time | "Tom has 5 homework assignments, each takes 20 minutes..." |
| Sharing | "There are 60 stickers to share among 4 friends..." |
| Savings | "Emma saves $25 per week for 8 weeks..." |

```python
from raptor.experiments import generate_gsm8k_synthetic

tasks = generate_gsm8k_synthetic(100)
print(tasks[0].question)
# "Sarah goes to the store and buys 4 oranges at $7 each..."
print(tasks[0].ground_truth)
# "19"
```

### `generate_math_synthetic(n, seed=42)`

Generates harder MATH-style problems covering:

| Category | Example |
|----------|---------|
| Algebra | "Solve for x: 5x + 12 = 42" |
| Geometry | "A rectangle has base 14 and height 9. What is its area?" |
| Number theory | "What is the sum of all positive divisors of 36?" |
| Combinatorics | "How many ways can you choose 3 items from 7?" |
| Sequences | "Sum of the first 10 terms of an arithmetic sequence with a₁=3, d=4" |

```python
from raptor.experiments import generate_math_synthetic

tasks = generate_math_synthetic(50)
```

### `generate_hotpotqa_synthetic(n, seed=42)`

Generates HotpotQA-style multi-hop reasoning questions:

| Type | Example |
|------|---------|
| Comparison | "New York has population 8,336,817. Chicago has 2,693,976. Which is larger?" |
| Bridge | "Albert Einstein was born in Germany and contributed to physics. What country was he born in?" |
| Intersection | "What do Python and Java have in common?" |
| Temporal | "World War I started in 1914 and the Moon landing was in 1969. How many years apart?" |

```python
from raptor.experiments import generate_hotpotqa_synthetic

tasks = generate_hotpotqa_synthetic(50)
```

---

## Running Experiments

### `run_experiment(config, tasks, agents, on_result=None)`

Runs the evaluation loop over tasks using the specified baseline mode.

**Parameters:**
- `config` — `ExperimentConfig` specifying mode, samples, etc.
- `tasks` — List of `BenchmarkTask` to evaluate
- `agents` — List of `ReasoningAgent` instances
- `on_result` — Optional callback `(result: EvalResult, index: int) → None`

**Returns:** List of `EvalResult`, one per task (up to `n_samples`).

```python
from raptor.experiments import run_experiment, ExperimentConfig, BaselineMode
from raptor.agents import MockReasoningAgent

# Create agents
agents = [
    MockReasoningAgent(
        agent_id=f"agent-{i}",
        reasoning_steps=["Let me think...", "The answer is 42"],
        final_answer="42",
    )
    for i in range(5)
]

# Run experiment
config = ExperimentConfig(
    baseline_mode=BaselineMode.SELF_CONSISTENCY,
    n_samples=50,
)
results = run_experiment(config, tasks, agents)
```

### Progress Tracking

Use the `on_result` callback for progress reporting:

```python
def progress(result: EvalResult, idx: int):
    status = "✓" if result.is_correct else "✗"
    print(f"  [{idx+1}/{len(tasks)}] {status} {result.task_id}")

results = run_experiment(config, tasks, agents, on_result=progress)
```

---

## Computing Metrics

### `compute_metrics(results, n_bins=10)`

Computes aggregated evaluation metrics from experiment results.

**Returns:** `ExperimentMetrics` with:

| Metric | Description |
|--------|-------------|
| `accuracy` | Fraction of tasks answered correctly |
| `ece` | Expected Calibration Error (lower is better) |
| `avg_rerolls` | Mean rerolls per task |
| `avg_steps` | Mean orchestration steps per task |
| `total_cost` | Sum of cost estimates across all tasks |
| `n_tasks` | Number of tasks evaluated |
| `n_correct` | Number of correct predictions |

```python
from raptor.experiments import compute_metrics

metrics = compute_metrics(results)
print(f"Accuracy: {metrics.accuracy:.1%}")
print(f"ECE: {metrics.ece:.4f}")
print(f"Avg rerolls: {metrics.avg_rerolls:.2f}")
print(f"Avg steps: {metrics.avg_steps:.2f}")
print(f"Total cost: {metrics.total_cost:.2f}")
```

### Expected Calibration Error (ECE)

ECE measures how well confidence scores match actual accuracy. It bins predictions by confidence, then computes the weighted average gap between bin accuracy and bin confidence:

```
ECE = Σ_b (|B_b| / n) · |acc(B_b) − conf(B_b)|
```

- **ECE = 0**: Perfect calibration (confidence matches accuracy exactly)
- **ECE > 0.1**: Poor calibration (confidence is misleading)

---

## Hyperparameter Sweeps

### `sweep_configs(base_config, param_grid)`

Generates `ExperimentConfig` objects for every combination of parameter values (Cartesian product).

**Supported parameters:**

| Key | Maps to | Type |
|-----|---------|------|
| `baseline_mode` | `ExperimentConfig.baseline_mode` | `BaselineMode` |
| `n_agents` | `ExperimentConfig.n_agents` | `int` |
| `max_rerolls` | `raptor_config.max_rerolls` | `int` |
| `switch_threshold` | `raptor_config.utility.switch_threshold` | `float` |
| `monotonicity_threshold` | `raptor_config.entropy.monotonicity_threshold` | `float` |
| `gain_weight` | `raptor_config.utility.weights["gain"]` | `float` |
| `confidence_weight` | `raptor_config.utility.weights["confidence"]` | `float` |
| `cost_penalty_weight` | `raptor_config.utility.weights["cost_penalty"]` | `float` |
| `redundancy_penalty_weight` | `raptor_config.utility.weights["redundancy_penalty"]` | `float` |
| `severity_weight` | `raptor_config.utility.weights["severity"]` | `float` |

**Example:**

```python
from raptor.experiments import sweep_configs, ExperimentConfig

base = ExperimentConfig(n_samples=50)

configs = sweep_configs(base, {
    "baseline_mode": [BaselineMode.SELF_CONSISTENCY, BaselineMode.RAPTOR_FULL],
    "n_agents": [3, 5, 7],
    "max_rerolls": [1, 3],
})
# Produces 2 × 3 × 2 = 12 configs

for cfg in configs:
    print(cfg.label)
    # "baseline_mode=BaselineMode.SELF_CONSISTENCY, n_agents=3, max_rerolls=1"
```

---

## Experiment Reports

### `ExperimentReport`

Aggregates results from multiple experiment runs into a comparison table.

```python
from raptor.experiments import ExperimentReport

report = ExperimentReport()

for cfg in configs:
    results = run_experiment(cfg, tasks, agents)
    metrics = compute_metrics(results)
    report.add_result(cfg, metrics)
```

### Markdown Output

```python
print(report.to_markdown())
```

Produces:

```markdown
# RAPTOR Experiment Report

| Config | Mode | Agents | Accuracy | ECE | Avg Rerolls | Avg Steps | Total Cost |
|--------|------|--------|----------|-----|-------------|-----------|------------|
| naive  | naive | 5     | 0.6200   | 0.1200 | 0.00     | 1.00      | 50.00      |
| raptor_full | raptor_full | 5 | 0.8400 | 0.0350 | 1.20 | 2.80 | 350.00 |

## Summary
- **Best accuracy:** raptor_full (0.8400)
- **Lowest cost:** naive (50.00)
- **Best calibration (lowest ECE):** raptor_full (0.0350)
```

### JSON/Dict Output

```python
json_str = report.to_json(indent=2)
data = report.to_dict()  # {"experiments": [...]}
```

---

## Complete Example

A full workflow: generate data, run all four baselines, compute metrics, compare.

```python
from raptor.agents import MockReasoningAgent
from raptor.experiments import (
    generate_gsm8k_synthetic,
    run_experiment,
    compute_metrics,
    ExperimentConfig,
    ExperimentReport,
    BaselineMode,
)

# 1. Generate synthetic tasks
tasks = generate_gsm8k_synthetic(50)

# 2. Create mock agents (in real usage, use OpenAIAgent or similar)
agents = [
    MockReasoningAgent(
        agent_id=f"agent-{i}",
        reasoning_steps=[
            "First, I identify the values in the problem.",
            "Next, I compute the intermediate result.",
            "Finally, I arrive at the answer.",
        ],
        final_answer="42",
    )
    for i in range(5)
]

# 3. Run all baselines
report = ExperimentReport()

for mode in BaselineMode:
    config = ExperimentConfig(
        baseline_mode=mode,
        n_agents=5,
        n_samples=50,
        label=mode.value,
    )
    results = run_experiment(config, tasks, agents)
    metrics = compute_metrics(results)
    report.add_result(config, metrics)

    print(f"{mode.value:20s}  acc={metrics.accuracy:.2%}  "
          f"ECE={metrics.ece:.4f}  cost={metrics.total_cost:.1f}")

# 4. Print comparison table
print(report.to_markdown())

# 5. Export to JSON
with open("experiment_results.json", "w") as f:
    f.write(report.to_json(indent=2))
```

---

## Cost Model

Cost estimates are in arbitrary units, roughly proportional to token consumption:

| Action | Cost per event |
|--------|---------------|
| Poll (per agent) | 1.0 |
| Reroll | 0.5 |
| Verify | 0.3 |
| Retrieve | 0.1 |
| Escalate | 2.0 |

Total cost varies by baseline mode:
- **NAIVE**: 1.0 (single agent call)
- **SELF_CONSISTENCY**: `n_agents × 1.0`
- **DISCOUQ_ONLY**: `n_agents × 1.0 + n_rerolls × 0.5`
- **RAPTOR_FULL**: `n_agents × 1.0 × n_steps + n_rerolls × n_agents × 0.5`

---

## Tips

- **Reproducibility**: All synthetic generators accept a `seed` parameter. Use the same seed for comparable results.
- **Real data**: Replace `generate_*_synthetic()` with real dataset loaders. Just produce `BenchmarkTask` objects.
- **Custom agents**: Any object with a `generate(prompt: str) → AgentResponse` method works. See `examples/custom_agent.py`.
- **Sweeps at scale**: `sweep_configs()` can produce hundreds of configs. Consider parallelizing with `multiprocessing` or distributing across machines.
- **Signal analysis**: Access per-task `signals_history` in `EvalResult` for fine-grained analysis of where RAPTOR's orchestrator made decisions.
