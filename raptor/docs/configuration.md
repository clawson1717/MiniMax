# Configuration Reference

RAPTOR is configured via nested Python dataclasses. This document describes every parameter, its default value, and its effect on system behavior.

## Quick Start

```python
from raptor import Config
from raptor.config import EntropyConfig, DisagreementConfig, UtilityConfig

# Default configuration — works well for most use cases
config = Config()

# Customized configuration
config = Config(
    entropy=EntropyConfig(monotonicity_threshold=1.0),
    disagreement=DisagreementConfig(mode="learn", n_agents=5),
    utility=UtilityConfig(switch_threshold=0.1),
    max_rerolls=3,
    max_steps=20,
)
```

---

## Top-Level: `Config`

The main configuration object containing all sub-configurations.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `disagreement` | `DisagreementConfig` | `DisagreementConfig()` | Disagreement monitoring settings |
| `entropy` | `EntropyConfig` | `EntropyConfig()` | Entropy trajectory settings |
| `utility` | `UtilityConfig` | `UtilityConfig()` | Utility-guided action selection |
| `max_rerolls` | `int` | `3` | Maximum number of reroll attempts per session |
| `max_steps` | `int` | `20` | Maximum orchestration steps before forced termination |
| `log_signal_history` | `bool` | `True` | Enable JSONL signal logging to disk |
| `log_dir` | `str` | `"raptor_logs"` | Directory for session log files |
| `model_name` | `str` | `"gpt-4o"` | Default model name (informational) |
| `temperature` | `float` | `0.7` | Default sampling temperature |
| `top_p` | `float` | `0.9` | Nucleus sampling parameter |

### Tuning Guidance

- **`max_rerolls`**: Higher values improve accuracy on hard problems at the cost of latency/tokens. 2–5 is typical.
- **`max_steps`**: Safety limit. Most sessions resolve in 1–3 steps. Set higher (20+) for complex multi-hop reasoning.
- **`log_signal_history`**: Disable for production to save disk I/O. Essential during development.

---

## Entropy Configuration: `EntropyConfig`

Controls entropy trajectory monitoring — the internal coherence signal.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `monotonicity_threshold` | `float` | `1.0` | Strictness of monotonicity check |
| `slope_window` | `int` | `3` | Window size for slope computation |
| `token_level` | `bool` | `True` | Per-token vs per-step entropy |
| `vocab_size` | `int` | `128_000` | Vocabulary size for calibration |

### Parameter Details

#### `monotonicity_threshold` (default: 1.0)

Controls how strictly monotonicity is enforced:
- `1.0` (default, strict): Entropy must decrease at **every** consecutive step. This matches the paper's finding that strict monotonicity is the strongest predictor.
- `< 1.0` (relaxed): Allows some fraction of non-decreasing steps. E.g., `0.8` means 80% of steps must decrease.

**Recommendation:** Keep at 1.0 unless you have evidence that relaxed thresholds work better for your domain.

#### `slope_window` (default: 3)

Not currently used in the main signal computation (OLS slope uses the full trajectory). Reserved for future windowed slope variants.

#### `token_level` (default: True)

- `True`: Compute entropy per token (higher resolution, more data points)
- `False`: Compute entropy per reasoning step (coarser, but simpler)

#### `vocab_size` (default: 128,000)

Used for log-probability calibration. Set to match your model's vocabulary:
- GPT-4/GPT-4o: ~128,000
- Claude: ~100,000
- Llama-3: ~128,256

---

## Disagreement Configuration: `DisagreementConfig`

Controls the DiscoUQ-style disagreement monitoring — the external diversity signal.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mode` | `str` | `"embed"` | Feature extraction mode |
| `n_agents` | `int` | `5` | Expected number of agents |
| `embedding_model` | `str` | `"sentence-transformers/all-MiniLM-L6-v2"` | Embedding model name |
| `use_linguistic_features` | `bool` | `True` | Enable linguistic feature extraction |
| `use_embedding_features` | `bool` | `True` | Enable embedding geometry features |
| `feature_weights` | `dict[str, float] \| None` | `None` | Custom weights (overrides defaults) |

### Parameter Details

#### `mode` (default: "embed")

Determines which features contribute to confidence calibration:

| Mode | Features Used | Best For |
|------|---------------|----------|
| `"llm"` | Linguistic only (evidence overlap, argument strength, divergence depth) | When embeddings aren't available |
| `"embed"` | Embedding geometry only (dispersion, cohesion) | Default — good balance |
| `"learn"` | All features combined | Maximum signal, highest accuracy |

**Recommendation:** Start with `"embed"` (default). Switch to `"learn"` if you need maximum calibration quality and can tolerate slightly more computation.

#### `n_agents` (default: 5)

The expected number of agents. Used primarily for configuration — the actual number polled can differ.

**Effect on signal quality:**
- 3 agents: Minimal ensemble. Works but noisier disagreement signals.
- 5 agents: Good balance of signal quality and cost.
- 7+ agents: Diminishing returns on signal quality, linear cost increase.

#### `feature_weights` (default: None)

Override default feature weights for calibration. Default weights:

```python
{
    "evidence_overlap": 2.0,
    "argument_strength": 1.5,
    "divergence_depth": -0.3,
    "dispersion": -2.5,
    "cohesion": 2.0,
}
```

Positive weights → higher value increases confidence. Negative weights → higher value decreases confidence.

A `"bias"` key is also supported (default: -1.5) and shifts the sigmoid center.

---

## Utility Configuration: `UtilityConfig`

Controls the utility-guided action selection engine.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `weights` | `dict[str, float]` | See below | Feature function weights |
| `action_costs` | `dict[OrchestrationAction, float]` | See below | Base cost per action |
| `switch_threshold` | `float` | `0.05` | Minimum margin to switch actions |
| `hysteresis_steps` | `int` | `1` | Consecutive proposals before action switch |

### Feature Weights (default)

```python
{
    "gain": 0.30,                # Expected accuracy improvement
    "confidence": 0.25,          # Combined calibrated confidence
    "cost_penalty": -0.15,       # Token/latency cost penalty
    "redundancy_penalty": -0.10, # Penalty for redundant actions
    "severity": 0.20,            # Error severity assessment
}
```

**Tuning guidance:**

| Weight | Increase to... | Decrease to... |
|--------|----------------|----------------|
| `gain` | Prioritize accuracy improvement | Reduce aggressiveness |
| `confidence` | Trust the calibration more | Rely more on other signals |
| `cost_penalty` | (more negative) Penalize expensive actions more | Allow more rerolls/escalations |
| `redundancy_penalty` | (more negative) Strongly penalize repeated actions | Allow more retries |
| `severity` | React more strongly to danger signals | Be more tolerant of uncertainty |

### Action Costs (default)

```python
{
    OrchestrationAction.RESPOND: 0.0,    # Free — just return the answer
    OrchestrationAction.REROLL: 0.5,     # Moderate — regenerate chain(s)
    OrchestrationAction.VERIFY: 0.3,     # Light — one verification pass
    OrchestrationAction.ESCALATE: 1.0,   # Expensive — stronger model/human
    OrchestrationAction.RETRIEVE: 0.4,   # Moderate — RAG retrieval
    OrchestrationAction.STOP: 0.0,       # Free — just stop
}
```

Costs are in arbitrary units and scaled by step number (later steps amplify costs to encourage termination).

### `switch_threshold` (default: 0.05)

The minimum utility margin the best action must have over the current action to trigger a switch.

- `0.0`: Switch immediately when any other action has higher utility
- `0.05` (default): Small margin prevents jitter on close calls
- `0.1+`: Conservative — requires strong evidence to change course

### `hysteresis_steps` (default: 1)

Number of consecutive steps a different action must be proposed before the switch takes effect.

- `1` (default): Immediate switch (threshold hysteresis still applies)
- `2+`: Requires sustained signal — prevents single-step noise from triggering changes

---

## Common Configuration Recipes

### High Accuracy (aggressive rerolling)

```python
config = Config(
    max_rerolls=5,
    max_steps=15,
    utility=UtilityConfig(
        weights={"gain": 0.40, "confidence": 0.20, "cost_penalty": -0.05,
                 "redundancy_penalty": -0.05, "severity": 0.30},
    ),
)
```

### Low Cost (conservative)

```python
config = Config(
    max_rerolls=1,
    max_steps=5,
    utility=UtilityConfig(
        weights={"gain": 0.20, "confidence": 0.30, "cost_penalty": -0.25,
                 "redundancy_penalty": -0.15, "severity": 0.10},
    ),
)
```

### High Reliability (safety-critical)

```python
config = Config(
    max_rerolls=3,
    max_steps=20,
    utility=UtilityConfig(
        weights={"gain": 0.25, "confidence": 0.15, "cost_penalty": -0.05,
                 "redundancy_penalty": -0.05, "severity": 0.50},
        switch_threshold=0.1,
        hysteresis_steps=2,
    ),
    disagreement=DisagreementConfig(mode="learn"),
)
```

### Fast Prototyping (minimal overhead)

```python
config = Config(
    max_rerolls=1,
    max_steps=3,
    log_signal_history=False,
    disagreement=DisagreementConfig(n_agents=3),
)
```
