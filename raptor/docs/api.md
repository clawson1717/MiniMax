# API Reference

Complete API documentation for all public classes and functions in RAPTOR, organized by module.

---

## `raptor.config` — Configuration

### `OrchestrationAction` (Enum)

Actions the RAPTOR orchestrator can select.

| Value | Description |
|-------|-------------|
| `RESPOND` | Accept current answer and return |
| `REROLL` | Regenerate reasoning chain(s) |
| `VERIFY` | Run a verification step |
| `ESCALATE` | Switch to stronger model or flag for human review |
| `RETRIEVE` | Trigger RAG retrieval for additional context |
| `STOP` | Halt reasoning — insufficient confidence to proceed |

### `Config`

Top-level RAPTOR configuration dataclass.

```python
@dataclass
class Config:
    disagreement: DisagreementConfig   # Disagreement monitoring settings
    entropy: EntropyConfig             # Entropy trajectory settings
    utility: UtilityConfig             # Utility-guided action selection settings
    max_rerolls: int = 3               # Maximum reroll attempts
    max_steps: int = 20                # Maximum orchestration steps per session
    log_signal_history: bool = True    # Enable JSONL signal logging
    log_dir: str = "raptor_logs"       # Directory for session log files
    model_name: str = "gpt-4o"        # Default model name
    temperature: float = 0.7          # Sampling temperature
    top_p: float = 0.9                # Nucleus sampling parameter
```

### `EntropyConfig`

Configuration for entropy trajectory monitoring.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `monotonicity_threshold` | `float` | `1.0` | Strictness threshold (1.0 = every step must decrease) |
| `slope_window` | `int` | `3` | Window size for computing entropy slope |
| `token_level` | `bool` | `True` | Compute entropy per token vs per step |
| `vocab_size` | `int` | `128_000` | Vocabulary size for log-prob calibration |

### `DisagreementConfig`

Configuration for DiscoUQ-style disagreement monitoring.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mode` | `str` | `"embed"` | Feature mode: `"llm"`, `"embed"`, or `"learn"` |
| `n_agents` | `int` | `5` | Number of agents in the ensemble |
| `embedding_model` | `str` | `"sentence-transformers/all-MiniLM-L6-v2"` | Embedding model name |
| `use_linguistic_features` | `bool` | `True` | Enable linguistic feature extraction |
| `use_embedding_features` | `bool` | `True` | Enable embedding geometry features |
| `feature_weights` | `dict[str, float] \| None` | `None` | Custom feature weights (overrides defaults) |

### `UtilityConfig`

Configuration for utility-guided action selection.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `weights` | `dict[str, float]` | `{"gain": 0.30, "confidence": 0.25, ...}` | Feature function weights |
| `action_costs` | `dict[OrchestrationAction, float]` | `{RESPOND: 0.0, REROLL: 0.5, ...}` | Cost per action type |
| `switch_threshold` | `float` | `0.05` | Minimum margin to switch actions |
| `hysteresis_steps` | `int` | `1` | Consecutive steps before action switch |

---

## `raptor.entropy_tracker` — Entropy Trajectory Monitoring

### `TrajectorySignal`

Signal emitted by the Entropy Tracker at each reasoning step.

| Field | Type | Description |
|-------|------|-------------|
| `n_steps` | `int` | Number of steps in the trajectory |
| `monotonicity` | `bool` | `True` if entropy decreased at every step |
| `entropy_slope` | `float` | Linear slope of entropy trajectory (negative = good) |
| `final_entropy` | `float` | Entropy value at the last step |
| `entropies` | `list[float]` | Full trajectory for visualization |
| `confidence_score` | `float` | 0–1 confidence derived from monotonicity + slope |

### `EntropyTracker`

Monitors entropy trajectory shape during chain-of-thought reasoning.

```python
class EntropyTracker:
    def __init__(self, config: EntropyConfig) -> None: ...
    def update(self, log_probs: np.ndarray) -> None: ...
    def update_from_step(self, step_token: str, step_log_probs: np.ndarray) -> None: ...
    def compute_signal(self) -> TrajectorySignal: ...
    def reset(self) -> None: ...
```

**Methods:**

| Method | Description |
|--------|-------------|
| `update(log_probs)` | Feed a new token's log-probability distribution into the tracker |
| `update_from_step(step_token, step_log_probs)` | Update from a reasoning step's output distribution |
| `compute_signal()` | Compute and return the `TrajectorySignal` from current trajectory |
| `reset()` | Clear the trajectory for a new reasoning chain |

---

## `raptor.disagreement` — Disagreement Monitoring

### `AgentResponse`

A single agent's response to a prompt.

| Field | Type | Description |
|-------|------|-------------|
| `agent_id` | `str` | Unique agent identifier |
| `reasoning_steps` | `list[str]` | Intermediate reasoning steps |
| `final_answer` | `str` | The agent's final answer |
| `embedding` | `np.ndarray \| None` | Optional pre-computed response embedding |

### `DisagreementSignal`

Signal emitted by the Disagreement Monitor.

| Field | Type | Description |
|-------|------|-------------|
| `evidence_overlap` | `float` | 0–1 fraction of shared evidence across agents |
| `argument_strength` | `float` | 0–1 average strength of each agent's argument |
| `divergence_depth` | `int` | Step index where agents first diverge |
| `dispersion` | `float` | Spread of agent response embeddings (std dev) |
| `cohesion` | `float` | Intra-cluster closeness (avg pairwise cosine sim) |
| `confidence_score` | `float` | 0–1 calibrated confidence from logistic model |
| `disagreement_tier` | `str` | `"low"`, `"medium"`, or `"weak"` |

### `DisagreementMonitor`

Monitors multi-agent disagreement using DiscoUQ-style feature extraction.

```python
class DisagreementMonitor:
    def __init__(self, config: DisagreementConfig) -> None: ...
    def compute_signal(
        self, responses: list[AgentResponse], mode: str | None = None
    ) -> DisagreementSignal: ...
```

**Methods:**

| Method | Description |
|--------|-------------|
| `compute_signal(responses, mode=None)` | Compute disagreement signal from multi-agent responses. `mode` overrides config: `"llm"`, `"embed"`, or `"learn"`. |

---

## `raptor.utility` — Utility Score Engine

### `ActionScore`

Utility score for a single action.

| Field | Type | Description |
|-------|------|-------------|
| `action` | `OrchestrationAction` | The action being scored |
| `utility` | `float` | Computed utility score |
| `breakdown` | `dict[str, float]` | Per-feature (φₖ) contributions |

### `UtilityEngine`

Computes utility scores for all actions given the fused signal vector.

```python
class UtilityEngine:
    def __init__(self, config: Config, mode: str = "fixed") -> None: ...
    def score_all(
        self,
        traj_signal: TrajectorySignal,
        disa_signal: DisagreementSignal,
        step_context: dict | None = None,
    ) -> list[ActionScore]: ...
    def select_best(
        self,
        scores: list[ActionScore],
        previous_action: OrchestrationAction | None = None,
    ) -> OrchestrationAction: ...
    def update_weights(
        self, features: dict, reward: float, predicted: float, learning_rate: float = 0.01
    ) -> None: ...
```

**Methods:**

| Method | Description |
|--------|-------------|
| `score_all(traj_signal, disa_signal, step_context)` | Compute utility scores for all 6 actions, sorted descending |
| `select_best(scores, previous_action)` | Select highest-utility action with hysteresis |
| `update_weights(features, reward, predicted, lr)` | Online SGD weight update (learned mode only) |

**Properties:**

| Property | Type | Description |
|----------|------|-------------|
| `weights` | `dict[str, float]` | Current feature weights |
| `mode` | `str` | `"fixed"` or `"learned"` |

**Feature Functions (φₖ):**

| Function | Description |
|----------|-------------|
| `φ_gain` | Expected accuracy improvement for each action given signal state |
| `φ_confidence` | Combined calibrated confidence from trajectory + disagreement |
| `φ_cost` | Step cost (base action cost scaled by step number) |
| `φ_redundancy` | Penalty for already-taken actions (duplicate retrieve, exceeded rerolls) |
| `φ_severity` | Error severity — how bad is getting it wrong in this signal state |

---

## `raptor.orchestrator` — RAPTOR Orchestrator

### `SignalVector`

Fused RAPTOR signal vector S(t).

| Field | Type | Description |
|-------|------|-------------|
| `monotonicity_flag` | `bool` | Whether entropy trajectory is monotone |
| `entropy_slope` | `float` | Linear slope of entropy trajectory |
| `disagreement_score` | `float` | 1 − confidence_score (higher = more disagreement) |
| `dispersion_score` | `float` | Embedding cluster spread |
| `cohesion_score` | `float` | Intra-cluster closeness |
| `divergence_depth` | `int` | Step index where agents diverged |

### `OrchestrationDecision`

Full output of the orchestrator for one step.

| Field | Type | Description |
|-------|------|-------------|
| `action` | `OrchestrationAction` | Selected action |
| `utility_score` | `float` | Utility score of the selected action |
| `action_breakdown` | `dict[str, float]` | Feature contributions |
| `traj_signal` | `TrajectorySignal` | Entropy trajectory signal |
| `disa_signal` | `DisagreementSignal` | Disagreement signal |
| `signal_vector` | `SignalVector` | Fused signal vector |
| `all_scores` | `list[ActionScore]` | Utility scores for all actions |
| `reason` | `str` | Human-readable explanation |

### `RAPTOROrchestrator`

Main RAPTOR controller — fuses signals and selects actions.

```python
class RAPTOROrchestrator:
    def __init__(self, config: Config) -> None: ...
    def step(
        self, prompt: str, agent_responses: list[AgentResponse],
        step_context: dict | None = None,
    ) -> OrchestrationDecision: ...
    def reroll_with_selection(
        self, agent_responses: list[AgentResponse], n_candidates: int = 3,
    ) -> AgentResponse: ...
    def reset(self) -> None: ...
```

**Methods:**

| Method | Description |
|--------|-------------|
| `step(prompt, agent_responses, step_context)` | Run one orchestration step: signals → utility → action |
| `reroll_with_selection(agent_responses, n_candidates)` | Select best candidate via entropy trajectory quality |
| `reset()` | Reset state for a new reasoning session |

**Properties:**

| Property | Type | Description |
|----------|------|-------------|
| `state` | `OrchestratorState` | Current internal state |
| `session_id` | `str` | Unique session identifier |

---

## `raptor.agents` — Agent Protocols & Polling

### `ReasoningAgent` (Protocol)

Abstract interface for reasoning-capable LLM agents. Runtime-checkable.

```python
@runtime_checkable
class ReasoningAgent(Protocol):
    def generate(self, prompt: str) -> AgentResponse: ...
```

### `StreamingReasoningAgent` (Protocol)

Extended agent protocol supporting streaming token-by-token output.

```python
@runtime_checkable
class StreamingReasoningAgent(Protocol):
    @property
    def agent_id(self) -> str: ...
    def stream(self, prompt: str) -> Iterator[AgentStreamEvent]: ...
```

### `AgentStreamEvent`

Event emitted by streaming agents.

| Field | Type | Description |
|-------|------|-------------|
| `event_type` | `StreamEventType` | `TOKEN`, `STEP_COMPLETE`, or `DONE` |
| `token` | `str \| None` | Token text (for TOKEN events) |
| `token_log_probs` | `TokenLogProbs \| None` | Log-probability data |
| `step_text` | `str \| None` | Full step text (for STEP_COMPLETE) |
| `step_index` | `int \| None` | Step index (0-based) |
| `final_answer` | `str \| None` | Final answer (for DONE) |
| `agent_id` | `str \| None` | Agent identifier |

### `PollResult`

Result of `poll_agents()`.

| Field | Type | Description |
|-------|------|-------------|
| `responses` | `list[AgentResponse]` | Successful responses |
| `errors` | `list[tuple[int, AgentError]]` | Failed agent indices and errors |
| `n_success` | `int` | Number of successful responses |
| `n_errors` | `int` | Number of errors |
| `all_succeeded` | `bool` | True if no errors |

### Functions

#### `poll_agents(agents, prompt, timeout=None, max_workers=None, fail_fast=False) → PollResult`

Poll multiple agents concurrently using a thread pool.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `agents` | `list[ReasoningAgent]` | — | Agents to poll |
| `prompt` | `str` | — | Prompt to send |
| `timeout` | `float \| None` | `None` | Max seconds to wait |
| `max_workers` | `int \| None` | `None` | Thread pool size (defaults to len(agents)) |
| `fail_fast` | `bool` | `False` | Raise on first failure |

#### `poll_agents_streaming(agents, prompt, on_event=None, timeout=None, max_workers=None) → PollResult`

Poll streaming agents concurrently, invoking `on_event` for each token.

#### `reroll(agent, prompt, n_candidates=3, timeout=None, concurrent=True) → list[AgentResponse]`

Generate multiple candidates from a single agent.

### Mock Agents (for Testing)

| Class | Description |
|-------|-------------|
| `MockReasoningAgent` | Returns pre-configured responses |
| `MockStreamingAgent` | Yields pre-configured tokens as stream events |
| `MockFailingAgent` | Always raises RuntimeError |
| `MockSlowAgent` | Sleeps before responding (for timeout testing) |
| `MockVariableAgent` | Returns different responses on successive calls |

---

## `raptor.integration` — LLM Adapters & Entry Point

### `OpenAIAgent`

ReasoningAgent adapter for OpenAI Chat Completions API.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `str` | `"gpt-4o"` | Model name |
| `api_key` | `str` | `""` | OpenAI API key |
| `agent_id` | `str \| None` | `None` | Agent identifier |
| `base_url` | `str` | `"https://api.openai.com"` | API base URL |
| `temperature` | `float` | `0.7` | Sampling temperature |
| `max_tokens` | `int` | `2048` | Max tokens to generate |
| `system_prompt` | `str \| None` | `None` | System message |
| `request_logprobs` | `bool` | `False` | Request log probabilities |
| `http_client` | `HttpClient \| None` | `None` | Injectable HTTP client |

### `AnthropicAgent`

ReasoningAgent adapter for Anthropic Messages API. Same parameter pattern as OpenAIAgent.

### `LocalLLMAgent`

ReasoningAgent adapter for local OpenAI-compatible endpoints (vLLM, llama.cpp, Ollama, LMStudio).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `endpoint` | `str` | `"http://localhost:8000"` | Full base URL |
| `model` | `str` | `"local-model"` | Model name/path |

### `RAPTORContext`

Serializable context object for a RAPTOR reasoning session.

| Method | Description |
|--------|-------------|
| `add_entry(role, content, action)` | Append to conversation history |
| `add_action(action)` | Record an orchestration action |
| `add_decision(decision)` | Record a full decision with signals |
| `to_dict()` | Serialize to JSON-compatible dict |
| `to_json(**kwargs)` | Serialize to JSON string |
| `from_dict(data, config)` | Deserialize from dict |

### `RAPTORResult`

Return type from `run_with_raptor()`.

| Field | Type | Description |
|-------|------|-------------|
| `final_answer` | `str` | Best answer from RAPTOR pipeline |
| `context` | `RAPTORContext` | Full session context |
| `steps_taken` | `int` | Number of orchestration steps |
| `final_action` | `OrchestrationAction` | Terminal action |
| `all_responses` | `list[AgentResponse]` | All collected responses |
| `escalated` | `bool` | True if ended via ESCALATE |
| `stopped` | `bool` | True if ended via STOP |

### `run_with_raptor(agents, prompt, config=None, max_steps=None, poll_timeout=None, on_decision=None, verify_prompt_fn=None) → RAPTORResult`

High-level entry point that runs the full RAPTOR pipeline.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `agents` | `list[ReasoningAgent]` | — | Agents to poll |
| `prompt` | `str` | — | Reasoning question/task |
| `config` | `Config \| None` | `None` | RAPTOR configuration |
| `max_steps` | `int \| None` | `None` | Override max steps |
| `poll_timeout` | `float \| None` | `None` | Per-poll timeout |
| `on_decision` | `Callable \| None` | `None` | Per-decision callback |
| `verify_prompt_fn` | `Callable \| None` | `None` | Custom verification prompt builder |

---

## `raptor.experiments` — Evaluation Harness

### `BaselineMode` (Enum)

| Value | Description |
|-------|-------------|
| `NAIVE` | Single agent, no RAPTOR |
| `SELF_CONSISTENCY` | Majority vote over N agents |
| `DISCOUQ_ONLY` | Disagreement monitoring only (no entropy/utility) |
| `RAPTOR_FULL` | Full RAPTOR pipeline |

### `BenchmarkTask`

| Field | Type | Description |
|-------|------|-------------|
| `question` | `str` | The question or prompt |
| `ground_truth` | `str` | Known correct answer |
| `dataset_name` | `str` | Benchmark name |
| `task_id` | `str` | Unique task identifier |
| `metadata` | `dict` | Extra data (difficulty, category) |

### `EvalResult`

| Field | Type | Description |
|-------|------|-------------|
| `task_id` | `str` | Links back to BenchmarkTask |
| `predicted` | `str` | Model's predicted answer |
| `correct` | `str` | Ground truth |
| `is_correct` | `bool` | Whether prediction matches truth |
| `n_rerolls` | `int` | Number of rerolls taken |
| `n_steps` | `int` | Number of orchestration steps |
| `cost_estimate` | `float` | Estimated cost |
| `confidence` | `float` | Calibrated confidence score |

### `ExperimentMetrics`

| Field | Type | Description |
|-------|------|-------------|
| `accuracy` | `float` | Fraction correct |
| `ece` | `float` | Expected Calibration Error |
| `avg_rerolls` | `float` | Mean rerolls per task |
| `avg_steps` | `float` | Mean steps per task |
| `total_cost` | `float` | Sum of cost estimates |

### Functions

#### `run_experiment(config, tasks, agents, on_result=None) → list[EvalResult]`

Run evaluation loop over tasks with pluggable baseline modes.

#### `compute_metrics(results, n_bins=10) → ExperimentMetrics`

Compute accuracy, ECE, reroll/step/cost statistics.

#### `sweep_configs(base_config, param_grid) → list[ExperimentConfig]`

Generate configs for Cartesian product parameter sweeps. Supported keys: `baseline_mode`, `n_agents`, `max_rerolls`, `switch_threshold`, `gain_weight`, `confidence_weight`, `cost_penalty_weight`, `redundancy_penalty_weight`, `severity_weight`, `monotonicity_threshold`.

#### Synthetic Data Generators

| Function | Description |
|----------|-------------|
| `generate_gsm8k_synthetic(n, seed=42)` | Grade-school math word problems |
| `generate_math_synthetic(n, seed=42)` | Harder math (algebra, geometry, number theory, combinatorics, sequences) |
| `generate_hotpotqa_synthetic(n, seed=42)` | Multi-hop reasoning questions |

### `ExperimentReport`

Aggregated results with comparison across configurations.

| Method | Description |
|--------|-------------|
| `add_result(config, metrics, results)` | Add one experiment's results |
| `to_markdown()` | Generate markdown comparison table |
| `to_json(**kwargs)` | Serialize to JSON |
| `to_dict()` | Return as dictionary |

---

## `raptor.dashboard` — Terminal Dashboard

### `DashboardRenderer`

Rich-based terminal renderer for RAPTOR signals.

| Method | Description |
|--------|-------------|
| `render_entropy_trajectory(entropies, monotonicity, slope, confidence)` | Entropy trajectory panel with non-monotone highlighting |
| `render_disagreement_signal(disa_signal)` | Disagreement features panel |
| `render_utility_scores(breakdown, chosen_action, chosen_utility, all_scores)` | Utility scores table |
| `render_reliability(monotonicity, disagreement_score)` | Color-coded reliability indicator |
| `render_step(record)` | Full step render (all 4 panels) |
| `render_session(records, delay=0.0)` | Sequential session replay |
| `render_session_summary(records)` | Compact summary table |

### Utility Functions

| Function | Description |
|----------|-------------|
| `load_session_log(path)` | Parse JSONL session log → `list[StepRecord]` |
| `list_session_logs(directory)` | List all session log files |
| `classify_reliability(monotonicity, disagreement_score)` | GREEN / YELLOW / RED classification |

### CLI

```bash
raptor-dashboard replay <logfile> [--delay 1.0]
raptor-dashboard step <logfile> --step N
raptor-dashboard summary <logfile>
raptor-dashboard list <directory>
```
