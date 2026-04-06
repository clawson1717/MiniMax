# Tutorial — Getting Started with RAPTOR

This tutorial walks you through RAPTOR's key features, from basic usage to running full experiments.

## Prerequisites

```bash
pip install raptor-llm
# Or from source:
git clone https://github.com/YOUR_HANDLE/raptor.git
cd raptor && pip install -e ".[dev]"
```

## 1. Understanding the Basics

RAPTOR monitors LLM reasoning quality in real time. It answers the question: "Should I trust this answer, or should I try again?"

The core loop is:
1. Ask multiple agents the same question
2. Analyze their reasoning chains (entropy trajectory) and disagreement
3. Decide what to do: respond, reroll, verify, escalate, retrieve, or stop

## 2. Your First RAPTOR Session

Let's start with mock agents to understand the flow without needing API keys:

```python
from raptor import Config, RAPTOROrchestrator
from raptor.agents import MockReasoningAgent, poll_agents
from raptor.config import OrchestrationAction

# Create 5 agents that all agree (high confidence scenario)
agents = [
    MockReasoningAgent(
        agent_id=f"agent-{i}",
        reasoning_steps=[
            "Step 1: Subtract 3 from both sides: 2x = 4",
            "Step 2: Divide both sides by 2: x = 2",
        ],
        final_answer="x = 2",
    )
    for i in range(5)
]

# Create orchestrator with default config
config = Config()
orchestrator = RAPTOROrchestrator(config)

# Poll agents
result = poll_agents(agents, "Solve: 2x + 3 = 7")
print(f"Got {result.n_success} responses, {result.n_errors} errors")

# Run one orchestration step
decision = orchestrator.step("Solve: 2x + 3 = 7", result.responses)
print(f"Action: {decision.action.value}")
print(f"Utility: {decision.utility_score:.3f}")
print(f"Monotone: {decision.traj_signal.monotonicity}")
print(f"Disagreement tier: {decision.disa_signal.disagreement_tier}")
print(f"Reason: {decision.reason}")
```

When all agents agree with coherent reasoning, RAPTOR will typically select **RESPOND** — it's confident in the answer.

## 3. Simulating Disagreement

Now let's see what happens when agents disagree:

```python
from raptor import Config, RAPTOROrchestrator
from raptor.agents import MockReasoningAgent, poll_agents

# Agents with different answers and reasoning
agents = [
    MockReasoningAgent("agent-0", ["Factoring..."], "x = 2"),
    MockReasoningAgent("agent-1", ["Substitution..."], "x = 2"),
    MockReasoningAgent("agent-2", ["Graphing..."], "x = 3"),      # Wrong!
    MockReasoningAgent("agent-3", ["Algebraic..."], "x = 2"),
    MockReasoningAgent("agent-4", ["Estimation..."], "x = 2.5"),  # Wrong!
]

config = Config()
orchestrator = RAPTOROrchestrator(config)
result = poll_agents(agents, "Solve: 2x + 3 = 7")
decision = orchestrator.step("Solve: 2x + 3 = 7", result.responses)

print(f"Action: {decision.action.value}")
print(f"Disagreement tier: {decision.disa_signal.disagreement_tier}")
print(f"Evidence overlap: {decision.disa_signal.evidence_overlap:.3f}")
print(f"Cohesion: {decision.disa_signal.cohesion:.3f}")
```

With disagreeing agents, RAPTOR may select **VERIFY** or **REROLL** depending on the signal severity.

## 4. The High-Level Entry Point

For most use cases, `run_with_raptor()` handles the full loop:

```python
from raptor.integration import run_with_raptor
from raptor.agents import MockReasoningAgent
from raptor import Config

# Create agents
agents = [
    MockReasoningAgent(
        f"agent-{i}",
        ["Parse the equation", "Apply algebraic rules", "Solve for x"],
        "x = 2",
    )
    for i in range(5)
]

# Run the full pipeline
result = run_with_raptor(
    agents=agents,
    prompt="Solve: 2x + 3 = 7",
    config=Config(max_steps=5, max_rerolls=2),
)

print(f"Final answer: {result.final_answer}")
print(f"Steps taken: {result.steps_taken}")
print(f"Final action: {result.final_action.value}")
print(f"Escalated: {result.escalated}")
print(f"Stopped: {result.stopped}")
```

The loop runs until RAPTOR decides to RESPOND, ESCALATE, STOP, or hits `max_steps`.

## 5. Using a Decision Callback

Track what RAPTOR decides at each step:

```python
from raptor.integration import run_with_raptor
from raptor.orchestrator import OrchestrationDecision
from raptor.agents import MockReasoningAgent
from raptor import Config

def on_decision(decision: OrchestrationDecision, step: int) -> None:
    print(f"  Step {step}: {decision.action.value} "
          f"(U={decision.utility_score:.3f}, "
          f"mono={decision.traj_signal.monotonicity}, "
          f"tier={decision.disa_signal.disagreement_tier})")

agents = [MockReasoningAgent(f"a{i}", ["think"], "42") for i in range(5)]

result = run_with_raptor(
    agents=agents,
    prompt="What is 6 × 7?",
    config=Config(max_steps=5),
    on_decision=on_decision,
)
```

## 6. Working with Signal Vectors

Access the raw signal data for custom analysis:

```python
from raptor import Config, RAPTOROrchestrator
from raptor.agents import MockReasoningAgent, poll_agents

agents = [MockReasoningAgent(f"a{i}", ["step1", "step2"], "42") for i in range(5)]
config = Config()
orchestrator = RAPTOROrchestrator(config)

result = poll_agents(agents, "What is 6 × 7?")
decision = orchestrator.step("What is 6 × 7?", result.responses)

# Access the fused signal vector
sv = decision.signal_vector
print(f"Monotonicity: {sv.monotonicity_flag}")
print(f"Entropy slope: {sv.entropy_slope:.4f}")
print(f"Disagreement: {sv.disagreement_score:.4f}")
print(f"Dispersion: {sv.dispersion_score:.4f}")
print(f"Cohesion: {sv.cohesion_score:.4f}")
print(f"Divergence depth: {sv.divergence_depth}")

# Access utility scores for all actions
for score in decision.all_scores:
    print(f"  {score.action.value}: U={score.utility:.4f}")
    for feat, val in score.breakdown.items():
        print(f"    {feat}: {val:.4f}")
```

## 7. Implementing a Custom Agent

Any object with a `generate(prompt: str) -> AgentResponse` method works:

```python
from raptor.disagreement import AgentResponse
from raptor.agents import ReasoningAgent

class MyAgent:
    """A custom agent that wraps any LLM API."""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id

    def generate(self, prompt: str) -> AgentResponse:
        # Call your LLM here
        reasoning = [
            f"Analyzing: {prompt[:50]}...",
            "Applying domain knowledge...",
            "Computing result...",
        ]
        answer = "computed answer"

        return AgentResponse(
            agent_id=self.agent_id,
            reasoning_steps=reasoning,
            final_answer=answer,
        )

# Verify it satisfies the protocol
assert isinstance(MyAgent("test"), ReasoningAgent)
```

## 8. Using Real LLM Providers

### OpenAI

```python
from raptor.integration import run_with_raptor, OpenAIAgent
from raptor import Config

agents = [
    OpenAIAgent(
        model="gpt-4o",
        api_key="sk-...",
        agent_id=f"gpt4-{i}",
        temperature=0.7 + i * 0.05,  # Slight temperature variation for diversity
    )
    for i in range(3)
]

result = run_with_raptor(agents=agents, prompt="Explain quantum entanglement.")
```

### Anthropic

```python
from raptor.integration import AnthropicAgent

agents = [
    AnthropicAgent(
        model="claude-sonnet-4-20250514",
        api_key="sk-ant-...",
        agent_id=f"claude-{i}",
    )
    for i in range(3)
]
```

### Local Models (vLLM, Ollama, etc.)

```python
from raptor.integration import LocalLLMAgent

agents = [
    LocalLLMAgent(
        endpoint="http://localhost:8000",
        model="meta-llama/Llama-3-8B",
        agent_id=f"llama-{i}",
    )
    for i in range(5)
]
```

### Mixed Ensemble

```python
from raptor.integration import OpenAIAgent, AnthropicAgent, LocalLLMAgent

agents = [
    OpenAIAgent(model="gpt-4o", api_key="sk-...", agent_id="gpt4"),
    OpenAIAgent(model="gpt-4o-mini", api_key="sk-...", agent_id="gpt4-mini"),
    AnthropicAgent(model="claude-sonnet-4-20250514", api_key="sk-ant-...", agent_id="claude"),
    LocalLLMAgent(endpoint="http://localhost:8000", model="llama-3-8b", agent_id="llama"),
]
```

## 9. Using the Dashboard

After running RAPTOR with logging enabled (default), replay sessions in the terminal:

```bash
# List available session logs
raptor-dashboard list raptor_logs/

# Replay step by step (1 second delay between steps)
raptor-dashboard replay raptor_logs/session_20260406_120000_000000.jsonl --delay 1.0

# View a specific step
raptor-dashboard step raptor_logs/session_20260406_120000_000000.jsonl --step 3

# Compact summary table
raptor-dashboard summary raptor_logs/session_20260406_120000_000000.jsonl
```

Or use the Python API:

```python
from raptor.dashboard import DashboardRenderer, load_session_log

renderer = DashboardRenderer()
records = load_session_log("raptor_logs/session_20260406_120000_000000.jsonl")

# Render all steps
renderer.render_session(records, delay=0.5)

# Or render a single step
renderer.render_step(records[0])

# Or show a compact summary
renderer.render_session_summary(records)
```

## 10. Running Experiments

Evaluate RAPTOR against baselines using synthetic benchmarks:

```python
from raptor.experiments import (
    generate_gsm8k_synthetic,
    generate_math_synthetic,
    run_experiment,
    compute_metrics,
    ExperimentConfig,
    ExperimentReport,
    BaselineMode,
)
from raptor.agents import MockReasoningAgent

# Generate synthetic tasks
gsm8k_tasks = generate_gsm8k_synthetic(100)
math_tasks = generate_math_synthetic(50)

# Create mock agents
agents = [
    MockReasoningAgent(f"agent-{i}", ["step 1", "step 2"], "42")
    for i in range(5)
]

# Run experiments across baselines
report = ExperimentReport()

for mode in [BaselineMode.NAIVE, BaselineMode.SELF_CONSISTENCY,
             BaselineMode.DISCOUQ_ONLY, BaselineMode.RAPTOR_FULL]:
    config = ExperimentConfig(
        baseline_mode=mode,
        n_samples=100,
        label=mode.value,
    )
    results = run_experiment(config, gsm8k_tasks, agents)
    metrics = compute_metrics(results)
    report.add_result(config, metrics)

    print(f"{mode.value}: accuracy={metrics.accuracy:.2%}, "
          f"ECE={metrics.ece:.4f}, cost={metrics.total_cost:.1f}")

# Generate markdown comparison table
print(report.to_markdown())
```

See [experiments.md](experiments.md) for more details on running and interpreting experiments.

## 11. Serializing Context

RAPTOR contexts are fully serializable for persistence or replay:

```python
from raptor.integration import run_with_raptor, RAPTORContext
from raptor.agents import MockReasoningAgent
from raptor import Config
import json

agents = [MockReasoningAgent(f"a{i}", ["think"], "42") for i in range(5)]
result = run_with_raptor(agents=agents, prompt="What is 6 × 7?")

# Serialize
context_json = result.context.to_json(indent=2)
print(context_json)

# Deserialize
ctx = RAPTORContext.from_dict(json.loads(context_json))
print(f"Prompt: {ctx.prompt}")
print(f"Actions taken: {ctx.action_history}")
print(f"Signal history: {len(ctx.signal_history)} steps")
```

## Next Steps

- **[Configuration Reference](configuration.md)** — Full parameter documentation
- **[Architecture](architecture.md)** — Deep dive into how RAPTOR works internally
- **[API Reference](api.md)** — Complete API documentation
- **[Experiments](experiments.md)** — Running benchmarks and interpreting results
