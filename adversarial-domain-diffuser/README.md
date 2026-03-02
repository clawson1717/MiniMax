# Adversarial-Domain-Diffuser (ADD)

A domain-specific agentic framework that uses **Adversarial Question Generation** to identify comprehension gaps and **Online Multi-Agent Diffusion Policies** to coordinate complex, interpretative reasoning tasks in specialized fields.

## Overview

The Adversarial-Domain-Diffuser (ADD) is designed to systematically improve domain-specific reasoning by combining two key techniques:

1. **Adversarial Question Generation**: Iteratively generates challenging questions that expose gaps between a target model's understanding and expert-level knowledge.

2. **Online Multi-Agent Diffusion (OMAD)**: Coordinates multiple specialized reasoning agents using diffusion-based generative policies, allowing for multimodal reasoning actions and entropy-augmented objectives.

The framework is particularly suited for specialized domains like legal reasoning, medical diagnosis, or scientific analysis where traditional evaluation methods may miss nuanced comprehension gaps.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Adversarial Domain Diffuser                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                    IntegratedAdversarialLoop                      │   │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐                 │   │
│  │  │Iteration 1 │─▶│Iteration 2 │─▶│Iteration N │                 │   │
│  │  │ Gap: 0.85  │  │ Gap: 0.62  │  │ Gap: 0.15  │                 │   │
│  │  └──────┬─────┘  └──────┬─────┘  └──────┬─────┘                 │   │
│  │         │               │               │                        │   │
│  │         ▼               ▼               ▼                        │   │
│  │  ┌─────────────────────────────────────────────────────────┐    │   │
│  │  │              Adversarial Generator                       │    │   │
│  │  │   Generates questions targeting comprehension gaps       │    │   │
│  │  └─────────────────────────────────────────────────────────┘    │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                             │                                          │
│  ┌──────────────────────────┼──────────────────────────────────────┐   │
│  │                    AgentEnvironment                              │   │
│  │                            │                                     │   │
│  │  ┌─────────────────────────▼──────────────────────────────┐     │   │
│  │  │                    OMAD Orchestrator                    │     │   │
│  │  │         (Online Multi-Agent Diffusion)                  │     │   │
│  │  │  ┌─────────┐  ┌─────────┐  ┌─────────┐                │     │   │
│  │  │  │Diffusion│  │Diffusion│  │Diffusion│                │     │   │
│  │  │  │Policy 1 │  │Policy 2 │  │Policy 3 │                │     │   │
│  │  │  └────┬────┘  └────┬────┘  └────┬────┘                │     │   │
│  │  └───────┼─────────────┼─────────────┼────────────────────┘     │   │
│  │          │             │             │                          │   │
│  │  ┌───────▼─────────────▼─────────────▼──────────────────────┐  │   │
│  │  │              Embodiment-based Grouper                     │  │   │
│  │  │   Clusters agents by cognitive morphology                 │  │   │
│  │  └──────────────────────────────────────────────────────────┘  │   │
│  │                            │                                     │   │
│  │  ┌─────────────────────────▼──────────────────────────────┐     │   │
│  │  │                  Reasoning Agents                        │     │   │
│  │  │  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐        │     │   │
│  │  │  │ Legal  │  │Medical │  │Physics │  │Ethics  │        │     │   │
│  │  │  └────────┘  └────────┘  └────────┘  └────────┘        │     │   │
│  │  └─────────────────────────────────────────────────────────┘     │   │
│  │                            │                                     │   │
│  │  ┌─────────────────────────▼──────────────────────────────┐     │   │
│  │  │                     Blackboard                          │     │   │
│  │  │              (Shared Knowledge Store)                    │     │   │
│  │  └─────────────────────────────────────────────────────────┘     │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                        Evaluator                                  │   │
│  │      Domain-specific benchmark evaluation (LegalBench, etc.)      │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Adversarial Generator (`src/adversarial_gen.py`)

The adversarial question generator creates semantically challenging questions based on the gap between a target model's response and expert-level knowledge.

```python
from src.adversarial_gen import AdversarialGenerator

generator = AdversarialGenerator(model_client=your_llm_client)

# Generate a challenging question
question = generator.generate_question(
    original_prompt="Explain contract formation",
    target_response="A contract requires offer, acceptance, and consideration...",
    expert_reference="Contract formation also requires capacity, legality, and mutual assent..."
)
```

### 2. Domain Reasoning Agent (`src/reasoning_agent.py`)

Specialized reasoning agents for specific domains. Each agent focuses on a particular field (legal, medical, scientific, etc.) using domain-specific knowledge.

```python
from src.reasoning_agent import DomainReasoningAgent

legal_agent = DomainReasoningAgent(domain="legal")
response = legal_agent.process("What are the elements of negligence?")
```

### 3. Diffusion Policy (`src/diffusion.py`)

Implements diffusion-based policies for representing multimodal reasoning actions. These policies allow agents to explore diverse reasoning paths.

```python
from src.diffusion import DiffusionPolicy

policy = DiffusionPolicy()
action = policy.sample_action(context="legal_reasoning")
```

### 4. OMAD Orchestrator (`src/omad.py`)

The Online Multi-Agent Diffusion orchestrator coordinates multiple diffusion policies with entropy-augmented objectives for effective multi-agent collaboration.

```python
from src.omad import OMADOrchestrator

orchestrator = OMADOrchestrator(
    agents={"agent_1": policy_1, "agent_2": policy_2},
    agent_metadata=[{"id": "agent_1", "domain": "legal"}, ...]
)

result = orchestrator.coordinate(query="Analyze this case...")
```

### 5. Embodiment-based Grouping (`src/grouping.py`)

Clusters specialized agents by "cognitive morphology" to reduce gradient conflicts during joint training. Inspired by cross-embodiment learning research.

```python
from src.grouping import EmbodimentGrouper

grouper = EmbodimentGrouper()
groups = grouper.group_agents([
    {"id": "legal_1", "morphology": {"expertise": "law"}},
    {"id": "medical_1", "morphology": {"expertise": "medicine"}},
    {"id": "legal_2", "morphology": {"expertise": "law"}},
])
# Groups: {0: ["legal_1", "legal_2"], 1: ["medical_1"]}
```

### 6. Agent Environment (`src/environment.py`)

Manages the shared blackboard and coordinates agent interactions within the OMAD framework.

```python
from src.environment import AgentEnvironment

env = AgentEnvironment(orchestrator=orchestrator)
env.register_agent(legal_agent)
env.register_agent(medical_agent)

result = env.process_query("What are the legal implications of this diagnosis?")
```

### 7. Integrated Adversarial Loop (`src/integrated_loop.py`)

The main orchestration layer that ties together adversarial generation with multi-agent diffusion for iterative gap closing.

```python
from src.integrated_loop import IntegratedAdversarialLoop

loop = IntegratedAdversarialLoop(
    agent_configs=[
        {"id": "legal_agent", "domain": "legal"},
        {"id": "medical_agent", "domain": "medical"},
    ],
    expert_reference="Expert-level explanation...",
    max_iterations=3
)

result = loop.run_iteration("Explain informed consent in medical treatment")
# result contains: final_gap_score, history, consensus_summary
```

### 8. Evaluator (`src/evaluation.py`)

Domain-specific evaluation using benchmarks like LegalBench and MedicalQA.

```python
from src.evaluation import Evaluator

evaluator = Evaluator()
evaluator.load_benchmarks()
evaluator.run_evaluation("legal", benchmark_items)
```

## Installation

```bash
pip install -r requirements.txt
```

Requirements:
- numpy
- pytest (for testing)

## Quick Start

```bash
# Run the demo
python -m src.main

# Run evaluation
python -m src.main --eval

# Evaluate specific domain
python -m src.main --eval --domain legal
```

## Programmatic Usage

```python
from src.integrated_loop import IntegratedAdversarialLoop
from src.evaluation import Evaluator

# Configure agents
agent_configs = [
    {"id": "agent_legal", "domain": "legal", "morphology": {"expertise": "law"}},
    {"id": "agent_medical", "domain": "medical", "morphology": {"expertise": "medicine"}},
    {"id": "agent_ethics", "domain": "ethics", "morphology": {"expertise": "philosophy"}},
]

# Expert reference for gap measurement
expert_text = """
Informed consent requires: (1) disclosure of material risks,
(2) patient capacity to understand, (3) voluntariness, and
(4) actual understanding by the patient.
"""

# Initialize and run the adversarial loop
loop = IntegratedAdversarialLoop(
    agent_configs=agent_configs,
    expert_reference=expert_text,
    max_iterations=3
)

result = loop.run_iteration(
    initial_context="What are the requirements for informed consent?"
)

print(f"Final Gap Score: {result['final_gap_score']}")
print(f"Iterations: {len(result['history'])}")

# Evaluate on domain benchmarks
evaluator = Evaluator()
evaluator.load_benchmarks()
evaluator.run_evaluation("legal", evaluator.benchmarks["legal"])
```

## Research Background

### Agentic Adversarial QA (Grari et al., 2026)
The adversarial question generation framework iteratively identifies comprehension gaps by comparing model responses against expert references. Key insight: adversarial probing reveals blind spots that standard evaluation misses.

### OMAD: Online Multi-Agent Diffusion Policies (Li et al., 2026)
Diffusion-based policies allow agents to represent multimodal reasoning actions. The entropy-augmented objectives encourage exploration while the joint distributional value function coordinates multi-agent collaboration.

### Cross-Embodiment Learning (Abe et al., 2026)
The "cognitive morphology" grouping strategy clusters agents by similarity in reasoning patterns, reducing gradient conflicts when training heterogeneous agent teams.

## Project Structure

```
adversarial-domain-diffuser/
├── src/
│   ├── __init__.py
│   ├── adversarial_gen.py    # Adversarial question generator
│   ├── reasoning_agent.py    # Domain-specific reasoning agents
│   ├── diffusion.py          # Diffusion policy implementation
│   ├── omad.py               # OMAD orchestrator
│   ├── grouping.py           # Embodiment-based agent grouping
│   ├── environment.py        # Multi-agent environment
│   ├── integrated_loop.py    # Main adversarial loop
│   ├── evaluation.py         # Domain benchmark evaluation
│   └── main.py               # CLI entry point
├── tests/
│   ├── test_adversarial_gen.py
│   ├── test_diffusion.py
│   ├── test_environment.py
│   ├── test_evaluation.py
│   ├── test_grouping.py
│   ├── test_integrated_loop.py
│   ├── test_omad.py
│   └── test_reasoning_agent.py
├── config/                    # Configuration files
├── data/                      # Datasets and samples
├── results/                   # Evaluation results (JSON)
├── README.md
└── requirements.txt
```

## Running Tests

```bash
pytest tests/ -v
```

All 28 tests should pass.

## Evaluation Results

Results are saved to the `results/` directory as JSON files containing:
- Domain name
- Accuracy metrics
- Gap scores per iteration
- Consensus summaries

## Future Enhancements

- Integration with real LLM backends for adversarial generation
- Additional domain benchmarks (financial, scientific)
- Visualization dashboard for gap-closing progress
- Distributed multi-agent deployment

## License

MIT

## Contributing

1. Fork the repository
2. Create a feature branch
3. Ensure all tests pass (`pytest tests/`)
4. Submit a pull request
