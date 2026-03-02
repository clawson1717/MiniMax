# Kepler-Skills-Distiller (KSD)

A framework for scientific equation discovery that distills expert reasoning into smaller, efficient models using pedagogically-inspired training.

## Overview

KSD implements a complete pipeline for:
1. **Scientific Discovery**: Using physics-guided reasoning to discover equations from data
2. **Skill Extraction**: Converting successful discovery trajectories into reusable "Expert Skills"
3. **Knowledge Distillation**: Training smaller "Scientific Apprentice" models using Bloom's Taxonomy curriculum

### Key Innovation

Unlike standard distillation, KSD uses **pedagogically-inspired data synthesis** - organizing training data by Bloom's Taxonomy tiers (Remember → Understand → Apply → Analyze → Evaluate → Create) to progressively build the apprentice's capabilities.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Kepler-Skills-Distiller                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────┐     ┌──────────────────┐                  │
│  │   Raw Data       │────▶│  Physical        │                  │
│  │   (observations) │     │  Property        │                  │
│  └──────────────────┘     │  Inferencer      │                  │
│                           └────────┬─────────┘                  │
│                                    │                            │
│                           ┌────────▼─────────┐                  │
│                           │  Kepler          │                  │
│                           │  Reasoning       │                  │
│                           │  Agent           │                  │
│                           └────────┬─────────┘                  │
│                                    │                            │
│                    ┌───────────────┴───────────────┐            │
│                    │                               │            │
│           ┌────────▼────────┐           ┌─────────▼──────┐     │
│           │  Expert Skill   │           │  Pedagogical   │     │
│           │  Generator      │           │  Data          │     │
│           └────────┬────────┘           │  Synthesizer   │     │
│                    │                    └─────────┬──────┘     │
│                    │                              │            │
│           ┌────────▼────────┐           ┌─────────▼──────┐     │
│           │  SkillsBench    │           │  Bloom's       │     │
│           │  Format Skills  │           │  Curriculum    │     │
│           └────────┬────────┘           └─────────┬──────┘     │
│                    │                              │            │
│                    └──────────────┬───────────────┘            │
│                                   │                            │
│                          ┌────────▼─────────┐                  │
│                          │  Mastery         │                  │
│                          │  Distillation    │                  │
│                          │  Loop            │                  │
│                          └────────┬─────────┘                  │
│                                   │                            │
│                          ┌────────▼─────────┐                  │
│                          │  Scientific      │                  │
│                          │  Apprentice      │                  │
│                          │  (Small Model)   │                  │
│                          └──────────────────┘                  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Components

### 1. Physical Property Inferencer (`src/inferencer.py`)

Infers physical constraints from raw data:
- **Dimensional consistency**: Unit relationships
- **Symmetries**: Time-translation, scale invariance
- **Conservation laws**: Constant quantities, sum-of-squares relationships

```python
from src.inferencer import PhysicalPropertyInferencer
import numpy as np

data = {
    "t": np.linspace(0, 10, 100),
    "x": np.sin(np.linspace(0, 10, 100)),
    "v": np.cos(np.linspace(0, 10, 100)),
}

inferencer = PhysicalPropertyInferencer(data)
properties = inferencer.infer_all()
# Returns: dimensional_consistency, symmetries, conservation_laws
```

### 2. Kepler Reasoning Agent (`src/reasoning_agent.py`)

Physics-guided discovery agent that uses inferred properties to constrain hypothesis generation.

```python
from src.reasoning_agent import KeplerReasoningAgent

agent = KeplerReasoningAgent(model_client=your_llm_client)
result = agent.discover(data)
# Returns: proposed_expressions, hypotheses, analysis
```

### 3. Expert Skill Generator (`src/skill_generator.py`)

Converts successful discovery trajectories into reusable procedural skills.

```python
from src.skill_generator import ExpertSkillGenerator

generator = ExpertSkillGenerator(skills_dir="skills")
skill = generator.generate_skill(
    trajectory=discovery_result,
    metrics={"elapsed": 1.5},
    skill_name="harmonic_oscillator_discovery",
)
```

### 4. Skill Schema (`src/skills.py`)

Structured JSON schema for scientific skills based on SkillsBench taxonomy:

```json
{
  "name": "skill_name",
  "domain": "physics",
  "description": "What this skill does",
  "procedural_steps": [
    "Step 1: Analyze data patterns",
    "Step 2: Check conservation laws",
    "Step 3: Formulate hypotheses"
  ],
  "prerequisites": ["basic_physics", "calculus"],
  "mastery_criteria": {
    "accuracy_threshold": 0.85,
    "consistency_checks": ["dimensional", "symmetry"]
  }
}
```

### 5. Pedagogical Data Synthesizer (`src/synthesizer.py`)

Creates training data organized by Bloom's Taxonomy:

| Tier | Cognitive Level | Training Focus |
|------|-----------------|----------------|
| 1 | Remember | Recognize equation forms |
| 2 | Understand | Explain physical meaning |
| 3 | Apply | Use equations for prediction |
| 4 | Analyze | Compare equation behaviors |
| 5 | Evaluate | Assess equation quality |
| 6 | Create | Synthesize novel equations |

```python
from src.synthesizer import create_curriculum_synthesizer

synthesizer = create_curriculum_synthesizer(
    data_path="data/training",
    output_dir="outputs/synthetic",
)

synthetic_data = synthesizer.synthesize_tier("Apply", n_samples=100)
```

### 6. Apprentice Model (`src/apprentice.py`)

Small model scaffold for distillation targets (Qwen-1.5B, Llama-3-1B, etc.).

```python
from src.apprentice import create_apprentice

apprentice = create_apprentice(
    model_name="Qwen/Qwen2-1.5B-Instruct",
    use_mock=False,  # Set True for testing
)
apprentice.load()
```

### 7. Mastery Distillation (`src/distillation.py`)

Training loop using Bloom's curriculum with mastery-based progression:

```python
from src.distillation import create_mastery_distiller

distiller = create_mastery_distiller(
    model_name="Qwen/Qwen2-1.5B-Instruct",
    data_path="data/training",
    output_dir="outputs/distillation",
)

result = distiller.run_curriculum(
    steps_per_tier=50,
    max_iterations=10,
)

# Result includes tier progress, mastery records, curriculum status
```

Key features:
- **Mastery threshold**: 85% accuracy before advancing to next tier
- **Adaptive pacing**: More iterations on weaker tiers
- **Progress tracking**: Per-tier mastery records

### 8. Equation Discovery Benchmark (`src/benchmark.py`)

Feynman Symbolic Regression Benchmark with 13 physics/math equations:

- **Classical Mechanics**: Newton's laws, kinetic energy, momentum, gravitation
- **Electromagnetism**: Coulomb's law, Ohm's law, electric power
- **Thermodynamics**: Ideal gas law, thermal energy
- **Mathematics**: Circle area, sphere volume, Pythagorean theorem

```python
from src.benchmark import EquationDiscoveryBenchmark, FeynmanDataset

dataset = FeynmanDataset()
benchmark = EquationDiscoveryBenchmark(output_dir="benchmark_results")

report = benchmark.run_benchmark(
    expert_model=kepler_agent,
    apprentice_model=apprentice,
    max_complexity=2,
)

print(f"Expert accuracy: {report.expert_accuracy:.1%}")
print(f"Apprentice accuracy: {report.apprentice_accuracy:.1%}")
print(f"Improvement ratio: {report.improvement_ratio:.1%}")
```

### 9. CLI (`src/cli.py`)

Command-line interface for the full pipeline:

```bash
# Run equation discovery
python -m src.cli discover --data-file data/experiment.json --verbose

# Run distillation
python -m src.cli distill --data data/training --steps-per-tier 50

# Run benchmark
python -m src.cli benchmark --mock --max-complexity 2 --compare

# Visualize progress
python -m src.cli visualize curriculum
python -m src.cli visualize capacity
python -m src.cli visualize skills

# List generated skills
python -m src.cli list-skills
```

## Papers & Techniques

### KeplerAgent (Yang et al., 2026)
Provides the physics-guided reasoning loop that infers physical properties to constrain symbolic regression. Key insight: constraining hypothesis space with physical priors improves discovery efficiency.

### SkillsBench (Li et al., 2026)
Informs the structure of "Expert Skills" - curated procedural knowledge that significantly boosts agent performance. Skills are stored as structured JSON with procedural steps, prerequisites, and mastery criteria.

### Pedagogically-Inspired Data Synthesis (He et al., 2026)
Provides the Bloom's Taxonomy-based curriculum for distilling reasoning into smaller models. Key insight: progressive difficulty aligned with cognitive levels produces better transfer than random data mixing.

## Installation

```bash
git clone https://github.com/clawson1717/kepler-skills-distiller.git
cd kepler-skills-distiller
pip install -r requirements.txt
```

### Requirements

- Python 3.10+
- numpy
- pytest (for testing)

Optional:
- transformers (for real model training)
- torch (for real model training)

## Testing

All 103 tests passing:

```bash
pytest tests/ -v
```

Test coverage:
- `test_inferencer.py`: Physical property inference
- `test_skills.py`: Skill schema and generation
- `test_reasoning_agent.py`: Discovery agent
- `test_synthesizer.py`: Pedagogical data synthesis
- `test_apprentice.py`: Apprentice model scaffold
- `test_distillation.py`: Mastery distillation loop
- `test_benchmark.py`: Feynman benchmark
- `test_cli.py`: CLI commands

## Project Structure

```
kepler-skills-distiller/
├── src/
│   ├── __init__.py
│   ├── inferencer.py        # Physical property inference
│   ├── reasoning_agent.py   # Kepler discovery agent
│   ├── skill_generator.py   # Expert skill extraction
│   ├── skills.py            # Skill schema definitions
│   ├── synthesizer.py       # Pedagogical data synthesis
│   ├── apprentice.py        # Apprentice model scaffold
│   ├── trainer.py           # Training utilities
│   ├── distillation.py      # Mastery distillation loop
│   ├── benchmark.py         # Equation discovery benchmark
│   └── cli.py               # Command-line interface
├── tests/
│   ├── test_inferencer.py
│   ├── test_skills.py
│   ├── test_reasoning_agent.py
│   ├── test_synthesizer.py
│   ├── test_apprentice.py
│   ├── test_distillation.py
│   ├── test_benchmark.py
│   └── test_cli.py
├── data/                    # Training datasets
├── skills/                  # Generated expert skills
├── requirements.txt
└── README.md
```

## Usage Example

Full pipeline example:

```python
import numpy as np
from src.inferencer import PhysicalPropertyInferencer
from src.reasoning_agent import KeplerReasoningAgent
from src.skill_generator import ExpertSkillGenerator
from src.apprentice import create_apprentice
from src.distillation import create_mastery_distiller
from src.benchmark import EquationDiscoveryBenchmark

# 1. Prepare data (simple harmonic motion)
t = np.linspace(0, 10, 100)
data = {
    "t": t,
    "x": np.sin(t),
    "v": np.cos(t),
}

# 2. Infer physical properties
inferencer = PhysicalPropertyInferencer(data)
properties = inferencer.infer_all()
print(f"Conservation laws: {properties['conservation_laws']}")

# 3. Run discovery with expert agent
agent = KeplerReasoningAgent(model_client=your_llm_client)
discovery = agent.discover(data)
print(f"Proposed equations: {discovery['proposed_expressions']}")

# 4. Generate expert skill
generator = ExpertSkillGenerator(skills_dir="skills")
skill = generator.generate_skill(
    trajectory=discovery,
    metrics={"elapsed": 1.5},
    skill_name="shm_discovery",
)
print(f"Generated skill: {skill.name}")

# 5. Distill to apprentice
distiller = create_mastery_distiller(
    model_name="Qwen/Qwen2-1.5B-Instruct",
    data_path="data/training",
    output_dir="outputs/distillation",
)
result = distiller.run_curriculum(steps_per_tier=50)

# 6. Benchmark performance
benchmark = EquationDiscoveryBenchmark()
report = benchmark.run_benchmark(
    expert_model=agent,
    apprentice_model=apprentice,
    max_complexity=2,
)
print(f"Improvement ratio: {report.improvement_ratio:.1%}")
```

## Key Design Decisions

### 1. Capacity-Based Compute Allocation
Unlike CATTS (where high uncertainty triggers more compute), KSD allocates more compute to high-capacity agents - they have more knowledge to leverage.

### 2. Bloom's Taxonomy Curriculum
Training progresses through cognitive levels:
- Lower tiers (Remember/Understand): Foundation
- Middle tiers (Apply/Analyze): Application
- Higher tiers (Evaluate/Create): Synthesis

### 3. Mastery-Based Progression
85% accuracy threshold before advancing tiers ensures solid foundation before building complexity.

## Future Work

1. **Real Model Training**: Integration with actual LLM training (currently mock-based for testing)
2. **Additional Benchmarks**: More symbolic regression datasets beyond Feynman
3. **Multi-Domain Skills**: Extend beyond physics to chemistry, biology
4. **Online Learning**: Continuous skill refinement from new discoveries
5. **Skill Composition**: Combining multiple skills for complex tasks

## License

MIT License

## Citation

If you use this framework, please cite the original papers:

```bibtex
@article{yang2026kepler,
  title={KeplerAgent: Think like a Scientist for Scientific Equation Discovery},
  author={Yang et al.},
  year={2026}
}

@article{li2026skillsbench,
  title={SkillsBench: Benchmarking Procedural Knowledge in AI Agents},
  author={Li et al.},
  year={2026}
}

@article{he2026pedagogical,
  title={Pedagogically-Inspired Data Synthesis for Reasoning Distillation},
  author={He et al.},
  year={2026}
}
```

## Acknowledgments

This project implements techniques from KeplerAgent, SkillsBench, and Pedagogically-Inspired Data Synthesis research papers.