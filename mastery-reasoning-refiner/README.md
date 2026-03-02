# Mastery-Reasoning-Refiner (MRR)

Mastery-Reasoning-Refiner is an RL-driven reasoning framework that iteratively refines thought traces using **iGRPO**, guided by a curriculum based on **Bloom's Taxonomy** and measured by **Information-Theoretic metrics**.

## Concept

The core philosophy of MRR is that reasoning ability is not a binary trait but a developmental process. By combining pedagogical theory with modern Reinforcement Learning, we enable agents to climb the ladder of cognitive complexity.

- **Bloom's Taxonomy:** Tasks are categorized into six tiers (Knowledge, Comprehension, Application, Analysis, Evaluation, Synthesis).
- **Mastery Learning & ZPD:** The trainer utilizes a "Zone of Proximal Development" (ZPD) controller to ensure the agent is always challenged but not overwhelmed, only advancing to higher Bloom tiers once mastery is demonstrated.
- **iGRPO (Iterative Group Relative Policy Optimization):** A self-feedback loop that allows the agent to refine its reasoning traces based on relative performance within a group of generated completions.
- **Information-Theoretic Analysis:** We measure "World Model Capacity" in bits to quantify how much structure and information the agent has internalized about its task environment.

## Architecture

```text
+-------------------+       +-----------------------+
|  Training Data    |------>|  ZPD Controller       |
|  (Bloom Tiers)    |       |  (Curriculum Manager) |
+-------------------+       +-----------+-----------+
                                        |
                                        v
+-------------------+       +-----------+-----------+
|  Reasoning Agent  |<------|  Mastery Trainer      |
|  (Policy Model)   |       |  (Orchestrator)       |
+---------+---------+       +-----------+-----------+
          |                             ^
          v                             |
+---------+---------+       +-----------+-----------+
|  Iterative GRPO   |<------|  Reward Model         |
|  (Refinement)     |       |  (Checklist-based)    |
+---------+---------+       +-----------+-----------+
          |                             |
          v                             v
+---------+-----------------------------+-----------+
|           Information-Theoretic Metrics           |
|        (Capacity, Complexity, Information)        |
+---------------------------------------------------+
```

## Core Modules

- **`src/agent.py` (`ReasoningAgent`):** The primary model interface. It handles prompt processing and thought trace generation.
- **`src/grpo.py` (`IterativeGRPO`):** Implements the iterative refinement logic. It generates multiple reasoning candidates and uses relative performance to update the policy.
- **`src/pedagogy.py` (`ZPDController`):** Manages the learning curriculum. It tracks success rates per Bloom tier and adjusts the difficulty of samples provided to the trainer.
- **`src/metrics.py` (`InformationTheoreticMetrics`):** Calculates metrics such as World Model Capacity (bits) and complexity. It provides an objective measure of the agent's internal representation growth.
- **`src/reward_model.py` (`ChecklistRewardModel`):** A rule-based and checklist-driven evaluator that scores reasoning traces based on accuracy, logic, and tier-specific requirements.
- **`src/trainer.py` (`MasteryTrainer`):** The central training loop that integrates the agent, rewards, metrics, and curriculum.
- **`src/cli.py`:** A unified command-line interface for training, evaluation, and visualization.

## Installation

```bash
# Clone the repository
git clone https://github.com/your-repo/mastery-reasoning-refiner.git
cd mastery-reasoning-refiner

# Install dependencies
pip install -r requirements.txt
```

## Usage

The project uses a unified CLI for all operations.

### Training
Run the training pipeline with ZPD-based curriculum:
```bash
python -m src.cli train --epochs 5 --lr 1e-5 --output results.json
```

### Evaluation
Evaluate the current model's world model capacity:
```bash
python -m src.cli evaluate
```

### Visualization
Visualize the relationship between Mastery and World Model Capacity:
```bash
python -m src.cli visualize
```

## References

This implementation is informed by the following research:

1.  **iGRPO:** DeepSeek-V3 Technical Report / NVIDIA Research on Iterative RL.
2.  **Pedagogically-Inspired Data Synthesis:** *Pedagogically-Inspired Data Synthesis for Mastery Learning in LLMs* (ICLR 2026).
3.  **Information-theoretic analysis of world models:** *Information-Theoretic Metrics for Evaluating World Models in Reasoning Agents* by José Faustino.
4.  **Bloom's Taxonomy:** Bloom, B. S. (1956). *Taxonomy of Educational Objectives*.
