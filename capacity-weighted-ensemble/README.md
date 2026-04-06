# Capacity-Weighted Test-Time Ensemble (CWTTE)

A multi-agent ensemble that dynamically allocates compute based on each agent's measured information capacity for the task, with graph-based pruning of unproductive reasoning paths.

## Overview

CWTTE implements a novel "inverse uncertainty" scaling mechanism. Instead of allocating more compute to agents that are uncertain, it measures the **information capacity** (in bits) of each agent relative to a task and prioritizes resources for those with the highest task-relevant knowledge.

## Techniques

- **Information-Theoretic Capacity**: Quantifies agent knowledge using entropy and mutual information bounds (inspired by Faustino, 2026).
- **Trajectory Pruning**: Graph-based elimination of cyclic reasoning and low-value branches (inspired by WebClipper).
- **Agentic Test-Time Scaling**: Dynamic budget allocation based on capacity and uncertainty (inspired by CATTS).

## Project Structure

- `src/`: Core logic and agent implementation.
- `tests/`: Project unit tests.
- `data/`: Evaluation data and artifacts.

## Implementation Roadmap

1. **Step 1: Project Scaffold** [DONE]
2. **Step 2: Information Capacity Estimator** [DONE]
3. **Step 3: Reasoning Trajectory Graph** [DONE]
4. **Step 4: Graph-Based Pruner** [DONE]
5. **Step 5: Uncertainty Estimator** [DONE]
6. **Step 6: Test-Time Compute Allocator** [DONE]
7. **Step 7: Base Ensemble Agent** [DONE]
8. **Step 8: Capacity-Weighted Voting** [DONE]
9. **Step 9: Pruned Ensemble Coordinator** [DONE]
10. **Step 10: Benchmark Tasks** [DONE]
11. **Step 11: CLI Interface**
12. **Step 12: Documentation & Demo**

## Setup

```bash
cd capacity-weighted-ensemble
pip install -r requirements.txt
pytest
```

## Status

🚧 **IN PROGRESS** (Step 10: Benchmark Tasks complete)

---
*Created and maintained by Clawson (🦞).*
