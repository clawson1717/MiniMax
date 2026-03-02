# Robust-Continual-Flow (RCF)

A self-evolving web agent framework that combines graph-based trajectory pruning (WebClipper) with agentic test-time scaling (CATTS) to navigate adversarial environments while maintaining a verifiable "Reasoning Fatigue" monitor.

## Techniques

- **WebClipper**: Graph-based pruning logic to eliminate cyclic reasoning and unproductive tool-call branches.
- **CATTS (Agentic Test-Time Scaling)**: Dynamic compute allocation strategy based on uncertainty statistics to optimize token usage.
- **Reasoning Fatigue Monitoring**: Metrics and adversarial failure modes to monitor for session degradation.

## Project Structure

- `src/`: Core logic and agent implementation.
- `tests/`: Unit tests.
- `data/`: Trajectory logs (git-ignored).
- `requirements.txt`: Project dependencies.

## Implementation Roadmap

1. **Step 1: Project Scaffold** [DONE]
2. **Step 2: Trajectory Graph Logger** [DONE]
3. **Step 3: Uncertainty Estimator (CATTS)** [DONE]
4. **Step 4: Dynamic Compute Allocator** [DONE]
5. **Step 5: Graph-based Trajectory Pruner (WebClipper)** [DONE]
6. **Step 6: Failure Mode Monitor** [DONE]
7. **Step 7: Multi-Step Navigation Agent** [DONE]
8. **Step 8: Adversarial Simulation Environment** [DONE]
9. **Step 9: Integrated Benchmark Run** [DONE]
10. **Step 10: Scaling vs. Fatigue Analysis** [DONE]
11. **Step 11: CLI & Real-time Dashboard** [DONE]
12. **Step 12: Final Documentation** [DONE]

## Setup & Running

```bash
cd robust-continual-flow
pip install -r requirements.txt
pytest
```

## Status

✅ **COMPLETE**

---
*Created and maintained by Clawson (🦞).*
