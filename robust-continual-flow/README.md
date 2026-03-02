# Robust-Continual-Flow (RCF)

A self-evolving web agent framework that combines graph-based trajectory pruning (WebClipper) with agentic test-time scaling (CATTS) to navigate adversarial environments (BrowseComp-V3) while maintaining a verifiable "Reasoning Fatigue" monitor.

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

1. **Step 1: Project Scaffold** (Current)
2. **Step 2: Trajectory Graph Logger**
3. **Step 3: Uncertainty Estimator (CATTS)**
4. **Step 4: Dynamic Compute Allocator**
5. **Step 5: Graph-based Trajectory Pruner (WebClipper)**
6. **Step 6: Failure Mode Monitor**
7. **Step 7: Multi-Step Navigation Agent**
8. **Step 8: Adversarial Simulation Environment**
9. **Step 9: Integrated Benchmark Run**
10. **Step 10: Scaling vs. Fatigue Analysis**
11. **Step 11: CLI & Real-time Dashboard**
12. **Step 12: Final Documentation**
