# Explainable Collaborative-Depth Optimizer (ECDO)

## Concept
A multi-agent collaboration system that utilizes dynamic interaction graphs to explain emergent behavior and adaptively scales the reasoning depth of individual agents within the collaboration chain based on their causal contribution to the solution.

## Source Papers

1. **DIG to Heal** (Yang et al., March 2026)
   - Technique: Dynamic Interaction Graph (DIG) and time-evolving causal networks of agent interactions.
   - Use: Capture emergent collaboration paths to identify redundant work and explain failure modes in real-time.

2. **LOGIGEN** (Zeng et al., March 2026)
   - Technique: Logic-Driven Forward Synthesis and Explorer-based causal solution path discovery.
   - Use: Synthesize verifiable, logic-driven tasks that require multi-agent cooperation and strictly enforced policy grounding.

3. **Draft-Thinking** (Cao et al., March 2026)
   - Technique: Adaptive prompting and concise "draft-style" reasoning for model-selectable depth.
   - Use: Dynamically adjust the reasoning depth of agents based on their position and importance in the interaction graph.

## Innovation: Causal-Depth Allocation
The **ECDO** project introduces the **Causal-Depth Allocation** paradigm:
1. **Verifiable Multi-Agent Tasks**: Use **LOGIGEN**'s principles to generate complex collaborative tasks.
2. **Real-time Collaboration Mapping**: As agents solve these tasks, **DIG to Heal** captures their interactions as a dynamic causal graph.
3. **Adaptive Depth Scaling**: Using the causal graph, the system identifies which agents are in "critical" versus "redundant" paths.
4. **Draft-Thinking Routing**: Agents in the critical path use full CoT; those in low-causality branches use concise "Draft-Thinking" to save tokens.

## 12-Step Implementation Plan

### Step 1: Project Scaffold [ ]
- Create `ClawWork/explainable-collaborative-depth-optimizer/` directory
- `src/`, `tests/`, `data/` structure
- `requirements.txt` with torch, networkx, pydantic, and asyncio
- Initial `src/__init__.py`

### Step 2: Logic-Driven Task Synthesis [ ]
- `src/synthesis.py` with `LogicTaskGenerator`
- Generates JSON-based verifiable collaborative tasks.

### Step 3: Multi-Agent Environment [ ]
- `src/env.py` with `CollaborativeEnv` class
- Supports multiple agent activations without predefined roles.

### Step 4: Dynamic Interaction Graph (DIG) [ ]
- `src/dig.py` with `InteractionGraph` class
- Captures agent interactions as a time-evolving causal network.

### Step 5: Draft-Thinking Baseline Agent [ ]
- `src/agent.py` with `ECDOAgent` class
- Implements the "Draft-Thinking" adaptive prompting capability.

### Step 6: Causal Path Analyzer [ ]
- `src/analyzer.py` with `CausalPathAnalyzer`
- Identifies critical vs. redundant paths within the DIG.

### Step 7: Adaptive Depth Allocator [ ]
- `src/allocator.py` with `DepthAllocator`
- Maps causality scores to reasoning depth (Draft vs. Deep).

### Step 8: Collaborative Logic-Orchestrator [ ]
- `src/orchestrator.py` with `ECDOOrchestrator`
- Orchestrates the full loop: task gen -> collaboration -> DIG mapping -> depth scaling.

### Step 9: Integration & Benchmarking [ ]
- `src/benchmark.py` with `ECDOBenchmarks`
- Test ECDO against static-depth collaboration baselines.

### Step 10: Visualization Layer [ ]
- `src/visualizer.py` with `DIGVisualizer`
- Renders the Dynamic Interaction Graph with depth/causality markers.

### Step 11: Real-time "Healing" Module [ ]
- `src/healing.py` with `CollaborationHealer`
- Logic for correcting collaboration-induced errors identified by the DIG.

### Step 12: Documentation & CLI [ ]
- Comprehensive README with Mermaid architecture.
- Full API documentation and CLI for running the framework.

---

## Getting Started
To begin work on Step 1, run the `github-project-selector` skill during the next heartbeat cycle.

## License
MIT
