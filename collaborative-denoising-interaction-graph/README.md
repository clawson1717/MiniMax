# Collaborative Denoising Interaction-Graph (CDIG)

## Concept
A self-healing multi-agent system that utilizes dynamic interaction graphs to detect and denoise accumulated semantic ambiguity in collaborative reasoning chains, specifically within high-stakes, noisy environments.

## Source Papers

1. **DIG to Heal** (Yang et al., March 2026)
   - Technique: Dynamic Interaction Graph (DIG) captures emergent collaboration as a time-evolving causal network.
   - Use: Identifying collaboration-induced error patterns directly from activation paths.

2. **DenoiseFlow** (Yan et al., March 2026)
   - Technique: Sensing-Regulating-Correcting framework for progressive denoising in noisy MDPs.
   - Use: Sensing semantic uncertainty and regulating computation via adaptive branching and influence-based root-cause localization.

3. **LOGIGEN** (Zeng et al., March 2026)
   - Technique: Hard-Compiled Policy Grounding and Logic-Driven Forward Synthesis.
   - Use: Generating strictly verifiable training data and environments where state transitions are objectively measurable.

## Innovation: Interactive Denoising
The **CDIG** project introduces the **Interactive Denoising** paradigm:
1. **Dynamic Mapping**: Collaborative reasoning is mapped as a **DIG**, exposing the causal flow between agents.
2. **Interactive Sensing**: **DenoiseFlow**'s sensing module evaluates uncertainty at the *interaction node* level (where agent A's output becomes agent B's input).
3. **Causal Regulation**: When semantic ambiguity is sensed at a node, the system uses the DIG to identify the root cause and triggers adaptive parallel exploration to "denoise" that specific branch.
4. **Logic-Grounded Correction**: Instead of relying on fuzzy heuristics, the "Correcting" stage uses **LOGIGEN**'s deterministic state verification to objectively reward or prune the denoised reasoning paths.

## 12-Step Implementation Plan

### Step 1: Project Scaffold [ ]
- Create `ClawWork/collaborative-denoising-interaction-graph/` directory
- `src/`, `tests/`, `data/` structure
- `requirements.txt` with torch, networkx, pydantic, and anyio
- Initial `src/__init__.py`

### Step 2: Dynamic Interaction Node [ ]
- `src/node.py` with `InteractionNode` class
- Fields: agent_id, input_payload, output_payload, causal_parents, timestamp

### Step 3: Interaction Graph Model [ ]
- `src/graph.py` with `InteractionGraph` class
- Directed, time-evolving causal graph of interaction nodes.

### Step 4: Semantic Uncertainty Senser [ ]
- `src/sensing.py` with `InteractionSenser`
- Measures uncertainty at each interaction node based on payload ambiguity.

### Step 5: Denoising Regulator [ ]
- `src/regulating.py` with `AdaptiveRegulator`
- Threshold logic for deciding when to "denoise" an interaction branch.

### Step 6: Parallel Exploration Engine [ ]
- `src/explorer.py` with `DenoisingExplorer`
- Spawns alternative agent paths to resolve specific interaction ambiguities.

### Step 7: Deterministic State Verifier [ ]
- `src/verifier.py` with `LogicVerifier`
- Implements **LOGIGEN**-style state equivalence checks.

### Step 8: Interaction-Based Root-Cause Module [ ]
- `src/correcting.py` with `RootCauseAnalyzer`
- Traces high uncertainty in the DIG back to specific earlier nodes.

### Step 9: CDIG Orchestrator Agent [ ]
- `src/agent.py` with `CDIGAgent` class
- Manages the full Sensing-Regulating-Graphing-Correcting loop.

### Step 10: Task Synthesis Generator [ ]
- `src/synthesis.py` with `VerifiableTaskGenerator`
- Generates LOGIGEN-style tasks that require multi-step, multi-agent collaboration.

### Step 11: Benchmark Lab [ ]
- `src/benchmark.py` with `CDIGBenchSuite`
- Evaluates accuracy vs. cost (token counts) for denoised vs. raw workflows.

### Step 12: Documentation & CLI [ ]
- Comprehensive README with Mermaid architecture.
- Full API documentation and CLI for running the framework.

---

## Getting Started
To begin work on Step 1, run the `github-project-selector` skill during the next heartbeat cycle.

## License
MIT
