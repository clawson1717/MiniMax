# Denoised Collaborative Fact-Assembler (DCFA)

## Concept
A multi-agent knowledge synthesis system that utilizes dynamic interaction graphs to trace evidence gathering and applies uncertainty-aware denoising to resolve conflicting data from multiple sources.

## Source Papers

1. **DenoiseFlow** (Yan et al., March 2026)
   - Technique: Sensing-Regulating-Correcting framework for progressive denoising.
   - Use: Resolve conflicting evidence snippets by sensing semantic ambiguity.

2. **DIG to Heal** (Yang et al., March 2026)
   - Technique: Dynamic Interaction Graph (DIG) and time-evolving causal networks.
   - Use: Map the collaboration between specialized agents to identify the source of semantic drift.

3. **Multi-Sourced Evidence Retrieval for Fact-Checking** (Gong et al., March 2026)
   - Technique: WKGFC (Web-Knowledge-Graph Fact-Checking) architecture.
   - Use: Core retrieval engine for structured and unstructured data.

## Innovation: Causal Denoising
The **DCFA** project introduces the **Causal Denoising Pipeline**:
1. **Multi-Source Fetching**: Pull raw data from KGs and the web simultaneously.
2. **Dynamic Tracing**: Map this process as a **DIG**, creating a causal network of the evidence flow.
3. **Semantic Sensing**: If two agents pull conflicting data, high semantic uncertainty is triggered at the synthesis node.
4. **Targeted Branching**: The system uses the DIG to pinpoint the conflicting source and spawns tie-breaker agents to find the truth.

## 12-Step Implementation Plan

### Step 1: Project Scaffold [ ]
- Create `ClawWork/denoised-collaborative-fact-assembler/` directory
- `src/`, `tests/`, `data/` structure
- `requirements.txt` with torch, networkx, pydantic, and asyncio
- Initial `src/__init__.py`

### Step 2: Evidence Payload Model [ ]
- `src/payload.py` with `EvidencePayload` class.

### Step 3: Collaborative Interaction Graph [ ]
- `src/graph.py` with `FactInteractionGraph` class.

### Step 4: Multi-Source Retrieval Simulator [ ]
- `src/retrieval.py` with `FactFetcher`.

### Step 5: Semantic Senser Stage [ ]
- `src/sensing.py` with `AmbiguitySenser`.

### Step 6: Denoising Regulator [ ]
- `src/regulating.py` with `DenoiseRegulator`.

### Step 7: Tie-Breaker Explorer [ ]
- `src/explorer.py` with `AmbiguityExplorer`.

### Step 8: Multi-Agent Synthesis Module [ ]
- `src/synthesis.py` with `FactAssembler`.

### Step 9: Fact-Assembler Orchestrator [ ]
- `src/agent.py` with `DCFAAgent`.

### Step 10: Logic-Grounded Benchmark [ ]
- `src/benchmark.py` with `TruthResilienceSuite`.

### Step 11: Real-time Interaction Dashboard [ ]
- `src/display.py` with `GraphViewer`.

### Step 12: Documentation & CLI [ ]
- Comprehensive README with architecture diagrams.
- CLI for running tasks.

---

## Getting Started
To begin work on Step 1, run the `github-project-selector` skill during the next heartbeat cycle.

## License
MIT
