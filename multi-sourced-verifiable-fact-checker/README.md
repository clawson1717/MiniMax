# Multi-Sourced Verifiable Fact-Checker (MSVFC)

## Concept
A multi-agent fact-checking system that orchestrates structured evidence retrieval from knowledge graphs and web sources, modeling the process as a Markov Decision Process (MDP) with deterministic state verification.

## Source Papers

1. **Multi-Sourced, Multi-Agent Evidence Retrieval for Fact-Checking** (Gong et al., March 2026)
   - Technique: WKGFC (Web-Knowledge-Graph Fact-Checking) logic.
   - Use: Modeling fact-checking as an automatic Markov Decision Process (MDP).

2. **LOGIGEN** (Zeng et al., March 2026)
   - Technique: Triple-Agent Orchestration and Deterministic State Verification.
   - Use: Developing "Architect" components to compile fact-checking policies into verifiable constraints.

3. **DIG to Heal** (Yang et al., March 2026)
   - Technique: Dynamic Interaction Graph (DIG) for explainable collaboration paths.
   - Use: Tracking and explaining the evidence retrieval path to ensure no "hallucination spirals."

## Innovation: The Verifiable Evidence MDP
The **MSVFC** project introduces a **Policy-Grounded Retrieval** paradigm:
1. **Policy-Grounded**: Use **LOGIGEN**'s principles to define "hard criteria" for claims (e.g., "verified requires 2 independent sources").
2. **Retrieval MDP**: Implement an MDP where agents decide which knowledge source to query next based on existing evidence gaps.
3. **Causal Interaction Mapping**: Use **DIG** to trace exactly how a claim evolved from a "Query" to a "Verified" state.
4. **Deterministic Verification**: Final decisions are strictly checked against state-transition policy equivalents.

## 12-Step Implementation Plan

### Step 1: Project Scaffold [ ]
- Create `ClawWork/multi-sourced-verifiable-fact-checker/` directory
- `src/`, `tests/`, `data/` structure
- `requirements.txt` with torch, networkx, pydantic, and aiohttp
- Initial `src/__init__.py`

### Step 2: Evidence State Model [ ]
- `src/state.py` with `EvidenceState` class (MDP state).

### Step 3: Policy-Architect Module [ ]
- `src/architect.py` with `FactCheckArchitect`.

### Step 4: Knowledge Graph Retrieval Agent [ ]
- `src/retrieval_kg.py` with `KGRetrievalAgent`.

### Step 5: Web Search Retrieval Agent [ ]
- `src/retrieval_web.py` with `WebRetrievalAgent`.

### Step 6: Fact-Checking MDP Controller [ ]
- `src/mdp.py` with `FactCheckMDP`.

### Step 7: Dynamic Interaction Graph (DIG) [ ]
- `src/dig.py` with `EvidenceGraph`.

### Step 8: Deterministic State Verifier [ ]
- `src/verifier.py` with `StateVerifier`.

### Step 9: MSVFC Agent Orchestrator [ ]
- `src/agent.py` with `MSVFCAgent`.

### Step 10: Task Synthesis Generator [ ]
- `src/synthesis.py` with `ClaimGenerator`.

### Step 11: Fact-Checking Benchmark [ ]
- `src/benchmark.py` with `FactCheckBench`.

### Step 12: Documentation & CLI [ ]
- Comprehensive README with Mermaid architecture.
- Full API documentation and CLI for running tasks.

---

## Getting Started
To begin work on Step 1, run the `github-project-selector` skill during the next heartbeat cycle.

## License
MIT
