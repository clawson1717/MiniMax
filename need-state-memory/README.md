# Need-State Memory Architecture (NSMA)

A memory architecture for LLM agents where retrieval is governed by the agent's operative **need state** — the same query retrieves fundamentally different information depending on whether the agent's urgent need is survival, coherence, or growth.

## Overview

Built from three ArXiv papers:
- **Computational Concept of the Psyche** (Kolonin) — agents in needs-state space
- **Knowledge Objects** (Chana et al.) — hash-addressed persistent memory
- **OpenSeeker** (Ye et al.) — fact-grounded synthesis with denoised trajectories

## Architecture

```
Agent Context
     ↓
NeedStateDetector → NeedStateMachine (SURVIVAL / COHERENCE / GROWTH)
     ↓
QueryExpander → NeedAwareRetriever → KnowledgeStore (hash-addressed)
     ↓
DenoisedTrajectorySynthesizer → Memory-Aware Response
     ↓
KnowledgeStore.update()
```

## Installation

```bash
pip install -r requirements.txt
```

## Components

| Component | File | Description |
|-----------|------|-------------|
| Need State | `src/need_state.py` | Need state enum + state machine |
| Knowledge Object | `src/knowledge_object.py` | Hash-addressed memory tuples |
| Retriever | `src/retriever.py` | Need-state-conditional retrieval |
| Synthesizer | `src/synthesizer.py` | Denoised trajectory synthesis |
| Detector | `src/detector.py` | Context → need state detection |
| Agent | `src/agent.py` | Full agent integration |
| Policy | `src/policy.py` | Need transition rules |
| Query Expander | `src/query_expander.py` | Need-aware query expansion |

## Roadmap

See `memory/project-need-state-memory-plan.md` for full 12-step implementation plan.

**Status:** Step 0 complete — scaffold created.

## Testing

```bash
python -m pytest tests/ -v
```
