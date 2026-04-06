# Soma — Self-Evolving Operational Memory Architecture

> Agents that learn by saving the code of what worked.

## Overview

Soma is a self-evolving agent memory framework where successful problem-solving trajectories are crystallized into **executable, hash-addressed Knowledge Subagents** — persistent, versioned units of capability that agents can retrieve, compose, and accumulate across sessions.

## Core Innovation

Combines three recent papers into a novel architecture:
- **AgentFactory**: Preserves successful solutions as executable Python subagents rather than textual prompts
- **Knowledge Objects**: Hash-addressed tuples eliminating memory capacity limits and compaction loss
- **Governed Memory**: Schema-enforced shared memory layer with formal governance

## Status

🟡 In Progress — 0/12 steps complete.

## Roadmap

| Step | Description | Status |
|------|-------------|--------|
| 1 | Project Scaffold | ⏳ Pending |
| 2 | Knowledge Object Model | ⏳ Pending |
| 3 | Subagent Representation | ⏳ Pending |
| 4 | Governed Memory Store | ⏳ Pending |
| 5 | Subagent Registry | ⏳ Pending |
| 6 | Belief Graph | ⏳ Pending |
| 7 | Subagent Factory | ⏳ Pending |
| 8 | Need State Tracker | ⏳ Pending |
| 9 | Agent Integration | ⏳ Pending |
| 10 | CLI Interface | ⏳ Pending |
| 11 | Benchmark Suite | ⏳ Pending |
| 12 | Final Documentation & README | ⏳ Pending |

## Full Plan

See `memory/project-soma-subagent-memory-governance-plan.md` for complete details.

## Quick Start

```bash
cd ClawWork/soma-subagent-memory-governance
pip install -r requirements.txt
pytest tests/ -v
```
