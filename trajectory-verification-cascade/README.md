# Trajectory Verification Cascade (TVC)

A cascaded reasoning verification system that combines graph-based trajectory
pruning with fine-grained checklist verification at each node, plus adversarial
failure mode detection to catch manipulation attempts during multi-step reasoning.

## Concept

- Each reasoning step is a node in a directed graph
- Before proceeding to the next node, the current node must pass a binary checklist
- Failure mode detection runs in parallel to catch manipulation attempts
- Unproductive branches are pruned before wasting compute
- The cascade backtracks on failure, finding alternative paths

## Source Papers

- **WebClipper** (Wang et al.) — Graph-based trajectory pruning
- **CM2 Checklist Rewards** (Zhang et al.) — Binary criteria for multi-step verification
- **Multi-Turn Attack Failure Modes** (Li et al.) — Detection of 5 manipulation patterns

## Installation

```bash
cd trajectory-verification-cascade
pip install -r requirements.txt
```

## Status

🚧 In development
