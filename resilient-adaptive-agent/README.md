# Resilient Adaptive Agent (RAA)

A web agent combining three ArXiv techniques: CATTS (dynamic compute allocation), Multi-Turn Attack resilience, and CM2 (checklist verification).

## Overview

The Resilient Adaptive Agent is a sophisticated web automation agent that:
- Dynamically allocates compute based on uncertainty (CATTS).
- Detects and recovers from adversarial attacks (based on 5 patterns).
- Uses fine-grained checklist verification (CM2) for atomic action validation.

## Techniques

- **CATTS (Agentic Test-Time Scaling)**: Dynamic compute allocation using uncertainty statistics from vote distributions.
- **Multi-Turn Attack Resilience**: Detection of 5 failure modes: Self-Doubt, Social Conformity, Suggestion Hijacking, Emotional Susceptibility, and Reasoning Fatigue.
- **CM2 Checklist Rewards**: Fine-grained binary criteria for multi-step RL-style verification.

## Project Structure

- `src/`: Core logic and agent implementation.
- `tests/`: Unit tests for components and the RAA orchestrator.

## Installation

```bash
cd resilient-adaptive-agent
npm install
```

## Quick Start

```javascript
const { RAA, Agent } = require('./src');

// Create an agent
const agent = new RAA({
  maxSteps: 20,
  verbose: true
});

// Run a task
const result = await agent.runTask({
  url: 'https://example.com',
  actions: [
    { type: 'navigate', url: 'https://example.com' },
    { type: 'click', selector: '#search' },
    { type: 'type', selector: '#query', text: 'test' }
  ]
});
```

## Status

✅ **COMPLETE**

---
*Created and maintained by Clawson (🦞).*
