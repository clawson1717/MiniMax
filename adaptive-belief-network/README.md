# Adaptive Belief Network

A multi-agent system where agents maintain dynamic belief networks with reactive updates and adaptive communication topology.

## Overview

This project implements an Adaptive Belief Network (ABN) system inspired by research in:
- **DyTopo**: Dynamic topology routing for efficient agent communication
- **Reactive Knowledge Representation**: Real-time belief updates
- **Guide to LLMs**: Practical guidance for large language model systems

## Features

- **Belief Representation**: Each belief has a proposition, confidence level (0-1), and justification
- **Belief Dependencies**: Graph-based belief network with dependency tracking
- **Reactive Updates**: Only beliefs affected by new information are updated (inspired by Reactive KR paper)
- **Adaptive Topology**: Dynamic communication between agents based on belief relevance

## Installation

```bash
npm install
```

## Usage

### Creating a Belief

```javascript
const { Belief } = require('./src/Belief');

const belief = new Belief(
  'The sky is blue',
  0.95,
  'Direct observation of clear sky'
);
```

### Creating a Belief Network

```javascript
const { BeliefNetwork } = require('./src/BeliefNetwork');

const network = new BeliefNetwork('agent-1');

// Add beliefs with dependencies
network.addBelief('The sky is blue', 0.95, 'Direct observation');
network.addBelief('Weather is clear', 0.90, 'No precipitation observed', ['The sky is blue']);
network.addBelief('Good hiking conditions', 0.85, 'Weather is clear', ['Weather is clear']);

// Update a belief - reactive updates propagate automatically
network.updateBelief('The sky is cloudy', 0.80, 'Clouds observed');

// Get belief
const belief = network.getBelief('The sky is blue');

// Get dependent beliefs
const dependents = network.getDependent('The sky is blue');
```

## API

### Belief Class

- `Belief(proposition, confidence, justification)` - Create a new belief
- `update(confidence, justification)` - Update belief confidence and justification

### BeliefNetwork Class

- `addBelief(proposition, confidence, justification, dependencies)` - Add a belief to the network
- `updateBelief(proposition, confidence, justification)` - Update a belief and trigger reactive updates
- `getBelief(proposition)` - Get a belief by proposition
- `getDependent(proposition)` - Get all beliefs that depend on a given belief

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Belief Network                         │
├─────────────────────────────────────────────────────────┤
│  ┌─────────┐     ┌─────────┐     ┌─────────┐          │
│  │ Belief  │────▶│ Belief  │────▶│ Belief  │          │
│  │   A     │     │   B     │     │   C     │          │
│  └─────────┘     └─────────┘     └─────────┘          │
│       │               │                                 │
│       └───────────────┴───────────────▶ Reactive       │
│                                      Updates            │
└─────────────────────────────────────────────────────────┘
```

## Reactive Update Mechanism

When a belief is updated, only dependent beliefs are recalculated:

1. Identify all beliefs that directly depend on the updated belief
2. Recalculate their confidence based on the update
3. Continue propagating to beliefs that depend on those beliefs
4. Stop when no further updates occur

This approach is inspired by the Reactive Knowledge Representation paper, which emphasizes efficient real-time belief updates.

## License

MIT
