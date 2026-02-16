# Adaptive Belief Network

A multi-agent system where agents maintain dynamic belief networks with reactive updates and adaptive communication topology.

## Overview

The Adaptive Belief Network (ABN) system is a JavaScript/Node.js implementation of a sophisticated multi-agent belief management system. Each agent maintains its own belief network where beliefs are represented as propositions with confidence levels and justifications. The system implements reactive belief propagation - when a belief changes, only dependent beliefs are recalculated, making updates efficient and scalable.

This project is inspired by research in dynamic topology routing, reactive knowledge representation, and practical large language model agent architectures.

## Features

- **Belief Representation**: Each belief has a proposition, confidence level (0-1), and justification
- **Belief Dependencies**: Graph-based belief network with dependency tracking
- **Reactive Updates**: Only beliefs affected by new information are updated
- **Adaptive Topology**: Dynamic communication between agents based on belief relevance
- **Multi-Agent Simulation**: Orchestrate complex multi-agent scenarios with time steps
- **Visualization**: Color-coded console output with belief evolution tracking
- **Persistence**: Save and load belief networks to/from JSON files
- **Analysis**: Tools for analyzing belief evolution and network statistics

## Installation

```bash
npm install
```

## Quick Start

```javascript
const { Agent, BeliefNetwork, Simulator, Visualizer } = require('./src');

// Create agents with belief networks
const alice = new Agent('alice-001', 'Alice');
const bob = new Agent('bob-001', 'Bob');

// Add beliefs with dependencies
alice.addBelief('The sky is blue', 0.95, 'Direct observation');
alice.addBelief('Weather is clear', 0.90, 'No precipitation', ['The sky is blue']);

// Create simulator and add agents
const sim = new Simulator({ maxTimeSteps: 10 });
sim.addAgent(alice);
sim.addAgent(bob);

// Run simulation
sim.run();

// Visualize results
const viz = new Visualizer();
viz.visualize(sim);
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Adaptive Belief Network                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                         Simulator                                │   │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐           │   │
│  │  │ Agent 1 │  │ Agent 2 │  │ Agent 3 │  │ Agent N │           │   │
│  │  │ ┌─────┐ │  │ ┌─────┐ │  │ ┌─────┐ │  │ ┌─────┐ │           │   │
│  │  │ │BN   │ │  │ │BN   │ │  │ │BN   │ │  │ │BN   │ │           │   │
│  │  │ └─────┘ │  │ └─────┘ │  │ └─────┘ │  │ └─────┘ │           │   │
│  │  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘           │   │
│  │       │            │            │            │                  │   │
│  │       └────────────┴────────────┴────────────┘                  │   │
│  │                          │                                        │   │
│  │                    ┌─────▼─────┐                                  │   │
│  │                    │  Message  │                                  │   │
│  │                    │  System   │                                  │   │
│  │                    └─────┬─────┘                                  │   │
│  └──────────────────────────┼──────────────────────────────────────┘   │
│                             │                                           │
│  ┌──────────────────────────┼──────────────────────────────────────┐   │
│  │                    Topology Manager                              │   │
│  │  ┌──────────────────────────────────────────────────────────┐  │   │
│  │  │  Dynamic Graph: Agents connect based on belief relevance │  │   │
│  │  └──────────────────────────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘

Belief Network Internal Structure:
┌─────────────────────────────────────────────────────────────────┐
│                      Belief Network                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│    ┌─────────┐      ┌─────────┐      ┌─────────┐              │
│    │ Belief A│─────▶│ Belief B│─────▶│ Belief C│              │
│    │ "Sky is │      │ "Weather│      │ "Good   │              │
│    │  blue"  │      │  clear" │      │hiking"  │              │
│    └─────────┘      └─────────┘      └─────────┘              │
│         │                │                                   │
│         └────────────────┼──────────────┐                    │
│                          ▼              ▼                    │
│                    ┌──────────┐   ┌──────────┐              │
│                    │ Update   │   │ Trigger  │              │
│                    │ Trigger  │   │ Manager  │              │
│                    └──────────┘   └──────────┘              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Key Concepts

### Beliefs

A belief consists of:
- **Proposition**: The statement being believed (e.g., "It will rain tomorrow")
- **Confidence**: A value between 0 and 1 indicating certainty
- **Justification**: The reason or evidence for the belief
- **History**: Track of how the belief has changed over time

### Belief Dependencies

Beliefs can depend on other beliefs. When a parent belief changes, dependent beliefs are automatically recalculated. This creates a network of interconnected knowledge.

### Reactive Updates

When a belief is updated, the system:
1. Identifies all beliefs that directly depend on the updated belief
2. Recalculates their confidence based on the change
3. Continues propagating to beliefs that depend on those beliefs
4. Stops when no further updates occur

This approach ensures only affected beliefs are recalculated, making the system efficient for large networks.

### Adaptive Topology

Agents communicate based on belief relevance. The topology manager dynamically adjusts connections between agents based on:
- Topic subscriptions
- Belief similarity
- Communication history

## Usage Examples

### Creating Beliefs

```javascript
const { Belief } = require('./src/Belief');

// Simple belief
const belief = new Belief(
  'The sky is blue',
  0.95,
  'Direct observation of clear sky'
);

// Update belief
belief.update(0.85, 'Clouds forming on horizon');
```

### Creating a Belief Network

```javascript
const { BeliefNetwork } = require('./src/BeliefNetwork');

const network = new BeliefNetwork('agent-1');

// Add beliefs with dependencies
network.addBelief('The sky is blue', 0.95, 'Direct observation');
network.addBelief('Weather is clear', 0.90, 'No precipitation', ['The sky is blue']);
network.addBelief('Good hiking conditions', 0.85, 'Weather is clear', ['Weather is clear']);

// Update triggers reactive updates
network.updateBelief('The sky is cloudy', 0.80, 'Clouds observed');

// Query the network
const belief = network.getBelief('The sky is blue');
const dependents = network.getDependent('Weather is clear');
```

### Creating Agents

```javascript
const { Agent } = require('./src/Agent');

const agent = new Agent('agent-001', 'Explorer');

// Add beliefs to agent's network
agent.addBelief('Water source nearby', 0.75, 'Heard flowing water');
agent.addBelief('Safe to drink', 0.60, 'Water looks clear', ['Water source nearby']);

// Get all beliefs
const allBeliefs = agent.getBeliefs();

// Subscribe to topics
agent.subscribe('weather');
agent.subscribe('navigation');
```

### Multi-Agent Simulation

```javascript
const { Simulator, Agent, Visualizer } = require('./src');

const sim = new Simulator({ maxTimeSteps: 20, tickDelay: 500 });

// Create and add agents
const alice = new Agent('alice', 'Alice');
alice.addBelief('Coffee needed', 0.9, 'Morning routine');

const bob = new Agent('bob', 'Bob');
bob.addBelief('Meeting at 10am', 0.95, 'Calendar entry');

sim.addAgent(alice);
sim.addAgent(bob);

// Run simulation
sim.run({ async: false });

// Visualize
const viz = new Visualizer();
viz.visualize(sim);
viz.printStats(sim);
```

### Agent Communication

```javascript
const { Agent, MessageSystem } = require('./src');

const alice = new Agent('alice', 'Alice');
const bob = new Agent('bob', 'Bob');

alice.addBelief('Stock prices rising', 0.80, 'Market analysis');

// Create message system
const messageSystem = new MessageSystem();
messageSystem.registerAgent(alice);
messageSystem.registerAgent(bob);

// Send belief update message
const message = {
  type: 'belief_update',
  senderId: 'alice',
  targetId: 'bob',
  payload: {
    proposition: 'Stock prices rising',
    confidence: 0.80,
    justification: 'Market analysis'
  }
};

messageSystem.sendMessage(message);

// Bob receives and processes
const response = bob.receiveMessage(message);
```

### Persistence

```javascript
const { Agent, Persistence } = require('./src');

const agent = new Agent('agent-1', 'Test Agent');
agent.addBelief('Test belief', 0.75, 'Test justification');

// Save to file
Persistence.save(agent, 'agent-state.json');

// Load from file
const loadedAgent = Persistence.load('agent-state.json');
```

### Visualization

```javascript
const { Simulator, Visualizer } = require('./src');

const sim = new Simulator({ maxTimeSteps: 10 });
// ... add agents and run simulation

const viz = new Visualizer({ useColors: true });

// Show current state
viz.visualize(sim);

// Compare time steps
viz.printComparison(sim, 0, 5);

// Track belief evolution
viz.printEvolution(sim, 'Your proposition here');

// Show statistics
viz.printStats(sim);
```

### Analysis

```javascript
const { Simulator, Analysis } = require('./src');

// Run simulation and get history
const sim = new Simulator({ maxTimeSteps: 20 });
// ... add agents and run

const history = sim.getHistory();

// Analyze belief changes
const changes = Analysis.detectBeliefChanges(history);

// Calculate convergence
const convergence = Analysis.calculateConvergence(history);

// Find anomalies
const anomalies = Analysis.detectAnomalies(history);
```

## API Reference

### Belief Class

```javascript
const { Belief } = require('./src/Belief');
```

**Constructor**
- `new Belief(proposition, confidence, justification)` - Create a belief
  - `proposition` (string): The statement being believed
  - `confidence` (number): Value between 0 and 1
  - `justification` (string): Reason for the belief

**Methods**
- `update(confidence, justification)` - Update belief values
- `toJSON()` - Get JSON representation
- `getHistory()` - Get change history

### BeliefNetwork Class

```javascript
const { BeliefNetwork } = require('./src/BeliefNetwork');
```

**Constructor**
- `new BeliefNetwork(agentId)` - Create a belief network

**Methods**
- `addBelief(proposition, confidence, justification, dependencies)` - Add belief
- `updateBelief(proposition, confidence, justification)` - Update and propagate
- `getBelief(proposition)` - Get single belief
- `getAllBeliefs()` - Get all beliefs
- `getDependent(proposition)` - Get dependent beliefs
- `getDependencies(proposition)` - Get belief dependencies
- `removeBelief(proposition)` - Remove a belief
- `clear()` - Clear all beliefs
- `toJSON()` - Export to JSON

### Agent Class

```javascript
const { Agent } = require('./src/Agent');
```

**Constructor**
- `new Agent(id, name)` - Create an agent

**Methods**
- `addBelief(proposition, confidence, justification, dependencies)` - Add belief
- `updateBelief(proposition, confidence, justification)` - Update belief
- `getBeliefs()` - Get all beliefs
- `getBelief(proposition)` - Get specific belief
- `receiveMessage(message)` - Process incoming message
- `subscribe(topic)` - Subscribe to topic
- `unsubscribe(topic)` - Unsubscribe from topic
- `isSubscribed(topic)` - Check subscription
- `getStats()` - Get agent statistics
- `getMessageHistory()` - Get message history
- `toJSON()` - Export to JSON

### Simulator Class

```javascript
const { Simulator } = require('./src/Simulator');
```

**Constructor**
- `new Simulator(options)` - Create simulator
  - `options.maxTimeSteps` (number): Maximum steps (default: 100)
  - `options.tickDelay` (number): Delay between steps in ms (default: 1000)

**Methods**
- `addAgent(agent, name)` - Add agent to simulation
- `removeAgent(agentId)` - Remove agent
- `getAgent(agentId)` - Get agent by ID
- `getAllAgents()` - Get all agents
- `run(config)` - Run simulation
  - `config.timeSteps` (number): Steps to run
  - `config.async` (boolean): Run asynchronously
- `pause()` - Pause simulation
- `resume()` - Resume simulation
- `stop()` - Stop simulation
- `getHistory(filters)` - Get simulation history
- `getBeliefStates()` - Get current belief states
- `getBeliefStatesAt(timeStep)` - Get beliefs at specific step
- `getStats()` - Get simulation statistics
- `reset()` - Reset simulation
- `on(event, callback)` - Add event listener
- `off(event, callback)` - Remove event listener

**Events**
- `simulationStart` - Simulation started
- `simulationComplete` - Simulation completed
- `simulationPaused` - Simulation paused
- `timeStepStart` - Time step started
- `timeStepEnd` - Time step ended
- `agentAdded` - Agent added
- `agentRemoved` - Agent removed

### TopologyManager Class

```javascript
const { TopologyManager } = require('./src/TopologyManager');
```

**Constructor**
- `new TopologyManager()` - Create topology manager

**Methods**
- `addAgent(agentId)` - Add agent to topology
- `removeAgent(agentId)` - Remove agent
- `addEdge(fromId, toId, weight)` - Add connection
- `removeEdge(fromId, toId)` - Remove connection
- `getNeighbors(agentId)` - Get connected agents
- `getPath(fromId, toId)` - Get communication path
- `updateWeights(agentId, beliefRelevance)` - Update edge weights
- `getGraph()` - Get topology graph

### MessageSystem Class

```javascript
const { MessageSystem } = require('./src/MessageSystem');
```

**Constructor**
- `new MessageSystem(options)` - Create message system

**Methods**
- `registerAgent(agent)` - Register agent
- `unregisterAgent(agentId)` - Unregister agent
- `sendMessage(message)` - Send message
- `broadcast(message, topic)` - Broadcast to topic
- `setRelevanceFilter(filter)` - Set relevance filter
- `getMessageHistory(agentId)` - Get message history
- `clearHistory()` - Clear message history

### Visualizer Class

```javascript
const { Visualizer } = require('./src/Visualizer');
```

**Constructor**
- `new Visualizer(options)` - Create visualizer
  - `options.useColors` (boolean): Enable colors (default: true)
  - `options.showTimestamps` (boolean): Show timestamps (default: true)
  - `options.indentSize` (number): Indentation size (default: 2)

**Methods**
- `visualize(simulator, options)` - Visualize simulator state
- `printState(states, options)` - Print belief state
- `printComparison(simulator, from, to)` - Print comparison
- `printStats(simulator)` - Print statistics
- `printEvolution(simulator, proposition)` - Print evolution
- `printSeparator()` - Print separator line
- `createBar(value, width)` - Create bar chart

### Persistence Module

```javascript
const { Persistence } = require('./src/Persistence');
```

**Methods**
- `Persistence.save(agent, filename)` - Save agent to file
- `Persistence.load(filename)` - Load agent from file
- `Persistence.saveNetwork(network, filename)` - Save network
- `Persistence.loadNetwork(filename)` - Load network

### Analysis Module

```javascript
const { Analysis } = require('./src/Analysis');
```

**Methods**
- `Analysis.detectBeliefChanges(history)` - Detect belief changes
- `Analysis.calculateConvergence(history)` - Calculate convergence
- `Analysis.detectAnomalies(history)` - Detect anomalies
- `Analysis.compareAgents(history, agent1, agent2)` - Compare agents
- `Analysis.getBeliefTrends(history, proposition)` - Get trends

### UpdateTrigger Class

```javascript
const { UpdateTrigger, UpdateTriggerManager } = require('./src/UpdateTrigger');
```

**Constructor**
- `new UpdateTrigger(type, proposition, data)` - Create trigger

**UpdateTriggerManager**
- `new UpdateTriggerManager()` - Create manager
- `register(type, handler)` - Register trigger handler
- `trigger(type, proposition, data)` - Trigger an event
- `getHistory()` - Get trigger history

## Running the Demo

```bash
# Run the demo script
npm start

# Or run directly
node examples/demo.js
```

## Running Tests

```bash
# Run all tests
npm test

# Run with coverage
npm run test:coverage
```

## Project Structure

```
adaptive-belief-network/
├── src/
│   ├── Belief.js           # Belief class
│   ├── BeliefNetwork.js    # Belief network with reactive updates
│   ├── Agent.js            # Agent with belief network
│   ├── TopologyManager.js  # Dynamic topology management
│   ├── MessageSystem.js    # Agent communication
│   ├── UpdateTrigger.js    # Update triggers
│   ├── Simulator.js        # Multi-agent simulation
│   ├── Visualizer.js       # Console visualization
│   ├── Persistence.js      # Save/load functionality
│   ├── Analysis.js         # Analysis tools
│   ├── cli.js             # Command-line interface
│   └── index.js           # Main exports
├── tests/                  # Unit tests
├── examples/              # Example scripts
├── package.json
└── README.md
```

## Theoretical Background

This implementation is inspired by several research areas:

### Reactive Knowledge Representation

The reactive update mechanism draws from research in reactive knowledge representation systems, where knowledge is updated efficiently by only recomputing affected portions of the knowledge base rather than rebuilding from scratch.

### Dynamic Topology Networks

The adaptive communication topology is inspired by research in dynamic routing and topology management for multi-agent systems, where communication pathways are dynamically adjusted based on relevance and efficiency.

### Belief Networks

The core belief representation follows Bayesian network principles, where beliefs have probabilistic confidence levels and dependencies between beliefs create directed acyclic graphs.

## ArXiv Sources and Inspiration

The following areas of research inform this implementation:

1. **Bayesian Networks**: Probabilistic graphical models for representing conditional dependencies
2. **Multi-Agent Systems**: Research on agent communication and coordination
3. **Knowledge Representation**: Reactive and efficient knowledge update mechanisms
4. **Dynamic Topology**: Adaptive routing and network management

## License

MIT

## Contributing

Contributions are welcome! Please ensure all tests pass before submitting pull requests.
