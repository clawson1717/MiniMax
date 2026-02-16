# Dynamic Negotiation Network

A multi-agent LLM negotiation system where agent communication topology dynamically adapts based on semantic matching of each agent's current needs and offers during the negotiation process.

## Overview

The Dynamic Negotiation Network (DNN) implements a novel approach to multi-agent negotiations inspired by research in dynamic topology routing and multi-agent LLM benchmarks. Unlike static communication structures where all agents can communicate with each other, DNN continuously reconstructs the communication graph based on semantic compatibility between what agents need and what they can offer.

This approach enables:
- **Efficient communication** - Agents only talk to relevant partners
- **Adaptive negotiations** - Topology evolves as needs are satisfied
- **Scalable coordination** - Network structure reflects actual compatibility
- **Semantic awareness** - Matching based on meaning, not just keywords

## Features

- **Dynamic Topology**: Communication graphs are reconstructed each round based on semantic matching
- **Agent Framework**: Flexible agent class with needs/offers/beliefs structure
- **Semantic Matching**: TF-IDF and hash-based embeddings for compatibility computation
- **Negotiation Protocols**: Round-based message passing with accept/reject/counter strategies
- **Strategy System**: Pluggable negotiation strategies (Accept, Reject, Counter, Random)
- **REST API**: Full HTTP API for managing negotiations programmatically
- **Comprehensive Evaluation**: Metrics including precision, recall, F1, payoff calculation

## Installation

```bash
npm install
```

## Quick Start

```javascript
const { Agent, AgentNetwork, NegotiationRunner } = require('./src');

// Create agents with needs and offers
const buyer = new Agent('buyer', {
  needs: ['discount', 'fast delivery'],
  offers: ['budget: $500', 'payment: immediate'],
  strategy: 'counter'
});

const seller = new Agent('seller', {
  needs: ['profit margin', 'quick payment'],
  offers: ['price: $600', 'delivery: 1 week'],
  strategy: 'accept'
});

// Create network and add agents
const network = new AgentNetwork();
network.addAgent(buyer);
network.addAgent(seller);

// Activate network (builds initial topology)
network.activate();

// Run negotiation
const runner = new NegotiationRunner(network, {
  maxRounds: 10,
  dynamicTopology: true,
  verbose: true
});

const result = runner.run();
console.log('Negotiation complete:', result);
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Dynamic Negotiation Network                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    NegotiationRunner                             │   │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐                        │   │
│  │  │ Round 1 │─▶│ Round 2 │─▶│ Round N │                        │   │
│  │  │         │  │         │  │         │                        │   │
│  │  │Topology │  │Topology │  │Topology │                        │   │
│  │  │Rebuild  │  │Rebuild  │  │Rebuild  │                        │   │
│  │  └────┬────┘  └────┬────┘  └────┬────┘                        │   │
│  │       │            │            │                              │   │
│  │       └────────────┴────────────┴────────────┐                 │   │
│  │                                             │                 │   │
│  │                    ┌─────────────────────────▼──────────┐     │   │
│  │                    │     NegotiationRound               │     │   │
│  │                    │  ┌──────────────────────────────┐  │     │   │
│  │                    │  │ Agent Negotiation Protocols  │  │     │   │
│  │                    │  │   offer → counter → accept   │  │     │   │
│  │                    │  └──────────────────────────────┘  │     │   │
│  │                    └────────────────────────────────────┘     │   │
│  └───────────────────────────────────────────────────────────────┘   │
│                             │                                          │
│  ┌──────────────────────────┼──────────────────────────────────────┐   │
│  │                    AgentNetwork                                  │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │   │
│  │  │   Agent 1   │  │   Agent 2   │  │   Agent N   │            │   │
│  │  │ ┌─────────┐ │  │ ┌─────────┐ │  │ ┌─────────┐ │            │   │
│  │  │ │ Needs   │ │  │ │ Needs   │ │  │ │ Needs   │ │            │   │
│  │  │ │ Offers  │ │  │ │ Offers  │ │  │ │ Offers  │ │            │   │
│  │  │ │ Beliefs │ │  │ │ Beliefs │ │  │ │ Beliefs │ │            │   │
│  │  │ └─────────┘ │  │ └─────────┘ │  │ └─────────┘ │            │   │
│  │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘            │   │
│  │         │                │                │                    │   │
│  │         └────────────────┼────────────────┘                    │   │
│  │                          │                                     │   │
│  │                   ┌──────▼──────┐                              │   │
│  │                   │  Message    │                              │   │
│  │                   │  System     │                              │   │
│  │                   └──────┬──────┘                              │   │
│  └──────────────────────────┼──────────────────────────────────────┘   │
│                             │                                           │
│  ┌──────────────────────────┼──────────────────────────────────────┐   │
│  │                    DynamicTopologyManager                        │   │
│  │  ┌──────────────────────────────────────────────────────────┐  │   │
│  │  │  Dynamic Graph: Edges based on semantic similarity       │  │   │
│  │  │                                                          │  │   │
│  │  │   Agent A ──0.85── Agent B                               │  │   │
│  │  │     │                  │                                 │  │   │
│  │  │   0.72               0.91                                │  │   │
│  │  │     │                  │                                 │  │   │
│  │  │   Agent C ──0.64── Agent D                               │  │   │
│  │  │                                                          │  │   │
│  │  │  Similarity = semantic_match(needs, offers)              │  │   │
│  │  └──────────────────────────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    SemanticMatcher                               │   │
│  │  ┌──────────────────────────────────────────────────────────┐  │   │
│  │  │  Embedding Methods: TF-IDF | Hash | Combined             │  │   │
│  │  │                                                          │  │   │
│  │  │  "fast delivery"  ──0.82──  "delivery: 1 week"          │  │   │
│  │  │  "discount"       ──0.75──  "price: $600"               │  │   │
│  │  └──────────────────────────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘

Negotiation Flow:
┌─────────────────────────────────────────────────────────────────┐
│                      Single Round                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Round Start                                                     │
│       │                                                          │
│       ▼                                                          │
│  ┌─────────────────┐                                             │
│  │ Rebuild Topology│  ← Semantic match agents' needs/offers     │
│  └────────┬────────┘                                             │
│           │                                                      │
│           ▼                                                      │
│  ┌─────────────────┐     ┌─────────────────┐                   │
│  │  Agent 1 Sends  │────▶│  Agent 2 Receives│                   │
│  │    Offers       │     │    Offers        │                   │
│  └─────────────────┘     └────────┬────────┘                   │
│                                   │                              │
│                                   ▼                              │
│                          ┌─────────────────┐                    │
│                          │ Strategy Eval   │                    │
│                          │  - Accept?      │                    │
│                          │  - Reject?      │                    │
│                          │  - Counter?     │                    │
│                          └────────┬────────┘                    │
│                                   │                              │
│                          ┌────────▼────────┐                    │
│                          │  Send Response  │                    │
│                          │ (accept/reject/ │                    │
│                          │  counter)       │                    │
│                          └─────────────────┘                    │
│                                                                  │
│  Round End                                                       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Key Concepts

### Agents

An agent consists of:
- **Needs**: What the agent wants to obtain (e.g., "fast delivery", "discount")
- **Offers**: What the agent can provide (e.g., "price: $500", "1 week delivery")
- **Beliefs**: Agent's beliefs about other agents (for reactive reasoning)
- **Strategy**: Negotiation strategy (accept, reject, counter, random)
- **History**: Record of all negotiation events

### Semantic Matching

The system uses semantic similarity to determine agent compatibility:

1. **TF-IDF Embeddings**: Term frequency-inverse document frequency vectors
2. **Hash Embeddings**: Random projection hash vectors for fast matching
3. **Combined**: Average of both methods for robust matching

Similarity scores range from 0 (completely different) to 1 (identical meaning).

### Dynamic Topology

Communication topology evolves based on semantic compatibility:

1. Each round, compute similarity between all agent pairs
2. Create edges when similarity exceeds threshold
3. Respect maximum neighbors constraint
4. Bidirectional edges require mutual compatibility

This ensures agents only communicate with relevant partners.

### Negotiation Rounds

Each round follows this protocol:

1. **Topology Update**: Rebuild communication graph
2. **Message Processing**: Agents process pending messages
3. **Strategy Evaluation**: Each agent evaluates offers using their strategy
4. **Response Generation**: Agents send accept/reject/counter responses
5. **Outcome Recording**: Track successful negotiations

### Strategies

Available negotiation strategies:

- **AcceptStrategy**: Accepts offers meeting similarity threshold
- **RejectStrategy**: Always rejects offers
- **CounterStrategy**: Proposes counter-offers based on gap analysis
- **RandomStrategy**: Random decisions with configurable probabilities

## Usage Examples

### Creating Agents

```javascript
const { Agent } = require('./src');

// Basic agent
const buyer = new Agent('buyer_1', {
  needs: ['discount', 'fast delivery'],
  offers: ['budget: $500', 'immediate payment']
});

// Agent with strategy
const seller = new Agent('seller_1', {
  needs: ['profit margin', 'quick payment'],
  offers: ['price: $600', 'delivery: 1 week'],
  strategy: 'counter'
});

// Add/remove needs and offers dynamically
buyer.addNeed('warranty');
buyer.removeNeed('discount');
seller.addOffer('extended warranty: 2 years');
```

### Semantic Matching

```javascript
const { SemanticMatcher } = require('./src');

// Create matcher with options
const matcher = new SemanticMatcher({
  method: 'tfidf',      // 'tfidf', 'hash', or 'combined'
  threshold: 0.3        // Minimum similarity for compatibility
});

// Compute similarity between need and offer
const score = matcher.computeSimilarity('fast delivery', 'delivery: 1 week');
console.log(`Similarity: ${score.toFixed(2)}`);

// Find best matching offer
const bestMatch = matcher.findBestMatch('discount', [
  'price: $600',
  'discount: 10%',
  'warranty: 1 year'
]);

// Check agent compatibility
const buyer = new Agent('buyer', { needs: ['fast delivery'], offers: ['$500'] });
const seller = new Agent('seller', { needs: ['quick payment'], offers: ['1 week delivery'] });

const compatible = matcher.findCompatibleAgents(buyer, [seller]);
console.log(`Compatible: ${compatible[0]?.meetsThreshold}`);

// Build full compatibility matrix
const agents = [buyer, seller];
const matrix = matcher.buildCompatibilityMatrix(agents);
```

### Dynamic Topology

```javascript
const { DynamicTopologyManager } = require('./src');

// Create topology manager
const topology = new DynamicTopologyManager({
  threshold: 0.3,        // Minimum similarity for edges
  maxNeighbors: 5,       // Maximum connections per agent
  bidirectional: true    // Require mutual compatibility
});

// Register agents
const buyer = new Agent('buyer', { needs: ['fast delivery'], offers: ['$500'] });
const seller = new Agent('seller', { needs: ['payment'], offers: ['fast delivery'] });

topology.registerAgent(buyer);
topology.registerAgent(seller);

// Rebuild topology based on current needs/offers
const stats = topology.rebuildTopology();
console.log(`Edges: ${stats.edgeCount}, Density: ${stats.density.toFixed(2)}`);

// Get neighbors for an agent
const neighbors = topology.getNeighbors('buyer');
console.log(`Buyer connected to: ${neighbors.join(', ')}`);

// Find path between agents
const path = topology.getPath('buyer', 'seller');
console.log(`Path: ${path?.join(' -> ')}`);

// Get weighted path (shortest path by similarity)
const weightedPath = topology.getWeightedPath('buyer', 'seller');
console.log(`Weighted path: ${weightedPath?.path.join(' -> ')}`);
```

### Running a Simple Negotiation

```javascript
const { Agent, AgentNetwork, NegotiationRunner } = require('./src');

// Create agents
const buyer = new Agent('buyer', {
  needs: ['discount', 'fast delivery'],
  offers: ['budget: $500', 'payment: immediate'],
  strategy: 'counter'
});

const seller = new Agent('seller', {
  needs: ['profit margin', 'quick payment'],
  offers: ['price: $450', 'delivery: 3 days'],
  strategy: 'accept'
});

// Create and configure network
const network = new AgentNetwork({
  topology: {
    threshold: 0.3,
    maxNeighbors: 3
  },
  maxRounds: 10
});

// Add agents and activate
network.addAgent(buyer);
network.addAgent(seller);
network.activate();

// Create runner with options
const runner = new NegotiationRunner(network, {
  maxRounds: 10,
  timeout: 30000,
  dynamicTopology: true,        // Enable topology adaptation
  topologyRebuildInterval: 1,   // Rebuild every round
  verbose: true
});

// Run negotiation
const result = runner.run();

// Examine results
console.log('Success:', result.success);
console.log('Rounds:', result.rounds);
console.log('Duration:', result.duration, 'ms');

// Check agent outcomes
result.agentResults.forEach(agent => {
  console.log(`${agent.id}: ${agent.needsSatisfied ? 'Satisfied' : 'Unsatisfied'}`);
  console.log(`  Remaining needs: ${agent.remainingNeeds.join(', ')}`);
});

// Get negotiation history
const history = runner.getHistory();
history.forEach(round => {
  console.log(`Round ${round.round}: ${round.statistics.messageCount} messages`);
});
```

### Using Different Strategies

```javascript
const { StrategyFactory } = require('./src');

// Available strategies
const strategies = StrategyFactory.getAvailableStrategies();
console.log('Available:', strategies); // ['accept', 'reject', 'counter', 'random']

// Create agents with different strategies
const flexibleBuyer = new Agent('flexible', {
  needs: ['product A'],
  offers: ['$100'],
  strategy: 'accept'  // Accepts if offer meets threshold
});

const toughNegotiator = new Agent('tough', {
  needs: ['product A'],
  offers: ['$150'],
  strategy: 'reject'  // Always rejects
});

const balancedNegotiator = new Agent('balanced', {
  needs: ['product A'],
  offers: ['$120'],
  strategy: 'counter' // Counters based on gap analysis
});

const randomNegotiator = new Agent('random', {
  needs: ['product A'],
  offers: ['$110'],
  strategy: 'random'  // Random accept/reject/counter
});

// Create strategy with custom options
const customAccept = StrategyFactory.create('accept', { threshold: 0.5 });
const customCounter = StrategyFactory.create('counter', { 
  counterThreshold: 0.4,
  concessionRate: 0.1 
});
```

### Dynamic Topology in Action

```javascript
const { Agent, AgentNetwork, NegotiationRunner } = require('./src');

// Create multiple agents with varying compatibility
const agents = [
  new Agent('tech_buyer', {
    needs: ['laptop', 'warranty'],
    offers: ['budget: $1000', 'cash payment']
  }),
  new Agent('tech_seller', {
    needs: ['quick sale', 'cash'],
    offers: ['laptop', '1 year warranty']
  }),
  new Agent('service_provider', {
    needs: ['referral', 'testimonial'],
    offers: ['installation', 'training']
  }),
  new Agent('consultant', {
    needs: ['project', 'payment'],
    offers: ['expertise', 'recommendation']
  })
];

// Create network with dynamic topology
const network = new AgentNetwork({
  topology: { threshold: 0.3, maxNeighbors: 3 }
});

agents.forEach(agent => network.addAgent(agent));
network.activate();

// Run negotiation with topology tracking
const runner = new NegotiationRunner(network, {
  maxRounds: 15,
  dynamicTopology: true,
  topologyRebuildInterval: 1,  // Rebuild every round
  verbose: true
});

const result = runner.run();

// Examine topology evolution
const topologyHistory = runner.getTopologyHistory();
topologyHistory.forEach(snapshot => {
  console.log(`\nRound ${snapshot.round}:`);
  console.log(`  Agents: ${snapshot.statistics.agentCount}`);
  console.log(`  Edges: ${snapshot.statistics.totalEdges}`);
  console.log(`  Changed: ${snapshot.changed ? 'Yes' : 'No'}`);
  
  // Show graph structure
  console.log('  Connections:');
  Object.entries(snapshot.graph).forEach(([agent, neighbors]) => {
    console.log(`    ${agent} -> [${neighbors.join(', ')}]`);
  });
});
```

### REST API Usage

```javascript
// Start the server
// npm run server

// Or programmatically
const { app, server } = require('./src/server');
```

#### Create a Negotiation

```bash
curl -X POST http://localhost:3000/api/negotiations \
  -H "Content-Type: application/json" \
  -d '{
    "agents": [
      {
        "template": "buyer",
        "needs": ["discount", "fast delivery"],
        "offers": ["$500", "immediate payment"],
        "strategy": "counter"
      },
      {
        "template": "seller",
        "needs": ["profit margin", "quick payment"],
        "offers": ["product X", "1 week delivery"],
        "strategy": "accept"
      }
    ],
    "options": {
      "maxRounds": 20,
      "timeout": 60000,
      "verbose": false,
      "topologyOptions": {
        "threshold": 0.3,
        "maxNeighbors": 5
      }
    }
  }'
```

Response:
```json
{
  "success": true,
  "negotiationId": "neg_1234567890_abc123",
  "status": "created",
  "agents": [...],
  "links": {
    "self": "/api/negotiations/neg_1234567890_abc123",
    "status": "/api/negotiations/neg_1234567890_abc123"
  }
}
```

#### Check Negotiation Status

```bash
curl http://localhost:3000/api/negotiations/neg_1234567890_abc123
```

#### List All Negotiations

```bash
curl http://localhost:3000/api/negotiations
```

#### Send a Message Between Agents

```bash
curl -X POST http://localhost:3000/api/negotiations/neg_1234567890_abc123/message \
  -H "Content-Type: application/json" \
  -d '{
    "from": "buyer",
    "to": "seller",
    "type": "offer",
    "content": {
      "offer": "$450 for product X"
    }
  }'
```

#### Step Negotiation Forward (Interactive Mode)

```bash
curl -X POST http://localhost:3000/api/negotiations/neg_1234567890_abc123/step
```

#### Reset a Negotiation

```bash
curl -X POST http://localhost:3000/api/negotiations/neg_1234567890_abc123/reset
```

#### Get Available Agent Templates

```bash
curl http://localhost:3000/api/agents
```

### Evaluation and Metrics

```javascript
const { Evaluator } = require('./src');

// Create evaluator
const evaluator = new Evaluator();

// Evaluate negotiation result
const evaluation = evaluator.evaluate(result);

console.log('Agreement reached:', evaluation.agreement);
console.log('Overall score:', evaluation.overallScore.toFixed(2));

// Per-agent scores
evaluation.agentScores.forEach(agent => {
  console.log(`${agent.id}:`);
  console.log(`  Satisfied: ${agent.needsSatisfied}`);
  console.log(`  Payoff: ${agent.payoff.toFixed(2)}`);
  console.log(`  Satisfaction rate: ${(agent.satisfactionRate * 100).toFixed(1)}%`);
});

// Negotiation metrics
console.log('Precision:', (evaluation.metrics.precision * 100).toFixed(1) + '%');
console.log('Recall:', (evaluation.metrics.recall * 100).toFixed(1) + '%');
console.log('F1 Score:', (evaluation.metrics.f1 * 100).toFixed(1) + '%');
console.log('Efficiency:', (evaluation.metrics.efficiency * 100).toFixed(1) + '%');
console.log('Success Rate:', (evaluation.metrics.successRate * 100).toFixed(1) + '%');

// Generate detailed report
const report = evaluator.generateReport(result);
console.log(report);

// Compare multiple negotiations
const results = [result1, result2, result3];
const comparison = evaluator.compare(results);
console.log('Best result index:', comparison.bestIndex);
console.log('Average F1:', comparison.averages.f1.toFixed(2));
```

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

## API Reference

### Agent Class

```javascript
const { Agent } = require('./src/Agent');
```

**Constructor**
- `new Agent(id, options)` - Create an agent
  - `id` (string): Unique identifier
  - `options.needs` (string[]): What the agent needs
  - `options.offers` (string[]): What the agent can offer
  - `options.beliefs` (Object): Initial beliefs about other agents
  - `options.strategy` (string): Negotiation strategy ('accept', 'reject', 'counter', 'random')
  - `options.utility` (number): Utility value (default: 1.0)

**Methods**
- `getNeeds()` - Get current needs
- `getOffers()` - Get current offers
- `setNeeds(newNeeds)` - Update needs
- `setOffers(newOffers)` - Update offers
- `addNeed(need)` - Add a need
- `removeNeed(need)` - Remove a need
- `addOffer(offer)` - Add an offer
- `removeOffer(offer)` - Remove an offer
- `updateBelief(agentId, belief)` - Update belief about another agent
- `getBelief(agentId)` - Get belief about another agent
- `recordEvent(event)` - Record a negotiation event
- `getHistory()` - Get negotiation history
- `clearHistory()` - Clear negotiation history
- `needsSatisfied()` - Check if all needs are satisfied
- `getStrategy()` - Get current strategy
- `setStrategy(strategy)` - Set strategy
- `respondToOffer(offer)` - Generate response to an offer
- `toJSON()` - Export to JSON
- `Agent.fromJSON(data)` - Create from JSON

### SemanticMatcher Class

```javascript
const { SemanticMatcher } = require('./src/SemanticMatcher');
```

**Constructor**
- `new SemanticMatcher(options)` - Create a semantic matcher
  - `options.method` (string): 'tfidf', 'hash', or 'combined' (default: 'tfidf')
  - `options.hashDimensions` (number): Dimensions for hash embeddings (default: 64)
  - `options.threshold` (number): Minimum similarity threshold (default: 0.3)

**Methods**
- `computeSimilarity(need, offer)` - Compute similarity between need and offer
- `computeSimilarityMatrix(needs, offers)` - Compute average similarity between arrays
- `findBestMatch(need, offers)` - Find best matching offer for a need
- `findCompatibleAgents(agent, candidates)` - Find compatible agents
- `buildCompatibilityMatrix(agents)` - Build full compatibility matrix
- `clearCache()` - Clear embeddings cache
- `getConfig()` - Get configuration

### DynamicTopologyManager Class

```javascript
const { DynamicTopologyManager } = require('./src/TopologyManager');
```

**Constructor**
- `new DynamicTopologyManager(options)` - Create topology manager
  - `options.matcherMethod` (string): Embedding method
  - `options.threshold` (number): Minimum similarity for edges (default: 0.3)
  - `options.maxNeighbors` (number): Maximum connections per agent (default: 5)
  - `options.bidirectional` (boolean): Require mutual compatibility (default: true)

**Methods**
- `registerAgent(agent)` - Register an agent
- `unregisterAgent(agentId)` - Unregister an agent
- `rebuildTopology()` - Rebuild entire topology
- `updateAgentTopology(agentId)` - Update topology for specific agent
- `getNeighbors(agentId)` - Get neighbor IDs
- `getNeighborAgents(agentId)` - Get neighbor agent objects
- `getPath(sourceId, targetId)` - Find path using BFS
- `getWeightedPath(sourceId, targetId)` - Find shortest weighted path
- `getGraph()` - Get adjacency list representation
- `getStatistics()` - Get topology statistics
- `getTopologyHistory()` - Get history of topology changes
- `areConnected(agentIdA, agentIdB)` - Check if agents are connected
- `getEdgeWeight(agentIdA, agentIdB)` - Get edge weight
- `getAgentsWithinHops(agentId, maxHops)` - Get agents within N hops
- `clearHistory()` - Clear topology history
- `reset()` - Reset topology

### AgentNetwork Class

```javascript
const { AgentNetwork } = require('./src/AgentNetwork');
```

**Constructor**
- `new AgentNetwork(options)` - Create agent network
  - `options.topology` (Object): Topology manager options
  - `options.maxRounds` (number): Maximum rounds (default: 10)
  - `options.onMessage` (Function): Message callback
  - `options.onRoundEnd` (Function): Round end callback

**Methods**
- `addAgent(agent)` - Add agent to network
- `removeAgent(agentId)` - Remove agent from network
- `getAgent(agentId)` - Get agent by ID
- `getAllAgents()` - Get all agents
- `rebuildTopology()` - Rebuild topology
- `getNeighbors(agentId)` - Get agent's neighbors
- `send(message)` - Send message between agents
- `broadcast(from, message)` - Broadcast to neighbors
- `getMessages(agentId)` - Get messages for agent
- `receiveMessage(agentId)` - Receive and clear messages
- `newRound()` - Start new negotiation round
- `getCurrentRound()` - Get current round number
- `getRoundHistory()` - Get round history
- `getRoundMessages(round)` - Get messages from specific round
- `isComplete()` - Check if negotiation is complete
- `getStatistics()` - Get network statistics
- `getPath(sourceId, targetId)` - Find path between agents
- `getWeightedPath(sourceId, targetId)` - Find weighted path
- `activate()` - Activate network
- `deactivate()` - Deactivate network
- `reset()` - Reset network state
- `toJSON()` - Export to JSON
- `getTopologyManager()` - Get topology manager instance

### NegotiationRound Class

```javascript
const { NegotiationRound } = require('./src/NegotiationRound');
```

**Constructor**
- `new NegotiationRound(network, options)` - Create negotiation round
  - `network` (AgentNetwork): The agent network
  - `options.strategyResolver` (Function): Strategy resolver function
  - `options.respectTopology` (boolean): Respect topology constraints

**Methods**
- `execute()` - Execute the negotiation round
- `getMessages()` - Get all messages from round
- `getMessagesByType(type)` - Get messages of specific type
- `getMessagesForAgent(agentId)` - Get messages involving agent
- `isComplete()` - Check if round is complete
- `getOutcome()` - Get round outcome
- `getStatistics()` - Get round statistics
- `getRoundNumber()` - Get round number

### NegotiationRunner Class

```javascript
const { NegotiationRunner } = require('./src/NegotiationRunner');
```

**Constructor**
- `new NegotiationRunner(network, options)` - Create negotiation runner
  - `network` (AgentNetwork): The agent network
  - `options.maxRounds` (number): Maximum rounds (default: 20)
  - `options.timeout` (number): Timeout in ms (default: 60000)
  - `options.verbose` (boolean): Enable verbose logging (default: false)
  - `options.dynamicTopology` (boolean): Enable topology adaptation (default: true)
  - `options.topologyRebuildInterval` (number): Rebuild every N rounds (default: 1)

**Methods**
- `run()` - Run full negotiation
- `getHistory()` - Get negotiation history
- `getRoundHistory(roundNumber)` - Get history for specific round
- `getResult()` - Get final negotiation result
- `getStatus()` - Get current status
- `getTopologyHistory()` - Get topology change history
- `getCurrentRound()` - Get current round number
- `reset()` - Reset for new negotiation
- `step()` - Run single step (interactive mode)
- `getSummary()` - Get negotiation summary

**Result Object**
```javascript
{
  success: boolean,           // Whether all needs satisfied
  status: string,             // 'completed', 'timeout', 'error', 'idle', 'running'
  rounds: number,             // Number of rounds executed
  duration: number,           // Duration in milliseconds
  dynamicTopology: boolean,   // Whether dynamic topology was enabled
  statistics: {
    totalOffers: number,
    totalAcceptances: number,
    totalRejections: number,
    totalCounters: number,
    agreementRate: number
  },
  agentResults: [{
    id: string,
    needsSatisfied: boolean,
    remainingNeeds: string[],
    historyLength: number
  }],
  history: [...],             // Round-by-round history
  topologyHistory: [...],     // Topology snapshots
  finalTopology: {...}        // Final topology state
}
```

### Strategy Classes

```javascript
const { 
  AcceptStrategy, 
  RejectStrategy, 
  CounterStrategy, 
  RandomStrategy,
  StrategyFactory 
} = require('./src/strategies');
```

#### AcceptStrategy

**Constructor**
- `new AcceptStrategy(options)`
  - `options.threshold` (number): Minimum similarity threshold (default: 0.3)

**Methods**
- `evaluate(context)` - Evaluate offer and return decision
- `getName()` - Get strategy name
- `getDescription()` - Get strategy description

#### RejectStrategy

**Constructor**
- `new RejectStrategy(options)`
  - `options.probability` (number): Rejection probability (default: 1.0)

**Methods**
- `evaluate(context)` - Always returns reject action

#### CounterStrategy

**Constructor**
- `new CounterStrategy(options)`
  - `options.counterThreshold` (number): Threshold to trigger counters (default: 0.2)
  - `options.concessionRate` (number): Concession rate per round (default: 0.1)

**Methods**
- `evaluate(context)` - Evaluate and return accept/counter/reject
- `reset()` - Reset round counter

#### RandomStrategy

**Constructor**
- `new RandomStrategy(options)`
  - `options.acceptProbability` (number): Probability of accepting (default: 0.3)
  - `options.counterProbability` (number): Probability of countering (default: 0.4)

**Methods**
- `evaluate(context)` - Random decision
- `setAcceptProbability(probability)` - Update accept probability
- `setCounterProbability(probability)` - Update counter probability

#### StrategyFactory

**Static Methods**
- `StrategyFactory.create(strategyName, options)` - Create strategy by name
- `StrategyFactory.getAvailableStrategies()` - Get list of available strategies

### Embedding Class

```javascript
const { Embedding } = require('./src/Embedding');
```

**Static Methods**
- `Embedding.tfidf(texts)` - Create TF-IDF embeddings
- `Embedding.hash(texts, dimensions)` - Create hash embeddings
- `Embedding.combined(texts, hashDimensions)` - Create combined embeddings
- `Embedding.cosineSimilarity(vec1, vec2)` - Compute cosine similarity
- `Embedding.euclideanDistance(vec1, vec2)` - Compute Euclidean distance

### Evaluator Class

```javascript
const { Evaluator } = require('./src/Evaluator');
```

**Constructor**
- `new Evaluator(options)` - Create evaluator

**Methods**
- `evaluate(negotiationResult)` - Evaluate negotiation outcome
- `calculatePayoff(agentResult, negotiationResult)` - Calculate agent payoff
- `isAgreement(negotiationResult)` - Check if agreement reached
- `getMetrics(negotiationResult)` - Get precision, recall, F1 metrics
- `compare(results)` - Compare multiple negotiation results
- `generateReport(negotiationResult)` - Generate detailed text report

### REST API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Health check |
| GET | `/api/agents` | List agent templates |
| GET | `/api/agents/:templateId` | Get specific template |
| GET | `/api/negotiations` | List all negotiations |
| POST | `/api/negotiations` | Create new negotiation |
| GET | `/api/negotiations/:id` | Get negotiation status |
| DELETE | `/api/negotiations/:id` | Delete negotiation |
| POST | `/api/negotiations/:id/message` | Send message between agents |
| GET | `/api/negotiations/:id/messages` | Get negotiation messages |
| POST | `/api/negotiations/:id/step` | Step negotiation forward |
| POST | `/api/negotiations/:id/reset` | Reset negotiation |

## Project Structure

```
dynamic-negotiation-network/
├── src/
│   ├── Agent.js                   # Agent class with needs/offers/beliefs
│   ├── SemanticMatcher.js         # Semantic matching using embeddings
│   ├── TopologyManager.js         # Dynamic topology management
│   ├── AgentNetwork.js            # Network of connected agents
│   ├── NegotiationRound.js        # Single negotiation round
│   ├── NegotiationRunner.js       # Multi-round negotiation orchestration
│   ├── Embedding.js               # TF-IDF and hash embeddings
│   ├── strategies.js              # Negotiation strategies
│   ├── Evaluator.js               # Outcome evaluation and metrics
│   ├── server.js                  # Express REST API server
│   ├── cli.js                     # Command-line interface
│   ├── demo.js                    # Demo script
│   └── index.js                   # Main exports
├── tests/                         # Unit tests
├── examples/                      # Example scripts
│   └── demo.js                    # Comprehensive demo
├── package.json
└── README.md
```

## ArXiv Sources and Theoretical Background

This implementation is informed by research in several areas:

### Dynamic Topology (DyTopo)

The dynamic topology approach is inspired by research on adaptive routing and topology management in multi-agent systems:

- **Dynamic Network Reconfiguration**: Communication graphs that adapt based on current context and requirements
- **Semantic Routing**: Routing decisions based on content meaning rather than fixed addresses
- **Topology Optimization**: Continuous optimization of network structure for efficient communication

Key principles from DyTopo research:
1. Topology reconstruction based on semantic matching
2. Bidirectional compatibility requirements for stable connections
3. Neighbor limits to prevent communication overhead
4. Weighted paths for optimal message routing

### AgenticPay Benchmark

The negotiation protocols and evaluation metrics draw from multi-agent LLM negotiation benchmarks:

- **Multi-round Negotiation**: Sequential offer-counteroffer protocols
- **Strategy Diversity**: Different agent behaviors (cooperative, competitive, neutral)
- **Outcome Evaluation**: Metrics including success rate, efficiency, and fairness
- **Belief Representation**: Agent beliefs about other agents' preferences and behaviors

Key benchmark considerations:
1. Fair evaluation across different strategy types
2. Measurement of both individual and collective outcomes
3. Efficiency metrics (rounds to agreement, message count)
4. Robustness to different negotiation scenarios

### Reactive Knowledge Representation

The belief update mechanism follows principles from reactive knowledge systems:

- **Belief State Management**: Each agent maintains beliefs about others
- **Event-Driven Updates**: Beliefs updated based on negotiation events
- **Temporal Tracking**: Timestamps for belief freshness
- **Context Preservation**: Beliefs preserve context and justification

### Semantic Matching Research

The semantic similarity computation uses established techniques:

- **TF-IDF Vectorization**: Term frequency-inverse document frequency for text similarity
- **Hash Embeddings**: Locality-sensitive hashing for fast approximate matching
- **Cosine Similarity**: Standard vector similarity metric
- **Multi-method Fusion**: Combining multiple embedding approaches

### Multi-Agent Negotiation Theory

The negotiation protocols implement established multi-agent negotiation concepts:

- **Alternating Offers Protocol**: Standard negotiation game structure
- **Strategy Spaces**: Accept, reject, and counter-offer actions
- **Utility Functions**: Payoff calculation based on needs satisfaction
- **Convergence Criteria**: Agreement detection and termination conditions

## License

MIT

## Contributing

Contributions are welcome! Please ensure all tests pass before submitting pull requests:

```bash
npm test
```

When contributing, please follow the existing code style and add tests for new features.
