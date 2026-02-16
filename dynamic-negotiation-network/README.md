# Dynamic Negotiation Network

A multi-agent LLM negotiation system where agent communication topology dynamically adapts based on semantic matching of each agent's current needs and offers during the negotiation process.

## Overview

This project implements a novel approach to multi-agent negotiations inspired by research from:

- **DyTopo**: Dynamic topology routing via semantic matching of agent "needs" and "offers"
- **AgenticPay**: Multi-agent LLM negotiation benchmark with 110+ tasks
- **Reactive Knowledge Representation**: Real-time belief updates for reactive reasoning

## Features

- **Dynamic Topology**: Communication graphs are reconstructed based on semantic matching
- **Agent Framework**: Flexible agent class with needs/offers attributes
- **Semantic Matching**: TF-IDF and hash-based embeddings for similarity computation
- **Negotiation Protocols**: Round-based message passing with accept/reject/counter strategies

## Installation

```bash
npm install
```

## Usage

```javascript
const { Agent, SemanticMatcher, DynamicTopologyManager } = require('./src');

// Create agents with needs and offers
const buyer = new Agent('buyer', {
  needs: ['discount', 'fast delivery', 'quality product'],
  offers: ['budget: $500', 'timeline: 2 weeks']
});

const seller = new Agent('seller', {
  needs: ['profit margin', 'quick payment', 'bulk order'],
  offers: ['price: $600', 'product: premium quality']
});

// Use semantic matcher to find compatible agents
const matcher = new SemanticMatcher();
const score = matcher.computeSimilarity(buyer.needs, seller.offers);

console.log('Compatibility score:', score);
```

## Architecture

### Core Components

1. **Agent**: Represents a negotiating party with needs, offers, and beliefs
2. **SemanticMatcher**: Computes semantic similarity between needs and offers
3. **Embedding**: TF-IDF or hash-based vector representations
4. **DynamicTopologyManager**: Builds and maintains communication graph
5. **AgentNetwork**: Connects agents in a network topology
6. **NegotiationRunner**: Orchestrates multi-round negotiations

## Development

Run tests:
```bash
npm test
```

Run demo:
```bash
npm run demo
```

## License

MIT License - see LICENSE file for details.
