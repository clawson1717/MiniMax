# Resilient Adaptive Agent (RAA)

A web agent combining three ArXiv techniques: CATTS (dynamic compute allocation), Multi-Turn Attack resilience, and CM2 (checklist verification).

## Overview

The Resilient Adaptive Agent is a sophisticated web automation agent that:
- Dynamically allocates compute based on uncertainty (CATTS)
- Detects and recovers from adversarial attacks
- Uses fine-grained checklist verification (CM2)

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    RAA (Main Agent)                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────────┐  ┌──────────────────────────────────┐ │
│  │  CATTSAllocator │  │    FailureModeDetector           │ │
│  │  - allocate()    │  │    - Self-Doubt                 │ │
│  │  - scaleUp/Down  │  │    - Social Conformity           │ │
│  │  - shouldRequery│  │    - Suggestion Hijacking        │ │
│  └──────────────────┘  │    - Emotional Susceptibility   │ │
│                         │    - Reasoning Fatigue            │ │
│  ┌──────────────────┐  └──────────────────────────────────┘ │
│  │  ResilienceRecovery │                                    │
│  │  - recoverFromX()  │  ┌────────────────────────────────┐ │
│  │  - selectStrategy │  │  SelfVerificationSystem        │ │
│  └──────────────────┘  │  - verifyAction()              │ │
│                         │  - verifyNavigation()             │ │
│  ┌──────────────────┐  └────────────────────────────────┘ │
│  │  ChecklistReward  │                                       │
│  │  - createChecklist│                                       │
│  │  - getReward()    │                                       │
│  └──────────────────┘                                       │
└─────────────────────────────────────────────────────────────┘
```

## Installation

```bash
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

## Components

### CATTSAllocator
Dynamic compute allocation using uncertainty statistics.

```javascript
const { CATTSAllocator } = require('./src');
const allocator = new CATTSAllocator();
const depth = allocator.allocateCompute(uncertaintyTracker);
```

### FailureModeDetector
Detects 5 adversarial attack patterns.

```javascript
const { FailureModeDetector } = require('./src');
const detector = new FailureModeDetector();
detector.analyzeTurn({ userInput: '...', agentResponse: '...', confidence: 0.8 });
const report = detector.getFailureReport();
```

### ResilienceRecovery
Recovery strategies for each failure mode.

```javascript
const { ResilienceRecovery } = require('./src');
const recovery = new ResilienceRecovery();
const result = recovery.recoverFromSelfDoubt({ currentConfidence: 0.5 });
```

### ChecklistReward
Fine-grained binary verification.

```javascript
const { ChecklistReward } = require('./src');
const checklist = new ChecklistReward();
const id = checklist.createChecklist('navigation');
checklist.evaluateCriterion(id, 'url_reachable', true);
const reward = checklist.getReward(id);
```

### SelfVerification
Atomic verification after each action.

```javascript
const { SelfVerification } = require('./src');
const verifier = new SelfVerification();
const result = verifier.verifyAction(action, actionResult, context);
```

## CLI Usage

```bash
# Run a task
node src/cli.js run --config config.json

# Initialize config
node src/cli.js init

# Show status
node src/cli.js status
```

## Testing

```bash
npm test
```

## ArXiv Sources

- **CATTS**: Agentic Test-Time Scaling - Dynamic compute allocation via uncertainty
- **Multi-Turn Attacks**: Consistency of LLMs under adversarial attacks
- **CM2**: Checklist Rewards for RL - Fine-grained binary verification

## License

MIT
