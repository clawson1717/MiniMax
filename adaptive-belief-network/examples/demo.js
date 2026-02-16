/**
 * Adaptive Belief Network Demo
 * 
 * This script demonstrates the key features of the Adaptive Belief Network system:
 * - Creating agents with belief networks
 * - Running a simulation
 * - Observing belief updates
 * - Visualizing results
 * 
 * Run with: node examples/demo.js
 */

const { 
  Agent, 
  BeliefNetwork, 
  Simulator, 
  Visualizer, 
  Persistence,
  Analysis,
  MessageSystem,
  TopologyManager
} = require('../src');

const fs = require('fs');

console.log('='.repeat(70));
console.log('       ADAPTIVE BELIEF NETWORK DEMO');
console.log('='.repeat(70));
console.log('');

// ============================================================================
// PART 1: Basic Belief Network
// ============================================================================
console.log('\n📌 PART 1: Creating a Belief Network\n');

const { Belief } = require('../src/Belief');

// Create a simple belief
const skyBelief = new Belief(
  'The sky is blue',
  0.95,
  'Direct observation from window'
);

console.log(`Created belief: "${skyBelief.proposition}"`);
console.log(`  Confidence: ${(skyBelief.confidence * 100).toFixed(1)}%`);
console.log(`  Justification: ${skyBelief.justification}`);

// Update the belief
skyBelief.update(0.75, 'Clouds are forming');
console.log(`\nAfter update:`);
console.log(`  Confidence: ${(skyBelief.confidence * 100).toFixed(1)}%`);
console.log(`  Justification: ${skyBelief.justification}`);

// ============================================================================
// PART 2: Belief Network with Dependencies
// ============================================================================
console.log('\n📌 PART 2: Belief Network with Dependencies\n');

const network = new BeliefNetwork('explorer-001');

console.log('Adding beliefs with dependencies...');

// Root belief - no dependencies
network.addBelief('The sky is blue', 0.95, 'Clear sky observed');

// Second level - depends on sky
network.addBelief('Weather is clear', 0.90, 'No precipitation, blue sky', 
  ['The sky is blue']);

// Third level - depends on weather
network.addBelief('Good hiking conditions', 0.85, 'Weather is clear, warm temperature', 
  ['Weather is clear']);

console.log('\nInitial belief network:');
const allBeliefs = network.getAllBeliefs();
allBeliefs.forEach(b => {
  console.log(`  • "${b.proposition}" (${(b.confidence * 100).toFixed(1)}%)`);
});

// Trigger reactive update
console.log('\nUpdating "The sky is blue" to 0.60 (cloudy)...');
network.updateBelief('The sky is blue', 0.60, 'Clouds observed on horizon');

console.log('\nAfter reactive update:');
const updatedBeliefs = network.getAllBeliefs();
updatedBeliefs.forEach(b => {
  console.log(`  • "${b.proposition}" (${(b.confidence * 100).toFixed(1)}%) - ${b.justification}`);
});

// Show dependent beliefs
console.log('\nBeliefs dependent on "The sky is blue":');
const dependents = network.getDependent('The sky is blue');
dependents.forEach(b => {
  console.log(`  → "${b.proposition}"`);
});

// ============================================================================
// PART 3: Creating Agents
// ============================================================================
console.log('\n📌 PART 3: Creating Agents with Belief Networks\n');

const alice = new Agent('alice-001', 'Alice');
const bob = new Agent('bob-001', 'Bob');
const charlie = new Agent('charlie-001', 'Charlie');

console.log('Created agents:');
console.log(`  • ${alice.name} (${alice.id})`);
console.log(`  • ${bob.name} (${bob.id})`);
console.log(`  • ${charlie.name} (${charlie.id})`);

// Add beliefs to Alice
alice.addBelief('Coffee needed', 0.9, 'Morning routine - tired');
alice.addBelief('Meeting at 10am', 0.95, 'Calendar entry');
alice.addBelief('Team project on track', 0.75, 'Last update from team lead');

// Add beliefs to Bob
bob.addBelief('Stock prices rising', 0.80, 'Market analysis');
bob.addBelief('Technology sector strong', 0.85, 'Tech stocks up', ['Stock prices rising']);
bob.addBelief('Good investment opportunities', 0.70, 'Multiple sectors looking good', ['Technology sector strong']);

// Add beliefs to Charlie
charlie.addBelief('Weather forecast: sunny', 0.90, 'Weather app prediction');
charlie.addBelief('Good day for outdoor activity', 0.80, 'Weather forecast: sunny', ['Weather forecast: sunny']);

console.log('\nAgent beliefs:');
console.log(`  Alice: ${alice.getBeliefs().length} beliefs`);
console.log(`  Bob: ${bob.getBeliefs().length} beliefs`);
console.log(`  Charlie: ${charlie.getBeliefs().length} beliefs`);

// Subscribe to topics
alice.subscribe('weather');
alice.subscribe('finance');
bob.subscribe('finance');
charlie.subscribe('weather');

console.log('\nTopic subscriptions:');
console.log(`  Alice: ${Array.from(alice.subscriptions).join(', ')}`);
console.log(`  Bob: ${Array.from(bob.subscriptions).join(', ')}`);
console.log(`  Charlie: ${Array.from(charlie.subscriptions).join(', ')}`);

// ============================================================================
// PART 4: Multi-Agent Simulation
// ============================================================================
console.log('\n📌 PART 4: Running Multi-Agent Simulation\n');

const sim = new Simulator({ maxTimeSteps: 15, tickDelay: 100 });

// Add agents to simulation
sim.addAgent(alice);
sim.addAgent(bob);
sim.addAgent(charlie);

console.log('Simulating 15 time steps...');

// Set up event listeners
sim.on('timeStepStart', ({ timeStep }) => {
  if (timeStep % 5 === 0) {
    console.log(`  Step ${timeStep} started...`);
  }
});

sim.on('simulationComplete', ({ totalTimeSteps, historyLength }) => {
  console.log(`\n✅ Simulation complete!`);
  console.log(`   Total time steps: ${totalTimeSteps}`);
  console.log(`   History entries: ${historyLength}`);
});

// Run simulation synchronously
sim.run({ async: false });

// ============================================================================
// PART 5: Visualization
// ============================================================================
console.log('\n📌 PART 5: Visualization\n');

const viz = new Visualizer({ useColors: true });

console.log('Current belief states:');
viz.visualize(sim);

console.log('\nSimulation statistics:');
viz.printStats(sim);

// Compare belief changes
console.log('\n📌 PART 6: Belief Evolution\n');
viz.printComparison(sim, 0, 10);

// Track specific belief evolution
const evolution = sim.getHistory();
if (evolution.length > 0) {
  const allProps = new Set();
  Object.values(evolution[0].agents).forEach(agent => {
    agent.beliefs.forEach(b => allProps.add(b.proposition));
  });
  
  const firstProp = Array.from(allProps)[0];
  if (firstProp) {
    console.log(`\nEvolution of belief: "${firstProp}"`);
    viz.printEvolution(sim, firstProp);
  }
}

// ============================================================================
// PART 7: Message Passing
// ============================================================================
console.log('\n📌 PART 7: Agent Communication\n');

// Create message system with topology - register actual agents
const msgTopology = new TopologyManager();
msgTopology.registerAgent(alice);
msgTopology.registerAgent(bob);
msgTopology.registerAgent(charlie);
const messageSystem = new MessageSystem(msgTopology);

// Alice sends a belief update to Bob
console.log('Alice sending message to Bob about team project...');
const result = messageSystem.send(
  'alice-001', 
  'bob-001', 
  'belief_update', 
  {
    proposition: 'Team project on track',
    confidence: 0.95,
    justification: 'Completed milestone ahead of schedule'
  }
);
console.log(`Message sent: ${result ? 'Success' : 'Failed'}`);

// Bob queries Alice about a belief
console.log('\nBob querying Alice about meeting...');
const queryResult = messageSystem.send(
  'bob-001',
  'alice-001',
  'query',
  {
    queryType: 'get_belief',
    proposition: 'Meeting at 10am'
  }
);
console.log(`Query sent: ${queryResult ? 'Success' : 'Failed'}`);

// Get message history
const msgHistory = messageSystem.getMessageLog();
console.log(`\nMessage log entries: ${msgHistory.length}`);

// ============================================================================
// PART 8: Topology Manager
// ============================================================================
console.log('\n📌 PART 8: Dynamic Topology\n');

// Create fresh topology for demonstration - register actual agents
const demoTopology = new TopologyManager();
demoTopology.registerAgent(alice);
demoTopology.registerAgent(bob);
demoTopology.registerAgent(charlie);

// Rebuild topology to discover connections based on relevance
demoTopology.rebuildTopology();

console.log('Network topology (neighbors):');
console.log(`  Alice → ${demoTopology.getNeighbors('alice-001').join(', ') || '(none)'}`);
console.log(`  Bob → ${demoTopology.getNeighbors('bob-001').join(', ') || '(none)'}`);
console.log(`  Charlie → ${demoTopology.getNeighbors('charlie-001').join(', ') || '(none)'}`);

// Send a message through topology
console.log('\nSending message from Alice to Charlie...');
const topoResult = demoTopology.sendTo('alice-001', 'charlie-001', {
  type: 'test',
  content: 'Hello through topology!'
});
console.log(`Message delivered: ${topoResult ? 'Success' : 'Failed'}`);

// ============================================================================
// PART 9: Analysis
// ============================================================================
console.log('\n📌 PART 9: Analysis\n');

const history = sim.getHistory();

// Compare first and last states
if (history.length >= 2) {
  const firstState = history[0];
  const lastState = history[history.length - 1];
  
  // Get agents' belief networks and compare
  const aliceFirst = firstState.agents['alice-001'];
  const aliceLast = lastState.agents['alice-001'];
  
  if (aliceFirst && aliceLast) {
    console.log(`Alice's belief count: ${aliceFirst.beliefs.length} → ${aliceLast.beliefs.length}`);
  }
  
  // Compute basic statistics
  let totalConfidence = 0;
  let beliefCount = 0;
  
  for (const state of history) {
    for (const agentData of Object.values(state.agents)) {
      for (const belief of agentData.beliefs) {
        totalConfidence += belief.confidence;
        beliefCount++;
      }
    }
  }
  
  const avgConfidence = beliefCount > 0 ? totalConfidence / beliefCount : 0;
  console.log(`Average confidence across simulation: ${(avgConfidence * 100).toFixed(1)}%`);
  console.log(`Total belief observations: ${beliefCount}`);
}

// ============================================================================
// PART 10: Persistence
// ============================================================================
console.log('\n📌 PART 10: Persistence\n');

// Save agent network state using sync export
const saveFile = 'demo-agent-state.json';
const exportedJson = Persistence.exportNetwork(alice.beliefNetwork);
fs.writeFileSync(saveFile, exportedJson, 'utf-8');
console.log(`Saved Alice's network state to ${saveFile}`);

// Load network state using sync import
const loadedJson = fs.readFileSync(saveFile, 'utf-8');
const loadedNetwork = Persistence.importNetwork(loadedJson, 'loaded-agent');
console.log(`Loaded network with ${loadedNetwork.beliefs.size} beliefs`);

// Clean up
fs.unlinkSync(saveFile);
console.log('Cleaned up temporary file');

// ============================================================================
// SUMMARY
// ============================================================================
console.log('\n' + '='.repeat(70));
console.log('                    DEMO COMPLETE');
console.log('='.repeat(70));

console.log(`
This demo showcased:

✅ PART 1: Creating beliefs with propositions, confidence, and justification
✅ PART 2: Building belief networks with dependencies and reactive updates
✅ PART 3: Creating agents with their own belief networks
✅ PART 4: Running multi-agent simulations over time
✅ PART 5: Visualizing simulation results with color-coded output
✅ PART 6: Tracking belief evolution over time
✅ PART 7: Agent-to-agent message passing
✅ PART 8: Dynamic communication topology
✅ PART 9: Analyzing belief network evolution
✅ PART 10: Persisting and loading agent states

Next steps:
- Explore the API in src/index.js
- Run unit tests: npm test
- Check out the CLI: node src/cli.js --help
- Build your own multi-agent scenarios!
`);
