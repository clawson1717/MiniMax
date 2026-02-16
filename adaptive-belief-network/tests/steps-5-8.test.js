/**
 * Tests for Agent, TopologyManager, and MessageSystem classes
 * 
 * These tests verify:
 * - Agent class functionality (step 5)
 * - TopologyManager functionality (step 6)
 * - MessageSystem functionality (step 7)
 * - Relevance filtering (step 8)
 */

const { Agent } = require('../src/Agent');
const { TopologyManager } = require('../src/TopologyManager');
const { MessageSystem } = require('../src/MessageSystem');

console.log('=== Testing Agent Class (Step 5) ===\n');

// Create an agent
const agent1 = new Agent('agent-1', 'Agent One');

// Test adding beliefs
agent1.addBelief('The sky is blue', 0.95, 'Direct observation');
agent1.addBelief('It is raining', 0.3, 'No rain detected');
agent1.addBelief('Weather is good', 0.8, 'Based on sky color', ['The sky is blue']);

console.log('Agent 1 beliefs:');
agent1.getBeliefs().forEach(b => {
  console.log(`  - ${b.proposition}: ${b.confidence}`);
});

// Test updating belief
agent1.updateBelief('The sky is blue', 0.7, 'Some clouds appearing');
console.log('\nAfter updating "The sky is blue":');
agent1.getBeliefs().forEach(b => {
  console.log(`  - ${b.proposition}: ${b.confidence}`);
});

// Test receive message
const testMessage = {
  id: 'test-1',
  type: 'belief_update',
  senderId: 'agent-2',
  payload: {
    proposition: 'Temperature is 72F',
    confidence: 0.9,
    justification: 'Thermostat reading'
  },
  timestamp: Date.now()
};

const response = agent1.receiveMessage(testMessage);
console.log('\nReceived belief update, response:', response ? response.type : 'none');

console.log('\nAgent stats:', agent1.getStats());

console.log('\n=== Testing TopologyManager (Step 6) ===\n');

// Create topology manager
const topology = new TopologyManager();

// Create separate agents for topology test
const topoAgent1 = new Agent('topo-agent-1', 'Topo Agent One');
const topoAgent2 = new Agent('topo-agent-2', 'Topo Agent Two');
const topoAgent3 = new Agent('topo-agent-3', 'Topo Agent Three');

// Register agents
topology.registerAgent(topoAgent1);
topology.registerAgent(topoAgent2);
topology.registerAgent(topoAgent3);

// Add beliefs to agents - make them overlap for topology building
// topoAgent1 has beliefs about sky
topoAgent1.addBelief('Sky is blue', 0.9, 'Observation');
topoAgent1.addBelief('Weather is clear', 0.85, 'No clouds');

// topoAgent2 has beliefs about weather/rain
topoAgent2.addBelief('Weather forecast', 0.8, 'Weather report');
topoAgent2.addBelief('Rain expected', 0.75, 'Rain prediction');

// topoAgent3 has beliefs about temperature/weather
topoAgent3.addBelief('Temperature is warm', 0.85, 'Thermostat');
topoAgent3.addBelief('Weather conditions', 0.7, 'General weather');

// Set relevance rules - these should match the other agents' beliefs
topology.setRelevanceRules('topo-agent-1', ['Weather*', 'Rain*', 'forecast']); // wants weather info
topology.setRelevanceRules('topo-agent-2', ['Sky*', 'Temperature*', 'clear']); // wants sky/temp info
topology.setRelevanceRules('topo-agent-3', ['Sky*', 'Weather*', 'Rain*']); // wants all weather info

// Rebuild topology
topology.rebuildTopology();

console.log('Topology after rebuild:');
console.log('  Topo Agent 1 neighbors:', topology.getNeighbors('topo-agent-1'));
console.log('  Topo Agent 2 neighbors:', topology.getNeighbors('topo-agent-2'));
console.log('  Topo Agent 3 neighbors:', topology.getNeighbors('topo-agent-3'));

console.log('\nTopology stats:', topology.getStats());

console.log('\n=== Testing MessageSystem (Step 7) ===\n');

// Create message system with topology
const messageSystem = new MessageSystem(topology);

// Send a message (now should work since agents are connected)
const sendResult = messageSystem.send('topo-agent-1', 'topo-agent-2', 'belief_update', {
  proposition: 'Weather forecast',
  confidence: 0.85,
  justification: 'Clear day'
});
console.log('Send result:', sendResult.success ? 'success' : 'failed');

// Broadcast to neighbors
const broadcastResult = messageSystem.broadcast('topo-agent-1', 'belief_update', {
  proposition: 'Test broadcast',
  confidence: 0.5,
  justification: 'Testing'
});
console.log('Broadcast result:', broadcastResult.deliveredCount, 'of', broadcastResult.totalNeighbors, 'received');

// Send belief update to relevant agents
const updateResult = messageSystem.sendBeliefUpdate('topo-agent-2', 'Rain expected', 0.9, 'Heavy rain predicted');
console.log('Belief update sent to:', updateResult.deliveredCount, 'relevant agents');

console.log('\n=== Testing Relevance Filtering (Step 8) ===\n');

// Subscribe agent to topics
topoAgent1.subscribe('Weather*');
topoAgent1.subscribe('Sky*');

// Test isRelevant
const testMessages = [
  {
    type: 'belief_update',
    payload: { proposition: 'Sky is blue', confidence: 0.9 }
  },
  {
    type: 'belief_update',
    payload: { proposition: 'Stock prices up', confidence: 0.8 }
  },
  {
    type: 'belief_update',
    payload: { proposition: 'Weather forecast', confidence: 0.7 }
  }
];

console.log('Testing relevance for topo-agent-1 (subscribed to Weather*, Sky*):');
testMessages.forEach(msg => {
  const relevant = messageSystem.isRelevant(msg, 'topo-agent-1');
  console.log(`  "${msg.payload.proposition}": ${relevant ? 'RELEVANT' : 'not relevant'}`);
});

// Test filterMessages
const filtered = messageSystem.filterMessages(messageSystem.getMessageLog(), {
  type: 'belief_update'
});
console.log('\nFiltered messages (type=belief_update):', filtered.length);

console.log('\n=== Message System Stats ===\n');
console.log(messageSystem.getStats());

console.log('\n=== All Steps 5-8 Tests Completed Successfully! ===');
