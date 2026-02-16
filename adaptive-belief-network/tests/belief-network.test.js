/**
 * Basic test for Adaptive Belief Network
 */

const { Belief, BeliefNetwork } = require('../src/index');

console.log('=== Testing Belief Class ===\n');

// Test 1: Create a belief
const belief1 = new Belief('The sky is blue', 0.95, 'Direct observation');
console.log('Created belief:', belief1.toJSON());

// Test 2: Update a belief
belief1.update(0.80, 'Clouds appearing');
console.log('Updated belief:', belief1.toJSON());

// Test 3: Get history
console.log('History:', belief1.getHistory());

console.log('\n=== Testing BeliefNetwork Class ===\n');

// Test 4: Create a belief network
const network = new BeliefNetwork('agent-1');
console.log('Created network for agent:', network.agentId);

// Test 5: Add beliefs with dependencies
network.addBelief('The sky is blue', 0.95, 'Direct observation');
network.addBelief('Weather is clear', 0.90, 'No precipitation', ['The sky is blue']);
network.addBelief('Good hiking conditions', 0.85, 'Weather is clear', ['Weather is clear']);
console.log('Added 3 beliefs with dependencies');

// Test 6: Get a belief
const skyBelief = network.getBelief('The sky is blue');
console.log('Get belief "The sky is blue":', skyBelief.toJSON());

// Test 7: Get dependent beliefs
const dependents = network.getDependent('The sky is blue');
console.log('Beliefs dependent on "The sky is blue":', dependents.map(b => b.proposition));

// Test 8: Update belief with reactive propagation
console.log('\nBefore update:');
network.getAllBeliefs().forEach(b => console.log(`  ${b.proposition}: ${b.confidence}`));

const updated = network.updateBelief('The sky is blue', 0.50, 'Heavy clouds');

console.log('\nAfter updating "The sky is blue" to 0.50:');
network.getAllBeliefs().forEach(b => console.log(`  ${b.proposition}: ${b.confidence}`));

console.log('\nUpdated beliefs (including propagated):', updated.map(b => b.proposition));

// Test 9: Network stats
console.log('\nNetwork stats:', network.getStats());

// Test 10: Update log
console.log('\nUpdate log:');
network.getUpdateLog().forEach(log => console.log(`  ${log.type}: ${log.proposition}`));

console.log('\n=== All tests completed successfully! ===');
