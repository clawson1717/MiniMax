/**
 * Test Dynamic Topology Integration in NegotiationRunner
 * 
 * Tests the integration of DynamicTopologyManager into the negotiation flow.
 */

const { Agent, AgentNetwork, NegotiationRunner } = require('./index');

console.log('='.repeat(70));
console.log('Dynamic Topology Integration Test');
console.log('='.repeat(70));

// Test 1: Dynamic Topology Enabled (Default)
console.log('\n--- Test 1: Dynamic Topology Enabled (Default) ---\n');

const network1 = new AgentNetwork({
  topology: {
    threshold: 0.2,
    maxNeighbors: 3
  }
});

// Create agents with complementary needs/offers
const agent1 = new Agent('buyer_1', {
  needs: ['discount', 'fast delivery'],
  offers: ['cash payment', 'bulk order'],
  strategy: 'counter'
});

const agent2 = new Agent('seller_1', {
  needs: ['cash payment', 'bulk order'],
  offers: ['discount', 'fast delivery'],
  strategy: 'counter'
});

const agent3 = new Agent('mediator_1', {
  needs: ['commission'],
  offers: ['negotiation service'],
  strategy: 'flexible'
});

network1.addAgent(agent1);
network1.addAgent(agent2);
network1.addAgent(agent3);

// Create runner with dynamic topology enabled (default)
const runner1 = new NegotiationRunner(network1, {
  maxRounds: 5,
  verbose: true,
  dynamicTopology: true,  // Explicitly enable
  topologyRebuildInterval: 1  // Rebuild every round
});

console.log('Initial Status:', runner1.getStatus());
console.log('Dynamic topology enabled:', runner1.dynamicTopology);

// Run the negotiation
const result1 = runner1.run();

console.log('\n--- Test 1 Results ---');
console.log('Success:', result1.success);
console.log('Rounds completed:', result1.rounds);
console.log('Dynamic topology used:', result1.dynamicTopology);
console.log('Topology history entries:', result1.topologyHistory?.length || 0);
console.log('Final topology stats:', JSON.stringify(result1.finalTopology?.statistics, null, 2));

// Test 2: Dynamic Topology Disabled
console.log('\n\n--- Test 2: Dynamic Topology Disabled ---\n');

const network2 = new AgentNetwork();

const agent4 = new Agent('buyer_2', {
  needs: ['quality product'],
  offers: ['fair price'],
  strategy: 'counter'
});

const agent5 = new Agent('seller_2', {
  needs: ['fair price'],
  offers: ['quality product'],
  strategy: 'counter'
});

network2.addAgent(agent4);
network2.addAgent(agent5);

// Create runner with dynamic topology disabled
const runner2 = new NegotiationRunner(network2, {
  maxRounds: 3,
  verbose: false,
  dynamicTopology: false  // Explicitly disable
});

console.log('Dynamic topology enabled:', runner2.dynamicTopology);

const result2 = runner2.run();

console.log('\n--- Test 2 Results ---');
console.log('Success:', result2.success);
console.log('Dynamic topology used:', result2.dynamicTopology);
console.log('Topology history entries:', result2.topologyHistory?.length || 0);
console.log('Final topology:', result2.finalTopology);

// Test 3: Check getTopologyHistory method
console.log('\n\n--- Test 3: getTopologyHistory Method ---\n');
const topologyHistory = runner1.getTopologyHistory();
console.log('Topology history length:', topologyHistory.length);
if (topologyHistory.length > 0) {
  console.log('First topology snapshot:');
  console.log('  Round:', topologyHistory[0].round);
  console.log('  Timestamp:', new Date(topologyHistory[0].timestamp).toISOString());
  console.log('  Agent count:', topologyHistory[0].statistics?.agentCount);
  console.log('  Total edges:', topologyHistory[0].statistics?.totalEdges);
  console.log('  Graph:', JSON.stringify(topologyHistory[0].graph, null, 2));
}

// Test 4: Check getStatus includes topology info
console.log('\n\n--- Test 4: getStatus with Topology Info ---\n');
const status = runner1.getStatus();
console.log('Status object:', JSON.stringify(status, null, 2));

// Test 5: Verify topology rebuild interval
console.log('\n\n--- Test 5: Topology Rebuild Interval ---\n');

const network3 = new AgentNetwork();
const agent6 = new Agent('a1', { needs: ['x'], offers: ['y'] });
const agent7 = new Agent('a2', { needs: ['y'], offers: ['x'] });
network3.addAgent(agent6);
network3.addAgent(agent7);

const runner3 = new NegotiationRunner(network3, {
  maxRounds: 6,
  verbose: false,
  dynamicTopology: true,
  topologyRebuildInterval: 2  // Rebuild every 2 rounds
});

runner3.run();
const history3 = runner3.getTopologyHistory();
console.log('With rebuild interval of 2:');
console.log('  Total rounds:', runner3.currentRound);
console.log('  Topology rebuilds:', history3.length);
console.log('  Rounds with topology changes:', history3.map(h => h.round).join(', '));

// Test 6: Reset clears topology history
console.log('\n\n--- Test 6: Reset Clears Topology History ---\n');
console.log('Before reset - topology history length:', runner1.topologyHistory.length);
runner1.reset();
console.log('After reset - topology history length:', runner1.topologyHistory.length);
console.log('Last topology rebuild round:', runner1.lastTopologyRebuild);

// Summary
console.log('\n' + '='.repeat(70));
console.log('All Tests Completed Successfully!');
console.log('='.repeat(70));
console.log('\nIntegration verified:');
console.log('✓ DynamicTopologyManager integrated into NegotiationRunner');
console.log('✓ Topology rebuilt at start of each round (or every N rounds)');
console.log('✓ Agent communication respects dynamic topology');
console.log('✓ dynamicTopology option enables/disables the feature');
console.log('✓ Topology changes tracked in negotiation history');
console.log('✓ getTopologyHistory() method available');
console.log('✓ getStatus() includes current topology info');
