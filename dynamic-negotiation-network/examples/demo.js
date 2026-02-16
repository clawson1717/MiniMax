/**
 * Dynamic Negotiation Network - Comprehensive Demo
 * 
 * This demo showcases all key features of the Dynamic Negotiation Network:
 * - Creating agents with needs/offers
 * - Semantic matching between agents
 * - Dynamic topology adaptation
 * - Multi-round negotiations
 * - Different negotiation strategies
 * - REST API usage
 * - Evaluation and metrics
 */

const { 
  Agent, 
  SemanticMatcher, 
  Embedding,
  DynamicTopologyManager,
  AgentNetwork, 
  NegotiationRunner,
  Evaluator,
  StrategyFactory
} = require('../src');

// Helper function for formatted output
function printSection(title) {
  console.log('\n' + '='.repeat(70));
  console.log('  ' + title);
  console.log('='.repeat(70));
}

function printSubsection(title) {
  console.log('\n--- ' + title + ' ---');
}

// ============================================================
// Part 1: Agent Creation and Basic Operations
// ============================================================

function demoAgentCreation() {
  printSection('PART 1: AGENT CREATION AND BASIC OPERATIONS');
  
  // Create a buyer agent
  const buyer = new Agent('tech_buyer', {
    needs: ['laptop', 'fast delivery', 'warranty', 'tech support'],
    offers: ['budget: $1200', 'payment: immediate', 'positive review'],
    strategy: 'counter'
  });
  
  // Create a seller agent
  const seller = new Agent('tech_seller', {
    needs: ['profit margin', 'quick payment', 'good review'],
    offers: ['laptop: gaming model', 'delivery: 2 days', 'warranty: 2 years', 'tech support: 24/7'],
    strategy: 'accept'
  });
  
  // Create a competitor agent
  const competitor = new Agent('competitor', {
    needs: ['market share', 'customer acquisition'],
    offers: ['laptop: budget model', 'delivery: next day', 'price: $900'],
    strategy: 'aggressive'
  });
  
  // Display agent information
  printSubsection('Created Agents');
  
  [buyer, seller, competitor].forEach(agent => {
    console.log(`\nAgent: ${agent.id}`);
    console.log(`  Strategy: ${agent.getStrategy()}`);
    console.log(`  Needs:    ${agent.getNeeds().join(', ')}`);
    console.log(`  Offers:   ${agent.getOffers().join(', ')}`);
  });
  
  // Demonstrate dynamic modification
  printSubsection('Dynamic Agent Modification');
  
  console.log('\nBefore modification:');
  console.log(`  Buyer needs: ${buyer.getNeeds().join(', ')}`);
  
  buyer.addNeed('student discount');
  buyer.removeNeed('tech support');
  
  console.log('After adding "student discount" and removing "tech support":');
  console.log(`  Buyer needs: ${buyer.getNeeds().join(', ')}`);
  
  // Demonstrate beliefs
  printSubsection('Agent Beliefs (Reactive Knowledge)');
  
  buyer.updateBelief('tech_seller', {
    reputation: 'high',
    priceFlexibility: 'medium',
    lastContact: Date.now()
  });
  
  console.log(`\nBuyer's belief about seller:`);
  console.log(`  ${JSON.stringify(buyer.getBelief('tech_seller'), null, 2)}`);
  
  return { buyer, seller, competitor };
}

// ============================================================
// Part 2: Semantic Matching
// ============================================================

function demoSemanticMatching() {
  printSection('PART 2: SEMANTIC MATCHING');
  
  // Create matcher with TF-IDF (default)
  const matcher = new SemanticMatcher({
    method: 'tfidf',
    threshold: 0.2
  });
  
  printSubsection('Direct Similarity Computation');
  
  // Show similarity between needs and offers
  const examples = [
    ['fast delivery', 'delivery: 2 days'],
    ['discount', 'price: $1000'],
    ['warranty', 'warranty: 2 years'],
    ['laptop', 'gaming laptop'],
    ['tech support', 'customer service']
  ];
  
  examples.forEach(([need, offer]) => {
    const score = matcher.computeSimilarity(need, offer);
    console.log(`  "${need}" <-> "${offer}": ${score.toFixed(4)}`);
  });
  
  printSubsection('Finding Best Matches');
  
  const need = 'discount';
  const offers = [
    'price: $1200',
    'discount: 10% off',
    'free shipping',
    'price match guarantee'
  ];
  
  const bestMatch = matcher.findBestMatch(need, offers);
  console.log(`\nBest match for "${need}":`);
  console.log(`  Offer: ${bestMatch.offer}`);
  console.log(`  Score: ${bestMatch.score.toFixed(4)}`);
  console.log(`  Meets threshold: ${bestMatch.meetsThreshold}`);
  
  printSubsection('Agent Compatibility');
  
  // Create agents for compatibility testing
  const buyer = new Agent('buyer_match', {
    needs: ['laptop', 'fast delivery', 'warranty'],
    offers: ['budget: $1200', 'immediate payment']
  });
  
  const seller = new Agent('seller_match', {
    needs: ['profit', 'quick payment'],
    offers: ['laptop', 'delivery: 2 days', 'warranty: 1 year']
  });
  
  const serviceProvider = new Agent('service_match', {
    needs: ['referral'],
    offers: ['installation', 'training']
  });
  
  const candidates = [seller, serviceProvider];
  const compatible = matcher.findCompatibleAgents(buyer, candidates);
  
  console.log(`\nCompatible agents for buyer:`);
  compatible.forEach(match => {
    console.log(`\n  ${match.agent.id}:`);
    console.log(`    Combined Score: ${match.score.toFixed(4)}`);
    console.log(`    Need->Offer:    ${match.needOfferScore.toFixed(4)}`);
    console.log(`    Offer->Need:    ${match.offerNeedScore.toFixed(4)}`);
    console.log(`    Meets threshold: ${match.meetsThreshold}`);
  });
  
  printSubsection('Compatibility Matrix');
  
  const agents = [buyer, seller, serviceProvider];
  const matrix = matcher.buildCompatibilityMatrix(agents);
  
  console.log('\n         ' + agents.map(a => a.id.padEnd(15)).join(''));
  agents.forEach((agent, i) => {
    const row = matrix[i].map(v => v.toFixed(2).padStart(6)).join('  ');
    console.log(`${agent.id.padEnd(8)} ${row}`);
  });
  
  printSubsection('Embedding Methods Comparison');
  
  const texts = ['fast delivery', 'quick shipping', 'express delivery', 'slow boat'];
  
  console.log('\nTF-IDF Embeddings (first 5 dimensions):');
  const tfidfEmbeds = Embedding.tfidf(texts);
  texts.forEach((text, i) => {
    const dims = tfidfEmbeds[i].slice(0, 5).map(v => v.toFixed(3)).join(', ');
    console.log(`  "${text}": [${dims}...]`);
  });
  
  console.log('\nHash Embeddings (first 5 dimensions):');
  const hashEmbeds = Embedding.hash(texts, 32);
  texts.forEach((text, i) => {
    const dims = hashEmbeds[i].slice(0, 5).map(v => v.toFixed(3)).join(', ');
    console.log(`  "${text}": [${dims}...]`);
  });
  
  return { matcher };
}

// ============================================================
// Part 3: Dynamic Topology
// ============================================================

function demoDynamicTopology() {
  printSection('PART 3: DYNAMIC TOPOLOGY');
  
  // Create topology manager
  const topology = new DynamicTopologyManager({
    threshold: 0.3,
    maxNeighbors: 3,
    bidirectional: true
  });
  
  printSubsection('Creating Agent Network');
  
  // Create a diverse set of agents
  const agents = [
    new Agent('electronics_buyer', {
      needs: ['laptop', 'warranty', 'support'],
      offers: ['budget: $1000', 'cash payment']
    }),
    new Agent('electronics_seller', {
      needs: ['profit', 'quick sale'],
      offers: ['laptop', '2 year warranty', 'tech support']
    }),
    new Agent('furniture_buyer', {
      needs: ['sofa', 'delivery', 'assembly'],
      offers: ['budget: $800', 'cash']
    }),
    new Agent('furniture_seller', {
      needs: ['sale', 'referral'],
      offers: ['sofa', 'delivery', 'assembly service']
    }),
    new Agent('consultant', {
      needs: ['project', 'payment'],
      offers: ['expertise', 'recommendations']
    })
  ];
  
  // Register all agents
  agents.forEach(agent => {
    topology.registerAgent(agent);
    console.log(`  Registered: ${agent.id}`);
  });
  
  printSubsection('Building Initial Topology');
  
  // Build topology
  const stats = topology.rebuildTopology();
  console.log(`\nTopology Statistics:`);
  console.log(`  Agents: ${stats.agentCount}`);
  console.log(`  Edges: ${stats.edgeCount}`);
  console.log(`  Density: ${(stats.density * 100).toFixed(1)}%`);
  
  printSubsection('Network Connections');
  
  const graph = topology.getGraph();
  console.log('\nAdjacency List:');
  Object.entries(graph).forEach(([agentId, neighbors]) => {
    const neighborStr = neighbors.length > 0 ? neighbors.join(', ') : '(none)';
    console.log(`  ${agentId} -> [${neighborStr}]`);
  });
  
  printSubsection('Path Finding');
  
  // Find path between agents
  const path1 = topology.getPath('electronics_buyer', 'electronics_seller');
  console.log(`\nPath (buyer -> seller): ${path1 ? path1.join(' -> ') : 'No path found'}`);
  
  const path2 = topology.getPath('furniture_buyer', 'furniture_seller');
  console.log(`Path (furniture buyer -> seller): ${path2 ? path2.join(' -> ') : 'No path found'}`);
  
  const path3 = topology.getPath('electronics_buyer', 'furniture_buyer');
  console.log(`Path (electronics -> furniture): ${path3 ? path3.join(' -> ') : 'No path found'}`);
  
  // Weighted path
  const weightedPath = topology.getWeightedPath('electronics_buyer', 'electronics_seller');
  if (weightedPath) {
    console.log(`\nWeighted path: ${weightedPath.path.join(' -> ')}`);
    console.log(`Total weight: ${weightedPath.totalWeight.toFixed(4)}`);
  }
  
  printSubsection('Topology Adaptation');
  
  // Simulate agent needs changing
  console.log('\nSimulating changing needs...');
  
  const consultant = agents.find(a => a.id === 'consultant');
  console.log(`\nBefore: Consultant needs = [${consultant.getNeeds().join(', ')}]`);
  
  // Change consultant's needs to be more compatible with electronics
  consultant.setNeeds(['electronics project', 'consulting fee']);
  console.log(`After:  Consultant needs = [${consultant.getNeeds().join(', ')}]`);
  
  // Rebuild topology
  const newStats = topology.rebuildTopology();
  console.log(`\nNew Topology Statistics:`);
  console.log(`  Edges: ${newStats.edgeCount}`);
  console.log(`  Density: ${(newStats.density * 100).toFixed(1)}%`);
  
  const newGraph = topology.getGraph();
  console.log('\nNew Adjacency List:');
  Object.entries(newGraph).forEach(([agentId, neighbors]) => {
    const neighborStr = neighbors.length > 0 ? neighbors.join(', ') : '(none)';
    console.log(`  ${agentId} -> [${neighborStr}]`);
  });
  
  return { topology, agents };
}

// ============================================================
// Part 4: Multi-Round Negotiation
// ============================================================

function demoNegotiation() {
  printSection('PART 4: MULTI-ROUND NEGOTIATION');
  
  printSubsection('Setting up Negotiation');
  
  // Create buyer and seller with complementary needs/offers
  const buyer = new Agent('negotiation_buyer', {
    needs: ['laptop', 'fast delivery', 'warranty'],
    offers: ['budget: $800', 'immediate payment', 'positive review'],
    strategy: 'counter'
  });
  
  const seller = new Agent('negotiation_seller', {
    needs: ['sale', 'quick payment', 'good review'],
    offers: ['laptop', '2 day delivery', '1 year warranty', 'price: $750'],
    strategy: 'accept'
  });
  
  console.log('\nBuyer:');
  console.log(`  Needs:  ${buyer.getNeeds().join(', ')}`);
  console.log(`  Offers: ${buyer.getOffers().join(', ')}`);
  
  console.log('\nSeller:');
  console.log(`  Needs:  ${seller.getNeeds().join(', ')}`);
  console.log(`  Offers: ${seller.getOffers().join(', ')}`);
  
  // Create network
  const network = new AgentNetwork({
    topology: {
      threshold: 0.2,
      maxNeighbors: 5
    },
    maxRounds: 10
  });
  
  network.addAgent(buyer);
  network.addAgent(seller);
  network.activate();
  
  console.log('\nNetwork activated. Topology built.');
  
  printSubsection('Running Negotiation');
  
  // Create runner with dynamic topology enabled
  const runner = new NegotiationRunner(network, {
    maxRounds: 10,
    timeout: 30000,
    dynamicTopology: true,
    topologyRebuildInterval: 1,
    verbose: false  // We'll print our own output
  });
  
  console.log('\nRunning negotiation with dynamic topology...\n');
  
  // Run the negotiation
  const result = runner.run();
  
  printSubsection('Negotiation Results');
  
  console.log(`\nStatus: ${result.status}`);
  console.log(`Success: ${result.success}`);
  console.log(`Rounds: ${result.rounds}`);
  console.log(`Duration: ${result.duration}ms`);
  
  console.log('\nStatistics:');
  console.log(`  Total Offers: ${result.statistics.totalOffers}`);
  console.log(`  Acceptances: ${result.statistics.totalAcceptances}`);
  console.log(`  Rejections: ${result.statistics.totalRejections}`);
  console.log(`  Counter-offers: ${result.statistics.totalCounters}`);
  console.log(`  Agreement Rate: ${(result.statistics.agreementRate * 100).toFixed(1)}%`);
  
  console.log('\nAgent Results:');
  result.agentResults.forEach(agent => {
    console.log(`\n  ${agent.id}:`);
    console.log(`    Needs Satisfied: ${agent.needsSatisfied}`);
    console.log(`    Remaining Needs: [${agent.remainingNeeds.join(', ')}]`);
    console.log(`    History Entries: ${agent.historyLength}`);
  });
  
  printSubsection('Round-by-Round History');
  
  const history = runner.getHistory();
  history.forEach(round => {
    console.log(`\n  Round ${round.round}:`);
    console.log(`    Messages: ${round.statistics.messageCount}`);
    console.log(`    Offers: ${round.statistics.offersMade}`);
    console.log(`    Acceptances: ${round.statistics.acceptances}`);
    
    if (round.topologyChange) {
      console.log(`    Topology: ${round.topologyChange.statistics.totalEdges} edges`);
    }
  });
  
  return { result, runner, network };
}

// ============================================================
// Part 5: Strategies
// ============================================================

function demoStrategies() {
  printSection('PART 5: NEGOTIATION STRATEGIES');
  
  printSubsection('Available Strategies');
  
  const strategies = StrategyFactory.getAvailableStrategies();
  console.log(`\nAvailable strategies: ${strategies.join(', ')}`);
  
  strategies.forEach(name => {
    const strategy = StrategyFactory.create(name);
    console.log(`\n  ${name}:`);
    console.log(`    ${strategy.getDescription()}`);
  });
  
  printSubsection('Strategy Evaluation Demo');
  
  // Create test agent and context
  const testAgent = new Agent('test_agent', {
    needs: ['laptop', 'discount'],
    offers: ['$1000', 'quick payment']
  });
  
  const context = {
    agent: testAgent,
    offer: 'laptop with 10% discount',
    roundNumber: 1
  };
  
  console.log('\nEvaluating offer "laptop with 10% discount":\n');
  
  strategies.forEach(name => {
    const strategy = StrategyFactory.create(name);
    const decision = strategy.evaluate(context);
    console.log(`  ${name.padEnd(10)} -> ${decision.action.toUpperCase()} (${decision.reason})`);
  });
  
  printSubsection('Strategy Comparison in Negotiation');
  
  // Run negotiations with different strategy combinations
  const strategyPairs = [
    ['accept', 'accept'],
    ['counter', 'counter'],
    ['accept', 'reject'],
    ['counter', 'accept']
  ];
  
  console.log('\nRunning negotiations with different strategy pairs...\n');
  
  const results = [];
  
  strategyPairs.forEach(([buyerStrat, sellerStrat]) => {
    const buyer = new Agent(`buyer_${buyerStrat}`, {
      needs: ['product', 'discount'],
      offers: ['$100', 'cash'],
      strategy: buyerStrat
    });
    
    const seller = new Agent(`seller_${sellerStrat}`, {
      needs: ['sale', 'profit'],
      offers: ['product', 'discount: 10%'],
      strategy: sellerStrat
    });
    
    const network = new AgentNetwork({
      topology: { threshold: 0.2 },
      maxRounds: 5
    });
    
    network.addAgent(buyer);
    network.addAgent(seller);
    network.activate();
    
    const runner = new NegotiationRunner(network, {
      maxRounds: 5,
      dynamicTopology: false,
      verbose: false
    });
    
    const result = runner.run();
    results.push({ pair: `${buyerStrat} vs ${sellerStrat}`, result });
    
    console.log(`  ${buyerStrat.padEnd(8)} vs ${sellerStrat.padEnd(8)}: ` +
                `${result.success ? 'SUCCESS' : 'PARTIAL'} ` +
                `(${result.rounds} rounds, ${result.statistics.totalAcceptances} accepts)`);
  });
  
  return { results };
}

// ============================================================
// Part 6: Evaluation and Metrics
// ============================================================

function demoEvaluation(negotiationResult) {
  printSection('PART 6: EVALUATION AND METRICS');
  
  printSubsection('Comprehensive Evaluation');
  
  const evaluator = new Evaluator();
  const evaluation = evaluator.evaluate(negotiationResult);
  
  console.log(`\nAgreement Reached: ${evaluation.agreement ? 'Yes' : 'No'}`);
  console.log(`Overall Score: ${evaluation.overallScore.toFixed(2)}`);
  console.log(`Status: ${evaluation.status}`);
  console.log(`Rounds: ${evaluation.rounds}`);
  console.log(`Duration: ${evaluation.duration}ms`);
  
  printSubsection('Agent Scores');
  
  evaluation.agentScores.forEach(agent => {
    console.log(`\n  ${agent.id}:`);
    console.log(`    Needs Satisfied: ${agent.needsSatisfied}`);
    console.log(`    Payoff: ${agent.payoff.toFixed(2)}`);
    console.log(`    Satisfaction Rate: ${(agent.satisfactionRate * 100).toFixed(1)}%`);
  });
  
  printSubsection('Negotiation Metrics');
  
  const metrics = evaluation.metrics;
  console.log(`\n  Core Metrics:`);
  console.log(`    Precision: ${(metrics.precision * 100).toFixed(1)}%`);
  console.log(`    Recall: ${(metrics.recall * 100).toFixed(1)}%`);
  console.log(`    F1 Score: ${(metrics.f1 * 100).toFixed(1)}%`);
  
  console.log(`\n  Negotiation-Specific Metrics:`);
  console.log(`    Success Rate: ${(metrics.successRate * 100).toFixed(1)}%`);
  console.log(`    Efficiency: ${(metrics.efficiency * 100).toFixed(1)}%`);
  console.log(`    Conflict Rate: ${(metrics.conflictRate * 100).toFixed(1)}%`);
  console.log(`    Flexibility: ${(metrics.flexibility * 100).toFixed(1)}%`);
  
  console.log(`\n  Message Counts:`);
  console.log(`    Total Offers: ${metrics.counts.totalOffers}`);
  console.log(`    Acceptances: ${metrics.counts.totalAcceptances}`);
  console.log(`    Rejections: ${metrics.counts.totalRejections}`);
  console.log(`    Counter-offers: ${metrics.counts.totalCounters}`);
  
  printSubsection('Generated Report');
  
  const report = evaluator.generateReport(negotiationResult);
  console.log('\n' + report);
  
  return { evaluation };
}

// ============================================================
// Part 7: Complex Multi-Agent Scenario
// ============================================================

function demoComplexScenario() {
  printSection('PART 7: COMPLEX MULTI-AGENT SCENARIO');
  
  printSubsection('Scenario: Technology Marketplace');
  
  console.log('\nCreating a complex marketplace with multiple buyers and sellers...\n');
  
  // Create diverse agents
  const agents = [
    // Buyers
    new Agent('gamer_buyer', {
      needs: ['gaming laptop', 'high performance', 'warranty'],
      offers: ['budget: $2000', 'cash payment'],
      strategy: 'counter'
    }),
    new Agent('student_buyer', {
      needs: ['budget laptop', 'student discount', 'durability'],
      offers: ['budget: $600', 'student id'],
      strategy: 'accept'
    }),
    new Agent('business_buyer', {
      needs: ['business laptop', 'reliability', 'support contract'],
      offers: ['budget: $1500', 'corporate account'],
      strategy: 'counter'
    }),
    
    // Sellers
    new Agent('premium_seller', {
      needs: ['high profit', 'quick sale'],
      offers: ['gaming laptop', 'premium warranty', 'priority support'],
      strategy: 'accept'
    }),
    new Agent('budget_seller', {
      needs: ['volume sales', 'market share'],
      offers: ['budget laptop', 'student discount', 'basic warranty'],
      strategy: 'accept'
    }),
    new Agent('business_seller', {
      needs: ['corporate clients', 'long term contracts'],
      offers: ['business laptop', 'enterprise support', 'bulk discounts'],
      strategy: 'counter'
    }),
    
    // Service providers
    new Agent('warranty_provider', {
      needs: ['customers', 'revenue'],
      offers: ['extended warranty', 'accident protection', 'tech support'],
      strategy: 'accept'
    }),
    new Agent('delivery_service', {
      needs: ['delivery contracts', 'good reviews'],
      offers: ['same day delivery', 'express shipping', 'tracking'],
      strategy: 'accept'
    })
  ];
  
  console.log('Created agents:');
  agents.forEach(agent => {
    console.log(`  - ${agent.id} (${agent.getStrategy()} strategy)`);
  });
  
  // Create network
  const network = new AgentNetwork({
    topology: {
      threshold: 0.25,
      maxNeighbors: 4
    },
    maxRounds: 15
  });
  
  agents.forEach(agent => network.addAgent(agent));
  network.activate();
  
  const initialStats = network.getStatistics();
  console.log(`\nInitial Network:`);
  console.log(`  Agents: ${initialStats.agentCount}`);
  console.log(`  Total Connections: ${initialStats.totalEdges}`);
  console.log(`  Average Degree: ${initialStats.averageDegree.toFixed(2)}`);
  
  printSubsection('Running Complex Negotiation');
  
  const runner = new NegotiationRunner(network, {
    maxRounds: 15,
    timeout: 60000,
    dynamicTopology: true,
    topologyRebuildInterval: 2,  // Rebuild every 2 rounds
    verbose: false
  });
  
  console.log('\nRunning multi-agent negotiation...\n');
  
  const result = runner.run();
  
  console.log(`Negotiation Complete!`);
  console.log(`  Status: ${result.status}`);
  console.log(`  Rounds: ${result.rounds}`);
  console.log(`  Duration: ${result.duration}ms`);
  
  printSubsection('Final Results');
  
  console.log('\nAgent Outcomes:');
  result.agentResults.forEach(agent => {
    const status = agent.needsSatisfied ? '✓ SATISFIED' : '✗ UNSATISFIED';
    const remaining = agent.remainingNeeds.length > 0 
      ? ` (remaining: ${agent.remainingNeeds.join(', ')})` 
      : '';
    console.log(`  ${agent.id.padEnd(20)} ${status}${remaining}`);
  });
  
  console.log('\nStatistics:');
  console.log(`  Total Offers: ${result.statistics.totalOffers}`);
  console.log(`  Acceptances: ${result.statistics.totalAcceptances}`);
  console.log(`  Rejections: ${result.statistics.totalRejections}`);
  console.log(`  Counter-offers: ${result.statistics.totalCounters}`);
  console.log(`  Success Rate: ${(result.statistics.agreementRate * 100).toFixed(1)}%`);
  
  printSubsection('Topology Evolution');
  
  const topologyHistory = runner.getTopologyHistory();
  console.log(`\nTopology changed ${topologyHistory.length} times:`);
  
  topologyHistory.forEach((snapshot, i) => {
    console.log(`\n  Change ${i + 1} (Round ${snapshot.round}):`);
    console.log(`    Agents: ${snapshot.statistics.agentCount}`);
    console.log(`    Edges: ${snapshot.statistics.totalEdges}`);
    console.log(`    Density: ${(snapshot.statistics.totalEdges / 
      (snapshot.statistics.agentCount * (snapshot.statistics.agentCount - 1)) * 100).toFixed(1)}%`);
    
    // Show which agents are connected
    const connections = Object.entries(snapshot.graph)
      .filter(([_, neighbors]) => neighbors.length > 0)
      .map(([agent, neighbors]) => `${agent}→[${neighbors.join(',')}]`)
      .join(', ');
    console.log(`    Active: ${connections}`);
  });
  
  // Evaluate the complex scenario
  const evaluator = new Evaluator();
  const evaluation = evaluator.evaluate(result);
  
  printSubsection('Complex Scenario Metrics');
  
  console.log(`\n  F1 Score: ${(evaluation.metrics.f1 * 100).toFixed(1)}%`);
  console.log(`  Efficiency: ${(evaluation.metrics.efficiency * 100).toFixed(1)}%`);
  console.log(`  Overall Score: ${evaluation.overallScore.toFixed(2)}`);
  
  return { result, evaluation };
}

// ============================================================
// Main Demo Execution
// ============================================================

async function runDemo() {
  console.log('\n' + '█'.repeat(70));
  console.log('█' + ' '.repeat(68) + '█');
  console.log('█' + '    DYNAMIC NEGOTIATION NETWORK - COMPREHENSIVE DEMO'.padStart(58).padEnd(68) + '█');
  console.log('█' + ' '.repeat(68) + '█');
  console.log('█'.repeat(70));
  
  try {
    // Part 1: Agent Creation
    const { buyer, seller, competitor } = demoAgentCreation();
    
    // Part 2: Semantic Matching
    const { matcher } = demoSemanticMatching();
    
    // Part 3: Dynamic Topology
    const { topology, agents } = demoDynamicTopology();
    
    // Part 4: Multi-Round Negotiation
    const { result, runner, network } = demoNegotiation();
    
    // Part 5: Strategies
    const { results } = demoStrategies();
    
    // Part 6: Evaluation
    const { evaluation } = demoEvaluation(result);
    
    // Part 7: Complex Scenario
    const { result: complexResult, evaluation: complexEval } = demoComplexScenario();
    
    // Summary
    printSection('DEMO COMPLETE');
    
    console.log('\nThis demo showcased:');
    console.log('  ✓ Agent creation with needs, offers, and strategies');
    console.log('  ✓ Semantic matching using TF-IDF and hash embeddings');
    console.log('  ✓ Dynamic topology that adapts to agent compatibility');
    console.log('  ✓ Multi-round negotiations with message passing');
    console.log('  ✓ Different negotiation strategies (accept, reject, counter, random)');
    console.log('  ✓ Comprehensive evaluation with precision, recall, F1 metrics');
    console.log('  ✓ Complex multi-agent scenarios with topology evolution');
    
    console.log('\nNext steps:');
    console.log('  • Run the REST API: npm run server');
    console.log('  • Run tests: npm test');
    console.log('  • Read the full documentation in README.md');
    console.log('  • Experiment with your own agent configurations');
    
    console.log('\n' + '='.repeat(70));
    console.log('Thank you for exploring the Dynamic Negotiation Network!');
    console.log('='.repeat(70) + '\n');
    
  } catch (error) {
    console.error('\nError running demo:', error);
    process.exit(1);
  }
}

// Run the demo if this file is executed directly
if (require.main === module) {
  runDemo();
}

module.exports = { runDemo };
