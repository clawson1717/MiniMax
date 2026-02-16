#!/usr/bin/env node
/**
 * CLI Interface for Dynamic Negotiation Network
 * 
 * Demonstrates:
 * - Creating 2-3 agents with different needs/offers
 * - Running negotiations
 * - Displaying results
 */

const { 
  Agent, 
  SemanticMatcher, 
  AgentNetwork, 
  DynamicTopologyManager 
} = require('./index');
const NegotiationRunner = require('./NegotiationRunner');
const Evaluator = require('./Evaluator');
const { StrategyFactory } = require('./strategies');

// Color codes for terminal output
const colors = {
  reset: '\x1b[0m',
  bright: '\x1b[1m',
  dim: '\x1b[2m',
  red: '\x1b[31m',
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m',
  magenta: '\x1b[35m',
  cyan: '\x1b[36m'
};

function log(msg, color = 'reset') {
  console.log(`${colors[color]}${msg}${colors.reset}`);
}

function logHeader(msg) {
  console.log('\n' + colors.bright + colors.cyan + '='.repeat(60) + colors.reset);
  console.log(colors.bright + colors.cyan + msg + colors.reset);
  console.log(colors.bright + colors.cyan + '='.repeat(60) + colors.reset);
}

function logSubHeader(msg) {
  console.log(colors.bright + '\n--- ' + msg + ' ---' + colors.reset);
}

/**
 * Create demo agents with different needs and offers
 */
function createDemoAgents() {
  logSubHeader('Creating Agents');
  
  // Agent 1: Buyer
  const buyer = new Agent('buyer_1', {
    needs: ['discount', 'fast delivery', 'quality product', 'warranty'],
    offers: ['budget: $500', 'payment: immediate', 'quantity: 10 units'],
    strategy: 'counter'
  });
  log(`Created: ${buyer.id}`, 'green');
  log(`  Needs: ${buyer.getNeeds().join(', ')}`, 'dim');
  log(`  Offers: ${buyer.getOffers().join(', ')}`, 'dim');
  
  // Agent 2: Seller
  const seller = new Agent('seller_1', {
    needs: ['profit margin', 'quick payment', 'bulk order'],
    offers: ['price: $600', 'product: premium quality', 'delivery: 1 week'],
    strategy: 'accept'
  });
  log(`Created: ${seller.id}`, 'green');
  log(`  Needs: ${seller.getNeeds().join(', ')}`, 'dim');
  log(`  Offers: ${seller.getOffers().join(', ')}`, 'dim');
  
  // Agent 3: Competitor/Mediator
  const competitor = new Agent('competitor_1', {
    needs: ['market share', 'customer satisfaction', 'partnership'],
    offers: ['price: $550', 'product: good quality', 'delivery: 3 days', 'discount: 5%'],
    strategy: 'random'
  });
  log(`Created: ${competitor.id}`, 'green');
  log(`  Needs: ${competitor.getNeeds().join(', ')}`, 'dim');
  log(`  Offers: ${competitor.getOffers().join(', ')}`, 'dim');
  
  return [buyer, seller, competitor];
}

/**
 * Create and configure the agent network
 */
function createNetwork(agents) {
  logSubHeader('Creating Agent Network');
  
  // Use lower threshold for demo to ensure connections
  const network = new AgentNetwork({
    topology: {
      threshold: 0.01,
      maxNeighbors: 10,
      bidirectional: false  // Allow one-way connections for demo
    }
  });
  
  // Add agents to network
  for (const agent of agents) {
    network.addAgent(agent);
    log(`Added agent: ${agent.id}`, 'green');
  }
  
  // Rebuild topology to establish connections
  network.topology.rebuildTopology();
  
  // Check if we have connections, if not make fully connected for demo
  let hasConnections = false;
  for (const agent of agents) {
    const neighbors = network.getNeighbors(agent.id);
    if (neighbors.length > 0) {
      hasConnections = true;
      break;
    }
  }
  
  // Force fully connected for demo purposes if no connections
  if (!hasConnections) {
    log('Forcing fully connected topology for demo...', 'yellow');
    // Directly manipulate the graph for demo
    for (const agent of agents) {
      const otherAgents = agents.filter(a => a.id !== agent.id);
      for (const other of otherAgents) {
        network.topology.graph.get(agent.id).add(other.id);
      }
    }
  }
  
  log(`Network created with ${agents.length} agents`, 'green');
  
  // Show connections (getNeighbors returns Agent objects)
  for (const agent of agents) {
    const neighbors = network.getNeighbors(agent.id);
    const neighborIds = neighbors.map(a => a.id).join(', ');
    log(`  ${agent.id} connected to: ${neighborIds || 'none'}`, 'dim');
  }
  
  return network;
}

/**
 * Demonstrate different strategies
 */
function demonstrateStrategies() {
  logSubHeader('Demonstrating Strategies');
  
  const agent = new Agent('demo_agent', {
    needs: ['discount', 'warranty'],
    offers: ['price: $100', 'warranty: 1 year']
  });
  
  const strategies = [
    { name: 'accept', options: { threshold: 0.3 } },
    { name: 'reject', options: { probability: 0.9 } },
    { name: 'counter', options: { concessionRate: 0.2 } },
    { name: 'random', options: { acceptProbability: 0.4, counterProbability: 0.3 } }
  ];
  
  for (const { name, options } of strategies) {
    const strategy = StrategyFactory.create(name, options);
    const context = {
      agent,
      offer: 'discount: 20%',
      senderOffers: ['price: $90']
    };
    const decision = strategy.evaluate(context);
    log(`\n${name.toUpperCase()} Strategy:`, 'yellow');
    log(`  Decision: ${decision.action}`, 'dim');
    log(`  Reason: ${decision.reason}`, 'dim');
  }
}

/**
 * Run negotiation demo
 */
function runNegotiationDemo(network, agents) {
  logSubHeader('Running Negotiation');
  
  // Create runner with options
  const runner = new NegotiationRunner(network, {
    maxRounds: 10,
    timeout: 30000,
    verbose: false
  });
  
  log(`Starting negotiation (max ${runner.maxRounds} rounds, ${runner.timeout/1000}s timeout)...`, 'cyan');
  
  // Run negotiation
  const result = runner.run();
  
  return result;
}

/**
 * Display negotiation results
 */
function displayResults(result, evaluator) {
  logSubHeader('Negotiation Results');
  
  // Handle error status
  if (result.status === 'error') {
    log(`\nStatus: ${colors.red}Error${colors.reset}`);
    log(`Message: ${result.message || result.error || 'Unknown error'}`);
    log(`\nNote: The negotiation encountered an issue. Check the error details above.`);
    return;
  }
  
  // Get detailed evaluation
  const evaluation = evaluator.evaluate(result);
  
  log(`\nStatus: ${colors.yellow}${result.status}${colors.reset}`);
  log(`Agreement Reached: ${evaluation.agreement ? colors.green + 'Yes' : colors.red + 'No'}${colors.reset}`);
  log(`Total Rounds: ${result.rounds || 0}`);
  log(`Duration: ${result.duration || 0}ms`);
  
  console.log('\n' + colors.bright + 'Statistics:' + colors.reset);
  const stats = result.statistics || { totalOffers: 0, totalAcceptances: 0, totalRejections: 0, totalCounters: 0, agreementRate: 0 };
  log(`  Total Offers: ${stats.totalOffers}`);
  log(`  Acceptances: ${colors.green}${stats.totalAcceptances}${colors.reset}`);
  log(`  Rejections: ${colors.red}${stats.totalRejections}${colors.reset}`);
  log(`  Counter-offers: ${colors.yellow}${stats.totalCounters}${colors.reset}`);
  log(`  Agreement Rate: ${(stats.agreementRate * 100).toFixed(1)}%`);
  
  console.log('\n' + colors.bright + 'Agent Outcomes:' + colors.reset);
  const agentResults = result.agentResults || [];
  for (const agentResult of agentResults) {
    const statusColor = agentResult.needsSatisfied ? 'green' : 'red';
    log(`\n  ${agentResult.id}:`, 'bright');
    log(`    Needs Satisfied: ${colors[statusColor]}${agentResult.needsSatisfied ? 'Yes' : 'No'}${colors.reset}`);
    log(`    Remaining Needs: ${(agentResult.remainingNeeds?.length || 0) > 0 ? agentResult.remainingNeeds.join(', ') : 'None'}`);
  }
  
  console.log('\n' + colors.bright + 'Performance Metrics:' + colors.reset);
  log(`  Precision: ${(evaluation.metrics.precision * 100).toFixed(1)}%`);
  log(`  Recall: ${(evaluation.metrics.recall * 100).toFixed(1)}%`);
  log(`  F1 Score: ${(evaluation.metrics.f1 * 100).toFixed(1)}%`);
  log(`  Success Rate: ${(evaluation.metrics.successRate * 100).toFixed(1)}%`);
  log(`  Efficiency: ${(evaluation.metrics.efficiency * 100).toFixed(1)}%`);
  
  console.log('\n' + colors.bright + 'Overall Score: ' + colors.cyan + evaluation.overallScore.toFixed(2) + colors.reset);
}

/**
 * Run a second demo with different configuration
 */
function runAlternativeDemo() {
  logHeader('ALTERNATIVE DEMO: Two-Agent Negotiation');
  
  // Create simpler setup with 2 agents
  const buyer2 = new Agent('buyer_2', {
    needs: ['low price', 'quick delivery'],
    offers: ['budget: $200', 'payment: cash'],
    strategy: 'accept'
  });
  
  const seller2 = new Agent('seller_2', {
    needs: ['profit'],
    offers: ['price: $250', 'delivery: 5 days'],
    strategy: 'counter'
  });
  
  log(`Created ${buyer2.id} and ${seller2.id}`, 'green');
  
  // Create network
  const network2 = new AgentNetwork();
  network2.addAgent(buyer2);
  network2.addAgent(seller2);
  network2.topology.rebuildTopology();
  
  // Run negotiation
  const runner2 = new NegotiationRunner(network2, {
    maxRounds: 5,
    timeout: 10000,
    verbose: false
  });
  
  const result2 = runner2.run();
  
  // Evaluate
  const evaluator = new Evaluator();
  const eval2 = evaluator.evaluate(result2);
  
  log(`\nStatus: ${result2.status}`);
  log(`Agreement: ${eval2.agreement ? 'Yes' : 'No'}`);
  log(`Rounds: ${result2.rounds}`);
  log(`Overall Score: ${eval2.overallScore.toFixed(2)}`);
  
  return result2;
}

/**
 * Main function
 */
function main() {
  console.clear();
  logHeader('Dynamic Negotiation Network - CLI Demo');
  
  try {
    // Demo 1: Demonstrate strategies
    demonstrateStrategies();
    
    // Demo 2: Full negotiation with 3 agents
    logHeader('MAIN DEMO: Three-Agent Negotiation');
    
    // Create agents
    const agents = createDemoAgents();
    
    // Create network
    const network = createNetwork(agents);
    
    // Run negotiation
    const result = runNegotiationDemo(network, agents);
    
    // Evaluate and display results
    const evaluator = new Evaluator();
    displayResults(result, evaluator);
    
    // Demo 3: Alternative configuration
    runAlternativeDemo();
    
    // Final summary
    logHeader('Demo Complete');
    log('All demonstrations completed successfully!', 'green');
    log('\nTo use in your own code:', 'bright');
    log(`
  const { Agent, AgentNetwork } = require('./src/index');
  const NegotiationRunner = require('./src/NegotiationRunner');
  const Evaluator = require('./src/Evaluator');
  
  // Create agents and network
  const agent1 = new Agent('agent1', { needs: [...], offers: [...] });
  const network = new AgentNetwork();
  network.addAgent(agent1);
  // ... add more agents
  
  // Run negotiation
  const runner = new NegotiationRunner(network, { maxRounds: 10 });
  const result = runner.run();
  
  // Evaluate
  const evaluator = new Evaluator();
  const evaluation = evaluator.evaluate(result);
  console.log(evaluation);
`);
    
  } catch (error) {
    log(`\nError: ${error.message}`, 'red');
    console.error(error.stack);
    process.exit(1);
  }
}

// Run if executed directly
if (require.main === module) {
  main();
}

module.exports = { main, createDemoAgents, createNetwork, runNegotiationDemo };
