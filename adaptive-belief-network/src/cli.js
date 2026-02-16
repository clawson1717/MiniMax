#!/usr/bin/env node

/**
 * CLI Interface for Adaptive Belief Network
 * 
 * Commands:
 * - init: Initialize a new simulation
 * - run: Run a simulation
 * - visualize: Visualize simulation results
 * - demo: Run a demo showing agent belief evolution
 */

const { Simulator } = require('./Simulator');
const { Agent } = require('./Agent');
const { Visualizer } = require('./Visualizer');
const { UpdateTriggerManager } = require('./UpdateTrigger');
const fs = require('fs');
const path = require('path');

// Parse command line arguments
const args = process.argv.slice(2);
const command = args[0];

// CLI Help text
const HELP = `
Adaptive Belief Network CLI

Usage: node cli.js <command> [options]

Commands:
  init [config]     Initialize a new simulation
  run [config]      Run a simulation
  visualize [file]  Visualize simulation from file
  demo              Run demo showing belief evolution

Options:
  --agents, -a      Number of agents (default: 3)
  --steps, -s       Number of time steps (default: 20)
  --output, -o      Output file path
  --no-colors       Disable colored output
  --help, -h        Show this help message

Examples:
  node cli.js init
  node cli.js run --agents 5 --steps 50
  node cli.js visualize results.json
  node cli.js demo
`;

// Parse options
function parseOptions(args) {
  const options = {
    agents: 3,
    steps: 20,
    output: null,
    colors: true,
    config: null
  };
  
  for (let i = 1; i < args.length; i++) {
    const arg = args[i];
    
    if (arg === '--agents' || arg === '-a') {
      options.agents = parseInt(args[++i], 10);
    } else if (arg === '--steps' || arg === '-s') {
      options.steps = parseInt(args[++i], 10);
    } else if (arg === '--output' || arg === '-o') {
      options.output = args[++i];
    } else if (arg === '--no-colors') {
      options.colors = false;
    } else if (arg === '--help' || arg === '-h') {
      console.log(HELP);
      process.exit(0);
    } else if (!arg.startsWith('-')) {
      options.config = arg;
    }
  }
  
  return options;
}

// Command: init - Initialize a new simulation
async function initCommand(options) {
  console.log('Initializing Adaptive Belief Network Simulation...\n');
  
  const simulator = new Simulator({
    maxTimeSteps: options.steps,
    tickDelay: 500
  });
  
  const visualizer = new Visualizer({ useColors: options.colors });
  
  // Create agents
  const agentNames = ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'];
  
  for (let i = 0; i < options.agents; i++) {
    const name = agentNames[i] || `Agent${i + 1}`;
    const agent = simulator.addAgent(`agent_${i + 1}`, name);
    
    // Add initial beliefs
    agent.addBelief(
      'The sky is blue',
      0.9,
      'Direct observation of clear sky'
    );
    
    agent.addBelief(
      'Gravity exists',
      0.95,
      'Personal experience with falling objects'
    );
    
    agent.addBelief(
      'Climate change is real',
      0.6 + (Math.random() * 0.3),
      'Media reports and scientific consensus'
    );
    
    agent.addBelief(
      'Artificial intelligence will benefit humanity',
      0.5 + (Math.random() * 0.3),
      'Technological progress trends'
    );
    
    console.log(`Added agent: ${name}`);
  }
  
  console.log(`\nSimulation initialized with ${options.agents} agents`);
  console.log(`Time steps: ${options.steps}`);
  
  // Save config if output specified
  if (options.output) {
    const config = {
      agents: options.agents,
      steps: options.steps,
      initialBeliefs: simulator.getBeliefStates()
    };
    fs.writeFileSync(options.output, JSON.stringify(config, null, 2));
    console.log(`Configuration saved to: ${options.output}`);
  }
  
  return simulator;
}

// Command: run - Run a simulation
async function runCommand(options) {
  console.log('Running Adaptive Belief Network Simulation...\n');
  
  let simulator;
  
  // Load config if specified
  if (options.config) {
    try {
      const config = JSON.parse(fs.readFileSync(options.config, 'utf8'));
      options.agents = config.agents || options.agents;
      options.steps = config.steps || options.steps;
    } catch (e) {
      console.error(`Failed to load config: ${e.message}`);
    }
  }
  
  // Initialize simulator
  simulator = await initCommand({ ...options, output: null });
  
  const visualizer = new Visualizer({ useColors: options.colors });
  
  console.log('\n');
  visualizer.printSeparator('═');
  console.log('Starting Simulation...');
  visualizer.printSeparator('═');
  console.log('');
  
  // Add some triggers to simulate different events
  const triggerManager = simulator.triggerManager;
  
  // Simulate evidence at step 5
  setTimeout(() => {
    console.log('\n[EVENT] New evidence discovered...\n');
    
    const agent = simulator.getAgent('agent_1');
    if (agent) {
      triggerManager.trigger('evidence', 'Climate change is real', {
        agentId: agent.id,
        confidence: 0.95,
        justification: 'New scientific study confirms climate data',
        source: 'scientific_journal'
      });
    }
  }, 100);
  
  // Simulate persuasion at step 10
  setTimeout(() => {
    console.log('\n[EVENT] Persuasion attempt...\n');
    
    const agent1 = simulator.getAgent('agent_2');
    const agent2 = simulator.getAgent('agent_3');
    
    if (agent1 && agent2) {
      triggerManager.trigger('persuasion', 'AI will benefit humanity', {
        agentId: agent1.id,
        confidence: 0.85,
        justification: 'Convincing argument from Bob',
        source: agent2.id
      });
    }
  }, 200);
  
  // Run simulation synchronously
  simulator.run({ async: false });
  
  console.log('\n');
  visualizer.printSeparator('═');
  console.log('Simulation Complete!');
  visualizer.printSeparator('═');
  console.log('');
  
  // Print final state
  visualizer.visualize(simulator);
  
  // Print statistics
  console.log('');
  visualizer.printStats(simulator);
  
  // Save results if output specified
  if (options.output) {
    const results = {
      config: {
        agents: options.agents,
        steps: options.steps
      },
      history: simulator.getHistory(),
      finalState: simulator.getBeliefStates(),
      stats: simulator.getStats()
    };
    fs.writeFileSync(options.output, JSON.stringify(results, null, 2));
    console.log(`\nResults saved to: ${options.output}`);
  }
  
  return simulator;
}

// Command: visualize - Visualize saved simulation
async function visualizeCommand(options) {
  if (!options.config) {
    console.error('Error: Please specify a file to visualize');
    console.log('Usage: node cli.js visualize <file>');
    process.exit(1);
  }
  
  const filePath = path.resolve(options.config);
  
  if (!fs.existsSync(filePath)) {
    console.error(`Error: File not found: ${filePath}`);
    process.exit(1);
  }
  
  const data = JSON.parse(fs.readFileSync(filePath, 'utf8'));
  
  // Recreate simulator from data
  const simulator = new Simulator({
    maxTimeSteps: data.config?.steps || 20,
    tickDelay: 500
  });
  
  // Reconstruct agents and beliefs
  if (data.finalState) {
    for (const [agentId, agentData] of Object.entries(data.finalState)) {
      const agent = simulator.addAgent(agentId, agentData.name);
      for (const belief of agentData.beliefs) {
        agent.addBelief(belief.proposition, belief.confidence, belief.justification);
      }
    }
  }
  
  // Restore history
  if (data.history) {
    simulator.history = data.history;
    simulator.currentTimeStep = data.history.length;
  }
  
  const visualizer = new Visualizer({ useColors: options.colors });
  
  console.log(`Visualizing simulation from: ${filePath}\n`);
  
  // Show initial state
  if (data.history && data.history.length > 0) {
    const firstState = data.history[0];
    visualizer.printState(firstState.agents, { title: 'Initial State' });
  }
  
  // Show final state
  console.log('');
  visualizer.visualize(simulator, { timeStep: simulator.currentTimeStep - 1 });
  
  // Show comparison if we have multiple time steps
  if (data.history && data.history.length > 1) {
    console.log('');
    visualizer.printComparison(simulator, 0, data.history.length - 1);
  }
  
  // Show stats
  console.log('');
  visualizer.printStats(simulator);
}

// Command: demo - Run a demo showing agent belief evolution
async function demoCommand(options) {
  console.log('╔════════════════════════════════════════════════════════════╗');
  console.log('║     ADAPTIVE BELIEF NETWORK - DEMONSTRATION               ║');
  console.log('╚════════════════════════════════════════════════════════════╝');
  console.log('');
  
  const visualizer = new Visualizer({ useColors: true });
  
  // Create simulator
  const simulator = new Simulator({
    maxTimeSteps: 15,
    tickDelay: 800
  });
  
  // Add demo agents with diverse initial beliefs
  const agent1 = simulator.addAgent('scientist', 'Dr. Smith');
  agent1.addBelief('Climate change is caused by human activity', 0.92, 'Scientific research and data');
  agent1.addBelief('Vaccines are safe and effective', 0.95, 'Medical studies');
  agent1.addBelief('AI poses existential risks', 0.75, 'Expert opinions and literature');
  
  const agent2 = simulator.addAgent('journalist', 'Ms. Johnson');
  agent2.addBelief('Climate change is caused by human activity', 0.65, 'News reports');
  agent2.addBelief('Vaccines are safe and effective', 0.70, 'Media coverage');
  agent2.addBelief('AI poses existential risks', 0.45, 'Various articles');
  
  const agent3 = simulator.addAgent('citizen', 'Mr. Williams');
  agent3.addBelief('Climate change is caused by human activity', 0.40, 'Mixed information');
  agent3.addBelief('Vaccines are safe and effective', 0.55, 'Personal experiences');
  agent3.addBelief('AI poses existential risks', 0.30, 'Popular media');
  
  console.log('Initial Belief States:\n');
  simulator._recordState();
  visualizer.visualize(simulator, { timeStep: 0 });
  
  // Event 1: New scientific evidence
  console.log('\n📢 EVENT 1: Major scientific study published on climate change\n');
  
  // Directly update agent beliefs based on evidence
  const scientist = simulator.getAgent('scientist');
  const journalist = simulator.getAgent('journalist');
  const citizen = simulator.getAgent('citizen');
  
  if (scientist) {
    scientist.updateBelief('Climate change is caused by human activity', 0.98, 'Comprehensive meta-analysis of climate data');
  }
  if (journalist) {
    journalist.updateBelief('Climate change is caused by human activity', 0.85, 'Breaking news coverage of study');
  }
  
  await new Promise(r => setTimeout(r, 500));
  // Record state and visualize
  simulator._recordState();
  visualizer.visualize(simulator, { timeStep: simulator.history.length - 1 });
  
  // Event 2: Social media discussion
  console.log('\n📢 EVENT 2: Viral social media debate on AI risks\n');
  
  if (journalist) {
    journalist.updateBelief('AI poses existential risks', 0.80, 'Compelling argument from tech expert');
  }
  
  await new Promise(r => setTimeout(r, 500));
  simulator._recordState();
  visualizer.visualize(simulator, { timeStep: simulator.history.length - 1 });
  
  // Event 3: Personal observation
  console.log('\n📢 EVENT 3: Personal experience with extreme weather\n');
  
  if (citizen) {
    citizen.updateBelief('Climate change is caused by human activity', 0.75, 'Witnessed unprecedented weather patterns');
  }
  
  await new Promise(r => setTimeout(r, 500));
  simulator._recordState();
  visualizer.visualize(simulator, { timeStep: simulator.history.length - 1 });
  
  // Event 4: Logical inference
  console.log('\n📢 EVENT 4: Logical inference from new information\n');
  
  // Add a new belief based on inference
  if (scientist) {
    scientist.addBelief('AI will benefit humanity', 0.60, 'Based on updated risk assessment');
  }
  
  await new Promise(r => setTimeout(r, 500));
  simulator._recordState();
  visualizer.visualize(simulator, { timeStep: simulator.history.length - 1 });
  
  // Run remaining steps
  console.log('\n');
  visualizer.printSeparator('=');
  console.log('CONTINUING SIMULATION...');
  visualizer.printSeparator('=');
  
  simulator.run({ async: false });
  
  console.log('\n');
  visualizer.printSeparator('=');
  console.log('FINAL BELIEF STATES');
  visualizer.printSeparator('=');
  
  visualizer.visualize(simulator);
  
  // Show evolution for a key proposition
  console.log('\n');
  visualizer.printEvolution(simulator, 'Climate change is caused by human activity');
  
  // Show comparison
  console.log('\n');
  visualizer.printComparison(simulator, 0, 14);
  
  // Final stats
  console.log('\n');
  visualizer.printStats(simulator);
  
  console.log('\n');
  console.log('╔════════════════════════════════════════════════════════════╗');
  console.log('║                    DEMO COMPLETE                            ║');
  console.log('╚════════════════════════════════════════════════════════════╝');
}

// Main entry point
async function main() {
  const options = parseOptions(args);
  
  // Apply color setting to all commands
  if (!options.colors) {
    // Override color settings globally
    const originalWrite = console.log;
    console.log = (...args) => {
      originalWrite(...args.map(arg => {
        if (typeof arg === 'string') {
          return arg.replace(/\x1b\[[0-9;]*m/g, '');
        }
        return arg;
      }));
    };
  }
  
  try {
    switch (command) {
      case 'init':
        await initCommand(options);
        break;
        
      case 'run':
        await runCommand(options);
        break;
        
      case 'visualize':
        await visualizeCommand(options);
        break;
        
      case 'demo':
        await demoCommand(options);
        break;
        
      case 'help':
      case '-h':
      case '--help':
        console.log(HELP);
        break;
        
      default:
        if (command) {
          console.log(`Unknown command: ${command}`);
          console.log('Run "node cli.js --help" for usage information');
        } else {
          // Default to demo
          await demoCommand(options);
        }
    }
  } catch (error) {
    console.error('Error:', error.message);
    console.error(error.stack);
    process.exit(1);
  }
}

main();
