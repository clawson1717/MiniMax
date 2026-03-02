#!/usr/bin/env node

/**
 * CLI Tool for Resilient Adaptive Agent
 * Provides command-line interface for running RAA tasks, initialization, and status checks
 */

const { Command } = require('commander');
const fs = require('fs');
const path = require('path');
const { RAA } = require('./RAA');

/**
 * Load configuration from file
 * @param {string} configPath - Path to config file
 * @returns {Object} Configuration object
 */
function loadConfig(configPath) {
  try {
    const absolutePath = path.resolve(configPath);
    if (!fs.existsSync(absolutePath)) {
      throw new Error(`Config file not found: ${absolutePath}`);
    }
    const configContent = fs.readFileSync(absolutePath, 'utf-8');
    return JSON.parse(configContent);
  } catch (error) {
    if (error instanceof SyntaxError) {
      throw new Error(`Invalid JSON in config file: ${error.message}`);
    }
    throw error;
  }
}

/**
 * Initialize a new RAA agent configuration
 * @param {string} configPath - Path where config should be saved
 * @param {boolean} verbose - Enable verbose output
 */
function initAgent(configPath, verbose = false) {
  const defaultConfig = {
    name: 'RAA',
    maxSteps: 100,
    verbose: false,
    agentConfig: {
      headless: true,
      timeout: 30000
    },
    uncertaintyConfig: {
      uncertaintyThreshold: 0.3,
      windowSize: 10
    },
    cattsConfig: {
      minComputeUnits: 1,
      maxComputeUnits: 10,
      allocationStrategy: 'uncertainty_based'
    },
    failureDetectorConfig: {
      suspicionThreshold: 0.7,
      consecutiveFailureLimit: 3
    },
    resilienceConfig: {
      enableCheckpoints: true,
      maxRecoveryAttempts: 3
    },
    verificationConfig: {
      strictMode: true,
      maxRetries: 2
    }
  };

  const absolutePath = path.resolve(configPath);
  fs.writeFileSync(absolutePath, JSON.stringify(defaultConfig, null, 2));
  
  if (verbose) {
    console.log('Initialized RAA configuration at:', absolutePath);
  } else {
    console.log('Configuration saved to:', absolutePath);
  }
}

/**
 * Run a task with the RAA agent
 * @param {string} task - Task description or URL
 * @param {Object} options - Command options
 */
async function runTask(task, options) {
  const verbose = options.verbose || false;
  const maxSteps = options.maxSteps || 100;
  const configPath = options.config;

  let config = {
    maxSteps,
    verbose
  };

  // Load config from file if provided
  if (configPath) {
    try {
      const fileConfig = loadConfig(configPath);
      config = { ...fileConfig, ...config };
    } catch (error) {
      console.error('Error loading config:', error.message);
      process.exit(1);
    }
  }

  if (verbose) {
    console.log('Starting RAA with config:', JSON.stringify(config, null, 2));
  }

  const agent = new RAA(config);

  try {
    if (verbose) {
      console.log('Initializing RAA...');
    }
    await agent.initialize();

    if (verbose) {
      console.log('Running task:', task);
    }
    const result = await agent.runTask(task);

    console.log('\n=== Task Result ===');
    console.log('Status:', result.success ? 'SUCCESS' : 'FAILED');
    console.log('Steps taken:', result.steps.length);
    console.log('Final state:', result.finalState);

    if (verbose) {
      console.log('\nDetailed results:', JSON.stringify(result, null, 2));
    }

    process.exit(result.success ? 0 : 1);
  } catch (error) {
    console.error('Error running task:', error.message);
    if (verbose) {
      console.error(error.stack);
    }
    process.exit(1);
  }
}

/**
 * Show agent status
 * @param {Object} options - Command options
 */
async function showStatus(options) {
  const verbose = options.verbose || false;
  const configPath = options.config;

  let config = { verbose };
  
  if (configPath) {
    try {
      const fileConfig = loadConfig(configPath);
      config = { ...fileConfig, ...config };
    } catch (error) {
      console.error('Error loading config:', error.message);
      process.exit(1);
    }
  }

  const agent = new RAA(config);

  try {
    await agent.initialize();

    console.log('\n=== RAA Status ===');
    console.log('Name:', agent.name);
    console.log('Initialized:', agent.isInitialized);
    console.log('Running:', agent.isRunning);
    console.log('Max Steps:', agent.maxSteps);
    console.log('Verbose:', agent.verbose);

    console.log('\n=== Metrics ===');
    console.log('Tasks Completed:', agent.metrics.tasksCompleted);
    console.log('Tasks Failed:', agent.metrics.tasksFailed);
    console.log('Total Actions:', agent.metrics.totalActions);
    console.log('Successful Verifications:', agent.metrics.successfulVerifications);
    console.log('Failed Verifications:', agent.metrics.failedVerifications);
    console.log('Recoveries Applied:', agent.metrics.recoveriesApplied);

    if (verbose) {
      console.log('\n=== Full Status ===');
      console.log(JSON.stringify(agent.getStatus(), null, 2));
    }
  } catch (error) {
    console.error('Error getting status:', error.message);
    if (verbose) {
      console.error(error.stack);
    }
    process.exit(1);
  }
}

/**
 * Run tests
 * @param {Object} options - Command options
 */
function runTests(options) {
  const verbose = options.verbose || false;
  
  const { execSync } = require('child_process');
  
  try {
    if (verbose) {
      console.log('Running npm test...');
    }
    execSync('npm test', { stdio: 'inherit', cwd: __dirname + '/..' });
  } catch (error) {
    console.error('Tests failed');
    process.exit(1);
  }
}

// Create CLI program
const program = new Command();

// Configure program
program
  .name('raa')
  .description('CLI tool for Resilient Adaptive Agent')
  .version('0.1.0');

// Run command
program
  .command('run <task>')
  .description('Run a task with the RAA agent')
  .option('-c, --config <path>', 'Path to config file')
  .option('-v, --verbose', 'Enable verbose output')
  .option('-m, --max-steps <n>', 'Maximum steps', parseInt)
  .action(runTask);

// Init command
program
  .command('init')
  .description('Initialize a new RAA agent config')
  .option('-c, --config <path>', 'Path to save config file', 'raa-config.json')
  .option('-v, --verbose', 'Enable verbose output')
  .action((options) => initAgent(options.config, options.verbose));

// Status command
program
  .command('status')
  .description('Show agent status')
  .option('-c, --config <path>', 'Path to config file')
  .option('-v, --verbose', 'Enable verbose output')
  .action(showStatus);

// Test command
program
  .command('test')
  .description('Run tests')
  .option('-v, --verbose', 'Enable verbose output')
  .action(runTests);

// Parse arguments
program.parse(process.argv);

// Show help if no command provided
if (process.argv.length === 2) {
  program.help();
}
