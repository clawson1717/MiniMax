/**
 * Express REST API Server for Dynamic Negotiation Network
 * 
 * Provides HTTP endpoints for:
 * - Health checks
 * - Agent template management
 * - Negotiation creation and management
 * - Message passing between agents
 */

const express = require('express');
const cors = require('cors');
const crypto = require('crypto');

// Import existing modules
const Agent = require('./Agent');
const AgentNetwork = require('./AgentNetwork');
const NegotiationRunner = require('./NegotiationRunner');

// Create Express app
const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json());

// Store negotiations in a Map with UUID keys
const negotiations = new Map();

// Sample agent templates
const agentTemplates = [
  {
    id: 'supplier-agent',
    name: 'Supplier Agent',
    description: 'A supplier looking to sell goods and services',
    needs: ['cash payment', 'long-term contract'],
    offers: ['raw materials: $500/ton', 'bulk shipping', 'quality assurance'],
    strategy: 'counter',
    utility: 1.0
  },
  {
    id: 'buyer-agent',
    name: 'Buyer Agent',
    description: 'A buyer looking to purchase goods at competitive prices',
    needs: ['raw materials: $400/ton', 'fast delivery'],
    offers: ['cash payment', 'bulk orders', 'referrals'],
    strategy: 'accept',
    utility: 0.9
  },
  {
    id: 'mediator-agent',
    name: 'Mediator Agent',
    description: 'A neutral mediator facilitating negotiations',
    needs: ['fair deal', 'transaction fee'],
    offers: ['dispute resolution', 'contract drafting', 'market insights'],
    strategy: 'random',
    utility: 0.8
  },
  {
    id: 'distributor-agent',
    name: 'Distributor Agent',
    description: 'A distributor managing logistics and supply chain',
    needs: ['storage space', 'reliable transport'],
    offers: ['last-mile delivery', 'inventory management', 'tracking systems'],
    strategy: 'counter',
    utility: 0.85
  }
];

/**
 * Generate a UUID v4
 * @returns {string} UUID
 */
function generateUUID() {
  return crypto.randomUUID();
}

/**
 * Create a new negotiation session
 * @param {Object} config - Negotiation configuration
 * @returns {Object} Negotiation session object
 */
function createNegotiation(config) {
  const id = generateUUID();
  const { agents: agentConfigs, topic, maxRounds = 10 } = config;
  
  // Create agent network
  const network = new AgentNetwork({ maxRounds });
  
  // Create agents from config
  const agents = [];
  for (const agentConfig of agentConfigs) {
    const agentId = agentConfig.name || `agent-${generateUUID().slice(0, 8)}`;
    const agent = new Agent(agentId, {
      needs: agentConfig.needs || [],
      offers: agentConfig.offers || [],
      strategy: agentConfig.strategy || 'counter',
      utility: agentConfig.utility || 1.0
    });
    network.addAgent(agent);
    agents.push({
      id: agentId,
      name: agentConfig.name,
      needs: agentConfig.needs,
      offers: agentConfig.offers,
      strategy: agentConfig.strategy
    });
  }
  
  // Create negotiation runner
  const runner = new NegotiationRunner(network, {
    maxRounds,
    timeout: 60000,
    verbose: false
  });
  
  // Build initial topology
  network.rebuildTopology();
  
  const negotiation = {
    id,
    topic: topic || 'General Negotiation',
    status: 'created', // created, running, completed, timeout, error
    maxRounds,
    agents,
    network,
    runner,
    messages: [],
    createdAt: Date.now(),
    startedAt: null,
    completedAt: null,
    result: null
  };
  
  negotiations.set(id, negotiation);
  return negotiation;
}

// ============================================
// API Routes
// ============================================

/**
 * GET /api/health
 * Health check endpoint
 */
app.get('/api/health', (req, res) => {
  res.json({
    status: 'ok',
    timestamp: Date.now()
  });
});

/**
 * GET /api/agents
 * Return list of agent templates
 */
app.get('/api/agents', (req, res) => {
  res.json({
    templates: agentTemplates,
    count: agentTemplates.length
  });
});

/**
 * POST /api/negotiations
 * Create a new negotiation
 * Body: { agents: [{name, needs, offers}], topic, maxRounds }
 */
app.post('/api/negotiations', (req, res) => {
  try {
    const { agents, topic, maxRounds } = req.body;
    
    // Validate request
    if (!agents || !Array.isArray(agents) || agents.length < 2) {
      return res.status(400).json({
        error: 'At least 2 agents are required'
      });
    }
    
    // Validate each agent config
    for (const agent of agents) {
      if (!agent.name) {
        return res.status(400).json({
          error: 'Each agent must have a name'
        });
      }
    }
    
    // Create negotiation
    const negotiation = createNegotiation({
      agents,
      topic,
      maxRounds: maxRounds || 10
    });
    
    res.status(201).json({
      id: negotiation.id,
      status: negotiation.status,
      agents: negotiation.agents,
      topic: negotiation.topic,
      maxRounds: negotiation.maxRounds,
      createdAt: negotiation.createdAt
    });
  } catch (error) {
    console.error('Error creating negotiation:', error);
    res.status(500).json({
      error: 'Failed to create negotiation',
      message: error.message
    });
  }
});

/**
 * GET /api/negotiations/:id
 * Get negotiation status/result
 */
app.get('/api/negotiations/:id', (req, res) => {
  const { id } = req.params;
  const negotiation = negotiations.get(id);
  
  if (!negotiation) {
    return res.status(404).json({
      error: 'Negotiation not found'
    });
  }
  
  // Get current runner status if available
  let runnerStatus = null;
  let runnerResult = null;
  
  if (negotiation.runner) {
    runnerStatus = negotiation.runner.getStatus();
    runnerResult = negotiation.runner.getResult();
  }
  
  res.json({
    id: negotiation.id,
    topic: negotiation.topic,
    status: runnerStatus || negotiation.status,
    maxRounds: negotiation.maxRounds,
    currentRound: negotiation.runner ? negotiation.runner.getCurrentRound() : 0,
    agents: negotiation.agents,
    messages: negotiation.messages,
    createdAt: negotiation.createdAt,
    startedAt: negotiation.startedAt,
    completedAt: negotiation.completedAt,
    result: runnerResult || negotiation.result
  });
});

/**
 * POST /api/negotiations/:id/message
 * Add a message to negotiation
 * Body: { from, to, content, type }
 */
app.post('/api/negotiations/:id/message', (req, res) => {
  const { id } = req.params;
  const { from, to, content, type = 'message' } = req.body;
  
  const negotiation = negotiations.get(id);
  
  if (!negotiation) {
    return res.status(404).json({
      error: 'Negotiation not found'
    });
  }
  
  // Validate required fields
  if (!from || !to || !content) {
    return res.status(400).json({
      error: 'Missing required fields: from, to, content'
    });
  }
  
  // Check if agents exist in the network
  const network = negotiation.network;
  const fromAgent = network.getAgent(from);
  const toAgent = network.getAgent(to);
  
  if (!fromAgent) {
    return res.status(400).json({
      error: `Sender agent '${from}' not found in negotiation`
    });
  }
  
  if (!toAgent) {
    return res.status(400).json({
      error: `Receiver agent '${to}' not found in negotiation`
    });
  }
  
  // Create message
  const message = {
    id: generateUUID(),
    from,
    to,
    type,
    content,
    timestamp: Date.now()
  };
  
  // Add to negotiation messages
  negotiation.messages.push(message);
  
  // Also send through the network for processing
  try {
    network.send({
      from,
      to,
      type,
      content
    });
    
    res.status(201).json({
      success: true,
      message: {
        id: message.id,
        from: message.from,
        to: message.to,
        type: message.type,
        timestamp: message.timestamp
      }
    });
  } catch (error) {
    console.error('Error sending message:', error);
    res.status(500).json({
      error: 'Failed to send message',
      message: error.message
    });
  }
});

/**
 * POST /api/negotiations/:id/run
 * Run the negotiation (execute all rounds)
 */
app.post('/api/negotiations/:id/run', (req, res) => {
  const { id } = req.params;
  const negotiation = negotiations.get(id);
  
  if (!negotiation) {
    return res.status(404).json({
      error: 'Negotiation not found'
    });
  }
  
  try {
    // Activate network
    negotiation.network.activate();
    
    // Mark as started
    if (negotiation.status === 'created') {
      negotiation.status = 'running';
      negotiation.startedAt = Date.now();
    }
    
    // Run the negotiation
    const result = negotiation.runner.run();
    
    // Update status
    negotiation.status = result.status;
    negotiation.completedAt = Date.now();
    negotiation.result = result;
    
    res.json({
      id: negotiation.id,
      status: negotiation.status,
      result
    });
  } catch (error) {
    console.error('Error running negotiation:', error);
    negotiation.status = 'error';
    res.status(500).json({
      error: 'Failed to run negotiation',
      message: error.message
    });
  }
});

/**
 * POST /api/negotiations/:id/step
 * Execute a single step/round of the negotiation
 */
app.post('/api/negotiations/:id/step', (req, res) => {
  const { id } = req.params;
  const negotiation = negotiations.get(id);
  
  if (!negotiation) {
    return res.status(404).json({
      error: 'Negotiation not found'
    });
  }
  
  try {
    // Activate network on first step
    if (negotiation.status === 'created') {
      negotiation.network.activate();
      negotiation.status = 'running';
      negotiation.startedAt = Date.now();
    }
    
    // Execute one step
    const stepResult = negotiation.runner.step();
    
    // Update status if complete
    if (stepResult.isComplete) {
      negotiation.status = negotiation.runner.getStatus();
      negotiation.completedAt = Date.now();
      negotiation.result = negotiation.runner.getResult();
    }
    
    res.json({
      id: negotiation.id,
      status: negotiation.status,
      round: stepResult.round,
      isComplete: stepResult.isComplete,
      result: stepResult
    });
  } catch (error) {
    console.error('Error executing step:', error);
    res.status(500).json({
      error: 'Failed to execute step',
      message: error.message
    });
  }
});

/**
 * DELETE /api/negotiations/:id
 * Delete a negotiation
 */
app.delete('/api/negotiations/:id', (req, res) => {
  const { id } = req.params;
  
  if (!negotiations.has(id)) {
    return res.status(404).json({
      error: 'Negotiation not found'
    });
  }
  
  negotiations.delete(id);
  
  res.json({
    success: true,
    message: 'Negotiation deleted'
  });
});

/**
 * GET /api/negotiations
 * List all negotiations
 */
app.get('/api/negotiations', (req, res) => {
  const allNegotiations = Array.from(negotiations.values()).map(n => ({
    id: n.id,
    topic: n.topic,
    status: n.runner ? n.runner.getStatus() : n.status,
    agentCount: n.agents.length,
    currentRound: n.runner ? n.runner.getCurrentRound() : 0,
    maxRounds: n.maxRounds,
    createdAt: n.createdAt
  }));
  
  res.json({
    negotiations: allNegotiations,
    count: allNegotiations.length
  });
});

// Error handling middleware
app.use((err, req, res, next) => {
  console.error('Express error:', err);
  res.status(500).json({
    error: 'Internal server error',
    message: err.message
  });
});

// 404 handler
app.use((req, res) => {
  res.status(404).json({
    error: 'Not found',
    path: req.path
  });
});

// Start server
if (require.main === module) {
  app.listen(PORT, () => {
    console.log(`Dynamic Negotiation Network API server running on port ${PORT}`);
    console.log(`Health check: http://localhost:${PORT}/api/health`);
  });
}

module.exports = app;
