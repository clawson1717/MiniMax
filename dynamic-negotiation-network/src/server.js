/**
 * Express.js REST API Server for Dynamic Negotiation Network
 * 
 * Provides RESTful endpoints for managing multi-agent negotiations
 * with dynamic topology-based communication.
 */

const express = require('express');
const cors = require('cors');
const Agent = require('./Agent');
const NegotiationRunner = require('./NegotiationRunner');
const AgentNetwork = require('./AgentNetwork');
const SemanticMatcher = require('./SemanticMatcher');
const TopologyManager = require('./TopologyManager');
const { StrategyFactory } = require('./strategies');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(express.json());
app.use(cors());

// In-memory storage for active negotiations
const negotiations = new Map();

// Predefined agent templates
const AGENT_TEMPLATES = [
  {
    id: 'buyer',
    name: 'Buyer Agent',
    description: 'An agent looking to purchase goods or services',
    defaultNeeds: ['Product A', 'Service B'],
    defaultOffers: ['$500', 'Payment terms: Net 30'],
    defaultStrategy: 'counter'
  },
  {
    id: 'seller',
    name: 'Seller Agent',
    description: 'An agent offering goods or services for sale',
    defaultNeeds: ['Payment guarantee', 'Long-term contract'],
    defaultOffers: ['Product X', 'Premium Service'],
    defaultStrategy: 'accept'
  },
  {
    id: 'mediator',
    name: 'Mediator Agent',
    description: 'An agent that facilitates negotiations between parties',
    defaultNeeds: ['Fair outcome', 'Agreement reached'],
    defaultOffers: ['Compromise proposal', 'Arbitration'],
    defaultStrategy: 'counter'
  },
  {
    id: 'supplier',
    name: 'Supplier Agent',
    description: 'An agent providing raw materials or components',
    defaultNeeds: ['Bulk orders', 'Timely payment'],
    defaultOffers: ['Raw materials', 'Components', 'Wholesale pricing'],
    defaultStrategy: 'accept'
  },
  {
    id: 'distributor',
    name: 'Distributor Agent',
    description: 'An agent distributing products to end customers',
    defaultNeeds: ['Exclusive rights', 'Marketing support'],
    defaultOffers: ['Market access', 'Customer network', 'Logistics'],
    defaultStrategy: 'counter'
  },
  {
    id: 'competitor',
    name: 'Competitor Agent',
    description: 'An agent competing for the same resources',
    defaultNeeds: ['Market share', 'Better pricing'],
    defaultOffers: ['Lower prices', 'Superior features'],
    defaultStrategy: 'reject'
  }
];

// Available strategies
const AVAILABLE_STRATEGIES = StrategyFactory.getAvailableStrategies();

/**
 * Generate a unique negotiation ID
 * @returns {string} - Unique ID
 */
function generateNegotiationId() {
  return `neg_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
}

/**
 * Generate a unique agent ID
 * @returns {string} - Unique ID
 */
function generateAgentId(prefix = 'agent') {
  return `${prefix}_${Date.now()}_${Math.random().toString(36).substr(2, 6)}`;
}

/**
 * Create an agent from template or custom configuration
 * @param {Object} config - Agent configuration
 * @returns {Agent} - Created agent
 */
function createAgent(config = {}) {
  const id = config.id || generateAgentId();
  
  // If template is specified, use template defaults
  let templateDefaults = {};
  if (config.template) {
    const template = AGENT_TEMPLATES.find(t => t.id === config.template);
    if (template) {
      templateDefaults = {
        needs: template.defaultNeeds,
        offers: template.defaultOffers,
        strategy: template.defaultStrategy
      };
    }
  }
  
  return new Agent(id, {
    needs: config.needs || templateDefaults.needs || [],
    offers: config.offers || templateDefaults.offers || [],
    strategy: config.strategy || templateDefaults.strategy || 'counter',
    utility: config.utility || 1.0
  });
}

// ============================================
// Health Check Endpoint
// ============================================

/**
 * GET /api/health
 * Health check endpoint
 */
app.get('/api/health', (req, res) => {
  res.json({
    status: 'healthy',
    timestamp: new Date().toISOString(),
    uptime: process.uptime(),
    version: '1.0.0',
    activeNegotiations: negotiations.size
  });
});

// ============================================
// Agent Templates Endpoints
// ============================================

/**
 * GET /api/agents
 * List available agent templates
 */
app.get('/api/agents', (req, res) => {
  res.json({
    templates: AGENT_TEMPLATES,
    strategies: AVAILABLE_STRATEGIES,
    count: AGENT_TEMPLATES.length
  });
});

/**
 * GET /api/agents/:templateId
 * Get a specific agent template
 */
app.get('/api/agents/:templateId', (req, res) => {
  const { templateId } = req.params;
  const template = AGENT_TEMPLATES.find(t => t.id === templateId);
  
  if (!template) {
    return res.status(404).json({
      error: 'Template not found',
      templateId
    });
  }
  
  res.json({
    template,
    strategies: AVAILABLE_STRATEGIES
  });
});

// ============================================
// Negotiation Endpoints
// ============================================

/**
 * POST /api/negotiations
 * Create and start a new negotiation
 * 
 * Request body:
 * {
 *   agents: [
 *     { template: 'buyer', needs: ['...'], offers: ['...'], strategy: 'counter' },
 *     { template: 'seller', needs: ['...'], offers: ['...'], strategy: 'accept' }
 *   ],
 *   options: {
 *     maxRounds: 20,
 *     timeout: 60000,
 *     verbose: false,
 *     topologyOptions: { threshold: 0.3, maxNeighbors: 5 }
 *   }
 * }
 */
app.post('/api/negotiations', (req, res) => {
  try {
    const { agents: agentConfigs = [], options = {} } = req.body;
    
    // Validate request
    if (!Array.isArray(agentConfigs) || agentConfigs.length < 2) {
      return res.status(400).json({
        error: 'Bad request',
        message: 'At least 2 agents are required to start a negotiation'
      });
    }
    
    // Generate negotiation ID
    const negotiationId = generateNegotiationId();
    
    // Create agent network
    const networkOptions = options.topologyOptions || {};
    const network = new AgentNetwork({
      topology: networkOptions,
      maxRounds: options.maxRounds || 20
    });
    
    // Create and add agents to network
    const agents = [];
    for (const config of agentConfigs) {
      const agent = createAgent(config);
      network.addAgent(agent);
      agents.push(agent);
    }
    
    // Activate the network and build topology
    network.activate();
    
    // Create negotiation runner
    const runner = new NegotiationRunner(network, {
      maxRounds: options.maxRounds || 20,
      timeout: options.timeout || 60000,
      verbose: options.verbose || false
    });
    
    // Store negotiation
    const negotiation = {
      id: negotiationId,
      status: 'created',
      createdAt: Date.now(),
      network,
      runner,
      agents,
      options,
      messages: []
    };
    
    negotiations.set(negotiationId, negotiation);
    
    // Start the negotiation asynchronously
    setImmediate(() => {
      negotiation.status = 'running';
      const result = runner.run();
      negotiation.status = result.status;
      negotiation.completedAt = Date.now();
      negotiation.result = result;
    });
    
    // Return immediate response with negotiation ID
    res.status(201).json({
      success: true,
      negotiationId,
      status: 'created',
      message: 'Negotiation created and started',
      agents: agents.map(a => ({
        id: a.id,
        needs: a.getNeeds(),
        offers: a.getOffers(),
        strategy: a.getStrategy()
      })),
      topology: network.getStatistics(),
      links: {
        self: `/api/negotiations/${negotiationId}`,
        status: `/api/negotiations/${negotiationId}`,
        message: `/api/negotiations/${negotiationId}/message`
      }
    });
    
  } catch (error) {
    console.error('Error creating negotiation:', error);
    res.status(500).json({
      error: 'Internal server error',
      message: error.message
    });
  }
});

/**
 * GET /api/negotiations
 * List all negotiations
 */
app.get('/api/negotiations', (req, res) => {
  const negotiationList = Array.from(negotiations.values()).map(neg => ({
    id: neg.id,
    status: neg.status,
    createdAt: neg.createdAt,
    completedAt: neg.completedAt,
    agentCount: neg.agents.length,
    currentRound: neg.runner ? neg.runner.getCurrentRound() : 0,
    maxRounds: neg.options.maxRounds || 20,
    links: {
      self: `/api/negotiations/${neg.id}`
    }
  }));
  
  res.json({
    negotiations: negotiationList,
    count: negotiationList.length
  });
});

/**
 * GET /api/negotiations/:id
 * Get negotiation status and results
 */
app.get('/api/negotiations/:id', (req, res) => {
  const { id } = req.params;
  const negotiation = negotiations.get(id);
  
  if (!negotiation) {
    return res.status(404).json({
      error: 'Not found',
      message: `Negotiation with ID '${id}' not found`
    });
  }
  
  const runner = negotiation.runner;
  const network = negotiation.network;
  
  const response = {
    id: negotiation.id,
    status: negotiation.status,
    createdAt: new Date(negotiation.createdAt).toISOString(),
    completedAt: negotiation.completedAt ? new Date(negotiation.completedAt).toISOString() : null,
    duration: negotiation.completedAt ? negotiation.completedAt - negotiation.createdAt : Date.now() - negotiation.createdAt,
    agents: negotiation.agents.map(agent => ({
      id: agent.id,
      needs: agent.getNeeds(),
      offers: agent.getOffers(),
      strategy: agent.getStrategy(),
      needsSatisfied: agent.needsSatisfied(),
      historyLength: agent.getHistory().length
    })),
    network: {
      statistics: network.getStatistics(),
      topology: network.topology.getGraph(),
      currentRound: network.getCurrentRound()
    },
    progress: {
      currentRound: runner.getCurrentRound(),
      maxRounds: negotiation.options.maxRounds || 20,
      status: runner.getStatus(),
      summary: runner.getSummary()
    },
    messages: negotiation.messages,
    links: {
      self: `/api/negotiations/${id}`,
      message: `/api/negotiations/${id}/message`
    }
  };
  
  // Include result if negotiation is complete
  if (negotiation.result) {
    response.result = negotiation.result;
  }
  
  res.json(response);
});

/**
 * DELETE /api/negotiations/:id
 * Delete a negotiation
 */
app.delete('/api/negotiations/:id', (req, res) => {
  const { id } = req.params;
  const negotiation = negotiations.get(id);
  
  if (!negotiation) {
    return res.status(404).json({
      error: 'Not found',
      message: `Negotiation with ID '${id}' not found`
    });
  }
  
  negotiations.delete(id);
  
  res.json({
    success: true,
    message: `Negotiation '${id}' deleted successfully`
  });
});

// ============================================
// Message Endpoints
// ============================================

/**
 * POST /api/negotiations/:id/message
 * Send a message between agents in a negotiation
 * 
 * Request body:
 * {
 *   from: 'agent_id',
 *   to: 'agent_id',
 *   type: 'offer' | 'counter' | 'accept' | 'reject' | 'query',
 *   content: {
 *     offer: '...',
 *     message: '...',
 *     ...
 *   }
 * }
 */
app.post('/api/negotiations/:id/message', (req, res) => {
  try {
    const { id } = req.params;
    const { from, to, type = 'message', content = {} } = req.body;
    
    const negotiation = negotiations.get(id);
    
    if (!negotiation) {
      return res.status(404).json({
        error: 'Not found',
        message: `Negotiation with ID '${id}' not found`
      });
    }
    
    // Validate required fields
    if (!from || !to) {
      return res.status(400).json({
        error: 'Bad request',
        message: 'Both "from" and "to" agent IDs are required'
      });
    }
    
    const network = negotiation.network;
    
    // Validate agents exist
    const sender = network.getAgent(from);
    const receiver = network.getAgent(to);
    
    if (!sender) {
      return res.status(400).json({
        error: 'Bad request',
        message: `Sender agent '${from}' not found in negotiation`
      });
    }
    
    if (!receiver) {
      return res.status(400).json({
        error: 'Bad request',
        message: `Receiver agent '${to}' not found in negotiation`
      });
    }
    
    // Validate message type
    const validTypes = ['offer', 'counter', 'accept', 'reject', 'query', 'message'];
    if (!validTypes.includes(type)) {
      return res.status(400).json({
        error: 'Bad request',
        message: `Invalid message type. Must be one of: ${validTypes.join(', ')}`
      });
    }
    
    // Send the message
    const success = network.send({
      from,
      to,
      type,
      content
    });
    
    if (!success) {
      return res.status(500).json({
        error: 'Internal server error',
        message: 'Failed to send message'
      });
    }
    
    // Record the message in negotiation history
    const messageRecord = {
      id: `msg_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      from,
      to,
      type,
      content,
      timestamp: Date.now(),
      round: network.getCurrentRound()
    };
    
    negotiation.messages.push(messageRecord);
    
    // Record event in sender's history
    sender.recordEvent({
      type: 'sent',
      to,
      messageType: type,
      content
    });
    
    // Record event in receiver's history
    receiver.recordEvent({
      type: 'received',
      from,
      messageType: type,
      content
    });
    
    // If this is an interactive negotiation (not auto-run), step the runner
    if (negotiation.status === 'interactive' || negotiation.status === 'running') {
      // Check if we should advance the round
      const pendingMessages = network.getPendingMessageCount(to);
      if (pendingMessages > 0) {
        // Let the receiver process the message
        const receiverMessages = network.receiveMessage(to);
        
        // Generate response based on receiver's strategy
        const strategyName = receiver.getStrategy() || 'counter';
        const strategy = StrategyFactory.create(strategyName);
        
        const context = {
          agent: receiver,
          messages: receiverMessages,
          offer: content.offer || content.counterOffer,
          senderOffers: content.senderOffers || [],
          roundNumber: network.getCurrentRound()
        };
        
        const decision = strategy.evaluate(context);
        
        messageRecord.response = {
          agentId: to,
          decision,
          timestamp: Date.now()
        };
      }
    }
    
    res.status(201).json({
      success: true,
      message: 'Message sent successfully',
      messageId: messageRecord.id,
      negotiationId: id,
      from,
      to,
      type,
      round: network.getCurrentRound(),
      timestamp: new Date(messageRecord.timestamp).toISOString(),
      links: {
        negotiation: `/api/negotiations/${id}`,
        self: `/api/negotiations/${id}/message`
      }
    });
    
  } catch (error) {
    console.error('Error sending message:', error);
    res.status(500).json({
      error: 'Internal server error',
      message: error.message
    });
  }
});

/**
 * GET /api/negotiations/:id/messages
 * Get all messages for a negotiation
 */
app.get('/api/negotiations/:id/messages', (req, res) => {
  const { id } = req.params;
  const { agentId, round, type } = req.query;
  
  const negotiation = negotiations.get(id);
  
  if (!negotiation) {
    return res.status(404).json({
      error: 'Not found',
      message: `Negotiation with ID '${id}' not found`
    });
  }
  
  let messages = negotiation.messages;
  
  // Apply filters if provided
  if (agentId) {
    messages = messages.filter(m => m.from === agentId || m.to === agentId);
  }
  
  if (round) {
    const roundNum = parseInt(round, 10);
    messages = messages.filter(m => m.round === roundNum);
  }
  
  if (type) {
    messages = messages.filter(m => m.type === type);
  }
  
  res.json({
    negotiationId: id,
    count: messages.length,
    messages: messages.sort((a, b) => a.timestamp - b.timestamp),
    links: {
      negotiation: `/api/negotiations/${id}`
    }
  });
});

// ============================================
// Advanced Negotiation Control Endpoints
// ============================================

/**
 * POST /api/negotiations/:id/step
 * Step the negotiation forward by one round (for interactive mode)
 */
app.post('/api/negotiations/:id/step', (req, res) => {
  const { id } = req.params;
  const negotiation = negotiations.get(id);
  
  if (!negotiation) {
    return res.status(404).json({
      error: 'Not found',
      message: `Negotiation with ID '${id}' not found`
    });
  }
  
  const runner = negotiation.runner;
  
  if (runner.getStatus() === 'completed' || runner.getStatus() === 'timeout' || runner.getStatus() === 'error') {
    return res.status(400).json({
      error: 'Bad request',
      message: 'Negotiation has already finished'
    });
  }
  
  const stepResult = runner.step();
  
  // Update negotiation status
  if (stepResult.isComplete) {
    negotiation.status = runner.getStatus();
    negotiation.completedAt = Date.now();
    negotiation.result = runner.getResult();
  }
  
  res.json({
    success: true,
    negotiationId: id,
    step: stepResult,
    status: runner.getStatus(),
    currentRound: runner.getCurrentRound(),
    summary: runner.getSummary()
  });
});

/**
 * POST /api/negotiations/:id/reset
 * Reset a negotiation to its initial state
 */
app.post('/api/negotiations/:id/reset', (req, res) => {
  const { id } = req.params;
  const negotiation = negotiations.get(id);
  
  if (!negotiation) {
    return res.status(404).json({
      error: 'Not found',
      message: `Negotiation with ID '${id}' not found`
    });
  }
  
  // Reset runner and network
  negotiation.runner.reset();
  negotiation.network.reset();
  negotiation.network.activate();
  
  // Reset negotiation state
  negotiation.status = 'idle';
  negotiation.messages = [];
  negotiation.completedAt = null;
  negotiation.result = null;
  
  res.json({
    success: true,
    message: 'Negotiation reset successfully',
    negotiationId: id,
    status: negotiation.status
  });
});

// ============================================
// Error Handling
// ============================================

// 404 handler
app.use((req, res) => {
  res.status(404).json({
    error: 'Not found',
    message: `Endpoint ${req.method} ${req.path} not found`,
    availableEndpoints: [
      'GET /api/health',
      'GET /api/agents',
      'GET /api/agents/:templateId',
      'GET /api/negotiations',
      'POST /api/negotiations',
      'GET /api/negotiations/:id',
      'DELETE /api/negotiations/:id',
      'POST /api/negotiations/:id/message',
      'GET /api/negotiations/:id/messages',
      'POST /api/negotiations/:id/step',
      'POST /api/negotiations/:id/reset'
    ]
  });
});

// Global error handler
app.use((err, req, res, next) => {
  console.error('Server error:', err);
  res.status(500).json({
    error: 'Internal server error',
    message: err.message,
    stack: process.env.NODE_ENV === 'development' ? err.stack : undefined
  });
});

// ============================================
// Start Server
// ============================================

const server = app.listen(PORT, () => {
  console.log(`🚀 Dynamic Negotiation Network Server running on port ${PORT}`);
  console.log(`📊 Health check: http://localhost:${PORT}/api/health`);
  console.log(`🤖 Agent templates: http://localhost:${PORT}/api/agents`);
  console.log(`📝 API Documentation:`);
  console.log(`   POST /api/negotiations - Create a new negotiation`);
  console.log(`   GET  /api/negotiations/:id - Get negotiation status`);
  console.log(`   POST /api/negotiations/:id/message - Send message between agents`);
});

// Graceful shutdown
process.on('SIGTERM', () => {
  console.log('SIGTERM received, shutting down gracefully');
  server.close(() => {
    console.log('Server closed');
    process.exit(0);
  });
});

process.on('SIGINT', () => {
  console.log('SIGINT received, shutting down gracefully');
  server.close(() => {
    console.log('Server closed');
    process.exit(0);
  });
});

module.exports = { app, server, negotiations };