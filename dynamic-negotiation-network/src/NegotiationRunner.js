/**
 * NegotiationRunner
 * 
 * Orchestrates multi-round negotiations between agents.
 * Manages execution, timeout, history tracking, and result retrieval.
 * 
 * Step 14: Integrated DynamicTopologyManager for adaptive communication topology
 */

const NegotiationRound = require('./NegotiationRound');
const { StrategyFactory } = require('./strategies');
const DynamicTopologyManager = require('./TopologyManager');

class NegotiationRunner {
  /**
   * Create a NegotiationRunner
   * @param {AgentNetwork} network - The agent network
   * @param {Object} options - Configuration options @param {number
   *} options.maxRounds - Maximum number of negotiation rounds
   * @param {number} options.timeout - Timeout in milliseconds
   * @param {boolean} options.verbose - Enable verbose logging
   * @param {boolean} options.dynamicTopology - Enable dynamic topology rebuilding (default: true)
   * @param {number} options.topologyRebuildInterval - Rebuild topology every N rounds (default: 1)
   */
  constructor(network, options = {}) {
    this.network = network;
    this.maxRounds = options.maxRounds || 20;
    this.timeout = options.timeout || 60000; // 1 minute default
    this.verbose = options.verbose || false;
    
    // Dynamic topology settings
    this.dynamicTopology = options.dynamicTopology !== false; // default true
    this.topologyRebuildInterval = options.topologyRebuildInterval || 1;
    
    // State
    this.rounds = [];
    this.history = [];
    this.currentRound = 0;
    this.status = 'idle'; // idle, running, completed, timeout, error
    this.startTime = null;
    this.endTime = null;
    this.result = null;
    
    // Strategy cache
    this.strategies = new Map();
    
    // Initialize dynamic topology manager
    if (this.dynamicTopology) {
      this.topologyManager = new DynamicTopologyManager({
        threshold: options.topologyThreshold || 0.3,
        maxNeighbors: options.topologyMaxNeighbors || 5
      });
      
      // Register all agents with topology manager
      const agents = this.network.getAllAgents();
      agents.forEach(agent => this.topologyManager.registerAgent(agent));
      
      // Build initial topology
      this.topologyManager.buildTopology();
      this._log('Dynamic topology initialized');
    }
    
    // Track topology changes
    this.topologyHistory = [];
  }

  /**
   * Run the full negotiation
   * @returns {Object} - Final negotiation result
   */
  run() {
    if (this.status === 'running') {
      return {
        success: false,
        message: 'Negotiation already in progress'
      };
    }
    
    this.status = 'running';
    this.startTime = Date.now();
    this.rounds = [];
    this.history = [];
    this.currentRound = 0;
    this.topologyHistory = [];
    
    this._log('Starting negotiation...');
    if (this.dynamicTopology) {
      this._log('Dynamic topology enabled - communication graph will adapt to agent needs/offers');
    }
    
    try {
      // Execute rounds until agreement or timeout
      while (this.currentRound < this.maxRounds) {
        // Check timeout
        if (Date.now() - this.startTime > this.timeout) {
          this.status = 'timeout';
          this.endTime = Date.now();
          this._log(`Timeout reached after ${this.currentRound} rounds`);
          break;
        }
        
        // Check if all agents have satisfied needs
        if (this._checkAgreement()) {
          this.status = 'completed';
          this.endTime = Date.now();
          this._log('Agreement reached!');
          break;
        }
        
        // Rebuild topology if dynamic topology is enabled
        if (this.dynamicTopology && this.currentRound % this.topologyRebuildInterval === 0) {
          this._rebuildTopology();
        }
        
        // Execute next round
        const roundResult = this._executeRound();
        this.rounds.push(roundResult);
        
        if (!roundResult.success) {
          this.status = 'error';
          this.endTime = Date.now();
          break;
        }
        
        this.currentRound++;
      }
      
      // Finalize result if not already set
      if (this.status === 'running') {
        this.status = this.currentRound >= this.maxRounds ? 'completed' : 'completed';
        this.endTime = Date.now();
      }
      
      return this.getResult();
      
    } catch (error) {
      this.status = 'error';
      this.endTime = Date.now();
      this.result = {
        success: false,
        error: error.message,
        status: this.status
      };
      return this.result;
    }
  }

  /**
   * Rebuild the communication topology based on current agent needs/offers
   * @private
   */
  _rebuildTopology() {
    if (!this.topologyManager) return;
    
    this._log(`Rebuilding topology at round ${this.currentRound + 1}...`);
    
    // Update agent registrations (in case needs/offers changed)
    const agents = this.network.getAllAgents();
    agents.forEach(agent => {
      if (!this.topologyManager.agents.has(agent.id)) {
        this.topologyManager.registerAgent(agent);
      }
    });
    
    // Rebuild topology
    const previousGraph = this._getTopologySnapshot();
    this.topologyManager.buildTopology();
    const newGraph = this._getTopologySnapshot();
    
    // Record topology change
    const change = {
      round: this.currentRound + 1,
      timestamp: Date.now(),
      previous: previousGraph,
      current: newGraph,
      changed: JSON.stringify(previousGraph) !== JSON.stringify(newGraph)
    };
    
    this.topologyHistory.push(change);
    
    if (change.changed) {
      this._log('  Topology adapted to changing agent needs/offers');
    }
  }

  /**
   * Get a snapshot of the current topology
   * @private
   * @returns {Object} - Topology snapshot
   */
  _getTopologySnapshot() {
    if (!this.topologyManager) return null;
    
    const snapshot = {};
    for (const [agentId, neighbors] of this.topologyManager.graph.entries()) {
      snapshot[agentId] = Array.from(neighbors);
    }
    return snapshot;
  }

  /**
   * Get the current communication topology
   * @returns {Object|null} - Current topology or null if disabled
   */
  getCurrentTopology() {
    return this._getTopologySnapshot();
  }

  /**
   * Get topology change history
   * @returns {Object[]} - Array of topology changes
   */
  getTopologyHistory() {
    return [...this.topologyHistory];
  }

  /**
   * Check if two agents can communicate based on current topology
   * @param {string} agentId1 - First agent ID
   * @param {string} agentId2 - Second agent ID
   * @returns {boolean} - True if agents can communicate
   */
  canCommunicate(agentId1, agentId2) {
    if (!this.dynamicTopology || !this.topologyManager) {
      // If dynamic topology disabled, all agents can communicate
      return true;
    }
    
    return this.topologyManager.areNeighbors(agentId1, agentId2);
  }

  /**
   * Get valid communication partners for an agent
   * @param {string} agentId - Agent ID
   * @returns {string[]} - Array of agent IDs that can communicate with this agent
   */
  getCommunicationPartners(agentId) {
    if (!this.dynamicTopology || !this.topologyManager) {
      // If dynamic topology disabled, all other agents are valid
      const allAgents = this.network.getAllAgents();
      return allAgents.map(a => a.id).filter(id => id !== agentId);
    }
    
    const neighbors = this.topologyManager.getNeighbors(agentId);
    return neighbors ? Array.from(neighbors) : [];
  }

  /**
   * Execute a single negotiation round
   * @private
   * @returns {Object} - Round execution result
   */
  _executeRound() {
    this._log(`\n--- Round ${this.currentRound + 1} ---`);
    
    // Log current topology state if dynamic
    if (this.dynamicTopology && this.topologyManager) {
      const topology = this._getTopologySnapshot();
      const edgeCount = Object.values(topology).reduce((sum, neighbors) => sum + neighbors.length, 0) / 2;
      this._log(`  Current topology: ${Object.keys(topology).length} agents, ${edgeCount} connections`);
    }
    
    // Create new round
    const round = new NegotiationRound(this.network, {
      strategyResolver: (agent, messages) => this._resolveStrategy(agent, messages)
    });
    
    // Execute round
    const result = round.execute();
    
    // Record round in history
    this.history.push({
      round: this.currentRound + 1,
      timestamp: Date.now(),
      messages: round.getMessages(),
      statistics: round.getStatistics(),
      outcome: round.getOutcome(),
      topology: this.dynamicTopology ? this._getTopologySnapshot() : null
    });
    
    this._log(`Round ${this.currentRound + 1} complete:`);
    this._log(`  Offers: ${result.offers}, Acceptances: ${result.acceptances}, Rejections: ${result.rejections}`);
    
    return result;
  }

  /**
   * Resolve strategy for an agent
   * @private
   */
  _resolveStrategy(agent, messages) {
    const strategyName = agent.getStrategy() || 'counter';
    
    // Get or create strategy instance
    if (!this.strategies.has(agent.id)) {
      const strategy = StrategyFactory.create(strategyName);
      this.strategies.set(agent.id, { strategy, name: strategyName });
    }
    
    const { strategy } = this.strategies.get(agent.id);
    
    // Build context
    const context = {
      agent,
      roundNumber: this.currentRound,
      messages
    };
    
    // Get last message if available
    if (messages.length > 0) {
      const lastMessage = messages[messages.length - 1];
      context.offer = lastMessage.content?.offer || lastMessage.content?.counterOffer;
      context.senderOffers = lastMessage.content?.senderOffers || [];
    }
    
    // Evaluate with strategy
    const decision = strategy.evaluate(context);
    
    // Map to negotiation actions
    return {
      shouldOffer: decision.action === 'counter' || decision.action === 'accept',
      shouldAccept: decision.action === 'accept',
      shouldReject: decision.action === 'reject',
      shouldCounter: decision.action === 'counter',
      decision,
      strategy: strategyName
    };
  }

  /**
   * Check if agreement has been reached
   * @private
   * @returns {boolean}
   */
  _checkAgreement() {
    const agents = this.network.getAllAgents();
    
    // Check if all agents have satisfied needs
    const allSatisfied = agents.every(agent => agent.needsSatisfied());
    
    if (allSatisfied) {
      return true;
    }
    
    // Check recent history for acceptances
    const recentRounds = this.history.slice(-3);
    let recentAcceptances = 0;
    
    for (const round of recentRounds) {
      recentAcceptances += round.statistics.acceptances;
    }
    
    // Agreement if multiple acceptances in recent rounds
    return recentAcceptances >= 3;
  }

  /**
   * Get negotiation history
   * @returns {Object[]} - Array of round history entries
   */
  getHistory() {
    return [...this.history];
  }

  /**
   * Get history for a specific round
   * @param {number} roundNumber - Round number (1-indexed)
   * @returns {Object|null}
   */
  getRoundHistory(roundNumber) {
    return this.history[roundNumber - 1] || null;
  }

  /**
   * Get full negotiation result
   * @returns {Object} - Final outcome
   */
  getResult() {
    if (this.result) {
      return this.result;
    }
    
    const agents = this.network.getAllAgents();
    
    // Calculate final statistics
    let totalOffers = 0;
    let totalAcceptances = 0;
    let totalRejections = 0;
    let totalCounters = 0;
    
    for (const round of this.rounds) {
      totalOffers += round.offers;
      totalAcceptances += round.acceptances;
      totalRejections += round.rejections;
      totalCounters += round.counters;
    }
    
    // Determine success
    const allSatisfied = agents.every(agent => agent.needsSatisfied());
    const successRate = agents.length > 0 
      ? agents.filter(a => a.needsSatisfied()).length / agents.length 
      : 0;
    
    this.result = {
      success: allSatisfied,
      status: this.status,
      rounds: this.currentRound,
      duration: this.endTime - this.startTime,
      statistics: {
        totalOffers,
        totalAcceptances,
        totalRejections,
        totalCounters,
        agreementRate: successRate
      },
      agentResults: agents.map(agent => ({
        id: agent.id,
        needsSatisfied: agent.needsSatisfied(),
        remainingNeeds: agent.getNeeds(),
        historyLength: agent.getHistory().length
      })),
      history: this.history,
      topology: {
        dynamicTopologyEnabled: this.dynamicTopology,
        topologyChanges: this.topologyHistory.length,
        finalTopology: this._getTopologySnapshot()
      }
    };
    
    return this.result;
  }

  /**
   * Get current status
   * @returns {Object} - Status object with topology info
   */
  getStatus() {
    return {
      status: this.status,
      currentRound: this.currentRound,
      maxRounds: this.maxRounds,
      dynamicTopology: this.dynamicTopology,
      topologyChanges: this.topologyHistory.length,
      currentTopology: this._getTopologySnapshot()
    };
  }

  /**
   * Reset the runner for a new negotiation
   */
  reset() {
    this.rounds = [];
    this.history = [];
    this.currentRound = 0;
    this.status = 'idle';
    this.startTime = null;
    this.endTime = null;
    this.result = null;
    this.strategies.clear();
    this.topologyHistory = [];
    
    // Reset topology manager
    if (this.topologyManager) {
      this.topologyManager.graph.clear();
      this.topologyManager.edgeWeights.clear();
    }
    
    // Clear agent histories
    const agents = this.network.getAllAgents();
    for (const agent of agents) {
      agent.clearHistory();
    }
  }

  /**
   * Log message if verbose mode is enabled
   * @private
   */
  _log(message) {
    if (this.verbose) {
      console.log(message);
    }
  }

  /**
   * Run a single step (for debugging/interactive use)
   * @returns {Object} - Step result
   */
  step() {
    if (this.status === 'completed' || this.status === 'timeout' || this.status === 'error') {
      return {
        success: false,
        message: 'Negotiation already finished'
      };
    }
    
    this.status = 'running';
    
    if (!this.startTime) {
      this.startTime = Date.now();
    }
    
    // Rebuild topology if needed
    if (this.dynamicTopology && this.currentRound % this.topologyRebuildInterval === 0) {
      this._rebuildTopology();
    }
    
    const roundResult = this._executeRound();
    this.rounds.push(roundResult);
    this.currentRound++;
    
    // Check if done
    if (this._checkAgreement()) {
      this.status = 'completed';
      this.endTime = Date.now();
    } else if (this.currentRound >= this.maxRounds) {
      this.status = 'completed';
      this.endTime = Date.now();
    } else if (Date.now() - this.startTime > this.timeout) {
      this.status = 'timeout';
      this.endTime = Date.now();
    }
    
    return {
      success: true,
      round: this.currentRound,
      status: this.status,
      result: roundResult,
      topology: this._getTopologySnapshot(),
      isComplete: this.status === 'completed' || this.status === 'timeout'
    };
  }

  /**
   * Get summary of negotiation
   * @returns {Object}
   */
  getSummary() {
    return {
      status: this.status,
      currentRound: this.currentRound,
      maxRounds: this.maxRounds,
      timeout: this.timeout,
      duration: this.endTime ? this.endTime - this.startTime : null,
      rounds: this.rounds.length,
      historyLength: this.history.length,
      dynamicTopology: this.dynamicTopology,
      topologyChanges: this.topologyHistory.length
    };
  }
}

module.exports = NegotiationRunner;
