/**
 * NegotiationRunner
 * 
 * Orchestrates multi-round negotiations between agents.
 * Manages execution, timeout, history tracking, and result retrieval.
 */

const NegotiationRound = require('./NegotiationRound');
const { StrategyFactory } = require('./strategies');

class NegotiationRunner {
  /**
   * Create a NegotiationRunner
   * @param {AgentNetwork} network - The agent network
   * @param {Object} options - Configuration options @param {number
   *} options.maxRounds - Maximum number of negotiation rounds
   * @param {number} options.timeout - Timeout in milliseconds
   * @param {boolean} options.verbose - Enable verbose logging
   */
  constructor(network, options = {}) {
    this.network = network;
    this.maxRounds = options.maxRounds || 20;
    this.timeout = options.timeout || 60000; // 1 minute default
    this.verbose = options.verbose || false;
    
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
    
    this._log('Starting negotiation...');
    
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
   * Execute a single negotiation round
   * @private
   * @returns {Object} - Round execution result
   */
  _executeRound() {
    this._log(`\n--- Round ${this.currentRound + 1} ---`);
    
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
      outcome: round.getOutcome()
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
      history: this.history
    };
    
    return this.result;
  }

  /**
   * Get current status
   * @returns {string}
   */
  getStatus() {
    return this.status;
  }

  /**
   * Get current round number
   * @returns {number}
   */
  getCurrentRound() {
    return this.currentRound;
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
      historyLength: this.history.length
    };
  }
}

module.exports = NegotiationRunner;
