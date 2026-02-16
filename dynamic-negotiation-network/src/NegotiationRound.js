/**
 * NegotiationRound Class
 * 
 * Manages a single negotiation round within the AgentNetwork.
 * Handles message processing, agent responses, and outcome determination.
 */

class NegotiationRound {
  /**
   * Create a new NegotiationRound
   * @param {AgentNetwork} network - The agent network
   * @param {Object} options - Configuration options
   * @param {Function} options.strategyResolver - Function to resolve agent strategies
   */
  constructor(network, options = {}) {
    this.network = network;
    this.roundNumber = network.getCurrentRound();
    this.strategyResolver = options.strategyResolver || this._defaultStrategyResolver;
    
    // Round state
    this.messages = [];
    this.outcomes = new Map();
    this.completed = false;
    
    // Statistics
    this.startTime = null;
    this.endTime = null;
    this.offersMade = 0;
    this.acceptances = 0;
    this.rejections = 0;
    this.counters = 0;
  }

  /**
   * Execute the negotiation round
   * Each agent communicates with neighbors based on their strategy
   * @returns {Object} - Round execution results
   */
  execute() {
    if (this.completed) {
      return {
        success: false,
        message: 'Round already completed'
      };
    }
    
    this.startTime = Date.now();
    
    // Get all agents in the network
    const agents = this.network.getAllAgents();
    
    // Process each agent
    for (const agent of agents) {
      this._processAgent(agent);
    }
    
    this.endTime = Date.now();
    this.completed = true;
    
    return {
      success: true,
      round: this.roundNumber,
      messages: this.messages.length,
      offers: this.offersMade,
      acceptances: this.acceptances,
      rejections: this.rejections,
      counters: this.counters,
      duration: this.endTime - this.startTime
    };
  }

  /**
   * Process a single agent's turn in the round
   * @private
   */
  _processAgent(agent) {
    const neighbors = this.network.getNeighbors(agent.id);
    
    if (neighbors.length === 0) {
      return;
    }
    
    // Get pending messages for this agent
    const pendingMessages = this.network.receiveMessage(agent.id);
    
    // Process each message
    for (const message of pendingMessages) {
      this._processMessage(agent, message);
    }
    
    // Agent makes moves based on strategy
    const strategy = this.strategyResolver(agent, pendingMessages);
    
    if (strategy.shouldOffer && agent.getOffers().length > 0) {
      // Make offers to neighbors
      for (const neighbor of neighbors) {
        if (this._shouldCommunicate(agent, neighbor)) {
          this._sendOffer(agent, neighbor);
        }
      }
    }
    
    // Check if agent's needs are satisfied after this round
    if (agent.needsSatisfied()) {
      this.outcomes.set(agent.id, {
        satisfied: true,
        timestamp: Date.now()
      });
    }
  }

  /**
   * Process an incoming message
   * @private
   */
  _processMessage(agent, message) {
    const { from, type, content } = message;
    
    // Record the message
    this.messages.push({
      ...message,
      processed: true,
      processedAt: Date.now()
    });
    
    // Update agent's belief about sender
    agent.updateBelief(from, {
      lastMessageType: type,
      lastMessageContent: content,
      lastContact: Date.now()
    });
    
    // Handle message based on type
    switch (type) {
      case 'offer':
        this._handleOffer(agent, from, content);
        break;
      case 'counter':
        this._handleCounter(agent, from, content);
        break;
      case 'accept':
        this._handleAccept(agent, from, content);
        break;
      case 'reject':
        this._handleReject(agent, from, content);
        break;
      case 'query':
        this._handleQuery(agent, from, content);
        break;
      default:
        // Generic message handling
        agent.recordEvent({
          type: 'message_received',
          from,
          content
        });
    }
  }

  /**
   * Handle an incoming offer
   * @private
   */
  _handleOffer(agent, from, content) {
    const offer = content.offer || content;
    
    // Check if offer satisfies any needs
    const matchedNeed = agent.getNeeds().find(need =>
      offer.toLowerCase().includes(need.toLowerCase())
    );
    
    if (matchedNeed) {
      // Accept the offer
      this.network.send({
        from: agent.id,
        to: from,
        type: 'accept',
        content: {
          acceptedOffer: offer,
          satisfiedNeed: matchedNeed
        }
      });
      this.acceptances++;
      
      // Remove the satisfied need
      agent.removeNeed(matchedNeed);
      
      // Record event
      agent.recordEvent({
        type: 'offer_accepted',
        from,
        offer,
        need: matchedNeed
      });
    } else {
      // Make counter-offer
      const counterOffer = agent.getOffers()[0];
      if (counterOffer) {
        this.network.send({
          from: agent.id,
          to: from,
          type: 'counter',
          content: {
            originalOffer: offer,
            counterOffer
          }
        });
        this.counters++;
      } else {
        // Reject if no counter-offer available
        this.network.send({
          from: agent.id,
          to: from,
          type: 'reject',
          content: { offer }
        });
        this.rejections++;
      }
    }
    
    this.offersMade++;
  }

  /**
   * Handle a counter-offer
   * @private
   */
  _handleCounter(agent, from, content) {
    const { originalOffer, counterOffer } = content;
    
    // Consider the counter-offer
    agent.recordEvent({
      type: 'counter_received',
      from,
      originalOffer,
      counterOffer
    });
    
    // Evaluate counter-offer against needs
    const matchedNeed = agent.getNeeds().find(need =>
      counterOffer.toLowerCase().includes(need.toLowerCase())
    );
    
    if (matchedNeed) {
      // Accept counter-offer
      this.network.send({
        from: agent.id,
        to: from,
        type: 'accept',
        content: {
          acceptedOffer: counterOffer,
          satisfiedNeed: matchedNeed
        }
      });
      this.acceptances++;
      agent.removeNeed(matchedNeed);
    } else {
      // Reject counter-offer
      this.network.send({
        from: agent.id,
        to: from,
        type: 'reject',
        content: { offer: counterOffer }
      });
      this.rejections++;
    }
    
    this.counters++;
  }

  /**
   * Handle an acceptance
   * @private
   */
  _handleAccept(agent, from, content) {
    const { acceptedOffer, satisfiedNeed } = content;
    
    // Record successful negotiation
    agent.recordEvent({
      type: 'offer_accepted_by',
      from,
      acceptedOffer,
      need: satisfiedNeed
    });
    
    // Update outcome
    this.outcomes.set(agent.id, {
      satisfied: true,
      timestamp: Date.now()
    });
    
    this.acceptances++;
  }

  /**
   * Handle a rejection
   * @private
   */
  _handleReject(agent, from, content) {
    const { offer } = content;
    
    agent.recordEvent({
      type: 'offer_rejected_by',
      from,
      offer
    });
    
    this.rejections++;
  }

  /**
   * Handle a query message
   * @private
   */
  _handleQuery(agent, from, content) {
    // Respond to queries about needs/offers
    this.network.send({
      from: agent.id,
      to: from,
      type: 'response',
      content: {
        needs: agent.getNeeds(),
        offers: agent.getOffers()
      }
    });
  }

  /**
   * Send an offer from one agent to another
   * @private
   */
  _sendOffer(sender, receiver) {
    const offer = sender.getOffers()[Math.floor(Math.random() * sender.getOffers().length)];
    
    const success = this.network.send({
      from: sender.id,
      to: receiver.id,
      type: 'offer',
      content: { offer }
    });
    
    if (success) {
      this.offersMade++;
      sender.recordEvent({
        type: 'offer_sent',
        to: receiver.id,
        offer
      });
    }
  }

  /**
   * Determine if two agents should communicate
   * @private
   */
  _shouldCommunicate(agent, neighbor) {
    // Check if agent has needs that neighbor might satisfy
    const agentNeeds = agent.getNeeds();
    const neighborOffers = neighbor.getOffers();
    
    for (const need of agentNeeds) {
      for (const offer of neighborOffers) {
        if (need.toLowerCase().includes(offer.toLowerCase()) ||
            offer.toLowerCase().includes(need.toLowerCase())) {
          return true;
        }
      }
    }
    
    return true; // Default to communicating
  }

  /**
   * Default strategy resolver
   * @private
   */
  _defaultStrategyResolver(agent, pendingMessages) {
    return {
      shouldOffer: true,
      shouldAccept: true,
      shouldQuery: pendingMessages.length === 0
    };
  }

  /**
   * Get all messages from this round
   * @returns {Object[]}
   */
  getMessages() {
    return [...this.messages];
  }

  /**
   * Get messages of a specific type
   * @param {string} type
   * @returns {Object[]}
   */
  getMessagesByType(type) {
    return this.messages.filter(m => m.type === type);
  }

  /**
   * Get messages involving a specific agent
   * @param {string} agentId
   * @returns {Object[]}
   */
  getMessagesForAgent(agentId) {
    return this.messages.filter(m => m.from === agentId || m.to === agentId);
  }

  /**
   * Check if the round is complete
   * @returns {boolean}
   */
  isComplete() {
    return this.completed;
  }

  /**
   * Get the round outcome
   * @returns {Object}
   */
  getOutcome() {
    if (!this.completed) {
      return {
        complete: false,
        message: 'Round not yet completed'
      };
    }
    
    // Aggregate outcomes by agent
    const agentOutcomes = {};
    for (const [agentId, outcome] of this.outcomes.entries()) {
      agentOutcomes[agentId] = outcome;
    }
    
    // Check which agents have satisfied needs
    const agents = this.network.getAllAgents();
    for (const agent of agents) {
      if (!this.outcomes.has(agent.id)) {
        agentOutcomes[agent.id] = {
          satisfied: agent.needsSatisfied(),
          timestamp: Date.now()
        };
      }
    }
    
    return {
      complete: true,
      round: this.roundNumber,
      duration: this.endTime - this.startTime,
      statistics: {
        totalMessages: this.messages.length,
        offersMade: this.offersMade,
        acceptances: this.acceptances,
        rejections: this.rejections,
        counters: this.counters
      },
      agentOutcomes,
      successfulNegotiations: this.acceptances,
      failedNegotiations: this.rejections
    };
  }

  /**
   * Get round statistics
   * @returns {Object}
   */
  getStatistics() {
    return {
      round: this.roundNumber,
      completed: this.completed,
      messageCount: this.messages.length,
      offersMade: this.offersMade,
      acceptances: this.acceptances,
      rejections: this.rejections,
      counters: this.counters,
      duration: this.endTime ? this.endTime - this.startTime : null
    };
  }

  /**
   * Get the round number
   * @returns {number}
   */
  getRoundNumber() {
    return this.roundNumber;
  }
}

module.exports = NegotiationRound;
