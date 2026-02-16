/**
 * Agent Class
 * 
 * Represents an agent in the adaptive belief network system.
 * Each agent maintains its own belief network and can communicate
 * with other agents via the message system.
 * 
 * Properties:
 * - id: Unique identifier
 * - name: Human-readable name
 * - beliefNetwork: The agent's belief network
 * 
 * Methods:
 * - addBelief(): Add a new belief to the agent's network
 * - updateBelief(): Update an existing belief
 * - getBeliefs(): Get all beliefs
 * - receiveMessage(): Process incoming messages
 */

const { BeliefNetwork } = require('./BeliefNetwork');

class Agent {
  /**
   * Create a new Agent
   * @param {string} id - Unique identifier for the agent
   * @param {string} name - Human-readable name for the agent
   */
  constructor(id, name) {
    this.id = id;
    this.name = name;
    this.beliefNetwork = new BeliefNetwork(id);
    this.messageHistory = [];
    this.subscriptions = new Set(); // Topics this agent is interested in
  }

  /**
   * Add a new belief to the agent's belief network
   * @param {string} proposition - The proposition statement
   * @param {number} confidence - Confidence level (0-1)
   * @param {string} justification - Justification for the belief
   * @param {string[]} [dependencies=[]] - List of propositions this belief depends on
   * @returns {Belief} - The added belief
   */
  addBelief(proposition, confidence, justification, dependencies = []) {
    return this.beliefNetwork.addBelief(proposition, confidence, justification, dependencies);
  }

  /**
   * Update an existing belief in the agent's network
   * @param {string} proposition - The proposition to update
   * @param {number} confidence - New confidence level (0-1)
   * @param {string} justification - New justification
   * @returns {Array<Belief>} - Array of updated beliefs (including propagated updates)
   */
  updateBelief(proposition, confidence, justification) {
    return this.beliefNetwork.updateBelief(proposition, confidence, justification);
  }

  /**
   * Get all beliefs from the agent's network
   * @returns {Array<Belief>} - Array of all beliefs
   */
  getBeliefs() {
    return this.beliefNetwork.getAllBeliefs();
  }

  /**
   * Get a specific belief by proposition
   * @param {string} proposition - The proposition to look up
   * @returns {Belief|undefined} - The belief or undefined if not found
   */
  getBelief(proposition) {
    return this.beliefNetwork.getBelief(proposition);
  }

  /**
   * Receive and process a message from another agent
   * @param {Object} message - The message object
   * @param {string} message.type - Message type (belief_update, query, response)
   * @param {string} message.senderId - ID of the sending agent
   * @param {Object} message.payload - Message payload
   * @param {number} message.timestamp - Message timestamp
   * @returns {Object|null} - Response message if applicable, null otherwise
   */
  receiveMessage(message) {
    // Store message in history
    this.messageHistory.push({
      ...message,
      receivedAt: Date.now()
    });

    switch (message.type) {
      case 'belief_update':
        return this._handleBeliefUpdate(message);
      
      case 'query':
        return this._handleQuery(message);
      
      case 'response':
        return this._handleResponse(message);
      
      default:
        console.warn(`Unknown message type: ${message.type}`);
        return null;
    }
  }

  /**
   * Handle a belief update message
   * @param {Object} message - The belief update message
   * @returns {Object|null} - Response message if belief was updated
   * @private
   */
  _handleBeliefUpdate(message) {
    const { proposition, confidence, justification, senderId } = message.payload;
    
    // Check if we already have this belief
    const existingBelief = this.beliefNetwork.getBelief(proposition);
    
    if (existingBelief) {
      // Update existing belief with weighted average
      const oldConfidence = existingBelief.confidence;
      const newConfidence = (oldConfidence + confidence) / 2;
      existingBelief.update(
        newConfidence,
        `Updated based on info from agent ${senderId}: ${justification}`
      );
      
      return {
        type: 'response',
        senderId: this.id,
        targetId: senderId,
        payload: {
          requestId: message.id,
          status: 'belief_updated',
          proposition,
          newConfidence
        },
        timestamp: Date.now()
      };
    } else {
      // Add new belief
      this.beliefNetwork.addBelief(
        proposition,
        confidence,
        `Received from agent ${senderId}: ${justification}`
      );
      
      return {
        type: 'response',
        senderId: this.id,
        targetId: senderId,
        payload: {
          requestId: message.id,
          status: 'belief_added',
          proposition
        },
        timestamp: Date.now()
      };
    }
  }

  /**
   * Handle a query message
   * @param {Object} message - The query message
   * @returns {Object} - Response message with requested information
   * @private
   */
  _handleQuery(message) {
    const { queryType, proposition, senderId } = message.payload;
    
    let responseData = {};
    
    switch (queryType) {
      case 'get_belief':
        const belief = this.beliefNetwork.getBelief(proposition);
        responseData = belief ? belief.toJSON() : null;
        break;
        
      case 'get_all_beliefs':
        responseData = this.beliefNetwork.getAllBeliefs().map(b => b.toJSON());
        break;
        
      case 'get_related':
        const deps = this.beliefNetwork.getDependencies(proposition);
        const dependents = this.beliefNetwork.getDependent(proposition);
        responseData = {
          dependencies: deps.map(b => b.toJSON()),
          dependents: dependents.map(b => b.toJSON())
        };
        break;
        
      default:
        responseData = { error: 'Unknown query type' };
    }
    
    return {
      type: 'response',
      senderId: this.id,
      targetId: senderId,
      payload: {
        requestId: message.id,
        queryType,
        data: responseData
      },
      timestamp: Date.now()
    };
  }

  /**
   * Handle a response message
   * @param {Object} message - The response message
   * @returns {null} - No further action needed for responses
   * @private
   */
  _handleResponse(message) {
    // Process response - could trigger callbacks or update state
    // For now, just return null (no further action)
    return null;
  }

  /**
   * Subscribe to a topic for relevant message filtering
   * @param {string} topic - The topic to subscribe to
   */
  subscribe(topic) {
    this.subscriptions.add(topic);
  }

  /**
   * Unsubscribe from a topic
   * @param {string} topic - The topic to unsubscribe from
   */
  unsubscribe(topic) {
    this.subscriptions.delete(topic);
  }

  /**
   * Check if agent is subscribed to a topic
   * @param {string} topic - The topic to check
   * @returns {boolean} - True if subscribed
   */
  isSubscribed(topic) {
    return this.subscriptions.has(topic);
  }

  /**
   * Get agent statistics
   * @returns {Object} - Statistics object
   */
  getStats() {
    return {
      id: this.id,
      name: this.name,
      beliefCount: this.beliefNetwork.beliefs.size,
      messageCount: this.messageHistory.length,
      subscriptions: Array.from(this.subscriptions)
    };
  }

  /**
   * Get message history
   * @returns {Array} - Array of received messages
   */
  getMessageHistory() {
    return [...this.messageHistory];
  }

  /**
   * Clear message history
   */
  clearMessageHistory() {
    this.messageHistory = [];
  }

  /**
   * Get JSON representation of the agent
   * @returns {Object} - Agent data as JSON
   */
  toJSON() {
    return {
      id: this.id,
      name: this.name,
      beliefNetwork: this.beliefNetwork.toJSON(),
      stats: this.getStats()
    };
  }
}

module.exports = { Agent };
