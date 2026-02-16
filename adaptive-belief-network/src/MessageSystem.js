/**
 * MessageSystem Class
 * 
 * Handles agent communication and message propagation.
 * Supports different message types and implements relevance filtering.
 * 
 * Message Types:
 * - belief_update: Notify agents of belief changes
 * - query: Request information from other agents
 * - response: Reply to queries
 * 
 * Features:
 * - Send messages to specific agents
 * - Broadcast messages to neighbors or all agents
 * - Relevance filtering for message routing
 * - Message queuing for async processing
 */

const { TopologyManager } = require('./TopologyManager');

class MessageSystem {
  /**
   * Create a new MessageSystem
   * @param {TopologyManager} topologyManager - Optional topology manager instance
   */
  constructor(topologyManager = null) {
    this.topologyManager = topologyManager || new TopologyManager();
    this.messageLog = [];
    this.pendingMessages = [];
    this.messageIdCounter = 0;
    this.filters = new Set(); // Global message filters
  }

  /**
   * Generate a unique message ID
   * @returns {string} - Unique message ID
   * @private
   */
  _generateMessageId() {
    return `msg_${Date.now()}_${this.messageIdCounter++}`;
  }

  /**
   * Send a message to a specific agent
   * @param {string} fromAgentId - Sender agent ID
   * @param {string} toAgentId - Receiver agent ID
   * @param {string} type - Message type (belief_update, query, response)
   * @param {Object} payload - Message payload
   * @returns {Object|null} - Result object or null if failed
   */
  send(fromAgentId, toAgentId, type, payload) {
    const message = {
      id: this._generateMessageId(),
      type,
      senderId: fromAgentId,
      targetId: toAgentId,
      payload,
      timestamp: Date.now(),
      status: 'pending'
    };

    // Apply global filters
    if (!this._passesFilters(message)) {
      message.status = 'filtered';
      this._logMessage(message);
      return { success: false, reason: 'filtered', message };
    }

    // Apply relevance filtering
    if (!this.isRelevant(message, toAgentId)) {
      message.status = 'filtered';
      this._logMessage(message);
      return { success: false, reason: 'not_relevant', message };
    }

    // Send via topology manager
    const success = this.topologyManager.sendTo(fromAgentId, toAgentId, message);
    
    message.status = success ? 'delivered' : 'failed';
    this._logMessage(message);

    return { success, messageId: message.id, message };
  }

  /**
   * Broadcast a message to all neighbors of an agent
   * @param {string} fromAgentId - Sender agent ID
   * @param {string} type - Message type
   * @param {Object} payload - Message payload
   * @returns {Object} - Result with delivery count
   */
  broadcast(fromAgentId, type, payload) {
    const message = {
      id: this._generateMessageId(),
      type,
      senderId: fromAgentId,
      payload,
      timestamp: Date.now(),
      status: 'pending'
    };

    // Apply global filters first
    if (!this._passesFilters(message)) {
      message.status = 'filtered';
      this._logMessage(message);
      return { success: false, reason: 'filtered', deliveredCount: 0, message };
    }

    // Get neighbors
    const neighbors = this.topologyManager.getNeighbors(fromAgentId);
    let deliveredCount = 0;
    const results = [];

    for (const neighborId of neighbors) {
      // Create individual message for relevance check
      const neighborMessage = {
        ...message,
        targetId: neighborId
      };

      // Check relevance for this specific neighbor
      if (!this.isRelevant(neighborMessage, neighborId)) {
        results.push({ neighborId, success: false, reason: 'not_relevant' });
        continue;
      }

      // Send the message
      const result = this.topologyManager.sendTo(fromAgentId, neighborId, neighborMessage);
      
      if (result) {
        deliveredCount++;
        results.push({ neighborId, success: true });
      } else {
        results.push({ neighborId, success: false, reason: 'delivery_failed' });
      }
    }

    this._logMessage({
      ...message,
      status: deliveredCount > 0 ? 'broadcast' : 'failed',
      deliveredCount
    });

    return {
      success: deliveredCount > 0,
      deliveredCount,
      totalNeighbors: neighbors.length,
      results
    };
  }

  /**
   * Broadcast a message to all agents in the network
   * @param {string} fromAgentId - Sender agent ID
   * @param {string} type - Message type
   * @param {Object} payload - Message payload
   * @returns {Object} - Result with delivery count
   */
  broadcastAll(fromAgentId, type, payload) {
    const message = {
      id: this._generateMessageId(),
      type,
      senderId: fromAgentId,
      payload,
      timestamp: Date.now(),
      status: 'pending'
    };

    // Apply global filters first
    if (!this._passesFilters(message)) {
      message.status = 'filtered';
      this._logMessage(message);
      return { success: false, reason: 'filtered', deliveredCount: 0, message };
    }

    const agents = this.topologyManager.getAllAgents();
    let deliveredCount = 0;
    const results = [];

    for (const agent of agents) {
      if (agent.id === fromAgentId) continue;

      // Create individual message for relevance check
      const agentMessage = {
        ...message,
        targetId: agent.id
      };

      // Check relevance for this specific agent
      if (!this.isRelevant(agentMessage, agent.id)) {
        results.push({ agentId: agent.id, success: false, reason: 'not_relevant' });
        continue;
      }

      // Send the message
      const result = agent.receiveMessage(agentMessage);
      
      if (result !== null) {
        deliveredCount++;
        results.push({ agentId: agent.id, success: true });
      } else {
        results.push({ agentId: agent.id, success: false, reason: 'delivery_failed' });
      }
    }

    this._logMessage({
      ...message,
      status: deliveredCount > 0 ? 'broadcast_all' : 'failed',
      deliveredCount
    });

    return {
      success: deliveredCount > 0,
      deliveredCount,
      totalAgents: agents.length - 1,
      results
    };
  }

  /**
   * Send a belief update to relevant agents
   * @param {string} fromAgentId - Sender agent ID
   * @param {string} proposition - The proposition that was updated
   * @param {number} confidence - New confidence level
   * @param {string} justification - Justification for the update
   * @returns {Object} - Result with delivery count
   */
  sendBeliefUpdate(fromAgentId, proposition, confidence, justification) {
    const payload = {
      proposition,
      confidence,
      justification
    };

    // Find agents that care about this proposition
    const relevantAgents = this.topologyManager.getRelevantAgents(proposition);
    
    let deliveredCount = 0;
    const results = [];

    for (const agentId of relevantAgents) {
      if (agentId === fromAgentId) continue;

      const result = this.send(fromAgentId, agentId, 'belief_update', payload);
      
      if (result.success) {
        deliveredCount++;
      }
      results.push({ agentId, ...result });
    }

    return {
      success: deliveredCount > 0,
      deliveredCount,
      totalRelevant: relevantAgents.length - 1,
      results
    };
  }

  /**
   * Query another agent for information
   * @param {string} fromAgentId - Sender agent ID
   * @param {string} toAgentId - Receiver agent ID
   * @param {string} queryType - Type of query (get_belief, get_all_beliefs, get_related)
   * @param {Object} queryParams - Query parameters
   * @returns {Promise<Object>} - Promise resolving to the response
   */
  async query(fromAgentId, toAgentId, queryType, queryParams = {}) {
    const payload = {
      queryType,
      ...queryParams,
      senderId: fromAgentId
    };

    // Send query message
    const result = this.send(fromAgentId, toAgentId, 'query', payload);
    
    return result;
  }

  /**
   * Receive a message (for external message injection)
   * @param {Object} message - The message to receive
   * @returns {Object|null} - Result or null if no handler
   */
  receive(message) {
    const { targetId, type, payload } = message;
    const targetAgent = this.topologyManager.getAgent(targetId);

    if (!targetAgent) {
      return { success: false, reason: 'agent_not_found' };
    }

    // Apply filters
    if (!this._passesFilters(message) || !this.isRelevant(message, targetId)) {
      return { success: false, reason: 'filtered' };
    }

    // Process message
    const response = targetAgent.receiveMessage(message);
    
    this._logMessage({
      ...message,
      status: response ? 'received' : 'failed'
    });

    return { success: true, response };
  }

  /**
   * Check if a message is relevant to a specific agent
   * @param {Object} message - The message to check
   * @param {string} agentId - The agent ID to check relevance for
   * @returns {boolean} - True if relevant
   */
  isRelevant(message, agentId) {
    const agent = this.topologyManager.getAgent(agentId);
    
    if (!agent) {
      return false;
    }

    // Get agent's subscriptions and belief topics
    const subscriptions = agent.subscriptions;
    
    // If agent has no subscriptions, receive all messages
    if (subscriptions.size === 0) {
      return true;
    }

    // Check message type and payload for relevance
    switch (message.type) {
      case 'belief_update':
        const proposition = message.payload?.proposition;
        if (proposition) {
          // Check if any subscription matches
          for (const sub of subscriptions) {
            if (this._matchesRelevance(proposition, sub)) {
              return true;
            }
          }
        }
        break;

      case 'query':
        // Queries are relevant if agent has related beliefs
        const queryProposition = message.payload?.proposition;
        if (queryProposition && agent.getBelief(queryProposition)) {
          return true;
        }
        break;

      case 'response':
        // Responses are always relevant to the original sender
        return message.senderId === agentId;
    }

    // Default: not relevant
    return false;
  }

  /**
   * Check if a proposition matches a relevance rule
   * @param {string} proposition - The proposition
   * @param {string} rule - The relevance rule
   * @returns {boolean} - True if matches
   * @private
   */
  _matchesRelevance(proposition, rule) {
    // Exact match
    if (proposition === rule) return true;
    
    // Wildcard matching (* at end)
    if (rule.endsWith('*')) {
      const prefix = rule.slice(0, -1);
      return proposition.startsWith(prefix);
    }
    
    // Wildcard matching (* at start)
    if (rule.startsWith('*')) {
      const suffix = rule.slice(1);
      return proposition.endsWith(suffix);
    }
    
    // Partial match
    return proposition.includes(rule);
  }

  /**
   * Filter messages based on criteria
   * @param {Object} message - The message to filter
   * @returns {Array<Object>} - Filtered messages
   */
  filterMessages(messages, criteria = {}) {
    let filtered = [...messages];

    // Filter by type
    if (criteria.type) {
      filtered = filtered.filter(m => m.type === criteria.type);
    }

    // Filter by sender
    if (criteria.senderId) {
      filtered = filtered.filter(m => m.senderId === criteria.senderId);
    }

    // Filter by status
    if (criteria.status) {
      filtered = filtered.filter(m => m.status === criteria.status);
    }

    // Filter by time range
    if (criteria.after) {
      filtered = filtered.filter(m => m.timestamp >= criteria.after);
    }
    if (criteria.before) {
      filtered = filtered.filter(m => m.timestamp <= criteria.before);
    }

    // Filter by relevance to an agent
    if (criteria.agentId) {
      filtered = filtered.filter(m => this.isRelevant(m, criteria.agentId));
    }

    return filtered;
  }

  /**
   * Add a global message filter
   * @param {Function} filterFn - Filter function (message) => boolean
   */
  addFilter(filterFn) {
    this.filters.add(filterFn);
  }

  /**
   * Remove a global message filter
   * @param {Function} filterFn - Filter function to remove
   */
  removeFilter(filterFn) {
    this.filters.delete(filterFn);
  }

  /**
   * Check if a message passes all global filters
   * @param {Object} message - The message to check
   * @returns {boolean} - True if passes all filters
   * @private
   */
  _passesFilters(message) {
    for (const filter of this.filters) {
      if (!filter(message)) {
        return false;
      }
    }
    return true;
  }

  /**
   * Log a message to the message log
   * @param {Object} message - The message to log
   * @private
   */
  _logMessage(message) {
    this.messageLog.push({
      ...message,
      loggedAt: Date.now()
    });
  }

  /**
   * Get message log
   * @param {Object} criteria - Optional filter criteria
   * @returns {Array} - Filtered message log
   */
  getMessageLog(criteria = {}) {
    if (Object.keys(criteria).length === 0) {
      return [...this.messageLog];
    }
    return this.filterMessages(this.messageLog, criteria);
  }

  /**
   * Clear message log
   */
  clearMessageLog() {
    this.messageLog = [];
  }

  /**
   * Get system statistics
   * @returns {Object} - Statistics object
   */
  getStats() {
    const totalMessages = this.messageLog.length;
    const deliveredMessages = this.messageLog.filter(m => m.status === 'delivered' || m.status === 'broadcast').length;
    const filteredMessages = this.messageLog.filter(m => m.status === 'filtered').length;

    return {
      totalMessages,
      deliveredMessages,
      filteredMessages,
      successRate: totalMessages > 0 ? deliveredMessages / totalMessages : 0,
      topologyStats: this.topologyManager.getStats()
    };
  }

  /**
   * Get the topology manager
   * @returns {TopologyManager} - The topology manager
   */
  getTopologyManager() {
    return this.topologyManager;
  }
}

module.exports = { MessageSystem };
