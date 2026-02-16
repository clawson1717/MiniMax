/**
 * AgentNetwork Class
 * 
 * Connects agents to the dynamic topology and provides message passing capabilities.
 * Implements round-based communication for multi-agent negotiations.
 */

const DynamicTopologyManager = require('./TopologyManager');

class AgentNetwork {
  /**
   * Create a new AgentNetwork
   * @param {Object} options - Configuration options
   * @param {Object} options.topology - Topology manager options
   */
  constructor(options = {}) {
    this.topology = new DynamicTopologyManager(options.topology || {});
    
    // Agent registry by ID
    this.agents = new Map();
    
    // Message inbox for each agent
    this.inboxes = new Map();
    
    // Round management
    this.currentRound = 0;
    this.roundHistory = [];
    this.maxRounds = options.maxRounds || 10;
    
    // Message history
    this.messageHistory = [];
    
    // Network state
    this.isActive = false;
    
    // Callbacks for message handling
    this.onMessage = options.onMessage || null;
    this.onRoundEnd = options.onRoundEnd || null;
  }

  /**
   * Add an agent to the network
   * @param {Agent} agent - The agent to add
   * @returns {boolean} - Success status
   */
  addAgent(agent) {
    if (this.agents.has(agent.id)) {
      console.warn(`Agent ${agent.id} already exists in network`);
      return false;
    }
    
    this.agents.set(agent.id, agent);
    this.topology.registerAgent(agent);
    this.inboxes.set(agent.id, []);
    
    return true;
  }

  /**
   * Remove an agent from the network
   * @param {string} agentId - Agent ID to remove
   * @returns {boolean} - Success status
   */
  removeAgent(agentId) {
    if (!this.agents.has(agentId)) {
      console.warn(`Agent ${agentId} not found in network`);
      return false;
    }
    
    this.agents.delete(agentId);
    this.topology.unregisterAgent(agentId);
    this.inboxes.delete(agentId);
    
    // Clear any pending messages for this agent
    for (const inbox of this.inboxes.values()) {
      const remaining = inbox.filter(m => m.to !== agentId && m.from !== agentId);
      inbox.length = 0;
      inbox.push(...remaining);
    }
    
    return true;
  }

  /**
   * Get an agent by ID
   * @param {string} agentId
   * @returns {Agent|null}
   */
  getAgent(agentId) {
    return this.agents.get(agentId) || null;
  }

  /**
   * Get all agents in the network
   * @returns {Agent[]}
   */
  getAllAgents() {
    return Array.from(this.agents.values());
  }

  /**
   * Rebuild the network topology based on current agent states
   * @returns {Object} - Topology statistics
   */
  rebuildTopology() {
    return this.topology.rebuildTopology();
  }

  /**
   * Get neighbors for a specific agent
   * @param {string} agentId
   * @returns {Agent[]}
   */
  getNeighbors(agentId) {
    return this.topology.getNeighborAgents(agentId);
  }

  /**
   * Send a message from one agent to another
   * @param {Object} message - Message object
   * @param {string} message.from - Sender agent ID
   * @param {string} message.to - Receiver agent ID
   * @param {string} message.type - Message type (offer, counter, accept, reject, query)
   * @param {*} message.content - Message content
   * @returns {boolean} - Success status
   */
  send(message) {
    const { from, to, type, content } = message;
    
    // Validate sender and receiver
    if (!this.agents.has(from)) {
      console.error(`Sender ${from} not found in network`);
      return false;
    }
    if (!this.agents.has(to)) {
      console.error(`Receiver ${to} not found in network`);
      return false;
    }
    
    // Validate connection
    if (!this.topology.areConnected(from, to)) {
      console.warn(`No direct connection between ${from} and ${to}`);
      // Still deliver message if we want to allow non-local messages
    }
    
    // Create message object
    const messageObj = {
      id: this._generateMessageId(),
      from,
      to,
      type: type || 'message',
      content,
      timestamp: Date.now(),
      round: this.currentRound,
      delivered: false
    };
    
    // Add to receiver's inbox
    const inbox = this.inboxes.get(to);
    inbox.push(messageObj);
    
    // Record in history
    this.messageHistory.push(messageObj);
    
    // Update sender's belief about receiver
    const senderAgent = this.agents.get(from);
    const receiverAgent = this.agents.get(to);
    senderAgent.updateBelief(to, { lastContact: Date.now(), messageType: type });
    receiverAgent.updateBelief(from, { lastContact: Date.now() });
    
    // Trigger callback
    if (this.onMessage) {
      this.onMessage(messageObj);
    }
    
    return true;
  }

  /**
   * Broadcast a message to all neighbors of an agent
   * @param {string} from - Sender agent ID
   * @param {Object} message - Message to broadcast (without 'to' field)
   * @returns {number} - Number of messages sent
   */
  broadcast(from, message) {
    const neighbors = this.topology.getNeighbors(from);
    let sentCount = 0;
    
    for (const neighbor of neighbors) {
      const success = this.send({
        from,
        to: neighbor.id,
        ...message
      });
      if (success) sentCount++;
    }
    
    return sentCount;
  }

  /**
   * Get messages for a specific agent in the current round
   * @param {string} agentId
   * @returns {Object[]}
   */
  getMessages(agentId) {
    const inbox = this.inboxes.get(agentId);
    if (!inbox) return [];
    
    return inbox.filter(m => m.round === this.currentRound);
  }

  /**
   * Receive and clear messages for an agent
   * @param {string} agentId
   * @returns {Object[]}
   */
  receiveMessage(agentId) {
    const inbox = this.inboxes.get(agentId);
    if (!inbox) return [];
    
    const messages = inbox.filter(m => m.round === this.currentRound && !m.read);
    
    // Mark messages as read
    messages.forEach(m => m.read = true);
    
    return messages;
  }

  /**
   * Start a new negotiation round
   * @returns {Object} - Round information
   */
  newRound() {
    if (this.currentRound >= this.maxRounds) {
      return {
        success: false,
        message: 'Maximum rounds reached'
      };
    }
    
    this.currentRound++;
    
    const roundInfo = {
      round: this.currentRound,
      startTime: Date.now(),
      agentCount: this.agents.size,
      messages: []
    };
    
    this.roundHistory.push(roundInfo);
    
    return {
      success: true,
      round: this.currentRound,
      message: `Started round ${this.currentRound}`
    };
  }

  /**
   * Get the current round number
   * @returns {number}
   */
  getCurrentRound() {
    return this.currentRound;
  }

  /**
   * Get round history
   * @returns {Object[]}
   */
  getRoundHistory() {
    return [...this.roundHistory];
  }

  /**
   * Get messages from a specific round
   * @param {number} round - Round number
   * @returns {Object[]}
   */
  getRoundMessages(round) {
    return this.messageHistory.filter(m => m.round === round);
  }

  /**
   * Check if the negotiation is complete
   * @returns {boolean}
   */
  isComplete() {
    // Check if max rounds reached
    if (this.currentRound >= this.maxRounds) {
      return true;
    }
    
    // Check if all agents have resolved their needs
    let allSatisfied = true;
    for (const agent of this.agents.values()) {
      if (!agent.needsSatisfied()) {
        allSatisfied = false;
        break;
      }
    }
    
    return allSatisfied;
  }

  /**
   * Get pending messages for an agent
   * @param {string} agentId
   * @returns {number}
   */
  getPendingMessageCount(agentId) {
    const inbox = this.inboxes.get(agentId);
    if (!inbox) return 0;
    
    return inbox.filter(m => m.to === agentId && !m.read).length;
  }

  /**
   * Clear messages for a specific agent
   * @param {string} agentId
   */
  clearInbox(agentId) {
    const inbox = this.inboxes.get(agentId);
    if (inbox) {
      inbox.length = 0;
    }
  }

  /**
   * Clear all inboxes
   */
  clearAllInboxes() {
    for (const inbox of this.inboxes.values()) {
      inbox.length = 0;
    }
  }

  /**
   * Get network statistics
   * @returns {Object}
   */
  getStatistics() {
    const topologyStats = this.topology.getStatistics();
    
    let totalMessages = 0;
    let unreadMessages = 0;
    for (const inbox of this.inboxes.values()) {
      totalMessages += inbox.length;
      unreadMessages += inbox.filter(m => !m.read).length;
    }
    
    return {
      agentCount: this.agents.size,
      currentRound: this.currentRound,
      maxRounds: this.maxRounds,
      totalMessages,
      unreadMessages,
      isActive: this.isActive,
      isComplete: this.isComplete(),
      ...topologyStats
    };
  }

  /**
   * Find path between two agents
   * @param {string} sourceId
   * @param {string} targetId
   * @returns {string[]|null}
   */
  getPath(sourceId, targetId) {
    return this.topology.getPath(sourceId, targetId);
  }

  /**
   * Get weighted path between two agents
   * @param {string} sourceId
   * @param {string} targetId
   * @returns {Object|null}
   */
  getWeightedPath(sourceId, targetId) {
    return this.topology.getWeightedPath(sourceId, targetId);
  }

  /**
   * Activate the network
   */
  activate() {
    this.isActive = true;
    this.rebuildTopology();
  }

  /**
   * Deactivate the network
   */
  deactivate() {
    this.isActive = false;
  }

  /**
   * Reset the network state
   */
  reset() {
    this.currentRound = 0;
    this.roundHistory = [];
    this.messageHistory = [];
    this.clearAllInboxes();
    this.isActive = false;
  }

  /**
   * Generate a unique message ID
   * @private
   */
  _generateMessageId() {
    return `msg_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  /**
   * Export network state to JSON
   * @returns {Object}
   */
  toJSON() {
    return {
      agents: Array.from(this.agents.values()).map(a => a.toJSON()),
      topology: this.topology.getGraph(),
      currentRound: this.currentRound,
      roundHistory: this.roundHistory,
      messageCount: this.messageHistory.length,
      statistics: this.getStatistics()
    };
  }

  /**
   * Get the topology manager
   * @returns {DynamicTopologyManager}
   */
  getTopologyManager() {
    return this.topology;
  }
}

module.exports = AgentNetwork;
