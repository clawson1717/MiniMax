/**
 * TopologyManager Class
 * 
 * Manages the dynamic communication graph between agents.
 * The topology adapts based on which agents need to know what,
 * implementing relevance-based routing.
 * 
 * Features:
 * - Dynamic topology reconstruction based on relevance
 * - Efficient neighbor queries
 * - Message routing to specific agents
 * - Support for relevance filtering
 */

class TopologyManager {
  /**
   * Create a new TopologyManager
   */
  constructor() {
    this.agents = new Map(); // agentId -> Agent
    this.topology = new Map(); // agentId -> Set of neighbor agentIds
    this.relevanceRules = new Map(); // agentId -> Set of relevant propositions
    this.messageQueue = []; // Queued messages for processing
  }

  /**
   * Register an agent with the topology manager
   * @param {Agent} agent - The agent to register
   */
  registerAgent(agent) {
    this.agents.set(agent.id, agent);
    this.topology.set(agent.id, new Set());
    this.relevanceRules.set(agent.id, new Set());
  }

  /**
   * Unregister an agent from the topology manager
   * @param {string} agentId - The ID of the agent to unregister
   */
  unregisterAgent(agentId) {
    // Remove from topology
    this.topology.delete(agentId);
    this.agents.delete(agentId);
    this.relevanceRules.delete(agentId);

    // Remove from all neighbor sets
    for (const neighbors of this.topology.values()) {
      neighbors.delete(agentId);
    }
  }

  /**
   * Rebuild the communication topology based on relevance rules
   * Each agent connects to other agents whose beliefs are relevant to them
   */
  rebuildTopology() {
    // Clear existing topology
    for (const neighbors of this.topology.values()) {
      neighbors.clear();
    }

    const agentIds = Array.from(this.agents.keys());

    // For each agent, find relevant neighbors
    for (const agentId of agentIds) {
      const agent = this.agents.get(agentId);
      const relevantPropositions = this.relevanceRules.get(agentId);

      if (!agent || !relevantPropositions) continue;

      // Find other agents that have relevant beliefs
      for (const otherId of agentIds) {
        if (agentId === otherId) continue;

        const otherAgent = this.agents.get(otherId);
        if (!otherAgent) continue;

        // Check if other agent has any relevant propositions
        const otherBeliefs = otherAgent.getBeliefs();
        
        for (const belief of otherBeliefs) {
          // Check if this belief's proposition matches any relevance rule
          for (const relevantProp of relevantPropositions) {
            if (this._isRelevant(belief.proposition, relevantProp)) {
              // Add bidirectional connection
              this.topology.get(agentId).add(otherId);
              this.topology.get(otherId).add(agentId);
              break;
            }
          }
        }
      }
    }
  }

  /**
   * Check if a proposition is relevant to a relevance rule
   * Supports wildcards and partial matching
   * @param {string} proposition - The proposition to check
   * @param {string} rule - The relevance rule
   * @returns {boolean} - True if relevant
   * @private
   */
  _isRelevant(proposition, rule) {
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
   * Set relevance rules for an agent
   * @param {string} agentId - The agent ID
   * @param {string[]} rules - Array of relevance rules (propositions/topics)
   */
  setRelevanceRules(agentId, rules) {
    const agentRules = this.relevanceRules.get(agentId);
    if (agentRules) {
      agentRules.clear();
      for (const rule of rules) {
        agentRules.add(rule);
      }
    }
  }

  /**
   * Add a relevance rule for an agent
   * @param {string} agentId - The agent ID
   * @param {string} rule - The relevance rule to add
   */
  addRelevanceRule(agentId, rule) {
    const agentRules = this.relevanceRules.get(agentId);
    if (agentRules) {
      agentRules.add(rule);
    }
  }

  /**
   * Get neighbors of an agent
   * @param {string} agentId - The agent ID
   * @returns {Array<string>} - Array of neighbor agent IDs
   */
  getNeighbors(agentId) {
    const neighbors = this.topology.get(agentId);
    return neighbors ? Array.from(neighbors) : [];
  }

  /**
   * Get neighbor agents (full objects)
   * @param {string} agentId - The agent ID
   * @returns {Array<Agent>} - Array of neighbor agents
   */
  getNeighborAgents(agentId) {
    const neighborIds = this.getNeighbors(agentId);
    return neighborIds
      .map(id => this.agents.get(id))
      .filter(agent => agent !== undefined);
  }

  /**
   * Send a message to a specific agent
   * @param {string} fromAgentId - Sender agent ID
   * @param {string} toAgentId - Receiver agent ID
   * @param {Object} message - The message to send
   * @returns {boolean} - True if message was delivered
   */
  sendTo(fromAgentId, toAgentId, message) {
    // Check if agents exist
    const fromAgent = this.agents.get(fromAgentId);
    const toAgent = this.agents.get(toAgentId);

    if (!fromAgent || !toAgent) {
      console.error(`Cannot send message: agent not found`);
      return false;
    }

    // Check if they're connected (for directed topology)
    const neighbors = this.topology.get(fromAgentId);
    if (neighbors && !neighbors.has(toAgentId)) {
      console.warn(`Agents ${fromAgentId} and ${toAgentId} are not connected`);
      return false;
    }

    // Deliver message
    const response = toAgent.receiveMessage({
      ...message,
      senderId: fromAgentId,
      timestamp: Date.now()
    });

    return response !== null;
  }

  /**
   * Broadcast a message to all neighbors of an agent
   * @param {string} fromAgentId - Sender agent ID
   * @param {Object} message - The message to broadcast
   * @returns {number} - Number of agents that received the message
   */
  broadcast(fromAgentId, message) {
    const neighbors = this.getNeighbors(fromAgentId);
    let deliveredCount = 0;

    for (const neighborId of neighbors) {
      if (this.sendTo(fromAgentId, neighborId, message)) {
        deliveredCount++;
      }
    }

    return deliveredCount;
  }

  /**
   * Broadcast a message to all agents in the network
   * @param {string} fromAgentId - Sender agent ID
   * @param {Object} message - The message to broadcast
   * @returns {number} - Number of agents that received the message
   */
  broadcastAll(fromAgentId, message) {
    let deliveredCount = 0;

    for (const [agentId, agent] of this.agents.entries()) {
      if (agentId === fromAgentId) continue;
      
      const response = agent.receiveMessage({
        ...message,
        senderId: fromAgentId,
        timestamp: Date.now()
      });

      if (response !== null) {
        deliveredCount++;
      }
    }

    return deliveredCount;
  }

  /**
   * Get all agents relevant to a specific proposition
   * Used for targeted message propagation
   * @param {string} proposition - The proposition to find relevant agents for
   * @returns {Array<string>} - Array of agent IDs that care about this proposition
   */
  getRelevantAgents(proposition) {
    const relevantAgents = [];

    for (const [agentId, rules] of this.relevanceRules.entries()) {
      for (const rule of rules) {
        if (this._isRelevant(proposition, rule)) {
          relevantAgents.push(agentId);
          break;
        }
      }
    }

    return relevantAgents;
  }

  /**
   * Get the current topology as JSON
   * @returns {Object} - Topology structure as JSON
   */
  toJSON() {
    const topologyJson = {};
    for (const [agentId, neighbors] of this.topology.entries()) {
      topologyJson[agentId] = Array.from(neighbors);
    }

    return {
      agents: Array.from(this.agents.keys()),
      topology: topologyJson,
      relevanceRules: Object.fromEntries(
        Array.from(this.relevanceRules.entries()).map(([k, v]) => [k, Array.from(v)])
      )
    };
  }

  /**
   * Get topology statistics
   * @returns {Object} - Statistics object
   */
  getStats() {
    let totalConnections = 0;
    let maxConnections = 0;
    let minConnections = Infinity;

    for (const neighbors of this.topology.values()) {
      const count = neighbors.size;
      totalConnections += count;
      maxConnections = Math.max(maxConnections, count);
      minConnections = Math.min(minConnections, count);
    }

    const agentCount = this.agents.size;

    return {
      agentCount,
      totalConnections: totalConnections / 2, // Each connection counted twice
      avgConnections: agentCount > 0 ? totalConnections / (2 * agentCount) : 0,
      maxConnections,
      minConnections: agentCount > 0 ? minConnections : 0,
      density: agentCount > 1 ? (totalConnections / 2) / (agentCount * (agentCount - 1) / 2) : 0
    };
  }

  /**
   * Get registered agent by ID
   * @param {string} agentId - The agent ID
   * @returns {Agent|undefined} - The agent or undefined
   */
  getAgent(agentId) {
    return this.agents.get(agentId);
  }

  /**
   * Get all registered agents
   * @returns {Array<Agent>} - Array of all agents
   */
  getAllAgents() {
    return Array.from(this.agents.values());
  }

  /**
   * Clear the topology (disconnect all agents)
   */
  clearTopology() {
    for (const neighbors of this.topology.values()) {
      neighbors.clear();
    }
  }
}

module.exports = { TopologyManager };
