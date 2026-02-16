/**
 * DynamicTopologyManager Class
 * 
 * Reconstructs communication graphs based on semantic matching of agent needs/offers.
 * Implements DyTopo-inspired approach for dynamic topology routing.
 */

const SemanticMatcher = require('./SemanticMatcher');

class DynamicTopologyManager {
  /**
   * Create a new DynamicTopologyManager
   * @param {Object} options - Configuration options
   * @param {string} options.matcherMethod - Embedding method: 'tfidf', 'hash', 'combined' (default: 'tfidf')
   * @param {number} options.threshold - Minimum similarity threshold for edges (default: 0.3)
   * @param {number} options.maxNeighbors - Maximum neighbors per agent (default: 5)
   * @param {boolean} options.bidirectional - Require mutual compatibility for edge (default: true)
   */
  constructor(options = {}) {
    this.matcher = new SemanticMatcher({
      method: options.matcherMethod || 'tfidf',
      threshold: options.threshold || 0.3
    });
    this.threshold = options.threshold || 0.3;
    this.maxNeighbors = options.maxNeighbors || 5;
    this.bidirectional = options.bidirectional !== false;
    
    // Graph structure: adjacency list
    this.graph = new Map();
    
    // Edge weights for pathfinding
    this.edgeWeights = new Map();
    
    // Agent registry
    this.agents = new Map();
    
    // Topology history for analysis
    this.topologyHistory = [];
  }

  /**
   * Register an agent with the topology manager
   * @param {Agent} agent - The agent to register
   */
  registerAgent(agent) {
    this.agents.set(agent.id, agent);
    if (!this.graph.has(agent.id)) {
      this.graph.set(agent.id, new Set());
    }
  }

  /**
   * Unregister an agent from the topology manager
   * @param {string} agentId - Agent ID to remove
   */
  unregisterAgent(agentId) {
    this.agents.delete(agentId);
    this.graph.delete(agentId);
    
    // Remove agent from all neighbor sets
    for (const neighbors of this.graph.values()) {
      neighbors.delete(agentId);
    }
    
    // Clean up edge weights
    this.edgeWeights.delete(agentId);
  }

  /**
   * Rebuild the entire topology based on current agent needs/offers
   * This is the core DyTopo-inspired method
   * @returns {Object} - Topology statistics
   */
  rebuildTopology() {
    const agentList = Array.from(this.agents.values());
    
    // Clear current graph
    this.graph.clear();
    this.edgeWeights.clear();
    
    // Initialize all agents in graph
    for (const agent of agentList) {
      this.graph.set(agent.id, new Set());
    }
    
    let edgeCount = 0;
    
    // Compute compatibility and build edges
    for (let i = 0; i < agentList.length; i++) {
      for (let j = 0; j < agentList.length; j++) {
        if (i === j) continue;
        
        const agentA = agentList[i];
        const agentB = agentList[j];
        
        // Compute bidirectional compatibility
        const aToB = this.matcher.computeSimilarityMatrix(
          agentA.getNeeds(),
          agentB.getOffers()
        );
        const bToA = this.matcher.computeSimilarityMatrix(
          agentB.getNeeds(),
          agentA.getOffers()
        );
        
        // Combined score
        const combinedScore = (aToB + bToA) / 2;
        
        // Check if edge should be created
        let shouldConnect = combinedScore >= this.threshold;
        
        // For bidirectional mode, both directions must meet threshold
        if (this.bidirectional) {
          shouldConnect = shouldConnect && aToB >= this.threshold && bToA >= this.threshold;
        }
        
        if (shouldConnect) {
          // Check neighbor limits
          const neighborsA = this.graph.get(agentA.id);
          const neighborsB = this.graph.get(agentB.id);
          
          if (neighborsA.size < this.maxNeighbors && neighborsB.size < this.maxNeighbors) {
            neighborsA.add(agentB.id);
            neighborsB.add(agentA.id);
            
            // Store edge weight (inverse of similarity for pathfinding)
            const edgeKey = `${agentA.id}->${agentB.id}`;
            this.edgeWeights.set(edgeKey, 1 - combinedScore);
            
            edgeCount++;
          }
        }
      }
    }
    
    // Record topology state
    const topologyState = {
      timestamp: Date.now(),
      agentCount: agentList.length,
      edgeCount,
      density: edgeCount > 0 ? edgeCount / (agentList.length * (agentList.length - 1)) : 0
    };
    this.topologyHistory.push(topologyState);
    
    return topologyState;
  }

  /**
   * Update topology for a specific agent (partial rebuild)
   * @param {string} agentId - Agent to update
   * @returns {Object} - Updated neighbors
   */
  updateAgentTopology(agentId) {
    const agent = this.agents.get(agentId);
    if (!agent) {
      throw new Error(`Agent ${agentId} not found`);
    }
    
    // Get all other agents
    const otherAgents = Array.from(this.agents.values()).filter(a => a.id !== agentId);
    
    // Clear existing connections for this agent
    const currentNeighbors = this.graph.get(agentId);
    for (const neighborId of currentNeighbors) {
      this.graph.get(neighborId).delete(agentId);
    }
    currentNeighbors.clear();
    
    // Find compatible agents
    const compatibility = this.matcher.findCompatibleAgents(agent, otherAgents);
    
    // Add up to maxNeighbors
    const neighbors = this.graph.get(agentId);
    for (let i = 0; i < Math.min(compatibility.length, this.maxNeighbors); i++) {
      const match = compatibility[i];
      neighbors.add(match.agent.id);
      this.graph.get(match.agent.id).add(agentId);
      
      // Update edge weight
      const edgeKey = `${agentId}->${match.agent.id}`;
      this.edgeWeights.set(edgeKey, 1 - match.score);
    }
    
    return {
      agentId,
      neighborCount: neighbors.size,
      neighbors: Array.from(neighbors)
    };
  }

  /**
   * Get neighbors for a specific agent
   * @param {string} agentId
   * @returns {string[]} - Array of neighbor agent IDs
   */
  getNeighbors(agentId) {
    const neighbors = this.graph.get(agentId);
    if (!neighbors) {
      return [];
    }
    return Array.from(neighbors);
  }

  /**
   * Get neighbor agents (not just IDs)
   * @param {string} agentId
   * @returns {Agent[]} - Array of neighbor agents
   */
  getNeighborAgents(agentId) {
    const neighborIds = this.getNeighbors(agentId);
    return neighborIds
      .map(id => this.agents.get(id))
      .filter(Boolean);
  }

  /**
   * Find path between two agents using BFS
   * @param {string} sourceId - Source agent ID
   * @param {string} targetId - Target agent ID
   * @returns {string[]|null} - Path as array of agent IDs, or null if no path
   */
  getPath(sourceId, targetId) {
    if (!this.graph.has(sourceId) || !this.graph.has(targetId)) {
      return null;
    }
    
    // BFS for shortest path
    const queue = [[sourceId]];
    const visited = new Set([sourceId]);
    
    while (queue.length > 0) {
      const path = queue.shift();
      const current = path[path.length - 1];
      
      if (current === targetId) {
        return path;
      }
      
      const neighbors = this.graph.get(current);
      for (const neighbor of neighbors) {
        if (!visited.has(neighbor)) {
          visited.add(neighbor);
          queue.push([...path, neighbor]);
        }
      }
    }
    
    return null; // No path found
  }

  /**
   * Find weighted path between two agents using Dijkstra's algorithm
   * @param {string} sourceId - Source agent ID
   * @param {string} targetId - Target agent ID
   * @returns {Object|null} - Path with total weight, or null if no path
   */
  getWeightedPath(sourceId, targetId) {
    if (!this.graph.has(sourceId) || !this.graph.has(targetId)) {
      return null;
    }
    
    // Dijkstra's algorithm
    const distances = new Map();
    const previous = new Map();
    const visited = new Set();
    
    // Initialize distances
    for (const agentId of this.agents.keys()) {
      distances.set(agentId, Infinity);
    }
    distances.set(sourceId, 0);
    
    // Priority queue (simple array, could be optimized)
    const queue = [sourceId];
    
    while (queue.length > 0) {
      // Get node with minimum distance
      queue.sort((a, b) => distances.get(a) - distances.get(b));
      const current = queue.shift();
      
      if (current === targetId) {
        // Reconstruct path
        const path = [];
        let node = targetId;
        while (node) {
          path.unshift(node);
          node = previous.get(node);
        }
        return {
          path,
          totalWeight: distances.get(targetId)
        };
      }
      
      if (visited.has(current)) continue;
      visited.add(current);
      
      // Update distances to neighbors
      const neighbors = this.graph.get(current);
      for (const neighbor of neighbors) {
        if (visited.has(neighbor)) continue;
        
        const edgeKey = `${current}->${neighbor}`;
        const weight = this.edgeWeights.get(edgeKey) || 1;
        const newDistance = distances.get(current) + weight;
        
        if (newDistance < distances.get(neighbor)) {
          distances.set(neighbor, newDistance);
          previous.set(neighbor, current);
        }
        
        if (!queue.includes(neighbor)) {
          queue.push(neighbor);
        }
      }
    }
    
    return null; // No path found
  }

  /**
   * Get the complete graph structure
   * @returns {Object} - Graph as adjacency list
   */
  getGraph() {
    const result = {};
    for (const [agentId, neighbors] of this.graph.entries()) {
      result[agentId] = Array.from(neighbors);
    }
    return result;
  }

  /**
   * Get topology statistics
   * @returns {Object}
   */
  getStatistics() {
    let totalEdges = 0;
    let avgDegree = 0;
    
    for (const neighbors of this.graph.values()) {
      totalEdges += neighbors.size;
    }
    avgDegree = this.agents.size > 0 ? totalEdges / this.agents.size : 0;
    
    return {
      agentCount: this.agents.size,
      totalEdges,
      averageDegree: avgDegree,
      maxNeighbors: this.maxNeighbors,
      threshold: this.threshold,
      bidirectional: this.bidirectional
    };
  }

  /**
   * Get topology history
   * @returns {Object[]}
   */
  getTopologyHistory() {
    return [...this.topologyHistory];
  }

  /**
   * Check if two agents are directly connected
   * @param {string} agentIdA
   * @param {string} agentIdB
   * @returns {boolean}
   */
  areConnected(agentIdA, agentIdB) {
    const neighbors = this.graph.get(agentIdA);
    return neighbors ? neighbors.has(agentIdB) : false;
  }

  /**
   * Get edge weight between two agents
   * @param {string} agentIdA
   * @param {string} agentIdB
   * @returns {number|null}
   */
  getEdgeWeight(agentIdA, agentIdB) {
    const edgeKey = `${agentIdA}->${agentIdB}`;
    return this.edgeWeights.get(edgeKey) ?? null;
  }

  /**
   * Get all agents within N hops of an agent
   * @param {string} agentId
   * @param {number} maxHops
   * @returns {Object} - Map of agent IDs to hop distance
   */
  getAgentsWithinHops(agentId, maxHops) {
    if (!this.graph.has(agentId)) {
      return {};
    }
    
    const result = new Map();
    const queue = [[agentId, 0]];
    const visited = new Set([agentId]);
    
    while (queue.length > 0) {
      const [current, hops] = queue.shift();
      
      if (hops > maxHops) continue;
      
      if (hops > 0) {
        result.set(current, hops);
      }
      
      const neighbors = this.graph.get(current);
      for (const neighbor of neighbors) {
        if (!visited.has(neighbor)) {
          visited.add(neighbor);
          queue.push([neighbor, hops + 1]);
        }
      }
    }
    
    return Object.fromEntries(result);
  }

  /**
   * Clear topology history
   */
  clearHistory() {
    this.topologyHistory = [];
  }

  /**
   * Reset the topology
   */
  reset() {
    this.graph.clear();
    this.edgeWeights.clear();
    this.topologyHistory = [];
  }
}

module.exports = DynamicTopologyManager;
