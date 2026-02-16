/**
 * Persistence Module
 * 
 * Provides JSON-based persistence for belief networks, agents, and topologies.
 * Supports full state preservation including belief history, message history,
 * and network structure.
 * 
 * Methods:
 * - save(): Save belief network to JSON file
 * - load(): Load belief network from JSON file
 * - exportNetwork(): Export network as JSON string
 * - importNetwork(): Import network from JSON string
 */

const fs = require('fs');
const path = require('path');

const { Belief } = require('./Belief');
const { BeliefNetwork } = require('./BeliefNetwork');
const { Agent } = require('./Agent');
const { TopologyManager } = require('./TopologyManager');

class Persistence {
  /**
   * Save a belief network to a JSON file
   * @param {BeliefNetwork} network - The belief network to save
   * @param {string} filePath - Path to save the file
   * @returns {Promise<boolean>} - True if successful
   */
  static async save(network, filePath) {
    try {
      const data = network.toJSON();
      const serialized = JSON.stringify(data, null, 2);
      await fs.promises.writeFile(filePath, serialized, 'utf-8');
      return true;
    } catch (error) {
      console.error('Error saving belief network:', error.message);
      throw error;
    }
  }

  /**
   * Load a belief network from a JSON file
   * @param {string} filePath - Path to the file to load
   * @param {string} agentId - Agent ID for the loaded network
   * @returns {Promise<BeliefNetwork>} - The loaded belief network
   */
  static async load(filePath, agentId) {
    try {
      const serialized = await fs.promises.readFile(filePath, 'utf-8');
      const data = JSON.parse(serialized);
      return Persistence.importNetwork(data, agentId);
    } catch (error) {
      console.error('Error loading belief network:', error.message);
      throw error;
    }
  }

  /**
   * Export a belief network to a JSON string
   * @param {BeliefNetwork} network - The belief network to export
   * @returns {string} - JSON string representation
   */
  static exportNetwork(network) {
    const data = network.toJSON();
    return JSON.stringify(data, null, 2);
  }

  /**
   * Import a belief network from a JSON object or string
   * @param {Object|string} json - JSON object or string
   * @param {string} agentId - Agent ID for the imported network
   * @returns {BeliefNetwork} - The imported belief network
   */
  static importNetwork(json, agentId) {
    // Parse string if needed
    const data = typeof json === 'string' ? JSON.parse(json) : json;

    // Create new network
    const network = new BeliefNetwork(agentId || data.agentId);

    // Restore beliefs
    if (data.beliefs) {
      for (const beliefData of data.beliefs) {
        // Get dependencies for this belief
        const dependencies = data.dependencies ? data.dependencies[beliefData.proposition] || [] : [];
        
        try {
          network.addBelief(
            beliefData.proposition,
            beliefData.confidence,
            beliefData.justification,
            dependencies
          );

          // Restore timestamp
          const belief = network.getBelief(beliefData.proposition);
          if (belief && beliefData.timestamp) {
            belief.timestamp = beliefData.timestamp;
          }
        } catch (error) {
          console.warn(`Could not restore belief "${beliefData.proposition}":`, error.message);
        }
      }
    }

    return network;
  }

  /**
   * Save an agent to a JSON file
   * @param {Agent} agent - The agent to save
   * @param {string} filePath - Path to save the file
   * @returns {Promise<boolean>} - True if successful
   */
  static async saveAgent(agent, filePath) {
    try {
      const data = agent.toJSON();
      const serialized = JSON.stringify(data, null, 2);
      await fs.promises.writeFile(filePath, serialized, 'utf-8');
      return true;
    } catch (error) {
      console.error('Error saving agent:', error.message);
      throw error;
    }
  }

  /**
   * Load an agent from a JSON file
   * @param {string} filePath - Path to the file to load
   * @returns {Promise<Agent>} - The loaded agent
   */
  static async loadAgent(filePath) {
    try {
      const serialized = await fs.promises.readFile(filePath, 'utf-8');
      const data = JSON.parse(serialized);
      return Persistence.importAgent(data);
    } catch (error) {
      console.error('Error loading agent:', error.message);
      throw error;
    }
  }

  /**
   * Export an agent to a JSON string
   * @param {Agent} agent - The agent to export
   * @returns {string} - JSON string representation
   */
  static exportAgent(agent) {
    const data = agent.toJSON();
    return JSON.stringify(data, null, 2);
  }

  /**
   * Import an agent from a JSON object or string
   * @param {Object|string} json - JSON object or string
   * @returns {Agent} - The imported agent
   */
  static importAgent(json) {
    const data = typeof json === 'string' ? JSON.parse(json) : json;

    // Create new agent
    const agent = new Agent(data.id, data.name);

    // Restore belief network
    if (data.beliefNetwork) {
      agent.beliefNetwork = Persistence.importNetwork(data.beliefNetwork, data.id);
    }

    // Restore subscriptions
    if (data.stats && data.stats.subscriptions) {
      for (const topic of data.stats.subscriptions) {
        agent.subscribe(topic);
      }
    }

    return agent;
  }

  /**
   * Save a topology manager to a JSON file
   * @param {TopologyManager} topologyManager - The topology manager to save
   * @param {string} filePath - Path to save the file
   * @returns {Promise<boolean>} - True if successful
   */
  static async saveTopology(topologyManager, filePath) {
    try {
      const data = topologyManager.toJSON();
      const serialized = JSON.stringify(data, null, 2);
      await fs.promises.writeFile(filePath, serialized, 'utf-8');
      return true;
    } catch (error) {
      console.error('Error saving topology:', error.message);
      throw error;
    }
  }

  /**
   * Load a topology manager from a JSON file
   * @param {string} filePath - Path to the file to load
   * @param {Agent[]} agents - Array of agents to register
   * @returns {Promise<TopologyManager>} - The loaded topology manager
   */
  static async loadTopology(filePath, agents = []) {
    try {
      const serialized = await fs.promises.readFile(filePath, 'utf-8');
      const data = JSON.parse(serialized);
      return Persistence.importTopology(data, agents);
    } catch (error) {
      console.error('Error loading topology:', error.message);
      throw error;
    }
  }

  /**
   * Export a topology manager to a JSON string
   * @param {TopologyManager} topologyManager - The topology manager to export
   * @returns {string} - JSON string representation
   */
  static exportTopology(topologyManager) {
    const data = topologyManager.toJSON();
    return JSON.stringify(data, null, 2);
  }

  /**
   * Import a topology manager from a JSON object or string
   * @param {Object|string} json - JSON object or string
   * @param {Agent[]} agents - Array of agents to register
   * @returns {TopologyManager} - The imported topology manager
   */
  static importTopology(json, agents = []) {
    const data = typeof json === 'string' ? JSON.parse(json) : json;

    // Create new topology manager
    const topologyManager = new TopologyManager();

    // Register agents
    for (const agent of agents) {
      topologyManager.registerAgent(agent);
    }

    // Restore relevance rules
    if (data.relevanceRules) {
      for (const [agentId, rules] of Object.entries(data.relevanceRules)) {
        topologyManager.setRelevanceRules(agentId, rules);
      }
    }

    // Rebuild topology
    topologyManager.rebuildTopology();

    return topologyManager;
  }

  /**
   * Save complete simulation state (agents + topology)
   * @param {Agent[]} agents - Array of agents
   * @param {TopologyManager} topologyManager - The topology manager
   * @param {string} filePath - Path to save the file
   * @returns {Promise<boolean>} - True if successful
   */
  static async saveSimulation(agents, topologyManager, filePath) {
    try {
      const data = {
        agents: agents.map(agent => agent.toJSON()),
        topology: topologyManager.toJSON(),
        savedAt: Date.now()
      };
      const serialized = JSON.stringify(data, null, 2);
      await fs.promises.writeFile(filePath, serialized, 'utf-8');
      return true;
    } catch (error) {
      console.error('Error saving simulation:', error.message);
      throw error;
    }
  }

  /**
   * Load complete simulation state
   * @param {string} filePath - Path to the file to load
   * @returns {Promise<{agents: Agent[], topologyManager: TopologyManager}>} - Loaded simulation
   */
  static async loadSimulation(filePath) {
    try {
      const serialized = await fs.promises.readFile(filePath, 'utf-8');
      const data = JSON.parse(serialized);

      // Load agents
      const agents = [];
      if (data.agents) {
        for (const agentData of data.agents) {
          const agent = Persistence.importAgent(agentData);
          agents.push(agent);
        }
      }

      // Load topology
      const topologyManager = Persistence.importTopology(data.topology, agents);

      return { agents, topologyManager };
    } catch (error) {
      console.error('Error loading simulation:', error.message);
      throw error;
    }
  }

  /**
   * Export complete simulation to JSON string
   * @param {Agent[]} agents - Array of agents
   * @param {TopologyManager} topologyManager - The topology manager
   * @returns {string} - JSON string representation
   */
  static exportSimulation(agents, topologyManager) {
    const data = {
      agents: agents.map(agent => agent.toJSON()),
      topology: topologyManager.toJSON(),
      exportedAt: Date.now()
    };
    return JSON.stringify(data, null, 2);
  }

  /**
   * Import simulation from JSON string
   * @param {string} jsonString - JSON string
   * @returns {Promise<{agents: Agent[], topologyManager: TopologyManager}>} - Imported simulation
   */
  static async importSimulation(jsonString) {
    const data = typeof jsonString === 'string' ? JSON.parse(jsonString) : jsonString;

    // Load agents
    const agents = [];
    if (data.agents) {
      for (const agentData of data.agents) {
        const agent = Persistence.importAgent(agentData);
        agents.push(agent);
      }
    }

    // Load topology
    const topologyManager = Persistence.importTopology(data.topology, agents);

    return { agents, topologyManager };
  }
}

module.exports = { Persistence };
