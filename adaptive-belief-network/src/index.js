/**
 * Adaptive Belief Network
 * 
 * Main entry point for the adaptive-belief-network package.
 * 
 * Exports:
 * - Belief: Class for representing agent beliefs
 * - BeliefNetwork: Class for managing belief networks with reactive updates
 * - Agent: Class for representing agents with belief networks
 * - TopologyManager: Class for managing dynamic communication topology
 * - MessageSystem: Class for agent communication and message propagation
 * - UpdateTrigger: Class for belief update triggers
 * - UpdateTriggerManager: Class for managing trigger events
 * - Simulator: Class for orchestrating multi-agent simulation
 * - Visualizer: Class for console output visualization
 */

const { Belief } = require('./Belief');
const { BeliefNetwork } = require('./BeliefNetwork');
const { Agent } = require('./Agent');
const { TopologyManager } = require('./TopologyManager');
const { MessageSystem } = require('./MessageSystem');
const { UpdateTrigger, UpdateTriggerManager } = require('./UpdateTrigger');
const { Simulator } = require('./Simulator');
const { Visualizer } = require('./Visualizer');

module.exports = {
  Belief,
  BeliefNetwork,
  Agent,
  TopologyManager,
  MessageSystem,
  UpdateTrigger,
  UpdateTriggerManager,
  Simulator,
  Visualizer
};
