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
 */

const { Belief } = require('./Belief');
const { BeliefNetwork } = require('./BeliefNetwork');
const { Agent } = require('./Agent');
const { TopologyManager } = require('./TopologyManager');
const { MessageSystem } = require('./MessageSystem');

module.exports = {
  Belief,
  BeliefNetwork,
  Agent,
  TopologyManager,
  MessageSystem
};
