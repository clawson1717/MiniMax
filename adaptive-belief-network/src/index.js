/**
 * Adaptive Belief Network
 * 
 * Main entry point for the adaptive-belief-network package.
 * 
 * Exports:
 * - Belief: Class for representing agent beliefs
 * - BeliefNetwork: Class for managing belief networks with reactive updates
 */

const { Belief } = require('./Belief');
const { BeliefNetwork } = require('./BeliefNetwork');

module.exports = {
  Belief,
  BeliefNetwork
};
