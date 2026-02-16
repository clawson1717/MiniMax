/**
 * Resilient Adaptive Agent
 * Web agent with CATTS compute allocation, attack resilience, and checklist verification
 */

const { Agent, MockBrowser } = require('./Agent');
const { Action, ActionTypes, ActionFactory } = require('./Action');
const { UncertaintyTracker } = require('./UncertaintyTracker');
const { CATTSAllocator } = require('./CATTSAllocator');

module.exports = {
  // Core classes
  Agent,
  MockBrowser,
  Action,
  
  // CATTS-style uncertainty tracking and compute allocation
  UncertaintyTracker,
  CATTSAllocator,
  
  // Action types and factory
  ActionTypes,
  ActionFactory
};
