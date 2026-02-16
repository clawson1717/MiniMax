/**
 * Resilient Adaptive Agent
 * Web agent with CATTS compute allocation, attack resilience, and checklist verification
 */

const { Agent, MockBrowser } = require('./Agent');
const { Action, ActionTypes, ActionFactory } = require('./Action');
const { UncertaintyTracker } = require('./UncertaintyTracker');

module.exports = {
  // Core classes
  Agent,
  MockBrowser,
  Action,
  
  // CATTS-style uncertainty tracking
  UncertaintyTracker,
  
  // Action types and factory
  ActionTypes,
  ActionFactory
};
