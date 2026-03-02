/**
 * Resilient Adaptive Agent
 * Web agent with CATTS compute allocation, attack resilience, and checklist verification
 */

const { Agent, MockBrowser } = require('./Agent');
const { Action, ActionTypes, ActionFactory } = require('./Action');
const { UncertaintyTracker } = require('./UncertaintyTracker');
const { CATTSAllocator } = require('./CATTSAllocator');
const { FailureModeDetector } = require('./FailureModeDetector');
const { ResilienceRecoverySystem } = require('./ResilienceRecovery');
const { ChecklistReward } = require('./ChecklistReward');
const { SelfVerificationSystem, VerificationStatus } = require('./SelfVerification');
const { RAA } = require('./RAA');

module.exports = {
  // Core classes
  Agent,
  MockBrowser,
  Action,
  
  // RAA Main Integration Class
  RAA,
  
  // CATTS-style uncertainty tracking and compute allocation
  UncertaintyTracker,
  CATTSAllocator,
  
  // Multi-turn attack resilience
  FailureModeDetector,
  ResilienceRecoverySystem,
  
  // CM2-style checklist reward verification
  ChecklistReward,
  
  // CM2-style self-verification system
  SelfVerificationSystem,
  VerificationStatus,
  
  // Action types and factory
  ActionTypes,
  ActionFactory
};
